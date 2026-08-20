//! The tuner's driver. One binary, one subcommand per operational tool.
//!
//! Three groups, and the grouping is the point:
//!
//! - **One recording**: `track` runs the partial tracker over it, `estimate`
//!   runs the whole per-note analysis and, given a base preset, writes the
//!   estimates out as a preset file.
//! - **The preset factory**, in the order the stages run: `survey` is stage 1
//!   — everything an isolated recorded note can identify, over a whole sample
//!   library at once — and `fit`, `sympathetic`, `tail`, `noise` and `mics`
//!   are stage 2, which is render-and-measure.
//! - **The standing boards and audits**: `bench`, `compass`, `melody` and
//!   `chain` each write a document into `renders/` that a milestone is read
//!   off; `score`, `brilliance` and `residuals` print; `ab` renders.
//!
//! The drivers themselves are [`tools`]; `track`, `estimate` and `survey` are
//! still below, because they predate the split and have no engine in them.

mod tools;

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use piano_tuner::estimate::decay::{DecayConfig, DecayCurve};
use piano_tuner::estimate::directivity::{balance_drift, pan_spread_for_drift, DirectivityConfig};
use piano_tuner::estimate::hammer::HammerConfig;
use piano_tuner::estimate::noise::{
    fit_noise_screened, NoiseConfig, NoiseScreening, MAX_MECHANISM_LEVEL_DB,
};
use piano_tuner::estimate::spread::{SigmaSpread, SpreadConfig};
use piano_tuner::pipeline::{analyze_note, NoteConfig};
use piano_tuner::preset::{equal_temperament, key_index, Preset, PresetBuilder};
use piano_tuner::survey::{
    measure_mechanism, pooled_velocity_map, HammerReport, Survey, SurveyConfig,
};
use piano_tuner::{
    audio, cents, Error, InharmonicModel, PartialTracker, Result, SampleLibrary, StftConfig,
    TrackerConfig, SAMPLE_RATE,
};

const USAGE: &str = "\
piano-tuner — offline analysis and parameter estimation for piano-emulator

usage:
  piano-tuner <command> [arguments]

one recording:
  track <in.wav|in.flac> --f0 <hz>     the partial trajectories
  estimate <in.wav|in.flac> --f0 <hz>  the whole per-note analysis

the library adapter, run once per library rather than once per fit:
  adapt <library-id> --root <dir> [--out <f.sfz>] [--resample]
        writes the instrument definition a library does not ship, from its
        LibrarySpec over the files actually on disk, and (--resample) brings
        a tree published at another rate onto the engine's clock in one
        offline pass. `adapt --list` names the libraries described.

the preset factory, in the order the stages run:
  survey <instrument.sfz> --preset <base.toml> [options]
        stage 1: everything an isolated recorded note identifies, over a
        whole sample library
  fit <instrument.sfz> --preset <base.toml> [--out <f>] [--stage <name>]...
        stage 2, the per-note fits. Stages, in order, and all five run when
        --stage is not given: false_beat, strike_direction, detune,
        partial_gains, texture. --stage partials is the sixth and runs
        alone: it is not re-entrant and is fitted from the survey base.
        Also: --key <n>, --draw-over-measured (motion stages);
        --keys <a,b,..>, --cache <dir> (partials).
  sympathetic <instrument.sfz> --preset <base.toml> [--out <f>]
        stage 2, render-and-measure: notes.duplex, the halo coupling and
        [voicing.bridge], notes.pan_spread
  tail [data/salamander] [preset.toml] [--key <n>] [--passes <n>] [--out <f>]
        stage 2, the upper partials' decay: notes.partial_sigma_scale and
        notes.synthesized_decay
  level [data/salamander] [preset.toml] [--passes <n>] [--out <f>]
        stage 2, a key's own loudness against the recording of the same
        key: a shrunk, compass-smoothed per-key gain written through
        notes.partial_gains' own pinning (DECISIONS.md 457, re-opening 272)
  noise [data/salamander] [preset.toml] [--key <n>] [--out <f>]
        [--stage balance|mechanism] [--base presets/default.toml]
        stage 2, the mechanism's balance: [noise.strike]'s level and
        velocity law, inverted on the engine's own attack against the
        recordings' at the recorded keys. --stage mechanism is the other
        four events instead — key_off, damper_lift, pedal_down, pedal_up,
        read off the library's own mechanism recordings and screened
        against MAX_MECHANISM_LEVEL_DB, a group that fails it inheriting
        --base's table rather than writing one (DECISIONS.md 531). It has
        no render in it, is re-entrant, and may write in place.
  mics [data/salamander] [preset.toml] [--out <f>] [--stage <name>]...
        stage 2, the microphone pair: [voicing.mics]. --stage geometry
        inverts spacing_m and span_m from the recording's own interchannel
        delays; --stage profile prints the recording's sixth-octave
        interchannel curve; --stage coherence closes width and
        diffuse_coherence and --stage modal the board's mode-controlled
        band, both on the engine's render against the same recordings.
        All but profile run when --stage is not given. --no-holdout skips
        the held-out velocity check.

the listening material, per preset, against its OWN library:
  listen <data-dir> <preset.toml> [renders/<name>]
        the melody line and a pedalled chord phrase, engine and that
        library's own recordings, each normalised separately, with a
        README.md naming which of the tune's keys are genuine takes

the standing boards, each writing its own document:
  bench [data] [renders/realism] [preset.toml]     -> REALISM.md
  compass [data] [renders/compass] [preset.toml] [keys...]
                                                   -> COMPASS.md
  melody [data] [renders/melody] [preset.toml] [flags]
                                                   -> MELODY.md
  chain [data] [renders/chain] [preset.toml]       -> CHAIN.md
  stereo [data] [renders/stereo] [preset.toml]     -> STEREO.md
        the one board that is not a mono sum: the same music through the
        pan-pot, the capsule pair, the shipped preset and the recording

the audits:
  score [preset.toml] [data]              Columns A and B, cell by cell
  brilliance [data] [preset.toml] [shelf_db] [shelf_hz] [--trim <n>]
                                          2-6 and 6-12 kHz against the
                                          recordings, per key and phrase
  residuals [data] [preset.toml] [cache]  the whole residual census
  ab [data] [renders/salamander-ab]       A/B renders of both presets

the one-shot instruments behind DECISIONS.md are not here: they are the
forensics/ crate, outside the workspace's default members. See its README.

options:
  --f0 <hz>          fundamental of the recorded note (required)
  --b <coefficient>  inharmonicity B used to seed the partial search [0]
  --partials <n>     highest partial index to look for [80]
  --window <n>       analysis window in samples [65536]
  --hop <n>          hop in samples [480]
  --pad <n>          zero-pad the transform to n times the window [2]
  --out <file>       track: write the trajectories as JSON
                     estimate/survey: write the preset (needs --preset)

  (track and estimate only; every other command documents its own flags
   in its module header and in the list above)

estimate only:
  --key <n>          MIDI key of the recorded note, for the preset table
  --preset <f.toml>  base preset to write the estimates into

survey only:
  --preset <f.toml>  base preset the estimates are written into (required)
  --cache <dir>      cache tracked trajectories here, and reuse them
  --refresh          re-track every recording even if it is cached
  --threads <n>      workers; 0 asks the machine [0]
  --keys <a,b,..>    survey only these MIDI keys
  --name <name>      name of the written preset
  --credit <text>    attribution, written as a comment at the head of the
                     preset and into its description
  --velocity-map     write the fitted hammer velocity map into the preset
";

fn main() -> ExitCode {
    match run(std::env::args().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("piano-tuner: {error}");
            ExitCode::FAILURE
        }
    }
}

/// The subcommands print their own diagnosis and carry it out as
/// `Box<dyn Error>` — which is what their `main` returned when they were
/// examples — so the dispatcher's error type is the wider one and each
/// message reaches the terminal exactly as its own tool wrote it.
type Exit = std::result::Result<(), Box<dyn std::error::Error>>;

fn run(args: Vec<String>) -> Exit {
    let rest = || args[1..].to_vec();
    match args.first().map(String::as_str) {
        Some("adapt") => tools::adapt::run(rest()),
        Some("track") => Ok(track(&args[1..])?),
        Some("estimate") => Ok(estimate(&args[1..])?),
        Some("survey") => Ok(survey(&args[1..])?),
        Some("fit") => tools::fit::run(rest()),
        Some("sympathetic") => Ok(tools::sympathetic::run(rest())?),
        Some("level") => tools::level::run(rest()),
        Some("listen") => tools::listen::run(rest()),
        Some("tail") => tools::tail::run(rest()),
        Some("bench") => tools::bench::run(rest()),
        Some("compass") => tools::compass::run(rest()),
        Some("melody") => tools::melody::run(rest()),
        Some("noise") => tools::noise::run(rest()),
        Some("mics") => tools::mics::run(rest()),
        Some("radiation") => tools::radiation::run(rest()),
        Some("chain") => tools::chain::run(rest()),
        Some("stereo") => tools::stereo::run(rest()),
        Some("score") => tools::score::run(rest()),
        Some("brilliance") => tools::brilliance::run(rest()),
        Some("residuals") => tools::residuals::run(rest()),
        Some("ab") => tools::ab::run(rest()),
        Some("help") | Some("--help") | Some("-h") | None => {
            print!("{USAGE}");
            Ok(())
        }
        Some(other) => Err(Error::Config(format!("unknown command {other:?}\n\n{USAGE}")).into()),
    }
}

fn track(args: &[String]) -> Result<()> {
    let options = parse_options(args)?;
    let recording = audio::load_at(&options.input, SAMPLE_RATE)?;
    let signal = recording.mono();
    println!(
        "{}: {:.2} s, {} channel(s), {} Hz",
        options.input,
        recording.duration_s(),
        recording.channel_count(),
        recording.sample_rate
    );

    let seed = InharmonicModel::new(options.f0, options.b);
    let tracker = PartialTracker::new(options.tracker()?)?;
    let trajectories = tracker
        .track(&signal, f64::from(SAMPLE_RATE), seed)
        .with_source(options.input.clone());

    if let Some(path) = &options.out {
        trajectories.write_json(path)?;
        println!(
            "wrote {} partials / {} points to {path}",
            trajectories.tracks.len(),
            trajectories.point_count()
        );
        return Ok(());
    }

    println!("onset {:.4} s, {} partials", trajectories.onset_s, trajectories.tracks.len());
    println!("   k     seed Hz  measured Hz    cents   peak amp   frames    span s");
    for track in &trajectories.tracks {
        let measured = track.weighted_frequency().unwrap_or(f64::NAN);
        let peak = track.peak().map(|p| p.amplitude).unwrap_or(0.0);
        let span = track.end_s().unwrap_or(0.0) - track.start_s().unwrap_or(0.0);
        println!(
            "{:4}  {:10.3}  {:11.3}  {:7.2}  {:9.6}  {:7}  {:8.2}",
            track.k,
            seed.partial(track.k),
            measured,
            cents(seed.partial(track.k), measured),
            peak,
            track.len(),
            span
        );
    }
    Ok(())
}

/// Options shared by the two subcommands.
struct Options {
    input: String,
    f0: f64,
    b: f64,
    partials: u32,
    window: usize,
    hop: usize,
    pad: usize,
    key: Option<u8>,
    preset: Option<String>,
    out: Option<String>,
}

fn parse_options(args: &[String]) -> Result<Options> {
    let mut input: Option<String> = None;
    let mut f0: Option<f64> = None;
    let mut options = Options {
        input: String::new(),
        f0: 0.0,
        b: 0.0,
        partials: 80,
        window: 1 << 16,
        hop: 480,
        pad: 2,
        key: None,
        preset: None,
        out: None,
    };
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].clone();
        match arg.as_str() {
            "--f0" => f0 = Some(parse(value(args, &mut i, &arg)?, &arg)?),
            "--b" => options.b = parse(value(args, &mut i, &arg)?, &arg)?,
            "--partials" => options.partials = parse(value(args, &mut i, &arg)?, &arg)?,
            "--window" => options.window = parse(value(args, &mut i, &arg)?, &arg)?,
            "--hop" => options.hop = parse(value(args, &mut i, &arg)?, &arg)?,
            "--pad" => options.pad = parse(value(args, &mut i, &arg)?, &arg)?,
            "--key" => options.key = Some(parse(value(args, &mut i, &arg)?, &arg)?),
            "--preset" => options.preset = Some(value(args, &mut i, &arg)?.to_string()),
            "--out" => options.out = Some(value(args, &mut i, &arg)?.to_string()),
            other if other.starts_with('-') => {
                return Err(Error::Config(format!("unknown option {other:?}")))
            }
            other => input = Some(other.to_string()),
        }
        i += 1;
    }
    options.input = input.ok_or_else(|| Error::Config("no input file".into()))?;
    options.f0 = f0.ok_or_else(|| Error::Config("--f0 is required".into()))?;
    Ok(options)
}

impl Options {
    fn tracker(&self) -> Result<TrackerConfig> {
        Ok(TrackerConfig {
            stft: StftConfig::padded(self.window, self.hop, self.pad)?,
            max_partials: self.partials,
            ..TrackerConfig::default()
        })
    }
}

/// Runs the whole per-note analysis and reports what it found.
fn estimate(args: &[String]) -> Result<()> {
    let options = parse_options(args)?;
    let recording = audio::load_at(&options.input, SAMPLE_RATE)?;
    let signal = recording.mono();
    let config = NoteConfig {
        tracker: options.tracker()?,
        ..NoteConfig::default()
    };
    let analysis = analyze_note(
        &signal,
        f64::from(SAMPLE_RATE),
        InharmonicModel::new(options.f0, options.b),
        &config,
    )?;

    let model = analysis.inharmonic.model;
    println!(
        "{}: {:.2} s, onset {:.3} s",
        options.input,
        recording.duration_s(),
        analysis.trajectories.onset_s
    );
    println!(
        "  f0            {:.4} Hz ({:+.2} cents from the seed), {} partials fitted, \
         residual {:.2} cents",
        model.f0_hz,
        cents(options.f0, model.f0_hz),
        analysis.inharmonic.used.len(),
        analysis.inharmonic.residual_cents
    );
    println!("  B             {:.4e}", model.b);
    let curve = analysis.decays.curve;
    println!(
        "  decay         sigma0 {:.3} /s, sigma1 {:.3} /s  (T60 {:.2} s at the fundamental)",
        curve.sigma0,
        curve.sigma1,
        curve.t60_at(model.f0_hz)
    );
    let split = analysis.decays.polarization;
    println!(
        "  polarization  {:.2} dB, decay ratio {:.3}, over {} partials",
        split.gain_db, split.decay_ratio, split.partials
    );
    match &analysis.unison {
        Some(unison) => println!(
            "  unison        {:.3} cents ({:.3} Hz at the fundamental), confidence {:.2}",
            unison.detune_cents,
            unison.beat_hz_at(model.f0_hz),
            unison.confidence
        ),
        None => println!("  unison        no beat found"),
    }
    match &analysis.strike {
        Some(strike) => println!(
            "  strike        {:.4} of the speaking length, residual {:.2} dB",
            strike.position, strike.residual_db
        ),
        None => println!("  strike        no comb null in range"),
    }
    println!("   k   frequency Hz    T60 s    a(0)      residual dB");
    for fit in &analysis.decays.partials {
        println!(
            "{:4}  {:12.3}  {:7.3}  {:9.6}  {:8.2}",
            fit.k,
            fit.frequency_hz,
            fit.t60(),
            fit.initial_amplitude(),
            fit.residual_db
        );
    }

    let Some(base) = &options.preset else {
        return Ok(());
    };
    let key = options
        .key
        .ok_or_else(|| Error::Config("--preset needs --key".into()))?;
    let preset = PresetBuilder::new(Preset::load(base)?)
        .description(format!(
            "estimated by piano-tuner from {}, key {key}",
            options.input
        ))
        .polarization(split)
        .note(analysis.estimate(key))
        .build()?;
    match &options.out {
        Some(path) => {
            preset.save(path)?;
            println!("wrote {path}");
        }
        None => print!("{}", preset.to_toml()),
    }
    Ok(())
}

// ------------------------------------------------------------------ survey

struct SurveyOptions {
    sfz: String,
    base: String,
    out: Option<String>,
    name: Option<String>,
    credit: Option<String>,
    keys: Option<Vec<u8>>,
    velocity_map: bool,
    config: SurveyConfig,
}

fn parse_survey(args: &[String]) -> Result<SurveyOptions> {
    let mut options = SurveyOptions {
        sfz: String::new(),
        base: String::new(),
        out: None,
        name: None,
        credit: None,
        keys: None,
        velocity_map: false,
        config: SurveyConfig::default(),
    };
    let mut sfz: Option<String> = None;
    let mut base: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].clone();
        match arg.as_str() {
            "--preset" => base = Some(value(args, &mut i, &arg)?.to_string()),
            "--out" => options.out = Some(value(args, &mut i, &arg)?.to_string()),
            "--name" => options.name = Some(value(args, &mut i, &arg)?.to_string()),
            "--credit" => options.credit = Some(value(args, &mut i, &arg)?.to_string()),
            "--cache" => {
                options.config.cache_dir = Some(PathBuf::from(value(args, &mut i, &arg)?))
            }
            "--refresh" => options.config.refresh_cache = true,
            "--threads" => options.config.threads = parse(value(args, &mut i, &arg)?, &arg)?,
            "--velocity-map" => options.velocity_map = true,
            "--keys" => {
                let list = value(args, &mut i, &arg)?;
                options.keys = Some(
                    list.split(',')
                        .map(|k| parse(k.trim(), &arg))
                        .collect::<Result<Vec<u8>>>()?,
                );
            }
            other if other.starts_with('-') => {
                return Err(Error::Config(format!("unknown option {other:?}")))
            }
            other => sfz = Some(other.to_string()),
        }
        i += 1;
    }
    options.sfz = sfz.ok_or_else(|| Error::Config("no instrument file".into()))?;
    options.base = base.ok_or_else(|| Error::Config("survey needs --preset".into()))?;
    Ok(options)
}

/// Stage 1 over a whole sample library.
fn survey(args: &[String]) -> Result<()> {
    let options = parse_survey(args)?;
    let mut library = SampleLibrary::from_sfz(&options.sfz)?;
    if let Some(keys) = &options.keys {
        library = library.restricted_to(keys);
    }
    println!(
        "{}: {} keys, {} recordings",
        options.sfz,
        library.key_count(),
        library.sample_count()
    );

    let total = library.sample_count();
    let mut done = 0usize;
    let started = Instant::now();
    let survey = Survey::run(&library, &options.config, |sample, result| {
        done += 1;
        if let Err(error) = result {
            eprintln!("  ! {}: {error}", sample.path.display());
        }
        // Progress on one line: this run is minutes long and silence looks
        // like a hang.
        eprint!(
            "\r  {done}/{total} recordings, {:.0} s elapsed   ",
            started.elapsed().as_secs_f64()
        );
    });
    eprintln!();

    let base = Preset::load(&options.base)?;
    let decay = options.config.note.decay;
    report_notes(&survey, &base, &decay);

    let spread_config = SpreadConfig::default();
    report_spread(&survey, &base, &spread_config);
    let pan_spread = report_directivity(&library, &options.config);
    let noise = report_mechanism(&library, &base);

    let hammers = report_hammer(&survey, &base);
    let map = pooled_velocity_map(&hammers).ok();
    if let Some(map) = &map {
        println!(
            "\nvelocity map over {} layers: {:.3}..{:.3} m/s (residual {:.3} in ln v)",
            hammers.iter().map(|h| h.layers.len()).sum::<usize>(),
            map.velocity_min,
            map.velocity_max,
            map.residual
        );
    }
    if !survey.failures.is_empty() {
        println!("\n{} recordings failed:", survey.failures.len());
        for failure in &survey.failures {
            println!(
                "  key {:>3} layer {:>2}: {} ({})",
                failure.key,
                failure.layer,
                failure.reason,
                failure.path.display()
            );
        }
    }

    let Some(out) = &options.out else {
        return Ok(());
    };
    let credit = options.credit.clone().unwrap_or_default();
    // The gate's refusals go into `description`, which is this schema's own
    // free-form provenance field ("which piano, which recordings, which
    // pipeline run"): a mechanism table left at the base preset's value is not
    // a measurement of this piano and the file has to say so rather than
    // leaving the reader to recognise §5's numbers. `DECISIONS.md` 531.
    let described = format!(
        "estimated by piano-tuner from {}{}{credit}",
        options.sfz,
        if credit.is_empty() { "" } else { "; " }
    );
    let described = match &noise {
        Some((_, screening)) => screening.describe(&described),
        None => described,
    };
    let mut builder = survey.builder(base, &decay).description(described);
    if let Some(name) = &options.name {
        builder = builder.name(name.clone());
    }
    if options.velocity_map {
        builder = builder.velocity_map(
            map.ok_or_else(|| Error::Estimate("no velocity map was fitted".into()))?,
        );
    }
    // `spread` is measured and printed above and deliberately **not** written.
    // `voicing.unison_sigma_scale` has been inert since `DECISIONS.md` 225 —
    // the per-string decay split it existed to carry is an output of the
    // coupled construction, not an input — so a survey that wrote it put a
    // number into every preset it emitted that nothing computes with, and the
    // engine warned about it on every load (item 324). The measurement is worth
    // printing: it is the recordings' own drift, and it is what
    // `tuner/tests/calibration.rs` still closes the construction against.
    if let Some(pan_spread) = pan_spread.filter(|s| *s > 0.0) {
        builder = builder.pan_spread(pan_spread as f32);
    }
    if let Some((noise, _)) = noise {
        builder = builder.noise(noise);
    }
    let preset = builder.build()?;
    // The attribution goes in twice: as a comment, where a human reading the
    // file sees it, and in `description`, where it survives being loaded and
    // written back out by anything that does not preserve comments.
    let mut text = wrapped_comment(&credit);
    if !text.is_empty() {
        text.push('\n');
    }
    text.push_str(&preset.to_toml());
    std::fs::write(out, text)?;
    println!("\nwrote {out}");
    Ok(())
}

/// `text` as TOML comment lines, wrapped to something a person can read.
fn wrapped_comment(text: &str) -> String {
    const WIDTH: usize = 76;
    let mut out = String::new();
    for paragraph in text.lines() {
        let mut line = String::from("#");
        for word in paragraph.split_whitespace() {
            if line.len() + 1 + word.len() > WIDTH && line.len() > 1 {
                out.push_str(&line);
                out.push('\n');
                line = String::from("#");
            }
            line.push(' ');
            line.push_str(word);
        }
        out.push_str(&line);
        out.push('\n');
    }
    out
}

/// The per-note table: what was measured, against what the base preset had.
fn report_notes(survey: &Survey, base: &Preset, decay: &DecayConfig) {
    let factor = survey.vertical_factor(base);
    println!(
        "\n key  lay   f0 Hz    stretch      B       vs base       B4      bands    sigma0  sigma1   T60 f0 \
           base  T60 f4   detune  base   comb  width  written"
    );
    for note in &survey.notes {
        let index = key_index(note.key).expect("a surveyed key is on the keyboard");
        let base_b = f64::from(base.notes.inharmonicity_b[index]);
        // The pitch as reported, whether or not it is written: a note the
        // survey rejects is the one the reader most wants to see.
        let f0 = note
            .partial_hz(1)
            .map(|f1| f1 / (1.0 + note.inharmonicity_b().unwrap_or(base_b)).sqrt())
            .unwrap_or(f64::NAN);
        let b = note.inharmonicity_b().unwrap_or(f64::NAN);
        let curve = note.decay_curve(factor, decay);
        let base_curve = DecayCurve {
            sigma0: f64::from(base.notes.sigma0[index]),
            sigma1: f64::from(base.notes.sigma1[index]),
            residual: 0.0,
        };
        let show = |t: Option<f64>| t.map_or("-".to_string(), |t| format!("{t:.2}s"));
        println!(
            "{:>4} {:>4} {:>9.3} {:+7.2}c {:9.3e} {:+7.1}% {:>10} {:>10} {:>7} {:>7} {:>7} {:>6} \
             {:>7} {:>7} {:>6} {:>6} {:>6}  {}",
            note.key,
            note.layers.len(),
            f0,
            1200.0 * (f0 / equal_temperament(note.key)).log2(),
            b,
            100.0 * (b / base_b - 1.0),
            match note.inharmonicity_b4() {
                Some(b4) if b4 != 0.0 => format!("{b4:+.2e}"),
                Some(_) => "0".to_string(),
                None => "-".to_string(),
            },
            note.band_ratio().map_or("-".to_string(), |(ratio, sigmas)| {
                format!("{ratio:.2}/{sigmas:.1}s")
            }),
            curve.map_or("-".to_string(), |c| format!("{:.3}", c.sigma0)),
            curve.map_or("-".to_string(), |c| format!("{:.3}", c.sigma1)),
            show(curve.map(|c| c.t60_at(f0))),
            show(Some(base_curve.t60_at(f0))),
            show(curve.map(|c| c.t60_at(4.0 * f0))),
            note.detune_cents()
                .map_or("-".to_string(), |c| format!("{c:.2}c")),
            format!("{:.2}c", base.notes.detune_cents[index]),
            note.strike_position()
                .map_or("-".to_string(), |x| format!("{x:.3}")),
            // Measured on every survey and written only when asked for: the
            // width comes out of the same comb as the strike position and
            // carries the same microphone confound (`DECISIONS.md` 93, 130).
            note.contact_width()
                .map_or("-".to_string(), |w| format!("{w:.3}")),
            if note.tuning(base_b).is_some() { "yes" } else { "NO" },
        );
    }
    if let Some(split) = survey.polarization() {
        println!(
            "\npolarization: {:.2} dB, decay ratio {:.3} (base {:.2} dB, {:.3})",
            split.gain_db,
            split.decay_ratio,
            base.voicing.horizontal_gain_db,
            base.voicing.horizontal_decay_ratio
        );
    }
}

/// The per-string decay spread, note by note, and what it pools to.
///
/// Printed and not written: `voicing.unison_sigma_scale` is inert
/// (`DECISIONS.md` 225, 324). What it measures is still the recordings' own
/// drift, and the compass it prints it over is how a reader sees whether the
/// instrument's unisons hand over at all.
fn report_spread(survey: &Survey, base: &Preset, config: &SpreadConfig) {
    let notes = survey.spreads(base, config);
    println!("\n key  strings   detune   drift      spread");
    for note in &notes {
        println!(
            "{:>4} {:>8} {:>8} {:>7} {:>11}",
            note.key,
            note.strings,
            format!("{:.2}c", note.detune_cents),
            note.drift_cents()
                .map_or("-".to_string(), |c| format!("{c:.2}c")),
            match note.spread {
                Some(s) if note.saturated => format!("{s:.3} (sat)"),
                Some(s) => format!("{s:.3}"),
                None => "-".to_string(),
            },
        );
    }
    let pooled = SigmaSpread::pooled(&notes, config);
    println!(
        "\nunison sigma scale: {:?} (from {:?} notes; {:?} more drifted further than their own \
         unison and were not pooled)",
        pooled
            .rows()
            .iter()
            .map(|row| row.scale.clone())
            .collect::<Vec<_>>(),
        pooled.notes,
        pooled.saturated,
    );
}

/// The mechanism's own recordings: the `[noise]` section, measured.
fn report_mechanism(
    library: &piano_tuner::SampleLibrary,
    base: &Preset,
) -> Option<(piano_tuner::preset::NoiseTables, NoiseScreening)> {
    let config = NoiseConfig::default();
    let measurements = measure_mechanism(library, &config);
    if measurements.is_empty() {
        println!("\nno mechanism recordings in this library");
        return None;
    }
    let (fitted, screening) = fit_noise_screened(&measurements, &base.noise, &config);
    print_mechanism(&measurements, &fitted, &screening);
    Some((fitted, screening))
}

/// The measured table, the gate's verdict on it, and what was written.
///
/// Shared with `tools::noise`'s own mechanism stage so that both roads to the
/// same tables print the same evidence (`DECISIONS.md` 531).
pub fn print_mechanism(
    measurements: &piano_tuner::estimate::noise::MechanismMeasurements,
    fitted: &piano_tuner::preset::NoiseTables,
    screening: &NoiseScreening,
) {
    println!(
        "\n mechanism   key   re strike   decay to -40 dB   centroid   against   plausible"
    );
    let rows = [
        ("key_off", &measurements.key_off),
        ("pedal_down", &measurements.pedal_down),
        ("pedal_up", &measurements.pedal_up),
    ];
    for (name, metrics) in rows {
        for metric in metrics.iter() {
            println!(
                "{name:>10} {:>5} {:>11.1} {:>17.3} {:>10.0} {:>9} {:>11}",
                metric.key.map_or("-".to_string(), |k| k.to_string()),
                metric.level_db,
                metric.decay_s,
                metric.centroid_hz,
                metric.reference_key,
                if metric.level_db <= MAX_MECHANISM_LEVEL_DB {
                    "yes"
                } else {
                    "HOT"
                },
            );
        }
    }
    println!(
        "\nthe plausibility gate, at {MAX_MECHANISM_LEVEL_DB:.1} dB against the group's own \
         notes (DECISIONS.md 531):"
    );
    for (name, screen) in screening.events() {
        if !screen.recorded() {
            println!("  {name:>12}  not recorded by this library — inherited");
            continue;
        }
        println!(
            "  {name:>12}  {} of {} plausible, hottest {:+.2} dB — {}",
            screen.kept,
            screen.read,
            screen.hottest_db,
            if screen.accepted() {
                "written"
            } else {
                "REFUSED, inherited from the base preset"
            }
        );
    }
    println!(
        "\nkey-off: {} anchors, {:.0} Hz, {:.3} s, {:.1} dB of velocity",
        fitted.key_off.level_db.len(),
        fitted.key_off.centroid_hz,
        fitted.key_off.decay_s,
        fitted.key_off.velocity_db
    );
}

/// How far each note's stereo balance travels while it decays, and the
/// `voicing.polarization_pan_spread` that reproduces it.
///
/// The loudest layer of every sampled key, in stereo — which is the one thing
/// in the survey that cannot come from the trajectory cache, because the cache
/// holds the mono sum. One note per key, not sixteen: the drift is a property
/// of how the instrument radiates, and `docs/history/TUNING_REPORT.md` §5 measured it on the
/// loudest layer for the same reason (a soft note's high partials are in the
/// floor by 2 s and the floor has a balance of its own).
fn report_directivity(library: &piano_tuner::SampleLibrary, config: &SurveyConfig) -> Option<f64> {
    let directivity = DirectivityConfig::default();
    println!("\n key   partials   drift 0.3->2s");
    let mut drifts: Vec<f64> = Vec::new();
    for key in library.keys() {
        let Some(sample) = library.layers(key).last() else {
            continue;
        };
        let Ok(recording) = piano_tuner::audio::load_at(&sample.path, SAMPLE_RATE) else {
            continue;
        };
        if recording.channel_count() < 2 {
            continue;
        }
        let Ok(note_config) = config.note_config(equal_temperament(key)) else {
            continue;
        };
        match balance_drift(
            &recording.channels[0],
            &recording.channels[1],
            equal_temperament(key),
            f64::from(SAMPLE_RATE),
            &note_config,
            &directivity,
        ) {
            Ok(drift) => {
                println!(
                    "{key:>4} {:>10} {:>15.2}",
                    drift.partials, drift.drift_db
                );
                drifts.push(drift.drift_db);
            }
            Err(error) => println!("{key:>4}          -   {error}"),
        }
    }
    if drifts.is_empty() {
        return None;
    }
    drifts.sort_by(f64::total_cmp);
    let median = drifts[drifts.len() / 2];
    let spread = pan_spread_for_drift(median);
    println!(
        "\nstereo drift: {median:.2} dB median over {} keys -> polarization_pan_spread {spread:.3}{}",
        drifts.len(),
        if spread >= piano_tuner::estimate::directivity::MAX_PAN_SPREAD {
            " (the engine's ceiling; the instrument drifts further than it can reach)"
        } else {
            ""
        }
    );
    Some(spread)
}

/// The felt fit, note by note. Reported rather than written: the recording has
/// no newtons in it (see `survey`'s module header).
fn report_hammer(survey: &Survey, base: &Preset) -> Vec<HammerReport> {
    let config = HammerConfig::default();
    let mut reports = Vec::new();
    println!("\n key    felt p   base p     mass g   softest..loudest m/s   residual");
    for note in &survey.notes {
        match survey.hammer(note.key, base, &config) {
            Ok(report) => {
                let index = key_index(note.key).expect("a surveyed key");
                let speeds = &report.fit.velocities;
                println!(
                    "{:>4}  {:>7.3}  {:>7.3}  {:>9.2}  {:>9.3}..{:<9.3}  {:>6.2} dB",
                    note.key,
                    report.fit.felt.exponent,
                    base.notes.hammer_exponent[index],
                    1000.0 * report.fit.felt.mass,
                    speeds.first().copied().unwrap_or(f64::NAN),
                    speeds.last().copied().unwrap_or(f64::NAN),
                    report.fit.residual_db
                );
                reports.push(report);
            }
            Err(error) => println!("{:>4}  no felt fit: {error}", note.key),
        }
    }
    reports
}

fn value<'a>(args: &'a [String], i: &mut usize, flag: &str) -> Result<&'a str> {
    *i += 1;
    args.get(*i)
        .map(String::as_str)
        .ok_or_else(|| Error::Config(format!("{flag} needs a value")))
}

fn parse<T: std::str::FromStr>(text: &str, flag: &str) -> Result<T> {
    text.parse()
        .map_err(|_| Error::Config(format!("{flag}: cannot parse {text:?}")))
}
