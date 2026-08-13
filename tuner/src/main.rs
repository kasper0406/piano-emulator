//! The tuner's driver: `track` runs the partial tracker over one recording,
//! `estimate` runs the whole per-note analysis on it and, given a base preset,
//! writes the estimates out as a preset file, and `survey` does that for every
//! note of a whole sample library at once.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use piano_tuner::estimate::decay::{DecayConfig, DecayCurve};
use piano_tuner::estimate::hammer::HammerConfig;
use piano_tuner::pipeline::{analyze_note, NoteConfig};
use piano_tuner::preset::{equal_temperament, key_index, Preset, PresetBuilder};
use piano_tuner::survey::{pooled_velocity_map, HammerReport, Survey, SurveyConfig};
use piano_tuner::{
    audio, cents, Error, InharmonicModel, PartialTracker, Result, SampleLibrary, StftConfig,
    TrackerConfig, SAMPLE_RATE,
};

const USAGE: &str = "\
piano-tuner — offline analysis for piano-emulator parameter estimation

usage:
  piano-tuner track <input.wav|input.flac> --f0 <hz> [options]
  piano-tuner estimate <input.wav|input.flac> --f0 <hz> [options]
  piano-tuner survey <instrument.sfz> --preset <base.toml> [options]

options:
  --f0 <hz>          fundamental of the recorded note (required)
  --b <coefficient>  inharmonicity B used to seed the partial search [0]
  --partials <n>     highest partial index to look for [80]
  --window <n>       analysis window in samples [65536]
  --hop <n>          hop in samples [480]
  --pad <n>          zero-pad the transform to n times the window [2]
  --out <file>       track: write the trajectories as JSON
                     estimate/survey: write the preset (needs --preset)

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

fn run(args: Vec<String>) -> Result<()> {
    match args.first().map(String::as_str) {
        Some("track") => track(&args[1..]),
        Some("estimate") => estimate(&args[1..]),
        Some("survey") => survey(&args[1..]),
        Some("help") | Some("--help") | Some("-h") | None => {
            print!("{USAGE}");
            Ok(())
        }
        Some(other) => Err(Error::Config(format!("unknown command {other:?}\n\n{USAGE}"))),
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
    let mut builder = survey.builder(base, &decay).description(format!(
        "estimated by piano-tuner from {}{}{credit}",
        options.sfz,
        if credit.is_empty() { "" } else { "; " }
    ));
    if let Some(name) = &options.name {
        builder = builder.name(name.clone());
    }
    if options.velocity_map {
        builder = builder.velocity_map(
            map.ok_or_else(|| Error::Estimate("no velocity map was fitted".into()))?,
        );
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
        "\n key  lay   f0 Hz    stretch      B       vs base   sigma0  sigma1   T60 f0   base  \
          T60 f4   detune  base   comb  written"
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
            "{:>4} {:>4} {:>9.3} {:+7.2}c {:9.3e} {:+7.1}% {:>7} {:>7} {:>7} {:>6} {:>7} {:>7} \
             {:>6} {:>6}  {}",
            note.key,
            note.layers.len(),
            f0,
            1200.0 * (f0 / equal_temperament(note.key)).log2(),
            b,
            100.0 * (b / base_b - 1.0),
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
