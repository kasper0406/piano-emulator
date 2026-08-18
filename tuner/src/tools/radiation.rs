//! `piano-tuner radiation` — fits `[soundboard.radiation]`, the strings'
//! radiated response *between* their partials, and writes it into a preset.
//!
//! ```sh
//! cargo run --release -p piano-tuner -- radiation \
//!     data/salamander presets/salamander-c5.toml --out presets/salamander-c5.toml
//! ```
//!
//! # What is being fitted, and why it is not the recording's mono
//!
//! `DECISIONS.md` 407-411 is the whole derivation and 410 states the target in
//! one sentence: **the recording's pooled level-matched mono share divided by
//! the pair's own mono transfer**, never the recording's mono directly.
//!
//! The reason is the defect itself. The recording is a spaced pair straddling a
//! nodal line of the soundboard between about 175 and 300 Hz; its two capsules
//! carry energy its own mono sum does not, and item 407(b) measured how much —
//! the recording's pooled pair-over-mono reads **+9.4 dB at 180 Hz**. So the
//! recording's mono sum is *not* what the piano radiated: it is what the piano
//! radiated, minus what the fold-down cancelled. Every fit in this repository
//! that scored a source against it has therefore been fitting the hole into the
//! source, which is exactly what item 407 found `notes.partial_gains` had done.
//!
//! Both halves are measurable and [`piano_tuner::realism::mono_columns`]
//! prints them side by side:
//!
//! | column | what it is |
//! |---|---|
//! | `required` | `REF pair − ENG pair`, the pair's own mono transfer |
//! | `standing` | pooled level-matched `ENG mono − REF mono`, where the source is |
//! | `deficit` | `required − standing`, what this fit owes the band |
//!
//! The acceptance is item 411's, in item 408's words: **the standing column
//! rises to meet the required column**. It is not any per-channel board and it
//! is not the fold-down landing on the recording's mono — after this fit the
//! engine's mono deliberately stands *above* the recording's by the required
//! column, because that is the headroom the nodal mechanism of item 406(a) then
//! spends. Item 411's ordering rule is the whole point: the rotation is the
//! second half of a two-milestone repair and this is the first.
//!
//! # The loop
//!
//! One round is: render the thirty recorded keys, measure the table, add the
//! deficit to the declared curve, subtract the offset that keeps the engine's
//! own 100-810 Hz energy where it was, write, repeat.
//!
//! The offset is not cosmetic and it is not free choice. The pooled statistic
//! is a *share* — every take is normalised on its own 100-810 Hz total — so
//! adding a constant to the whole curve moves no column at all: `standing` after
//! a colouration `c` is `standing + c_i − 10 log10 Σ s_j 10^(c_j/10)`, which is
//! invariant under `c → c + a`. The shape is determined by the target and the
//! level is not, so the level is *chosen*, and it is chosen to be the one that
//! leaves the engine's span energy alone — `Σ s_j 10^(c_j/10) = 1`, with `s`
//! the engine's own pooled band shares. Any other choice would move every
//! loudness board in the repository for no measured reason.
//!
//! The loop is closed on the engine's own render rather than on the filter
//! design, so what converges is what the instrument does and not what the
//! cascade was asked for.

use std::path::{Path, PathBuf};

use rayon::prelude::*;

use piano_emulator::preset::{MicVoicing, Preset, Radiation};
use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_emulator::types::Event;
use piano_tuner::realism::{self, MonoBands, MonoColumn};
use piano_tuner::{Audio, SampleLibrary, SAMPLE_RATE};

/// The velocity every stereo board and every forensic instrument in this
/// milestone reads, so this fit reads it too.
const FIT_VELOCITY: u8 = 90;
/// Seconds of note kept, matching `tuner/tests/stereo.rs` and
/// `mono_mechanism`.
const RENDER_S: f64 = 3.0;
const PREROLL: usize = realism::STEREO_PREROLL_SAMPLES;
const PREROLL_S: f64 = PREROLL as f64 / 48_000.0;

/// Rounds the closed-on-render loop takes unless it converges first.
const DEFAULT_ROUNDS: usize = 8;
/// The loop stops when every band's deficit is inside this, dB.
const DEFAULT_TOLERANCE: f64 = 0.25;
/// How much of each round's deficit is applied. One is the plain fixed-point
/// iteration; the flag exists because a loop that oscillates should be damped
/// rather than stopped early.
const DEFAULT_DAMPING: f64 = 1.0;

static ENGINE_CACHE: std::sync::OnceLock<piano_tuner::renders::EngineRenders> =
    std::sync::OnceLock::new();

fn render_engine(preset: &Preset, key: u8) -> Audio {
    if let Some(cache) = ENGINE_CACHE.get() {
        return cache.note(
            preset,
            piano_tuner::renders::NoteSpec::new(key, FIT_VELOCITY, PREROLL_S + RENDER_S, PREROLL),
        );
    }
    let events = [RenderEvent::new(
        PREROLL_S as f32,
        Event::NoteOn {
            key,
            vel: u16::from(FIT_VELOCITY),
        },
    )];
    let (left, right) = render_to_buffer(preset, &events, (PREROLL_S + RENDER_S) as f32);
    Audio::new(
        SAMPLE_RATE,
        vec![left[PREROLL..].to_vec(), right[PREROLL..].to_vec()],
    )
    .expect("the engine renders stereo")
}

/// The recording of one key, trimmed and cached with `tuner/tests/stereo.rs`'s
/// own fingerprint so the fit and the gate share one set of files on disk.
fn render_reference(
    data: &Path,
    sfz: &Path,
    key: u8,
    velocity: u8,
) -> Result<Audio, piano_tuner::Error> {
    use piano_tuner::cache;
    use piano_tuner::sampler::SAMPLER_VERSION;
    use piano_tuner::{Sampler, SamplerEvent, TimedEvent};

    let mut print = cache::Fingerprint::new();
    print
        .str("tests/stereo/reference")
        .u64(u64::from(SAMPLER_VERSION))
        .file(sfz)?
        .u64(u64::from(SAMPLE_RATE))
        .u64(u64::from(key))
        .u64(u64::from(velocity))
        .f64(RENDER_S);
    let path = cache::reference_dir(data).join(format!(
        "stereo-key{key:03}-v{velocity:03}-{}.wav",
        print.hex()
    ));
    cache::audio(&path, || {
        let mut sampler = Sampler::new(sfz)?;
        let events = [TimedEvent::new(
            0.0,
            SamplerEvent::NoteOn { key, vel: velocity },
        )];
        let rendered = sampler.render(&events, RENDER_S + 0.2)?;
        let mono = rendered.mono();
        let onset = piano_tuner::detect_onset(&mono, f64::from(SAMPLE_RATE));
        let skip = (onset * f64::from(SAMPLE_RATE)).round() as usize;
        let frames = (RENDER_S * f64::from(SAMPLE_RATE)) as usize;
        let channels: Vec<Vec<f32>> = rendered
            .channels
            .iter()
            .map(|c| {
                (0..frames)
                    .map(|n| c.get(skip + n).copied().unwrap_or(0.0))
                    .collect()
            })
            .collect();
        Audio::new(SAMPLE_RATE, channels)
    })
}

/// The instrument the table is measured on: the preset with the mode-controlled
/// band deleted.
///
/// It is the **bare** engine on both sides of `required` for the reason item
/// 405 gives — the lobe is the thing a nodal mechanism replaces, so the
/// headroom a mechanism has is measured against a pair that does not already
/// have one — and it changes nothing about `standing`, because the lobe leaves
/// `(L + R)/2` bit-identical at every setting.
fn bare(preset: &Preset) -> Preset {
    let mut p = preset.clone();
    if let Some(mics) = preset.voicing.mics {
        p.voicing.mics = Some(MicVoicing { modal: None, ..mics });
    }
    p
}

/// The curve after one round: `declared + damping · deficit`, offset so the
/// engine's own span energy is where it was. See the module header.
fn advance(declared: &[f32], columns: &[MonoColumn], damping: f64) -> Vec<f32> {
    let step: Vec<f64> = columns.iter().map(|c| damping * c.deficit_db()).collect();
    let scale: f64 = columns
        .iter()
        .zip(&step)
        .map(|(c, s)| c.engine_share * 10.0f64.powf(s / 10.0))
        .sum();
    let offset = 10.0 * scale.log10();
    declared
        .iter()
        .zip(&step)
        .map(|(&d, &s)| {
            (f64::from(d) + s - offset).clamp(
                f64::from(piano_emulator::preset::MIN_RADIATION_GAIN_DB),
                f64::from(piano_emulator::preset::MAX_RADIATION_GAIN_DB),
            ) as f32
        })
        .collect()
}

/// **Where the fitted span sits against the rest of the instrument.**
///
/// The fit's statistic is a share inside 100-810 Hz, and a share has no uniform
/// component: `Σ s = Σ r = 1` on both sides, so a *constant* standing column
/// must be zero, while the required column's own share-weighted mean is not.
/// The two can therefore never be equal, and what the loop can converge to is
/// `standing = required − K` with the same `K` in every band. `K` is the one
/// degree of freedom of the curve this statistic cannot see, and this is the
/// measurement that decides it instead of taste: the span's own level against
/// the whole take, engine less recording, read on the mono sum **and** on the
/// pair average.
///
/// The pair average is the one that matters. It is what the two capsules
/// carry — the best proxy either side has for what was radiated — and it is
/// blind to the fold-down. If the engine's span already carries the recording's
/// share of pair energy, then the source's *level* over the span is right, `K`
/// is not a source defect at all, and adding it would move every loudness board
/// in the repository for no reason a measurement asked for.
fn span_level(takes: &[(MonoBands, MonoBands)]) -> (f64, f64) {
    let (mut rp, mut rm, mut ep, mut em) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (reference, engine) in takes {
        let (p, m) = reference.span_over_take();
        rp += p;
        rm += m;
        let (p, m) = engine.span_over_take();
        ep += p;
        em += m;
    }
    (10.0 * (ep / rp).log10(), 10.0 * (em / rm).log10())
}

fn report(round: usize, grid: &[f64], columns: &[MonoColumn], declared: &[f32]) {
    println!(
        "\n### round {round} — item 408's table on this render\n\n\
         | Hz | REF pair | ENG pair | **required** | **standing** | deficit | declared dB |"
    );
    println!("|---:|---:|---:|---:|---:|---:|---:|");
    for (i, c) in columns.iter().enumerate() {
        println!(
            "| {:.0} | {:+.2} | {:+.2} | **{:+.2}** | **{:+.2}** | {:+.2} | {:+.2} |",
            grid[i],
            c.reference_pair_db,
            c.engine_pair_db,
            c.required_db,
            c.standing_db,
            c.deficit_db(),
            declared[i]
        );
    }
    // The uniform component the statistic cannot see is reported separately
    // from the shape it can: `K` is the share-weighted mean of the deficit and
    // the spread is what is left after it is taken out — the number this fit is
    // actually converging.
    let (k, spread) = uniform_and_spread(columns);
    let worst = columns
        .iter()
        .map(|c| c.deficit_db().abs())
        .fold(0.0f64, f64::max);
    println!(
        "\nworst deficit **{worst:.2} dB**; uniform component **K = {k:+.2} dB**, \
         shape residual (peak-to-peak about K) **{spread:.3} dB**"
    );
}

/// The deficit column split into the part a source colouration can move and
/// the part it cannot: `K`, the engine-share-weighted mean, and the
/// peak-to-peak spread of `deficit − K` over the bands.
fn uniform_and_spread(columns: &[MonoColumn]) -> (f64, f64) {
    let k: f64 = columns
        .iter()
        .map(|c| c.engine_share * c.deficit_db())
        .sum();
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for c in columns {
        let d = c.deficit_db() - k;
        lo = lo.min(d);
        hi = hi.max(d);
    }
    (k, hi - lo)
}

/// **Hold the partials where the per-key fit put them.**
///
/// The colouration is a response on the whole drive, so it moves a key's
/// partials exactly as much as it moves the floor between them — and item 410
/// asked for the floor. This divides `notes.partial_gains` by the realised
/// response at each partial's own frequency, which leaves every fitted partial
/// where `piano-tuner fit --stage partial_gains` put it and lets the
/// colouration act only where no partial is.
///
/// It is a flag rather than the default because the two readings are different
/// questions and both are owed. Without it the fit is item 410's as written and
/// the whole band moves; with it, what moves is only the part of the band no
/// per-partial table can reach, which is the part item 408 charged the deficit
/// to — and how far *that* reaches is the measurement.
#[must_use = "the count of railed gains is the measurement this variant takes"]
fn hold_partials(preset: &mut Preset, curve: &Radiation) -> usize {
    let f0 = preset.notes.f0_hz.clone();
    let b = preset.notes.inharmonicity_b.clone();
    let mut hz = Vec::new();
    for (k, row) in preset.notes.partial_gains.iter().enumerate() {
        for i in 0..row.len() {
            let n = (i + 1) as f32;
            hz.push(f64::from(f0[k] * n * (1.0 + b[k] * n * n).sqrt()));
        }
    }
    let response = piano_emulator::soundboard::radiation_response_db(curve, &hz);
    let mut at = 0;
    let mut railed = 0;
    for row in preset.notes.partial_gains.iter_mut() {
        for g in row.iter_mut() {
            let wanted = *g * 10.0f32.powf(-(response[at] as f32) / 20.0);
            // The schema's own rail on a per-partial gain, which this
            // compensation is entitled to reach and not to break. A partial
            // that is already near the floor cannot be pushed further down to
            // make room for a colouration, and how many of them there are is
            // the measurement this variant exists to take.
            let held = wanted.clamp(
                piano_emulator::string::MIN_PARTIAL_GAIN,
                piano_emulator::string::MAX_PARTIAL_GAIN,
            );
            if held != wanted {
                railed += 1;
            }
            *g = held;
            at += 1;
        }
    }
    railed
}

struct Options {
    data: PathBuf,
    preset: PathBuf,
    out: Option<PathBuf>,
    hold_partials: bool,
    rounds: usize,
    tolerance: f64,
    damping: f64,
}

fn parse(args: Vec<String>) -> Result<Options, Box<dyn std::error::Error>> {
    let mut positional: Vec<String> = Vec::new();
    let mut out = None;
    let mut rounds = DEFAULT_ROUNDS;
    let mut tolerance = DEFAULT_TOLERANCE;
    let mut damping = DEFAULT_DAMPING;
    let mut hold = false;
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" => out = Some(PathBuf::from(args.next().ok_or("--out needs a path")?)),
            "--hold-partials" => hold = true,
            "--rounds" => rounds = args.next().ok_or("--rounds needs a number")?.parse()?,
            "--tolerance" => tolerance = args.next().ok_or("--tolerance needs a number")?.parse()?,
            "--damping" => damping = args.next().ok_or("--damping needs a number")?.parse()?,
            other if other.starts_with("--") => return Err(format!("unknown flag {other}").into()),
            other => positional.push(other.to_string()),
        }
    }
    if positional.len() != 2 {
        return Err("usage: radiation <data-dir> <preset.toml> [--out <file>] \
                    [--hold-partials] [--rounds n] [--tolerance dB] [--damping f]"
            .into());
    }
    Ok(Options {
        data: PathBuf::from(&positional[0]),
        preset: PathBuf::from(&positional[1]),
        out,
        hold_partials: hold,
        rounds,
        tolerance,
        damping,
    })
}

pub fn run(args: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
    let options = parse(args)?;
    let sfz = options
        .data
        .join("SalamanderGrandPiano-V3+20200602.sfz");
    if !sfz.exists() {
        eprintln!(
            "the reference piano is not here: {}\nrun data/fetch_salamander.sh first (707 MiB).",
            sfz.display()
        );
        std::process::exit(2);
    }
    let _ = ENGINE_CACHE.set(piano_tuner::renders::EngineRenders::at_data_root(&options.data));

    let grid = realism::mono_grid();
    let library = SampleLibrary::from_sfz(&sfz)?;
    let recorded = realism::RecordedKeys::from_library(&library)?;
    let keys: Vec<u8> = recorded.keys().to_vec();

    println!(
        "# `piano-tuner radiation`\n\n{} recorded keys at v{FIT_VELOCITY}, {} sixth-octave bands \
         over {:.0}-{:.0} Hz, base {}",
        keys.len(),
        grid.len(),
        realism::MONO_SPAN_HZ.0,
        realism::MONO_SPAN_HZ.1,
        options.preset.display()
    );

    // The recording's side, measured once: it does not move when the engine
    // does.
    let reference: Vec<MonoBands> = keys
        .par_iter()
        .map(|&key| -> Result<MonoBands, piano_tuner::Error> {
            let take = render_reference(&options.data, &sfz, key, FIT_VELOCITY)?;
            Ok(MonoBands::of(
                &take.channels[0],
                &take.channels[1],
                f64::from(SAMPLE_RATE),
                &grid,
            ))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut preset = Preset::load(&options.preset)?;
    let mut declared: Vec<f32> = match &preset.soundboard.radiation {
        Some(r) if r.hz.len() == grid.len() => r.gain_db.clone(),
        Some(r) => {
            return Err(format!(
                "the preset declares {} radiation bands and this fit works on {}; \
                 delete the section or re-fit it on the same grid",
                r.hz.len(),
                grid.len()
            )
            .into())
        }
        None => vec![0.0; grid.len()],
    };
    let centres: Vec<f32> = grid.iter().map(|&hz| hz as f32).collect();

    let mut converged = None;
    let mut last: Vec<MonoColumn> = Vec::new();
    let mut measured: Vec<(MonoBands, MonoBands)> = Vec::new();
    let railed;
    for round in 0..=options.rounds {
        let mut candidate = preset.clone();
        candidate.soundboard.radiation = if declared.iter().all(|&g| g == 0.0) && round == 0 {
            preset.soundboard.radiation.clone()
        } else {
            Some(Radiation {
                hz: centres.clone(),
                gain_db: declared.clone(),
            })
        };
        if options.hold_partials {
            if let Some(curve) = candidate.soundboard.radiation.clone() {
                let _ = hold_partials(&mut candidate, &curve);
            }
        }
        candidate.validate()?;
        let probe = bare(&candidate);
        let engine: Vec<MonoBands> = keys
            .par_iter()
            .map(|&key| {
                let take = render_engine(&probe, key);
                MonoBands::of(
                    &take.channels[0],
                    &take.channels[1],
                    f64::from(SAMPLE_RATE),
                    &grid,
                )
            })
            .collect();
        measured = reference.iter().cloned().zip(engine).collect();
        let columns = realism::mono_columns(&grid, &measured);
        report(round, &grid, &columns, &declared);
        let (_, spread) = uniform_and_spread(&columns);
        last = columns.clone();
        if spread <= options.tolerance {
            converged = Some(round);
            break;
        }
        if round == options.rounds {
            break;
        }
        declared = advance(&declared, &columns, options.damping);
    }

    preset.soundboard.radiation = Some(Radiation {
        hz: centres,
        gain_db: declared.clone(),
    });
    if options.hold_partials {
        let curve = preset.soundboard.radiation.clone().expect("just written");
        railed = hold_partials(&mut preset, &curve);
        let total: usize = preset.notes.partial_gains.iter().map(Vec::len).sum();
        println!(
            "\n**`--hold-partials`**: {railed} of {total} fitted per-partial gains were pushed \
             through the schema's own rail by the compensation and are clamped there."
        );
    }
    preset.validate()?;

    // What the instrument will actually do with the curve, read at the declared
    // centres and at the points *between* them — the ripple the band-integrated
    // statistic the loop closes on is blind to by construction.
    let curve = preset
        .soundboard
        .radiation
        .clone()
        .expect("the curve was just written");
    let fine: Vec<f64> = {
        let ratio = 2.0f64.powf(1.0 / 24.0);
        let mut hz = grid[0] / ratio;
        let mut out = Vec::new();
        while hz <= grid[grid.len() - 1] * ratio {
            out.push(hz);
            hz *= ratio;
        }
        out
    };
    let realised_fine = piano_emulator::soundboard::radiation_response_db(&curve, &fine);
    let realised = piano_emulator::soundboard::radiation_response_db(&curve, &grid);
    println!("\n## the fitted curve, and what the cascade realises\n\n| Hz | declared dB | realised dB |");
    println!("|---:|---:|---:|");
    for ((hz, g), r) in grid.iter().zip(&declared).zip(&realised) {
        println!("| {hz:.0} | {g:+.2} | {r:+.2} |");
    }
    let ripple = fine
        .iter()
        .zip(&realised_fine)
        .fold(f64::NEG_INFINITY, |m, (_, &r)| m.max(r))
        - fine
            .iter()
            .zip(&realised_fine)
            .fold(f64::INFINITY, |m, (_, &r)| m.min(r));
    println!(
        "\nRealised at 1/24 octave over the same span: {:.2} dB peak-to-peak.",
        ripple
    );
    print!("  ");
    for (hz, r) in fine.iter().zip(&realised_fine) {
        print!("{hz:.0}:{r:+.1}  ");
    }
    println!();
    let (k, spread) = uniform_and_spread(&last);
    let (pair_level, mono_level) = span_level(&measured);
    println!(
        "\n## the uniform component, and what it is\n\n\
         The shape converged to a **flat** deficit column: `standing = required − K` with \
         **K = {k:+.2} dB** and a peak-to-peak spread of **{spread:.3} dB** across the nineteen \
         bands. `K` is invisible to the fit's own statistic — a share has no uniform component — \
         so it is decided here instead:\n\n\
         * the span's **pair average** against the whole take, engine less recording: \
         **{pair_level:+.2} dB**\n\
         * the span's **mono sum** against the whole take, engine less recording: \
         **{mono_level:+.2} dB**\n\n\
         The pair average is what the capsules carry and is the closest either side has to what \
         was radiated. A source colouration is what sets it."
    );
    match converged {
        Some(round) => println!(
            "\n**Converged after {round} rounds**: the shape residual is inside {:.2} dB.",
            options.tolerance
        ),
        None => println!(
            "\n**Not converged in {} rounds.** Shape residual {spread:.3} dB.",
            options.rounds
        ),
    }
    println!("engine render cache: {}", piano_tuner::renders::stats_line());

    if let Some(out) = &options.out {
        preset.save(out)?;
        println!("wrote {}", out.display());
    } else {
        println!("(no --out: nothing was written)");
    }
    Ok(())
}
