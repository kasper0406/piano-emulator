//! Stage 2, the mechanism's balance: how loud the hammer is against the note it
//! belongs to, engine against the recording of the same key, and the level of
//! `[noise.strike]` that puts the two on top of each other.
//!
//! `DECISIONS.md` 338-341. `estimate::attack` fits this event's *colour* and a
//! level referenced to the note's **peak**; `fit --stage partials` then corrects
//! that level on the engine's own render. Neither closes on a ratio, and a ratio
//! is what a listener hears — so anything that moves the attack's tonal content
//! without moving the note's peak moves the balance and nothing in the factory
//! notices. That is what happened between the milestone this event was fitted in
//! and the one that found it: the event never moved and the instrument around it
//! did.
//!
//! # What is measured
//!
//! Per recorded key and per velocity, on isolated notes:
//!
//! * the **recording's** attack tonality
//!   ([`piano_tuner::estimate::attack::noise_to_tone_db`]) — the arithmetic over
//!   the geometric mean of the power spectrum of the first 30 ms from its own
//!   onset, which is a noise-to-tone ratio needing no level match;
//! * the **engine's**, as the preset ships;
//! * the **engine's with the event silenced** — the tonal attack alone;
//! * the offset on the event's level that puts the second on the first.
//!
//! **Recorded keys only** (`DECISIONS.md` 328): every row is the engine at a key
//! against a recording *of that key*. The transposed keys are still played by
//! everything else in the repository and are not scored here.
//!
//! # Why the inversion is exact
//!
//! Two renders per note — with the event and without it — and the sample-wise
//! difference **is** the event, through the board, the master gain and its own
//! filters. Every other level of it is then
//! [`mix`](piano_tuner::estimate::attack::mix) and no render is repeated. So the
//! answer is not a search over presets and not a prediction: it is the same
//! output-referenced inversion `CombLine`, the damper line and `strike_offset`
//! are (`DECISIONS.md` 199, 203, 211), with the estimator between the render and
//! the number removed.
//!
//! # What it writes
//!
//! `--out <file>` applies the fitted correction to `[noise.strike]`: the level
//! at the nominal drive to every anchor, and the slope in drive to
//! `velocity_db`. Those are the event's only two level fields, and they are
//! exactly what a line through the per-note offsets has.
//!
//! **It is re-entrant.** The correction is measured on whatever preset it is
//! given, so running it over its own output measures a corrected instrument and
//! asks for nothing more; `estimate::attack`'s `the_balance_is_a_fixed_point`
//! gates the arithmetic and the second pass over the shipped preset returns
//! −0.00 dB on both fields.
//!
//! ```sh
//! cargo run --release -p piano-tuner -- noise \
//!     [data/salamander] [presets/salamander-c5.toml] [--out <f>] [--key <n>]
//! ```

use std::cell::RefCell;
use std::path::{Path, PathBuf};

use rayon::prelude::*;

use piano_emulator::preset::{NoiseAnchor, Preset, SILENT_LEVEL_DB};
use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_emulator::types::Event;
use piano_tuner::cache;
use piano_tuner::estimate::attack::{
    balance_reading, fit_balance, BalanceReading, BalanceVerdict,
};
use piano_tuner::estimate::melody::note_onset;
use piano_tuner::realism::RecordedKeys;
use piano_tuner::sampler::SAMPLER_VERSION;
use piano_tuner::{Audio, SampleLibrary, Sampler, TimedEvent, SAMPLE_RATE};

/// The velocities the balance is read at.
///
/// Five, spanning the library's own range, because the correction has a **slope
/// in drive** and a slope needs more than the nominal point: measured on the
/// preset this tool was written for, the offset the balance asks for runs from
/// −17 dB at velocity 24 to −3 dB at velocity 110, which is a velocity law and
/// not a level. They are MIDI velocities rather than layer indices so that the
/// same list means the same thing on a library with a different number of
/// layers.
pub const VELOCITIES: [u8; 5] = [24, 48, 72, 88, 110];

/// Seconds of note rendered, and how long the key is held.
const HOLD_S: f64 = 0.5;
const RENDER_S: f64 = 0.8;
const PREROLL_S: f64 = 0.05;
const SR: f64 = SAMPLE_RATE as f64;

/// Fewest readings a correction may be fitted from.
///
/// Thirty is one velocity at every recorded key, or five velocities at six of
/// them; under it the line is being drawn through a register rather than
/// through the compass.
const MIN_READINGS: usize = 30;

const MAX_CACHED_BUFFERS: usize = 8;

thread_local! {
    static SAMPLER: RefCell<Option<Sampler>> = const { RefCell::new(None) };
}

fn with_sampler<T>(
    sfz: &Path,
    body: impl FnOnce(&mut Sampler) -> Result<T, piano_tuner::Error>,
) -> Result<T, piano_tuner::Error> {
    SAMPLER.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            *slot = Some(Sampler::new(sfz)?);
        }
        let sampler = slot.as_mut().expect("a player was just built");
        let out = body(sampler);
        if sampler.cached_buffers() > MAX_CACHED_BUFFERS {
            sampler.clear_cache();
        }
        out
    })
}

/// The same preset with `[noise.strike]` silenced outright.
pub fn without_strike(preset: &Preset) -> Preset {
    let mut out = preset.clone();
    out.noise.strike.level_db = vec![NoiseAnchor {
        key: 21,
        db: SILENT_LEVEL_DB,
    }];
    out
}

fn render_engine(preset: &Preset, key: u8, vel: u8) -> Vec<f32> {
    let events = [
        RenderEvent::new(PREROLL_S as f32, Event::NoteOn { key, vel }),
        RenderEvent::new(
            (PREROLL_S + HOLD_S) as f32,
            Event::NoteOff { key, vel: 64 },
        ),
    ];
    let (left, right) = render_to_buffer(preset, &events, RENDER_S as f32);
    left.iter().zip(&right).map(|(&l, &r)| 0.5 * (l + r)).collect()
}

fn render_reference(
    sampler: &mut Sampler,
    key: u8,
    vel: u8,
) -> Result<Audio, piano_tuner::Error> {
    let events = TimedEvent::note(PREROLL_S, key, vel, HOLD_S);
    sampler.render(&events, RENDER_S)
}

pub fn run(args: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
    let positional: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    let data = PathBuf::from(
        positional
            .first()
            .map(|s| s.as_str())
            .unwrap_or("data/salamander"),
    );
    let preset_path = PathBuf::from(
        positional
            .get(1)
            .map(|s| s.as_str())
            .unwrap_or("presets/salamander-c5.toml"),
    );
    let mut out: Option<PathBuf> = None;
    let mut only: Option<Vec<u8>> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => {
                out = Some(PathBuf::from(&args[i + 1]));
                i += 1;
            }
            "--key" => {
                only = Some(vec![args[i + 1].parse()?]);
                i += 1;
            }
            _ => {}
        }
        i += 1;
    }

    let sfz = data.join("SalamanderGrandPiano-V3+20200602.sfz");
    if !sfz.exists() {
        eprintln!(
            "the reference piano is not here: {}\nrun data/fetch_salamander.sh first (707 MiB).",
            sfz.display()
        );
        std::process::exit(2);
    }
    if out.as_deref() == Some(preset_path.as_path()) {
        return Err("--out may not be the preset being measured".into());
    }

    let preset = Preset::load(&preset_path)?;
    let quiet = without_strike(&preset);
    let library = SampleLibrary::from_sfz(&sfz)?;
    let recorded = RecordedKeys::from_library(&library)?;
    let keys: Vec<u8> = match &only {
        Some(k) => k.clone(),
        None => recorded.keys().to_vec(),
    };

    let reference_cache = cache::reference_dir(&data);
    let mut reference_key = cache::Fingerprint::new();
    reference_key
        .str("noise-balance/reference")
        .u64(u64::from(SAMPLER_VERSION))
        .file(&sfz)?
        .u64(u64::from(SAMPLE_RATE))
        .f64(RENDER_S)
        .f64(HOLD_S);

    println!(
        "the mechanism's balance: {} recorded keys x {} velocities, engine on {}",
        keys.len(),
        VELOCITIES.len(),
        preset_path.display()
    );
    println!(
        "  [noise.strike] as it stands: level {:+.2} .. {:+.2} dB over {} anchors, \
         velocity_db {:.2}, centroid {:.0} Hz, band {:.0} Hz, decay {:.3} s",
        preset
            .noise
            .strike
            .level_db
            .iter()
            .map(|a| a.db)
            .fold(f32::INFINITY, f32::min),
        preset
            .noise
            .strike
            .level_db
            .iter()
            .map(|a| a.db)
            .fold(f32::NEG_INFINITY, f32::max),
        preset.noise.strike.level_db.len(),
        preset.noise.strike.velocity_db,
        preset.noise.strike.centroid_hz,
        preset.noise.strike.bandwidth_hz,
        preset.noise.strike.decay_s,
    );

    let cells: Vec<(u8, u8)> = keys
        .iter()
        .flat_map(|&key| VELOCITIES.iter().map(move |&vel| (key, vel)))
        .collect();
    let readings: Vec<BalanceReading> = cells
        .par_iter()
        .map(|&(key, vel)| -> Result<BalanceReading, piano_tuner::Error> {
            let engine = render_engine(&preset, key, vel);
            let tone = render_engine(&quiet, key, vel);
            let burst: Vec<f32> = engine.iter().zip(&tone).map(|(&a, &b)| a - b).collect();
            let mut cell_print = reference_key;
            cell_print.u64(u64::from(key)).u64(u64::from(vel));
            let path = reference_cache.join(format!(
                "balance-key{key:03}-v{vel:03}-{}.wav",
                cell_print.hex()
            ));
            let reference = cache::audio(&path, || {
                with_sampler(&sfz, |s| render_reference(s, key, vel))
            })?;
            let reference = reference.mono();
            Ok(balance_reading(
                key,
                vel,
                &reference,
                note_onset(&reference, SR, PREROLL_S),
                &tone,
                &burst,
                note_onset(&engine, SR, PREROLL_S),
                SR,
            ))
        })
        .collect::<Result<Vec<_>, _>>()?;

    println!(
        "\nattack tonality of the first 30 ms, dB — a line spectrum is large, a continuum is zero\n"
    );
    println!(
        "{:>4} {:>4} {:>9} {:>9} {:>9} {:>9} {:>10}",
        "key", "vel", "reference", "engine", "tone only", "eng-ref", "offset"
    );
    for r in &readings {
        println!(
            "{:>4} {:>4} {:>9.2} {:>9.2} {:>9.2} {:>+9.2} {:>10}",
            r.key,
            r.midi_velocity,
            r.reference_db,
            r.engine_db,
            r.tone_db,
            r.engine_db - r.reference_db,
            match (r.offset_db, r.verdict) {
                (Some(db), _) => format!("{db:+.2}"),
                (None, BalanceVerdict::Floor) => "floor".to_string(),
                (None, _) => "ceiling".to_string(),
            }
        );
    }

    let column = |pick: &dyn Fn(&BalanceReading) -> f64| -> (f64, f64) {
        let mut v: Vec<f64> = readings.iter().map(pick).filter(|x| x.is_finite()).collect();
        if v.is_empty() {
            return (f64::NAN, f64::NAN);
        }
        v.sort_by(f64::total_cmp);
        (
            v[v.len() / 2],
            v.iter().map(|x| x.abs()).sum::<f64>() / v.len() as f64,
        )
    };
    let (imbalance, imbalance_abs) = column(&|r| r.engine_db - r.reference_db);
    let (tone_only, tone_only_abs) = column(&|r| r.tone_db - r.reference_db);
    println!(
        "\nengine minus reference: median {imbalance:+.2} dB, mean |·| {imbalance_abs:.2} dB \
         over {} notes",
        readings.len()
    );
    println!(
        "the same with the event silenced: median {tone_only:+.2} dB, mean |·| {tone_only_abs:.2} dB \
         — the sign says which side of the piano the tonal attack is on"
    );
    for &vel in &VELOCITIES {
        let mut a: Vec<f64> = readings
            .iter()
            .filter(|r| r.midi_velocity == vel)
            .map(|r| r.engine_db - r.reference_db)
            .filter(|x| x.is_finite())
            .collect();
        let mut b: Vec<f64> = readings
            .iter()
            .filter(|r| r.midi_velocity == vel)
            .filter_map(|r| r.offset_db)
            .collect();
        a.sort_by(f64::total_cmp);
        b.sort_by(f64::total_cmp);
        if a.is_empty() {
            continue;
        }
        println!(
            "  vel {vel:>3}: engine-ref {:+6.2} dB   offset {} (n {})",
            a[a.len() / 2],
            if b.is_empty() {
                "     —".to_string()
            } else {
                format!("{:+6.2}", b[b.len() / 2])
            },
            b.len()
        );
    }

    let Some(fit) = fit_balance(&readings, MIN_READINGS) else {
        println!(
            "\nnot enough readings inverted to fit a correction ({} of {})",
            readings.iter().filter(|r| r.offset_db.is_some()).count(),
            readings.len()
        );
        return Ok(());
    };
    println!(
        "\nthe correction, Theil-Sen through {} inverted readings ({} floor, {} ceiling):",
        fit.closed, fit.floor, fit.ceiling
    );
    println!("  level at the nominal drive  {:+.2} dB", fit.level_db);
    println!(
        "  velocity_db                 {:.2} -> {:.2}  ({:+.2})",
        preset.noise.strike.velocity_db,
        f64::from(preset.noise.strike.velocity_db) + fit.velocity_db,
        fit.velocity_db
    );
    println!("  scatter about the line      {:.2} dB", fit.scatter_db);

    if let Some(path) = out {
        let mut written = preset.clone();
        for anchor in written.noise.strike.level_db.iter_mut() {
            anchor.db += fit.level_db as f32;
        }
        written.noise.strike.velocity_db += fit.velocity_db as f32;
        written.validate()?;
        written.save(&path)?;
        println!("\n{}", path.display());
        for anchor in &written.noise.strike.level_db {
            println!("  key {:>3}  {:+.3} dB", anchor.key, anchor.db);
        }
        println!("  velocity_db {:.3}", written.noise.strike.velocity_db);
    }
    Ok(())
}
