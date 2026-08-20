//! **A key's own loudness against the recording of the same key** — the
//! quantity `DECISIONS.md` 272 measured, named and decided not to write
//! anywhere, and item 457 re-opened.
//!
//! [`piano_tuner::estimate::level`] holds the estimator and the argument for its
//! shape; this is the driver. It renders every key of the compass and, at the
//! thirty the library recorded, the recording of the same key and the same key
//! out of the neighbouring velocity layer; measures the A-weighted energy of the
//! note's head on the **mono fold-down**; and writes the shrunk, smoothed,
//! capped per-key level into `notes.partial_gains` through the field's own
//! pinning, [`flatten_row`].
//!
//! # Why `notes.partial_gains` and not a table of its own
//!
//! Item 272 tried `notes.bridge_gain` and measured it worse: a per-key level is
//! exactly what that table is, but it is also what `DECISIONS.md` 44 calibrated
//! the engine's flattened compass on and what item 282's decomposition is
//! about, and hiding a library's gain inside a physical table is how a
//! measurement becomes a lie. `notes.partial_gains` is where a key's *radiated*
//! level already lives: `estimate::shaping::energy_offset` takes the scalar out
//! of the row on purpose, `flatten_row` puts one back into it, and
//! `fit::motion`'s `LEVEL_BAND_DB` already licenses a fitted row to carry two
//! sigmas of it as a side effect. What this stage changes is that the level is
//! now written **on purpose, against the recording, and smoothed across the
//! compass** instead of being whatever the row's shape happened to drag with it.
//!
//! # The loop
//!
//! Closed on the render, like every other stage here (items 137, 199, 211, 264,
//! 273, 300): pass one measures each key's deficit and fits the curve, and each
//! pass after it measures what is **left** of the target and corrects the row
//! by it, damped. The target is fitted once and never re-fitted, which is the
//! whole of the difference between this and the per-key free gain item 272
//! refused — a target that were re-fitted every pass would converge on each
//! key's own measurement however much of it is the library's take-to-take gain.
//!
//! ```sh
//! cargo run --release -p piano-tuner -- level \
//!     data/salamander presets/salamander-c5.toml
//! cargo run --release -p piano-tuner -- level \
//!     data/salamander presets/salamander-c5.toml --passes 3 --out /tmp/levelled.toml
//! ```

use std::cell::RefCell;
use std::path::{Path, PathBuf};

use rayon::prelude::*;

use piano_emulator::preset::Preset;
use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_emulator::types::Event;
use piano_tuner::cache;
use piano_tuner::estimate::level::{LevelCurve, LevelPoint, MAX_LEVEL_DB};
use piano_tuner::estimate::melody::a_weighted_db;
use piano_tuner::estimate::shaping::{flatten_row, ShapingConfig};
use piano_tuner::realism::VelocityLayers;
use piano_tuner::sampler::SAMPLER_VERSION;
use piano_tuner::{Audio, SampleLibrary, Sampler, SamplerEvent, TimedEvent, SAMPLE_RATE};

/// The velocity every render here is taken at.
///
/// The melody's own, because the percept this stage exists for is a melody:
/// `DECISIONS.md` 453's ladder, the `c4_ledger` instrument and the melody
/// board's `loudness` column are all at this velocity, so the number this stage
/// closes is the number they read.
const VELOCITY: u8 = piano_tuner::realism::ODE_MELODY_VEL;
/// Seconds of note rendered.
const RENDER_S: f64 = 2.6;
/// The window the level is read over: the note's head, from its own onset.
///
/// The same 0.45 s `forensics/src/bin/c4_ledger.rs` reads and the same span the
/// melody board's head window covers. A level over the *whole* note is a decay
/// as much as a level, and the two defects of item 453 are separate.
const HEAD_S: f64 = 0.45;
const PREROLL_S: f64 = 0.05;
const FIRST_KEY: u8 = 21;
const LAST_KEY: u8 = 108;
const SR: f64 = SAMPLE_RATE as f64;
const MAX_CACHED_BUFFERS: usize = 8;

/// Share of one pass's remaining error applied. The row's cells multiply
/// amplitudes and the note's rendered level also contains its mechanism noise,
/// its halo and the partials the row does not cover, so the response to a lift
/// is near one and not one; damping turns that into convergence.
const DAMPING: f64 = 0.8;

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

fn render_engine(preset: &Preset, key: u8) -> Vec<f32> {
    let events = [RenderEvent::new(
        PREROLL_S as f32,
        Event::NoteOn {
            key,
            vel: u16::from(VELOCITY),
        },
    )];
    let (left, right) = render_to_buffer(preset, &events, (PREROLL_S + RENDER_S) as f32);
    let skip = (PREROLL_S * SR) as usize;
    Audio::new(
        SAMPLE_RATE,
        vec![left[skip..].to_vec(), right[skip..].to_vec()],
    )
    .expect("the engine renders stereo")
    .mono()
}

fn render_reference(sampler: &mut Sampler, key: u8, vel: u8) -> Result<Audio, piano_tuner::Error> {
    let events = [TimedEvent::new(0.0, SamplerEvent::NoteOn { key, vel })];
    let rendered = sampler.render(&events, RENDER_S + 0.2)?;
    let mono = rendered.mono();
    let onset = piano_tuner::detect_onset(&mono, SR);
    let skip = (onset * SR).round() as usize;
    let frames = (RENDER_S * SR) as usize;
    let cut = |c: &Vec<f32>| -> Vec<f32> {
        (0..frames)
            .map(|n| c.get(skip + n).copied().unwrap_or(0.0))
            .collect()
    };
    Audio::new(SAMPLE_RATE, rendered.channels.iter().map(cut).collect())
}

/// The A-weighted energy of a note's head, dB.
fn head_db(mono: &[f32]) -> f64 {
    let n = ((HEAD_S * SR) as usize).min(mono.len());
    a_weighted_db(&mono[..n], SR)
}

fn note_name(key: u8) -> String {
    const NAMES: [&str; 12] = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    format!("{}{}", NAMES[usize::from(key) % 12], i32::from(key) / 12 - 1)
}

pub fn run(args: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
    let mut args = args.into_iter();
    let data = PathBuf::from(args.next().unwrap_or_else(|| "data/salamander".into()));
    let preset_path = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "presets/salamander-c5.toml".into()),
    );
    let rest: Vec<String> = args.collect();
    let flag = |name: &str| -> Option<&String> {
        rest.iter()
            .position(|a| a == name)
            .and_then(|i| rest.get(i + 1))
    };
    let out: Option<PathBuf> = flag("--out").map(PathBuf::from);
    let passes: usize = flag("--passes").and_then(|v| v.parse().ok()).unwrap_or(0);
    // Whichever library this tree is, rather than Salamander's own filename:
    // `adapter::instrument_path` resolves a described library through its
    // LibrarySpec and an undescribed one by its single map (DECISIONS.md 521).
    let sfz = piano_tuner::adapter::instrument_path(&data)?;
    if !sfz.exists() {
        eprintln!("the reference piano is not here: {}", sfz.display());
        std::process::exit(2);
    }
    let mut preset = Preset::load(&preset_path)?;
    let library = SampleLibrary::from_sfz(&sfz)?;
    let mut sampled: Vec<u8> = library.samples().map(|s| s.key).collect();
    sampled.sort_unstable();
    sampled.dedup();
    let keys: Vec<u8> = (FIRST_KEY..=LAST_KEY).collect();
    let shaping = ShapingConfig::default();

    let reference_cache = cache::reference_dir(&data);
    let layers = VelocityLayers::from_library(&library)?;
    let alt_velocity = layers.alternate(VELOCITY);
    let fingerprint = |what: &str, vel: u8| -> Result<cache::Fingerprint, piano_tuner::Error> {
        let mut k = cache::Fingerprint::new();
        k.str(what)
            .u64(u64::from(SAMPLER_VERSION))
            .file(&sfz)?
            .u64(u64::from(SAMPLE_RATE))
            .u64(u64::from(vel))
            .f64(RENDER_S);
        Ok(k)
    };
    let own_key = fingerprint("level/reference", VELOCITY)?;
    let alt_key = fingerprint("level/alt-layer", alt_velocity)?;

    println!(
        "per-key level: engine on {}, reference {}\n\
         {} keys at velocity {VELOCITY}, A-weighted over the first {HEAD_S} s of the mono sum",
        preset_path.display(),
        sfz.display(),
        keys.len(),
    );

    // The reference side never moves, so it is measured once.
    let reference: Vec<(u8, Option<(f64, f64)>)> = keys
        .par_iter()
        .map(|&key| -> Result<(u8, Option<(f64, f64)>), piano_tuner::Error> {
            if !sampled.contains(&key) {
                return Ok((key, None));
            }
            let read = |what: &str, mut k: cache::Fingerprint, vel: u8| {
                k.u64(u64::from(key));
                let path = reference_cache.join(format!("{what}-key{key:03}-{}.wav", k.hex()));
                cache::audio(&path, || {
                    with_sampler(&sfz, |s| render_reference(s, key, vel))
                })
                .map(|a| head_db(&a.mono()))
            };
            let own = read("level", own_key, VELOCITY)?;
            let alt = read("level-alt", alt_key, alt_velocity)?;
            Ok((key, Some((own, alt))))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let engine_levels = |preset: &Preset| -> Vec<(u8, f64)> {
        keys.par_iter()
            .map(|&key| (key, head_db(&render_engine(preset, key))))
            .collect()
    };

    let before = engine_levels(&preset);
    let points: Vec<LevelPoint> = reference
        .iter()
        .filter_map(|&(key, r)| {
            let (own, alt) = r?;
            let engine = before.iter().find(|&&(k, _)| k == key)?.1;
            Some(LevelPoint {
                key,
                error_db: engine - own,
                take_db: (own - alt).abs(),
            })
        })
        .collect();
    let curve = LevelCurve::fit(&points);
    println!(
        "\nthe curve: common offset {:+.2} dB (the engine's gain against the library's, \
         nobody's error)\n  \
         line exp({:+.4}{:+.5}·key) (r {:+.3}, n {}), take-to-take sigma {:.2} dB, \
         residual sigma {:.2} dB, median belief {:.2}\n  \
         written: {:+.2} at A0, {:+.2} at D#3, {:+.2} at C4, {:+.2} at A4, {:+.2} at C5, \
         {:+.2} at C8 — capped at ±{MAX_LEVEL_DB:.2}",
        curve.offset_db,
        curve.line.intercept,
        curve.line.slope,
        curve.line.correlation,
        curve.line.points,
        curve.take_sigma_db,
        curve.residual_sigma_db,
        curve.shrink,
        curve.at(21),
        curve.at(51),
        curve.at(60),
        curve.at(69),
        curve.at(72),
        curve.at(108),
    );

    // ---- the loop ----------------------------------------------------------
    //
    // Each pass measures what is left of the target on the render and corrects
    // the row by it. The target itself is the curve above and is never
    // re-fitted: see the module header.
    let mut lift: Vec<f64> = vec![0.0; keys.len()];
    let mut current = before.clone();
    for pass in 1..=passes {
        for (i, &key) in keys.iter().enumerate() {
            let index = usize::from(key - FIRST_KEY);
            let got = current[i].1 - before[i].1;
            let want = curve.at(key);
            let step = DAMPING * (want - got);
            if !step.is_finite() || step.abs() < 0.01 {
                continue;
            }
            lift[i] = (lift[i] + step).clamp(-2.0 * MAX_LEVEL_DB, 2.0 * MAX_LEVEL_DB);
            // A key whose `notes.partial_gains` row is empty has nowhere to
            // carry a level: on this library that is A7 and C8, the two keys
            // the sampler recorded and whose banks measured nothing
            // (`DECISIONS.md` 275-276, 284). Reported in the table's `moved`
            // column as a zero, never as a silent skip.
            let row = &preset.notes.partial_gains[index];
            if row.is_empty() {
                continue;
            }
            preset.notes.partial_gains[index] = flatten_row(row, 1.0, step, &shaping);
        }
        preset.validate()?;
        current = engine_levels(&preset);
        let worst = keys
            .iter()
            .enumerate()
            .filter(|(_, k)| {
                sampled.contains(k) && !preset.notes.partial_gains[usize::from(**k - FIRST_KEY)].is_empty()
            })
            .map(|(i, &k)| (current[i].1 - before[i].1 - curve.at(k)).abs())
            .fold(0.0f64, f64::max);
        println!("  pass {pass}: worst recorded key still {worst:.2} dB from its target");
    }

    if let Some(path) = &out {
        preset.save(path)?;
        println!("wrote {}", path.display());
    }

    // ---- the table ---------------------------------------------------------
    println!(
        "\n{:>4} {:>5} {:>9} {:>9} {:>9} {:>8} {:>8} {:>8} {:>8}",
        "key", "note", "engine", "recording", "take", "err", "seam", "target", "moved"
    );
    for (i, &key) in keys.iter().enumerate() {
        let Some((own, alt)) = reference[i].1 else {
            continue;
        };
        let err = before[i].1 - own;
        println!(
            "{:>4} {:>5} {:>9.2} {:>9.2} {:>9.2} {:>8.2} {:>+8.2} {:>+8.2} {:>+8.2}",
            key,
            note_name(key),
            before[i].1,
            own,
            (own - alt).abs(),
            err,
            err - curve.offset_db,
            curve.at(key),
            current[i].1 - before[i].1,
        );
    }
    let seams: Vec<f64> = points
        .iter()
        .map(|p| (p.error_db - curve.offset_db).abs())
        .collect();
    let worst_before = seams.iter().copied().fold(0.0f64, f64::max);
    let after: Vec<f64> = keys
        .iter()
        .enumerate()
        .filter(|(_, k)| sampled.contains(k))
        .filter_map(|(i, _)| {
            let (own, _) = reference[i].1?;
            Some(current[i].1 - own)
        })
        .collect();
    let centre = {
        let mut v = after.clone();
        v.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        v.get(v.len() / 2).copied().unwrap_or(0.0)
    };
    let worst_after = after
        .iter()
        .map(|e| (e - centre).abs())
        .fold(0.0f64, f64::max);
    println!(
        "\nworst recorded key's departure from the register's median level: \
         {worst_before:.2} dB before, {worst_after:.2} dB after"
    );
    Ok(())
}
