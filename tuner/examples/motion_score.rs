//! Columns A and B alone, per cell, for one preset — the fast half of
//! `realism_bench` so a fit can be iterated against the gates without paying
//! for six phrases of mel spectrograms.
//!
//! Same cells, same code (`piano_tuner::realism::motion_columns`) and therefore
//! the same four numbers `renders/realism/REALISM.md` publishes; this one also
//! prints every cell, which is what says *which* cell fails.
//!
//! ```text
//! cargo run --release -p piano-tuner --example motion_score -- [preset.toml] [data/salamander]
//! ```

use std::path::PathBuf;

use piano_emulator::preset::Preset;
use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_emulator::types::Event;
use piano_tuner::realism::{
    self, MotionCell, A1_GATE, A2_GATE, B1_GATE_DB, B2_GATE, MOTION_KEYS, MOTION_PARTIALS,
    MOTION_REFERENCE_VELOCITY, MOTION_VELOCITIES,
};
use piano_tuner::{audio, detect_onset, SampleLibrary, SAMPLE_RATE};

const RENDER_S: f64 = 4.5;
const PREROLL_S: f64 = 0.05;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let preset_path = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "presets/salamander-c5.toml".into()),
    );
    let data = PathBuf::from(args.next().unwrap_or_else(|| "data/salamander".into()));
    let sfz = data.join("SalamanderGrandPiano-V3+20200602.sfz");

    let preset = Preset::load(&preset_path)?;
    let library = SampleLibrary::from_sfz(&sfz)?;
    let sample_rate = f64::from(SAMPLE_RATE);

    let mut cells: Vec<MotionCell> = Vec::new();
    println!(
        "{:<4} {:>2} {:>4} | {:>7} {:>7} | {:>6} {:>6} | {:>7} {:>7} | {:>6} {:>6}",
        "key", "k", "vel", "J_eng", "J_ref", "L_eng", "L_ref", "D_eng", "D_ref", "R_eng", "R_ref"
    );
    for &(key, name) in &MOTION_KEYS {
        let params = preset.string_params(key);
        let partial_hz: Vec<f64> = (1..=MOTION_PARTIALS)
            .map(|k| f64::from(params.partial_freq(k as usize)))
            .collect();
        for &velocity in &MOTION_VELOCITIES {
            let engine = measure_render(&preset, key, velocity, &partial_hz);
            let reference = library
                .layers(key)
                .iter()
                .find(|s| (s.lovel..=s.hivel).contains(&velocity))
                .and_then(|sample| {
                    let mono = audio::load_at(&sample.path, SAMPLE_RATE).ok()?.mono();
                    let start = (detect_onset(&mono, sample_rate) * sample_rate).round() as usize;
                    let frames = (RENDER_S * sample_rate) as usize;
                    let cut: Vec<f64> = (0..frames)
                        .map(|n| f64::from(mono.get(start + n).copied().unwrap_or(0.0)))
                        .collect();
                    Some(realism::measure_partials(&cut, &partial_hz))
                })
                .unwrap_or_else(|| vec![None; partial_hz.len()]);
            for k in 1..=MOTION_PARTIALS {
                let cell = MotionCell {
                    key,
                    k,
                    velocity,
                    engine: engine[k as usize - 1],
                    reference: reference[k as usize - 1],
                };
                let f = |m: Option<piano_tuner::motion::Motion>,
                         g: fn(&piano_tuner::motion::Motion) -> f64| {
                    m.map_or(f64::NAN, |m| g(&m))
                };
                println!(
                    "{name:<4} {k:>2} {velocity:>4} | {:>7.3} {:>7.3} | {:>6.3} {:>6.3} | \
                     {:>7.2} {:>7.2} | {:>6.3} {:>6.3}",
                    f(cell.engine, |m| m.band_cents),
                    f(cell.reference, |m| m.band_cents),
                    f(cell.engine, |m| m.placement()),
                    f(cell.reference, |m| m.placement()),
                    f(cell.engine, |m| m.beat_depth_db),
                    f(cell.reference, |m| m.beat_depth_db),
                    f(cell.engine, |m| m.beat_rate_hz),
                    f(cell.reference, |m| m.beat_rate_hz),
                );
                cells.push(cell);
            }
        }
    }

    let c = realism::motion_columns(&cells);
    println!("\npreset {}", preset_path.display());
    println!(
        "A1 {:6.3} (<= {A1_GATE})   A2 {:6.3} (>= {A2_GATE})   \
         B1 {:6.3} dB (<= {B1_GATE_DB})   B2 {:6.3} (>= {B2_GATE})",
        c.if_mismatch, c.if_placement, c.beat_depth_error_db, c.velocity_coherence
    );
    println!(
        "cells {} / velocity cells {}; B2 halves freq {:.3} ({:.3} vs {:.3} c) depth {:.3} \
         ({:.2} vs {:.2} dB)",
        c.cells,
        c.velocity_cells,
        c.velocity_coherence_freq,
        c.spread_cents.0,
        c.spread_cents.1,
        c.velocity_coherence_depth,
        c.spread_depth_db.0,
        c.spread_depth_db.1
    );
    println!("passes: {}", c.passes());

    // The per-cell contributions to A1 and B1, worst first: the two failing
    // columns are means, and a mean is not actionable without its terms.
    let mut a1: Vec<(f64, String)> = Vec::new();
    let mut b1: Vec<(f64, String)> = Vec::new();
    for cell in cells.iter().filter(|c| c.velocity == MOTION_REFERENCE_VELOCITY) {
        let (Some(e), Some(r)) = (cell.engine, cell.reference) else {
            continue;
        };
        let name = MOTION_KEYS
            .iter()
            .find(|(k, _)| *k == cell.key)
            .map_or("?", |(_, n)| *n);
        let (a, b) = (e.floored_cents(), r.floored_cents());
        a1.push((
            a.max(b) / a.min(b),
            format!("{name} k={} {:.3} vs {:.3} c", cell.k, e.band_cents, r.band_cents),
        ));
        b1.push((
            (e.beat_depth_db - r.beat_depth_db).abs(),
            format!(
                "{name} k={} {:.2} vs {:.2} dB",
                cell.k, e.beat_depth_db, r.beat_depth_db
            ),
        ));
    }
    a1.sort_by(|x, y| y.0.total_cmp(&x.0));
    b1.sort_by(|x, y| y.0.total_cmp(&x.0));
    println!("\nA1 worst cells:");
    for (v, s) in a1.iter().take(8) {
        println!("  {v:6.2}x  {s}");
    }
    println!("B1 worst cells:");
    for (v, s) in b1.iter().take(8) {
        println!("  {v:6.2} dB  {s}");
    }
    Ok(())
}

fn measure_render(
    preset: &Preset,
    key: u8,
    velocity: u8,
    partial_hz: &[f64],
) -> Vec<Option<piano_tuner::motion::Motion>> {
    let events = [RenderEvent::new(
        PREROLL_S as f32,
        Event::NoteOn { key, vel: velocity },
    )];
    let (left, right) = render_to_buffer(preset, &events, (PREROLL_S + RENDER_S) as f32);
    let skip = (PREROLL_S * f64::from(SAMPLE_RATE)) as usize;
    let mono: Vec<f64> = left
        .iter()
        .zip(&right)
        .skip(skip)
        .map(|(&l, &r)| 0.5 * (f64::from(l) + f64::from(r)))
        .collect();
    realism::measure_partials(&mono, partial_hz)
}
