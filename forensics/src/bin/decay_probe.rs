//! Whether a partial's decay rate is a **measurement** on this library, or a
//! function of the span it was fitted over.
//!
//! `c4_ledger` establishes that C4's fitted `notes.partial_sigma_scale` cell on
//! the fundamental — 0.534 where the recording asks for about 0.79 — is the
//! whole of the held-octave collapse. This asks the question one level down:
//! the cell is a straight line in dB through the recording's own envelope, and
//! a three-string unison's fundamental does not decay along a straight line, it
//! beats. If the fitted slope depends on where the fit stops, the cell is not a
//! property of the piano.
//!
//! Per key and per partial it prints the least-squares slope of the dB envelope
//! over four spans, the slope's spread across those spans, and the RMS of the
//! envelope about its own line over the longest of them — the beat depth that
//! is doing the biasing, in the same units.
//!
//! ```sh
//! cargo run --release -p forensics --bin decay_probe -- data/salamander 51 54 57 60 63 66 69 72 75
//! ```

use std::path::PathBuf;

use piano_tuner::estimate::brilliance::narrowband_db;
use piano_tuner::{realism, Sampler, SamplerEvent, TimedEvent, SAMPLE_RATE};

const VELOCITY: u8 = realism::ODE_MELODY_VEL;
const RENDER_S: f64 = 3.2;
/// Spans, in seconds from the note's own onset, that the slope is fitted over.
const SPANS: [(f64, f64); 4] = [(0.10, 0.60), (0.10, 1.00), (0.10, 1.50), (0.30, 2.00)];
const PARTIALS: usize = 4;
/// A partial inside this of its own late floor is not decaying, it is the floor.
const FLOOR_MARGIN_DB: f64 = 6.0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let data = PathBuf::from(args.next().unwrap_or_else(|| "data/salamander".into()));
    let keys: Vec<u8> = args.filter_map(|a| a.parse().ok()).collect();
    let keys = if keys.is_empty() {
        vec![51, 54, 57, 60, 63, 66, 69, 72, 75]
    } else {
        keys
    };
    let sfz = data.join("SalamanderGrandPiano-V3+20200602.sfz");
    let mut sampler = Sampler::new(&sfz)?;

    println!(
        "the recording's own partial decays, velocity {VELOCITY}\n\
         key   k       hz | {} | spread  x  | wobble dB",
        SPANS
            .iter()
            .map(|s| format!("{:>5.2}-{:<4.2}", s.0, s.1))
            .collect::<Vec<_>>()
            .join("  ")
    );
    for key in keys {
        let mono = reference_mono(&mut sampler, key)?;
        let f0 = 440.0 * 2.0f64.powf((f64::from(key) - 69.0) / 12.0);
        for k in 1..=PARTIALS {
            let env = narrowband_db(&mono, f0 * k as f64, f64::from(SAMPLE_RATE));
            let floor = late_floor(&env);
            let slopes: Vec<f64> = SPANS
                .iter()
                .map(|&(a, b)| slope_db_per_s(&env, a, b, floor))
                .collect();
            let finite: Vec<f64> = slopes.iter().copied().filter(|s| s.is_finite()).collect();
            let (lo, hi) = finite.iter().fold((f64::MAX, f64::MIN), |a, &s| {
                (a.0.min(s), a.1.max(s))
            });
            let ratio = if finite.len() > 1 && lo.abs() > 1e-9 {
                hi.abs().max(lo.abs()) / hi.abs().min(lo.abs())
            } else {
                f64::NAN
            };
            println!(
                "{key:>3} {k:>3} {:>8.1} | {} | {:>6.1} {:>4.2} | {:>9.2}",
                f0 * k as f64,
                slopes
                    .iter()
                    .map(|s| format!("{s:>10.1}"))
                    .collect::<String>(),
                hi - lo,
                ratio,
                residual_rms(&env, 0.10, 1.50, floor),
            );
        }
    }
    Ok(())
}

/// The envelope's own floor: the median of its last 300 ms.
fn late_floor(env: &[f64]) -> f64 {
    if env.len() < 400 {
        return f64::NEG_INFINITY;
    }
    let mut tail: Vec<f64> = env[env.len() - 300..].to_vec();
    tail.sort_by(|a, b| a.partial_cmp(b).unwrap());
    tail[tail.len() / 2]
}

fn points(env: &[f64], from_s: f64, to_s: f64, floor: f64) -> Vec<(f64, f64)> {
    let (a, b) = (
        (from_s * 1000.0) as usize,
        ((to_s * 1000.0) as usize).min(env.len()),
    );
    if a >= b {
        return Vec::new();
    }
    env[a..b]
        .iter()
        .enumerate()
        .filter(|(_, &v)| v.is_finite() && v > floor + FLOOR_MARGIN_DB)
        .map(|(i, &v)| ((a + i) as f64 / 1000.0, v))
        .collect()
}

fn slope_db_per_s(env: &[f64], from_s: f64, to_s: f64, floor: f64) -> f64 {
    let p = points(env, from_s, to_s, floor);
    if p.len() < 50 {
        return f64::NAN;
    }
    let n = p.len() as f64;
    let mx = p.iter().map(|q| q.0).sum::<f64>() / n;
    let my = p.iter().map(|q| q.1).sum::<f64>() / n;
    let num: f64 = p.iter().map(|q| (q.0 - mx) * (q.1 - my)).sum();
    let den: f64 = p.iter().map(|q| (q.0 - mx).powi(2)).sum();
    if den <= 0.0 {
        f64::NAN
    } else {
        num / den
    }
}

fn residual_rms(env: &[f64], from_s: f64, to_s: f64, floor: f64) -> f64 {
    let p = points(env, from_s, to_s, floor);
    if p.len() < 50 {
        return f64::NAN;
    }
    let n = p.len() as f64;
    let mx = p.iter().map(|q| q.0).sum::<f64>() / n;
    let my = p.iter().map(|q| q.1).sum::<f64>() / n;
    let num: f64 = p.iter().map(|q| (q.0 - mx) * (q.1 - my)).sum();
    let den: f64 = p.iter().map(|q| (q.0 - mx).powi(2)).sum();
    let slope = if den > 0.0 { num / den } else { 0.0 };
    (p.iter()
        .map(|q| (q.1 - (my + slope * (q.0 - mx))).powi(2))
        .sum::<f64>()
        / n)
        .sqrt()
}

fn reference_mono(sampler: &mut Sampler, key: u8) -> Result<Vec<f32>, piano_tuner::Error> {
    let events = [TimedEvent::new(
        0.0,
        SamplerEvent::NoteOn {
            key,
            vel: VELOCITY,
        },
    )];
    let rendered = sampler.render(&events, RENDER_S + 0.3)?;
    let mono = rendered.mono();
    let onset = piano_tuner::detect_onset(&mono, f64::from(SAMPLE_RATE));
    let skip = ((onset * f64::from(SAMPLE_RATE)).round() as usize).min(mono.len());
    Ok(mono[skip..].to_vec())
}
