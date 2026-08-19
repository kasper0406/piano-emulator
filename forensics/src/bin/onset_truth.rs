//! The ground truth the onset detectors of `onset_probe` are graded against:
//! the hammer itself, found by the one thing in a melody that is broadband.
//!
//! A piano strike puts energy into 2-6 kHz that a sounding note's tail does not
//! have, so the largest rise of a **high-passed** 2 ms envelope is an onset
//! detector with no low-frequency ripple in it at all — it cannot be fooled by
//! a fundamental whose period is longer than the block, which is the failure
//! `onset_probe` prints. It is too expensive to run per note inside a board
//! (one biquad pass per window) and it is unnecessary there once the block is
//! long enough; here it is the referee.
//!
//! Prints, per note and per side: the HF-truth onset, and a dump of the
//! broadband 2 ms envelope around it in dB so that the shape of the attack —
//! not just its time — can be read.
//!
//! ```sh
//! cargo run --release -p forensics --bin onset_truth -- [engine.wav] [reference.wav] [key]
//! ```

use piano_tuner::audio;
use piano_tuner::estimate::melody;

const BLOCK_MS: f64 = 2.0;
const WIDE: (f64, f64) = (0.05, 0.15);

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let engine = args
        .next()
        .unwrap_or_else(|| "renders/melody/ode_soprano_engine.wav".into());
    let reference = args
        .next()
        .unwrap_or_else(|| "renders/melody/ode_soprano_reference.wav".into());
    let dump_key: Option<u8> = args.next().and_then(|a| a.parse().ok());

    let notes = melody::line_notes();
    for (label, path) in [("engine", &engine), ("reference", &reference)] {
        let audio = audio::load(path)?;
        let sr = f64::from(audio.sample_rate);
        let mono = audio.mono();
        let hf = highpass(&mono, sr, 2000.0);
        println!("\n{label}  {path}");
        println!("  #  key   grid s |  hf truth ms |  broadband 2 ms envelope, dB, from grid-20 ms");
        for (i, note) in notes.iter().enumerate() {
            if !note.measurable() {
                continue;
            }
            let truth = rise(&hf, sr, note.onset_s, WIDE.0, WIDE.1, BLOCK_MS);
            let strip = if dump_key.is_none_or(|k| k == note.key) {
                envelope_strip(&mono, sr, note.onset_s - 0.02, 0.16, BLOCK_MS)
            } else {
                String::new()
            };
            println!(
                "{i:>3}  {:>3}  {:>7.3} |  {:>+8.1} | {strip}",
                note.key,
                note.onset_s,
                1000.0 * (truth - note.onset_s)
            );
        }
    }
    Ok(())
}

/// One-pole-per-stage 2nd-order Butterworth high-pass, applied forward only:
/// the phase it adds is a fraction of a millisecond at 2 kHz and the question
/// here is resolved at two.
fn highpass(x: &[f32], sample_rate: f64, cutoff: f64) -> Vec<f32> {
    let w = (std::f64::consts::PI * cutoff / sample_rate).tan();
    let k = std::f64::consts::SQRT_2;
    let norm = 1.0 / (1.0 + k * w + w * w);
    let (b0, b1, b2) = (norm, -2.0 * norm, norm);
    let a1 = 2.0 * (w * w - 1.0) * norm;
    let a2 = (1.0 - k * w + w * w) * norm;
    let (mut x1, mut x2, mut y1, mut y2) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    x.iter()
        .map(|&s| {
            let x0 = f64::from(s);
            let y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
            x2 = x1;
            x1 = x0;
            y2 = y1;
            y1 = y0;
            y0 as f32
        })
        .collect()
}

fn rise(
    signal: &[f32],
    sample_rate: f64,
    near_s: f64,
    back_s: f64,
    forward_s: f64,
    block_ms: f64,
) -> f64 {
    let block = ((sample_rate * block_ms * 1e-3) as usize).max(1);
    let from = ((((near_s - back_s) * sample_rate) as isize).max(0)) as usize;
    let to = ((((near_s + forward_s) * sample_rate) as usize) + block).min(signal.len());
    if from + 4 * block >= to {
        return near_s;
    }
    let env: Vec<f64> = signal[from..to]
        .chunks(block)
        .map(|c| (c.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / c.len() as f64).sqrt())
        .collect();
    let step = 3usize;
    let mut best = (0usize, f64::MIN);
    for i in 0..env.len().saturating_sub(step) {
        let r = env[i + step] - env[i];
        if r > best.1 {
            best = (i, r);
        }
    }
    from as f64 / sample_rate + best.0 as f64 * block as f64 / sample_rate
}

fn envelope_strip(signal: &[f32], sample_rate: f64, from_s: f64, span_s: f64, block_ms: f64) -> String {
    let block = ((sample_rate * block_ms * 1e-3) as usize).max(1);
    let from = ((from_s * sample_rate).max(0.0)) as usize;
    let to = (((from_s + span_s) * sample_rate) as usize).min(signal.len());
    if from >= to {
        return String::new();
    }
    signal[from..to]
        .chunks(block)
        .map(|c| {
            let r = (c.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / c.len() as f64)
                .sqrt();
            format!("{:>5.0}", 20.0 * r.max(1e-9).log10())
        })
        .collect()
}
