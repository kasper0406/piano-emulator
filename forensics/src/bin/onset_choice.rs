//! Which onset detector the melody board should use, decided by grading every
//! candidate against the hammer rather than against an argument.
//!
//! `onset_truth` establishes that on both sides of the melody render every
//! note's hammer is within a few milliseconds of its grid time. This sweeps the
//! two knobs — the envelope's block length and the band it is taken over — over
//! both sides of the line and prints, per candidate, the worst and the median
//! absolute miss against that truth. It is the table `DECISIONS.md` 452 quotes.
//!
//! ```sh
//! cargo run --release -p forensics --bin onset_choice
//! ```

use piano_tuner::audio;
use piano_tuner::estimate::melody;

const TRUTH_MS: f64 = 0.0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let engine = args
        .next()
        .unwrap_or_else(|| "renders/melody/ode_soprano_engine.wav".into());
    let reference = args
        .next()
        .unwrap_or_else(|| "renders/melody/ode_soprano_reference.wav".into());

    let slow = std::env::args().any(|a| a == "--slow");
    let notes = if slow {
        melody::slow_line_notes()
    } else {
        melody::line_notes()
    };
    let (engine, reference) = if slow {
        (
            "renders/melody/ode_pitches_slow_engine.wav".to_string(),
            "renders/melody/ode_pitches_slow_reference.wav".to_string(),
        )
    } else {
        (engine, reference)
    };
    let sides: Vec<(&str, Vec<f32>, f64)> = [("engine", &engine), ("reference", &reference)]
        .iter()
        .map(|(label, path)| {
            let a = audio::load(path).expect("render");
            let sr = f64::from(a.sample_rate);
            (*label, a.mono(), sr)
        })
        .collect();

    println!("candidate                    | side       worst ms   median ms   >10 ms   at");
    for &(hp, order) in &[(0.0f64, 1u32), (2000.0, 1), (2000.0, 2), (2000.0, 3), (1000.0, 2), (3000.0, 2)] {
        for &block_ms in &[1.0f64, 2.0, 3.0, 5.0] {
            for &fwd in &[0.12f64, 0.15] {
                let name = format!(
                    "{:>5} Hz hp x{order}, {block_ms:>3.0} ms, +{:>3.0} ms",
                    hp as u32, 1000.0 * fwd
                );
                for (label, mono, sr) in &sides {
                    let mut band = mono.clone();
                    if hp > 0.0 {
                        for _ in 0..order {
                            band = highpass(&band, *sr, hp);
                        }
                    }
                    let mut misses: Vec<(f64, u8)> = Vec::new();
                    for note in notes.iter().filter(|n| n.measurable()) {
                        let t = rise(&band, *sr, note.onset_s, 0.05, fwd, block_ms);
                        misses.push((1000.0 * (t - note.onset_s) - TRUTH_MS, note.key));
                    }
                    let mut abs: Vec<f64> = misses.iter().map(|m| m.0.abs()).collect();
                    abs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let worst = misses
                        .iter()
                        .cloned()
                        .fold((0.0f64, 0u8), |acc, m| if m.0.abs() > acc.0.abs() { m } else { acc });
                    let bad = misses.iter().filter(|m| m.0.abs() > 10.0).count();
                    println!(
                        "{name} | {label:>9}   {:>+8.1}   {:>+9.1}   {bad:>6}   key {}",
                        worst.0,
                        abs[abs.len() / 2],
                        worst.1
                    );
                }
            }
        }
    }
    Ok(())
}

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
