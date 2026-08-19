//! **The tune's own loudness contour**, in the units a listener hears it in:
//! A-weighted energy in 125 ms steps through the fast Ode line, engine against
//! the recording, with the step *into* and *out of* every C4 called out.
//!
//! `DECISIONS.md` 453's owner-facing statement of the level defect is not a
//! ladder number, it is this: *the tune dips into every C4 and jumps out of it
//! where the recording lifts into it and drops out*. The ladder says how big
//! the error is; this says what it does to the music.
//!
//! Both lines are read off the melody board's own rendered files so that this
//! measures exactly what that board played, and the onset of each note is the
//! grid time the board schedules it at — no detector, because the quantity is
//! an energy over a fixed window and not a time.
//!
//! ```sh
//! cargo run --release -p forensics --bin melody_contour -- renders/melody
//! cargo run --release -p forensics --bin melody_contour -- /tmp/melody-before
//! ```

use std::f64::consts::TAU;
use std::path::PathBuf;

use piano_tuner::estimate::melody::soprano;
use piano_tuner::{audio, SamplerEvent, SAMPLE_RATE};

/// The step a listener integrates over: an eighth of a second.
const STEP_S: f64 = 0.125;
/// The key the complaint is about.
const C4: u8 = 60;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "renders/melody".into()),
    );
    let phrase = soprano();
    // Every note's grid time and key, in order.
    let mut notes: Vec<(f64, u8)> = phrase
        .events
        .iter()
        .filter_map(|e| match e.event {
            SamplerEvent::NoteOn { key, .. } => Some((e.time_s, key)),
            _ => None,
        })
        .collect();
    notes.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("finite"));

    println!("A-weighted {STEP_S:.3} s steps through the fast line, {}", dir.display());
    println!(
        "{:>7} {:>5} | {:>9} {:>9} | {:>9} {:>9}",
        "at s", "note", "eng in", "eng out", "ref in", "ref out"
    );
    let engine = audio::load(dir.join("ode_soprano_engine.wav"))?.mono();
    let reference = audio::load(dir.join("ode_soprano_reference.wav"))?.mono();
    let mut steps: Vec<(f64, f64, f64, f64)> = Vec::new();
    for (i, &(t, key)) in notes.iter().enumerate() {
        if key != C4 || i == 0 || i + 1 >= notes.len() {
            continue;
        }
        let (prev, next) = (notes[i - 1].0, notes[i + 1].0);
        let step = |mono: &[f32]| -> (f64, f64) {
            let at = |from: f64| a_weighted_db(window(mono, from, STEP_S), f64::from(SAMPLE_RATE));
            // Each note is read over the same window — its own first
            // [`STEP_S`] — so "into C4" is how much louder C4 is than the note
            // before it and "out of C4" is how much louder the note after it
            // is than C4. Negative in and positive out is a **dip**, which is
            // the owner's percept stated as arithmetic.
            (at(t) - at(prev), at(next) - at(t))
        };
        let (ei, eo) = step(&engine);
        let (ri, ro) = step(&reference);
        println!("{t:>7.2} {:>5} | {ei:>+9.2} {eo:>+9.2} | {ri:>+9.2} {ro:>+9.2}", "C4");
        steps.push((ei, eo, ri, ro));
    }
    if steps.is_empty() {
        println!("  no C4 with a neighbour on either side");
        return Ok(());
    }
    let mean = |pick: fn(&(f64, f64, f64, f64)) -> f64| -> f64 {
        steps.iter().map(pick).sum::<f64>() / steps.len() as f64
    };
    println!(
        "\nmean over {} occurrences: engine {:+.2} in / {:+.2} out, \
         recording {:+.2} in / {:+.2} out",
        steps.len(),
        mean(|s| s.0),
        mean(|s| s.1),
        mean(|s| s.2),
        mean(|s| s.3),
    );
    Ok(())
}

fn window(mono: &[f32], from_s: f64, len_s: f64) -> &[f32] {
    let sr = f64::from(SAMPLE_RATE);
    let a = ((from_s.max(0.0) * sr) as usize).min(mono.len());
    let b = (a + (len_s * sr) as usize).min(mono.len());
    &mono[a..b]
}

fn a_weighted_db(window: &[f32], sample_rate: f64) -> f64 {
    if window.len() < 64 {
        return f64::NAN;
    }
    let n = window.len().next_power_of_two();
    let mut planner = rustfft::FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    let mut buf: Vec<rustfft::num_complex::Complex<f64>> = (0..n)
        .map(|i| {
            let v = f64::from(window.get(i).copied().unwrap_or(0.0));
            let w = if i < window.len() {
                0.5 - 0.5 * (TAU * i as f64 / window.len() as f64).cos()
            } else {
                0.0
            };
            rustfft::num_complex::Complex::new(v * w, 0.0)
        })
        .collect();
    fft.process(&mut buf);
    let bin = sample_rate / n as f64;
    let mut sum = 0.0;
    for (k, c) in buf.iter().take(n / 2).enumerate().skip(1) {
        sum += c.norm_sqr() * a_weight(k as f64 * bin);
    }
    10.0 * (sum / (n as f64 * window.len() as f64)).max(1e-30).log10()
}

fn a_weight(f: f64) -> f64 {
    let f2 = f * f;
    let num = 12194.0f64.powi(2) * f2 * f2;
    let den = (f2 + 20.6f64.powi(2))
        * ((f2 + 107.7f64.powi(2)) * (f2 + 737.9f64.powi(2))).sqrt()
        * (f2 + 12194.0f64.powi(2));
    let a = num / den.max(1e-30);
    a * a * 10.0f64.powf(0.2)
}
