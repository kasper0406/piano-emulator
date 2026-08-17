//! What a `notes.partial_gains` row does to a key's **level**, measured on the
//! render rather than argued from the row.
//!
//! `DECISIONS.md` 231 and the header of `estimate::shaping::measured_over_rendered`
//! both claim the row is loudness-neutral because its geometric mean is one.
//! This asks the engine. For each key it renders the note twice through the same
//! preset — once as shipped and once with that key's row emptied — and reports
//! the two quantities a listener actually has: the **strike peak** and the
//! **0.2-2.0 s RMS**. The difference between them is what the row moved, and it
//! is not zero.
//!
//! Beside it, three statistics of the row itself, so the arithmetic and the
//! render can be read against each other:
//!
//! * `log mean` — the mean of `20 log10 g_k`, which is what "geometric mean 1"
//!   asserts is zero.
//! * `power mean` — `10 log10 mean(g_k^2)`, which is what a level made of a
//!   *sum of powers* sees. For a zero-log-mean row with a log-spread of `s` dB
//!   these differ by about `s^2 ln 10 / 40` dB — Jensen's inequality, and the
//!   whole of the leak.
//! * `weighted` — the same power mean weighted by the engine's own rendered
//!   partial amplitudes, i.e. the level change the row predicts for this key.
//!
//! ```sh
//! cargo run --release -p forensics --bin gain_level \
//!     -- presets/salamander-c5.toml 60 96 99
//! ```

use std::path::PathBuf;

use piano_emulator::preset::Preset;
use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_emulator::types::Event;
use piano_tuner::SAMPLE_RATE;

const VELOCITY: u8 = 90;
const RENDER_S: f32 = 2.4;
const PREROLL_S: f32 = 0.05;
const RMS_WINDOW: (f64, f64) = (0.2, 2.0);
const FIRST_KEY: u8 = 21;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let preset_path = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "presets/salamander-c5.toml".into()),
    );
    let keys: Vec<u8> = {
        let listed: Vec<u8> = args.filter_map(|a| a.parse().ok()).collect();
        if listed.is_empty() {
            (21..=108).collect()
        } else {
            listed
        }
    };

    let base = Preset::load(&preset_path)?;
    println!(
        "key  n   log mean  power mean  weighted |  peak with/without   RMS with/without |  moved"
    );
    for key in keys {
        let index = usize::from(key - FIRST_KEY);
        let row = base.notes.partial_gains[index].clone();
        if row.is_empty() {
            // An unfitted key is its own bare render: printed so the band a
            // fitted key has to land inside can be taken over the whole compass.
            let (peak, rms) = levels(&base, key);
            println!(
                "{key:>3}   -         -           -         - | \
                 {peak:>7.2}/{peak:>7.2}  {rms:>7.2}/{rms:>7.2} |  +0.00 peak  +0.00 RMS"
            );
            continue;
        }
        let mut bare = base.clone();
        bare.notes.partial_gains[index] = Vec::new();

        let (with_peak, with_rms) = levels(&base, key);
        let (without_peak, without_rms) = levels(&bare, key);

        let logs: Vec<f64> = row.iter().map(|&g| 20.0 * f64::from(g).log10()).collect();
        let log_mean = logs.iter().sum::<f64>() / logs.len() as f64;
        let power_mean = 10.0
            * (row.iter().map(|&g| f64::from(g).powi(2)).sum::<f64>() / row.len() as f64).log10();
        let weighted = weighted_power_mean(&bare, key, &row);

        println!(
            "{key:>3} {:>3}  {log_mean:>+8.2}  {power_mean:>+10.2}  {weighted:>+8.2} | \
             {with_peak:>7.2}/{without_peak:>7.2}  {with_rms:>7.2}/{without_rms:>7.2} | \
             {:>+6.2} peak {:>+6.2} RMS",
            row.len(),
            with_peak - without_peak,
            with_rms - without_rms,
        );
    }
    Ok(())
}

/// `(peak dBFS, RMS dBFS over [`RMS_WINDOW`])` of one key struck alone.
fn levels(preset: &Preset, key: u8) -> (f64, f64) {
    let events = [RenderEvent::new(PREROLL_S, Event::NoteOn { key, vel: u16::from(VELOCITY) })];
    let (left, right) = render_to_buffer(preset, &events, RENDER_S);
    let sr = f64::from(SAMPLE_RATE);
    let mono: Vec<f64> = left
        .iter()
        .zip(&right)
        .map(|(&l, &r)| 0.5 * f64::from(l + r))
        .collect();
    let peak = mono.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let onset = (f64::from(PREROLL_S) * sr) as usize;
    let lo = onset + (RMS_WINDOW.0 * sr) as usize;
    let hi = (onset + (RMS_WINDOW.1 * sr) as usize).min(mono.len());
    let window = &mono[lo.min(mono.len())..hi];
    let rms = if window.is_empty() {
        0.0
    } else {
        (window.iter().map(|x| x * x).sum::<f64>() / window.len() as f64).sqrt()
    };
    (db(peak), db(rms))
}

fn db(x: f64) -> f64 {
    20.0 * x.max(1e-12).log10()
}

/// The power mean of the row weighted by the *engine's own* partial amplitudes:
/// the level change the row predicts, if a level were a sum of partial powers
/// and nothing else.
fn weighted_power_mean(bare: &Preset, key: u8, row: &[f32]) -> f64 {
    let params = bare.string_params(key);
    let amplitudes: Vec<f64> = (1..=row.len())
        .map(|k| {
            let f = f64::from(params.partial_freq(k));
            if f >= 0.5 * f64::from(SAMPLE_RATE) {
                0.0
            } else {
                // A plain 1/k envelope times the engine's own excitation comb is
                // enough weighting to tell a fundamental from a 40th partial.
                let comb = (f64::from(k as u32) * std::f64::consts::PI
                    * f64::from(params.strike_position))
                .sin()
                .abs()
                .max(f64::from(params.comb_floor));
                comb / f64::from(k as u32)
            }
        })
        .collect();
    let total: f64 = amplitudes.iter().map(|a| a * a).sum();
    if total <= 0.0 {
        return 0.0;
    }
    let lifted: f64 = amplitudes
        .iter()
        .zip(row)
        .map(|(a, &g)| (a * f64::from(g)).powi(2))
        .sum();
    10.0 * (lifted / total).log10()
}
