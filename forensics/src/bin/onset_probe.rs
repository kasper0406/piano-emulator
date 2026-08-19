//! Where the melody board's per-note window actually starts, on both sides of
//! the line, and what the block length of the envelope it is found with does to
//! that.
//!
//! `estimate::melody::note_onset` refines each note's grid time into the
//! largest rise of a **1 ms** RMS envelope over `ONSET_SEARCH_S`. One
//! millisecond is a fraction of a period at the bottom of the melodic register
//! — C4's is 3.8 ms — so that envelope is not an envelope at all there, it is
//! the waveform's own ripple, and the largest rise in it can land on any later
//! cycle of the note instead of on the hammer. This prints, per note of the
//! soprano line and per side, the offset from the grid that each block length
//! finds, so the miss can be read as a number rather than argued.
//!
//! ```sh
//! cargo run --release -p forensics --bin onset_probe -- \
//!     renders/melody/ode_soprano_engine.wav renders/melody/ode_soprano_reference.wav
//! ```

use piano_tuner::audio;
use piano_tuner::estimate::melody;

/// Block lengths tried, in milliseconds.
const BLOCKS_MS: [f64; 4] = [1.0, 2.0, 3.0, 5.0];

/// The window each search runs in, seconds either side of the grid time.
const WIDE: (f64, f64) = (0.05, 0.15);

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let engine = args
        .next()
        .unwrap_or_else(|| "renders/melody/ode_soprano_engine.wav".into());
    let reference = args
        .next()
        .unwrap_or_else(|| "renders/melody/ode_soprano_reference.wav".into());

    let notes = melody::line_notes();
    for (label, path) in [("engine", &engine), ("reference", &reference)] {
        let audio = audio::load(path)?;
        let sr = f64::from(audio.sample_rate);
        let mono = audio.mono();
        println!("\n{label}  {path}   ({:.2} s)", audio.duration_s());
        println!(
            "  #  key   grid s |{}",
            BLOCKS_MS
                .iter()
                .map(|b| format!("  {b:>4.0} ms"))
                .collect::<String>()
        );
        for (i, note) in notes.iter().enumerate() {
            if !note.measurable() {
                continue;
            }
            let offsets: String = BLOCKS_MS
                .iter()
                .map(|&block_ms| {
                    let t = strike_near_block(
                        &mono, sr, note.onset_s, WIDE.0, WIDE.1, block_ms,
                    );
                    format!("  {:>+7.1}", 1000.0 * (t - note.onset_s))
                })
                .collect();
            let shipped = melody::note_onset(&mono, sr, note.onset_s);
            println!(
                "{i:>3}  {:>3}  {:>7.3} |{offsets}   shipped {:>+7.1} ms",
                note.key,
                note.onset_s,
                1000.0 * (shipped - note.onset_s)
            );
        }
    }

    // Per-key summary: the melody board reads one number per pitch, so the
    // question is whether one pitch is systematically mis-windowed.
    println!("\nmedian offset by key, ms (block ms across)");
    for (label, path) in [("engine", &engine), ("reference", &reference)] {
        let audio = audio::load(path)?;
        let sr = f64::from(audio.sample_rate);
        let mono = audio.mono();
        let mut keys: Vec<u8> = notes.iter().filter(|n| n.measurable()).map(|n| n.key).collect();
        keys.sort_unstable();
        keys.dedup();
        for key in keys {
            let cells: String = BLOCKS_MS
                .iter()
                .map(|&block_ms| {
                    let mut offs: Vec<f64> = notes
                        .iter()
                        .filter(|n| n.measurable() && n.key == key)
                        .map(|n| {
                            1000.0
                                * (strike_near_block(
                                    &mono, sr, n.onset_s, WIDE.0, WIDE.1, block_ms,
                                ) - n.onset_s)
                        })
                        .collect();
                    offs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    format!("  {:>+7.1}", offs[offs.len() / 2])
                })
                .collect();
            println!("  {label:>9}  key {key:>3} |{cells}");
        }
    }
    Ok(())
}

/// [`realism::strike_near`] with the envelope's block length exposed.
///
/// Kept here rather than called, so that this instrument measures the shipped
/// detector's *shape* even after the shipped one is changed.
fn strike_near_block(
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
    let envelope: Vec<f64> = signal[from..to]
        .chunks(block)
        .map(|c| (c.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>() / c.len() as f64).sqrt())
        .collect();
    let step = 3usize;
    let mut best = (0usize, f64::MIN);
    for i in 0..envelope.len().saturating_sub(step) {
        let rise = envelope[i + step] - envelope[i];
        if rise > best.1 {
            best = (i, rise);
        }
    }
    from as f64 / sample_rate + best.0 as f64 * block as f64 / sample_rate
}
