//! One key, struck alone, with one fitted table removed at a time.
//!
//! `compass_scan` says *which* key does not fit the rest. This says *what in
//! the preset is responsible*, by the only method that settles it: take the
//! candidate out and render again. Five tables can shape a single note's
//! spectrum after the construction has had its say — `notes.partial_gains`,
//! `notes.partial_sigma_scale`, `notes.false_beat`, `notes.comb_floor` and
//! `notes.detune_cents` — and each is removed on its own and then all together,
//! so the attribution is a subtraction rather than an argument.
//!
//! The statistic every row is ranked on is **`error`**: the mean absolute
//! difference between the engine's partial levels and the recording's, in dB,
//! after the common offset between the two has been removed. A common offset is
//! a gain and a gain is not a timbre; what is left is the shape of the harmonic
//! series, which is what a listener hears as the note's colour.
//!
//! ```sh
//! cargo run --release -p piano-tuner --example key_probe \
//!     -- data/salamander presets/salamander-c5.toml 33 36 39
//! ```

use std::f64::consts::TAU;
use std::path::PathBuf;

use piano_emulator::preset::Preset;
use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_emulator::types::Event;
use piano_tuner::{realism, Audio, Sampler, SamplerEvent, TimedEvent, SAMPLE_RATE};

const VELOCITY: u8 = 90;
const RENDER_S: f64 = 3.6;
const PREROLL_S: f64 = 0.05;
const PARTIALS: usize = 12;
const WINDOW: (f64, f64) = (0.10, 1.10);
const FIRST_KEY: u8 = 21;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let data = PathBuf::from(args.next().unwrap_or_else(|| "data/salamander".into()));
    let preset_path = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "presets/salamander-c5.toml".into()),
    );
    let keys: Vec<u8> = args.filter_map(|a| a.parse().ok()).collect();
    let keys = if keys.is_empty() { vec![36] } else { keys };

    let sfz = data.join("SalamanderGrandPiano-V3+20200602.sfz");
    let base = Preset::load(&preset_path)?;
    let mut sampler = Sampler::new(&sfz)?;

    for key in keys {
        let i = usize::from(key - FIRST_KEY);
        let params = base.string_params(key);
        let partial_hz: Vec<f64> = (1..=PARTIALS)
            .map(|k| f64::from(params.partial_freq(k)))
            .collect();
        let reference = reference_levels(&mut sampler, key, &partial_hz)?;

        println!(
            "\nkey {key} ({}), f0 {:.2} Hz, {} string(s)",
            note_name(key),
            partial_hz[0],
            base.notes.unison[i]
        );
        println!(
            "  fitted row lengths: gains {}, sigma_scale {}, false_beat {}, comb_floor {:.3}, detune {:.3} cents",
            base.notes.partial_gains.get(i).map_or(0, Vec::len),
            base.notes.partial_sigma_scale.get(i).map_or(0, Vec::len),
            base.notes.false_beat.get(i).map_or(0, Vec::len),
            base.notes.comb_floor[i],
            base.notes.detune_cents[i],
        );

        let ablations: Vec<Ablation> = vec![
            ("shipped", Box::new(|_: &mut Preset| {})),
            (
                "no partial_gains",
                Box::new(move |p: &mut Preset| clear_row(&mut p.notes.partial_gains, i)),
            ),
            (
                "no sigma_scale",
                Box::new(move |p: &mut Preset| clear_row(&mut p.notes.partial_sigma_scale, i)),
            ),
            (
                "no false_beat",
                Box::new(move |p: &mut Preset| clear_row(&mut p.notes.false_beat, i)),
            ),
            (
                "no comb_floor",
                Box::new(move |p: &mut Preset| p.notes.comb_floor[i] = 0.0),
            ),
            (
                "gaps filled",
                Box::new(move |p: &mut Preset| {
                    if let Some(row) = p.notes.partial_gains.get_mut(i) {
                        fill_gaps(row);
                    }
                }),
            ),
            (
                "all four off",
                Box::new(move |p: &mut Preset| {
                    clear_row(&mut p.notes.partial_gains, i);
                    clear_row(&mut p.notes.partial_sigma_scale, i);
                    clear_row(&mut p.notes.false_beat, i);
                    p.notes.comb_floor[i] = 0.0;
                }),
            ),
        ];

        println!(
            "  {:<18} {:>7} {:>10} {:>8} {:>7} {:>7}",
            "ablation", "error", "irregular", "level", "beat", "jitter"
        );
        for (name, edit) in &ablations {
            let mut preset = base.clone();
            edit(&mut preset);
            let audio = render(&preset, key);
            let mono = audio.mono();
            let levels = partial_levels(&mono, &partial_hz);
            let signal: Vec<f64> = mono.iter().map(|&v| f64::from(v)).collect();
            let motions = realism::measure_partials(&signal, &partial_hz);
            let beats: Vec<f64> = motions.iter().flatten().map(|m| m.beat_depth_db).collect();
            let jit: Vec<f64> = motions.iter().flatten().map(|m| m.floored_cents()).collect();
            println!(
                "  {:<18} {:>7.2} {:>10.2} {:>8.1} {:>7.2} {:>7.2}",
                name,
                shape_error(&levels, &reference),
                irregularity(&levels),
                amp_db(realism::rms(&mono[frame(WINDOW.0)..frame(WINDOW.1).min(mono.len())])),
                median(&beats),
                median(&jit),
            );
        }

        // The per-partial ledger, on the shipped preset, next to the row that
        // is supposed to be closing the gap.
        let audio = render(&base, key);
        let levels = partial_levels(&audio.mono(), &partial_hz);
        let offset = common_offset(&levels, &reference);
        let gains = base.notes.partial_gains.get(i).cloned().unwrap_or_default();
        println!(
            "\n  {:>2} {:>9} {:>9} {:>9} {:>9} {:>10}",
            "k", "hz", "engine", "recording", "residual", "gain dB"
        );
        for k in 0..PARTIALS {
            let g = gains.get(k).copied().unwrap_or(1.0);
            println!(
                "  {:>2} {:>9.1} {:>9.1} {:>9.1} {:>+9.1} {:>+10.1}{}",
                k + 1,
                partial_hz[k],
                levels[k],
                reference[k],
                levels[k] - offset - reference[k],
                20.0 * f64::from(g).log10(),
                if (f64::from(g) - 0.05).abs() < 1e-6 || (f64::from(g) - 20.0).abs() < 1e-6 {
                    "  <- railed"
                } else {
                    ""
                }
            );
        }
    }
    Ok(())
}

/// Replaces every unmeasured entry of a gain row — the exact `1.0` the fit
/// writes when a partial was measured on neither side — with what a degree-2
/// polynomial in `ln k` through the *measured* entries says at that partial.
///
/// This is the ablation the whole attribution turns on. If the row's damage is
/// its holes rather than its measured values, filling them is the whole repair
/// and nothing that was actually measured has to move.
fn fill_gaps(row: &mut [f32]) {
    let points: Vec<(f64, f64)> = row
        .iter()
        .enumerate()
        .filter(|(_, &g)| g != 1.0 && g > 0.0)
        .map(|(k, &g)| (((k + 1) as f64).ln(), f64::from(g).ln()))
        .collect();
    if points.len() < 4 {
        return;
    }
    let x: Vec<f64> = points.iter().map(|p| p.0).collect();
    let y: Vec<f64> = points.iter().map(|p| p.1).collect();
    let weights = vec![1.0; x.len()];
    let Some(poly) = piano_tuner::numeric::weighted_polyfit(&x, &y, &weights, 2) else {
        return;
    };
    for (k, g) in row.iter_mut().enumerate() {
        if *g == 1.0 {
            *g = (piano_tuner::numeric::poly_eval(&poly, ((k + 1) as f64).ln()).exp() as f32)
                .clamp(0.05, 20.0);
        }
    }
}

/// A named edit to the preset, applied to a fresh clone before each render.
type Ablation = (&'static str, Box<dyn Fn(&mut Preset)>);

fn clear_row<T>(table: &mut [Vec<T>], i: usize) {
    if let Some(row) = table.get_mut(i) {
        row.clear();
    }
}

fn render(preset: &Preset, key: u8) -> Audio {
    let events = [RenderEvent::new(
        PREROLL_S as f32,
        Event::NoteOn {
            key,
            vel: VELOCITY,
        },
    )];
    let (left, right) = render_to_buffer(preset, &events, (PREROLL_S + RENDER_S) as f32);
    let skip = frame(PREROLL_S);
    Audio::new(
        SAMPLE_RATE,
        vec![left[skip..].to_vec(), right[skip..].to_vec()],
    )
    .expect("the engine renders stereo")
}

fn reference_levels(
    sampler: &mut Sampler,
    key: u8,
    partial_hz: &[f64],
) -> Result<Vec<f64>, piano_tuner::Error> {
    let events = [TimedEvent::new(
        0.0,
        SamplerEvent::NoteOn {
            key,
            vel: VELOCITY,
        },
    )];
    let rendered = sampler.render(&events, RENDER_S + 0.2)?;
    let mono = rendered.mono();
    let onset = piano_tuner::detect_onset(&mono, f64::from(SAMPLE_RATE));
    let skip = (onset * f64::from(SAMPLE_RATE)).round() as usize;
    let cut: Vec<f32> = mono[skip.min(mono.len())..].to_vec();
    Ok(partial_levels(&cut, partial_hz))
}

fn frame(seconds: f64) -> usize {
    (seconds * f64::from(SAMPLE_RATE)) as usize
}

/// Level of each partial over [`WINDOW`], dB, by projection onto the partial's
/// own frequency through a Hann window with a +-3 bin skirt.
fn partial_levels(mono: &[f32], partial_hz: &[f64]) -> Vec<f64> {
    let sr = f64::from(SAMPLE_RATE);
    let lo = frame(WINDOW.0).min(mono.len());
    let hi = frame(WINDOW.1).min(mono.len());
    let window = &mono[lo..hi];
    if window.is_empty() {
        return vec![f64::NEG_INFINITY; partial_hz.len()];
    }
    let n = window.len();
    let taper: Vec<f64> = (0..n)
        .map(|i| 0.5 - 0.5 * (TAU * i as f64 / n as f64).cos())
        .collect();
    let bin = sr / n as f64;
    partial_hz
        .iter()
        .map(|&hz| {
            let mut power = 0.0;
            for d in -3..=3i32 {
                let f = hz + f64::from(d) * bin;
                if f <= 0.0 || f >= 0.45 * sr {
                    continue;
                }
                let (mut re, mut im) = (0.0f64, 0.0f64);
                let w = TAU * f / sr;
                for (i, (&s, &t)) in window.iter().zip(&taper).enumerate() {
                    let phase = w * i as f64;
                    let v = f64::from(s) * t;
                    re += v * phase.cos();
                    im -= v * phase.sin();
                }
                power += re * re + im * im;
            }
            amp_db(2.0 * power.sqrt() / n as f64)
        })
        .collect()
}

/// The gain between the two spectra: the median of the per-partial differences,
/// which no single railed partial can move.
fn common_offset(engine: &[f64], reference: &[f64]) -> f64 {
    let diffs: Vec<f64> = engine
        .iter()
        .zip(reference)
        .filter(|(a, b)| a.is_finite() && b.is_finite())
        .map(|(a, b)| a - b)
        .collect();
    median(&diffs)
}

/// Mean absolute residual after the common offset is removed, dB.
fn shape_error(engine: &[f64], reference: &[f64]) -> f64 {
    let offset = common_offset(engine, reference);
    let residuals: Vec<f64> = engine
        .iter()
        .zip(reference)
        .filter(|(a, b)| a.is_finite() && b.is_finite())
        .map(|(a, b)| (a - offset - b).abs())
        .collect();
    if residuals.is_empty() {
        return 0.0;
    }
    residuals.iter().sum::<f64>() / residuals.len() as f64
}

fn irregularity(levels_db: &[f64]) -> f64 {
    let usable: Vec<f64> = levels_db.iter().copied().filter(|d| d.is_finite()).collect();
    if usable.len() < 2 {
        return 0.0;
    }
    let steps: Vec<f64> = usable.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
    steps.iter().sum::<f64>() / steps.len() as f64
}

fn median(values: &[f64]) -> f64 {
    let mut v: Vec<f64> = values.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

fn amp_db(amp: f64) -> f64 {
    if amp > 0.0 {
        20.0 * amp.log10()
    } else {
        f64::NEG_INFINITY
    }
}

fn note_name(key: u8) -> String {
    const NAMES: [&str; 12] = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    format!("{}{}", NAMES[usize::from(key) % 12], i32::from(key) / 12 - 1)
}
