//! What a key's **level** and its **octave partial's decay** are, on the engine
//! and on the recording of the same key, and which fitted table in the preset
//! is responsible for each.
//!
//! `key_probe` answers the neighbouring question and deliberately cannot answer
//! this one: it removes the common offset between the two sides before it
//! scores them, because a gain is not a timbre. The complaint this instrument
//! was built for is the gain. A listener said *that C sounds off*; the melody
//! render puts C4 4-5 dB under the engine's own melodic trend and 8-9 dB under
//! where the recording puts its own C4, and over a held note the engine's C4
//! loses its octave partial while the recording's keeps it.
//!
//! Three quantities per render, all on the mono fold-down, all from the note's
//! own onset:
//!
//! * `head` / `whole` — A-weighted energy over 0-0.45 s and 0-2.0 s, dB. The
//!   A-weighting is the only reason a level can be compared *across* keys at
//!   all: 261.6 Hz and 392.0 Hz are 2.3 dB apart on that curve before anything
//!   about the piano is considered.
//! * `oct` — `10 log10(E(2 f0) / E(f0))` over 0-1.5 s, both partials taken
//!   through the same heterodyne the rest of the tuner uses. Positive is an
//!   octave-dominant note, which is what a real C4 is.
//! * the per-partial ledger — each of k=1..6 at 0.10 s and at 1.50 s, so that
//!   `oct` can be read as two decays rather than as one ratio.
//!
//! Every quantity is printed for the shipped preset, for one fitted table
//! removed at a time, and for the recording, so that an attribution is a
//! subtraction.
//!
//! ```sh
//! cargo run --release -p forensics --bin c4_ledger -- \
//!     data/salamander presets/salamander-c5.toml 60 62 64 65 67 51
//! ```

use std::f64::consts::TAU;
use std::path::PathBuf;

use piano_emulator::preset::Preset;
use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_emulator::types::Event;
use piano_tuner::estimate::brilliance::narrowband_db;
use piano_tuner::{realism, Audio, Sampler, SamplerEvent, TimedEvent, SAMPLE_RATE};

/// The melody's own velocity, so that this measures the note the board and the
/// listener heard rather than a different one.
const VELOCITY: u8 = realism::ODE_MELODY_VEL;
const RENDER_S: f64 = 2.6;
const PREROLL_S: f64 = 0.05;
const PARTIALS: usize = 6;
const HEAD_S: f64 = 0.45;
const WHOLE_S: f64 = 2.0;
const OCTAVE_S: f64 = 1.5;
const EARLY_S: f64 = 0.10;
const LATE_S: f64 = 1.50;
const FIRST_KEY: u8 = 21;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let data = PathBuf::from(args.next().unwrap_or_else(|| "data/salamander".into()));
    let preset_path = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "presets/salamander-c5.toml".into()),
    );
    let keys: Vec<u8> = args.filter_map(|a| a.parse().ok()).collect();
    let keys = if keys.is_empty() {
        vec![60, 62, 64, 65, 67]
    } else {
        keys
    };

    let sfz = data.join("SalamanderGrandPiano-V3+20200602.sfz");
    let base = Preset::load(&preset_path)?;
    let mut sampler = Sampler::new(&sfz)?;

    // Pass one: the shipped engine against the recording, one row per key, so
    // that the register trend both sides have can be read off one table.
    println!(
        "shipped engine vs recording, velocity {VELOCITY}\n\
         key  f0      | engine head  whole    oct | recording head  whole    oct | \
         d(head) d(whole) d(oct)"
    );
    for &key in &keys {
        let hz = partial_hz(&base, key);
        let engine = Row::of(&render(&base, key).mono(), &hz);
        let reference = match reference_mono(&mut sampler, key) {
            Ok(mono) => Some(Row::of(&mono, &hz)),
            Err(e) => {
                eprintln!("  key {key}: no recording ({e})");
                None
            }
        };
        match reference {
            Some(r) => println!(
                "{key:>3}  {:>7.2} | {:>11.2} {:>6.2} {:>+6.2} | {:>15.2} {:>6.2} {:>+6.2} | \
                 {:>+7.2} {:>+8.2} {:>+6.2}",
                hz[0],
                engine.head_db,
                engine.whole_db,
                engine.octave_db,
                r.head_db,
                r.whole_db,
                r.octave_db,
                engine.head_db - r.head_db,
                engine.whole_db - r.whole_db,
                engine.octave_db - r.octave_db,
            ),
            None => println!(
                "{key:>3}  {:>7.2} | {:>11.2} {:>6.2} {:>+6.2} | {:>15} {:>6} {:>6} |",
                hz[0], engine.head_db, engine.whole_db, engine.octave_db, "-", "-", "-"
            ),
        }
    }

    // Pass two: per key, the ablations and the per-partial ledger.
    for &key in &keys {
        let i = usize::from(key - FIRST_KEY);
        let hz = partial_hz(&base, key);
        println!(
            "\nkey {key}, f0 {:.2} Hz, {} string(s), gains {}, sigma_scale {}, comb_floor {:.3}",
            hz[0],
            base.notes.unison[i],
            base.notes.partial_gains.get(i).map_or(0, Vec::len),
            base.notes.partial_sigma_scale.get(i).map_or(0, Vec::len),
            base.notes.comb_floor[i],
        );

        let ablations: Vec<(&str, Box<dyn Fn(&mut Preset)>)> = vec![
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
                "no duplex",
                Box::new(move |p: &mut Preset| {
                    if let Some(d) = p.notes.duplex.get_mut(i) {
                        *d = Default::default();
                    }
                }),
            ),
            (
                "all of them",
                Box::new(move |p: &mut Preset| {
                    clear_row(&mut p.notes.partial_gains, i);
                    clear_row(&mut p.notes.partial_sigma_scale, i);
                    clear_row(&mut p.notes.false_beat, i);
                    p.notes.comb_floor[i] = 0.0;
                }),
            ),
        ];
        println!(
            "  {:<18} {:>7} {:>7} {:>7} | k1..k6 at {EARLY_S:.2} s -> {LATE_S:.2} s, dB",
            "ablation", "head", "whole", "oct"
        );
        for (name, edit) in &ablations {
            let mut preset = base.clone();
            edit(&mut preset);
            let mono = render(&preset, key).mono();
            let row = Row::of(&mono, &hz);
            println!(
                "  {:<18} {:>7.2} {:>7.2} {:>+7.2} | {}",
                name,
                row.head_db,
                row.whole_db,
                row.octave_db,
                row.ledger()
            );
        }
        if let Ok(mono) = reference_mono(&mut sampler, key) {
            let row = Row::of(&mono, &hz);
            println!(
                "  {:<18} {:>7.2} {:>7.2} {:>+7.2} | {}",
                "the recording",
                row.head_db,
                row.whole_db,
                row.octave_db,
                row.ledger()
            );
        }
    }
    Ok(())
}

struct Row {
    head_db: f64,
    whole_db: f64,
    octave_db: f64,
    early: Vec<f64>,
    late: Vec<f64>,
}

impl Row {
    fn of(mono: &[f32], hz: &[f64]) -> Row {
        let sr = f64::from(SAMPLE_RATE);
        let (mut early, mut late) = (Vec::new(), Vec::new());
        for &f in hz {
            let env = narrowband_db(mono, f, sr);
            early.push(at_ms(&env, EARLY_S));
            late.push(at_ms(&env, LATE_S));
        }
        Row {
            head_db: a_weighted_db(&mono[..frame(HEAD_S).min(mono.len())], sr),
            whole_db: a_weighted_db(&mono[..frame(WHOLE_S).min(mono.len())], sr),
            octave_db: octave_ratio_db(mono, hz[0], sr),
            early,
            late,
        }
    }

    fn ledger(&self) -> String {
        self.early
            .iter()
            .zip(&self.late)
            .map(|(e, l)| format!(" {e:>6.1}->{l:>6.1}"))
            .collect()
    }
}

/// `10 log10(E(2 f0) / E(f0))` over the first [`OCTAVE_S`] seconds.
///
/// Both partials come out of the same heterodyne, so the ratio is of two
/// quantities measured with one filter; and it is an *energy* ratio over the
/// window rather than a level at an instant, because the question is what the
/// note sounds like while it is held and not what it is at one moment.
fn octave_ratio_db(mono: &[f32], f0: f64, sample_rate: f64) -> f64 {
    let e1 = band_energy(mono, f0, sample_rate);
    let e2 = band_energy(mono, 2.0 * f0, sample_rate);
    10.0 * (e2 / e1.max(1e-30)).max(1e-30).log10()
}

fn band_energy(mono: &[f32], hz: f64, sample_rate: f64) -> f64 {
    let env = narrowband_db(mono, hz, sample_rate);
    let n = ((OCTAVE_S * 1000.0) as usize).min(env.len());
    if n == 0 {
        return 0.0;
    }
    env[..n]
        .iter()
        .map(|&db| 10.0f64.powf(db / 10.0))
        .sum::<f64>()
        / n as f64
}

fn at_ms(env: &[f64], seconds: f64) -> f64 {
    let i = (seconds * 1000.0) as usize;
    env.get(i).copied().unwrap_or(f64::NEG_INFINITY)
}

/// A-weighted energy of a window, dB.
///
/// Taken in the frequency domain rather than through a filter: a filter would
/// have a settling time comparable with the 0.45 s window, and the window is
/// the measurement.
fn a_weighted_db(window: &[f32], sample_rate: f64) -> f64 {
    if window.len() < 64 {
        return f64::NEG_INFINITY;
    }
    let n = window.len().next_power_of_two();
    let mut planner = rustfft::FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    let mut buf: Vec<rustfft::num_complex::Complex<f64>> = (0..n)
        .map(|i| {
            let v = window.get(i).copied().unwrap_or(0.0);
            let w = if i < window.len() {
                0.5 - 0.5 * (TAU * i as f64 / window.len() as f64).cos()
            } else {
                0.0
            };
            rustfft::num_complex::Complex::new(f64::from(v) * w, 0.0)
        })
        .collect();
    fft.process(&mut buf);
    let bin = sample_rate / n as f64;
    let mut sum = 0.0;
    for (k, c) in buf.iter().take(n / 2).enumerate().skip(1) {
        let f = k as f64 * bin;
        sum += c.norm_sqr() * a_weight(f);
    }
    10.0 * (sum / (n as f64 * window.len() as f64)).max(1e-30).log10()
}

/// IEC 61672 A-weighting, as a power gain.
fn a_weight(f: f64) -> f64 {
    let f2 = f * f;
    let num = 12194.0f64.powi(2) * f2 * f2;
    let den = (f2 + 20.6f64.powi(2))
        * ((f2 + 107.7f64.powi(2)) * (f2 + 737.9f64.powi(2))).sqrt()
        * (f2 + 12194.0f64.powi(2));
    let a = num / den.max(1e-30);
    // +2.0 dB normalisation so that 1 kHz reads unity, as the standard does.
    a * a * 10.0f64.powf(0.2)
}

fn partial_hz(preset: &Preset, key: u8) -> Vec<f64> {
    let params = preset.string_params(key);
    (1..=PARTIALS)
        .map(|k| f64::from(params.partial_freq(k)))
        .collect()
}

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
            vel: u16::from(VELOCITY),
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

fn frame(seconds: f64) -> usize {
    (seconds * f64::from(SAMPLE_RATE)) as usize
}
