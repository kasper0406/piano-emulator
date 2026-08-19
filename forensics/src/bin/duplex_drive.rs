//! What actually reaches an undamped segment, in the three units it can arrive
//! in — the instrument behind `DECISIONS.md` 481.
//!
//! `DECISIONS.md` 260 left the duplex gap with a diagnosis and no arithmetic:
//! "the segments need broadband drive". *How much* broadband drive, and from
//! where, is a question about three signals whose levels nothing in the repo
//! had ever put side by side:
//!
//! * the **hammer's own force pulse** (newtons, what the string's excitation
//!   buffer receives), which is broadband;
//! * the key's **own bridge force** (the engine's signal unit, what `Voice`
//!   hands the segments), which is a line spectrum;
//! * the **resonance bus** drive, which is the same line spectra of every other
//!   key, attenuated by the coupling.
//!
//! A resonator with a bandwidth under a hertz does not care about any of their
//! *peaks*: what it integrates is the drive's transform **at its own centre
//! frequency**, `X(w) = sum_n x[n] e^{-j w n}`, and the mode's peak amplitude is
//! `g |X(w)|`. So that is what this prints, per candidate segment frequency,
//! for a single strike of one key: the three drives' `|X(w)|` in their own
//! units, the ratio between them in dB, and — with the gains the engine builds
//! — the amplitude each one leaves in the segment.
//!
//! ```sh
//! cargo run --release -p forensics --bin duplex_drive -- 72 90
//! ```

use piano_emulator::preset::Preset;
use piano_emulator::types::{key_index, BLOCK, SAMPLE_RATE};

/// Cents off the nearest partial to probe, either side: a real rear duplex is
/// tuned sharp of nominal by tens of cents (Öberg & Askenfelt), and the whole
/// question is what a drive carries *between* the partials.
const OFFSETS_CENTS: [f64; 5] = [0.0, 12.0, 25.0, 52.0, -38.0];
/// Partials whose neighbourhood is probed.
const PARTIALS: [u32; 3] = [5, 9, 15];
/// How long the strike is followed, seconds.
const WINDOW_S: f32 = 0.5;

/// `|X(w)|` of a signal at one frequency, accumulated the way a resonator does.
fn transform(x: &[f32], hz: f64) -> f64 {
    let w = std::f64::consts::TAU * hz / f64::from(SAMPLE_RATE);
    let (mut re, mut im) = (0.0f64, 0.0f64);
    for (n, &v) in x.iter().enumerate() {
        let phase = w * n as f64;
        re += f64::from(v) * phase.cos();
        im -= f64::from(v) * phase.sin();
    }
    re.hypot(im)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let key: u8 = args.next().and_then(|a| a.parse().ok()).unwrap_or(72);
    let vel: u16 = args.next().and_then(|a| a.parse().ok()).unwrap_or(90);
    let preset = Preset::default();
    let index = key_index(key).expect("a key");
    let f0 = f64::from(preset.f0(key));
    let b = f64::from(preset.notes.inharmonicity_b[index]);
    let scale = piano_emulator::string::bridge_excitation_scale_per_hz(
        &preset.string_params(key),
        &preset.voicing,
    );

    // (1) the hammer's own pulse, in newtons.
    let mut hammer = piano_emulator::hammer::Hammer::new(preset.hammer_params(key));
    hammer.strike_midi(vel);
    let pulse: Vec<f32> = hammer.pulse().to_vec();
    let area: f32 = pulse.iter().sum::<f32>() / SAMPLE_RATE;
    let peak = pulse.iter().fold(0.0f32, |m, &x| m.max(x.abs()));

    // (2) the key's own bridge force, as `Voice::process` hands it to the
    // segments: one strike, no pedal, no bus, no segments.
    let mut string = piano_emulator::string::PianoString::new(
        preset.string_params(key),
        &preset.voicing,
        preset.partial_shaping(key),
    );
    string.set_damper(0.0);
    let mut own = Vec::new();
    let mut cursor = 0usize;
    let blocks = (WINDOW_S * SAMPLE_RATE / BLOCK as f32) as usize;
    for _ in 0..blocks {
        let mut block = [0.0f32; BLOCK];
        {
            let exc = string.excitation_mut();
            for (i, e) in exc.iter_mut().enumerate() {
                if let Some(&f) = pulse.get(cursor + i) {
                    *e += f;
                }
            }
        }
        cursor += BLOCK;
        string.process(&mut block);
        own.extend_from_slice(&block);
    }
    let mut burst = vec![0.0f32; own.len()];
    let n = pulse.len().min(burst.len());
    burst[..n].copy_from_slice(&pulse[..n]);

    println!(
        "key {key} vel {vel}: f0 {f0:.2} Hz, bridge scale {scale:.4}, hammer pulse \
         peak {peak:.2} N over {} samples ({:.4} N.s), own bridge force peak {:.4}",
        pulse.len(),
        area,
        own.iter().fold(0.0f32, |m, &x| m.max(x.abs()))
    );
    println!(
        "{:>10}  {:>10}  {:>12}  {:>12}  {:>9}  {:>11}  {:>11}",
        "partial", "cents", "|X| burst", "|X| own", "burst/own", "seg(burst)", "seg(own)"
    );
    for k in PARTIALS {
        let partial = f64::from(k) * f0 * (1.0 + b * f64::from(k) * f64::from(k)).sqrt();
        for cents in OFFSETS_CENTS {
            let hz = partial * (cents / 1200.0).exp2();
            let xb = transform(&burst, hz);
            let xo = transform(&own, hz);
            // The gain the engine builds for a 0 dB segment.
            let g = f64::from(scale) * hz / f64::from(SAMPLE_RATE);
            println!(
                "{k:>10}  {cents:>+10.1}  {xb:>12.4e}  {xo:>12.4e}  {:>+9.1}  {:>11.4e}  {:>11.4e}",
                20.0 * (xb / xo).log10(),
                g * xb,
                g * xo
            );
        }
    }
}
