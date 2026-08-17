//! Where a note's 0.5-2.0 s brightness comes from, partial by partial, engine
//! against the recording of the same key — the instrument behind
//! `DECISIONS.md` 334-336.
//!
//! `melody`'s tail `hf` column is a **share**: 2-6 kHz over the whole band, so
//! one number confounds "the highs died" with "the fundamental rings too long".
//! This separates them. For every recorded key of the melody's register it
//! plays the slow line's own note — one strike at `ODE_MELODY_VEL`, held
//! `TAIL_HOLD_S` — through the engine and through the sampler, and prints per
//! partial the **level at 0.5 s** and the **fall over 0.5 -> 2.0 s** on both
//! sides, grouped by the band the correction curve is pinned in.
//!
//! ```text
//! cargo run --release -p forensics --bin tail_seam -- [preset] [key ...]
//! ```

use piano_emulator::preset::Preset;
use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_emulator::types::Event;
use piano_tuner::estimate::tail::partial_envelopes;
use piano_tuner::{SampleLibrary, Sampler, SamplerEvent, TimedEvent, SAMPLE_RATE};

const SFZ: &str = "data/salamander/SalamanderGrandPiano-V3+20200602.sfz";
const VEL: u8 = 88;
const HOLD_S: f64 = 2.2;
const DUR_S: f64 = 2.6;
const FROM_S: f64 = 0.5;
const TO_S: f64 = 2.0;
/// Frames of `partial_envelopes` per second.
const HOP_S: f64 = 0.005;

fn mono(channels: &[Vec<f32>]) -> Vec<f32> {
    (0..channels[0].len())
        .map(|i| channels.iter().map(|c| c[i]).sum::<f32>() / channels.len() as f32)
        .collect()
}

/// Median dB over a 40 ms window centred on `at_s`, from the envelope's start.
fn level_at(env: &[f64], at_s: f64) -> Option<f64> {
    let centre = (at_s / HOP_S).round() as usize;
    let half = 4usize;
    if centre + half >= env.len() {
        return None;
    }
    let mut v: Vec<f64> = env[centre.saturating_sub(half)..centre + half].to_vec();
    v.sort_by(f64::total_cmp);
    Some(v[v.len() / 2])
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let preset_path = args
        .first()
        .cloned()
        .unwrap_or_else(|| "presets/salamander-c5.toml".to_string());
    let keys: Vec<u8> = if args.len() > 1 {
        args[1..].iter().map(|a| a.parse().expect("key")).collect()
    } else {
        vec![51, 54, 57, 60, 63, 66, 69, 72, 75]
    };
    let preset = Preset::load(std::path::Path::new(&preset_path)).expect("preset");
    let library = SampleLibrary::from_sfz(SFZ).expect("library");
    let _ = library;
    let mut sampler = Sampler::new(SFZ).expect("sampler");
    let sr = f64::from(SAMPLE_RATE);

    println!("tail seam: {preset_path}, one strike at vel {VEL} held {HOLD_S} s");
    println!("levels in dB at {FROM_S} s and the fall to {TO_S} s; ratio = fall(r)/fall(e), over 1 = the engine rings too long\n");

    for key in keys {
        let params = preset.string_params(key);
        let f0 = f64::from(params.partial_freq(1));
        let hz: Vec<f64> = (1..=piano_tuner::series::PARTIALS)
            .map(|k| f64::from(params.partial_freq(k)))
            .collect();
        let events = [
            RenderEvent::new(
                0.05,
                Event::NoteOn {
                    key,
                    vel: u16::from(VEL),
                },
            ),
            RenderEvent::new((0.05 + HOLD_S) as f32, Event::NoteOff { key, vel: 64 }),
        ];
        let (l, r) = render_to_buffer(&preset, &events, DUR_S as f32);
        let engine = mono(&[l, r]);
        let ref_events = [
            TimedEvent::new(0.05, SamplerEvent::NoteOn { key, vel: VEL }),
            TimedEvent::new(0.05 + HOLD_S, SamplerEvent::NoteOff { key, vel: 64 }),
        ];
        let rendered = sampler.render(&ref_events, DUR_S).expect("render");
        let reference = mono(&rendered.channels);

        let e = partial_envelopes(&engine, &hz, f0, sr);
        let f = partial_envelopes(&reference, &hz, f0, sr);
        // The envelopes start at the first complete window, so 0.5 s from the
        // strike is 0.5 s from the note-on plus the window's own fill; both
        // signals are struck at the same time and read the same way.
        println!("key {key}  f0 {f0:.1} Hz");
        println!(
            "  {:>3} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
            "k", "Hz", "e@0.5", "r@0.5", "e rel", "r rel", "e fall", "r fall", "ratio"
        );
        let (mut e_ref, mut r_ref) = (f64::NAN, f64::NAN);
        // Band power sums, linear, at each instant: 2-6 kHz and the whole note.
        let mut sums = [[0.0f64; 2]; 4]; // [e@.5, r@.5, e@2, r@2][hf, all]
        for k in 1..=hz.len().min(48) {
            let (ee, ff) = (&e[k - 1], &f[k - 1]);
            let (Some(e0), Some(e1), Some(f0l), Some(f1)) = (
                level_at(ee, FROM_S),
                level_at(ee, TO_S),
                level_at(ff, FROM_S),
                level_at(ff, TO_S),
            ) else {
                continue;
            };
            if k == 1 {
                e_ref = e0;
                r_ref = f0l;
            }
            let hf = hz[k - 1] >= 2000.0 && hz[k - 1] < 6000.0;
            for (slot, level) in [(0usize, e0), (1, f0l), (2, e1), (3, f1)] {
                let p = 10f64.powf(level / 10.0);
                sums[slot][1] += p;
                if hf {
                    sums[slot][0] += p;
                }
            }
            if f0l < -110.0 {
                continue;
            }
            let (ef, rf) = (e0 - e1, f0l - f1);
            let ratio = if ef.abs() > 0.5 { rf / ef } else { f64::NAN };
            println!(
                "  {:>3} {:8.0} {:8.1} {:8.1} {:8.1} {:8.1} {:8.1} {:8.1} {:8.2}",
                k,
                hz[k - 1],
                e0,
                f0l,
                e0 - e_ref,
                f0l - r_ref,
                ef,
                rf,
                ratio
            );
        }
        let share = |s: [f64; 2]| 10.0 * (s[0] / s[1]).max(1e-30).log10();
        println!(
            "  hf share (2-6k over all, dB): at 0.5 s engine {:.2} reference {:.2} (error {:+.2}) | at 2.0 s engine {:.2} reference {:.2} (error {:+.2})\n",
            share(sums[0]),
            share(sums[1]),
            share(sums[0]) - share(sums[1]),
            share(sums[2]),
            share(sums[3]),
            share(sums[2]) - share(sums[3]),
        );
    }
}
