//! Where the top octave's notes end, and what ends them.
//!
//! `renders/compass/COMPASS.md` flagged keys 104-108 on `decay`, `beat` and
//! `jitter` at scores no other key came near (A7 `decay` z **-116**, engine
//! -176.0 dB/s against a neighbourhood of -19.9 and a recording of -0.7). The
//! three flags are one event: the note is switched off, and a signal that stops
//! has an infinite decay slope, an envelope span equal to the whole dynamic
//! range and a phase that is noise.
//!
//! This is the tool that says *which* threshold switches it off, on
//! measurements rather than on the docstrings. Three candidates, and the
//! columns that separate them:
//!
//! 1. **The culling floor.** [`piano_emulator::types::CULL_AMPLITUDE`] is one
//!    global number in internal units justified as "-90 dBFS at the master".
//!    `cull dBFS` is what it actually comes to at the master for each key,
//!    measured as the ratio between the full render's 0.10-1.10 s RMS and the
//!    bare string's over the same window. Over the compass it reads **-107.8 to
//!    -119.7 dBFS** — 22 dB under its own documented design point, and flat, so
//!    there is nothing for a per-bank version of the constant to fix.
//! 2. **The idle threshold.** `IDLE_ENERGY` lets `Voice::process` take the
//!    branch that writes nothing at all, so it is where a note *ends*. This is
//!    the one: with `CULL_AMPLITUDE` alone dropped to 1e-12 nothing about A7
//!    moves, and with `IDLE_ENERGY` alone dropped its `decay` goes -176.0 ->
//!    -21.4, `beat` 97.9 -> 14.4 and `jitter` 7.77 -> 1.52. (The first control
//!    run for this looked like the cull and was not: `CULL_AMPLITUDE` is also
//!    what `ResonanceBus::is_active` tests, so lowering it keeps the bus alive,
//!    keeps the string driven, and suppresses culling instead of measuring it.)
//! 3. **The T60 solve and the level calibration.** `sigma` / `T60` per mode and
//!    `peak` against `ref peak`. Neither is what switched the note off.
//!
//! **`eng -60` against `ref -60` is not a decay comparison, and this file used
//! to say it was** — "the engine's top octave is 60 dB down at 1.3-1.9 s where
//! the recordings take 3.4-3.7". Both columns are **broadband**, and at those
//! keys the recording's late broadband energy is its room and its neighbouring
//! strings: the note's own partial is 72-92 dB under the peak out there while
//! the broadband level is only ~58 dB under it. Measured on the partial
//! instead, the engine's top-octave fundamental lasts **1.5 to 2.9 times
//! longer** than the recording's, which is the opposite defect.
//! `piano-tuner brilliance` is where that measurement lives and
//! `DECISIONS.md` 293 is the refusal.
//!
//! `cull @` is the analytic crossing time of each partial's loudest and
//! longest-lived mode; `cliff @s` is the first instant after which the render is
//! identically zero; `render dBFS` and `recording dBFS` are the two envelopes at
//! the same instants, which is the comparison the audibility argument has to be
//! made on.
//!
//! ```sh
//! cargo run --release -p forensics --bin top_octave \
//!     -- data/salamander presets/salamander-c5.toml 96 108
//! ```

use std::path::PathBuf;

use piano_emulator::hammer::Hammer;
use piano_emulator::preset::Preset;
use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_emulator::string::PianoString;
use piano_emulator::types::{Event, BLOCK, CULL_AMPLITUDE};
use piano_tuner::{Audio, Sampler, SamplerEvent, TimedEvent, SAMPLE_RATE};

const VELOCITY: u8 = 90;
const RENDER_S: f32 = 3.6;
const PREROLL_S: f32 = 0.05;

/// Where the mode amplitudes are read: past the hammer pulse and past the
/// first coupled beat, but early enough that nothing has been culled yet.
const PROBE_S: f32 = 0.05;

fn db(x: f64) -> f64 {
    20.0 * x.max(1e-30).log10()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let data = PathBuf::from(args.next().unwrap_or_else(|| "data/salamander".into()));
    let preset_path = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "presets/salamander-c5.toml".into()),
    );
    let first: u8 = args.next().and_then(|a| a.parse().ok()).unwrap_or(96);
    let last: u8 = args.next().and_then(|a| a.parse().ok()).unwrap_or(108);

    let preset = Preset::load(&preset_path)?;
    let sfz = data.join("SalamanderGrandPiano-V3+20200602.sfz");
    let mut sampler = Sampler::new(&sfz).ok();

    println!("cull floor CULL_AMPLITUDE = {CULL_AMPLITUDE:.4e} (internal units)\n");
    println!(
        "{:>4} {:>7} {:>3} {:>8} {:>8} {:>9} {:>9} {:>8} {:>8}",
        "key", "f0", "np", "peak", "ref peak", "cull dBFS", "cliff @s", "eng -60", "ref -60"
    );

    for key in first..=last {
        // --- the rendered truth -------------------------------------------
        let events = [RenderEvent::new(
            PREROLL_S,
            Event::NoteOn {
                key,
                vel: u16::from(VELOCITY),
            },
        )];
        let (left, right) = render_to_buffer(&preset, &events, PREROLL_S + RENDER_S);
        let skip = (PREROLL_S * SAMPLE_RATE as f32) as usize;
        let mono: Vec<f32> = left[skip..]
            .iter()
            .zip(&right[skip..])
            .map(|(&l, &r)| 0.5 * (l + r))
            .collect();
        let peak = mono.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
        // First instant after which the render is identically zero.
        let cliff = mono
            .iter()
            .rposition(|&x| x != 0.0)
            .map(|i| (i + 1) as f64 / f64::from(SAMPLE_RATE))
            .unwrap_or(0.0);
        // The engine's own -60 dB point, by exactly the recipe the recording's
        // is taken with below: the last 20 ms window still standing 60 dB over
        // the note's peak.
        let idle = {
            let w = (0.02 * f64::from(SAMPLE_RATE)) as usize;
            let mut t60 = 0.0;
            let mut i = 0;
            while i + w <= mono.len() {
                let rms = (mono[i..i + w]
                    .iter()
                    .map(|&x| f64::from(x) * f64::from(x))
                    .sum::<f64>()
                    / w as f64)
                    .sqrt();
                if db(rms) > db(f64::from(peak)) - 60.0 {
                    t60 = (i + w) as f64 / f64::from(SAMPLE_RATE);
                }
                i += w;
            }
            t60
        };

        // --- the string on its own, mode by mode --------------------------
        let params = preset.string_params(key);
        let mut string = PianoString::new(params, &preset.voicing, preset.partial_shaping(key));
        let mut hammer = Hammer::new(preset.hammer_params(key));
        hammer.strike_midi(u16::from(VELOCITY));
        let blocks = (PROBE_S * SAMPLE_RATE as f32) as usize / BLOCK;
        let mut buf = vec![0.0f32; BLOCK];
        let mut string_peak = 0.0f32;
        for _ in 0..blocks {
            buf.fill(0.0);
            hammer.add_pulse(string.excitation_mut(), 0, 1.0);
            hammer.advance(BLOCK);
            string.process(&mut buf);
            string_peak = string_peak.max(buf.iter().fold(0.0f32, |a, &x| a.max(x.abs())));
        }

        let np = string.partial_count();

        // The dB the chain adds between a mode's internal amplitude and the
        // master, measured where it matters: the *ringing* note, not the
        // attack. The bare string is run on with no further excitation and its
        // 0.10-1.10 s RMS compared with the full render's over the same window.
        // Taking it at the attack instead would fold in the hammer's own noise
        // and the board's transient, which are not what a lone decaying mode
        // goes through.
        let mut solo = PianoString::new(params, &preset.voicing, preset.partial_shaping(key));
        let mut solo_hammer = Hammer::new(preset.hammer_params(key));
        solo_hammer.strike_midi(u16::from(VELOCITY));
        let mut bare = vec![0.0f32; (f64::from(SAMPLE_RATE) * 1.2) as usize / BLOCK * BLOCK];
        for chunk in bare.chunks_mut(BLOCK) {
            solo_hammer.add_pulse(solo.excitation_mut(), 0, 1.0);
            solo_hammer.advance(BLOCK);
            solo.process(chunk);
        }
        let win = |v: &[f32], a: f64, b: f64| -> f64 {
            let lo = (a * f64::from(SAMPLE_RATE)) as usize;
            let hi = ((b * f64::from(SAMPLE_RATE)) as usize).min(v.len());
            if hi <= lo {
                return 0.0;
            }
            (v[lo..hi]
                .iter()
                .map(|&x| f64::from(x) * f64::from(x))
                .sum::<f64>()
                / (hi - lo) as f64)
                .sqrt()
        };
        // `bare` starts at the strike; `mono` has the preroll already cut.
        let chain_db = db(win(&mono, 0.10, 1.10)) - db(win(&bare, 0.10, 1.10));
        let _ = string_peak;
        let cull_dbfs = db(f64::from(CULL_AMPLITUDE)) + chain_db;

        // --- the recording ------------------------------------------------
        let (ref_peak, ref_60) = match sampler.as_mut() {
            Some(s) => {
                let ev = [TimedEvent::new(
                    0.0,
                    SamplerEvent::NoteOn { key, vel: VELOCITY },
                )];
                let a: Audio = s.render(&ev, f64::from(RENDER_S) + 0.2)?;
                let m = a.mono();
                let p = m.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
                // When the recording's own 20 ms RMS last stands 60 dB under
                // its peak — the note's audible length in the reference.
                let win = (0.02 * f64::from(SAMPLE_RATE)) as usize;
                let mut t60 = 0.0;
                let mut i = 0;
                while i + win <= m.len() {
                    let rms = (m[i..i + win]
                        .iter()
                        .map(|&x| f64::from(x) * f64::from(x))
                        .sum::<f64>()
                        / win as f64)
                        .sqrt();
                    if db(rms) > db(f64::from(p)) - 60.0 {
                        t60 = (i + win) as f64 / f64::from(SAMPLE_RATE);
                    }
                    i += win;
                }
                // The recording's own floor: the quietest 20 ms window it has
                // anywhere in the note, which on a real recording is its noise
                // and not its silence. Nothing the engine does below this is
                // measurable against the reference at all.
                let mut floor = f64::INFINITY;
                let mut i = 0;
                while i + win <= m.len() {
                    let rms = (m[i..i + win]
                        .iter()
                        .map(|&x| f64::from(x) * f64::from(x))
                        .sum::<f64>()
                        / win as f64)
                        .sqrt();
                    floor = floor.min(db(rms));
                    i += win;
                }
                let at = |t: f64| -> f64 {
                    let lo = (t * f64::from(SAMPLE_RATE)) as usize;
                    let hi = (lo + win).min(m.len());
                    if hi <= lo {
                        return f64::NAN;
                    }
                    db((m[lo..hi]
                        .iter()
                        .map(|&x| f64::from(x) * f64::from(x))
                        .sum::<f64>()
                        / (hi - lo) as f64)
                        .sqrt())
                };
                println!(
                    "       recording dBFS  peak {:>7.1}  floor {:>7.1}  2.0s {:>7.1}  2.5s {:>7.1}  3.0s {:>7.1}  3.5s {:>7.1}",
                    db(f64::from(p)),
                    floor,
                    at(2.0),
                    at(2.5),
                    at(3.0),
                    at(3.5),
                );
                if key % 12 == 0 {
                    s.clear_cache();
                }
                (db(f64::from(p)), t60)
            }
            None => (f64::NAN, f64::NAN),
        };

        // --- the compass's own three motion metrics, same recipe -----------
        let partial_hz: Vec<f64> = (1..=8).map(|k| f64::from(params.partial_freq(k))).collect();
        let signal: Vec<f64> = mono.iter().map(|&v| f64::from(v)).collect();
        let motions = piano_tuner::realism::measure_partials(&signal, &partial_hz);
        let measured: Vec<_> = motions.iter().flatten().copied().collect();
        let med = |mut v: Vec<f64>| -> f64 {
            if v.is_empty() {
                return f64::NAN;
            }
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let n = v.len();
            if n % 2 == 1 {
                v[n / 2]
            } else {
                0.5 * (v[n / 2 - 1] + v[n / 2])
            }
        };
        let decay = med(measured.iter().map(|m| m.tail_db_s).collect());
        let beat = med(measured.iter().map(|m| m.beat_depth_db).collect());
        let jitter = med(measured.iter().map(|m| m.floored_cents()).collect());

        println!(
            "{key:>4} {:>7.1} {np:>3} {:>8.1} {:>8.1} {:>9.1} {:>9.2} {:>8.2} {:>8.2}  | decay {:>9.2} beat {:>7.2} jitter {:>6.2}",
            params.partial_freq(1),
            db(f64::from(peak)),
            ref_peak,
            cull_dbfs,
            cliff,
            idle,
            ref_60,
            decay,
            beat,
            jitter,
        );
        // What the render is actually at, in dBFS, across the motion window —
        // so a claim that the cull "cuts an audible tail" can be checked
        // against the level of the tail it cuts.
        let prof: Vec<String> = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
            .iter()
            .map(|&t| format!("{t:.1}s {:>7.1}", db(win(&mono, t, t + 0.1))))
            .collect();
        println!("       render dBFS  {}", prof.join("  "));

        // Per-partial detail.
        for k in 1..=np.min(8) {
            let amps = string.partial_amplitudes(k);
            let modes = string.partial_modes(k);
            let (loud, &a_max) = amps
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();
            let sigma = f64::from(modes[loud].sigma);
            let t60 = 6.9078 / sigma;
            // Time from the probe at which the loudest mode crosses the floor.
            let cull_at = if a_max > CULL_AMPLITUDE {
                f64::from(PROBE_S) + (f64::from(a_max / CULL_AMPLITUDE)).ln() / sigma
            } else {
                0.0
            };
            // The tail is set by whichever mode of the group survives longest,
            // which is not in general the loudest one: a horizontal mode can
            // start 20 dB down and still outlive the vertical one.
            let survive = |i: usize| -> f64 {
                let a = f64::from(amps[i]);
                let s = f64::from(modes[i].sigma).max(1e-6);
                if a > f64::from(CULL_AMPLITUDE) {
                    (a / f64::from(CULL_AMPLITUDE)).ln() / s
                } else {
                    0.0
                }
            };
            let last = (0..amps.len())
                .max_by(|&a, &b| survive(a).partial_cmp(&survive(b)).unwrap())
                .unwrap();
            println!(
                "       k{k:<2} f {:>8.1}  loud |s| {:>10.3e} ({:>7.1} dBFS) sigma {:>7.2} T60 {:>6.2} cull@ {:>6.2}  | last mode sigma {:>7.2} T60 {:>6.2} cull@ {:>6.2}",
                f64::from(modes[loud].hz),
                f64::from(a_max),
                db(f64::from(a_max)) + chain_db,
                sigma,
                t60,
                cull_at,
                f64::from(modes[last].sigma),
                6.9078 / f64::from(modes[last].sigma).max(1e-6),
                f64::from(PROBE_S) + survive(last),
            );
        }
    }
    Ok(())
}
