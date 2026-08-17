//! Independent verification of the per-channel repair (`DECISIONS.md` 392-394).
//!
//! Own code on purpose: everything here is computed from fresh renders with
//! this file's own spectra, not the gate's accumulators. Run once on the
//! fixed tree (`after`) and once with `engine/src/soundboard.rs` stashed
//! (`before`); the tag is only a label on the printout.
//!
//! ```text
//! cargo run --release -p forensics --bin channel_verify -- <tag> <section>
//! sections: notch  noise  melody  image  perf  all
//! ```

use std::path::Path;
use std::time::Instant;

use piano_emulator::preset::{MicVoicing, Preset};
use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_emulator::types::Event;
use piano_tuner::audio::Audio;
use piano_tuner::estimate::melody::{self, Window};
use piano_tuner::realism::{self, Phrase};
use piano_tuner::sampler::{engine_events, SamplerEvent, SAMPLER_VERSION};
use piano_tuner::{cache, Sampler, TimedEvent, SAMPLE_RATE};

use rustfft::num_complex::Complex32;
use rustfft::FftPlanner;

const SFZ: &str = "data/salamander/SalamanderGrandPiano-V3+20200602.sfz";
const DATA: &str = "data/salamander";
const KEYS: [u8; 6] = [54, 57, 60, 63, 66, 69];
const RENDER_S: f64 = 3.0;
const PREROLL: usize = realism::STEREO_PREROLL_SAMPLES;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let tag = args.first().cloned().unwrap_or_else(|| "run".into());
    let section = args.get(1).cloned().unwrap_or_else(|| "all".into());
    let preset = Preset::load(Path::new("presets/salamander-c5.toml"))?;
    println!("### channel_verify [{tag}] ###\n");
    if section == "notch" || section == "all" {
        notch(&preset)?;
    }
    if section == "noise" || section == "all" {
        noise(&preset)?;
    }
    if section == "melody" || section == "all" {
        melody_section(&preset)?;
    }
    if section == "image" || section == "all" {
        image(&preset)?;
    }
    if section == "perf" || section == "all" {
        perf(&preset);
    }
    if section == "phase" {
        phase(&preset)?;
    }
    Ok(())
}

/// Interchannel cross-spectrum phase across the lobe's band: how fast the
/// angle between the two loudspeakers turns with frequency. A real AB pair is
/// a short delay (a phase slope of a fraction of a millisecond); a nodal line
/// is a flat 180. Nothing in the repository scores this.
fn phase(preset: &Preset) -> Result<(), Box<dyn std::error::Error>> {
    println!("== 6. interchannel phase across 140-560 Hz ==\n");
    println!("unwrapped cross-spectrum phase, total rotation over the span (degrees),");
    println!("and the worst 1/6-octave-local slope expressed as group delay (ms).\n");
    println!(
        "{:<5} {:<10} {:>12} {:>14}",
        "key", "take", "rot_deg", "worst_gd_ms"
    );
    const N: usize = 16384;
    for &key in &[57u8, 60, 63] {
        let reference = reference_key(key, 90)?;
        let (el, er) = render_key(preset, key, 90);
        for (name, l, r) in [
            ("reference", &reference.channels[0], &reference.channels[1]),
            ("engine", &el, &er),
        ] {
            let mut planner = FftPlanner::<f32>::new();
            let fft = planner.plan_fft_forward(N);
            let window: Vec<f32> = (0..N)
                .map(|i| 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / N as f64).cos() as f32)
                .collect();
            let mut cross = vec![(0.0f64, 0.0f64); N / 2 + 1];
            let mut start = 0usize;
            while start + N <= l.len().min(r.len()) {
                let mut a: Vec<Complex32> = (0..N)
                    .map(|i| Complex32::new(l[start + i] * window[i], 0.0))
                    .collect();
                let mut b: Vec<Complex32> = (0..N)
                    .map(|i| Complex32::new(r[start + i] * window[i], 0.0))
                    .collect();
                fft.process(&mut a);
                fft.process(&mut b);
                for (slot, (x, y)) in cross.iter_mut().zip(a.iter().zip(b.iter())) {
                    let c = x * y.conj();
                    slot.0 += f64::from(c.re);
                    slot.1 += f64::from(c.im);
                }
                start += N / 2;
            }
            // Sixth-octave points 140-560 Hz, phase unwrapped point to point.
            let mut f = 140.0f64;
            let half = 2f64.powf(1.0 / 12.0);
            let mut prev: Option<f64> = None;
            let mut total = 0.0f64;
            let mut worst_gd = 0.0f64;
            let mut prev_f = f;
            while f <= 560.0 {
                let (mut re, mut im) = (0.0, 0.0);
                for (i, c) in cross.iter().enumerate() {
                    let g = bin_hz(i) * 16384.0 / N as f64;
                    if g >= f / half && g < f * half {
                        re += c.0;
                        im += c.1;
                    }
                }
                let mut ph = im.atan2(re);
                if let Some(p) = prev {
                    while ph - p > std::f64::consts::PI {
                        ph -= std::f64::consts::TAU;
                    }
                    while ph - p < -std::f64::consts::PI {
                        ph += std::f64::consts::TAU;
                    }
                    total += ph - p;
                    let gd = (ph - p).abs() / (std::f64::consts::TAU * (f - prev_f));
                    worst_gd = worst_gd.max(gd);
                }
                prev = Some(ph);
                prev_f = f;
                f *= 2f64.powf(1.0 / 6.0);
            }
            println!(
                "{:<5} {:<10} {:>12.0} {:>14.2}",
                melody::note_name(key),
                name,
                total.to_degrees(),
                worst_gd * 1e3
            );
        }
    }
    println!();
    Ok(())
}

fn panpot(preset: &Preset) -> Preset {
    let mut p = preset.clone();
    p.voicing.mics = None;
    p
}

fn pair_only(preset: &Preset) -> Preset {
    let mut p = preset.clone();
    if let Some(m) = preset.voicing.mics {
        p.voicing.mics = Some(MicVoicing { modal: None, ..m });
    }
    p
}

fn render_key(preset: &Preset, key: u8, vel: u8) -> (Vec<f32>, Vec<f32>) {
    let preroll_s = PREROLL as f32 / SAMPLE_RATE as f32;
    let events = [RenderEvent::new(
        preroll_s,
        Event::NoteOn {
            key,
            vel: u16::from(vel),
        },
    )];
    let (l, r) = render_to_buffer(preset, &events, preroll_s + RENDER_S as f32);
    (l[PREROLL..].to_vec(), r[PREROLL..].to_vec())
}

fn reference_key(key: u8, vel: u8) -> Result<Audio, Box<dyn std::error::Error>> {
    let sfz = Path::new(SFZ);
    let mut print = cache::Fingerprint::new();
    print
        .str("forensics/channel_verify/reference")
        .u64(u64::from(SAMPLER_VERSION))
        .file(sfz)?
        .u64(u64::from(SAMPLE_RATE))
        .u64(u64::from(key))
        .u64(u64::from(vel))
        .f64(RENDER_S);
    let dir = cache::reference_dir(Path::new(DATA));
    let path = dir.join(format!("cv-key{key:03}-v{vel:03}-{}.wav", print.hex()));
    let audio = cache::audio(&path, || {
        let mut sampler = Sampler::new(sfz)?;
        let events = [TimedEvent::new(0.0, SamplerEvent::NoteOn { key, vel })];
        let rendered = sampler.render(&events, RENDER_S + 0.2)?;
        let mono = rendered.mono();
        let onset = piano_tuner::detect_onset(&mono, f64::from(SAMPLE_RATE));
        let skip = (onset * f64::from(SAMPLE_RATE)).round() as usize;
        let frames = (RENDER_S * f64::from(SAMPLE_RATE)) as usize;
        let channels: Vec<Vec<f32>> = rendered
            .channels
            .iter()
            .map(|c| {
                (0..frames)
                    .map(|n| c.get(skip + n).copied().unwrap_or(0.0))
                    .collect()
            })
            .collect();
        Audio::new(SAMPLE_RATE, channels)
    })?;
    Ok(audio)
}

fn render_phrase(preset: &Preset, phrase: &Phrase) -> Audio {
    let events: Vec<RenderEvent> = engine_events::to_render_events(&phrase.events);
    let (l, r) = render_to_buffer(preset, &events, phrase.duration_s as f32);
    Audio::new(SAMPLE_RATE, vec![l, r]).expect("stereo")
}

fn reference_phrase(phrase: &Phrase) -> Result<Audio, Box<dyn std::error::Error>> {
    let sfz = Path::new(SFZ);
    let mut key = cache::Fingerprint::new();
    key.str("tests/melody/reference")
        .u64(u64::from(SAMPLER_VERSION))
        .file(sfz)?
        .u64(u64::from(SAMPLE_RATE))
        .str(phrase.name)
        .str("reference")
        .f64(phrase.duration_s);
    let path = cache::reference_dir(Path::new(DATA))
        .join(format!("melody-{}-reference-{}.wav", phrase.name, key.hex()));
    let rendered = cache::audio(&path, || {
        let mut sampler = Sampler::new(sfz)?;
        sampler.render(&phrase.events, phrase.duration_s)
    })?;
    Ok(melody::align_reference(&rendered, phrase.events[0].time_s))
}

// ---------------------------------------------------------------------------
// Spectra
// ---------------------------------------------------------------------------

/// Welch power spectrum, 16384-point Hann, half overlap: 2.9 Hz bins, fine
/// enough that a notch a few Hz wide in the low mids is not smoothed away.
fn power_spectrum(signal: &[f32]) -> Vec<f64> {
    const N: usize = 16384;
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N);
    let window: Vec<f32> = (0..N)
        .map(|i| 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / N as f64).cos() as f32)
        .collect();
    let mut acc = vec![0.0f64; N / 2 + 1];
    let mut frames = 0usize;
    let mut start = 0usize;
    while start + N <= signal.len() {
        let mut buffer: Vec<Complex32> = (0..N)
            .map(|i| Complex32::new(signal[start + i] * window[i], 0.0))
            .collect();
        fft.process(&mut buffer);
        for (slot, c) in acc.iter_mut().zip(buffer.iter().take(N / 2 + 1)) {
            *slot += f64::from(c.norm_sqr());
        }
        frames += 1;
        start += N / 2;
    }
    if frames > 0 {
        acc.iter_mut().for_each(|s| *s /= frames as f64);
    }
    acc
}

fn bin_hz(i: usize) -> f64 {
    i as f64 * f64::from(SAMPLE_RATE) / 16384.0
}

fn band_energy(power: &[f64], lo: f64, hi: f64) -> f64 {
    power
        .iter()
        .enumerate()
        .filter(|(i, _)| {
            let f = bin_hz(*i);
            f >= lo && f < hi
        })
        .map(|(_, &p)| p)
        .sum::<f64>()
        .max(1e-300)
}

/// Sixth-octave smoothed dB curve of `power` between `lo` and `hi`.
fn sixth_octave(power: &[f64], lo: f64, hi: f64) -> Vec<(f64, f64)> {
    let mut out = Vec::new();
    let mut f = lo;
    let step = 2f64.powf(1.0 / 6.0);
    let half = 2f64.powf(1.0 / 12.0);
    while f <= hi {
        let e = band_energy(power, f / half, f * half);
        let n = power
            .iter()
            .enumerate()
            .filter(|(i, _)| {
                let g = bin_hz(*i);
                g >= f / half && g < f * half
            })
            .count()
            .max(1);
        out.push((f, 10.0 * (e / n as f64).log10()));
        f *= step;
    }
    out
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn percentile(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let i = ((v.len() - 1) as f64 * p).round() as usize;
    v[i]
}

// ---------------------------------------------------------------------------
// 1. Per-channel notch statistics, engine vs reference
// ---------------------------------------------------------------------------

fn notch(preset: &Preset) -> Result<(), Box<dyn std::error::Error>> {
    println!("== 1. per-channel notch statistics, 100-900 Hz ==\n");
    println!("(a) the mic stage's own per-channel response: sixth-octave");
    println!("    10log10(P_engine/P_panpot) per channel; `depth` = median - min over");
    println!("    100-900 Hz (a notch the stage itself carved), `at` its frequency.\n");
    let flat = panpot(preset);
    println!(
        "{:<5} {:<4} {:>9} {:>9} {:>7} {:>9} {:>9}",
        "key", "ch", "median", "min", "at_Hz", "depth_dB", "max_dB"
    );
    for &key in &KEYS {
        let (el, er) = render_key(preset, key, 90);
        let (pl, pr) = render_key(&flat, key, 90);
        for (ch, e, p) in [("L", &el, &pl), ("R", &er, &pr)] {
            let a = sixth_octave(&power_spectrum(e), 100.0, 900.0);
            let b = sixth_octave(&power_spectrum(p), 100.0, 900.0);
            let curve: Vec<(f64, f64)> = a
                .iter()
                .zip(&b)
                .map(|(x, y)| (x.0, x.1 - y.1))
                .collect();
            let vals: Vec<f64> = curve.iter().map(|c| c.1).collect();
            let med = median(&mut vals.clone());
            let (at, min) = curve
                .iter()
                .fold((0.0, f64::MAX), |acc, c| if c.1 < acc.1 { (c.0, c.1) } else { acc });
            let max = curve.iter().map(|c| c.1).fold(f64::MIN, f64::max);
            println!(
                "{:<5} {:<4} {:>9.2} {:>9.2} {:>7.0} {:>9.2} {:>9.2}",
                melody::note_name(key),
                ch,
                med,
                min,
                at,
                med - min,
                max
            );
        }
    }
    println!();
    println!("(b) interchannel |L-R| statistics, engine vs the real AB pair, sixth-octave");
    println!("    | |L|-|R| | in dB over 100-900 Hz: median, p90, max. The recording's own");
    println!("    is the target (DECISIONS.md 393: 2.5-4.5 dB median, p90 8-16).\n");
    println!(
        "{:<5} {:<10} {:>8} {:>8} {:>8}",
        "key", "take", "median", "p90", "max"
    );
    for &key in &KEYS {
        let reference = reference_key(key, 90)?;
        let (el, er) = render_key(preset, key, 90);
        for (name, l, r) in [
            ("reference", &reference.channels[0], &reference.channels[1]),
            ("engine", &el, &er),
        ] {
            let a = sixth_octave(&power_spectrum(l), 100.0, 900.0);
            let b = sixth_octave(&power_spectrum(r), 100.0, 900.0);
            let mut diff: Vec<f64> = a.iter().zip(&b).map(|(x, y)| (x.1 - y.1).abs()).collect();
            println!(
                "{:<5} {:<10} {:>8.2} {:>8.2} {:>8.2}",
                melody::note_name(key),
                name,
                median(&mut diff.clone()),
                percentile(&mut diff.clone(), 0.9),
                percentile(&mut diff, 1.0)
            );
        }
    }
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// 2. Noise-to-tone per channel, three keys x three velocities
// ---------------------------------------------------------------------------

fn noise(preset: &Preset) -> Result<(), Box<dyn std::error::Error>> {
    println!("== 2. attack noise-to-tone per channel, 3 keys x 3 velocities ==\n");
    println!("noise_to_tone_db (estimate::attack) on each channel and the mono sum;");
    println!("`dev` = engine - reference. Positive dev = engine noisier there.\n");
    println!(
        "{:<5} {:>4} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} | {:>7} {:>7} {:>7}",
        "key", "vel", "ref L", "ref R", "ref M", "eng L", "eng R", "eng M", "devL", "devR", "devM"
    );
    let ntt =
        |s: &[f32]| piano_tuner::estimate::attack::noise_to_tone_db(s, 0.0, f64::from(SAMPLE_RATE));
    for &key in &[54u8, 60, 66] {
        for &vel in &[45u8, 90, 115] {
            let reference = reference_key(key, vel)?;
            let (el, er) = render_key(preset, key, vel);
            let em: Vec<f32> = el.iter().zip(&er).map(|(a, b)| 0.5 * (a + b)).collect();
            let rm = reference.mono();
            let (rl, rr) = (&reference.channels[0], &reference.channels[1]);
            let vals = [
                ntt(rl),
                ntt(rr),
                ntt(&rm),
                ntt(&el),
                ntt(&er),
                ntt(&em),
            ];
            println!(
                "{:<5} {:>4} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8.2} | {:>7.2} {:>7.2} {:>7.2}",
                melody::note_name(key),
                vel,
                vals[0],
                vals[1],
                vals[2],
                vals[3],
                vals[4],
                vals[5],
                vals[3] - vals[0],
                vals[4] - vals[1],
                vals[5] - vals[2],
            );
        }
    }
    println!();
    println!("worse-channel dev per key/vel is the number `piano-tuner noise` now reports.\n");
    Ok(())
}

// ---------------------------------------------------------------------------
// 3. C4 per-channel evenness on the melody line
// ---------------------------------------------------------------------------

fn melody_section(preset: &Preset) -> Result<(), Box<dyn std::error::Error>> {
    println!("== 3. the melody line: per-note pair-over-mono and loudness evenness ==\n");
    let line = melody::line_for(Window::Head);
    let notes = melody::line_notes_for(Window::Head);
    let engine = render_phrase(preset, &line);
    let pair = render_phrase(&pair_only(preset), &line);
    let reference = reference_phrase(&line)?;
    let sr = f64::from(SAMPLE_RATE);
    println!("per-note channel metric 10log10((E_L+E_R)/2E_M) over 30-300 ms of each");
    println!("strike, median per key, and per-note PAIR level departure from the line's");
    println!("Theil-Sen trend (loudness evenness — what `stands out` means).\n");
    let takes: Vec<(&str, &Audio)> = vec![
        ("reference", &reference),
        ("engine", &engine),
        ("pair-only", &pair),
    ];
    print!("{:<22}", "take");
    let keys = melody::line_keys();
    for &k in &keys {
        print!(" {:>13}", melody::note_name(k));
    }
    println!("  {:>7}", "spread");
    for (name, audio) in &takes {
        let detect = audio.mono();
        // channel metric per note
        let mut ch: Vec<(u8, Vec<f64>)> = Vec::new();
        let mut lvl: Vec<(u8, Vec<f64>)> = Vec::new();
        for note in notes.iter().filter(|n| n.measurable()) {
            let strike = melody::note_onset(&detect, sr, note.onset_s);
            let lo = ((strike + 0.03) * sr) as usize;
            let hi = (((strike + 0.30) * sr) as usize).min(audio.channels[0].len());
            if hi <= lo {
                continue;
            }
            let (l, r) = (&audio.channels[0][lo..hi], &audio.channels[1][lo..hi]);
            let el: f64 = l.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
            let er: f64 = r.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
            let em: f64 = l
                .iter()
                .zip(r)
                .map(|(&a, &b)| {
                    let m = 0.5 * (f64::from(a) + f64::from(b));
                    m * m
                })
                .sum();
            let c = 10.0 * ((el + er) / (2.0 * em.max(1e-300))).log10();
            let level = 10.0 * ((el + er) / (hi - lo) as f64).max(1e-300).log10();
            for (store, v) in [(&mut ch, c), (&mut lvl, level)] {
                match store.iter_mut().find(|(k, _)| *k == note.key) {
                    Some((_, list)) => list.push(v),
                    None => store.push((note.key, vec![v])),
                }
            }
        }
        for (label, store, trend) in [("chan", &mut ch, false), ("level", &mut lvl, true)] {
            store.sort_by_key(|(k, _)| *k);
            let medians: Vec<(u8, f64)> = store
                .iter_mut()
                .map(|(k, v)| (*k, median(v)))
                .collect();
            let shown: Vec<f64> = if trend {
                let points: Vec<(f64, f64)> =
                    medians.iter().map(|(k, x)| (f64::from(*k), *x)).collect();
                let (slope, intercept) = melody::theil_sen(&points);
                medians
                    .iter()
                    .map(|(k, x)| x - (intercept + slope * f64::from(*k)))
                    .collect()
            } else {
                medians.iter().map(|(_, x)| *x).collect()
            };
            print!("{:<22}", format!("{name} {label}"));
            for v in &shown {
                print!(" {:>13.2}", v);
            }
            let spread = shown.iter().cloned().fold(f64::MIN, f64::max)
                - shown.iter().cloned().fold(f64::MAX, f64::min);
            println!("  {:>7.2}", spread);
        }
    }
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// 4. Image, mono bound, determinism
// ---------------------------------------------------------------------------

fn image(preset: &Preset) -> Result<(), Box<dyn std::error::Error>> {
    println!("== 4. coherence table, mono fold-down, determinism ==\n");
    let flat = panpot(preset);
    for phrase in [melody::line_for(Window::Head), realism::chords_pedal()] {
        let engine = render_phrase(preset, &phrase);
        let reference = reference_phrase(&phrase)?;
        let ei = realism::stereo_image_of(&engine)?;
        let ri = realism::stereo_image_of(&reference)?;
        println!("phrase `{}` r0 per band (engine / reference):", phrase.name);
        print!("  ");
        for (b, &(name, _, _)) in realism::STEREO_BANDS.iter().enumerate() {
            print!(
                "{}: {:+.3}/{:+.3}  ",
                name, ei.bands[b].r0, ri.bands[b].r0
            );
        }
        println!();
        print!("  mid/side dB          : ");
        for (b, _) in realism::STEREO_BANDS.iter().enumerate() {
            print!(
                "{:+.1}/{:+.1}  ",
                ei.bands[b].mid_side_db, ri.bands[b].mid_side_db
            );
        }
        println!();
        // Mono fold-down bound against the pan-pot.
        let pan = render_phrase(&flat, &phrase);
        let em: Vec<f32> = engine.channels[0]
            .iter()
            .zip(&engine.channels[1])
            .map(|(a, b)| 0.5 * (a + b))
            .collect();
        let pm: Vec<f32> = pan.channels[0]
            .iter()
            .zip(&pan.channels[1])
            .map(|(a, b)| 0.5 * (a + b))
            .collect();
        let worst = em
            .iter()
            .zip(&pm)
            .map(|(a, b)| f64::from(a - b).abs())
            .fold(0.0f64, f64::max);
        let peak = em.iter().map(|&x| f64::from(x).abs()).fold(0.0f64, f64::max);
        println!(
            "  mono fold-down vs pan-pot: worst sample |diff| = {:.3e} (peak {:.3}) -> {:.1} dB down\n",
            worst,
            peak,
            20.0 * (worst / peak.max(1e-30)).log10()
        );
    }
    // Determinism: the same phrase rendered twice in-process must be identical.
    let line = melody::line_for(Window::Head);
    let a = render_phrase(preset, &line);
    let b = render_phrase(preset, &line);
    let same = a.channels == b.channels;
    // A stable digest for cross-process comparison.
    let mut hash: u64 = 0xcbf29ce484222325;
    for c in &a.channels {
        for &x in c {
            hash ^= u64::from(x.to_bits());
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    println!("determinism: two in-process renders identical = {same}; digest = {hash:016x}\n");
    Ok(())
}

fn perf(preset: &Preset) {
    let start = Instant::now();
    let mut sink = 0.0f32;
    for _ in 0..3 {
        let (l, _r) = render_key(preset, 60, 90);
        sink += l[1000];
    }
    let per = start.elapsed().as_secs_f64() / 3.0;
    println!(
        "== 5. perf == three 3 s C4 renders: {:.3} s per render (sink {sink:e})\n",
        per
    );
}
