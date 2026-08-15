//! Independent verification harness. Own metric implementations on purpose:
//! Blackman window instead of Hann, peak-search per partial instead of a fixed
//! skirt at the engine's nominal frequency, its own presence rule. Not part of
//! the shipped tooling; written to cross-check the compass/fit reports.
//!
//! Modes:
//!   solo      - per-key metrics (engine vs recording) + neighbour residuals
//!   neutral   - level moved by each fitted partial_gains row (with vs without)
//!   ff        - every key at vel 127 on both presets: channel peak + limiter samples
//!   treble    - envelope of keys 100-108: 2-3 s slope, zero-sample check, engine+recording
//!   idle      - bit-exact silence of an event-free render
//!   chord     - DECISIONS 42 clauses on both presets
//!
//! cargo run --release -p piano-tuner --example verify_scan -- <mode>

#![allow(clippy::type_complexity)]

use std::f64::consts::TAU;
use std::path::PathBuf;

use piano_emulator::preset::Preset;
use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_emulator::soundboard::LIMIT_THRESHOLD;
use piano_emulator::types::Event;
use piano_tuner::{Sampler, SamplerEvent, TimedEvent, SAMPLE_RATE};

const VEL: u8 = 90;
const PREROLL_S: f64 = 0.05;
const RENDER_S: f64 = 3.6;
const FIRST_KEY: u8 = 21;
const LAST_KEY: u8 = 108;

fn db(x: f64) -> f64 {
    20.0 * x.max(1e-15).log10()
}

fn engine_note(preset: &Preset, key: u8, vel: u8, seconds: f64) -> Vec<f32> {
    let events = [RenderEvent::new(PREROLL_S as f32, Event::NoteOn { key, vel })];
    let (l, r) = render_to_buffer(preset, &events, (PREROLL_S + seconds) as f32);
    let skip = (PREROLL_S * f64::from(SAMPLE_RATE)) as usize;
    l[skip..]
        .iter()
        .zip(&r[skip..])
        .map(|(&a, &b)| 0.5 * (a + b))
        .collect()
}

fn reference_note(sampler: &mut Sampler, key: u8, vel: u8, seconds: f64) -> Vec<f32> {
    let events = [TimedEvent::new(0.0, SamplerEvent::NoteOn { key, vel })];
    let rendered = sampler.render(&events, seconds + 0.2).expect("sampler");
    let mono = rendered.mono();
    let onset = piano_tuner::detect_onset(&mono, f64::from(SAMPLE_RATE));
    let skip = (onset * f64::from(SAMPLE_RATE)).round() as usize;
    let frames = (seconds * f64::from(SAMPLE_RATE)) as usize;
    (0..frames)
        .map(|n| mono.get(skip + n).copied().unwrap_or(0.0))
        .collect()
}

fn rms_db(x: &[f32], sr: f64, t0: f64, t1: f64) -> f64 {
    let lo = ((t0 * sr) as usize).min(x.len());
    let hi = ((t1 * sr) as usize).min(x.len());
    if hi <= lo {
        return f64::NEG_INFINITY;
    }
    let w = &x[lo..hi];
    db((w.iter().map(|&s| f64::from(s) * f64::from(s)).sum::<f64>() / w.len() as f64).sqrt())
}

/// Windowed projection at one frequency: Blackman taper, power at `hz`.
fn power_at(win: &[f32], taper: &[f64], sr: f64, hz: f64) -> f64 {
    if hz <= 0.0 || hz >= 0.47 * sr {
        return 0.0;
    }
    let (mut re, mut im) = (0.0f64, 0.0f64);
    let w = TAU * hz / sr;
    for (i, (&s, &t)) in win.iter().zip(taper).enumerate() {
        let ph = w * i as f64;
        let v = f64::from(s) * t;
        re += v * ph.cos();
        im -= v * ph.sin();
    }
    re * re + im * im
}

/// Level of a partial: peak of the projection over a +-45-cent grid around the
/// nominal frequency, then power integrated over the found peak +-2 bins.
/// Returns (level_db, floor_db, found_hz).
fn partial_level(win: &[f32], taper: &[f64], sr: f64, hz: f64, spacing: f64) -> (f64, f64, f64) {
    let bin = sr / win.len() as f64;
    let lo = hz * (2.0f64).powf(-45.0 / 1200.0);
    let hi = hz * (2.0f64).powf(45.0 / 1200.0);
    // Never search past the midpoint to a neighbouring partial.
    let lo = lo.max(hz - 0.45 * spacing);
    let hi = hi.min(hz + 0.45 * spacing);
    let steps = (((hi - lo) / (0.5 * bin)).ceil() as usize).clamp(1, 60);
    let mut best = (0.0f64, hz);
    for i in 0..=steps {
        let f = lo + (hi - lo) * i as f64 / steps as f64;
        let p = power_at(win, taper, sr, f);
        if p > best.0 {
            best = (p, f);
        }
    }
    let mut power = 0.0;
    for d in -2..=2i32 {
        power += power_at(win, taper, sr, best.1 + f64::from(d) * bin);
    }
    let n = win.len() as f64;
    let level = db(2.0 * power.sqrt() / n);
    // Local floor midway to each neighbour, lower side.
    let f1 = power_at(win, taper, sr, hz - 0.45 * spacing);
    let f2 = power_at(win, taper, sr, hz + 0.45 * spacing);
    let floor = db(2.0 * f1.min(f2).sqrt() / n);
    (level, floor, best.1)
}

struct Spectrum {
    levels: Vec<f64>,
    present: Vec<bool>,
}

fn spectrum(mono: &[f32], sr: f64, partial_hz: &[f64]) -> Spectrum {
    let lo = (0.12 * sr) as usize;
    let hi = ((1.12 * sr) as usize).min(mono.len());
    let win = &mono[lo..hi];
    let n = win.len();
    // Blackman.
    let taper: Vec<f64> = (0..n)
        .map(|i| {
            let x = TAU * i as f64 / n as f64;
            0.42 - 0.5 * x.cos() + 0.08 * (2.0 * x).cos()
        })
        .collect();
    let spacing = partial_hz[0];
    let mut levels = Vec::new();
    let mut present = Vec::new();
    for &hz in partial_hz {
        if hz >= 0.47 * sr {
            levels.push(f64::NEG_INFINITY);
            present.push(false);
            continue;
        }
        let (l, floor, _) = partial_level(win, &taper, sr, hz, spacing);
        levels.push(l);
        present.push(l.is_finite() && l - floor >= 10.0);
    }
    Spectrum { levels, present }
}

impl Spectrum {
    fn sounding(&self) -> Vec<(usize, f64)> {
        self.levels
            .iter()
            .zip(&self.present)
            .enumerate()
            .filter(|(_, (l, &p))| p && l.is_finite())
            .map(|(i, (&l, _))| (i + 1, l))
            .collect()
    }
    fn irregular(&self) -> f64 {
        let s = self.sounding();
        let steps: Vec<f64> = s
            .windows(2)
            .filter(|w| w[1].0 - w[0].0 <= 2)
            .map(|w| (w[1].1 - w[0].1).abs())
            .collect();
        if steps.is_empty() {
            0.0
        } else {
            steps.iter().sum::<f64>() / steps.len() as f64
        }
    }
    fn centroid(&self) -> f64 {
        let (mut num, mut den) = (0.0, 0.0);
        for (k, l) in self.sounding() {
            let p = 10f64.powf(l / 10.0);
            num += p * k as f64;
            den += p;
        }
        if den <= 0.0 {
            0.0
        } else {
            12.0 * (num / den).log2()
        }
    }
    fn match_db(&self, other: &Spectrum) -> f64 {
        let mut diffs: Vec<f64> = self
            .levels
            .iter()
            .zip(&self.present)
            .zip(other.levels.iter().zip(&other.present))
            .filter(|((a, &pa), (b, &pb))| pa && pb && a.is_finite() && b.is_finite())
            .map(|((&a, _), (&b, _))| a - b)
            .collect();
        if diffs.len() < 3 {
            return f64::NAN;
        }
        diffs.sort_by(f64::total_cmp);
        let med = diffs[diffs.len() / 2];
        diffs.iter().map(|d| (d - med).abs()).sum::<f64>() / diffs.len() as f64
    }
}

fn partial_freqs(preset: &Preset, key: u8, n: usize) -> Vec<f64> {
    let params = preset.string_params(key);
    (1..=n).map(|k| f64::from(params.partial_freq(k))).collect()
}

fn fitted_keys(preset: &Preset) -> Vec<u8> {
    (FIRST_KEY..=LAST_KEY)
        .filter(|&k| !preset.notes.partial_gains[usize::from(k - FIRST_KEY)].is_empty())
        .collect()
}

fn mode_solo() {
    let preset = Preset::load(std::path::Path::new("presets/salamander-c5.toml")).expect("preset");
    let sfz = PathBuf::from("data/salamander/SalamanderGrandPiano-V3+20200602.sfz");
    let mut sampler = Sampler::new(&sfz).expect("sampler");
    let sr = f64::from(SAMPLE_RATE);
    let fitted = fitted_keys(&preset);

    println!("key fit  lvl_e   lvl_r   irr_e irr_r  cen_e cen_r  match");
    let mut rows: Vec<(u8, bool, f64, f64, f64, f64, f64, f64, f64)> = Vec::new();
    for key in FIRST_KEY..=LAST_KEY {
        let hz = partial_freqs(&preset, key, 12);
        let e = engine_note(&preset, key, VEL, RENDER_S);
        let r = reference_note(&mut sampler, key, VEL, RENDER_S);
        if key % 12 == 0 {
            sampler.clear_cache();
        }
        let se = spectrum(&e, sr, &hz);
        let sre = spectrum(&r, sr, &hz);
        let (le, lr) = (rms_db(&e, sr, 0.10, 1.10), rms_db(&r, sr, 0.10, 1.10));
        let row = (
            key,
            fitted.contains(&key),
            le,
            lr,
            se.irregular(),
            sre.irregular(),
            se.centroid(),
            sre.centroid(),
            se.match_db(&sre),
        );
        println!(
            "{:>3} {}  {:7.1} {:7.1}  {:5.1} {:5.1}  {:5.1} {:5.1}  {:5.1}",
            row.0,
            if row.1 { "y" } else { "-" },
            row.2,
            row.3,
            row.4,
            row.5,
            row.6,
            row.7,
            row.8
        );
        rows.push(row);
    }

    // Neighbour residuals on level, same-N stringing, 8 nearest.
    let unison: Vec<usize> = (FIRST_KEY..=LAST_KEY)
        .map(|k| usize::from(preset.notes.unison[usize::from(k - FIRST_KEY)]))
        .collect();
    fn residual(
        rows: &[(u8, bool, f64, f64, f64, f64, f64, f64, f64)],
        unison: &[usize],
        levels: &[f64],
        i: usize,
    ) -> f64 {
        let mut nb: Vec<(u16, f64)> = (0..rows.len())
            .filter(|&j| j != i && unison[j] == unison[i])
            .map(|j| ((rows[j].0 as i16 - rows[i].0 as i16).unsigned_abs(), levels[j]))
            .collect();
        nb.sort_by_key(|&(d, _)| d);
        let mut vals: Vec<f64> = nb.iter().take(8).map(|&(_, v)| v).collect();
        vals.sort_by(f64::total_cmp);
        levels[i] - vals[vals.len() / 2]
    }
    let e_levels: Vec<f64> = rows.iter().map(|r| r.2).collect();
    let r_levels: Vec<f64> = rows.iter().map(|r| r.3).collect();
    println!("\nlevel residual vs 8 nearest same-N neighbours (engine | recording):");
    let mut worst_fit = (0u8, 0.0f64);
    let mut sum_fit = 0.0;
    let mut n_fit = 0;
    for i in 0..rows.len() {
        let re = residual(&rows, &unison, &e_levels, i);
        let rr = residual(&rows, &unison, &r_levels, i);
        if rows[i].1 {
            sum_fit += re.abs();
            n_fit += 1;
            if re.abs() > worst_fit.1 {
                worst_fit = (rows[i].0, re.abs());
            }
        }
        if re.abs() > 2.5 || rr.abs() > 2.5 {
            println!(
                "  {:>3} {}  engine {:+6.2}  recording {:+6.2}",
                rows[i].0,
                if rows[i].1 { "y" } else { "-" },
                re,
                rr
            );
        }
    }
    println!(
        "fitted keys: mean |level residual| {:.2} dB, worst {:.2} dB at key {}",
        sum_fit / n_fit as f64,
        worst_fit.1,
        worst_fit.0
    );

    // Roughness distance to own recording over fitted keys 21..=84.
    let mut d: Vec<(u8, f64, f64, f64)> = rows
        .iter()
        .filter(|r| r.1 && r.0 <= 84)
        .map(|r| (r.0, r.4, r.5, (r.4 - r.5).abs()))
        .collect();
    let mean = d.iter().map(|x| x.3).sum::<f64>() / d.len() as f64;
    let within = d.iter().filter(|x| x.3 <= 1.5).count();
    d.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
    println!(
        "\nfitted keys A0-C6: mean |irr_e - irr_r| {:.2} dB, {}/{} inside 1.5 dB; worst:",
        mean,
        within,
        d.len()
    );
    for x in d.iter().take(6) {
        println!("  key {:>3}  e {:.1}  r {:.1}  |d| {:.1}", x.0, x.1, x.2, x.3);
    }
}

fn mode_neutral() {
    let preset = Preset::load(std::path::Path::new("presets/salamander-c5.toml")).expect("preset");
    let sr = f64::from(SAMPLE_RATE);
    println!("key  d_peak   d_rms(0.10-1.10)  (with-row minus without-row, dB)");
    let mut worst = (0u8, 0.0f64);
    for key in fitted_keys(&preset) {
        let idx = usize::from(key - FIRST_KEY);
        let mut bare = preset.clone();
        bare.notes.partial_gains[idx] = Vec::new();
        let a = engine_note(&preset, key, VEL, 2.0);
        let b = engine_note(&bare, key, VEL, 2.0);
        let peak = |x: &[f32]| db(x.iter().fold(0.0f64, |m, &s| m.max(f64::from(s).abs())));
        let dp = peak(&a) - peak(&b);
        let dr = rms_db(&a, sr, 0.10, 1.10) - rms_db(&b, sr, 0.10, 1.10);
        if dr.abs() > worst.1 {
            worst = (key, dr.abs());
        }
        println!("{key:>3}  {dp:+6.2}  {dr:+6.2}");
    }
    println!("worst |RMS moved| {:.2} dB at key {}", worst.1, worst.0);
}

fn mode_ff() {
    for path in ["presets/default.toml", "presets/salamander-c5.toml"] {
        let preset = Preset::load(std::path::Path::new(path)).expect("preset");
        println!("\n{path}: every key vel 127, 1.0 s");
        let mut worst = (0u8, -200.0f64);
        let mut total_over = 0usize;
        for key in FIRST_KEY..=LAST_KEY {
            let events = [RenderEvent::new(0.0, Event::NoteOn { key, vel: 127 })];
            let (l, r) = render_to_buffer(&preset, &events, 1.0);
            let chan = l
                .iter()
                .chain(r.iter())
                .fold(0.0f64, |m, &x| m.max(f64::from(x).abs()));
            let over = l
                .iter()
                .chain(r.iter())
                .filter(|x| x.abs() > LIMIT_THRESHOLD)
                .count();
            total_over += over;
            if db(chan) > worst.1 {
                worst = (key, db(chan));
            }
            if over > 0 {
                println!("  key {key}: {over} samples over the limiter, peak {:.2} dBFS", db(chan));
            }
        }
        println!(
            "  loudest key {} at {:.2} dBFS channel; limiter samples total {}",
            worst.0, worst.1, total_over
        );
    }
}

fn mode_treble() {
    let preset = Preset::load(std::path::Path::new("presets/salamander-c5.toml")).expect("preset");
    let sfz = PathBuf::from("data/salamander/SalamanderGrandPiano-V3+20200602.sfz");
    let mut sampler = Sampler::new(&sfz).expect("sampler");
    let sr = f64::from(SAMPLE_RATE);
    println!("key  slope2-3s e/r (dB/s)   tail-rms 3.0-3.5s e/r (dBFS)  zero-tail?");
    for key in 100..=LAST_KEY {
        let e = engine_note(&preset, key, VEL, RENDER_S);
        let r = reference_note(&mut sampler, key, VEL, RENDER_S);
        // 50 ms block RMS envelope, straight-line fit 2.0-3.0 s.
        let slope = |x: &[f32]| -> f64 {
            let blocks: Vec<(f64, f64)> = (0..20)
                .map(|i| {
                    let t = 2.0 + 0.05 * i as f64;
                    (t + 0.025, rms_db(x, sr, t, t + 0.05))
                })
                .filter(|(_, l)| l.is_finite())
                .collect();
            let n = blocks.len() as f64;
            let mx = blocks.iter().map(|b| b.0).sum::<f64>() / n;
            let my = blocks.iter().map(|b| b.1).sum::<f64>() / n;
            let num: f64 = blocks.iter().map(|b| (b.0 - mx) * (b.1 - my)).sum();
            let den: f64 = blocks.iter().map(|b| (b.0 - mx) * (b.0 - mx)).sum();
            num / den
        };
        let tail_zero = {
            let lo = (3.0 * sr) as usize;
            e[lo..].iter().all(|&s| s == 0.0)
        };
        println!(
            "{key:>3}  {:+8.1} / {:+8.1}   {:7.1} / {:7.1}   {}",
            slope(&e),
            slope(&r),
            rms_db(&e, sr, 3.0, 3.5),
            rms_db(&r, sr, 3.0, 3.5),
            if tail_zero { "ZERO" } else { "rings" }
        );
    }
}

fn mode_idle() {
    for path in ["presets/default.toml", "presets/salamander-c5.toml"] {
        let preset = Preset::load(std::path::Path::new(path)).expect("preset");
        let (l, r) = render_to_buffer(&preset, &[], 3.0);
        let nonzero = l.iter().chain(r.iter()).filter(|&&x| x != 0.0).count();
        println!("{path}: {} samples, {} nonzero", l.len() + r.len(), nonzero);
    }
}

fn mode_chord() {
    const CHORD: [u8; 10] = [36, 43, 48, 52, 55, 60, 64, 67, 72, 76];
    for path in ["presets/default.toml", "presets/salamander-c5.toml"] {
        let preset = Preset::load(std::path::Path::new(path)).expect("preset");
        println!("\n{path}:");
        let strike = |keys: &[u8], vel: u8, secs: f32| {
            let events: Vec<RenderEvent> = keys
                .iter()
                .map(|&k| RenderEvent::new(0.0, Event::NoteOn { key: k, vel }))
                .collect();
            let (l, r) = render_to_buffer(&preset, &events, secs);
            let chan = l
                .iter()
                .chain(r.iter())
                .fold(0.0f64, |m, &x| m.max(f64::from(x).abs()));
            let mono = l
                .iter()
                .zip(&r)
                .fold(0.0f64, |m, (&a, &b)| m.max(f64::from(a + b).abs()));
            let over = l
                .iter()
                .chain(r.iter())
                .filter(|x| x.abs() > LIMIT_THRESHOLD)
                .count();
            (db(mono), db(chan), over)
        };
        let (m, c, o) = strike(&[60], 80, 3.0);
        println!("  mf C4 vel80:   mono {m:7.2} chan {c:7.2} limiter {o}");
        let (m, c, o) = strike(&[60], 127, 3.0);
        println!("  ff C4 vel127:  mono {m:7.2} chan {c:7.2} limiter {o}");
        let (m, c, o) = strike(&[108], 127, 2.0);
        println!("  C8 vel127:     mono {m:7.2} chan {c:7.2} limiter {o}");
        let (m, c, o) = strike(&CHORD, 127, 3.0);
        println!(
            "  ten-note ff:   mono {m:7.2} chan {c:7.2} limiter {o}  (headroom {:+.2} dB)",
            db(f64::from(LIMIT_THRESHOLD)) - c
        );
    }
}

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "solo".into());
    match mode.as_str() {
        "solo" => mode_solo(),
        "neutral" => mode_neutral(),
        "ff" => mode_ff(),
        "treble" => mode_treble(),
        "idle" => mode_idle(),
        "chord" => mode_chord(),
        other => eprintln!("unknown mode {other}"),
    }
}
