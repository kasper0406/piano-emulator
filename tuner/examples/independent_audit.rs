//! Independent audio audit of the sympathetic milestone, measured with its own
//! DSP rather than the tuner's.
//!
//! Everything spectral here — the FFT, the windowed tone projection, the T60
//! fits, the band split — is implemented in this file, deliberately not calling
//! `piano_tuner::residual` / `estimate::halo`, so that the numbers it prints
//! are an independent check on both the engine and the tuner's own
//! measurement code. Only rendering (the engine), sample-library discovery and
//! audio decoding (the tuner) are reused.
//!
//! Sections, one per audit criterion:
//!
//! 1. between-partial census at C4/C6/C7 across velocity, recordings beside
//!    before/after renders, at two window lengths;
//! 2. release-resonance halo at C3/C5 against the `harmL*` targets;
//! 3. duplex segments: residual peaks at the table's frequencies after a
//!    staccato release, against the same render with the table emptied;
//! 4. cross-note bloom through the bridge (the Cartling signature);
//! 5. decay-rate coupling: a partial on a bridge peak, shipped against unity
//!    bridge;
//! 6. mid-range regression at A2/C4, before against after;
//! 7. 60 s stability fuzz at the fitted preset and at validation-boundary
//!    couplings;
//! 8. per-register stereo drift;
//! 9. what the `harmL*` targets are actually made of, and the like-for-like
//!    comparison criterion 2 is not;
//! 10. which engine path the treble aftersound gap of criterion 1 lives on.
//!
//! ```text
//! cargo run --release --example independent_audit -- [before.toml] [1 2 ...]
//! ```

use std::f64::consts::TAU;
use std::path::PathBuf;

use piano_emulator::preset::Preset as EnginePreset;
use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_emulator::types::{Event, PedalEvent};
use piano_tuner::{audio, SampleLibrary, SAMPLE_RATE};
use rustfft::{num_complex::Complex, FftPlanner};

const SR: f64 = SAMPLE_RATE as f64;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let after = EnginePreset::load(&repo.join("presets/salamander-c5.toml"))?;

    let mut args = std::env::args().skip(1).peekable();
    let before_path = match args.peek() {
        Some(a) if a.ends_with(".toml") => PathBuf::from(args.next().expect("peeked")),
        _ => PathBuf::from("/tmp/before_2daa84d.toml"),
    };
    let sections: Vec<String> = args.collect();
    let wanted = |s: &str| sections.is_empty() || sections.iter().any(|a| a == s);
    let before = EnginePreset::load(&before_path).ok();
    if before.is_none() {
        println!("(no before preset at {}; before rows skipped)", before_path.display());
    }
    let library =
        SampleLibrary::from_sfz(repo.join("data/salamander/SalamanderGrandPiano-V3+20200602.sfz"))
            .ok();

    if wanted("1") {
        census(library.as_ref(), before.as_ref(), &after);
    }
    if wanted("2") {
        halo(before.as_ref(), &after);
    }
    if wanted("3") {
        duplex(&after);
    }
    if wanted("4") {
        bloom(&after);
    }
    if wanted("5") {
        decay_coupling(&after);
    }
    if wanted("6") {
        if let Some(before) = before.as_ref() {
            regression(before, &after);
        }
    }
    if wanted("7") {
        fuzz(&after);
    }
    if wanted("8") {
        drift(before.as_ref(), &after);
    }
    if wanted("9") {
        provenance(library.as_ref(), &repo, &after);
    }
    if wanted("10") {
        aftersound_paths(library.as_ref(), &after);
    }
    Ok(())
}

// ----------------------------------------------------------------- primitives

fn mono(l: &[f32], r: &[f32]) -> Vec<f32> {
    l.iter().zip(r).map(|(&a, &b)| 0.5 * (a + b)).collect()
}

fn render_mono(preset: &EnginePreset, events: &[RenderEvent], seconds: f32) -> Vec<f32> {
    let (l, r) = render_to_buffer(preset, events, seconds);
    mono(&l, &r)
}

fn db(x: f64) -> f64 {
    20.0 * x.max(1e-30).log10()
}

fn peak(signal: &[f32]) -> f64 {
    signal.iter().fold(0.0f64, |m, &v| m.max(f64::from(v.abs())))
}

fn rms(signal: &[f32]) -> f64 {
    if signal.is_empty() {
        return 0.0;
    }
    (signal.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / signal.len() as f64).sqrt()
}

fn at(t: f64) -> usize {
    (t * SR) as usize
}

/// Hann-windowed magnitude spectrum (own FFT path, `rustfft` directly).
fn magnitude_spectrum(signal: &[f32], start: usize, window: usize) -> Option<Vec<f64>> {
    if start + window > signal.len() {
        return None;
    }
    let mut buf: Vec<Complex<f64>> = (0..window)
        .map(|n| {
            let w = 0.5 - 0.5 * (TAU * n as f64 / window as f64).cos();
            Complex::new(f64::from(signal[start + n]) * w, 0.0)
        })
        .collect();
    FftPlanner::new().plan_fft_forward(window).process(&mut buf);
    Some(buf[..window / 2].iter().map(|c| c.norm()).collect())
}

/// Amplitude of a Hann-weighted projection onto `hz` — a one-bin DFT at the
/// exact frequency, so no picket-fence bias.
fn tone_level(signal: &[f32], start: usize, window: usize, hz: f64) -> f64 {
    if start + window > signal.len() {
        return 0.0;
    }
    let (mut re, mut im, mut wsum) = (0.0f64, 0.0f64, 0.0f64);
    for n in 0..window {
        let w = 0.5 - 0.5 * (TAU * n as f64 / window as f64).cos();
        let phase = TAU * hz * n as f64 / SR;
        let x = f64::from(signal[start + n]) * w;
        re += x * phase.cos();
        im -= x * phase.sin();
        wsum += w;
    }
    2.0 * re.hypot(im) / wsum
}

/// Least-squares line through `(t, y)`; returns (slope, intercept).
fn line_fit(points: &[(f64, f64)]) -> (f64, f64) {
    let n = points.len() as f64;
    let (sx, sy): (f64, f64) = points.iter().fold((0.0, 0.0), |a, p| (a.0 + p.0, a.1 + p.1));
    let (sxx, sxy): (f64, f64) = points
        .iter()
        .fold((0.0, 0.0), |a, p| (a.0 + p.0 * p.0, a.1 + p.0 * p.1));
    let slope = (n * sxy - sx * sy) / (n * sxx - sx * sx).max(1e-12);
    (slope, (sy - slope * sx) / n)
}

/// T60 of one partial from a linear fit to its dB envelope over `t0..t1`.
fn partial_t60(signal: &[f32], hz: f64, t0: f64, t1: f64) -> f64 {
    let window = 4800; // 100 ms
    let mut points = Vec::new();
    let mut t = t0;
    while t + 0.1 < t1 {
        let level = tone_level(signal, at(t), window, hz);
        if level > 1e-12 {
            points.push((t, db(level)));
        }
        t += 0.05;
    }
    if points.len() < 8 {
        return f64::NAN;
    }
    let (slope, _) = line_fit(&points);
    if slope >= -1e-3 {
        return f64::INFINITY;
    }
    -60.0 / slope
}

/// Refines a partial's frequency by scanning the projection over ±40 cents.
fn refine_freq(signal: &[f32], start: usize, window: usize, seed_hz: f64) -> f64 {
    let mut best = (0.0f64, seed_hz);
    let mut cents = -40.0f64;
    while cents <= 40.0 {
        let hz = seed_hz * (cents / 1200.0).exp2();
        let level = tone_level(signal, start, window, hz);
        if level > best.0 {
            best = (level, hz);
        }
        cents += 0.5;
    }
    best.1
}

/// The engine's partial layout for one key, up to `top_hz`.
fn partials(preset: &EnginePreset, key: u8, top_hz: f64) -> Vec<f64> {
    let i = usize::from(key - 21);
    let f0 = f64::from(preset.notes.f0_hz[i]);
    let b = f64::from(preset.notes.inharmonicity_b[i]);
    let b4 = preset
        .notes
        .inharmonicity_b4
        .get(i)
        .map_or(0.0, |&v| f64::from(v));
    let mut out = Vec::new();
    for k in 1..=120u32 {
        let kf = f64::from(k);
        let radicand = 1.0 + b * kf * kf + b4 * kf.powi(4);
        if radicand <= 0.0 {
            break;
        }
        let f = kf * f0 * radicand.sqrt();
        if f > top_hz {
            break;
        }
        out.push(f);
    }
    out
}

/// First sample that reaches 5 % of the signal's peak — onset of a recording.
fn onset(signal: &[f32]) -> usize {
    let threshold = (peak(signal) * 0.05) as f32;
    signal.iter().position(|v| v.abs() >= threshold).unwrap_or(0)
}

/// Power between the partials over power in them, dB — `TUNING_REPORT.md` §4's
/// definition, reimplemented: Hann frame, guard of four bins around each
/// partial, band from 0.75·f1 to 12 kHz.
fn between_partials_db(signal: &[f32], start: usize, window: usize, layout: &[f64]) -> f64 {
    let Some(magnitude) = magnitude_spectrum(signal, start, window) else {
        return f64::NAN;
    };
    let bin_hz = SR / window as f64;
    let guard = 4.0 * bin_hz;
    let band = (0.75 * layout[0], 12_000.0);
    let (mut in_partials, mut between) = (0.0f64, 0.0f64);
    for (bin, &value) in magnitude.iter().enumerate() {
        let f = bin as f64 * bin_hz;
        if f < band.0 || f > band.1 {
            continue;
        }
        let power = value * value;
        if layout.iter().any(|&p| (p - f).abs() <= guard) {
            in_partials += power;
        } else {
            between += power;
        }
    }
    if in_partials <= 0.0 || between <= 0.0 {
        return f64::NEG_INFINITY;
    }
    10.0 * (between / in_partials).log10()
}

/// Silences every mechanism burst so a key-off thump is not counted as halo.
fn silence_mechanism(preset: &EnginePreset) -> EnginePreset {
    let mut quiet = preset.clone();
    for event in [
        &mut quiet.noise.key_off,
        &mut quiet.noise.damper_lift,
        &mut quiet.noise.pedal_down,
        &mut quiet.noise.pedal_up,
    ] {
        for anchor in &mut event.level_db {
            anchor.db = -200.0;
        }
    }
    quiet
}

/// Largest `voicing.resonance_coupling` the validator accepts for this preset.
fn max_legal_coupling(preset: &EnginePreset) -> f32 {
    let (mut lo, mut hi) = (0.0f32, 2.0f32);
    for _ in 0..40 {
        let mid = 0.5 * (lo + hi);
        let mut candidate = preset.clone();
        candidate.voicing.resonance_coupling = mid;
        if candidate.validate().is_ok() {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

// ------------------------------------------------------------- 1. the census

fn census(library: Option<&SampleLibrary>, before: Option<&EnginePreset>, after: &EnginePreset) {
    println!("\n=== 1. between-partial energy, one second after the strike");
    println!("       (own FFT; guard 4 bins; band 0.75*f1..12 kHz; windows 85 ms and 341 ms)");
    println!(
        "\n{:>12} {:>4} {:>4} {:>12} {:>12} {:>12} {:>12}",
        "source", "key", "vel", "between@0", "between@1s", "@1s (341ms)", "@2s (341ms)"
    );
    for key in [60u8, 84, 96] {
        let mut velocities = vec![40u8, 90, 108];
        if key == 96 {
            velocities.insert(2, 68);
        }
        for vel in velocities {
            if let Some(library) = library {
                if let Some(sample) = library
                    .layers(key)
                    .iter()
                    .find(|s| (s.lovel..=s.hivel).contains(&vel))
                {
                    if let Ok(recording) = audio::load_at(&sample.path, SAMPLE_RATE) {
                        let signal = recording.mono();
                        let layout = partials(after, key, 12_000.0);
                        let start = onset(&signal);
                        row("salamander", key, sample.midi_velocity(), &signal, start, &layout);
                    }
                }
            }
            let layout = partials(after, key, 12_000.0);
            if let Some(before) = before {
                let signal = render_note(before, key, vel, 8.0);
                row("before", key, vel, &signal, 0, &layout);
            }
            let signal = render_note(after, key, vel, 8.0);
            row("after", key, vel, &signal, 0, &layout);
        }
        println!();
    }

    fn row(source: &str, key: u8, vel: u8, signal: &[f32], start: usize, layout: &[f64]) {
        println!(
            "{source:>12} {key:>4} {vel:>4} {:>12.1} {:>12.1} {:>12.1} {:>12.1}",
            between_partials_db(signal, start, 4096, layout),
            between_partials_db(signal, start + at(1.0), 4096, layout),
            between_partials_db(signal, start + at(1.0), 16_384, layout),
            between_partials_db(signal, start + at(2.0), 16_384, layout),
        );
    }
}

fn render_note(preset: &EnginePreset, key: u8, vel: u8, seconds: f32) -> Vec<f32> {
    render_mono(preset, &[RenderEvent::new(0.0, Event::NoteOn { key, vel })], seconds)
}

// --------------------------------------------------------------- 2. the halo

fn halo(before: Option<&EnginePreset>, after: &EnginePreset) {
    println!("\n=== 2. release-resonance halo at C3/C5 (targets: peak -31 dB / -39 dB, +-3)");
    println!("       (strike vel 90, release at 1 s, mechanism silenced, uncoupled render");
    println!("        subtracted; quoted against the peak of a strike of the same key)");
    println!(
        "\n{:>8} {:>4} {:>10} {:>10} {:>12} {:>12} {:>16}",
        "preset", "key", "peak dB", "rms dB", "@rel+0.5s", "@rel+1.5s", "20 dB ring s"
    );
    for key in [48u8, 72] {
        for (label, preset) in [("before", before), ("after", Some(after))] {
            let Some(preset) = preset else { continue };
            let quiet = silence_mechanism(preset);
            let mut bare = quiet.clone();
            bare.voicing.resonance_coupling = 0.0;
            bare.notes.duplex = Vec::new();
            let events = [
                RenderEvent::new(0.0, Event::NoteOn { key, vel: 90 }),
                RenderEvent::new(1.0, Event::NoteOff { key, vel: 64 }),
            ];
            let with = render_mono(&quiet, &events, 5.0);
            let without = render_mono(&bare, &events, 5.0);
            let halo: Vec<f32> = with
                .iter()
                .zip(&without)
                .skip(at(1.05))
                .map(|(&a, &b)| a - b)
                .collect();
            let strike = render_note(&quiet, key, 90, 2.0);
            let reference = peak(&strike);
            let halo_peak = peak(&halo);
            let halo_rms = rms(&halo[..at(2.0).min(halo.len())]);
            // 50 ms envelope; time from its post-release peak to 20 dB below it.
            let envelope: Vec<f64> = halo
                .chunks(at(0.05))
                .map(rms)
                .collect();
            let (peak_i, peak_v) = envelope
                .iter()
                .enumerate()
                .fold((0, 0.0f64), |m, (i, &v)| if v > m.1 { (i, v) } else { m });
            let ring = envelope[peak_i..]
                .iter()
                .position(|&v| v < peak_v * 0.1)
                .map_or(f64::INFINITY, |i| i as f64 * 0.05);
            // Windowed RMS 0.5 s and 1.5 s after the release (halo[0] is the
            // release), for the "rings 1-2 s" half of the target.
            let window = |t: f64| {
                let start = at(t).min(halo.len());
                let end = (start + at(0.5)).min(halo.len());
                db(rms(&halo[start..end])) - db(reference)
            };
            println!(
                "{label:>8} {key:>4} {:>10.1} {:>10.1} {:>12.1} {:>12.1} {:>16.2}",
                db(halo_peak) - db(reference),
                db(halo_rms) - db(reference),
                window(0.5),
                window(1.5),
                ring
            );
        }
    }
}

// ------------------------------------------------------------ 3. the duplex

fn duplex(after: &EnginePreset) {
    println!("\n=== 3. duplex segments after a staccato treble release");
    println!("       (strike at 0.2 s vel 108, off at 0.35 s; levels re strike peak, at the");
    println!("        table's own frequencies; 'empty' is the same render, table removed)");
    let keys_with_tables: Vec<u8> = after
        .notes
        .duplex
        .iter()
        .enumerate()
        .filter(|(_, row)| !row.is_empty())
        .map(|(i, _)| 21 + i as u8)
        .collect();
    println!("       keys with tables: {keys_with_tables:?}");
    let mut empty = after.clone();
    empty.notes.duplex = Vec::new();
    // The loudest legal version of the same table: every gain raised by the
    // largest uniform dB shift the validator accepts (schema cap +6 dB/entry).
    let boost_db = {
        let (mut lo, mut hi) = (0.0f32, 80.0f32);
        for _ in 0..30 {
            let mid = 0.5 * (lo + hi);
            if boosted(after, mid).validate().is_ok() {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    };
    let loud = boosted(after, boost_db);
    println!("       largest legal uniform boost of the shipped table: +{boost_db:.1} dB\n");
    println!(
        "{:>10} {:>4} {:>9} {:>14} {:>14} {:>14} {:>10}",
        "variant", "key", "segment", "@rel+0.4 dB", "@rel+1.0 dB", "@rel+1.6 dB", "T60 s"
    );
    for &key in [84u8, 90, 96].iter().filter(|k| keys_with_tables.contains(k)) {
        let row = &after.notes.duplex[usize::from(key - 21)];
        let events = [
            RenderEvent::new(0.2, Event::NoteOn { key, vel: 108 }),
            RenderEvent::new(0.35, Event::NoteOff { key, vel: 64 }),
        ];
        let reference_render = render_mono(&silence_mechanism(after), &events, 4.0);
        let reference = db(peak(&reference_render));
        let empty_render = render_mono(&silence_mechanism(&empty), &events, 4.0);
        for (label, preset) in [("shipped", after), ("boosted", &loud)] {
            // The segments isolated: this render minus the same one with the
            // table removed. Anything here came from the duplex bank alone.
            let signal = render_mono(&silence_mechanism(preset), &events, 4.0);
            let difference: Vec<f32> = signal
                .iter()
                .zip(&empty_render)
                .map(|(&a, &b)| a - b)
                .collect();
            println!(
                "{label:>10} {key:>4}  contribution peak {:>6.1} dB re strike",
                db(peak(&difference)) - reference
            );
            for segment in row {
                let hz = f64::from(segment.hz);
                let level = |t: f64| db(tone_level(&difference, at(0.35 + t), 4800, hz)) - reference;
                let t60 = partial_t60(&difference, hz, 0.8, 2.4);
                println!(
                    "{label:>10} {key:>4} {:>6.0}Hz {:>14.1} {:>14.1} {:>14.1} {:>10.2}",
                    hz,
                    level(0.4),
                    level(1.0),
                    level(1.6),
                    t60
                );
            }
        }
        println!();
    }

    fn boosted(preset: &EnginePreset, delta_db: f32) -> EnginePreset {
        let mut out = preset.clone();
        for row in &mut out.notes.duplex {
            for segment in row {
                segment.gain_db = (segment.gain_db + delta_db).min(6.0);
            }
        }
        out
    }
}

// ------------------------------------------------------- 4. cross-note bloom

fn bloom(after: &EnginePreset) {
    println!("\n=== 4. cross-note bloom (Cartling): a bass note gains energy from a");
    println!("       coincident treble strike; G4 struck into C3 (G4 = C3's 3rd partial)");
    let quiet = silence_mechanism(after);
    let c3 = partials(after, 48, 4000.0);

    // (a) sounding C3 held; G4 struck at 3.0 s, damped at 3.2 s. Levels of
    // C3's partials at 4.5 s, with the strike against without it, and the
    // treble alone as the bound on its own residual.
    let held = [RenderEvent::new(0.0, Event::NoteOn { key: 48, vel: 85 })];
    let struck = [
        RenderEvent::new(0.0, Event::NoteOn { key: 48, vel: 85 }),
        RenderEvent::new(3.0, Event::NoteOn { key: 67, vel: 108 }),
        RenderEvent::new(3.2, Event::NoteOff { key: 67, vel: 64 }),
    ];
    let treble = [
        RenderEvent::new(3.0, Event::NoteOn { key: 67, vel: 108 }),
        RenderEvent::new(3.2, Event::NoteOff { key: 67, vel: 64 }),
    ];
    let with = render_mono(&quiet, &struck, 6.0);
    let without = render_mono(&quiet, &held, 6.0);
    let alone = render_mono(&quiet, &treble, 6.0);
    println!("\n   (a) C3 sounding, held; C3 partial levels at 4.5 s (dBFS):");
    println!(
        "{:>10} {:>12} {:>12} {:>14} {:>12}",
        "partial", "with G4", "without", "treble alone", "bloom dB"
    );
    for (k, &hz) in c3.iter().enumerate().take(6) {
        let level = |s: &[f32]| db(tone_level(s, at(4.5), 9600, hz));
        println!(
            "{:>8}{:>2} {:>12.1} {:>12.1} {:>14.1} {:>12.2}",
            format!("{hz:.0}Hz k="),
            k + 1,
            level(&with),
            level(&without),
            level(&alone),
            level(&with) - level(&without)
        );
    }

    // (b) the same with C3 pressed silently, so everything at its partials
    // after the treble is damped arrived through the bridge.
    let silent_with = [
        RenderEvent::new(0.0, Event::KeyDown { key: 48 }),
        RenderEvent::new(0.5, Event::NoteOn { key: 67, vel: 108 }),
        RenderEvent::new(0.7, Event::NoteOff { key: 67, vel: 64 }),
    ];
    let silent_without = [
        RenderEvent::new(0.5, Event::NoteOn { key: 67, vel: 108 }),
        RenderEvent::new(0.7, Event::NoteOff { key: 67, vel: 64 }),
    ];
    let with = render_mono(&quiet, &silent_with, 4.0);
    let without = render_mono(&quiet, &silent_without, 4.0);
    let reference = db(peak(&with));
    println!("\n   (b) C3 pressed silently; levels at C3's 3rd partial re G4 strike peak:");
    for t in [1.2, 2.2, 3.2] {
        println!(
            "       t={t:.1}s  with press {:>7.1} dB   without {:>7.1} dB",
            db(tone_level(&with, at(t), 9600, c3[2])) - reference,
            db(tone_level(&without, at(t), 9600, c3[2])) - reference
        );
    }
}

// -------------------------------------------------- 5. decay-rate coupling

fn decay_coupling(after: &EnginePreset) {
    println!("\n=== 5. decay-rate coupling: T60 of a partial on a bridge peak, shipped");
    println!("       bridge against unity bridge (PHYSICS section 4's second signature)");
    let Some(bridge) = after.voicing.bridge.as_ref() else {
        println!("   (no bridge section in the preset)");
        return;
    };
    // The strongest positive peaks, and the note partial nearest each.
    let mut candidates: Vec<(f64, f64, u8, usize, f64)> = Vec::new(); // gain, peak hz, key, k, cents
    for peak in bridge.peaks.iter().filter(|p| p.gain_db >= 3.0) {
        for key in 21u8..=96 {
            for (k, &hz) in partials(after, key, 8000.0).iter().enumerate() {
                let cents = 1200.0 * (hz / f64::from(peak.hz)).log2().abs();
                candidates.push((f64::from(peak.gain_db), f64::from(peak.hz), key, k + 1, cents));
            }
        }
    }
    candidates.sort_by(|a, b| a.4.total_cmp(&b.4));
    candidates.truncate(3);
    let mut unity = after.clone();
    unity.voicing.bridge = None;
    let mut uncoupled = after.clone();
    uncoupled.voicing.resonance_coupling = 0.0;
    println!(
        "\n{:>26} {:>10} {:>14} {:>13} {:>13}",
        "partial on peak", "peak gain", "T60 shipped", "T60 unity", "T60 uncoupld"
    );
    for (gain, peak_hz, key, k, cents) in candidates {
        let layout = partials(after, key, 8000.0);
        let hz = layout[k - 1];
        let events = [RenderEvent::new(0.0, Event::NoteOn { key, vel: 80 })];
        let t60 = |p: &EnginePreset| {
            let signal = render_mono(p, &events, 6.5);
            partial_t60(&signal, refine_freq(&signal, at(0.3), 16_384, hz), 0.5, 4.0)
        };
        println!(
            "{:>14} k{k:<2} ({cents:>4.1}c off {peak_hz:.0}Hz) {gain:>6.1}dB {:>11.2}s {:>11.2}s {:>11.2}s",
            format!("key {key} {hz:.1}Hz"),
            t60(after),
            t60(&unity),
            t60(&uncoupled)
        );
    }
}

// ------------------------------------------------------ 6. mid-range guard

fn regression(before: &EnginePreset, after: &EnginePreset) {
    println!("\n=== 6. mid-range regression guard: A2 and C4, before against after");
    println!(
        "\n{:>4} {:>26} {:>15} {:>15} {:>13} {:>11}",
        "key", "worst partial move (c)", "T60 k1 b/a s", "T60 k3 b/a s", "RMS 0-3s dB", "peak dB"
    );
    for key in [45u8, 60] {
        let b = render_note(before, key, 90, 5.0);
        let a = render_note(after, key, 90, 5.0);
        let layout = partials(after, key, 6000.0);
        let mut worst = 0.0f64;
        for &hz in layout.iter().take(8) {
            let fb = refine_freq(&b, at(0.3), 16_384, hz);
            let fa = refine_freq(&a, at(0.3), 16_384, hz);
            let cents = 1200.0 * (fa / fb).log2();
            if cents.abs() > worst.abs() {
                worst = cents;
            }
        }
        let t60_pair = |k: usize| {
            (
                partial_t60(&b, layout[k - 1], 0.3, 3.5),
                partial_t60(&a, layout[k - 1], 0.3, 3.5),
            )
        };
        let (t1b, t1a) = t60_pair(1);
        let (t3b, t3a) = t60_pair(3);
        println!(
            "{key:>4} {worst:>26.2} {:>7.2}/{:<7.2} {:>7.2}/{:<7.2} {:>13.2} {:>11.2}",
            t1b,
            t1a,
            t3b,
            t3a,
            db(rms(&a[..at(3.0)])) - db(rms(&b[..at(3.0)])),
            db(peak(&a)) - db(peak(&b)),
        );
    }
}

// ---------------------------------------------------------- 7. stability fuzz

fn fuzz(after: &EnginePreset) {
    println!("\n=== 7. stability fuzz: 60 s of dense pedal-down playing");
    println!("       (trend is the slope of 1 s-block RMS in dB over the last 20 s, while");
    println!("        the playing is still dense; bounded means finite and under 0 dBFS)");
    let boundary = {
        let mut p = after.clone();
        p.voicing.resonance_coupling = max_legal_coupling(after) * 0.999;
        p
    };
    let extreme = {
        let mut p = after.clone();
        if let Some(bridge) = p.voicing.bridge.as_mut() {
            for anchor in &mut bridge.backbone {
                anchor.gain_db = 20.0;
            }
            for peak in &mut bridge.peaks {
                peak.gain_db = peak.gain_db.max(10.0);
            }
        }
        p.voicing.resonance_coupling = 0.001;
        p.voicing.resonance_coupling = max_legal_coupling(&p) * 0.999;
        p
    };
    println!(
        "       couplings: shipped {:.6}, boundary {:.6}, extreme bridge {:.6}",
        after.voicing.resonance_coupling,
        boundary.voicing.resonance_coupling,
        extreme.voicing.resonance_coupling
    );
    let events = dense_pedal_down();
    println!(
        "\n{:>16} {:>10} {:>10} {:>22} {:>16} {:>10}",
        "variant", "finite", "peak dBFS", "RMS 0-20/20-40/40-60", "trend dB/s", "verdict"
    );
    for (label, preset) in [
        ("shipped", after),
        ("boundary", &boundary),
        ("extreme bridge", &extreme),
    ] {
        assert!(preset.validate().is_ok(), "{label} preset must be legal");
        let (l, r) = render_to_buffer(preset, &events, 60.0);
        let signal = mono(&l, &r);
        let finite = signal.iter().all(|v| v.is_finite());
        let blocks: Vec<f64> = signal.chunks(at(1.0)).map(|c| db(rms(c))).collect();
        let late: Vec<(f64, f64)> = blocks[40..60]
            .iter()
            .enumerate()
            .map(|(i, &v)| (i as f64, v))
            .collect();
        let (slope, _) = line_fit(&late);
        let p = db(peak(&signal).max(peak(&l)).max(peak(&r)));
        let bounded = finite && p < 0.0 && slope < 0.5;
        println!(
            "{label:>16} {finite:>10} {p:>10.1} {:>6.1}/{:>6.1}/{:>6.1} dB {slope:>13.3} {:>10}",
            db(rms(&signal[..at(20.0)])),
            db(rms(&signal[at(20.0)..at(40.0)])),
            db(rms(&signal[at(40.0)..])),
            if bounded { "BOUNDED" } else { "SUSPECT" }
        );
    }

    fn dense_pedal_down() -> Vec<RenderEvent> {
        let mut state = 0x243F_6A88_85A3_08D3u64;
        let mut next = move || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        let mut events = vec![RenderEvent::new(0.0, Event::Pedal(PedalEvent::Sustain(1.0)))];
        let mut t = 0.05f32;
        while t < 59.0 {
            let key = 21 + (next() % 88) as u8;
            let vel = 30 + (next() % 98) as u8;
            let hold = 0.1 + (next() % 400) as f32 / 1000.0;
            events.push(RenderEvent::new(t, Event::NoteOn { key, vel }));
            events.push(RenderEvent::new(t + hold, Event::NoteOff { key, vel: 64 }));
            t += 0.04 + (next() % 100) as f32 / 1000.0;
        }
        events.sort_by(|a, b| a.time_s.total_cmp(&b.time_s));
        events
    }
}

// ------------------------------------------------------------ 8. stereo drift

fn drift(before: Option<&EnginePreset>, after: &EnginePreset) {
    println!("\n=== 8. per-register stereo drift (recordings' band: 1.2 to 6.2 dB)");
    println!("       (median over partials of |(L-R)@2s - (L-R)@0.3s|, vel 108)");
    println!(
        "\n{:>4} {:>10} {:>14} {:>14}",
        "key", "partials", "before dB", "after dB"
    );
    for key in [21u8, 45, 60, 72, 84, 96] {
        let measure = |preset: &EnginePreset| -> (usize, f64) {
            let events = [RenderEvent::new(0.0, Event::NoteOn { key, vel: 108 })];
            let (l, r) = render_to_buffer(preset, &events, 4.5);
            let layout = partials(preset, key, 8000.0);
            let level = |c: &[f32], hz: f64, t: f64| tone_level(c, at(t), 8192, hz);
            let loudest = layout
                .iter()
                .map(|&hz| level(&l, hz, 0.3).max(level(&r, hz, 0.3)))
                .fold(0.0f64, f64::max);
            let mut deltas: Vec<f64> = Vec::new();
            for &hz in layout.iter().take(30) {
                let (l0, r0) = (level(&l, hz, 0.3), level(&r, hz, 0.3));
                let (l2, r2) = (level(&l, hz, 2.0), level(&r, hz, 2.0));
                let audible = l0.min(r0) > loudest * 10f64.powf(-45.0 / 20.0)
                    && l2.min(r2) > loudest * 10f64.powf(-70.0 / 20.0);
                if audible {
                    let early = db(l0) - db(r0);
                    let late = db(l2) - db(r2);
                    deltas.push((late - early).abs());
                }
            }
            deltas.sort_by(f64::total_cmp);
            if deltas.is_empty() {
                (0, f64::NAN)
            } else {
                (deltas.len(), deltas[deltas.len() / 2])
            }
        };
        let (partials_after, after_db) = measure(after);
        let before_db = before.map_or(f64::NAN, |p| measure(p).1);
        println!("{key:>4} {partials_after:>10} {before_db:>14.2} {after_db:>14.2}");
    }
}

// ------------------------------------------- 9. what the halo target is made of

/// The `harmL*` recordings, taken apart, and the engine measured the way they
/// were.
///
/// `TUNING_REPORT.md` §5 quotes `harmLC3` at −30.7 dB and `harmLC5` at
/// −39.0 dB relative to a velocity-90 strike of the same key, and reads them as
/// "a target for the sympathetic-coupling fit". Criterion 2 measures the engine
/// against those numbers by rendering the same gesture *twice* — once whole,
/// once with the coupling and the segments removed — and subtracting, i.e. it
/// measures the engine's **sympathetic component alone**. Those are only the
/// same quantity if the recording contains nothing but sympathetic resonance.
///
/// Two measurements decide it, and neither needs a guess about how the sample
/// was recorded:
///
/// * how the recording's energy divides between the struck key's **own**
///   partials and everything else. A damper takes a few tenths of a second to
///   stop a wound string, so a release recording that is mostly the note's own
///   partials is mostly the note's own damped tail;
/// * the same split on the engine's whole post-release signal, which contains
///   the damped tail *and* the halo, rendered with the damper working.
fn provenance(library: Option<&SampleLibrary>, repo: &std::path::Path, after: &EnginePreset) {
    println!("\n=== 9. what the harmL* targets are made of");
    println!("       (release recording vs the engine's whole post-release signal, split");
    println!("        into the struck key's own partials and everything else)");
    println!(
        "\n{:>16} {:>4} {:>12} {:>12} {:>12} {:>10} {:>13}",
        "signal", "key", "peak re str", "own partials", "the rest", "own share", "20 dB ring s"
    );
    for (key, name) in [(48u8, "harmLC3"), (72, "harmLC5")] {
        let layout = partials(after, key, 12_000.0);

        // The recording, at the level the SFZ plays it (`volume=-4`), against
        // the loudest layer of the same key's strike.
        if let Some(library) = library {
            let path = repo.join(format!("data/salamander/samples/{name}.flac"));
            let strike = library
                .layers(key)
                .iter()
                .find(|s| (s.lovel..=s.hivel).contains(&90u8))
                .and_then(|s| audio::load_at(&s.path, SAMPLE_RATE).ok());
            if let (Ok(release), Some(strike)) = (audio::load_at(&path, SAMPLE_RATE), strike) {
                let release = release.mono();
                let strike = strike.mono();
                let start = onset(&release);
                let (own, rest) = partial_split(&release, start, &layout);
                let level = db(peak(&release[start..])) - 4.0 - db(peak(&strike));
                row(name, key, level, own, rest, ring_seconds(&release[start..]));
            }
        }

        // The engine, same gesture, nothing subtracted: strike, release, and
        // what is still there a twentieth of a second later.
        let quiet = silence_mechanism(after);
        let events = [
            RenderEvent::new(0.0, Event::NoteOn { key, vel: 90 }),
            RenderEvent::new(1.0, Event::NoteOff { key, vel: 64 }),
        ];
        let whole = render_mono(&quiet, &events, 5.0);
        let start = at(1.05);
        let strike_peak = peak(&render_note(&quiet, key, 90, 2.0));
        let (own, rest) = partial_split(&whole, start, &layout);
        row(
            "engine, whole",
            key,
            db(peak(&whole[start..])) - db(strike_peak),
            own,
            rest,
            ring_seconds(&whole[start..]),
        );

        // ... and the sympathetic component of it on its own, which is what
        // criterion 2 quotes: the same render minus the uncoupled one.
        let mut bare = quiet.clone();
        bare.voicing.resonance_coupling = 0.0;
        bare.notes.duplex = Vec::new();
        let uncoupled = render_mono(&bare, &events, 5.0);
        let halo: Vec<f32> = whole
            .iter()
            .zip(&uncoupled)
            .map(|(&a, &b)| a - b)
            .collect();
        let (own, rest) = partial_split(&halo, start, &layout);
        row(
            "engine, halo",
            key,
            db(peak(&halo[start..])) - db(strike_peak),
            own,
            rest,
            ring_seconds(&halo[start..]),
        );
        println!();
    }

    fn row(signal: &str, key: u8, level: f64, own: f64, rest: f64, ring: f64) {
        let total = own + rest;
        println!(
            "{signal:>16} {key:>4} {level:>12.1} {:>12.1} {:>12.1} {:>9.0}% {:>13.2}",
            db(own.sqrt()) - db(total.sqrt()),
            db(rest.sqrt()) - db(total.sqrt()),
            100.0 * own / total.max(f64::MIN_POSITIVE),
            ring
        );
    }
}

/// Power in the bins within two of a partial of `layout`, and power in all the
/// others, over a 341 ms window — long enough that a wound bass string's
/// partials are resolved from each other.
fn partial_split(signal: &[f32], start: usize, layout: &[f64]) -> (f64, f64) {
    let window = 16_384;
    let Some(spectrum) = magnitude_spectrum(signal, start, window) else {
        return (0.0, 0.0);
    };
    let bin_hz = SR / window as f64;
    let (mut own, mut rest) = (0.0, 0.0);
    for (i, &m) in spectrum.iter().enumerate() {
        let hz = i as f64 * bin_hz;
        if !(30.0..=12_000.0).contains(&hz) {
            continue;
        }
        let near = layout.iter().any(|&f| (hz - f).abs() <= 2.0 * bin_hz);
        if near {
            own += m * m;
        } else {
            rest += m * m;
        }
    }
    (own, rest)
}

/// Time from a signal's loudest 50 ms window to the first one 20 dB below it —
/// the "rings 1–2 s" half of `TUNING_REPORT.md` §5's target, measured the same
/// way on a recording and on a render.
fn ring_seconds(signal: &[f32]) -> f64 {
    let envelope: Vec<f64> = signal.chunks(at(0.05)).map(rms).collect();
    let (peak_i, peak_v) = envelope
        .iter()
        .enumerate()
        .fold((0, 0.0f64), |m, (i, &v)| if v > m.1 { (i, v) } else { m });
    envelope[peak_i..]
        .iter()
        .position(|&v| v < peak_v * 0.1)
        .map_or(f64::INFINITY, |i| i as f64 * 0.05)
}

// ------------------------------------- 10. where the treble aftersound gap is

/// Criterion 1 measures a 21.5 dB deficit at C7 above the leakage floor. This
/// asks which path in the engine that deficit is on, by taking each candidate
/// to an extreme and re-measuring.
///
/// `TUNING_REPORT.md`'s backlog item 5 says of this gap that "the level is a
/// coupling parameter (`resonance.rs`, one per-register field)". That is the
/// claim under test, and the sympathetic coupling is therefore taken all the
/// way to `Preset::max_safe_coupling` — the largest value the stability
/// contract will certify — rather than merely moved.
fn aftersound_paths(library: Option<&SampleLibrary>, after: &EnginePreset) {
    println!("\n=== 10. which path the treble aftersound gap is on");
    println!("       (between-partial energy, 341 ms window, velocity 108; each row is one");
    println!("        path taken to an extreme, everything else as shipped)");
    for key in [96u8, 84] {
        let layout = partials(after, key, 12_000.0);
        println!(
            "\n{:>22} {:>4} {:>10} {:>10} {:>10}",
            "variant", "key", "@0", "@1 s", "@2 s"
        );
        if let Some(library) = library {
            if let Some(sample) = library
                .layers(key)
                .iter()
                .find(|s| (s.lovel..=s.hivel).contains(&108u8))
            {
                if let Ok(recording) = audio::load_at(&sample.path, SAMPLE_RATE) {
                    let signal = recording.mono();
                    let start = onset(&signal);
                    println!(
                        "{:>22} {key:>4} {:>10.1} {:>10.1} {:>10.1}",
                        "salamander",
                        between_partials_db(&signal, start, 16_384, &layout),
                        between_partials_db(&signal, start + at(1.0), 16_384, &layout),
                        between_partials_db(&signal, start + at(2.0), 16_384, &layout),
                    );
                }
            }
        }

        let mut variants: Vec<(&str, EnginePreset)> = Vec::new();
        variants.push(("shipped", after.clone()));
        let mut bare = after.clone();
        bare.voicing.resonance_coupling = 0.0;
        bare.notes.duplex = Vec::new();
        variants.push(("no coupling at all", bare));
        let mut loud = after.clone();
        loud.voicing.resonance_coupling = loud.max_safe_coupling();
        variants.push(("coupling at its bound", loud));
        let mut board = after.clone();
        board.soundboard.board_mix = 1.0;
        variants.push(("board mix 1.0", board));
        let mut dry = after.clone();
        dry.soundboard.board_mix = 0.0;
        variants.push(("no board at all", dry));
        let mut wet = after.clone();
        wet.soundboard.fdn_t60_lf *= 4.0;
        wet.soundboard.fdn_t60_hf *= 4.0;
        variants.push(("diffuse field T60 x4", wet));

        for (name, preset) in variants {
            if preset.validate().is_err() {
                println!("{name:>22} {key:>4}   (not a legal preset)");
                continue;
            }
            let signal = render_note(&preset, key, 108, 8.0);
            println!(
                "{name:>22} {key:>4} {:>10.1} {:>10.1} {:>10.1}",
                between_partials_db(&signal, 0, 16_384, &layout),
                between_partials_db(&signal, at(1.0), 16_384, &layout),
                between_partials_db(&signal, at(2.0), 16_384, &layout),
            );
        }
    }
}
