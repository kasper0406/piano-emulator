//! Independent audio verification of the Milestone A engine changes.
//!
//! Re-measures, from rendered audio and from the Salamander recordings,
//! everything the milestone claims — without trusting the gate report:
//!
//! 1. Wound-bass partial placement (A0/C1/D#1) against the layout
//!    `presets/salamander-c5.toml` writes, full series and trusted prefix.
//! 2. Stereo drift 0.3 s -> 2 s at `polarization_pan_spread` 0.4 and 0.
//! 3. Key-off / pedal noise levels, decay, centroid, band limit, determinism.
//! 4. Silently-held-key sympathetic response with the pedal up.
//! 5. Derivative-outlier click scan around note-off / pedal / half-pedal.
//! 6. NaN / clipping / DC in dense material.
//! 7. Fundamental drift on notes with the fitted per-string sigma spread.
//!
//! ```text
//! cargo run --release -p forensics --bin verify_milestone_a
//! ```

use std::path::PathBuf;

use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_emulator::types::{Event, PedalEvent};
use piano_tuner::estimate::inharmonic::{trusted_prefix, InharmonicConfig};
use piano_tuner::pipeline::{analyze_trajectories, track_refined};
use piano_tuner::preset::{equal_temperament, key_index, Preset};
use piano_tuner::residual::{
    frame_spectrum, partial_levels, partial_residuals, transient_metrics, ResidualConfig,
};
use piano_tuner::survey::{trajectories_for, SurveyConfig};
use piano_tuner::trajectory::InharmonicModel;
use piano_tuner::{SampleLibrary, SAMPLE_RATE};

const SR: f64 = SAMPLE_RATE as f64;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let root = repo.join("data/salamander");
    let preset_path = repo.join("presets/salamander-c5.toml");
    let cache = repo.join("data/cache/salamander");

    let library = SampleLibrary::from_sfz(root.join("SalamanderGrandPiano-V3+20200602.sfz"))?;
    let tuner_preset = Preset::load(&preset_path)?;
    let engine_preset = piano_emulator::preset::Preset::load(&preset_path)?;
    let config = SurveyConfig {
        cache_dir: Some(cache),
        ..SurveyConfig::default()
    };
    let residual = ResidualConfig::default();

    let sections: Vec<String> = std::env::args().skip(1).collect();
    let wanted = |s: &str| sections.is_empty() || sections.iter().any(|a| a == s);

    if wanted("1") {
        bass_pass(&library, &tuner_preset, &config, &residual);
        bass_layers(&library, &tuner_preset, &config, &residual, 21);
    }
    if wanted("2") {
        stereo_pass(&engine_preset, &config);
    }
    if wanted("3") {
        noise_pass(&engine_preset);
        strike_reference_pass(&engine_preset, &repo)?;
    }
    if wanted("4") {
        sympathetic_pass(&engine_preset, &tuner_preset);
    }
    if wanted("5") {
        click_pass(&engine_preset);
    }
    if wanted("6") {
        dense_pass(&engine_preset);
    }
    if wanted("7") {
        drift_pass(&engine_preset, &config, &residual);
    }
    Ok(())
}

// ------------------------------------------------- 1. wound-bass partial placement

fn bass_pass(
    library: &SampleLibrary,
    preset: &Preset,
    config: &SurveyConfig,
    residual: &ResidualConfig,
) {
    println!("\n=== 1. wound-bass partial placement vs the layout the preset writes\n");
    println!("        (TUNING_REPORT section 1 published 15.7 / 13.1 / 15.4 cents RMS over the");
    println!("         whole tracked series; the milestone claims 8.79 / 6.33 / 4.57 over the");
    println!("         partials whose index the tracker can be believed)\n");
    println!(" key  layers   full rms   full worst(k)   trusted rms   trusted worst(k)  trusted n");
    for key in [21u8, 24, 27] {
        let mut full_rms = Vec::new();
        let mut full_worst = Vec::new();
        let mut trusted_rms = Vec::new();
        let mut trusted_worst = Vec::new();
        let mut trusted_n = Vec::new();
        let model = written_model(preset, key);
        for sample in library.layers(key) {
            let Ok(note_config) = config.note_config(equal_temperament(key)) else {
                continue;
            };
            let Ok(trajectories) = trajectories_for(sample, &note_config, config) else {
                continue;
            };
            let Ok(analysis) = analyze_trajectories(trajectories, &note_config) else {
                continue;
            };
            let residuals = partial_residuals(&analysis, residual);
            let trusted = trusted_prefix(
                &residuals
                    .iter()
                    .map(|r| (r.k, r.frequency_hz))
                    .collect::<Vec<_>>(),
                &InharmonicConfig::default(),
            );
            let cents: Vec<(u32, f64)> = residuals
                .iter()
                .map(|r| (r.k, model.cents_from_partial(r.k, r.frequency_hz)))
                .collect();
            full_rms.push(rms(cents.iter().map(|&(_, c)| c)));
            full_worst.push(worst(&cents));
            let head = &cents[..trusted.min(cents.len())];
            trusted_rms.push(rms(head.iter().map(|&(_, c)| c)));
            trusted_worst.push(worst(head));
            trusted_n.push(trusted as f64);
        }
        let m = |v: &[f64]| median(v.iter().copied()).unwrap_or(f64::NAN);
        let mw = |v: &[(f64, u32)]| {
            let w = median(v.iter().map(|&(c, _)| c)).unwrap_or(f64::NAN);
            let k = median(v.iter().map(|&(_, k)| f64::from(k))).unwrap_or(f64::NAN);
            format!("{w:6.1}c ({k:.0})")
        };
        println!(
            "{key:>4} {:>7}  {:>8.2}c {:>15} {:>12.2}c {:>18} {:>10.0}",
            full_rms.len(),
            m(&full_rms),
            mw(&full_worst),
            m(&trusted_rms),
            mw(&trusted_worst),
            m(&trusted_n),
        );
    }
}

/// Layer-by-layer detail for one key: where does the trusted-prefix residual
/// live, and in particular what happens at the fundamental.
fn bass_layers(
    library: &SampleLibrary,
    preset: &Preset,
    config: &SurveyConfig,
    residual: &ResidualConfig,
    key: u8,
) {
    println!("\n--- 1b. key {key} layer by layer (trusted prefix only)\n");
    println!(" layer  vel   trusted n   rms all   rms k>=2   k=1 cents   k=1 level dB");
    let model = written_model(preset, key);
    for (layer, sample) in library.layers(key).iter().enumerate() {
        let Ok(note_config) = config.note_config(equal_temperament(key)) else {
            continue;
        };
        let Ok(trajectories) = trajectories_for(sample, &note_config, config) else {
            continue;
        };
        let Ok(analysis) = analyze_trajectories(trajectories, &note_config) else {
            continue;
        };
        let residuals = partial_residuals(&analysis, residual);
        let trusted = trusted_prefix(
            &residuals
                .iter()
                .map(|r| (r.k, r.frequency_hz))
                .collect::<Vec<_>>(),
            &InharmonicConfig::default(),
        );
        let head = &residuals[..trusted.min(residuals.len())];
        let all = rms(head.iter().map(|r| model.cents_from_partial(r.k, r.frequency_hz)));
        let tail = rms(
            head.iter()
                .filter(|r| r.k >= 2)
                .map(|r| model.cents_from_partial(r.k, r.frequency_hz)),
        );
        let k1 = head.iter().find(|r| r.k == 1);
        println!(
            "{layer:>6} {:>4} {:>10} {:>9.2}c {:>9.2}c {:>10} {:>12}",
            sample.midi_velocity(),
            trusted,
            all,
            tail,
            k1.map_or("-".to_string(), |r| format!(
                "{:+.1}",
                model.cents_from_partial(r.k, r.frequency_hz)
            )),
            k1.map_or("-".to_string(), |r| format!("{:.1}", r.level_db)),
        );
    }
}

fn worst(cents: &[(u32, f64)]) -> (f64, u32) {
    cents.iter().fold((0.0, 0), |(w, wk), &(k, c)| {
        if c.abs() > w {
            (c.abs(), k)
        } else {
            (w, wk)
        }
    })
}

fn written_model(preset: &Preset, key: u8) -> InharmonicModel {
    let index = key_index(key).unwrap();
    InharmonicModel::with_b4(
        f64::from(preset.notes.f0_hz[index]),
        f64::from(preset.notes.inharmonicity_b[index]),
        f64::from(preset.notes.inharmonicity_b4[index]),
    )
}

// ----------------------------------------------------------- 2. stereo drift

fn stereo_pass(preset: &piano_emulator::preset::Preset, config: &SurveyConfig) {
    println!("\n=== 2. stereo drift 0.3 s -> 2 s (median over partials of |delta(2s) - delta(0.3s)|)\n");
    println!("        recordings measured 1.2-6.2 dB; engine before the milestone 0.02-0.14 dB\n");
    println!(" key   drift @ spread=0.4   drift @ spread=0");
    let mut zero = preset.clone();
    zero.voicing.polarization_pan_spread = 0.0;
    for key in [21u8, 45, 60, 72, 84, 96] {
        let with = stereo_drift(preset, key, config);
        let without = stereo_drift(&zero, key, config);
        let show = |v: Option<f64>| v.map_or("-".to_string(), |v| format!("{v:.2} dB"));
        println!("{key:>4} {:>20} {:>18}", show(with), show(without));
    }
}

fn stereo_drift(
    preset: &piano_emulator::preset::Preset,
    key: u8,
    config: &SurveyConfig,
) -> Option<f64> {
    let events = [RenderEvent::new(0.0, Event::NoteOn { key, vel: 90 })];
    let (left, right) = render_to_buffer(preset, &events, 4.0);
    let mono: Vec<f32> = left.iter().zip(&right).map(|(&l, &r)| 0.5 * (l + r)).collect();
    let note_config = config.note_config(equal_temperament(key)).ok()?;
    let (trajectories, _) = track_refined(
        &mono,
        SR,
        InharmonicModel::harmonic(equal_temperament(key)),
        &note_config,
    )
    .ok()?;
    let loudest = trajectories
        .tracks
        .iter()
        .filter_map(|t| t.peak())
        .map(|p| p.amplitude)
        .fold(0.0f64, f64::max);
    let frequencies: Vec<f64> = trajectories
        .tracks
        .iter()
        .filter(|t| t.peak().is_some_and(|p| p.amplitude >= loudest * 1e-3))
        .filter_map(|t| t.weighted_frequency())
        .collect();
    let window = note_config.tracker.stft.window;
    let guard = 4.0 * SR / window as f64;
    let deltas = |seconds: f64| -> Option<Vec<Option<f64>>> {
        let start = ((trajectories.onset_s + seconds) * SR) as usize;
        let l = frame_spectrum(&left, start, window, 1).ok()?;
        let r = frame_spectrum(&right, start, window, 1).ok()?;
        let (l, r) = (
            partial_levels(&l, SR, window, &frequencies, guard),
            partial_levels(&r, SR, window, &frequencies, guard),
        );
        Some(
            l.into_iter()
                .zip(r)
                .map(|(l, r)| Some(20.0 * (l? / r?).log10()).filter(|d| d.is_finite()))
                .collect(),
        )
    };
    let early = deltas(0.3)?;
    let late = deltas(2.0)?;
    median(
        early
            .into_iter()
            .zip(late)
            .filter_map(|(a, b)| Some((b? - a?).abs())),
    )
}

// -------------------------------------------------------------- 3. noise levels

fn noise_pass(preset: &piano_emulator::preset::Preset) {
    println!("\n=== 3. mechanism noise: measured on renders vs the preset and the report\n");

    // Reference: peak of a velocity-90 strike of the same key, measured on the
    // stereo magnitude sqrt(l^2 + r^2), which an equal-power pan preserves —
    // a mono average would weight the two pan positions differently by up to
    // 3 dB, and the noise and the reference sit at different effective pans.
    let strike_peak = |key: u8| -> f64 {
        let events = [RenderEvent::new(0.0, Event::NoteOn { key, vel: 90 })];
        let (l, r) = render_to_buffer(preset, &events, 3.0);
        f64::from(
            l.iter()
                .zip(&r)
                .map(|(&l, &r)| (l * l + r * r).sqrt())
                .fold(0.0f32, f32::max),
        )
    };

    // The report's section-5 measured table (peak re vel-90 strike of the key).
    let report: &[(u8, f64)] = &[(21, -37.3), (57, -30.2), (60, -35.4), (72, -25.4), (96, -33.5)];
    let anchors: Vec<(u8, f64)> = preset
        .noise
        .key_off
        .level_db
        .iter()
        .map(|a| (a.key, f64::from(a.db)))
        .collect();

    println!("key-off (release velocity 64, i.e. the tabulated level; level is the mean of");
    println!("four events, because a single burst's peak scatters 2-3 dB by seed):");
    println!(" key   measured    preset asks   report table   decay(-40dB)   centroid   >2kHz energy");
    for &(key, table) in report {
        let reference = strike_peak(key);
        let offs = [2.5f64, 5.0, 7.5, 10.0];
        let mut events = Vec::new();
        for &at in &offs {
            events.push(RenderEvent::new(at as f32 - 0.4, Event::KeyDown { key }));
            events.push(RenderEvent::new(at as f32, Event::NoteOff { key, vel: 64 }));
        }
        let (l, r) = render_to_buffer(preset, &events, 12.5);
        let magnitude: Vec<f32> = l
            .iter()
            .zip(&r)
            .map(|(&l, &r)| (l * l + r * r).sqrt())
            .collect();
        let mono: Vec<f32> = l.iter().zip(&r).map(|(&l, &r)| 0.5 * (l + r)).collect();
        let mut levels = Vec::new();
        let mut first = None;
        for &at in &offs {
            let range = (at * SR) as usize..((at + 2.0) * SR) as usize;
            let peak = magnitude[range.clone()].iter().fold(0.0f32, |m, &x| m.max(x.abs()));
            levels.push(20.0 * (f64::from(peak) / reference).log10());
            if let Some(m) = transient_metrics(&mono[range], SR) {
                first.get_or_insert(m);
            }
        }
        let Some(m) = first else { continue };
        let level = levels.iter().sum::<f64>() / levels.len() as f64;
        let asked = interp(&anchors, key);
        let slice = &mono[(offs[0] * SR) as usize..((offs[0] + 2.0) * SR) as usize];
        println!(
            "{key:>4} {level:>9.1} dB {asked:>10.1} dB {table:>11.1} dB {:>10.3} s {:>8.0} Hz {:>11.1} dB",
            m.decay_s,
            m.centroid_hz,
            high_band_db(slice, 2000.0),
        );
    }

    // Pedal down: sustain crossing up with every damper free plays the
    // tabulated level. Quoted against a velocity-90 C4 strike as in the report.
    let c4 = strike_peak(60);
    let down_events = [RenderEvent::new(0.5, Event::Pedal(PedalEvent::Sustain(1.0)))];
    let (dl, dr) = render_to_buffer(preset, &down_events, 8.5);
    let down_mag: Vec<f32> = dl
        .iter()
        .zip(&dr)
        .map(|(&l, &r)| (l * l + r * r).sqrt())
        .collect();
    let down: Vec<f32> = dl.iter().zip(&dr).map(|(&l, &r)| 0.5 * (l + r)).collect();
    let slice = &down[(0.5 * SR) as usize..];
    let peak = down_mag[(0.5 * SR) as usize..].iter().fold(0.0f32, |m, &x| m.max(x));
    if let Some(m) = transient_metrics(slice, SR) {
        println!("\npedal-down (preset asks {:.1} dB, report table -35.8 dB re C4 strike):", -38.3);
        println!(
            "   measured {:.1} dB   decay {:.2} s (report 5.76)   centroid {:.0} Hz (report 77)   >2kHz {:.1} dB",
            20.0 * (f64::from(peak) / c4).log10(),
            m.decay_s,
            m.centroid_hz,
            high_band_db(slice, 2000.0),
        );
    }

    // Pedal up, isolated by subtracting a render where the pedal stays down.
    let up_events = [
        RenderEvent::new(0.5, Event::Pedal(PedalEvent::Sustain(1.0))),
        RenderEvent::new(4.0, Event::Pedal(PedalEvent::Sustain(0.0))),
    ];
    let (ul, ur) = render_to_buffer(preset, &up_events, 8.5);
    let diff_mag: Vec<f32> = ul
        .iter()
        .zip(&ur)
        .zip(dl.iter().zip(&dr))
        .map(|((&al, &ar), (&bl, &br))| {
            let (l, r) = (al - bl, ar - br);
            (l * l + r * r).sqrt()
        })
        .collect();
    let diff: Vec<f32> = ul
        .iter()
        .zip(&ur)
        .zip(dl.iter().zip(&dr))
        .map(|((&al, &ar), (&bl, &br))| 0.5 * ((al - bl) + (ar - br)))
        .collect();
    let slice = &diff[(4.0 * SR) as usize..];
    let peak = diff_mag[(4.0 * SR) as usize..].iter().fold(0.0f32, |m, &x| m.max(x));
    if let Some(m) = transient_metrics(slice, SR) {
        println!("\npedal-up (preset asks {:.1} dB, report table -42.4 dB re C4 strike):", -45.6);
        println!(
            "   measured {:.1} dB   decay {:.2} s (report 0.32)   centroid {:.0} Hz (report 187)   >2kHz {:.1} dB",
            20.0 * (f64::from(peak) / c4).log10(),
            m.decay_s,
            m.centroid_hz,
            high_band_db(slice, 2000.0),
        );
    }

    // Determinism: the same event list must render the same bits.
    let phrase = click_phrase();
    let (al, ar) = render_to_buffer(preset, &phrase, 7.0);
    let (bl, br) = render_to_buffer(preset, &phrase, 7.0);
    let same = al.iter().zip(&bl).all(|(a, b)| a.to_bits() == b.to_bits())
        && ar.iter().zip(&br).all(|(a, b)| a.to_bits() == b.to_bits());
    println!("\ndeterminism: two identical renders bit-identical: {same}");
}

/// The strike each preset's mechanism levels are quoted against, which is now
/// what `engine/src/calibrate.rs` measures for itself when an engine is built
/// (`DECISIONS.md` 145): the two presets' velocity-90 peaks differ by 1.4-1.9 dB,
/// and while the engine anchored its bursts to one constant measured on the
/// default preset that difference went straight into the rendered level. Also
/// re-measures the pedal-down decay by envelope slope, which a single deep fade
/// of narrowband noise cannot fool.
fn strike_reference_pass(
    preset: &piano_emulator::preset::Preset,
    repo: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- 3b. vel-90 strike peaks, salamander preset vs default (the noise reference)\n");
    let default = piano_emulator::preset::Preset::load(&repo.join("presets/default.toml"))?;
    println!(" key   salamander peak   default peak   difference");
    for key in [21u8, 57, 60, 72, 96] {
        let peak = |p: &piano_emulator::preset::Preset| -> f64 {
            let events = [RenderEvent::new(0.0, Event::NoteOn { key, vel: 90 })];
            let (l, r) = render_to_buffer(p, &events, 3.0);
            let mono: Vec<f32> = l.iter().zip(&r).map(|(&l, &r)| 0.5 * (l + r)).collect();
            transient_metrics(&mono, SR).map_or(f64::NAN, |m| m.peak)
        };
        let (s, d) = (peak(preset), peak(&default));
        println!(
            "{key:>4} {:>13.1} dBFS {:>10.1} dBFS {:>+10.1} dB",
            20.0 * s.log10(),
            20.0 * d.log10(),
            20.0 * (s / d).log10(),
        );
    }

    let events = [RenderEvent::new(0.5, Event::Pedal(PedalEvent::Sustain(1.0)))];
    let (l, r) = render_to_buffer(preset, &events, 8.5);
    let mono: Vec<f32> = l.iter().zip(&r).map(|(&l, &r)| 0.5 * (l + r)).collect();
    let slice = &mono[(0.5 * SR) as usize..];
    if let Some(t40) = slope_decay_to_40db(slice, 0.5, 7.5) {
        println!("\npedal-down decay by envelope slope over 0.5-7.5 s: {t40:.2} s to -40 dB");
    }

    // The same key-off measurement on the default preset, whose noise table is
    // the report's own measured entries: separates this measurement chain from
    // the salamander preset's voicing.
    println!("\n--- 3c. key-off on the default preset (its table IS the report's)\n");
    println!(" key   measured    preset asks");
    for key in [21u8, 57, 60, 72, 96] {
        let reference = {
            let events = [RenderEvent::new(0.0, Event::NoteOn { key, vel: 90 })];
            let (l, r) = render_to_buffer(&default, &events, 3.0);
            f64::from(
                l.iter()
                    .zip(&r)
                    .map(|(&l, &r)| (l * l + r * r).sqrt())
                    .fold(0.0f32, f32::max),
            )
        };
        let offs = [2.5f64, 5.0, 7.5, 10.0];
        let mut events = Vec::new();
        for &at in &offs {
            events.push(RenderEvent::new(at as f32 - 0.4, Event::KeyDown { key }));
            events.push(RenderEvent::new(at as f32, Event::NoteOff { key, vel: 64 }));
        }
        let (l, r) = render_to_buffer(&default, &events, 12.5);
        let magnitude: Vec<f32> = l
            .iter()
            .zip(&r)
            .map(|(&l, &r)| (l * l + r * r).sqrt())
            .collect();
        let levels: Vec<f64> = offs
            .iter()
            .map(|&at| {
                let peak = magnitude[(at * SR) as usize..((at + 2.0) * SR) as usize]
                    .iter()
                    .fold(0.0f32, |m, &x| m.max(x));
                20.0 * (f64::from(peak) / reference).log10()
            })
            .collect();
        let anchors: Vec<(u8, f64)> = default
            .noise
            .key_off
            .level_db
            .iter()
            .map(|a| (a.key, f64::from(a.db)))
            .collect();
        println!(
            "{key:>4} {:>9.1} dB {:>10.1} dB",
            levels.iter().sum::<f64>() / levels.len() as f64,
            interp(&anchors, key),
        );
    }
    Ok(())
}

/// Time to fall 40 dB, from a least-squares line through the 100 ms RMS
/// envelope in dB between `from` and `to` seconds of the slice.
fn slope_decay_to_40db(signal: &[f32], from: f64, to: f64) -> Option<f64> {
    let window = (0.1 * SR) as usize;
    let mut points = Vec::new();
    let mut start = (from * SR) as usize;
    while start + window <= signal.len().min((to * SR) as usize) {
        let chunk = &signal[start..start + window];
        let rms = (chunk.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>()
            / chunk.len() as f64)
            .sqrt();
        if rms > 0.0 {
            points.push((start as f64 / SR, 20.0 * rms.log10()));
        }
        start += window;
    }
    if points.len() < 8 {
        return None;
    }
    let n = points.len() as f64;
    let mx = points.iter().map(|p| p.0).sum::<f64>() / n;
    let my = points.iter().map(|p| p.1).sum::<f64>() / n;
    let (num, den) = points.iter().fold((0.0, 0.0), |(num, den), &(x, y)| {
        (num + (x - mx) * (y - my), den + (x - mx) * (x - mx))
    });
    let slope = num / den;
    (slope < 0.0).then(|| -40.0 / slope)
}

/// Energy above `split_hz` relative to the whole band, dB, over the first
/// 0.68 s of the slice.
fn high_band_db(signal: &[f32], split_hz: f64) -> f64 {
    let window = 32768.min(signal.len().next_power_of_two() / 2).max(4096);
    let Ok(spectrum) = frame_spectrum(signal, 0, window, 1) else {
        return f64::NAN;
    };
    let bin_hz = SR / window as f64;
    let split = (split_hz / bin_hz) as usize;
    let total: f64 = spectrum.iter().map(|&a| f64::from(a) * f64::from(a)).sum();
    let high: f64 = spectrum[split.min(spectrum.len())..]
        .iter()
        .map(|&a| f64::from(a) * f64::from(a))
        .sum();
    10.0 * (high / total).log10()
}

fn interp(anchors: &[(u8, f64)], key: u8) -> f64 {
    let x = f64::from(key);
    let mut sorted = anchors.to_vec();
    sorted.sort_by_key(|&(k, _)| k);
    if x <= f64::from(sorted[0].0) {
        return sorted[0].1;
    }
    for pair in sorted.windows(2) {
        let (k0, v0) = (f64::from(pair[0].0), pair[0].1);
        let (k1, v1) = (f64::from(pair[1].0), pair[1].1);
        if x <= k1 {
            return v0 + (v1 - v0) * (x - k0) / (k1 - k0);
        }
    }
    sorted.last().unwrap().1
}

// ------------------------------------------------- 4. silent-key sympathetic response

fn sympathetic_pass(preset: &piano_emulator::preset::Preset, tuner_preset: &Preset) {
    println!("\n=== 4. silently held C3 answering a struck G4, pedal up\n");
    let model = written_model(tuner_preset, 48);
    let frequencies: Vec<f64> = (1..=8).map(|k| model.partial(k)).collect();

    let render = |hold: bool| -> (Vec<f32>, Vec<f32>) {
        let mut events = vec![
            RenderEvent::new(0.5, Event::NoteOn { key: 67, vel: 100 }),
            RenderEvent::new(2.0, Event::NoteOff { key: 67, vel: 110 }),
        ];
        if hold {
            events.insert(0, RenderEvent::new(0.1, Event::KeyDown { key: 48 }));
        }
        render_to_buffer(preset, &events, 6.0)
    };
    let level_at = |signal: &[f32], seconds: f64| -> Vec<Option<f64>> {
        let window = 16384;
        let guard = 4.0 * SR / window as f64;
        let start = (seconds * SR) as usize;
        frame_spectrum(signal, start, window, 1)
            .map(|s| partial_levels(&s, SR, window, &frequencies, guard))
            .unwrap_or_default()
    };

    let (hl, hr) = render(true);
    let held: Vec<f32> = hl.iter().zip(&hr).map(|(&l, &r)| 0.5 * (l + r)).collect();
    let (cl, cr) = render(false);
    let control: Vec<f32> = cl.iter().zip(&cr).map(|(&l, &r)| 0.5 * (l + r)).collect();

    // The two renders share every G4 sample (the engine is deterministic), so
    // the difference is exactly what holding C3 added: its damper-lift noise
    // early on, then whatever its string radiates.
    let diff: Vec<f32> = held.iter().zip(&control).map(|(&h, &c)| h - c).collect();
    println!("   RMS of (held - control), i.e. C3's own contribution:");
    for (from, to) in [(0.6, 1.9), (2.2, 3.0), (3.0, 4.0), (4.0, 5.8)] {
        let chunk = &diff[(from * SR) as usize..(to * SR) as usize];
        let rms = (chunk.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>()
            / chunk.len() as f64)
            .sqrt();
        println!("      {from:.1}-{to:.1} s: {:.1} dBFS", 20.0 * rms.log10().max(-200.0));
    }
    println!("   level at C3's partials, held vs control, 0.5 s after G4's release (t = 2.5 s):");
    let held_levels = level_at(&held, 2.5);
    let control_levels = level_at(&control, 2.5);
    for (k, (h, c)) in held_levels.iter().zip(&control_levels).enumerate() {
        if let (Some(h), Some(c)) = (h, c) {
            println!(
                "   k={} ({:>6.1} Hz): held {:>7.1} dBFS, control {:>7.1} dBFS, excess {:+.1} dB",
                k + 1,
                frequencies[k],
                20.0 * h.log10(),
                20.0 * c.log10(),
                20.0 * (h / c).log10(),
            );
        }
    }
    let held_late = level_at(&held, 3.5);
    if let Some(Some(h)) = held_late.get(2) {
        println!(
            "   held level at C3 k=3 one second later (t = 3.5 s): {:.1} dBFS",
            20.0 * h.log10()
        );
    }

    // The preparation itself must be almost silent: peak of the held render
    // before the strike (this is the damper-lift noise, nothing else).
    let before: &[f32] = &held[..(0.45 * SR) as usize];
    let peak = before.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    println!("   peak before the G4 strike (damper-lift noise only): {:.1} dBFS", 20.0 * f64::from(peak).log10());
}

// ------------------------------------------------------------- 5. click scan

fn click_phrase() -> Vec<RenderEvent> {
    vec![
        RenderEvent::new(0.30, Event::NoteOn { key: 60, vel: 90 }),
        RenderEvent::new(0.50, Event::NoteOn { key: 21, vel: 70 }),
        RenderEvent::new(1.30, Event::NoteOff { key: 60, vel: 127 }),
        RenderEvent::new(2.00, Event::NoteOff { key: 21, vel: 30 }),
        RenderEvent::new(2.30, Event::Pedal(PedalEvent::Sustain(1.0))),
        RenderEvent::new(2.50, Event::NoteOn { key: 55, vel: 85 }),
        RenderEvent::new(2.55, Event::NoteOn { key: 72, vel: 85 }),
        RenderEvent::new(3.50, Event::Pedal(PedalEvent::Sustain(0.45))),
        RenderEvent::new(4.20, Event::Pedal(PedalEvent::Sustain(0.55))),
        RenderEvent::new(4.50, Event::Pedal(PedalEvent::Sustain(0.45))),
        RenderEvent::new(5.00, Event::NoteOff { key: 55, vel: 64 }),
        RenderEvent::new(5.20, Event::NoteOff { key: 72, vel: 64 }),
        RenderEvent::new(5.50, Event::Pedal(PedalEvent::Sustain(0.0))),
    ]
}

fn click_pass(preset: &piano_emulator::preset::Preset) {
    println!("\n=== 5. click scan: note-offs, pedal crossings, half-pedal\n");
    let (left, right) = render_to_buffer(preset, &click_phrase(), 7.0);
    for (name, channel) in [("left", &left), ("right", &right)] {
        let all = derivative_outliers(channel, 0.0, 1e-4);
        let clicks: Vec<_> = all.iter().filter(|o| o.1 > 12.0).collect();
        println!(
            "   {name}: {} outlier(s) above ratio 12 with |step| > 1e-4; largest ratio anywhere {:.1} at {:.3} s (|step| {:.2e})",
            clicks.len(),
            all.first().map_or(0.0, |o| o.1),
            all.first().map_or(0.0, |o| o.0),
            all.first().map_or(0.0, |o| o.2),
        );
        for o in clicks.iter().take(8) {
            println!("      at {:.3} s: ratio {:.1}, step {:.2e}", o.0, o.1, o.2);
        }
    }
}

/// Samples whose first difference stands `threshold` times above the RMS first
/// difference of the surrounding 43 ms, with |difference| at least `floor`.
/// Returned worst first as (seconds, ratio, step).
fn derivative_outliers(signal: &[f32], threshold: f64, floor: f64) -> Vec<(f64, f64, f64)> {
    let window = 2048usize;
    let d: Vec<f64> = signal
        .windows(2)
        .map(|w| f64::from(w[1]) - f64::from(w[0]))
        .collect();
    let mut out = Vec::new();
    for (start, chunk) in d.chunks(window).enumerate() {
        let rms = (chunk.iter().map(|&x| x * x).sum::<f64>() / chunk.len() as f64).sqrt();
        for (i, &x) in chunk.iter().enumerate() {
            let ratio = x.abs() / rms.max(1e-12);
            if ratio > threshold && x.abs() > floor {
                out.push(((start * window + i) as f64 / SR, ratio, x.abs()));
            }
        }
    }
    out.sort_by(|a, b| b.1.total_cmp(&a.1));
    out
}

// ------------------------------------------------------------ 6. dense material

fn dense_pass(preset: &piano_emulator::preset::Preset) {
    println!("\n=== 6. dense material: NaN / clipping / DC over 45 s of random playing\n");
    let mut state = 0x9E3779B97F4A7C15u64;
    let mut next = move || {
        state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    };
    let mut events = Vec::new();
    let mut t = 0.1f32;
    while t < 42.0 {
        let key = 21 + (next() % 88) as u8;
        let vel = 20 + (next() % 108) as u8;
        let hold = 0.2 + (next() % 2300) as f32 / 1000.0;
        events.push(RenderEvent::new(t, Event::NoteOn { key, vel }));
        events.push(RenderEvent::new(t + hold, Event::NoteOff { key, vel: 64 }));
        t += 0.03 + (next() % 120) as f32 / 1000.0;
    }
    for i in 0..14 {
        let at = 1.5 + 3.0 * i as f32;
        let value = if i % 2 == 0 { 1.0 } else { 0.0 };
        events.push(RenderEvent::new(at, Event::Pedal(PedalEvent::Sustain(value))));
    }
    let (left, right) = render_to_buffer(preset, &events, 45.0);
    for (name, channel) in [("left", &left), ("right", &right)] {
        let nan = channel.iter().filter(|x| !x.is_finite()).count();
        let peak = channel.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        let clipped = channel.iter().filter(|x| x.abs() >= 1.0).count();
        let dc = channel.iter().map(|&x| f64::from(x)).sum::<f64>() / channel.len() as f64;
        let all = derivative_outliers(channel, 0.0, 1e-4);
        let clicks = all.iter().filter(|o| o.1 > 12.0).count();
        println!(
            "   {name}: {} non-finite, peak {:.2} dBFS, {} samples at or above 1.0, DC {:.1} dB, \
             {} derivative outliers above 12 (largest {:.1} at {:.3} s)",
            nan,
            20.0 * f64::from(peak).log10(),
            clipped,
            20.0 * dc.abs().max(1e-12).log10(),
            clicks,
            all.first().map_or(0.0, |o| o.1),
            all.first().map_or(0.0, |o| o.0),
        );
        for o in all.iter().take(3) {
            let near: Vec<String> = events
                .iter()
                .filter(|e| (f64::from(e.time_s) - o.0).abs() < 0.06)
                .map(|e| format!("{:?}@{:.2}", e.event, e.time_s))
                .collect();
            println!(
                "      at {:.3} s: ratio {:.1}, step {:.2e}   nearby: {}",
                o.0,
                o.1,
                o.2,
                near.join(", "),
            );
        }
    }
}

// --------------------------------------------- 7. fundamental drift from sigma spread

fn drift_pass(
    preset: &piano_emulator::preset::Preset,
    config: &SurveyConfig,
    residual: &ResidualConfig,
) {
    println!("\n=== 7. fundamental drift over its first 20 dB, fitted sigma spread vs unity\n");
    println!("        (recordings: F#3 -31.9 c, C4 -1.7 c; old engine control -2.0..+0.7 c)\n");
    let mut unity = preset.clone();
    for row in &mut unity.voicing.unison_sigma_scale {
        for v in &mut row.scale {
            *v = 1.0;
        }
    }
    // A positive control at the size DECISIONS.md 105 tested, so the
    // measurement's own sensitivity is on the table next to the answer.
    let mut wide = preset.clone();
    if let Some(row) = wide.voicing.unison_sigma_scale.get_mut(2) {
        row.scale = vec![0.7, 1.0, 1.3];
    }
    println!(" key       fitted spread              unity          [0.7,1,1.3] control");
    println!("        glide     f(1s->6s)     glide     f(1s->6s)     glide     f(1s->6s)");
    for key in [54u8, 60] {
        let show = |v: Option<f64>| v.map_or("-".to_string(), |v| format!("{v:+.2}c"));
        let row = |p: &piano_emulator::preset::Preset| {
            (
                fundamental_drift(p, key, config, residual),
                fundamental_time_drift(p, key, config),
            )
        };
        let (ag, at) = row(preset);
        let (bg, bt) = row(&unity);
        let (cg, ct) = row(&wide);
        println!(
            "{key:>4} {:>9} {:>12} {:>9} {:>12} {:>9} {:>12}",
            show(ag),
            show(at),
            show(bg),
            show(bt),
            show(cg),
            show(ct),
        );
    }
}

/// Frequency of the fundamental at 6 s against 1 s, in cents, straight off the
/// track — the drift against *time* that `DECISIONS.md` 124 argues for.
fn fundamental_time_drift(
    preset: &piano_emulator::preset::Preset,
    key: u8,
    config: &SurveyConfig,
) -> Option<f64> {
    let events = [RenderEvent::new(0.0, Event::NoteOn { key, vel: 90 })];
    let (l, r) = render_to_buffer(preset, &events, 8.0);
    let mono: Vec<f32> = l.iter().zip(&r).map(|(&l, &r)| 0.5 * (l + r)).collect();
    let note_config = config.note_config(equal_temperament(key)).ok()?;
    let (trajectories, _) = track_refined(
        &mono,
        SR,
        InharmonicModel::harmonic(equal_temperament(key)),
        &note_config,
    )
    .ok()?;
    let track = trajectories.track(1)?;
    let early = track.frequency_at(trajectories.onset_s + 1.0)?;
    let late = track.frequency_at(trajectories.onset_s + 6.0)?;
    Some(piano_tuner::cents(early, late))
}

fn fundamental_drift(
    preset: &piano_emulator::preset::Preset,
    key: u8,
    config: &SurveyConfig,
    residual: &ResidualConfig,
) -> Option<f64> {
    let events = [RenderEvent::new(0.0, Event::NoteOn { key, vel: 90 })];
    let (l, r) = render_to_buffer(preset, &events, 8.0);
    let mono: Vec<f32> = l.iter().zip(&r).map(|(&l, &r)| 0.5 * (l + r)).collect();
    let note_config = config.note_config(equal_temperament(key)).ok()?;
    let (trajectories, _) = track_refined(
        &mono,
        SR,
        InharmonicModel::harmonic(equal_temperament(key)),
        &note_config,
    )
    .ok()?;
    let analysis = analyze_trajectories(trajectories, &note_config).ok()?;
    partial_residuals(&analysis, residual)
        .iter()
        .find(|p| p.k == 1)
        .and_then(|p| p.glide_cents)
}

// ---------------------------------------------------------------- small tools

fn median(values: impl Iterator<Item = f64>) -> Option<f64> {
    let mut v: Vec<f64> = values.filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return None;
    }
    v.sort_by(f64::total_cmp);
    let mid = v.len() / 2;
    Some(if v.len() % 2 == 0 {
        0.5 * (v[mid - 1] + v[mid])
    } else {
        v[mid]
    })
}

fn rms(values: impl Iterator<Item = f64>) -> f64 {
    let v: Vec<f64> = values.filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return f64::NAN;
    }
    (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt()
}
