//! Independent verification that the timbre ladder in `renders/timbre-ladder/`
//! is what it claims to be, before anyone listens to it.
//!
//! Seven checks, each printed as PASS/FAIL with the number it was decided on:
//!
//! 1. `00_source.wav` is the library recording: cut the library file to the
//!    cached onset with the ladder's own recipe and compare sample by sample,
//!    modulo one fitted gain and the edge fades.
//! 2. `01_resynth_full.wav` tracks the cached trajectories: re-measure its
//!    per-partial envelope at every cached track point (Hann-windowed complex
//!    projection at the point's own measured frequency) and report the mean
//!    absolute error in dB against the cached amplitude, after removing the
//!    one global level-match gain.
//! 3. `07_engine.wav` is the shipped engine: render the note again through
//!    `render_to_buffer` and compare sample by sample modulo one fitted gain
//!    (with a determinism control: the fresh render against itself, twice).
//! 4. Level matching holds: RMS over 0.2–2 s after the strike, every rung
//!    against `00`.
//! 5. No clicks or edge artifacts: first/last sample, largest sample-to-sample
//!    step, peak, and non-finite scan over every file.
//! 6. Two ANALYSIS.md metrics recomputed from the WAVs with re-implemented
//!    DSP: the roughness row (spline and poly RMS) and the per-partial
//!    linewidth, against the numbers the file prints.
//! 7. The ladder is honestly constructed: no two files byte-identical, and
//!    pairwise correlations only high where construction says they must be
//!    (`06`=`07`+attack and `08`=`01`+attack agree past the residual and
//!    differ inside it; every other pair stays apart).
//!
//! ```text
//! cargo run --release -p forensics --bin verify_ladder
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use piano_emulator::preset::Preset as EnginePreset;
use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_emulator::types::Event;
use piano_tuner::estimate::FitSpan;
use piano_tuner::numeric::{poly_eval, weighted_polyfit};
use piano_tuner::pipeline::analyze_trajectories;
use piano_tuner::preset::equal_temperament;
use piano_tuner::survey::{trajectories_for, SurveyConfig};
use piano_tuner::{audio, NoteAnalysis, Sample, SampleLibrary, SAMPLE_RATE};
use rustfft::{num_complex::Complex64, FftPlanner};

const SR: f64 = SAMPLE_RATE as f64;
const PREROLL: usize = 512;
const NOTE_FRAMES: usize = 4 * SAMPLE_RATE as usize;
const TOTAL_FRAMES: usize = PREROLL + NOTE_FRAMES;
const KEYS: [u8; 3] = [60, 45, 84];
const VELOCITY: u8 = 90;
const MATCH_LO_S: f64 = 0.2;
const MATCH_HI_S: f64 = 2.0;

const RUNGS: [&str; 10] = [
    "00_source",
    "01_resynth_full",
    "02_meas_amp_law_decay",
    "03_smooth_amp_meas_decay",
    "04_engine_rough",
    "05_engine_linewidth",
    "06_engine_attack",
    "07_engine",
    "08_resynth_plus_attack",
    "09_engine_modal_control",
];

/// ANALYSIS.md numbers spot-checked by check 6, transcribed from the file:
/// (key name, rung, spline RMS dB, poly RMS dB) …
const REPORTED_ROUGHNESS: [(&str, usize, f64, f64); 3] = [
    ("C4", 0, 7.63, 7.80),
    ("C4", 7, 4.81, 4.79),
    ("C6", 7, 11.06, 9.86),
];
/// … and (key name, rung, k, linewidth cents).
const REPORTED_WIDTH: [(&str, usize, u32, f64); 3] =
    [("C4", 0, 1, 6.71), ("C4", 0, 2, 3.43), ("A2", 0, 1, 15.74)];

// Metric-1/2 constants, matching ANALYSIS.md §0.
const ROUGHNESS_T_S: f64 = 0.3;
const ROUGHNESS_WINDOW: usize = 8192;
const ROUGHNESS_FFT: usize = 65_536;
const SPLINE_LAMBDA: f64 = 1.0;
const WIDTH_LO_S: f64 = 1.5;
const WIDTH_HI_S: f64 = 3.5;
const WIDTH_FFT: usize = 1 << 19;

struct Outcome {
    check: &'static str,
    pass: bool,
    detail: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let root = repo.join("data/salamander");
    let renders = repo.join("renders/timbre-ladder");
    let preset = EnginePreset::load(&repo.join("presets/salamander-c5.toml"))?;
    let library = SampleLibrary::from_sfz(root.join("SalamanderGrandPiano-V3+20200602.sfz"))?;
    let config = SurveyConfig {
        cache_dir: Some(repo.join("data/cache/salamander")),
        ..SurveyConfig::default()
    };

    let mut outcomes: Vec<Outcome> = Vec::new();
    for key in KEYS {
        verify_key(key, &library, &preset, &config, &renders, &mut outcomes)?;
    }

    println!();
    println!("| check | verdict | numbers |");
    println!("|:--|:--|:--|");
    let mut failures = 0;
    for o in &outcomes {
        if !o.pass {
            failures += 1;
        }
        println!(
            "| {} | {} | {} |",
            o.check,
            if o.pass { "PASS" } else { "FAIL" },
            o.detail
        );
    }
    println!();
    println!(
        "{} checks, {} failed",
        outcomes.len(),
        failures
    );
    Ok(())
}

fn verify_key(
    key: u8,
    library: &SampleLibrary,
    preset: &EnginePreset,
    config: &SurveyConfig,
    renders: &Path,
    outcomes: &mut Vec<Outcome>,
) -> Result<(), Box<dyn std::error::Error>> {
    let name = note_name(key);
    let dir = renders.join(&name);
    println!("== {name} ==");

    let sample = layer_for(library, key, VELOCITY)?;
    let note_config = config.note_config(equal_temperament(key))?;
    let trajectories = trajectories_for(sample, &note_config, config)?;
    let onset_s = trajectories.onset_s;
    let span = FitSpan::from_trajectories(&trajectories);
    let analysis = analyze_trajectories(trajectories, &note_config)?;

    let rungs: Vec<(Vec<f32>, Vec<f32>)> = RUNGS
        .iter()
        .map(|stem| {
            let clip = audio::load_wav(dir.join(format!("{stem}.wav")))?;
            let l = clip.channels[0].clone();
            let r = clip.channels[clip.channel_count() - 1].clone();
            Ok::<_, Box<dyn std::error::Error>>((l, r))
        })
        .collect::<Result<_, _>>()?;

    check_source(key, sample, onset_s, &rungs[0], outcomes)?;
    check_resynth(key, &analysis, span, &rungs[1], outcomes);
    check_engine(key, preset, &rungs[7], outcomes);
    check_levels(key, &rungs, outcomes);
    check_edges(key, &rungs, outcomes);
    check_metrics(&name, &analysis, &rungs, outcomes);
    check_distinct(key, &dir, &rungs, outcomes);
    Ok(())
}

// ------------------------------------------------------- 1. rung 00 = source

fn check_source(
    key: u8,
    sample: &Sample,
    onset_s: f64,
    rung: &(Vec<f32>, Vec<f32>),
    outcomes: &mut Vec<Outcome>,
) -> Result<(), Box<dyn std::error::Error>> {
    let recording = audio::load_at(&sample.path, SAMPLE_RATE)?;
    let start = (onset_s * SR).round() as isize - PREROLL as isize;
    let channel = |i: usize| -> Vec<f32> {
        let source = &recording.channels[i.min(recording.channel_count() - 1)];
        (0..TOTAL_FRAMES)
            .map(|n| {
                let index = start + n as isize;
                if index < 0 {
                    0.0
                } else {
                    source.get(index as usize).copied().unwrap_or(0.0)
                }
            })
            .collect()
    };
    let (cut_l, cut_r) = (channel(0), channel(1));

    // Compare away from the ladder's fades: 2 ms in, 30 ms out.
    let lo = 200;
    let hi = TOTAL_FRAMES - 1600;
    let gain = fit_gain(&[&cut_l[lo..hi], &cut_r[lo..hi]], &[&rung.0[lo..hi], &rung.1[lo..hi]]);
    let residual = residual_db(
        &[&cut_l[lo..hi], &cut_r[lo..hi]],
        &[&rung.0[lo..hi], &rung.1[lo..hi]],
        gain,
    );
    let pass = residual < -60.0;
    outcomes.push(Outcome {
        check: leak(format!("1. {}: 00 is the library recording", note_name(key))),
        pass,
        detail: format!("gain {:.4}, residual {:.1} dB re source", gain, residual),
    });
    Ok(())
}

// --------------------------------------- 2. rung 01 tracks the trajectories

fn check_resynth(
    key: u8,
    analysis: &NoteAnalysis,
    span: FitSpan,
    rung: &(Vec<f32>, Vec<f32>),
    outcomes: &mut Vec<Outcome>,
) {
    let mono: Vec<f64> = rung
        .0
        .iter()
        .zip(&rung.1)
        .map(|(&l, &r)| 0.5 * (f64::from(l) + f64::from(r)))
        .collect();

    // One projection window: four periods of the fundamental, clamped like the
    // ladder's own analysis window so neighbours stay separated.
    let f0 = analysis
        .decays
        .partials
        .iter()
        .map(|f| f.frequency_hz / f64::from(f.k))
        .fold(f64::INFINITY, f64::min);
    let window_n = {
        let n = ((4.0 / f0).clamp(0.020, 0.040) * SR).round() as usize;
        n + n % 2
    };

    let mut diffs: Vec<f64> = Vec::new();
    for fit in &analysis.decays.partials {
        let Some(track) = analysis.trajectories.track(fit.k) else {
            continue;
        };
        let peak = track
            .points
            .iter()
            .map(|p| p.amplitude)
            .fold(0.0f64, f64::max);
        if peak <= 0.0 {
            continue;
        }
        for point in &track.points {
            let t = point.time_s - span.onset_s;
            if point.time_s < span.start_s
                || !(0.15..=3.5).contains(&t)
                || point.amplitude <= peak * 1e-3
                || !point.frequency_hz.is_finite()
            {
                continue;
            }
            let centre = PREROLL + (t * SR).round() as usize;
            if centre + window_n / 2 >= mono.len() {
                continue;
            }
            let (re, im) = project(&mono, point.frequency_hz, centre, window_n);
            let measured = (re * re + im * im).sqrt();
            if measured <= 0.0 {
                continue;
            }
            diffs.push(20.0 * (measured / point.amplitude).log10());
        }
    }
    // The file carries one global level-match gain; remove it as the median
    // offset, then the mean absolute error is the tracking error.
    let offset = median(&mut diffs.clone()).unwrap_or(0.0);
    let mae = diffs.iter().map(|d| (d - offset).abs()).sum::<f64>() / diffs.len().max(1) as f64;
    let pass = !diffs.is_empty() && mae <= 1.5;
    outcomes.push(Outcome {
        check: leak(format!(
            "2. {}: 01 tracks the cached trajectories",
            note_name(key)
        )),
        pass,
        detail: format!(
            "{} track points, level-match offset {:+.2} dB, mean |err| {:.2} dB",
            diffs.len(),
            offset,
            mae
        ),
    });
}

// ------------------------------------------------- 3. rung 07 = the engine

fn check_engine(
    key: u8,
    preset: &EnginePreset,
    rung: &(Vec<f32>, Vec<f32>),
    outcomes: &mut Vec<Outcome>,
) {
    let render = |_: ()| -> (Vec<f32>, Vec<f32>) {
        let events = [RenderEvent::new(
            PREROLL as f32 / SAMPLE_RATE as f32,
            Event::NoteOn { key, vel: u16::from(VELOCITY) },
        )];
        let (mut l, mut r) =
            render_to_buffer(preset, &events, TOTAL_FRAMES as f32 / SAMPLE_RATE as f32);
        l.resize(TOTAL_FRAMES, 0.0);
        r.resize(TOTAL_FRAMES, 0.0);
        (l, r)
    };
    let (al, ar) = render(());
    let (bl, br) = render(());
    let determinism = residual_db(&[&al, &ar], &[&bl, &br], 1.0);

    let lo = 200;
    let hi = TOTAL_FRAMES - 1600;
    let gain = fit_gain(&[&al[lo..hi], &ar[lo..hi]], &[&rung.0[lo..hi], &rung.1[lo..hi]]);
    let residual = residual_db(
        &[&al[lo..hi], &ar[lo..hi]],
        &[&rung.0[lo..hi], &rung.1[lo..hi]],
        gain,
    );
    let pass = residual < -60.0;
    outcomes.push(Outcome {
        check: leak(format!("3. {}: 07 is the shipped engine", note_name(key))),
        pass,
        detail: format!(
            "gain {:.4}, residual {:.1} dB re render (render-vs-render {})",
            gain,
            residual,
            if determinism.is_finite() {
                format!("{determinism:.1} dB")
            } else {
                "identical".into()
            }
        ),
    });
}

// ------------------------------------------------------- 4. level matching

fn check_levels(key: u8, rungs: &[(Vec<f32>, Vec<f32>)], outcomes: &mut Vec<Outcome>) {
    let lo = PREROLL + (MATCH_LO_S * SR) as usize;
    let hi = PREROLL + (MATCH_HI_S * SR) as usize;
    let level = |(l, r): &(Vec<f32>, Vec<f32>)| -> f64 {
        let sum: f64 = (lo..hi)
            .map(|i| f64::from(l[i]).powi(2) + f64::from(r[i]).powi(2))
            .sum();
        (sum / (2 * (hi - lo)) as f64).sqrt()
    };
    let reference = level(&rungs[0]);
    let mut worst = 0.0f64;
    for rung in rungs.iter().skip(1) {
        let db = 20.0 * (level(rung) / reference).log10();
        if db.abs() > worst.abs() {
            worst = db;
        }
    }
    outcomes.push(Outcome {
        check: leak(format!("4. {}: rungs level-matched to 00", note_name(key))),
        pass: worst.abs() <= 0.5,
        detail: format!("worst deviation {worst:+.3} dB over 0.2-2 s"),
    });
}

// ------------------------------------------------- 5. clicks and edges

fn check_edges(key: u8, rungs: &[(Vec<f32>, Vec<f32>)], outcomes: &mut Vec<Outcome>) {
    let mut worst_edge = 0.0f64;
    let mut worst_step = 0.0f64;
    let mut worst_peak = 0.0f64;
    let mut finite = true;
    for (l, r) in rungs {
        for channel in [l, r] {
            finite &= channel.iter().all(|v| v.is_finite());
            worst_edge = worst_edge
                .max(f64::from(channel[0]).abs())
                .max(f64::from(*channel.last().expect("non-empty")).abs());
            worst_peak = channel
                .iter()
                .fold(worst_peak, |m, &v| m.max(f64::from(v).abs()));
            worst_step = channel
                .windows(2)
                .fold(worst_step, |m, w| m.max(f64::from(w[1] - w[0]).abs()));
        }
    }
    let pass = finite && worst_edge <= 1e-3 && worst_step <= 0.25 && worst_peak < 1.0;
    outcomes.push(Outcome {
        check: leak(format!("5. {}: no clicks or edge artifacts", note_name(key))),
        pass,
        detail: format!(
            "edges <= {:.1e}, max step {:.3}, peak {:.3}, finite {}",
            worst_edge, worst_step, worst_peak, finite
        ),
    });
}

// --------------------------------------- 6. ANALYSIS.md numbers recomputed

fn check_metrics(
    name: &str,
    analysis: &NoteAnalysis,
    rungs: &[(Vec<f32>, Vec<f32>)],
    outcomes: &mut Vec<Outcome>,
) {
    let mut partials: Vec<(u32, f64)> = analysis
        .decays
        .partials
        .iter()
        .filter(|fit| fit.frequency_hz.is_finite() && fit.frequency_hz > 0.0)
        .map(|fit| (fit.k, fit.frequency_hz))
        .collect();
    partials.sort_by_key(|&(k, _)| k);
    let f0 = partials[0].1 / f64::from(partials[0].0);

    for &(key_name, rung, spline_db, poly_db) in &REPORTED_ROUGHNESS {
        if key_name != name {
            continue;
        }
        let mono: Vec<f64> = rungs[rung]
            .0
            .iter()
            .zip(&rungs[rung].1)
            .map(|(&l, &r)| 0.5 * (f64::from(l) + f64::from(r)))
            .collect();
        let (spline, poly) = roughness(&mono, &partials, f0);
        let err = (spline - spline_db).abs().max((poly - poly_db).abs());
        outcomes.push(Outcome {
            check: leak(format!(
                "6. {name}: ANALYSIS.md roughness of rung {:02}",
                rung
            )),
            pass: err <= 0.05,
            detail: format!(
                "spline {spline:.2} dB (reported {spline_db:.2}), poly {poly:.2} dB (reported {poly_db:.2})"
            ),
        });
    }

    for &(key_name, rung, k, cents) in &REPORTED_WIDTH {
        if key_name != name {
            continue;
        }
        let mono: Vec<f64> = rungs[rung]
            .0
            .iter()
            .zip(&rungs[rung].1)
            .map(|(&l, &r)| 0.5 * (f64::from(l) + f64::from(r)))
            .collect();
        let lo = (PREROLL + (WIDTH_LO_S * SR) as usize).min(mono.len());
        let hi = (PREROLL + (WIDTH_HI_S * SR) as usize).min(mono.len());
        let spectrum = magnitude_spectrum(&mono[lo..hi], WIDTH_FFT);
        let f = partials
            .iter()
            .find(|&&(order, _)| order == k)
            .map(|&(_, f)| f)
            .unwrap_or(f64::NAN);
        let width = partial_width_cents(&spectrum, f, f0).unwrap_or(f64::NAN);
        outcomes.push(Outcome {
            check: leak(format!(
                "6. {name}: ANALYSIS.md linewidth k={k} of rung {:02}",
                rung
            )),
            pass: (width - cents).abs() <= 0.05,
            detail: format!("{width:.2} cents (reported {cents:.2})"),
        });
    }
}

// ------------------------------------------- 7. honest construction

fn check_distinct(
    key: u8,
    dir: &Path,
    rungs: &[(Vec<f32>, Vec<f32>)],
    outcomes: &mut Vec<Outcome>,
) {
    // Byte-identity, on the files themselves.
    let mut digests: HashMap<u64, Vec<&str>> = HashMap::new();
    for stem in RUNGS {
        let bytes = std::fs::read(dir.join(format!("{stem}.wav"))).unwrap_or_default();
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for b in bytes {
            hash = (hash ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01B3);
        }
        digests.entry(hash).or_default().push(stem);
    }
    let duplicates: Vec<String> = digests
        .values()
        .filter(|v| v.len() > 1)
        .map(|v| v.join("=") )
        .collect();

    // Pairwise correlation over the whole file. The construction itself names
    // the pairs that are allowed to agree: (06,07) and (08,01) differ only by
    // the attack residual — identical past its 0.2 s fade-out, different
    // inside the first 150 ms — and 09 is a re-implementation of the engine's
    // string and board, so (07,09) and (06,09) correlate to the degree that
    // the engine beyond the string is small. For those two the honest-copy
    // question is different: 09 must be independently *computed* audio, not
    // 07's samples with a gain on them, so the gain-fitted residual must sit
    // far above the −150 dB floor check 3 measures for a true re-render.
    let attack_hi = PREROLL + (0.15 * SR) as usize;
    let tail_lo = PREROLL + (0.21 * SR) as usize;
    let mut expected_ok = true;
    let mut expected_detail = String::new();
    let mut unexpected: Vec<String> = Vec::new();
    let mut highest_other = (String::new(), 0.0f64);
    for i in 0..rungs.len() {
        for j in i + 1..rungs.len() {
            let whole = correlation(&rungs[i], &rungs[j], 0, TOTAL_FRAMES);
            let pair = (RUNGS[i], RUNGS[j]);
            if (i, j) == (6, 7) || (i, j) == (1, 8) {
                let attack = correlation(&rungs[i], &rungs[j], PREROLL, attack_hi);
                // Same audio past the residual, different audio inside it.
                let tail = correlation(&rungs[i], &rungs[j], tail_lo, TOTAL_FRAMES - 1600);
                expected_ok &= tail > 0.999999 && attack < 0.999;
                expected_detail.push_str(&format!(
                    "{}~{}: tail r={:.6}, attack r={:.3}; ",
                    &pair.0[..2],
                    &pair.1[..2],
                    tail,
                    attack
                ));
            } else if (i, j) == (7, 9) || (i, j) == (6, 9) {
                let (x, y) = (&rungs[i], &rungs[j]);
                let gain = fit_gain(&[&x.0, &x.1], &[&y.0, &y.1]);
                let residual = residual_db(&[&x.0, &x.1], &[&y.0, &y.1], gain);
                expected_ok &= residual > -80.0;
                expected_detail.push_str(&format!(
                    "{}~{}: r={:.4}, gain-fitted residual {:.0} dB; ",
                    &pair.0[..2],
                    &pair.1[..2],
                    whole,
                    residual
                ));
            } else if whole > 0.99 {
                unexpected.push(format!("{}~{} r={:.4}", &pair.0[..2], &pair.1[..2], whole));
            } else if whole > highest_other.1 {
                highest_other = (format!("{}~{}", &pair.0[..2], &pair.1[..2]), whole);
            }
        }
    }

    let pass = duplicates.is_empty() && unexpected.is_empty() && expected_ok;
    outcomes.push(Outcome {
        check: leak(format!("7. {}: rungs are distinct audio", note_name(key))),
        pass,
        detail: format!(
            "{}{}{}next-highest r: {} {:.3}",
            if duplicates.is_empty() {
                String::new()
            } else {
                format!("byte-identical: {}; ", duplicates.join(", "))
            },
            if unexpected.is_empty() {
                String::new()
            } else {
                format!("unexpectedly correlated: {}; ", unexpected.join(", "))
            },
            expected_detail,
            highest_other.0,
            highest_other.1
        ),
    });
}

// --------------------------------------------------------------- shared DSP

/// Hann-windowed complex projection of `signal` onto `f`, centred on `centre`.
fn project(signal: &[f64], f: f64, centre: usize, window_n: usize) -> (f64, f64) {
    let half = window_n / 2;
    let start = centre.saturating_sub(half);
    let omega = std::f64::consts::TAU * f / SR;
    let mut re = 0.0;
    let mut im = 0.0;
    let mut weight = 0.0;
    for i in 0..window_n {
        let index = start + i;
        if index >= signal.len() {
            break;
        }
        let w = 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / window_n as f64).cos();
        let phase = omega * index as f64;
        re += w * signal[index] * phase.cos();
        im -= w * signal[index] * phase.sin();
        weight += w;
    }
    (2.0 * re / weight.max(1e-30), 2.0 * im / weight.max(1e-30))
}

/// Least-squares gain putting `x` onto `y`.
fn fit_gain(x: &[&[f32]], y: &[&[f32]]) -> f64 {
    let mut xy = 0.0f64;
    let mut xx = 0.0f64;
    for (a, b) in x.iter().zip(y) {
        for (&u, &v) in a.iter().zip(b.iter()) {
            xy += f64::from(u) * f64::from(v);
            xx += f64::from(u) * f64::from(u);
        }
    }
    if xx > 0.0 {
        xy / xx
    } else {
        0.0
    }
}

/// RMS of `y - gain*x` relative to the RMS of `gain*x`, in dB.
fn residual_db(x: &[&[f32]], y: &[&[f32]], gain: f64) -> f64 {
    let mut err = 0.0f64;
    let mut reference = 0.0f64;
    for (a, b) in x.iter().zip(y) {
        for (&u, &v) in a.iter().zip(b.iter()) {
            let scaled = gain * f64::from(u);
            err += (f64::from(v) - scaled).powi(2);
            reference += scaled * scaled;
        }
    }
    if err == 0.0 {
        return f64::NEG_INFINITY;
    }
    10.0 * (err / reference.max(1e-30)).log10()
}

/// Normalised correlation of two stereo buffers over `[lo, hi)`.
fn correlation(a: &(Vec<f32>, Vec<f32>), b: &(Vec<f32>, Vec<f32>), lo: usize, hi: usize) -> f64 {
    let mut ab = 0.0f64;
    let mut aa = 0.0f64;
    let mut bb = 0.0f64;
    for (x, y) in [(&a.0, &b.0), (&a.1, &b.1)] {
        for i in lo..hi.min(x.len()).min(y.len()) {
            let (u, v) = (f64::from(x[i]), f64::from(y[i]));
            ab += u * v;
            aa += u * u;
            bb += v * v;
        }
    }
    ab / (aa.sqrt() * bb.sqrt()).max(1e-30)
}

/// ANALYSIS.md metric 1, re-implemented: per-partial deviation at 0.3 s from a
/// penalised octave spline and from a degree-2 polynomial in `ln k`, dB RMS.
fn roughness(mono: &[f64], partials: &[(u32, f64)], f0: f64) -> (f64, f64) {
    let centre = PREROLL + (ROUGHNESS_T_S * SR) as usize;
    let start = centre.saturating_sub(ROUGHNESS_WINDOW / 2).min(mono.len());
    let end = (start + ROUGHNESS_WINDOW).min(mono.len());
    let spectrum = magnitude_spectrum(&mono[start..end], ROUGHNESS_FFT);
    let bin = SR / ROUGHNESS_FFT as f64;

    let mut x = Vec::new();
    let mut lnk = Vec::new();
    let mut y = Vec::new();
    for &(k, f) in partials {
        let Some((_, level_db)) = peak_near(&spectrum, bin, f, guard_hz(f, f0)) else {
            continue;
        };
        if !level_db.is_finite() {
            continue;
        }
        x.push(f.log2());
        lnk.push(f64::from(k).ln());
        y.push(level_db);
    }
    let spline_fit = octave_spline(&x, &y);
    let spline: Vec<f64> = y.iter().zip(&spline_fit).map(|(&v, &s)| v - s).collect();
    let weights = vec![1.0; y.len()];
    let coefficients =
        weighted_polyfit(&lnk, &y, &weights, 2.min(y.len() - 1)).unwrap_or_default();
    let poly: Vec<f64> = lnk
        .iter()
        .zip(&y)
        .map(|(&u, &v)| v - poly_eval(&coefficients, u))
        .collect();
    (rms(&spline), rms(&poly))
}

fn guard_hz(f: f64, f0: f64) -> f64 {
    (0.03 * f).min(0.45 * f0)
}

fn rms(values: &[f64]) -> f64 {
    (values.iter().map(|v| v * v).sum::<f64>() / values.len().max(1) as f64).sqrt()
}

fn magnitude_spectrum(samples: &[f64], fft_size: usize) -> Vec<f64> {
    let n = samples.len().min(fft_size);
    let mut buffer = vec![Complex64::new(0.0, 0.0); fft_size];
    let mut weight = 0.0;
    for (i, value) in samples.iter().take(n).enumerate() {
        let w = 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / n as f64).cos();
        weight += w;
        buffer[i] = Complex64::new(value * w, 0.0);
    }
    FftPlanner::new().plan_fft_forward(fft_size).process(&mut buffer);
    let scale = 2.0 / weight.max(1e-30);
    buffer[..fft_size / 2]
        .iter()
        .map(|c| c.norm() * scale)
        .collect()
}

fn amp_db(a: f64) -> f64 {
    20.0 * a.max(1e-30).log10()
}

fn peak_bin_near(spectrum: &[f64], bin: f64, f: f64, guard: f64) -> Option<(usize, f64)> {
    let lo = (((f - guard) / bin).ceil().max(1.0)) as usize;
    let hi = (((f + guard) / bin).floor() as usize).min(spectrum.len().saturating_sub(2));
    if hi <= lo {
        return None;
    }
    let mut best = lo;
    for i in lo..=hi {
        if spectrum[i] > spectrum[best] {
            best = i;
        }
    }
    Some((best, amp_db(spectrum[best])))
}

fn peak_near(spectrum: &[f64], bin: f64, f: f64, guard: f64) -> Option<(f64, f64)> {
    let (index, _) = peak_bin_near(spectrum, bin, f, guard)?;
    if index == 0 || index + 1 >= spectrum.len() {
        return Some((index as f64 * bin, amp_db(spectrum[index])));
    }
    let (a, b, c) = (
        amp_db(spectrum[index - 1]),
        amp_db(spectrum[index]),
        amp_db(spectrum[index + 1]),
    );
    let denominator = a - 2.0 * b + c;
    let offset = if denominator.abs() > 1e-12 {
        (0.5 * (a - c) / denominator).clamp(-0.5, 0.5)
    } else {
        0.0
    };
    Some(((index as f64 + offset) * bin, b - 0.25 * (a - c) * offset))
}

/// ANALYSIS.md metric 2, re-implemented: −6 dB width in cents of the peak
/// nearest `f` on the Hann-windowed 1.5–3.5 s segment.
fn partial_width_cents(spectrum: &[f64], f: f64, f0: f64) -> Option<f64> {
    let bin = SR / WIDTH_FFT as f64;
    let guard = guard_hz(f, f0);
    let (peak_bin, peak_db) = peak_bin_near(spectrum, bin, f, guard)?;
    let target = peak_db - 6.0;
    let db = |i: usize| -> f64 { amp_db(spectrum[i]) };
    let limit = (guard / bin) as usize;

    let mut low = None;
    let mut i = peak_bin;
    while i > 0 && peak_bin - i < limit {
        if db(i - 1) <= target {
            let (a, b) = (db(i - 1), db(i));
            let frac = if (b - a).abs() > 1e-12 {
                (target - a) / (b - a)
            } else {
                0.0
            };
            low = Some(((i - 1) as f64 + frac) * bin);
            break;
        }
        i -= 1;
    }
    let mut high = None;
    let mut i = peak_bin;
    while i + 1 < spectrum.len() && i - peak_bin < limit {
        if db(i + 1) <= target {
            let (a, b) = (db(i), db(i + 1));
            let frac = if (a - b).abs() > 1e-12 {
                (a - target) / (a - b)
            } else {
                0.0
            };
            high = Some((i as f64 + frac) * bin);
            break;
        }
        i += 1;
    }
    let (low, high) = (low?, high?);
    if !(low > 0.0 && high > low) {
        return None;
    }
    Some(1200.0 * (high / low).log2())
}

/// The penalised octave spline of ANALYSIS.md metric 1: linear B-splines on an
/// octave grid in `log2 f`, second-difference penalty `SPLINE_LAMBDA`.
fn octave_spline(x: &[f64], y: &[f64]) -> Vec<f64> {
    let lo = x.iter().cloned().fold(f64::INFINITY, f64::min).floor();
    let hi = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max).ceil();
    let knots: Vec<f64> = {
        let count = ((hi - lo).round() as usize).max(1) + 1;
        (0..count).map(|i| lo + i as f64).collect()
    };
    let n = knots.len();
    let hat = |j: usize, u: f64| -> f64 { (1.0 - (u - knots[j]).abs()).max(0.0) };

    let mut normal = vec![0.0; n * n];
    let mut rhs = vec![0.0; n];
    for (&u, &value) in x.iter().zip(y) {
        let row: Vec<f64> = (0..n).map(|j| hat(j, u)).collect();
        for i in 0..n {
            rhs[i] += row[i] * value;
            for j in 0..n {
                normal[i * n + j] += row[i] * row[j];
            }
        }
    }
    if n >= 3 {
        for j in 1..n - 1 {
            let stencil = [(j - 1, 1.0), (j, -2.0), (j + 1, 1.0)];
            for &(a, va) in &stencil {
                for &(b, vb) in &stencil {
                    normal[a * n + b] += SPLINE_LAMBDA * va * vb;
                }
            }
        }
    }
    for i in 0..n {
        normal[i * n + i] += 1e-9;
    }
    let coefficients = solve(&mut normal, &mut rhs, n).unwrap_or_else(|| vec![0.0; n]);
    x.iter()
        .map(|&u| (0..n).map(|j| coefficients[j] * hat(j, u)).sum())
        .collect()
}

/// Gaussian elimination with partial pivoting for the spline's normal system.
fn solve(a: &mut [f64], b: &mut [f64], n: usize) -> Option<Vec<f64>> {
    for column in 0..n {
        let pivot = (column..n).max_by(|&i, &j| {
            a[i * n + column]
                .abs()
                .partial_cmp(&a[j * n + column].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
        if a[pivot * n + column].abs() < 1e-14 {
            return None;
        }
        if pivot != column {
            for j in 0..n {
                a.swap(column * n + j, pivot * n + j);
            }
            b.swap(column, pivot);
        }
        for row in column + 1..n {
            let factor = a[row * n + column] / a[column * n + column];
            for j in column..n {
                a[row * n + j] -= factor * a[column * n + j];
            }
            b[row] -= factor * b[column];
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let mut sum = b[row];
        for j in row + 1..n {
            sum -= a[row * n + j] * x[j];
        }
        x[row] = sum / a[row * n + row];
    }
    Some(x)
}

fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    Some(if values.len() % 2 == 1 {
        values[mid]
    } else {
        0.5 * (values[mid - 1] + values[mid])
    })
}

fn layer_for(
    library: &SampleLibrary,
    key: u8,
    velocity: u8,
) -> Result<&Sample, Box<dyn std::error::Error>> {
    library
        .layers(key)
        .iter()
        .find(|s| (s.lovel..=s.hivel).contains(&velocity))
        .ok_or_else(|| format!("key {key} has no layer covering velocity {velocity}").into())
}

fn note_name(key: u8) -> String {
    const NAMES: [&str; 12] = [
        "C", "Cs", "D", "Ds", "E", "F", "Fs", "G", "Gs", "A", "As", "B",
    ];
    format!("{}{}", NAMES[usize::from(key) % 12], i32::from(key) / 12 - 1)
}

fn leak(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}
