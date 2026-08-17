//! The measurements behind `docs/history/TUNING_REPORT.md`: what stage 1 could not fit, and
//! what the engine's own renders return from the same code.
//!
//! Six passes, each printed as a table:
//!
//! 1. **Trajectory residuals**, every sampled key and every velocity layer,
//!    read through the survey's trajectory cache: how far the measured partials
//!    stand from `k f0 sqrt(1 + B k^2)`, how far each partial's pitch slides
//!    while it decays, what the two-exponential envelope model leaves over, and
//!    how much the measured excitation spectrum scatters around the smooth
//!    envelope times `sin(k pi x)` that the engine's mode gains are.
//! 2. **The same numbers per velocity layer** for a few keys, because a
//!    nonlinearity shows itself as a residual that grows with the blow.
//! 3. **The engine's own renders**, analysed by the same pipeline. This is the
//!    control: a residual that comes back on material the model generated is
//!    the estimator's, and only the difference is the piano's.
//! 4. **A spectrum census**: every peak of a sustained frame classified as a
//!    transverse partial, a phantom at `f_i +- f_j`, another key's pitch, or
//!    unexplained — and the broadband energy between all of them, at the strike
//!    and one second later.
//! 5. **Stereo balance** per partial, which the engine's one pan per key cannot
//!    reproduce if it varies.
//! 6. **The mechanism recordings** Salamander ships and the engine does not
//!    model at all: 88 key-off samples, four pedal actions, and the
//!    string-resonance samples.
//!
//! ```text
//! cargo run --release -p piano-tuner -- residuals \
//!     [data/salamander] [presets/salamander-c5.toml] [data/cache/salamander]
//! ```
//!
//! Runs in a few minutes against a warm trajectory cache (`data/cache/salamander`).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_emulator::types::Event;
use piano_tuner::estimate::inharmonic::{
    fit_inharmonic_partials, trusted_prefix, InharmonicConfig,
};
use piano_tuner::estimate::strike::{fit_strike_position, StrikeConfig};
use piano_tuner::pipeline::{analyze_trajectories, track_refined, NoteAnalysis};
use piano_tuner::preset::{equal_temperament, Preset};
use piano_tuner::residual::{
    band_split, classify_peaks, excitation_scatter, frame_spectrum, partial_levels,
    partial_residuals, stereo_balance, transient_metrics, PeakClass, ResidualConfig,
};
use piano_tuner::stft::find_peaks;
use piano_tuner::survey::{trajectories_for, SurveyConfig};
use piano_tuner::trajectory::InharmonicModel;
use piano_tuner::{audio, Sample, SampleLibrary, SAMPLE_RATE};

/// Keys the audio-domain passes work on: the sampled keys nearest to one per
/// octave, so that every register is measured without decoding the library
/// twice over.
const CENSUS_KEYS: [u8; 8] = [21, 33, 45, 57, 60, 72, 84, 96];

/// Velocity layers the audio-domain passes use: soft, mezzo-forte and loud.
const CENSUS_LAYERS: [u8; 3] = [3, 8, 13];

/// Velocity the control renders are struck at, matching `ab`.
const CONTROL_VELOCITY: u8 = 90;

const CONTROL_SECONDS: f32 = 8.0;

pub fn run(args: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
    let mut args = args.into_iter();
    let root = PathBuf::from(args.next().unwrap_or_else(|| "data/salamander".into()));
    let preset_path = args
        .next()
        .unwrap_or_else(|| "presets/salamander-c5.toml".into());
    let cache = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "data/cache/salamander".into()),
    );

    let library = SampleLibrary::from_sfz(root.join("SalamanderGrandPiano-V3+20200602.sfz"))?;
    let preset = Preset::load(&preset_path)?;
    let config = SurveyConfig {
        cache_dir: Some(cache),
        ..SurveyConfig::default()
    };
    let residual = ResidualConfig::default();

    let (summaries, excitation) = trajectory_pass(&library, &preset, &config, &residual);
    report_by_key(&summaries);
    report_by_velocity(&summaries, &[21, 36, 48, 60, 72, 84]);
    report_partials(&library, &config, &residual, &[(21, 11), (36, 11), (60, 11), (84, 11)]);
    report_excitation_by_frequency(&excitation);
    report_attack_glide(&library, &preset_path, &[36, 48, 60, 72])?;
    report_control(&preset, &preset_path, &config, &residual)?;
    census_pass(&library, &preset, &config, &residual, &preset_path)?;
    phantom_pass(&library, &config, &preset_path)?;
    stereo_pass(&library, &config, &preset_path)?;
    mechanism_pass(&library, &root)?;
    Ok(())
}

// ------------------------------------------------------ trajectory-domain pass

/// What one recording's residuals came to.
#[derive(Clone, Copy, Debug)]
struct LayerSummary {
    key: u8,
    midi_velocity: u8,
    partials: usize,
    /// RMS and worst deviation of the measured partials from the fitted
    /// stiff-string law, in cents.
    inharmonic_rms_cents: f64,
    inharmonic_worst_cents: f64,
    inharmonic_worst_k: u32,
    /// Pitch slide of the fundamental over its first 20 dB, cents.
    glide_k1: Option<f64>,
    /// Median slide over every measured partial, cents.
    glide_median: Option<f64>,
    /// Median envelope-fit residual over the partials, dB.
    envelope_rms_db: f64,
    /// Median systematic part of it at the end of the fitted span, dB.
    envelope_trend_db: Option<f64>,
    /// Scatter of the measured excitation spectrum around comb times envelope.
    scatter_rms_db: Option<f64>,
    scatter_worst_db: Option<f64>,
    /// The same deviation, against the partial layout the *preset* writes for
    /// this key rather than against this layer's own fit: what the engine will
    /// actually put on the bridge, against what was recorded.
    preset_rms_cents: f64,
    preset_worst_cents: f64,
    preset_worst_k: u32,
    /// The same deviation over the partials whose *index* the tracker can be
    /// believed — everything below the first skipped partial
    /// (`estimate::inharmonic::trusted_prefix`). Above the skip the tracked
    /// frequencies are real and their `k` is one too low, so a deviation
    /// measured there is the tracker's and not the instrument's.
    preset_rms_trusted_cents: f64,
    trusted_partials: usize,
    /// Inharmonicity fitted to the low partials alone and to the high ones
    /// alone. The engine's string has one `B` for the whole series; a real
    /// wound string does not, and the ratio is by how much.
    b_low: Option<f64>,
    b_high: Option<f64>,
}

fn summarise(
    key: u8,
    midi_velocity: u8,
    analysis: &NoteAnalysis,
    config: &ResidualConfig,
    written: Option<InharmonicModel>,
) -> (LayerSummary, Vec<(f64, f64)>) {
    let residuals = partial_residuals(analysis, config);
    let trusted = trusted_prefix(
        &residuals
            .iter()
            .map(|r| (r.k, r.frequency_hz))
            .collect::<Vec<_>>(),
        &InharmonicConfig::default(),
    );
    // The fit's own residual is reported over the partials the fit was allowed
    // to believe. Above the first skipped partial the tracked frequencies are
    // real and their index is one too low, so what a residual measures there is
    // the tracker rather than the string — at A0 it comes to 800 cents.
    let cents: Vec<f64> = residuals[..trusted]
        .iter()
        .map(|r| r.model_cents)
        .collect();
    let (worst, worst_k) = residuals[..trusted]
        .iter()
        .fold((0.0f64, 0u32), |(worst, k), r| {
            if r.model_cents.abs() > worst {
                (r.model_cents.abs(), r.k)
            } else {
                (worst, k)
            }
        });
    let preset: Vec<f64> = written
        .map(|model| {
            residuals
                .iter()
                .map(|r| model.cents_from_partial(r.k, r.frequency_hz))
                .collect()
        })
        .unwrap_or_default();
    let (preset_worst, preset_worst_k) = residuals.iter().zip(&preset).fold(
        (0.0f64, 0u32),
        |(worst, k), (r, &cents)| {
            if cents.abs() > worst {
                (cents.abs(), r.k)
            } else {
                (worst, k)
            }
        },
    );
    let strike = fit_strike_position(
        &analysis.decays.excitation_spectrum(),
        &StrikeConfig::default(),
    )
    .ok();
    let spectrum = analysis.decays.excitation_spectrum();
    let scatter = strike.as_ref().and_then(|fit| excitation_scatter(&spectrum, fit));
    // The same residual partial by partial, against frequency rather than
    // against `k`: what a soundboard imprints is a function of frequency and is
    // shared between neighbouring notes, what an estimator's noise does is
    // neither.
    let by_frequency: Vec<(f64, f64)> = strike
        .as_ref()
        .map(|fit| {
            spectrum
                .iter()
                .filter_map(|&(k, amplitude)| {
                    let fit_amplitude = fit.amplitude(k);
                    let measured = analysis.decays.fit(k)?;
                    (amplitude > 0.0 && fit_amplitude > 0.0).then(|| {
                        (
                            measured.frequency_hz,
                            20.0 * (amplitude / fit_amplitude).log10(),
                        )
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    let summary = LayerSummary {
        key,
        midi_velocity,
        partials: residuals.len(),
        inharmonic_rms_cents: rms(&cents),
        inharmonic_worst_cents: worst,
        inharmonic_worst_k: worst_k,
        // Only partials that stand within 20 dB of the loudest are asked where
        // their pitch went: below that a track spends its measurements in the
        // recording's floor, and what it reports is the floor's frequency. A0's
        // fundamental — 43 dB below its own third partial — is the case that
        // forces this.
        glide_k1: residuals
            .iter()
            .find(|r| r.k == 1 && r.level_db > -20.0)
            .and_then(|r| r.glide_cents),
        glide_median: median(
            residuals
                .iter()
                .filter(|r| r.level_db > -20.0)
                .filter_map(|r| r.glide_cents),
        ),
        envelope_rms_db: median(residuals.iter().map(|r| r.envelope_residual_db)).unwrap_or(f64::NAN),
        envelope_trend_db: median(residuals.iter().filter_map(|r| r.envelope_trend_db)),
        scatter_rms_db: scatter.map(|s| s.rms_db),
        scatter_worst_db: scatter.map(|s| s.worst_db),
        preset_rms_cents: rms(&preset),
        preset_worst_cents: preset_worst,
        preset_worst_k,
        preset_rms_trusted_cents: rms(&preset[..trusted.min(preset.len())]),
        trusted_partials: trusted,
        b_low: band_inharmonicity(&residuals, 1, 8),
        b_high: band_inharmonicity(&residuals, 14, 26),
    };
    (summary, by_frequency)
}

/// `B` fitted to one stretch of the partial series alone.
///
/// The stiff-string law has one `B` for the whole series. Over a narrow band of
/// `k` the pair `(f0, B)` is correlated — a smaller `B` can be traded against a
/// higher `f0` — so neither number means much on its own; what the two bands
/// compare is the *curvature* of the measured series, low against high, and
/// that is what a single `B` has to reproduce and cannot when the two disagree.
fn band_inharmonicity(residuals: &[piano_tuner::residual::PartialResidual], lo: u32, hi: u32) -> Option<f64> {
    let partials: Vec<(u32, f64)> = residuals
        .iter()
        .filter(|r| r.level_db > -40.0 && (lo..=hi).contains(&r.k))
        .map(|r| (r.k, r.frequency_hz))
        .collect();
    if partials.len() < 5 {
        return None;
    }
    fit_inharmonic_partials(
        &partials,
        &InharmonicConfig {
            min_partials: 4,
            reject_cents: 40.0,
            ..InharmonicConfig::default()
        },
    )
    .ok()
    .map(|fit| fit.model.b)
}

/// Every recording of the library, through the trajectory cache.
fn trajectory_pass(
    library: &SampleLibrary,
    preset: &Preset,
    config: &SurveyConfig,
    residual: &ResidualConfig,
) -> (Vec<LayerSummary>, Vec<(u8, f64, f64)>) {
    let samples: Vec<&Sample> = library.samples().collect();
    let next = AtomicUsize::new(0);
    let mut out: Vec<LayerSummary> = Vec::new();
    let mut excitation: Vec<(u8, f64, f64)> = Vec::new();
    let workers = std::thread::available_parallelism().map_or(1, |n| n.get());
    eprintln!("reading {} recordings on {workers} threads", samples.len());
    std::thread::scope(|scope| {
        let (tx, rx) = std::sync::mpsc::channel();
        for _ in 0..workers {
            let (next, samples, tx) = (&next, &samples, tx.clone());
            scope.spawn(move || loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                let Some(sample) = samples.get(index) else {
                    return;
                };
                let summary = analyse(sample, config).map(|analysis| {
                    summarise(
                        sample.key,
                        sample.midi_velocity(),
                        &analysis,
                        residual,
                        written_model(preset, sample.key),
                    )
                });
                if tx.send(summary).is_err() {
                    return;
                }
            });
        }
        drop(tx);
        for (summary, by_frequency) in rx.into_iter().flatten() {
            excitation.extend(by_frequency.into_iter().map(|(f, db)| (summary.key, f, db)));
            out.push(summary);
        }
    });
    out.sort_by_key(|s| (s.key, s.midi_velocity));
    (out, excitation)
}

fn analyse(sample: &Sample, config: &SurveyConfig) -> piano_tuner::Result<NoteAnalysis> {
    let note_config = config.note_config(equal_temperament(sample.key))?;
    let trajectories = trajectories_for(sample, &note_config, config)?;
    analyze_trajectories(trajectories, &note_config)
}

fn report_by_key(summaries: &[LayerSummary]) {
    println!("\n=== 1. trajectory residuals, median over the layers of each key\n");
    println!(
        " key   n  partials   inharm rms  worst (k)   glide k1  glide med   env rms  env trend  \
         excite rms  excite worst   preset rms  worst (k)  trusted   preset rms(t)    B low     B high   ratio"
    );
    for (key, group) in grouped(summaries) {
        let show = |v: Option<f64>| v.map_or("-".to_string(), |v| format!("{v:+7.2}"));
        let low = median(group.iter().filter_map(|s| s.b_low));
        let high = median(group.iter().filter_map(|s| s.b_high));
        println!(
            "{key:>4} {:>3}  {:>8}  {:>10.2}c {:>7.1}c ({:>2}) {:>9} {:>10} {:>9.2} {:>10} {:>11} \
             {:>13} {:>11.2}c {:>7.1}c ({:>2}) {:>8} {:>14.2}c {:>9} {:>10} {:>7}",
            group.len(),
            median(group.iter().map(|s| s.partials as f64)).unwrap_or(f64::NAN) as usize,
            median(group.iter().map(|s| s.inharmonic_rms_cents)).unwrap_or(f64::NAN),
            median(group.iter().map(|s| s.inharmonic_worst_cents)).unwrap_or(f64::NAN),
            median(group.iter().map(|s| f64::from(s.inharmonic_worst_k))).unwrap_or(f64::NAN)
                as u32,
            show(median(group.iter().filter_map(|s| s.glide_k1))),
            show(median(group.iter().filter_map(|s| s.glide_median))),
            median(group.iter().map(|s| s.envelope_rms_db)).unwrap_or(f64::NAN),
            show(median(group.iter().filter_map(|s| s.envelope_trend_db))),
            show(median(group.iter().filter_map(|s| s.scatter_rms_db))),
            show(median(group.iter().filter_map(|s| s.scatter_worst_db))),
            median(group.iter().map(|s| s.preset_rms_cents)).unwrap_or(f64::NAN),
            median(group.iter().map(|s| s.preset_worst_cents)).unwrap_or(f64::NAN),
            median(group.iter().map(|s| f64::from(s.preset_worst_k))).unwrap_or(f64::NAN) as u32,
            median(group.iter().map(|s| s.trusted_partials as f64)).unwrap_or(f64::NAN) as usize,
            median(group.iter().map(|s| s.preset_rms_trusted_cents)).unwrap_or(f64::NAN),
            low.map_or("-".to_string(), |b| format!("{b:.2e}")),
            high.map_or("-".to_string(), |b| format!("{b:.2e}")),
            low.zip(high)
                .map_or("-".to_string(), |(l, h)| format!("{:.2}", h / l)),
        );
    }
}

fn report_by_velocity(summaries: &[LayerSummary], keys: &[u8]) {
    println!("\n=== 2. the same, layer by layer, for a few keys\n");
    println!(" key  vel  partials   inharm rms   glide k1  glide med   env rms  env trend  excite rms");
    for &key in keys {
        for summary in summaries.iter().filter(|s| s.key == key) {
            let show = |v: Option<f64>| v.map_or("-".to_string(), |v| format!("{v:+7.2}"));
            println!(
                "{key:>4} {:>4} {:>9} {:>11.2}c {:>10} {:>10} {:>9.2} {:>10} {:>11}",
                summary.midi_velocity,
                summary.partials,
                summary.inharmonic_rms_cents,
                show(summary.glide_k1),
                show(summary.glide_median),
                summary.envelope_rms_db,
                show(summary.envelope_trend_db),
                show(summary.scatter_rms_db),
            );
        }
        println!();
    }
}

/// Every partial of a few individual recordings, so the shape of a residual can
/// be seen rather than summarised.
fn report_partials(
    library: &SampleLibrary,
    config: &SurveyConfig,
    residual: &ResidualConfig,
    wanted: &[(u8, u8)],
) {
    println!("\n=== 2b. partial by partial, one loud layer of four keys\n");
    for &(key, layer) in wanted {
        let Some(sample) = library.layers(key).get(usize::from(layer)) else {
            continue;
        };
        let Ok(analysis) = analyse(sample, config) else {
            continue;
        };
        println!(
            "key {key}, layer {layer} (vel {}), fitted f0 {:.3} Hz, B {:.3e}",
            sample.midi_velocity(),
            analysis.inharmonic.model.f0_hz,
            analysis.inharmonic.model.b,
        );
        println!(
            "   used by the fit: {:?}   rejected: {:?}",
            analysis.inharmonic.used,
            analysis
                .inharmonic
                .rejected
                .iter()
                .map(|&(k, cents)| format!("{k}:{cents:+.0}c"))
                .collect::<Vec<String>>(),
        );
        println!("    k   frequency Hz   level dB   model cents   glide cents   env rms   env trend");
        for r in partial_residuals(&analysis, residual).iter().take(24) {
            println!(
                "{:5} {:14.3} {:10.1} {:13.2} {:>13} {:9.2} {:>11}",
                r.k,
                r.frequency_hz,
                r.level_db,
                r.model_cents,
                r.glide_cents.map_or("-".to_string(), |c| format!("{c:+.2}")),
                r.envelope_residual_db,
                r.envelope_trend_db
                    .map_or("-".to_string(), |d| format!("{d:+.2}")),
            );
        }
        println!();
    }
}

/// Whether the excitation-spectrum scatter belongs to the note or to the
/// instrument.
///
/// The engine's per-mode input gains are one smooth curve times `sin(k pi x)`,
/// so every decibel the measured spectrum scatters around that is unreachable —
/// but the *fix* depends on where the scatter lives. If it is a function of
/// frequency, it is the bridge and the soundboard, one admittance curve shared
/// by the whole instrument, and it costs a table and no arithmetic per sample.
/// If it belongs to each note separately, nothing short of a per-note per-mode
/// gain table will do.
///
/// Measured by binning every partial of every note in thirds of an octave and
/// asking how much of the scatter survives averaging over the notes: the mean
/// of a bin is the part all the notes in it agree on, the spread within it is
/// the part they do not.
fn report_excitation_by_frequency(points: &[(u8, f64, f64)]) {
    println!("\n=== 1b. excitation-spectrum scatter, by frequency and by note\n");
    println!("  band Hz    notes  partials   mean dB   spread across notes dB");
    let bins = 24;
    let (low, high) = (60.0f64, 8000.0f64);
    for bin in 0..bins {
        let lo = low * (high / low).powf(bin as f64 / bins as f64);
        let hi = low * (high / low).powf((bin + 1) as f64 / bins as f64);
        let inside: Vec<&(u8, f64, f64)> = points
            .iter()
            .filter(|&&(_, f, db)| (lo..hi).contains(&f) && db.is_finite())
            .collect();
        if inside.len() < 20 {
            continue;
        }
        // One number per note first, so that a note with fifty partials in the
        // band does not outvote a note with two.
        let mut keys: Vec<u8> = inside.iter().map(|&&(key, _, _)| key).collect();
        keys.sort_unstable();
        keys.dedup();
        let per_note: Vec<f64> = keys
            .iter()
            .filter_map(|&key| {
                median(
                    inside
                        .iter()
                        .filter(|&&&(k, _, _)| k == key)
                        .map(|&&(_, _, db)| db),
                )
            })
            .collect();
        let mean = per_note.iter().sum::<f64>() / per_note.len() as f64;
        let spread = (per_note.iter().map(|db| (db - mean).powi(2)).sum::<f64>()
            / per_note.len() as f64)
            .sqrt();
        println!(
            "{lo:>6.0}-{hi:<6.0} {:>5} {:>9} {:>9.2} {:>24.2}",
            per_note.len(),
            inside.len(),
            mean,
            spread,
        );
    }
}

/// The fundamental's frequency through the first second, on a window short
/// enough to see the attack.
///
/// The survey's windows are between 85 ms and 680 ms long, and a tension
/// nonlinearity spends itself in the first few tens of milliseconds — so the
/// glide column of the tables above is measured through a low-pass that could
/// hide most of it. This pass re-tracks a few notes at four periods per window
/// and prints the pitch against time directly, next to the engine's render of
/// the same note, where the answer must be a flat line.
fn report_attack_glide(
    library: &SampleLibrary,
    preset_path: &str,
    keys: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== 2c. the fundamental's pitch through the attack, short windows\n");
    let config = SurveyConfig {
        cache_dir: None,
        window_periods: 4.0,
        min_window: 1 << 11,
        max_window: 1 << 13,
        hop_divisor: 32,
        ..SurveyConfig::default()
    };
    let engine = piano_emulator::preset::Preset::load(Path::new(preset_path))?;
    let times = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6];
    println!(
        "source  key  vel  window ms   f1 at 1.6 s   cents at {:?} s",
        &times[..times.len() - 1]
    );
    for &key in keys {
        let Ok(note_config) = config.note_config(equal_temperament(key)) else {
            continue;
        };
        let window_ms = 1000.0 * note_config.tracker.stft.window as f64 / f64::from(SAMPLE_RATE);
        let row = |source: &str, midi_velocity: u8, signal: &[f32]| {
            let Ok((trajectories, _)) = track_refined(
                signal,
                f64::from(SAMPLE_RATE),
                InharmonicModel::harmonic(equal_temperament(key)),
                &note_config,
            ) else {
                return;
            };
            let Some(track) = trajectories.track(1) else {
                return;
            };
            let last = times[times.len() - 1];
            let Some(reference) = track.frequency_at(trajectories.onset_s + last) else {
                return;
            };
            let slide: Vec<String> = times[..times.len() - 1]
                .iter()
                .map(|&t| {
                    track
                        .frequency_at(trajectories.onset_s + t)
                        .map_or("-".to_string(), |f| {
                            format!("{:+.2}", piano_tuner::cents(reference, f))
                        })
                })
                .collect();
            println!(
                "{source:>10} {key:>4} {midi_velocity:>4} {window_ms:>10.1} {reference:>13.3}   {}",
                slide.join("  ")
            );
        };
        for layer in [CENSUS_LAYERS[0], CENSUS_LAYERS[2]] {
            let Some(sample) = library.layers(key).get(usize::from(layer)) else {
                continue;
            };
            let Ok(recording) = audio::load_at(&sample.path, SAMPLE_RATE) else {
                continue;
            };
            row("salamander", sample.midi_velocity(), &recording.mono());
        }
        let (left, right) = render_note(&engine, key);
        let mono: Vec<f32> = left.iter().zip(&right).map(|(&l, &r)| 0.5 * (l + r)).collect();
        row("engine", CONTROL_VELOCITY, &mono);
    }
    Ok(())
}

// -------------------------------------------------------------- engine control

/// The same measurements on notes the engine rendered from the estimated
/// preset, where the model is true by construction.
fn report_control(
    preset: &Preset,
    preset_path: &str,
    config: &SurveyConfig,
    residual: &ResidualConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== 3. the engine's own renders through the same estimators (the control)\n");
    println!(
        " key  partials   inharm rms  worst (k)   glide k1  glide med   env rms  env trend  \
         excite rms"
    );
    let engine = piano_emulator::preset::Preset::load(Path::new(preset_path))?;
    for key in CENSUS_KEYS {
        let (left, right) = render_note(&engine, key);
        let mono: Vec<f32> = left.iter().zip(&right).map(|(&l, &r)| 0.5 * (l + r)).collect();
        let note_config = config.note_config(equal_temperament(key))?;
        let Ok((trajectories, _)) = track_refined(
            &mono,
            f64::from(SAMPLE_RATE),
            InharmonicModel::harmonic(equal_temperament(key)),
            &note_config,
        ) else {
            println!("{key:>4}  no tracking");
            continue;
        };
        let Ok(analysis) = analyze_trajectories(trajectories, &note_config) else {
            println!("{key:>4}  no analysis");
            continue;
        };
        let (summary, _) =
            summarise(key, CONTROL_VELOCITY, &analysis, residual, written_model(preset, key));
        let show = |v: Option<f64>| v.map_or("-".to_string(), |v| format!("{v:+7.2}"));
        println!(
            "{key:>4} {:>9} {:>12.2}c {:>7.1}c ({:>2}) {:>10} {:>10} {:>9.2} {:>10} {:>11}",
            summary.partials,
            summary.inharmonic_rms_cents,
            summary.inharmonic_worst_cents,
            summary.inharmonic_worst_k,
            show(summary.glide_k1),
            show(summary.glide_median),
            summary.envelope_rms_db,
            show(summary.envelope_trend_db),
            show(summary.scatter_rms_db),
        );
    }
    Ok(())
}

fn render_note(preset: &piano_emulator::preset::Preset, key: u8) -> (Vec<f32>, Vec<f32>) {
    let events = [RenderEvent::new(
        0.0,
        Event::NoteOn {
            key,
            vel: u16::from(CONTROL_VELOCITY),
        },
    )];
    render_to_buffer(preset, &events, CONTROL_SECONDS)
}

// ------------------------------------------------------------ spectrum census

/// What a sustained frame of a recording is made of, and what lies between the
/// partials at the strike and a second later.
fn census_pass(
    library: &SampleLibrary,
    preset: &Preset,
    config: &SurveyConfig,
    residual: &ResidualConfig,
    preset_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== 4. spectrum census: what radiates that is not a transverse partial\n");
    println!(
        "source  key  vel  peaks  transv  phantom  loudest  neighbour  loudest  unexpl  loudest  \
         between@0  between@1s   loudest unexplained, Hz"
    );
    let engine = piano_emulator::preset::Preset::load(Path::new(preset_path))?;
    for key in CENSUS_KEYS {
        for layer in CENSUS_LAYERS {
            let Some(sample) = library.layers(key).get(usize::from(layer)) else {
                continue;
            };
            let Ok(recording) = audio::load_at(&sample.path, SAMPLE_RATE) else {
                continue;
            };
            census_one(
                "salamander",
                key,
                sample.midi_velocity(),
                &recording.mono(),
                preset,
                config,
                residual,
            );
        }
        let (left, right) = render_note(&engine, key);
        let mono: Vec<f32> = left.iter().zip(&right).map(|(&l, &r)| 0.5 * (l + r)).collect();
        census_one("engine", key, CONTROL_VELOCITY, &mono, preset, config, residual);
    }
    Ok(())
}

fn census_one(
    source: &str,
    key: u8,
    midi_velocity: u8,
    signal: &[f32],
    preset: &Preset,
    config: &SurveyConfig,
    residual: &ResidualConfig,
) {
    let note_config = match config.note_config(equal_temperament(key)) {
        Ok(config) => config,
        Err(_) => return,
    };
    let Ok((trajectories, fit)) = track_refined(
        signal,
        f64::from(SAMPLE_RATE),
        InharmonicModel::harmonic(equal_temperament(key)),
        &note_config,
    ) else {
        return;
    };
    // The partials as measured, not as modelled: a census asks what is left
    // over once everything the note actually put in the spectrum is accounted
    // for, and the tracker's frequencies are that.
    let loudest = trajectories
        .tracks
        .iter()
        .filter_map(|t| t.peak())
        .map(|p| p.amplitude)
        .fold(0.0f64, f64::max);
    let partials: Vec<(u32, f64)> = trajectories
        .tracks
        .iter()
        .filter(|t| {
            t.peak()
                .is_some_and(|p| p.amplitude >= loudest * 10f64.powf(-residual.level_db / 20.0))
        })
        .filter_map(|t| t.weighted_frequency().map(|f| (t.k, f)))
        .collect();
    if partials.is_empty() {
        return;
    }
    // Every other key of the instrument, so a peak at a neighbour's pitch is
    // recognised as one. Only fundamentals: a sympathetically excited string
    // radiates mostly there, and its higher partials coincide with the struck
    // note's own by construction.
    let neighbours: Vec<(u8, f64)> = (21..=108)
        .filter(|&k| k != key)
        .filter_map(|k| preset.f0(k).map(|f| (k, f64::from(f))))
        .collect();

    let window = note_config.tracker.stft.window;
    let onset = (trajectories.onset_s * f64::from(SAMPLE_RATE)) as usize;
    let sustain = onset + (f64::from(SAMPLE_RATE) * 0.5) as usize;
    let Ok(spectrum) = frame_spectrum(signal, sustain, window, 2) else {
        return;
    };
    let fft_size = window * 2;
    let mut peaks = Vec::new();
    find_peaks(&spectrum, f64::from(SAMPLE_RATE), fft_size, -70.0, &mut peaks);
    // Only the band the note itself occupies. Below its fundamental a
    // recording holds room rumble and the microphone's own offset — peaks of
    // several hertz that are louder than a treble note's partials and have
    // nothing to do with the instrument.
    let band = (0.75 * fit.model.partial(1), 12_000.0);
    peaks.retain(|p| (band.0..=band.1).contains(&p.frequency_hz));
    // Levels are quoted against the loudest transverse partial in the same
    // frame, not against the fundamental: a bass note's fundamental is often
    // not its loudest partial, and a residual quoted against it would read as
    // louder than the note.
    let lobe = 4.0 * f64::from(SAMPLE_RATE) / window as f64;
    let reference = partial_levels(
        &spectrum,
        f64::from(SAMPLE_RATE),
        fft_size,
        &partials.iter().map(|&(_, f)| f).collect::<Vec<f64>>(),
        lobe,
    )
    .into_iter()
    .flatten()
    .fold(0.0f64, f64::max)
    .max(f64::MIN_POSITIVE);
    let census = classify_peaks(&peaks, &partials, &neighbours, reference, lobe, residual);
    let count = |wanted: fn(&PeakClass) -> bool| census.iter().filter(|p| wanted(&p.class)).count();
    let loudest_of = |wanted: fn(&PeakClass) -> bool| {
        census
            .iter()
            .filter(|p| wanted(&p.class))
            .map(|p| p.level_db)
            .fold(f64::NEG_INFINITY, f64::max)
    };

    // The broadband energy between the partials, at the strike and a second
    // later, on a window short enough to keep the two apart.
    let short = 4096;
    let guard = (4.0 * f64::from(SAMPLE_RATE) / short as f64).max(3.0);
    let frequencies: Vec<f64> = partials.iter().map(|&(_, f)| f).collect();
    let between = |start: usize| -> f64 {
        frame_spectrum(signal, start, short, 1).map_or(f64::NAN, |frame| {
            band_split(
                &frame,
                f64::from(SAMPLE_RATE),
                short,
                &frequencies,
                guard,
                band,
            )
            .between_db()
        })
    };

    // The frequencies of the loudest unexplained peaks, so that a reader can
    // see whether they repeat between velocity layers — a resonance of the
    // instrument does, a noise floor does not.
    let mut unexplained: Vec<&piano_tuner::residual::ClassifiedPeak> = census
        .iter()
        .filter(|p| p.class == PeakClass::Unexplained)
        .collect();
    unexplained.sort_by(|a, b| b.level_db.total_cmp(&a.level_db));
    let loudest_frequencies: Vec<String> = unexplained
        .iter()
        .take(4)
        .map(|p| format!("{:.0}", p.frequency_hz))
        .collect();

    println!(
        "{source:>10} {key:>4} {midi_velocity:>4} {:>6} {:>7} {:>8} {:>8.1} {:>10} {:>8.1} \
         {:>7} {:>8.1} {:>10.1} {:>11.1}   {}",
        census.len(),
        count(|c| matches!(c, PeakClass::Transverse { .. })),
        count(|c| matches!(c, PeakClass::Combination { .. })),
        loudest_of(|c| matches!(c, PeakClass::Combination { .. })),
        count(|c| matches!(c, PeakClass::Neighbour { .. })),
        loudest_of(|c| matches!(c, PeakClass::Neighbour { .. })),
        count(|c| matches!(c, PeakClass::Unexplained)),
        loudest_of(|c| matches!(c, PeakClass::Unexplained)),
        between(onset),
        between(onset + (f64::from(SAMPLE_RATE)) as usize),
        loudest_frequencies.join(" "),
    );
}

// ------------------------------------------------------ phantom partial growth

/// The decisive test for a quadratic nonlinearity, note by note and layer by
/// layer.
///
/// A transverse string that stretches as it vibrates drives the bridge at every
/// sum and difference of its partial frequencies, with an amplitude
/// proportional to the *product* of the two partials that made it. So the level
/// at `f_i + f_j` must rise one dB for every dB the product rises — two dB per
/// dB of the note's own level — while anything linear (a partial leaking in,
/// a neighbouring string, the noise floor) rises one for one with the note. The
/// sixteen velocity layers of a sampled piano measure that slope directly.
///
/// The probe has to be chosen with care. `f_i + f_j` stands flat of transverse
/// partial `i + j` by only `3 f0 B i j (i+j) / 2` hertz, which for low `i, j` is
/// less than the width of the analysis window's own main lobe: probing there
/// measures partial `i + j`, whatever is or is not beside it. The pairs are
/// therefore chosen per note as the ones whose sum stands at least two main
/// lobes clear of *every* measured partial, and the clearance is reported with
/// the answer.
fn phantom_pass(
    library: &SampleLibrary,
    config: &SurveyConfig,
    preset_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== 4b. phantom partials: how their level grows with the note's\n");
    println!(
        "source  key  pair    clear of partials   layers   slope vs product   slope vs note   \
         margin over floor   level re loudest"
    );
    let engine = piano_emulator::preset::Preset::load(Path::new(preset_path))?;
    for key in [21u8, 33, 45, 57, 60, 72] {
        let Ok(note_config) = config.note_config(equal_temperament(key)) else {
            continue;
        };
        let window = note_config.tracker.stft.window;
        let lobe = 4.0 * f64::from(SAMPLE_RATE) / window as f64;
        let layers = library.layers(key);
        // The probes are chosen once per key, on the loudest layer, and then
        // measured on every layer: a regression needs the same quantity at each
        // velocity.
        let Some(loudest) = layers.last() else { continue };
        let Ok(reference) = trajectories_for(loudest, &note_config, config) else {
            continue;
        };
        let probes = choose_probes(&reference, lobe);
        if probes.is_empty() {
            println!("{key:>4}  no combination stands clear of the partials");
            continue;
        }

        let salamander: Vec<PhantomPoint> = layers
            .iter()
            .filter_map(|sample| {
                let recording = audio::load_at(&sample.path, SAMPLE_RATE).ok()?;
                let trajectories = trajectories_for(sample, &note_config, config).ok()?;
                phantom_points(&recording.mono(), &trajectories, &probes, window)
            })
            .flatten()
            .collect();
        let control: Vec<PhantomPoint> = layers
            .iter()
            .filter_map(|sample| {
                let events = [RenderEvent::new(
                    0.0,
                    Event::NoteOn {
                        key,
                        vel: u16::from(sample.midi_velocity()),
                    },
                )];
                let (left, right) = render_to_buffer(&engine, &events, CONTROL_SECONDS);
                let mono: Vec<f32> =
                    left.iter().zip(&right).map(|(&l, &r)| 0.5 * (l + r)).collect();
                let (trajectories, _) = track_refined(
                    &mono,
                    f64::from(SAMPLE_RATE),
                    InharmonicModel::harmonic(equal_temperament(key)),
                    &note_config,
                )
                .ok()?;
                phantom_points(&mono, &trajectories, &probes, window)
            })
            .flatten()
            .collect();

        for (source, points) in [("salamander", &salamander), ("engine", &control)] {
            for &(pair, clearance) in &probes {
                let group: Vec<&PhantomPoint> =
                    points.iter().filter(|p| p.pair == pair).collect();
                if group.len() < 4 {
                    continue;
                }
                let slope = |x: fn(&PhantomPoint) -> f64| {
                    least_squares_slope(
                        &group.iter().map(|p| x(p)).collect::<Vec<f64>>(),
                        &group.iter().map(|p| p.phantom_db).collect::<Vec<f64>>(),
                    )
                };
                println!(
                    "{source:>10} {key:>4}  f{}+f{}   {:>14.1} Hz   {:>6}   {:>16}   {:>13}   \
                     {:>17.1}   {:>16.1}",
                    pair.0,
                    pair.1,
                    clearance,
                    group.len(),
                    slope(|p| p.drive_db).map_or("-".into(), |s| format!("{s:+.2}")),
                    slope(|p| p.note_db).map_or("-".into(), |s| format!("{s:+.2}")),
                    median(group.iter().map(|p| p.phantom_db - p.background_db))
                        .unwrap_or(f64::NAN),
                    median(group.iter().map(|p| p.phantom_db - p.note_db))
                        .unwrap_or(f64::NAN),
                );
            }
        }
    }
    Ok(())
}

/// The pairs whose sum frequency is far enough from every measured partial to
/// be probed, with how far, worst pair first.
fn choose_probes(
    trajectories: &piano_tuner::NoteTrajectories,
    lobe_hz: f64,
) -> Vec<((u32, u32), f64)> {
    let loudest = trajectories
        .tracks
        .iter()
        .filter_map(|t| t.peak())
        .map(|p| p.amplitude)
        .fold(0.0f64, f64::max);
    let audible: Vec<(u32, f64)> = trajectories
        .tracks
        .iter()
        .filter(|t| t.peak().is_some_and(|p| p.amplitude >= loudest * 1e-2))
        .filter_map(|t| t.weighted_frequency().map(|f| (t.k, f)))
        .collect();
    let all: Vec<f64> = trajectories
        .tracks
        .iter()
        .filter_map(|t| t.weighted_frequency())
        .collect();
    let mut probes: Vec<((u32, u32), f64)> = Vec::new();
    for (a, &(i, fi)) in audible.iter().enumerate() {
        for &(j, fj) in &audible[a..] {
            if i < 3 || j > 9 {
                continue;
            }
            let sum = fi + fj;
            let clearance = all
                .iter()
                .map(|&f| (f - sum).abs())
                .fold(f64::INFINITY, f64::min);
            if clearance >= 2.0 * lobe_hz {
                probes.push(((i, j), clearance));
            }
        }
    }
    probes.sort_by(|a, b| b.1.total_cmp(&a.1));
    probes.truncate(4);
    probes
}

/// One recording's measurement at each probe.
#[derive(Clone, Copy, Debug)]
struct PhantomPoint {
    pair: (u32, u32),
    /// Level at `f_i + f_j`, dB (the recording's own scale, which is common to
    /// its velocity layers).
    phantom_db: f64,
    /// Level just beside it, where nothing is predicted: the local floor.
    background_db: f64,
    /// Level of the product `a_i a_j` that a quadratic mechanism would drive
    /// it with, dB.
    drive_db: f64,
    /// Level of the note itself — its loudest partial — dB.
    note_db: f64,
}

fn phantom_points(
    signal: &[f32],
    trajectories: &piano_tuner::NoteTrajectories,
    probes: &[((u32, u32), f64)],
    window: usize,
) -> Option<Vec<PhantomPoint>> {
    let start = ((trajectories.onset_s + 0.3) * f64::from(SAMPLE_RATE)) as usize;
    let spectrum = frame_spectrum(signal, start, window, 2).ok()?;
    let sr = f64::from(SAMPLE_RATE);
    let fft_size = window * 2;
    let guard = 2.0 * sr / window as f64;
    let frequency = |k: u32| trajectories.track(k).and_then(|t| t.weighted_frequency());
    let level = |f: f64| {
        partial_levels(&spectrum, sr, fft_size, &[f], guard)[0].map(|a| 20.0 * a.log10())
    };
    let f1 = frequency(1)?;
    let note_db = (1..=8)
        .filter_map(frequency)
        .filter_map(level)
        .fold(f64::NEG_INFINITY, f64::max);
    let mut out = Vec::new();
    for &((i, j), _) in probes {
        let (Some(fi), Some(fj)) = (frequency(i), frequency(j)) else {
            continue;
        };
        let Some(phantom_db) = level(fi + fj) else {
            continue;
        };
        // Two probes a third of a partial spacing to either side, where nothing
        // is predicted at all: the quieter of them is the floor this peak has
        // to stand above to be a peak.
        let (Some(below), Some(above)) = (level(fi + fj - 0.35 * f1), level(fi + fj + 0.35 * f1))
        else {
            continue;
        };
        let (Some(li), Some(lj)) = (level(fi), level(fj)) else {
            continue;
        };
        out.push(PhantomPoint {
            pair: (i, j),
            phantom_db,
            background_db: below.min(above),
            drive_db: li + lj,
            note_db,
        });
    }
    Some(out)
}

fn least_squares_slope(x: &[f64], y: &[f64]) -> Option<f64> {
    let n = x.len() as f64;
    if x.len() < 3 || x.len() != y.len() {
        return None;
    }
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;
    let (num, den) = x.iter().zip(y).fold((0.0, 0.0), |(num, den), (&x, &y)| {
        (num + (x - mean_x) * (y - mean_y), den + (x - mean_x).powi(2))
    });
    (den > 0.0).then(|| num / den)
}

// --------------------------------------------------------------------- stereo

/// How the partials of one note divide between the two channels, measured on
/// the recordings and on the engine's renders.
fn stereo_pass(
    library: &SampleLibrary,
    config: &SurveyConfig,
    preset_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== 5. stereo balance per partial (left minus right, dB)\n");
    println!(
        "source  key  vel  partials   median@0.3s  spread@0.3s   median@2s  spread@2s  \
         drift 0.3->2s"
    );
    let engine = piano_emulator::preset::Preset::load(Path::new(preset_path))?;
    for key in CENSUS_KEYS {
        if let Some(sample) = library.layers(key).get(usize::from(CENSUS_LAYERS[2])) {
            if let Ok(recording) = audio::load_at(&sample.path, SAMPLE_RATE) {
                if recording.channel_count() >= 2 {
                    stereo_one(
                        "salamander",
                        key,
                        sample.midi_velocity(),
                        &recording.channels[0],
                        &recording.channels[1],
                        config,
                    );
                }
            }
        }
        let (left, right) = render_note(&engine, key);
        stereo_one("engine", key, CONTROL_VELOCITY, &left, &right, config);
    }
    Ok(())
}

fn stereo_one(
    source: &str,
    key: u8,
    midi_velocity: u8,
    left: &[f32],
    right: &[f32],
    config: &SurveyConfig,
) {
    let Ok(note_config) = config.note_config(equal_temperament(key)) else {
        return;
    };
    let mono: Vec<f32> = left.iter().zip(right).map(|(&l, &r)| 0.5 * (l + r)).collect();
    let Ok((trajectories, _)) = track_refined(
        &mono,
        f64::from(SAMPLE_RATE),
        InharmonicModel::harmonic(equal_temperament(key)),
        &note_config,
    ) else {
        return;
    };
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
    let guard = 4.0 * f64::from(SAMPLE_RATE) / window as f64;
    let at = |seconds: f64| -> Option<piano_tuner::residual::StereoBalance> {
        let start = ((trajectories.onset_s + seconds) * f64::from(SAMPLE_RATE)) as usize;
        let l = frame_spectrum(left, start, window, 1).ok()?;
        let r = frame_spectrum(right, start, window, 1).ok()?;
        let sr = f64::from(SAMPLE_RATE);
        stereo_balance(
            &partial_levels(&l, sr, window, &frequencies, guard),
            &partial_levels(&r, sr, window, &frequencies, guard),
        )
    };
    let (early, late) = (at(0.3), at(2.0));

    // What moved, partial by partial. A pan is one number for the whole note
    // and the engine has it; a balance that drifts as the note decays means the
    // two channels are hearing different decay rates, which one panned mono
    // voice cannot do however the pan is set.
    let deltas = |seconds: f64| -> Option<Vec<Option<f64>>> {
        let start = ((trajectories.onset_s + seconds) * f64::from(SAMPLE_RATE)) as usize;
        let l = frame_spectrum(left, start, window, 1).ok()?;
        let r = frame_spectrum(right, start, window, 1).ok()?;
        let sr = f64::from(SAMPLE_RATE);
        let (l, r) = (
            partial_levels(&l, sr, window, &frequencies, guard),
            partial_levels(&r, sr, window, &frequencies, guard),
        );
        Some(
            l.into_iter()
                .zip(r)
                .map(|(l, r)| Some(20.0 * (l? / r?).log10()).filter(|d| d.is_finite()))
                .collect(),
        )
    };
    let drift = deltas(0.3).zip(deltas(2.0)).and_then(|(early, late)| {
        median(
            early
                .into_iter()
                .zip(late)
                .filter_map(|(a, b)| Some((b? - a?).abs())),
        )
    });

    let show = |b: Option<piano_tuner::residual::StereoBalance>, get: fn(&piano_tuner::residual::StereoBalance) -> f64| {
        b.map_or("-".to_string(), |b| format!("{:+.2}", get(&b)))
    };
    println!(
        "{source:>10} {key:>4} {midi_velocity:>4} {:>9} {:>13} {:>12} {:>11} {:>10} {:>10}",
        early.map_or(0, |b| b.partials),
        show(early, |b| b.median_db),
        show(early, |b| b.spread_db),
        show(late, |b| b.median_db),
        show(late, |b| b.spread_db),
        drift.map_or("-".to_string(), |d| format!("{d:.2}")),
    );
}

// ------------------------------------------------------------------ mechanism

/// The recordings of the action rather than of the strings: what the engine
/// does not model at all, measured against a struck note of the same key.
fn mechanism_pass(
    library: &SampleLibrary,
    root: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== 6. the mechanism: key-off, pedal and string-resonance recordings\n");
    println!(
        "recording        key   plays at   peak re strike   rms re strike   decay to -40 dB   \
         centroid Hz   length s"
    );
    let samples = root.join("samples");
    // The layer a mezzo-forte strike would trigger, which is what these noises
    // are heard against in playing.
    let reference = |key: u8| -> Option<f64> {
        let sample = library
            .layers(key)
            .iter()
            .find(|s| (s.lovel..=s.hivel).contains(&CONTROL_VELOCITY))?;
        let recording = audio::load_at(&sample.path, SAMPLE_RATE).ok()?;
        transient_metrics(&recording.mono(), f64::from(SAMPLE_RATE)).map(|m| m.peak)
    };
    let reference_rms = |key: u8| -> Option<f64> {
        let sample = library
            .layers(key)
            .iter()
            .find(|s| (s.lovel..=s.hivel).contains(&CONTROL_VELOCITY))?;
        let recording = audio::load_at(&sample.path, SAMPLE_RATE).ok()?;
        let mono = recording.mono();
        // Over the first second, which is the note's prompt sound, so that a
        // long tail does not make the noise look louder than it is.
        let n = (SAMPLE_RATE as usize).min(mono.len());
        transient_metrics(&mono[..n], f64::from(SAMPLE_RATE)).map(|m| m.rms)
    };

    // Every level is quoted as the instrument plays it, which is not how the
    // file is stored: the SFZ attenuates each of these groups on the way out
    // (`SalamanderGrandPiano-V3+20200602.sfz` lines 514, 548, 593, 686 and
    // 691), and the key-off group is attenuated by 37 dB. Comparing the raw
    // files would say a damper landing is as loud as the note.
    let row = |name: &str, path: PathBuf, key: Option<u8>, plays_at_db: f64| {
        let Ok(recording) = audio::load_at(&path, SAMPLE_RATE) else {
            return;
        };
        let Some(metrics) = transient_metrics(&recording.mono(), f64::from(SAMPLE_RATE)) else {
            return;
        };
        let against = key.unwrap_or(60);
        let peak_db = reference(against).map(|p| plays_at_db + 20.0 * (metrics.peak / p).log10());
        let rms_db = reference_rms(against).map(|r| plays_at_db + 20.0 * (metrics.rms / r).log10());
        let show = |v: Option<f64>| v.map_or("-".to_string(), |v| format!("{v:+.1} dB"));
        println!(
            "{name:>14} {:>5} {:>10.0} {:>15} {:>15} {:>17} {:>13.0} {:>10.2}",
            key.map_or("-".to_string(), |k| k.to_string()),
            plays_at_db,
            show(peak_db),
            show(rms_db),
            if metrics.decay_s.is_finite() {
                format!("{:.3} s", metrics.decay_s)
            } else {
                "never".to_string()
            },
            metrics.centroid_hz,
            metrics.duration_s,
        );
    };

    for key in CENSUS_KEYS {
        row(
            &format!("rel{}", key - 20),
            samples.join(format!("rel{}.flac", key - 20)),
            Some(key),
            -37.0,
        );
    }
    for (name, gain) in [
        ("pedalD1", -20.0),
        ("pedalD2", -20.0),
        ("pedalU1", -19.0),
        ("pedalU2", -19.0),
    ] {
        row(name, samples.join(format!("{name}.flac")), None, gain);
    }
    for (name, gain) in [
        ("harmLC3", -4.0),
        ("harmSC3", 0.0),
        ("harmLC5", -4.0),
        ("harmSC5", 0.0),
    ] {
        row(name, samples.join(format!("{name}.flac")), Some(48), gain);
    }
    Ok(())
}

/// The partial layout the preset writes for one key: what the engine will
/// synthesize when that key is struck.
fn written_model(preset: &Preset, key: u8) -> Option<InharmonicModel> {
    let index = piano_tuner::preset::key_index(key)?;
    Some(InharmonicModel::with_b4(
        f64::from(preset.f0(key)?),
        f64::from(preset.notes.inharmonicity_b[index]),
        f64::from(preset.notes.inharmonicity_b4[index]),
    ))
}

// ---------------------------------------------------------------- small tools

fn grouped(summaries: &[LayerSummary]) -> BTreeMap<u8, Vec<&LayerSummary>> {
    let mut out: BTreeMap<u8, Vec<&LayerSummary>> = BTreeMap::new();
    for summary in summaries {
        out.entry(summary.key).or_default().push(summary);
    }
    out
}

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

fn rms(values: &[f64]) -> f64 {
    let finite: Vec<f64> = values.iter().copied().filter(|x| x.is_finite()).collect();
    if finite.is_empty() {
        return f64::NAN;
    }
    (finite.iter().map(|x| x * x).sum::<f64>() / finite.len() as f64).sqrt()
}
