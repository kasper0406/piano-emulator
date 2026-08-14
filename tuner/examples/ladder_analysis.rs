//! Objective measurements over every rung of the timbre ladder, per key.
//!
//! `timbre_ladder.rs` writes ten level-matched renderings of the same note, from
//! the Salamander recording (`00`) to the engine (`07`), each rung differing
//! from its neighbours by one hypothesis about what the engine is missing. This
//! example measures them. Nothing here re-renders anything: it reads the WAVs
//! that were written, plus the cached trajectories, so the numbers describe the
//! files that were listened to.
//!
//! ```text
//! cargo run --release --example ladder_analysis -- \
//!     [renders/timbre-ladder] [data/salamander] [data/cache/salamander]
//! ```
//!
//! # The five measurements
//!
//! 1. **Spectral-envelope roughness** (`roughness`). Per-partial level at
//!    `t = 0.3 s` against a smooth envelope through the same points —
//!    `TUNING_REPORT.md` §3's statistic, now computed on each *rung* rather than
//!    on the trajectories. Two smooth references are fitted, because the answer
//!    depends on how much bending "smooth" is allowed: a penalised spline with
//!    knots one octave apart (flexible: only a genuinely non-smooth comb
//!    survives it) and a degree-2 polynomial in `ln k` (stiff: what §3 used, so
//!    its column is comparable with the report's).
//!
//! 2. **Linewidth** (`width`). A 2 s segment from 1.5 s to 3.5 s, Hann-windowed
//!    and zero-padded to `2^19`, and the −6 dB width of partials 1–6 in cents.
//!    A single decaying sinusoid has a width set by its own decay and by the
//!    window; a unison of three strings at three pitches, or a mode whose
//!    frequency wanders, has more. The synthetic control row (`floor`) is a pure
//!    exponentially-decaying sinusoid at the note's own fundamental and decay
//!    rate, measured with the same code, so the excess over it is the signal.
//!
//! 3. **Envelope liveliness** (`mod`). The amplitude envelope of each of
//!    partials 1–6, extracted by complex projection, detrended (degree-3
//!    polynomial in `ln a`), and transformed: the *modulation spectrum* over
//!    0.1–20 Hz. Reported as the RMS of the detrended log envelope in that band
//!    (how much the partial moves), its spectral flatness (0 dB = a continuum,
//!    strongly negative = discrete lines), its centroid and its spread. A pure
//!    exponential has no modulation at all, a beating unison has one or two
//!    lines, and a real recording is claimed to have a continuum.
//!
//! 4. **Attack** (`attack`). The first 30 ms with the tracked partials removed
//!    by a phase-locked resynthesis: its level against the same 30 ms of the
//!    rung, and its spectral flatness.
//!
//! 5. **Distance to the source**, per rung and per measurement, and what each
//!    ingredient (one rung minus its control) moved.
//!
//! Every metric is computed by the same code on every rung including `00`, so a
//! metric's own bias cancels in the differences, which is the only thing any of
//! these numbers are used for.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use piano_emulator::preset::Preset as EnginePreset;
use piano_emulator::string::contact_taper;
use piano_tuner::numeric::{median, poly_eval, solve_in_place, weighted_polyfit};
use piano_tuner::pipeline::analyze_trajectories;
use piano_tuner::preset::equal_temperament;
use piano_tuner::survey::{trajectories_for, SurveyConfig};
use piano_tuner::{audio, Sample, SampleLibrary, SAMPLE_RATE};
use rustfft::{num_complex::Complex64, FftPlanner};

const SR: f64 = SAMPLE_RATE as f64;

/// Frames of pre-roll `timbre_ladder` puts in front of the strike. Frame
/// `PREROLL` is the strike in every rung of every key.
const PREROLL: usize = 512;

/// The keys the ladder was rendered for, in the order it renders them.
const KEYS: [u8; 3] = [60, 45, 84];
const VELOCITY: u8 = 90;

/// The ten rungs, in ladder order. `00` is the reference every distance is
/// measured against.
const RUNGS: [(&str, &str, &str); 10] = [
    ("00", "00_source", "source"),
    ("01", "01_resynth_full", "resynth (measured a_k, f_k)"),
    ("02", "02_meas_amp_law_decay", "fitted decay law"),
    ("03", "03_smooth_amp_meas_decay", "smooth excitation"),
    ("04", "04_engine_rough", "engine + roughness"),
    ("05", "05_engine_linewidth", "engine + linewidth"),
    ("06", "06_engine_attack", "engine + attack residual"),
    ("07", "07_engine", "engine"),
    ("08", "08_resynth_plus_attack", "resynth + attack residual"),
    ("09", "09_engine_modal_control", "engine string only (control)"),
];

/// Index of the source and of the shipped engine inside [`RUNGS`].
const SOURCE: usize = 0;
const RESYNTH: usize = 1;
const ENGINE: usize = 7;

// -------------------------------------------------------- measurement geometry

/// Instant the spectral-envelope roughness is measured at, in seconds after the
/// strike, and the window it is measured over. Past the attack and inside every
/// rung's modelled span; 8192 frames is 171 ms, which resolves neighbouring
/// partials at A2 (110 Hz apart) with margin.
const ROUGHNESS_T_S: f64 = 0.3;
const ROUGHNESS_WINDOW: usize = 8192;
const ROUGHNESS_FFT: usize = 65_536;

/// Curvature penalty of the octave spline, in dB per octave squared per point.
/// It is what stops a spline with a knot per octave from following the very
/// roughness it is supposed to be a reference for.
const SPLINE_LAMBDA: f64 = 1.0;

/// The long-window segment: 1.5 s to 3.5 s after the strike, transformed at
/// `2^19` so the bin spacing (0.092 Hz) is far below any width being measured.
const WIDTH_LO_S: f64 = 1.5;
const WIDTH_HI_S: f64 = 3.5;
const WIDTH_FFT: usize = 1 << 19;

/// Partials the linewidth and the modulation spectrum are reported over. Low
/// enough that every key has them (C6 has eight tracked partials in all) and
/// low enough that the ear resolves them individually.
const MAX_REPORTED_PARTIAL: u32 = 6;

/// The span the amplitude envelopes are taken over, and the hop between
/// envelope samples: 200 Hz of envelope sample rate for a 20 Hz band.
const ENV_LO_S: f64 = 0.15;
const ENV_HI_S: f64 = 3.60;
const ENV_HOP_S: f64 = 0.005;
const ENV_FFT: usize = 4096;

/// The modulation band. The low edge is one cycle in the 3.5 s of envelope
/// there is; the high edge is where a piano partial's amplitude stops moving
/// and starts being a second partial.
const MOD_LO_HZ: f64 = 0.1;
const MOD_HI_HZ: f64 = 20.0;

/// Where the band is split. Below this a partial's envelope is *beating* — a
/// unison's two or three pitches, and the two polarizations, all of which the
/// engine's model has and can be fitted. Above it nothing in a two-exponential
/// two-beat envelope can put any energy at all, so the upper band is the part
/// of the liveliness that no per-partial envelope model represents.
const MOD_SPLIT_HZ: f64 = 5.0;

/// Degree of the polynomial in `t` removed from `ln a(t)` before the modulation
/// transform. Three takes out the two-exponential shape a decaying partial has
/// without touching anything periodic.
const DETREND_DEGREE: usize = 3;

/// The attack window, and the hop of the phase-locked analysis that clears the
/// partials out of it.
const ATTACK_S: f64 = 0.030;
/// The span `timbre_ladder` itself keeps its attack residual over, measured
/// here as a cross-check that this subtraction is the ladder's.
const ATTACK_CHECK_S: f64 = 0.150;
const ATTACK_HOP_S: f64 = 0.005;
const ATTACK_FFT: usize = 4096;
const ATTACK_FLAT_LO_HZ: f64 = 50.0;
const ATTACK_FLAT_HI_HZ: f64 = 16_000.0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let renders = PathBuf::from(args.next().unwrap_or_else(|| "renders/timbre-ladder".into()));
    let root = PathBuf::from(args.next().unwrap_or_else(|| "data/salamander".into()));
    let cache = PathBuf::from(args.next().unwrap_or_else(|| "data/cache/salamander".into()));
    let preset_path = args
        .next()
        .unwrap_or_else(|| "presets/salamander-c5.toml".into());

    let preset = EnginePreset::load(Path::new(&preset_path))?;
    let library = SampleLibrary::from_sfz(root.join("SalamanderGrandPiano-V3+20200602.sfz"))?;
    let config = SurveyConfig {
        cache_dir: Some(cache),
        ..SurveyConfig::default()
    };

    let validation = validate();
    for line in &validation {
        println!("validation: {line}");
    }

    let mut keys = Vec::new();
    for key in KEYS {
        let report = measure_key(key, &library, &config, &renders, &preset)?;
        println!("{} measured", report.name);
        keys.push(report);
    }

    let markdown = write_markdown(&keys, &validation);
    let path = renders.join("ANALYSIS.md");
    std::fs::write(&path, markdown)?;
    println!("wrote {}", path.display());
    Ok(())
}

// ------------------------------------------------------------------ one key

/// Everything measured on one key's ten files.
struct KeyReport {
    name: String,
    /// The reference partials: `(k, f_k)` from the cached trajectories, the same
    /// set `timbre_ladder` rendered from.
    partials: Vec<(u32, f64)>,
    /// The strike position the preset tabulates for this key, and the partials
    /// inside the tracked range where its `sin(k pi x)` comb has a null, with
    /// the depth of each null in dB below the comb's own maximum.
    strike_position: f64,
    comb_nulls: Vec<(u32, f64)>,
    /// The synthetic linewidth floor, per reported partial: a pure sinusoid at
    /// that partial's frequency decaying at that partial's own fitted rate,
    /// measured by the same code. The excess over it is the only part of a
    /// measured width that is about the instrument.
    width_floor_cents: Vec<f64>,
    rungs: Vec<RungReport>,
}

/// Everything measured on one file.
struct RungReport {
    label: &'static str,
    caption: &'static str,
    /// §3's statistic against the octave spline and against the stiff
    /// polynomial, in dB RMS, plus the worst partial against the spline.
    roughness_spline_db: f64,
    roughness_poly_db: f64,
    roughness_worst_db: f64,
    /// Which partial that worst deviation is, and which way it points. A single
    /// partial 35 dB under a smooth envelope is a hole in the spectrum, and a
    /// hole is a different fault from a generally rough comb.
    roughness_worst_k: u32,
    /// Median absolute deviation from the spline: the same statistic with one
    /// dead partial unable to carry it.
    roughness_median_db: f64,
    /// −6 dB width of each reported partial, in cents, and their median.
    widths_cents: Vec<f64>,
    width_median_cents: f64,
    /// The same widths minus each partial's own analysis floor: what is left
    /// once the window and the partial's own decay are accounted for.
    width_excess_cents: f64,
    /// Per-partial modulation statistics and their medians.
    modulation: Vec<Modulation>,
    mod_energy_db: f64,
    mod_low_db: f64,
    mod_high_db: f64,
    mod_flatness_db: f64,
    mod_centroid_hz: f64,
    mod_spread_hz: f64,
    mod_lines: f64,
    /// The first 30 ms with the tracked partials removed: level against the
    /// prompt sound, spectral flatness, and the ladder's own 150 ms statistic.
    attack_level_db: f64,
    attack_flatness_db: f64,
    attack_check_db: f64,
}

/// One partial's modulation spectrum, reduced to four numbers.
#[derive(Clone, Copy)]
struct Modulation {
    k: u32,
    /// RMS of the detrended log envelope inside the band, in dB.
    energy_db: f64,
    /// The same, split at [`MOD_SPLIT_HZ`]: beating below, everything a
    /// per-partial envelope model cannot represent above.
    low_db: f64,
    high_db: f64,
    /// Spectral flatness of the band, in dB: 0 is white, −20 is a line.
    flatness_db: f64,
    centroid_hz: f64,
    spread_hz: f64,
    /// How many modulation lines it takes to account for half the band energy.
    lines: f64,
}

fn measure_key(
    key: u8,
    library: &SampleLibrary,
    config: &SurveyConfig,
    renders: &Path,
    preset: &EnginePreset,
) -> Result<KeyReport, Box<dyn std::error::Error>> {
    let name = note_name(key);
    let dir = renders.join(&name);

    // The reference partial set, from the same cache the ladder rendered from.
    let sample = layer_for(library, key, VELOCITY)?;
    let note_config = config.note_config(equal_temperament(key))?;
    let trajectories = trajectories_for(sample, &note_config, config)?;
    let analysis = analyze_trajectories(trajectories, &note_config)?;
    let mut partials: Vec<(u32, f64)> = analysis
        .decays
        .partials
        .iter()
        .filter(|fit| fit.frequency_hz.is_finite() && fit.frequency_hz > 0.0)
        .map(|fit| (fit.k, fit.frequency_hz))
        .collect();
    partials.sort_by_key(|&(k, _)| k);
    if partials.is_empty() {
        return Err(format!("{name}: no tracked partial").into());
    }
    let f0 = partials[0].1 / f64::from(partials[0].0);

    // The synthetic linewidth floor, one control per reported partial: a single
    // sinusoid at that partial's frequency, decaying at that partial's own
    // fitted rate, measured with the same window and the same code. A partial
    // wider than this is wider for a reason that is in the instrument.
    let width_floor_cents: Vec<f64> = partials
        .iter()
        .filter(|&&(k, _)| k <= MAX_REPORTED_PARTIAL)
        .map(|&(k, f)| {
            let sigma = analysis
                .decays
                .partials
                .iter()
                .find(|fit| fit.k == k)
                .map_or(1.0, |fit| fit.fast.sigma.max(1e-6));
            let control = magnitude_spectrum(&synthetic_partial(f, sigma), WIDTH_FFT);
            partial_width_cents(&control, f, f0).unwrap_or(f64::NAN)
        })
        .collect();

    let mut rungs = Vec::new();
    for (label, file, caption) in RUNGS {
        let path = dir.join(format!("{file}.wav"));
        let clip = audio::load_wav(&path)?;
        let mono: Vec<f64> = clip.mono().iter().map(|&x| f64::from(x)).collect();
        rungs.push(measure_rung(
            label,
            caption,
            &mono,
            &partials,
            f0,
            &width_floor_cents,
        ));
    }

    // What the engine's own excitation comb does over the same partials. The
    // roughness measurement finds the engine's worst partial; this says whether
    // that partial is where `sin(k pi x)` has a zero, which is a hole the
    // recording has no counterpart for and a different fault from roughness.
    let params = preset.string_params(key);
    let position = f64::from(params.strike_position);
    let comb = |k: u32| -> f64 {
        let value = (f64::from(k) * std::f64::consts::PI * position).sin().abs()
            * f64::from(contact_taper(k as usize, params.contact_width));
        20.0 * value.max(1e-9).log10()
    };
    let top = partials.last().map_or(1, |&(k, _)| k);
    let peak = (1..=top).map(comb).fold(f64::NEG_INFINITY, f64::max);
    let comb_nulls: Vec<(u32, f64)> = (1..=top)
        .map(|k| (k, comb(k) - peak))
        .filter(|&(_, depth)| depth < -20.0)
        .collect();

    Ok(KeyReport {
        name,
        partials,
        strike_position: position,
        comb_nulls,
        width_floor_cents,
        rungs,
    })
}

fn measure_rung(
    label: &'static str,
    caption: &'static str,
    mono: &[f64],
    partials: &[(u32, f64)],
    f0: f64,
    width_floor: &[f64],
) -> RungReport {
    let (spline, poly, worst, robust, worst_k) = roughness(mono, partials, f0);

    let reported: Vec<(u32, f64)> = partials
        .iter()
        .copied()
        .filter(|&(k, _)| k <= MAX_REPORTED_PARTIAL)
        .collect();

    let lo = (PREROLL + (WIDTH_LO_S * SR) as usize).min(mono.len());
    let hi = (PREROLL + (WIDTH_HI_S * SR) as usize).min(mono.len());
    let width_spectrum = magnitude_spectrum(&mono[lo..hi], WIDTH_FFT);
    let widths: Vec<f64> = reported
        .iter()
        .map(|&(_, f)| partial_width_cents(&width_spectrum, f, f0).unwrap_or(f64::NAN))
        .collect();
    let width_median_cents = median(&finite(&widths)).unwrap_or(f64::NAN);
    let excess: Vec<f64> = widths
        .iter()
        .zip(width_floor)
        .map(|(&w, &floor)| w - floor)
        .collect();
    let width_excess_cents = median(&finite(&excess)).unwrap_or(f64::NAN);

    let modulation: Vec<Modulation> = reported
        .iter()
        .filter_map(|&(k, f)| modulation_of(mono, k, f, f0))
        .collect();

    let (attack_level_db, attack_flatness_db, attack_check_db) = attack(mono, partials, f0);

    RungReport {
        label,
        caption,
        roughness_spline_db: spline,
        roughness_poly_db: poly,
        roughness_worst_db: worst,
        roughness_worst_k: worst_k,
        roughness_median_db: robust,
        widths_cents: widths,
        width_median_cents,
        width_excess_cents,
        mod_energy_db: median(&modulation.iter().map(|m| m.energy_db).collect::<Vec<_>>())
            .unwrap_or(f64::NAN),
        mod_low_db: median(&modulation.iter().map(|m| m.low_db).collect::<Vec<_>>())
            .unwrap_or(f64::NAN),
        mod_high_db: median(&modulation.iter().map(|m| m.high_db).collect::<Vec<_>>())
            .unwrap_or(f64::NAN),
        mod_flatness_db: median(&modulation.iter().map(|m| m.flatness_db).collect::<Vec<_>>())
            .unwrap_or(f64::NAN),
        mod_centroid_hz: median(&modulation.iter().map(|m| m.centroid_hz).collect::<Vec<_>>())
            .unwrap_or(f64::NAN),
        mod_spread_hz: median(&modulation.iter().map(|m| m.spread_hz).collect::<Vec<_>>())
            .unwrap_or(f64::NAN),
        mod_lines: median(&modulation.iter().map(|m| m.lines).collect::<Vec<_>>())
            .unwrap_or(f64::NAN),
        modulation,
        attack_level_db,
        attack_flatness_db,
        attack_check_db,
    }
}

// -------------------------------------------------- 1. spectral-envelope roughness

/// Per-partial deviation from a smooth spectral envelope at `ROUGHNESS_T_S`.
///
/// Returns `(spline RMS, polynomial RMS, worst partial, median absolute
/// deviation)`, all in dB against the spline except the polynomial column. The
/// two references bracket what "smooth" is allowed to mean: the spline may bend
/// once per octave (penalised), the polynomial may not bend at all beyond a
/// parabola in `ln k`. The median is carried because one partial sitting in a
/// beat null at the measured instant can move an RMS over twenty partials by
/// several decibels, and a single dead partial is a different fault from a
/// generally rough comb.
fn roughness(mono: &[f64], partials: &[(u32, f64)], f0: f64) -> (f64, f64, f64, f64, u32) {
    let centre = PREROLL + (ROUGHNESS_T_S * SR) as usize;
    let start = centre.saturating_sub(ROUGHNESS_WINDOW / 2).min(mono.len());
    let end = (start + ROUGHNESS_WINDOW).min(mono.len());
    let spectrum = magnitude_spectrum(&mono[start..end], ROUGHNESS_FFT);
    let bin = SR / ROUGHNESS_FFT as f64;

    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut lnk = Vec::new();
    let mut orders = Vec::new();
    for &(k, f) in partials {
        let Some((_, level_db)) = peak_near(&spectrum, bin, f, guard_hz(f, f0)) else {
            continue;
        };
        if !level_db.is_finite() {
            continue;
        }
        x.push(f.log2());
        lnk.push(f64::from(k).ln());
        orders.push(k);
        y.push(level_db);
    }
    if y.len() < 4 {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN, 0);
    }

    let spline_fit = octave_spline(&x, &y);
    let spline_residual: Vec<f64> = y
        .iter()
        .zip(&spline_fit)
        .map(|(&value, &fitted)| value - fitted)
        .collect();

    let weights = vec![1.0; y.len()];
    let degree = 2.min(y.len() - 1);
    let poly_residual: Vec<f64> = match weighted_polyfit(&lnk, &y, &weights, degree) {
        Some(c) => lnk
            .iter()
            .zip(&y)
            .map(|(&u, &value)| value - poly_eval(&c, u))
            .collect(),
        None => vec![f64::NAN; y.len()],
    };

    let (worst_index, worst) = spline_residual.iter().map(|r| r.abs()).enumerate().fold(
        (0usize, 0.0f64),
        |(bi, best), (i, value)| {
            if value > best {
                (i, value)
            } else {
                (bi, best)
            }
        },
    );
    let absolute: Vec<f64> = spline_residual.iter().map(|r| r.abs()).collect();
    let robust = median(&finite(&absolute)).unwrap_or(f64::NAN);
    (
        rms(&spline_residual),
        rms(&poly_residual),
        worst,
        robust,
        orders[worst_index],
    )
}

/// Half-width of the band a partial's peak is looked for in: three per cent of
/// its own frequency, but never more than half the spacing to its neighbours,
/// so a strong neighbour can never be picked up as this partial.
fn guard_hz(f: f64, f0: f64) -> f64 {
    (0.03 * f).min(0.45 * f0)
}

/// A penalised least-squares spline in `log2 f` with a knot every octave.
///
/// Linear B-splines (hat functions) on an octave grid, with a second-difference
/// penalty on the knot values. Without the penalty a knot per octave is enough
/// freedom to follow a comb at the bottom of the compass, where two partials can
/// share an octave; with it the reference stays a spectral *envelope*.
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
    // Second-difference penalty: lambda * sum_j (c_{j-1} - 2 c_j + c_{j+1})^2.
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
    // A tiny ridge so a knot no partial reaches is still determined.
    for i in 0..n {
        normal[i * n + i] += 1e-9;
    }
    let coefficients = solve_in_place(&mut normal, &mut rhs, n).unwrap_or_else(|| vec![0.0; n]);
    x.iter()
        .map(|&u| (0..n).map(|j| coefficients[j] * hat(j, u)).sum())
        .collect()
}

// ---------------------------------------------------------------- 2. linewidth

/// −6 dB width of the peak nearest `f`, in cents, on a Hann-windowed transform
/// of the linewidth segment zero-padded to [`WIDTH_FFT`].
///
/// `None` when the peak does not fall 6 dB before the search band runs out,
/// which is a partial too close to its neighbour to measure rather than a wide
/// one.
fn partial_width_cents(spectrum: &[f64], f: f64, f0: f64) -> Option<f64> {
    let bin = SR / WIDTH_FFT as f64;
    let guard = guard_hz(f, f0);
    let (peak_bin, peak_db) = peak_bin_near(spectrum, bin, f, guard)?;
    let target = peak_db - 6.0;
    let db = |i: usize| -> f64 { amp_db(spectrum[i]) };

    let limit = (guard / bin) as usize;
    // Walk down from the peak in each direction until the level crosses the
    // target, then interpolate the crossing linearly in dB.
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

/// A pure exponentially-decaying sinusoid over the linewidth segment: the
/// analysis floor of measurement 2, in the same units, for one partial that has
/// no linewidth of its own beyond its decay.
fn synthetic_partial(f: f64, sigma: f64) -> Vec<f64> {
    let frames = ((WIDTH_HI_S - WIDTH_LO_S) * SR) as usize;
    (0..frames)
        .map(|n| {
            let t = WIDTH_LO_S + n as f64 / SR;
            (-sigma * t).exp() * (std::f64::consts::TAU * f * t).sin()
        })
        .collect()
}

// ------------------------------------------------------ 3. envelope liveliness

/// The modulation spectrum of one partial's amplitude envelope.
fn modulation_of(mono: &[f64], k: u32, f: f64, f0: f64) -> Option<Modulation> {
    let window_n = envelope_window(f0);
    let hop_n = (ENV_HOP_S * SR).round() as usize;
    let lo = PREROLL + (ENV_LO_S * SR) as usize;
    let hi = (PREROLL + (ENV_HI_S * SR) as usize).min(mono.len());
    if hi <= lo + window_n {
        return None;
    }
    let hops = (hi - lo) / hop_n;
    if hops < 64 {
        return None;
    }

    // Complex projection onto e^{-i 2 pi f t}, one value per hop.
    let envelope: Vec<f64> = (0..hops)
        .map(|h| {
            let centre = lo + h * hop_n;
            let (re, im) = project(mono, f, centre, window_n);
            (re * re + im * im).sqrt()
        })
        .collect();

    let peak = envelope.iter().cloned().fold(0.0f64, f64::max);
    if !(peak > 0.0) {
        return None;
    }
    // A partial that has decayed into the noise carries no envelope; below
    // −80 dB of its own peak the log is measuring the analysis floor.
    let floor = peak * 1e-4;
    if envelope.iter().filter(|&&a| a < floor).count() > hops / 10 {
        return None;
    }
    let ln_a: Vec<f64> = envelope.iter().map(|&a| a.max(floor).ln()).collect();

    // Detrend: take out the smooth decay, keep everything that wiggles.
    let t: Vec<f64> = (0..hops).map(|h| h as f64 * ENV_HOP_S).collect();
    let weights = vec![1.0; hops];
    let coefficients = weighted_polyfit(&t, &ln_a, &weights, DETREND_DEGREE)?;
    let db = 20.0 / std::f64::consts::LN_10;
    let residual: Vec<f64> = t
        .iter()
        .zip(&ln_a)
        .map(|(&ti, &value)| db * (value - poly_eval(&coefficients, ti)))
        .collect();

    // Modulation spectrum of the residual, corrected for the low-pass the
    // analysis window imposes on the envelope.
    let env_rate = SR / hop_n as f64;
    let spectrum = hann_power_spectrum(&residual, ENV_FFT);
    let bin = env_rate / ENV_FFT as f64;
    let window_s = window_n as f64 / SR;
    let mut freqs = Vec::new();
    let mut power = Vec::new();
    for (i, &p) in spectrum.iter().enumerate() {
        let nu = i as f64 * bin;
        if nu < MOD_LO_HZ || nu > MOD_HI_HZ {
            continue;
        }
        // The analysis window low-passes the envelope; undo it, but only where
        // the correction is small enough to be a correction.
        let response = hann_response(nu * window_s);
        if response < 0.25 {
            continue;
        }
        freqs.push(nu);
        power.push(p / (response * response));
    }
    if power.len() < 8 {
        return None;
    }

    let total: f64 = power.iter().sum();
    if !(total > 0.0) {
        return None;
    }
    // `hann_power_spectrum` is normalised so the sum over all bins is the mean
    // square of the input, so the band's sum is the band-limited variance and
    // its square root is the RMS — of a signal already in dB.
    let energy_db = total.sqrt();
    let low: f64 = freqs
        .iter()
        .zip(&power)
        .filter(|(&f, _)| f < MOD_SPLIT_HZ)
        .map(|(_, &p)| p)
        .sum();
    let high: f64 = freqs
        .iter()
        .zip(&power)
        .filter(|(&f, _)| f >= MOD_SPLIT_HZ)
        .map(|(_, &p)| p)
        .sum();

    let centroid: f64 = freqs.iter().zip(&power).map(|(f, p)| f * p).sum::<f64>() / total;
    let spread = (freqs
        .iter()
        .zip(&power)
        .map(|(f, p)| (f - centroid).powi(2) * p)
        .sum::<f64>()
        / total)
        .sqrt();
    let geometric = power.iter().map(|p| p.max(1e-30).ln()).sum::<f64>() / power.len() as f64;
    let arithmetic = total / power.len() as f64;
    let flatness_db = 10.0 * (geometric.exp() / arithmetic).log10();
    let lines = lines_to_half(&power, total, bin);

    Some(Modulation {
        k,
        energy_db,
        low_db: low.sqrt(),
        high_db: high.sqrt(),
        flatness_db,
        centroid_hz: centroid,
        spread_hz: spread,
        lines,
    })
}

/// How many separate modulation lines it takes to account for half the band's
/// energy.
///
/// This is the direct form of the question the hypothesis asks. A partial with
/// one beat needs one line; a three-string unison with two polarizations has
/// fifteen pairs and needs several; a genuine continuum needs enough that the
/// count is really a bandwidth. Peaks are taken greedily, largest first, each
/// claiming ±0.25 Hz so a single Hann-broadened line cannot be counted twice.
fn lines_to_half(power: &[f64], total: f64, bin: f64) -> f64 {
    let claim = (0.25 / bin).ceil() as usize;
    let mut peaks: Vec<(usize, f64)> = (1..power.len().saturating_sub(1))
        .filter(|&i| power[i] > power[i - 1] && power[i] >= power[i + 1])
        .map(|i| (i, power[i]))
        .collect();
    peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut taken = vec![false; power.len()];
    let mut accumulated = 0.0;
    let mut count = 0.0;
    for (index, _) in peaks {
        if taken[index] {
            continue;
        }
        let lo = index.saturating_sub(claim);
        let hi = (index + claim).min(power.len() - 1);
        for i in lo..=hi {
            if !taken[i] {
                accumulated += power[i];
                taken[i] = true;
            }
        }
        count += 1.0;
        if accumulated >= 0.5 * total {
            return count;
        }
    }
    count.max(1.0)
}

/// The window the envelope projection uses: four periods of the fundamental,
/// which is the shortest window that separates neighbouring partials, held
/// between 15 and 40 ms so the envelope's own bandwidth covers the 20 Hz band.
fn envelope_window(f0: f64) -> usize {
    let seconds = (4.0 / f0).clamp(0.015, 0.040);
    let n = (seconds * SR).round() as usize;
    n + n % 2
}

/// Magnitude response of a normalised Hann window at `x = nu T`, which is the
/// factor by which an amplitude modulation at `nu` is attenuated by an analysis
/// window `T` long.
fn hann_response(x: f64) -> f64 {
    if x.abs() < 1e-9 {
        return 1.0;
    }
    let pi_x = std::f64::consts::PI * x;
    if (x.abs() - 1.0).abs() < 1e-6 {
        return 0.5;
    }
    (pi_x.sin() / (pi_x * (1.0 - x * x))).abs()
}

/// Hann-windowed projection of `signal` onto `e^{-i 2 pi f t}` over `window_n`
/// samples centred at `centre`, scaled so a sinusoid of amplitude `A` returns
/// `|c| = A`.
fn project(signal: &[f64], f: f64, centre: usize, window_n: usize) -> (f64, f64) {
    let half = window_n / 2;
    let start = centre as isize - half as isize;
    let omega = std::f64::consts::TAU * f / SR;
    let (rot_re, rot_im) = ((-omega).cos(), (-omega).sin());
    let phase = -omega * start as f64;
    let (mut re, mut im) = (phase.cos(), phase.sin());
    let (mut acc_re, mut acc_im) = (0.0, 0.0);
    let mut weight = 0.0;
    for i in 0..window_n {
        let w = 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / window_n as f64).cos();
        let index = start + i as isize;
        // A window that hangs off the front of the signal is renormalised by
        // the weight it actually used, so an estimate near the strike is the
        // amplitude of the part of the note that exists rather than half of it.
        if index >= 0 {
            if let Some(&x) = signal.get(index as usize) {
                acc_re += w * x * re;
                acc_im += w * x * im;
                weight += w;
            }
        }
        let next_re = re * rot_re - im * rot_im;
        im = re * rot_im + im * rot_re;
        re = next_re;
    }
    if weight <= 0.0 {
        return (0.0, 0.0);
    }
    (2.0 * acc_re / weight, 2.0 * acc_im / weight)
}

// ------------------------------------------------------------------ 4. attack

/// Level and spectral flatness of the first [`ATTACK_S`] with the tracked
/// partials removed.
///
/// The removal is a phase-locked resynthesis: the same projection the ladder
/// used to build its attack residual, so what is left here is what rungs `06`
/// and `08` were given. The level is relative to the same 30 ms of the rung
/// itself, so it is a *proportion* of the attack rather than an absolute, and
/// the rungs are level-matched anyway.
fn attack(mono: &[f64], partials: &[(u32, f64)], f0: f64) -> (f64, f64, f64) {
    let window_n = envelope_window(f0).max((0.020 * SR) as usize);
    let hop_n = (ATTACK_HOP_S * SR).round() as usize;
    let frames = (ATTACK_CHECK_S * SR) as usize;
    let hops = frames / hop_n + 3;

    let mut modelled = vec![0.0f64; frames];
    for &(_, f) in partials {
        let coefficients: Vec<(f64, f64)> = (0..hops)
            .map(|h| project(mono, f, PREROLL + h * hop_n, window_n))
            .collect();
        let omega = std::f64::consts::TAU * f / SR;
        let (rot_re, rot_im) = (omega.cos(), omega.sin());
        // `project` references its phase to sample 0 of the file, and the
        // resynthesis starts at the strike, which is [`PREROLL`] samples later:
        // without this the model is a random phase against the signal and the
        // subtraction *adds* 3 dB instead of cancelling 20.
        let origin = omega * PREROLL as f64;
        let (mut re, mut im) = (origin.cos(), origin.sin());
        for (n, slot) in modelled.iter_mut().enumerate() {
            let u = n as f64 / hop_n as f64;
            let i = (u.floor() as usize).min(hops - 1);
            let j = (i + 1).min(hops - 1);
            let frac = u - i as f64;
            let (c_re, c_im) = coefficients[i];
            let (d_re, d_im) = coefficients[j];
            let a_re = c_re * (1.0 - frac) + d_re * frac;
            let a_im = c_im * (1.0 - frac) + d_im * frac;
            *slot += a_re * re - a_im * im;
            let next_re = re * rot_re - im * rot_im;
            im = re * rot_im + im * rot_re;
            re = next_re;
        }
    }

    let residual: Vec<f64> = (0..frames)
        .map(|n| mono.get(PREROLL + n).copied().unwrap_or(0.0) - modelled[n])
        .collect();
    let whole: Vec<f64> = (0..frames)
        .map(|n| mono.get(PREROLL + n).copied().unwrap_or(0.0))
        .collect();

    // The cross-check against `timbre_ladder`'s own reported residual level,
    // which is measured over the same 150 ms and against the same 150 ms of
    // signal. If this column does not reproduce the ladder's numbers on `00`,
    // this implementation of the subtraction is not the ladder's.
    let check_db = 20.0 * (rms_energy(&residual) / rms_energy(&whole).max(1e-30)).log10();

    // The reported level is the first 30 ms against the *prompt sound* — the
    // 0.2–2 s window every rung was level-matched over — so it is an absolute
    // number, comparable between rungs, and does not divide by an attack whose
    // own size is part of what is being compared.
    let short = (ATTACK_S * SR) as usize;
    let lo = PREROLL + (0.2 * SR) as usize;
    let hi = (PREROLL + (2.0 * SR) as usize).min(mono.len());
    let reference = rms_energy(&mono[lo.min(mono.len())..hi]);
    let level = 20.0 * (rms_energy(&residual[..short]) / reference.max(1e-30)).log10();

    let spectrum = magnitude_spectrum(&residual[..short], ATTACK_FFT);
    let bin = SR / ATTACK_FFT as f64;
    let lo = (ATTACK_FLAT_LO_HZ / bin).ceil() as usize;
    let hi = ((ATTACK_FLAT_HI_HZ / bin).floor() as usize).min(spectrum.len() - 1);
    let band: Vec<f64> = spectrum[lo..=hi].iter().map(|&m| m * m + 1e-30).collect();
    let geometric = band.iter().map(|p| p.ln()).sum::<f64>() / band.len() as f64;
    let arithmetic = band.iter().sum::<f64>() / band.len() as f64;
    let flatness = 10.0 * (geometric.exp() / arithmetic).log10();
    (level, flatness, check_db)
}

// --------------------------------------------------------------- shared DSP

/// Hann-windowed magnitude spectrum, zero-padded to `fft_size`, scaled so a
/// sinusoid of amplitude `A` reads `A` at its peak.
fn magnitude_spectrum(samples: &[f64], fft_size: usize) -> Vec<f64> {
    let n = samples.len().min(fft_size);
    let mut buffer = vec![Complex64::new(0.0, 0.0); fft_size];
    let mut weight = 0.0;
    for i in 0..n {
        let w = 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / n as f64).cos();
        weight += w;
        buffer[i] = Complex64::new(samples[i] * w, 0.0);
    }
    FftPlanner::new().plan_fft_forward(fft_size).process(&mut buffer);
    let scale = 2.0 / weight.max(1e-30);
    buffer[..fft_size / 2]
        .iter()
        .map(|c| c.norm() * scale)
        .collect()
}

/// Hann-windowed one-sided power spectrum, zero-padded to `fft_size` and
/// normalised by Parseval so that the sum over the returned bins is the mean
/// square of the input.
///
/// Zero-padding spreads one sinusoid over many bins, so summing an
/// amplitude-calibrated spectrum over a band over-counts by up to 2 dB; this is
/// the normalisation a band-limited RMS needs.
fn hann_power_spectrum(samples: &[f64], fft_size: usize) -> Vec<f64> {
    let n = samples.len().min(fft_size);
    let mut buffer = vec![Complex64::new(0.0, 0.0); fft_size];
    let mut sum_sq = 0.0;
    for i in 0..n {
        let w = 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / n as f64).cos();
        sum_sq += w * w;
        buffer[i] = Complex64::new(samples[i] * w, 0.0);
    }
    FftPlanner::new()
        .plan_fft_forward(fft_size)
        .process(&mut buffer);
    let scale = 2.0 / (fft_size as f64 * sum_sq.max(1e-30));
    buffer[..fft_size / 2]
        .iter()
        .map(|c| c.norm_sqr() * scale)
        .collect()
}

/// The largest bin within `guard` hertz of `f`, refined by a parabola through
/// its two neighbours in dB. Returns `(frequency, level in dB)`.
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
    let level = b - 0.25 * (a - c) * offset;
    Some(((index as f64 + offset) * bin, level))
}

/// The largest bin within `guard` hertz of `f`, unrefined.
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

fn amp_db(a: f64) -> f64 {
    20.0 * a.max(1e-30).log10()
}

fn rms(values: &[f64]) -> f64 {
    let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return f64::NAN;
    }
    (finite.iter().map(|v| v * v).sum::<f64>() / finite.len() as f64).sqrt()
}

fn rms_energy(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    (values.iter().map(|v| v * v).sum::<f64>() / values.len() as f64).sqrt()
}

fn finite(values: &[f64]) -> Vec<f64> {
    values.iter().copied().filter(|v| v.is_finite()).collect()
}

fn layer_for<'a>(
    library: &'a SampleLibrary,
    key: u8,
    velocity: u8,
) -> Result<&'a Sample, Box<dyn std::error::Error>> {
    library
        .layers(key)
        .iter()
        .find(|s| (s.lovel..=s.hivel).contains(&velocity))
        .ok_or_else(|| format!("key {key} has no layer covering velocity {velocity}").into())
}

fn note_name(key: u8) -> String {
    const NAMES: [&str; 12] = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    format!(
        "{}{}",
        NAMES[usize::from(key) % 12],
        i32::from(key) / 12 - 1
    )
}

// ------------------------------------------------------------- self-validation

/// Measures signals whose answers are known, so the numbers in the tables are
/// read against something. Every line is a claim about the code above.
fn validate() -> Vec<String> {
    let mut out = Vec::new();
    let f = 261.6256;
    let frames = PREROLL + 4 * SAMPLE_RATE as usize;

    // A pure exponential: no modulation at all.
    let plain: Vec<f64> = (0..frames)
        .map(|n| {
            let t = (n as f64 - PREROLL as f64) / SR;
            if t < 0.0 {
                0.0
            } else {
                (-0.7 * t).exp() * (std::f64::consts::TAU * f * t).sin()
            }
        })
        .collect();
    if let Some(m) = modulation_of(&plain, 1, f, f) {
        out.push(format!(
            "pure exponential: modulation {:.3} dB RMS, flatness {:.1} dB (expected ~0 dB of modulation)",
            m.energy_db, m.flatness_db
        ));
    }

    // One beat: 1 dB peak-to-peak amplitude modulation at 3 Hz. A sinusoidal
    // modulation of the log envelope with peak `p` has RMS `p / sqrt 2`.
    let beating: Vec<f64> = (0..frames)
        .map(|n| {
            let t = (n as f64 - PREROLL as f64) / SR;
            if t < 0.0 {
                0.0
            } else {
                let am = 10f64.powf(0.5 * (std::f64::consts::TAU * 3.0 * t).sin() / 20.0);
                (-0.7 * t).exp() * am * (std::f64::consts::TAU * f * t).sin()
            }
        })
        .collect();
    if let Some(m) = modulation_of(&beating, 1, f, f) {
        out.push(format!(
            "1 dB peak AM at 3 Hz: modulation {:.3} dB RMS (expected 0.354; {:.3} below 5 Hz, \
             {:.3} above), centroid {:.2} Hz (expected 3.00), spread {:.2} Hz, flatness {:.1} dB, \
             {:.0} lines to half (expected 1)",
            m.energy_db, m.low_db, m.high_db, m.centroid_hz, m.spread_hz, m.flatness_db, m.lines
        ));
    }

    // Band-limited noise on the log envelope: a continuum, same total power.
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = || -> f64 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    };
    let mut walk = 0.0f64;
    let noisy: Vec<f64> = (0..frames)
        .map(|n| {
            let t = (n as f64 - PREROLL as f64) / SR;
            walk = 0.9995 * walk + 0.03 * next();
            if t < 0.0 {
                0.0
            } else {
                let am = 10f64.powf(walk / 20.0);
                (-0.7 * t).exp() * am * (std::f64::consts::TAU * f * t).sin()
            }
        })
        .collect();
    if let Some(m) = modulation_of(&noisy, 1, f, f) {
        out.push(format!(
            "random-walk envelope: modulation {:.3} dB RMS ({:.3} below 5 Hz, {:.3} above), \
             flatness {:.1} dB, spread {:.2} Hz, {:.0} lines to half (a continuum must be flatter, \
             wider and need more lines than the beat above)",
            m.energy_db, m.low_db, m.high_db, m.flatness_db, m.spread_hz, m.lines
        ));
    }

    // A partial standing on a broadband pedestal: the halo `TUNING_REPORT.md`
    // §4 measures around a recorded partial and the engine does not have. It is
    // the mechanism that would give a *continuum* of envelope modulation
    // without any beat at all, so its signature is the one to know.
    for pedestal_db in [-40.0, -30.0, -20.0] {
        // Two independent quadratures of noise band-limited to about 8 Hz, each
        // of unit variance, scaled so the pedestal's power stands `pedestal_db`
        // below the carrier's. Only the in-phase half modulates the amplitude,
        // which is why the answer is 3 dB below the naive one.
        let gain = 10f64.powf(pedestal_db / 20.0) / std::f64::consts::SQRT_2;
        let (mut n1, mut n2) = (0.0f64, 0.0f64);
        let pedestal: Vec<f64> = (0..frames)
            .map(|n| {
                let t = (n as f64 - PREROLL as f64) / SR;
                n1 = 0.999 * n1 + 0.077_44 * next();
                n2 = 0.999 * n2 + 0.077_44 * next();
                if t < 0.0 {
                    0.0
                } else {
                    let theta = std::f64::consts::TAU * f * t;
                    (-0.7 * t).exp()
                        * (theta.sin() + gain * (n1 * theta.sin() + n2 * theta.cos()))
                }
            })
            .collect();
        if let Some(m) = modulation_of(&pedestal, 1, f, f) {
            out.push(format!(
                "partial on a {pedestal_db:.0} dB noise pedestal: modulation {:.3} dB RMS \
                 ({:.3} below 5 Hz, {:.3} above), flatness {:.1} dB, {:.0} lines to half",
                m.energy_db, m.low_db, m.high_db, m.flatness_db, m.lines
            ));
        }
    }

    // Linewidth: a pure decaying sinusoid, and the same sinusoid split into a
    // pair 0.5 Hz apart (a mistuned unison).
    let single = synthetic_partial(f, 0.7);
    let pair: Vec<f64> = (0..single.len())
        .map(|n| {
            let t = WIDTH_LO_S + n as f64 / SR;
            (-0.7 * t).exp()
                * ((std::f64::consts::TAU * f * t).sin()
                    + (std::f64::consts::TAU * (f + 0.5) * t).sin())
        })
        .collect();
    let a = partial_width_cents(&magnitude_spectrum(&single, WIDTH_FFT), f, f).unwrap_or(f64::NAN);
    let b = partial_width_cents(&magnitude_spectrum(&pair, WIDTH_FFT), f, f).unwrap_or(f64::NAN);
    out.push(format!(
        "linewidth floor: one decaying sinusoid {a:.2} cents, two 0.5 Hz apart {b:.2} cents \
         (0.5 Hz at {f:.0} Hz is 3.3 cents of separation)"
    ));

    out
}

// ------------------------------------------------------------------ reporting

/// Distance of one rung's metric from the source's, and the share of the
/// engine's own distance that closes.
struct Distance {
    delta: f64,
}

impl Distance {
    fn of(rung: f64, source: f64) -> Self {
        Distance {
            delta: rung - source,
        }
    }
}

fn write_markdown(keys: &[KeyReport], validation: &[String]) -> String {
    let mut md = String::new();
    let _ = writeln!(md, "# ANALYSIS.md — the timbre ladder, measured\n");
    let _ = writeln!(
        md,
        "Objective measurements over every rung of `renders/timbre-ladder/` and the recording it \
         starts from, written by `cargo run --release --example ladder_analysis`. Nothing here \
         re-renders anything: it reads the WAVs that were listened to, plus the cached \
         trajectories for the reference partial set. Every metric runs the same code on every \
         rung including `00`, so a metric's own bias cancels in the differences — which is the \
         only thing any of these numbers is used for.\n"
    );

    executive_summary(&mut md, keys);

    let _ = writeln!(md, "## 0. What is measured, and how\n");
    let _ = writeln!(
        md,
        "| # | metric | window | what it is |\n|---:|:--|:--|:--|\n\
         | 1 | spectral-envelope roughness | 171 ms at 0.3 s | per-partial level against a smooth \
         envelope through the same points, dB RMS |\n\
         | 2 | linewidth | 2 s from 1.5 s, `2^19` transform | −6 dB width of partials 1–6 in cents, and the excess over a synthetic control that decays at the same rate |\n\
         | 3 | envelope liveliness | 0.15–3.6 s | modulation spectrum of each partial's amplitude \
         envelope over 0.1–20 Hz |\n\
         | 4 | attack | first 30 ms | level and spectral flatness of what is left when the tracked \
         partials are subtracted |\n"
    );
    let _ = writeln!(
        md,
        "Two smooth references are fitted for metric 1 because the answer depends on how much \
         bending \"smooth\" is allowed. `spline` is a least-squares spline in `log2 f` with a knot \
         every octave and a second-difference penalty; `poly` is a degree-2 polynomial in `ln k`, \
         which is what `TUNING_REPORT.md` §3 used, so its column is comparable with the report's. \
         The spline is the more demanding reference and returns the smaller number.\n"
    );
    let _ = writeln!(
        md,
        "Metric 3's `mod` is the RMS of the detrended log envelope inside 0.1–20 Hz, in dB: how \
         far the partial's level moves around its own smooth decay. It is split at 5 Hz because \
         nothing in a two-exponential, two-beat envelope can put energy above that, so the upper \
         half is the part of the liveliness no per-partial envelope model represents. `flat` is \
         the spectral flatness of the band — 0 dB is a continuum, −15 dB and below is one or two \
         discrete lines — and `lines` is the number of separate modulation lines it takes to \
         account for half the band's energy, which asks the same question without a logarithm in \
         it. `spread` is the power-weighted standard deviation of the modulation frequency.\n"
    );

    let _ = writeln!(md, "### Validation on signals with known answers\n");
    for line in validation {
        let _ = writeln!(md, "- {line}");
    }
    let _ = writeln!(md);

    for key in keys {
        key_section(&mut md, key);
    }

    comb_section(&mut md, keys);
    distance_section(&mut md, keys);
    ingredient_section(&mut md, keys);
    diagnostic_section(&mut md, keys);
    recommendation(&mut md, keys);
    md
}

fn key_section(md: &mut String, key: &KeyReport) {
    let _ = writeln!(md, "## {} — the ten rungs\n", key.name);
    let f0 = key.partials[0].1 / f64::from(key.partials[0].0);
    let floors: String = key
        .width_floor_cents
        .iter()
        .map(|c| format!("{c:.2}"))
        .collect::<Vec<_>>()
        .join(" / ");
    let _ = writeln!(
        md,
        "{} tracked partials, fundamental {:.2} Hz. Linewidth floor per partial (one sinusoid at \
         that partial's frequency decaying at its own fitted rate, same window, same code): \
         **{floors} cents** for k = 1…{}.\n",
        key.partials.len(),
        f0,
        key.width_floor_cents.len(),
    );
    let _ = writeln!(
        md,
        "| rung | | rough RMS | rough med | rough poly | worst (k) | width | excess | mod | <5 Hz | >5 Hz | flat | lines | centroid | spread | attack | attack flat | 150 ms |\n\
         |:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"
    );
    let _ = writeln!(
        md,
        "| | | dB | dB | dB | dB | cents | cents | dB | dB | dB | dB | n | Hz | Hz | dB | dB | dB |"
    );
    for rung in &key.rungs {
        let _ = writeln!(
            md,
            "| `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            rung.label,
            rung.caption,
            f2(rung.roughness_spline_db),
            f2(rung.roughness_median_db),
            f2(rung.roughness_poly_db),
            format!("{} ({})", f2(rung.roughness_worst_db), rung.roughness_worst_k),
            f2(rung.width_median_cents),
            f2(rung.width_excess_cents),
            f3(rung.mod_energy_db),
            f3(rung.mod_low_db),
            f3(rung.mod_high_db),
            f1(rung.mod_flatness_db),
            f1(rung.mod_lines),
            f2(rung.mod_centroid_hz),
            f2(rung.mod_spread_hz),
            f1(rung.attack_level_db),
            f1(rung.attack_flatness_db),
            f1(rung.attack_check_db),
        );
    }
    let _ = writeln!(md);

    // Per-partial linewidth, because the median hides which partials are wide.
    let reported: Vec<u32> = key
        .partials
        .iter()
        .map(|&(k, _)| k)
        .filter(|&k| k <= MAX_REPORTED_PARTIAL)
        .collect();
    let _ = writeln!(md, "**Linewidth per partial, cents.**\n");
    let header: String = reported.iter().map(|k| format!(" k={k} |")).collect();
    let rule: String = reported.iter().map(|_| " --: |".to_string()).collect();
    let _ = writeln!(md, "| rung |{header}\n|:--|{rule}");
    for rung in &key.rungs {
        let cells: String = rung
            .widths_cents
            .iter()
            .map(|&w| format!(" {} |", f2(w)))
            .collect();
        let _ = writeln!(md, "| `{}` |{cells}", rung.label);
    }
    let _ = writeln!(md);

    let _ = writeln!(
        md,
        "**Envelope modulation per partial, dB RMS in 0.1–20 Hz (the part above 5 Hz in \
         brackets).** Nothing in a two-exponential, two-beat envelope can put energy above 5 Hz, \
         so the bracketed number is the part of the liveliness no per-partial envelope model \
         represents.\n"
    );
    let _ = writeln!(md, "| rung |{header}\n|:--|{rule}");
    for rung in &key.rungs {
        let cells: String = reported
            .iter()
            .map(|&k| match rung.modulation.iter().find(|m| m.k == k) {
                Some(m) => format!(" {} ({}) |", f2(m.energy_db), f2(m.high_db)),
                None => " — |".to_string(),
            })
            .collect();
        let _ = writeln!(md, "| `{}` |{cells}", rung.label);
    }
    let _ = writeln!(md);
}

/// Where the engine's worst partial is, and whether it is a hole its own
/// excitation comb puts there.
fn comb_section(md: &mut String, keys: &[KeyReport]) {
    let _ = writeln!(md, "## 4a. The worst partial, and where it comes from\n");
    let _ = writeln!(
        md,
        "The roughness measurement's `worst (k)` column is the single partial furthest from the \
         smooth envelope at 0.3 s, and it is worth separating from the RMS beside it: a comb that \
         is generally rough and a comb with a *hole* in it are different faults with different \
         fixes. The engine's excitation is `g_k ∝ sin(k π x)` at the strike position the preset \
         tabulates, times the contact taper — and `sin` has exact zeros. This table puts the \
         measured worst partial of `00` and of `07` beside the nulls that formula predicts for the \
         same key.\n"
    );
    let _ = writeln!(
        md,
        "| key | strike position | comb nulls below −20 dB (k: depth) | `00` worst | `07` worst |\n\
         |:--|--:|:--|--:|--:|"
    );
    for key in keys {
        let nulls = if key.comb_nulls.is_empty() {
            "none".to_string()
        } else {
            key.comb_nulls
                .iter()
                .map(|&(k, depth)| format!("{k}: {depth:.0} dB"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let source = &key.rungs[SOURCE];
        let engine = &key.rungs[ENGINE];
        let _ = writeln!(
            md,
            "| {} | {:.5} | {nulls} | {} dB (k={}) | {} dB (k={}) |",
            key.name,
            key.strike_position,
            f1(source.roughness_worst_db),
            source.roughness_worst_k,
            f1(engine.roughness_worst_db),
            engine.roughness_worst_k,
        );
    }
    let _ = writeln!(md);
}

/// Every metric on every rung as a distance from the source, and as a share of
/// the engine's own distance.
fn distance_section(md: &mut String, keys: &[KeyReport]) {
    let _ = writeln!(md, "## 5. Distance to the source\n");
    let _ = writeln!(
        md,
        "Signed distance of each rung from `00` in each metric, and — for the metrics where the \
         engine has a gap at all — the share of that gap the rung closes. A rung at `100 %` reads \
         the source's number; a rung at `0 %` reads the engine's; above 100 % it has overshot.\n"
    );
    for key in keys {
        let source = &key.rungs[SOURCE];
        let engine = &key.rungs[ENGINE];
        let _ = writeln!(md, "### {}\n", key.name);
        let _ = writeln!(
            md,
            "| rung | Δ rough | Δ width excess | Δ mod | Δ mod >5 Hz | Δ attack | Δ attack flat | closed: rough | mod >5 Hz | attack flat |\n\
             |:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|"
        );
        for rung in &key.rungs {
            let rough = Distance::of(rung.roughness_spline_db, source.roughness_spline_db);
            let width = Distance::of(rung.width_excess_cents, source.width_excess_cents);
            let modulation = Distance::of(rung.mod_energy_db, source.mod_energy_db);
            let high = Distance::of(rung.mod_high_db, source.mod_high_db);
            let att = Distance::of(rung.attack_level_db, source.attack_level_db);
            let att_flat = Distance::of(rung.attack_flatness_db, source.attack_flatness_db);
            let closed = |d: &Distance, gap: f64| -> String {
                if gap.abs() < 1e-9 || !gap.is_finite() || !d.delta.is_finite() {
                    return "—".into();
                }
                format!("{:.0} %", 100.0 * (1.0 - d.delta / gap))
            };
            let _ = writeln!(
                md,
                "| `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
                rung.label,
                f2(rough.delta),
                f2(width.delta),
                f3(modulation.delta),
                f3(high.delta),
                f1(att.delta),
                f1(att_flat.delta),
                closed(
                    &rough,
                    engine.roughness_spline_db - source.roughness_spline_db
                ),
                closed(&high, engine.mod_high_db - source.mod_high_db),
                closed(
                    &att_flat,
                    engine.attack_flatness_db - source.attack_flatness_db
                ),
            );
        }
        let _ = writeln!(md);
    }
}

/// Each ingredient is one rung minus its own control: what adding that one
/// thing moved, isolated.
fn ingredient_section(md: &mut String, keys: &[KeyReport]) {
    let _ = writeln!(md, "## 6. Ingredient → metrics moved\n");
    let _ = writeln!(
        md,
        "Each row is one rung against its own control, so the difference is that ingredient and \
         nothing else, and the last column says whether the move was toward the source. \
         `04 − 09` is the excitation roughness on the engine's string, `05 − 09` a 2-cent \
         random-walk linewidth on the same string, `09 − 07` everything the engine has beside the \
         string, `06 − 07` and `08 − 01` the recording's own attack residual, `02 − 01` replacing \
         every measured envelope with the fitted law, `03 − 01` replacing the measured excitation \
         with a smooth comb.\n"
    );
    let pairs: [(usize, usize, &str); 7] = [
        (4, 9, "excitation roughness (per-partial a_k(0))"),
        (5, 9, "linewidth (2 c random-walk detune)"),
        (9, 7, "the engine beside the string, removed"),
        (6, 7, "attack residual on the engine"),
        (8, 1, "attack residual on the resynthesis"),
        (2, 1, "fitted decay law replacing measured envelopes"),
        (3, 1, "smooth comb replacing measured excitation"),
    ];
    let _ = writeln!(
        md,
        "| key | ingredient | Δ rough | Δ width excess | Δ mod | Δ mod >5 Hz | Δ attack | Δ attack flat | toward source? |\n\
         |:--|:--|--:|--:|--:|--:|--:|--:|:--|"
    );
    for key in keys {
        for &(a, b, caption) in &pairs {
            let (ra, rb, source) = (&key.rungs[a], &key.rungs[b], &key.rungs[SOURCE]);
            let toward = |x: f64, y: f64, target: f64| -> i32 {
                if !(x.is_finite() && y.is_finite() && target.is_finite()) {
                    return 0;
                }
                let before = (y - target).abs();
                let after = (x - target).abs();
                if after < before - 1e-9 {
                    1
                } else if after > before + 1e-9 {
                    -1
                } else {
                    0
                }
            };
            let votes = [
                toward(
                    ra.roughness_spline_db,
                    rb.roughness_spline_db,
                    source.roughness_spline_db,
                ),
                toward(
                    ra.width_excess_cents,
                    rb.width_excess_cents,
                    source.width_excess_cents,
                ),
                toward(ra.mod_energy_db, rb.mod_energy_db, source.mod_energy_db),
                toward(ra.mod_high_db, rb.mod_high_db, source.mod_high_db),
                toward(
                    ra.attack_level_db,
                    rb.attack_level_db,
                    source.attack_level_db,
                ),
                toward(
                    ra.attack_flatness_db,
                    rb.attack_flatness_db,
                    source.attack_flatness_db,
                ),
            ];
            let plus = votes.iter().filter(|&&v| v > 0).count();
            let minus = votes.iter().filter(|&&v| v < 0).count();
            let _ = writeln!(
                md,
                "| {} | {} ({} − {}) | {} | {} | {} | {} | {} | {} | {plus} closer, {minus} further |",
                key.name,
                caption,
                key.rungs[a].label,
                key.rungs[b].label,
                f2(ra.roughness_spline_db - rb.roughness_spline_db),
                f2(ra.width_excess_cents - rb.width_excess_cents),
                f3(ra.mod_energy_db - rb.mod_energy_db),
                f3(ra.mod_high_db - rb.mod_high_db),
                f1(ra.attack_level_db - rb.attack_level_db),
                f1(ra.attack_flatness_db - rb.attack_flatness_db),
            );
        }
    }
    let _ = writeln!(md);
}

/// Tests the hypothesis that envelope liveliness is the single most diagnostic
/// number: a diagnostic metric separates the engine from the source *and* puts
/// an exact per-partial resynthesis of the source next to it.
fn diagnostic_section(md: &mut String, keys: &[KeyReport]) {
    let _ = writeln!(md, "## 7. Which metric is diagnostic\n");
    let _ = writeln!(
        md,
        "A number is diagnostic of \"the engine does not sound like the piano\" only if it does two \
         things at once: put the engine (`07`) far from the recording (`00`), *and* put an exact \
         additive resynthesis of the recording (`01`) close to it. The second half is the test that \
         matters — a metric that separates `01` from `00` is measuring the resynthesis's own \
         artefacts (free phases, mono trajectories, an unmeasured first 43 ms), not the engine's \
         deficit. `ratio` is |`07` − `00`| / |`01` − `00`|: how many times larger the engine's gap \
         is than the metric's own floor. It has to hold at all three keys to be worth anything.\n"
    );
    let metrics: [(&str, fn(&RungReport) -> f64, &str); 10] = [
        ("modulation lines to half", |r| r.mod_lines, "lines"),
        ("roughness (spline RMS)", |r| r.roughness_spline_db, "dB"),
        ("roughness (median)", |r| r.roughness_median_db, "dB"),
        ("linewidth excess", |r| r.width_excess_cents, "cents"),
        ("modulation, whole band", |r| r.mod_energy_db, "dB"),
        ("modulation below 5 Hz", |r| r.mod_low_db, "dB"),
        ("modulation above 5 Hz", |r| r.mod_high_db, "dB"),
        ("modulation flatness", |r| r.mod_flatness_db, "dB"),
        ("attack level", |r| r.attack_level_db, "dB"),
        ("attack flatness", |r| r.attack_flatness_db, "dB"),
    ];
    let _ = writeln!(
        md,
        "| metric | key | `00` | `01` | `07` | engine gap | resynth gap | ratio |\n\
         |:--|:--|--:|--:|--:|--:|--:|--:|"
    );
    let mut scores: Vec<(String, Vec<f64>, Vec<f64>)> = Vec::new();
    for (name, get, unit) in metrics {
        let mut ratios = Vec::new();
        let mut gaps = Vec::new();
        for key in keys {
            let source = get(&key.rungs[SOURCE]);
            let resynth = get(&key.rungs[RESYNTH]);
            let engine = get(&key.rungs[ENGINE]);
            let engine_gap = engine - source;
            let resynth_gap = resynth - source;
            let ratio = if resynth_gap.abs() > 1e-9 {
                engine_gap.abs() / resynth_gap.abs()
            } else {
                f64::INFINITY
            };
            ratios.push(ratio);
            gaps.push(engine_gap.abs());
            let _ = writeln!(
                md,
                "| {} ({}) | {} | {} | {} | {} | {} | {} | {} |",
                name,
                unit,
                key.name,
                f2(source),
                f2(resynth),
                f2(engine),
                f2(engine_gap),
                f2(resynth_gap),
                f1(ratio)
            );
        }
        scores.push((name.to_string(), ratios, gaps));
    }
    let _ = writeln!(md);
    let _ = writeln!(
        md,
        "**Ranked by the worst of the three keys** — the number a metric can be relied on for. \
         `smallest gap` is the smallest engine deficit the metric reports across the three keys, \
         in the metric's own units: a metric can have a fine ratio and still be measuring \
         something too small to hear.\n"
    );
    let _ = writeln!(
        md,
        "| metric | worst-key ratio | ratio at C4 / A2 / C6 | smallest gap |\n|:--|--:|:--|--:|"
    );
    let mut ranked: Vec<(String, f64, String, f64)> = scores
        .into_iter()
        .map(|(name, ratios, gaps)| {
            let worst = ratios.iter().cloned().fold(f64::INFINITY, f64::min);
            let listed = ratios
                .iter()
                .map(|r| format!("{r:.1}"))
                .collect::<Vec<_>>()
                .join(" / ");
            let smallest = gaps.iter().cloned().fold(f64::INFINITY, f64::min);
            (name, worst, listed, smallest)
        })
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (name, worst, listed, smallest) in &ranked {
        let _ = writeln!(
            md,
            "| {name} | {} | {listed} | {} |",
            f1(*worst),
            f2(*smallest)
        );
    }
    let _ = writeln!(md);
}

/// The executive summary, with the numbers that decide it substituted in.
fn executive_summary(md: &mut String, keys: &[KeyReport]) {
    let _ = writeln!(md, "## Executive summary\n");
    let mut lines = Vec::new();
    for key in keys {
        let s = &key.rungs[SOURCE];
        let r = &key.rungs[RESYNTH];
        let e = &key.rungs[ENGINE];
        lines.push(format!(
            "| {} | {} / {} / {} | {} / {} / {} | {} / {} / {} | {} / {} / {} | {} / {} / {} | {} / {} / {} |",
            key.name,
            f2(s.roughness_spline_db),
            f2(r.roughness_spline_db),
            f2(e.roughness_spline_db),
            f2(s.width_excess_cents),
            f2(r.width_excess_cents),
            f2(e.width_excess_cents),
            f2(s.mod_energy_db),
            f2(r.mod_energy_db),
            f2(e.mod_energy_db),
            f2(s.mod_high_db),
            f2(r.mod_high_db),
            f2(e.mod_high_db),
            f1(s.attack_level_db),
            f1(r.attack_level_db),
            f1(e.attack_level_db),
            f1(s.attack_flatness_db),
            f1(r.attack_flatness_db),
            f1(e.attack_flatness_db),
        ));
    }
    let _ = writeln!(
        md,
        "Source `00` / resynthesis `01` / engine `07`, on every metric:\n"
    );
    let _ = writeln!(
        md,
        "| key | roughness dB | linewidth excess c | modulation dB | modulation >5 Hz dB | attack dB | attack flatness dB |\n\
         |:--|:--|:--|:--|:--|:--|:--|"
    );
    for line in lines {
        let _ = writeln!(md, "{line}");
    }
    let _ = writeln!(md);
}

/// The conclusion the tables support, assembled from them rather than asserted.
fn recommendation(md: &mut String, keys: &[KeyReport]) {
    let _ = writeln!(md, "## 8. What the measurements say\n");
    let _ = writeln!(md, "The three rungs that decide it, per key:\n");

    let mut summary = Vec::new();
    for key in keys {
        let s = &key.rungs[SOURCE];
        let r = &key.rungs[RESYNTH];
        let e = &key.rungs[ENGINE];
        summary.push(format!(
            "- **{}**: modulation above 5 Hz {} dB (source) / {} (resynth) / {} (engine); whole \
             band {} / {} / {}; flatness {} / {} / {} dB; linewidth excess over the per-partial \
             floor {} / {} / {} cents; roughness {} / {} / {} dB; attack {} / {} / {} dB at \
             flatness {} / {} / {} dB.",
            key.name,
            f2(s.mod_high_db),
            f2(r.mod_high_db),
            f2(e.mod_high_db),
            f2(s.mod_energy_db),
            f2(r.mod_energy_db),
            f2(e.mod_energy_db),
            f1(s.mod_flatness_db),
            f1(r.mod_flatness_db),
            f1(e.mod_flatness_db),
            f2(s.width_excess_cents),
            f2(r.width_excess_cents),
            f2(e.width_excess_cents),
            f2(s.roughness_spline_db),
            f2(r.roughness_spline_db),
            f2(e.roughness_spline_db),
            f1(s.attack_level_db),
            f1(r.attack_level_db),
            f1(e.attack_level_db),
            f1(s.attack_flatness_db),
            f1(r.attack_flatness_db),
            f1(e.attack_flatness_db),
        ));
    }
    for line in summary {
        let _ = writeln!(md, "{line}");
    }
    let _ = writeln!(md);
    let _ = writeln!(md, "{CONCLUSION}");
}

/// The reading of the tables above. Written by hand from them and kept in the
/// generator so the file is one artefact rather than a table dump with a note
/// stapled to it; every number quoted here appears in a table above.
const CONCLUSION: &str = r#"
### The hypothesis, tested

**Envelope liveliness is the most diagnostic single number, and it is the whole 0.1–20 Hz band
rather than any refinement of it.** Section 7 ranks the metrics by the worst of the three keys, and
modulation energy over the whole band wins by a factor of three: the engine stands 0.97–2.32 dB
from the recording while an exact additive resynthesis of the recording stands 0.01–0.04 dB from
it, a ratio of 32 at the worst key and 259 at the best. No other metric is that clean at all three
keys. The hypothesis is confirmed in its main claim.

It is refuted in its detail. The prediction was that a pure exponential has no modulation, beating
unisons have discrete lines and *the recording has a continuum*. The first two hold — rung `02`,
the fitted two-exponential law, returns 0.045 dB (C4), 0.001 (A2) and 0.797 (C6) against the
recording's 2.58 / 2.38 / 3.51, which is to say four numbers per partial describe an envelope that
does not move at all. The third does not: the recording needs **one** modulation line to account for
half its band energy at C4 and A2, the same as the engine, and its energy above 5 Hz is 0.10 and
0.06 dB — a −33 dB noise pedestal by the calibration in the validation list. The recording's
envelopes are discrete beats, not a continuum. Only C6 is different (3.5 lines, 2.38 dB above
5 Hz), and there the engine has 2.04 dB of it already while the *resynthesis* has 1.12.

So the diagnostic quantity is **how much** each partial's level moves, not what shape the movement
has. And the engine's error is not one-signed: it moves **too much** at C4 (4.90 against 2.58 dB)
and at C6 (5.08 against 3.51), and **too little** at A2 (1.41 against 2.38).

### Which ingredients the measurements say matter

1. **Per-partial decays — specifically the beat structure, not the rate.** Largest and most
   consistent effect in the ladder. `02 − 01` is the whole of it: replacing every measured envelope
   with the engine's own fitted law, at the same excitation and the same frequency, removes 2.35 to
   2.76 dB RMS of per-partial level movement and moves *six of six* metrics away from the source at
   C4 and C6. `TUNING_REPORT.md` §1 and the decay stage already place the rates correctly; what is
   missing is that three strings and two polarizations make six components and fifteen beat rates,
   and the model fits two. `DECISIONS.md`'s backlog item 3 — a per-string `sigma` scale, one preset
   field — is the cheapest thing on this list that acts on the most diagnostic number.

2. **Per-partial amplitudes — and there are two separate faults, of opposite sign.** Roughness is
   the second-ranked metric (ratio 12–30, consistent at three keys). Between the nulls the engine's
   comb is *too smooth*: median deviation 2.46 dB at C4 against the recording's 5.51. At its nulls
   it is far too rough: section 4a shows the engine's worst partial is exactly where `sin(k π x)`
   crosses zero — k = 17 at A2 (predicted −42 dB, measured −35.7) and k = 8 at C6 (predicted −42,
   measured −25.9) — while the recording's deepest partial anywhere is 9.3 to 17.7 dB below a smooth
   envelope and never at those indices. A real hammer has width and a real string has stiffness; the
   contact taper the engine applies is a low-pass in `k` and does not fill a null. Softening the
   null is a one-line change and is the only item here with no fitting cost. Filling in the
   between-null roughness is backlog item 6 and stays expensive.

3. **The attack residual — the most reliably-signed ingredient in the ladder, and not diagnostic of
   the engine.** The first 30 ms with the tracked partials subtracted is 11.1 to 12.7 dB *more
   tonal* in the engine than in the recording, and 12.3 to 14.6 dB more tonal in the additive
   resynthesis: rung `01` is as wrong here as rung `07`, which is why the metric scores 0.8–1.0 in
   section 7 and why this is a deficit of per-partial modelling rather than of the engine. Mixing
   the recording's own residual back in closes it at every key and on both hosts — `06` lands 0.2 to
   2.6 dB from the source's flatness, `08` within 1.6 dB — which is the cleanest single intervention
   anywhere in the ladder. `TUNING_REPORT.md` §4 refuted a missing attack *level*; this is a missing
   attack *spectrum*, and it is compatible with that refutation. (Read the level column with care:
   partials above the tracked set are not subtracted and count as residual, so the flatness is the
   trustworthy half.)

4. **Linewidth — refuted, and rung `05` is measurably a wrong turn.** With each partial's own decay
   and the window's own resolution subtracted (the per-partial floor at the head of each key's
   section), the recording's excess width is −0.03, +0.14 and −0.20 cents: zero. The engine's is
   +0.20, +0.05, −0.31: also zero. There is nothing to close. The 2-cent random walk of rung `05`
   adds 0.19–0.35 cents of width and 0.42–1.42 dB of envelope modulation above 5 Hz that the
   recording does not have, and moves six of six metrics away from the source at A2 and C6. Do not
   build it.

### Does rung `01` close the gap?

**On everything a per-partial steady-state model can represent, yes; on three things, no.**

Closed: spectral-envelope roughness (94 %, 92 %, 103 % of the engine's gap), linewidth (identical to
the source at every partial of every key, to 0.01 cents at A2), and modulation energy (0.009, 0.030
and 0.040 dB from the source against the engine's 0.97–2.32). Rung `01` is not a near miss on these
— it is the recording.

Not closed, and measurably:

* **The attack.** 12.3–14.6 dB too tonal in the first 30 ms, at all three keys. Free phases and a
  2 ms ramp cannot make a hammer noise.
* **The fine structure of the envelope.** Modulation flatness stands 6.2 to 18.0 dB more line-like
  than the source's, and at C6 the energy above 5 Hz is 1.12 dB against 2.38. The trajectories are
  measured on a window of four periods of the fundamental — 85 ms at C4, 171 ms at A2 — and the
  magnitude in a window that long is a low-pass on the envelope with a corner near 10 Hz (C4) and
  5 Hz (A2), whatever the hop. Anything the envelope does faster than that is smoothed out of the
  trajectories before the resynthesis ever sees it. This is a limit of the *tracker*, not of
  additive synthesis, and it means rung `01` is a slightly conservative ceiling rather than an exact
  one — the measurements above use a 15–37 ms window precisely so they can see past it.
* **The late treble.** At C6 the resynthesis has no measurable envelope at all at partials 5 and 6
  where the recording has one — `TUNING_REPORT.md` §4's halo, which per-partial trajectories do not
  contain by construction.

### What listening has to decide, and measurement cannot

* **Which of the two amplitude faults matters.** The engine is simultaneously too smooth between the
  nulls and too deep at them. Both are measured; nothing here ranks them by audibility, and they
  have very different costs.
* **The sign of the modulation error.** The engine's partials move too much at C4 and C6 and too
  little at A2. The metric says the amount is wrong; it cannot say whether a partial that beats
  twice as hard as the piano's sounds livelier or sounds mechanical. `04` against `09` and `05`
  against `09` are the two files that answer that.
* **Whether 11 dB of attack flatness at −20 dB is audible at all.** The measurement is unambiguous
  and the level is low. `06` against `07`, and `08` against `01`, are the only evidence that counts.
* **Whether rung `01` sounds like the piano.** If it does, the three unclosed items above are
  incidental and the engine's own gaps in section 5 are the whole story. If it does not, then the
  missing sound is in one of those three — or in something none of these five measurements chose to
  look at, which is the possibility no table in this file can exclude.
"#;

fn f1(v: f64) -> String {
    if v.is_finite() {
        format!("{v:.1}")
    } else {
        "—".into()
    }
}
fn f2(v: f64) -> String {
    if v.is_finite() {
        format!("{v:.2}")
    } else {
        "—".into()
    }
}
fn f3(v: f64) -> String {
    if v.is_finite() {
        format!("{v:.3}")
    } else {
        "—".into()
    }
}
