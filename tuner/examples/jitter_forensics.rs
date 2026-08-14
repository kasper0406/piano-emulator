//! Jitter forensics: the instantaneous frequency of a composite partial,
//! measured on the engine and on the recording it is fitted to, and attributed
//! to the mechanism that produces it.
//!
//! ```text
//! cargo run --release --example jitter_forensics -- \
//!     [data/salamander] [presets/salamander-c5.toml] [renders/jitter]
//! ```
//!
//! # The percept this is built to measure
//!
//! Every partial the engine renders is a **sum of independent sinusoids**: one
//! per unison string per polarization, two to six of them, at frequency offsets
//! the preset fixes once and for all (`notes.detune_cents` as a ratio,
//! `voicing.horizontal_offset_hz` as a constant number of hertz) and with decay
//! rates that differ per string (`voicing.unison_sigma_scale`) and per partial
//! (`notes.partial_sigma_scale`). Nothing couples them, so their beats never
//! change rate and never lose phase: the sum's *instantaneous frequency* swings
//! back and forth at exactly those beat rates forever, and the swing is
//! unbounded wherever two components pass through equal amplitude — at an exact
//! null the composite phase steps by pi in no time at all.
//!
//! A real unison is three **coupled** oscillators (Weinreich, JASA 62 1474,
//! 1977): the bridge admittance ties them into eigenmodes with shifted
//! frequencies and split decay rates, and the composite partial that radiates
//! has a linewidth near zero — which is exactly what
//! `renders/timbre-ladder/ANALYSIS.md` measures on the source recordings
//! (linewidth excess −0.03 / +0.14 / −0.20 cents at C4 / A2 / C6). So the
//! recording's own numbers are the pass bar, on every statistic here.
//!
//! # What is measured
//!
//! For partials 1..[`MAX_PARTIAL`] of four keys, on the recording and on every
//! bisection render:
//!
//! 1. **The frequency track.** The signal is transformed once, a Gaussian
//!    band-pass of time-constant [`SMOOTH_SIGMA_S`] is centred on the partial's
//!    own spectral peak, and the inverse transform is the partial's analytic
//!    signal. Demodulating it at that carrier and differentiating the phase
//!    gives the instantaneous frequency, already smoothed by the filter and by
//!    nothing else. Reported as RMS deviation in cents from the partial's own
//!    power-weighted mean frequency, the 95th percentile of that deviation, and
//!    the number of separate excursions past ±[`EXCURSION_CENTS`] per second —
//!    all three inside [`MOD_LO_HZ`]–[`MOD_HI_HZ`], because a partial's phase is
//!    also perturbed by everything else radiating near it and that perturbation
//!    is broadband while a beat is not. The unrestricted RMS is reported beside
//!    them, and so is its power-weighted version, which is what says whether a
//!    partial's wobble happens while it is loud or only at the null of a beat.
//!
//! 2. **The modulation spectrum of that frequency track** over
//!    [`MOD_LO_HZ`]–[`MOD_HI_HZ`]. Free-running components beat at fixed rates,
//!    so their frequency track is a few **discrete lines**; a coupled or
//!    noise-driven instrument gives a continuum. Reported as spectral flatness
//!    (0 dB is a continuum, strongly negative is a line), the line-to-continuum
//!    ratio (peak bin over median bin) and how many bins it takes to account
//!    for half the band's energy.
//!
//! 3. **The amplitude side of the same demodulation** — beat depth and the same
//!    line statistics on the log envelope, which is the "metronomic beating"
//!    half of the hypothesis.
//!
//! Five synthetic controls run through the identical code: one exponentially
//! decaying sinusoid (the floor every number here is measured against), the same
//! on three white-noise pedestals (how much of a row can be a measurement of the
//! background), and two equal-amplitude sinusoids [`CONTROL_BEAT_HZ`] apart (the
//! artefact in its pure form). None of them is a fit to anything; they are what
//! the statistics read when the answer is known.
//!
//! # The bisection
//!
//! [`VARIANTS`] disables one mechanism at a time in a clone of the shipped
//! preset — never in the engine, which this example only reads — and re-renders
//! the same note. Each variant's numbers against the shipped preset's are that
//! mechanism's contribution; `all_off` is the control that says how much of
//! what is left belongs to the rest of the instrument (the soundboard's diffuse
//! field, the resonance bus, the hammer's noise).
//!
//! Every render is also written to the output directory, level-matched to the
//! recording over the same window, so the numbers can be listened to.

use std::f64::consts::TAU;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use piano_emulator::preset::{Preset as EnginePreset, UnisonSigmaScale, Voicing};
use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_emulator::resonance::BridgeFilter;
use piano_emulator::string::{contact_taper, StringParams};
use piano_emulator::types::{key_index, Event, MAX_UNISON};
use piano_tuner::{audio, detect_onset, Sample, SampleLibrary, SAMPLE_RATE};
use rustfft::{num_complex::Complex64, FftPlanner};

const SR: f64 = SAMPLE_RATE as f64;

/// The four keys, in the order they are reported. Two of them (C4, A2) are the
/// keys `renders/timbre-ladder/ANALYSIS.md` already measures a linewidth on, so
/// its recording numbers are directly comparable; A4 and C6 add a triple unison
/// whose detuning is wide enough for the beats to land inside the audible
/// modulation band.
const KEYS: [(u8, &str); 4] = [(60, "C4"), (45, "A2"), (84, "C6"), (69, "A4")];

/// Partials reported per key. Low enough that every key has them and low enough
/// that the ear resolves them individually.
const MAX_PARTIAL: usize = 4;

/// Velocity every bisection render is struck at, and the two extra velocities
/// the shipped preset is rendered at to test the amplitude-equality prediction.
const VELOCITY: u8 = 90;
const EXTRA_VELOCITIES: [u8; 2] = [40, 120];

/// Silence before the strike in every render, in seconds.
const PREROLL_S: f64 = 0.05;
/// Length of every written file, past the preroll.
const NOTE_S: f64 = 4.0;
/// How long the engine is actually rendered for: past the analysis window with
/// room for the band-pass filter's own tail.
const RENDER_S: f64 = 4.5;

/// The analysis window, in seconds since the strike. Past the attack and the
/// hammer's noise, and inside the part of the record every key still sounds in.
const T0_S: f64 = 0.3;
const T1_S: f64 = 3.0;

/// Time-domain standard deviation of the Gaussian band-pass, i.e. the smoothing
/// the frequency track gets. Its frequency-domain width is
/// `1 / (2 pi SMOOTH_SIGMA_S)` = 31.8 Hz, so the nearest neighbouring partial of
/// the lowest key here (A2, 110 Hz apart) is 3.5 sigma out and 54 dB down.
const SMOOTH_SIGMA_S: f64 = 0.005;

/// Rate the demodulated track is decimated to before the phase is
/// differentiated. Far above the filter's own 32 Hz bandwidth, so nothing is
/// lost, and far below the carrier, so no phase difference can wrap.
const TRACK_HZ: f64 = 1000.0;

/// A frequency excursion this far from the partial's mean pitch is counted, and
/// the count is of separate *runs* rather than of samples. Three cents is about
/// the pitch discrimination limit for a sustained tone in this register, and it
/// is the tolerance `SPEC.md`'s own tuning test allows.
const EXCURSION_CENTS: f64 = 3.0;

/// The modulation band both spectra are reported over. Matches
/// `renders/timbre-ladder/ANALYSIS.md` metric 3 so the two are comparable. The
/// 2.7 s track resolves 0.37 Hz, which is stated with the numbers.
const MOD_LO_HZ: f64 = 0.1;
const MOD_HI_HZ: f64 = 20.0;

/// Transform length the band-pass is applied in: 5.46 s at 48 kHz, longer than
/// anything analysed, so the filter never wraps into its own input.
const FFT_N: usize = 1 << 18;

/// The synthetic control pair's separation, chosen inside the engine's own
/// range of beat rates (`voicing.horizontal_offset_hz` is 0.27–0.52 Hz).
const CONTROL_BEAT_HZ: f64 = 0.35;

/// Window every render's level is matched over, in seconds since the strike.
const MATCH_LO_S: f64 = 0.2;
const MATCH_HI_S: f64 = 2.0;

/// Fades applied to every written file, so nothing can click at either edge.
const FADE_IN_S: f64 = 0.002;
const FADE_OUT_S: f64 = 0.030;

// ------------------------------------------------------------- the bisection

/// One rung of the bisection: a label, what it does, and the edit it makes to a
/// clone of the shipped preset. The engine is never touched; every one of these
/// is a preset a user could write.
type Edit = fn(&mut EnginePreset);

const VARIANTS: &[(&str, &str, Edit)] = &[
    ("engine", "shipped preset (control)", |_| {}),
    ("no_detune", "notes.detune_cents = 0", |p| {
        for d in &mut p.notes.detune_cents {
            *d = 0.0;
        }
    }),
    (
        "no_hoffset",
        "voicing.horizontal_offset_hz = 0",
        |p| p.voicing.horizontal_offset_hz = vec![0.0; MAX_UNISON],
    ),
    (
        "flat_unison_sigma",
        "voicing.unison_sigma_scale = ones",
        |p| p.voicing.unison_sigma_scale = unity_sigma_scale(),
    ),
    ("no_partial_gains", "notes.partial_gains dropped", |p| {
        p.notes.partial_gains = Vec::new()
    }),
    (
        "no_partial_sigma",
        "notes.partial_sigma_scale dropped",
        |p| p.notes.partial_sigma_scale = Vec::new(),
    ),
    ("no_coupling", "voicing.unison_coupling = 0", |p| {
        p.voicing.unison_coupling = 0.0
    }),
    ("single_string", "notes.unison = 1 (no unison at all)", |p| {
        for u in &mut p.notes.unison {
            *u = 1;
        }
    }),
    ("all_off", "every row above at once", |p| {
        for d in &mut p.notes.detune_cents {
            *d = 0.0;
        }
        p.voicing.horizontal_offset_hz = vec![0.0; MAX_UNISON];
        p.voicing.unison_sigma_scale = unity_sigma_scale();
        p.notes.partial_gains = Vec::new();
        p.notes.partial_sigma_scale = Vec::new();
        p.voicing.unison_coupling = 0.0;
    }),
];

/// `preset::unity_sigma_scale`, which is private there.
fn unity_sigma_scale() -> Vec<UnisonSigmaScale> {
    (1..=MAX_UNISON)
        .map(|n| UnisonSigmaScale {
            scale: vec![1.0; n],
        })
        .collect()
}

// ------------------------------------------------------------ the measurement

/// One partial's demodulated track, on the [`TRACK_HZ`] grid.
struct Track {
    /// Power-weighted mean instantaneous frequency, Hz — the pitch the
    /// deviations below are quoted against.
    mean_hz: f64,
    /// The partial's own bin over the median bin of the neighbourhood it was
    /// found in, dB: a signal-to-background density that says how much of a row
    /// could be a measurement of what is radiating *beside* the partial rather
    /// than of the partial itself.
    peak_db: f64,
    /// Instantaneous frequency deviation from `mean_hz`, cents.
    cents: Vec<f64>,
    /// Envelope of the same demodulation, dB.
    amp_db: Vec<f64>,
    /// `|y|^2`, normalised to a maximum of one — the weight the amplitude-
    /// weighted deviation uses.
    weight: Vec<f64>,
}

const MIN_PEAK_DB: f64 = 10.0;

/// Everything one partial of one signal contributes to the tables.
struct PartialStats {
    mean_hz: f64,
    /// [`Track::peak_db`], carried through so a row can be written from the
    /// statistics alone.
    peak_db: f64,
    /// RMS of the frequency deviation inside the modulation band, cents — the
    /// headline number, and the one that is not a measurement of the noise
    /// floor. A partial's phase is perturbed by everything else that radiates
    /// near it (the diffuse field, the room, the sympathetic bus), and that
    /// perturbation is broadband: it fills the whole 500 Hz the 1 kHz grid can
    /// express, while a beat between two components of the partial itself is
    /// one line under 20 Hz. Restricting to [`MOD_LO_HZ`]–[`MOD_HI_HZ`] keeps
    /// the second and rejects most of the first — the `all_off` render, which
    /// is one sinusoid per partial by construction, reads 0.00–0.01 cents here
    /// and up to 8.7 cents unrestricted.
    band_cents: f64,
    /// 95th percentile of `|deviation|` inside the same band, cents.
    p95_cents: f64,
    /// Separate runs past ±[`EXCURSION_CENTS`] inside the same band, per second.
    excursions_per_s: f64,
    /// RMS of the *unrestricted* deviation, cents: the same track with the
    /// broadband perturbation left in, so the two columns together say how much
    /// of a signal's jitter is its own beating and how much is everything else.
    raw_cents: f64,
    /// The unrestricted deviation weighted by `|y|^2`: what the excursions are
    /// worth once the fact that the largest of them happen at the quietest
    /// instants is taken into account.
    weighted_cents: f64,
    /// Modulation spectrum of the frequency track.
    freq_mod: ModStats,
    /// Peak-to-trough span of the detrended log envelope, dB (p95 − p5).
    beat_depth_db: f64,
    /// Modulation spectrum of the log envelope.
    amp_mod: ModStats,
}

/// The line-versus-continuum statistics of one modulation spectrum.
#[derive(Clone, Copy, Default)]
struct ModStats {
    /// RMS of the detrended track inside the band, in the track's own unit.
    rms: f64,
    /// Spectral flatness of the band, dB. Zero is a continuum; −20 dB and below
    /// is one or two discrete lines.
    flatness_db: f64,
    /// Peak bin over median bin, dB — the line-to-continuum ratio.
    line_db: f64,
    /// How many bins it takes to account for half the band's energy.
    lines_to_half: usize,
    /// Power-weighted centre of the band, Hz.
    centroid_hz: f64,
}

// Deliberately *not* here: an autocorrelation "metronome" statistic. It was
// written, measured and removed. The engine's beat rates at these keys are
// 0.05-5 Hz, so a period of two to twenty seconds has fewer than two cycles
// inside a 2.7 s window and no autocorrelation can find it; and inside the
// 0.1-20 Hz band a component near the bottom edge correlates with itself at
// every short lag simply because it is smooth, which is why the `all_off`
// render — one sinusoid per partial, 0.4 dB of envelope movement in total —
// scored 0.78-0.89 on it. What the tables do say about regularity is
// load-bearing and does not need it: the engine's numbers are identical to two
// decimals at velocities 40, 90 and 120 while the recording's are not, and the
// beat inventory above is a list of rates the preset fixes once.

/// The forward transform of one signal, computed once and reused by every
/// partial of it.
struct Spectrum {
    bins: Vec<Complex64>,
}

impl Spectrum {
    /// `signal` starts at the strike.
    fn new(signal: &[f64], planner: &mut FftPlanner<f64>) -> Spectrum {
        let mut bins: Vec<Complex64> = (0..FFT_N)
            .map(|n| Complex64::new(signal.get(n).copied().unwrap_or(0.0), 0.0))
            .collect();
        planner.plan_fft_forward(FFT_N).process(&mut bins);
        Spectrum { bins }
    }

    /// Frequency of bin `m`.
    fn hz(m: usize) -> f64 {
        m as f64 * SR / FFT_N as f64
    }

    /// The strongest bin within `±half_width` of `nominal`, refined by a
    /// parabolic fit, and how far it stands over the band's median magnitude.
    fn peak_near(&self, nominal: f64, half_width: f64) -> (f64, f64) {
        let bin = |hz: f64| ((hz * FFT_N as f64 / SR).round() as isize).max(1) as usize;
        let lo = bin(nominal - half_width).max(1);
        let hi = bin(nominal + half_width).min(FFT_N / 2 - 2);
        if hi <= lo {
            return (nominal, 0.0);
        }
        let mag = |m: usize| self.bins[m].norm();
        let mut best = lo;
        for m in lo..=hi {
            if mag(m) > mag(best) {
                best = m;
            }
        }
        let mut band: Vec<f64> = (lo..=hi).map(mag).collect();
        band.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = band[band.len() / 2].max(f64::MIN_POSITIVE);
        let peak_db = 20.0 * (mag(best) / median).log10();
        // Parabolic interpolation in log magnitude: the Gaussian band-pass is
        // centred on the result, and a bin is 0.18 Hz wide, so this only has to
        // be good to a fraction of a bin.
        let (a, b, c) = (
            mag(best - 1).max(f64::MIN_POSITIVE).ln(),
            mag(best).max(f64::MIN_POSITIVE).ln(),
            mag(best + 1).max(f64::MIN_POSITIVE).ln(),
        );
        let denom = a - 2.0 * b + c;
        let delta = if denom.abs() > 1e-12 {
            (0.5 * (a - c) / denom).clamp(-0.5, 0.5)
        } else {
            0.0
        };
        (Spectrum::hz(best) + delta * SR / FFT_N as f64, peak_db)
    }

    /// The partial's analytic signal, band-passed by a Gaussian centred on
    /// `carrier` and demodulated down to zero, decimated to [`TRACK_HZ`] and
    /// cut to the analysis window.
    fn demodulate(&self, carrier: f64, planner: &mut FftPlanner<f64>) -> (Vec<Complex64>, f64) {
        // Never wider than a quarter of the carrier: a Gaussian centred at
        // 110 Hz with a 32 Hz width would be cut off at DC, and a cut-off
        // Gaussian is a sharp edge whose ringing is not the partial's phase.
        let sigma_f = (1.0 / (TAU * SMOOTH_SIGMA_S)).min(carrier / 4.0);
        let mut z = vec![Complex64::new(0.0, 0.0); FFT_N];
        // Positive frequencies only, at twice the amplitude: that is the
        // analytic signal, and the Gaussian is what separates this partial from
        // its neighbours. Six sigma out the weight is 1.5e-8, so the sum is
        // over the partial's own neighbourhood and nothing else.
        let span = (6.0 * sigma_f * FFT_N as f64 / SR).ceil() as usize;
        let centre = (carrier * FFT_N as f64 / SR).round() as usize;
        let lo = centre.saturating_sub(span).max(1);
        let hi = (centre + span).min(FFT_N / 2 - 1);
        for (m, bin) in z.iter_mut().enumerate().take(hi + 1).skip(lo) {
            let u = (Spectrum::hz(m) - carrier) / sigma_f;
            *bin = self.bins[m] * (2.0 * (-0.5 * u * u).exp());
        }
        planner.plan_fft_inverse(FFT_N).process(&mut z);
        let scale = 1.0 / FFT_N as f64;
        let step = (SR / TRACK_HZ).round() as usize;
        let from = (T0_S * SR) as usize;
        // One extra sample at each end: the phase is differentiated, and the
        // deviation is quoted over exactly [T0_S, T1_S].
        let to = (T1_S * SR) as usize + step;
        let track: Vec<Complex64> = (from..=to)
            .step_by(step)
            .map(|n| {
                let phase = -TAU * carrier * n as f64 / SR;
                z[n] * scale * Complex64::from_polar(1.0, phase)
            })
            .collect();
        (track, TRACK_HZ)
    }
}

/// Demodulates partial `k` of `signal` and returns its track, or `None` if the
/// partial is not present over its own noise.
fn track_partial(
    spectrum: &Spectrum,
    nominal_hz: f64,
    search_half_width: f64,
    planner: &mut FftPlanner<f64>,
) -> Option<Track> {
    let (carrier_hz, peak_db) = spectrum.peak_near(nominal_hz, search_half_width);
    if peak_db < MIN_PEAK_DB {
        return None;
    }
    let (y, rate) = spectrum.demodulate(carrier_hz, planner);
    if y.len() < 3 {
        return None;
    }
    // Instantaneous frequency from the phase increment. `arg(y[j+1] conj y[j])`
    // is the increment already wrapped into (−pi, pi], which cannot alias here:
    // the filter is 32 Hz wide and the grid is 1 kHz.
    let mut inst = Vec::with_capacity(y.len() - 1);
    let mut weight = Vec::with_capacity(y.len() - 1);
    for j in 0..y.len() - 1 {
        let d = y[j + 1] * y[j].conj();
        inst.push(carrier_hz + d.arg() * rate / TAU);
        // The weight is the geometric mean of the two endpoints' powers, so it
        // sits at the same instant the increment does.
        weight.push((y[j].norm_sqr() * y[j + 1].norm_sqr()).sqrt());
    }
    let total: f64 = weight.iter().sum();
    if total.is_nan() || total <= 0.0 {
        return None;
    }
    let mean_hz: f64 = inst
        .iter()
        .zip(&weight)
        .map(|(f, w)| f * w / total)
        .sum::<f64>();
    if mean_hz.is_nan() || mean_hz <= 0.0 {
        return None;
    }
    let cents: Vec<f64> = inst
        .iter()
        .map(|f| {
            if *f > 0.0 {
                1200.0 * (f / mean_hz).log2()
            } else {
                // A negative instantaneous frequency is what an exact null
                // looks like on this grid; it is a real excursion, and clamping
                // it to the widest value the grid can express is the honest
                // reading rather than dropping it.
                -1200.0 * (rate / 2.0 / mean_hz).log2().abs()
            }
        })
        .collect();
    let peak_power = weight.iter().cloned().fold(0.0f64, f64::max);
    let amp_db: Vec<f64> = weight
        .iter()
        .map(|w| 10.0 * w.max(1e-300).log10())
        .collect();
    let weight: Vec<f64> = weight.iter().map(|w| w / peak_power).collect();
    Some(Track {
        mean_hz,
        peak_db,
        cents,
        amp_db,
        weight,
    })
}

/// Every statistic of one track.
fn statistics(track: &Track) -> PartialStats {
    let n = track.cents.len() as f64;
    let raw_cents = (track.cents.iter().map(|c| c * c).sum::<f64>() / n).sqrt();
    let wsum: f64 = track.weight.iter().sum();
    let weighted_cents = (track
        .cents
        .iter()
        .zip(&track.weight)
        .map(|(c, w)| c * c * w)
        .sum::<f64>()
        / wsum.max(f64::MIN_POSITIVE))
    .sqrt();

    // The frequency track is detrended linearly — a partial whose pitch drifts
    // as its unison dies (`TUNING_REPORT.md` §6) is drift and not jitter — and
    // the log envelope cubically, which is what a two-exponential decay looks
    // like over three seconds and is the detrend `ANALYSIS.md` metric 3 uses.
    let detrended_cents = detrended(&track.cents, 1);
    let freq_mod = mod_spectrum(&detrended_cents, TRACK_HZ);
    let band = band_limited(&detrended_cents, TRACK_HZ);
    let band_cents = (band.iter().map(|c| c * c).sum::<f64>() / n).sqrt();
    let mut abs: Vec<f64> = band.iter().map(|c| c.abs()).collect();
    abs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p95_cents = abs[((abs.len() - 1) as f64 * 0.95) as usize];

    // Runs, not samples: one slow swing past the threshold and back is one
    // excursion however many milliseconds it spends out there.
    let mut runs = 0usize;
    let mut inside = true;
    for c in &band {
        if c.abs() > EXCURSION_CENTS {
            if inside {
                runs += 1;
            }
            inside = false;
        } else {
            inside = true;
        }
    }
    let span_s = n / TRACK_HZ;

    let residual = detrended(&track.amp_db, 3);
    let amp_mod = mod_spectrum(&residual, TRACK_HZ);
    let mut sorted = band_limited(&residual, TRACK_HZ);
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let at = |q: f64| sorted[((sorted.len() - 1) as f64 * q) as usize];
    PartialStats {
        mean_hz: track.mean_hz,
        peak_db: track.peak_db,
        band_cents,
        p95_cents,
        excursions_per_s: runs as f64 / span_s,
        raw_cents,
        weighted_cents,
        freq_mod,
        beat_depth_db: at(0.95) - at(0.05),
        amp_mod,
    }
}

/// `x` with everything outside [`MOD_LO_HZ`]–[`MOD_HI_HZ`] removed, zero phase.
///
/// The track is already detrended, so it has no step at the wrap and a
/// rectangular mask is the right filter: this is a restriction of the same
/// Parseval sum the modulation spectrum reports, put back in the time domain so
/// the excursions can be counted where they happen.
fn band_limited(x: &[f64], rate: f64) -> Vec<f64> {
    let n = x.len();
    if n < 16 {
        return x.to_vec();
    }
    let mut planner = FftPlanner::<f64>::new();
    let mut buf: Vec<Complex64> = x.iter().map(|&v| Complex64::new(v, 0.0)).collect();
    planner.plan_fft_forward(n).process(&mut buf);
    let bin_hz = rate / n as f64;
    for (m, b) in buf.iter_mut().enumerate() {
        let hz = if m <= n / 2 {
            m as f64 * bin_hz
        } else {
            (n - m) as f64 * bin_hz
        };
        if !(MOD_LO_HZ..=MOD_HI_HZ).contains(&hz) {
            *b = Complex64::new(0.0, 0.0);
        }
    }
    planner.plan_fft_inverse(n).process(&mut buf);
    buf.iter().map(|c| c.re / n as f64).collect()
}

/// `x` with a least-squares polynomial of `degree` in time removed.
fn detrended(x: &[f64], degree: usize) -> Vec<f64> {
    let n = x.len();
    let cols = degree + 1;
    // Normal equations on the Vandermonde in `u = 2 j / (n - 1) - 1`, which
    // keeps the matrix well conditioned for the degrees used here.
    let u = |j: usize| 2.0 * j as f64 / (n - 1).max(1) as f64 - 1.0;
    let mut a = vec![0.0f64; cols * cols];
    let mut b = vec![0.0f64; cols];
    let mut p = vec![1.0f64; cols];
    for (j, &value) in x.iter().enumerate() {
        p[0] = 1.0;
        for c in 1..cols {
            p[c] = p[c - 1] * u(j);
        }
        for r in 0..cols {
            for c in 0..cols {
                a[r * cols + c] += p[r] * p[c];
            }
            b[r] += p[r] * value;
        }
    }
    // Gaussian elimination with partial pivoting.
    for c in 0..cols {
        let mut pivot = c;
        for r in c + 1..cols {
            if a[r * cols + c].abs() > a[pivot * cols + c].abs() {
                pivot = r;
            }
        }
        if a[pivot * cols + c].abs() < 1e-12 {
            return x.to_vec();
        }
        for k in 0..cols {
            a.swap(c * cols + k, pivot * cols + k);
        }
        b.swap(c, pivot);
        for r in 0..cols {
            if r == c {
                continue;
            }
            let f = a[r * cols + c] / a[c * cols + c];
            for k in c..cols {
                a[r * cols + k] -= f * a[c * cols + k];
            }
            b[r] -= f * b[c];
        }
    }
    let coeff: Vec<f64> = (0..cols).map(|c| b[c] / a[c * cols + c]).collect();
    (0..n)
        .map(|j| {
            let mut p = 1.0;
            let mut fit = 0.0;
            for &c in &coeff {
                fit += c * p;
                p *= u(j);
            }
            x[j] - fit
        })
        .collect()
}

/// The modulation spectrum of an already-detrended track: how much of it is in
/// the band, and whether the band is lines or a continuum.
fn mod_spectrum(x: &[f64], rate: f64) -> ModStats {
    let n = x.len();
    if n < 16 {
        return ModStats::default();
    }
    // Hann, un-padded: zero padding would interpolate the spectrum and inflate
    // both flatness and the line ratio, and this measurement is about how many
    // *resolved* lines there are.
    let window: Vec<f64> = (0..n)
        .map(|j| 0.5 - 0.5 * (TAU * j as f64 / n as f64).cos())
        .collect();
    let window_energy: f64 = window.iter().map(|w| w * w).sum();
    let mut buf: Vec<Complex64> = x
        .iter()
        .zip(&window)
        .map(|(&v, &w)| Complex64::new(v * w, 0.0))
        .collect();
    FftPlanner::<f64>::new().plan_fft_forward(n).process(&mut buf);

    let bin_hz = rate / n as f64;
    let lo = (MOD_LO_HZ / bin_hz).ceil().max(1.0) as usize;
    let hi = ((MOD_HI_HZ / bin_hz).floor() as usize).min(n / 2 - 1);
    if hi <= lo {
        return ModStats::default();
    }
    let power: Vec<f64> = (lo..=hi).map(|m| buf[m].norm_sqr()).collect();
    let total: f64 = power.iter().sum();
    if total.is_nan() || total <= 0.0 {
        return ModStats::default();
    }
    // Parseval, with the window's own energy divided out: the RMS the band
    // would have on its own.
    let rms = (2.0 * total / (n as f64 * window_energy)).sqrt();
    let mean = total / power.len() as f64;
    let log_mean = power
        .iter()
        .map(|p| p.max(1e-300).ln())
        .sum::<f64>()
        / power.len() as f64;
    let flatness_db = 10.0 * (log_mean.exp() / mean).log10();
    let mut sorted = power.clone();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let median = sorted[sorted.len() / 2].max(1e-300);
    let line_db = 10.0 * (sorted[0] / median).log10();
    let mut acc = 0.0;
    let mut lines_to_half = 0usize;
    for p in &sorted {
        acc += p;
        lines_to_half += 1;
        if acc >= 0.5 * total {
            break;
        }
    }
    let centroid_hz = power
        .iter()
        .enumerate()
        .map(|(i, p)| (lo + i) as f64 * bin_hz * p)
        .sum::<f64>()
        / total;
    ModStats {
        rms,
        flatness_db,
        line_db,
        lines_to_half,
        centroid_hz,
    }
}

// ------------------------------------------------------- what the preset says

/// One sinusoid the engine builds for a partial: which string, which
/// polarization, its frequency, its decay rate and its amplitude at the strike.
struct Component {
    string: usize,
    horizontal: bool,
    hz: f64,
    sigma: f64,
    amp: f64,
}

/// `engine::string::radiated_damping`, which is private there.
fn radiated_damping(params: &StringParams, voicing: &Voicing, partials: usize) -> Vec<f32> {
    let share = match &voicing.bridge {
        Some(bridge) if bridge.radiated_share > 0.0 => bridge.radiated_share,
        _ => return vec![1.0; partials],
    };
    let modes = BridgeFilter::peaks_only(voicing.bridge.as_ref().expect("checked above"));
    (1..=partials)
        .map(|k| {
            let excess = modes.magnitude(params.partial_freq(k)) - 1.0;
            (1.0 + share * excess).clamp(0.25, 4.0)
        })
        .collect()
}

/// Every sinusoid the engine's `string.rs` puts into partial `k` of `key`,
/// rebuilt from the preset by the same arithmetic. Read-only: this predicts
/// what the measurement should find, and the measurement is what decides.
fn components(preset: &EnginePreset, key: u8, k: usize) -> Vec<Component> {
    let params = preset.string_params(key);
    let voicing = &preset.voicing;
    let i = key_index(key).expect("key in range");
    let row = |table: &Vec<Vec<f32>>| -> f32 {
        table
            .get(i)
            .and_then(|r| r.get(k - 1))
            .copied()
            .unwrap_or(1.0)
    };
    let gain_k = row(&preset.notes.partial_gains);
    let sigma_k = row(&preset.notes.partial_sigma_scale);
    let radiated = radiated_damping(&params, voicing, params.partial_count());
    let vertical_factor = voicing.vertical_decay_factor();
    let horizontal_gain = 10f32.powf(voicing.horizontal_gain_db / 20.0);
    let comb = (k as f32 * std::f32::consts::PI * params.strike_position).sin();
    let comb = if params.comb_floor > 0.0 {
        comb.signum() * (comb * comb + params.comb_floor * params.comb_floor).sqrt()
    } else {
        comb
    };
    let base_amp = (comb * contact_taper(k, params.contact_width) * gain_k).abs();
    let mut out = Vec::new();
    for s in 0..params.unison {
        let detune = voicing.detune_ratio(s, params.unison, params.detune_cents);
        let sigma = params.partial_sigma(k)
            * sigma_k
            * vertical_factor
            * voicing.sigma_scale(s, params.unison)
            * radiated[k - 1];
        let f = params.partial_freq(k) * detune;
        let share = voicing.strike_share(s, params.unison);
        out.push(Component {
            string: s,
            horizontal: false,
            hz: f64::from(f),
            sigma: f64::from(sigma),
            amp: f64::from(base_amp * share),
        });
        out.push(Component {
            string: s,
            horizontal: true,
            hz: f64::from(f + voicing.horizontal_offset_hz[s]),
            sigma: f64::from(sigma * voicing.horizontal_decay_ratio),
            amp: f64::from(base_amp * share * horizontal_gain),
        });
    }
    out
}

/// `v0`, `h2`: which string, which polarization.
fn component_name(c: &Component) -> String {
    format!("{}{}", if c.horizontal { "h" } else { "v" }, c.string)
}

/// The pair of components that pass through equal amplitude *loudest*, the time
/// they do it at, the beat rate between them, and how far under the partial's
/// own peak that meeting happens.
///
/// This is the instant the hypothesis predicts the frequency track to swing
/// hardest: two sinusoids of equal amplitude a beat apart have an instantaneous
/// frequency that runs through infinity at the null between them, and how much
/// that is heard depends on how loud the pair still is when it happens.
fn equality_time(components: &[Component]) -> Option<(f64, f64, f64, String)> {
    let level_at = |t: f64| {
        components
            .iter()
            .map(|c| c.amp * (-c.sigma * t).exp())
            .fold(0.0f64, f64::max)
    };
    let mut best: Option<(f64, f64, f64, String)> = None;
    for (i, a) in components.iter().enumerate() {
        for b in &components[i + 1..] {
            if a.amp <= 0.0 || b.amp <= 0.0 || (a.sigma - b.sigma).abs() < 1e-9 {
                continue;
            }
            let t = (a.amp / b.amp).ln() / (a.sigma - b.sigma);
            if t.is_nan() || t <= 0.0 || t > T1_S {
                continue;
            }
            // How far the meeting pair is under whatever is loudest at that
            // instant: 0 dB is two equal components with nothing over them,
            // which is a full null, and −20 dB is a ripple.
            let pair = a.amp * (-a.sigma * t).exp();
            let under = 20.0 * (pair / level_at(t).max(f64::MIN_POSITIVE)).log10();
            if best.as_ref().map_or(true, |(_, _, u, _)| under > *u) {
                best = Some((
                    t,
                    (a.hz - b.hz).abs(),
                    under,
                    format!("{}/{}", component_name(a), component_name(b)),
                ));
            }
        }
    }
    best
}

// ------------------------------------------------------------------- signals

type Stereo = (Vec<f32>, Vec<f32>);

/// One thing to be measured and listened to: stereo audio whose frame 0 is the
/// strike, plus the mono sum the measurement runs on.
struct Signal {
    label: String,
    stereo: Stereo,
    mono: Vec<f64>,
}

impl Signal {
    fn new(label: impl Into<String>, stereo: Stereo) -> Signal {
        let n = stereo.0.len().min(stereo.1.len());
        let mono = (0..n)
            .map(|i| 0.5 * (f64::from(stereo.0[i]) + f64::from(stereo.1[i])))
            .collect();
        Signal {
            label: label.into(),
            stereo,
            mono,
        }
    }
}

/// The library layer of `key` that a strike at `velocity` would trigger.
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

/// The recording, on the engine's clock, cut so that frame 0 is the strike.
fn recording(sample: &Sample) -> Result<Stereo, Box<dyn std::error::Error>> {
    let audio = audio::load_at(&sample.path, SAMPLE_RATE)?;
    let onset = detect_onset(&audio.mono(), SR);
    let start = (onset * SR).round() as usize;
    let frames = (RENDER_S * SR) as usize;
    let channel = |i: usize| -> Vec<f32> {
        let source = &audio.channels[i.min(audio.channel_count() - 1)];
        (0..frames)
            .map(|n| source.get(start + n).copied().unwrap_or(0.0))
            .collect()
    };
    Ok((channel(0), channel(1)))
}

/// The engine's render of one note through its public API, cut so that frame 0
/// is the strike.
fn render(preset: &EnginePreset, key: u8, vel: u8) -> Stereo {
    let events = [RenderEvent::new(
        PREROLL_S as f32,
        Event::NoteOn { key, vel },
    )];
    let (left, right) = render_to_buffer(preset, &events, (PREROLL_S + RENDER_S) as f32);
    let skip = (PREROLL_S * SR) as usize;
    let cut = |c: Vec<f32>| -> Vec<f32> {
        let mut v: Vec<f32> = c.into_iter().skip(skip).collect();
        v.resize((RENDER_S * SR) as usize, 0.0);
        v
    };
    (cut(left), cut(right))
}

/// One exponentially decaying sinusoid, optionally on a white-noise pedestal,
/// and two of them a beat apart at equal amplitude — the floor, the floor with
/// a background, and the artefact, through the identical measurement.
///
/// `noise_db` is quoted against the sinusoid's amplitude at the middle of the
/// analysis window, broadband; what it comes out as *inside* the partial's own
/// neighbourhood is the `S/N` column, which is measured and not assumed.
fn synthetic(hz: f64, sigma: f64, pair: bool, noise_db: Option<f64>) -> Stereo {
    let frames = (RENDER_S * SR) as usize;
    let mut rng = 0x2545_f491_4f6c_dd1du64;
    let mut next = || {
        // xorshift64*, so the control is the same file every run.
        rng ^= rng >> 12;
        rng ^= rng << 25;
        rng ^= rng >> 27;
        (rng.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let mid = (-sigma * 0.5 * (T0_S + T1_S)).exp();
    let noise = noise_db.map_or(0.0, |db| 10f64.powf(db / 20.0) * mid * 12f64.sqrt() / 2.0);
    let mut out = vec![0.0f32; frames];
    for (n, v) in out.iter_mut().enumerate() {
        let t = n as f64 / SR;
        let e = (-sigma * t).exp();
        let mut x = (TAU * hz * t).sin() * e;
        if pair {
            // The same decay on both, so they stay equal in amplitude and the
            // null is exact — the worst case the hypothesis describes.
            x += (TAU * (hz + CONTROL_BEAT_HZ) * t + 0.7).sin() * e;
        }
        *v = (0.2 * (x + noise * next())) as f32;
    }
    (out.clone(), out)
}

// ------------------------------------------------------------------- output

fn rms(left: &[f32], right: &[f32], from: usize, to: usize) -> f64 {
    let to = to.min(left.len()).min(right.len());
    if to <= from {
        return 0.0;
    }
    let sum: f64 = (from..to)
        .map(|i| f64::from(left[i]).powi(2) + f64::from(right[i]).powi(2))
        .sum();
    (sum / (2 * (to - from)) as f64).sqrt()
}

fn match_rms(audio: &Stereo) -> f64 {
    rms(
        &audio.0,
        &audio.1,
        (MATCH_LO_S * SR) as usize,
        (MATCH_HI_S * SR) as usize,
    )
}

/// Writes the listening set for one key: every signal matched to the
/// recording's level over [`MATCH_LO_S`]–[`MATCH_HI_S`], then one common gain if
/// anything would clip. One common guard and not one per file, because the
/// whole point is that the files are at the same level.
fn write_set(dir: &Path, signals: &[Signal]) -> Result<(), Box<dyn std::error::Error>> {
    let reference = match_rms(&signals[0].stereo);
    if reference.is_nan() || reference <= 0.0 {
        return Err(format!("{}: the reference is silent", dir.display()).into());
    }
    let gains: Vec<f64> = signals
        .iter()
        .map(|s| {
            let level = match_rms(&s.stereo);
            if level > 0.0 {
                reference / level
            } else {
                0.0
            }
        })
        .collect();
    let peak = signals
        .iter()
        .zip(&gains)
        .map(|(s, &g)| {
            s.stereo
                .0
                .iter()
                .chain(s.stereo.1.iter())
                .fold(0.0f64, |m, &v| m.max(f64::from(v).abs()))
                * g
        })
        .fold(0.0f64, f64::max);
    let common = if peak > 0.891 { 0.891 / peak } else { 1.0 };
    for (i, (signal, &gain)) in signals.iter().zip(&gains).enumerate() {
        let path = dir.join(format!("{i:02}_{}.wav", signal.label));
        write_wav(&path, &signal.stereo, gain * common)?;
    }
    Ok(())
}

fn write_wav(path: &Path, (left, right): &Stereo, gain: f64) -> Result<(), hound::Error> {
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let frames = (NOTE_S * SR) as usize;
    let fade_in = (FADE_IN_S * SR) as usize;
    let fade_out = (FADE_OUT_S * SR) as usize;
    let mut writer = hound::WavWriter::create(path, spec)?;
    for n in 0..frames {
        let mut envelope = gain;
        if n < fade_in {
            envelope *= 0.5 - 0.5 * (std::f64::consts::PI * n as f64 / fade_in as f64).cos();
        }
        if n + fade_out > frames {
            let u = (n + fade_out - frames) as f64 / fade_out as f64;
            envelope *= 0.5 + 0.5 * (std::f64::consts::PI * u).cos();
        }
        let at = |c: &[f32]| f64::from(c.get(n).copied().unwrap_or(0.0)) * envelope;
        writer.write_sample(at(left) as f32)?;
        writer.write_sample(at(right) as f32)?;
    }
    writer.finalize()
}


// ------------------------------------------------------------- table writing

/// The header both measurement tables share.
///
/// `cents` is the frequency deviation inside the modulation band, which is the
/// column to read first; `raw` is the same with the broadband background left
/// in, and `wRMS` is `raw` weighted by the partial's own power. `S/N` is how
/// far the partial stands over the density of whatever else is in its
/// neighbourhood, so a large `raw` next to a small `S/N` is a measurement of
/// the background rather than of the partial.
fn header_row(first: &str) -> String {
    format!(
        "| {first} | mean Hz | S/N dB | cents | p95 | exc/s | raw | wRMS | flat dB | line dB \
         | n | centroid Hz | depth dB | mod dB | flat dB | line dB | n |\n\
         |:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"
    )
}

fn stats_row(label: &str, s: &PartialStats) -> String {
    format!(
        "| {label} | {:.2} | {:.0} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.1} | {:.1} | {} \
         | {:.2} | {:.2} | {:.2} | {:.1} | {:.1} | {} |",
        s.mean_hz,
        s.peak_db,
        s.band_cents,
        s.p95_cents,
        s.excursions_per_s,
        s.raw_cents,
        s.weighted_cents,
        s.freq_mod.flatness_db,
        s.freq_mod.line_db,
        s.freq_mod.lines_to_half,
        s.freq_mod.centroid_hz,
        s.beat_depth_db,
        s.amp_mod.rms,
        s.amp_mod.flatness_db,
        s.amp_mod.line_db,
        s.amp_mod.lines_to_half,
    )
}

// ---------------------------------------------------------------------- main

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let root = PathBuf::from(args.next().unwrap_or_else(|| "data/salamander".into()));
    let preset_path = args
        .next()
        .unwrap_or_else(|| "presets/salamander-c5.toml".into());
    let out = PathBuf::from(args.next().unwrap_or_else(|| "renders/jitter".into()));

    let library = SampleLibrary::from_sfz(root.join("SalamanderGrandPiano-V3+20200602.sfz"))?;
    let base = EnginePreset::load(Path::new(&preset_path))?;
    base.validate()?;
    std::fs::create_dir_all(&out)?;

    let mut planner = FftPlanner::<f64>::new();
    let mut report = String::new();
    writeln!(
        report,
        "# JITTER.md - the instantaneous frequency of a composite partial\n\n\
         Written by `cargo run --release --example jitter_forensics`. Preset \
         `{preset_path}`, velocity {VELOCITY} unless a row says otherwise. Every number \
         is measured over {T0_S}-{T1_S} s after the strike on the mono sum, with the same \
         code on every signal including the recording, so the metric's own bias cancels \
         in the differences.\n\n\
         The demodulation is a Gaussian band-pass of time-constant {:.0} ms ({:.1} Hz \
         wide, and never wider than a quarter of the carrier) centred on the partial's own \
         spectral peak; the frequency track is its phase derivative on a {TRACK_HZ:.0} Hz \
         grid. The modulation spectra are over {MOD_LO_HZ}-{MOD_HI_HZ} Hz at a resolution \
         of {:.2} Hz, which is the {:.1} s track's own limit.\n\n\
         **Reading a row.** `mean Hz` is the partial's power-weighted mean frequency and \
         everything after it is quoted against that. `S/N dB` is the partial's own bin over \
         the median bin of its neighbourhood - a decaying sinusoid's own Lorentzian skirt \
         puts a ceiling near 44 dB on it (see the controls), so it is a floor indicator and \
         not an absolute. Then come **the frequency track in cents** - `cents` (RMS inside \
         the modulation band, the column to read first), `p95`, `exc/s` (separate swings \
         past +-{EXCURSION_CENTS} cents per second), `raw` (the same RMS with the broadband \
         background left in) and `wRMS` (`raw` weighted by the partial's own power, which \
         is what a null-crossing excursion is worth given that it happens where the partial \
         is quietest) - then **that track's modulation spectrum** (`flat`, `line`, `n` bins \
         to half the band's energy, `centroid`), and last **the log envelope**: `depth` \
         peak-to-trough dB, `mod` RMS dB, and the same three line statistics.\n",
        SMOOTH_SIGMA_S * 1000.0,
        1.0 / (TAU * SMOOTH_SIGMA_S),
        1.0 / (T1_S - T0_S),
        T1_S - T0_S,
    )?;

    // ---- the synthetic controls, first, because every other number is read
    // against them.
    writeln!(
        report,
        "\n## 0. Controls: what the measurement reads when the answer is known\n\n\
         All five are built at C4's own fundamental and decay rate and go through the \
         identical code path. The first is the floor; the middle three say how much of a \
         row can be a measurement of whatever radiates *beside* the partial (read against \
         the `S/N` column, which is measured the same way on every signal below); the last \
         is the artefact in its pure form — two free-running components of equal amplitude \
         {CONTROL_BEAT_HZ} Hz apart, which is inside `voicing.horizontal_offset_hz`'s own \
         range.\n\n\
         {}",
        header_row("control")
    )?;
    let control_params = base.string_params(60);
    let control_hz = f64::from(control_params.partial_freq(1));
    let control_sigma =
        f64::from(control_params.partial_sigma(1) * base.voicing.vertical_decay_factor());
    for (label, pair, noise) in [
        ("one decaying sinusoid", false, None),
        ("+ white noise at -60 dB", false, Some(-60.0)),
        ("+ white noise at -40 dB", false, Some(-40.0)),
        ("+ white noise at -20 dB", false, Some(-20.0)),
        ("two equal, 0.35 Hz apart", true, None),
    ] {
        let signal = Signal::new(
            "control",
            synthetic(control_hz, control_sigma, pair, noise),
        );
        let spectrum = Spectrum::new(&signal.mono, &mut planner);
        if let Some(track) = track_partial(&spectrum, control_hz, 0.35 * control_hz, &mut planner) {
            writeln!(report, "{}", stats_row(label, &statistics(&track)))?;
        }
    }

    for (key, name) in KEYS {
        let dir = out.join(name);
        std::fs::create_dir_all(&dir)?;
        let params = base.string_params(key);
        let f0 = f64::from(params.partial_freq(1));

        // ---- every signal for this key, in the order the files are numbered.
        let mut signals: Vec<Signal> = Vec::new();
        signals.push(Signal::new(
            "recording",
            recording(layer_for(&library, key, VELOCITY)?)?,
        ));
        for (label, _, edit) in VARIANTS {
            let mut preset = base.clone();
            edit(&mut preset);
            preset
                .validate()
                .map_err(|e| format!("variant {label} is not a legal preset: {e:?}"))?;
            signals.push(Signal::new(*label, render(&preset, key, VELOCITY)));
        }
        for vel in EXTRA_VELOCITIES {
            signals.push(Signal::new(
                format!("engine_vel{vel:03}"),
                render(&base, key, vel),
            ));
            signals.push(Signal::new(
                format!("recording_vel{vel:03}"),
                recording(layer_for(&library, key, vel)?)?,
            ));
        }
        write_set(&dir, &signals)?;

        // ---- what the preset predicts, before anything is measured.
        writeln!(
            report,
            "\n## {name} (key {key}, {} string{}, detune {:.3} cents)\n",
            params.unison,
            if params.unison == 1 { "" } else { "s" },
            params.detune_cents,
        )?;
        writeln!(
            report,
            "**What the shipped preset builds.** Every partial is this many independent \
             sinusoids, at these fixed offsets, with these decay rates; `equal at` is when \
             the two loudest of them pass through equal amplitude, which is where the \
             hypothesis says the frequency track swings hardest. `under dB` is how far \
             that meeting pair sits below whatever is loudest at that instant: 0 dB is a \
             full null.\n\n\
             | k | components | beat rates Hz | equal at s | pair | pair Hz | under dB |\n\
             |--:|--:|:--|--:|:--|--:|--:|"
        )?;
        for k in 1..=MAX_PARTIAL {
            let comps = components(&base, key, k);
            let mut beats: Vec<f64> = Vec::new();
            for (i, a) in comps.iter().enumerate() {
                for b in &comps[i + 1..] {
                    let d = (a.hz - b.hz).abs();
                    if d > 1e-4 {
                        beats.push(d);
                    }
                }
            }
            beats.sort_by(|a, b| a.partial_cmp(b).unwrap());
            beats.dedup_by(|a, b| (*a - *b).abs() < 1e-3);
            let list = beats
                .iter()
                .map(|b| format!("{b:.3}"))
                .collect::<Vec<_>>()
                .join(", ");
            let meeting = equality_time(&comps)
                .map(|(t, d, u, who)| format!("{t:.2} | {who} | {d:.3} | {u:.1}"))
                .unwrap_or_else(|| "- | - | - | -".into());
            writeln!(report, "| {k} | {} | {list} | {meeting} |", comps.len())?;
        }

        // ---- the measurement. One transform per signal, every partial of it
        // read out of the same transform.
        let measured: Vec<Vec<Option<PartialStats>>> = signals
            .iter()
            .map(|signal| {
                let spectrum = Spectrum::new(&signal.mono, &mut planner);
                (1..=MAX_PARTIAL)
                    .map(|k| {
                        let nominal = f64::from(params.partial_freq(k));
                        track_partial(&spectrum, nominal, 0.35 * f0, &mut planner)
                            .map(|t| statistics(&t))
                    })
                    .collect()
            })
            .collect();

        // The two summaries the attribution is read off: one column per
        // partial, one row per signal.
        for (title, note, get) in [
            (
                "Frequency jitter",
                "RMS of the instantaneous-frequency deviation inside 0.1-20 Hz, in cents. The recording's row is the pass bar.",
                (|s: &PartialStats| s.band_cents) as fn(&PartialStats) -> f64,
            ),
            (
                "Frequency-track flatness",
                "spectral flatness of the track's modulation spectrum, dB. Near 0 dB is a continuum, -20 dB and below is one or two discrete lines. A caveat that matters below: a *deep* beat spikes the frequency track at every null, and a train of spikes is broadband however regular it is, so a flat row with a large beat depth beside it is a deep periodic beat and not a continuum.",
                |s: &PartialStats| s.freq_mod.flatness_db,
            ),
            (
                "Beat depth",
                "peak-to-trough span of the log envelope inside the same band, dB.",
                |s: &PartialStats| s.beat_depth_db,
            ),
            (
                "Envelope flatness",
                "the line-versus-continuum question asked of the amplitude, dB.",
                |s: &PartialStats| s.amp_mod.flatness_db,
            ),
            (
                "Where the jitter sits (wRMS / raw)",
                "the power-weighted deviation over the plain one - *where* in the note the \
                 wobble happens. A wobble that rides on the partial while it is loud gives \
                 about 1; a wobble that is a spike at the null of a beat, where the partial \
                 has almost no amplitude left, gives a small fraction. Read it only where \
                 the jitter is large enough to be about something: the `all_off` render has \
                 0.00-0.01 cents of it and its ratio is a ratio of two noise floors.",
                |s: &PartialStats| {
                    if s.raw_cents > 0.0 {
                        s.weighted_cents / s.raw_cents
                    } else {
                        0.0
                    }
                },
            ),
        ] {
            writeln!(
                report,
                "\n**{name} - {title}.** {note}\n\n| signal | k=1 | k=2 | k=3 | k=4 |\n\
                 |:--|--:|--:|--:|--:|"
            )?;
            for (signal, row) in signals.iter().zip(&measured) {
                let cells: Vec<String> = row
                    .iter()
                    .map(|s| s.as_ref().map_or("-".into(), |s| format!("{:.2}", get(s))))
                    .collect();
                writeln!(report, "| {} | {} |", signal.label, cells.join(" | "))?;
            }
        }

        // ---- and the full reading, partial by partial.
        for k in 1..=MAX_PARTIAL {
            let nominal = f64::from(params.partial_freq(k));
            writeln!(
                report,
                "\n### {name} partial {k} (nominal {nominal:.2} Hz)\n\n{}",
                header_row("signal")
            )?;
            for (signal, row) in signals.iter().zip(&measured) {
                match &row[k - 1] {
                    Some(s) => writeln!(report, "{}", stats_row(&signal.label, s))?,
                    None => writeln!(
                        report,
                        "| {} | not present over its own background | | | | | | | | | | | | | | | |",
                        signal.label
                    )?,
                }
            }
        }
        println!("{name}: {} signals written to {}", signals.len(), dir.display());
    }

    writeln!(
        report,
        "\n## The listening set\n\n\
         `{}/<note>/NN_<label>.wav`: {NOTE_S} s, stereo, every file matched to the \
         recording's RMS over {MATCH_LO_S}-{MATCH_HI_S} s with one common headroom gain \
         over the whole set, so what differs between two files is not their level. The \
         bisection labels are:\n",
        out.display()
    )?;
    for (label, what, _) in VARIANTS {
        writeln!(report, "- `{label}` - {what}")?;
    }
    writeln!(
        report,
        "\nThe rungs are deliberately **not** orthogonal, and reading them as if they were \
         is the one mistake this table invites. `notes.detune_cents` offsets whole strings \
         by a *ratio* and `voicing.horizontal_offset_hz` offsets each string's horizontal \
         polarization by a fixed number of *hertz*; zeroing either one leaves the other's \
         beats standing, and can make the remaining ones deeper by removing what used to \
         fill their nulls. The two rungs that bracket the whole mechanism are \
         `single_string`, which leaves exactly one pair of components beating at one \
         `horizontal_offset_hz`, and `all_off`, which leaves one sinusoid per partial and \
         is therefore this measurement's own floor."
    )?;
    writeln!(
        report,
        "- `engine_velNNN` / `recording_velNNN` - the shipped preset and the library layer \
         at velocities {:?}, which is the amplitude-equality prediction: if the percept is \
         two components passing through each other, it has to move with how hard the note \
         is struck.",
        EXTRA_VELOCITIES
    )?;

    std::fs::write(out.join("JITTER.md"), &report)?;
    println!("wrote {}", out.join("JITTER.md").display());
    Ok(())
}
