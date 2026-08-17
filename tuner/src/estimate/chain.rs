//! The recording chain: the part of engine-vs-reference that is a microphone
//! and a room rather than a piano.
//!
//! Every number in `renders/realism/REALISM.md` compares a **near-anechoic**
//! render against a **recording**. The engine's output is the soundboard's own
//! diffuse field (`DECISIONS.md` 19: `T60 ~ 0.4 s` falling to 0.1 s at 8 kHz,
//! and it lives *inside* the board path) plus a pan-pot; the Salamander
//! recordings carry a microphone pair, its placement, the lid, the room those
//! sessions were in, and whatever mastering the release had. `PHYSICS.md` §8
//! and §9 say that is two stages the instrument does not have, and `TUNING.md`
//! stage 2 puts a placeholder for the first of them — "a static linear filter
//! applied to the engine output before loss, a ~40-band cepstrally-smooth
//! log-magnitude EQ (optionally + one short early-reflection IR later)".
//!
//! This module is that absorber, built and **measured** rather than optimised
//! into: the point is not to make the engine score better, it is to find out
//! how much of the standing gap a static chain can absorb at all, because that
//! is the difference between "the instrument is wrong" and "the comparison is".
//!
//! # The two halves, and what each can be
//!
//! **(a) The smooth spectral transfer** ([`fit_eq`], [`ChainEq`]). A microphone,
//! a lid and a room colour a spectrum smoothly in `ln f`; a partial series does
//! not. [`EQ_BANDS`] bands log-spaced over [`EQ_F_MIN`]..[`EQ_F_MAX`], each
//! band's gain the **median** over many matched note pairs of what the
//! recording has there that the render does not, then [`cepstral_smooth`]ed to
//! [`CEPSTRAL_ORDER`] coefficients so the curve cannot chase one key's partials.
//! Forty free numbers against a spectrum is a lot of freedom; twelve cepstral
//! coefficients that must hold across every key and velocity at once is not.
//!
//! **(b) The spatial and temporal part** ([`StereoSignature`], [`EnergyDecay`],
//! [`RoomStage`]). A magnitude curve is blind to everything that makes a
//! recording sound like a place: the interchannel decorrelation of a mic pair,
//! the discrete early reflections, the tail that outlives the note. Those are
//! measured here off the material that has them — the reference's own two
//! channels, and the library's *mechanism* recordings, which are the only
//! impulsive events in it — and turned into a minimal matching stage: a few
//! delayed, filtered reflections plus a decorrelated exponential-noise tail,
//! every parameter a reading rather than a taste.
//!
//! # What is knowably NOT identifiable from this material
//!
//! Listed here because the honest half of this experiment is the half that
//! says what the numbers cannot mean.
//!
//! 1. **Chain colour against instrument error.** A smooth static EQ fitted on
//!    engine-vs-reference band ratios cannot tell a microphone's response from
//!    a systematic error in the excitation model: both are smooth in `ln f` and
//!    both are constant across the compass. Nothing in this material breaks
//!    that degeneracy. What *can* be measured is **consistency** — fit the
//!    curve on one half of the keys and on the other half independently and see
//!    whether the two agree ([`curve_agreement`]) — and consistency bounds how
//!    much of the gap is *shaped like* a static chain. It is not causation and
//!    is never reported as such.
//! 2. **Absolute pre-delay.** Every sample in the library is trimmed to its own
//!    onset (`crate::sampler` re-detects it), so the direct sound's flight time
//!    to the microphone is gone. Reflection delays are measurable only
//!    *relative* to the direct arrival; mic distance is not recoverable.
//! 3. **The chain's phase.** Two signals that are not the same excitation give
//!    magnitude ratios and nothing else. Whether the real chain is
//!    minimum-phase (a microphone and a room mostly are) or not cannot be read
//!    off this material, so [`ChainEq`] is **linear-phase by choice**, stated
//!    rather than fitted, and its group delay is compensated exactly.
//! 4. **Room tail against string and board decay.** The late energy of a struck
//!    note is the string, the board's diffuse field, the sympathetic halo and
//!    the room, superposed. Nothing separates them on a note. The mechanism
//!    recordings are used instead precisely because they have no string in
//!    them — but they have their own mechanical duration, so what [`schroeder_db`]
//!    reads off them is an **upper bound** on the room's decay, not the room's
//!    decay.
//! 5. **Mic geometry against the instrument's own spatial extent.** Interchannel
//!    correlation versus lag mixes microphone spacing, the bass and treble
//!    bridges being metres apart, and the room's early field. One instrument in
//!    one session cannot separate three causes of one number.
//! 6. **Chain nonlinearity.** A compressor or tape in the chain would make the
//!    fitted curve depend on level. That is *testable* here and is tested —
//!    fit the curve on the soft layers and on the loud layers and compare — but
//!    a null result bounds it rather than excluding it.
//! 7. **What the reference's room does between notes.** The reference is a
//!    *sampler*: each note carries its own recording's room, and Salamander's
//!    note-off applies a release fade that truncates that room tail with the
//!    note. A room stage convolved over a whole engine phrase does not truncate.
//!    The two are equal while notes ring and differ after every damper, and the
//!    difference is the reference's, not the model's.

use crate::audio::Audio;
use crate::error::{Error, Result};
use crate::estimate::texture::Draw;

use rustfft::{num_complex::Complex32, FftPlanner};

// ---------------------------------------------------------------------------
// The band layout
// ---------------------------------------------------------------------------

/// Bands in the log-frequency EQ. `TUNING.md`'s "~40-band" absorber, taken
/// literally: forty bands over the nine octaves below carry 0.215 octave each,
/// about two and a half semitones, which is the width of a mel band up where
/// the argument is and finer than any microphone or room feature.
pub const EQ_BANDS: usize = 40;

/// Bottom of the fitted range. Below it the lowest key's fundamental is the
/// only thing in the band and the recording's own rumble is comparable to it.
pub const EQ_F_MIN: f64 = 40.0;

/// Top of the fitted range: `realism::MEL_F_MAX`, because above it the metric
/// this chain is measured against does not look.
pub const EQ_F_MAX: f64 = 16_000.0;

/// Cepstral coefficients kept by [`cepstral_smooth`].
///
/// Twelve over forty bands means the fastest ripple the curve can carry has a
/// period of about one octave, which is a room mode group or a microphone's
/// presence peak and is *not* a partial. This is the whole statistical
/// discipline of half (a): the EQ is allowed to be a chain and is not allowed
/// to be a spectrum.
pub const CEPSTRAL_ORDER: usize = 12;

/// How far under a note's own loudest band a band may sit and still be read.
///
/// Under it the band is the recording's floor on one side and the render's
/// board field on the other, and their ratio is a ratio of two floors.
pub const BAND_FLOOR_DB: f64 = -70.0;

/// Fewest matched notes a band needs before its median is a reading.
pub const MIN_BAND_ITEMS: usize = 8;

/// Taps in the linear-phase FIR [`ChainEq::impulse_response`] designs.
///
/// 4096 at 48 kHz is 85 ms and an 11.7 Hz design grid — finer than the lowest
/// band is wide. The group delay is exactly half of it and is removed.
pub const EQ_TAPS: usize = 4096;

/// Band edges, log-spaced: `EQ_BANDS + 1` of them.
pub fn band_edges() -> Vec<f64> {
    let ratio = (EQ_F_MAX / EQ_F_MIN).powf(1.0 / EQ_BANDS as f64);
    (0..=EQ_BANDS)
        .map(|i| EQ_F_MIN * ratio.powi(i as i32))
        .collect()
}

/// Geometric centre of each band.
pub fn band_centres() -> Vec<f64> {
    let edges = band_edges();
    (0..EQ_BANDS)
        .map(|i| (edges[i] * edges[i + 1]).sqrt())
        .collect()
}

/// Power per band of a bin-power spectrum (DC first, non-negative half).
pub fn band_powers(power: &[f64], sample_rate: f64) -> Vec<f64> {
    let edges = band_edges();
    (0..EQ_BANDS)
        .map(|i| crate::estimate::brilliance::band(power, sample_rate, (edges[i], edges[i + 1])))
        .collect()
}

// ---------------------------------------------------------------------------
// (a) The smooth spectral transfer
// ---------------------------------------------------------------------------

/// One matched pair's band powers: the same note, played twice.
#[derive(Clone, Debug)]
pub struct EqSample {
    /// Band powers of the engine's render.
    pub engine: Vec<f64>,
    /// Band powers of the recording.
    pub reference: Vec<f64>,
    /// What this pair is, for reporting: key and velocity.
    pub key: u8,
    pub velocity: u8,
}

/// What [`fit_eq`] found.
#[derive(Clone, Debug)]
pub struct EqFit {
    /// Per-band median of `reference − engine` in dB, after each pair's own
    /// broadband level is divided out. The reading, before smoothing.
    pub raw_db: Vec<f64>,
    /// [`cepstral_smooth`]ed to [`CEPSTRAL_ORDER`], mean removed. The curve.
    pub smooth_db: Vec<f64>,
    /// Pairs that contributed to each band.
    pub counts: Vec<usize>,
    /// Median absolute deviation of each band's readings, dB: how much of the
    /// band's number is a chain and how much is one key's partials.
    pub scatter_db: Vec<f64>,
    /// Lowest and highest band that [`MIN_BAND_ITEMS`] pairs could read.
    /// Outside them the curve is **held flat**: see [`fit_eq`].
    pub read_range: (usize, usize),
    /// Pairs offered.
    pub items: usize,
}

impl EqFit {
    /// The curve as a filter.
    pub fn eq(&self) -> ChainEq {
        ChainEq {
            gains_db: self.smooth_db.clone(),
        }
    }
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        0.5 * (values[n / 2 - 1] + values[n / 2])
    }
}

/// Fit the chain EQ from matched pairs.
///
/// Each pair is level-matched on the **sum of its own bands** first, so the
/// curve carries no part of `OUTPUT_GAIN` and no part of the 13-22 dB
/// engine-against-recording level error `COMPASS.md` reports. Each band's
/// reading is then the recording's excess there in dB, and each band's number
/// is the **median** over pairs: a mean would let one key whose 3 kHz band is
/// empty in the render and full in the recording write the chain.
pub fn fit_eq(items: &[EqSample]) -> EqFit {
    let mut per_band: Vec<Vec<f64>> = vec![Vec::new(); EQ_BANDS];
    let floor = 10f64.powf(BAND_FLOOR_DB / 10.0);
    for item in items {
        let total_e: f64 = item.engine.iter().sum();
        let total_r: f64 = item.reference.iter().sum();
        if total_e <= 0.0 || total_r <= 0.0 {
            continue;
        }
        let scale = total_r / total_e;
        let peak_r = item.reference.iter().cloned().fold(0.0f64, f64::max);
        let peak_e = item.engine.iter().cloned().fold(0.0f64, f64::max) * scale;
        for (readings, (&engine, &reference)) in per_band
            .iter_mut()
            .zip(item.engine.iter().zip(item.reference.iter()))
        {
            let e = engine * scale;
            if e <= peak_e * floor || reference <= peak_r * floor {
                continue;
            }
            readings.push(10.0 * (reference / e).log10());
        }
    }

    let mut raw_db = vec![0.0; EQ_BANDS];
    let mut scatter_db = vec![0.0; EQ_BANDS];
    let mut counts = vec![0usize; EQ_BANDS];
    for (b, readings) in per_band.iter_mut().enumerate() {
        counts[b] = readings.len();
        if counts[b] < MIN_BAND_ITEMS {
            raw_db[b] = f64::NAN;
            scatter_db[b] = f64::NAN;
            continue;
        }
        let m = median(readings);
        raw_db[b] = m;
        let mut dev: Vec<f64> = readings.iter().map(|v| (v - m).abs()).collect();
        scatter_db[b] = median(&mut dev);
    }

    // A band nobody could read is filled from its neighbours before smoothing:
    // a hole in the middle of a DCT is a ripple everywhere, and the honest
    // statement about an unread band is "whatever its neighbours say", not zero.
    let filled = fill_holes(&raw_db);
    let mut smooth_db = cepstral_smooth(&filled, CEPSTRAL_ORDER);

    // Outside the range the material could read, the curve is held flat rather
    // than continued. A truncated DCT overshoots at an edge, and the top bands
    // are exactly where a note has least energy: three readings at 15 kHz were
    // continued into a −19 dB shelf that nothing measured. Flat is the only
    // thing an unread band licenses.
    let first = (0..EQ_BANDS)
        .find(|&b| counts[b] >= MIN_BAND_ITEMS)
        .unwrap_or(0);
    let last = (0..EQ_BANDS)
        .rev()
        .find(|&b| counts[b] >= MIN_BAND_ITEMS)
        .unwrap_or(EQ_BANDS - 1);
    for b in 0..first {
        smooth_db[b] = smooth_db[first];
    }
    for b in last + 1..EQ_BANDS {
        smooth_db[b] = smooth_db[last];
    }

    let mean = smooth_db.iter().sum::<f64>() / EQ_BANDS as f64;
    for g in smooth_db.iter_mut() {
        *g -= mean;
    }

    EqFit {
        raw_db,
        smooth_db,
        counts,
        scatter_db,
        read_range: (first, last),
        items: items.len(),
    }
}

/// Replace non-finite entries by linear interpolation between the nearest
/// finite ones, holding the end values flat.
pub fn fill_holes(curve: &[f64]) -> Vec<f64> {
    let n = curve.len();
    let known: Vec<usize> = (0..n).filter(|&i| curve[i].is_finite()).collect();
    if known.is_empty() {
        return vec![0.0; n];
    }
    (0..n)
        .map(|i| {
            if curve[i].is_finite() {
                return curve[i];
            }
            let before = known.iter().rev().find(|&&j| j < i).copied();
            let after = known.iter().find(|&&j| j > i).copied();
            match (before, after) {
                (Some(a), Some(b)) => {
                    let t = (i - a) as f64 / (b - a) as f64;
                    curve[a] * (1.0 - t) + curve[b] * t
                }
                (Some(a), None) => curve[a],
                (None, Some(b)) => curve[b],
                (None, None) => 0.0,
            }
        })
        .collect()
}

/// Keep the lowest `order` DCT-II coefficients of a curve sampled on the band
/// grid and rebuild it — the cepstral smoothing `TUNING.md` asks for, done on
/// the log-frequency axis where a chain is smooth rather than on the linear one
/// where it is not.
pub fn cepstral_smooth(curve: &[f64], order: usize) -> Vec<f64> {
    let n = curve.len();
    if n == 0 {
        return Vec::new();
    }
    let order = order.min(n).max(1);
    let scale = std::f64::consts::PI / n as f64;
    let coeff: Vec<f64> = (0..order)
        .map(|k| {
            (2.0 / n as f64)
                * curve
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| v * (scale * (i as f64 + 0.5) * k as f64).cos())
                    .sum::<f64>()
        })
        .collect();
    (0..n)
        .map(|i| {
            coeff[0] / 2.0
                + (1..order)
                    .map(|k| coeff[k] * (scale * (i as f64 + 0.5) * k as f64).cos())
                    .sum::<f64>()
        })
        .collect()
}

/// How far apart two independently fitted curves are, in dB: the mean absolute
/// difference and the Pearson correlation of the two shapes.
///
/// This is the only evidence in this module about whether a *global* chain
/// exists. Two halves of the compass have nothing in common but the microphone
/// and the room; if their curves agree, a static chain of that shape is what
/// they share, and if they do not, what was fitted was each half's own keys.
pub fn curve_agreement(a: &[f64], b: &[f64]) -> (f64, f64) {
    let n = a.len().min(b.len());
    if n == 0 {
        return (f64::NAN, f64::NAN);
    }
    let mean_abs = (0..n).map(|i| (a[i] - b[i]).abs()).sum::<f64>() / n as f64;
    let ma = a[..n].iter().sum::<f64>() / n as f64;
    let mb = b[..n].iter().sum::<f64>() / n as f64;
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let (x, y) = (a[i] - ma, b[i] - mb);
        sxy += x * y;
        sxx += x * x;
        syy += y * y;
    }
    let r = if sxx > 0.0 && syy > 0.0 {
        sxy / (sxx * syy).sqrt()
    } else {
        f64::NAN
    };
    (mean_abs, r)
}

/// The fitted chain EQ: one gain per [`band_centres`] entry, applied as a
/// linear-phase FIR.
#[derive(Clone, Debug)]
pub struct ChainEq {
    pub gains_db: Vec<f64>,
}

impl ChainEq {
    /// A flat chain: the null the collapse table is measured against.
    pub fn flat() -> ChainEq {
        ChainEq {
            gains_db: vec![0.0; EQ_BANDS],
        }
    }

    pub fn from_db(gains_db: Vec<f64>) -> Result<ChainEq> {
        if gains_db.len() != EQ_BANDS {
            return Err(Error::Config(format!(
                "a chain EQ has {EQ_BANDS} bands, not {}",
                gains_db.len()
            )));
        }
        Ok(ChainEq { gains_db })
    }

    /// The curve's gain at an arbitrary frequency: linear in dB against `ln f`
    /// between band centres, held flat outside the fitted range.
    pub fn gain_db_at(&self, hz: f64) -> f64 {
        let centres = band_centres();
        if self.gains_db.is_empty() {
            return 0.0;
        }
        if hz <= centres[0] {
            return self.gains_db[0];
        }
        if hz >= centres[centres.len() - 1] {
            return self.gains_db[self.gains_db.len() - 1];
        }
        let i = centres.partition_point(|&c| c <= hz) - 1;
        let t = (hz.ln() - centres[i].ln()) / (centres[i + 1].ln() - centres[i].ln());
        self.gains_db[i] * (1.0 - t) + self.gains_db[i + 1] * t
    }

    /// A symmetric, windowed, linear-phase FIR realising the curve. Its group
    /// delay is `taps / 2` samples, which [`ChainEq::apply`] removes.
    pub fn impulse_response(&self, taps: usize, sample_rate: f64) -> Vec<f32> {
        let n = taps.max(16) & !1;
        let mut planner = FftPlanner::<f32>::new();
        let inverse = planner.plan_fft_inverse(n);
        let mut buffer = vec![Complex32::new(0.0, 0.0); n];
        for j in 0..=n / 2 {
            let hz = j as f64 * sample_rate / n as f64;
            let mag = 10f64.powf(self.gain_db_at(hz) / 20.0) as f32;
            buffer[j] = Complex32::new(mag, 0.0);
            if j > 0 && j < n / 2 {
                buffer[n - j] = Complex32::new(mag, 0.0);
            }
        }
        inverse.process(&mut buffer);
        // Circular shift to centre the (symmetric) response, then window it.
        let mut h = vec![0.0f32; n];
        for (i, tap) in h.iter_mut().enumerate() {
            let from = (i + n / 2) % n;
            let w = 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / n as f64).cos();
            *tap = buffer[from].re / n as f32 * w as f32;
        }
        h
    }

    /// Filter a signal, keeping its length and its alignment.
    pub fn apply_mono(&self, x: &[f32], sample_rate: f64) -> Vec<f32> {
        let h = self.impulse_response(EQ_TAPS, sample_rate);
        let full = convolve(x, &h);
        let delay = h.len() / 2;
        (0..x.len())
            .map(|i| full.get(i + delay).copied().unwrap_or(0.0))
            .collect()
    }

    /// Filter every channel of a render.
    pub fn apply(&self, audio: &Audio) -> Audio {
        let sr = f64::from(audio.sample_rate);
        Audio {
            sample_rate: audio.sample_rate,
            channels: audio
                .channels
                .iter()
                .map(|c| self.apply_mono(c, sr))
                .collect(),
        }
    }
}

/// Linear convolution by FFT. `x.len() + h.len() - 1` samples out.
pub fn convolve(x: &[f32], h: &[f32]) -> Vec<f32> {
    if x.is_empty() || h.is_empty() {
        return Vec::new();
    }
    let n = (x.len() + h.len() - 1).next_power_of_two();
    let mut planner = FftPlanner::<f32>::new();
    let forward = planner.plan_fft_forward(n);
    let inverse = planner.plan_fft_inverse(n);
    let mut a: Vec<Complex32> = (0..n)
        .map(|i| Complex32::new(x.get(i).copied().unwrap_or(0.0), 0.0))
        .collect();
    let mut b: Vec<Complex32> = (0..n)
        .map(|i| Complex32::new(h.get(i).copied().unwrap_or(0.0), 0.0))
        .collect();
    forward.process(&mut a);
    forward.process(&mut b);
    for (x, y) in a.iter_mut().zip(b.iter()) {
        *x *= *y;
    }
    inverse.process(&mut a);
    let scale = 1.0 / n as f32;
    a.into_iter()
        .take(x.len() + h.len() - 1)
        .map(|c| c.re * scale)
        .collect()
}

// ---------------------------------------------------------------------------
// (b) The spatial and temporal signature
// ---------------------------------------------------------------------------

/// Widest lag the interchannel correlation is searched over. Five milliseconds
/// is 1.7 m of air: wider than any mic pair and wider than the first reflection
/// off a lid, and narrow enough that a piano partial's own period does not wrap
/// the search into a periodicity measurement above 200 Hz.
pub const MAX_LAG_S: f64 = 0.005;

/// The octave bands the stereo signature and the tail decay are read in.
pub const SPATIAL_BANDS: [(f64, f64); 6] = [
    (63.0, 125.0),
    (125.0, 250.0),
    (250.0, 500.0),
    (500.0, 2000.0),
    (2000.0, 6000.0),
    (6000.0, 12000.0),
];

/// One band's interchannel behaviour.
#[derive(Clone, Copy, Debug)]
pub struct BandCoherence {
    pub lo_hz: f64,
    pub hi_hz: f64,
    /// Largest normalised cross-correlation over the searched lags.
    pub peak_r: f64,
    /// Where it happened, in milliseconds; positive means the **right** channel
    /// leads, because the correlation is `Σ_t L[t+τ]·R[t]` and a delayed right
    /// channel peaks at negative `τ`. Pinned by
    /// `a_pan_pot_correlates_at_one_and_a_delayed_pair_does_not`, which reads
    /// −2.00 ms for a right channel delayed by 2 ms.
    pub lag_ms: f64,
    /// The value at lag zero, which is what a pan-pot gives.
    pub zero_r: f64,
}

/// How two channels of one recording relate: the measurement a pan-pot fails.
#[derive(Clone, Debug)]
pub struct StereoSignature {
    pub broadband: BandCoherence,
    pub per_band: Vec<BandCoherence>,
}

fn spectrum(signal: &[f32], n: usize, planner: &mut FftPlanner<f32>) -> Vec<Complex32> {
    let forward = planner.plan_fft_forward(n);
    let mut buffer: Vec<Complex32> = (0..n)
        .map(|i| Complex32::new(signal.get(i).copied().unwrap_or(0.0), 0.0))
        .collect();
    forward.process(&mut buffer);
    buffer
}

fn masked(spec: &[Complex32], sample_rate: f64, band: Option<(f64, f64)>) -> Vec<Complex32> {
    let n = spec.len();
    let Some((lo, hi)) = band else {
        return spec.to_vec();
    };
    let bin = |hz: f64| (hz * n as f64 / sample_rate).round() as usize;
    let (blo, bhi) = (bin(lo).max(1), bin(hi).min(n / 2));
    let mut out = vec![Complex32::new(0.0, 0.0); n];
    for j in blo..=bhi {
        out[j] = spec[j];
        out[n - j] = spec[n - j];
    }
    out
}

fn coherence(
    left: &[Complex32],
    right: &[Complex32],
    sample_rate: f64,
    band: Option<(f64, f64)>,
    planner: &mut FftPlanner<f32>,
) -> BandCoherence {
    let n = left.len();
    let (a, b) = (
        masked(left, sample_rate, band),
        masked(right, sample_rate, band),
    );
    // Parseval gives each side's energy without a second inverse transform.
    let ea: f64 = a.iter().map(|c| f64::from(c.norm_sqr())).sum::<f64>() / n as f64;
    let eb: f64 = b.iter().map(|c| f64::from(c.norm_sqr())).sum::<f64>() / n as f64;
    let mut cross: Vec<Complex32> = a.iter().zip(b.iter()).map(|(x, y)| *x * y.conj()).collect();
    let inverse = planner.plan_fft_inverse(n);
    inverse.process(&mut cross);
    let norm = (ea * eb).sqrt().max(1e-30);
    let max_lag = ((MAX_LAG_S * sample_rate).round() as usize)
        .min(n / 4)
        .max(1);
    let value = |lag: isize| -> f64 {
        let idx = if lag >= 0 {
            lag as usize
        } else {
            n - (-lag) as usize
        };
        f64::from(cross[idx].re) / n as f64 / norm
    };
    let zero_r = value(0);
    let mut best = (0isize, f64::NEG_INFINITY);
    for lag in -(max_lag as isize)..=(max_lag as isize) {
        let v = value(lag).abs();
        if v > best.1 {
            best = (lag, v);
        }
    }
    let (lo, hi) = band.unwrap_or((0.0, sample_rate / 2.0));
    BandCoherence {
        lo_hz: lo,
        hi_hz: hi,
        peak_r: value(best.0),
        lag_ms: best.0 as f64 / sample_rate * 1000.0,
        zero_r,
    }
}

/// Interchannel correlation against lag, broadband and per octave band.
///
/// The engine's own answer to this is 1.0 at lag 0 in every band by
/// construction — `soundboard::pan_for_key` scales one signal into two — so any
/// departure the recording shows is a property of the chain and of nothing the
/// preset can carry.
pub fn stereo_signature(left: &[f32], right: &[f32], sample_rate: f64) -> Result<StereoSignature> {
    if left.is_empty() || right.is_empty() {
        return Err(Error::Config(
            "a stereo signature needs two channels".into(),
        ));
    }
    let n = (left.len().max(right.len()) * 2).next_power_of_two();
    let mut planner = FftPlanner::<f32>::new();
    let a = spectrum(left, n, &mut planner);
    let b = spectrum(right, n, &mut planner);
    let broadband = coherence(&a, &b, sample_rate, None, &mut planner);
    let per_band = SPATIAL_BANDS
        .iter()
        .map(|&band| coherence(&a, &b, sample_rate, Some(band), &mut planner))
        .collect();
    Ok(StereoSignature {
        broadband,
        per_band,
    })
}

/// A Schroeder backward integration in dB, normalised to 0 at the start.
///
/// Truncated where the signal reaches [`SCHROEDER_MARGIN_DB`] over its own
/// noise floor, because integrating a floor turns it into a straight line with
/// a slope, which is the classic way to report a room that is not there.
pub fn schroeder_db(mono: &[f32]) -> Vec<f64> {
    let n = mono.len();
    if n == 0 {
        return Vec::new();
    }
    let power: Vec<f64> = mono.iter().map(|&x| f64::from(x) * f64::from(x)).collect();
    // The floor: the quietest tenth of the signal, taken from its tail.
    let tail_from = n * 9 / 10;
    let floor = if tail_from < n {
        power[tail_from..].iter().sum::<f64>() / (n - tail_from) as f64
    } else {
        0.0
    };
    let cut_at = 10f64.powf(SCHROEDER_MARGIN_DB / 10.0) * floor;
    // Where a 1 ms moving average last stands over the cut.
    let mut end = n;
    let window = (n / 200).max(16);
    let mut running: f64 = power[..window.min(n)].iter().sum();
    let mut last_over = 0usize;
    for i in window..n {
        running += power[i] - power[i - window];
        if running / window as f64 > cut_at {
            last_over = i;
        }
    }
    if last_over > window {
        end = last_over;
    }
    let mut cumulative = vec![0.0f64; end + 1];
    for i in (0..end).rev() {
        cumulative[i] = cumulative[i + 1] + power[i];
    }
    let total = cumulative[0].max(1e-30);
    cumulative[..end]
        .iter()
        .map(|&c| 10.0 * (c / total).max(1e-30).log10())
        .collect()
}

/// How far over its own floor a signal must stand for [`schroeder_db`] to keep
/// integrating it.
pub const SCHROEDER_MARGIN_DB: f64 = 10.0;

/// Decay time extrapolated to 60 dB from the slope between two levels on a
/// Schroeder curve. `None` when the curve never reaches `to_db`.
pub fn decay_time(curve_db: &[f64], sample_rate: f64, from_db: f64, to_db: f64) -> Option<f64> {
    let first = curve_db.iter().position(|&v| v <= from_db)?;
    let last = curve_db.iter().position(|&v| v <= to_db)?;
    if last <= first {
        return None;
    }
    let span = (last - first) as f64 / sample_rate;
    Some(span * 60.0 / (from_db - to_db))
}

/// What one impulsive recording says about the space it was made in.
#[derive(Clone, Debug)]
pub struct EnergyDecay {
    /// Early decay time: the 0 → −10 dB slope extrapolated to 60 dB.
    pub edt_s: Option<f64>,
    /// The −5 → −25 dB slope extrapolated to 60 dB, the usual T20.
    pub t20_s: Option<f64>,
    /// The same per [`SPATIAL_BANDS`] band.
    pub per_band: Vec<(f64, f64, Option<f64>)>,
}

/// Read a decay off one impulsive recording, broadband and per band.
pub fn energy_decay(mono: &[f32], sample_rate: f64) -> EnergyDecay {
    let broad = schroeder_db(mono);
    let mut planner = FftPlanner::<f32>::new();
    let n = (mono.len() * 2).next_power_of_two();
    let spec = spectrum(mono, n, &mut planner);
    let inverse = planner.plan_fft_inverse(n);
    let per_band = SPATIAL_BANDS
        .iter()
        .map(|&(lo, hi)| {
            let mut band = masked(&spec, sample_rate, Some((lo, hi)));
            inverse.process(&mut band);
            let filtered: Vec<f32> = band
                .iter()
                .take(mono.len())
                .map(|c| c.re / n as f32)
                .collect();
            let curve = schroeder_db(&filtered);
            (lo, hi, decay_time(&curve, sample_rate, -5.0, -25.0))
        })
        .collect();
    EnergyDecay {
        edt_s: decay_time(&broad, sample_rate, 0.0, -10.0),
        t20_s: decay_time(&broad, sample_rate, -5.0, -25.0),
        per_band,
    }
}

/// A discrete arrival after the direct sound.
#[derive(Clone, Copy, Debug)]
pub struct Reflection {
    pub delay_s: f64,
    pub gain_db: f64,
    /// −1 fully left, +1 fully right. Reflections arrive from somewhere.
    pub side: f64,
}

/// Where the envelope of an impulsive recording rises again after the direct
/// sound: candidate early reflections, as `(delay s, level dB under direct)`.
///
/// A *candidate* and not a reflection: see the module's identifiability note 4.
/// A key-off recording is a mechanical event with its own duration, so a rise
/// 12 ms after the first one may be the damper landing rather than a wall.
pub fn reflection_candidates(
    mono: &[f32],
    sample_rate: f64,
    from_s: f64,
    to_s: f64,
    margin_db: f64,
) -> Vec<(f64, f64)> {
    let smooth = (0.0005 * sample_rate).round().max(4.0) as usize;
    let env: Vec<f64> = {
        let mut out = vec![0.0; mono.len()];
        let mut running = 0.0;
        for i in 0..mono.len() {
            running += f64::from(mono[i]) * f64::from(mono[i]);
            if i >= smooth {
                running -= f64::from(mono[i - smooth]) * f64::from(mono[i - smooth]);
            }
            out[i] = (running / smooth as f64).sqrt();
        }
        out
    };
    let direct = env.iter().cloned().fold(0.0f64, f64::max).max(1e-30);
    let (lo, hi) = (
        (from_s * sample_rate) as usize,
        ((to_s * sample_rate) as usize).min(env.len().saturating_sub(1)),
    );
    if lo + 2 >= hi {
        return Vec::new();
    }
    // The decaying trend the rise has to beat: a running minimum from the left,
    // which is what "the envelope stopped falling and rose again" means.
    let mut out = Vec::new();
    let mut trough = env[lo];
    let mut i = lo + 1;
    while i < hi {
        if env[i] < trough {
            trough = env[i];
        } else if env[i] > env[i - 1] && env[i] >= env[i + 1] {
            let rise = 20.0 * (env[i] / trough.max(1e-30)).log10();
            if rise >= margin_db {
                out.push((i as f64 / sample_rate, 20.0 * (env[i] / direct).log10()));
                trough = env[i];
            }
        }
        i += 1;
    }
    out
}

/// The minimal matching stage: a direct sound, a few measured reflections, and
/// a decorrelated exponential-noise tail with a measured decay per band.
///
/// Everything here is a *reading*: `reflections` from
/// [`reflection_candidates`], `tail_t60` from [`energy_decay`], `tail_level_db`
/// from the late-to-direct energy ratio of the same recordings. The two
/// channels' tails are drawn from **independent** streams, which is the whole
/// spatial content of the stage: a pan-pot's two channels correlate at 1.0 and
/// a room's do not.
#[derive(Clone, Debug)]
pub struct RoomStage {
    pub reflections: Vec<Reflection>,
    /// Where the diffuse tail starts, in seconds after the direct sound.
    pub tail_onset_s: f64,
    /// Total tail energy under the direct sound, in dB.
    pub tail_level_db: f64,
    /// `(lo, hi, T60)` per band.
    pub tail_t60: Vec<(f64, f64, f64)>,
    /// First-order roll-off applied to the reflection train.
    pub reflection_lowpass_hz: f64,
    /// The draw the tail's noise comes from, so a render is reproducible.
    pub seed: u64,
}

impl RoomStage {
    /// A stage that does nothing: the null.
    pub fn none() -> RoomStage {
        RoomStage {
            reflections: Vec::new(),
            tail_onset_s: 0.0,
            tail_level_db: f64::NEG_INFINITY,
            tail_t60: Vec::new(),
            reflection_lowpass_hz: 20_000.0,
            seed: 0,
        }
    }

    fn tail_length_s(&self) -> f64 {
        self.tail_t60
            .iter()
            .map(|&(_, _, t)| t)
            .fold(0.0f64, f64::max)
            .max(0.0)
    }

    /// The stage's own two-channel impulse response, direct sound included.
    pub fn impulse_response(&self, sample_rate: f64) -> Vec<Vec<f32>> {
        let tail_end = self.tail_onset_s + self.tail_length_s();
        let last_tap = self
            .reflections
            .iter()
            .map(|r| r.delay_s)
            .fold(0.0f64, f64::max);
        let n = ((tail_end.max(last_tap) + 0.01) * sample_rate).ceil() as usize + 1;
        let mut channels = vec![vec![0.0f32; n.max(1)]; 2];

        // Direct.
        for c in channels.iter_mut() {
            c[0] = 1.0;
        }

        // Reflections, then one first-order roll-off over the train alone: air
        // and a soft surface both take the top off an arrival, and one measured
        // cutoff for all of them is the least the material licenses.
        let mut train = vec![vec![0.0f32; n]; 2];
        for r in &self.reflections {
            let idx = (r.delay_s * sample_rate).round() as usize;
            if idx == 0 || idx >= n {
                continue;
            }
            let g = 10f64.powf(r.gain_db / 20.0);
            let left = ((1.0 - r.side) * 0.5).clamp(0.0, 1.0).sqrt();
            let right = ((1.0 + r.side) * 0.5).clamp(0.0, 1.0).sqrt();
            train[0][idx] += (g * left) as f32;
            train[1][idx] += (g * right) as f32;
        }
        let a = (-std::f64::consts::TAU * self.reflection_lowpass_hz / sample_rate).exp();
        for (c, t) in channels.iter_mut().zip(train.iter()) {
            let mut state = 0.0f64;
            for i in 0..n {
                state = f64::from(t[i]) * (1.0 - a) + state * a;
                c[i] += state as f32;
            }
        }

        // The tail: independent noise per channel, shaped band by band.
        if self.tail_level_db.is_finite() && !self.tail_t60.is_empty() {
            let start = (self.tail_onset_s * sample_rate).round() as usize;
            let target = 10f64.powf(self.tail_level_db / 10.0);
            for (ch, out) in channels.iter_mut().enumerate() {
                let mut draw = Draw::for_key(ch as u8, self.seed);
                let noise: Vec<f32> = (0..n).map(|_| draw.normal() as f32).collect();
                let mut planner = FftPlanner::<f32>::new();
                let size = (n * 2).next_power_of_two();
                let spec = spectrum(&noise, size, &mut planner);
                let inverse = planner.plan_fft_inverse(size);
                let mut tail = vec![0.0f64; n];
                for &(lo, hi, t60) in &self.tail_t60 {
                    let mut band = masked(&spec, sample_rate, Some((lo, hi)));
                    inverse.process(&mut band);
                    for i in start..n {
                        let t = (i - start) as f64 / sample_rate;
                        let env = 10f64.powf(-3.0 * t / t60.max(1e-3));
                        tail[i] += f64::from(band[i].re) / size as f64 * env;
                    }
                }
                let energy: f64 = tail.iter().map(|v| v * v).sum();
                if energy > 0.0 {
                    let gain = (target / energy).sqrt();
                    for i in 0..n {
                        out[i] += (tail[i] * gain) as f32;
                    }
                }
            }
        }
        channels
    }

    /// Convolve a render with the stage. Length and alignment are preserved:
    /// the direct sound is the response's first sample, so nothing moves.
    pub fn apply(&self, audio: &Audio) -> Audio {
        let sr = f64::from(audio.sample_rate);
        let ir = self.impulse_response(sr);
        let channels: Vec<Vec<f32>> = audio
            .channels
            .iter()
            .enumerate()
            .map(|(c, x)| {
                let h = &ir[c.min(ir.len() - 1)];
                let full = convolve(x, h);
                full.into_iter().take(x.len()).collect()
            })
            .collect();
        Audio {
            sample_rate: audio.sample_rate,
            channels,
        }
    }
}

/// The whole chain: colour then space, in that order.
///
/// `PHYSICS.md` §8 and §9 are two stages and this is both of them, kept
/// separable so the collapse table can say which half moved which column.
#[derive(Clone, Debug)]
pub struct Chain {
    pub eq: ChainEq,
    pub room: RoomStage,
}

impl Chain {
    pub fn apply(&self, audio: &Audio) -> Audio {
        self.room.apply(&self.eq.apply(audio))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn white(n: usize, seed: u64) -> Vec<f32> {
        let mut draw = Draw::for_key(1, seed);
        (0..n).map(|_| draw.normal() as f32 * 0.1).collect()
    }

    fn band_power_of(signal: &[f32], sample_rate: f64) -> Vec<f64> {
        let n = signal.len().next_power_of_two();
        let mut planner = FftPlanner::<f32>::new();
        let spec = spectrum(signal, n, &mut planner);
        let power: Vec<f64> = spec[..=n / 2]
            .iter()
            .map(|c| f64::from(c.norm_sqr()))
            .collect();
        band_powers(&power, sample_rate)
    }

    #[test]
    fn the_band_grid_is_log_spaced_and_covers_the_stated_range() {
        let edges = band_edges();
        assert_eq!(edges.len(), EQ_BANDS + 1);
        assert!((edges[0] - EQ_F_MIN).abs() < 1e-9);
        assert!((edges[EQ_BANDS] - EQ_F_MAX).abs() < 1e-6);
        let first = edges[1] / edges[0];
        for i in 1..EQ_BANDS {
            assert!((edges[i + 1] / edges[i] - first).abs() < 1e-9);
        }
    }

    #[test]
    fn cepstral_smoothing_keeps_a_smooth_curve_and_removes_a_jagged_one() {
        // A one-octave-wide bump survives; alternating band-to-band noise does not.
        let centres = band_centres();
        let smooth_in: Vec<f64> = centres
            .iter()
            .map(|&f| 6.0 * (-((f.ln() - 3000f64.ln()).powi(2)) / (2.0 * 0.5f64.powi(2))).exp())
            .collect();
        let jagged: Vec<f64> = (0..EQ_BANDS)
            .map(|i| if i % 2 == 0 { 6.0 } else { -6.0 })
            .collect();
        let mut draw = Draw::for_key(4, 5);
        let scatter: Vec<f64> = (0..EQ_BANDS).map(|_| draw.normal() * 6.0).collect();
        let a = cepstral_smooth(&smooth_in, CEPSTRAL_ORDER);
        let b = cepstral_smooth(&jagged, CEPSTRAL_ORDER);
        let c = cepstral_smooth(&scatter, CEPSTRAL_ORDER);
        let err_a = a
            .iter()
            .zip(smooth_in.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f64, f64::max);
        let rms = |v: &[f64]| (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt();
        assert!(
            err_a < 1.0,
            "a smooth bump should survive, worst error {err_a:.2} dB"
        );
        // A comb at the band grid's own Nyquist is the worst case for a
        // truncated DCT and is not removed outright: 6 dB of comb leaves under
        // 2, which is 10 dB of power. Band-to-band scatter — what a partial
        // series actually looks like on this grid — is left near nothing.
        assert!(
            rms(&b) < 2.0,
            "a band-to-band comb should be cut hard, {:.2} dB rms",
            rms(&b)
        );
        // Keeping `order` of `n` coefficients keeps `order / n` of a white
        // input's variance, so `sqrt(12 / 40) = 0.55` is the theoretical figure
        // and the test is that the implementation is at it rather than past it.
        assert!(
            rms(&c) < 0.65 * rms(&scatter),
            "scatter should fall to about 0.55 of itself, {:.2} -> {:.2} dB rms",
            rms(&scatter),
            rms(&c)
        );
    }

    #[test]
    fn the_eq_realises_the_curve_it_was_given() {
        let sr = 48_000.0;
        let centres = band_centres();
        let target: Vec<f64> = centres
            .iter()
            .map(|&f| {
                6.0 * (-((f.ln() - 4000f64.ln()).powi(2)) / (2.0 * 0.6f64.powi(2))).exp() - 2.0
            })
            .collect();
        let eq = ChainEq::from_db(cepstral_smooth(&target, CEPSTRAL_ORDER)).expect("40 bands");
        let x = white(1 << 16, 7);
        let y = eq.apply_mono(&x, sr);
        let (bx, by) = (band_power_of(&x, sr), band_power_of(&y, sr));
        // Bands well inside the fitted range and well above the window's own
        // resolution: the lowest two bands are 40-46 Hz, narrower than the FIR.
        let mut worst: f64 = 0.0;
        for b in 6..EQ_BANDS - 1 {
            let moved = 10.0 * (by[b] / bx[b]).log10();
            worst = worst.max((moved - eq.gains_db[b]).abs());
        }
        assert!(
            worst < 1.0,
            "the filter should realise its own curve, worst {worst:.2} dB"
        );
    }

    #[test]
    fn the_eq_keeps_length_and_alignment() {
        let sr = 48_000.0;
        let eq = ChainEq::from_db(vec![0.0; EQ_BANDS]).expect("40 bands");
        let mut x = vec![0.0f32; 4096];
        x[1000] = 1.0;
        let y = eq.apply_mono(&x, sr);
        assert_eq!(y.len(), x.len());
        let peak = y
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).expect("finite"))
            .expect("a peak")
            .0;
        assert_eq!(peak, 1000, "a flat EQ must not move the signal in time");
    }

    #[test]
    fn a_fit_recovers_a_transfer_it_was_given() {
        // Twenty synthetic "keys", each a different random spectrum; the
        // reference is that spectrum through a known smooth curve. The fit must
        // return the curve and not the spectra.
        let centres = band_centres();
        let truth: Vec<f64> = centres
            .iter()
            .map(|&f| 4.0 * (f / 1000.0).ln().tanh() - 1.5)
            .collect();
        let truth = cepstral_smooth(&truth, CEPSTRAL_ORDER);
        let mean = truth.iter().sum::<f64>() / EQ_BANDS as f64;
        let truth: Vec<f64> = truth.iter().map(|v| v - mean).collect();
        let mut draw = Draw::for_key(3, 11);
        let items: Vec<EqSample> = (0..20)
            .map(|i| {
                let engine: Vec<f64> = (0..EQ_BANDS)
                    .map(|_| 10f64.powf(draw.normal() * 0.8))
                    .collect();
                let reference: Vec<f64> = engine
                    .iter()
                    .zip(truth.iter())
                    .map(|(&e, &g)| e * 10f64.powf(g / 10.0))
                    .collect();
                EqSample {
                    engine,
                    reference,
                    key: 21 + i as u8,
                    velocity: 90,
                }
            })
            .collect();
        let fit = fit_eq(&items);
        let worst = fit
            .smooth_db
            .iter()
            .zip(truth.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(
            worst < 0.6,
            "the fit should return the transfer, worst {worst:.2} dB"
        );
    }

    #[test]
    fn the_fit_carries_no_part_of_the_level() {
        // The same material with the engine 20 dB quiet must fit the same curve:
        // a level error is not a chain and must not land in one.
        let mut draw = Draw::for_key(5, 13);
        let base: Vec<EqSample> = (0..16)
            .map(|i| {
                let engine: Vec<f64> = (0..EQ_BANDS)
                    .map(|_| 10f64.powf(draw.normal() * 0.5))
                    .collect();
                let reference: Vec<f64> = engine.iter().map(|&e| e * 2.0).collect();
                EqSample {
                    engine,
                    reference,
                    key: 30 + i as u8,
                    velocity: 90,
                }
            })
            .collect();
        let quiet: Vec<EqSample> = base
            .iter()
            .map(|s| EqSample {
                engine: s.engine.iter().map(|&e| e * 0.01).collect(),
                reference: s.reference.clone(),
                key: s.key,
                velocity: s.velocity,
            })
            .collect();
        let a = fit_eq(&base);
        let b = fit_eq(&quiet);
        let (mean_abs, _) = curve_agreement(&a.smooth_db, &b.smooth_db);
        assert!(
            mean_abs < 1e-6,
            "a level offset moved the curve by {mean_abs:.4} dB"
        );
        let flat = a.smooth_db.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(
            flat < 1e-6,
            "a pure gain should fit a flat curve, {flat:.4} dB"
        );
    }

    #[test]
    fn the_curve_is_held_flat_outside_what_the_material_could_read() {
        // Material with energy only in bands 6..=30: the ends must be flat and
        // must not continue the trend the DCT would like to draw.
        let mut draw = Draw::for_key(6, 17);
        let items: Vec<EqSample> = (0..20)
            .map(|i| {
                let engine: Vec<f64> = (0..EQ_BANDS)
                    .map(|b| {
                        if (6..=30).contains(&b) {
                            10f64.powf(draw.normal() * 0.4)
                        } else {
                            1e-12
                        }
                    })
                    .collect();
                let reference: Vec<f64> = engine
                    .iter()
                    .enumerate()
                    .map(|(b, &e)| e * 10f64.powf((b as f64 - 18.0) * 0.4 / 10.0))
                    .collect();
                EqSample {
                    engine,
                    reference,
                    key: 21 + i as u8,
                    velocity: 90,
                }
            })
            .collect();
        let fit = fit_eq(&items);
        assert_eq!(fit.read_range, (6, 30));
        for b in 0..6 {
            assert!((fit.smooth_db[b] - fit.smooth_db[6]).abs() < 1e-9);
        }
        for b in 31..EQ_BANDS {
            assert!((fit.smooth_db[b] - fit.smooth_db[30]).abs() < 1e-9);
        }
        // And the read range still carries the transfer it was given: +0.4 dB
        // per band, which is 9.6 dB across the 24 bands that were read.
        let span = fit.smooth_db[30] - fit.smooth_db[6];
        assert!(
            (span - 9.6).abs() < 1.5,
            "the read range should keep its slope, {span:.2} dB"
        );
    }

    #[test]
    fn a_band_nobody_could_read_is_filled_from_its_neighbours() {
        let mut curve = vec![f64::NAN; EQ_BANDS];
        for (i, c) in curve.iter_mut().enumerate() {
            if !(10..=20).contains(&i) {
                *c = i as f64;
            }
        }
        let filled = fill_holes(&curve);
        assert!(filled.iter().all(|v| v.is_finite()));
        assert!(
            (filled[15] - 15.0).abs() < 1e-9,
            "a straight line should be interpolated"
        );
    }

    #[test]
    fn a_pan_pot_correlates_at_one_and_a_delayed_pair_does_not() {
        let sr = 48_000.0;
        let x = white(1 << 14, 21);
        let panned: Vec<f32> = x.iter().map(|&v| v * 0.6).collect();
        let sig = stereo_signature(&x, &panned, sr).expect("two channels");
        assert!(
            sig.broadband.zero_r > 0.99,
            "a pan-pot must read 1.0 at lag zero, read {:.3}",
            sig.broadband.zero_r
        );
        let delay = 96; // 2 ms
        let mut shifted = vec![0.0f32; x.len()];
        shifted[delay..].copy_from_slice(&x[..x.len() - delay]);
        let sig = stereo_signature(&x, &shifted, sr).expect("two channels");
        assert!(
            sig.broadband.zero_r.abs() < 0.2,
            "a 2 ms shift decorrelates at lag zero"
        );
        assert!(
            (sig.broadband.lag_ms + 2.0).abs() < 0.1,
            "and peaks at the shift, read {:.2} ms",
            sig.broadband.lag_ms
        );
    }

    #[test]
    fn a_known_exponential_decay_is_recovered() {
        let sr = 48_000.0;
        let t60 = 0.4;
        let n = (2.0 * sr) as usize;
        let mut draw = Draw::for_key(9, 31);
        let signal: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f64 / sr;
                (draw.normal() * 10f64.powf(-3.0 * t / t60)) as f32
            })
            .collect();
        let curve = schroeder_db(&signal);
        let t20 = decay_time(&curve, sr, -5.0, -25.0).expect("a decay");
        assert!(
            (t20 - t60).abs() < 0.05,
            "T20 should recover the 0.4 s decay, read {t20:.3} s"
        );
    }

    #[test]
    fn the_room_stage_decorrelates_and_keeps_the_direct_sound_in_place() {
        let sr = 48_000.0;
        let room = RoomStage {
            reflections: vec![
                Reflection {
                    delay_s: 0.011,
                    gain_db: -9.0,
                    side: -0.4,
                },
                Reflection {
                    delay_s: 0.019,
                    gain_db: -12.0,
                    side: 0.5,
                },
            ],
            tail_onset_s: 0.025,
            tail_level_db: -8.0,
            tail_t60: vec![
                (250.0, 500.0, 0.5),
                (500.0, 2000.0, 0.45),
                (2000.0, 6000.0, 0.3),
            ],
            reflection_lowpass_hz: 6_000.0,
            seed: 4,
        };
        let ir = room.impulse_response(sr);
        assert_eq!(ir.len(), 2);
        assert!(
            (ir[0][0] - 1.0).abs() < 0.05,
            "the direct sound is the first sample"
        );
        let x = white(1 << 15, 17);
        let audio = Audio::new(48_000, vec![x.clone(), x.clone()]).expect("stereo");
        let out = room.apply(&audio);
        assert_eq!(out.channels[0].len(), x.len());
        let before = stereo_signature(&x, &x, sr).expect("two channels");
        let after = stereo_signature(&out.channels[0], &out.channels[1], sr).expect("two channels");
        assert!(before.broadband.zero_r > 0.999);
        assert!(
            after.broadband.zero_r < 0.98,
            "the stage must decorrelate the pair, read {:.4}",
            after.broadband.zero_r
        );
    }

    #[test]
    fn the_reflection_finder_finds_the_arrivals_it_is_given() {
        let sr = 48_000.0;
        let n = (0.1 * sr) as usize;
        let mut x = vec![0.0f32; n];
        // A direct burst and two later, quieter ones.
        let mut draw = Draw::for_key(2, 41);
        let burst = |x: &mut Vec<f32>, at: f64, gain: f64, draw: &mut Draw| {
            let start = (at * sr) as usize;
            for i in 0..96 {
                x[start + i] += (draw.normal() * gain) as f32;
            }
        };
        burst(&mut x, 0.001, 1.0, &mut draw);
        burst(&mut x, 0.013, 0.3, &mut draw);
        burst(&mut x, 0.028, 0.15, &mut draw);
        let found = reflection_candidates(&x, sr, 0.004, 0.06, 6.0);
        let near = |t: f64| found.iter().any(|&(d, _)| (d - t).abs() < 0.004);
        assert!(near(0.013), "the 13 ms arrival should be found: {found:?}");
        assert!(near(0.028), "the 28 ms arrival should be found: {found:?}");
    }

    #[test]
    fn two_curves_that_agree_read_as_agreeing() {
        let a: Vec<f64> = (0..EQ_BANDS)
            .map(|i| (i as f64 / 10.0).sin() * 3.0)
            .collect();
        let b: Vec<f64> = a.iter().map(|v| v + 0.2).collect();
        let (mean_abs, r) = curve_agreement(&a, &b);
        assert!((mean_abs - 0.2).abs() < 1e-9);
        assert!(r > 0.999);
        let c: Vec<f64> = a.iter().map(|v| -v).collect();
        let (_, r) = curve_agreement(&a, &c);
        assert!(r < -0.999, "an inverted curve must not read as agreement");
    }
}
