//! `[noise.strike]`: the hammer's own noise, measured as what is left of a
//! struck note's first 150 ms when every partial of it has been subtracted.
//!
//! # Why this event exists at all, and why it is not a level
//!
//! `docs/history/TUNING_REPORT.md` §4 refuted a missing attack *transient*: broadband energy
//! between the partials over the first 85 ms came back within ~7 dB of the
//! engine's. What two later measurements convict instead is the attack's
//! **spectrum**:
//!
//! * `renders/realism/REALISM.md` finds the engine's attacks **+5.2 dB more
//!   tonal** than the recordings' over six phrases, worst on `staccato`
//!   (+11.3 dB).
//! * `renders/timbre-ladder/ANALYSIS.md` §8.3 finds the first 30 ms with every
//!   tracked partial subtracted 11.1 to 12.7 dB more tonal in the engine than in
//!   the recording — and closes it at all three keys by mixing the recording's
//!   own residual back in.
//!
//! So the fit's job is a *colour* at a level the recordings already have, not a
//! louder attack. The level written here is the measured residual's own peak
//! against the same key's velocity-90 strike, which is exactly the convention
//! every other `[noise]` level uses; it is 10 to 20 dB under the note, and
//! adding it moves the attack's level by a fraction of a decibel while moving
//! its flatness by the whole gap.
//!
//! # The measurement
//!
//! Per key and per velocity layer, on the recording's mono sum:
//!
//! 1. Take the partial frequencies the decay stage fitted — the ones that
//!    survived its level floor, so a "partial" in the noise is not subtracted as
//!    if it were one.
//! 2. Project the signal onto `e^{i 2 pi f_k t}` in short hopped windows and
//!    resynthesize what that measured
//!    ([`residual::track_complex`](crate::residual::track_complex)), which is
//!    the ladder's own subtraction.
//! 3. Measure the difference over the first [`AttackConfig::residual_s`]: its
//!    peak, its spectral centroid, the frequency below which
//!    [`AttackConfig::rolloff`] of its energy lies, its spectral flatness, and
//!    the exponential rate its envelope falls at.
//!
//! The four shape numbers are medians across the compass, because the schema
//! holds one of each; the level becomes compass anchors, and the velocity law is
//! the slope of level against drive that the sixteen layers draw.
//!
//! # The band the shape is fitted over, and why it is not the whole residual
//!
//! The residual has two regimes and the schema has one band-pass. Measured on
//! Salamander in octave bands, C4's residual peaks at **62–125 Hz**, an octave
//! *below* its own fundamental, and carries a second, far broader plateau from
//! 250 Hz to 4 kHz some 15 dB under it; A2's does the same; C7's is flat from
//! 31 Hz to 2 kHz. A shape fitted to the whole thing is a 100 Hz thump at every
//! key — which is a sound the engine already makes twice over: the action's four
//! events are centred at 77 to 300 Hz (`docs/history/TUNING_REPORT.md` §5) and the board's own
//! modes put a strike's low frequencies into the render whatever the hammer
//! does.
//!
//! So the fit is band-limited, and **the level is measured on the same
//! band-limited signal as the colour**, so that what the burst delivers is the
//! signal that was measured rather than a low thump's level wearing a broadband
//! spectrum's colour.
//!
//! # The band is where the engine is *missing* the residual, and it is measured
//!
//! [`MIN_STRIKE_BANDWIDTH_HZ`] … [`MAX_STRIKE_BANDWIDTH_HZ`] is the schema's own
//! range and the first fit used all of it. That was wrong, and the render said
//! so: the median of a residual whose energy peaks at 200–400 Hz put the burst
//! at a 344 Hz centroid under a 1 kHz limit, and `REALISM.md`'s attack column
//! moved 0.19 dB. Six decibels more of that burst bought another 0.18
//! (`DECISIONS.md` 206), which is the signature of an event playing where
//! nothing was missing.
//!
//! What is missing is measurable, and it is not where the residual is loudest.
//! In octave bands over the benchmark's own onsets the engine's first 30 ms sits
//! **−0.2 dB** against the recordings at 200–400 Hz, **−2.0** at 400–800,
//! **−6.3** at 800–1600, **−17.5** at 1.6–3.2 kHz, **−19.8** at 3.2–6.4 kHz and
//! **−13.9** at 6.4–10 kHz. The bottom of the residual is a sound the engine
//! already makes — its own low partials, the board's modes, and the action's
//! four events, which are centred at 77–300 Hz and stop at the engine's
//! `noise::BANDWIDTH_HZ` of 2 kHz. The top of it is a
//! sound nothing in the engine makes at all, and the schema contract's own
//! sentence for this event is that its bandwidth "reaches far above the 2 kHz
//! structure-borne ceiling of the action events".
//!
//! So [`deficit_band`] measures the band: the same third-octave density is taken
//! from the recording's onset residual and from the *engine's own* onset
//! residual of the same key at the same drive, each against its own note's peak,
//! and the band is the run of third-octaves around the largest deficit over
//! which the engine is at least [`AttackConfig::deficit_db`] short. The level,
//! the centroid, the limit and the decay are then all measured inside that band
//! ([`AttackConfig::band_hz`]) — which is the same discipline
//! `estimate::directivity` and `estimate::damper` use, a measurement referenced
//! to what the engine already does rather than to the recording alone.
//!
//! Two details of the colour follow from the shape the engine actually builds
//! (`engine::noise::EventShape`: a band-pass of fixed `Q` under a low-pass):
//!
//! * The centroid is the **geometric** power-weighted mean frequency. A
//!   constant-`Q` band-pass is symmetric in `log f`, so its centre is a
//!   geometric mean; an arithmetic one over a spectrum that spans six octaves is
//!   a number about the loudest octave and not about the band.
//! * The bandwidth is where the spectral **density** has fallen
//!   [`AttackConfig::band_limit_db`] below its own maximum, in third-octave
//!   bands — a band *limit*, which is what a low-pass corner is. An energy
//!   quantile answers a different question and, on a spectrum with a long flat
//!   tail, answers it with the loudest octave's edge.
//!
//! # Where the measurement is refused
//!
//! The projection needs four periods of the note's own fundamental to keep
//! neighbouring partials out of each other's window, and an attack measurement
//! needs a window far shorter than the 150 ms it lives in. Those two meet at
//! about **100 Hz**: below it, four periods is more than 40 ms and the
//! subtraction stops separating the partials at all — measured on A0, the
//! "residual" comes back 2.6 dB under the note itself, which is the note. Keys
//! under [`AttackConfig::min_f0_hz`] are therefore not measured, and their level
//! comes from the lowest anchor above them, held.
//!
//! # What the residual is not, and why the level is an upper bound
//!
//! Two things ride along inside it, both of which `ANALYSIS.md` names:
//!
//! * **Partials above the tracked set are not subtracted** and count as
//!   residual. That inflates the level in the bass, where a wound string has
//!   partials past the tracker's reach.
//! * **The analysis cannot follow an envelope faster than its own window.** The
//!   projection measures each partial over 20–40 ms and interpolates between
//!   hops, so a note whose partials rise faster than that leaves a subtraction
//!   error concentrated in exactly the milliseconds this event is measured over.
//!   Measured on synthetic notes with a known burst mixed in, that floor is
//!   about −27 dB for a 30 ms rise and about −16 dB for a 12 ms one — and
//!   `REALISM.md` puts real attacks at 13 to 22 ms.
//!
//! So the level written here is an **upper bound** on the hammer's noise, and
//! the fit is deliberately conservative about it: the level is the anchors'
//! median over neighbouring keys rather than any one key's own figure, and this
//! module reports the residual's *flatness* beside its level because the
//! flatness is the half `ANALYSIS.md` §8.3 found trustworthy. What settles it is
//! not this measurement but the render: `REALISM.md`'s attack column has both a
//! level and a tonality in it, and a strike noise fitted here has to move the
//! second without moving the first.

use crate::estimate::noise::{compass_anchors, NoiseConfig};
use crate::numeric::weighted_least_squares;
use crate::preset::{
    NoiseAnchor, StrikeNoise, MAX_STRIKE_BANDWIDTH_HZ, MAX_STRIKE_DECAY_S,
    MIN_STRIKE_BANDWIDTH_HZ, MIN_STRIKE_DECAY_S, NOMINAL_STRIKE_VELOCITY,
};
use crate::residual::onset_residual;
use crate::stft::Stft;
use crate::stft::StftConfig;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttackConfig {
    /// How much of the attack the residual is measured over. 150 ms is the span
    /// `ANALYSIS.md`'s `150 ms` column uses and the span the ladder's own
    /// residual is kept over.
    pub residual_s: f64,
    /// Fraction of the residual's energy that has to lie below the reported
    /// bandwidth.
    pub rolloff: f64,
    /// Block length of the envelope the decay rate is fitted on, seconds.
    pub envelope_block_s: f64,
    /// How far below its peak the residual's envelope is followed while its rate
    /// is fitted. Below this it is the note's own floor.
    pub envelope_range_db: f64,
    /// Fewest envelope blocks a rate may be fitted from.
    pub min_envelope_blocks: usize,
    /// Transform used for the colour measurements. A power of two spanning the
    /// residual window.
    pub spectrum_size: usize,
    /// How far the spectral density may fall below its own maximum and still be
    /// inside the burst's band.
    pub band_limit_db: f64,
    /// Bands per octave the density is measured in.
    pub bands_per_octave: usize,
    /// Lowest fundamental this measurement is attempted at, Hz. Below it four
    /// periods is longer than any window an attack can be measured in.
    pub min_f0_hz: f64,
    /// How far under the note a residual has to sit before it counts as one, dB.
    ///
    /// The other end of the same gate as [`AttackConfig::min_f0_hz`], and it
    /// bites at the other end of the compass. C8 has **two** partials under
    /// Nyquist, so a subtraction of its tracked partials removes two lines from
    /// a note that is mostly not lines, and what is left is the note: measured
    /// on Salamander, the "residual" of C8 came back **1.0 dB** under the note
    /// itself. A residual that close has not separated anything, and writing it
    /// as a level would put a hammer noise on the top octave as loud as the top
    /// octave.
    pub max_level_db: f64,
    /// Fewest partials the note must have had tracked before its residual is a
    /// residual. Three lines subtracted from a note that is mostly not lines is
    /// the same failure as [`AttackConfig::max_level_db`] catches, one step
    /// earlier and in the units that cause it.
    pub min_partials: usize,
    /// Fewest velocity layers a key needs before it anchors the compass. The
    /// level at the nominal velocity is read off a line, and a line through four
    /// points that scatter several dB each is not one.
    pub min_layers: usize,
    /// The band the level and the colour are measured in, Hz.
    ///
    /// The default is the schema's whole range, which is what
    /// [`density_bands`] reports a spectrum over and what [`deficit_band`]
    /// searches inside. A fit narrows it to what that search returns; see the
    /// module header for why measuring the residual where the engine already has
    /// one is measuring nothing.
    pub band_hz: (f64, f64),
    /// How far the engine's own onset residual has to sit below the recording's,
    /// in a third-octave band, before that band counts as missing.
    ///
    /// Six decibels, because that is a factor of four in power and comfortably
    /// outside the two or three decibels a residual's own peak scatters by
    /// between draws; the measured deficit at the top of the band is 14 to 20 dB
    /// and at the bottom is under one, so nothing here is decided by the exact
    /// threshold.
    pub deficit_db: f64,
}

impl Default for AttackConfig {
    fn default() -> Self {
        Self {
            residual_s: 0.150,
            rolloff: 0.95,
            envelope_block_s: 0.005,
            envelope_range_db: 30.0,
            min_envelope_blocks: 6,
            spectrum_size: 1 << 13,
            band_limit_db: 20.0,
            bands_per_octave: 3,
            min_f0_hz: 100.0,
            max_level_db: -6.0,
            min_partials: 4,
            min_layers: 8,
            band_hz: (
                f64::from(MIN_STRIKE_BANDWIDTH_HZ),
                f64::from(MAX_STRIKE_BANDWIDTH_HZ),
            ),
            deficit_db: 6.0,
        }
    }
}

impl AttackConfig {
    /// The same configuration measuring inside `band`, clamped to the schema's
    /// own range so that no search can ask for a burst the preset cannot hold.
    pub fn in_band(self, band: (f64, f64)) -> AttackConfig {
        let lo = band
            .0
            .clamp(f64::from(MIN_STRIKE_BANDWIDTH_HZ), f64::from(MAX_STRIKE_BANDWIDTH_HZ));
        let hi = band
            .1
            .clamp(lo, f64::from(MAX_STRIKE_BANDWIDTH_HZ));
        AttackConfig {
            band_hz: (lo, hi),
            ..self
        }
    }
}

/// The onset residual of one recording.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttackResidual {
    pub key: u8,
    /// Middle of the layer's MIDI velocity band — the drive this level was
    /// measured at.
    pub midi_velocity: u8,
    /// Peak of the residual, in dB relative to the peak of a velocity-90 strike
    /// of the same key, both at the level the instrument plays them.
    pub level_db: f64,
    /// Power-weighted mean frequency of the residual, Hz.
    pub centroid_hz: f64,
    /// Frequency below which [`AttackConfig::rolloff`] of its energy lies, Hz.
    pub bandwidth_hz: f64,
    /// Time for the residual's envelope to fall 40 dB, from the rate fitted over
    /// the window. Longer than the window itself, normally: the fit is a rate,
    /// not a stopwatch.
    pub decay_s: f64,
    /// Spectral flatness of the residual, in dB — 0 is a continuum, −40 is one
    /// or two lines. The half of this measurement `ANALYSIS.md` §8.3 found
    /// trustworthy.
    pub flatness_db: f64,
}

/// Measures the onset residual of one recording.
///
/// `reference_peak` is the peak of a velocity-90 strike of the same key, at the
/// level the instrument plays it; `signal` must be at that same level.
#[allow(clippy::too_many_arguments)]
pub fn residual_metrics(
    key: u8,
    midi_velocity: u8,
    signal: &[f32],
    sample_rate: f64,
    partial_hz: &[f64],
    onset_s: f64,
    reference_peak: f64,
    config: &AttackConfig,
) -> Option<AttackResidual> {
    if reference_peak <= 0.0 || !reference_peak.is_finite() {
        return None;
    }
    // Below this the projection cannot separate the note's own partials inside
    // any window an attack fits in, and what comes back is the note.
    let f0 = partial_hz.iter().copied().fold(f64::INFINITY, f64::min);
    if f0 < config.min_f0_hz || !f0.is_finite() || partial_hz.len() < config.min_partials {
        return None;
    }
    let residual = onset_residual(signal, sample_rate, partial_hz, onset_s, config.residual_s)?;
    // The level and the colour are measured on the same band-limited signal —
    // the band the event exists to fill, and the band the engine will play it
    // in. See the module header.
    let band = band_limited(&residual, sample_rate, config.band_hz.0, config.band_hz.1);
    let peak = band.iter().fold(0.0f64, |m, &x| m.max(x.abs()));
    if peak <= 0.0 {
        return None;
    }
    let level_db = 20.0 * (peak / reference_peak).log10();
    if level_db > config.max_level_db {
        return None;
    }
    let (centroid_hz, bandwidth_hz, flatness_db) = colour(&band, sample_rate, config)?;
    let decay_s = fitted_decay_s(&band, sample_rate, config)?;
    Some(AttackResidual {
        key,
        midi_velocity,
        level_db,
        centroid_hz,
        bandwidth_hz,
        decay_s,
        flatness_db,
    })
}

/// The band the engine's own attack is short of the recording's, in Hz.
///
/// Both arguments are third-octave *densities in dB against their own note's
/// peak* — [`density_bands`] normalised, which is what makes two renders at two
/// levels comparable. `recorded` is the recording's onset residual and `engine`
/// is the engine's own, measured with the same code from a render of the same
/// key at the same drive and with `[noise.strike]` silenced, so what comes back
/// is what the event has to supply rather than what it already supplies.
///
/// The band is the **contiguous run of third-octaves around the largest
/// deficit** over which the engine is at least [`AttackConfig::deficit_db`]
/// short. Contiguous because the engine plays one band-pass and a band-pass is
/// an interval; around the largest deficit because that is the part of the
/// spectrum the event exists for, and growing outwards from it stops at the
/// first octave the engine already has — which in the bass is the action's own
/// four events and in the treble is nothing at all.
///
/// `None` when nothing is missing by that margin, which is the honest answer for
/// a key whose attack the engine already reproduces.
pub fn deficit_band(
    recorded: &[(f64, f64)],
    engine: &[(f64, f64)],
    config: &AttackConfig,
) -> Option<(f64, f64)> {
    if recorded.len() != engine.len() || recorded.is_empty() {
        return None;
    }
    let deficit: Vec<(f64, f64)> = recorded
        .iter()
        .zip(engine)
        .filter(|((a, _), (b, _))| (a - b).abs() < 1.0)
        .map(|(&(hz, r), &(_, e))| (hz, r - e))
        .collect();
    if deficit.len() != recorded.len() {
        return None;
    }
    let peak = deficit
        .iter()
        .enumerate()
        .max_by(|a, b| a.1 .1.total_cmp(&b.1 .1))
        .map(|(i, _)| i)?;
    if deficit[peak].1 < config.deficit_db {
        return None;
    }
    let mut lo = peak;
    while lo > 0 && deficit[lo - 1].1 >= config.deficit_db {
        lo -= 1;
    }
    let mut hi = peak;
    while hi + 1 < deficit.len() && deficit[hi + 1].1 >= config.deficit_db {
        hi += 1;
    }
    Some((deficit[lo].0, deficit[hi].0))
}

/// A third-octave density in dB against its own maximum, which is the form
/// [`deficit_band`] compares two of.
pub fn density_db(bands: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let peak = bands.iter().fold(0.0f64, |m, &(_, p)| m.max(p));
    if peak <= 0.0 {
        return bands.iter().map(|&(hz, _)| (hz, 0.0)).collect();
    }
    bands
        .iter()
        .map(|&(hz, p)| (hz, 10.0 * (p.max(1e-30) / peak).log10()))
        .collect()
}

/// The residual inside the band the burst will occupy, exposed so that a
/// calibration gate can put a known burst through the same filter before
/// comparing levels with it.
pub fn band_limited_for_test(signal: &[f64], sample_rate: f64) -> Vec<f64> {
    band_limited(
        signal,
        sample_rate,
        f64::from(MIN_STRIKE_BANDWIDTH_HZ),
        f64::from(MAX_STRIKE_BANDWIDTH_HZ),
    )
}

/// The residual inside the band the burst will occupy: two cascaded one-poles
/// either side, which is skirt enough that neither edge decides the answer and
/// phase enough that nothing here reads it.
fn band_limited(residual: &[f64], sample_rate: f64, low_hz: f64, high_hz: f64) -> Vec<f64> {
    let mut out = residual.to_vec();
    for (cutoff, high) in [(low_hz, true), (high_hz, false)] {
        let a = (-std::f64::consts::TAU * cutoff / sample_rate).exp();
        for _ in 0..2 {
            let mut state = 0.0f64;
            for x in out.iter_mut() {
                state = (1.0 - a) * *x + a * state;
                *x = if high { *x - state } else { state };
            }
        }
    }
    out
}

/// Third-octave (or finer) power density of a signal: `(centre Hz, power)`.
///
/// Public because the spectrum a preset was fitted from is worth printing beside
/// the four numbers it was reduced to.
pub fn density_bands(
    signal: &[f64],
    sample_rate: f64,
    config: &AttackConfig,
) -> Option<Vec<(f64, f64)>> {
    let size = config.spectrum_size.max(64).next_power_of_two();
    let padded: Vec<f32> = (0..size)
        .map(|i| signal.get(i).copied().unwrap_or(0.0) as f32)
        .collect();
    let stft = Stft::new(StftConfig::padded(size, size, 1).ok()?).ok()?;
    let mut magnitude: Vec<f32> = Vec::new();
    stft.for_each_frame(&padded, 1.0, |_, frame| {
        magnitude.clear();
        magnitude.extend_from_slice(frame);
    });
    if magnitude.len() < 4 {
        return None;
    }
    let bin_hz = sample_rate / size as f64;
    let per_octave = config.bands_per_octave.max(1) as f64;
    let ratio = 2f64.powf(1.0 / per_octave);
    let half = ratio.sqrt();
    let mut bands = Vec::new();
    let mut centre = config.band_hz.0;
    while centre <= config.band_hz.1 {
        let (lo, hi) = (centre / half, centre * half);
        let power: f64 = magnitude
            .iter()
            .enumerate()
            .filter(|&(bin, _)| {
                let f = bin as f64 * bin_hz;
                f >= lo && f < hi
            })
            .map(|(_, &m)| f64::from(m) * f64::from(m))
            .sum();
        // Power *density*: a third-octave band is proportionally wider the
        // higher it sits, and comparing raw band energies would call any flat
        // spectrum a rising one.
        bands.push((centre, power / (hi - lo)));
        centre *= ratio;
    }
    (!bands.is_empty()).then_some(bands)
}

/// Centroid, band limit and flatness of one band-limited residual.
///
/// The centroid is the geometric power-weighted mean frequency and the band
/// limit is where the third-octave density has fallen
/// [`AttackConfig::band_limit_db`] below its own maximum; the module header says
/// why each is the right one for the shape the engine builds. The flatness is
/// the ordinary one — geometric over arithmetic mean of the density — and is
/// reported rather than written, because it is the half of this measurement
/// `ANALYSIS.md` §8.3 found trustworthy.
fn colour(band: &[f64], sample_rate: f64, config: &AttackConfig) -> Option<(f64, f64, f64)> {
    let bands = density_bands(band, sample_rate, config)?;
    let total: f64 = bands.iter().map(|&(_, p)| p).sum();
    if total <= 0.0 {
        return None;
    }
    let centroid_hz = (bands
        .iter()
        .map(|&(hz, p)| p * hz.ln())
        .sum::<f64>()
        / total)
        .exp();
    let peak = bands.iter().fold(0.0f64, |m, &(_, p)| m.max(p));
    let floor = peak * 10f64.powf(-config.band_limit_db / 10.0);
    let bandwidth_hz = bands
        .iter()
        .filter(|&&(_, p)| p >= floor)
        .map(|&(hz, _)| hz)
        .fold(config.band_hz.0, f64::max);
    let kept: Vec<f64> = bands.iter().map(|&(_, p)| p).filter(|&p| p > 0.0).collect();
    let flatness_db = if kept.is_empty() {
        0.0
    } else {
        let ln_mean = kept.iter().map(|p| p.ln()).sum::<f64>() / kept.len() as f64;
        let mean = kept.iter().sum::<f64>() / kept.len() as f64;
        10.0 * (ln_mean.exp() / mean).log10()
    };
    Some((centroid_hz, bandwidth_hz, flatness_db))
}

/// Time for the residual to fall 40 dB, from the exponential rate fitted to its
/// own envelope.
///
/// A rate rather than a stopwatch: the window is 150 ms and a hammer's noise
/// falls 40 dB in rather longer than that, so a "time to −40 dB" read straight
/// off the envelope would be infinite for every recording. The engine's
/// `decay_s` is the time constant of an exponential, and an exponential's rate
/// is measurable inside a window much shorter than its own decay.
fn fitted_decay_s(residual: &[f64], sample_rate: f64, config: &AttackConfig) -> Option<f64> {
    let block = ((config.envelope_block_s * sample_rate) as usize).max(1);
    let envelope: Vec<f64> = residual
        .chunks(block)
        .map(|chunk| (chunk.iter().map(|x| x * x).sum::<f64>() / chunk.len() as f64).sqrt())
        .collect();
    let (peak_block, &peak) = envelope
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))?;
    if peak <= 0.0 {
        return None;
    }
    let floor = peak * 10f64.powf(-config.envelope_range_db / 20.0);
    let points: Vec<(f64, f64)> = envelope[peak_block..]
        .iter()
        .enumerate()
        .take_while(|&(_, &a)| a > floor)
        .map(|(i, &a)| (i as f64 * block as f64 / sample_rate, a.ln()))
        .collect();
    if points.len() < config.min_envelope_blocks {
        return None;
    }
    let basis: Vec<f64> = points.iter().flat_map(|&(t, _)| [1.0, t]).collect();
    let y: Vec<f64> = points.iter().map(|&(_, a)| a).collect();
    let solution = weighted_least_squares(&basis, &y, &vec![1.0; points.len()], 2)?;
    let sigma = -solution[1];
    if sigma <= 0.0 || !sigma.is_finite() {
        // A residual that does not fall inside the window is one the window is
        // too short for, and the engine's longest legal strike is the honest
        // statement of that.
        return Some(f64::from(MAX_STRIKE_DECAY_S));
    }
    Some((40.0 / 20.0 * std::f64::consts::LN_10 / sigma).clamp(
        f64::from(MIN_STRIKE_DECAY_S),
        f64::from(MAX_STRIKE_DECAY_S),
    ))
}

/// What the fit found, beside the section it wrote.
#[derive(Clone, Debug, PartialEq)]
pub struct StrikeFitReport {
    pub strike: StrikeNoise,
    /// Keys that contributed a level, and recordings that contributed anything.
    pub keys: usize,
    pub recordings: usize,
    /// Per-key level at the nominal velocity, in key order — the line's value at
    /// drive `90/127`, before it was reduced to anchors.
    pub per_key_db: Vec<(u8, f64)>,
    /// Per-key slope of level against drive, dB over the full range.
    pub per_key_velocity_db: Vec<(u8, f64)>,
    /// Median residual flatness, dB: how far from a continuum what is being
    /// added actually is.
    pub flatness_db: f64,
}

/// Fits `[noise.strike]` from the measured residuals.
///
/// The level of each key is the value its own sixteen layers' line takes at
/// velocity [`NOMINAL_STRIKE_VELOCITY`] — a line rather than the layer nearest
/// 90, because the peak of a noise residual scatters several dB per draw and
/// sixteen of them constrain the line far better than one constrains a point.
/// The slope of that same line, in dB per unit of drive, *is* the engine's
/// `velocity_db`, whose law is `velocity_db · (v/127 − 90/127)` through the
/// tabulated level.
pub fn fit_strike(
    measurements: &[AttackResidual],
    base: &StrikeNoise,
    config: &AttackConfig,
    noise: &NoiseConfig,
) -> StrikeFitReport {
    let mut keys: Vec<u8> = measurements.iter().map(|m| m.key).collect();
    keys.sort_unstable();
    keys.dedup();

    let mut per_key_db: Vec<(u8, f64)> = Vec::new();
    let mut per_key_velocity_db: Vec<(u8, f64)> = Vec::new();
    for &key in &keys {
        let layers: Vec<&AttackResidual> =
            measurements.iter().filter(|m| m.key == key).collect();
        let Some((level, slope)) = velocity_line(&layers, config) else {
            continue;
        };
        per_key_db.push((key, level));
        per_key_velocity_db.push((key, slope));
    }

    let levels: Vec<(Option<u8>, f64)> =
        per_key_db.iter().map(|&(key, db)| (Some(key), db)).collect();
    let level_db =
        compass_anchors(&levels, 0.0, noise).unwrap_or_else(|| base.level_db.clone());
    let velocity_db = median(per_key_velocity_db.iter().map(|&(_, s)| s))
        .unwrap_or(f64::from(base.velocity_db));
    let centroid_hz =
        median(measurements.iter().map(|m| m.centroid_hz)).unwrap_or(f64::from(base.centroid_hz));
    let bandwidth_hz = median(measurements.iter().map(|m| m.bandwidth_hz))
        .unwrap_or(f64::from(base.bandwidth_hz))
        .clamp(
            f64::from(MIN_STRIKE_BANDWIDTH_HZ),
            f64::from(MAX_STRIKE_BANDWIDTH_HZ),
        );
    let decay_s = median(measurements.iter().map(|m| m.decay_s))
        .unwrap_or(f64::from(base.decay_s))
        .clamp(
            f64::from(MIN_STRIKE_DECAY_S),
            f64::from(MAX_STRIKE_DECAY_S),
        );
    let strike = StrikeNoise {
        // A burst centred outside its own band is refused by both crates, and
        // the two numbers are separate measurements of the same spectrum, so the
        // centroid is held under the rolloff rather than the pair being
        // rejected.
        centroid_hz: centroid_hz.min(bandwidth_hz) as f32,
        decay_s: decay_s as f32,
        bandwidth_hz: bandwidth_hz as f32,
        velocity_db: velocity_db as f32,
        level_db: if per_key_db.is_empty() {
            base.level_db.clone()
        } else {
            level_db
        },
    };
    StrikeFitReport {
        strike,
        keys: per_key_db.len(),
        recordings: measurements.len(),
        per_key_db,
        per_key_velocity_db,
        flatness_db: median(measurements.iter().map(|m| m.flatness_db)).unwrap_or(0.0),
    }
}

impl StrikeFitReport {
    /// The anchors as `(key, dB)`, for printing.
    pub fn anchors(&self) -> Vec<(u8, f32)> {
        self.strike
            .level_db
            .iter()
            .map(|a: &NoiseAnchor| (a.key, a.db))
            .collect()
    }
}

/// The level at the nominal velocity and the slope through it, in dB per unit of
/// drive, from one key's layers.
fn velocity_line(layers: &[&AttackResidual], config: &AttackConfig) -> Option<(f64, f64)> {
    let points: Vec<(f64, f64)> = layers
        .iter()
        .filter(|m| m.level_db.is_finite())
        .map(|m| (f64::from(m.midi_velocity) / 127.0, m.level_db))
        .collect();
    let nominal = f64::from(NOMINAL_STRIKE_VELOCITY) / 127.0;
    if points.len() < config.min_layers.max(2) {
        return None;
    }
    let basis: Vec<f64> = points.iter().flat_map(|&(x, _)| [1.0, x - nominal]).collect();
    let y: Vec<f64> = points.iter().map(|&(_, db)| db).collect();
    let solution = weighted_least_squares(&basis, &y, &vec![1.0; points.len()], 2)?;
    Some((solution[0], solution[1]))
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

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f64 = 48_000.0;

    /// The band comes back where the hole was put, and nowhere else.
    #[test]
    fn the_deficit_band_is_the_run_of_octaves_the_engine_is_short_in() {
        let config = AttackConfig::default();
        // Two flat densities, with the second dug 15 dB deep between 2 and
        // 5 kHz and 3 dB deep — inside the noise — between 400 and 700 Hz.
        let centres: Vec<f64> = {
            let mut hz = f64::from(MIN_STRIKE_BANDWIDTH_HZ);
            let mut out = Vec::new();
            while hz <= f64::from(MAX_STRIKE_BANDWIDTH_HZ) {
                out.push(hz);
                hz *= 2f64.powf(1.0 / 3.0);
            }
            out
        };
        let recorded: Vec<(f64, f64)> = centres.iter().map(|&hz| (hz, 0.0)).collect();
        let engine: Vec<(f64, f64)> = centres
            .iter()
            .map(|&hz| {
                let db = if (2_000.0..=5_000.0).contains(&hz) {
                    -15.0
                } else if (400.0..=700.0).contains(&hz) {
                    -3.0
                } else {
                    0.0
                };
                (hz, db)
            })
            .collect();
        let (lo, hi) = deficit_band(&recorded, &engine, &config).expect("a band");
        assert!(
            (1_800.0..=2_100.0).contains(&lo) && (4_000.0..=5_100.0).contains(&hi),
            "{lo} .. {hi}"
        );
    }

    /// An engine that already has the residual is told so, rather than being
    /// given a band anyway.
    #[test]
    fn nothing_missing_is_no_band() {
        let config = AttackConfig::default();
        let recorded: Vec<(f64, f64)> = (0..12).map(|i| (200.0 * 1.26f64.powi(i), 0.0)).collect();
        let engine: Vec<(f64, f64)> = recorded.iter().map(|&(hz, _)| (hz, -1.0)).collect();
        assert_eq!(deficit_band(&recorded, &engine, &config), None);
    }

    /// The measurement's own band is what narrows, and it never leaves the
    /// schema's range however it is asked to.
    #[test]
    fn a_narrowed_band_stays_inside_the_schemas_own() {
        let narrowed = AttackConfig::default().in_band((10.0, 96_000.0));
        assert_eq!(
            narrowed.band_hz,
            (
                f64::from(MIN_STRIKE_BANDWIDTH_HZ),
                f64::from(MAX_STRIKE_BANDWIDTH_HZ)
            )
        );
        let inverted = AttackConfig::default().in_band((5_000.0, 1_000.0));
        assert!(inverted.band_hz.0 <= inverted.band_hz.1);
    }

    /// A density in dB against its own maximum, which is what makes two renders
    /// at two levels comparable at all.
    #[test]
    fn a_density_in_decibels_is_against_its_own_maximum() {
        let bands = [(200.0, 1.0), (400.0, 4.0), (800.0, 0.04)];
        let db = density_db(&bands);
        assert!((db[1].1 - 0.0).abs() < 1e-9, "{db:?}");
        assert!((db[0].1 + 6.02).abs() < 0.02, "{db:?}");
        assert!((db[2].1 + 20.0).abs() < 0.02, "{db:?}");
    }

    /// A note: partials at `freqs` decaying slowly, plus a broadband burst at
    /// the strike at `noise_peak` relative to the note's own peak.
    fn note(freqs: &[f64], noise_peak: f64, seed: u64, frames: usize) -> (Vec<f32>, f64) {
        let mut state = seed | 1;
        let mut random = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        };
        let mut out = vec![0.0f64; frames];
        // A raised-cosine rise over 30 ms, which is what a real attack does
        // (`REALISM.md`: 13 ms in the engine, 18 in the recordings) with enough
        // margin that the analysis window can follow it. A note that reached
        // full amplitude on the first sample would be a step, and no hopped
        // analysis of any window length can follow a step.
        const RISE_S: f64 = 0.030;
        // The burst falls 40 dB in about 0.58 s, which is the order the fitted
        // events run at. A much faster one would be mostly inside the half
        // window `onset_residual` does not measure.
        const BURST_RATE: f64 = 8.0;
        for (i, &f) in freqs.iter().enumerate() {
            let a = 1.0 / (1.0 + i as f64);
            for (n, x) in out.iter_mut().enumerate() {
                let t = n as f64 / SR;
                let rise = if t >= RISE_S {
                    1.0
                } else {
                    0.5 - 0.5 * (std::f64::consts::PI * t / RISE_S).cos()
                };
                *x += a * rise * (-1.5 * t).exp() * (std::f64::consts::TAU * f * t).sin();
            }
        }
        let tonal_peak = out.iter().fold(0.0f64, |m, &x| m.max(x.abs()));
        for (n, x) in out.iter_mut().enumerate() {
            let t = n as f64 / SR;
            *x += noise_peak * tonal_peak * (-BURST_RATE * t).exp() * random();
        }
        let peak = out.iter().fold(0.0f64, |m, &x| m.max(x.abs()));
        (out.iter().map(|&x| x as f32).collect(), peak)
    }

    /// The residual's own peak in the band the burst is written for, in dB
    /// against the note — [`residual_metrics`]'s level, without its gates, so
    /// that a test about the floor can measure the floor.
    fn residual_level_db(signal: &[f32], freqs: &[f64], reference: f64) -> f64 {
        let config = AttackConfig::default();
        let residual =
            onset_residual(signal, SR, freqs, 0.0, config.residual_s).expect("a residual");
        let band = band_limited_for_test(&residual, SR);
        let peak = band.iter().fold(0.0f64, |m, &x| m.max(x.abs()));
        20.0 * (peak / reference).log10()
    }

    #[test]
    fn a_known_burst_under_a_note_comes_back_decibel_for_decibel_under_a_one_signed_loss() {
        // What a level measurement of this kind can and cannot claim. Three
        // things stand between the burst that was mixed in and the number that
        // comes back, all of them one-signed and all of them properties of the
        // measurement rather than of the estimator: the 200 Hz … 8 kHz band
        // limit does not write what is outside the band the engine plays, the
        // half window `onset_residual` does not measure takes the front off the
        // burst, and the phase-locked projection absorbs whatever noise is
        // coherent at a partial's own frequency. Together they are 4 to 7 dB
        // here, and they are a *bound*: the level written is never hotter than
        // the burst that produced it, which is the direction this milestone
        // requires. What the estimator has to do is move with the level, and
        // that it does decibel for decibel.
        let config = AttackConfig::default();
        let freqs: Vec<f64> = (1..=12).map(|k| 261.6 * f64::from(k)).collect();
        let mut measured: Vec<(f64, f64)> = Vec::new();
        for &asked in &[0.1, 0.3] {
            let (signal, peak) = note(&freqs, asked, 0x51ed_2701, (0.5 * SR) as usize);
            let metrics =
                residual_metrics(60, 90, &signal, SR, &freqs, 0.0, peak, &config).expect("fitted");
            let want = 20.0 * asked.log10();
            let loss = want - metrics.level_db;
            assert!(
                (3.0..7.0).contains(&loss),
                "asked {want:.1} dB, measured {:.1}, a loss of {loss:.1}",
                metrics.level_db
            );
            // A broadband burst reads as a continuum, not as lines.
            assert!(metrics.flatness_db > -12.0, "{metrics:?}");
            assert!(metrics.bandwidth_hz > 2_000.0, "{metrics:?}");
            measured.push((want, metrics.level_db));
        }
        let asked = measured[1].0 - measured[0].0;
        let got = measured[1].1 - measured[0].1;
        assert!(
            (got - asked).abs() < 1.5,
            "{asked:.1} dB of level became {got:.1} dB of measurement"
        );
    }

    #[test]
    fn a_burst_under_the_subtractions_own_floor_is_measured_as_a_bound_not_as_itself() {
        // The module header's honest half, as an assertion. The projection
        // cannot follow an envelope faster than its own window, so what is left
        // of a note whose partials rise in 30 ms is a floor — and a burst mixed
        // *under* that floor comes back at the floor rather than at its own
        // level. The measurement is an upper bound and never an underestimate,
        // which is the property a preset can be built on.
        let freqs: Vec<f64> = (1..=12).map(|k| 261.6 * f64::from(k)).collect();
        let (bare, bare_peak) = note(&freqs, 0.0, 0x51ed_2701, (0.5 * SR) as usize);
        let floor = residual_level_db(&bare, &freqs, bare_peak);
        assert!(floor < -20.0, "the subtraction's own floor is {floor:.1} dB");
        let (quiet, quiet_peak) = note(&freqs, 0.001, 0x51ed_2701, (0.5 * SR) as usize);
        let measured = residual_level_db(&quiet, &freqs, quiet_peak);
        let mixed = 20.0f64 * 0.001f64.log10();
        assert!(measured > mixed, "{measured:.1} against {mixed:.1} dB");
        assert!(
            (measured - floor).abs() < 3.0,
            "{measured:.1} against a floor of {floor:.1}"
        );
    }

    #[test]
    fn a_note_with_no_burst_under_it_is_measured_far_quieter_than_one_with() {
        let freqs: Vec<f64> = (1..=12).map(|k| 261.6 * f64::from(k)).collect();
        let (bare, bare_peak) = note(&freqs, 0.0, 0x51ed_2701, (0.5 * SR) as usize);
        let (noisy, noisy_peak) = note(&freqs, 0.3, 0x51ed_2701, (0.5 * SR) as usize);
        let quiet = residual_level_db(&bare, &freqs, bare_peak);
        let loud = residual_level_db(&noisy, &freqs, noisy_peak);
        assert!(loud - quiet > 12.0, "{loud:.1} vs {quiet:.1} dB");
    }

    fn residual(key: u8, vel: u8, level_db: f64) -> AttackResidual {
        AttackResidual {
            key,
            midi_velocity: vel,
            level_db,
            centroid_hz: 1_300.0,
            bandwidth_hz: 5_500.0,
            decay_s: 0.06,
            flatness_db: -6.0,
        }
    }

    #[test]
    fn the_velocity_law_is_the_slope_the_layers_draw_and_the_level_is_where_it_crosses_ninety() {
        let noise = NoiseConfig::default();
        let base = StrikeNoise::default();
        // Two keys, sixteen layers each, on a line of 30 dB over the full drive
        // range through −18 dB at velocity 90.
        let velocities: Vec<u8> = (0..16).map(|i| 8 + i * 7).collect();
        let measurements: Vec<AttackResidual> = [21u8, 60]
            .into_iter()
            .flat_map(|key| {
                velocities.iter().map(move |&v| {
                    let drive = f64::from(v) / 127.0 - 90.0 / 127.0;
                    residual(key, v, -18.0 + 30.0 * drive)
                })
            })
            .collect();
        let report = fit_strike(&measurements, &base, &AttackConfig::default(), &noise);
        assert_eq!(report.keys, 2);
        assert!(
            (f64::from(report.strike.velocity_db) - 30.0).abs() < 0.1,
            "{:?}",
            report.strike
        );
        assert!(report
            .strike
            .level_db
            .iter()
            .all(|a| (f64::from(a.db) + 18.0).abs() < 0.5));
        assert!((report.strike.centroid_hz - 1_300.0).abs() < 1.0);
        assert!((report.strike.bandwidth_hz - 5_500.0).abs() < 1.0);
        assert!((report.strike.decay_s - 0.06).abs() < 1e-3);
        // Nothing was measured for the strike, so nothing is written: the base
        // event, silence and all, survives.
        let empty = fit_strike(&[], &base, &AttackConfig::default(), &noise);
        assert_eq!(empty.strike.level_db, base.level_db);
    }

    #[test]
    fn a_centroid_above_the_rolloff_is_held_under_it_rather_than_refused() {
        let noise = NoiseConfig::default();
        let base = StrikeNoise::default();
        let measurements: Vec<AttackResidual> = (0..16)
            .map(|i| AttackResidual {
                centroid_hz: 6_000.0,
                bandwidth_hz: 4_000.0,
                ..residual(60, 8 + i * 7, -20.0)
            })
            .collect();
        let report = fit_strike(&measurements, &base, &AttackConfig::default(), &noise);
        assert!(report.strike.centroid_hz <= report.strike.bandwidth_hz);
    }
}

// ---------------------------------------------------------------------------
// The balance: how much of an attack is the mechanism and how much is the string
// ---------------------------------------------------------------------------
//
// Everything above fits `[noise.strike]` from the *recordings'* onset residual
// and then corrects the level on the engine's own render (`strike_offset`,
// `DECISIONS.md` 210-213). What that machinery never closes on is the thing a
// listener actually hears, which is a **ratio**: how loud the hammer is against
// the note it belongs to. The level it writes is referenced to the note's peak,
// and the note's peak is not where the tone the burst competes with lives —
// so a change anywhere else in the engine that moves the *attack's* tonal
// content moves this ratio without moving anything the fit reads.
//
// What follows measures that ratio, per note, against a recording of the same
// note, and inverts it exactly.

/// The statistic: [`crate::realism::attack_tonality_db`] of the first
/// [`crate::realism::ATTACK_WINDOW_S`] from a note's own onset — the arithmetic
/// over the geometric mean of its power spectrum, in dB.
///
/// A line spectrum reads large and positive, a continuum reads zero, so it *is*
/// a noise-to-tone ratio; it is the same number `REALISM.md`'s `attack` column
/// is a mean of, and it needs no level match, no partial subtraction and no
/// model of the note.
pub fn noise_to_tone_db(signal: &[f32], onset_s: f64, sample_rate: f64) -> f64 {
    let len = (crate::realism::ATTACK_WINDOW_S * sample_rate).round() as usize;
    let start = (onset_s * sample_rate).round().max(0.0) as usize;
    let end = start + len;
    if end > signal.len() || len < 64 {
        return f64::NAN;
    }
    crate::realism::attack_tonality_db(&signal[start..end], sample_rate)
}

/// How far either way a level is searched when the balance is inverted, dB.
///
/// Wide enough that the answer is never the bracket: the measured corrections
/// are a few decibels to a few tens, and a note whose recording is outside
/// ±40 dB of what this event can produce is a note the event cannot reach at
/// all, which is a finding ([`BalanceVerdict`]) rather than a number.
pub const BALANCE_REACH_DB: f64 = 40.0;

/// Why one note's balance did or did not invert.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BalanceVerdict {
    /// A level of this event puts the engine on the recording.
    Closed,
    /// The engine is **already noisier than the piano with the event silenced**.
    /// Adding a continuum only lowers tonality further, so no level reaches the
    /// recording and the excess is not this event's.
    Floor,
    /// Even [`BALANCE_REACH_DB`] more of it is not enough: the recording's
    /// attack is noisier than this event can make the engine.
    Ceiling,
}

/// One note's noise-to-tone reading, engine against the recording of the same
/// key.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BalanceReading {
    pub key: u8,
    pub midi_velocity: u8,
    /// The recording's attack tonality, dB.
    pub reference_db: f64,
    /// The engine's, as the preset ships.
    pub engine_db: f64,
    /// The engine's with the event silenced — the tonal attack alone.
    pub tone_db: f64,
    /// The offset on the event's level, in dB, that puts `engine_db` on
    /// `reference_db`.
    pub offset_db: Option<f64>,
    pub verdict: BalanceVerdict,
}

impl BalanceReading {
    /// Drive the reading was taken at, `v / 127`.
    pub fn drive(&self) -> f64 {
        f64::from(self.midi_velocity) / 127.0
    }
}

/// The engine's render with an additive event moved by `db`.
///
/// `tone` is a render with the event silenced and `burst` is the sample-wise
/// difference between that and a render with it at its tabulated level, so
/// `burst` **is** the event through the whole chain — the board's response, the
/// master gain and the burst's own filters included. Every other level of it is
/// then arithmetic, and no further render is needed. This is the property that
/// makes the inversion below exact rather than a search over presets:
/// `engine::voice::Voice::process` adds the noise bus to the string's output
/// and nothing after it is level-dependent at these amplitudes.
pub fn mix(tone: &[f32], burst: &[f32], db: f64) -> Vec<f32> {
    let gain = 10f64.powf(db / 20.0) as f32;
    tone.iter()
        .zip(burst)
        .map(|(&a, &b)| a + gain * b)
        .collect()
}

/// The offset on the event's level that puts the engine's attack tonality on
/// `target`, bisected over [`BALANCE_REACH_DB`].
///
/// Monotone by construction: mixing a continuum into a line spectrum can only
/// lower the arithmetic-over-geometric ratio, so tonality falls as the event
/// rises, and the crossing is unique.
pub fn balance_offset(
    tone: &[f32],
    burst: &[f32],
    onset_s: f64,
    sample_rate: f64,
    target: f64,
) -> Result<f64, BalanceVerdict> {
    let at = |db: f64| noise_to_tone_db(&mix(tone, burst, db), onset_s, sample_rate);
    let (mut lo, mut hi) = (-BALANCE_REACH_DB, BALANCE_REACH_DB);
    let (quiet, loud) = (at(lo), at(hi));
    if !target.is_finite() || !quiet.is_finite() || !loud.is_finite() {
        return Err(BalanceVerdict::Floor);
    }
    if target > quiet {
        return Err(BalanceVerdict::Floor);
    }
    if target < loud {
        return Err(BalanceVerdict::Ceiling);
    }
    // 24 halvings of 80 dB is under a ten-thousandth of a decibel.
    for _ in 0..24 {
        let mid = 0.5 * (lo + hi);
        if at(mid) > target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(0.5 * (lo + hi))
}

/// One note measured and inverted.
#[allow(clippy::too_many_arguments)]
pub fn balance_reading(
    key: u8,
    midi_velocity: u8,
    reference: &[f32],
    reference_onset_s: f64,
    tone: &[f32],
    burst: &[f32],
    engine_onset_s: f64,
    sample_rate: f64,
) -> BalanceReading {
    let reference_db = noise_to_tone_db(reference, reference_onset_s, sample_rate);
    let tone_db = noise_to_tone_db(tone, engine_onset_s, sample_rate);
    let engine_db = noise_to_tone_db(
        &mix(tone, burst, 0.0),
        engine_onset_s,
        sample_rate,
    );
    let (offset_db, verdict) =
        match balance_offset(tone, burst, engine_onset_s, sample_rate, reference_db) {
            Ok(db) => (Some(db), BalanceVerdict::Closed),
            Err(v) => (None, v),
        };
    BalanceReading {
        key,
        midi_velocity,
        reference_db,
        engine_db,
        tone_db,
        offset_db,
        verdict,
    }
}

/// The correction the whole population asks for: a level at the nominal drive
/// and a slope in drive, which are exactly `[noise.strike]`'s two level fields.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BalanceFit {
    /// dB to add to every `level_db` anchor.
    pub level_db: f64,
    /// dB to add to `velocity_db` — the slope of the correction in drive, which
    /// adds to the slope the event already has.
    pub velocity_db: f64,
    /// Notes that inverted, and notes that did not.
    pub closed: usize,
    pub floor: usize,
    pub ceiling: usize,
    /// Robust scatter of the readings about the fitted line, dB.
    pub scatter_db: f64,
}

/// Fits [`BalanceFit`] from the readings of a whole population.
///
/// **Theil-Sen, not least squares.** Two reasons, and both are properties of
/// this material rather than preferences. The readings are censored — a note
/// whose verdict is [`BalanceVerdict::Floor`] or [`BalanceVerdict::Ceiling`]
/// contributes no number at all, and censoring is not symmetric in drive — so
/// an estimator that a handful of extreme points can pull is the wrong one; and
/// the per-key scatter of this quantity is tens of decibels, because how much
/// hammer noise one recorded key has is a property of that key's own recording.
/// The median of the pairwise slopes is unmoved by either.
///
/// `min_notes` is the fewest readings a fit may be made from.
pub fn fit_balance(readings: &[BalanceReading], min_notes: usize) -> Option<BalanceFit> {
    let nominal = f64::from(NOMINAL_STRIKE_VELOCITY) / 127.0;
    let points: Vec<(f64, f64)> = readings
        .iter()
        .filter_map(|r| r.offset_db.map(|db| (r.drive() - nominal, db)))
        .filter(|&(_, db)| db.is_finite())
        .collect();
    if points.len() < min_notes.max(2) {
        return None;
    }
    let (velocity_db, level_db) = crate::estimate::melody::theil_sen(&points);
    let mut residuals: Vec<f64> = points
        .iter()
        .map(|&(x, y)| (y - (level_db + velocity_db * x)).abs())
        .collect();
    residuals.sort_by(f64::total_cmp);
    let scatter_db = 1.4826 * residuals[residuals.len() / 2];
    Some(BalanceFit {
        level_db,
        velocity_db,
        closed: points.len(),
        floor: readings
            .iter()
            .filter(|r| r.verdict == BalanceVerdict::Floor)
            .count(),
        ceiling: readings
            .iter()
            .filter(|r| r.verdict == BalanceVerdict::Ceiling)
            .count(),
        scatter_db,
    })
}

#[cfg(test)]
mod balance_tests {
    use super::*;

    const SR: f64 = 48_000.0;

    /// A line spectrum: eight harmonics of C4, unit amplitude.
    fn tone(len: usize) -> Vec<f32> {
        (0..len)
            .map(|n| {
                let t = n as f64 / SR;
                (1..=8)
                    .map(|k| (std::f64::consts::TAU * 261.6 * f64::from(k) * t).sin())
                    .sum::<f64>() as f32
                    * 0.1
            })
            .collect()
    }

    /// A continuum: a deterministic white sequence, at unit scale.
    fn burst(len: usize) -> Vec<f32> {
        let mut state = 0x9e37_79b9u32;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 8) as f32 / 8_388_608.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn a_mix_scales_the_event_and_leaves_the_tone_alone() {
        let (t, b) = (tone(1024), burst(1024));
        let at_zero = mix(&t, &b, 0.0);
        for ((&m, &a), &c) in at_zero.iter().zip(&t).zip(&b) {
            assert!((m - (a + c)).abs() < 1e-6);
        }
        // Six decibels is a factor of two on an amplitude.
        let louder = mix(&t, &b, 20.0 * 2f64.log10());
        for ((&m, &a), &c) in louder.iter().zip(&t).zip(&b) {
            assert!((m - (a + 2.0 * c)).abs() < 1e-5, "{m} vs {}", a + 2.0 * c);
        }
        // And forty decibels down is the tone to five figures.
        let quiet = mix(&t, &b, -100.0);
        for (&m, &a) in quiet.iter().zip(&t) {
            assert!((m - a).abs() < 1e-4);
        }
    }

    #[test]
    fn adding_a_continuum_to_a_line_spectrum_only_lowers_the_tonality() {
        let len = (0.030 * SR) as usize + 16;
        let (t, b) = (tone(len), burst(len));
        let mut last = f64::INFINITY;
        for db in [-60.0, -50.0, -40.0, -30.0, -20.0, -10.0, 0.0] {
            let now = noise_to_tone_db(&mix(&t, &b, db), 0.0, SR);
            assert!(
                now < last,
                "tonality rose from {last:.2} to {now:.2} at {db} dB of event"
            );
            last = now;
        }
    }

    #[test]
    fn the_balance_recovers_the_level_a_note_was_built_with() {
        let len = (0.030 * SR) as usize + 16;
        let (t, b) = (tone(len), burst(len));
        // Three notes, each built with a known amount of the event in it. The
        // inversion sees only the two components and the finished tonality.
        for planted in [-30.0, -18.0, -6.0] {
            let target = noise_to_tone_db(&mix(&t, &b, planted), 0.0, SR);
            let found = balance_offset(&t, &b, 0.0, SR, target).expect("a reachable level");
            assert!(
                (found - planted).abs() < 0.05,
                "planted {planted:.2} dB, recovered {found:.2} dB"
            );
        }
    }

    /// The property that makes the stage re-entrant: an engine already on the
    /// recording asks for nothing. Run the tool over its own output and the
    /// correction is zero, so a preset cannot ratchet.
    #[test]
    fn the_balance_is_a_fixed_point() {
        let len = (0.030 * SR) as usize + 16;
        let (t, b) = (tone(len), burst(len));
        let engine = mix(&t, &b, 0.0);
        let target = noise_to_tone_db(&engine, 0.0, SR);
        let again = balance_offset(&t, &b, 0.0, SR, target).expect("its own level is reachable");
        assert!(again.abs() < 1e-3, "a second pass asked for {again:+.4} dB");

        // And one pass lands where it aimed: move the event by the offset the
        // inversion returns and the tonality is the target, not near it.
        let wrong = mix(&t, &b, 9.0);
        let want = noise_to_tone_db(&wrong, 0.0, SR);
        let step = balance_offset(&t, &b, 0.0, SR, want).expect("reachable");
        let landed = noise_to_tone_db(&mix(&t, &b, step), 0.0, SR);
        assert!((landed - want).abs() < 0.01, "{landed:.3} against {want:.3}");
    }

    #[test]
    fn a_recording_this_event_cannot_reach_is_refused_from_either_side() {
        let len = (0.030 * SR) as usize + 16;
        let (t, b) = (tone(len), burst(len));
        // More tonal than the tone alone: no amount of continuum gets there,
        // and the excess is not this event's.
        let bare = noise_to_tone_db(&t, 0.0, SR);
        assert_eq!(
            balance_offset(&t, &b, 0.0, SR, bare + 6.0),
            Err(BalanceVerdict::Floor)
        );
        // Noisier than the loudest legal event makes it.
        let loudest = noise_to_tone_db(&mix(&t, &b, BALANCE_REACH_DB), 0.0, SR);
        assert_eq!(
            balance_offset(&t, &b, 0.0, SR, loudest - 6.0),
            Err(BalanceVerdict::Ceiling)
        );
    }

    fn reading(key: u8, vel: u8, offset: Option<f64>) -> BalanceReading {
        BalanceReading {
            key,
            midi_velocity: vel,
            reference_db: 30.0,
            engine_db: 25.0,
            tone_db: 35.0,
            offset_db: offset,
            verdict: if offset.is_some() {
                BalanceVerdict::Closed
            } else {
                BalanceVerdict::Floor
            },
        }
    }

    #[test]
    fn the_correction_is_the_level_at_the_nominal_drive_and_the_slope_through_it() {
        let nominal = f64::from(NOMINAL_STRIKE_VELOCITY) / 127.0;
        // A population on an exact line: −8 dB at the nominal drive, rising
        // 20 dB per unit of drive.
        let readings: Vec<BalanceReading> = [21u8, 45, 60, 84, 108]
            .into_iter()
            .flat_map(|key| {
                [24u8, 48, 72, 88, 110].into_iter().map(move |vel| {
                    let drive = f64::from(vel) / 127.0 - nominal;
                    reading(key, vel, Some(-8.0 + 20.0 * drive))
                })
            })
            .collect();
        let fit = fit_balance(&readings, 10).expect("enough readings");
        assert!((fit.level_db + 8.0).abs() < 1e-6, "{fit:?}");
        assert!((fit.velocity_db - 20.0).abs() < 1e-6, "{fit:?}");
        assert_eq!(fit.closed, 25);
        assert!(fit.scatter_db < 1e-9);

        // Theil-Sen, so a fifth of the population thrown far off the line moves
        // neither number: this material's per-key scatter is tens of decibels
        // and least squares would follow it.
        let mut bent = readings.clone();
        for r in bent.iter_mut().filter(|r| r.key == 21) {
            r.offset_db = Some(r.offset_db.unwrap() + 40.0);
        }
        let bent = fit_balance(&bent, 10).expect("enough readings");
        assert!((bent.level_db + 8.0).abs() < 1e-6, "{bent:?}");
        assert!((bent.velocity_db - 20.0).abs() < 1e-6, "{bent:?}");

        // Readings that did not invert are counted and not fitted.
        let refused: Vec<BalanceReading> = readings
            .iter()
            .map(|r| reading(r.key, r.midi_velocity, None))
            .collect();
        assert!(fit_balance(&refused, 10).is_none());
        let mut half = readings.clone();
        half.extend(refused);
        let half = fit_balance(&half, 10).expect("enough readings");
        assert_eq!((half.closed, half.floor), (25, 25));
    }
}
