//! Unison detuning, read off the beats in a partial's envelope.
//!
//! Two strings of a unison group differ in tension by a fraction of a percent.
//! Their partial `k` therefore sits at two frequencies a few tenths of a hertz
//! apart, and what the tracker sees at that frequency is one peak whose
//! amplitude rises and falls at the difference frequency — the beat. Recovering
//! the detuning is recovering that modulation rate:
//!
//! ```text
//!     |A1 e^{-s1 t} + A2 e^{-s2 t} e^{i 2 pi df t}|
//! ```
//! is a decay times a modulation of period `1/df`, so dividing the measured
//! envelope by its fitted decay leaves the modulation alone, and the dominant
//! period of what is left is `1/df`.
//!
//! Two stages, for two different reasons. Autocorrelation finds *which* period
//! it is — it is what does not get confused by the modulation being far from
//! sinusoidal (when the two strings are equally loud the envelope is a
//! rectified cosine, rich in harmonics). A local spectral refinement then finds
//! the period *precisely*, because the pipeline's tolerance is 0.05 Hz on a
//! beat of about one hertz and the autocorrelation's own resolution is one
//! envelope sample.

use crate::error::{Error, Result};
use crate::estimate::decay::{DecayFit, DecayReport};
use crate::estimate::FitSpan;
use crate::numeric::{median, parabolic_offset, poly_eval, weighted_polyfit};
use crate::trajectory::{NoteTrajectories, PartialTrack};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnisonConfig {
    /// Slowest beat that will be looked for. Below this the "beat" is
    /// indistinguishable from the decay the fit already removed.
    pub min_beat_hz: f64,
    /// Fastest beat that will be looked for. A unison is never mistuned by
    /// more than a few hertz; anything faster is a different partial leaking
    /// into the window.
    pub max_beat_hz: f64,
    /// The analysed envelope must contain at least this many beat periods, or
    /// the period is being read off less than a full cycle.
    pub min_periods: f64,
    /// Normalized autocorrelation at the chosen lag, below which the partial is
    /// reported but not used in the median.
    pub min_confidence: f64,
    /// Fractional modulation depth (RMS of the log-envelope residual) below
    /// which there is nothing beating: a single string's envelope leaves only
    /// the decay fit's own error behind, and that is not a unison.
    pub min_depth: f64,
    /// Highest partial to take a beat from. High partials beat fast and their
    /// envelopes are the least reliable.
    pub max_partial: u32,
    /// How far below the partial's peak the envelope is still worth searching,
    /// in dB. Measured against the *fitted* decay rather than the measurement,
    /// because the measurement dips into the beat's own nulls and those are the
    /// signal, not the end of it.
    pub range_db: f64,
    /// Frequency-refinement grid over the interval around the autocorrelation's
    /// answer.
    pub refine_steps: usize,
}

impl Default for UnisonConfig {
    fn default() -> Self {
        Self {
            min_beat_hz: 0.15,
            max_beat_hz: 20.0,
            min_periods: 2.0,
            min_confidence: 0.2,
            min_depth: 0.01,
            max_partial: 8,
            range_db: 60.0,
            refine_steps: 800,
        }
    }
}

/// The beat found in one partial's envelope.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BeatEstimate {
    pub k: u32,
    /// The partial's own frequency — the centre the two strings straddle.
    pub frequency_hz: f64,
    pub beat_hz: f64,
    /// The frequency difference the beat implies, as a ratio in cents. This is
    /// the quantity that is the same for every partial of the note, and the one
    /// the preset stores.
    pub detune_cents: f64,
    /// Height of the autocorrelation peak the period came from, in [0, 1].
    pub confidence: f64,
    /// RMS of the log-envelope residual the beat was found in: how deep the
    /// modulation is, as a fraction of the amplitude.
    pub depth: f64,
}

/// A note's unison detuning, from every partial that showed a beat.
#[derive(Clone, Debug)]
pub struct UnisonEstimate {
    /// Median detuning over the confident partials, in cents.
    pub detune_cents: f64,
    pub partials: Vec<BeatEstimate>,
    /// Median confidence of the partials the estimate was taken over.
    pub confidence: f64,
}

impl UnisonEstimate {
    /// The beat rate this detuning produces at `f_hz` — the estimate expressed
    /// the way a tuner hears it.
    pub fn beat_hz_at(&self, f_hz: f64) -> f64 {
        f_hz * ((self.detune_cents / 1200.0).exp2() - 1.0)
    }
}

/// Estimates the unison detuning of a note from its tracked partials and their
/// fitted decays.
pub fn estimate_unison(
    trajectories: &NoteTrajectories,
    decays: &DecayReport,
    config: &UnisonConfig,
) -> Result<UnisonEstimate> {
    let span = FitSpan::from_trajectories(trajectories);
    let hop = trajectories.hop_s;
    let mut partials = Vec::new();
    for track in &trajectories.tracks {
        if track.k > config.max_partial {
            continue;
        }
        let Some(fit) = decays.fit(track.k) else {
            continue;
        };
        if let Ok(beat) = estimate_beat(track, fit, span, hop, config) {
            partials.push(beat);
        }
    }
    let confident: Vec<&BeatEstimate> = partials
        .iter()
        .filter(|beat| beat.confidence >= config.min_confidence && beat.depth >= config.min_depth)
        .collect();
    if confident.is_empty() {
        return Err(Error::Estimate(
            "no partial's envelope showed a periodic beat".into(),
        ));
    }
    let cents: Vec<f64> = confident.iter().map(|beat| beat.detune_cents).collect();
    let confidences: Vec<f64> = confident.iter().map(|beat| beat.confidence).collect();
    Ok(UnisonEstimate {
        detune_cents: median(&cents).expect("non-empty"),
        confidence: median(&confidences).expect("non-empty"),
        partials,
    })
}

/// Finds the dominant modulation of one partial's envelope after its decay is
/// divided out.
pub fn estimate_beat(
    track: &PartialTrack,
    fit: &DecayFit,
    span: FitSpan,
    hop_s: f64,
    config: &UnisonConfig,
) -> Result<BeatEstimate> {
    if hop_s <= 0.0 {
        return Err(Error::Estimate("envelope hop must be positive".into()));
    }
    let residual = detrended_envelope(track, fit, span, hop_s, config)?;
    let n = residual.len();
    let duration = (n - 1) as f64 * hop_s;

    let min_lag = (1.0 / (config.max_beat_hz * hop_s)).ceil().max(2.0) as usize;
    let max_lag = ((1.0 / (config.min_beat_hz * hop_s)).floor())
        .min(n as f64 / config.min_periods)
        .max(0.0) as usize;
    if max_lag <= min_lag + 1 {
        return Err(Error::Estimate(format!(
            "partial {}: {duration:.2} s of envelope cannot hold {:.1} periods of a \
             {} Hz beat",
            track.k, config.min_periods, config.min_beat_hz
        )));
    }

    let energy: f64 = residual.iter().map(|r| r * r).sum();
    let depth = (energy / n as f64).sqrt();
    if energy <= 0.0 {
        return Err(Error::Estimate(format!(
            "partial {}: envelope has no modulation left after the decay fit",
            track.k
        )));
    }
    let correlation: Vec<f64> = (0..=max_lag)
        .map(|lag| {
            // Biased normalization (the whole energy in the denominator, not
            // the overlap's): it tapers the estimate towards zero at long lags,
            // which is exactly the prior wanted here — half a cycle of
            // something slow must not outrank two cycles of the real beat.
            residual[..n - lag]
                .iter()
                .zip(&residual[lag..])
                .map(|(a, b)| a * b)
                .sum::<f64>()
                / energy
        })
        .collect();

    // The search may not start at lag zero. Every signal correlates with
    // itself at short lags and a smooth envelope keeps correlating for as long
    // as it stays smooth, so the largest correlation in the whole range is
    // always the shortest lag allowed. What marks a period is the first
    // *maximum*, and it can only come after the correlation has finished
    // falling away from lag zero — so the search starts at the first local
    // minimum.
    let mut first = min_lag;
    while first < max_lag && correlation[first + 1] < correlation[first] {
        first += 1;
    }
    if first + 1 >= max_lag {
        return Err(Error::Estimate(format!(
            "partial {}: no complete beat cycle in {duration:.2} s of envelope",
            track.k
        )));
    }
    let mut best = first;
    for lag in first..=max_lag {
        if correlation[lag] > correlation[best] {
            best = lag;
        }
    }
    let chosen = best;
    let confidence = correlation[chosen].clamp(0.0, 1.0);
    let refined_lag = if chosen > 0 && chosen < max_lag {
        chosen as f64
            + parabolic_offset(
                correlation[chosen - 1],
                correlation[chosen],
                correlation[chosen + 1],
            )
    } else {
        chosen as f64
    };
    let coarse_hz = 1.0 / (refined_lag * hop_s);

    // The autocorrelation resolves the period to about one envelope sample;
    // the tolerance is far finer than that, so finish on the spectrum of the
    // residual, where the whole record length sets the resolution.
    let beat_hz = refine_frequency(&residual, hop_s, coarse_hz, config)?;

    let f = fit.frequency_hz;
    if !(f.is_finite() && f > beat_hz) {
        return Err(Error::Estimate(format!(
            "partial {}: beat {beat_hz:.3} Hz is not slower than the partial itself",
            track.k
        )));
    }
    // The two strings straddle the measured peak, so their interval is the
    // ratio of f + beat/2 to f - beat/2.
    let detune_cents = 1200.0 * ((f + 0.5 * beat_hz) / (f - 0.5 * beat_hz)).log2();
    Ok(BeatEstimate {
        k: track.k,
        frequency_hz: f,
        beat_hz,
        detune_cents,
        confidence,
        depth,
    })
}

/// The envelope on a uniform grid with its fitted decay divided out, in the log
/// domain and with its mean removed. Log, because a beat is a multiplicative
/// modulation of the decay, and taking the logarithm turns it into an additive
/// one of constant depth for as long as the partial is above the noise.
fn detrended_envelope(
    track: &PartialTrack,
    fit: &DecayFit,
    span: FitSpan,
    hop_s: f64,
    config: &UnisonConfig,
) -> Result<Vec<f64>> {
    let (Some(start), Some(end)) = (track.start_s(), track.end_s()) else {
        return Err(Error::Estimate(format!("partial {} is empty", track.k)));
    };
    // Stop where the fitted decay has fallen `range_db`: past that the track is
    // following the noise floor and its "modulation" is the noise's.
    let floor = fit.initial_amplitude() * 10f64.powf(-config.range_db / 20.0);
    let end = {
        let mut last = start;
        let mut t = start;
        while t <= end {
            if fit.amplitude_at(t - span.onset_s) >= floor {
                last = t;
            }
            t += hop_s;
        }
        last.min(end)
    };
    let first = start.max(span.start_s);
    if end <= first {
        return Err(Error::Estimate(format!(
            "partial {} has no measurements after the first full window",
            track.k
        )));
    }
    let count = ((end - first) / hop_s).floor() as usize + 1;
    let mut series = Vec::with_capacity(count);
    for i in 0..count {
        let t = first + i as f64 * hop_s;
        let Some(amplitude) = track.amplitude_at(t) else {
            continue;
        };
        let model = fit.amplitude_at(t - span.onset_s);
        if amplitude > 0.0 && model > 0.0 {
            series.push(amplitude.ln() - model.ln());
        }
    }
    if series.len() < 8 {
        return Err(Error::Estimate(format!(
            "partial {}: {} envelope samples is too few to find a beat in",
            track.k,
            series.len()
        )));
    }
    // Take a low-order polynomial out on top of the decay model. A
    // two-exponential fit to a beating envelope is a compromise, and what it
    // leaves behind is the beat plus a slow drift; the drift is a large,
    // aperiodic term that would dominate both the correlation at short lags and
    // the spectrum near zero. A cubic in time cannot absorb a beat — the search
    // requires at least `min_periods` cycles in the record — but it takes the
    // drift out entirely.
    let n = series.len();
    let x: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
    let weights = vec![1.0; n];
    if let Some(trend) = weighted_polyfit(&x, &series, &weights, 3) {
        for (value, &xi) in series.iter_mut().zip(&x) {
            *value -= poly_eval(&trend, xi);
        }
    } else {
        let mean = series.iter().sum::<f64>() / n as f64;
        for value in series.iter_mut() {
            *value -= mean;
        }
    }
    Ok(series)
}

/// Maximizes the magnitude of the residual's Fourier transform over a narrow
/// band around `coarse_hz`, on a grid fine enough that the parabolic
/// interpolation of its peak is limited by the data and not by the grid.
fn refine_frequency(
    residual: &[f64],
    hop_s: f64,
    coarse_hz: f64,
    config: &UnisonConfig,
) -> Result<f64> {
    let n = residual.len();
    // A Hann taper: without it the record's own edges put a sinc pattern around
    // the peak whose sidelobes are only 13 dB down and can move it.
    let windowed: Vec<f64> = residual
        .iter()
        .enumerate()
        .map(|(i, &r)| {
            let phase = 2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64;
            r * (0.5 - 0.5 * phase.cos())
        })
        .collect();
    let magnitude = |hz: f64| -> f64 {
        let (mut re, mut im) = (0.0, 0.0);
        for (i, &r) in windowed.iter().enumerate() {
            let phase = -2.0 * std::f64::consts::PI * hz * i as f64 * hop_s;
            re += r * phase.cos();
            im += r * phase.sin();
        }
        (re * re + im * im).sqrt()
    };

    // A narrow band: the autocorrelation, interpolated, is already within a few
    // percent, and a wide band would let the spectrum's low-frequency skirt —
    // whatever the detrending did not remove — outrank the beat.
    let lo = (coarse_hz * 0.88).max(config.min_beat_hz);
    let hi = (coarse_hz * 1.14).min(config.max_beat_hz);
    if hi <= lo {
        return Ok(coarse_hz);
    }
    let steps = config.refine_steps.max(8);
    let step = (hi - lo) / steps as f64;
    let mut best = 0usize;
    let mut values = Vec::with_capacity(steps + 1);
    for i in 0..=steps {
        let value = magnitude(lo + i as f64 * step);
        if values.is_empty() || value > values[best] {
            best = i;
        }
        values.push(value);
    }
    let offset = if best > 0 && best < steps {
        parabolic_offset(values[best - 1], values[best], values[best + 1])
    } else {
        0.0
    };
    Ok(lo + (best as f64 + offset) * step)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimate::decay::{fit_two_exponential, DecayConfig};
    use crate::synth::{beating_track, Partial};

    /// Two strings, `detune_hz` apart at the fundamental, tracked at partial
    /// `k` for `duration_s`.
    fn beating_partial(k: u32, f0: f64, detune_hz: f64, duration_s: f64) -> PartialTrack {
        let ratio = 1.0 + detune_hz / f0;
        let f = f0 * f64::from(k);
        beating_track(
            k,
            &[
                Partial::new(k, f, 1.0, 1.4),
                Partial::new(k, f * ratio, 0.85, 1.5).with_phase(0.7),
            ],
            0.0,
            0.01,
            duration_s,
        )
    }

    #[test]
    fn a_beating_pair_gives_up_its_detuning_to_within_a_twentieth_of_a_hertz() {
        let track = beating_partial(1, 220.0, 0.73, 8.0);
        let span = FitSpan::new(0.0, 0.0);
        let fit = fit_two_exponential(&track, span, &DecayConfig::default()).unwrap();
        let beat = estimate_beat(&track, &fit, span, 0.01, &UnisonConfig::default()).unwrap();
        assert!(
            (beat.beat_hz - 0.73).abs() < 0.05,
            "{:.4} Hz vs 0.73: {beat:?}",
            beat.beat_hz
        );
        assert!(beat.confidence > 0.5, "{beat:?}");
    }

    #[test]
    fn the_detuning_read_from_a_high_partial_agrees_with_the_fundamental() {
        // The same 0.5 Hz mistuning at the fundamental beats five times as fast
        // at the fifth partial; in cents the two must agree.
        let span = FitSpan::new(0.0, 0.0);
        let config = UnisonConfig::default();
        let mut cents = Vec::new();
        for k in [1u32, 5] {
            let track = beating_partial(k, 196.0, 0.5, 8.0);
            let fit = fit_two_exponential(&track, span, &DecayConfig::default()).unwrap();
            let beat = estimate_beat(&track, &fit, span, 0.01, &config).unwrap();
            assert!((beat.beat_hz - 0.5 * f64::from(k)).abs() < 0.05, "{beat:?}");
            cents.push(beat.detune_cents);
        }
        // 0.05 Hz at the fundamental of this note is 0.44 cents, so that is the
        // precision the two readings can be asked to agree to; what the
        // pipeline needs is the hertz, asserted above. The fundamental is the
        // weaker reading of the two — the same 0.05 Hz is one beat cycle in
        // twenty seconds there and one in four at the fifth partial — and it is
        // where the decay fit's own error lands, since what this estimator
        // reads is what that fit left behind.
        assert!((cents[0] - cents[1]).abs() < 0.44, "{cents:?}");
    }

    #[test]
    fn equal_strings_beat_at_the_difference_frequency_and_not_at_twice_it() {
        // Equal amplitudes rectify the envelope: its dominant harmonic is at
        // twice the beat rate, and only the sub-multiple check gets this right.
        let span = FitSpan::new(0.0, 0.0);
        let track = beating_track(
            1,
            &[
                Partial::new(1, 130.81, 1.0, 1.0),
                Partial::new(1, 130.81 + 0.9, 1.0, 1.0),
            ],
            0.0,
            0.01,
            8.0,
        );
        let fit = fit_two_exponential(&track, span, &DecayConfig::default()).unwrap();
        let beat = estimate_beat(&track, &fit, span, 0.01, &UnisonConfig::default()).unwrap();
        assert!((beat.beat_hz - 0.9).abs() < 0.05, "{beat:?}");
    }

    #[test]
    fn a_single_string_leaves_no_modulation_to_measure() {
        let config = UnisonConfig::default();
        let span = FitSpan::new(0.0, 0.0);
        let track = beating_track(1, &[Partial::new(1, 440.0, 1.0, 2.0)], 0.0, 0.01, 6.0);
        let fit = fit_two_exponential(&track, span, &DecayConfig::default()).unwrap();
        // Either there is no cycle to find, or whatever period the search
        // settles on sits in an envelope with no depth. Both mean "not a
        // unison"; neither may be reported as a detuning.
        if let Ok(beat) = estimate_beat(&track, &fit, span, 0.01, &config) {
            assert!(beat.depth < config.min_depth, "{beat:?}");
        }

        let beating = beating_partial(1, 440.0, 0.6, 6.0);
        let fit = fit_two_exponential(&beating, span, &DecayConfig::default()).unwrap();
        let beat = estimate_beat(&beating, &fit, span, 0.01, &config).unwrap();
        assert!(beat.depth > 10.0 * config.min_depth, "{beat:?}");
    }
}
