//! Robust fit of the stiff-string partial layout `f_k = k f0 sqrt(1 + B k^2)`
//! to the measured partial frequencies.
//!
//! The model is nonlinear in `(f0, B)` but linear after one substitution:
//! squaring and dividing by `k^2` gives
//!
//! ```text
//!     (f_k / k)^2 = f0^2 + f0^2 B k^2,
//! ```
//! a straight line in `k^2` whose intercept is `f0^2` and whose slope over the
//! intercept is `B`. So the fit is a weighted least-squares line, and all the
//! work is in being robust: one partial mis-tracked into a neighbour's peak, or
//! one sitting on a soundboard resonance that pulled it, would otherwise drag
//! `B` by more than the 2 % the pipeline needs.
//!
//! Robustness comes in two layers: a Theil-Sen line (median of pairwise slopes,
//! ~29 % breakdown) to start from, then iterated rejection of partials more
//! than `reject_cents` off the current model, refitting until the accepted set
//! stops changing.

use crate::error::{Error, Result};
use crate::estimate::level_floor;
use crate::numeric::{median, weighted_least_squares};
use crate::trajectory::{InharmonicModel, NoteTrajectories};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InharmonicConfig {
    /// Partials above this index are ignored. High partials have the longest
    /// lever on `B`, but they are also where the tracker is least sure and
    /// where a real string stops obeying the model.
    pub max_partial: u32,
    /// A track with fewer measurements than this is not trusted to have a
    /// frequency at all.
    pub min_points: usize,
    /// How far below the loudest partial a track may be and still be used, in
    /// dB. Above the note's highest real partial the tracker keeps looking, and
    /// what it finds there is the noise floor's own peaks — which have
    /// frequencies, and would otherwise be fitted as if they were partials.
    pub min_level_db: f64,
    /// How far below a track's own peak its measurements are still used, in dB.
    /// A partial goes on being tracked after it has decayed into the noise, and
    /// the frequencies it reports from down there are the noise's.
    pub range_db: f64,
    /// Fewest accepted partials the fit will report a result from. Two would
    /// determine the line; four leaves the rejection something to work with.
    pub min_partials: usize,
    /// A partial further than this from the current model is rejected and the
    /// line refitted without it.
    pub reject_cents: f64,
    pub max_iterations: usize,
}

impl Default for InharmonicConfig {
    fn default() -> Self {
        Self {
            max_partial: 40,
            min_points: 3,
            min_level_db: 60.0,
            range_db: 40.0,
            min_partials: 4,
            reject_cents: 20.0,
            max_iterations: 8,
        }
    }
}

#[derive(Clone, Debug)]
pub struct InharmonicFit {
    pub model: InharmonicModel,
    /// Partial indices the final line was fitted to, ascending.
    pub used: Vec<u32>,
    /// Partials the rejection threw out, with how far off they were.
    pub rejected: Vec<(u32, f64)>,
    /// RMS deviation of the accepted partials from the fitted model, in cents.
    pub residual_cents: f64,
    pub worst_cents: f64,
}

/// Fits `(f0, B)` to a note's tracked partials.
///
/// Each partial contributes one frequency and one uncertainty, both measured
/// from its own track: see [`measure_partial`].
pub fn fit_inharmonic(
    trajectories: &NoteTrajectories,
    config: &InharmonicConfig,
) -> Result<InharmonicFit> {
    let floor = level_floor(trajectories, config.min_level_db);
    let partials: Vec<(u32, f64, f64)> = trajectories
        .tracks
        .iter()
        .filter(|track| track.k >= 1 && track.k <= config.max_partial)
        .filter(|track| track.len() >= config.min_points)
        .filter(|track| track.peak().is_some_and(|peak| peak.amplitude >= floor))
        .filter_map(|track| {
            measure_partial(track, config.range_db)
                .map(|(frequency, variance)| (track.k, frequency, variance))
        })
        .collect();
    fit_measured_partials(&partials, config)
}

/// The same fit from bare `(k, f_k)` pairs, with nothing known about how well
/// each was measured.
pub fn fit_inharmonic_partials(
    partials: &[(u32, f64)],
    config: &InharmonicConfig,
) -> Result<InharmonicFit> {
    let measured: Vec<(u32, f64, f64)> =
        partials.iter().map(|&(k, f)| (k, f, 0.0)).collect();
    fit_measured_partials(&measured, config)
}

/// The fit proper: `(k, f_k, variance of f_k)` in, `(f0, B)` out.
pub fn fit_measured_partials(
    partials: &[(u32, f64, f64)],
    config: &InharmonicConfig,
) -> Result<InharmonicFit> {
    let partials: Vec<&(u32, f64, f64)> = partials
        .iter()
        .filter(|&&(k, f, _)| k >= 1 && f > 0.0)
        .collect();
    if partials.len() < config.min_partials {
        return Err(Error::Estimate(format!(
            "inharmonicity fit needs {} partials, got {}",
            config.min_partials,
            partials.len()
        )));
    }
    // Floor on a partial's frequency variance. Perfect data would otherwise
    // divide by zero, and no peak in a windowed spectrum is located better than
    // this anyway.
    const VARIANCE_FLOOR: f64 = 1e-8;
    // The linearized observation: y = (f_k / k)^2 against x = k^2.
    let samples: Vec<Sample> = partials
        .iter()
        .map(|&&(k, f, variance)| {
            let kf = f64::from(k);
            // An error in f becomes an error in y of 2 (f_k / k^2) times it, so
            // the weight is the reciprocal of that squared: inverse-variance
            // weighting in the space the line is fitted in. It is also why the
            // high partials, whose lever on B is longest, count most — but only
            // to the extent that they were measured as well.
            let jacobian = 2.0 * f / (kf * kf);
            Sample {
                k,
                frequency_hz: f,
                x: kf * kf,
                y: (f / kf) * (f / kf),
                weight: 1.0 / (jacobian * jacobian * variance.max(VARIANCE_FLOOR)),
            }
        })
        .collect();

    let (mut intercept, mut slope) = theil_sen(&samples)?;
    let mut used: Vec<usize> = (0..samples.len()).collect();
    for _ in 0..config.max_iterations {
        let model = model_from_line(intercept, slope)?;
        let accepted: Vec<usize> = (0..samples.len())
            .filter(|&i| {
                model
                    .cents_from_partial(samples[i].k, samples[i].frequency_hz)
                    .abs()
                    <= config.reject_cents
            })
            .collect();
        if accepted.len() < config.min_partials {
            break;
        }
        let converged = accepted == used;
        used = accepted;
        let (new_intercept, new_slope) = weighted_line(&samples, &used)?;
        let moved = (new_intercept - intercept).abs() > 1e-12 * intercept.abs()
            || (new_slope - slope).abs() > 1e-12 * slope.abs().max(1e-12);
        intercept = new_intercept;
        slope = new_slope;
        if converged && !moved {
            break;
        }
    }

    let model = model_from_line(intercept, slope)?;
    let mut residual = 0.0;
    let mut worst: f64 = 0.0;
    for &i in &used {
        let error = model.cents_from_partial(samples[i].k, samples[i].frequency_hz);
        residual += error * error;
        worst = worst.max(error.abs());
    }
    let rejected: Vec<(u32, f64)> = (0..samples.len())
        .filter(|i| !used.contains(i))
        .map(|i| {
            (
                samples[i].k,
                model.cents_from_partial(samples[i].k, samples[i].frequency_hz),
            )
        })
        .collect();

    Ok(InharmonicFit {
        model,
        used: used.iter().map(|&i| samples[i].k).collect(),
        rejected,
        residual_cents: (residual / used.len() as f64).sqrt(),
        worst_cents: worst,
    })
}

/// One partial's frequency and the variance of that estimate, from its own
/// track: the amplitude-weighted mean over the frames within `range_db` of the
/// track's peak, and the scatter of those frames about it.
///
/// Weighting by amplitude *squared* is the inverse-variance weighting for this
/// measurement: a windowed peak's frequency error scales as the reciprocal of
/// its signal-to-noise ratio, so its variance scales as the reciprocal of its
/// power. The scatter matters as much as the mean — it is what tells the fit
/// that the twentieth partial, tracked while decaying into the noise, is worth
/// less than the second, and without it a run of noisy high partials drags `B`.
pub fn measure_partial(track: &crate::trajectory::PartialTrack, range_db: f64) -> Option<(f64, f64)> {
    let peak = track.peak()?.amplitude;
    let floor = peak * 10f64.powf(-range_db / 20.0);
    let (mut sum_w, mut sum_w2, mut sum_wf) = (0.0, 0.0, 0.0);
    for point in &track.points {
        if point.amplitude < floor || point.frequency_hz <= 0.0 {
            continue;
        }
        let w = point.amplitude * point.amplitude;
        sum_w += w;
        sum_w2 += w * w;
        sum_wf += w * point.frequency_hz;
    }
    if sum_w <= 0.0 {
        return None;
    }
    let mean = sum_wf / sum_w;
    let mut scatter = 0.0;
    for point in &track.points {
        if point.amplitude < floor || point.frequency_hz <= 0.0 {
            continue;
        }
        let w = point.amplitude * point.amplitude;
        scatter += w * (point.frequency_hz - mean).powi(2);
    }
    // Effective sample size of a weighted mean; the variance of the mean is the
    // weighted variance divided by it.
    let effective = sum_w * sum_w / sum_w2;
    let variance = if effective > 1.0 {
        (scatter / sum_w) / effective
    } else {
        scatter / sum_w
    };
    Some((mean, variance))
}

struct Sample {
    k: u32,
    frequency_hz: f64,
    x: f64,
    y: f64,
    weight: f64,
}

/// `y = intercept + slope x` with `intercept = f0^2` and `slope = f0^2 B`.
fn model_from_line(intercept: f64, slope: f64) -> Result<InharmonicModel> {
    if !(intercept.is_finite() && slope.is_finite()) || intercept <= 0.0 {
        return Err(Error::Estimate(
            "inharmonicity fit produced a non-positive f0^2".into(),
        ));
    }
    // A negative slope is a negative B: a string whose partials converge, which
    // no string does. It means the data is noise-dominated; report the
    // harmonic layout rather than a square root of a negative number.
    Ok(InharmonicModel::new(
        intercept.sqrt(),
        (slope / intercept).max(0.0),
    ))
}

/// Median of the pairwise slopes, and the median residual as intercept. The
/// starting line for the rejection loop: it survives a minority of the partials
/// being wrong, which a least-squares line does not.
fn theil_sen(samples: &[Sample]) -> Result<(f64, f64)> {
    let mut slopes = Vec::with_capacity(samples.len() * (samples.len() - 1) / 2);
    for (i, a) in samples.iter().enumerate() {
        for b in &samples[i + 1..] {
            if (b.x - a.x).abs() > 1e-12 {
                slopes.push((b.y - a.y) / (b.x - a.x));
            }
        }
    }
    let slope = median(&slopes)
        .ok_or_else(|| Error::Estimate("inharmonicity fit needs distinct partials".into()))?;
    let intercepts: Vec<f64> = samples.iter().map(|s| s.y - slope * s.x).collect();
    let intercept = median(&intercepts).expect("samples are non-empty");
    Ok((intercept, slope))
}

fn weighted_line(samples: &[Sample], used: &[usize]) -> Result<(f64, f64)> {
    let mut basis = Vec::with_capacity(used.len() * 2);
    let mut y = Vec::with_capacity(used.len());
    let mut weights = Vec::with_capacity(used.len());
    for &i in used {
        basis.push(1.0);
        basis.push(samples[i].x);
        y.push(samples[i].y);
        weights.push(samples[i].weight);
    }
    let solution = weighted_least_squares(&basis, &y, &weights, 2)
        .ok_or_else(|| Error::Estimate("inharmonicity fit is singular".into()))?;
    Ok((solution[0], solution[1]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn partials(model: InharmonicModel, count: u32) -> Vec<(u32, f64)> {
        (1..=count).map(|k| (k, model.partial(k))).collect()
    }

    #[test]
    fn an_exact_partial_series_is_inverted_exactly() {
        let truth = InharmonicModel::new(261.626, 4.1e-4);
        let fit = fit_inharmonic_partials(&partials(truth, 24), &InharmonicConfig::default())
            .unwrap();
        assert!((fit.model.f0_hz - truth.f0_hz).abs() < 1e-6, "{fit:?}");
        assert!((fit.model.b - truth.b).abs() < 1e-9, "{fit:?}");
        assert!(fit.residual_cents < 1e-6);
        assert!(fit.rejected.is_empty());
    }

    #[test]
    fn b_survives_tracker_grade_frequency_noise() {
        // 0.05 Hz of error on every partial, the tracker's measured worst case,
        // in a deterministic zig-zag so the fit cannot average it away.
        let truth = InharmonicModel::new(110.31, 3.7e-4);
        let mut measured = partials(truth, 20);
        for (index, point) in measured.iter_mut().enumerate() {
            point.1 += if index % 2 == 0 { 0.05 } else { -0.05 };
        }
        let fit = fit_inharmonic_partials(&measured, &InharmonicConfig::default()).unwrap();
        assert!(
            (fit.model.b / truth.b - 1.0).abs() < 0.02,
            "B off by {:.3} %: {fit:?}",
            100.0 * (fit.model.b / truth.b - 1.0)
        );
        assert!((fit.model.f0_hz / truth.f0_hz - 1.0).abs() < 1e-3, "{fit:?}");
    }

    #[test]
    fn a_mistracked_partial_is_rejected_rather_than_fitted() {
        let truth = InharmonicModel::new(220.0, 8e-4);
        let mut measured = partials(truth, 16);
        // Partial 9 caught its neighbour's peak: a semitone out.
        measured[8].1 *= 2f64.powf(1.0 / 12.0);
        let fit = fit_inharmonic_partials(&measured, &InharmonicConfig::default()).unwrap();
        assert_eq!(fit.rejected.iter().map(|r| r.0).collect::<Vec<_>>(), vec![9]);
        assert!((fit.model.b / truth.b - 1.0).abs() < 0.02, "{fit:?}");
        assert!((fit.model.f0_hz - truth.f0_hz).abs() < 1e-4, "{fit:?}");
    }

    #[test]
    fn a_harmonic_series_fits_zero_inharmonicity() {
        let truth = InharmonicModel::harmonic(440.0);
        let fit = fit_inharmonic_partials(&partials(truth, 12), &InharmonicConfig::default())
            .unwrap();
        assert!(fit.model.b.abs() < 1e-9, "{fit:?}");
        assert!((fit.model.f0_hz - 440.0).abs() < 1e-6);
    }

    #[test]
    fn too_few_partials_is_an_error_and_not_a_guess() {
        let truth = InharmonicModel::new(440.0, 1e-3);
        assert!(fit_inharmonic_partials(&partials(truth, 3), &InharmonicConfig::default()).is_err());
    }
}
