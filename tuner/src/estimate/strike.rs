//! Strike position from the comb it puts in the excitation spectrum.
//!
//! A hammer striking at a fraction `x` of the speaking length excites mode `k`
//! in proportion to `sin(k pi x)`: the modes with a node at the strike point
//! are not excited at all. The measured time-zero amplitudes are therefore a
//! smooth spectral envelope — the hammer's pulse, the bridge, the microphone —
//! multiplied by a comb whose nulls sit at `k = 1/x, 2/x, ...`. The envelope is
//! unknown and uninteresting; the nulls are the measurement, and their spacing
//! gives `x` directly.
//!
//! The fit is therefore variable projection over a grid of `x`: for each
//! candidate, divide out its comb and ask how smooth what is left is. The
//! smoothness model is a low-order polynomial in `ln k` — enough to absorb any
//! spectral tilt or gentle curvature, far too stiff to absorb a comb.
//!
//! Two details keep it honest:
//!
//! * The comb is softened to `sqrt(sin^2 + floor^2)`. A real null is never
//!   empty: the hammer is not a point, the string is stiff, and the recording
//!   has a noise floor. Without a floor the model would predict minus infinity
//!   decibels and the fit would be driven entirely by whichever partial came
//!   closest to a node.
//! * The residual is Huberized. One partial sitting on a soundboard resonance
//!   is worth a bounded amount of evidence, not an unbounded one.

use crate::error::{Error, Result};
use crate::numeric::{golden_section, weighted_polyfit, poly_eval};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StrikeConfig {
    /// Search range, as a fraction of the speaking length. `sin(k pi x)` is
    /// symmetric about the middle of the string, so `x` and `1 - x` are the
    /// same spectrum and only the lower half is identifiable. Real hammers
    /// strike between 1/7 and 1/12 of the way along.
    pub min_position: f64,
    pub max_position: f64,
    /// Candidates on the search grid. Fine enough that the grid's own minimum
    /// is inside the basin of the true one, which is all it has to be — a local
    /// refinement finishes the job.
    pub grid: usize,
    /// Degree of the polynomial in `ln k` standing in for the spectral
    /// envelope.
    pub envelope_degree: usize,
    /// Floor under the comb, as a fraction of its peak.
    pub null_floor: f64,
    /// Huber threshold on the log-domain residual. 0.7 nepers is 6 dB.
    pub huber_delta: f64,
    pub irls_iterations: usize,
    /// Fewest partials the fit needs. Below about eight there is no null in
    /// range for a realistic strike point and nothing to measure.
    pub min_partials: usize,
}

impl Default for StrikeConfig {
    fn default() -> Self {
        Self {
            min_position: 0.03,
            max_position: 0.5,
            grid: 940,
            envelope_degree: 2,
            null_floor: 0.05,
            huber_delta: 0.7,
            irls_iterations: 4,
            min_partials: 8,
        }
    }
}

#[derive(Clone, Debug)]
pub struct StrikeFit {
    /// Strike point as a fraction of the speaking length.
    pub position: f64,
    /// Coefficients of the fitted `ln` envelope in ascending powers of `ln k`.
    pub envelope: Vec<f64>,
    /// RMS of the fit residual, in dB.
    pub residual_db: f64,
    pub partials: usize,
    null_floor: f64,
}

impl StrikeFit {
    /// The amplitude this fit predicts for partial `k`.
    pub fn amplitude(&self, k: u32) -> f64 {
        let kf = f64::from(k);
        poly_eval(&self.envelope, kf.ln()).exp() * comb(kf, self.position, self.null_floor)
    }

    /// The smooth part alone: the excitation spectrum with the strike comb
    /// divided out, which is what the hammer estimator wants to see.
    pub fn envelope_at(&self, k: u32) -> f64 {
        poly_eval(&self.envelope, f64::from(k).ln()).exp()
    }

    /// The fitted comb at partial `k`: the factor the hammer estimator divides
    /// out of a measured amplitude to see the hammer's own spectrum.
    pub fn comb_at(&self, k: u32) -> f64 {
        comb(f64::from(k), self.position, self.null_floor)
    }
}

/// Fits `x` to the time-zero amplitudes `(k, a_k)` of one note.
pub fn fit_strike_position(spectrum: &[(u32, f64)], config: &StrikeConfig) -> Result<StrikeFit> {
    let points: Vec<(f64, f64)> = spectrum
        .iter()
        .filter(|&&(k, a)| k >= 1 && a > 0.0 && a.is_finite())
        .map(|&(k, a)| (f64::from(k), a.ln()))
        .collect();
    if points.len() < config.min_partials {
        return Err(Error::Estimate(format!(
            "strike position needs {} partials, got {}",
            config.min_partials,
            points.len()
        )));
    }
    if config.max_position <= config.min_position || config.grid < 2 {
        return Err(Error::Config("strike search range is empty".into()));
    }

    let step = (config.max_position - config.min_position) / config.grid as f64;
    let objective = |x: f64| residual_at(&points, x, config).map_or(f64::MAX, |fit| fit.0);
    let mut best = config.min_position;
    let mut best_value = f64::MAX;
    for i in 0..=config.grid {
        let x = config.min_position + i as f64 * step;
        let value = objective(x);
        if value < best_value {
            best_value = value;
            best = x;
        }
    }
    // Refine inside the winning cell. The objective is smooth there — the comb
    // moves continuously with x — so a golden section converges to the exact
    // minimum in a few dozen evaluations.
    let (position, _) = golden_section(
        (best - step).max(config.min_position),
        (best + step).min(config.max_position),
        60,
        objective,
    );

    let (_, envelope, rms) = residual_at(&points, position, config)
        .ok_or_else(|| Error::Estimate("strike-position envelope fit is singular".into()))?;
    Ok(StrikeFit {
        position,
        envelope,
        residual_db: 8.685_889_638_065_035 * rms,
        partials: points.len(),
        null_floor: config.null_floor,
    })
}

/// The softened comb: `|sin(k pi x)|` with a floor under its nulls.
fn comb(k: f64, x: f64, floor: f64) -> f64 {
    let s = (k * std::f64::consts::PI * x).sin();
    (s * s + floor * floor).sqrt()
}

/// Huber loss of the best smooth envelope through the comb-corrected spectrum,
/// with that envelope and the plain RMS residual alongside it.
fn residual_at(
    points: &[(f64, f64)],
    x: f64,
    config: &StrikeConfig,
) -> Option<(f64, Vec<f64>, f64)> {
    let log_k: Vec<f64> = points.iter().map(|&(k, _)| k.ln()).collect();
    let corrected: Vec<f64> = points
        .iter()
        .map(|&(k, log_a)| log_a - comb(k, x, config.null_floor).ln())
        .collect();

    let mut weights = vec![1.0; points.len()];
    let mut envelope = weighted_polyfit(&log_k, &corrected, &weights, config.envelope_degree)?;
    for _ in 1..config.irls_iterations.max(1) {
        for (i, weight) in weights.iter_mut().enumerate() {
            let residual = (corrected[i] - poly_eval(&envelope, log_k[i])).abs();
            *weight = if residual <= config.huber_delta {
                1.0
            } else {
                config.huber_delta / residual
            };
        }
        envelope = weighted_polyfit(&log_k, &corrected, &weights, config.envelope_degree)?;
    }

    let mut loss = 0.0;
    let mut square = 0.0;
    for (i, &value) in corrected.iter().enumerate() {
        let residual = value - poly_eval(&envelope, log_k[i]);
        square += residual * residual;
        let absolute = residual.abs();
        loss += if absolute <= config.huber_delta {
            0.5 * residual * residual
        } else {
            config.huber_delta * (absolute - 0.5 * config.huber_delta)
        };
    }
    let n = points.len() as f64;
    Some((loss / n, envelope, (square / n).sqrt()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The spectrum a strike at `x` produces from an envelope falling as
    /// `k^-1.4`, which is roughly a mezzo-forte hammer's.
    fn spectrum(x: f64, count: u32, floor: f64) -> Vec<(u32, f64)> {
        (1..=count)
            .map(|k| {
                let kf = f64::from(k);
                (k, kf.powf(-1.4) * comb(kf, x, floor))
            })
            .collect()
    }

    #[test]
    fn the_comb_gives_up_the_strike_point_to_better_than_a_percent() {
        let config = StrikeConfig::default();
        for &truth in &[0.075, 0.115, 0.14, 0.22] {
            let fit = fit_strike_position(&spectrum(truth, 40, config.null_floor), &config).unwrap();
            assert!(
                (fit.position / truth - 1.0).abs() < 0.01,
                "x = {} from {truth}: {fit:?}",
                fit.position
            );
        }
    }

    #[test]
    fn a_deeper_null_than_the_model_expects_still_lands_inside_five_percent() {
        // The data's nulls go 20 dB below what the fit's floor allows, and the
        // envelope is not the power law the fit assumes either.
        let config = StrikeConfig::default();
        let truth = 0.118;
        let measured: Vec<(u32, f64)> = (1..=32)
            .map(|k| {
                let kf = f64::from(k);
                let envelope = kf.powf(-1.2) * (1.0 + 0.3 * (kf / 9.0).sin());
                (k, envelope * comb(kf, truth, 0.005))
            })
            .collect();
        let fit = fit_strike_position(&measured, &config).unwrap();
        assert!(
            (fit.position / truth - 1.0).abs() < 0.05,
            "x = {}: {fit:?}",
            fit.position
        );
    }

    #[test]
    fn one_partial_on_a_resonance_does_not_move_the_answer() {
        let config = StrikeConfig::default();
        let truth = 0.125;
        let mut measured = spectrum(truth, 32, config.null_floor);
        measured[10].1 *= 6.0; // +15 dB on partial 11
        let fit = fit_strike_position(&measured, &config).unwrap();
        assert!(
            (fit.position / truth - 1.0).abs() < 0.05,
            "x = {}: {fit:?}",
            fit.position
        );
    }

    #[test]
    fn a_spectrum_with_no_null_in_it_is_refused_rather_than_guessed() {
        // Six partials of a note struck at 1/8: the first null is at k = 8 and
        // was never measured, so there is nothing to fit.
        let config = StrikeConfig::default();
        assert!(fit_strike_position(&spectrum(0.125, 6, config.null_floor), &config).is_err());
    }

    #[test]
    fn the_fitted_spectrum_reproduces_its_input() {
        let config = StrikeConfig::default();
        let measured = spectrum(0.13, 30, config.null_floor);
        let fit = fit_strike_position(&measured, &config).unwrap();
        for &(k, amplitude) in &measured {
            let modelled = fit.amplitude(k);
            assert!(
                (modelled / amplitude - 1.0).abs() < 0.05,
                "k={k}: {modelled} vs {amplitude}"
            );
        }
        assert!(fit.residual_db < 0.5, "{fit:?}");
    }
}
