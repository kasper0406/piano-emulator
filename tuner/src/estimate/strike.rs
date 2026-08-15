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
//!
//! # The hammer's width
//!
//! `sin(k pi x)` is a *point* force. A real hammer touches 1–2 % of the speaking
//! length, so the excitation is that comb convolved with the contact profile:
//! for a raised-cosine patch of relative width `w` the comb is multiplied by
//! `cos^2(k pi w / 2)`, clamped at zero past `k w = 1` (`PHYSICS.md` §7 — and
//! this is the same taper `engine::string::contact_taper` applies, so what is
//! fitted here is what the engine will play).
//!
//! `w` is fitted as a second parameter of the same variable projection: a grid
//! over `w` around the grid over `x`, then a local refinement of each. Two
//! things make it a real measurement rather than a second envelope parameter:
//!
//! * The taper's signature is *not* smooth droop — a low-order polynomial in
//!   `ln k` absorbs that by construction. It is the cutoff at `k w = 1`, so a
//!   note whose partials do not reach `k ~ 1/w` carries little information
//!   about `w` and the fit says so by not improving.
//! * A nonzero width is therefore only reported when it earns
//!   [`StrikeConfig::min_width_gain`] of the residual. Below that the fit
//!   returns `None` and the preset's table keeps whatever it had, which for a
//!   base preset that never had the field is the point force it always was.

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
    /// Widest contact the search considers, as a fraction of the speaking
    /// length. The engine's own ceiling
    /// (`engine::string::MAX_CONTACT_WIDTH`); a real hammer is at 1–2 %.
    pub max_width: f64,
    /// Candidates on the width grid. Zero disables the width search entirely
    /// and the fit is the point-force one.
    pub width_grid: usize,
    /// How far into the measured spectrum the taper's own null must fall
    /// before a width is considered at all, as a multiple of `1/w`.
    ///
    /// This is what makes `w` identifiable rather than decorative. For
    /// `k w << 1` the taper is `1 - (k pi w / 2)^2` — a smooth droop, which is
    /// precisely what the fitted envelope exists to absorb, and a fit allowed
    /// to use `w` there will happily spend it on the hammer's own spectral
    /// rolloff instead (measured: a 24-partial spectrum with no taper in it at
    /// all gives up 13 % of its residual to a spurious 1.3 % width). Only the
    /// null at `k w = 1`, which no exponential-of-a-polynomial envelope can
    /// make, is evidence about the width — so widths whose null falls above the
    /// highest measured partial are not searched.
    pub min_cutoff_reach: f64,
    /// Fraction of the residual a nonzero width must remove before it is
    /// reported at all. A width is one more free parameter in a fit that
    /// already has a flexible envelope, and on a note whose partials stop well
    /// below `1/w` the two are nearly degenerate: the improvement is then a
    /// fraction of a percent, and what would be written into the preset is the
    /// envelope's rounding error.
    ///
    /// **0.1 until the coupled unison.** One partial is now `2N` eigenmodes
    /// with a derived decay split, so a note's time-zero spectrum carries more
    /// per-partial structure than a point-force comb times a smooth envelope
    /// can hold, and a spurious taper picks some of it up. Measured on C2
    /// through `tuner/tests/calibration.rs`: a render with **no** width earns
    /// **0.131** of the residual with a spurious `w = 0.018`, where the render
    /// with a real `w = 0.03` earns **0.324**. The threshold sits between them.
    pub min_width_gain: f64,
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
            max_width: 0.05,
            width_grid: 20,
            min_width_gain: 0.2,
            min_cutoff_reach: 1.0,
        }
    }
}

/// The engine's contact taper: a raised-cosine patch of relative width `w`
/// averages the strike comb over `w` of the string, which multiplies mode `k`
/// by `cos^2(k pi w / 2)` and takes it to nothing past `k w = 1`.
///
/// Clamped at the first null rather than letting the analytic form turn back
/// up, exactly as `engine::string::contact_taper` clamps it. `taper(k, 0.0)` is
/// exactly 1.
pub fn contact_taper(k: f64, width: f64) -> f64 {
    let phase = 0.5 * k * std::f64::consts::PI * width;
    if phase >= std::f64::consts::FRAC_PI_2 {
        0.0
    } else {
        let c = phase.cos();
        c * c
    }
}

#[derive(Clone, Debug)]
pub struct StrikeFit {
    /// Strike point as a fraction of the speaking length.
    pub position: f64,
    /// Width of the hammer's contact, as a fraction of the speaking length, or
    /// `None` where the spectrum did not pay for the extra parameter — see
    /// [`StrikeConfig::min_width_gain`].
    pub contact_width: Option<f64>,
    /// Coefficients of the fitted `ln` envelope in ascending powers of `ln k`.
    pub envelope: Vec<f64>,
    /// RMS of the fit residual, in dB.
    pub residual_db: f64,
    /// The same residual for the point-force fit at its own best position: what
    /// the width bought, in dB.
    pub residual_db_point: f64,
    pub partials: usize,
    null_floor: f64,
}

impl StrikeFit {
    /// The amplitude this fit predicts for partial `k`.
    pub fn amplitude(&self, k: u32) -> f64 {
        let kf = f64::from(k);
        poly_eval(&self.envelope, kf.ln()).exp() * self.comb_at(k)
    }

    /// The smooth part alone: the excitation spectrum with the strike comb
    /// divided out, which is what the hammer estimator wants to see.
    pub fn envelope_at(&self, k: u32) -> f64 {
        poly_eval(&self.envelope, f64::from(k).ln()).exp()
    }

    /// The fitted comb at partial `k`, contact taper included: the factor the
    /// hammer estimator divides out of a measured amplitude to see the hammer's
    /// own force spectrum. The taper belongs on this side of the division — it
    /// is a property of how the hammer couples to the string's modes, not of
    /// the force pulse the felt produces — which is why fitting it also moves
    /// the felt fit.
    pub fn comb_at(&self, k: u32) -> f64 {
        comb(
            f64::from(k),
            self.position,
            self.width(),
            self.null_floor,
        )
    }

    fn width(&self) -> f64 {
        self.contact_width.unwrap_or(0.0)
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

    // The point force first, at its own best position: it is the answer unless
    // a width earns its place, and its residual is what "earns" is measured
    // against.
    let (position, rms_point) = fit_at_width(&points, 0.0, config)?;
    let mut best = (position, 0.0, rms_point);
    // Only widths whose null stands inside the measured spectrum are searched.
    let top_k = points.iter().fold(0.0f64, |m, &(k, _)| m.max(k));
    let low = if top_k > 0.0 {
        config.min_cutoff_reach / top_k
    } else {
        f64::INFINITY
    };
    if low <= config.max_width && config.width_grid > 0 {
        // The two parameters are minimised alternately — the width searched at
        // a fixed strike point, the strike point re-fitted at the winning width
        // — rather than the whole grid over `x` being re-run at every candidate
        // `w`. They are very nearly separable: a taper is smooth in `k` and the
        // comb's nulls are not, so tapering the spectrum moves the fitted `x`
        // by well under a percent. Two rounds bring the width back to within
        // the same 5 % the joint search gives, for a fiftieth of the work — and
        // this path runs once per recording of a five-hundred-recording survey.
        let step = (config.max_width - low) / config.width_grid as f64;
        let mut position = position;
        for _ in 0..2 {
            let at =
                |w: f64| residual_at(&points, position, w, config).map_or(f64::MAX, |fit| fit.0);
            let mut width = low;
            let mut value = f64::MAX;
            for i in 0..=config.width_grid {
                let candidate = low + i as f64 * step;
                let loss = at(candidate);
                if loss < value {
                    value = loss;
                    width = candidate;
                }
            }
            if step > 0.0 {
                let (refined, _) = golden_section(
                    (width - step).max(low),
                    (width + step).min(config.max_width),
                    40,
                    at,
                );
                if at(refined) < value {
                    width = refined;
                }
            }
            let Ok((refit, rms)) = fit_at_width(&points, width, config) else {
                break;
            };
            position = refit;
            if rms < best.2 {
                best = (refit, width, rms);
            }
        }
    }
    // A width is one more parameter in a fit that already has an envelope; it
    // is only reported where it removed a real fraction of the residual.
    let (position, width, rms) = if best.1 > 0.0 && best.2 <= rms_point * (1.0 - config.min_width_gain)
    {
        best
    } else {
        (position, 0.0, rms_point)
    };

    let (_, envelope, rms_exact) = residual_at(&points, position, width, config)
        .ok_or_else(|| Error::Estimate("strike-position envelope fit is singular".into()))?;
    debug_assert!((rms_exact - rms).abs() < 1e-9 * rms.max(1e-9));
    Ok(StrikeFit {
        position,
        contact_width: (width > 0.0).then_some(width),
        envelope,
        residual_db: NEPERS_TO_DB * rms_exact,
        residual_db_point: NEPERS_TO_DB * rms_point,
        partials: points.len(),
        null_floor: config.null_floor,
    })
}

/// Decibels per neper: the residual is fitted in `ln` amplitude and reported in
/// dB.
const NEPERS_TO_DB: f64 = 8.685_889_638_065_035;

/// The best strike position at one contact width, and the RMS residual there.
fn fit_at_width(
    points: &[(f64, f64)],
    width: f64,
    config: &StrikeConfig,
) -> Result<(f64, f64)> {
    let step = (config.max_position - config.min_position) / config.grid as f64;
    let objective = |x: f64| residual_at(points, x, width, config).map_or(f64::MAX, |fit| fit.0);
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
    let (_, _, rms) = residual_at(points, position, width, config)
        .ok_or_else(|| Error::Estimate("strike-position envelope fit is singular".into()))?;
    Ok((position, rms))
}

/// The softened excitation shape: `|sin(k pi x)|` tapered by the hammer's
/// contact and with a floor under its nulls.
///
/// The floor goes under the *product*, which is what makes the taper's own zero
/// at `k w = 1` a deep null rather than minus infinity decibels — the same
/// reason the comb's nulls have one.
fn comb(k: f64, x: f64, width: f64, floor: f64) -> f64 {
    let s = (k * std::f64::consts::PI * x).sin() * contact_taper(k, width);
    (s * s + floor * floor).sqrt()
}

/// Huber loss of the best smooth envelope through the comb-corrected spectrum,
/// with that envelope and the plain RMS residual alongside it.
fn residual_at(
    points: &[(f64, f64)],
    x: f64,
    width: f64,
    config: &StrikeConfig,
) -> Option<(f64, Vec<f64>, f64)> {
    let log_k: Vec<f64> = points.iter().map(|&(k, _)| k.ln()).collect();
    let corrected: Vec<f64> = points
        .iter()
        .map(|&(k, log_a)| log_a - comb(k, x, width, config.null_floor).ln())
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
                (k, kf.powf(-1.4) * comb(kf, x, 0.0, floor))
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
                (k, envelope * comb(kf, truth, 0.0, 0.005))
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

    /// The spectrum a strike at `x` produces from the same envelope, with a
    /// hammer of relative width `w` averaging over the comb.
    fn tapered(x: f64, w: f64, count: u32, floor: f64) -> Vec<(u32, f64)> {
        (1..=count)
            .map(|k| {
                let kf = f64::from(k);
                (k, kf.powf(-1.4) * comb(kf, x, w, floor))
            })
            .collect()
    }

    #[test]
    fn the_contact_width_comes_back_where_its_null_is_in_the_spectrum() {
        let config = StrikeConfig::default();
        for &truth in &[0.02, 0.03, 0.04] {
            // Sixty partials: a bass note, where `k w` passes 1 and the taper
            // has a null of its own inside the measurement.
            let fit = fit_strike_position(&tapered(0.118, truth, 60, config.null_floor), &config)
                .unwrap();
            let width = fit.contact_width.expect("a width whose null was measured");
            assert!(
                (width / truth - 1.0).abs() < 0.05,
                "w = {width:.4} from {truth}: {fit:?}"
            );
            // ... and it did not buy that fit by moving the strike point.
            assert!((fit.position / 0.118 - 1.0).abs() < 0.02, "{fit:?}");
            assert!(fit.residual_db < fit.residual_db_point, "{fit:?}");
        }
    }

    #[test]
    fn a_point_force_spectrum_is_not_given_a_width() {
        // No taper in the data, and an envelope the fit's polynomial cannot
        // reproduce exactly — the felt's own rolloff, which is what a width
        // would otherwise be spent absorbing.
        let config = StrikeConfig::default();
        for count in [16u32, 24, 40, 60] {
            let measured: Vec<(u32, f64)> = (1..=count)
                .map(|k| {
                    let kf = f64::from(k);
                    let pulse = 1.0 / (1.0 + (kf * 130.8 / 1400.0).powi(2)).powf(1.1);
                    (k, pulse * comb(kf, 0.118, 0.0, config.null_floor))
                })
                .collect();
            let fit = fit_strike_position(&measured, &config).unwrap();
            assert_eq!(fit.contact_width, None, "{count} partials: {fit:?}");
            assert_eq!(fit.residual_db, fit.residual_db_point);
        }
    }

    #[test]
    fn a_width_whose_null_is_above_the_last_partial_is_not_guessed() {
        // The taper is real — 2 % — but 24 partials only reach `k w = 0.5`,
        // where it is a smooth droop and nothing else. Refused rather than
        // fitted at half its value.
        let config = StrikeConfig::default();
        let fit =
            fit_strike_position(&tapered(0.118, 0.02, 24, config.null_floor), &config).unwrap();
        assert_eq!(fit.contact_width, None, "{fit:?}");
        // Sixty partials of the same string do reach it.
        let fit =
            fit_strike_position(&tapered(0.118, 0.02, 60, config.null_floor), &config).unwrap();
        assert!((fit.contact_width.unwrap() / 0.02 - 1.0).abs() < 0.05, "{fit:?}");
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

