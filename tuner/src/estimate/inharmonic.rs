//! Robust fit of the stiff-string partial layout
//! `f_k = k f0 sqrt(1 + B k^2 + B4 k^4)` to the measured partial frequencies.
//!
//! The model is nonlinear in `(f0, B, B4)` but linear after one substitution:
//! squaring and dividing by `k^2` gives
//!
//! ```text
//!     (f_k / k)^2 = f0^2 + f0^2 B k^2 + f0^2 B4 k^4,
//! ```
//! a *polynomial in `x = k^2`* whose constant term is `f0^2`, whose linear
//! coefficient over it is `B` and whose quadratic coefficient over it is `B4`.
//! So the fit is a weighted least-squares polynomial, and all the work is in
//! being robust: one partial mis-tracked into a neighbour's peak, or one sitting
//! on a soundboard resonance that pulled it, would otherwise drag `B` by more
//! than the 2 % the pipeline needs.
//!
//! Robustness comes in two layers: a Theil-Sen line (median of pairwise slopes,
//! ~29 % breakdown) to start from, then iterated rejection of partials more
//! than `reject_cents` off the current model, refitting until the accepted set
//! stops changing.
//!
//! # The fourth-order term, and when it is not fitted
//!
//! One `B` describes the measured series to 1–5 cents from D#2 up and misplaces
//! partials by up to 78 cents below it (`docs/history/TUNING_REPORT.md` §1): a wound bass
//! string's `B` *falls* 25–37 % along its own series and the short wound tenor
//! strings' *rises* 24–45 %. The diagnostic that found it is a `B` fitted twice
//! over disjoint bands of `k` and the ratio of the two, and that is exactly the
//! statistic [`BandRatio`] carries here — because a `k^4` term fitted where
//! the series does not curve is a term fitted to noise, and it lands on the
//! high partials, which are the ones with the longest lever.
//!
//! So the quartic is only fitted where the two bands disagree by more than
//! their own uncertainty says they should. The uncertainty is measured, not
//! assumed: each band's `B` carries the standard error of its own weighted fit,
//! and the guard asks for a ratio at least
//! [`InharmonicConfig::band_sigmas`] standard deviations from 1. Everywhere
//! else `B4` is reported as an exact zero, which is the two-parameter law term
//! for term.

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
    /// Fit the signed fourth-order term at all. Off, the fit is the
    /// two-parameter one and `B4` is a hard zero.
    pub fourth_order: bool,
    /// The two bands of the diagnostic, as inclusive ranges of `k`.
    ///
    /// The low band is `docs/history/TUNING_REPORT.md` §1's own: partials 1–8, which is
    /// where a single `B` is always anchored. The high band is *not* — §1 used
    /// 14–26 and this is everything above the split, bounded only by
    /// [`Self::max_partial`] — and the difference was measured rather than
    /// chosen. What a `k^4` term does to a series grows as `k^4`, so the lever
    /// is at the very top of it: on a C2 rendered with a known `B4`, the guard
    /// stands 2.1 sigma from a flat ratio with the band running to partial 40,
    /// 1.1 with it capped at 26, and 1.3 with §1's 14–26 window. Only the first
    /// of those fires, so only the first can measure anything.
    ///
    /// The high band used to disagree with §1's on the wound bass — 1.01–1.03
    /// against 0.63–0.75 on A0, C1 and D#1 — and the reason turned out to be in
    /// the data rather than in the window: above partial 25 the tracker had
    /// skipped one and every index was one too low ([`trusted_prefix`],
    /// `DECISIONS.md` 131). With the mis-numbered top gone the two agree
    /// (0.80/0.68/0.65), and the wide band goes on being the one that recovers a
    /// coefficient it was given.
    pub band_low: (u32, u32),
    pub band_high: (u32, u32),
    /// Fewest partials each band needs before the two are compared. Two would
    /// determine a `B` each; four leaves the jackknife something to work with.
    pub band_min_partials: usize,
    /// How many standard deviations the two bands' `B` must differ by before a
    /// fourth-order term is fitted. The standard deviation is each band's own
    /// leave-one-out scatter, so this is a measurement of the note and not a
    /// constant: two sigmas on material where the estimator returns 0.06 cents
    /// of residual is a very different absolute threshold from two sigmas on a
    /// recording with 16.
    pub band_sigmas: f64,
    /// Largest the `B4 k^4` term may be, at the highest partial the fit used,
    /// as a fraction of the `B k^2` term there. The measured shifts are
    /// 25–45 % (`docs/history/TUNING_REPORT.md` §1); past this the quartic has run away and
    /// the two-parameter answer is reported instead.
    pub max_band_shift: f64,
    /// Multiple of the series' own local spacing at which a step from one
    /// partial to the next is read as a *skipped* partial rather than as a
    /// wide one. See [`trusted_prefix`]. Zero or non-finite switches the check
    /// off and fits whatever the tracker returned.
    pub series_break_ratio: f64,
    /// Consecutive spacings the check needs before it will call a break. Four
    /// gives the running median something to be a median of, and no note is
    /// trimmed on the strength of its first step.
    pub series_break_spacings: usize,
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
            fourth_order: true,
            band_low: (1, 8),
            band_high: (9, u32::MAX),
            band_min_partials: 4,
            band_sigmas: 2.0,
            max_band_shift: 0.6,
            series_break_ratio: 1.5,
            series_break_spacings: 4,
        }
    }
}

/// The two-band diagnostic: one `B` fitted to the low partials, one to the
/// high, and how far apart they are compared with how well each was measured.
///
/// Over a narrow band of `k` the pair `(f0, B)` is correlated — a smaller `B`
/// trades against a higher `f0` — so neither number means much on its own. What
/// the two bands compare is the *curvature* of the measured series, which is
/// exactly what one `B` has to reproduce and what the `k^4` term exists to fix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BandRatio {
    pub b_low: f64,
    pub b_high: f64,
    /// Leave-one-out standard error of each band's `B`.
    pub sigma_low: f64,
    pub sigma_high: f64,
    pub low_partials: usize,
    pub high_partials: usize,
}

impl BandRatio {
    /// `B(high) / B(low)`: 0.63–0.75 in the wound bass, 1.24–1.45 in the low
    /// tenor, and 0.93–1.08 where the two-parameter law holds.
    pub fn ratio(&self) -> f64 {
        self.b_high / self.b_low
    }

    /// Standard error of that ratio, from the two bands' own scatter.
    pub fn sigma(&self) -> f64 {
        let relative = |b: f64, sigma: f64| if b != 0.0 { sigma / b } else { f64::INFINITY };
        self.ratio().abs()
            * (relative(self.b_low, self.sigma_low).powi(2)
                + relative(self.b_high, self.sigma_high).powi(2))
            .sqrt()
    }

    /// How many standard deviations the ratio stands from 1. This is the whole
    /// guard: below the configured threshold the series has no curvature the
    /// second coefficient could describe that the noise does not also explain.
    pub fn sigmas_from_one(&self) -> f64 {
        let sigma = self.sigma();
        if !(sigma.is_finite() && sigma > 0.0) {
            return 0.0;
        }
        (self.ratio() - 1.0).abs() / sigma
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
    /// The two-band diagnostic, where the note had two bands to compare.
    pub bands: Option<BandRatio>,
    /// RMS residual of the two-parameter fit, in cents — what the note would
    /// have had without the fourth-order term. Equal to
    /// [`InharmonicFit::residual_cents`] whenever `B4` came back zero.
    pub residual_cents_2: f64,
}

impl InharmonicFit {
    /// Whether this fit put anything in the fourth-order term.
    pub fn is_fourth_order(&self) -> bool {
        self.model.b4 != 0.0
    }
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

/// How many of a measured series' partials stand before the tracker skipped
/// one — the prefix the fit may believe.
///
/// A tracked partial is only as good as its *index*. The tracker associates
/// peaks with a predicted series and refuses anything further than a fraction
/// of the local spacing from the prediction, so it cannot mistake partial
/// `k + 1` for partial `k` in one step. What it can do — and on Salamander's
/// wound bass strings does, from partial 25 up — is lose one: where the model
/// it is predicting from has drifted past half the local spacing, the peak for
/// `k` falls outside `k`'s window, `k` lands on whatever weak peak is inside
/// it, and every index above is one too low. The frequencies stay real; the
/// numbering does not, and the fit sees a series whose top is stretched.
///
/// It is visible in the series itself and nowhere else: a skip leaves a step
/// of two spacings where every other step is one. Stiffness widens the spacing
/// as `k` grows, but slowly — under 3 % per partial anywhere on this
/// instrument — so half a spacing of margin is unambiguous.
///
/// Measured on A0 (`DECISIONS.md`): tracks 18–24 land on the peaks that are
/// there to within a cent, track 25 lands 21 Hz above the peak the spectrum
/// has and 12 Hz below the next, and from 26 up every track is the partial
/// above its index. Fitting all of them is what gave A0 the `B` the preset
/// wrote, and with it the 78 cents of misplaced fundamental `docs/history/TUNING_REPORT.md`
/// §1 reports.
///
/// Only consecutive indices are compared: a partial that was never tracked at
/// all leaves a gap of two in `k` and is not evidence of anything.
///
/// A skip is also distinguished from a single partial *pulled* off its peak —
/// by a neighbour's sidelobe, or by a resonance sitting on it — which the
/// rejection loop already handles and which must not cost the note the rest of
/// its series. Both make one wide step; only a skip stays wide, because a
/// pulled partial comes back to the series at the next index. So the check
/// asks for two things: this step is wider than the local spacing allows, and
/// the step *across* it is wider still.
pub fn trusted_prefix(partials: &[(u32, f64)], config: &InharmonicConfig) -> usize {
    if !(config.series_break_ratio.is_finite() && config.series_break_ratio > 0.0) {
        return partials.len();
    }
    let ratio = config.series_break_ratio;
    let mut spacings: Vec<f64> = Vec::new();
    for i in 1..partials.len() {
        let (previous_k, previous_f) = partials[i - 1];
        let (k, f) = partials[i];
        if k != previous_k + 1 {
            continue;
        }
        let spacing = f - previous_f;
        if spacings.len() >= config.series_break_spacings.max(1) {
            let mut sorted = spacings.clone();
            sorted.sort_by(f64::total_cmp);
            let median = sorted[sorted.len() / 2];
            // The step across this one, where there is a next partial to
            // measure it to. A partial that was merely pulled has a short step
            // on its far side and spans two spacings in total.
            let across = partials
                .get(i + 1)
                .filter(|&&(next_k, _)| next_k == k + 1)
                .map(|&(_, next_f)| next_f - previous_f);
            if spacing > ratio * median
                && across.map_or(true, |d| d > (1.0 + ratio) * median)
            {
                return i;
            }
        }
        spacings.push(spacing);
    }
    partials.len()
}

/// The fit proper: `(k, f_k, variance of f_k)` in, `(f0, B)` out.
pub fn fit_measured_partials(
    partials: &[(u32, f64, f64)],
    config: &InharmonicConfig,
) -> Result<InharmonicFit> {
    let mut partials: Vec<&(u32, f64, f64)> = partials
        .iter()
        .filter(|&&(k, f, _)| k >= 1 && f > 0.0)
        .collect();
    partials.sort_by_key(|&&(k, _, _)| k);
    // Everything above the first skipped partial is mis-numbered, and a
    // mis-numbered partial has the longest lever of all on `B`.
    let series: Vec<(u32, f64)> = partials.iter().map(|&&(k, f, _)| (k, f)).collect();
    partials.truncate(trusted_prefix(&series, config));
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

    let (intercept, slope) = theil_sen(&samples)?;
    let seed = vec![intercept, slope];
    let (linear, used) = refine(&samples, &seed, 2, config)?;
    let model = model_from_coefficients(&linear)?;

    // The two-band diagnostic decides whether there is anything for a second
    // coefficient to describe, on the partials the rejection kept.
    let bands = config
        .fourth_order
        .then(|| band_ratio(&samples, &used, config))
        .flatten();
    let mut chosen = model;
    if let Some(bands) = &bands {
        if bands.sigmas_from_one() >= config.band_sigmas {
            let seed = [linear[0], linear[1], 0.0];
            if let Ok((quartic, quartic_used)) = refine(&samples, &seed, 3, config) {
                if let Some(candidate) =
                    largest_plausible(&quartic, &samples, &quartic_used, config)
                {
                    chosen = candidate;
                }
            }
        }
    }

    let deviations = |model: &InharmonicModel| -> (f64, f64) {
        let mut sum = 0.0;
        let mut worst = 0.0f64;
        for &i in &used {
            let error = model.cents_from_partial(samples[i].k, samples[i].frequency_hz);
            sum += error * error;
            worst = worst.max(error.abs());
        }
        ((sum / used.len() as f64).sqrt(), worst)
    };
    let (residual_cents, worst_cents) = deviations(&chosen);
    let (residual_cents_2, _) = deviations(&model);
    let rejected: Vec<(u32, f64)> = (0..samples.len())
        .filter(|i| !used.contains(i))
        .map(|i| {
            (
                samples[i].k,
                chosen.cents_from_partial(samples[i].k, samples[i].frequency_hz),
            )
        })
        .collect();

    Ok(InharmonicFit {
        model: chosen,
        used: used.iter().map(|&i| samples[i].k).collect(),
        rejected,
        residual_cents,
        worst_cents,
        bands,
        residual_cents_2,
    })
}

/// The rejection loop, over a polynomial of `terms` coefficients in `x = k^2`.
///
/// `terms = 2` is the straight line the two-parameter law is, `terms = 3` the
/// parabola the four-parameter one is. Both are refitted without whatever the
/// current model puts more than `reject_cents` away, until the accepted set and
/// the coefficients both stop moving.
fn refine(
    samples: &[Sample],
    seed: &[f64],
    terms: usize,
    config: &InharmonicConfig,
) -> Result<(Vec<f64>, Vec<usize>)> {
    let mut coefficients = seed.to_vec();
    let mut used: Vec<usize> = (0..samples.len()).collect();
    for _ in 0..config.max_iterations {
        let model = model_from_coefficients(&coefficients)?;
        let accepted: Vec<usize> = (0..samples.len())
            .filter(|&i| {
                model
                    .cents_from_partial(samples[i].k, samples[i].frequency_hz)
                    .abs()
                    <= config.reject_cents
            })
            .collect();
        if accepted.len() < config.min_partials.max(terms) {
            break;
        }
        let converged = accepted == used;
        used = accepted;
        let new = weighted_polynomial(samples, &used, terms)?;
        let moved = new.iter().zip(&coefficients).any(|(&a, &b)| {
            (a - b).abs() > 1e-12 * b.abs().max(1e-12)
        });
        coefficients = new;
        if converged && !moved {
            break;
        }
    }
    Ok((coefficients, used))
}

/// As much of the fitted `k^4` term as still leaves a partial layout.
///
/// The unconstrained fit is taken whole where it is [`plausible`]. Where it is
/// not — and in the wound bass it usually is not, because the shape
/// `docs/history/TUNING_REPORT.md` §1 measures is larger than one `k^4` term can carry
/// without folding the top of the series — the term is shrunk rather than
/// dropped: the coefficient is scaled towards zero and `(f0, B)` refitted
/// against what is left, by bisection on the largest fraction that is still a
/// layout. A partial correction is a better instrument than none — A0's
/// partials 14–26 sit at 390–760 Hz, where the ear resolves them individually —
/// and it is the same rule `PresetBuilder` applies to an interpolated
/// coefficient, so what the estimator reports is what the file can hold.
fn largest_plausible(
    quartic: &[f64],
    samples: &[Sample],
    used: &[usize],
    config: &InharmonicConfig,
) -> Option<InharmonicModel> {
    let at = |fraction: f64| -> Option<InharmonicModel> {
        let c2 = quartic.get(2).copied()? * fraction;
        let linear = fit_with_fixed_quartic(samples, used, c2)?;
        model_from_coefficients(&[linear[0], linear[1], c2]).ok()
    };
    let whole = at(1.0)?;
    if plausible(&whole, samples, used, config) {
        return Some(whole);
    }
    let (mut lo, mut hi) = (0.0, 1.0);
    for _ in 0..40 {
        let mid = 0.5 * (lo + hi);
        match at(mid) {
            Some(model) if plausible(&model, samples, used, config) => lo = mid,
            _ => hi = mid,
        }
    }
    // Zero is the two-parameter fit, which is always a layout; anything at or
    // below the resolution of this search is that fit and is reported as it.
    (lo > 0.0).then(|| at(lo)).flatten()
}

/// `(f0^2, f0^2 B)` refitted with the quartic coefficient held at `c2`.
fn fit_with_fixed_quartic(samples: &[Sample], used: &[usize], c2: f64) -> Option<Vec<f64>> {
    if used.len() < 2 {
        return None;
    }
    let mut basis = Vec::with_capacity(used.len() * 2);
    let mut y = Vec::with_capacity(used.len());
    let mut weights = Vec::with_capacity(used.len());
    for &i in used {
        basis.push(1.0);
        basis.push(samples[i].x);
        y.push(samples[i].y - c2 * samples[i].x * samples[i].x);
        weights.push(samples[i].weight);
    }
    weighted_least_squares(&basis, &y, &weights, 2)
}

/// Whether a fourth-order fit is a partial layout and not a runaway.
///
/// Three things are asked of it, and each has cost a real fit: `B` itself must
/// stay non-negative (the engine's tables are), the series must stay ordered
/// over the partials that were fitted, and the `k^4` term must not have grown
/// past [`InharmonicConfig::max_band_shift`] of the `k^2` term at the top of
/// them. A candidate that fails any of them is dropped in favour of the
/// two-parameter answer rather than clamped into range: a clamped coefficient
/// is neither what was measured nor what the two-band test asked for.
fn plausible(
    model: &InharmonicModel,
    samples: &[Sample],
    used: &[usize],
    config: &InharmonicConfig,
) -> bool {
    if !(model.f0_hz.is_finite() && model.f0_hz > 0.0 && model.b4.is_finite()) {
        return false;
    }
    if model.b < 0.0 || !model.b.is_finite() {
        return false;
    }
    let Some(&top) = used.iter().max_by_key(|&&i| samples[i].k) else {
        return false;
    };
    let k = f64::from(samples[top].k);
    let quadratic = model.b * k * k;
    let quartic = model.b4 * k * k * k * k;
    if quadratic <= 0.0 || quartic.abs() > config.max_band_shift * quadratic {
        return false;
    }
    let mut previous = 0.0;
    for k in 1..=samples[top].k {
        let f = model.partial(k);
        if !f.is_finite() || f <= previous {
            return false;
        }
        previous = f;
    }
    true
}

/// One `B` per band of `k`, each with the leave-one-out standard error of its
/// own fit. `None` where either band is too short to fit.
fn band_ratio(
    samples: &[Sample],
    used: &[usize],
    config: &InharmonicConfig,
) -> Option<BandRatio> {
    let band = |(first, last): (u32, u32)| -> Vec<usize> {
        used.iter()
            .copied()
            .filter(|&i| (first..=last).contains(&samples[i].k))
            .collect()
    };
    let (low, high) = (band(config.band_low), band(config.band_high));
    if low.len() < config.band_min_partials || high.len() < config.band_min_partials {
        return None;
    }
    let (b_low, sigma_low) = band_b(samples, &low)?;
    let (b_high, sigma_high) = band_b(samples, &high)?;
    Some(BandRatio {
        b_low,
        b_high,
        sigma_low,
        sigma_high,
        low_partials: low.len(),
        high_partials: high.len(),
    })
}

/// `B` over one band, and how well it is determined there.
///
/// The uncertainty is a jackknife: the band is refitted with each of its
/// partials left out in turn, and the scatter of those answers is what says
/// whether the two bands really disagree. It costs `n` two-by-two solves and,
/// unlike a formula, it counts whatever actually makes this band's `B` uncertain
/// — a short band, one loud outlier the rejection kept, or a partial measured
/// badly.
fn band_b(samples: &[Sample], band: &[usize]) -> Option<(f64, f64)> {
    let b_of = |set: &[usize]| -> Option<f64> {
        let solution = weighted_polynomial_opt(samples, set, 2)?;
        (solution[0] > 0.0).then(|| solution[1] / solution[0])
    };
    let full = b_of(band)?;
    let n = band.len();
    let mut left_out = Vec::with_capacity(n);
    for skip in 0..n {
        let set: Vec<usize> = band
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != skip)
            .map(|(_, &index)| index)
            .collect();
        if let Some(b) = b_of(&set) {
            left_out.push(b);
        }
    }
    if left_out.len() < 3 {
        return None;
    }
    let mean = left_out.iter().sum::<f64>() / left_out.len() as f64;
    let scatter: f64 = left_out.iter().map(|b| (b - mean).powi(2)).sum();
    let variance = scatter * (left_out.len() - 1) as f64 / left_out.len() as f64;
    Some((full, variance.sqrt()))
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

/// `y = c0 + c1 x + c2 x^2` with `c0 = f0^2`, `c1 = f0^2 B` and `c2 = f0^2 B4`.
/// A two-coefficient argument is the two-parameter law, `B4 = 0`.
fn model_from_coefficients(c: &[f64]) -> Result<InharmonicModel> {
    if c.iter().any(|v| !v.is_finite()) || !c.first().is_some_and(|&c0| c0 > 0.0) {
        return Err(Error::Estimate(
            "inharmonicity fit produced a non-positive f0^2".into(),
        ));
    }
    let intercept = c[0];
    // A negative slope is a negative B: a string whose partials converge, which
    // no string does. It means the data is noise-dominated; report the
    // harmonic layout rather than a square root of a negative number. `B4` is
    // *not* clamped — its sign is the finding (`docs/history/TUNING_REPORT.md` §1) — and
    // what keeps it a layout is `plausible`.
    Ok(InharmonicModel::with_b4(
        intercept.sqrt(),
        (c[1] / intercept).max(0.0),
        c.get(2).map_or(0.0, |&c2| c2 / intercept),
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

/// Weighted least squares of `y` against `1, x, x^2, ...` over `used`.
fn weighted_polynomial(samples: &[Sample], used: &[usize], terms: usize) -> Result<Vec<f64>> {
    weighted_polynomial_opt(samples, used, terms)
        .ok_or_else(|| Error::Estimate("inharmonicity fit is singular".into()))
}

fn weighted_polynomial_opt(samples: &[Sample], used: &[usize], terms: usize) -> Option<Vec<f64>> {
    if used.len() < terms {
        return None;
    }
    let mut basis = Vec::with_capacity(used.len() * terms);
    let mut y = Vec::with_capacity(used.len());
    let mut weights = Vec::with_capacity(used.len());
    for &i in used {
        let mut power = 1.0;
        for _ in 0..terms {
            basis.push(power);
            power *= samples[i].x;
        }
        y.push(samples[i].y);
        weights.push(samples[i].weight);
    }
    weighted_least_squares(&basis, &y, &weights, terms)
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

    /// What Salamander's A0 does from partial 25 up: the tracker loses one
    /// partial and every index above it names the partial above itself. The
    /// frequencies are all real, so nothing about them looks wrong except the
    /// step where the skip happened — and fitting them shifts `B` by far more
    /// than the 2 % the pipeline is allowed.
    #[test]
    fn a_skipped_partial_costs_the_series_its_top_and_not_its_answer() {
        let truth = InharmonicModel::new(27.5, 3.1e-4);
        let mut measured = partials(truth, 40);
        for point in measured.iter_mut().skip(24) {
            point.1 = truth.partial(point.0 + 1);
        }
        // A few cents of scatter, deterministic and alternating, because that
        // is what stops the rejection loop from separating the mis-numbered
        // top of the series by itself: on exact data the low partials pin the
        // line and the skip rejects cleanly, and on a recording they do not.
        for (index, point) in measured.iter_mut().enumerate() {
            point.1 *= 2f64.powf(if index % 3 == 0 { 4.0 } else { -2.0 } / 1200.0);
        }
        assert_eq!(trusted_prefix(&measured, &InharmonicConfig::default()), 24);

        let fit = fit_inharmonic_partials(&measured, &InharmonicConfig::default()).unwrap();
        assert!(
            (fit.model.b / truth.b - 1.0).abs() < 0.02,
            "B off by {:.1} %: {fit:?}",
            100.0 * (fit.model.b / truth.b - 1.0)
        );
        assert!(fit.used.iter().all(|&k| k <= 24), "{:?}", fit.used);

        // ... and the same series fitted whole, which is what the survey did
        // before, is wrong by much more than that.
        let unchecked = fit_inharmonic_partials(
            &measured,
            &InharmonicConfig {
                series_break_ratio: 0.0,
                ..InharmonicConfig::default()
            },
        )
        .unwrap();
        assert!(
            (unchecked.model.b / truth.b - 1.0).abs() > 0.05,
            "the skip cost nothing, so the check is measuring nothing: {unchecked:?}"
        );
    }

    /// A partial pulled off its own peak makes one wide step too, and it must
    /// not cost the note the rest of its series: it comes back at the next
    /// index, and the rejection loop is what deals with it.
    #[test]
    fn a_partial_pulled_off_its_peak_is_not_a_skipped_one() {
        let truth = InharmonicModel::new(220.0, 8e-4);
        let mut measured = partials(truth, 16);
        measured[8].1 *= 2f64.powf(1.0 / 12.0);
        assert_eq!(
            trusted_prefix(&measured, &InharmonicConfig::default()),
            measured.len()
        );
    }

    /// A partial the tracker never found leaves a gap of two in `k`, which is
    /// a wide step in hertz and no evidence at all about the numbering.
    #[test]
    fn a_partial_that_was_never_tracked_is_not_a_skipped_one() {
        let truth = InharmonicModel::new(220.0, 8e-4);
        let mut measured = partials(truth, 16);
        measured.remove(9);
        assert_eq!(
            trusted_prefix(&measured, &InharmonicConfig::default()),
            measured.len()
        );
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

    fn quartic(model: InharmonicModel, count: u32) -> Vec<(u32, f64)> {
        (1..=count).map(|k| (k, model.partial(k))).collect()
    }

    #[test]
    fn an_exact_fourth_order_series_gives_up_both_coefficients() {
        // A wound bass string's shape: `B` at the top of the series is 25 %
        // below `B` at the bottom, which is `docs/history/TUNING_REPORT.md` §1's A0.
        let truth = InharmonicModel::with_b4(27.5, 3.0e-4, -1.9e-7);
        let fit = fit_inharmonic_partials(&quartic(truth, 26), &InharmonicConfig::default())
            .unwrap();
        assert!(fit.is_fourth_order(), "{fit:?}");
        assert!(
            (fit.model.b4 / truth.b4 - 1.0).abs() < 0.05,
            "B4 {:.3e} vs {:.3e}: {fit:?}",
            fit.model.b4,
            truth.b4
        );
        assert!((fit.model.b / truth.b - 1.0).abs() < 0.02, "{fit:?}");
        assert!((fit.model.f0_hz / truth.f0_hz - 1.0).abs() < 1e-4, "{fit:?}");
        // The two-parameter fit of the same series cannot place the partials,
        // which is the whole finding.
        assert!(fit.residual_cents < 0.2 * fit.residual_cents_2, "{fit:?}");
        // And the diagnostic that let it be fitted reads the ratio the report
        // reads.
        let bands = fit.bands.expect("two bands");
        assert!(bands.ratio() < 0.8, "{bands:?}");
    }

    #[test]
    fn the_tenor_side_of_the_break_comes_back_with_the_other_sign() {
        // F#1 to C2: the high partials are *sharper* than one `B` predicts, so
        // the correction is positive. A signed coefficient or nothing.
        let truth = InharmonicModel::with_b4(46.25, 7.5e-5, 3.0e-8);
        let fit = fit_inharmonic_partials(&quartic(truth, 26), &InharmonicConfig::default())
            .unwrap();
        assert!(fit.model.b4 > 0.0, "{fit:?}");
        assert!((fit.model.b4 / truth.b4 - 1.0).abs() < 0.05, "{fit:?}");
        assert!(fit.bands.expect("two bands").ratio() > 1.2);
    }

    #[test]
    fn a_two_parameter_series_is_not_given_a_fourth_order_term() {
        // Tracker-grade noise on a series that obeys the two-parameter law:
        // the two bands agree inside their own scatter, so nothing is fitted
        // to the disagreement.
        let truth = InharmonicModel::new(110.31, 3.7e-4);
        let mut measured = partials(truth, 24);
        for (index, point) in measured.iter_mut().enumerate() {
            point.1 += if index % 2 == 0 { 0.05 } else { -0.05 };
        }
        let fit = fit_inharmonic_partials(&measured, &InharmonicConfig::default()).unwrap();
        assert_eq!(fit.model.b4, 0.0, "{fit:?}");
        assert_eq!(fit.residual_cents, fit.residual_cents_2);
        let bands = fit.bands.expect("two bands");
        assert!(
            bands.sigmas_from_one() < InharmonicConfig::default().band_sigmas,
            "{bands:?} is {:.2} sigmas from 1",
            bands.sigmas_from_one()
        );
    }

    #[test]
    fn a_note_with_one_band_is_never_given_a_fourth_order_term() {
        // Six partials: a treble note, where there is no high band to compare
        // the low one with and the `k^4` term would be fitted to whatever the
        // top two partials did.
        let truth = InharmonicModel::with_b4(1046.5, 2.7e-3, -1.0e-5);
        let fit = fit_inharmonic_partials(&quartic(truth, 6), &InharmonicConfig::default())
            .unwrap();
        assert_eq!(fit.model.b4, 0.0, "{fit:?}");
        assert!(fit.bands.is_none(), "{fit:?}");
    }

    #[test]
    fn the_fourth_order_fit_can_be_switched_off() {
        let truth = InharmonicModel::with_b4(27.5, 3.0e-4, -1.9e-7);
        let config = InharmonicConfig {
            fourth_order: false,
            ..InharmonicConfig::default()
        };
        let fit = fit_inharmonic_partials(&quartic(truth, 26), &config).unwrap();
        assert_eq!(fit.model.b4, 0.0);
        assert!(fit.bands.is_none());
    }

    #[test]
    fn too_few_partials_is_an_error_and_not_a_guess() {
        let truth = InharmonicModel::new(440.0, 1e-3);
        assert!(fit_inharmonic_partials(&partials(truth, 3), &InharmonicConfig::default()).is_err());
    }
}
