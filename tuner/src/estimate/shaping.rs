//! The three per-partial fields, measured: `notes.comb_floor`,
//! `notes.partial_gains` and `notes.partial_sigma_scale`.
//!
//! All three are *direct* estimates. Nothing here searches a parameter space:
//! the gains are a ratio of two measured spectra, the sigma scales a ratio of
//! two measured rates, and the floor is the one number that makes a monotone
//! function of it hit a measured target, found by bisection.
//!
//! # The excitation: two faults of opposite sign
//!
//! `renders/timbre-ladder/ANALYSIS.md` §8.2 convicts the engine's excitation
//! twice over, and the two convictions point opposite ways:
//!
//! * **Between the nulls the comb is too smooth.** The recording's per-partial
//!   deviation from a smooth envelope is 5–10 dB (`TUNING_REPORT.md` §3) against
//!   the engine's 2–5, and the roughness is *not* shared between notes at the
//!   same frequency, so it cannot be a bridge curve — it has to be per note, per
//!   partial. That is [`partial_gains`].
//! * **At the nulls it is far too rough.** `sin(k pi x)` has exact zeros; A2's
//!   k = 17 lands 42 dB down where the recording's deepest partial *anywhere* is
//!   9.3 to 17.7 dB down and never at that index. That is [`comb_floor`].
//!
//! # How the two are kept from double-counting the same null
//!
//! **The floor decides how deep the engine's comb is allowed to go; the gains
//! place every partial inside that.** The floor is fitted first, from the depth
//! of the recording's own deepest partial below a smooth envelope, and the gains
//! are then measured against the comb *with that floor already in it*
//! ([`EngineComb`]). Neither mechanism can carry the other's correction, because
//! the floor is in the denominator of every gain by construction — this is a
//! property of the arithmetic, not a rule about which keys get which.
//!
//! Two consequences worth stating:
//!
//! * Where a key's bare comb is *already* shallower than the recording's deepest
//!   partial, the fitted floor is exactly zero — a floor can only lift a null,
//!   and lifting one the measurement did not ask for would be an invention. The
//!   gains then carry that key's whole deepest dip, which they can, because a
//!   dip inside 9.3–17.7 dB is inside their ±20 dB range.
//! * That range is the reason the split has to be this way round and not the
//!   other. A bare null 42 dB deep needs **+25 dB** of gain to fill and the
//!   schema allows +20: gains alone cannot do it, and gains fitted against a
//!   bare comb would all pile up against their own ceiling at exactly the
//!   partials the floor exists for.
//!
//! # What the gains are measured against, and why that is the engine's comb
//!
//! Not against the *fitted* comb of `estimate::strike`. The preset does not
//! write a fitted strike position — a close microphone's own `sin(k pi x)` comb
//! is not separable from the hammer's (`DECISIONS.md` 93), so the survey reports
//! it and leaves the table alone — and what the engine will actually play is the
//! base preset's own strike position, contact width and floor. The gains are the
//! correction from *that* comb to the recording, which is what makes the
//! rendered spectrum the recorded one. The microphone's comb rides along inside
//! them, deliberately: the preset targets the recording.
//!
//! # Why they are velocity-independent
//!
//! Each layer's smooth envelope is fitted to that layer's own spectrum, so
//! everything that varies with the blow — the level, the tilt, the felt's
//! rolloff — is absorbed by the envelope before the ratio is taken. What is left
//! is the same per-partial pattern in all sixteen layers, and the value written
//! is their median with the outliers dropped. The fitted envelope is a
//! least-squares one in the log domain, so the residuals it leaves sum to zero:
//! a row of gains has geometric mean 1 and writing it does not move the note's
//! loudness.

use crate::estimate::decay::{DecayCurve, DecayReport};
use crate::estimate::DecayConfig;
use crate::numeric::{poly_eval, weighted_polyfit};
use crate::preset::{
    MAX_PARTIAL_GAIN, MAX_PARTIAL_SIGMA_SCALE, MIN_PARTIAL_GAIN, MIN_PARTIAL_SIGMA_SCALE,
};

/// Decibels per neper.
const NEPERS_TO_DB: f64 = 8.685_889_638_065_035;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShapingConfig {
    /// Degree of the polynomial in `ln k` standing in for the smooth spectral
    /// envelope — the same reference `TUNING_REPORT.md` §3 measured its
    /// roughness against and the same one `estimate::strike` fits under its
    /// comb.
    pub envelope_degree: usize,
    /// Huber threshold on the log-domain residual of that envelope, in nepers.
    /// 0.7 is 6 dB.
    pub huber_delta: f64,
    pub irls_iterations: usize,
    /// Fewest partials a layer must have measured before its spectrum is used
    /// for anything here. Below this the envelope has as many parameters as
    /// data.
    pub min_partials: usize,
    /// Fewest velocity layers a partial must be measured in before a value is
    /// written for it. A partial that only three of sixteen layers could see is
    /// a partial near the noise floor.
    pub min_layers: usize,
    /// Outlier rejection, in dB: layers more than this far from the layers'
    /// median are dropped and the median re-taken.
    pub outlier_db: f64,
    /// Fewest partials of a key's series that must lie inside the measured
    /// spectrum before a comb floor is fitted at all. A key whose partials stop
    /// before its comb's first null has measured nothing about the null.
    pub min_floor_partials: usize,
    /// Bisection steps for the floor. Fifty halvings of `0..=0.5` is far below
    /// the last bit of the `f32` the preset stores.
    pub floor_bisections: usize,
    /// Largest fit residual a partial's envelope may have and still contribute a
    /// decay correction, in dB. The two-exponential law describes a real
    /// partial to about 4 dB whatever produced it (`TUNING_REPORT.md` §2); a
    /// partial fitted worse than that has not measured its own rate.
    pub max_decay_residual_db: f64,
}

impl Default for ShapingConfig {
    fn default() -> Self {
        Self {
            envelope_degree: 2,
            huber_delta: 0.7,
            irls_iterations: 4,
            min_partials: 6,
            min_layers: 5,
            outlier_db: 6.0,
            min_floor_partials: 8,
            floor_bisections: 50,
            max_decay_residual_db: 4.0,
        }
    }
}

/// The excitation comb the engine will really play for one key: the preset's own
/// strike position, contact width and comb floor.
///
/// A magnitude, not a gain — the sign `engine::string::comb_magnitude` keeps is
/// the phase partial `k` starts at and cancels in every ratio taken here.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EngineComb {
    pub strike_position: f64,
    pub contact_width: f64,
    pub comb_floor: f64,
}

impl EngineComb {
    pub fn new(strike_position: f64, contact_width: f64, comb_floor: f64) -> Self {
        Self {
            strike_position,
            contact_width,
            comb_floor,
        }
    }

    /// `sqrt(sin^2(k pi x) + floor^2) * cos^2(k pi w / 2)`, term for term as the
    /// engine builds it.
    pub fn magnitude(&self, k: u32) -> f64 {
        let kf = f64::from(k);
        let sine = (kf * std::f64::consts::PI * self.strike_position).sin();
        let comb = if self.comb_floor > 0.0 {
            (sine * sine + self.comb_floor * self.comb_floor).sqrt()
        } else {
            sine.abs()
        };
        comb * crate::estimate::strike::contact_taper(kf, self.contact_width)
    }

    /// How far the comb's deepest point over `partials` stands below its crest
    /// over the same set, in dB. This is the quantity the floor is fitted
    /// against, and it is what `ANALYSIS.md` §4a's "comb nulls below −20 dB"
    /// column reports.
    pub fn deepest_db(&self, partials: &[u32]) -> f64 {
        let (mut low, mut high) = (f64::INFINITY, 0.0f64);
        for &k in partials {
            let c = self.magnitude(k);
            low = low.min(c);
            high = high.max(c);
        }
        if !(low.is_finite() && high > 0.0) {
            return 0.0;
        }
        20.0 * (low.max(1e-30) / high).log10()
    }
}

/// What one key's fits found, with the coverage that produced them.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NoteShaping {
    pub key: u8,
    /// Whether this key measured enough partials for a comb floor to be
    /// inverted from its deepest one at all. The floor itself is
    /// [`CombLine::floor_for`]'s answer, and needs renders this function does
    /// not do.
    pub floor_measurable: bool,
    /// Depth of the recording's deepest partial below a smooth envelope, dB
    /// (negative), medianed over the layers, and the partial index it sat at in
    /// the median layer.
    pub deepest_db: Option<f64>,
    pub deepest_k: Option<u32>,
    /// The same depth for the engine's *bare* comb (floor 0) over the partials
    /// the recording measured: the number the floor exists to move.
    pub bare_comb_db: f64,
    /// RMS of the per-partial deviation from the smooth envelope, dB — §3's
    /// roughness, reported so a fit can be read against the table it came from.
    pub roughness_db: f64,
    /// The row written into `notes.partial_gains`, trailing 1.0s trimmed.
    pub gains: Vec<f32>,
    /// Partials whose measured gain hit a bound.
    pub clamped_gains: usize,
    /// The row written into `notes.partial_sigma_scale`.
    pub sigma_scale: Vec<f32>,
    /// Partials that passed the decay stage's own trust gates.
    pub trusted_rates: usize,
    /// Partials the decay stage returned at all.
    pub offered_rates: usize,
}

impl NoteShaping {
    /// Span of the written gains, in dB, as `(quietest, loudest)`.
    pub fn gain_span_db(&self) -> (f64, f64) {
        let mut span = (0.0f64, 0.0f64);
        for &g in &self.gains {
            let db = 20.0 * f64::from(g).log10();
            span.0 = span.0.min(db);
            span.1 = span.1.max(db);
        }
        span
    }
}

/// Fits all three fields for one key, from every layer of it.
///
/// `comb` carries the strike position and contact width the preset will play;
/// its own `comb_floor` is ignored and replaced by the fitted one, so a caller
/// cannot accidentally fit the gains against a floor that is not the one being
/// written.
#[allow(clippy::too_many_arguments)]
pub fn fit_note(
    key: u8,
    layers: &[&DecayReport],
    comb: EngineComb,
    curve: DecayCurve,
    split: DecaySplit,
    decay: &DecayConfig,
    config: &ShapingConfig,
) -> NoteShaping {
    let spectra: Vec<Vec<(u32, f64)>> = layers
        .iter()
        .map(|report| {
            report
                .partials
                .iter()
                .filter(|fit| fit.k >= 1 && fit.initial_amplitude() > 0.0)
                .map(|fit| (fit.k, fit.initial_amplitude()))
                .collect()
        })
        .filter(|spectrum: &Vec<(u32, f64)>| spectrum.len() >= config.min_partials)
        .collect();

    let mut out = NoteShaping {
        key,
        ..NoteShaping::default()
    };
    if spectra.is_empty() {
        return out;
    }

    // ---- the deepest partial, and the floor that reproduces it
    let measured: Vec<u32> = union_of_partials(&spectra);
    let bare = EngineComb {
        comb_floor: 0.0,
        ..comb
    };
    out.bare_comb_db = bare.deepest_db(&measured);
    out.roughness_db = median(
        spectra
            .iter()
            .filter_map(|spectrum| roughness_rms_db(spectrum, config)),
    )
    .unwrap_or(0.0);
    if let Some((db, k)) = measured_deepest(&spectra, config) {
        out.deepest_db = Some(db);
        out.deepest_k = Some(k);
        out.floor_measurable = measured.len() >= config.min_floor_partials;
    }
    let _ = bare;

    out.gains = partial_gains(&spectra, comb, config);
    out.clamped_gains = out
        .gains
        .iter()
        .filter(|&&g| g == MIN_PARTIAL_GAIN || g == MAX_PARTIAL_GAIN)
        .count();

    let (sigma_scale, trusted, offered) =
        partial_sigma_scale(layers, curve, split, decay, config);
    out.sigma_scale = sigma_scale;
    out.trusted_rates = trusted;
    out.offered_rates = offered;
    out
}

/// What the engine's own excitation measures like at several comb floors, for
/// one key: the line [`CombLine::floor_for`] inverts.
///
/// # Why a line and not a formula
///
/// The floor is defined by what it does to the *deepest partial*, and how deep
/// a partial measures is not how deep the comb is. Two things stand between
/// them, and both are large:
///
/// * **Leakage.** A partial 20 dB under its neighbours, measured in a window
///   whose sidelobes are 31 dB down, reads several dB high. On the engine's own
///   render of C4 — where the comb is exactly `sin(k pi x)` and its null over the
///   measured partials is 17.3 dB deep — the deepest partial *measures*
///   **11.0 dB** down.
/// * **The smooth reference.** A comb null is a broad feature at low `k`, and the
///   degree-2 envelope [`deepest_partial`] fits absorbs part of a broad dip.
///
/// Neither is a defect: the recording is measured through exactly the same two,
/// and a fit that corrected for them on one side only would be wrong by their
/// difference. So the engine is asked directly — render the key at a few floors,
/// measure each render's deepest partial with the code the recording was
/// measured with, and invert. `estimate::directivity` (`DECISIONS.md` 137–138)
/// and `estimate::damper` invert lines measured on the engine for the same
/// reason.
///
/// The interpolation is piecewise-linear in the **amplitude** the depth
/// represents rather than in the decibel: measured on the engine, `10^(dB/20)`
/// against the floor is very nearly a straight line (C4: 0.282, 0.375, 0.485,
/// 0.630 at floors 0, 0.12, 0.24, 0.4), because a floor adds in quadrature under
/// a null and the null is where the deepest partial is.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CombLine {
    pub key: u8,
    /// `(comb_floor, deepest partial of the render, dB under a smooth envelope)`.
    pub probes: Vec<(f64, f64)>,
}

impl CombLine {
    /// The floor whose render's deepest partial measures `deepest_db`.
    ///
    /// `Some(0.0)` where the bare comb already measures shallower than the
    /// recording — a floor can only lift a null, and a lift the recording did
    /// not ask for is an invention. `None` where the probes do not draw a line.
    pub fn floor_for(&self, deepest_db: f64) -> Option<f64> {
        if !deepest_db.is_finite() {
            return None;
        }
        let mut probes: Vec<(f64, f64)> = self
            .probes
            .iter()
            .filter(|&&(floor, db)| floor.is_finite() && db.is_finite() && db <= 0.0)
            .map(|&(floor, db)| (floor, 10f64.powf(db / 20.0)))
            .collect();
        if probes.len() < 2 {
            return None;
        }
        probes.sort_by(|a, b| a.0.total_cmp(&b.0));
        let target = 10f64.powf(deepest_db / 20.0);
        if target <= probes[0].1 {
            return Some(probes[0].0);
        }
        if target >= probes[probes.len() - 1].1 {
            return Some(probes[probes.len() - 1].0);
        }
        for pair in probes.windows(2) {
            let ((f0, r0), (f1, r1)) = (pair[0], pair[1]);
            if target >= r0 && target <= r1 {
                if r1 <= r0 {
                    return Some(f0);
                }
                return Some(f0 + (f1 - f0) * (target - r0) / (r1 - r0));
            }
        }
        None
    }

    /// Whether the answer landed on an end of the probed range rather than
    /// inside it.
    pub fn saturated(&self, floor: f64) -> bool {
        let ends = self
            .probes
            .iter()
            .fold((f64::INFINITY, 0.0f64), |(lo, hi), &(f, _)| {
                (lo.min(f), hi.max(f))
            });
        floor <= ends.0 || floor >= ends.1
    }
}

/// The depth of the deepest partial of a key, medianed over its layers: the
/// number [`CombLine::floor_for`] is inverted at.
pub fn measured_deepest(
    spectra: &[Vec<(u32, f64)>],
    config: &ShapingConfig,
) -> Option<(f64, u32)> {
    let mut depths: Vec<(f64, u32)> = spectra
        .iter()
        .filter_map(|spectrum| deepest_partial(spectrum, config))
        .collect();
    if depths.is_empty() {
        return None;
    }
    depths.sort_by(|a, b| a.0.total_cmp(&b.0));
    Some(depths[depths.len() / 2])
}

/// The per-partial gains: the layers' median of `a_k(0)` over what a smooth
/// envelope times the engine's own comb puts at the same partial.
pub fn partial_gains(
    spectra: &[Vec<(u32, f64)>],
    comb: EngineComb,
    config: &ShapingConfig,
) -> Vec<f32> {
    let mut per_partial: Vec<Vec<f64>> = Vec::new();
    for spectrum in spectra {
        // The envelope is fitted to the spectrum with the comb *already divided
        // out*, so what it absorbs is the hammer, the bridge and the microphone
        // — everything smooth in `ln k` — and what is left is the roughness.
        let points: Vec<(f64, f64)> = spectrum
            .iter()
            .filter_map(|&(k, a)| {
                let c = comb.magnitude(k);
                (a > 0.0 && c > 0.0).then(|| (f64::from(k).ln(), a.ln() - c.ln()))
            })
            .collect();
        if points.len() <= config.envelope_degree {
            continue;
        }
        let x: Vec<f64> = points.iter().map(|p| p.0).collect();
        let y: Vec<f64> = points.iter().map(|p| p.1).collect();
        let Some(envelope) = robust_polyfit(&x, &y, config) else {
            continue;
        };
        for (i, &(k, _)) in spectrum
            .iter()
            .filter(|&&(k, a)| a > 0.0 && comb.magnitude(k) > 0.0)
            .enumerate()
        {
            let index = k as usize;
            if per_partial.len() < index {
                per_partial.resize(index, Vec::new());
            }
            per_partial[index - 1].push(y[i] - poly_eval(&envelope, x[i]));
        }
    }

    let mut gains: Vec<f32> = per_partial
        .iter()
        .map(|values| {
            if values.len() < config.min_layers {
                return 1.0;
            }
            let ln_gain = robust_median(values, config.outlier_db / NEPERS_TO_DB);
            match ln_gain {
                Some(ln) => (ln.exp() as f32).clamp(MIN_PARTIAL_GAIN, MAX_PARTIAL_GAIN),
                None => 1.0,
            }
        })
        .collect();
    // Trailing neutral entries say nothing the row's absence does not.
    while gains.last() == Some(&1.0) {
        gains.pop();
    }
    gains
}

/// The split the engine's composite envelope is built from: what a `sigma` table
/// entry has to be multiplied by to become the vertical bank's rate, and what
/// that rate has to be multiplied by to become a T60.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecaySplit {
    pub horizontal_gain_db: f64,
    pub horizontal_decay_ratio: f64,
}

impl DecaySplit {
    /// `vertical_decay_factor`, mirrored from [`crate::preset`].
    pub fn vertical_factor(&self) -> f64 {
        crate::preset::vertical_decay_factor(self.horizontal_gain_db, self.horizontal_decay_ratio)
    }

    /// The constant `sigma_v * T60` of the composite envelope
    /// `(e^{-s t} + g e^{-r s t}) / (1 + g)`.
    ///
    /// The envelope depends on `sigma_v` and `t` only through their product, so
    /// one bisection at `sigma_v = 1` gives the constant for every partial of
    /// every note that shares this split — which every note does, the split
    /// being global in the engine.
    pub fn t60_constant(&self) -> f64 {
        let g = 10f64.powf(self.horizontal_gain_db / 20.0);
        let r = self.horizontal_decay_ratio;
        let envelope = |x: f64| ((-x).exp() + g * (-r * x).exp()) / (1.0 + g);
        let (mut lo, mut hi) = (0.0, 1.0);
        while envelope(hi) > 1e-3 && hi < 1e9 {
            hi *= 2.0;
        }
        for _ in 0..100 {
            let mid = 0.5 * (lo + hi);
            if envelope(mid) > 1e-3 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }
}

/// The per-partial decay corrections: the layers' median of the rate a partial
/// actually decayed at over what the note's own `sigma(f)` law predicts, with
/// the row normalised so that it redistributes the note's damping instead of
/// retuning it.
///
/// Returns the row, how many partials passed the trust gates, and how many fits
/// the decay stage offered.
///
/// # Which measured rate, and why not the prompt one
///
/// The measured rate is inverted from the partial's **T60** through the engine's
/// own composite envelope — `sigma_v = t60_constant / T60` — and not read off
/// the two-exponential fit's fast component.
///
/// `DecayFit::fast.sigma` is the right quantity for the `sigma(f)` *line*
/// (`survey::prompt_decay_curve` uses it, and `TUNING_REPORT.md` §1 says why:
/// the last twenty decibels of a T60 are extrapolation and the prompt rate is
/// what the ear calls the decay). It is the wrong quantity for a *per-partial*
/// correction, and the gate says so by how much: on the engine's own renders of
/// A1, where every partial is on its law by construction and the answer must be
/// 1.0 everywhere, the prompt rate returns **1.73, 1.72, 1.62, 1.40, 1.43, 1.33,
/// 1.12, 1.18** for partials one to eight — the two-exponential decomposition
/// trading amplitude against rate between its own components, differently at
/// every partial. The same renders through the composite T60 return **1.05,
/// 1.05, 1.04, 1.03, 1.04, 1.03, 1.01, 1.02**. The gap is the measurement, not
/// the instrument.
///
/// # The trust gates, and the normalisation
///
/// The gates are the decay stage's own: a partial whose fitted decay reaches
/// further past the end of the record than [`DecayConfig::max_t60_ratio`] allows
/// has not been *seen* decay (the same test that keeps it out of the `sigma(f)`
/// line), one fitted from fewer than [`DecayConfig::min_points`] measurements
/// has not been fitted, and one whose envelope residual is worse than the ~4 dB
/// the law manages on any material (`TUNING_REPORT.md` §2) has not measured its
/// own rate. A partial that fails them is 1.0.
///
/// The row is then divided by its own geometric mean, for the reason
/// `voicing.unison_sigma_scale` is required to average to one: `notes.sigma0`
/// and `notes.sigma1` decide how long the note rings, this decides how its
/// partials share that, and a row with a mean in it would silently retune the
/// note behind the table's back. It also makes the field independent of *which*
/// convention the note's own `sigma(f)` line was fitted in, which is what lets
/// the prompt-rate line above and the T60 measurement here live in one preset.
pub fn partial_sigma_scale(
    layers: &[&DecayReport],
    curve: DecayCurve,
    split: DecaySplit,
    decay: &DecayConfig,
    config: &ShapingConfig,
) -> (Vec<f32>, usize, usize) {
    let mut per_partial: Vec<Vec<f64>> = Vec::new();
    let mut offered = 0usize;
    let vertical_factor = split.vertical_factor();
    let t60_constant = split.t60_constant();
    if !(vertical_factor.is_finite() && vertical_factor > 0.0 && t60_constant > 0.0) {
        return (Vec::new(), 0, 0);
    }
    for report in layers {
        for fit in &report.partials {
            if fit.k == 0 {
                continue;
            }
            offered += 1;
            let t60 = fit.t60();
            let trustworthy = t60.is_finite()
                && t60 > 0.0
                && fit.fast.sigma > 0.0
                && fit.frequency_hz > 0.0
                && fit.points >= decay.min_points
                && fit.residual_db.abs() <= config.max_decay_residual_db
                && t60 <= decay.max_t60_ratio * fit.span_s;
            if !trustworthy {
                continue;
            }
            // The table's convention: the engine builds the vertical bank at
            // `partial_sigma(k) * vertical_factor`, and that bank plus its
            // horizontal partner falls 60 dB in `t60_constant / sigma_v`.
            let measured = t60_constant / t60 / vertical_factor;
            let law = curve.sigma0 + curve.sigma1 * (fit.frequency_hz / 1000.0).powi(2);
            if !(law.is_finite() && law > 0.0 && measured.is_finite() && measured > 0.0) {
                continue;
            }
            let index = fit.k as usize;
            if per_partial.len() < index {
                per_partial.resize(index, Vec::new());
            }
            per_partial[index - 1].push((measured / law).ln());
        }
    }

    let mut trusted = 0usize;
    let mut logs: Vec<Option<f64>> = per_partial
        .iter()
        .map(|values| {
            if values.len() < config.min_layers {
                return None;
            }
            let ln = robust_median(values, config.outlier_db / NEPERS_TO_DB)?;
            trusted += 1;
            Some(ln)
        })
        .collect();
    // The mean of the *measured* entries only: an unmeasured partial is 1.0
    // because nothing was learned about it, not because it was measured at the
    // row's average.
    let measured: Vec<f64> = logs.iter().flatten().copied().collect();
    if measured.is_empty() {
        return (Vec::new(), 0, offered);
    }
    let mean = measured.iter().sum::<f64>() / measured.len() as f64;
    for slot in logs.iter_mut().flatten() {
        *slot -= mean;
    }
    let mut scales: Vec<f32> = logs
        .iter()
        .map(|slot| match slot {
            Some(ln) => {
                (ln.exp() as f32).clamp(MIN_PARTIAL_SIGMA_SCALE, MAX_PARTIAL_SIGMA_SCALE)
            }
            None => 1.0,
        })
        .collect();
    while scales.last() == Some(&1.0) {
        scales.pop();
    }
    (scales, trusted, offered)
}

/// The deepest partial of one spectrum below a smooth envelope through it, in dB
/// (negative), and which partial it was.
///
/// The reference is the same degree-2 polynomial in `ln k` `TUNING_REPORT.md` §3
/// measured its roughness against — a stiffer reference than the octave spline
/// `ANALYSIS.md` also reports, which is why the depths it returns are the larger
/// of the two.
pub fn deepest_partial(spectrum: &[(u32, f64)], config: &ShapingConfig) -> Option<(f64, u32)> {
    let (residuals, keys) = envelope_residuals(spectrum, config)?;
    let mut worst = (0.0f64, 0u32);
    for (r, k) in residuals.iter().zip(&keys) {
        let db = NEPERS_TO_DB * r;
        if db < worst.0 {
            worst = (db, *k);
        }
    }
    (worst.1 > 0).then_some(worst)
}

/// RMS of the same residuals — `TUNING_REPORT.md` §3's number, in dB.
pub fn roughness_rms_db(spectrum: &[(u32, f64)], config: &ShapingConfig) -> Option<f64> {
    let (residuals, _) = envelope_residuals(spectrum, config)?;
    let n = residuals.len() as f64;
    Some(NEPERS_TO_DB * (residuals.iter().map(|r| r * r).sum::<f64>() / n).sqrt())
}

/// Deviation of every measured partial from a smooth envelope through the
/// spectrum, in nepers, with the partial indices beside them.
fn envelope_residuals(
    spectrum: &[(u32, f64)],
    config: &ShapingConfig,
) -> Option<(Vec<f64>, Vec<u32>)> {
    let points: Vec<(u32, f64, f64)> = spectrum
        .iter()
        .filter(|&&(k, a)| k >= 1 && a > 0.0 && a.is_finite())
        .map(|&(k, a)| (k, f64::from(k).ln(), a.ln()))
        .collect();
    if points.len() <= config.envelope_degree.max(config.min_partials - 1) {
        return None;
    }
    let x: Vec<f64> = points.iter().map(|p| p.1).collect();
    let y: Vec<f64> = points.iter().map(|p| p.2).collect();
    let envelope = robust_polyfit(&x, &y, config)?;
    Some((
        x.iter()
            .zip(&y)
            .map(|(&x, &y)| y - poly_eval(&envelope, x))
            .collect(),
        points.iter().map(|p| p.0).collect(),
    ))
}

/// Least squares in the log domain, reweighted so that one partial sitting on a
/// soundboard resonance is worth a bounded amount of evidence about the
/// envelope's shape — `estimate::strike`'s own scheme, on the same material.
fn robust_polyfit(x: &[f64], y: &[f64], config: &ShapingConfig) -> Option<Vec<f64>> {
    let mut weights = vec![1.0; x.len()];
    let mut envelope = weighted_polyfit(x, y, &weights, config.envelope_degree)?;
    for _ in 1..config.irls_iterations.max(1) {
        for (i, weight) in weights.iter_mut().enumerate() {
            let residual = (y[i] - poly_eval(&envelope, x[i])).abs();
            *weight = if residual <= config.huber_delta {
                1.0
            } else {
                config.huber_delta / residual
            };
        }
        envelope = weighted_polyfit(x, y, &weights, config.envelope_degree)?;
    }
    Some(envelope)
}

/// The median with the outliers dropped: the median of everything within
/// `tolerance` of the median.
///
/// Not a mean of the survivors — a layer whose envelope fit went wrong is wrong
/// by tens of decibels, not by a few, and there are sixteen of them.
fn robust_median(values: &[f64], tolerance: f64) -> Option<f64> {
    let first = median(values.iter().copied())?;
    let kept: Vec<f64> = values
        .iter()
        .copied()
        .filter(|v| (v - first).abs() <= tolerance)
        .collect();
    median(kept.into_iter()).or(Some(first))
}

/// Every partial index any layer measured, ascending.
fn union_of_partials(spectra: &[Vec<(u32, f64)>]) -> Vec<u32> {
    let mut keys: Vec<u32> = spectra
        .iter()
        .flat_map(|spectrum| spectrum.iter().map(|&(k, _)| k))
        .collect();
    keys.sort_unstable();
    keys.dedup();
    keys
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
    use crate::estimate::decay::{DecayFit, EnvelopeBeats, Exponential, PolarizationSplit};
    use crate::preset::MAX_PARTIAL_GAIN;

    fn comb() -> EngineComb {
        EngineComb::new(0.1175, 0.0, 0.0)
    }

    /// A spectrum with a known per-partial pattern on it: a smooth envelope
    /// times the engine's comb times `pattern`, at a level that changes from
    /// layer to layer the way a velocity layer does.
    fn spectrum(comb: EngineComb, pattern: &[f64], level: f64, tilt: f64) -> Vec<(u32, f64)> {
        (1..=pattern.len() as u32)
            .map(|k| {
                let kf = f64::from(k);
                let envelope = level * kf.powf(tilt);
                (k, envelope * comb.magnitude(k) * pattern[k as usize - 1])
            })
            .collect()
    }

    #[test]
    fn a_known_per_partial_pattern_comes_back_whatever_the_layer_did_to_the_level() {
        let config = ShapingConfig::default();
        let mut pattern: Vec<f64> = (1..=24).map(|k| 1.0 + 0.3 * ((k as f64) * 1.7).sin()).collect();
        pattern[8] = 2.2;
        pattern[13] = 0.45;
        // Sixteen layers, 40 dB apart end to end and with the felt's own tilt
        // moving under them: everything an envelope absorbs.
        let spectra: Vec<Vec<(u32, f64)>> = (0..16)
            .map(|i| {
                let level = 10f64.powf(-2.0 + 0.13 * f64::from(i));
                spectrum(comb(), &pattern, level, -1.6 - 0.02 * f64::from(i))
            })
            .collect();
        let gains = partial_gains(&spectra, comb(), &config);
        assert_eq!(gains.len(), pattern.len());
        // The pattern is recovered up to the one thing a *smooth* reference
        // cannot separate from it: whatever part of the pattern is itself smooth
        // in `ln k`, which the fitted envelope absorbs by construction. What
        // that leaves is a small common scale and a gentle tilt, and the two
        // spikes — the part of the pattern no envelope can be — come back
        // exactly.
        let errors: Vec<f64> = gains
            .iter()
            .zip(&pattern)
            .map(|(&g, &p)| 20.0 * (f64::from(g) / p).log10())
            .collect();
        let rms = (errors.iter().map(|e| e * e).sum::<f64>() / errors.len() as f64).sqrt();
        assert!(rms < 1.0, "{rms:.2} dB RMS: {errors:?}");
        assert!(errors[8].abs() < 1.0, "the +7 dB spike: {:.2} dB", errors[8]);
        assert!(errors[13].abs() < 1.0, "the -7 dB dip: {:.2} dB", errors[13]);
        // The row does not move the note's loudness: the envelope under it is a
        // least-squares fit in the log domain, so the residuals it leaves — the
        // gains — have geometric mean 1.
        let mean: f64 = gains.iter().map(|&g| f64::from(g).ln()).sum::<f64>()
            / gains.len() as f64;
        assert!(mean.exp().ln().abs() < 0.05, "geometric mean {}", mean.exp());
    }

    #[test]
    fn a_partial_only_a_couple_of_layers_saw_is_left_at_one() {
        let config = ShapingConfig::default();
        let pattern = vec![1.0; 20];
        let mut spectra: Vec<Vec<(u32, f64)>> = (0..16)
            .map(|_| spectrum(comb(), &pattern, 1.0, -1.5))
            .collect();
        // Partial 20 survives in three layers out of sixteen, at +12 dB.
        for (i, s) in spectra.iter_mut().enumerate() {
            if i >= 3 {
                s.retain(|&(k, _)| k < 20);
            } else {
                s.last_mut().expect("non-empty").1 *= 4.0;
            }
        }
        let gains = partial_gains(&spectra, comb(), &config);
        assert!(
            gains.len() < 20 || (f64::from(gains[19]) - 1.0).abs() < 1e-6,
            "{gains:?}"
        );
    }

    #[test]
    // One of the measured depths below is -6.28 dB, which clippy reads as an
    // approximation of tau.
    #[allow(clippy::approx_constant)]
    fn the_floor_is_read_off_the_line_the_engine_draws_and_nowhere_else() {
        // The engine's own numbers at C4 (`gate_probe`, `presets/default.toml`
        // with the diffuse field off): floors 0, 0.06, 0.12, 0.24 and 0.4 put
        // the deepest measured partial at these depths.
        let line = CombLine {
            key: 60,
            probes: vec![
                (0.0, -10.97),
                (0.06, -10.33),
                (0.12, -8.51),
                (0.24, -6.28),
                (0.40, -4.02),
            ],
        };
        for &(floor, db) in &line.probes {
            let back = line.floor_for(db).expect("a line");
            assert!(
                (back - floor).abs() < 1e-6,
                "a probe at {floor} came back as {back}"
            );
        }
        // Between the probes, in the amplitude domain the depths are nearly a
        // straight line in.
        let between = line.floor_for(-7.3).expect("a line");
        assert!((0.12..0.24).contains(&between), "{between}");
        // A recording deeper than the bare comb measures asks for no floor, and
        // one shallower than the deepest probe saturates rather than
        // extrapolating.
        assert_eq!(line.floor_for(-20.0), Some(0.0));
        assert_eq!(line.floor_for(-1.0), Some(0.40));
        assert!(line.saturated(0.0) && line.saturated(0.4) && !line.saturated(0.12));
        // Two probes that say the parameter did nothing are not a line.
        let flat = CombLine {
            key: 60,
            probes: vec![(0.0, -9.0)],
        };
        assert_eq!(flat.floor_for(-8.0), None);
    }

    #[test]
    fn the_gains_are_measured_against_the_comb_the_floor_makes() {
        // The other half of the split the module header documents: with the
        // floor in the reference the gains are one, and against the bare comb
        // the same spectrum asks for more than the schema allows — at exactly
        // the partial the floor exists for.
        let config = ShapingConfig::default();
        let bare = comb();
        let floored = EngineComb {
            comb_floor: 0.2,
            ..bare
        };
        let pattern = vec![1.0; 30];
        let spectra: Vec<Vec<(u32, f64)>> = (0..16)
            .map(|i| spectrum(floored, &pattern, 10f64.powf(-0.1 * f64::from(i)), -1.5))
            .collect();
        let gains = partial_gains(&spectra, floored, &config);
        for (k, &g) in gains.iter().enumerate() {
            let db = 20.0 * f64::from(g).log10();
            assert!(db.abs() < 1.0, "partial {} at {db:.2} dB", k + 1);
        }
        let against_bare = partial_gains(&spectra, bare, &config);
        assert!(
            against_bare.iter().any(|&g| g == MAX_PARTIAL_GAIN),
            "{against_bare:?}"
        );
    }

    fn fit(k: u32, frequency_hz: f64, sigma: f64, residual_db: f64, span_s: f64) -> DecayFit {
        DecayFit {
            k,
            frequency_hz,
            fast: Exponential {
                amplitude: 1.0,
                sigma,
            },
            slow: Exponential {
                amplitude: 0.0,
                sigma,
            },
            beats: EnvelopeBeats::default(),
            residual_db,
            points: 200,
            span_s,
        }
    }

    fn report(partials: Vec<DecayFit>) -> DecayReport {
        DecayReport {
            partials,
            curve: DecayCurve {
                sigma0: 0.0,
                sigma1: 0.0,
                residual: 0.0,
            },
            polarization: PolarizationSplit {
                gain_db: -20.0,
                decay_ratio: 0.3,
                partials: 0,
            },
        }
    }

    /// The engine's default split, which every note in a preset shares.
    fn split() -> DecaySplit {
        DecaySplit {
            horizontal_gain_db: -12.0,
            horizontal_decay_ratio: 0.29,
        }
    }

    #[test]
    fn a_partial_that_decays_off_its_notes_law_comes_back_as_the_ratio() {
        let config = ShapingConfig::default();
        let decay = DecayConfig::default();
        let split = split();
        let curve = DecayCurve {
            sigma0: 1.2,
            sigma1: 6.0,
            residual: 0.0,
        };
        // A `DecayFit` with no slow component falls 60 dB in `ln(1000)/sigma`,
        // and the split's own constant divided by the vertical factor undoes
        // itself, so a fit built at `law * scale` is a partial decaying at
        // `scale` times its note's law — which is what this measures.
        let truth = |k: u32| match k {
            3 => 1.8,
            6 => 0.4,
            _ => 1.0,
        };
        let reports: Vec<DecayReport> = (0..16)
            .map(|_| {
                report(
                    (1..=8)
                        .map(|k| {
                            let f = 261.6 * f64::from(k);
                            let law = curve.sigma0 + curve.sigma1 * (f / 1000.0).powi(2);
                            fit(k, f, law * truth(k), 1.0, 60.0)
                        })
                        .collect(),
                )
            })
            .collect();
        let borrowed: Vec<&DecayReport> = reports.iter().collect();
        let (scales, trusted, offered) =
            partial_sigma_scale(&borrowed, curve, split, &decay, &config);
        assert_eq!(offered, 16 * 8);
        // Every partial passed the gates: a written 1.0 is a measurement that
        // the law was right there.
        assert_eq!(trusted, 8, "{scales:?}");
        // The row is normalised: it redistributes the note's damping and does
        // not retune it, so what comes back is the pattern divided by its own
        // geometric mean.
        let mean: f64 =
            ((1..=8).map(|k| truth(k).ln()).sum::<f64>() / 8.0).exp();
        assert!((mean - 0.9598).abs() < 1e-3, "{mean}");
        for k in 1..=8u32 {
            let got = f64::from(scales[k as usize - 1]);
            let want = truth(k) / mean;
            assert!(
                (got / want - 1.0).abs() < 0.01,
                "partial {k}: {want:.4} came back as {got:.4}"
            );
        }
        // ... and the geometric mean of what was written is one.
        let written: f64 = (scales.iter().map(|&s| f64::from(s).ln()).sum::<f64>()
            / scales.len() as f64)
            .exp();
        assert!((written - 1.0).abs() < 1e-4, "{written}");
    }

    #[test]
    fn a_partial_the_decay_stage_does_not_trust_is_left_at_one() {
        let config = ShapingConfig::default();
        let decay = DecayConfig::default();
        let split = split();
        let curve = DecayCurve {
            sigma0: 1.0,
            sigma1: 0.0,
            residual: 0.0,
        };
        // Partial 2's own T60 reaches seven times past the end of a one-second
        // record and the decay stage's gate is six; partial 3 is fitted with a
        // 9 dB residual, which is worse than the envelope law manages on any
        // material. Neither has measured anything, so only partial 1 is left —
        // and one partial says nothing about how a note's damping is
        // *distributed*, which the normalisation states by returning nothing.
        let reports: Vec<DecayReport> = (0..16)
            .map(|_| {
                report(vec![
                    fit(1, 261.6, 2.0, 1.0, 60.0),
                    fit(2, 523.2, 1.0, 1.0, 1.0),
                    fit(3, 784.8, 4.0, 9.0, 60.0),
                ])
            })
            .collect();
        let borrowed: Vec<&DecayReport> = reports.iter().collect();
        let (scales, trusted, _) = partial_sigma_scale(&borrowed, curve, split, &decay, &config);
        assert_eq!(trusted, 1, "{scales:?}");
        assert!(scales.is_empty(), "{scales:?}");
    }
}
