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
//! least-squares one in the log domain, so the residuals it leaves sum to zero
//! and a row of gains has geometric mean 1.
//!
//! **That is not the same as leaving the note's loudness alone**, and
//! `DECISIONS.md` 272 is the measurement of the difference: a level is a sum of
//! powers, so a row with a zero log mean and a spread of `s` dB multiplies a
//! note's power by about `s^2 ln 10 / 40` dB — up to +25 dB of rendered RMS on
//! the shipped rows. The live fit ([`measured_over_rendered`]) is pinned on the
//! power instead ([`energy_offset`]) and then on the render itself; the two
//! functions above it in this file are the history the schema came through and
//! their geometric means are stated as what they are, not as a level
//! guarantee.

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
    /// Fewest partials a layer must have measured on **both** sides before it
    /// contributes to [`measured_over_rendered`]. Lower than
    /// [`ShapingConfig::min_partials`], which gates a *polynomial* fit and
    /// therefore needs more points than it has parameters: this is a
    /// partial-by-partial ratio with no parameters at all, and the top of the
    /// compass has three partials under Nyquist.
    pub min_gain_partials: usize,
    /// How far under a spectrum's own loudest partial a partial may be and still
    /// contribute to [`measured_over_rendered`], in dB. Below it the tracker is
    /// measuring its own floor on one side or the other and the ratio is a ratio
    /// of two noises.
    pub max_gain_range_db: f64,
    /// How many robust sigmas of a key's **own** cell distribution a written
    /// cell may reach, before the schema's rails are applied at all.
    ///
    /// The schema's `MIN_PARTIAL_GAIN`/`MAX_PARTIAL_GAIN` are ±26.02 dB, which
    /// is a limit on what the *file* may say and not a statement about what any
    /// key measured. Fitted against them, five keys wrote a cell at exactly
    /// ±26.02 — C7 and D#7 wrote it on their **fundamental** — and a rail is not
    /// a measurement. The rail here is the key's own: `RAIL_SIGMAS` times
    /// `1.4826 MAD` of its cells' departures from the row's centre, floored at
    /// [`ShapingConfig::min_rail_db`] so a key whose cells genuinely agree can
    /// still carry a real dip.
    pub rail_sigmas: f64,
    /// Floor on that rail, in dB: no key is railed tighter than this however
    /// well its cells agree.
    pub min_rail_db: f64,
    /// Fewest cells a row must have before its own MAD is allowed to set the
    /// rail. Below this a spread is not a distribution — C7 and D#7 have three
    /// cells each, and three numbers one of which is the outlier have a MAD the
    /// outlier itself sets. Such a row is railed at
    /// [`ShapingConfig::min_rail_db`]: with three partials the fit cannot tell a
    /// per-partial correction from a level, and must not claim to.
    pub min_rail_cells: usize,
    /// Partials the step statistic [`temper`] bisects against is taken over.
    ///
    /// Twelve, which is `compass_scan`'s own `PARTIALS`: the acceptance
    /// criterion for this regularisation is that report's `irregular` column, so
    /// the statistic fitted against is the statistic scored against. Taken over
    /// all 48 instead, a bass key's target is set by the scatter of its own
    /// fortieth partial near the tracker's floor — measured, C4's target goes
    /// 7.6 dB (over twelve) to 13.5 (over forty-eight), and a row licensed at
    /// 13.5 is the row that renders at 13.9 and was heard.
    pub step_partials: usize,
    /// How closely the bisected smoother has to land on the recording's own
    /// step statistic, in dB, before [`temper`] stops.
    pub step_tolerance_db: f64,
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
            min_gain_partials: 3,
            max_gain_range_db: 60.0,
            max_decay_residual_db: 4.0,
            rail_sigmas: 3.0,
            min_rail_db: 6.0,
            min_rail_cells: 5,
            step_partials: 12,
            step_tolerance_db: 0.05,
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

/// The per-partial multiplier that puts the engine's **smooth spectral
/// envelope** on the recording's.
///
/// # The half of the excitation [`partial_gains`] deliberately threw away
///
/// [`partial_gains`] fits a polynomial in `ln k` to each layer's spectrum and
/// writes only what is left over — the roughness — on the reasoning that
/// everything smooth in `ln k` is the hammer, the bridge and the microphone, and
/// that the engine has models for all three. That reasoning is half right. The
/// envelope really does contain the part of the blow that changes with velocity,
/// which a velocity-independent table must not carry. What it also contains is
/// the engine's own error in that envelope, and *that* does not change with
/// velocity, and nothing else in the schema can carry it.
///
/// The measurement that convicts it is one number: at C4, velocity 90, the
/// recording's strongest partial is **k = 2** and the engine's is **k = 1**. A
/// note whose fundamental leads where the recording's second partial does is a
/// note the ear places an octave differently at the attack — which is what a
/// listener reported of the jitter A/B pair, before any of it was measured. It
/// has been true since the felt fits landed in Phase E, and the roughness gains
/// are innocent of it by construction: their geometric mean is 1 and their
/// reference is a polynomial that has already absorbed the tilt.
///
/// # What this does instead
///
/// The same polynomial, fitted twice — once to the recording and once to a
/// **render of the engine** at the same key and the same velocity, measured by
/// the same tracker — and the difference of the two written into the same table.
/// This is `estimate::directivity`'s pattern and `CombLine`'s and
/// `strike_offset`'s (`DECISIONS.md` 137, 199, 211): a quantity that is only
/// meaningful as "how far is the engine from the recording" is inverted *on the
/// engine*, so everything both signals go through — the tracker's own bias, the
/// window, the microphone's comb — divides out instead of being modelled.
///
/// Two properties it keeps, because they are what makes the table legal:
///
/// * **The geometric mean is 1.** The difference is offset so that its mean over
///   the partials both spectra measured is zero, exactly as the roughness half
///   is normalised. This was written as "the level does not move" and it is not
///   that — see the module header and `DECISIONS.md` 272. Superseded by
///   [`measured_over_rendered`], which pins the power.
/// * **It is velocity-independent.** Both envelopes are taken at the reference
///   velocity, so what is written is the *mismatch* at that velocity and not the
///   blow's own tilt, which is still absorbed on both sides.
///
/// Returns `None` when either spectrum has too few partials to fit an envelope
/// to, which is a key that measured nothing rather than a key that needs no
/// correction.
pub fn envelope_tilt(
    recording: &[(u32, f64)],
    engine: &[(u32, f64)],
    partials: usize,
    config: &ShapingConfig,
) -> Option<Vec<f32>> {
    let fit = |spectrum: &[(u32, f64)]| -> Option<(Vec<f64>, u32, u32)> {
        let points: Vec<(f64, f64)> = spectrum
            .iter()
            .filter(|&&(k, a)| k >= 1 && a > 0.0)
            .map(|&(k, a)| (f64::from(k).ln(), a.ln()))
            .collect();
        if points.len() < config.min_partials.max(config.envelope_degree + 2) {
            return None;
        }
        let x: Vec<f64> = points.iter().map(|p| p.0).collect();
        let y: Vec<f64> = points.iter().map(|p| p.1).collect();
        let lo = spectrum.iter().filter(|&&(_, a)| a > 0.0).map(|&(k, _)| k).min()?;
        let hi = spectrum.iter().filter(|&&(_, a)| a > 0.0).map(|&(k, _)| k).max()?;
        Some((robust_polyfit(&x, &y, config)?, lo, hi))
    };
    let (recorded, r_lo, r_hi) = fit(recording)?;
    let (rendered, e_lo, e_hi) = fit(engine)?;
    // Only where both were measured: extrapolating either polynomial past its
    // own data is how a degree-3 fit turns into a ±40 dB correction.
    let (lo, hi) = (r_lo.max(e_lo), r_hi.min(e_hi).min(partials as u32));
    if hi <= lo {
        return None;
    }
    let difference = |k: u32| poly_eval(&recorded, f64::from(k).ln()) - poly_eval(&rendered, f64::from(k).ln());
    let mean: f64 =
        (lo..=hi).map(difference).sum::<f64>() / f64::from(hi - lo + 1);
    Some(
        (1..=partials)
            .map(|k| {
                let k = (k as u32).clamp(lo, hi);
                ((difference(k) - mean).exp() as f32)
                    .clamp(MIN_PARTIAL_GAIN, MAX_PARTIAL_GAIN)
            })
            .collect(),
    )
}

/// The per-partial gains as the schema now defines them: the **full** measured
/// ratio `a_k(0) recorded / a_k(0) as the engine itself renders it`.
///
/// # Why this replaces the roughness fit and the envelope tilt both
///
/// [`partial_gains`] writes the residual left after a smooth polynomial in
/// `ln k` is divided out; [`envelope_tilt`] writes the difference between two
/// such polynomials. `DECISIONS.md` 237 settled what the field *is* — "`a_k(0)`
/// measured over `a_k(0)` as the engine's own excitation model predicts it" —
/// and once the engine is rendered and measured rather than predicted from a
/// formula, the two halves are one subtraction and there is no reason to take it
/// in two pieces. The smooth part and the rough part of the same log ratio do
/// not need separating: nothing downstream reads them apart.
///
/// # Three properties, all of them load-bearing
///
/// * **Re-entrant.** The probe the engine is rendered from must have the key's
///   own row *cleared*, and then the ratio is absolute: running the fit on its
///   own output returns the same table. This is what `fit_partials`'s header
///   warns is not true of the roughness fit, and it is true here by
///   construction rather than by discipline.
/// * **Layer-robust.** Each layer is rendered at its own velocity, so the
///   engine's velocity law is applied on both sides and what is left is the
///   velocity-independent mismatch. Every layer's ratio is levelled (its own mean
///   over the partials both spectra measured is removed) before the layers are
///   combined, so a layer that is simply louder does not tilt the median.
/// * **The level does not move.** The written row is normalised so that the
///   *power* it puts through the engine's own rendered spectrum is the power
///   that was already there — [`energy_offset`]. This is not the geometric mean
///   the two halves above used, and the difference between the two is the whole
///   of `DECISIONS.md` 272's leak.
///
/// What is written per partial is not the raw ratio: [`write_row`] reads each
/// cell as a measurement with an error bar, shrinks it toward the row's own
/// smooth curve by how much the cell's velocity layers agreed about it, and
/// gives a partial the layers could not reach the curve outright instead of a
/// `1.0` that would be a tooth. [`temper`] then smooths what is left until the
/// series the row *predicts the engine will render* is no more jagged than the
/// recording's own. Their headers carry the measurements that say why.
///
/// Returns `None` when no layer measured enough partials on both sides.
pub fn measured_over_rendered(
    recorded: &[Vec<(u32, f64)>],
    rendered: &[Vec<(u32, f64)>],
    partials: usize,
    config: &ShapingConfig,
) -> Option<Vec<f32>> {
    measured_over_rendered_report(recorded, rendered, partials, config).map(|row| row.gains)
}

/// One key's fitted gain row and the four numbers that say how it was
/// disciplined.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GainRow {
    /// The row itself, trailing 1.0s trimmed.
    pub gains: Vec<f32>,
    /// The broadband level the fit found: how much louder the recording is than
    /// the engine over the partials the row was measured on, in dB, **after**
    /// the row's shape is taken out. Removed from the row by [`energy_offset`]
    /// and reported here rather than written — see `DECISIONS.md` 272.
    pub level_db: f64,
    /// The rail this key's own cells earned, in dB, and how many cells reached
    /// it.
    pub rail_db: f64,
    pub railed: usize,
    /// The recording's own mean absolute step between adjacent partials, dB —
    /// the target [`temper`] bisects against.
    pub target_step_db: f64,
    /// The same statistic on the series the row predicts the engine will
    /// render. Within [`ShapingConfig::step_tolerance_db`] of the target unless
    /// the smoother ran out of range.
    pub written_step_db: f64,
    /// The smoothing the bisection had to use to get there. Zero is a row whose
    /// raw cells were already no rougher than the recording.
    pub lambda: f64,
}

/// The same fit, with the discipline reported.
pub fn measured_over_rendered_report(
    recorded: &[Vec<(u32, f64)>],
    rendered: &[Vec<(u32, f64)>],
    partials: usize,
    config: &ShapingConfig,
) -> Option<GainRow> {
    let mut per_partial: Vec<Vec<f64>> = vec![Vec::new(); partials];
    for (recording, engine) in recorded.iter().zip(rendered) {
        let engine_at: std::collections::BTreeMap<u32, f64> = engine
            .iter()
            .filter(|&&(_, a)| a > 0.0)
            .map(|&(k, a)| (k, a))
            .collect();
        // Only partials both sides measured *well*: a partial 60 dB under its
        // own note is at the tracker's floor on either signal, and a ratio of
        // two floors is a ratio of two noises.
        let loudest = |spectrum: &[(u32, f64)]| {
            spectrum
                .iter()
                .map(|&(_, a)| a)
                .fold(0.0f64, f64::max)
                .max(f64::MIN_POSITIVE)
        };
        let (recorded_top, engine_top) = (loudest(recording), loudest(engine));
        let floor = 10f64.powf(-config.max_gain_range_db / 20.0);
        let ratios: Vec<(u32, f64)> = recording
            .iter()
            .filter(|&&(k, a)| a > 0.0 && k >= 1 && (k as usize) <= partials)
            .filter(|&&(_, a)| a >= floor * recorded_top)
            .filter_map(|&(k, a)| engine_at.get(&k).map(|e| (k, a, *e)))
            .filter(|&(_, _, e)| e >= floor * engine_top)
            .map(|(k, a, e)| (k, (a / e).ln()))
            .collect();
        if ratios.len() < config.min_gain_partials {
            continue;
        }
        // Levelled per layer, by the **median**: what is compared between layers
        // is the *shape* of the mismatch, not how loud that layer happened to be
        // rendered, and a mean over a table that spans 40 dB is set by its
        // extremes.
        let Some(centre) = median(ratios.iter().map(|&(_, r)| r)) else {
            continue;
        };
        for (k, r) in ratios {
            per_partial[k as usize - 1].push(r - centre);
        }
    }
    let mut measured: Vec<Option<Cell>> = per_partial
        .iter()
        .map(|values| {
            (values.len() >= config.min_layers.min(recorded.len().max(1)))
                .then(|| Cell::of(values, config))
                .flatten()
        })
        .collect();
    let seen: Vec<f64> = measured.iter().flatten().map(|c| c.centre).collect();
    if seen.is_empty() {
        return None;
    }
    // A provisional pin, which decides only where the smooth curve sits: the
    // level the row is finally written at is [`energy_offset`]'s, taken on the
    // engine's own spectrum after the shape is settled.
    let offset = median(seen.iter().copied()).expect("seen is not empty");
    // The key's own rail, before the schema's. A cell past it is not this key's
    // measurement.
    let (rail_db, railed) = rail_cells(&mut measured, offset, config);

    let written = write_row(&measured, offset, config);

    // The two spectra the discipline is measured against: the reference layer
    // of each side, which is the velocity the compass strikes at.
    let reference = |set: &[Vec<(u32, f64)>]| -> Vec<Option<f64>> {
        let Some(spectrum) = set.get(set.len() / 2) else {
            return vec![None; partials];
        };
        let mut out = vec![None; partials];
        for &(k, a) in spectrum {
            if k >= 1 && (k as usize) <= partials && a > 0.0 {
                out[k as usize - 1] = Some(a.ln());
            }
        }
        out
    };
    let recording_line = reference(recorded);
    let engine_line = reference(rendered);

    let (tempered, target_step, written_step, lambda) =
        temper(&written, &measured, &engine_line, &recording_line, config);

    // The level, last and on the engine's own spectrum: the row multiplies
    // amplitudes and a level is a sum of *powers*, so a row pinned in the log
    // domain moves one and a row pinned here does not.
    let (levelled, level_db) = energy_offset(&tempered, &engine_line);

    let mut gains: Vec<f32> = levelled
        .iter()
        .map(|value| match value {
            Some(ln) => (ln.exp() as f32).clamp(MIN_PARTIAL_GAIN, MAX_PARTIAL_GAIN),
            None => 1.0,
        })
        .collect();
    while gains.last() == Some(&1.0) {
        gains.pop();
    }
    (!gains.is_empty()).then_some(GainRow {
        gains,
        level_db,
        rail_db,
        railed,
        target_step_db: target_step,
        written_step_db: written_step,
        lambda,
    })
}

/// Clips every cell to the key's **own** measured spread, and reports the rail
/// and how many cells reached it.
///
/// # Why the schema's rails are the wrong ones to fit against
///
/// `MIN_PARTIAL_GAIN`/`MAX_PARTIAL_GAIN` are ±26.02 dB. They are a statement
/// about what a preset file may contain, and fitting against them makes the
/// *file format* the estimator's prior. Measured on the shipped preset, five
/// keys wrote a cell at exactly ±26.02 dB and two of them — C7 and D#7 — wrote
/// it on the **fundamental**, where it is 26 dB of the note's whole power. A
/// cell at a rail is not a measurement of the piano, it is the fit running out
/// of room, and running out of room is what a fit does when the two spectra it
/// divides disagree for a reason neither of them is about.
///
/// The rail used instead is the key's own: `rail_sigmas` times `1.4826 MAD` of
/// its cells' departures from the row's centre, floored at `min_rail_db` and
/// capped by the schema. On the shipped rows that is 6 dB at C7 (whose three
/// cells otherwise read +26.0, 0.0, −1.2) and 24 dB at C2, whose cells really
/// do span that — which is the point: the rail is measured, so a key with a
/// genuine deep dip keeps it and a key with one wild cell does not.
fn rail_cells(measured: &mut [Option<Cell>], offset: f64, config: &ShapingConfig) -> (f64, usize) {
    let deviations: Vec<f64> = measured
        .iter()
        .flatten()
        .map(|c| c.centre - offset)
        .collect();
    let schema = f64::from(MAX_PARTIAL_GAIN).ln().min(-f64::from(MIN_PARTIAL_GAIN).ln());
    let Some(centre) = median(deviations.iter().copied()) else {
        return (NEPERS_TO_DB * schema, 0);
    };
    let mad = median(deviations.iter().map(|d| (d - centre).abs())).unwrap_or(0.0);
    let earned = if deviations.len() >= config.min_rail_cells {
        config.rail_sigmas * 1.4826 * mad
    } else {
        0.0
    };
    let rail = earned.max(config.min_rail_db / NEPERS_TO_DB).min(schema);
    let mut railed = 0usize;
    for cell in measured.iter_mut().flatten() {
        let deviation = cell.centre - offset;
        if deviation > rail + centre {
            cell.centre = offset + centre + rail;
            railed += 1;
        } else if deviation < centre - rail {
            cell.centre = offset + centre - rail;
            railed += 1;
        }
    }
    (NEPERS_TO_DB * rail, railed)
}

/// Smooths the written row until the harmonic series it predicts the engine will
/// render is **no more jagged than the recording's own**, and returns
/// `(row, target, achieved, lambda)` in dB.
///
/// # The measurement this exists for
///
/// `renders/compass/COMPASS.md`'s `irregular` — the mean absolute step between
/// adjacent partial levels — is 4 dB *higher* on the fitted keys than on the
/// recordings they were fitted to, and 5 dB *lower* on the 58 keys with no row
/// at all. A listener found the second half of that as one note of a melody that
/// "sounds different from the rest": C4 renders at 13.9 dB of `irregular`
/// against its own recording's 7.0 and its unfitted neighbours' 5-6.
///
/// The row cannot be blamed for having steps in it — the recording has 5-10 dB
/// of measured scatter and reproducing it is the entire reason the field exists.
/// What it can be blamed for is having *more* steps than the recording, because
/// every one of those is the tracker's noise written into the instrument. So the
/// criterion is neither "smooth" nor "as rough as the rails allow" but **the
/// recording's own roughness, per key**, and it is available inside the fit:
/// both spectra are already in hand.
///
/// # How
///
/// A Whittaker smoother on the row's log cells — minimise
/// `sum p_k (d_k - y_k)^2 + lambda sum (d_{k+1} - d_k)^2 / dk` — with `p_k` the
/// precision the cell earned from its own velocity layers, so a cell the layers
/// agreed on resists smoothing and a cell they scattered about does not. The one
/// free number, `lambda`, is not a constant: it is **bisected per key** until
/// the predicted series `ln e_k + d_k` has the recording's mean absolute step.
/// A key whose raw cells are already smoother than its recording keeps them
/// untouched (`lambda = 0`) — the target is the recording's roughness, and this
/// never removes roughness the recording has.
///
/// The statistic is taken over adjacent partials **both** sides measured, so
/// the recording's own gaps do not count as steps on either side of the
/// comparison.
fn temper(
    written: &[Option<f64>],
    measured: &[Option<Cell>],
    engine: &[Option<f64>],
    recording: &[Option<f64>],
    config: &ShapingConfig,
) -> (Vec<Option<f64>>, f64, f64, f64) {
    // The partial pairs the statistic is defined on: the compass's own range,
    // and only where both sides measured both members of the pair.
    let top = written.len().min(config.step_partials).saturating_sub(1);
    let pairs: Vec<usize> = (0..top)
        .filter(|&i| {
            written[i].is_some()
                && written[i + 1].is_some()
                && engine.get(i).copied().flatten().is_some()
                && engine.get(i + 1).copied().flatten().is_some()
                && recording.get(i).copied().flatten().is_some()
                && recording.get(i + 1).copied().flatten().is_some()
        })
        .collect();
    let step = |line: &dyn Fn(usize) -> f64| -> f64 {
        if pairs.is_empty() {
            return 0.0;
        }
        NEPERS_TO_DB * pairs.iter().map(|&i| (line(i + 1) - line(i)).abs()).sum::<f64>()
            / pairs.len() as f64
    };
    let at = |line: &[Option<f64>], i: usize| line.get(i).copied().flatten().unwrap_or(0.0);
    let target = step(&|i| at(recording, i));

    // The cells the smoother works on, in row order, with their own precisions.
    let index: Vec<usize> = (0..written.len()).filter(|&i| written[i].is_some()).collect();
    let y: Vec<f64> = index.iter().map(|&i| written[i].expect("some")).collect();
    let floor = measured
        .iter()
        .flatten()
        .map(|c| c.variance)
        .fold(f64::INFINITY, f64::min)
        .max(1e-6);
    let precision: Vec<f64> = index
        .iter()
        .map(|&i| match measured.get(i).copied().flatten() {
            Some(cell) => 1.0 / cell.variance.max(floor),
            // A hole filled from the curve is already smooth and carries no
            // evidence of its own: it follows its neighbours.
            None => 1.0 / floor * 1e-3,
        })
        .collect();

    let rebuild = |smoothed: &[f64]| -> Vec<Option<f64>> {
        let mut out = vec![None; written.len()];
        for (slot, &i) in index.iter().enumerate() {
            out[i] = Some(smoothed[slot]);
        }
        out
    };
    let achieved = |smoothed: &[f64]| -> f64 {
        let row = rebuild(smoothed);
        step(&|i| at(engine, i) + at(&row, i))
    };

    if pairs.is_empty() || index.len() < 3 || target <= 0.0 {
        return (written.to_vec(), target, achieved(&y), 0.0);
    }
    let raw = achieved(&y);
    if raw <= target + config.step_tolerance_db {
        // Already no rougher than the piano: nothing to take out, and taking
        // any out would be taking out the measurement.
        return (written.to_vec(), target, raw, 0.0);
    }
    // Bisect in `ln lambda`. The smoother is monotone in it — more weight on the
    // differences is never a rougher row — so a bisection is exact.
    let gaps: Vec<f64> = index.windows(2).map(|w| (w[1] - w[0]) as f64).collect();
    let (mut lo, mut hi) = (1e-4f64, 1e-4f64);
    let mut best = y.clone();
    for _ in 0..40 {
        hi *= 4.0;
        let smoothed = whittaker(&y, &precision, &gaps, hi);
        if achieved(&smoothed) <= target {
            best = smoothed;
            break;
        }
        lo = hi;
    }
    if achieved(&best) > target + config.step_tolerance_db {
        // The smoother saturated: even a straight line through the cells is
        // rougher than the recording, which happens where the engine's own
        // series is the jagged one. The row is not the place to fix that.
        let got = achieved(&best);
        return (rebuild(&best), target, got, hi);
    }
    for _ in 0..60 {
        let mid = (lo * hi).sqrt();
        let smoothed = whittaker(&y, &precision, &gaps, mid);
        let got = achieved(&smoothed);
        let close = (got - target).abs() <= config.step_tolerance_db;
        if got > target {
            lo = mid;
            if close {
                best = smoothed;
            }
        } else {
            hi = mid;
            best = smoothed;
        }
        if close {
            break;
        }
    }
    let got = achieved(&best);
    (rebuild(&best), target, got, hi)
}

/// Whittaker–Henderson: minimises `sum p_i (d_i - y_i)^2 + lambda sum (d_{i+1} -
/// d_i)^2 / gap_i`, by the tridiagonal solve its normal equations are.
///
/// The gap divisor is what lets the row have holes in it: two cells four
/// partials apart are a quarter as strongly tied as two adjacent ones, which is
/// the same weighting a linear interpolation between them would imply.
fn whittaker(y: &[f64], precision: &[f64], gaps: &[f64], lambda: f64) -> Vec<f64> {
    let n = y.len();
    if n == 0 {
        return Vec::new();
    }
    let (mut diag, mut upper, mut lower, mut rhs) =
        (precision.to_vec(), vec![0.0; n], vec![0.0; n], Vec::with_capacity(n));
    for i in 0..n {
        rhs.push(precision[i] * y[i]);
    }
    for (i, &gap) in gaps.iter().enumerate() {
        let w = lambda / gap.max(1.0);
        diag[i] += w;
        diag[i + 1] += w;
        upper[i] = -w;
        lower[i + 1] = -w;
    }
    // Thomas.
    for i in 1..n {
        let factor = lower[i] / diag[i - 1];
        diag[i] -= factor * upper[i - 1];
        rhs[i] -= factor * rhs[i - 1];
    }
    let mut out = vec![0.0; n];
    out[n - 1] = rhs[n - 1] / diag[n - 1];
    for i in (0..n - 1).rev() {
        out[i] = (rhs[i] - upper[i] * out[i + 1]) / diag[i];
    }
    out
}

/// Puts the row at the level that leaves the note's **power** where it was, and
/// returns the level it took out, in dB.
///
/// # The leak this closes
///
/// `DECISIONS.md` 231 and every version of this field since have normalised the
/// row in the *log* domain — geometric mean 1, or the median partial unmoved —
/// and called that loudness-neutral. It is not, and the gap is Jensen's
/// inequality. A level is a sum of **powers**: a row of log-gains with mean zero
/// and spread `s` dB multiplies the note's power by about `s^2 ln 10 / 40` dB.
/// Measured on the shipped preset, whose rows have log means of +0.1 to +8.3 dB,
/// the *rendered* strike peak moves by up to **+18.9 dB** and the 0.2-2 s RMS by
/// **+25.5 dB** (key 96), because the row's spread is 27 dB and one railed cell
/// sits on the fundamental. That is the whole of the compass's family-2 level
/// spikes and of `DECISIONS.md` 266's two keys inside the master limiter.
///
/// So the row is pinned on the quantity that is actually conserved:
///
/// ```text
/// sum_k (e_k g_k)^2 = sum_k e_k^2
/// ```
///
/// over the partials the row is written for, with `e_k` the engine's own
/// rendered spectrum — which the fit already has, since it is the denominator of
/// every cell. Partials the row does not write are unchanged on both sides and
/// drop out. The scalar this removes is real per-key information — how much
/// louder the recording is than the engine at this key — and it is *reported*
/// as [`GainRow::level_db`] rather than written anywhere.
///
/// # Where the removed level does *not* go, measured
///
/// Two homes were tried for it and both are worse than reporting it.
///
/// * **`notes.bridge_gain`.** A per-key level is exactly what that table is, and
///   the shift would be uniform over the note rather than over the row's own
///   range. But the shift is not the piano's voicing, and "no smooth trend down
///   the compass" is now a decomposition rather than three keys' values
///   (`DECISIONS.md` 282). Over the 28 fitted keys the removed level has a
///   standard deviation of **4.82 dB**, and a smooth polynomial in key explains
///   **1.2 % of its variance at degree 1, 21 % at degree 3 and 26 % at
///   degree 4**; the degree-3 residual's lag-1 autocorrelation across keys three
///   semitones apart is **+0.08**. It is white. A `bridge_gain` curve is by
///   construction smooth, so there is no version of this table that can carry
///   four fifths of the quantity, and interpolating it onto the 58 unsampled
///   keys is interpolating along a correlation that is not there.
///
///   The same holds from the other side. The engine-versus-recording `level`
///   error over all 88 keys of `renders/compass/COMPASS.md` is 15.38 dB of
///   common offset plus **3.34 dB rms** of key-to-key error; a degree-1 trend
///   removes 1.2 % of that and degree 5 removes 26 %. Split on the sampler's own
///   three-key zones it is 1.61 dB within a zone — the pitch-shift ramp — and
///   2.26 dB between zones, and the between-zone series has a lag-1
///   autocorrelation of **−0.09**. So what the fit is chasing is the library's
///   take-to-take gain, and writing it would hide that inside a physical table
///   that item 44 calibrated on the *engine's* flattened compass.
///
///   Nor is it the recording's own per-key level: across the 28 keys the removed
///   level correlates with the *recording's* `level` residual against its own
///   eight nearest same-`N` neighbours at **r = +0.16** raw, **+0.22**
///   detrended. Carried in full it puts F#5 at `level` z **+6.7** — −32.58 dBFS
///   against a neighbourhood of −50.45 — where the recording's own F#5 sits at
///   +0.2, and F#6 at +4.5 against the recording's +0.1.
/// * **The whole bank.** Padding the row out to the key's every partial with the
///   removed level makes the row exactly loudness-neutral with no step anywhere.
///   Measured, it costs more than the step does: the padding writes the level
///   over partials nothing measured, and `compass_scan`'s `centroid` — a
///   power-weighted mean partial index — reads it as a colour change. D#4 went
///   `centroid` z **+7.4 → +9.9** and A1, E3 and F#1 gained flags they did not
///   have, against a `match` improvement of 1-2 dB at the three keys with the
///   largest shifts. So an unmeasured partial keeps its `1.0` and the step at
///   the end of the row is accepted, with this measurement as the reason.
fn energy_offset(row: &[Option<f64>], engine: &[Option<f64>]) -> (Vec<Option<f64>>, f64) {
    let weights: Vec<(usize, f64)> = row
        .iter()
        .enumerate()
        .filter_map(|(i, cell)| {
            cell.and(engine.get(i).copied().flatten())
                .map(|ln| (i, (2.0 * ln).exp()))
        })
        .collect();
    let before: f64 = weights.iter().map(|&(_, w)| w).sum();
    let after: f64 = weights
        .iter()
        .map(|&(i, w)| w * (2.0 * row[i].expect("some")).exp())
        .sum();
    if !(before > 0.0 && after > 0.0) {
        return (row.to_vec(), 0.0);
    }
    let offset = 0.5 * (after / before).ln();
    (
        row.iter().map(|cell| cell.map(|ln| ln - offset)).collect(),
        NEPERS_TO_DB * offset,
    )
}

/// The same row with its **roughness** scaled by `keep` and `level_db` added to
/// every cell.
///
/// The decomposition is the one the field's own argument rests on: a degree-2
/// polynomial in `ln k` is the *tilt* — `DECISIONS.md` 231's 7.5 dB at C4, the
/// engine's error in its own smooth envelope, and the half of the correction
/// that a smooth model could in principle absorb — and what is left over is the
/// per-partial scatter. Only the second is scaled. A loop that flattened the row
/// as a whole would give back the octave-displaced attack that item 231 exists
/// to repair, in exchange for the jaggedness this one exists to repair.
///
/// `keep = 1` is the row unchanged. `keep = 0` is the tilt alone, which is what
/// the field was before the roughness half was ever written.
pub fn flatten_row(row: &[f32], keep: f64, level_db: f64, config: &ShapingConfig) -> Vec<f32> {
    if row.is_empty() {
        return Vec::new();
    }
    let lift = level_db / NEPERS_TO_DB;
    let x: Vec<f64> = (1..=row.len()).map(|k| (k as f64).ln()).collect();
    let y: Vec<f64> = row.iter().map(|&g| f64::from(g).ln()).collect();
    let curve = (row.len() > config.envelope_degree + 1)
        .then(|| robust_polyfit(&x, &y, config))
        .flatten();
    let mut out: Vec<f32> = x
        .iter()
        .zip(&y)
        .map(|(&x, &y)| {
            let smooth = curve.as_ref().map_or(0.0, |c| poly_eval(c, x));
            (((smooth + keep * (y - smooth) + lift).exp()) as f32)
                .clamp(MIN_PARTIAL_GAIN, MAX_PARTIAL_GAIN)
        })
        .collect();
    while out.last() == Some(&1.0) {
        out.pop();
    }
    out
}

/// One partial's correction as the velocity layers measured it: where they
/// agree it is, and how far apart they are about that.
#[derive(Clone, Copy, Debug)]
struct Cell {
    /// Robust median of the layers' levelled log ratios, nepers.
    centre: f64,
    /// Variance of that median, nepers squared — how much of the cell is the
    /// measurement rather than the piano.
    variance: f64,
}

impl Cell {
    fn of(values: &[f64], config: &ShapingConfig) -> Option<Cell> {
        let centre = robust_median(values, config.outlier_db / NEPERS_TO_DB)?;
        let n = values.len().max(1) as f64;
        let spread =
            (values.iter().map(|v| (v - centre).powi(2)).sum::<f64>() / n).sqrt();
        // The standard error of a median is about 1.25 times a mean's.
        Some(Cell {
            centre,
            variance: (1.25 * spread).powi(2) / n,
        })
    }
}

/// The row as written: every partial's correction shrunk toward the smooth
/// curve by how much its own velocity layers agreed about it, and every partial
/// the layers could not reach given the smooth curve outright.
///
/// # Why the measured cells cannot be written as measured either
///
/// `DECISIONS.md` 231's argument for this field is that the per-partial pattern
/// is **the same in every velocity layer** — that is what makes it a property of
/// the string and the bridge rather than of the blow, and it is the entire
/// justification for a velocity-independent table carrying it. A cell whose
/// layers disagree by twenty decibels is not that pattern; it is three noisy
/// numbers, and their median is a noisy number. Writing it anyway puts the
/// tracker's own scatter into the instrument: measured on the fitted keys, the
/// engine's rendered spectrum came out **4 dB more jagged than the recording it
/// was fitted to**, while the 58 keys with no row at all came out 5 dB
/// *smoother* — a compass that lurches every third key, which is what a listener
/// finds as one note that does not fit.
///
/// So each cell is read as a measurement with an error bar and combined with the
/// row's own smooth curve the way two measurements of the same thing are:
/// `written = p(k) + w (m_k − p(k))`, with `w = t² / (t² + v_k)`, `v_k` the
/// variance of the cell's own median over its layers and `t²` the variance the
/// row's departures from the curve have in excess of it. A cell its layers agree
/// on keeps all of itself; a cell they disagree on keeps the part of itself the
/// row can vouch for. Nothing is thresholded and no cell is discarded.
///
/// The curve is the same Huber-weighted degree-2 polynomial in `ln k` that
/// [`envelope_tilt`] fits, now weighted by `1 / v_k` so that a cell nobody
/// agrees about does not bend it either.
fn write_row(measured: &[Option<Cell>], offset: f64, config: &ShapingConfig) -> Vec<Option<f64>> {
    let deviation = |cell: &Cell| cell.centre - offset;
    let points: Vec<(f64, f64, f64)> = measured
        .iter()
        .enumerate()
        .filter_map(|(i, cell)| {
            cell.map(|c| (((i + 1) as f64).ln(), deviation(&c), c.variance))
        })
        .collect();
    let raw = |i: usize| measured[i].map(|c| deviation(&c));
    // Too few points to fit a curve to: the cells stand as measured, which is
    // what the treble's three-partial rows get and is the only honest answer
    // there.
    if points.len() < config.envelope_degree + 2 {
        return (0..measured.len()).map(raw).collect();
    }
    let (Some(lo), Some(hi)) = (
        measured.iter().position(Option::is_some),
        measured.iter().rposition(Option::is_some),
    ) else {
        return (0..measured.len()).map(raw).collect();
    };
    let floor = points
        .iter()
        .map(|&(_, _, v)| v)
        .fold(f64::INFINITY, f64::min)
        .max(1e-6);
    let x: Vec<f64> = points.iter().map(|p| p.0).collect();
    let y: Vec<f64> = points.iter().map(|p| p.1).collect();
    let precision: Vec<f64> = points.iter().map(|&(_, _, v)| 1.0 / v.max(floor)).collect();
    let Some(curve) = huber_polyfit(&x, &y, &precision, config) else {
        return (0..measured.len()).map(raw).collect();
    };
    // The variance the row's departures from the curve have beyond what the
    // layers' own disagreement explains: the signal, as against the noise.
    let n = points.len() as f64;
    let excess = points
        .iter()
        .map(|&(xk, yk, v)| (yk - poly_eval(&curve, xk)).powi(2) - v)
        .sum::<f64>()
        / n;
    let signal = excess.max(0.0);
    (0..measured.len())
        .map(|i| {
            let at = poly_eval(&curve, ((i + 1) as f64).ln());
            match measured[i] {
                Some(cell) => {
                    let w = if signal + cell.variance > 0.0 {
                        signal / (signal + cell.variance)
                    } else {
                        1.0
                    };
                    Some(at + w * (deviation(&cell) - at))
                }
                // Interior, and *surrounded*: past the last measured partial
                // there is nothing to interpolate, a degree-2 polynomial in
                // `ln k` extrapolates by tens of decibels, and a row that ends
                // in genuine `1.0`s is a row that gets trimmed.
                None => (i > lo && i < hi && run_length(measured, i) <= MAX_FILL_RUN)
                    .then_some(at),
            }
        })
        .collect()
}

/// How many partials in a row may be interpolated at once.
///
/// A hole with a measurement on each side of it is an interpolation. A run of
/// three or more is a **gap**, and what a degree-2 polynomial in `ln k` says in
/// the middle of one is a guess — measured, and it costs: filling the long runs
/// as well moved the scoreboard's `scale_mf` and `staccato` up 0.14 and 0.13 dB
/// of log-mel, because the rows that have them are the short treble ones where
/// the curve has few points to stand on and the partials it reaches are the
/// 5 kHz bands those two phrases live in. Filling only the short runs keeps C2's
/// repair — its holes come in ones and twos — and gives that back.
const MAX_FILL_RUN: usize = 2;

/// How many consecutive unmeasured partials `i` belongs to.
fn run_length(measured: &[Option<Cell>], i: usize) -> usize {
    let mut lo = i;
    while lo > 0 && measured[lo - 1].is_none() {
        lo -= 1;
    }
    let mut hi = i;
    while hi + 1 < measured.len() && measured[hi + 1].is_none() {
        hi += 1;
    }
    hi - lo + 1
}

/// [`robust_polyfit`] with prior weights: the Huber iteration multiplies them
/// rather than replacing them, so a point can be down-weighted for being an
/// outlier, for being uncertain, or for both.
fn huber_polyfit(
    x: &[f64],
    y: &[f64],
    prior: &[f64],
    config: &ShapingConfig,
) -> Option<Vec<f64>> {
    let mut weights = prior.to_vec();
    let mut curve = weighted_polyfit(x, y, &weights, config.envelope_degree)?;
    for _ in 1..config.irls_iterations.max(1) {
        for (i, weight) in weights.iter_mut().enumerate() {
            let residual = (y[i] - poly_eval(&curve, x[i])).abs();
            *weight = prior[i]
                * if residual <= config.huber_delta {
                    1.0
                } else {
                    config.huber_delta / residual
                };
        }
        curve = weighted_polyfit(x, y, &weights, config.envelope_degree)?;
    }
    Some(curve)
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

    /// A known tilt, put in and taken out.
    ///
    /// The engine's spectrum is the recording's with a 6 dB per octave slope on
    /// it — the shape of the defect this exists for, and the size of it: at C4
    /// the fitted `envelope_tilt` reads **-6.53 dB** on the fundamental against
    /// **-0.05** on the fourth partial, a 7.5 dB span, which is a note whose
    /// strongest partial is `k = 1` where the recording's is `k = 2`.
    #[test]
    fn a_known_tilt_comes_back_out_of_the_two_envelopes() {
        let config = ShapingConfig::default();
        // A smooth, plausible spectrum: -9 dB per octave with a little curvature.
        let recording: Vec<(u32, f64)> = (1..=24u32)
            .map(|k| {
                let x = f64::from(k).ln();
                (k, (-1.5 * x - 0.1 * x * x).exp())
            })
            .collect();
        // The "engine" is that, 6 dB per octave steeper.
        let engine: Vec<(u32, f64)> = recording
            .iter()
            .map(|&(k, a)| (k, a * f64::from(k).powf(-1.0)))
            .collect();
        let tilt = envelope_tilt(&recording, &engine, 24, &config).expect("both fit");
        assert_eq!(tilt.len(), 24);
        // The correction is `k`, normalised so its geometric mean over the
        // measured partials is 1 — the same rule the roughness half obeys, and
        // the reason writing this table cannot move the note's loudness.
        let mean: f64 = tilt.iter().map(|&g| f64::from(g).ln()).sum::<f64>() / 24.0;
        assert!(mean.abs() < 1e-3, "the tilt moved the level by {mean} nepers");
        // ... so what is written is `k` divided by the geometric mean of `k`
        // over the measured partials, which for 1..=24 is 9.66.
        let geometric: f64 =
            ((1..=24u32).map(|k| f64::from(k).ln()).sum::<f64>() / 24.0).exp();
        for k in 1..=24usize {
            let got = f64::from(tilt[k - 1]) * geometric;
            assert!(
                (got / k as f64 - 1.0).abs() < 0.05,
                "partial {k}: the correction came back {got}, expected {k}"
            );
        }
        // Two spectra with the same envelope need no correction at all.
        let flat = envelope_tilt(&recording, &recording, 24, &config).expect("both fit");
        for (k, g) in flat.iter().enumerate() {
            assert!(
                (f64::from(*g) - 1.0).abs() < 1e-3,
                "partial {}: {g} where nothing differs",
                k + 1
            );
        }
        // A spectrum with nothing in it is a key that measured nothing, not a
        // key that needs no correction.
        assert!(envelope_tilt(&recording, &[], 24, &config).is_none());
    }

    // ---- the full measured ratio ----------------------------------------

    /// A spectrum with one level per partial, as the tracker hands one over.
    fn flat_spectrum(levels: &[f64]) -> Vec<(u32, f64)> {
        levels
            .iter()
            .enumerate()
            .map(|(i, &db)| (i as u32 + 1, 10f64.powf(db / 20.0)))
            .collect()
    }

    /// A known per-partial error between two spectra comes back as the gain that
    /// cancels it — up to the one scalar the row is not allowed to carry, which
    /// is the level. What is pinned is the *power* the row puts through the
    /// engine's own spectrum, so the shape comes back exactly and the common
    /// offset is whatever energy neutrality asks for.
    #[test]
    fn the_full_ratio_returns_the_error_between_the_two_spectra() {
        let engine: Vec<f64> = (0..12).map(|k| -2.0 * k as f64).collect();
        // The recording is the same envelope with a tilt and two rough partials
        // on it, plus 6 dB of level nobody should see in the answer.
        let error = [3.0, -4.0, 0.0, 0.0, 1.5, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0];
        let recorded: Vec<f64> = engine
            .iter()
            .zip(&error)
            .map(|(e, x)| e + x + 6.0)
            .collect();
        let config = ShapingConfig::default();
        let gains = measured_over_rendered(
            &[flat_spectrum(&recorded), flat_spectrum(&recorded), flat_spectrum(&recorded)],
            &[flat_spectrum(&engine), flat_spectrum(&engine), flat_spectrum(&engine)],
            12,
            &config,
        )
        .expect("a fit");
        let db = |g: f32| 20.0 * f64::from(g).log10();
        // The shape: every cell is the known error plus one common level.
        let common = db(gains[0]) - error[0];
        for (k, (&want, &got)) in error.iter().zip(gains.iter()).enumerate() {
            assert!(
                (db(got) - want - common).abs() < 0.05,
                "partial {}: {:.2} dB against {want} + {common:.2}",
                k + 1,
                db(got)
            );
        }
        // ... and that level is the one that leaves the note's *power* where it
        // was, which is the property `DECISIONS.md` 272 replaced the geometric
        // mean with. Measured the way a level meter would: the engine's own
        // partial powers, with and without the row on them.
        let engine_power: Vec<f64> = engine.iter().map(|db| 10f64.powf(db / 10.0)).collect();
        let before: f64 = engine_power.iter().sum();
        let after: f64 = engine_power
            .iter()
            .zip(gains.iter())
            .map(|(p, &g)| p * f64::from(g).powi(2))
            .sum();
        let moved = 10.0 * (after / before).log10();
        assert!(moved.abs() < 0.1, "the row moved the note's power by {moved:.2} dB");
    }

    /// Re-entrancy, which is the property the fit is built for: applying the
    /// answer to the engine and fitting again returns nothing more to do.
    #[test]
    fn applying_the_full_ratio_leaves_nothing_to_fit() {
        let engine: Vec<f64> = (0..10).map(|k| -1.5 * k as f64).collect();
        let error = [2.0, -3.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0];
        let recorded: Vec<f64> = engine.iter().zip(&error).map(|(e, x)| e + x).collect();
        let config = ShapingConfig::default();
        let three = |v: &[f64]| vec![flat_spectrum(v), flat_spectrum(v), flat_spectrum(v)];
        let first =
            measured_over_rendered(&three(&recorded), &three(&engine), 10, &config).expect("a fit");
        // The engine, with the answer in it.
        let corrected: Vec<f64> = engine
            .iter()
            .enumerate()
            .map(|(i, e)| e + 20.0 * f64::from(first.get(i).copied().unwrap_or(1.0)).log10())
            .collect();
        let second = measured_over_rendered(&three(&recorded), &three(&corrected), 10, &config);
        let residual = second.map_or(0.0, |g| {
            g.iter()
                .map(|v| (20.0 * f64::from(*v).log10()).abs())
                .fold(0.0f64, f64::max)
        });
        assert!(residual < 0.05, "a second pass still moves partials by {residual:.3} dB");
    }

    /// A partial at the tracker's floor on either side is not a measurement, so
    /// the ratio measured there does not reach the table.
    #[test]
    fn a_partial_in_the_floor_is_not_given_a_gain() {
        let engine: Vec<f64> = (0..10).map(|k| -1.0 * k as f64).collect();
        let mut recorded = engine.clone();
        // One real error, so the row is not empty — a table of nothing but ones
        // is correctly returned as `None`, which is a key with nothing to write.
        recorded[1] += 4.0;
        // ... and one partial 80 dB under this spectrum's loudest, which is past
        // `max_gain_range_db` and is the tracker's floor rather than a partial.
        recorded[7] = -80.0;
        let config = ShapingConfig::default();
        let three = |v: &[f64]| vec![flat_spectrum(v), flat_spectrum(v), flat_spectrum(v)];
        let gains = measured_over_rendered(&three(&recorded), &three(&engine), 10, &config)
            .expect("a fit");
        let db = |i: usize| 20.0 * f64::from(gains.get(i).copied().unwrap_or(1.0)).log10();
        // Against its neighbours, not against 0 dB: the row carries one common
        // level now (the energy pin), so "the floor reading did not reach the
        // table" is "this cell reads what the cells around it read".
        assert!(
            (db(7) - db(6)).abs() < 1.0 && (db(7) - db(8)).abs() < 1.0,
            "the floor reading is 79 dB down and reached the table as {:.2} dB \
             against {:.2} and {:.2}: {gains:?}",
            db(7),
            db(6),
            db(8)
        );
    }

    /// What that partial gets *instead*, and why `1.0` is not it.
    ///
    /// `DECISIONS.md` 267. The row here is a real correction of about −20 dB
    /// across the low partials with one partial unmeasurable in the middle of
    /// it. A `1.0` written there is not a missing correction, it is a 20 dB
    /// tooth in a comb — which is what C2 had fourteen of. The contract is that
    /// a hole lands between the measured partials on either side of it.
    #[test]
    fn an_unmeasured_partial_is_filled_from_the_curve_and_not_left_as_a_tooth() {
        let engine: Vec<f64> = (0..10).map(|k| -1.0 * k as f64).collect();
        let mut recorded = engine.clone();
        // A steep smooth tilt: the engine is 20 dB too loud at k=1 and right by
        // k=10, which no `1.0` in the middle of can be neutral in.
        for (k, r) in recorded.iter_mut().enumerate() {
            *r -= 20.0 * (1.0 - k as f64 / 9.0);
        }
        recorded[4] = -90.0; // past the floor: unmeasurable, not a measurement
        let config = ShapingConfig::default();
        let three = |v: &[f64]| vec![flat_spectrum(v), flat_spectrum(v), flat_spectrum(v)];
        let gains = measured_over_rendered(&three(&recorded), &three(&engine), 10, &config)
            .expect("a fit");
        let db = |i: usize| 20.0 * f64::from(gains.get(i).copied().unwrap_or(1.0)).log10();
        let (below, hole, above) = (db(3), db(4), db(5));
        assert!(
            below < hole && hole < above,
            "the hole at k=5 reads {hole:.2} dB, outside its neighbours {below:.2} and {above:.2}"
        );
        assert!(
            (hole - 0.5 * (below + above)).abs() < 1.5,
            "the hole at k=5 reads {hole:.2} dB, not between {below:.2} and {above:.2}"
        );
    }

    /// The leak `DECISIONS.md` 272 closed, in the one line that states it: a row
    /// whose *log* mean is zero is not loudness-neutral, and the row this fit
    /// writes is neutral on the quantity a level is made of.
    #[test]
    fn a_log_neutral_row_is_not_level_neutral_and_the_written_one_is() {
        // A flat engine spectrum and a recording that is the same spectrum with
        // ±12 dB of alternating scatter on it: a pattern whose geometric mean is
        // exactly 1 and whose power is 6.8 dB up.
        let engine: Vec<f64> = vec![0.0; 12];
        let scatter: Vec<f64> = (0..12)
            .map(|k| if k % 2 == 0 { 12.0 } else { -12.0 })
            .collect();
        let recorded: Vec<f64> = engine.iter().zip(&scatter).map(|(e, s)| e + s).collect();
        // What the old normalisation would have written, and what it costs.
        let log_neutral = 10.0
            * (scatter.iter().map(|db| 10f64.powf(db / 10.0)).sum::<f64>() / 12.0).log10();
        assert!(
            log_neutral > 6.0,
            "a zero-mean ±12 dB row lifts the power by {log_neutral:.2} dB"
        );

        let config = ShapingConfig::default();
        let three = |v: &[f64]| vec![flat_spectrum(v), flat_spectrum(v), flat_spectrum(v)];
        let row = measured_over_rendered_report(&three(&recorded), &three(&engine), 12, &config)
            .expect("a fit");
        // The level it took out is that lift, and it is reported rather than
        // written.
        assert!(
            (row.level_db - log_neutral).abs() < 1.5,
            "the fit reports {:.2} dB of level against the {log_neutral:.2} it took out",
            row.level_db
        );
        let power: f64 = row
            .gains
            .iter()
            .map(|&g| f64::from(g).powi(2))
            .sum::<f64>()
            / row.gains.len() as f64;
        let moved = 10.0 * power.log10();
        assert!(
            moved.abs() < 0.5,
            "the written row moves a flat note's power by {moved:.2} dB"
        );
    }

    /// A cell that is nobody's measurement is clipped to the key's own spread,
    /// not to the schema's ±26 dB.
    #[test]
    fn a_cell_past_the_keys_own_spread_is_railed_to_it() {
        let engine: Vec<f64> = vec![0.0; 12];
        // Eleven cells inside ±2 dB and one at +40: the key's own MAD says the
        // twelfth is not this key's measurement.
        let mut recorded = vec![0.0, 1.5, -1.0, 2.0, -1.5, 1.0, -2.0, 0.5, 1.8, -0.8, 1.2];
        recorded.push(40.0);
        let config = ShapingConfig::default();
        let three = |v: &[f64]| vec![flat_spectrum(v), flat_spectrum(v), flat_spectrum(v)];
        let row = measured_over_rendered_report(&three(&recorded), &three(&engine), 12, &config)
            .expect("a fit");
        assert!(
            row.rail_db < 10.0 && row.railed >= 1,
            "rail {:.2} dB, {} railed",
            row.rail_db,
            row.railed
        );
        let db = |g: f32| 20.0 * f64::from(g).log10();
        assert!(
            db(row.gains[11]) < 12.0,
            "the +40 dB cell reached the table as {:.2} dB",
            db(row.gains[11])
        );
        // ... and the schema's own rails are still the outer bound.
        for &g in &row.gains {
            assert!((MIN_PARTIAL_GAIN..=MAX_PARTIAL_GAIN).contains(&g), "{g}");
        }
    }

    /// The row is smoothed until the series it predicts is no rougher than the
    /// recording's — and not one decibel further, because the recording's own
    /// roughness is the measurement the field exists to carry.
    #[test]
    fn the_row_is_smoothed_to_the_recordings_roughness_and_no_further() {
        let config = ShapingConfig::default();
        let three = |v: &[f64]| vec![flat_spectrum(v), flat_spectrum(v), flat_spectrum(v)];
        let engine: Vec<f64> = (0..12).map(|k| -1.0 * k as f64).collect();

        // A recording that really is rough: the row has to keep all of it, and
        // the loop is not allowed to smooth a thing.
        let rough: Vec<f64> = engine
            .iter()
            .enumerate()
            .map(|(k, e)| e + if k % 2 == 0 { 5.0 } else { -5.0 })
            .collect();
        let kept = measured_over_rendered_report(&three(&rough), &three(&engine), 12, &config)
            .expect("a fit");
        assert_eq!(kept.lambda, 0.0, "a matched roughness was smoothed anyway");
        assert!(
            (kept.written_step_db - kept.target_step_db).abs() < 1.0,
            "target {:.2} against written {:.2}",
            kept.target_step_db,
            kept.written_step_db
        );
    }

    /// A cell whose velocity layers disagree is not the velocity-independent
    /// pattern the field is defined as, and is written closer to the row's own
    /// smooth curve than one they agree on.
    #[test]
    fn a_cell_the_layers_disagree_about_is_shrunk_toward_the_curve() {
        let engine: Vec<f64> = (0..10).map(|k| -1.0 * k as f64).collect();
        // Two identical departures from a flat row: k=3 every layer agrees on,
        // k=7 they scatter about by ±12 dB. Same median, different evidence.
        let layers: Vec<Vec<(u32, f64)>> = [-12.0, 0.0, 12.0]
            .iter()
            .map(|&scatter| {
                let mut r = engine.clone();
                r[2] += 8.0;
                r[6] += 8.0 + scatter;
                flat_spectrum(&r)
            })
            .collect();
        let config = ShapingConfig::default();
        let rendered: Vec<Vec<(u32, f64)>> =
            (0..3).map(|_| flat_spectrum(&engine)).collect();
        let gains =
            measured_over_rendered(&layers, &rendered, 10, &config).expect("a fit");
        let db = |i: usize| 20.0 * f64::from(gains.get(i).copied().unwrap_or(1.0)).log10();
        assert!(
            db(2) > db(6) + 3.0,
            "the agreed cell reads {:.2} dB and the disputed one {:.2}",
            db(2),
            db(6)
        );
    }
}
