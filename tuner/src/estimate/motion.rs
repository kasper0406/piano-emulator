//! The two motion mechanisms, inverted from the recordings: `notes.false_beat`
//! and `[voicing.strike_direction]`.
//!
//! `FUNDAMENTALS.md` §7.4 is the diagnosis this module promotes into an
//! estimator. Having removed the unison beat completely — the coupled
//! construction of `DECISIONS.md` 223 locks the bass and midrange, and measures
//! 0.01–0.17 dB of beat depth at C4's fundamental where the recording measures
//! 9.5 — the prototype could ask what is left in the recording, and the answer
//! was a **second component 4–7 dB down, 0.7–1.5 Hz away, at a spacing that does
//! not scale with the partial number**:
//!
//! | implied companion, dB / Hz | k=1 | k=2 | k=3 | k=4 |
//! |:--|--:|--:|--:|--:|
//! | C4 recording | −6.1 / 1.11 | −3.6 / 1.48 | −6.6 / 0.74 | −16.9 / 0.74 |
//! | A2 recording | −26.6 / 1.11 | −21.7 / 1.48 | −5.5 / 0.74 | −5.6 / 0.74 |
//! | C6 recording | −4.8 / 2.22 | −6.1 / 4.07 | −2.6 / 4.44 | −8.1 / 5.19 |
//!
//! Two mechanisms are fitted from those two columns, and one falsification runs
//! between them.
//!
//! # 1. The false beat: per key, per partial, `{k, hz, db}`
//!
//! [`fit_false_beat`] inverts each partial's measured beat depth for the
//! amplitude ratio of a two-component pair ([`crate::motion::Motion::companion_db`],
//! `D = 20 log10((1+r)/(1−r))`) and takes the rate from the envelope's own sign
//! changes ([`crate::motion::Motion::beat_rate_hz`]). That is the whole
//! inversion: the schema is written in exactly these units because they are the
//! units the measurement comes back in.
//!
//! **The falsifiability check is the point of the module.** `DECISIONS.md` 233
//! states it as a condition on the fit rather than as a property of the
//! mechanism: *fitted per string and per partial, `delta` must come back
//! uncorrelated across `k` or it is not a false beat*. A beat from a unison
//! **mistuning** is a frequency ratio, so partial `k` beats at `k` times the
//! fundamental's rate and the correlation between `k` and the rate is +1; a beat
//! from the wire's own geometry has no reason to know what `k` is. So
//! [`fit_false_beat`] measures that correlation per key and **writes nothing at
//! all** for a key whose rates track `k` ([`FalseBeatVerdict::ScalesWithPartial`]).
//! On the Salamander library that rejects the treble, where §7.4 already
//! observed the rates rising 2.22 → 5.19 with `k` and where the fitted unison
//! detune is genuinely wide enough to beat on its own.
//!
//! What the measured depth contains, stated once: it is the beat of everything
//! in the partial, unison included. In the register this fit writes into, that
//! is not an approximation worth correcting — the coupled unison's own
//! contribution there is a hundredth of a decibel — and in the register where it
//! would be, the correlation test has already refused the key.
//!
//! # 2. The strike direction: one global velocity law, from 16 layers
//!
//! [`fit_strike_direction`] uses the *same* inversion at every velocity layer
//! the library has. `FUNDAMENTALS.md` §7.3's second refutation is that nothing
//! in a linear string model can make the beat structure depend on velocity
//! except the **direction** of the strike vector, and `DECISIONS.md` 235 built
//! the handle: `leak_db(v) = horizontal_gain_db + lerp(vh_db_at_pp,
//! vh_db_at_ff, t)` with `t = (vel − 1)/126`. The companion's level moves
//! decibel for decibel with that offset, so the fit is a regression of the
//! measured companion level on `t`:
//!
//! ```text
//!     db(t) = alpha + beta t ,   beta = vh_db_at_ff - vh_db_at_pp
//! ```
//!
//! and the one remaining degree of freedom is fixed by the requirement that the
//! law be **neutral at the reference velocity** — `vh_db_at_pp + beta t_ref = 0`
//! — so that writing it does not move the note every other table was fitted on.
//! That is the same discipline `unison_sigma_scale`'s mean-of-1 constraint uses
//! (`DECISIONS.md` 105) and the same one [`crate::estimate::shaping`] applies to
//! the gains.
//!
//! The regression is over every (key, partial) cell that survived the false-beat
//! gates, because those are the cells that have a companion to measure: pooling
//! is what makes one global field identifiable from data whose per-cell scatter
//! is several decibels.

use crate::motion::Motion;
use crate::numeric::median;
use crate::preset::{
    FalseBeat, StrikeDirection, MAX_FALSE_BEATS_PER_KEY, MAX_FALSE_BEAT_DB, MAX_FALSE_BEAT_HZ,
    MAX_STRIKE_DIRECTION_DB, MIN_FALSE_BEAT_DB, MIN_FALSE_BEAT_HZ,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotionConfig {
    /// Highest partial a row is written for. The schema allows eight entries per
    /// key and the ear resolves about that many.
    pub max_partial: u32,
    /// How far a partial must stand over its own neighbourhood before its
    /// envelope is a measurement of the partial rather than of the background,
    /// in dB. [`crate::motion::MIN_PEAK_DB`] is the floor; this is the gate the
    /// fit adds on top of it.
    pub min_peak_db: f64,
    /// Fewest partials of a key that must have measured a companion before the
    /// correlation test means anything — and therefore before any row is
    /// written. Two points always lie on a line.
    pub min_partials: usize,
    /// How much better the flat-in-`k` model must fit the measured rates than
    /// the proportional-to-`k` one before the key is written, as a ratio of
    /// residual sums of squares. 1.0 is "no worse"; above it is a margin.
    pub flat_model_margin: f64,
    /// Huber threshold on the strike-direction regression's residual, in dB.
    pub huber_db: f64,
    pub irls_iterations: usize,
    /// Fewest cells the velocity regression needs before it returns anything.
    pub min_velocity_cells: usize,
    /// Smallest pianissimo-to-fortissimo swing that is written at all, in dB.
    /// Under it the regression has found no velocity dependence, and the field
    /// written is exactly zero — which `DECISIONS.md` 238 requires to be the
    /// absent section, bit for bit.
    pub min_swing_db: f64,
}

impl Default for MotionConfig {
    fn default() -> Self {
        Self {
            max_partial: MAX_FALSE_BEATS_PER_KEY as u32,
            min_peak_db: 15.0,
            min_partials: 3,
            flat_model_margin: 1.0,
            huber_db: 4.0,
            irls_iterations: 4,
            min_velocity_cells: 12,
            min_swing_db: 0.5,
        }
    }
}

/// What one partial's envelope implied, before any gate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Companion {
    pub k: u32,
    /// Where the second component sits, in hertz from the first.
    pub hz: f64,
    /// How loud it is, in dB under the first.
    pub db: f64,
    /// The beat depth it was inverted from, dB.
    pub depth_db: f64,
    /// The partial's own signal-to-neighbourhood, dB.
    pub peak_db: f64,
}

impl Companion {
    /// Whether this row is inside the schema's own bounds — a rate a false beat
    /// can have and a level the engine can be asked for.
    pub fn in_range(&self) -> bool {
        (f64::from(MIN_FALSE_BEAT_HZ)..=f64::from(MAX_FALSE_BEAT_HZ)).contains(&self.hz)
            && (f64::from(MIN_FALSE_BEAT_DB)..=f64::from(MAX_FALSE_BEAT_DB)).contains(&self.db)
    }
}

/// Why a key was or was not written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FalseBeatVerdict {
    /// Rows written.
    Written,
    /// Too few partials measured a companion at all: nothing to test and
    /// nothing to write.
    TooFewPartials,
    /// The rates track the partial number, so the beat is the unison's and the
    /// mechanism does not apply here.
    ScalesWithPartial,
    /// Every measured companion fell outside the schema's range — too quiet to
    /// name or beating outside 0.2–3 Hz.
    NoneInRange,
}

/// One key's fit.
#[derive(Clone, Debug)]
pub struct FalseBeatFit {
    pub key: u8,
    /// Everything measured, gates and all, so a coverage report can say what was
    /// seen as well as what was written.
    pub measured: Vec<Companion>,
    /// What goes in the preset. Empty is a legitimate answer.
    pub rows: Vec<FalseBeat>,
    /// Pearson correlation between the partial number and the fitted rate —
    /// reported, not gated on. See [`FalseBeatFit::model_ratio`].
    pub rate_correlation: f64,
    /// `RSS(rate = s k) / RSS(rate = c)`: how much better a **flat** rate
    /// describes this key's partials than a rate **proportional to `k`**. Above
    /// one the wire wins and the key is written; at or below it the tuning wins
    /// and it is not.
    ///
    /// This is `DECISIONS.md` 233's falsification stated as a model comparison
    /// rather than as a correlation threshold, and it has to be: the two models
    /// have one parameter each, so their residuals are directly comparable,
    /// while a correlation of 0.6 over five partials is not evidence of
    /// anything either way — it decided A4 and C6 opposite ways on two readings
    /// of the same recordings that differed only in how the rate was estimated.
    pub model_ratio: f64,
    pub verdict: FalseBeatVerdict,
}

/// Inverts one key's measured partials into `notes.false_beat` rows.
///
/// `motions` is the key's partials at the reference velocity, each with the
/// partial number it was measured at. Partials that measured nothing are simply
/// absent.
pub fn fit_false_beat(key: u8, motions: &[(u32, Motion)], config: &MotionConfig) -> FalseBeatFit {
    let mut measured: Vec<Companion> = motions
        .iter()
        .filter(|(k, m)| *k >= 1 && *k <= config.max_partial && m.peak_db >= config.min_peak_db)
        .filter_map(|(k, m)| {
            m.companion_db().map(|db| Companion {
                k: *k,
                hz: m.beat_rate_hz,
                db,
                depth_db: m.beat_depth_db,
                peak_db: m.peak_db,
            })
        })
        .collect();
    measured.sort_by_key(|c| c.k);

    // The falsification runs on the rows that would be *written*, not on every
    // partial that returned a number: a companion 26 dB down has a rate the
    // measurement barely resolves, and the claim being tested is the claim the
    // preset makes.
    let candidates: Vec<Companion> = measured.iter().copied().filter(Companion::in_range).collect();
    let mut fit = FalseBeatFit {
        key,
        rate_correlation: correlation(
            &candidates.iter().map(|c| f64::from(c.k)).collect::<Vec<_>>(),
            &candidates.iter().map(|c| c.hz).collect::<Vec<_>>(),
        ),
        model_ratio: model_ratio(&candidates),
        measured,
        rows: Vec::new(),
        verdict: FalseBeatVerdict::NoneInRange,
    };
    if candidates.is_empty() {
        return fit;
    }
    if candidates.len() < config.min_partials {
        fit.verdict = FalseBeatVerdict::TooFewPartials;
        return fit;
    }
    // A rate that tracks `k` is a frequency ratio, which is the unison, which
    // this mechanism is not.
    if fit.model_ratio < config.flat_model_margin {
        fit.verdict = FalseBeatVerdict::ScalesWithPartial;
        return fit;
    }
    fit.rows = candidates
        .iter()
        .take(MAX_FALSE_BEATS_PER_KEY)
        .map(|c| FalseBeat {
            k: c.k as u16,
            hz: c.hz as f32,
            db: c.db as f32,
        })
        .collect();
    fit.verdict = FalseBeatVerdict::Written;
    fit
}

/// One cell of the velocity regression: a companion level measured at one
/// velocity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VelocityCell {
    /// Which key × partial this reading belongs to. The regression is a
    /// **within** estimator: every cell has its own companion level, and only
    /// how that level *moves* with velocity is evidence about a global law, so
    /// each group's own mean is removed before the slope is taken. Pooling the
    /// raw levels instead measures the scatter between cells, which is 5.5 dB
    /// on this library and swamps what is being fitted.
    pub group: u32,
    /// The MIDI velocity the layer is the recording of.
    pub velocity: u8,
    /// The companion level measured there, dB.
    pub db: f64,
}

impl VelocityCell {
    /// The engine's own abscissa: `t = (max(vel, 1) − 1) / 126`.
    pub fn t(&self) -> f64 {
        f64::from(self.velocity.max(1) - 1) / 126.0
    }
}

/// The velocity law, and what it was fitted from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StrikeDirectionFit {
    /// `vh_db_at_ff − vh_db_at_pp`: the whole pianissimo-to-fortissimo swing of
    /// the horizontal leak, in dB.
    pub swing_db: f64,
    /// Where the law is pinned to zero, so that the reference velocity's own
    /// fits are unmoved.
    pub reference_t: f64,
    /// Correlation of the within-group regression, which is what says whether
    /// the recording has a *common* velocity dependence at all.
    pub correlation: f64,
    /// Median of the per-cell slopes, dB per unit `t`. The regression above is
    /// the pooled version of this; quoting both is what says whether one law
    /// describes the cells or only their average.
    pub median_cell_slope: f64,
    /// Interquartile range of those per-cell slopes.
    pub cell_slope_iqr: f64,
    /// How many cells had enough velocities to have a slope of their own.
    pub groups: usize,
    /// RMS of the regression's residual, dB.
    pub residual_db: f64,
    pub cells: usize,
    /// The field, ready to write.
    pub direction: StrikeDirection,
}

/// Fits `[voicing.strike_direction]`'s two decibel fields from how the measured
/// companion level moves with velocity.
///
/// Returns `None` when there are too few cells for a regression to mean
/// anything, which is a library that did not sample enough velocity layers
/// rather than an instrument with no velocity dependence.
pub fn fit_strike_direction(
    cells: &[VelocityCell],
    reference_velocity: u8,
    config: &MotionConfig,
) -> Option<StrikeDirectionFit> {
    if cells.len() < config.min_velocity_cells {
        return None;
    }
    // Within-group centring: each cell's own level is a nuisance parameter and
    // is removed with it.
    let mut groups: std::collections::BTreeMap<u32, Vec<(f64, f64)>> =
        std::collections::BTreeMap::new();
    for cell in cells {
        groups.entry(cell.group).or_default().push((cell.t(), cell.db));
    }
    let mut t: Vec<f64> = Vec::with_capacity(cells.len());
    let mut db: Vec<f64> = Vec::with_capacity(cells.len());
    let mut cell_slopes: Vec<f64> = Vec::new();
    for points in groups.values() {
        if points.len() < 2 {
            continue;
        }
        let n = points.len() as f64;
        let mt = points.iter().map(|p| p.0).sum::<f64>() / n;
        let md = points.iter().map(|p| p.1).sum::<f64>() / n;
        for &(x, y) in points {
            t.push(x - mt);
            db.push(y - md);
        }
        let xs: Vec<f64> = points.iter().map(|p| p.0).collect();
        let ys: Vec<f64> = points.iter().map(|p| p.1).collect();
        if let Some((_, slope)) = huber_line(&xs, &ys, config) {
            cell_slopes.push(slope);
        }
    }
    if t.len() < config.min_velocity_cells {
        return None;
    }
    let group_count = groups.len();
    cell_slopes.sort_by(f64::total_cmp);
    let quantile = |q: f64| {
        if cell_slopes.is_empty() {
            0.0
        } else {
            cell_slopes[((cell_slopes.len() - 1) as f64 * q).round() as usize]
        }
    };
    let (median_cell_slope, cell_slope_iqr) = (quantile(0.5), quantile(0.75) - quantile(0.25));
    let (intercept, slope) = huber_line(&t, &db, config)?;
    let reference_t = f64::from(reference_velocity.max(1) - 1) / 126.0;
    // The law is the *shape* of the regression, moved so that it adds nothing at
    // the velocity every other table was fitted at.
    let swing = if slope.abs() < config.min_swing_db {
        0.0
    } else {
        slope.clamp(
            -2.0 * f64::from(MAX_STRIKE_DIRECTION_DB),
            2.0 * f64::from(MAX_STRIKE_DIRECTION_DB),
        )
    };
    let pp = (-swing * reference_t).clamp(
        -f64::from(MAX_STRIKE_DIRECTION_DB),
        f64::from(MAX_STRIKE_DIRECTION_DB),
    );
    let ff = (swing * (1.0 - reference_t)).clamp(
        -f64::from(MAX_STRIKE_DIRECTION_DB),
        f64::from(MAX_STRIKE_DIRECTION_DB),
    );
    let residual_db = (t
        .iter()
        .zip(&db)
        .map(|(t, d)| {
            let e = d - (intercept + slope * t);
            e * e
        })
        .sum::<f64>()
        / t.len() as f64)
        .sqrt();
    Some(StrikeDirectionFit {
        swing_db: swing,
        reference_t,
        correlation: correlation(&t, &db),
        median_cell_slope,
        cell_slope_iqr,
        groups: group_count,
        residual_db,
        cells: cells.len(),
        direction: StrikeDirection {
            vh_db_at_pp: pp as f32,
            vh_db_at_ff: ff as f32,
            // Nothing in the recordings identifies the per-string share tilt on
            // its own — it moves the same beat depth the leak does, and one
            // regression cannot carry two parameters that enter it the same
            // way. Left neutral, deliberately, rather than split arbitrarily.
            share_tilt: 0.0,
        },
    })
}

/// How far the engine's own rendered velocity spread moves with the swing —
/// `CombLine`'s and `DamperLine`'s pattern (`DECISIONS.md` 214, 183), and the
/// half of this fit that cannot come off the recording.
///
/// # Why the regression alone is not the fit
///
/// [`fit_strike_direction`] measures what the recording's companion level does
/// *in common* across cells as velocity rises, and on the Salamander library
/// that is **+0.55 dB** over the whole velocity range against a per-cell slope
/// IQR of **4.95 dB**: the velocity dependence is overwhelmingly per key and per
/// partial, and `[voicing.strike_direction]` is one global law. Writing +0.55 dB
/// because that is the common part would be reading the wrong quantity — the
/// column the field exists to move (`B2`, velocity coherence) is a *spread*, not
/// a trend, and a spread does not care that half the cells move the other way.
///
/// So the recording sets two things and the engine sets the third: the recording
/// gives the **sign** (the common trend's) and the **size of the target** (the
/// mean per-cell spread of the beat depth across the three velocities Column B
/// is defined at), and the engine is rendered at a handful of swings to find
/// which one produces that spread. That is the same inversion-on-the-engine
/// discipline as `estimate::directivity` and `CombLine`: a quantity only
/// meaningful as "how far is the engine from the recording" is inverted on the
/// engine, so the measurement's own bias divides out.
#[derive(Clone, Debug, Default)]
pub struct SwingLine {
    /// `(swing dB, mean per-cell spread of the beat depth across velocities)`,
    /// ascending in the swing. The first probe should be zero, which is the
    /// velocity-independent construction and reads its own floor.
    pub probes: Vec<(f64, f64)>,
}

impl SwingLine {
    /// The swing whose rendered spread is `target`, by linear interpolation
    /// between the two probes that bracket it.
    ///
    /// Returns the largest probe when the target is above everything measured —
    /// which is the honest answer, "as far as the field goes and still short" —
    /// and `None` when the line does not rise at all, which is a probe set that
    /// measured nothing.
    pub fn swing_for(&self, target: f64) -> Option<f64> {
        let probes = &self.probes;
        if probes.len() < 2 {
            return None;
        }
        let (first, last) = (probes[0], probes[probes.len() - 1]);
        if last.1 <= first.1 || !last.1.is_finite() {
            return None;
        }
        if target <= first.1 {
            return Some(first.0);
        }
        if target >= last.1 {
            return Some(last.0);
        }
        for pair in probes.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            if target >= a.1 && target <= b.1 && b.1 > a.1 {
                return Some(a.0 + (b.0 - a.0) * (target - a.1) / (b.1 - a.1));
            }
        }
        None
    }
}

/// The field a swing and a sign make, pinned so that the reference velocity is
/// unmoved — the same normalisation [`fit_strike_direction`] applies.
pub fn strike_direction_for(swing_db: f64, reference_velocity: u8) -> StrikeDirection {
    let reference_t = f64::from(reference_velocity.max(1) - 1) / 126.0;
    let limit = f64::from(MAX_STRIKE_DIRECTION_DB);
    StrikeDirection {
        vh_db_at_pp: (-swing_db * reference_t).clamp(-limit, limit) as f32,
        vh_db_at_ff: (swing_db * (1.0 - reference_t)).clamp(-limit, limit) as f32,
        share_tilt: 0.0,
    }
}

/// The spread of a per-cell statistic across velocities, which is Column B2's
/// own quantity: `max − min` over the velocities of each cell, averaged.
pub fn velocity_spread(cells: &[Vec<f64>]) -> f64 {
    let spreads: Vec<f64> = cells
        .iter()
        .filter(|values| values.len() >= 2)
        .map(|values| {
            let lo = values.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            hi - lo
        })
        .collect();
    if spreads.is_empty() {
        return 0.0;
    }
    spreads.iter().sum::<f64>() / spreads.len() as f64
}

/// `RSS(rate = s k) / RSS(rate = c)` over the measured partials — the two
/// one-parameter models `FUNDAMENTALS.md` §7.4 sets against each other, fitted
/// and compared. Returns 0 for fewer than two points, which no caller writes on.
fn model_ratio(measured: &[Companion]) -> f64 {
    if measured.len() < 2 {
        return 0.0;
    }
    let n = measured.len() as f64;
    let mean = measured.iter().map(|c| c.hz).sum::<f64>() / n;
    let flat: f64 = measured.iter().map(|c| (c.hz - mean).powi(2)).sum();
    let (mut num, mut den) = (0.0, 0.0);
    for c in measured {
        num += f64::from(c.k) * c.hz;
        den += f64::from(c.k) * f64::from(c.k);
    }
    let slope = if den > 0.0 { num / den } else { 0.0 };
    let proportional: f64 = measured
        .iter()
        .map(|c| (c.hz - slope * f64::from(c.k)).powi(2))
        .sum();
    if flat <= 0.0 {
        return f64::INFINITY;
    }
    proportional / flat
}

/// Pearson correlation, or 0 when either side does not vary.
fn correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }
    let mean = |v: &[f64]| v[..n].iter().sum::<f64>() / n as f64;
    let (mx, my) = (mean(x), mean(y));
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    for i in 0..n {
        let (dx, dy) = (x[i] - mx, y[i] - my);
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    if sxx <= 0.0 || syy <= 0.0 {
        return 0.0;
    }
    sxy / (sxx * syy).sqrt()
}

/// A straight line through `(x, y)` by iteratively reweighted least squares with
/// a Huber weight, which is what keeps one layer whose partial went through a
/// null from setting the slope.
fn huber_line(x: &[f64], y: &[f64], config: &MotionConfig) -> Option<(f64, f64)> {
    let n = x.len();
    if n < 2 {
        return None;
    }
    let mut weight = vec![1.0f64; n];
    let mut line = (0.0, 0.0);
    for _ in 0..config.irls_iterations.max(1) {
        let (mut sw, mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for i in 0..n {
            let w = weight[i];
            sw += w;
            sx += w * x[i];
            sy += w * y[i];
            sxx += w * x[i] * x[i];
            sxy += w * x[i] * y[i];
        }
        let denom = sw * sxx - sx * sx;
        if denom.abs() <= 1e-12 || !denom.is_finite() {
            return None;
        }
        let slope = (sw * sxy - sx * sy) / denom;
        let intercept = (sy - slope * sx) / sw;
        line = (intercept, slope);
        let residuals: Vec<f64> = (0..n).map(|i| (y[i] - intercept - slope * x[i]).abs()).collect();
        let scale = median(&residuals)
            .filter(|m| *m > 1e-9)
            .unwrap_or(config.huber_db)
            .max(config.huber_db * 0.25);
        for i in 0..n {
            let r = residuals[i] / scale;
            weight[i] = if r <= 1.0 { 1.0 } else { 1.0 / r };
        }
    }
    Some(line)
}

// ---------------------------------------------------------------------------
// The closed loop: what the render actually does with the row
// ---------------------------------------------------------------------------

/// One partial's target and the state of the search for it.
#[derive(Clone, Copy, Debug, PartialEq)]
struct LoopState {
    k: u32,
    /// The recording's beat depth, dB — what the level is solved against.
    target_depth_db: f64,
    /// The recording's in-band frequency deviation, cents — what the rate is
    /// solved against.
    target_cents: f64,
    /// Bisection bracket on the level, dB, and the point being probed.
    lo_db: f64,
    hi_db: f64,
    db: f64,
    hz: f64,
    /// What the render read with this key's row cleared: the depth the engine's
    /// own unison already produces without any wire defect.
    baseline_depth_db: f64,
    baseline_cents: f64,
    /// The best point the search has visited, and its cost. `None` is the
    /// baseline — no row at all — which is a candidate like any other and wins
    /// whenever the engine's own unison is already closer to the piano than
    /// anything the mechanism can add.
    best: Option<(f64, f64)>,
    best_cost: f64,
    /// Whether the level bisection has anything left to do: cleared when the
    /// baseline is already past the recording's depth, since a companion can
    /// only add.
    dropped: bool,
}

/// Solving `notes.false_beat` on the engine instead of on the recording.
///
/// # Why the recording's own inversion is not the answer
///
/// [`fit_false_beat`] reads the recording's beat depth, inverts
/// `D = 20 log10((1+r)/(1−r))` into a companion level and writes it. That is the
/// right reading of the *recording*, and `DECISIONS.md` 234 already recorded
/// that it is not the level the *render* comes back with: 13 dB of asked level
/// moved the rendered depth by under 3 dB, because the asked level is quoted
/// against one block's coherent sum while the depth is measured on the whole
/// partial — `2N` eigenmodes, of which the vertical unison is already beating on
/// its own. Measured against the shipped rows, the two disagree by up to
/// **16 dB** (A4 k=3: asked −14.8 dB for a 3.2 dB depth, rendered 19.6 dB), and
/// `B1` is a mean of exactly that disagreement.
///
/// So the level is not asserted, it is **solved on the render**: bisect the
/// asked level until the rendered depth is the recording's. That is the same
/// inversion-on-the-engine discipline as [`SwingLine`],
/// `estimate::directivity` and `estimate::shaping::measured_over_rendered` —
/// a quantity that only means anything as "how far is the engine from the
/// recording" is inverted where the engine is.
///
/// # Two targets, two knobs
///
/// A companion at ratio `r` a distance `δ` away sets the beat depth through `r`
/// alone and the in-band frequency deviation through **both**: to first order
/// `J ≈ (1200/ln 2) · δ r / (√2 f)`. Depth and deviation are therefore
/// independent coordinates of `(r, δ)`, and the loop solves both — bisection on
/// the level for the depth, a damped multiplicative step on the rate for the
/// deviation.
///
/// This is a change of objective, and it is falsifiable and falsified: at A2
/// k=1 the recording's 0.81 dB of depth and 1.44 cents of deviation imply
/// `δ = 2.8 Hz`, while its *envelope* beats at 0.59 Hz. The two disagree by 4.8x,
/// which is a measurement saying the recording's fundamental is **not** two
/// components — the deviation is larger than any two-component pair of that
/// depth can make at that rate. The rate written is the one that reproduces the
/// motion the ear is shown to detect (`FUNDAMENTALS.md` §II.1: coherent slow FM
/// at 3–5 cents of threshold), not the one that reproduces the envelope line,
/// and `DECISIONS.md` 250 records the choice with the number that forced it.
///
/// # Cost
///
/// One render per round per key, and every partial of the key is measured from
/// it, so the loop costs [`FalseBeatLoop::ROUNDS`] renders per key however many
/// rows it writes.
#[derive(Clone, Debug)]
pub struct FalseBeatLoop {
    states: Vec<LoopState>,
    round: usize,
}

impl FalseBeatLoop {
    /// Rounds of render-and-observe after the baseline.
    ///
    /// The two knobs are only *approximately* independent — a rate fast enough
    /// to beat several times inside the 2.7 s window and one slow enough not to
    /// complete a cycle in it read different depths at the same level — so the
    /// schedule solves them in the order that makes them independent: the level
    /// is bisected to convergence first, *then* the rate is stepped from a
    /// deviation read at a settled level, then the level is re-bisected. Both
    /// rate steps therefore land on rounds where the depth is already right,
    /// and neither is ever the last word.
    pub const ROUNDS: usize = 24;
    /// The rounds the rate is stepped at: after the first eight halvings of the
    /// schema's 40 dB band (0.16 dB of bracket), and after five more each time.
    /// Three of them, because the rate has to be able to travel: the recording's
    /// A2 fundamental needs 4.8x the rate its envelope beats at, and one
    /// [`FalseBeatLoop::MAX_RATE_STEP`] does not get there.
    pub const RATE_ROUNDS: [usize; 3] = [8, 13, 18];
    /// How far the level bracket is re-opened around the current point after a
    /// rate step, dB. Wide enough to contain the whole effect of a rate that
    /// has at most [`FalseBeatLoop::MAX_RATE_STEP`] moved.
    pub const REOPEN_DB: f64 = 8.0;
    /// Largest single multiplicative step the rate may take, so one cell whose
    /// deviation is a null spike cannot throw the rate to a rail in one round.
    pub const MAX_RATE_STEP: f64 = 2.0;
    /// The slowest rate this fit will write, in Hz, whatever the schema allows:
    /// one cycle inside the analysis window
    /// ([`crate::motion::WINDOW_HI_S`] − [`crate::motion::WINDOW_LO_S`] = 2.7 s,
    /// 0.37 Hz), which is the modulation spectrum's own resolution
    /// (`renders/jitter/JITTER.md`). Under it neither statistic the loop reads
    /// is a measurement of the beat: the depth is a fraction of a cycle of it
    /// and the deviation is a slow drift.
    pub const MIN_FITTED_HZ: f64 = 1.0 / (crate::motion::WINDOW_HI_S - crate::motion::WINDOW_LO_S);

    /// Seeds the loop from the recording's own reading of the key.
    ///
    /// `rows` is [`fit_false_beat`]'s answer — which partials this key writes at
    /// all, and the rate each one's envelope beats at, which is where the search
    /// starts. `targets` is the same recording's [`Motion`] per partial, which
    /// is what the search runs against.
    pub fn new(rows: &[FalseBeat], targets: &[(u32, Motion)]) -> FalseBeatLoop {
        let states = rows
            .iter()
            .filter_map(|row| {
                let k = u32::from(row.k);
                let target = targets.iter().find(|(t, _)| *t == k).map(|(_, m)| *m)?;
                Some(LoopState {
                    k,
                    target_depth_db: target.beat_depth_db,
                    target_cents: target.floored_cents(),
                    lo_db: f64::from(MIN_FALSE_BEAT_DB),
                    hi_db: f64::from(MAX_FALSE_BEAT_DB),
                    db: f64::from(row.db),
                    hz: f64::from(row.hz),
                    baseline_depth_db: f64::NAN,
                    baseline_cents: f64::NAN,
                    best: None,
                    best_cost: f64::INFINITY,
                    dropped: false,
                })
            })
            .collect();
        FalseBeatLoop { states, round: 0 }
    }

    /// The rows to render next. Empty on the first round: round zero is the
    /// baseline, the key with its own table cleared, which is what says whether
    /// there is anything to write at all.
    pub fn rows(&self) -> Vec<FalseBeat> {
        if self.round == 0 {
            return Vec::new();
        }
        self.states
            .iter()
            .filter(|s| !s.dropped)
            .map(|s| FalseBeat {
                k: s.k as u16,
                hz: s.hz.clamp(Self::MIN_FITTED_HZ, f64::from(MAX_FALSE_BEAT_HZ)) as f32,
                db: s.db.clamp(f64::from(MIN_FALSE_BEAT_DB), f64::from(MAX_FALSE_BEAT_DB)) as f32,
            })
            .collect()
    }

    /// Whether another round is owed.
    pub fn running(&self) -> bool {
        self.round <= Self::ROUNDS
    }

    /// Folds in one render of [`FalseBeatLoop::rows`].
    ///
    /// `rendered` is the engine's own [`Motion`] per partial of the same key at
    /// the same velocity, measured with the same code as the recording's, so
    /// the measurement's bias divides out of the comparison.
    pub fn observe(&mut self, rendered: &[(u32, Motion)]) {
        let round = self.round;
        for state in &mut self.states {
            let Some(seen) = rendered.iter().find(|(k, _)| *k == state.k).map(|(_, m)| *m) else {
                // The render did not resolve the partial. Nothing can be
                // concluded from it, so the state stands.
                continue;
            };
            let cost = state.cost(&seen);
            if round == 0 {
                state.baseline_depth_db = seen.beat_depth_db;
                state.baseline_cents = seen.floored_cents();
                state.best_cost = cost;
                // A companion only adds. If the unison is already deeper than
                // the piano, no level in the schema reaches the target, so the
                // search has nowhere to go and the baseline stands.
                state.dropped = seen.beat_depth_db >= state.target_depth_db;
                state.db = 0.5 * (state.lo_db + state.hi_db);
                continue;
            }
            if state.dropped {
                continue;
            }
            if cost < state.best_cost {
                state.best_cost = cost;
                state.best = Some((state.db, state.hz));
            }
            if seen.beat_depth_db < state.target_depth_db {
                state.lo_db = state.db;
            } else {
                state.hi_db = state.db;
            }
            if Self::RATE_ROUNDS.contains(&round) {
                // The deviation is linear in the rate at fixed level, so the
                // step is the ratio — damped and clamped, because a partial
                // that is spiking at a null reads a deviation the linear model
                // does not describe.
                let step = (state.target_cents / seen.floored_cents())
                    .clamp(1.0 / Self::MAX_RATE_STEP, Self::MAX_RATE_STEP);
                state.hz =
                    (state.hz * step).clamp(Self::MIN_FITTED_HZ, f64::from(MAX_FALSE_BEAT_HZ));
                state.lo_db = (state.db - Self::REOPEN_DB).max(f64::from(MIN_FALSE_BEAT_DB));
                state.hi_db = (state.db + Self::REOPEN_DB).min(f64::from(MAX_FALSE_BEAT_DB));
            }
            state.db = 0.5 * (state.lo_db + state.hi_db);
        }
        self.round += 1;
    }

    /// The rows the loop settled on: per partial, the **best point the search
    /// visited**, not the last one.
    ///
    /// Keeping the best rather than the endpoint is not a convenience. The
    /// landscape is not smooth: where the total partial passes close to an exact
    /// null the frequency track spikes, and a decibel of level can move the
    /// measured deviation from 2.6 to 10.2 cents (C4 k=1, measured). A bisection
    /// is still the right *search* — the depth is monotone in the level, so it
    /// walks straight to the right register — but its final point is whichever
    /// side of a halving it landed on, and that is not the point to write.
    pub fn solved(&self) -> Vec<FalseBeat> {
        let mut rows: Vec<FalseBeat> = self
            .states
            .iter()
            .filter_map(|s| {
                let (db, hz) = s.best?;
                Some(FalseBeat {
                    k: s.k as u16,
                    hz: hz.clamp(Self::MIN_FITTED_HZ, f64::from(MAX_FALSE_BEAT_HZ)) as f32,
                    db: db.clamp(f64::from(MIN_FALSE_BEAT_DB), f64::from(MAX_FALSE_BEAT_DB)) as f32,
                })
            })
            .collect();
        rows.sort_by_key(|r| r.k);
        rows.truncate(MAX_FALSE_BEATS_PER_KEY);
        rows
    }

    /// Per partial: what the render was asked for and what the recording asked
    /// of it — `(k, written, baseline depth, target depth, hz)`, for the report
    /// that has to say why a partial went unwritten.
    pub fn trace(&self) -> Vec<(u32, bool, f64, f64, f64)> {
        self.states
            .iter()
            .map(|s| {
                (
                    s.k,
                    s.best.is_some(),
                    s.baseline_depth_db,
                    s.target_depth_db,
                    s.best.map_or(f64::NAN, |(_, hz)| hz),
                )
            })
            .collect()
    }

    /// How many partials came back with no row: the engine's own unison was
    /// already closer to the recording than anything the mechanism could add.
    /// This is the count that says how much of `B1` the unison owns rather than
    /// the wire.
    pub fn unwritten(&self) -> usize {
        self.states.iter().filter(|s| s.best.is_none()).count()
    }
}

impl LoopState {
    /// How far one rendered partial is from the recording, in the units the two
    /// columns are gated in: a beat depth is worth [`B1_SCALE_DB`] and a
    /// frequency deviation is worth a factor of [`A1_SCALE`], so a point that is
    /// one gate-width out on either reads 1.
    ///
    /// The two terms are `B1`'s and `A1`'s own per-cell contributions, which is
    /// deliberate and has to be said plainly: after this change the columns
    /// aggregate the quantity the fit minimises, so `A1` and `B1` stop being an
    /// independent test **of the fit**. What stays independent is everything
    /// else — `A2` (placement) and `B2` (velocity coherence) are not in this
    /// objective and are not free to follow it, 26 of the 30 fitted keys are
    /// outside the column's four, and the jitter forensics' flatness and
    /// placement tables are measured on statistics nothing here touches.
    fn cost(&self, seen: &Motion) -> f64 {
        let depth = (seen.beat_depth_db - self.target_depth_db).abs() / B1_SCALE_DB;
        let ratio = (seen.floored_cents() / self.target_cents).ln().abs() / A1_SCALE.ln();
        depth + ratio
    }
}

/// The gate widths [`LoopState::cost`] is quoted in: `FUNDAMENTALS.md` §II.3's
/// own `B1 <= 3 dB` and `A1 <= 2.0`.
const B1_SCALE_DB: f64 = 3.0;
const A1_SCALE: f64 = 2.0;

#[cfg(test)]
mod tests {
    use super::*;

    fn motion(depth_db: f64, rate_hz: f64) -> Motion {
        Motion {
            mean_hz: 440.0,
            peak_db: 40.0,
            band_cents: 1.0,
            raw_cents: 1.5,
            weighted_cents: 1.0,
            beat_depth_db: depth_db,
            beat_rate_hz: rate_hz,
            prompt_db_s: -12.0,
            tail_db_s: -6.0,
            aftersound_db: 10.0,
        }
    }

    /// The depth that a companion `db` under the partial produces, which is the
    /// inverse of the inversion under test.
    fn depth_for(db: f64) -> f64 {
        let r = 10f64.powf(db / 20.0);
        20.0 * ((1.0 + r) / (1.0 - r)).log10()
    }

    /// `FUNDAMENTALS.md` §7.4's C4 row, put through the estimator: flat in `k`,
    /// so it is written, and the levels and rates come back as the table quotes
    /// them.
    #[test]
    fn a_flat_in_k_companion_is_written_as_the_false_beat_it_is() {
        let cells: Vec<(u32, Motion)> = [(1, -6.1, 1.11), (2, -3.6, 1.48), (3, -6.6, 0.74)]
            .iter()
            .map(|&(k, db, hz)| (k, motion(depth_for(db), hz)))
            .collect();
        let fit = fit_false_beat(60, &cells, &MotionConfig::default());
        assert_eq!(fit.verdict, FalseBeatVerdict::Written);
        assert_eq!(fit.rows.len(), 3);
        assert!((f64::from(fit.rows[0].db) + 6.1).abs() < 0.05, "{:?}", fit.rows);
        assert!((f64::from(fit.rows[1].hz) - 1.48).abs() < 0.01, "{:?}", fit.rows);
        assert!(fit.model_ratio > 1.0, "model ratio {}", fit.model_ratio);
    }

    /// The falsification: rates proportional to `k` are a unison mistuning, and
    /// the key is left empty however clean the measurement is.
    #[test]
    fn rates_that_scale_with_the_partial_number_are_refused() {
        // Every rate inside the schema's own 0.2..=3.0 Hz, so what is refused is
        // the shape and not the range.
        let cells: Vec<(u32, Motion)> = (1..=6)
            .map(|k| (k, motion(depth_for(-5.0), 0.45 * f64::from(k))))
            .collect();
        let fit = fit_false_beat(84, &cells, &MotionConfig::default());
        assert_eq!(fit.verdict, FalseBeatVerdict::ScalesWithPartial);
        assert!(fit.rows.is_empty());
        assert!(fit.model_ratio < 1.0, "model ratio {}", fit.model_ratio);
        // The measurement is still reported, so a coverage table can say what
        // was seen as well as what was written.
        assert_eq!(fit.measured.len(), 6);
    }

    /// A companion too quiet to name, or beating outside the schema's range, is
    /// coverage rather than a row. The floor moved to −40 dB at
    /// `DECISIONS.md` 249, so what is "too quiet" moved with it — the point of
    /// the test is the gate, not the number.
    #[test]
    fn a_companion_outside_the_schemas_range_is_not_written() {
        let cells: Vec<(u32, Motion)> = [(1, -48.0, 1.0), (2, -45.0, 3.4), (3, -52.0, 0.1)]
            .iter()
            .map(|&(k, db, hz)| (k, motion(depth_for(db), hz)))
            .collect();
        let fit = fit_false_beat(45, &cells, &MotionConfig::default());
        assert_eq!(fit.verdict, FalseBeatVerdict::NoneInRange);
        assert!(fit.rows.is_empty());
        assert_eq!(fit.measured.len(), 3);
    }

    /// Fewer partials than the correlation test needs is not a licence to skip
    /// the test.
    #[test]
    fn a_key_with_two_measured_partials_is_not_written() {
        let cells: Vec<(u32, Motion)> = [(1, -6.0, 1.0), (2, -6.0, 1.1)]
            .iter()
            .map(|&(k, db, hz)| (k, motion(depth_for(db), hz)))
            .collect();
        let fit = fit_false_beat(45, &cells, &MotionConfig::default());
        assert_eq!(fit.verdict, FalseBeatVerdict::TooFewPartials);
        assert!(fit.rows.is_empty());
    }

    /// A known velocity law comes back out of the regression, and comes back
    /// **neutral at the reference velocity** — which is what keeps every table
    /// fitted at velocity 90 where it was.
    #[test]
    fn a_known_velocity_law_comes_back_pinned_at_the_reference() {
        let config = MotionConfig::default();
        let (pp, ff) = (-5.0, 4.0);
        let cells: Vec<VelocityCell> = (1..=16)
            .map(|layer| {
                let velocity = (layer * 8) as u8;
                let t = f64::from(velocity.max(1) - 1) / 126.0;
                VelocityCell {
                    group: 0,
                    velocity,
                    db: -7.0 + pp + (ff - pp) * t,
                }
            })
            .collect();
        let fit = fit_strike_direction(&cells, 90, &config).expect("enough cells");
        assert!(
            (fit.swing_db - (ff - pp)).abs() < 0.05,
            "swing {}",
            fit.swing_db
        );
        let at = |t: f64| {
            f64::from(fit.direction.vh_db_at_pp)
                + (f64::from(fit.direction.vh_db_at_ff) - f64::from(fit.direction.vh_db_at_pp)) * t
        };
        assert!(at(fit.reference_t).abs() < 1e-4, "{}", at(fit.reference_t));
        assert!(fit.correlation > 0.99, "{}", fit.correlation);
        assert!(fit.residual_db < 0.01, "{}", fit.residual_db);
    }

    /// A recording with no velocity dependence writes a neutral field, which is
    /// the same instrument the section's absence describes.
    #[test]
    fn a_velocity_independent_recording_writes_a_neutral_law() {
        let cells: Vec<VelocityCell> = (1..=16)
            .map(|layer| VelocityCell {
                group: 0,
                velocity: (layer * 8) as u8,
                db: -6.4,
            })
            .collect();
        let fit = fit_strike_direction(&cells, 90, &MotionConfig::default()).expect("cells");
        assert!(fit.swing_db.abs() < 1e-6, "{}", fit.swing_db);
        assert_eq!(fit.direction.vh_db_at_pp, 0.0);
        assert_eq!(fit.direction.vh_db_at_ff, 0.0);
    }

    /// One layer whose partial went through a null does not set the slope.
    #[test]
    fn the_regression_is_robust_to_one_wild_layer() {
        let clean: Vec<VelocityCell> = (1..=16)
            .map(|layer| {
                let velocity = (layer * 8) as u8;
                let t = f64::from(velocity.max(1) - 1) / 126.0;
                VelocityCell {
                    group: 0,
                    velocity,
                    db: -7.0 - 5.0 + 9.0 * t,
                }
            })
            .collect();
        let mut wild = clean.clone();
        wild[3].db += 40.0;
        let a = fit_strike_direction(&clean, 90, &MotionConfig::default()).expect("cells");
        let b = fit_strike_direction(&wild, 90, &MotionConfig::default()).expect("cells");
        assert!(
            (a.swing_db - b.swing_db).abs() < 1.0,
            "{} vs {}",
            a.swing_db,
            b.swing_db
        );
    }

    // -- the closed loop ----------------------------------------------------

    /// A stand-in engine: a partial that already beats `baseline` deep on its
    /// own, plus whatever companion the loop asks for, added as an amplitude
    /// ratio and read back as a depth. Compressive by construction, which is
    /// the property `DECISIONS.md` 234 measured and the reason the level cannot
    /// be asserted.
    fn engine(baseline_depth_db: f64, rows: &[FalseBeat], f_hz: f64) -> Vec<(u32, Motion)> {
        let r_of = |d: f64| {
            let x = 10f64.powf(d / 20.0);
            (x - 1.0) / (x + 1.0)
        };
        let base = r_of(baseline_depth_db);
        (1..=8u32)
            .map(|k| {
                let row = rows.iter().find(|r| u32::from(r.k) == k);
                let added = row.map_or(0.0, |r| 10f64.powf(f64::from(r.db) / 20.0));
                let hz = row.map_or(0.0, |r| f64::from(r.hz));
                let r = (base + added).min(0.999);
                let depth = 20.0 * ((1.0 + r) / (1.0 - r)).log10();
                // The first-order deviation of a two-component partial, which
                // is the relation the rate step inverts.
                let cents = 1200.0 / std::f64::consts::LN_2 * hz * r
                    / (std::f64::consts::SQRT_2 * f_hz * f64::from(k));
                let mut m = motion(depth, hz);
                m.band_cents = cents;
                m.raw_cents = cents.max(1e-9);
                m.weighted_cents = cents;
                (k, m)
            })
            .collect()
    }

    fn run(seed: &[FalseBeat], targets: &[(u32, Motion)], baseline: f64, f_hz: f64) -> FalseBeatLoop {
        let mut loops = FalseBeatLoop::new(seed, targets);
        while loops.running() {
            let rendered = engine(baseline, &loops.rows(), f_hz);
            loops.observe(&rendered);
        }
        loops
    }

    /// The loop reaches a depth the open-loop inversion misses, because the
    /// engine's own unison is already beating underneath it.
    #[test]
    fn the_level_is_solved_on_the_render_and_not_asserted_from_the_recording() {
        let target_depth = 9.0;
        let mut want = motion(target_depth, 0.8);
        // The deviation a companion of that depth 0.8 Hz away makes at C4,
        // so that the rate has nowhere to go and the level is the whole fit.
        want.band_cents = 1.783;
        want.raw_cents = 2.0;
        want.weighted_cents = 1.7;
        let seed = [FalseBeat { k: 1, hz: 0.8, db: -6.0 }];
        let solved = run(&seed, &[(1, want)], 3.0, 261.6).solved();
        assert_eq!(solved.len(), 1);
        let rendered = engine(3.0, &solved, 261.6);
        let depth = rendered[0].1.beat_depth_db;
        assert!(
            (depth - target_depth).abs() < 0.2,
            "solved {:.2} dB -> rendered {depth:.2} dB, wanted {target_depth}",
            solved[0].db
        );
        // And it is *not* what the recording's own inversion asks for: that
        // reads the whole depth as the companion, ignoring the baseline.
        let open_loop = want.companion_db().expect("a depth inverts");
        assert!(
            f64::from(solved[0].db) < open_loop - 1.0,
            "closed {:.2} vs open {open_loop:.2}",
            solved[0].db
        );
    }

    /// A partial the engine already beats harder than the recording does is not
    /// written at all: the mechanism can only add.
    #[test]
    fn a_partial_the_unison_already_out_beats_is_left_unwritten_rather_than_forced() {
        let mut want = motion(3.0, 0.8);
        want.band_cents = 0.5;
        want.raw_cents = 1.0;
        want.weighted_cents = 0.5;
        let seed = [FalseBeat { k: 3, hz: 0.8, db: -6.0 }];
        let loops = run(&seed, &[(3, want)], 12.0, 261.6);
        assert_eq!(loops.unwritten(), 1);
        assert!(loops.solved().is_empty());
        let (k, written, baseline, target, _) = loops.trace()[0];
        assert_eq!((k, written), (3, false));
        assert!(baseline > target, "{baseline} vs {target}");
    }

    /// The rate carries the frequency deviation, and the loop moves it there:
    /// same depth, four times the deviation, four times the rate.
    #[test]
    fn the_rate_is_solved_against_the_deviation_the_depth_cannot_reach() {
        let f = 110.0;
        let depth = 0.81;
        let mut want = motion(depth, 0.59);
        want.band_cents = 1.44;
        want.raw_cents = 2.0;
        want.weighted_cents = 1.4;
        let seed = [FalseBeat { k: 1, hz: 0.59, db: -20.0 }];
        let solved = run(&seed, &[(1, want)], 0.02, f).solved();
        assert_eq!(solved.len(), 1);
        let rendered = engine(0.02, &solved, f);
        assert!(
            (rendered[0].1.beat_depth_db - depth).abs() < 0.15,
            "depth {:.3}",
            rendered[0].1.beat_depth_db
        );
        assert!(
            (rendered[0].1.band_cents - 1.44).abs() < 0.25,
            "cents {:.3} at {} Hz",
            rendered[0].1.band_cents,
            solved[0].hz
        );
        // The envelope beat the recording shows is 0.59 Hz; the rate that
        // reproduces its *frequency* motion is several times that, which is the
        // measurement saying the recording's fundamental is not two components.
        assert!(f64::from(solved[0].hz) > 2.0, "{}", solved[0].hz);
    }

    /// Nothing the loop writes can leave the schema, whatever it is asked for.
    #[test]
    fn the_loop_never_leaves_the_schemas_own_bounds() {
        let mut want = motion(19.0, 8.0);
        want.band_cents = 40.0;
        want.raw_cents = 40.0;
        want.weighted_cents = 40.0;
        let seed: Vec<FalseBeat> = (1..=8)
            .map(|k| FalseBeat { k, hz: 0.5, db: -6.0 })
            .collect();
        let targets: Vec<(u32, Motion)> = (1..=8).map(|k| (k, want)).collect();
        for row in run(&seed, &targets, 0.01, 261.6).solved() {
            assert!((MIN_FALSE_BEAT_HZ..=MAX_FALSE_BEAT_HZ).contains(&row.hz), "{row:?}");
            assert!((MIN_FALSE_BEAT_DB..=MAX_FALSE_BEAT_DB).contains(&row.db), "{row:?}");
        }
    }
}
