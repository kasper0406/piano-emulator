//! **A key's own loudness**: how far the engine's note sits from the recording
//! of the same note once the two instruments' common gain is out, what part of
//! that is worth writing, and where across the compass it may be carried.
//!
//! `DECISIONS.md` 272 measured this quantity, named it
//! [`GainRow::level_db`](crate::estimate::shaping::GainRow::level_db), tried two
//! homes for it, measured both to be worse than nothing and **decided not to
//! write it anywhere**. Item 453 re-opened that decision on the evidence: on the
//! recorded ladder the engine's C4 sits **8.96 dB** and its D#3 **8.27 dB**
//! under where the recording of the same key sits, everything else inside
//! ±1.1 dB, and no stage of this factory fits, scores or gates a key's absolute
//! level at all. Item 457 is what replaced the decision; this module is the
//! estimator it turns on.
//!
//! # Item 272's objection was right and is the whole design
//!
//! *"What the fit is chasing is the library's take-to-take gain."* Over the 28
//! fitted keys the removed level has a standard deviation of 4.82 dB and a
//! smooth polynomial in the key explains **1.2 % of it at degree 1 and 26 % at
//! degree 4**; the residual's lag-1 autocorrelation across the sampled keys is
//! +0.08 — white. Carried in full it puts F#5 **17.9 dB over its own
//! neighbours**. So the two obvious estimators are both wrong: a per-key free
//! gain memorises the library's noise, and a smooth curve cannot carry a defect
//! that is not smooth.
//!
//! What is right is the estimator that sits between them, and it is not a
//! compromise but the standard answer to exactly this shape of problem: each
//! key's own measurement, **shrunk toward the compass's smooth curve by how much
//! of its spread is noise** ([`LevelCurve::fit`]). The noise is not assumed —
//! it is measured on the library itself, as the scatter between two takes of one
//! key (the neighbouring velocity layer), which is the same floor
//! `realism::VelocityLayers` and the melody board's `seam_floor` are built from.
//! A key whose deficit is one take-to-take sigma is shrunk almost to the curve;
//! a key three sigmas out keeps most of what it measured. That is the sentence
//! item 453 asked for: *capture the −9 dB outliers without memorising the
//! noise*.
//!
//! # Three rails, each of them a measurement
//!
//! * **The common offset is removed first.** The engine's master gain against
//!   the library's mastering is about 15 dB and is nobody's error; what is
//!   fitted is each key's departure from the register's own median, which is
//!   the melody board's `seam` and the one statistic a median-based gate cannot
//!   see (`DECISIONS.md` 456).
//! * **Only recorded keys are fitted** (`DECISIONS.md` 328). A transposed
//!   reference note's level is the neighbour's take through a resampler and its
//!   gain is the neighbour's gain.
//! * **Nothing is written past [`MAX_LEVEL_DB`]**, which is the *recording's*
//!   own worst key-to-key level residual and not a number anyone chose.

use crate::estimate::texture::LogLine;

/// The largest per-key level this estimator will ever write, in dB.
///
/// The piano's own worst: measured over the 88 keys of
/// `renders/compass/COMPASS.md`, the **recording's** `level` residual against
/// its own eight nearest same-`N` neighbours has a robust sigma of 1.48 dB, a
/// p90 of 3.25 and a worst of **6.40** (`DECISIONS.md` 272's own table, which
/// is where `fit::motion::LEVEL_BAND_DB`'s 2.96 — two sigmas of the same
/// quantity — comes from). A per-key gain larger than the largest one the
/// instrument itself has is not a measurement of the instrument.
///
/// It is deliberately larger than `LEVEL_BAND_DB`, and the two are about
/// different things: that band bounds how far a row fitted for its *shape* may
/// drag a level as a side effect, where this bounds a level written **on
/// purpose** against the recording of the same key. Two sigmas is the right
/// bound on an accident; the distribution's own worst is the right bound on an
/// intention.
pub const MAX_LEVEL_DB: f64 = 6.40;

/// Least take-to-take sigma the shrinkage will assume, in dB.
///
/// The measured floor is the library's, and at a few keys two velocity layers
/// of one note agree to a hundredth of a decibel — which would ask this
/// estimator to believe every decibel it measures. A quarter of a decibel is far
/// under the 1.48 dB of key-to-key scatter the same recordings have and is a
/// bound on the arithmetic rather than on the measurement.
pub const MIN_TAKE_SIGMA_DB: f64 = 0.25;

/// One recorded key's measurement: its own level deficit and the take-to-take
/// distance measured beside it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LevelPoint {
    pub key: u8,
    /// `engine - recording`, A-weighted, on the mono fold-down, in dB. The
    /// common offset is **not** removed here: [`LevelCurve::fit`] removes it,
    /// because it is a property of the set and not of the key.
    pub error_db: f64,
    /// `|recording - the same key out of the neighbouring velocity layer|`, dB:
    /// how far one key of this library moves between two takes of itself.
    pub take_db: f64,
}

/// The per-key level this stage writes, across the whole compass.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LevelCurve {
    /// The smooth part: a straight line in the key through the recorded keys'
    /// own deficits, in dB.
    pub line: LogLine,
    /// The common offset removed before anything was fitted, dB — the engine's
    /// gain against the library's, which is nobody's error.
    pub offset_db: f64,
    /// The take-to-take sigma the shrinkage used, dB.
    pub take_sigma_db: f64,
    /// The spread of the deficits about the line, dB.
    pub residual_sigma_db: f64,
    /// The **median** over the recorded keys of the share of its own departure
    /// from the line that this estimator believed. Reported, not applied: the
    /// shrinkage is per key, `1 - (take_sigma / departure)^2`, so a key one
    /// take-to-take sigma from the line keeps nothing and a key nine decibels
    /// from it keeps almost all of it. A population statistic here would be the
    /// wrong shape — a compass that is flat except for two keys has a robust
    /// residual sigma of nearly zero, and a shrinkage taken from *that* would
    /// throw away exactly the two keys the estimator exists for.
    pub shrink: f64,
    /// The recorded keys and the level finally written at each, ascending.
    pub points: Vec<(u8, f64)>,
}

impl LevelCurve {
    /// Fits the curve from the recorded keys' measurements.
    ///
    /// The line is fitted in **dB against the key** — `LogLine` is a line in the
    /// log of its values, so the deficits are exponentiated into it and the
    /// answer comes back out in dB, which keeps one interpolator in the
    /// repository rather than two.
    pub fn fit(points: &[LevelPoint]) -> LevelCurve {
        let mut sorted: Vec<LevelPoint> = points
            .iter()
            .copied()
            .filter(|p| p.error_db.is_finite())
            .collect();
        sorted.sort_by_key(|p| p.key);
        if sorted.is_empty() {
            return LevelCurve::default();
        }
        // 1. The common offset: the median error over the recorded keys. It is
        //    the engine's gain against the library's and no key's fault.
        let offset_db = median(sorted.iter().map(|p| p.error_db).collect());
        let deficits: Vec<(f64, f64)> = sorted
            .iter()
            .map(|p| (f64::from(p.key), p.error_db - offset_db))
            .collect();
        // 2. The smooth part.
        let line = LogLine::fit(
            &deficits
                .iter()
                .map(|&(k, d)| (k, d.exp()))
                .collect::<Vec<_>>(),
        );
        let at_line = |key: f64| -> f64 {
            let v = line.at(key as u8);
            if v > 0.0 { v.ln() } else { 0.0 }
        };
        // 3. The two spreads the shrinkage is the ratio of.
        let residuals: Vec<f64> = deficits.iter().map(|&(k, d)| d - at_line(k)).collect();
        let residual_sigma_db = robust_sigma(&residuals);
        // The noise this estimator is shrunk against: how far one key of this
        // library moves between two takes of itself. The median of it and not
        // its scatter — the quantity is already a distance.
        let take_sigma_db = median(
            sorted
                .iter()
                .map(|p| p.take_db)
                .filter(|d| d.is_finite())
                .collect(),
        )
        .max(MIN_TAKE_SIGMA_DB);
        // 4. What is written at each key that measured: the line plus the share
        //    of its own departure the noise leaves standing, capped. The share
        //    is **per key** — see `LevelCurve::shrink`.
        let believe = |residual: f64| -> f64 {
            if residual.abs() <= 0.0 {
                return 0.0;
            }
            (1.0 - (take_sigma_db / residual).powi(2)).clamp(0.0, 1.0)
        };
        let shrink = median(residuals.iter().map(|&r| believe(r)).collect());
        let written: Vec<(u8, f64)> = sorted
            .iter()
            .zip(&deficits)
            .zip(&residuals)
            .map(|((p, &(k, _)), &residual)| {
                let want = -(at_line(k) + believe(residual) * residual);
                (p.key, want.clamp(-MAX_LEVEL_DB, MAX_LEVEL_DB))
            })
            .collect();
        LevelCurve {
            line,
            offset_db,
            take_sigma_db,
            residual_sigma_db,
            shrink,
            points: written,
        }
    }

    /// The level to write at any key, in dB: exact at a key that measured,
    /// interpolated between them, and tapered back to the smooth line an octave
    /// past the last one that did.
    ///
    /// The same device `tail::LowDecay::at` and `tail::DecayModel::at` use, for
    /// the same reason (`DECISIONS.md` 320(d), 335): the departures of the keys
    /// that measured are white across the compass, so interpolating them is all
    /// that may be done with them and **drawing** their scatter would land a
    /// random gain on every key between two measured ones — which is the defect
    /// this stage exists to remove, rebuilt one register at a time.
    pub fn at(&self, key: u8) -> f64 {
        if self.points.is_empty() {
            return 0.0;
        }
        let smooth = |k: u8| -> f64 {
            let v = self.line.at(k);
            let db = if v > 0.0 { -v.ln() } else { 0.0 };
            db.clamp(-MAX_LEVEL_DB, MAX_LEVEL_DB)
        };
        let x = f64::from(key);
        let (first, last) = (self.points[0].0, self.points[self.points.len() - 1].0);
        // Past the ends: the departure tapers to nothing over an octave, so a
        // key twelve semitones above the last measured one is the line alone.
        let taper = |edge: u8| -> f64 {
            let distance = (x - f64::from(edge)).abs();
            (1.0 - distance / 12.0).clamp(0.0, 1.0)
        };
        let departure = if key <= first {
            (self.points[0].1 - smooth(first)) * taper(first)
        } else if key >= last {
            (self.points[self.points.len() - 1].1 - smooth(last)) * taper(last)
        } else {
            let i = self
                .points
                .iter()
                .position(|&(k, _)| k > key)
                .expect("key is inside the range");
            let (ka, va) = self.points[i - 1];
            let (kb, vb) = self.points[i];
            let t = (x - f64::from(ka)) / (f64::from(kb) - f64::from(ka));
            (va - smooth(ka)) * (1.0 - t) + (vb - smooth(kb)) * t
        };
        (smooth(key) + departure).clamp(-MAX_LEVEL_DB, MAX_LEVEL_DB)
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.retain(|x| x.is_finite());
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// `1.4826 MAD` about the median — robust, because the quantity being measured
/// has outliers in it by construction and one F#5 must not be allowed to set
/// the width of the distribution it is an outlier of.
fn robust_sigma(values: &[f64]) -> f64 {
    let v: Vec<f64> = values.iter().copied().filter(|x| x.is_finite()).collect();
    if v.len() < 3 {
        return 0.0;
    }
    let centre = median(v.clone());
    1.4826 * median(v.iter().map(|x| (x - centre).abs()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(key: u8, error_db: f64, take_db: f64) -> LevelPoint {
        LevelPoint {
            key,
            error_db,
            take_db,
        }
    }

    /// The estimator's whole purpose in one assertion: a key that is nine
    /// decibels down keeps most of it, and a key one take-to-take sigma down
    /// keeps almost none.
    #[test]
    fn an_outlier_survives_the_shrinkage_and_the_librarys_own_noise_does_not() {
        // Nine keys at the common offset, one nine decibels under it, and a
        // take-to-take scatter of half a decibel.
        let mut points: Vec<LevelPoint> = (0..9)
            .map(|i| point(48 + 3 * i, -15.0 + 0.4 * f64::from(i % 3) - 0.4, 0.5))
            .collect();
        points[3].error_db = -15.0 - 9.0;
        let curve = LevelCurve::fit(&points);
        let outlier = curve.at(points[3].key);
        assert!(
            outlier > 3.0,
            "the nine-decibel key kept only {outlier:.2} dB"
        );
        for p in points.iter().filter(|p| (p.error_db + 15.0).abs() < 1.0) {
            assert!(
                curve.at(p.key).abs() < 1.5,
                "key {} moved {:.2} dB on nothing",
                p.key,
                curve.at(p.key)
            );
        }
    }

    /// Nothing but the library's own take-to-take gain: the estimator writes
    /// nothing at all, which is item 272's objection honoured rather than
    /// argued with.
    #[test]
    fn a_compass_of_pure_take_to_take_noise_is_written_as_nothing() {
        let noise = [0.9, -1.1, 0.7, -0.8, 1.0, -0.6, 0.8, -1.0, 0.9, -0.7];
        let points: Vec<LevelPoint> = noise
            .iter()
            .enumerate()
            .map(|(i, &n)| point(45 + 3 * i as u8, -15.0 + n, 1.6))
            .collect();
        let curve = LevelCurve::fit(&points);
        assert!(curve.shrink < 0.5, "shrink {}", curve.shrink);
        for p in &points {
            assert!(
                curve.at(p.key).abs() < 0.5,
                "key {} moved {:.2} dB on noise alone",
                p.key,
                curve.at(p.key)
            );
        }
    }

    /// Nothing is ever written past the piano's own worst key-to-key level
    /// residual, however far the measurement goes.
    #[test]
    fn nothing_is_written_past_the_pianos_own_worst_key_to_key_level() {
        let mut points: Vec<LevelPoint> = (0..9).map(|i| point(48 + 3 * i, -15.0, 0.2)).collect();
        points[4].error_db = -60.0;
        let curve = LevelCurve::fit(&points);
        assert!(
            curve.at(points[4].key) <= MAX_LEVEL_DB + 1e-9,
            "wrote {:.2} dB",
            curve.at(points[4].key)
        );
        assert!(curve.at(points[4].key) >= MAX_LEVEL_DB - 1e-9);
    }

    /// A key nobody recorded reads the interpolation of its neighbours and
    /// never a value of its own, and a key an octave past the last recorded one
    /// reads the smooth line.
    #[test]
    fn an_unrecorded_key_is_interpolated_and_the_ends_taper() {
        let points = vec![
            point(48, -15.0, 0.2),
            point(51, -21.0, 0.2),
            point(54, -15.0, 0.2),
            point(57, -15.0, 0.2),
            point(60, -21.0, 0.2),
        ];
        let curve = LevelCurve::fit(&points);
        let (a, mid, b) = (curve.at(48), curve.at(49), curve.at(51));
        assert!(
            mid > a.min(b) - 1e-6 && mid < b.max(a) + 1e-6,
            "{a:.2} {mid:.2} {b:.2}"
        );
        // Past the last recorded key the departure tapers to nothing over an
        // octave, so the curve is its own smooth line there.
        let smooth = |k: u8| -(curve.line.at(k).ln());
        let departure = |k: u8| (curve.at(k) - smooth(k)).abs();
        assert!(
            departure(61) > departure(66) && departure(66) > departure(72),
            "{:.3} {:.3} {:.3}",
            departure(61),
            departure(66),
            departure(72)
        );
        assert!(departure(72) < 1e-6, "{:.4}", departure(72));
    }
}
