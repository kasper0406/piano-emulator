//! Spreading estimates measured at some notes across all 88 keys.
//!
//! A sample library gives one recording every minor third, so two of every
//! three keys are never measured. Every quantity the estimators produce —
//! inharmonicity, decay rates, strike position, hammer stiffness — varies
//! smoothly and monotonically-ish with pitch, so the missing keys come from an
//! interpolating curve through the measured ones.
//!
//! The curve is a **monotone cubic** (Fritsch-Carlson) in semitones, which is
//! log-frequency. Why not a plain cubic spline: a spline through data that
//! rises steeply at one end — `B` climbs two orders of magnitude across the
//! compass — overshoots between the last two points, and an overshoot in `B` or
//! in a decay rate is not a slightly wrong note, it is a note whose partials go
//! to the wrong place or whose damping is negative. Fritsch-Carlson limits the
//! node slopes so the interpolant cannot leave the interval its neighbouring
//! data spans, at the cost of some smoothness at the nodes. That is the right
//! trade for a table nobody differentiates.
//!
//! Quantities that live on a ratio scale (`B`, decay rates, felt stiffness) are
//! interpolated in the log domain: halving between two neighbours must look the
//! same wherever it happens in the compass, and a linear-domain curve through
//! `1e-4` and `1e-2` spends its whole range near the top.

use crate::error::{Error, Result};
use crate::preset::{index_to_key, NUM_KEYS};

/// A monotone cubic through measurements at some subset of the keys.
#[derive(Clone, Debug)]
pub struct CompassCurve {
    /// Sample positions, ascending. In semitones — MIDI key numbers — which is
    /// a log-frequency axis.
    xs: Vec<f64>,
    /// Sample values, in the interpolation domain (already logged when
    /// `log_values`).
    ys: Vec<f64>,
    /// Node slopes, limited to keep the interpolant monotone.
    slopes: Vec<f64>,
    log_values: bool,
}

impl CompassCurve {
    /// Builds the curve through `samples`, which are `(key, value)` pairs in
    /// any order. Duplicate keys are averaged.
    pub fn new(samples: &[(f64, f64)], log_values: bool) -> Result<Self> {
        let mut points: Vec<(f64, f64)> = samples
            .iter()
            .filter(|(_, y)| y.is_finite() && (!log_values || *y > 0.0))
            .map(|&(x, y)| (x, if log_values { y.ln() } else { y }))
            .collect();
        points.sort_by(|a, b| a.0.total_cmp(&b.0));
        // Average duplicates rather than letting the last one win: two velocity
        // layers of the same note are two measurements of one number.
        let mut xs: Vec<f64> = Vec::with_capacity(points.len());
        let mut ys: Vec<f64> = Vec::with_capacity(points.len());
        let mut counts: Vec<f64> = Vec::with_capacity(points.len());
        for (x, y) in points {
            if let Some(last) = xs.last().copied() {
                if (x - last).abs() < 1e-9 {
                    let i = ys.len() - 1;
                    counts[i] += 1.0;
                    ys[i] += (y - ys[i]) / counts[i];
                    continue;
                }
            }
            xs.push(x);
            ys.push(y);
            counts.push(1.0);
        }
        if xs.is_empty() {
            return Err(Error::Estimate(
                "compass interpolation needs at least one measured note".into(),
            ));
        }
        let slopes = monotone_slopes(&xs, &ys);
        Ok(Self {
            xs,
            ys,
            slopes,
            log_values,
        })
    }

    pub fn from_keys(samples: &[(u8, f64)], log_values: bool) -> Result<Self> {
        let points: Vec<(f64, f64)> = samples.iter().map(|&(k, v)| (f64::from(k), v)).collect();
        Self::new(&points, log_values)
    }

    pub fn samples(&self) -> usize {
        self.xs.len()
    }

    /// The curve at `x`, in the original domain.
    ///
    /// Outside the measured range the curve continues along the end segment's
    /// limited slope rather than flattening off. Clamping would be safer if the
    /// data ended anywhere near the middle of the compass, but it does not: the
    /// gap is at most a couple of semitones at either end, where `B` and the
    /// decay rates are changing fastest and holding them constant is the
    /// visibly wrong answer.
    pub fn value_at(&self, x: f64) -> f64 {
        let value = self.interpolate(x);
        if self.log_values {
            value.exp()
        } else {
            value
        }
    }

    pub fn value_at_key(&self, key: u8) -> f64 {
        self.value_at(f64::from(key))
    }

    fn interpolate(&self, x: f64) -> f64 {
        let n = self.xs.len();
        if n == 1 {
            return self.ys[0];
        }
        if x <= self.xs[0] {
            return self.ys[0] + self.slopes[0] * (x - self.xs[0]);
        }
        if x >= self.xs[n - 1] {
            return self.ys[n - 1] + self.slopes[n - 1] * (x - self.xs[n - 1]);
        }
        let upper = self.xs.partition_point(|&xi| xi <= x).clamp(1, n - 1);
        let (i, j) = (upper - 1, upper);
        let h = self.xs[j] - self.xs[i];
        let t = (x - self.xs[i]) / h;
        // Cubic Hermite basis.
        let t2 = t * t;
        let t3 = t2 * t;
        (2.0 * t3 - 3.0 * t2 + 1.0) * self.ys[i]
            + (t3 - 2.0 * t2 + t) * h * self.slopes[i]
            + (-2.0 * t3 + 3.0 * t2) * self.ys[j]
            + (t3 - t2) * h * self.slopes[j]
    }
}

/// Fritsch-Carlson node slopes: the three-point average, limited so that each
/// segment stays monotone in the direction its own data goes.
fn monotone_slopes(xs: &[f64], ys: &[f64]) -> Vec<f64> {
    let n = xs.len();
    if n == 1 {
        return vec![0.0];
    }
    let secants: Vec<f64> = (0..n - 1)
        .map(|i| (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]))
        .collect();
    let mut slopes = vec![0.0; n];
    slopes[0] = secants[0];
    slopes[n - 1] = secants[n - 2];
    for i in 1..n - 1 {
        // A local extremum in the data must be an extremum of the curve too,
        // which is what the sign test enforces.
        slopes[i] = if secants[i - 1] * secants[i] <= 0.0 {
            0.0
        } else {
            0.5 * (secants[i - 1] + secants[i])
        };
    }
    for i in 0..n - 1 {
        if secants[i] == 0.0 {
            slopes[i] = 0.0;
            slopes[i + 1] = 0.0;
            continue;
        }
        let alpha = slopes[i] / secants[i];
        let beta = slopes[i + 1] / secants[i];
        let magnitude = alpha * alpha + beta * beta;
        if magnitude > 9.0 {
            let scale = 3.0 / magnitude.sqrt();
            slopes[i] = scale * alpha * secants[i];
            slopes[i + 1] = scale * beta * secants[i];
        }
    }
    slopes
}

/// Interpolates measurements at some keys onto all 88, A0 to C8.
pub fn interpolate_keys(samples: &[(u8, f64)], log_values: bool) -> Result<Vec<f64>> {
    let curve = CompassCurve::from_keys(samples, log_values)?;
    Ok((0..NUM_KEYS)
        .map(|index| curve.value_at_key(index_to_key(index)))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_curve_passes_through_every_measurement() {
        let samples = [(21u8, 1.0e-4), (48, 3.0e-4), (60, 4.0e-4), (84, 1.2e-3), (108, 1.0e-2)];
        let curve = CompassCurve::from_keys(&samples, true).unwrap();
        for (key, value) in samples {
            assert!(
                (curve.value_at_key(key) / value - 1.0).abs() < 1e-12,
                "key {key}"
            );
        }
    }

    #[test]
    fn a_steeply_rising_table_is_interpolated_without_overshoot() {
        // The default preset's B curve: a plain cubic spline overshoots between
        // the last two anchors, which would put a treble note's partials in the
        // wrong place.
        let samples = [(21u8, 1.0e-4), (48, 3.0e-4), (60, 4.0e-4), (84, 1.2e-3), (108, 1.0e-2)];
        let table = interpolate_keys(&samples, true).unwrap();
        assert_eq!(table.len(), NUM_KEYS);
        for pair in table.windows(2) {
            assert!(pair[1] >= pair[0], "B is not monotone: {pair:?}");
        }
        assert!(table[0] >= 1.0e-4 - 1e-12 && *table.last().unwrap() <= 1.0e-2 + 1e-12);
    }

    #[test]
    fn a_local_maximum_stays_a_local_maximum() {
        // Non-monotone data must not be flattened, but must not ring either:
        // the interpolant may not leave the range of its neighbouring samples.
        let samples = [(21u8, 1.0), (33, 3.0), (45, 2.0), (57, 2.5)];
        let curve = CompassCurve::from_keys(&samples, false).unwrap();
        for key in 21u8..=57 {
            let value = curve.value_at_key(key);
            assert!((0.9..=3.1).contains(&value), "key {key}: {value}");
        }
        assert!(curve.value_at_key(33) > curve.value_at_key(30));
        assert!(curve.value_at_key(33) > curve.value_at_key(36));
    }

    #[test]
    fn minor_third_spacing_recovers_a_smooth_truth_closely() {
        // Sample a smooth curve every three semitones, interpolate, and check
        // the notes that were never measured. TUNING.md's risk note: this is
        // the error the minor-third spacing costs on a smooth quantity.
        let truth = |key: f64| (0.0004 * (10f64).powf((key - 60.0) / 40.0)).ln();
        let samples: Vec<(u8, f64)> = (21..=108)
            .step_by(3)
            .map(|key| (key as u8, truth(f64::from(key)).exp()))
            .collect();
        let curve = CompassCurve::from_keys(&samples, true).unwrap();
        for key in 21u8..=108 {
            let error = curve.value_at_key(key) / truth(f64::from(key)).exp() - 1.0;
            assert!(error.abs() < 0.002, "key {key}: {:.4} %", 100.0 * error);
        }
    }

    #[test]
    fn one_measurement_is_a_constant_and_not_an_error() {
        let table = interpolate_keys(&[(60u8, 0.5)], false).unwrap();
        assert!(table.iter().all(|&v| (v - 0.5).abs() < 1e-12));
    }

    #[test]
    fn ends_extrapolate_along_the_data_rather_than_flattening() {
        let samples = [(24u8, 1.0), (36, 2.0), (48, 3.0)];
        let curve = CompassCurve::from_keys(&samples, false).unwrap();
        assert!((curve.value_at_key(21) - 0.75).abs() < 1e-9);
        assert!((curve.value_at_key(51) - 3.25).abs() < 1e-9);
    }
}
