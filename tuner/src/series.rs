//! One note's harmonic series as `renders/compass/COMPASS.md` measures it: a
//! level per partial, whether the partial is really there, and the three
//! statistics the compass scores a key on.
//!
//! This lived inside `examples/compass_scan.rs` until `DECISIONS.md` 272, and
//! moving it here is not tidying. The gains fit is now **bisected against
//! `irregularity`** — the row is smoothed until the series the engine renders is
//! no more jagged than the recording's — and the compass then scores the result
//! on the same number. A fit whose objective is a re-implementation of its own
//! acceptance test is a fit that can pass by disagreeing with it, so there is
//! one implementation and both call it.
//!
//! The projection is Goertzel-style onto each partial's own frequency through a
//! Hann window rather than a DFT bin, because the window is a second long, its
//! bins are 1 Hz apart, and the two modes of a coupled unison are a tenth of
//! that: what is wanted is the partial's energy *however it is split inside its
//! own neighbourhood*, so a skirt of ±3 bins is summed in power.

use std::f64::consts::TAU;

/// How far a partial must stand over the signal *between* partials before it is
/// counted as present, in dB.
///
/// Without this the spectrum metrics measure the analysis floor at the top of
/// the compass, where a key's twelfth partial is past the rendered band on one
/// side and past Nyquist on the other: two floor readings differ by whatever the
/// floor happens to be doing, and the treble reads a 25 dB `irregular` that has
/// nothing to do with the note. The test is a *local* one — the level midway to
/// each neighbouring partial — so a partial that is genuinely 26 dB under its
/// neighbours still counts, which is the whole point.
pub const PRESENCE_DB: f64 = 8.0;

/// Partials the compass takes its spectrum metrics over.
///
/// Twelve reaches 785 Hz at A0 and 12.5 kHz at C8; above that a bass note's
/// partials are closer together than the band-pass is wide.
pub const PARTIALS: usize = 12;

/// Window the level and the spectrum are measured over, seconds from the strike:
/// past the hammer's noise and inside the part every key still sounds in.
pub const WINDOW_S: (f64, f64) = (0.10, 1.10);

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Series {
    /// Level of partial `k`, dB.
    pub levels_db: Vec<f64>,
    /// Whether partial `k` stands [`PRESENCE_DB`] over the signal between it and
    /// its neighbours.
    pub present: Vec<bool>,
}

impl Series {
    pub fn measure(window: &[f32], partial_hz: &[f64], sr: f64) -> Series {
        if window.is_empty() || partial_hz.is_empty() {
            return Series {
                levels_db: vec![f64::NEG_INFINITY; partial_hz.len()],
                present: vec![false; partial_hz.len()],
            };
        }
        let n = window.len();
        // Hann, so a neighbouring partial 1/T away is 31 dB down.
        let taper: Vec<f64> = (0..n)
            .map(|i| 0.5 - 0.5 * (TAU * i as f64 / n as f64).cos())
            .collect();
        let bin = sr / n as f64;
        let at = |hz: f64| -> f64 {
            if hz <= 0.0 || hz >= 0.45 * sr {
                return f64::NEG_INFINITY;
            }
            let mut power = 0.0;
            for d in -3..=3i32 {
                let f = hz + f64::from(d) * bin;
                if f <= 0.0 {
                    continue;
                }
                let (mut re, mut im) = (0.0f64, 0.0f64);
                let w = TAU * f / sr;
                for (i, (&s, &t)) in window.iter().zip(&taper).enumerate() {
                    let phase = w * i as f64;
                    let v = f64::from(s) * t;
                    re += v * phase.cos();
                    im -= v * phase.sin();
                }
                power += re * re + im * im;
            }
            amp_db(2.0 * power.sqrt() / n as f64)
        };
        let levels_db: Vec<f64> = partial_hz.iter().map(|&hz| at(hz)).collect();
        // The floor is measured where the note is not: 45 % of the way to each
        // neighbouring partial, whichever side reads lower.
        let spacing = partial_hz[0];
        let present: Vec<bool> = partial_hz
            .iter()
            .zip(&levels_db)
            .map(|(&hz, &db)| {
                let floor = at(hz - 0.45 * spacing).min(at(hz + 0.45 * spacing));
                db.is_finite() && (!floor.is_finite() || db - floor >= PRESENCE_DB)
            })
            .collect();
        Series {
            levels_db,
            present,
        }
    }

    /// The partials that are really there, as `(k, level)`.
    pub fn sounding(&self) -> Vec<(usize, f64)> {
        self.levels_db
            .iter()
            .enumerate()
            .zip(&self.present)
            .filter(|((_, db), &p)| p && db.is_finite())
            .map(|((i, &db), _)| (i + 1, db))
            .collect()
    }

    /// Power-weighted mean partial index, expressed in semitones over `f0`.
    ///
    /// Register-free by construction: a note whose energy sits on its second
    /// partial reads 12 semitones whether it is A0 or C8, so the number can be
    /// compared across the compass without a trend to remove first.
    pub fn centroid_semitones(&self) -> f64 {
        let mut num = 0.0;
        let mut den = 0.0;
        for (k, db) in self.sounding() {
            let p = 10f64.powf(db / 10.0);
            num += p * k as f64;
            den += p;
        }
        if den <= 0.0 {
            return 0.0;
        }
        12.0 * (num / den).log2()
    }

    /// Mean absolute per-partial distance from another spectrum of the same
    /// note, dB, after the common offset between the two is removed.
    ///
    /// The offset is the **median** of the per-partial differences, so one
    /// railed partial cannot move it. Removing it is what makes this a measure
    /// of *colour* rather than of loudness: a note that is uniformly 6 dB quiet
    /// is a level error, which `level` already reports, and it is not what a
    /// listener calls a note that does not fit.
    ///
    /// Taken only where both spectra have the partial: a partial the recording
    /// does not have is not a distance, it is a missing measurement.
    pub fn distance_from(&self, other: &Series) -> f64 {
        let pairs: Vec<(f64, f64)> = self
            .levels_db
            .iter()
            .zip(&self.present)
            .zip(other.levels_db.iter().zip(&other.present))
            .filter(|((a, &pa), (b, &pb))| pa && pb && a.is_finite() && b.is_finite())
            .map(|((&a, _), (&b, _))| (a, b))
            .collect();
        if pairs.len() < 3 {
            return f64::NAN;
        }
        let offset = median(&pairs.iter().map(|(a, b)| a - b).collect::<Vec<_>>());
        pairs.iter().map(|(a, b)| (a - offset - b).abs()).sum::<f64>() / pairs.len() as f64
    }

    /// Mean absolute step between the levels of adjacent sounding partials, dB.
    ///
    /// Every mechanism in the engine that shapes a spectrum is smooth in `ln k`
    /// — the hammer, the bridge admittance, the strike comb, the microphone —
    /// with exactly one exception, the fitted `notes.partial_gains` row, which
    /// is a free number per partial. So this metric cannot be made large by the
    /// physics, and a key that reads large has a table problem.
    ///
    /// Steps are taken only between partials that are adjacent or one apart: a
    /// step across a long silent stretch of the series is a step across whatever
    /// the note stopped doing, not a step in the note.
    pub fn irregularity(&self) -> f64 {
        let sounding = self.sounding();
        let steps: Vec<f64> = sounding
            .windows(2)
            .filter(|w| w[1].0 - w[0].0 <= 2)
            .map(|w| (w[1].1 - w[0].1).abs())
            .collect();
        if steps.is_empty() {
            return 0.0;
        }
        steps.iter().sum::<f64>() / steps.len() as f64
    }
}

pub fn amp_db(amp: f64) -> f64 {
    if amp > 0.0 {
        20.0 * amp.log10()
    } else {
        f64::NEG_INFINITY
    }
}

fn median(values: &[f64]) -> f64 {
    let mut v: Vec<f64> = values.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    let mid = v.len() / 2;
    if v.len() % 2 == 0 {
        0.5 * (v[mid - 1] + v[mid])
    } else {
        v[mid]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A synthetic series with a known step in it comes back with that step,
    /// and a smooth one comes back smooth.
    #[test]
    fn the_irregularity_is_the_mean_absolute_step_between_partials() {
        let smooth = Series {
            levels_db: (0..8).map(|k| -3.0 * f64::from(k)).collect(),
            present: vec![true; 8],
        };
        assert!((smooth.irregularity() - 3.0).abs() < 1e-9);
        let mut jagged = smooth.clone();
        jagged.levels_db[3] += 12.0;
        // Two steps changed by 12 dB each: 3 -> 15 and 3 -> 9 (the sign flips).
        assert!(jagged.irregularity() > smooth.irregularity() + 2.0);
        // A partial that is not there is not a step of its own: the series
        // steps across it, one partial's worth of slope at a time, and a gap of
        // three or more is not stepped across at all.
        let mut gapped = smooth.clone();
        gapped.present[3] = false;
        // Five steps of 3 dB and one of 6 dB, across the hole.
        assert!((gapped.irregularity() - (5.0 * 3.0 + 6.0) / 6.0).abs() < 1e-9);
        for i in 2..6 {
            gapped.present[i] = false;
        }
        assert!((gapped.irregularity() - 3.0).abs() < 1e-9);
    }
}
