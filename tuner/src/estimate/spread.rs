//! Per-string decay spread, from the drift of a beating composite partial.
//!
//! **Measured and printed, never written.** This module fitted
//! `voicing.unison_sigma_scale`, which has been inert since the unison became a
//! coupled eigenproblem (`DECISIONS.md` 225): the split it inverts is an
//! *output* of that construction, which pushes a group's decay rates apart by a
//! factor of 4.6 at C4's fundamental without being asked, so writing it in as
//! well would count the same physics twice. `survey` stopped writing it in item
//! 324 and still prints it, because what it measures is the recordings' own
//! drift and that is a fact about the piano either way — and it is what
//! `tuner/tests/calibration.rs` closes the construction against. Read the rest
//! of this header as the derivation of a measurement, not of a preset field.
//!
//! The engine gives every string of a unison the same damping law. A real group
//! does not, and `docs/history/TUNING_REPORT.md` §6 found the signature: the *measured*
//! fundamental of some tenor notes moves as it decays — by up to 32 cents over
//! its first 20 dB at F#3, against −2.0 to +0.7 cents on the engine's own
//! renders, which structurally cannot do it. A single string cannot move its
//! pitch and neither can a unison whose strings share one `sigma`; what moves a
//! composite partial's frequency is a group that is mistuned **and** decays
//! unevenly, so that the survivor's pitch takes over.
//!
//! # What the drift is worth in decay rates
//!
//! Two strings a full unison width `d` apart, equally struck, decaying at
//! `sigma (1 -+ s)`. Two components inside one main lobe are one peak, and a
//! magnitude peak picker lands on their *amplitude*-weighted centre:
//!
//! ```text
//!     F(t) = f_mean + (d/2) * tanh(ln(a2/a1) / 2)
//! ```
//! with `ln(a2/a1) = -(sigma2 - sigma1) t = -2 s sigma t`. Waiting until the
//! partial has fallen `D` dB puts `sigma t = D ln 10 / 20`, so the whole drift
//! over that window is
//!
//! ```text
//!     |Delta| = (d/2) * tanh(s D ln 10 / 20)
//! ```
//! which inverts to `s` directly. Both factors are measured: `d` by the unison
//! estimator, whose beat is what the two strings' interval *is*, and `Delta` by
//! [`partial_drift`] — bin medians of the tracked frequency and a least-squares
//! line through them, which is §6's own construction.
//!
//! Read in cents on both sides, so the ratio `2 |Delta| / d` is dimensionless
//! and the note's pitch drops out. Every partial of the note measures the same
//! `s`, which is what makes the median over the low partials a check rather
//! than an average.
//!
//! Two details are what make the number come back rather than come back a
//! quarter light:
//!
//! * The drift is regressed against **time**, not against the partial's own
//!   level as [`residual::track_glide`](crate::residual::track_glide) does. A
//!   unison's beat period is comparable to the 20 dB window — C4's is 2.5 s
//!   against a 2–4 s window — so over that window the level is not a clock, and
//!   using it as one returns a spread 25 % light (measured, on synthetic
//!   groups). Time is a clock. What sets the *end* of the window is still the
//!   level: the last frame standing above `D` dB under the partial's peak.
//! * The inversion predicts the *measurement* and not the model. A
//!   least-squares line through a `tanh` does not pass through its ends, so
//!   what [`partial_drift`] reports is not `(d/2) tanh(...)`; `predicted_ratio`
//!   puts the same line through the same bins of the same `tanh` and the
//!   inversion is of that.
//!
//! # What it cannot say
//!
//! * **Which** string is the fast one. The drift's sign says which end of the
//!   group survives at *this* note, and §6's table changes sign from key to key
//!   (−31.9 cents at F#3, +3.7 at A4). The engine has one row per group size for
//!   the whole instrument, so what is pooled is the size of the spread; the row
//!   puts the slow string first and lets `unison_layout` decide which detuning
//!   that is.
//! * **How the spread is distributed** inside a group of three. The model above
//!   is the two-string one, applied to the pair that ends up dominating; a row
//!   of three is written as `[1-s, 1, 1+s]`, which has that pair at its ends.
//! * Anything at all once `2 |Delta| / d` approaches 1. There the survivor has
//!   already taken the partial over and `tanh` is flat: the measurement
//!   saturates, and what is reported is the largest spread it can still
//!   distinguish, flagged as such.
//! * **How wide the pair really is.** The inversion divides by the group's
//!   unison width because that is the interval the tuner can measure. It is not
//!   the only interval in there: the engine's own two polarizations are offset
//!   by 0.27–0.52 Hz, which at C4 is 1.8–3.4 cents against a 2.83-cent unison,
//!   and they decay at very different rates by construction. A drift of a given
//!   size divided by too small a width comes back as too large a spread, and on
//!   the engine's renders it does — by up to a factor of two at C4 (the gate
//!   measures it). What this estimator writes is therefore the *character* of
//!   the voicing, bounded, and not a coefficient to three figures; making it
//!   one needs the polarization offsets, which stage 1 does not measure.

use crate::preset::{UnisonSigmaScale, MAX_SIGMA_SCALE, MAX_UNISON, MIN_SIGMA_SCALE};
use crate::trajectory::{cents, NoteTrajectories, PartialTrack};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpreadConfig {
    /// How far the partial must decay for the drift to be measured over, in dB.
    /// §6's own window; long enough for the survivor to take over, short enough
    /// to stay above a recording's floor.
    pub drop_db: f64,
    /// Bins the drift is regressed over. Five across the window is §6's own
    /// resolution (4 dB bins over 20), and coarse enough that each bin holds
    /// several frames of a beating envelope rather than one.
    pub bins: usize,
    /// Highest partial a drift is read from. Every partial of the note carries
    /// the same `s`, but the high ones of a three-string group are six
    /// components at six frequencies and drift for reasons this model does not
    /// contain.
    pub max_partial: u32,
    /// Drift below which nothing is claimed, in cents.
    ///
    /// A quarter of a cent: below it a three-cent unison's ratio is under 0.17,
    /// which inverts to a spread of four percent — less than the width of the
    /// pooled median over a compass. §6's own control column (−2.0 to +0.7
    /// cents of drift on renders that cannot drift) is the floor on *library*
    /// material with the survey's long windows, and is the reason the gate
    /// measures this estimator's floor rather than assuming one.
    pub min_drift_cents: f64,
    /// Largest `2 |Delta| / d` the inversion is trusted at. `tanh` is flat
    /// above it, so a measurement that reaches it is reported at the ceiling
    /// and marked saturated rather than inverted to a number the data cannot
    /// support.
    pub max_ratio: f64,
    /// Ceiling on the spread itself. `1 + s` and `1 - s` are decay-rate
    /// multipliers and the engine bounds those to `0.5..=2.0`.
    pub max_spread: f64,
    /// Fewest notes a group size needs before its own row is written from its
    /// own notes rather than from the instrument's median.
    pub min_notes: usize,
    /// Drift, in cents, that this instrument produces with *one* damping law —
    /// subtracted before the inversion.
    ///
    /// Zero, and left at zero by the survey, because a recording of somebody
    /// else's piano does not come with a control. The engine's renders do, and
    /// they say the baseline is not nothing: a group whose strings share one
    /// `sigma` still drifts by 0.3–1.2 cents, because its strings are coupled
    /// through the bridge and because its two polarizations are themselves a
    /// pair with different rates and different frequencies. The gate measures
    /// it, and this is where it goes.
    pub baseline_cents: f64,
}

impl Default for SpreadConfig {
    fn default() -> Self {
        Self {
            drop_db: 20.0,
            bins: 5,
            max_partial: 4,
            min_drift_cents: 0.25,
            max_ratio: 0.9,
            max_spread: 0.45,
            min_notes: 3,
            baseline_cents: 0.0,
        }
    }
}

/// What one note's beating partials say about its strings' damping.
#[derive(Clone, Debug, PartialEq)]
pub struct NoteSpread {
    pub key: u8,
    /// Strings in the group, from the preset. One string cannot beat and is
    /// never measured.
    pub strings: usize,
    /// The group's full detune width, in cents, as the unison estimator
    /// measured it.
    pub detune_cents: f64,
    /// Per-partial drift over the first [`SpreadConfig::drop_db`], in cents,
    /// positive when the partial falls — see [`partial_drift`].
    pub drifts: Vec<(u32, f64)>,
    /// The recovered spread `s`, or `None` where the note said nothing.
    pub spread: Option<f64>,
    /// Whether the inversion hit [`SpreadConfig::max_ratio`], i.e. the drift
    /// is at or past what the model can tell apart.
    pub saturated: bool,
}

impl NoteSpread {
    /// Median of the per-partial drifts, in cents — §6's own number, signed.
    ///
    /// Signed and not absolute: the sign says which end of the group survives,
    /// it is the same for every partial of a note, and folding it away turns a
    /// baseline that points one way and a spread that points the other into two
    /// numbers that cannot be told apart.
    pub fn drift_cents(&self) -> Option<f64> {
        median(self.drifts.iter().map(|&(_, cents)| cents))
    }
}

/// Measures one note's decay spread.
///
/// `detune_cents` is the group's **full** width, not the dominant beat: for a
/// group of three it is the interval between the outer strings, which is the
/// pair the survivor argument is about.
pub fn note_spread(
    key: u8,
    strings: usize,
    detune_cents: f64,
    trajectories: &NoteTrajectories,
    config: &SpreadConfig,
) -> NoteSpread {
    note_spread_over(key, strings, detune_cents, [trajectories], config)
}

/// The same, over every velocity layer of the note at once.
///
/// The drifts are pooled rather than the answers: a layer contributes one
/// number per partial, they all measure the same `s`, and the median over the
/// pool is what a note with sixteen layers and four usable partials has to say.
pub fn note_spread_over<'a>(
    key: u8,
    strings: usize,
    detune_cents: f64,
    layers: impl IntoIterator<Item = &'a NoteTrajectories>,
    config: &SpreadConfig,
) -> NoteSpread {
    let drifts: Vec<(u32, f64)> = layers
        .into_iter()
        .flat_map(|trajectories| trajectories.tracks.iter())
        .filter(|track| track.k >= 1 && track.k <= config.max_partial)
        .filter_map(|track| partial_drift(track, config).map(|cents| (track.k, cents)))
        .filter(|(_, cents)| cents.is_finite())
        .collect();
    let mut note = NoteSpread {
        key,
        strings,
        detune_cents,
        drifts,
        spread: None,
        saturated: false,
    };
    if strings < 2 {
        return note;
    }
    let Some(drift) = note.drift_cents() else {
        return note;
    };
    let (spread, saturated) = spread_from_drift(drift, detune_cents, config);
    note.spread = spread;
    note.saturated = saturated;
    note
}

/// The inversion alone: a measured drift and the group's width in, a spread
/// and whether it saturated out.
pub fn spread_from_drift(
    drift_cents: f64,
    detune_cents: f64,
    config: &SpreadConfig,
) -> (Option<f64>, bool) {
    if !(detune_cents.is_finite() && detune_cents > 0.0 && drift_cents.is_finite()) {
        return (None, false);
    }
    let drift = (drift_cents - config.baseline_cents).abs();
    if drift < config.min_drift_cents {
        // Measured, and measured to be nothing: a group whose strings share one
        // damping law drifts by less than this, so the note contributes a zero
        // to the pool rather than dropping out of it.
        return (Some(0.0), false);
    }
    let ratio = 2.0 * drift / detune_cents;
    let ceiling = predicted_ratio(config.max_spread, config).min(config.max_ratio);
    if ratio >= ceiling {
        return (Some(config.max_spread), true);
    }
    (Some(invert_ratio(ratio, config)), false)
}

/// How far one partial's tracked frequency moves while it decays
/// [`SpreadConfig::drop_db`], in cents. Positive is falling.
///
/// Bin medians against time and a least-squares line through them: medians
/// because a beating composite crosses every frequency in its range several
/// times per bin and one frame must not decide the answer, and a line because
/// what the model predicts is a slope, not two endpoints. The window ends at
/// the *last* frame above the level threshold rather than the first one under
/// it — a beat null takes the envelope 20 dB down and brings it back, and
/// stopping there would measure a fifth of the note.
pub fn partial_drift(track: &PartialTrack, config: &SpreadConfig) -> Option<f64> {
    let peak = track.peak()?;
    let floor = peak.amplitude * 10f64.powf(-config.drop_db / 20.0);
    let start = track.points.first()?.time_s;
    let end = track
        .points
        .iter()
        .rev()
        .find(|point| point.amplitude >= floor)?
        .time_s;
    let span = end - start;
    let bins = config.bins.max(2);
    if span <= 0.0 {
        return None;
    }
    let mut buckets: Vec<Vec<f64>> = vec![Vec::new(); bins];
    for point in &track.points {
        let u = (point.time_s - start) / span;
        if !(0.0..1.0).contains(&u) || point.frequency_hz <= 0.0 {
            continue;
        }
        buckets[(u * bins as f64) as usize].push(point.frequency_hz);
    }
    let points: Vec<(f64, f64)> = buckets
        .iter()
        .enumerate()
        .filter(|(_, bucket)| bucket.len() >= 2)
        .map(|(bin, bucket)| {
            let mut sorted = bucket.clone();
            sorted.sort_by(f64::total_cmp);
            ((bin as f64 + 0.5) / bins as f64, sorted[sorted.len() / 2])
        })
        .collect();
    if points.len() < 3 {
        return None;
    }
    let n = points.len() as f64;
    let mean_x = points.iter().map(|&(x, _)| x).sum::<f64>() / n;
    let mean_y = points.iter().map(|&(_, y)| y).sum::<f64>() / n;
    let (num, den) = points.iter().fold((0.0, 0.0), |(num, den), &(x, y)| {
        (num + (x - mean_x) * (y - mean_y), den + (x - mean_x).powi(2))
    });
    if den <= 0.0 {
        return None;
    }
    let slope = num / den;
    let at = |u: f64| mean_y + slope * (u - mean_x);
    Some(cents(at(1.0), at(0.0)))
}

/// The drift a spread of `s` produces, as a fraction of half the group's width.
///
/// Not `tanh` itself: what [`track_glide`] reports is the interval between the
/// ends of a *least-squares line* through the frequency against the partial's
/// own level, taken over the same amplitude bins, and a straight line through a
/// `tanh` does not pass through its ends. Predicting the measurement instead of
/// the model is worth a quarter of the answer — inverting the `tanh` directly
/// returns a spread 25 % light at `s = 0.3`.
fn predicted_ratio(spread: f64, config: &SpreadConfig) -> f64 {
    // `sigma t` at the end of the window: the group's mean decay is what puts
    // the partial `drop_db` down, which is the same relation the measurement's
    // own end point stands on.
    let last = config.drop_db * std::f64::consts::LN_10 / 20.0;
    let bins = config.bins.max(2);
    let points: Vec<(f64, f64)> = (0..bins)
        .map(|bin| {
            let u = (bin as f64 + 0.5) / bins as f64;
            (u, (spread * u * last).tanh())
        })
        .collect();
    let n = points.len() as f64;
    let mean_x = points.iter().map(|&(x, _)| x).sum::<f64>() / n;
    let mean_y = points.iter().map(|&(_, y)| y).sum::<f64>() / n;
    let (num, den) = points.iter().fold((0.0, 0.0), |(num, den), &(x, y)| {
        (num + (x - mean_x) * (y - mean_y), den + (x - mean_x).powi(2))
    });
    if den <= 0.0 {
        return 0.0;
    }
    num / den
}

/// `predicted_ratio` inverted, by bisection. Monotone in `s` below saturation,
/// so there is nothing cleverer to do and nothing to go wrong.
fn invert_ratio(ratio: f64, config: &SpreadConfig) -> f64 {
    let (mut lo, mut hi) = (0.0, config.max_spread);
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if predicted_ratio(mid, config) < ratio {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// The instrument's decay spread: one number per unison size, pooled over the
/// notes of that size.
#[derive(Clone, Debug, PartialEq)]
pub struct SigmaSpread {
    /// `spread[n - 1]` is `s` for a group of `n` strings.
    pub spread: [f64; MAX_UNISON],
    /// How many notes of each size the median was taken over.
    pub notes: [usize; MAX_UNISON],
    /// How many notes of each size saturated out and were therefore *not*
    /// counted — see [`SigmaSpread::pooled`].
    pub saturated: [usize; MAX_UNISON],
}

impl SigmaSpread {
    /// The neutral answer: every string of every group on the note's own law.
    pub fn none() -> Self {
        Self {
            spread: [0.0; MAX_UNISON],
            notes: [0; MAX_UNISON],
            saturated: [0; MAX_UNISON],
        }
    }

    /// Pools per-note measurements into one row per group size.
    ///
    /// A group size with fewer than [`SpreadConfig::min_notes`] measured notes
    /// falls back to the median over the whole instrument, and a run with
    /// nothing usable in it at all returns [`SigmaSpread::none`] — ones, which
    /// is the shared damping law the engine had before.
    ///
    /// **A note that saturated is not pooled.** Saturation means the drift is
    /// larger than the model can explain at *any* spread, which is a statement
    /// that the model does not describe that note — not a measurement of the
    /// largest spread. Counting it as the ceiling would make a median of a
    /// clamp: on Salamander, where two thirds of the tenor and treble notes
    /// drift further than their own unison width, that is the difference
    /// between writing a ±45 % row (the ceiling) and a ±12 % one (the median of
    /// the notes the model fits). The saturated notes are counted separately
    /// so a driver can say how many there were, which on that library is the
    /// most interesting number in the table.
    pub fn pooled(notes: &[NoteSpread], config: &SpreadConfig) -> Self {
        let measured: Vec<&NoteSpread> = notes
            .iter()
            .filter(|n| n.spread.is_some() && !n.saturated)
            .collect();
        let overall = median(measured.iter().filter_map(|n| n.spread));
        let mut pooled = SigmaSpread::none();
        for size in 1..=MAX_UNISON {
            let own: Vec<f64> = measured
                .iter()
                .filter(|n| n.strings == size)
                .filter_map(|n| n.spread)
                .collect();
            pooled.notes[size - 1] = own.len();
            pooled.saturated[size - 1] = notes
                .iter()
                .filter(|n| n.strings == size && n.saturated)
                .count();
            let value = if own.len() >= config.min_notes {
                median(own.into_iter())
            } else {
                overall
            };
            // A single string has nothing to spread against, whatever the rest
            // of the instrument did.
            pooled.spread[size - 1] = if size == 1 {
                0.0
            } else {
                value.unwrap_or(0.0).clamp(0.0, config.max_spread)
            };
        }
        pooled
    }

    /// The preset's rows: `[1]`, `[1-s, 1+s]`, `[1-s, 1, 1+s]` — the slow
    /// string first, since which string of a group survives is a per-note fact
    /// and a global row cannot carry it.
    ///
    /// Each row averages to exactly 1 — the engine checks it, because the row
    /// is a redistribution of the note's damping and not a second decay control
    /// beside `notes.sigma0` — and every entry is inside the engine's bounds by
    /// construction.
    pub fn rows(&self) -> Vec<UnisonSigmaScale> {
        (1..=MAX_UNISON)
            .map(|n| {
                // The row is 1 -+ s, so a spread past half the distance from
                // 1 to either engine bound would put a string outside them.
                let ceiling = f64::from(MAX_SIGMA_SCALE - 1.0).min(f64::from(1.0 - MIN_SIGMA_SCALE));
                let s = self.spread[n - 1].clamp(0.0, ceiling);
                let scale = match n {
                    1 => vec![1.0],
                    2 => vec![1.0 - s, 1.0 + s],
                    _ => {
                        let mut row = vec![1.0 - s, 1.0, 1.0 + s];
                        row.truncate(n);
                        row
                    }
                };
                UnisonSigmaScale {
                    scale: scale
                        .into_iter()
                        .map(|x| (x as f32).clamp(MIN_SIGMA_SCALE, MAX_SIGMA_SCALE))
                        .collect(),
                }
            })
            .collect()
    }

    /// Whether anything was measured at all.
    pub fn is_neutral(&self) -> bool {
        self.spread.iter().all(|&s| s == 0.0)
    }
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
    use crate::synth::{beating_track, Partial};

    /// A note whose two strings sit `detune_cents` apart and decay at
    /// `sigma (1 -+ spread)`, tracked at partials 1 to 4.
    pub(super) fn beating_note(f0: f64, detune_cents: f64, spread: f64, sigma: f64) -> NoteTrajectories {
        let ratio = (detune_cents / 1200.0).exp2();
        let hop = 0.01;
        let tracks: Vec<PartialTrack> = (1..=4u32)
            .map(|k| {
                let f = f0 * f64::from(k);
                beating_track(
                    k,
                    &[
                        Partial::new(k, f / ratio.sqrt(), 1.0, sigma * (1.0 - spread)),
                        Partial::new(k, f * ratio.sqrt(), 1.0, sigma * (1.0 + spread))
                            .with_phase(0.3),
                    ],
                    0.0,
                    hop,
                    40.0,
                )
            })
            .collect();
        NoteTrajectories {
            source: String::new(),
            note: None,
            sample_rate: 48_000.0,
            window_s: 0.17,
            hop_s: hop,
            seed: crate::trajectory::InharmonicModel::harmonic(f0),
            onset_s: 0.0,
            tracks,
        }
    }

    #[test]
    fn a_group_whose_strings_decay_unevenly_gives_up_how_unevenly() {
        let config = SpreadConfig::default();
        for &truth in &[0.15, 0.3] {
            let note = note_spread(54, 2, 3.0, &beating_note(185.0, 3.0, truth, 1.2), &config);
            let spread = note.spread.expect("a measured spread");
            assert!(
                (spread / truth - 1.0).abs() < 0.15,
                "s = {spread:.3} from {truth}: {note:?}"
            );
            assert!(!note.saturated, "{note:?}");
        }
    }

    #[test]
    fn a_group_that_shares_one_damping_law_is_measured_to_have_no_spread() {
        let config = SpreadConfig::default();
        let note = note_spread(54, 2, 3.0, &beating_note(185.0, 3.0, 0.0, 1.2), &config);
        assert_eq!(note.spread, Some(0.0), "{note:?}");
        // A single string has no unison at all, whatever its partials did.
        let single = note_spread(24, 1, 0.0, &beating_note(65.0, 0.0, 0.3, 1.2), &config);
        assert_eq!(single.spread, None);
    }

    #[test]
    fn a_drift_past_what_the_model_can_tell_apart_is_flagged_and_capped() {
        let config = SpreadConfig::default();
        // The drift is most of the group's own width: the survivor has taken
        // the partial over and the inversion is flat there.
        let note = NoteSpread {
            key: 54,
            strings: 3,
            detune_cents: 2.0,
            drifts: vec![(1, -1.9)],
            spread: None,
            saturated: false,
        };
        let measured = note_spread_from(&note, &config);
        assert!(measured.saturated, "{measured:?}");
        assert!(measured.spread.unwrap() <= config.max_spread);
    }

    /// The inversion alone, on drifts that were measured elsewhere.
    fn note_spread_from(note: &NoteSpread, config: &SpreadConfig) -> NoteSpread {
        let mut out = note.clone();
        let drift = out.drift_cents().expect("a drift");
        let (spread, saturated) = spread_from_drift(drift, out.detune_cents, config);
        out.spread = spread;
        out.saturated = saturated;
        out
    }

    #[test]
    fn the_rows_average_to_one_and_stay_inside_the_engines_bounds() {
        for spread in [0.0, 0.2, 0.45] {
            let pooled = SigmaSpread {
                spread: [0.0, spread, spread],
                notes: [0, 4, 9],
                saturated: [0; MAX_UNISON],
            };
            let rows = pooled.rows();
            assert_eq!(rows.len(), MAX_UNISON);
            for (i, row) in rows.iter().enumerate() {
                assert_eq!(row.scale.len(), i + 1);
                let mean = row.scale.iter().sum::<f32>() / (i + 1) as f32;
                assert!((mean - 1.0).abs() < 1e-6, "{row:?}");
                assert!(row.scale.iter().all(|&s| (MIN_SIGMA_SCALE..=MAX_SIGMA_SCALE).contains(&s)));
            }
            assert_eq!(rows[0].scale, vec![1.0]);
        }
    }

    #[test]
    fn a_size_with_too_few_notes_borrows_the_instruments_median() {
        let config = SpreadConfig::default();
        let note = |key: u8, strings: usize, spread: f64| NoteSpread {
            key,
            strings,
            detune_cents: 3.0,
            drifts: Vec::new(),
            spread: Some(spread),
            saturated: false,
        };
        // Four three-string notes agree on 0.3; the one two-string note is not
        // enough to write its own row, so it takes the instrument's median.
        let notes = vec![
            note(60, 3, 0.3),
            note(62, 3, 0.3),
            note(64, 3, 0.28),
            note(65, 3, 0.32),
            note(40, 2, 0.9),
        ];
        let pooled = SigmaSpread::pooled(&notes, &config);
        assert_eq!(pooled.notes, [0, 1, 4]);
        assert!((pooled.spread[2] - 0.3).abs() < 0.02, "{pooled:?}");
        assert!((pooled.spread[1] - 0.3).abs() < 0.02, "{pooled:?}");
        assert_eq!(pooled.spread[0], 0.0);
        // Nothing measured at all is ones, not a guess.
        assert!(SigmaSpread::pooled(&[], &config).is_neutral());
    }

    #[test]
    fn a_note_the_model_cannot_explain_is_counted_and_not_pooled() {
        let config = SpreadConfig::default();
        let note = |key: u8, spread: f64, saturated: bool| NoteSpread {
            key,
            strings: 3,
            detune_cents: 1.0,
            drifts: Vec::new(),
            spread: Some(spread),
            saturated,
        };
        // Three notes the model fits and four whose drift is wider than their
        // own unison: the row is the median of the three, and the four are
        // reported rather than counted as the ceiling they were clamped to.
        let notes = vec![
            note(60, 0.10, false),
            note(62, 0.12, false),
            note(64, 0.14, false),
            note(66, config.max_spread, true),
            note(68, config.max_spread, true),
            note(70, config.max_spread, true),
            note(72, config.max_spread, true),
        ];
        let pooled = SigmaSpread::pooled(&notes, &config);
        assert_eq!(pooled.notes[2], 3);
        assert_eq!(pooled.saturated[2], 4);
        assert!((pooled.spread[2] - 0.12).abs() < 1e-9, "{pooled:?}");
    }
}

