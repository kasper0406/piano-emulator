//! Where a note's two polarizations sit in the stereo image, from how far its
//! balance moves while it decays.
//!
//! `docs/history/TUNING_REPORT.md` §5: the median per-partial left-minus-right level of a
//! Salamander recording moves **1.2–6.2 dB** between 0.3 s and 2 s after the
//! strike, and the engine's render of the same note moved 0.02–0.14 dB. The
//! engine's could not move at all, for a structural reason — it panned one mono
//! voice per key, so whatever the pan, the ratio between the channels was fixed
//! for the life of the note. `voicing.polarization_pan_spread` is what removed
//! that: the horizontal polarization renders at `pan + spread` and the vertical
//! at `pan - spread`, and because the two decay at different rates the balance
//! now travels from one to the other as the fast plane dies.
//!
//! # What is measured, and what is inverted
//!
//! [`balance_drift`] is §5's own measurement, done here rather than in the
//! report's driver: the note's partials are located once from the mono sum, and
//! their levels are read out of a left frame and a right frame at two times.
//! The statistic is the *median over partials* of `|Δ(late) − Δ(early)|`, which
//! is what §5 tabulates — the median rather than the mean because a partial
//! that has decayed into the floor at 2 s reports the floor's balance and not
//! the string's.
//!
//! The inversion is deliberately not a model. What a spread of `s` does to the
//! measured drift depends on the key's own pan, on how much of the signal
//! arrives through the soundboard's diffuse path (which is not panned at all
//! and compresses every balance towards the middle), and on how far apart the
//! two polarizations' decay rates are — three things the engine already knows
//! and this crate would have to mirror to predict. So the relation is
//! *measured* instead, on the engine, over the same eight keys and the same two
//! times: it is a straight line in `s` to within a tenth of a dB, and
//! [`DRIFT_PER_SPREAD_DB`] and [`DRIFT_AT_ZERO_DB`] are its slope and
//! intercept. `tuner/tests/calibration.rs` is where that claim is checked
//! against the engine rather than asserted.

use crate::error::Result;
use crate::pipeline::{track_refined, NoteConfig};
use crate::residual::{frame_spectrum, partial_levels};
use crate::trajectory::InharmonicModel;

/// Median drift, in dB, that the engine's own renders show per unit of
/// `voicing.polarization_pan_spread`. Measured over keys 21, 33, 45, 57, 60,
/// 72, 84 and 96 at velocity 90 by `forensics/src/bin/drift_line.rs`, at five
/// spreads from 0 to the ceiling: **0.739, 1.356, 2.441, 2.913, 3.963 dB**,
/// a straight line to within 0.170 dB.
///
/// **On `presets/default.toml`, and it cannot be measured on
/// `presets/salamander-c5.toml`** as this line used to claim: that preset
/// carries a per-key `notes.pan_spread` table, which overrides the global
/// parameter, so the sweep is exactly flat — slope 0.000 at every spread. The
/// old value 8.3 is close to right for the wrong reason
/// (`DECISIONS.md` 279).
///
/// **And it is the finished chain that has this slope.** With the soundboard's
/// diffuse path removed the same sweep runs **15.84** dB per unit, because the
/// board is not panned at all and compresses every balance toward the middle —
/// the header above says so and the number is now measured. A gate that renders
/// at `board_mix = 0` is checking this constant on a different instrument.
pub const DRIFT_PER_SPREAD_DB: f64 = 8.0;

/// Median drift the same measurement returns at a spread of zero: not zero,
/// because a note's partials do not all decay at the same rate and the diffuse
/// field is not a constant either. Subtracted before the inversion, so a
/// recording that drifts no more than the engine already does asks for no
/// spread at all.
///
/// 0.68 dB, the intercept of the same eight-key fit. It read 0.32 while the
/// slope read 8.3, and the pair was taken with the board out of the chain,
/// where the intercept really is near zero (-0.23) — put the board back and an
/// unspread instrument drifts two thirds of a decibel on its own.
pub const DRIFT_AT_ZERO_DB: f64 = 0.68;

/// The engine's ceiling on the parameter (`soundboard::MAX_PAN_SPREAD`).
pub const MAX_PAN_SPREAD: f64 = 0.4;

/// The band `docs/history/TUNING_REPORT.md` §5 measured the recordings' drift in: 1.2 dB at
/// A0 to 6.2 dB at C6, over the whole compass and every key it sampled.
///
/// [`pan_spread_table`] aims at each key's *own* recorded drift clamped into
/// this band, and the clamp is the point of the field. A recording that drifts
/// 10.7 dB at C6 is a measurement of eight partials at the end of a treble
/// note's decay, where three of them are in the recording's floor; asking the
/// engine to reproduce it exactly is asking it to reproduce the floor. The
/// band is what the report is willing to claim, so it is what the fit aims at.
pub const MEASURED_DRIFT_BAND: (f64, f64) = (1.2, 6.2);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DirectivityConfig {
    /// The two times the balance is read at, in seconds after the onset.
    /// §5's own: 0.3 s is past the attack and 2 s is where a middle-register
    /// note has handed over to its slow polarization.
    pub early_s: f64,
    pub late_s: f64,
    /// A partial is only followed if its peak stands within this of the note's
    /// loudest. Below it the track spends both frames in the recording's floor,
    /// and the floor has a balance of its own.
    pub level_db: f64,
    /// Fewest partials a note must contribute before its drift is reported.
    pub min_partials: usize,
}

impl Default for DirectivityConfig {
    fn default() -> Self {
        Self {
            early_s: 0.3,
            late_s: 2.0,
            level_db: 60.0,
            min_partials: 3,
        }
    }
}

/// One note's stereo drift: the median over its partials of how far the
/// left-minus-right level moved between the two times.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BalanceDrift {
    pub drift_db: f64,
    pub partials: usize,
}

/// Measures [`BalanceDrift`] on a stereo recording of a note whose pitch is
/// `f0_hz`.
///
/// The partials are located on the mono sum, so both channels are read at the
/// same frequencies — a per-channel track would find the same partial at
/// slightly different places and report the difference as a level.
pub fn balance_drift(
    left: &[f32],
    right: &[f32],
    f0_hz: f64,
    sample_rate: f64,
    note_config: &NoteConfig,
    config: &DirectivityConfig,
) -> Result<BalanceDrift> {
    let mono: Vec<f32> = left
        .iter()
        .zip(right)
        .map(|(&l, &r)| 0.5 * (l + r))
        .collect();
    let (trajectories, _) = track_refined(
        &mono,
        sample_rate,
        InharmonicModel::harmonic(f0_hz),
        note_config,
    )?;
    let loudest = trajectories
        .tracks
        .iter()
        .filter_map(|track| track.peak())
        .map(|peak| peak.amplitude)
        .fold(0.0f64, f64::max);
    let floor = loudest * 10f64.powf(-config.level_db / 20.0);
    let frequencies: Vec<f64> = trajectories
        .tracks
        .iter()
        .filter(|track| track.peak().is_some_and(|peak| peak.amplitude >= floor))
        .filter_map(|track| track.weighted_frequency())
        .collect();

    let window = note_config.tracker.stft.window;
    // Half the main lobe of the window, which is how far a partial may sit
    // from where the tracker put it and still be the same partial.
    let guard_hz = 4.0 * sample_rate / window as f64;
    let deltas = |seconds: f64| -> Result<Vec<Option<f64>>> {
        let start = ((trajectories.onset_s + seconds) * sample_rate) as usize;
        let l = frame_spectrum(left, start, window, 1)?;
        let r = frame_spectrum(right, start, window, 1)?;
        let levels = |spectrum: &[f32]| {
            partial_levels(spectrum, sample_rate, window, &frequencies, guard_hz)
        };
        Ok(levels(&l)
            .into_iter()
            .zip(levels(&r))
            .map(|(l, r)| Some(20.0 * (l? / r?).log10()).filter(|d| d.is_finite()))
            .collect())
    };
    let (early, late) = (deltas(config.early_s)?, deltas(config.late_s)?);
    let mut moved: Vec<f64> = early
        .into_iter()
        .zip(late)
        .filter_map(|(a, b)| Some((b? - a?).abs()))
        .collect();
    if moved.len() < config.min_partials {
        return Err(crate::error::Error::Estimate(format!(
            "a stereo drift needs {} partials measured in both frames, got {}",
            config.min_partials,
            moved.len()
        )));
    }
    moved.sort_by(f64::total_cmp);
    Ok(BalanceDrift {
        drift_db: moved[moved.len() / 2],
        partials: moved.len(),
    })
}

/// The `voicing.polarization_pan_spread` whose renders drift as far as a
/// recording that drifted `measured_db`.
///
/// Clamped to the engine's own range at both ends: a recording that moves no
/// more than the engine's own partials already do asks for nothing, and one
/// that moves further than the ceiling can reach gets the ceiling — which is
/// what Salamander does, and is reported rather than hidden.
pub fn pan_spread_for_drift(measured_db: f64) -> f64 {
    if !measured_db.is_finite() {
        return 0.0;
    }
    ((measured_db - DRIFT_AT_ZERO_DB) / DRIFT_PER_SPREAD_DB).clamp(0.0, MAX_PAN_SPREAD)
}

/// The line one key's drift follows against the spread, measured on the engine
/// at two spreads rather than assumed from the compass-wide constants.
///
/// [`DRIFT_PER_SPREAD_DB`] is one number for the whole instrument, and the
/// instrument does not have one: at the ceiling the engine's own renders drift
/// 0.24 dB at A0, 1.26 at A2, 3.33 at C4, 8.67 at C5 and 5.59 at C7
/// (`docs/history/TUNING_REPORT.md` §5, Milestone A update) against the recordings' 1.24,
/// 4.73, 3.96, 5.33 and 5.85. A spread fitted to the compass median therefore
/// undershoots the bass by a factor of five and *overshoots* C5 and C6 — which
/// is what `notes.pan_spread` exists to fix, and what this inverts.
///
/// Both ends are measurements on the engine, so nothing here models the
/// diffuse field, the key's own pan or the polarizations' decay ratio: they are
/// in the two numbers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KeyDriftLine {
    pub key: u8,
    /// Drift the engine shows at spread 0 and at spread [`MAX_PAN_SPREAD`], dB.
    pub at_zero_db: f64,
    pub at_ceiling_db: f64,
}

impl KeyDriftLine {
    /// The spread whose renders drift as far as a recording that drifted
    /// `measured_db`, for this key.
    ///
    /// Clamped into the engine's range at both ends. A key whose line has no
    /// slope — the bass, where the pan has almost nowhere left to move —
    /// cannot be inverted, and asks for the ceiling if the recording drifts
    /// more than the engine does and for nothing if it does not. Saying so is
    /// better than dividing by a slope of 0.02 dB and writing a spread of 60.
    pub fn spread_for(&self, measured_db: f64) -> f64 {
        if !measured_db.is_finite() {
            return 0.0;
        }
        let slope = (self.at_ceiling_db - self.at_zero_db) / MAX_PAN_SPREAD;
        if slope <= MIN_USABLE_SLOPE {
            return if measured_db > self.at_zero_db {
                MAX_PAN_SPREAD
            } else {
                0.0
            };
        }
        ((measured_db - self.at_zero_db) / slope).clamp(0.0, MAX_PAN_SPREAD)
    }
}

/// Slope, in dB of drift per unit of spread, below which a key's line carries
/// no information. A tenth of the compass-wide [`DRIFT_PER_SPREAD_DB`]: below
/// that, a 1 dB error in the measured drift moves the answer by more than the
/// whole range of the parameter.
pub const MIN_USABLE_SLOPE: f64 = 0.83;

/// Fits `notes.pan_spread` — one spread per key — from the drift measured on
/// the recordings and the two lines measured on the engine.
///
/// `measured` and the two engine columns are sparse (a library samples a
/// subset of the compass); the answer is all 88 keys, filled in by the same
/// monotone-cubic interpolation across the compass every other per-note table
/// uses. Keys the recordings never covered therefore inherit their neighbours'
/// spread rather than the compass median, which is the whole point: the median
/// is what overshot.
pub fn pan_spread_table(measured: &[(u8, f64)], lines: &[KeyDriftLine]) -> Result<Vec<f32>> {
    let mut points: Vec<(u8, f64)> = Vec::new();
    for &(key, drift_db) in measured {
        let Some(line) = lines.iter().find(|l| l.key == key) else {
            continue;
        };
        let target = drift_db.clamp(MEASURED_DRIFT_BAND.0, MEASURED_DRIFT_BAND.1);
        points.push((key, line.spread_for(target)));
    }
    if points.is_empty() {
        return Err(crate::error::Error::Estimate(
            "no key had both a measured drift and an engine line".into(),
        ));
    }
    points.sort_by_key(|&(key, _)| key);
    points.dedup_by_key(|&mut (key, _)| key);
    // Linear, not logarithmic: a spread of zero is a legitimate answer for a
    // key whose image the recordings say does not move, and log of zero is not.
    Ok(crate::estimate::compass::interpolate_keys(&points, false)?
        .into_iter()
        .map(|s| s.clamp(0.0, MAX_PAN_SPREAD) as f32)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_recording_that_drifts_no_more_than_the_engine_asks_for_no_spread() {
        assert_eq!(pan_spread_for_drift(0.0), 0.0);
        assert_eq!(pan_spread_for_drift(DRIFT_AT_ZERO_DB), 0.0);
        assert_eq!(pan_spread_for_drift(f64::NAN), 0.0);
    }

    #[test]
    fn the_inversion_is_the_measured_line_and_stops_at_the_engines_ceiling() {
        // The middle of the line, to the resolution the constants carry.
        let spread =
            pan_spread_for_drift(DRIFT_AT_ZERO_DB + 0.5 * MAX_PAN_SPREAD * DRIFT_PER_SPREAD_DB);
        assert!((spread - 0.5 * MAX_PAN_SPREAD).abs() < 1e-12, "{spread}");
        assert_eq!(pan_spread_for_drift(100.0), MAX_PAN_SPREAD);
    }

    /// The per-key inversion is the per-key line, and the compass-wide
    /// constant is not: at C5 the engine moves twice as far per unit of spread
    /// as the median key, so the same recorded drift asks for half the spread.
    #[test]
    fn a_key_is_inverted_on_its_own_line_and_not_on_the_compass_median() {
        // The Milestone A columns of `docs/history/TUNING_REPORT.md` §5, as lines.
        let c5 = KeyDriftLine {
            key: 72,
            at_zero_db: 0.32,
            at_ceiling_db: 8.67,
        };
        let a2 = KeyDriftLine {
            key: 45,
            at_zero_db: 0.32,
            at_ceiling_db: 1.26,
        };
        // C5's recording drifts 5.33 dB. The compass line says 0.40 — the
        // ceiling — and C5's own line says 0.24.
        assert!((pan_spread_for_drift(5.33) - MAX_PAN_SPREAD).abs() < 1.0e-12);
        let own = c5.spread_for(5.33);
        assert!((own - 0.240).abs() < 0.005, "{own}");
        // A2's recording drifts 4.73 dB, which its own line cannot reach at
        // all: it gets the ceiling, and honestly.
        assert_eq!(a2.spread_for(4.73), MAX_PAN_SPREAD);
        assert_eq!(c5.spread_for(0.0), 0.0);
        assert_eq!(c5.spread_for(f64::NAN), 0.0);
    }

    /// A key whose image the spread cannot move — the bass, panned to the edge
    /// already — is not inverted through a slope of nearly zero.
    #[test]
    fn a_key_the_spread_cannot_move_is_not_divided_by_its_own_noise() {
        let a0 = KeyDriftLine {
            key: 21,
            at_zero_db: 0.20,
            at_ceiling_db: 0.24,
        };
        assert_eq!(a0.spread_for(1.24), MAX_PAN_SPREAD);
        assert_eq!(a0.spread_for(0.10), 0.0);
    }

    #[test]
    fn the_table_covers_the_compass_and_stays_inside_the_engines_range() {
        let lines: Vec<KeyDriftLine> = [21u8, 45, 60, 72, 96]
            .into_iter()
            .zip([0.24, 1.26, 3.33, 8.67, 5.59])
            .map(|(key, at_ceiling_db)| KeyDriftLine {
                key,
                at_zero_db: 0.32,
                at_ceiling_db,
            })
            .collect();
        let measured: Vec<(u8, f64)> = [21u8, 45, 60, 72, 96]
            .into_iter()
            .zip([1.24, 4.73, 3.96, 5.33, 5.85])
            .collect();
        let table = pan_spread_table(&measured, &lines).unwrap();
        assert_eq!(table.len(), 88);
        // Every target is inside the band the report is willing to claim.
        assert!(table.iter().all(|&s| (0.0..=MAX_PAN_SPREAD as f32).contains(&s)));
        // The measured keys keep their own answers: the interpolant passes
        // through its data, like every other per-note table.
        for (key, drift) in measured {
            let line = lines.iter().find(|l| l.key == key).unwrap();
            let index = usize::from(key - 21);
            let target = drift.clamp(MEASURED_DRIFT_BAND.0, MEASURED_DRIFT_BAND.1);
            assert!(
                (f64::from(table[index]) - line.spread_for(target)).abs() < 1.0e-6,
                "key {key}"
            );
        }
        // And the finding that motivated the field: C5 asks for much less than
        // the bass, where the global scalar had put them at the same 0.4.
        assert!(table[usize::from(72u8 - 21)] < table[usize::from(21u8 - 21)]);
    }

    /// The drift is a magnitude, so a note that swings the other way asks for
    /// the same spread — the parity of the sign belongs to the engine, which
    /// alternates it by key so that 88 voices do not pile their aftersound up
    /// on one side.
    #[test]
    fn a_drift_is_measured_as_a_magnitude() {
        let sample_rate = 48_000.0;
        let seconds = 4.0;
        let n = (sample_rate * seconds) as usize;
        // Two partials, each divided between the channels by a ratio that
        // moves over time in opposite directions.
        let (mut left, mut right) = (vec![0.0f32; n], vec![0.0f32; n]);
        for i in 0..n {
            let t = i as f64 / sample_rate;
            let mut l = 0.0;
            let mut r = 0.0;
            for (k, direction) in [(1u32, 1.0f64), (2, -1.0)] {
                let phase = std::f64::consts::TAU * 220.0 * f64::from(k) * t;
                let envelope = (-1.5 * t).exp();
                // One channel loses 3 dB more per second than the other.
                let tilt = (-0.35 * direction * t).exp();
                l += envelope * tilt * phase.sin();
                r += envelope / tilt * phase.sin();
            }
            left[i] = l as f32;
            right[i] = r as f32;
        }
        let config = crate::survey::SurveyConfig::default();
        let note_config = config.note_config(220.0).unwrap();
        let drift = balance_drift(
            &left,
            &right,
            220.0,
            sample_rate,
            &note_config,
            &DirectivityConfig {
                min_partials: 2,
                ..DirectivityConfig::default()
            },
        )
        .unwrap();
        // The channels tilt against each other at 0.35 nepers per second
        // each, so the balance moves 2 * 8.686 * 0.35 dB per second, over the
        // 1.7 s between the two frames.
        let expected = 2.0 * 8.685_889 * 0.35 * (2.0 - 0.3);
        assert!(
            (drift.drift_db - expected).abs() < 1.5,
            "{drift:?} against {expected:.2} dB"
        );
    }
}
