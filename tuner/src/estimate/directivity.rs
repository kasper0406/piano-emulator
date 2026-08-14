//! Where a note's two polarizations sit in the stereo image, from how far its
//! balance moves while it decays.
//!
//! `TUNING_REPORT.md` §5: the median per-partial left-minus-right level of a
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
/// 72, 84 and 96 at velocity 90 on `presets/salamander-c5.toml`.
pub const DRIFT_PER_SPREAD_DB: f64 = 8.3;

/// Median drift the same measurement returns at a spread of zero: not zero,
/// because a note's partials do not all decay at the same rate and the diffuse
/// field is not a constant either. Subtracted before the inversion, so a
/// recording that drifts no more than the engine already does asks for no
/// spread at all.
pub const DRIFT_AT_ZERO_DB: f64 = 0.32;

/// The engine's ceiling on the parameter (`soundboard::MAX_PAN_SPREAD`).
pub const MAX_PAN_SPREAD: f64 = 0.4;

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
