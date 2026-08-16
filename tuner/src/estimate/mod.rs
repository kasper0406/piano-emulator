//! The estimators: pure functions from measured partial trajectories to the
//! numbers a preset is made of.
//!
//! Each one inverts one part of the model `SPEC.md` describes, and each is
//! unit-tested against synthetic trajectories with known parameters — see
//! `TUNING.md`'s self-calibration gate for what that does and does not prove.
//! The chain, and the order the estimators must run in, is:
//!
//! ```text
//! trajectories ─┬─> inharmonic::fit  ──────────────> f0, B, B4
//!               ├─> decay::fit_decays ─────────────> sigma(f), polarization split
//!               │        │
//!               │        ├─> unison::estimate ─────> detune
//!               │        │        └─> spread::note_spread -> per-string sigma
//!               │        └─> excitation spectrum
//!               │                 ├─> strike::fit ─> strike position, contact width
//!               │                 └─> hammer::fit ─> K, p, mass, velocity map
//!               └─> (across notes) compass::interpolate -> all 88 keys
//!
//! recordings of the action ──> noise::fit_noise ───> the [noise] section
//! stereo recordings of notes ─> directivity::balance_drift -> pan spread
//! release resonances ────────> duplex::residual_modes ──> notes.duplex
//! per-partial beat depth/rate > motion::fit_false_beat -> notes.false_beat
//!         the same, across velocities > motion::fit_strike_direction
//! engine renders (stage 2) ──> halo::refine ───────────> coupling, [voicing.bridge]
//! ```
//!
//! Nothing here reads audio or does a transform; everything reads
//! [`NoteTrajectories`](crate::trajectory::NoteTrajectories), or — for
//! [`noise`], whose material is a recording of the mechanism and has no
//! partials to track — measurements a caller has already taken from one. That
//! is what makes the estimators cheap to iterate on: the expensive STFT pass is
//! cached to disk once and the fits run against it in milliseconds.

pub mod attack;
pub mod brilliance;
pub mod chain;
pub mod compass;
pub mod damper;
pub mod decay;
pub mod directivity;
pub mod duplex;
pub mod halo;
pub mod hammer;
pub mod inharmonic;
pub mod melody;
pub mod motion;
pub mod noise;
pub mod shaping;
pub mod spread;
pub mod strike;
pub mod tail;
pub mod texture;
pub mod unison;

pub use attack::{fit_strike, residual_metrics, AttackConfig, AttackResidual, StrikeFitReport};
pub use compass::{interpolate_keys, CompassCurve};
pub use damper::{band_release, tail_decay_s, DamperConfig, DamperLine};
pub use directivity::{
    balance_drift, pan_spread_for_drift, pan_spread_table, BalanceDrift, DirectivityConfig,
    KeyDriftLine,
};
pub use duplex::{duplex_row, residual_modes, DuplexConfig, ResidualMode};
pub use motion::{
    fit_false_beat, fit_strike_direction, strike_direction_for, Companion, FalseBeatFit,
    FalseBeatVerdict, MotionConfig, StrikeDirectionFit, SwingLine, VelocityCell,
};
pub use halo::{
    between_partials, peaks_from_body_modes, refine as refine_halo, BetweenPartials, HaloConfig,
    HaloError, HaloTarget, HaloVoicing,
};
pub use decay::{DecayConfig, DecayCurve, DecayFit, DecayReport, Exponential, PolarizationSplit};
pub use hammer::{
    contact_pulse, fit_hammer, fit_velocity_map, ContactConfig, FeltParams, ForcePulse, HammerConfig,
    HammerFit, LayerSpectrum, SpectrumPoint, SpectrumWeighting, VelocityMap,
};
pub use inharmonic::{fit_inharmonic, BandRatio, InharmonicConfig, InharmonicFit};
pub use noise::{fit_noise, EventMetrics, MechanismMeasurements, NoiseConfig};
pub use shaping::{
    fit_note as fit_shaping, measured_deepest, partial_gains, partial_sigma_scale, CombLine,
    DecaySplit, EngineComb, NoteShaping, ShapingConfig,
};
pub use spread::{note_spread, spread_from_drift, NoteSpread, SigmaSpread, SpreadConfig};
pub use strike::{contact_taper, fit_strike_position, StrikeConfig, StrikeFit};
pub use unison::{estimate_unison, BeatEstimate, UnisonConfig, UnisonEstimate};

use crate::trajectory::NoteTrajectories;

/// The amplitude below which a track is not a partial of this note.
///
/// The tracker looks for a partial everywhere the seed model predicts one, all
/// the way up to the Nyquist limit, but a real note runs out of partials long
/// before that — and what the tracker finds above the last one is the noise
/// floor's own peaks, which have frequencies and envelopes and would be fitted
/// as if they were partials. Every estimator that reads a whole note therefore
/// works `level_db` down from the loudest partial and no further.
pub fn level_floor(trajectories: &NoteTrajectories, level_db: f64) -> f64 {
    let loudest = trajectories
        .tracks
        .iter()
        .filter_map(|track| track.peak())
        .map(|peak| peak.amplitude)
        .fold(0.0, f64::max);
    loudest * 10f64.powf(-level_db / 20.0)
}

/// Which part of a recording an envelope fit is allowed to look at.
///
/// Two different instants matter and confusing them is a factor-of-two error in
/// every amplitude the estimators extrapolate back to the strike:
///
/// * `onset_s` is the strike — the origin every fitted envelope is expressed
///   against, so that `a(0)` is the excitation the hammer delivered.
/// * `start_s` is the first measurement that may be used. A frame is
///   timestamped at the centre of its window, so a frame centred less than half
///   a window after the strike measured part of the silence before it and reads
///   too low. Fitting starts at the first window that lies entirely after the
///   strike.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FitSpan {
    pub onset_s: f64,
    pub start_s: f64,
}

impl FitSpan {
    pub fn new(onset_s: f64, start_s: f64) -> Self {
        Self { onset_s, start_s }
    }

    pub fn from_trajectories(trajectories: &NoteTrajectories) -> Self {
        Self {
            onset_s: trajectories.onset_s,
            start_s: trajectories.onset_s + 0.5 * trajectories.window_s,
        }
    }
}
