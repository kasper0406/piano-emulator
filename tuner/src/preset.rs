//! The preset file the estimators write.
//!
//! This is the tuner's end of `engine/src/preset.rs`: the same schema, written
//! and read independently. The two crates do not share the type, on purpose —
//! the tuner does not depend on the engine (`DECISIONS.md` item 57), and the
//! file is the interface between them. What keeps them honest is a test that
//! reads the engine's own `presets/default.toml` through this module and writes
//! it back byte for byte: a schema drift on either side breaks it immediately.
//!
//! Numbers are `f32` here for the same reason they are `f32` there — that is
//! the precision the engine computes in, and rounding at the file boundary
//! rather than at load time means what a preset says is what gets played. They
//! are written as the shortest decimal that reads back as the same `f32`, so a
//! table of estimates stays readable by the human who has to sanity-check it.
//!
//! # Building a preset from estimates
//!
//! [`PresetBuilder`] takes a base preset — normally the hand-tuned default,
//! which supplies everything stage 1 cannot measure from isolated notes
//! (soundboard, coupling, damper profile) — plus one [`NoteEstimate`] per
//! measured note, and fills all 88 keys by monotone-cubic interpolation across
//! the compass. Measured notes keep their own values exactly: the interpolant
//! passes through its data.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::estimate::compass::CompassCurve;
use crate::estimate::decay::{DecayCurve, DecayReport, PolarizationSplit};
use crate::estimate::hammer::{HammerFit, VelocityMap};
use crate::estimate::inharmonic::InharmonicFit;
use crate::estimate::strike::StrikeFit;
use crate::estimate::unison::UnisonEstimate;
use crate::response::{duplex_magnitude, BridgeResponse};

/// Keys on the instrument, A0 (MIDI 21) to C8 (MIDI 108).
pub const NUM_KEYS: usize = 88;
pub const LOWEST_KEY: u8 = 21;
pub const HIGHEST_KEY: u8 = 108;
/// Most strings in a unison group.
pub const MAX_UNISON: usize = 3;
/// Ceilings on the two feedback couplings, mirroring `engine::string::
/// MAX_UNISON_COUPLING` and `engine::resonance::MAX_COUPLING`. Both are loop
/// gains — a string's own bridge force returns to its excitation one block
/// later, through its unison siblings and through the resonance bus — so a
/// value that reaches unity sustains itself, and the engine refuses one.
pub const MAX_UNISON_COUPLING: f32 = 0.05;
pub const MAX_RESONANCE_COUPLING: f32 = 0.05;
/// Widest hammer contact a preset may declare, as a fraction of the speaking
/// length (`engine::string::MAX_CONTACT_WIDTH`).
pub const MAX_CONTACT_WIDTH: f32 = 0.05;
/// Bounds on a per-string decay multiplier (`engine::string::{MIN,MAX}_SIGMA_SCALE`).
pub const MIN_SIGMA_SCALE: f32 = 0.5;
pub const MAX_SIGMA_SCALE: f32 = 2.0;
/// Deepest a preset may soften the strike comb's nulls, as a fraction of the
/// comb's crest (`engine::string::MAX_COMB_FLOOR`).
pub const MAX_COMB_FLOOR: f32 = 0.5;
/// Bounds on a per-partial excitation gain — a factor of twenty either way
/// (`engine::string::{MIN,MAX}_PARTIAL_GAIN`).
///
/// Widened from ±20 dB when the field's semantics changed: it is the **full**
/// measured ratio of a recorded partial to the engine's own prediction of it,
/// not the roughness residual left after a smooth envelope has been divided out,
/// so it now carries the engine's envelope error as well as the roughness — 7.5
/// dB of tilt over C4's first four partials on its own (`DECISIONS.md` 231).
pub const MIN_PARTIAL_GAIN: f32 = 0.05;
pub const MAX_PARTIAL_GAIN: f32 = 20.0;
/// Bounds on a `notes.false_beat` row (`engine::preset`). A within-string split
/// is a defect of one wire at one partial; the rate band is the one the
/// mechanism was measured in (0.74–1.48 Hz at C4 and A2, 2.22–5.19 at C6) and
/// the level band runs from inaudible (−40 dB, 0.17 dB of beat depth, low
/// enough to reach the fundamentals whose depth is under a decibel and whose
/// frequency still moves a cent — `DECISIONS.md` 249) up to two planes of equal
/// strength.
pub const MAX_FALSE_BEATS_PER_KEY: usize = 8;
pub const MIN_FALSE_BEAT_HZ: f32 = 0.2;
pub const MAX_FALSE_BEAT_HZ: f32 = 3.0;
pub const MIN_FALSE_BEAT_DB: f32 = -40.0;
pub const MAX_FALSE_BEAT_DB: f32 = 0.0;
/// Bounds on `[voicing.strike_direction]` (`engine::preset`): the velocity law
/// for the *direction* of the hammer's excitation vector, which is the one place
/// velocity enters a linear string model.
pub const MAX_STRIKE_DIRECTION_DB: f32 = 12.0;
pub const MAX_SHARE_TILT: f32 = 0.2;
/// Bounds on a per-partial correction to the *fitted* decay rate
/// (`engine::string::{MIN,MAX}_PARTIAL_SIGMA_SCALE`).
pub const MIN_PARTIAL_SIGMA_SCALE: f32 = 0.25;
pub const MAX_PARTIAL_SIGMA_SCALE: f32 = 4.0;
/// Bounds on `[noise.strike]`, mirroring `engine::preset`. The bandwidth is the
/// one field the four action events do not have: their 2 kHz ceiling is
/// Askenfelt's structure-borne measurement of the *action*, and a hammer meeting
/// a string is not structure-borne.
pub const MIN_STRIKE_BANDWIDTH_HZ: f32 = 200.0;
pub const MAX_STRIKE_BANDWIDTH_HZ: f32 = 8_000.0;
pub const MIN_STRIKE_DECAY_S: f32 = 0.02;
pub const MAX_STRIKE_DECAY_S: f32 = 0.3;
/// Velocity at which `[noise.strike]`'s tabulated level is the level played
/// (`engine::preset::NOMINAL_STRIKE_VELOCITY`).
pub const NOMINAL_STRIKE_VELOCITY: u8 = 90;
/// The level at which the engine does not play an event at all
/// (`engine::preset::SILENT_LEVEL_DB`).
pub const SILENT_LEVEL_DB: f32 = -200.0;
/// Largest displacement between the two polarizations of one key, either side
/// of that key's pan (`engine::soundboard::MAX_PAN_SPREAD`). The engine's
/// `MAX_PAN + MAX_PAN_SPREAD` is 1, so at the ceiling the outer polarization of
/// the outermost key sits exactly hard left or hard right.
pub const MAX_PAN_SPREAD: f32 = 0.4;
/// Shortest and longest a mechanism event may last, seconds
/// (`engine::preset::{MIN,MAX}_NOISE_DECAY_S`).
pub const MIN_NOISE_DECAY_S: f32 = 0.01;
pub const MAX_NOISE_DECAY_S: f32 = 10.0;
/// Widest unison spread a key may declare, in cents
/// (`engine::preset::MAX_DETUNE_CENTS`).
///
/// The spread multiplies the *top* of the partial series, so it spends the band
/// between the partial cap and Nyquist; a whole semitone is also not one note.
pub const MAX_DETUNE_CENTS: f32 = 100.0;
/// Slowest decay rate a mode may be given, 1/s (`engine::string::MIN_MODE_SIGMA`)
/// and therefore the floor on `notes.sigma0`.
///
/// Arithmetic, not musical: the resonator's pole radius is `exp(-sigma/48000)`
/// in `f32`, which rounds to exactly one under about 5.7e-3. A T60 of 345 s is
/// longer than any piano string rings.
pub const MIN_MODE_SIGMA: f32 = 0.02;
/// Partials the engine builds a bank from: the cap, and the fraction of the
/// sample rate above which a partial is not admitted (`engine::types`).
pub const MAX_PARTIALS: u32 = 80;
const MAX_PARTIAL_RATIO: f64 = 0.45;
const SAMPLE_RATE: f64 = 48_000.0;

/// Index into the per-note tables, or `None` off the keyboard.
pub fn key_index(key: u8) -> Option<usize> {
    (LOWEST_KEY..=HIGHEST_KEY)
        .contains(&key)
        .then(|| usize::from(key - LOWEST_KEY))
}

pub fn index_to_key(index: usize) -> u8 {
    LOWEST_KEY + index as u8
}

/// Equal-tempered pitch of a key, A4 = 440 Hz. The tuner writes measured
/// pitches into a preset; this is only the reference the stretch of a real
/// instrument is expressed against.
pub fn equal_temperament(key: u8) -> f64 {
    440.0 * ((f64::from(key) - 69.0) / 12.0).exp2()
}

// ------------------------------------------------------------- the schema

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Preset {
    pub name: String,
    pub description: String,
    pub voicing: Voicing,
    pub hammer: HammerVoicing,
    pub soundboard: SoundboardVoicing,
    pub notes: NoteTables,
    /// The action's own sounds. Absent means the levels `TUNING_REPORT.md` §5
    /// measured — the one defaulted field here whose default is not neutral,
    /// exactly as in the engine.
    #[serde(default, skip_serializing_if = "is_default_noise")]
    pub noise: NoiseTables,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Voicing {
    #[serde(serialize_with = "short::scalar")]
    pub excitation_scale: f32,
    /// Input gain of the horizontal polarization relative to the vertical one.
    /// Estimated: the level ratio of the two exponentials in a partial's decay.
    #[serde(serialize_with = "short::scalar")]
    pub horizontal_gain_db: f32,
    /// Horizontal decay rate as a fraction of the vertical one. Estimated: the
    /// rate ratio of the same two exponentials.
    #[serde(serialize_with = "short::scalar")]
    pub horizontal_decay_ratio: f32,
    #[serde(serialize_with = "short::list")]
    pub horizontal_offset_hz: Vec<f32>,
    #[serde(serialize_with = "short::scalar")]
    pub unison_coupling: f32,
    #[serde(serialize_with = "short::scalar")]
    pub resonance_coupling: f32,
    /// How far apart the two polarizations sit in the stereo image, as a pan
    /// displacement either side of the key's own position. Because the two
    /// decay at very different rates, a nonzero spread makes a single note's
    /// balance *move* while it rings — the 1.2–6.2 dB of drift
    /// `TUNING_REPORT.md` §5 measured on the recordings against 0.02–0.14 dB on
    /// the engine's own renders. Zero, the default, is the old mono-per-key
    /// image.
    #[serde(
        default,
        skip_serializing_if = "is_zero",
        serialize_with = "short::scalar"
    )]
    pub polarization_pan_spread: f32,
    /// The bridge admittance `B(f)` on the sympathetic bus's drive path.
    /// Absent — the default — is the unity filter, the spectrally flat halo the
    /// engine has always had. See [`BridgeVoicing`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bridge: Option<BridgeVoicing>,
    pub unison_layout: Vec<UnisonLayout>,
    /// Decay-rate multipliers for the individual strings of a unison, one row
    /// per group size exactly like [`Voicing::unison_layout`]. Estimated from
    /// the drift of a beating composite partial's measured frequency
    /// (`TUNING_REPORT.md` §6): strings that are mistuned *and* decay at
    /// different rates move that frequency as the survivor takes over, and one
    /// shared damping law cannot. All ones is the shared law.
    #[serde(
        default = "unity_sigma_scale",
        skip_serializing_if = "is_unity_sigma_scale"
    )]
    pub unison_sigma_scale: Vec<UnisonSigmaScale>,
    pub damper_weight: Vec<DamperAnchor>,
    /// How the hammer's blow changes *direction* with velocity: the
    /// vertical/horizontal ratio at the two ends of the velocity range, and how
    /// far the group's per-string share asymmetry tilts between them. Absent —
    /// the default — is the fitted, velocity-independent strike vector, which is
    /// what every preset written so far has. Nothing fits it yet
    /// (`FUNDAMENTALS.md` §7.7's last row).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strike_direction: Option<StrikeDirection>,
}

/// The velocity law for the strike vector's direction. See
/// `engine::preset::StrikeDirection` for what each field means and why the model
/// has nowhere else to put a velocity dependence.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StrikeDirection {
    /// Offset on [`Voicing::horizontal_gain_db`] at MIDI velocity 1, in dB.
    #[serde(serialize_with = "short::scalar")]
    pub vh_db_at_pp: f32,
    /// The same offset at MIDI velocity 127.
    #[serde(serialize_with = "short::scalar")]
    pub vh_db_at_ff: f32,
    /// Full pianissimo-to-fortissimo swing of the group's share asymmetry, as a
    /// fraction of that asymmetry, taken about mid-velocity.
    #[serde(serialize_with = "short::scalar")]
    pub share_tilt: f32,
}

/// One within-string split: the two transverse planes of **one wire** at
/// genuinely different frequencies, which is Capleton's false beat (JASA 115(2),
/// 2004) and the only mechanism `FUNDAMENTALS.md` §7.4's measurements support
/// for the companion each recorded mid and low partial carries — 4–7 dB down,
/// 0.7–1.5 Hz away, at a spacing that does **not** scale with the partial
/// number. See `engine::preset::FalseBeat`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FalseBeat {
    /// Which partial of this key is split, 1-based; at most one entry per
    /// partial.
    pub k: u16,
    /// The split, in hertz.
    #[serde(serialize_with = "short::scalar")]
    pub hz: f32,
    /// How loud the companion stands, in dB relative to the loudest mode of the
    /// same partial — the quantity a measured beat depth inverts to.
    #[serde(serialize_with = "short::scalar")]
    pub db: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UnisonLayout {
    #[serde(serialize_with = "short::list")]
    pub detune: Vec<f32>,
    #[serde(serialize_with = "short::list")]
    pub share: Vec<f32>,
}

/// The decay-rate multipliers of one size of unison group.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UnisonSigmaScale {
    /// One multiplier per string, applied to both of that string's
    /// polarizations. The row averages to 1, so it redistributes the note's
    /// damping rather than adding a second decay control beside `notes.sigma0`.
    #[serde(serialize_with = "short::list")]
    pub scale: Vec<f32>,
}

/// The bridge's driving-point admittance, as the shape of one filter.
///
/// The tuner's copy of `engine::preset::BridgeVoicing`. A string terminates on
/// a bridge with a complex mobility `Y(f)`, not on a node, and because every
/// string shares one board the mobility is what makes sympathetic coupling
/// frequency-selective. `PHYSICS.md` §4 gives the shape to fit: mean mobility
/// ≈ 1.3e-3 s/kg over 100–1000 Hz with ±10–15 dB of fluctuation, sharp and
/// well-separated peaks below ~500 Hz, falling in the treble, with the plate/
/// rib transition at Ege & Boutillon's `f_lim ≈ 1.1 kHz`. Hence the split into
/// a smooth [`backbone`](BridgeVoicing::backbone) and discrete
/// [`peaks`](BridgeVoicing::peaks) rather than one long modal bank.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BridgeVoicing {
    /// Mean mobility, as gains at frequencies interpolated in log `f`.
    /// 2 to [`MAX_BRIDGE_ANCHORS`] anchors, strictly ascending in `hz`.
    pub backbone: Vec<BridgeAnchor>,
    /// Discrete bridge resonances, at most [`MAX_BRIDGE_PEAKS`] of them.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub peaks: Vec<BridgePeak>,
    /// Share of each partial's decay rate that is loss into the board, and so
    /// follows the *fluctuation* of the admittance — the peaks alone, since the
    /// mean is already inside the fitted `notes.sigma0` and `notes.sigma1`.
    /// `sigma_k <- sigma_k · (1 + share · (|P(f_k)| − 1))`, clamped to a factor
    /// of four either way by the engine. Zero (the default, and an absent
    /// field) leaves every string bit for bit what it was.
    #[serde(default, skip_serializing_if = "is_zero", serialize_with = "short::scalar")]
    pub radiated_share: f32,
}

/// One anchor of the admittance backbone.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BridgeAnchor {
    #[serde(serialize_with = "short::scalar")]
    pub hz: f32,
    #[serde(serialize_with = "short::scalar")]
    pub gain_db: f32,
}

/// One discrete bridge resonance.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BridgePeak {
    #[serde(serialize_with = "short::scalar")]
    pub hz: f32,
    #[serde(serialize_with = "short::scalar")]
    pub q: f32,
    /// Peak gain over the backbone, dB. Negative is an anti-resonance.
    #[serde(serialize_with = "short::scalar")]
    pub gain_db: f32,
}

/// Bounds on a `[voicing.bridge]` section, mirroring `engine::preset`. None of
/// them is what makes the filter *safe*: that is [`Preset::validate`]'s
/// loop-gain check against the response the engine will actually realise
/// ([`crate::response::BridgeResponse`]).
pub const MAX_BRIDGE_ANCHORS: usize = 24;
pub const MAX_BRIDGE_PEAKS: usize = 40;
pub const MIN_BRIDGE_HZ: f32 = 20.0;
pub const MAX_BRIDGE_HZ: f32 = 16_000.0;
pub const MIN_BRIDGE_GAIN_DB: f32 = -40.0;
pub const MAX_BRIDGE_GAIN_DB: f32 = 20.0;
pub const MIN_BRIDGE_Q: f32 = 0.5;
pub const MAX_BRIDGE_Q: f32 = 50.0;
/// Largest share of a partial's decay the admittance may be given
/// (`engine::preset::MAX_RADIATED_SHARE`).
pub const MAX_RADIATED_SHARE: f32 = 0.9;
/// Largest `resonance_coupling · max|B|` a preset may ask for
/// (`engine::resonance::MAX_BRIDGE_LOOP_GAIN`), and largest
/// `resonance_coupling · max|B| · (Σ|D| + max|D|)`
/// (`engine::duplex::MAX_DUPLEX_LOOP_GAIN`).
pub const MAX_BRIDGE_LOOP_GAIN: f32 = 0.25;
pub const MAX_DUPLEX_LOOP_GAIN: f32 = 0.25;

/// One duplex or aliquot segment of a key, as a resonance.
///
/// The tuner's copy of `engine::preset::DuplexMode`. `hz` is a **measured**
/// frequency and never `k·f0`: Öberg & Askenfelt found real rear-duplex tuning
/// sharp of nominal by an average approaching +50 cents, with ~25 cents of
/// scatter inside a single trichord, and that scatter is the sound
/// (`PHYSICS.md` §3). `gain_db` is the segment's response *at its own
/// frequency* per unit of the bridge force driving it, normalised so that
/// `t60_s` changes how long it rings without changing how loud it is — which is
/// what makes the two estimable separately, since a peak level and a decay time
/// are separate measurements.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DuplexMode {
    #[serde(serialize_with = "short::scalar")]
    pub hz: f32,
    #[serde(serialize_with = "short::scalar")]
    pub gain_db: f32,
    #[serde(serialize_with = "short::scalar")]
    pub t60_s: f32,
}

/// Bounds on a `notes.duplex` row, mirroring `engine::preset`.
pub const MAX_DUPLEX_MODES: usize = 6;
pub const MIN_DUPLEX_HZ: f32 = 200.0;
pub const MAX_DUPLEX_HZ: f32 = 18_000.0;
pub const MIN_DUPLEX_GAIN_DB: f32 = -60.0;
pub const MAX_DUPLEX_GAIN_DB: f32 = 6.0;
pub const MIN_DUPLEX_T60_S: f32 = 0.05;
pub const MAX_DUPLEX_T60_S: f32 = 3.0;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DamperAnchor {
    #[serde(serialize_with = "short::scalar")]
    pub hz: f32,
    #[serde(serialize_with = "short::scalar")]
    pub weight: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HammerVoicing {
    /// Hammer speed at MIDI velocity 1 and 127, m/s. Estimated jointly with
    /// the felt, from the excitation spectra of a note's velocity layers.
    #[serde(serialize_with = "short::scalar")]
    pub velocity_min: f32,
    #[serde(serialize_with = "short::scalar")]
    pub velocity_max: f32,
    #[serde(serialize_with = "short::scalar")]
    pub felt_hysteresis: f32,
    #[serde(serialize_with = "short::scalar")]
    pub una_corda_stiffness: f32,
    #[serde(serialize_with = "short::scalar")]
    pub reflection_gain: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SoundboardVoicing {
    #[serde(serialize_with = "short::scalar")]
    pub board_mix: f32,
    #[serde(serialize_with = "short::scalar")]
    pub body_mix: f32,
    #[serde(serialize_with = "short::scalar")]
    pub board_level: f32,
    #[serde(serialize_with = "short::scalar")]
    pub shelf_hz: f32,
    #[serde(serialize_with = "short::scalar")]
    pub shelf_gain_db: f32,
    #[serde(serialize_with = "short::scalar")]
    pub fdn_t60_lf: f32,
    #[serde(serialize_with = "short::scalar")]
    pub fdn_t60_hf: f32,
    #[serde(serialize_with = "short::scalar")]
    pub fdn_hf_hz: f32,
    pub body_modes: Vec<BodyMode>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BodyMode {
    #[serde(serialize_with = "short::scalar")]
    pub hz: f32,
    #[serde(serialize_with = "short::scalar")]
    pub q: f32,
    #[serde(serialize_with = "short::scalar")]
    pub gain: f32,
}

/// The per-note tables, 88 entries each, indexed by [`key_index`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NoteTables {
    /// The tuning. Equal temperament in the default preset; the measured
    /// stretch of a real instrument in an estimated one.
    #[serde(serialize_with = "short::list")]
    pub f0_hz: Vec<f32>,
    #[serde(serialize_with = "short::list")]
    pub inharmonicity_b: Vec<f32>,
    /// Fourth-order coefficient B4 of `f_k = k f0 sqrt(1 + B k^2 + B4 k^4)`,
    /// **signed**: a wound bass string's series curves one way and the short
    /// wound tenor strings' the other (`TUNING_REPORT.md` §1). Absent means
    /// zero, which is the two-parameter law exactly.
    #[serde(
        default = "zero_table",
        skip_serializing_if = "is_zero_table",
        serialize_with = "short::list"
    )]
    pub inharmonicity_b4: Vec<f32>,
    #[serde(serialize_with = "short::list")]
    pub strike_position: Vec<f32>,
    /// Width of the hammer's contact with the string, as a fraction of the
    /// speaking length. Absent means zero, which is the point force the strike
    /// comb alone describes.
    #[serde(
        default = "zero_table",
        skip_serializing_if = "is_zero_table",
        serialize_with = "short::list"
    )]
    pub contact_width: Vec<f32>,
    /// Soft floor under the strike comb's nulls, one per key, as a fraction of
    /// the comb's crest: the excitation magnitude of partial `k` becomes
    /// `sqrt(sin^2(k pi x) + floor^2)` before the contact taper and the
    /// per-partial gain. Absent means zero, which is the bare comb.
    ///
    /// `sin(k pi x)` has exact zeros and a hammer with width on a stiff string
    /// terminated on a bridge does not: the engine's worst partial is measurably
    /// *at* those zeros, 42 dB down, where the recording's deepest partial
    /// anywhere is 9.3 to 17.7 dB down (`renders/timbre-ladder/ANALYSIS.md`
    /// §4a). [`estimate::shaping`](crate::estimate::shaping) fits it, and fits
    /// it *before* [`NoteTables::partial_gains`] so that the two cannot
    /// double-count the same null.
    #[serde(
        default = "zero_table",
        skip_serializing_if = "is_zero_table",
        serialize_with = "short::list"
    )]
    pub comb_floor: Vec<f32>,
    /// Per-partial linear gain multipliers on the excitation comb, one row per
    /// key, 1-based in the partial index.
    ///
    /// `TUNING_REPORT.md` §3's backlog item 6: the measured excitation is
    /// 5–10 dB rougher than any smooth envelope times `sin(k pi x)` and the
    /// roughness is not shared between notes at the same frequency, so it cannot
    /// be a bridge curve. Absent — the default — is one everywhere; a row may be
    /// **short** (the tracker measures as far up the series as the recording
    /// lets it) and everything past its end is exactly 1.0; a row *longer* than
    /// the key's partial count is refused, naming the key.
    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        serialize_with = "short::table"
    )]
    pub partial_gains: Vec<Vec<f32>>,
    /// Per-partial multipliers on the note's fitted `sigma(f)` law, with the
    /// same shape rules as [`NoteTables::partial_gains`]. Applied before the
    /// polarization split, so both banks and the damper profile follow it
    /// (`TUNING_REPORT.md` §2: the envelope law describes a real partial to
    /// about 4 dB whatever produced it, and the residual is per partial).
    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        serialize_with = "short::table"
    )]
    pub partial_sigma_scale: Vec<Vec<f32>>,
    #[serde(serialize_with = "short::list")]
    pub sigma0: Vec<f32>,
    #[serde(serialize_with = "short::list")]
    pub sigma1: Vec<f32>,
    pub unison: Vec<u8>,
    #[serde(serialize_with = "short::list")]
    pub detune_cents: Vec<f32>,
    #[serde(serialize_with = "short::list")]
    pub impedance: Vec<f32>,
    #[serde(serialize_with = "short::list")]
    pub damper_sigma: Vec<f32>,
    #[serde(serialize_with = "short::list")]
    pub bridge_gain: Vec<f32>,
    #[serde(serialize_with = "short::list")]
    pub hammer_mass: Vec<f32>,
    #[serde(serialize_with = "short::list")]
    pub hammer_stiffness: Vec<f32>,
    #[serde(serialize_with = "short::list")]
    pub hammer_exponent: Vec<f32>,
    /// The key's duplex and aliquot segments, up to [`MAX_DUPLEX_MODES`] per
    /// key. Empty — the default, and absent from the file — is the instrument
    /// with no segments at all. A table that is present has one row per key,
    /// and a row may be empty: `estimate::duplex` writes one only where the
    /// release recordings actually carry a long-lived residual, and an invented
    /// row would be a harmonic ratio pretending to be a measurement.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub duplex: Vec<Vec<DuplexMode>>,
    /// Within-string splits: which partials of which key beat against
    /// themselves, and how hard ([`FalseBeat`]). Empty — the default, and absent
    /// from the file — is the instrument with no false beats at all. A table
    /// that is present has one row per key, and a row may be empty: a
    /// well-drawn wire in good condition has no measurable split, and most of
    /// them are.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub false_beat: Vec<Vec<FalseBeat>>,
    /// Per-key override of `voicing.polarization_pan_spread`. Empty — the
    /// default — means the global scalar applies to the whole compass. The
    /// compass does not want one number: at the engine's ceiling of 0.4 the
    /// drift it produces is 0.24 dB at A0 and 8.67 dB at C5 against the
    /// recordings' 1.24 and 5.33 (`TUNING_REPORT.md` §5).
    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        serialize_with = "short::list"
    )]
    pub pan_spread: Vec<f32>,
}

/// The four mechanism events, and what each of them sounds like.
///
/// The estimator's end of `engine::preset::NoiseTables`. `TUNING_REPORT.md` §5
/// measured the table the engine defaults to; [`estimate::noise`](crate::estimate::noise)
/// re-fits it from a library's own release and pedal recordings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NoiseTables {
    pub key_off: EventNoise,
    pub damper_lift: EventNoise,
    pub pedal_down: EventNoise,
    pub pedal_up: EventNoise,
    /// The hammer arriving: the one mechanism event that happens *under* a note
    /// rather than beside one, and the only one that is broadband well past the
    /// action's 2 kHz ceiling — hence its own [`StrikeNoise::bandwidth_hz`].
    ///
    /// Absent — the default — is **silence**, and silence is the neutral value
    /// here where it is not for the other four: nothing in any library isolates
    /// a blow, so a level written by default would be a guess with the authority
    /// of a measurement. [`estimate::attack`](crate::estimate::attack) measures
    /// it from the onset residual of the struck-note recordings themselves.
    #[serde(default, skip_serializing_if = "is_silent_strike")]
    pub strike: StrikeNoise,
}

/// The hammer's own noise: how loud, how long, what colour, and how far up.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StrikeNoise {
    /// Spectral centre of the burst, Hz.
    #[serde(serialize_with = "short::scalar")]
    pub centroid_hz: f32,
    /// Time to fall 40 dB, seconds. Short: the residual is an attack, and the
    /// window it is measured in is 30 to 150 ms.
    #[serde(serialize_with = "short::scalar")]
    pub decay_s: f32,
    /// Upper band limit of the burst, Hz — the field the other four events do
    /// not have.
    #[serde(serialize_with = "short::scalar")]
    pub bandwidth_hz: f32,
    /// How far the level travels, in dB, over the full velocity range, through
    /// the tabulated level at velocity [`NOMINAL_STRIKE_VELOCITY`].
    #[serde(serialize_with = "short::scalar")]
    pub velocity_db: f32,
    /// Peak level, in dB relative to a velocity-90 strike of the same key,
    /// anchored at the keys it was measured at — the same convention, and the
    /// same output-referenced calibration, as the other four events.
    pub level_db: Vec<NoiseAnchor>,
}

/// Silence, which is what a preset that does not describe the hammer's noise
/// asks for. The shape fields carry a plausible hammer so that a preset which
/// writes only `level_db` gets one; the level is −200 dB, which the engine
/// refuses to render at all, so the neutral value is bit-exact silence.
impl Default for StrikeNoise {
    fn default() -> StrikeNoise {
        StrikeNoise {
            centroid_hz: 1_200.0,
            decay_s: 0.05,
            bandwidth_hz: 6_000.0,
            velocity_db: 24.0,
            level_db: vec![NoiseAnchor {
                key: LOWEST_KEY,
                db: SILENT_LEVEL_DB,
            }],
        }
    }
}

fn is_silent_strike(strike: &StrikeNoise) -> bool {
    *strike == StrikeNoise::default()
}

/// One mechanism event: how loud, how long, and what colour.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventNoise {
    /// Spectral centre of the burst, Hz.
    #[serde(serialize_with = "short::scalar")]
    pub centroid_hz: f32,
    /// Time to fall 40 dB, seconds.
    #[serde(serialize_with = "short::scalar")]
    pub decay_s: f32,
    /// How far the level travels, in dB, over the event's full drive range:
    /// release velocity for the key events, the fraction of the dampers that
    /// move for the pedal ones.
    #[serde(serialize_with = "short::scalar")]
    pub velocity_db: f32,
    /// Peak level, in dB relative to a velocity-90 strike of the same key,
    /// anchored at the keys it was measured at. A global event carries a single
    /// anchor.
    pub level_db: Vec<NoiseAnchor>,
}

/// One measured point of a mechanism event's level across the compass.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NoiseAnchor {
    pub key: u8,
    #[serde(serialize_with = "short::scalar")]
    pub db: f32,
}

impl NoiseTables {
    /// The four events with their field names, for validation.
    pub fn events(&self) -> [(&'static str, &EventNoise); 4] {
        [
            ("key_off", &self.key_off),
            ("damper_lift", &self.damper_lift),
            ("pedal_down", &self.pedal_down),
            ("pedal_up", &self.pedal_up),
        ]
    }
}

/// The mechanism as `TUNING_REPORT.md` §5 measured it — the engine's own
/// default, duplicated here because the file is the interface between the two
/// crates and a section written at its default must not appear in it.
impl Default for NoiseTables {
    fn default() -> NoiseTables {
        const KEY_OFF: [(u8, f32); 5] = [
            (21, -37.3),
            (57, -30.2),
            (60, -35.4),
            (72, -25.4),
            (96, -33.5),
        ];
        let anchors = |table: &[(u8, f32)], offset: f32| -> Vec<NoiseAnchor> {
            table
                .iter()
                .map(|&(key, db)| NoiseAnchor {
                    key,
                    db: db + offset,
                })
                .collect()
        };
        NoiseTables {
            key_off: EventNoise {
                centroid_hz: 190.0,
                decay_s: 0.24,
                velocity_db: 12.0,
                level_db: anchors(&KEY_OFF, 0.0),
            },
            damper_lift: EventNoise {
                centroid_hz: 300.0,
                decay_s: 0.08,
                velocity_db: 12.0,
                level_db: anchors(&KEY_OFF, -6.0),
            },
            pedal_down: EventNoise {
                centroid_hz: 77.0,
                decay_s: 5.76,
                velocity_db: 6.0,
                level_db: vec![NoiseAnchor {
                    key: LOWEST_KEY,
                    db: -35.8,
                }],
            },
            pedal_up: EventNoise {
                centroid_hz: 187.0,
                decay_s: 0.32,
                velocity_db: 6.0,
                level_db: vec![NoiseAnchor {
                    key: LOWEST_KEY,
                    db: -42.4,
                }],
            },
            strike: StrikeNoise::default(),
        }
    }
}

fn is_default_noise(noise: &NoiseTables) -> bool {
    *noise == NoiseTables::default()
}

/// A per-note table that may be absent from the file, all 88 entries neutral.
fn zero_table() -> Vec<f32> {
    vec![0.0; NUM_KEYS]
}

fn is_zero_table(table: &[f32]) -> bool {
    table.len() == NUM_KEYS && table.iter().all(|&x| x == 0.0)
}

fn is_zero(value: &f32) -> bool {
    *value == 0.0
}

/// The neutral [`Voicing::unison_sigma_scale`]: every string of every group
/// size on the note's own damping law.
pub fn unity_sigma_scale() -> Vec<UnisonSigmaScale> {
    (1..=MAX_UNISON)
        .map(|n| UnisonSigmaScale {
            scale: vec![1.0; n],
        })
        .collect()
}

fn is_unity_sigma_scale(rows: &[UnisonSigmaScale]) -> bool {
    rows.len() == MAX_UNISON
        && rows
            .iter()
            .enumerate()
            .all(|(i, row)| row.scale.len() == i + 1 && row.scale.iter().all(|&s| s == 1.0))
}

impl Preset {
    pub fn load(path: impl AsRef<Path>) -> Result<Preset> {
        Preset::from_toml(&std::fs::read_to_string(path)?)
    }

    pub fn from_toml(text: &str) -> Result<Preset> {
        let preset: Preset = toml::from_str(text)?;
        preset.validate()?;
        Ok(preset)
    }

    pub fn to_toml(&self) -> String {
        toml::to_string_pretty(self).expect("a preset is always serializable")
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        std::fs::write(path, self.to_toml())?;
        Ok(())
    }

    /// The same invariants the engine checks when it loads a preset.
    ///
    /// Duplicated rather than deferred: a pipeline that writes a file the
    /// engine will refuse has failed, and it should say so where the number was
    /// produced rather than hours later at playback.
    pub fn validate(&self) -> Result<()> {
        let n = &self.notes;
        let tables: [(&str, &Vec<f32>, bool); 12] = [
            ("f0_hz", &n.f0_hz, true),
            ("inharmonicity_b", &n.inharmonicity_b, false),
            ("strike_position", &n.strike_position, true),
            ("sigma0", &n.sigma0, true),
            ("sigma1", &n.sigma1, false),
            ("detune_cents", &n.detune_cents, false),
            ("impedance", &n.impedance, true),
            ("damper_sigma", &n.damper_sigma, false),
            ("bridge_gain", &n.bridge_gain, true),
            ("hammer_mass", &n.hammer_mass, true),
            ("hammer_stiffness", &n.hammer_stiffness, true),
            ("hammer_exponent", &n.hammer_exponent, true),
        ];
        for (name, table, positive) in tables {
            if table.len() != NUM_KEYS {
                return Err(Error::Preset(format!(
                    "notes.{name} has {} entries, expected {NUM_KEYS}",
                    table.len()
                )));
            }
            for (i, &v) in table.iter().enumerate() {
                if !v.is_finite() || v < 0.0 || (positive && v == 0.0) {
                    return Err(Error::Preset(format!(
                        "notes.{name}[{i}] is {v}, expected a finite {} number",
                        if positive { "positive" } else { "non-negative" }
                    )));
                }
            }
        }
        if let Some(i) = n.strike_position.iter().position(|&x| x >= 1.0) {
            return Err(Error::Preset(format!(
                "notes.strike_position[{i}] is {}, expected 0 < x < 1",
                n.strike_position[i]
            )));
        }
        // The two tables the loop above could only check the sign of. The
        // spread multiplies every partial's frequency, so it spends the headroom
        // between the partial cap and Nyquist; and `sigma0` is the floor of the
        // whole loss budget the eigensolve's roots come out of, which under the
        // engine's `MIN_MODE_SIGMA` is decided by the solver's own rounding
        // rather than by the fit. Mirrors `engine::preset::Preset::validate`
        // (`DECISIONS.md` 257).
        for (i, &c) in n.detune_cents.iter().enumerate() {
            within(&format!("notes.detune_cents[{i}]"), c, 0.0, MAX_DETUNE_CENTS)?;
        }
        for (i, &s) in n.sigma0.iter().enumerate() {
            if s < MIN_MODE_SIGMA {
                return Err(Error::Preset(format!(
                    "notes.sigma0[{i}] is {s}, expected at least {MIN_MODE_SIGMA} \
                     (a T60 of 345 s)"
                )));
            }
        }
        // The fourth-order inharmonicity is the one signed table — the sign is
        // the finding (`TUNING_REPORT.md` §1) — so entry by entry only
        // finiteness can be checked. What the value has to *do* is checked
        // below, against the series it produces.
        table_length("inharmonicity_b4", n.inharmonicity_b4.len())?;
        for (i, &b4) in n.inharmonicity_b4.iter().enumerate() {
            finite(&format!("notes.inharmonicity_b4[{i}]"), b4)?;
        }
        table_length("contact_width", n.contact_width.len())?;
        for (i, &w) in n.contact_width.iter().enumerate() {
            within(
                &format!("notes.contact_width[{i}]"),
                w,
                0.0,
                MAX_CONTACT_WIDTH,
            )?;
        }
        // Checked here rather than with the two ragged tables below, because the
        // partial-series loop reads it and the ragged tables are checked against
        // the count that loop produces.
        table_length("comb_floor", n.comb_floor.len())?;
        for (i, &floor) in n.comb_floor.iter().enumerate() {
            within(&format!("notes.comb_floor[{i}]"), floor, 0.0, MAX_COMB_FLOOR)?;
        }
        // Either absent — the global spread applies — or a whole compass of
        // them, each inside the range the scalar itself is held to, because
        // each reaches the soundboard as a pan displacement in the same way.
        if !n.pan_spread.is_empty() {
            table_length("pan_spread", n.pan_spread.len())?;
            for (i, &s) in n.pan_spread.iter().enumerate() {
                within(&format!("notes.pan_spread[{i}]"), s, 0.0, MAX_PAN_SPREAD)?;
            }
        }
        if n.unison.len() != NUM_KEYS {
            return Err(Error::Preset(format!(
                "notes.unison has {} entries, expected {NUM_KEYS}",
                n.unison.len()
            )));
        }
        if let Some((i, &u)) = n
            .unison
            .iter()
            .enumerate()
            .find(|(_, &u)| u == 0 || usize::from(u) > MAX_UNISON)
        {
            return Err(Error::Preset(format!(
                "notes.unison[{i}] is {u}, expected 1..={MAX_UNISON}"
            )));
        }

        // `f_k = k f0 sqrt(1 + B k^2 + B4 k^4)` is only a partial layout while
        // the radicand stays positive and the series stays ordered: with `B4`
        // signed and its term growing four times as fast in the exponent, a
        // coefficient that is harmless on the low partials can fold the top of
        // the series back down or take it under the root. Checked over every
        // partial the law could reach up to the Nyquist cap — NOT over
        // `partial_count()`, whose `take_while` stops at the first non-finite
        // frequency and would let a radicand that jumps straight negative
        // truncate the series silently — because the tuner has to refuse the
        // number *where it was produced*.
        // Partials the engine will really build for each key, filled in by the
        // loop below and used by `validate_partial_tables` afterwards: a ragged
        // row is measured against a bank that exists, not against one the layout
        // would have refused.
        let mut partial_counts = [0usize; NUM_KEYS];
        for (i, count) in partial_counts.iter_mut().enumerate() {
            let layout = PartialLayout::of(self, i);
            let limit = (MAX_PARTIAL_RATIO * SAMPLE_RATE) as f32;
            let mut previous = 0.0f32;
            *count = layout.partial_count();
            for k in 1..=MAX_PARTIALS {
                let radicand = layout.radicand(k);
                if !(radicand.is_finite() && radicand > 0.0) {
                    return Err(Error::Preset(format!(
                        "notes.inharmonicity_b[{i}] = {} with inharmonicity_b4[{i}] = {} \
                         puts partial {k} under a root of {radicand}",
                        layout.b, layout.b4
                    )));
                }
                let f = layout.partial_hz(k);
                if !f.is_finite() || f <= previous {
                    return Err(Error::Preset(format!(
                        "notes.inharmonicity_b[{i}] = {} with inharmonicity_b4[{i}] = {} \
                         puts partial {k} at {f} Hz, not above the {previous} Hz before it",
                        layout.b, layout.b4
                    )));
                }
                // The legitimate end of the series: past the cap the engine
                // never builds the partial. Except at `k = 1`, which is not an
                // end: `partial_count` floors at one, so a key whose own
                // fundamental is past the cap still gets a bank — of one
                // partial, outside the band the resonator is defined in.
                if f >= limit {
                    if k == 1 {
                        return Err(Error::Preset(format!(
                            "notes.f0_hz[{i}] = {} puts the key's own fundamental at {f} Hz, \
                             at or past the {limit} Hz cap the partial series stops at",
                            layout.f0
                        )));
                    }
                    break;
                }
                previous = f;
            }
        }

        self.validate_partial_tables(&partial_counts)?;
        self.validate_false_beats(&partial_counts)?;

        let v = &self.voicing;
        positive("voicing.excitation_scale", v.excitation_scale)?;
        positive("voicing.horizontal_decay_ratio", v.horizontal_decay_ratio)?;
        finite("voicing.horizontal_gain_db", v.horizontal_gain_db)?;
        within(
            "voicing.unison_coupling",
            v.unison_coupling,
            0.0,
            MAX_UNISON_COUPLING,
        )?;
        within(
            "voicing.resonance_coupling",
            v.resonance_coupling,
            0.0,
            MAX_RESONANCE_COUPLING,
        )?;
        // A displacement either side of a pan position that already reaches the
        // engine's `MAX_PAN`: the ceiling is what puts the outer polarization of
        // the outermost key exactly hard left or hard right, never past it.
        within(
            "voicing.polarization_pan_spread",
            v.polarization_pan_spread,
            0.0,
            MAX_PAN_SPREAD,
        )?;
        // The velocity law for the strike vector's direction reaches the mode
        // gains at note-on, so it is bounded exactly as the engine bounds it.
        if let Some(d) = &v.strike_direction {
            for (name, value) in [
                ("vh_db_at_pp", d.vh_db_at_pp),
                ("vh_db_at_ff", d.vh_db_at_ff),
            ] {
                within(
                    &format!("voicing.strike_direction.{name}"),
                    value,
                    -MAX_STRIKE_DIRECTION_DB,
                    MAX_STRIKE_DIRECTION_DB,
                )?;
            }
            within(
                "voicing.strike_direction.share_tilt",
                d.share_tilt,
                -MAX_SHARE_TILT,
                MAX_SHARE_TILT,
            )?;
        }
        if v.horizontal_offset_hz.len() != MAX_UNISON {
            return Err(Error::Preset(format!(
                "voicing.horizontal_offset_hz needs {MAX_UNISON} entries"
            )));
        }
        // The horizontal polarization is the same string in the other plane:
        // its partials are the vertical ones shifted by a fixed number of
        // hertz. A shift of a whole fundamental would move the bottom of the
        // compass by an octave, and one more negative than that would put the
        // first partial at a negative frequency, so the lowest note in the
        // tuning bounds them.
        let lowest_f0 = n.f0_hz.iter().copied().fold(f32::INFINITY, f32::min);
        for (i, &offset) in v.horizontal_offset_hz.iter().enumerate() {
            within(
                &format!("voicing.horizontal_offset_hz[{i}]"),
                offset,
                -lowest_f0,
                lowest_f0,
            )?;
        }
        if v.unison_layout.len() != MAX_UNISON
            || v.unison_layout
                .iter()
                .enumerate()
                .any(|(i, l)| l.detune.len() != i + 1 || l.share.len() != i + 1)
        {
            return Err(Error::Preset(format!(
                "voicing.unison_layout needs {MAX_UNISON} entries, the n-th with n \
                 detunings and n shares"
            )));
        }
        for (row, layout) in v.unison_layout.iter().enumerate() {
            let strings = row + 1;
            for (i, &detune) in layout.detune.iter().enumerate() {
                // A detuning is a fraction of the group's width, so a string
                // beyond ±1 is outside its own group. It also multiplies that
                // width in cents and is then exponentiated, which takes an
                // unbounded value out of the audible spectrum and past the
                // range of an `f32` frequency.
                within(
                    &format!("voicing.unison_layout[{row}].detune[{i}]"),
                    detune,
                    -1.0,
                    1.0,
                )?;
            }
            for (i, &share) in layout.share.iter().enumerate() {
                // The shares split one hammer blow and average to 1 across the
                // row, so no string takes a negative part of it (a hammer
                // cannot pull) or more than all of it.
                within(
                    &format!("voicing.unison_layout[{row}].share[{i}]"),
                    share,
                    0.0,
                    strings as f32,
                )?;
            }
        }
        if v.unison_sigma_scale.len() != MAX_UNISON
            || v.unison_sigma_scale
                .iter()
                .enumerate()
                .any(|(i, row)| row.scale.len() != i + 1)
        {
            return Err(Error::Preset(format!(
                "voicing.unison_sigma_scale needs {MAX_UNISON} entries, the n-th with n scales"
            )));
        }
        for (row, scales) in v.unison_sigma_scale.iter().enumerate() {
            let strings = row + 1;
            for (i, &scale) in scales.scale.iter().enumerate() {
                // A multiplier on a decay rate: zero or negative is a pole on
                // or outside the unit circle, i.e. a string that never stops.
                within(
                    &format!("voicing.unison_sigma_scale[{row}].scale[{i}]"),
                    scale,
                    MIN_SIGMA_SCALE,
                    MAX_SIGMA_SCALE,
                )?;
            }
            // The row is a *redistribution* of the note's damping, not a second
            // decay control: `notes.sigma0` alone decides how long the note
            // rings, and a row that did not average to 1 would silently retune
            // the whole compass's T60 out from under it.
            let mean = scales.scale.iter().sum::<f32>() / strings as f32;
            if (mean - 1.0).abs() > 1.0e-3 {
                return Err(Error::Preset(format!(
                    "voicing.unison_sigma_scale[{row}].scale averages {mean}, expected 1"
                )));
            }
        }
        if v.damper_weight.is_empty() {
            return Err(Error::Preset("voicing.damper_weight is empty".into()));
        }
        for anchor in &v.damper_weight {
            positive("voicing.damper_weight.hz", anchor.hz)?;
            finite("voicing.damper_weight.weight", anchor.weight)?;
        }
        // The engine walks these anchors in order and interpolates between
        // neighbours: out of order it reads the wrong pair, and two anchors at
        // one frequency divide by a zero span.
        if let Some(i) = v.damper_weight.windows(2).position(|w| w[0].hz >= w[1].hz) {
            return Err(Error::Preset(format!(
                "voicing.damper_weight[{}] is at {} Hz, not above the {} Hz before it",
                i + 1,
                v.damper_weight[i + 1].hz,
                v.damper_weight[i].hz
            )));
        }

        let h = &self.hammer;
        positive("hammer.velocity_min", h.velocity_min)?;
        positive("hammer.velocity_max", h.velocity_max)?;
        finite("hammer.felt_hysteresis", h.felt_hysteresis)?;
        positive("hammer.una_corda_stiffness", h.una_corda_stiffness)?;
        positive("hammer.reflection_gain", h.reflection_gain)?;

        let s = &self.soundboard;
        positive("soundboard.fdn_t60_lf", s.fdn_t60_lf)?;
        positive("soundboard.fdn_t60_hf", s.fdn_t60_hf)?;
        positive("soundboard.fdn_hf_hz", s.fdn_hf_hz)?;
        positive("soundboard.shelf_hz", s.shelf_hz)?;
        finite("soundboard.shelf_gain_db", s.shelf_gain_db)?;
        finite("soundboard.board_mix", s.board_mix)?;
        finite("soundboard.body_mix", s.body_mix)?;
        finite("soundboard.board_level", s.board_level)?;
        if s.body_modes.is_empty() {
            return Err(Error::Preset("soundboard.body_modes is empty".into()));
        }
        for mode in &s.body_modes {
            positive("soundboard.body_modes.hz", mode.hz)?;
            positive("soundboard.body_modes.q", mode.q)?;
            finite("soundboard.body_modes.gain", mode.gain)?;
        }

        // The mechanism events reach a biquad's coefficients and an exponential
        // envelope on the engine's audio path: a centroid at or past Nyquist is
        // a pole outside the unit circle, and a decay at zero is an event that
        // never ends.
        for (name, event) in self.noise.events() {
            validate_event(
                name,
                event.centroid_hz,
                event.decay_s,
                (MIN_NOISE_DECAY_S, MAX_NOISE_DECAY_S),
                event.velocity_db,
                &event.level_db,
            )?;
        }
        // The strike is the fifth event and the only one with a band limit of
        // its own, so it is checked here rather than in the loop above.
        let strike = &self.noise.strike;
        validate_event(
            "strike",
            strike.centroid_hz,
            strike.decay_s,
            (MIN_STRIKE_DECAY_S, MAX_STRIKE_DECAY_S),
            strike.velocity_db,
            &strike.level_db,
        )?;
        within(
            "noise.strike.bandwidth_hz",
            strike.bandwidth_hz,
            MIN_STRIKE_BANDWIDTH_HZ,
            MAX_STRIKE_BANDWIDTH_HZ,
        )?;
        // A band limit under the centroid is a burst whose energy is outside its
        // own band, and the two are then both describing the same thing badly.
        if strike.centroid_hz > strike.bandwidth_hz {
            return Err(Error::Preset(format!(
                "noise.strike.centroid_hz is {} but its bandwidth_hz is {}, so the burst is \
                 centred outside its own band",
                strike.centroid_hz, strike.bandwidth_hz
            )));
        }

        self.validate_bridge()?;
        self.validate_duplex()?;
        Ok(())
    }

    /// Checks the two ragged per-partial tables.
    ///
    /// Both are all-or-nothing across the compass — a preset either has one or
    /// does not — and inside a key both are allowed to be *short*: an estimator
    /// tracks as far up the series as the recording lets it, and every partial
    /// past the end of a row is 1.0. What is refused is a row that is **longer**
    /// than the key's partial count, with the key named, because a row that
    /// overruns the bank is a table measured on a different instrument (a
    /// different tuning, a different sample rate, or a different partial cap) and
    /// the entries the engine would silently drop are exactly the ones that say
    /// so.
    fn validate_partial_tables(&self, partial_counts: &[usize; NUM_KEYS]) -> Result<()> {
        for (name, table, low, high) in [
            (
                "partial_gains",
                &self.notes.partial_gains,
                MIN_PARTIAL_GAIN,
                MAX_PARTIAL_GAIN,
            ),
            (
                "partial_sigma_scale",
                &self.notes.partial_sigma_scale,
                MIN_PARTIAL_SIGMA_SCALE,
                MAX_PARTIAL_SIGMA_SCALE,
            ),
        ] {
            if table.is_empty() {
                continue;
            }
            if table.len() != NUM_KEYS {
                return Err(Error::Preset(format!(
                    "notes.{name} has {} rows, expected {NUM_KEYS} (or none at all)",
                    table.len()
                )));
            }
            for (i, row) in table.iter().enumerate() {
                let partials = partial_counts[i];
                if row.len() > partials {
                    return Err(Error::Preset(format!(
                        "notes.{name}[{i}] (key {}) has {} entries, but that key has only \
                         {partials} partials",
                        index_to_key(i),
                        row.len()
                    )));
                }
                for (k, &value) in row.iter().enumerate() {
                    within(&format!("notes.{name}[{i}][{k}]"), value, low, high)?;
                }
            }
        }
        Ok(())
    }

    /// The within-string splits: the engine's `validate_false_beats`, on the
    /// same bounds and with the same messages.
    fn validate_false_beats(&self, partial_counts: &[usize; NUM_KEYS]) -> Result<()> {
        let table = &self.notes.false_beat;
        if table.is_empty() {
            return Ok(());
        }
        if table.len() != NUM_KEYS {
            return Err(Error::Preset(format!(
                "notes.false_beat has {} rows, expected {NUM_KEYS} (or none at all)",
                table.len()
            )));
        }
        for (i, row) in table.iter().enumerate() {
            if row.len() > MAX_FALSE_BEATS_PER_KEY {
                return Err(Error::Preset(format!(
                    "notes.false_beat[{i}] has {} entries, expected at most \
                     {MAX_FALSE_BEATS_PER_KEY}",
                    row.len()
                )));
            }
            for (e, entry) in row.iter().enumerate() {
                let at = format!("notes.false_beat[{i}][{e}]");
                if entry.k == 0 || entry.k as usize > partial_counts[i] {
                    return Err(Error::Preset(format!(
                        "{at}.k is {}, but that key has partials 1..={}",
                        entry.k, partial_counts[i]
                    )));
                }
                within(&format!("{at}.hz"), entry.hz, MIN_FALSE_BEAT_HZ, MAX_FALSE_BEAT_HZ)?;
                within(&format!("{at}.db"), entry.db, MIN_FALSE_BEAT_DB, MAX_FALSE_BEAT_DB)?;
                if row[..e].iter().any(|other| other.k == entry.k) {
                    return Err(Error::Preset(format!(
                        "{at} splits partial {} a second time",
                        entry.k
                    )));
                }
            }
        }
        Ok(())
    }

    /// The bridge admittance's shape, and what it does to the coupling loop.
    ///
    /// The shape checks are the ordinary ones — a resonance at or past Nyquist
    /// is a pole outside the unit circle, a `Q` of zero divides by zero,
    /// anchors out of order read the wrong interpolation pair. The loop check
    /// is the one that matters and the reason this section can exist at all.
    ///
    /// # The derivation, which is the engine's
    ///
    /// The old stability argument was written for a *flat* bus: a string
    /// answers a steady drive at one of its own partials with at most about one
    /// signal unit per unit drive, so the tightest loop string → bus → string
    /// has gain `≈ resonance_coupling`, and bounding the coupling bounded the
    /// loop. With `B` in the path that loop has gain
    /// `≈ resonance_coupling · |B(f)|` at the frequency where it closes, and
    /// `B` is allowed well over unity at its resonances — forty cascaded peaks
    /// at +20 dB would multiply the loop by a thousand. So the quantity to
    /// bound is the *effective* coupling `resonance_coupling · max|B|`, and
    /// `max|B|` is a property of the **realised** filter — the fitted shelf
    /// cascade and the peaking sections that were actually built — and not of
    /// the anchors in the file. It is measured on a 512-point log grid from
    /// 20 Hz to 20 kHz plus every peak's own centre (the grid steps 1.4 %,
    /// which a `Q`-50 resonance hides between), by the mirror in
    /// [`crate::response`]. [`MAX_BRIDGE_LOOP_GAIN`] is a quarter of unity:
    /// 12 dB of margin against the worst string in the instrument, and four
    /// times more against any realistic cluster of coincident partials.
    fn validate_bridge(&self) -> Result<()> {
        let Some(bridge) = &self.voicing.bridge else {
            return Ok(());
        };
        let n = bridge.backbone.len();
        if !(2..=MAX_BRIDGE_ANCHORS).contains(&n) {
            return Err(Error::Preset(format!(
                "voicing.bridge.backbone has {n} anchors, expected 2..={MAX_BRIDGE_ANCHORS}"
            )));
        }
        for (i, a) in bridge.backbone.iter().enumerate() {
            within(
                &format!("voicing.bridge.backbone[{i}].hz"),
                a.hz,
                MIN_BRIDGE_HZ,
                MAX_BRIDGE_HZ,
            )?;
            within(
                &format!("voicing.bridge.backbone[{i}].gain_db"),
                a.gain_db,
                MIN_BRIDGE_GAIN_DB,
                MAX_BRIDGE_GAIN_DB,
            )?;
        }
        if let Some(i) = bridge.backbone.windows(2).position(|w| w[0].hz >= w[1].hz) {
            return Err(Error::Preset(format!(
                "voicing.bridge.backbone[{}] is at {} Hz, not above the {} Hz before it",
                i + 1,
                bridge.backbone[i + 1].hz,
                bridge.backbone[i].hz
            )));
        }
        if bridge.peaks.len() > MAX_BRIDGE_PEAKS {
            return Err(Error::Preset(format!(
                "voicing.bridge.peaks has {} entries, expected at most {MAX_BRIDGE_PEAKS}",
                bridge.peaks.len()
            )));
        }
        for (i, p) in bridge.peaks.iter().enumerate() {
            within(
                &format!("voicing.bridge.peaks[{i}].hz"),
                p.hz,
                MIN_BRIDGE_HZ,
                MAX_BRIDGE_HZ,
            )?;
            within(
                &format!("voicing.bridge.peaks[{i}].q"),
                p.q,
                MIN_BRIDGE_Q,
                MAX_BRIDGE_Q,
            )?;
            within(
                &format!("voicing.bridge.peaks[{i}].gain_db"),
                p.gain_db,
                MIN_BRIDGE_GAIN_DB,
                MAX_BRIDGE_GAIN_DB,
            )?;
        }

        within(
            "voicing.bridge.radiated_share",
            bridge.radiated_share,
            0.0,
            MAX_RADIATED_SHARE,
        )?;

        // Well formed above; safe here.
        let max_b = BridgeResponse::new(bridge).max_magnitude();
        if !max_b.is_finite() {
            return Err(Error::Preset(format!(
                "voicing.bridge has a response of {max_b} somewhere in the audio band"
            )));
        }
        let loop_gain = self.voicing.resonance_coupling * max_b;
        if loop_gain > MAX_BRIDGE_LOOP_GAIN {
            return Err(Error::Preset(format!(
                "voicing.bridge peaks at {:.1} dB, which with resonance_coupling = {} makes a \
                 sympathetic loop gain of {loop_gain}, past the {MAX_BRIDGE_LOOP_GAIN} the bus \
                 is stable under",
                20.0 * max_b.log10(),
                self.voicing.resonance_coupling
            )));
        }
        Ok(())
    }

    /// The duplex segments' shape, and what 88 permanently undamped banks do to
    /// the coupling loop.
    ///
    /// A different loop from the bridge's, and a worse one: a string's
    /// contribution to the coupling loop dies with the note, and a segment's
    /// never does. Segment `j` puts `D_j(f)` on the bus per unit of drive at
    /// `f` and gets back `resonance_coupling · B(f)` of every other segment's
    /// output one block later plus, in the worst case, its own — so the
    /// tightest loop any frequency can close is
    ///
    /// ```text
    /// resonance_coupling · max|B| · ( sum_j |D_j(f)| + max_j |D_j(f)| )
    /// ```
    ///
    /// evaluated at every segment's own centre, where a sum of resonances
    /// peaks, and bounded by [`MAX_DUPLEX_LOOP_GAIN`]. What it refuses is the
    /// preset that tunes every key's segments alike — which is also, per Öberg
    /// & Askenfelt, not what a piano does. A measured layout passes it by two
    /// orders of magnitude, and `estimate::duplex` reports the margin it
    /// achieved rather than trusting it.
    fn validate_duplex(&self) -> Result<()> {
        let table = &self.notes.duplex;
        if table.is_empty() {
            return Ok(());
        }
        if table.len() != NUM_KEYS {
            return Err(Error::Preset(format!(
                "notes.duplex has {} rows, expected {NUM_KEYS} (or none at all)",
                table.len()
            )));
        }
        for (i, row) in table.iter().enumerate() {
            if row.len() > MAX_DUPLEX_MODES {
                return Err(Error::Preset(format!(
                    "notes.duplex[{i}] has {} segments, expected at most {MAX_DUPLEX_MODES}",
                    row.len()
                )));
            }
            for (k, m) in row.iter().enumerate() {
                within(
                    &format!("notes.duplex[{i}][{k}].hz"),
                    m.hz,
                    MIN_DUPLEX_HZ,
                    MAX_DUPLEX_HZ,
                )?;
                within(
                    &format!("notes.duplex[{i}][{k}].gain_db"),
                    m.gain_db,
                    MIN_DUPLEX_GAIN_DB,
                    MAX_DUPLEX_GAIN_DB,
                )?;
                within(
                    &format!("notes.duplex[{i}][{k}].t60_s"),
                    m.t60_s,
                    MIN_DUPLEX_T60_S,
                    MAX_DUPLEX_T60_S,
                )?;
            }
        }

        let loop_gain = self.duplex_loop_gain();
        if loop_gain > MAX_DUPLEX_LOOP_GAIN {
            return Err(Error::Preset(format!(
                "notes.duplex closes an undamped loop of gain {loop_gain} with \
                 resonance_coupling = {}, past the {MAX_DUPLEX_LOOP_GAIN} the bus is stable \
                 under",
                self.voicing.resonance_coupling
            )));
        }
        Ok(())
    }

    /// The undamped loop gain `validate_duplex` bounds, so that a fit can
    /// report its margin instead of discovering it as a rejection.
    pub fn duplex_loop_gain(&self) -> f32 {
        let max_b = BridgeResponse::of(self.voicing.bridge.as_ref()).max_magnitude();
        self.voicing.resonance_coupling * max_b * self.duplex_response_factor()
    }

    /// `Σ_j |D_j(f)| + max_j |D_j(f)|` at its worst frequency: everything in
    /// the duplex loop bound except the coupling and the admittance.
    ///
    /// Zero for a preset with no segments. A fit that is about to move the
    /// coupling needs this *before* it renders, because the coupling ceiling is
    /// `MAX_LOOP_GAIN / (max|B| · max(1, this))` and discovering it as a
    /// validation failure eight renders in is not a fit.
    pub fn duplex_response_factor(&self) -> f32 {
        let table = &self.notes.duplex;
        if table.is_empty() {
            return 0.0;
        }
        let mut worst = 0.0f32;
        for probe in table.iter().flatten() {
            let (mut total, mut largest) = (0.0f32, 0.0f32);
            for row in table {
                let d = duplex_magnitude(row, probe.hz);
                total += d;
                largest = largest.max(d);
            }
            worst = worst.max(total + largest);
        }
        worst
    }

    /// The segments of one key, or nothing when the preset has no table.
    pub fn duplex_modes(&self, key: u8) -> &[DuplexMode] {
        key_index(key)
            .and_then(|i| self.notes.duplex.get(i))
            .map_or(&[], Vec::as_slice)
    }

    /// Pitch of a key according to this preset's tuning.
    pub fn f0(&self, key: u8) -> Option<f32> {
        key_index(key).map(|i| self.notes.f0_hz[i])
    }

    /// How far this preset's tuning is stretched from equal temperament at
    /// `key`, in cents.
    pub fn stretch_cents(&self, key: u8) -> Option<f64> {
        self.f0(key)
            .map(|f0| 1200.0 * (f64::from(f0) / equal_temperament(key)).log2())
    }
}

/// How much faster the engine runs its vertical modal bank than the per-note
/// `sigma` table says.
///
/// The tables hold the *whole note's* decay rate — both polarizations together,
/// which is what a T60 measured from a recording is. Inside the engine the
/// horizontal polarization starts `gain_db` down but decays `decay_ratio` times
/// as fast, so it alone is what is left at the end and it alone sets when the
/// note reaches -60 dB. Solving `g exp(-rho sigma_v T60) = 1e-3 (1 + g)` for
/// `sigma_v` gives the factor between the two, and every estimated decay has to
/// be written to the file on the table's side of it. This is the engine's own
/// `Voicing::vertical_decay_factor`, duplicated here because a tuner that
/// cannot convert between the two conventions cannot check its own output.
pub fn vertical_decay_factor(horizontal_gain_db: f64, horizontal_decay_ratio: f64) -> f64 {
    let gain = 10f64.powf(horizontal_gain_db / 20.0);
    (gain / (1.0e-3 * (1.0 + gain))).ln() / (horizontal_decay_ratio * 6.91)
}

impl Voicing {
    /// The fraction of a note's full detune width that its *dominant* beat
    /// spans, for a group of `unison` strings.
    ///
    /// This is what turns a measurement into a table entry. A three-string
    /// unison beats at three different rates at once, and what an envelope's
    /// autocorrelation picks out is the deepest modulation — the pair whose
    /// amplitudes are most nearly equal and largest, i.e. the pair with the
    /// largest product of hammer shares. That pair is not the outer one, so its
    /// interval is not the full width: with the default layout it spans 0.61 of
    /// it, and a preset written from the raw measurement would tune every
    /// three-string unison 40 % too narrow.
    ///
    /// Zero for a single string, which has nothing to beat against.
    pub fn dominant_beat_fraction(&self, unison: usize) -> f64 {
        let Some(layout) = self.unison_layout.get(unison.clamp(1, MAX_UNISON) - 1) else {
            return 0.0;
        };
        let mut best = (0.0f64, 0.0f64);
        for i in 0..layout.detune.len() {
            for j in i + 1..layout.detune.len() {
                let strength = f64::from(layout.share[i]) * f64::from(layout.share[j]);
                if strength > best.0 {
                    best = (
                        strength,
                        f64::from(layout.detune[i] - layout.detune[j]).abs(),
                    );
                }
            }
        }
        best.1
    }

    /// [`vertical_decay_factor`] for this voicing.
    pub fn vertical_decay_factor(&self) -> f64 {
        vertical_decay_factor(
            f64::from(self.horizontal_gain_db),
            f64::from(self.horizontal_decay_ratio),
        )
    }
}

/// One key's partial layout, in the engine's own arithmetic.
///
/// [`Preset::validate`] has to answer a question about what the engine will
/// *build* — how many partials the bank holds and whether their frequencies
/// stay ordered under a signed `B4` — so the layout is recomputed here in `f32`,
/// term for term as `engine::string::StringParams` computes it. A `f64`
/// paraphrase would disagree with the engine at the last bit, which is exactly
/// where a series that only just stays ordered lives.
struct PartialLayout {
    f0: f32,
    b: f32,
    b4: f32,
}

impl PartialLayout {
    fn of(preset: &Preset, index: usize) -> Self {
        Self {
            f0: preset.notes.f0_hz[index],
            b: preset.notes.inharmonicity_b[index],
            b4: preset.notes.inharmonicity_b4[index],
        }
    }

    fn radicand(&self, k: u32) -> f32 {
        let k = k as f32;
        let k2 = k * k;
        1.0 + self.b * k * k + self.b4 * k2 * k2
    }

    fn partial_hz(&self, k: u32) -> f32 {
        k as f32 * self.f0 * self.radicand(k).sqrt()
    }

    /// Partials the engine builds for this key, exactly as
    /// `engine::string::StringParams::partial_count` counts them: the series
    /// stops at the first partial at or above the Nyquist cap, and a key whose
    /// fundamental is already past it still has one mode.
    fn partial_count(&self) -> usize {
        let limit = (MAX_PARTIAL_RATIO * SAMPLE_RATE) as f32;
        (1..=MAX_PARTIALS)
            .take_while(|&k| self.partial_hz(k) < limit)
            .count()
            .max(1)
    }
}

/// The shape checks every `[noise]` event shares. The strike passes a different
/// decay range and carries a band limit the other four do not have, which is why
/// this takes the fields rather than an [`EventNoise`].
fn validate_event(
    name: &str,
    centroid_hz: f32,
    decay_s: f32,
    decay_range: (f32, f32),
    velocity_db: f32,
    level_db: &[NoiseAnchor],
) -> Result<()> {
    positive(&format!("noise.{name}.centroid_hz"), centroid_hz)?;
    within(
        &format!("noise.{name}.centroid_hz"),
        centroid_hz,
        1.0,
        (0.45 * SAMPLE_RATE) as f32,
    )?;
    within(
        &format!("noise.{name}.decay_s"),
        decay_s,
        decay_range.0,
        decay_range.1,
    )?;
    finite(&format!("noise.{name}.velocity_db"), velocity_db)?;
    if level_db.is_empty() {
        return Err(Error::Preset(format!("noise.{name}.level_db is empty")));
    }
    for (i, anchor) in level_db.iter().enumerate() {
        if key_index(anchor.key).is_none() {
            return Err(Error::Preset(format!(
                "noise.{name}.level_db[{i}].key is {}, which is not on the keyboard",
                anchor.key
            )));
        }
        // A mechanism event louder than the note it belongs to is not a
        // mechanism event; the measured range is -25 to -45 dB.
        if !anchor.db.is_finite() || anchor.db > 0.0 {
            return Err(Error::Preset(format!(
                "noise.{name}.level_db[{i}].db is {}, expected a finite level at or \
                 below 0 dB relative to a strike",
                anchor.db
            )));
        }
    }
    // The level is interpolated across the compass by walking the anchors in
    // order.
    if let Some(i) = level_db.windows(2).position(|w| w[0].key >= w[1].key) {
        return Err(Error::Preset(format!(
            "noise.{name}.level_db[{}] is at key {}, not above the key {} before it",
            i + 1,
            level_db[i + 1].key,
            level_db[i].key
        )));
    }
    Ok(())
}

fn table_length(name: &str, length: usize) -> Result<()> {
    if length == NUM_KEYS {
        Ok(())
    } else {
        Err(Error::Preset(format!(
            "notes.{name} has {length} entries, expected {NUM_KEYS}"
        )))
    }
}

fn finite(name: &str, value: f32) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(Error::Preset(format!("{name} is {value}")))
    }
}

fn positive(name: &str, value: f32) -> Result<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(Error::Preset(format!(
            "{name} is {value}, expected a finite positive number"
        )))
    }
}

fn within(name: &str, value: f32, low: f32, high: f32) -> Result<()> {
    if value.is_finite() && (low..=high).contains(&value) {
        Ok(())
    } else {
        Err(Error::Preset(format!(
            "{name} is {value}, expected a number in {low}..={high}"
        )))
    }
}

// ---------------------------------------------------------- the estimates

/// What the estimators found for one measured note. Every field is optional:
/// a recording that was too short for a decay fit still contributes its
/// inharmonicity.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NoteEstimate {
    pub key: u8,
    pub f0_hz: Option<f64>,
    pub inharmonicity_b: Option<f64>,
    /// The signed fourth-order coefficient. `Some(0.0)` is a measurement — the
    /// two-band diagnostic said the series has no curvature the second
    /// coefficient can describe — and `None` is the absence of one.
    pub inharmonicity_b4: Option<f64>,
    pub strike_position: Option<f64>,
    pub contact_width: Option<f64>,
    /// Soft floor under this key's comb nulls
    /// ([`estimate::shaping::comb_floor`](crate::estimate::shaping::comb_floor)).
    /// `Some(0.0)` is a measurement — the comb is already shallower than the
    /// recording's deepest partial and needs no floor — and `None` is the
    /// absence of one.
    pub comb_floor: Option<f64>,
    /// Extra decay rate the damper adds at full engagement, 1/s
    /// ([`estimate::damper`](crate::estimate::damper)).
    pub damper_sigma: Option<f64>,
    pub sigma0: Option<f64>,
    pub sigma1: Option<f64>,
    pub detune_cents: Option<f64>,
    pub hammer_mass: Option<f64>,
    pub hammer_stiffness: Option<f64>,
    pub hammer_exponent: Option<f64>,
}

impl NoteEstimate {
    pub fn new(key: u8) -> Self {
        Self {
            key,
            ..Default::default()
        }
    }

    /// Takes the tuning and the inharmonicity from an `(f0, B, B4)` fit.
    pub fn with_inharmonic(mut self, fit: &InharmonicFit) -> Self {
        self.f0_hz = Some(fit.model.f0_hz);
        self.inharmonicity_b = Some(fit.model.b);
        self.inharmonicity_b4 = Some(fit.model.b4);
        self
    }

    /// Takes the damping law from a decay fit.
    pub fn with_decay_curve(mut self, curve: &DecayCurve) -> Self {
        self.sigma0 = Some(curve.sigma0);
        self.sigma1 = Some(curve.sigma1);
        self
    }

    pub fn with_decays(self, report: &DecayReport) -> Self {
        self.with_decay_curve(&report.curve)
    }

    /// Takes the unison detuning. The estimate is the interval of the
    /// *dominant* beat; [`PresetBuilder::build`] widens it to the group's full
    /// spread through the base preset's unison layout.
    pub fn with_unison(mut self, unison: &UnisonEstimate) -> Self {
        self.detune_cents = Some(unison.detune_cents);
        self
    }

    /// Takes the strike point and the hammer's contact width from a comb fit.
    pub fn with_strike(mut self, fit: &StrikeFit) -> Self {
        self.strike_position = Some(fit.position);
        self.contact_width = fit.contact_width;
        self
    }

    pub fn with_hammer(mut self, fit: &HammerFit) -> Self {
        self.hammer_mass = Some(fit.felt.mass);
        self.hammer_stiffness = Some(fit.felt.stiffness);
        self.hammer_exponent = Some(fit.felt.exponent);
        self
    }
}

/// Assembles a preset from a base and a set of per-note estimates.
#[derive(Clone, Debug)]
pub struct PresetBuilder {
    base: Preset,
    notes: Vec<NoteEstimate>,
    polarization: Option<PolarizationSplit>,
    velocity_map: Option<VelocityMap>,
    sigma_scale: Option<Vec<UnisonSigmaScale>>,
    pan_spread: Option<f32>,
    noise: Option<NoiseTables>,
    partial_gains: Option<Vec<Vec<f32>>>,
    partial_sigma_scale: Option<Vec<Vec<f32>>>,
}

impl PresetBuilder {
    /// Starts from `base`. Everything stage 1 cannot see from isolated
    /// recordings — soundboard, body modes, coupling, damper profile, the
    /// unison group sizes, the bridge gains — is carried over from it
    /// unchanged.
    pub fn new(base: Preset) -> Self {
        Self {
            base,
            notes: Vec::new(),
            polarization: None,
            velocity_map: None,
            sigma_scale: None,
            pan_spread: None,
            noise: None,
            partial_gains: None,
            partial_sigma_scale: None,
        }
    }

    pub fn from_base_file(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self::new(Preset::load(path)?))
    }

    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.base.name = name.into();
        self
    }

    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.base.description = description.into();
        self
    }

    /// Adds one measured note. A second estimate for the same key replaces the
    /// first, so a caller may overwrite a note it re-measured.
    pub fn note(mut self, estimate: NoteEstimate) -> Self {
        self.notes.retain(|n| n.key != estimate.key);
        self.notes.push(estimate);
        self
    }

    /// The two-polarization split, which is global in the engine: one figure
    /// for the whole instrument, normally the median over the measured notes.
    pub fn polarization(mut self, split: PolarizationSplit) -> Self {
        self.polarization = Some(split);
        self
    }

    pub fn velocity_map(mut self, map: VelocityMap) -> Self {
        self.velocity_map = Some(map);
        self
    }

    /// The per-string decay spread, which is global in the engine: one row per
    /// unison size, normally
    /// [`SigmaSpread::rows`](crate::estimate::spread::SigmaSpread::rows) over
    /// the notes that showed the drift.
    pub fn sigma_scale(mut self, rows: Vec<UnisonSigmaScale>) -> Self {
        self.sigma_scale = Some(rows);
        self
    }

    /// How far apart the two polarizations are panned. Not estimated by stage
    /// 1: what a recording shows is the *drift* of one note's balance
    /// ([`residual::stereo_balance`](crate::residual::stereo_balance)), and
    /// turning that into a pan displacement needs the microphone geometry the
    /// recording does not carry.
    pub fn pan_spread(mut self, spread: f32) -> Self {
        self.pan_spread = Some(spread);
        self
    }

    /// The mechanism's own sounds, from
    /// [`estimate::noise`](crate::estimate::noise).
    pub fn noise(mut self, noise: NoiseTables) -> Self {
        self.noise = Some(noise);
        self
    }

    /// The per-partial excitation gains, one ragged row per key, from
    /// [`estimate::shaping`](crate::estimate::shaping). Not interpolated across
    /// the compass and deliberately so: `TUNING_REPORT.md` §3 measured that the
    /// roughness is *not* shared between notes at the same frequency, so a row
    /// invented for an unsampled key would be its neighbour's roughness under
    /// its own partials. An unsampled key gets an empty row, which is 1.0
    /// everywhere.
    pub fn partial_gains(mut self, rows: Vec<Vec<f32>>) -> Self {
        self.partial_gains = Some(rows);
        self
    }

    /// The per-partial decay corrections, same shape and same reasoning.
    pub fn partial_sigma_scale(mut self, rows: Vec<Vec<f32>>) -> Self {
        self.partial_sigma_scale = Some(rows);
        self
    }

    /// Interpolates every estimated quantity across the compass and returns the
    /// finished preset.
    pub fn build(&self) -> Result<Preset> {
        let mut preset = self.base.clone();
        for note in &self.notes {
            if key_index(note.key).is_none() {
                return Err(Error::Preset(format!(
                    "estimate for key {}, which is not on the keyboard",
                    note.key
                )));
            }
        }

        // The tuning is interpolated as its deviation from equal temperament,
        // not as frequency. Interpolating the frequencies themselves would let
        // a curve through every third note decide the pitch of the two between
        // them, and the ear hears a few cents there; interpolating the stretch
        // — which is smooth, a few cents wide, and what a tuner actually sets —
        // leaves the unmeasured notes exactly in tune with themselves.
        let stretch: Vec<(u8, f64)> = self
            .notes
            .iter()
            .filter_map(|note| {
                note.f0_hz
                    .map(|f0| (note.key, 1200.0 * (f0 / equal_temperament(note.key)).log2()))
            })
            .collect();
        let base_stretch: Vec<f64> = (0..NUM_KEYS)
            .map(|index| {
                let key = index_to_key(index);
                1200.0 * (f64::from(preset.notes.f0_hz[index]) / equal_temperament(key)).log2()
            })
            .collect();
        if let Some(cents) = fill(&base_stretch, &stretch, false)? {
            for (index, cents) in cents.iter().enumerate() {
                let key = index_to_key(index);
                preset.notes.f0_hz[index] =
                    (equal_temperament(key) * (cents / 1200.0).exp2()) as f32;
            }
        }

        // (table, sample, log domain, lower clamp, upper clamp)
        #[allow(clippy::type_complexity)]
        let fields: [(&str, fn(&NoteEstimate) -> Option<f64>, bool, f64, f64); 12] = [
            ("inharmonicity_b", |n| n.inharmonicity_b, true, 0.0, 1.0),
            // Signed, and a correction rather than a quantity: the base table
            // is zero everywhere, so `fill`'s "move the base curve onto the
            // measurement" rule degenerates to holding the nearest measured
            // value past the ends of the measured range, which is what a
            // correction wants. The clamp is only a sanity bound; what really
            // bounds `B4` is the partial series it has to leave ordered, and
            // that is enforced below.
            ("inharmonicity_b4", |n| n.inharmonicity_b4, false, -1.0, 1.0),
            (
                "contact_width",
                |n| n.contact_width,
                false,
                0.0,
                f64::from(MAX_CONTACT_WIDTH),
            ),
            // A strike point is a point on the string, and the comb cannot tell
            // x from 1 - x, so the table's half is the near one.
            ("strike_position", |n| n.strike_position, true, 1e-3, 0.49),
            // A floor, like `B4`, is a correction whose base table is zero
            // everywhere, so `fill` holds the nearest measured value past the
            // ends of the measured range rather than decaying to nothing —
            // which is what an unsampled key needs, since its comb has the same
            // nulls as its neighbours'.
            (
                "comb_floor",
                |n| n.comb_floor,
                false,
                0.0,
                f64::from(MAX_COMB_FLOOR),
            ),
            // The damper's grip. Interpolated in the log domain because it is a
            // rate that spans an order of magnitude across the compass.
            ("damper_sigma", |n| n.damper_sigma, true, 0.0, f64::MAX),
            // The two survey clamps are the schema's own bounds, not looser
            // ones: an interpolated table that the engine would refuse is a
            // preset the pipeline cannot write. Neither moves a fitted value —
            // the measured preset's smallest `sigma0` is 0.126 and its widest
            // spread is 3.89 cents.
            ("sigma0", |n| n.sigma0, true, f64::from(MIN_MODE_SIGMA), f64::MAX),
            ("sigma1", |n| n.sigma1, false, 0.0, f64::MAX),
            (
                "detune_cents",
                |n| n.detune_cents,
                true,
                0.0,
                f64::from(MAX_DETUNE_CENTS),
            ),
            ("hammer_mass", |n| n.hammer_mass, true, 1e-4, 1.0),
            ("hammer_stiffness", |n| n.hammer_stiffness, true, 1.0, f64::MAX),
            ("hammer_exponent", |n| n.hammer_exponent, false, 1.0, 6.0),
        ];
        for (name, get, log_values, low, high) in fields {
            let mut samples: Vec<(u8, f64)> = self
                .notes
                .iter()
                .filter_map(|note| get(note).map(|value| (note.key, value)))
                .collect();
            if name == "detune_cents" {
                // What was measured is the dominant beat of the group; what the
                // table holds is the group's full spread. A single-strung note
                // has no beat to have measured, so an estimate for one is
                // dropped rather than divided by zero.
                samples.retain_mut(|(key, cents)| {
                    let index = key_index(*key).expect("checked above");
                    let fraction = preset
                        .voicing
                        .dominant_beat_fraction(usize::from(preset.notes.unison[index]));
                    if fraction <= 0.0 {
                        return false;
                    }
                    *cents /= fraction;
                    true
                });
            }
            let table = table_of(&mut preset.notes, name);
            let base: Vec<f64> = table.iter().map(|&v| f64::from(v)).collect();
            let Some(filled) = fill(&base, &samples, log_values)? else {
                continue;
            };
            for (slot, value) in table.iter_mut().zip(filled) {
                *slot = value.clamp(low, high) as f32;
            }
        }

        // An interpolated `B4` is a coefficient nobody measured at that key,
        // and a negative one large enough to fold the top of the series back
        // down would be refused — for the whole preset, at the end of a survey
        // that took minutes. It is shrunk towards zero instead, which is the
        // two-parameter law the key had before, and only where the series it
        // produces is not a layout.
        for index in 0..NUM_KEYS {
            while !ordered_series(&preset, index) {
                let b4 = &mut preset.notes.inharmonicity_b4[index];
                if *b4 == 0.0 || !b4.is_finite() {
                    *b4 = 0.0;
                    break;
                }
                *b4 *= 0.5;
            }
        }

        if let Some(split) = self.polarization {
            preset.voicing.horizontal_gain_db = split.gain_db as f32;
            preset.voicing.horizontal_decay_ratio = split.decay_ratio as f32;
        }
        if let Some(map) = self.velocity_map {
            preset.hammer.velocity_min = map.velocity_min as f32;
            preset.hammer.velocity_max = map.velocity_max as f32;
        }
        if let Some(rows) = &self.sigma_scale {
            preset.voicing.unison_sigma_scale = rows.clone();
        }
        if let Some(spread) = self.pan_spread {
            preset.voicing.polarization_pan_spread = spread;
        }
        if let Some(noise) = &self.noise {
            preset.noise = noise.clone();
        }
        // Written after the partial series is settled: a row is only legal
        // against the bank the finished layout builds, and `B4` was shrunk above
        // for exactly the keys where that count could have moved.
        for (rows, table) in [
            (&self.partial_gains, &mut preset.notes.partial_gains),
            (
                &self.partial_sigma_scale,
                &mut preset.notes.partial_sigma_scale,
            ),
        ] {
            let Some(rows) = rows else { continue };
            *table = if rows.iter().all(Vec::is_empty) {
                // Every row neutral is the table's own absence, and an absent
                // table is what keeps a preset that measured nothing byte-stable.
                Vec::new()
            } else {
                rows.clone()
            };
        }
        preset.validate()?;
        Ok(preset)
    }
}

/// One per-note table, rebuilt from what was measured of it.
///
/// * Nothing measured: `None` — the base table stands.
/// * Two or more notes: the monotone-cubic compass curve through them. It
///   passes through its data, so a measured note keeps its own value exactly.
/// * **One** note: the base table *moved onto* the measurement — scaled in the
///   log domain, shifted in the linear one — rather than flattened to a
///   constant. One measurement says where the curve passes, not what shape it
///   has, and the base preset's shape is a far better guess than a horizontal
///   line. It is also what makes a single-note run useful: measure one A4 and
///   the whole instrument moves to that pitch.
///
/// Beyond the outermost measurements the same rule as for a single note takes
/// over: the base table moved onto the nearest measured key, rather than the
/// interpolant's end slope carried onwards. An end slope is fitted to the last
/// two data points and says nothing about what happens past them, and these
/// quantities climb steeply at the ends of the compass — a damping rate that
/// rises eightfold between the last two measured keys extrapolates to a top
/// note that dies in a fifth of a second, faster than a *damped* note, which is
/// not a piano. The base preset's shape is the better guess there for exactly
/// the reason it is with one measurement.
fn fill(base: &[f64], samples: &[(u8, f64)], log_values: bool) -> Result<Option<Vec<f64>>> {
    if samples.is_empty() {
        return Ok(None);
    }
    let index_of = |key: u8| {
        key_index(key).ok_or_else(|| {
            Error::Preset(format!("estimate for key {key}, which is not on the keyboard"))
        })
    };
    // Moves the whole base table onto `measured` at `key`.
    let moved = |key: u8, measured: f64| -> Result<Vec<f64>> {
        let anchor = base[index_of(key)?];
        if log_values {
            if !(anchor > 0.0 && measured > 0.0) {
                return Err(Error::Preset(format!(
                    "cannot scale a table through {measured} at key {key}"
                )));
            }
            let ratio = measured / anchor;
            Ok(base.iter().map(|&v| v * ratio).collect())
        } else {
            let offset = measured - anchor;
            Ok(base.iter().map(|&v| v + offset).collect())
        }
    };

    if samples.len() == 1 {
        return Ok(Some(moved(samples[0].0, samples[0].1)?));
    }
    let curve = CompassCurve::from_keys(samples, log_values)?;
    let lowest = samples.iter().map(|&(key, _)| key).min().expect("non-empty");
    let highest = samples.iter().map(|&(key, _)| key).max().expect("non-empty");
    let below = moved(lowest, curve.value_at_key(lowest))?;
    let above = moved(highest, curve.value_at_key(highest))?;
    Ok(Some(
        (0..base.len())
            .map(|index| {
                let key = index_to_key(index);
                if key < lowest {
                    below[index]
                } else if key > highest {
                    above[index]
                } else {
                    curve.value_at_key(key)
                }
            })
            .collect(),
    ))
}

/// Whether one key's partials stay under a positive root and in ascending
/// order — the same question [`Preset::validate`] asks, without the message.
fn ordered_series(preset: &Preset, index: usize) -> bool {
    let layout = PartialLayout::of(preset, index);
    // Over the full reachable range, not `partial_count()`, for the same
    // reason as `validate`: a radicand that jumps straight negative truncates
    // the count itself, and the series it leaves behind must not pass as
    // ordered.
    let limit = (MAX_PARTIAL_RATIO * SAMPLE_RATE) as f32;
    let mut previous = 0.0f32;
    for k in 1..=MAX_PARTIALS {
        let radicand = layout.radicand(k);
        let f = layout.partial_hz(k);
        if !(radicand.is_finite() && radicand > 0.0 && f.is_finite()) || f <= previous {
            return false;
        }
        if f >= limit {
            break;
        }
        previous = f;
    }
    true
}

fn table_of<'a>(notes: &'a mut NoteTables, name: &str) -> &'a mut Vec<f32> {
    match name {
        "inharmonicity_b" => &mut notes.inharmonicity_b,
        "inharmonicity_b4" => &mut notes.inharmonicity_b4,
        "contact_width" => &mut notes.contact_width,
        "comb_floor" => &mut notes.comb_floor,
        "damper_sigma" => &mut notes.damper_sigma,
        "strike_position" => &mut notes.strike_position,
        "sigma0" => &mut notes.sigma0,
        "sigma1" => &mut notes.sigma1,
        "detune_cents" => &mut notes.detune_cents,
        "hammer_mass" => &mut notes.hammer_mass,
        "hammer_stiffness" => &mut notes.hammer_stiffness,
        "hammer_exponent" => &mut notes.hammer_exponent,
        other => unreachable!("no per-note table named {other}"),
    }
}

/// Serializers that write an `f32` as the shortest decimal that reads back as
/// the same `f32` (`0.35`, not `0.34999999403953552`). The engine's preset
/// module does the same thing for the same reason; the two must agree or the
/// files stop round-tripping between them.
mod short {
    use serde::ser::{SerializeSeq, Serializer};

    fn widen(x: f32) -> f64 {
        x.to_string().parse().unwrap_or(x as f64)
    }

    pub fn scalar<S: Serializer>(x: &f32, s: S) -> std::result::Result<S::Ok, S::Error> {
        s.serialize_f64(widen(*x))
    }

    /// The same, one row per key: the per-partial tables are ragged, so they are
    /// a list of lists rather than a list.
    pub fn table<S: Serializer>(
        rows: &[Vec<f32>],
        s: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        let mut seq = s.serialize_seq(Some(rows.len()))?;
        for row in rows {
            let widened: Vec<f64> = row.iter().map(|&x| widen(x)).collect();
            seq.serialize_element(&widened)?;
        }
        seq.end()
    }

    pub fn list<S: Serializer>(v: &[f32], s: S) -> std::result::Result<S::Ok, S::Error> {
        let mut seq = s.serialize_seq(Some(v.len()))?;
        for &x in v {
            seq.serialize_element(&widen(x))?;
        }
        seq.end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The engine's own preset file, next to the repository root.
    pub(crate) fn default_preset_path() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../presets/default.toml")
    }

    fn default_preset() -> Preset {
        Preset::load(default_preset_path()).expect("presets/default.toml loads")
    }

    #[test]
    fn the_engines_preset_file_round_trips_through_this_schema_byte_for_byte() {
        // The whole contract between the two crates, in one assertion: this
        // module reads what the engine writes, and writes what the engine
        // reads. A field added on either side, or renamed, or reordered, fails
        // here.
        let text = std::fs::read_to_string(default_preset_path()).expect("read");
        let preset = Preset::from_toml(&text).expect("parse");
        assert_eq!(preset.to_toml(), text);
        assert_eq!(preset.notes.f0_hz.len(), NUM_KEYS);
    }

    /// `validate` is duplicated across the two crates rather than shared, so
    /// what keeps the copies honest is asserting them against each other on
    /// material both can read. Every mutation below is a value that reaches the
    /// engine's audio thread as a NaN in a mode frequency or an excitation, or
    /// as a feedback loop that sustains itself — the tuner must refuse it where
    /// the number was produced, not leave it to be found at playback.
    /// A preset that *uses* every refinement, written by the engine and read
    /// back here.
    ///
    /// The four defaulted fields and the `[noise]` section only appear in a
    /// file when they are not at their default, so the byte-for-byte round trip
    /// above never sees them: it is a test of a file in which they are all
    /// absent. This is the other half — the engine writes one with all of them
    /// present, and this module has to read it, validate it, and write the same
    /// bytes back. Field order, spelling and serialization style are all pinned
    /// by it.
    #[test]
    fn a_preset_that_uses_every_refinement_round_trips_through_the_engine() {
        let mut preset = piano_emulator::Preset::from_toml(
            &std::fs::read_to_string(default_preset_path()).expect("read"),
        )
        .expect("the engine reads its own preset");
        preset.notes.inharmonicity_b4[3] = -1.0e-9;
        preset.notes.inharmonicity_b4[40] = 2.5e-7;
        preset.notes.contact_width[0] = 0.02;
        preset.notes.contact_width[87] = 0.04;
        preset.voicing.polarization_pan_spread = 0.25;
        preset.voicing.unison_sigma_scale[1].scale = vec![0.8, 1.2];
        preset.voicing.unison_sigma_scale[2].scale = vec![0.75, 1.0, 1.25];
        preset.noise.key_off.centroid_hz = 210.0;
        preset.noise.pedal_down.decay_s = 4.5;
        preset.voicing.bridge = Some(piano_emulator::preset::BridgeVoicing {
            backbone: [(30.0, -12.0), (250.0, 1.5), (1_100.0, 0.0), (10_000.0, -13.0)]
                .into_iter()
                .map(|(hz, gain_db)| piano_emulator::preset::BridgeAnchor { hz, gain_db })
                .collect(),
            peaks: [(58.0, 22.0, 8.0), (188.0, 30.0, 6.5), (3_800.0, 4.0, 3.0)]
                .into_iter()
                .map(|(hz, q, gain_db)| piano_emulator::preset::BridgePeak { hz, q, gain_db })
                .collect(),
            radiated_share: 0.5,
        });
        preset.notes.duplex = vec![Vec::new(); NUM_KEYS];
        preset.notes.duplex[60] = vec![
            piano_emulator::preset::DuplexMode { hz: 3_121.5, gain_db: -22.5, t60_s: 1.2 },
            piano_emulator::preset::DuplexMode { hz: 4_703.0, gain_db: -27.0, t60_s: 0.8 },
        ];
        preset.notes.pan_spread = (0..NUM_KEYS).map(|i| 0.02 + 0.002 * i as f32).collect();
        // The five fields of this milestone: a per-key comb floor, two ragged
        // per-partial tables (one short row, one empty, one full) and the
        // hammer's own noise with a band limit the other four events do not
        // have.
        preset.notes.comb_floor[0] = 0.18;
        preset.notes.comb_floor[39] = 0.05;
        preset.notes.partial_gains = vec![Vec::new(); NUM_KEYS];
        preset.notes.partial_gains[39] = vec![1.4, 0.7, 1.0, 2.5];
        preset.notes.partial_gains[0] = vec![0.5, 1.75];
        preset.notes.partial_sigma_scale = vec![Vec::new(); NUM_KEYS];
        preset.notes.partial_sigma_scale[39] = vec![0.6, 1.0, 1.9];
        // The two motion mechanisms: a within-string split on one key, and the
        // velocity law for the strike vector's direction.
        preset.notes.false_beat = vec![Vec::new(); NUM_KEYS];
        preset.notes.false_beat[39] = vec![
            piano_emulator::preset::FalseBeat { k: 1, hz: 1.11, db: -6.1 },
            piano_emulator::preset::FalseBeat { k: 3, hz: 0.74, db: -6.6 },
        ];
        preset.voicing.strike_direction = Some(piano_emulator::preset::StrikeDirection {
            vh_db_at_pp: -2.5,
            vh_db_at_ff: 3.75,
            share_tilt: 0.06,
        });
        preset.noise.strike = piano_emulator::preset::StrikeNoise {
            centroid_hz: 1_450.0,
            decay_s: 0.06,
            bandwidth_hz: 7_000.0,
            velocity_db: 31.0,
            level_db: vec![
                piano_emulator::preset::NoiseAnchor { key: 21, db: -16.5 },
                piano_emulator::preset::NoiseAnchor { key: 60, db: -20.0 },
                piano_emulator::preset::NoiseAnchor { key: 96, db: -10.5 },
            ],
        };
        let text = preset.to_toml();
        assert!(text.contains("comb_floor"));
        assert!(text.contains("partial_gains"));
        assert!(text.contains("partial_sigma_scale"));
        assert!(text.contains("[noise.strike]"));
        assert!(text.contains("inharmonicity_b4"));
        assert!(text.contains("contact_width"));
        assert!(text.contains("polarization_pan_spread"));
        assert!(text.contains("unison_sigma_scale"));
        assert!(text.contains("[noise."));
        assert!(text.contains("voicing.bridge"));
        assert!(text.contains("duplex"));
        assert!(text.contains("pan_spread = ["));
        assert!(text.contains("false_beat"));
        assert!(text.contains("[voicing.strike_direction]"));

        let ours = Preset::from_toml(&text).expect("the tuner reads it");
        assert_eq!(ours.to_toml(), text);
        assert_eq!(ours.notes.inharmonicity_b4[40], 2.5e-7);
        assert_eq!(ours.voicing.polarization_pan_spread, 0.25);
        assert_eq!(ours.voicing.unison_sigma_scale[2].scale, vec![0.75, 1.0, 1.25]);
        assert_eq!(ours.noise.key_off.centroid_hz, 210.0);
        let bridge = ours.voicing.bridge.as_ref().expect("the bridge came through");
        assert_eq!(bridge.backbone.len(), 4);
        assert_eq!(bridge.peaks[1].q, 30.0);
        assert_eq!(ours.duplex_modes(81).len(), 2);
        assert_eq!(ours.duplex_modes(81)[0].hz, 3_121.5);
        assert_eq!(ours.duplex_modes(21).len(), 0);
        assert_eq!(ours.notes.pan_spread[0], 0.02);
        assert_eq!(ours.notes.comb_floor[0], 0.18);
        assert_eq!(ours.notes.partial_gains[39], vec![1.4, 0.7, 1.0, 2.5]);
        assert!(ours.notes.partial_gains[1].is_empty());
        assert_eq!(ours.notes.partial_sigma_scale[39], vec![0.6, 1.0, 1.9]);
        assert_eq!(ours.notes.false_beat[39].len(), 2);
        assert_eq!(ours.notes.false_beat[39][0].hz, 1.11);
        assert_eq!(ours.notes.false_beat[39][1].k, 3);
        assert!(ours.notes.false_beat[40].is_empty());
        let direction = ours
            .voicing
            .strike_direction
            .expect("the strike direction came through");
        assert_eq!(direction.vh_db_at_pp, -2.5);
        assert_eq!(direction.vh_db_at_ff, 3.75);
        assert_eq!(direction.share_tilt, 0.06);
        assert_eq!(ours.noise.strike.bandwidth_hz, 7_000.0);
        assert_eq!(ours.noise.strike.level_db.len(), 3);
        assert!(ours.validate().is_ok(), "{:?}", ours.validate().err());

        // And the two crates agree on the derived quantities the schema's
        // safety rests on, not merely on the numbers in the file: `max|B|`
        // comes out of a *fitted* shelf cascade, so a mirror that drifted would
        // let a preset through one crate and not the other.
        let theirs = piano_emulator::resonance::BridgeFilter::new(
            preset.voicing.bridge.as_ref().unwrap(),
        );
        let mine = crate::response::BridgeResponse::of(ours.voicing.bridge.as_ref());
        assert_eq!(mine.max_magnitude(), theirs.max_magnitude());
        for hz in [20.0f32, 58.0, 188.0, 440.0, 1_100.0, 3_800.0, 12_000.0] {
            assert_eq!(mine.magnitude(f64::from(hz)) as f32, theirs.magnitude(hz));
        }
        for hz in [3_121.5f32, 3_130.0, 4_703.0, 9_000.0] {
            assert_eq!(
                crate::response::duplex_magnitude(ours.duplex_modes(81), hz),
                piano_emulator::duplex::magnitude(preset.duplex_modes(81), hz)
            );
        }
    }

    /// A per-key spread table with one entry out of range.
    fn broken_spread(value: f32) -> Vec<f32> {
        let mut table = vec![0.1; NUM_KEYS];
        table[9] = value;
        table
    }

    fn bridge(backbone: &[(f32, f32)], peaks: &[(f32, f32, f32)]) -> BridgeVoicing {
        BridgeVoicing {
            backbone: backbone
                .iter()
                .map(|&(hz, gain_db)| BridgeAnchor { hz, gain_db })
                .collect(),
            peaks: peaks
                .iter()
                .map(|&(hz, q, gain_db)| BridgePeak { hz, q, gain_db })
                .collect(),
            radiated_share: 0.0,
        }
    }

    /// A well-formed per-partial row on one key, everything else neutral.
    fn ragged(rows: usize, key_index: usize, row: Vec<f32>) -> Vec<Vec<f32>> {
        let mut table = vec![Vec::new(); rows];
        if let Some(slot) = table.get_mut(key_index) {
            *slot = row;
        }
        table
    }

    /// A well-formed duplex table with one segment on one key broken.
    fn duplex_table(break_it: impl Fn(&mut DuplexMode)) -> Vec<Vec<DuplexMode>> {
        let mut table = vec![Vec::new(); NUM_KEYS];
        let mut mode = DuplexMode {
            hz: 4_213.0,
            gain_db: -28.0,
            t60_s: 1.1,
        };
        break_it(&mut mode);
        table[70] = vec![mode];
        table
    }

    /// A well-formed split table with the one entry broken.
    fn split_with(break_it: impl Fn(&mut FalseBeat)) -> Vec<Vec<FalseBeat>> {
        let mut entry = FalseBeat {
            k: 2,
            hz: 1.0,
            db: -6.0,
        };
        break_it(&mut entry);
        let mut table = vec![Vec::new(); NUM_KEYS];
        table[39] = vec![entry];
        table
    }

    #[test]
    fn both_crates_refuse_the_same_broken_voicing() {
        let breakages: [fn(&mut Preset); 110] = [
            // The two motion mechanisms, on the same bounds the engine states.
            |p| p.notes.false_beat = vec![Vec::new(); NUM_KEYS - 1],
            |p| p.notes.false_beat = split_with(|e| e.k = 0),
            |p| p.notes.false_beat = split_with(|e| e.hz = 0.05),
            |p| p.notes.false_beat = split_with(|e| e.hz = 4.0),
            |p| p.notes.false_beat = split_with(|e| e.db = 1.0),
            |p| p.notes.false_beat = split_with(|e| e.db = f32::NAN),
            |p| {
                p.voicing.strike_direction = Some(StrikeDirection {
                    vh_db_at_pp: 20.0,
                    vh_db_at_ff: 0.0,
                    share_tilt: 0.0,
                })
            },
            |p| {
                p.voicing.strike_direction = Some(StrikeDirection {
                    vh_db_at_pp: 0.0,
                    vh_db_at_ff: 0.0,
                    share_tilt: 0.5,
                })
            },
            |p| p.voicing.horizontal_offset_hz[1] = f32::NAN,
            |p| p.voicing.horizontal_offset_hz[0] = -100.0,
            |p| p.voicing.unison_coupling = f32::NAN,
            |p| p.voicing.unison_coupling = 1.0,
            |p| p.voicing.resonance_coupling = f32::NAN,
            |p| p.voicing.resonance_coupling = -0.01,
            |p| p.voicing.unison_layout[2].detune[1] = f32::NAN,
            |p| p.voicing.unison_layout[2].detune[0] = -1.5,
            |p| p.voicing.unison_layout[2].share[1] = f32::NAN,
            |p| p.voicing.unison_layout[1].share[0] = -0.1,
            |p| p.voicing.damper_weight[1].hz = f32::NAN,
            // Anchors out of ascending order: the engine interpolates between
            // neighbours and would read the wrong pair.
            |p| p.voicing.damper_weight[2].hz = 100.0,
            // The refinements, each of which reaches a mode frequency, an
            // excitation gain, a pan or a biquad on the audio thread.
            |p| p.notes.inharmonicity_b4[0] = f32::NAN,
            // A fourth-order coefficient large enough to fold A0's own series
            // back down, and one large enough to take it under the root.
            |p| p.notes.inharmonicity_b4[0] = -1.0e-6,
            |p| p.notes.inharmonicity_b4[0] = -2.0,
            |p| p.notes.inharmonicity_b4.pop().map(|_| ()).unwrap_or_default(),
            |p| p.notes.contact_width[3] = f32::NAN,
            |p| p.notes.contact_width[3] = 0.06,
            |p| p.notes.contact_width[3] = -0.01,
            |p| p.notes.contact_width.truncate(87),
            |p| p.voicing.polarization_pan_spread = f32::NAN,
            |p| p.voicing.polarization_pan_spread = 0.5,
            |p| p.voicing.polarization_pan_spread = -0.1,
            |p| p.voicing.unison_sigma_scale[2].scale[1] = f32::NAN,
            // Inside the bounds, but the row no longer redistributes the
            // note's damping — it retunes the whole compass's T60.
            |p| p.voicing.unison_sigma_scale[2].scale[1] = 1.5,
            |p| p.voicing.unison_sigma_scale[1].scale = vec![0.4, 1.6],
            |p| p.voicing.unison_sigma_scale[1].scale = vec![1.0],
            |p| p.voicing.unison_sigma_scale.truncate(2),
            |p| p.noise.key_off.centroid_hz = f32::NAN,
            |p| p.noise.key_off.centroid_hz = 30_000.0,
            |p| p.noise.key_off.decay_s = 0.0,
            |p| p.noise.key_off.decay_s = 20.0,
            |p| p.noise.pedal_down.velocity_db = f32::NAN,
            |p| p.noise.pedal_up.level_db.clear(),
            |p| p.noise.pedal_up.level_db[0].db = 3.0,
            |p| p.noise.key_off.level_db[1].key = 12,
            // Anchors out of ascending order, on the other side of the file.
            |p| p.noise.key_off.level_db[1].key = 100,
            // The per-key stereo spread: all 88 or none, each inside the range
            // the global scalar is held to, because each is a pan position.
            |p| p.notes.pan_spread = vec![0.1; NUM_KEYS - 1],
            |p| p.notes.pan_spread = vec![0.1; NUM_KEYS + 1],
            |p| p.notes.pan_spread = broken_spread(MAX_PAN_SPREAD + 0.01),
            |p| p.notes.pan_spread = broken_spread(-0.01),
            |p| p.notes.pan_spread = broken_spread(f32::NAN),
            // The bridge admittance. Shape first: a backbone needs at least
            // two anchors and they are interpolated in order, and every gain,
            // frequency and Q reaches a biquad on the audio thread.
            |p| p.voicing.bridge = Some(bridge(&[(100.0, 0.0)], &[])),
            |p| {
                p.voicing.bridge = Some(bridge(
                    &(0..=MAX_BRIDGE_ANCHORS)
                        .map(|i| (30.0 * 1.2f32.powi(i as i32), 0.0))
                        .collect::<Vec<_>>(),
                    &[],
                ))
            },
            |p| p.voicing.bridge = Some(bridge(&[(100.0, 0.0), (100.0, 1.0)], &[])),
            |p| p.voicing.bridge = Some(bridge(&[(400.0, 0.0), (100.0, 1.0)], &[])),
            |p| p.voicing.bridge = Some(bridge(&[(MIN_BRIDGE_HZ - 1.0, 0.0), (400.0, 0.0)], &[])),
            |p| p.voicing.bridge = Some(bridge(&[(100.0, 0.0), (MAX_BRIDGE_HZ + 1.0, 0.0)], &[])),
            |p| p.voicing.bridge = Some(bridge(&[(100.0, f32::NAN), (400.0, 0.0)], &[])),
            |p| {
                p.voicing.bridge =
                    Some(bridge(&[(100.0, MIN_BRIDGE_GAIN_DB - 1.0), (400.0, 0.0)], &[]))
            },
            |p| {
                p.voicing.bridge =
                    Some(bridge(&[(100.0, MAX_BRIDGE_GAIN_DB + 1.0), (400.0, 0.0)], &[]))
            },
            |p| {
                p.voicing.bridge = Some(bridge(
                    &[(100.0, 0.0), (400.0, 0.0)],
                    &(0..=MAX_BRIDGE_PEAKS)
                        .map(|i| (200.0 + 3.0 * i as f32, 5.0, 1.0))
                        .collect::<Vec<_>>(),
                ))
            },
            |p| p.voicing.bridge = Some(bridge(&[(100.0, 0.0), (400.0, 0.0)], &[(250.0, 0.0, 3.0)])),
            |p| {
                p.voicing.bridge = Some(bridge(
                    &[(100.0, 0.0), (400.0, 0.0)],
                    &[(250.0, MAX_BRIDGE_Q + 1.0, 3.0)],
                ))
            },
            |p| {
                p.voicing.bridge = Some(bridge(
                    &[(100.0, 0.0), (400.0, 0.0)],
                    &[(250.0, 10.0, f32::NAN)],
                ))
            },
            |p| {
                p.voicing.bridge = Some(bridge(
                    &[(100.0, 0.0), (400.0, 0.0)],
                    &[(MAX_BRIDGE_HZ + 1.0, 10.0, 3.0)],
                ))
            },
            // Well formed, every number in range, and past the loop bound: the
            // check both crates have to compute from the *realised* cascade
            // rather than read off the file. Eight stacked +20 dB peaks on one
            // frequency realise +160 dB, which nothing in the file says.
            |p| {
                p.voicing.resonance_coupling = MAX_RESONANCE_COUPLING;
                p.voicing.bridge = Some(bridge(
                    &[(100.0, 0.0), (400.0, 0.0)],
                    &[(1_000.0, 8.0, MAX_BRIDGE_GAIN_DB); 8],
                ));
            },
            // A backbone with no peaks at all, lifted bodily: +20 dB of mean
            // mobility against the ceiling coupling is 0.5, twice the bound.
            |p| {
                p.voicing.resonance_coupling = MAX_RESONANCE_COUPLING;
                p.voicing.bridge = Some(bridge(
                    &[(100.0, MAX_BRIDGE_GAIN_DB), (10_000.0, MAX_BRIDGE_GAIN_DB)],
                    &[],
                ));
            },
            // The comb floor: 88 entries, each a fraction of the comb's crest
            // that a real hammer on a real string could plausibly miss a node
            // by. Every one of them reaches an excitation gain on the audio
            // thread through `sqrt(sin^2 + floor^2)`.
            |p| p.notes.comb_floor[3] = f32::NAN,
            |p| p.notes.comb_floor[3] = MAX_COMB_FLOOR + 0.01,
            |p| p.notes.comb_floor[3] = -0.01,
            |p| p.notes.comb_floor.truncate(87),
            |p| p.notes.comb_floor.push(0.0),
            // The two ragged per-partial tables. All 88 rows or none; a row may
            // be short or empty, and a row longer than the key's own partial
            // count is a table measured on a different instrument. C8 (index 87)
            // has two partials at 48 kHz, so nine entries there is the case the
            // engine names the key for.
            |p| p.notes.partial_gains = vec![Vec::new(); NUM_KEYS - 1],
            |p| p.notes.partial_gains = vec![vec![1.0]; NUM_KEYS + 1],
            |p| p.notes.partial_gains = ragged(NUM_KEYS, 87, vec![1.0; 9]),
            |p| p.notes.partial_gains = ragged(NUM_KEYS, 0, vec![f32::NAN]),
            |p| p.notes.partial_gains = ragged(NUM_KEYS, 0, vec![MIN_PARTIAL_GAIN - 0.01]),
            |p| p.notes.partial_gains = ragged(NUM_KEYS, 0, vec![MAX_PARTIAL_GAIN + 0.01]),
            |p| p.notes.partial_gains = ragged(NUM_KEYS, 0, vec![0.0]),
            |p| p.notes.partial_gains = ragged(NUM_KEYS, 0, vec![-1.0]),
            |p| p.notes.partial_sigma_scale = vec![Vec::new(); NUM_KEYS - 1],
            |p| p.notes.partial_sigma_scale = vec![vec![1.0]; NUM_KEYS + 1],
            |p| p.notes.partial_sigma_scale = ragged(NUM_KEYS, 87, vec![1.0; 9]),
            |p| p.notes.partial_sigma_scale = ragged(NUM_KEYS, 0, vec![f32::NAN]),
            |p| {
                p.notes.partial_sigma_scale =
                    ragged(NUM_KEYS, 0, vec![MIN_PARTIAL_SIGMA_SCALE - 0.01])
            },
            |p| {
                p.notes.partial_sigma_scale =
                    ragged(NUM_KEYS, 0, vec![MAX_PARTIAL_SIGMA_SCALE + 0.01])
            },
            // A scale of zero is a pole on the unit circle: a partial that never
            // stops, damper or no damper.
            |p| p.notes.partial_sigma_scale = ragged(NUM_KEYS, 0, vec![0.0]),
            // The hammer's own noise: the fifth `[noise]` event, with a decay
            // range of its own and a band limit the other four do not have.
            |p| p.noise.strike.centroid_hz = f32::NAN,
            |p| p.noise.strike.decay_s = MIN_STRIKE_DECAY_S - 0.001,
            |p| p.noise.strike.decay_s = MAX_STRIKE_DECAY_S + 0.001,
            |p| p.noise.strike.bandwidth_hz = MIN_STRIKE_BANDWIDTH_HZ - 1.0,
            |p| p.noise.strike.bandwidth_hz = MAX_STRIKE_BANDWIDTH_HZ + 1.0,
            |p| p.noise.strike.velocity_db = f32::NAN,
            |p| p.noise.strike.level_db[0].db = 3.0,
            // In range, both of them, and the burst is centred outside its own
            // band — which nothing in either field says on its own.
            |p| {
                p.noise.strike.centroid_hz = 4_000.0;
                p.noise.strike.bandwidth_hz = 1_000.0;
            },
            // The duplex table: 88 rows or none, at most six per row, and
            // every number inside a range that keeps it a resonator.
            |p| p.notes.duplex = vec![Vec::new(); NUM_KEYS - 1],
            |p| {
                p.notes.duplex = duplex_table(|_| {});
                p.notes.duplex[7] = (0..=MAX_DUPLEX_MODES)
                    .map(|k| DuplexMode { hz: 4_000.0 + 11.0 * k as f32, gain_db: -30.0, t60_s: 1.0 })
                    .collect();
            },
            |p| p.notes.duplex = duplex_table(|m| m.hz = MIN_DUPLEX_HZ - 1.0),
            |p| p.notes.duplex = duplex_table(|m| m.hz = MAX_DUPLEX_HZ + 1.0),
            |p| p.notes.duplex = duplex_table(|m| m.hz = f32::NAN),
            |p| p.notes.duplex = duplex_table(|m| m.gain_db = MAX_DUPLEX_GAIN_DB + 1.0),
            |p| p.notes.duplex = duplex_table(|m| m.gain_db = MIN_DUPLEX_GAIN_DB - 1.0),
            |p| p.notes.duplex = duplex_table(|m| m.gain_db = f32::NAN),
            |p| p.notes.duplex = duplex_table(|m| m.t60_s = 0.0),
            |p| p.notes.duplex = duplex_table(|m| m.t60_s = MAX_DUPLEX_T60_S + 1.0),
            |p| p.notes.duplex = duplex_table(|m| m.t60_s = f32::NAN),
            // Well formed, in range, and past the undamped loop bound: six
            // segments at one frequency on all 88 keys, which nothing damps.
            // `PHYSICS.md` §3's point about scatter, as a rejection.
            |p| {
                p.voicing.resonance_coupling = MAX_RESONANCE_COUPLING;
                p.notes.duplex = vec![
                    vec![
                        DuplexMode {
                            hz: 4_000.0,
                            gain_db: MAX_DUPLEX_GAIN_DB,
                            t60_s: MAX_DUPLEX_T60_S,
                        };
                        MAX_DUPLEX_MODES
                    ];
                    NUM_KEYS
                ];
            },
            // The three tables that decide where a mode *lands*, each of which
            // both crates used to accept and the engine then panicked on inside
            // the eigensolve (`DECISIONS.md` 257): a fundamental past the cap
            // the series stops at, a spread that is not a unison, and a decay
            // rate under the floor the pole radius rounds at.
            |p| p.notes.f0_hz[NUM_KEYS - 1] = 25_000.0,
            |p| p.notes.detune_cents[0] = MAX_DETUNE_CENTS + 0.1,
            |p| p.notes.sigma0[3] = 1.0e-44,
            |p| p.notes.sigma0[3] = MIN_MODE_SIGMA * 0.5,
        ];
        for (i, break_it) in breakages.into_iter().enumerate() {
            let mut preset = default_preset();
            break_it(&mut preset);
            assert!(preset.validate().is_err(), "the tuner accepted breakage {i}");
            let text = preset.to_toml();
            assert!(
                piano_emulator::Preset::from_toml(&text).is_err(),
                "the engine accepted breakage {i}"
            );
        }
        // ... and the unbroken preset passes both, so the loop above is not
        // rejecting everything for an unrelated reason.
        let text = default_preset().to_toml();
        assert!(piano_emulator::Preset::from_toml(&text).is_ok());
    }

    #[test]
    fn the_default_preset_is_equal_tempered() {
        let preset = default_preset();
        assert!((f64::from(preset.f0(69).unwrap()) - 440.0).abs() < 1e-3);
        for key in LOWEST_KEY..=HIGHEST_KEY {
            assert!(preset.stretch_cents(key).unwrap().abs() < 0.01, "key {key}");
        }
    }

    #[test]
    fn estimates_reach_the_table_and_the_notes_between_them_are_interpolated() {
        let base = default_preset();
        let measured = [(48u8, 2.5e-4), (60, 4.4e-4), (72, 7.0e-4)];
        let mut builder = PresetBuilder::new(base.clone()).name("test");
        for (key, b) in measured {
            builder = builder.note(NoteEstimate {
                inharmonicity_b: Some(b),
                ..NoteEstimate::new(key)
            });
        }
        let preset = builder.build().unwrap();

        for (key, b) in measured {
            let index = key_index(key).unwrap();
            assert!(
                (f64::from(preset.notes.inharmonicity_b[index]) / b - 1.0).abs() < 1e-6,
                "measured key {key} did not survive"
            );
        }
        // Between two measurements the curve stays between them ...
        let between = preset.notes.inharmonicity_b[key_index(66).unwrap()];
        assert!(between > 4.4e-4 && between < 7.0e-4, "{between}");
        // ... and every other table is untouched.
        assert_eq!(preset.notes.impedance, base.notes.impedance);
        assert_eq!(preset.notes.bridge_gain, base.notes.bridge_gain);
        assert_eq!(preset.soundboard, base.soundboard);
        assert_eq!(preset.name, "test");
    }

    #[test]
    fn past_the_last_measurement_the_base_curve_takes_over() {
        // Three measured keys in the middle of the compass, each twice the base
        // preset's `B`. Between them the interpolant; outside them the base
        // table scaled onto the nearest measurement, which for data that is a
        // uniform doubling means the whole table doubles.
        let base = default_preset();
        let mut builder = PresetBuilder::new(base.clone());
        for key in [48u8, 60, 72] {
            let index = key_index(key).unwrap();
            builder = builder.note(NoteEstimate {
                inharmonicity_b: Some(2.0 * f64::from(base.notes.inharmonicity_b[index])),
                ..NoteEstimate::new(key)
            });
        }
        let preset = builder.build().unwrap();
        for key in [LOWEST_KEY, 30, 47, 73, 90, HIGHEST_KEY] {
            let index = key_index(key).unwrap();
            let ratio = f64::from(preset.notes.inharmonicity_b[index])
                / f64::from(base.notes.inharmonicity_b[index]);
            assert!((ratio - 2.0).abs() < 1e-5, "key {key}: ratio {ratio}");
        }
        // The end slope of the interpolant would have carried on rising: the
        // top of the compass is where the base curve climbs an order of
        // magnitude, and following its shape is what keeps C8 a piano note.
        let top = f64::from(preset.notes.inharmonicity_b[key_index(HIGHEST_KEY).unwrap()]);
        assert!(top < 1.0, "B at C8 ran away to {top}");
    }

    #[test]
    fn a_measured_stretch_is_written_as_the_tuning() {
        // A Railsback-ish stretch: 30 cents flat at A0, 20 sharp at C8.
        let stretch = |key: u8| -0.03 * (f64::from(key) - 90.0).powi(2) / 20.0;
        let mut builder = PresetBuilder::new(default_preset());
        for key in (21u8..=108).step_by(3) {
            builder = builder.note(NoteEstimate {
                f0_hz: Some(equal_temperament(key) * (stretch(key) / 1200.0).exp2()),
                ..NoteEstimate::new(key)
            });
        }
        let preset = builder.build().unwrap();
        for key in LOWEST_KEY..=HIGHEST_KEY {
            let measured = preset.stretch_cents(key).unwrap();
            assert!(
                (measured - stretch(key)).abs() < 0.05,
                "key {key}: {measured} vs {}",
                stretch(key)
            );
        }
    }

    #[test]
    fn one_measured_note_moves_the_base_curve_rather_than_flattening_it() {
        let base = default_preset();
        let index = key_index(60).unwrap();
        let measured = f64::from(base.notes.inharmonicity_b[index]) * 1.5;
        let preset = PresetBuilder::new(base.clone())
            .note(NoteEstimate {
                inharmonicity_b: Some(measured),
                // A4 measured at 442: the instrument is tuned two hertz sharp,
                // which says nothing about its stretch.
                f0_hz: None,
                ..NoteEstimate::new(60)
            })
            .build()
            .unwrap();
        assert!(
            (f64::from(preset.notes.inharmonicity_b[index]) / measured - 1.0).abs() < 1e-6,
            "the measured note did not survive"
        );
        for (estimated, original) in preset
            .notes
            .inharmonicity_b
            .iter()
            .zip(&base.notes.inharmonicity_b)
        {
            let ratio = f64::from(*estimated) / f64::from(*original);
            assert!((ratio - 1.5).abs() < 1e-5, "the curve's shape changed: {ratio}");
        }

        let retuned = PresetBuilder::new(base.clone())
            .note(NoteEstimate {
                f0_hz: Some(442.0),
                ..NoteEstimate::new(69)
            })
            .build()
            .unwrap();
        for key in LOWEST_KEY..=HIGHEST_KEY {
            let cents = retuned.stretch_cents(key).unwrap();
            assert!((cents - 7.85).abs() < 0.01, "key {key}: {cents} cents");
        }
    }

    #[test]
    fn a_measured_beat_is_widened_to_the_groups_full_spread() {
        let base = default_preset();
        // C4 is a triple. Its loudest pair spans 0.61 of the full width in the
        // default layout, so a beat measured at 1.76 cents is a 2.9-cent unison.
        let fraction = base.voicing.dominant_beat_fraction(3);
        assert!((fraction - 0.61).abs() < 1e-6, "{fraction}");
        assert_eq!(base.voicing.dominant_beat_fraction(2), 1.0);
        assert_eq!(base.voicing.dominant_beat_fraction(1), 0.0);

        let preset = PresetBuilder::new(base.clone())
            .note(NoteEstimate {
                detune_cents: Some(1.76),
                ..NoteEstimate::new(60)
            })
            .build()
            .unwrap();
        let measured = f64::from(preset.notes.detune_cents[key_index(60).unwrap()]);
        assert!((measured - 1.76 / 0.61).abs() < 0.01, "{measured} cents");

        // A single-strung bass note cannot beat, so an estimate there is
        // dropped rather than divided by a zero-wide pair.
        let untouched = PresetBuilder::new(base.clone())
            .note(NoteEstimate {
                detune_cents: Some(1.0),
                ..NoteEstimate::new(24)
            })
            .build()
            .unwrap();
        assert_eq!(untouched.notes.detune_cents, base.notes.detune_cents);
    }

    #[test]
    fn the_builder_refuses_to_write_a_preset_the_engine_would_reject() {
        let builder = PresetBuilder::new(default_preset()).note(NoteEstimate {
            // Sigma clamps at a floor rather than going negative, but an
            // exponent outside the felt model's range must be caught.
            hammer_exponent: Some(f64::NAN),
            ..NoteEstimate::new(60)
        });
        assert!(builder.build().is_err());

        let off_the_keyboard = PresetBuilder::new(default_preset()).note(NoteEstimate::new(12));
        assert!(off_the_keyboard.build().is_err());
    }

    #[test]
    fn the_global_voicing_estimates_land_in_the_right_fields() {
        let preset = PresetBuilder::new(default_preset())
            .polarization(PolarizationSplit {
                gain_db: -10.5,
                decay_ratio: 0.31,
                partials: 12,
            })
            .velocity_map(VelocityMap {
                velocity_min: 0.25,
                velocity_max: 5.5,
                residual: 0.0,
            })
            .build()
            .unwrap();
        assert_eq!(preset.voicing.horizontal_gain_db, -10.5);
        assert_eq!(preset.voicing.horizontal_decay_ratio, 0.31);
        assert_eq!(preset.hammer.velocity_min, 0.25);
        assert_eq!(preset.hammer.velocity_max, 5.5);
    }

    #[test]
    fn the_refinements_reach_their_own_tables_and_fields() {
        let base = default_preset();
        let preset = PresetBuilder::new(base.clone())
            .note(NoteEstimate {
                inharmonicity_b4: Some(-3.0e-9),
                contact_width: Some(0.018),
                ..NoteEstimate::new(36)
            })
            .note(NoteEstimate {
                inharmonicity_b4: Some(0.0),
                contact_width: Some(0.012),
                ..NoteEstimate::new(72)
            })
            .sigma_scale(vec![
                UnisonSigmaScale { scale: vec![1.0] },
                UnisonSigmaScale {
                    scale: vec![0.85, 1.15],
                },
                UnisonSigmaScale {
                    scale: vec![0.85, 1.0, 1.15],
                },
            ])
            .pan_spread(0.22)
            .noise(NoiseTables {
                key_off: EventNoise {
                    centroid_hz: 205.0,
                    ..NoiseTables::default().key_off
                },
                ..NoiseTables::default()
            })
            .build()
            .unwrap();
        assert_eq!(preset.notes.inharmonicity_b4[key_index(36).unwrap()], -3.0e-9);
        assert_eq!(preset.notes.inharmonicity_b4[key_index(72).unwrap()], 0.0);
        assert!((preset.notes.contact_width[key_index(36).unwrap()] - 0.018).abs() < 1e-9);
        // Between the two measured notes the interpolant; past them the
        // measured value held, because the base table for a correction is zero
        // everywhere and there is no shape to carry on.
        let between = preset.notes.contact_width[key_index(54).unwrap()];
        assert!(between > 0.012 && between < 0.018, "{between}");
        assert_eq!(preset.notes.contact_width[key_index(21).unwrap()], 0.018);
        assert_eq!(preset.voicing.polarization_pan_spread, 0.22);
        assert_eq!(preset.voicing.unison_sigma_scale[1].scale, vec![0.85, 1.15]);
        assert_eq!(preset.noise.key_off.centroid_hz, 205.0);
        // ... and a preset the engine will play.
        assert!(piano_emulator::Preset::from_toml(&preset.to_toml()).is_ok());
    }

    #[test]
    fn a_fourth_order_coefficient_that_would_fold_a_series_is_shrunk_not_refused() {
        // A0 is built with eighty partials, so the negative coefficient its own
        // measured partials 1-26 support takes the top of that bank under the
        // root. Interpolated onto a key nobody measured, that would refuse a
        // whole survey's preset at the last step; it is shrunk towards the
        // two-parameter law instead, which is what the key had before.
        let preset = PresetBuilder::new(default_preset())
            .note(NoteEstimate {
                inharmonicity_b4: Some(-1.0e-6),
                ..NoteEstimate::new(21)
            })
            .note(NoteEstimate {
                inharmonicity_b4: Some(-1.0e-6),
                ..NoteEstimate::new(108)
            })
            .build()
            .expect("a preset, not an error");
        let b4 = preset.notes.inharmonicity_b4[0];
        assert!(b4 < 0.0 && b4 > -1.0e-6, "A0 kept {b4}");
        assert!(piano_emulator::Preset::from_toml(&preset.to_toml()).is_ok());
    }

    #[test]
    fn a_preset_written_and_read_back_is_the_same_preset() {
        let preset = PresetBuilder::new(default_preset())
            .name("round-trip")
            .note(NoteEstimate {
                inharmonicity_b: Some(3.3e-4),
                sigma0: Some(0.55),
                sigma1: Some(0.6),
                detune_cents: Some(2.7),
                strike_position: Some(0.117),
                ..NoteEstimate::new(60)
            })
            .build()
            .unwrap();
        let back = Preset::from_toml(&preset.to_toml()).unwrap();
        assert_eq!(preset, back);
    }
}
