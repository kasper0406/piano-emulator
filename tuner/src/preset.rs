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
/// Largest displacement between the two polarizations of one key, either side
/// of that key's pan (`engine::soundboard::MAX_PAN_SPREAD`). The engine's
/// `MAX_PAN + MAX_PAN_SPREAD` is 1, so at the ceiling the outer polarization of
/// the outermost key sits exactly hard left or hard right.
pub const MAX_PAN_SPREAD: f32 = 0.4;
/// Shortest and longest a mechanism event may last, seconds
/// (`engine::preset::{MIN,MAX}_NOISE_DECAY_S`).
pub const MIN_NOISE_DECAY_S: f32 = 0.01;
pub const MAX_NOISE_DECAY_S: f32 = 10.0;
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
        for i in 0..NUM_KEYS {
            let layout = PartialLayout::of(self, i);
            let limit = (MAX_PARTIAL_RATIO * SAMPLE_RATE) as f32;
            let mut previous = 0.0f32;
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
                // never builds the partial.
                if f >= limit {
                    break;
                }
                previous = f;
            }
        }

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
            positive(&format!("noise.{name}.centroid_hz"), event.centroid_hz)?;
            within(
                &format!("noise.{name}.centroid_hz"),
                event.centroid_hz,
                1.0,
                (0.45 * SAMPLE_RATE) as f32,
            )?;
            within(
                &format!("noise.{name}.decay_s"),
                event.decay_s,
                MIN_NOISE_DECAY_S,
                MAX_NOISE_DECAY_S,
            )?;
            finite(&format!("noise.{name}.velocity_db"), event.velocity_db)?;
            if event.level_db.is_empty() {
                return Err(Error::Preset(format!("noise.{name}.level_db is empty")));
            }
            for (i, anchor) in event.level_db.iter().enumerate() {
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
            // The level is interpolated across the compass by walking the
            // anchors in order.
            if let Some(i) = event.level_db.windows(2).position(|w| w[0].key >= w[1].key) {
                return Err(Error::Preset(format!(
                    "noise.{name}.level_db[{}] is at key {}, not above the key {} before it",
                    i + 1,
                    event.level_db[i + 1].key,
                    event.level_db[i].key
                )));
            }
        }
        Ok(())
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

    fn partial_count(&self) -> u32 {
        let limit = (MAX_PARTIAL_RATIO * SAMPLE_RATE) as f32;
        (1..=MAX_PARTIALS)
            .take_while(|&k| self.partial_hz(k) < limit)
            .count()
            .max(1) as u32
    }
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
        let fields: [(&str, fn(&NoteEstimate) -> Option<f64>, bool, f64, f64); 10] = [
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
            ("sigma0", |n| n.sigma0, true, 1e-3, f64::MAX),
            ("sigma1", |n| n.sigma1, false, 0.0, f64::MAX),
            ("detune_cents", |n| n.detune_cents, true, 0.0, 50.0),
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
        let text = preset.to_toml();
        assert!(text.contains("inharmonicity_b4"));
        assert!(text.contains("contact_width"));
        assert!(text.contains("polarization_pan_spread"));
        assert!(text.contains("unison_sigma_scale"));
        assert!(text.contains("[noise."));

        let ours = Preset::from_toml(&text).expect("the tuner reads it");
        assert_eq!(ours.to_toml(), text);
        assert_eq!(ours.notes.inharmonicity_b4[40], 2.5e-7);
        assert_eq!(ours.voicing.polarization_pan_spread, 0.25);
        assert_eq!(ours.voicing.unison_sigma_scale[2].scale, vec![0.75, 1.0, 1.25]);
        assert_eq!(ours.noise.key_off.centroid_hz, 210.0);
    }

    #[test]
    fn both_crates_refuse_the_same_broken_voicing() {
        let breakages: [fn(&mut Preset); 37] = [
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
