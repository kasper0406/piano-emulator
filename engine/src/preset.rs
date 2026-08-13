//! The instrument's parameters, as data.
//!
//! Everything that voices the piano — the per-note tables and the global
//! voicing constants — lives in a [`Preset`] and is read from a TOML file. The
//! DSP modules hold no tuning numbers of their own: they take a `Preset` (or
//! the per-note views [`StringParams`] and [`HammerParams`] derived from it) at
//! construction and never look at anything else.
//!
//! [`Preset::default`] is the instrument as it was hand-tuned for v1, and
//! `presets/default.toml` is that same preset written out; loading it must
//! reproduce the built-in one exactly, which `preset::tests` pins.
//!
//! # Per-note tables, not curves
//!
//! Every per-note quantity is stored as 88 explicit values rather than as the
//! anchor curve it was originally written as. Automated estimation produces one
//! number per note — sometimes only for the notes it could measure, with the
//! rest interpolated — so a curve is the wrong shape for the file even though
//! it is the right shape for a human editing it. The curves survive as the code
//! that builds the default preset, which is also the natural place to write a
//! new preset by hand: edit the anchors, dump the preset, tune the numbers.
//!
//! # Numbers in the file
//!
//! The engine computes in `f32`, so the tables are `f32`. They are written out
//! as the shortest decimal that reads back as the same `f32`, so a value that
//! was typed as `0.35` stays `0.35` in the file instead of becoming
//! `0.34999999403953552`, and a full round trip through TOML is bit-exact.

use crate::hammer::HammerParams;
use crate::resonance::MAX_COUPLING;
use crate::string::{StringParams, MAX_UNISON_COUPLING};
use crate::types::{
    db_to_amp, index_to_note, interp_anchors, key_index, key_position, note_to_freq, MAX_UNISON,
    NUM_KEYS,
};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::Path;

/// A complete description of one instrument.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Preset {
    /// Short identifier, e.g. `default` or `salamander-c5`.
    pub name: String,
    /// Free-form provenance: which piano, which recordings, which pipeline run.
    pub description: String,
    pub voicing: Voicing,
    pub hammer: HammerVoicing,
    pub soundboard: SoundboardVoicing,
    pub notes: NoteTables,
}

/// Global string and coupling constants: the parts of the voicing that are not
/// per note.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Voicing {
    /// Turns the string's force on the bridge, in newtons, into the engine's
    /// internal signal unit. Purely gain staging.
    #[serde(serialize_with = "short::scalar")]
    pub excitation_scale: f32,
    /// Input gain of the horizontal polarization relative to the vertical one.
    #[serde(serialize_with = "short::scalar")]
    pub horizontal_gain_db: f32,
    /// Horizontal decay rate as a fraction of the vertical one — well below 1,
    /// which is what gives a piano note its double decay.
    #[serde(serialize_with = "short::scalar")]
    pub horizontal_decay_ratio: f32,
    /// Frequency offset of the horizontal polarization, one per unison string.
    /// The two planes see slightly different transverse stiffness, and no two
    /// strings of a group see the same difference.
    #[serde(serialize_with = "short::list")]
    pub horizontal_offset_hz: Vec<f32>,
    /// Bridge coupling within a unison group, as a fraction of the string's
    /// wave impedance.
    #[serde(serialize_with = "short::scalar")]
    pub unison_coupling: f32,
    /// Fraction of the sympathetic-resonance bus injected into each undamped
    /// string.
    #[serde(serialize_with = "short::scalar")]
    pub resonance_coupling: f32,
    /// One entry per unison group size, 1 to [`MAX_UNISON`] strings.
    pub unison_layout: Vec<UnisonLayout>,
    /// How firmly the damper felt grips a partial, as anchors interpolated in
    /// log frequency. Dampers hold low partials tightly and the top ones barely
    /// at all, which is the brief metallic zing on release.
    pub damper_weight: Vec<DamperAnchor>,
}

/// How the strings of one size of unison group are laid out.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UnisonLayout {
    /// Detuning of each string as a fraction of the note's detune width.
    ///
    /// Deliberately uneven. Evenly spaced detunings make the fundamentals
    /// coincide in antiphase at a fixed point of every beat cycle and cancel to
    /// nothing — the note is heard pumping to silence and back, which no piano
    /// does. With uneven spacing the beat rates are incommensurate and the
    /// cancellation never lines up.
    #[serde(serialize_with = "short::list")]
    pub detune: Vec<f32>,
    /// Share of the hammer's force each string receives. The hammer is not
    /// perfectly square to the strings, so the shares differ by a few percent;
    /// each row averages to 1, so the group's total excitation does not depend
    /// on how they are spread, and is paired with a detuning whose
    /// amplitude-weighted centre is nominal pitch.
    #[serde(serialize_with = "short::list")]
    pub share: Vec<f32>,
}

/// One point of the damper's frequency response.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DamperAnchor {
    #[serde(serialize_with = "short::scalar")]
    pub hz: f32,
    #[serde(serialize_with = "short::scalar")]
    pub weight: f32,
}

/// Hammer constants shared by the whole compass.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HammerVoicing {
    /// Hammer speed, m/s, at MIDI velocity 1 and 127. The mapping between them
    /// is exponential, so each MIDI step is a constant ratio.
    #[serde(serialize_with = "short::scalar")]
    pub velocity_min: f32,
    #[serde(serialize_with = "short::scalar")]
    pub velocity_max: f32,
    /// Hunt-Crossley hysteresis coefficient, s/m: the felt is stiffer while
    /// being compressed than while relaxing, so it returns less than it stored.
    #[serde(serialize_with = "short::scalar")]
    pub felt_hysteresis: f32,
    /// Stiffness multiplier under una corda — the hammer meets the strings off
    /// its worn centre line, where the felt is softer.
    #[serde(serialize_with = "short::scalar")]
    pub una_corda_stiffness: f32,
    /// Velocity reflection coefficient of the agraffe end of the speaking
    /// length. Below one because the termination is not rigid and string
    /// stiffness disperses the returning pulse.
    #[serde(serialize_with = "short::scalar")]
    pub reflection_gain: f32,
}

/// Soundboard, body and master-chain constants.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SoundboardVoicing {
    /// Direct/board balance: `out = (1 - mix) * panned + mix * board`.
    #[serde(serialize_with = "short::scalar")]
    pub board_mix: f32,
    /// Weight of the body-mode bank in the board's drive signal.
    #[serde(serialize_with = "short::scalar")]
    pub body_mix: f32,
    /// Broadband gain correction that makes the board path unity, so
    /// `board_mix` is a loudness-preserving crossfade.
    #[serde(serialize_with = "short::scalar")]
    pub board_level: f32,
    /// Master high shelf: corner frequency and asymptotic gain. Stands in for
    /// the drop in radiation efficiency above a few kHz.
    #[serde(serialize_with = "short::scalar")]
    pub shelf_hz: f32,
    #[serde(serialize_with = "short::scalar")]
    pub shelf_gain_db: f32,
    /// Reverberation time of the diffuse board field at DC and at
    /// `fdn_hf_hz`.
    #[serde(serialize_with = "short::scalar")]
    pub fdn_t60_lf: f32,
    #[serde(serialize_with = "short::scalar")]
    pub fdn_t60_hf: f32,
    #[serde(serialize_with = "short::scalar")]
    pub fdn_hf_hz: f32,
    /// Body modes. Their frequencies are kept off the equal-tempered grid so
    /// no single key is emphasised.
    pub body_modes: Vec<BodyMode>,
}

/// One resonance of the cabinet and soundboard.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BodyMode {
    #[serde(serialize_with = "short::scalar")]
    pub hz: f32,
    #[serde(serialize_with = "short::scalar")]
    pub q: f32,
    /// Peak gain relative to the board's drive signal.
    #[serde(serialize_with = "short::scalar")]
    pub gain: f32,
}

/// Per-note tables, one entry per key from A0 to C8, indexed by
/// [`crate::types::key_index`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NoteTables {
    /// Fundamental frequency, Hz. This is the tuning: the default table is
    /// equal temperament, and a stretch-tuned instrument writes the Railsback
    /// curve here. Every "what pitch is this key" question in the engine reads
    /// this table, not the equal-tempered formula.
    #[serde(serialize_with = "short::list")]
    pub f0_hz: Vec<f32>,
    /// Stiffness inharmonicity B in `f_k = k f0 sqrt(1 + B k^2)`.
    #[serde(serialize_with = "short::list")]
    pub inharmonicity_b: Vec<f32>,
    /// Hammer strike point as a fraction of the speaking length.
    #[serde(serialize_with = "short::list")]
    pub strike_position: Vec<f32>,
    /// Frequency-independent part of the partial decay rate, 1/s.
    #[serde(serialize_with = "short::list")]
    pub sigma0: Vec<f32>,
    /// Coefficient of `(f_k/1000)^2` in the partial decay rate, 1/s.
    #[serde(serialize_with = "short::list")]
    pub sigma1: Vec<f32>,
    /// Strings per note: 1, 2 or 3.
    pub unison: Vec<u8>,
    /// Full width of the unison detuning spread, in cents. Cents and not hertz:
    /// two strings of a unison differ in tension and `f_k ∝ sqrt(T)` for every
    /// partial at once, so the mistuning is a ratio.
    #[serde(serialize_with = "short::list")]
    pub detune_cents: Vec<f32>,
    /// Transverse wave impedance of one string, kg/s.
    #[serde(serialize_with = "short::list")]
    pub impedance: Vec<f32>,
    /// Extra decay rate applied by a fully engaged damper, 1/s.
    #[serde(serialize_with = "short::list")]
    pub damper_sigma: Vec<f32>,
    /// Fraction of this note's bridge force that becomes signal: the
    /// soundboard's coupling to this part of the compass, and the table that
    /// flattens the compass.
    #[serde(serialize_with = "short::list")]
    pub bridge_gain: Vec<f32>,
    /// Hammer head mass, kg.
    #[serde(serialize_with = "short::list")]
    pub hammer_mass: Vec<f32>,
    /// Felt stiffness K, in N/m^p — meaningful only together with the exponent
    /// of the same note.
    #[serde(serialize_with = "short::list")]
    pub hammer_stiffness: Vec<f32>,
    /// Felt nonlinearity exponent p.
    #[serde(serialize_with = "short::list")]
    pub hammer_exponent: Vec<f32>,
}

impl Preset {
    /// Reads and validates a preset file.
    pub fn load(path: &Path) -> Result<Preset, PresetError> {
        let text = std::fs::read_to_string(path).map_err(PresetError::Io)?;
        Preset::from_toml(&text)
    }

    pub fn from_toml(text: &str) -> Result<Preset, PresetError> {
        let preset: Preset = toml::from_str(text).map_err(PresetError::Parse)?;
        preset.validate()?;
        Ok(preset)
    }

    pub fn to_toml(&self) -> String {
        toml::to_string_pretty(self).expect("a preset is always serializable")
    }

    pub fn save(&self, path: &Path) -> Result<(), PresetError> {
        std::fs::write(path, self.to_toml()).map_err(PresetError::Io)
    }

    /// Checks the invariants the engine relies on.
    ///
    /// A preset file is untrusted input that goes straight into resonator
    /// coefficients and into an ODE integrated on the audio thread, and the DSP
    /// has no other guard: a negative decay rate is a pole outside the unit
    /// circle, a negative `B` puts a square root of a negative number in the
    /// partial layout, and a zero mass or stiffness divides by zero inside the
    /// hammer's contact model. All of that has to be refused here, with a
    /// message naming the field, rather than reached at note-on.
    pub fn validate(&self) -> Result<(), PresetError> {
        let n = &self.notes;
        // (name, table, must be strictly positive rather than just non-negative)
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
                return Err(PresetError::invalid(format!(
                    "notes.{name} has {} entries, expected {NUM_KEYS}",
                    table.len()
                )));
            }
            for (i, &v) in table.iter().enumerate() {
                if !v.is_finite() || v < 0.0 || (positive && v == 0.0) {
                    return Err(PresetError::invalid(format!(
                        "notes.{name}[{i}] is {v}, expected a finite {} number",
                        if positive { "positive" } else { "non-negative" }
                    )));
                }
            }
        }
        // A strike point outside the string is not a point on the string.
        if let Some(i) = n.strike_position.iter().position(|&x| x >= 1.0) {
            return Err(PresetError::invalid(format!(
                "notes.strike_position[{i}] is {}, expected 0 < x < 1",
                n.strike_position[i]
            )));
        }
        if n.unison.len() != NUM_KEYS {
            return Err(PresetError::invalid(format!(
                "notes.unison has {} entries, expected {NUM_KEYS}",
                n.unison.len()
            )));
        }
        for (i, &u) in n.unison.iter().enumerate() {
            if u == 0 || u as usize > MAX_UNISON {
                return Err(PresetError::invalid(format!(
                    "notes.unison[{i}] is {u}, expected 1..={MAX_UNISON}"
                )));
            }
        }

        let v = &self.voicing;
        // `excitation_scale` divides the unison bridge coupling, and the
        // horizontal polarization's decay is a fraction of the vertical one.
        positive("voicing.excitation_scale", v.excitation_scale)?;
        positive("voicing.horizontal_decay_ratio", v.horizontal_decay_ratio)?;
        finite("voicing.horizontal_gain_db", v.horizontal_gain_db)?;
        // Both couplings are loop gains: a string's own output comes back into
        // its excitation one block later, through its unison siblings and
        // through the resonance bus, and a loop that reaches unity sustains
        // itself until the state overflows to infinity and then to NaN.
        within("voicing.unison_coupling", v.unison_coupling, 0.0, MAX_UNISON_COUPLING)?;
        within("voicing.resonance_coupling", v.resonance_coupling, 0.0, MAX_COUPLING)?;
        if v.horizontal_offset_hz.len() != MAX_UNISON {
            return Err(PresetError::invalid(format!(
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
            return Err(PresetError::invalid(format!(
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
        if v.damper_weight.is_empty() {
            return Err(PresetError::invalid("voicing.damper_weight is empty"));
        }
        for anchor in &v.damper_weight {
            positive("voicing.damper_weight.hz", anchor.hz)?;
            finite("voicing.damper_weight.weight", anchor.weight)?;
        }
        // `damper_weight_at` walks the anchors in order and interpolates
        // between neighbours: out of order it reads the wrong pair, and two
        // anchors at one frequency divide by a zero span.
        if let Some(i) = v.damper_weight.windows(2).position(|w| w[0].hz >= w[1].hz) {
            return Err(PresetError::invalid(format!(
                "voicing.damper_weight[{}] is at {} Hz, not above the {} Hz before it",
                i + 1,
                v.damper_weight[i + 1].hz,
                v.damper_weight[i].hz
            )));
        }

        let h = &self.hammer;
        // The velocity map is a ratio between the two ends of its range.
        positive("hammer.velocity_min", h.velocity_min)?;
        positive("hammer.velocity_max", h.velocity_max)?;
        finite("hammer.felt_hysteresis", h.felt_hysteresis)?;
        positive("hammer.una_corda_stiffness", h.una_corda_stiffness)?;
        positive("hammer.reflection_gain", h.reflection_gain)?;

        let s = &self.soundboard;
        // The FDN's per-pass loss is set by 1 / T60, and each body mode's
        // bandwidth by f / Q.
        positive("soundboard.fdn_t60_lf", s.fdn_t60_lf)?;
        positive("soundboard.fdn_t60_hf", s.fdn_t60_hf)?;
        positive("soundboard.fdn_hf_hz", s.fdn_hf_hz)?;
        positive("soundboard.shelf_hz", s.shelf_hz)?;
        finite("soundboard.shelf_gain_db", s.shelf_gain_db)?;
        finite("soundboard.board_mix", s.board_mix)?;
        finite("soundboard.body_mix", s.body_mix)?;
        finite("soundboard.board_level", s.board_level)?;
        if s.body_modes.is_empty() {
            return Err(PresetError::invalid("soundboard.body_modes is empty"));
        }
        for mode in &s.body_modes {
            positive("soundboard.body_modes.hz", mode.hz)?;
            positive("soundboard.body_modes.q", mode.q)?;
            finite("soundboard.body_modes.gain", mode.gain)?;
        }
        Ok(())
    }

    /// The string parameters of one key. Panics if `key` is outside A0..C8 —
    /// callers hold a real key.
    pub fn string_params(&self, key: u8) -> StringParams {
        let i = self.index(key);
        let n = &self.notes;
        StringParams {
            f0: n.f0_hz[i],
            inharmonicity_b: n.inharmonicity_b[i],
            strike_position: n.strike_position[i],
            sigma0: n.sigma0[i],
            sigma1: n.sigma1[i],
            unison: n.unison[i] as usize,
            detune_cents: n.detune_cents[i],
            impedance: n.impedance[i],
            damper_sigma: n.damper_sigma[i],
            bridge_gain: n.bridge_gain[i],
        }
    }

    /// The hammer parameters of one key, global felt constants folded in.
    pub fn hammer_params(&self, key: u8) -> HammerParams {
        let i = self.index(key);
        let n = &self.notes;
        HammerParams {
            mass: n.hammer_mass[i],
            stiffness: n.hammer_stiffness[i],
            exponent: n.hammer_exponent[i],
            impedance: n.impedance[i],
            strings: n.unison[i] as f32,
            // The wave reaches the agraffe in x_strike * L / c, and L / c is
            // 1 / (2 f0), so the round trip is x_strike / f0.
            reflection_seconds: n.strike_position[i] / n.f0_hz[i],
            hysteresis: self.hammer.felt_hysteresis,
            una_corda_stiffness: self.hammer.una_corda_stiffness,
            reflection_gain: self.hammer.reflection_gain,
            velocity_min: self.hammer.velocity_min,
            velocity_max: self.hammer.velocity_max,
        }
    }

    /// Pitch of a key according to this preset's tuning.
    pub fn f0(&self, key: u8) -> f32 {
        self.notes.f0_hz[self.index(key)]
    }

    fn index(&self, key: u8) -> usize {
        key_index(key).expect("key outside A0..C8")
    }
}

/// Field checks used by [`Preset::validate`].
fn finite(name: &str, value: f32) -> Result<(), PresetError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(PresetError::invalid(format!("{name} is {value}")))
    }
}

fn positive(name: &str, value: f32) -> Result<(), PresetError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(PresetError::invalid(format!(
            "{name} is {value}, expected a finite positive number"
        )))
    }
}

fn within(name: &str, value: f32, low: f32, high: f32) -> Result<(), PresetError> {
    if value.is_finite() && (low..=high).contains(&value) {
        Ok(())
    } else {
        Err(PresetError::invalid(format!(
            "{name} is {value}, expected a number in {low}..={high}"
        )))
    }
}

impl Voicing {
    /// How firmly a fully engaged damper grips a partial at `f_hz`.
    pub fn damper_weight_at(&self, f_hz: f32) -> f32 {
        // Interpolated in log frequency: the anchors span decades and the felt
        // lets go per octave, not per hertz. `Preset::validate` has required
        // the anchor frequencies to be positive and strictly ascending, and the
        // logarithm keeps that order, which is what `interp_anchors` needs.
        let anchors: Vec<(f32, f32)> = self
            .damper_weight
            .iter()
            .map(|a| (a.hz.ln(), a.weight))
            .collect();
        interp_anchors(f_hz.max(f32::MIN_POSITIVE).ln(), &anchors)
    }

    /// Frequency ratio of unison string `i` of `n` against nominal pitch, given
    /// the group's full spread in cents.
    pub fn detune_ratio(&self, i: usize, n: usize, width_cents: f32) -> f32 {
        let cents = width_cents * self.unison_layout[n.clamp(1, MAX_UNISON) - 1].detune[i];
        (cents / 1200.0 * std::f32::consts::LN_2).exp()
    }

    /// Share of the hammer's force string `i` of `n` receives.
    pub fn strike_share(&self, i: usize, n: usize) -> f32 {
        self.unison_layout[n.clamp(1, MAX_UNISON) - 1].share[i]
    }

    /// How much faster the vertical polarization decays than the note as a
    /// whole.
    ///
    /// The horizontal polarization starts `horizontal_gain_db` down but decays
    /// `horizontal_decay_ratio` times as fast, so it is what is left at the end
    /// and it alone sets when the note reaches -60 dB. Solving
    /// `g_h exp(-rho sigma_v T60) = 1e-3 (1 + g_h)` for `sigma_v` gives the
    /// factor between the tabulated whole-note decay and the vertical bank's.
    pub fn vertical_decay_factor(&self) -> f32 {
        let gain = db_to_amp(self.horizontal_gain_db);
        (gain / (1.0e-3 * (1.0 + gain))).ln() / (self.horizontal_decay_ratio * 6.91)
    }
}

#[derive(Debug)]
pub enum PresetError {
    Io(std::io::Error),
    Parse(toml::de::Error),
    Invalid(String),
}

impl PresetError {
    fn invalid(message: impl Into<String>) -> PresetError {
        PresetError::Invalid(message.into())
    }
}

impl fmt::Display for PresetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PresetError::Io(e) => write!(f, "{e}"),
            PresetError::Parse(e) => write!(f, "{e}"),
            PresetError::Invalid(m) => write!(f, "invalid preset: {m}"),
        }
    }
}

impl std::error::Error for PresetError {}

// ------------------------------------------------------------ the default

/// Reference strike used to derive per-note felt stiffness.
const K_REF_VELOCITY: f32 = 3.0;

/// Coefficient of the `(f/1000)^2` term in the per-partial decay rate. Air and
/// internal friction losses grow with frequency, so high partials die first.
const SIGMA_FREQ_COEFF: f32 = 0.5;

impl Default for Preset {
    /// The instrument as hand-tuned for v1: the anchor curves the per-note
    /// tables were originally written as, evaluated for all 88 keys.
    fn default() -> Preset {
        let keys: Vec<u8> = (0..NUM_KEYS).map(index_to_note).collect();
        let table = |f: &dyn Fn(u8) -> f32| -> Vec<f32> { keys.iter().map(|&k| f(k)).collect() };

        let f0_hz = table(&note_to_freq);
        let strike_position = table(&|key| {
            interp_anchors(
                key_position(key),
                &[(0.0, 0.12), (0.55, 0.115), (1.0, 0.14)],
            )
        });
        let hammer_mass =
            table(&|key| interp_anchors(key_position(key), &[(0.0, 0.011), (1.0, 0.004)]));
        let hammer_exponent =
            table(&|key| interp_anchors(key_position(key), &[(0.0, 2.3), (1.0, 3.0)]));
        let hammer_stiffness = keys
            .iter()
            .enumerate()
            .map(|(i, &key)| {
                // K has units N/m^p and p varies across the compass, so a raw
                // table of it would be meaningless to write by hand. It is
                // derived instead from a target felt compression at a reference
                // strike, through the energy balance
                // `(1/2) m v^2 = K c^(p+1) / (p+1)`. The compressions are
                // chosen to land the contact durations on measured values:
                // ~4 ms in the bass, ~1.5 ms at C4, ~0.4 ms at the top.
                let c_ref = interp_anchors(
                    key_position(key),
                    &[
                        (0.0, 1.5e-3f32.ln()),
                        (0.28, 1.15e-3f32.ln()),
                        (0.59, 0.68e-3f32.ln()),
                        (1.0, 0.32e-3f32.ln()),
                    ],
                )
                .exp();
                let p = hammer_exponent[i];
                (p + 1.0) * hammer_mass[i] * K_REF_VELOCITY * K_REF_VELOCITY
                    / (2.0 * c_ref.powf(p + 1.0))
            })
            .collect();

        // Fundamental T60 anchors: 25 s at A0, 12 s at C4, 3 s at C6, 0.6 s at
        // C8, interpolated in log-T60 so the curve is smooth. These are
        // whole-note figures — see `Voicing::vertical_decay_factor`.
        let sigma1: Vec<f32> = vec![SIGMA_FREQ_COEFF; NUM_KEYS];
        let sigma0 = keys
            .iter()
            .enumerate()
            .map(|(i, &key)| {
                let t60 = interp_anchors(
                    key_position(key),
                    &[
                        (key_position(21), 25.0f32.ln()),
                        (key_position(60), 12.0f32.ln()),
                        (key_position(84), 3.0f32.ln()),
                        (key_position(108), 0.6f32.ln()),
                    ],
                )
                .exp();
                let sigma_fundamental = 6.91 / t60;
                (sigma_fundamental - sigma1[i] * (f0_hz[i] / 1000.0).powi(2)).max(0.01)
            })
            .collect();

        Preset {
            name: "default".to_string(),
            description: "piano-emulator v1, hand-tuned. Equal temperament; \
                          per-note tables evaluated from the design curves in preset.rs."
                .to_string(),
            voicing: Voicing {
                excitation_scale: 0.40,
                horizontal_gain_db: -12.0,
                horizontal_decay_ratio: 0.29,
                horizontal_offset_hz: vec![0.35, 0.52, 0.27],
                unison_coupling: 0.02,
                resonance_coupling: 0.012,
                unison_layout: vec![
                    UnisonLayout {
                        detune: vec![0.0],
                        share: vec![1.0],
                    },
                    UnisonLayout {
                        detune: vec![-0.47, 0.53],
                        share: vec![1.06, 0.94],
                    },
                    UnisonLayout {
                        detune: vec![-0.5, 0.11, 0.5],
                        share: vec![1.09, 1.0, 0.91],
                    },
                ],
                damper_weight: [(500.0, 1.0), (2000.0, 0.9), (6000.0, 0.35), (12000.0, 0.2)]
                    .into_iter()
                    .map(|(hz, weight)| DamperAnchor { hz, weight })
                    .collect(),
            },
            hammer: HammerVoicing {
                velocity_min: 0.2,
                velocity_max: 6.0,
                felt_hysteresis: 0.15,
                una_corda_stiffness: 0.7,
                reflection_gain: 0.85,
            },
            soundboard: SoundboardVoicing {
                board_mix: 0.35,
                body_mix: 0.7,
                board_level: 1.44,
                shelf_hz: 4_000.0,
                shelf_gain_db: -4.0,
                fdn_t60_lf: 0.4,
                fdn_t60_hf: 0.1,
                fdn_hf_hz: 8_000.0,
                body_modes: DEFAULT_BODY_MODES
                    .iter()
                    .map(|&(hz, q, gain)| BodyMode { hz, q, gain })
                    .collect(),
            },
            notes: NoteTables {
                f0_hz,
                inharmonicity_b: table(&default_inharmonicity),
                strike_position,
                sigma0,
                sigma1,
                unison: keys.iter().map(|&k| default_unison_count(k)).collect(),
                // 0.28 Hz at C2 through 0.45 at C4 to 2.4 Hz at C8 — a beat
                // period of 3-6 s where a pianist would hear it.
                detune_cents: table(&|key| {
                    interp_anchors(key_position(key), &[(0.0, 3.5), (1.0, 2.0)])
                }),
                impedance: table(&default_impedance),
                // Release T60 0.3 s in the bass falling to 0.1 s in the treble.
                damper_sigma: table(&|key| {
                    6.91 / interp_anchors(key_position(key), &[(0.0, 0.3), (1.0, 0.1)])
                }),
                bridge_gain: table(&default_bridge_gain),
                hammer_mass,
                hammer_stiffness,
                hammer_exponent,
            },
        }
    }
}

/// Inharmonicity B: ~1e-4 for the wound bass strings, dipping around C3, then
/// rising steeply through the short thick treble strings to ~1e-2 at C8.
fn default_inharmonicity(key: u8) -> f32 {
    interp_anchors(
        key_position(key),
        &[
            (key_position(21), 1.0e-4f32.ln()),
            (key_position(48), 3.0e-4f32.ln()),
            (key_position(60), 4.0e-4f32.ln()),
            (key_position(84), 1.2e-3f32.ln()),
            (key_position(108), 1.0e-2f32.ln()),
        ],
    )
    .exp()
}

/// Unison group size: single strings up to B1, pairs to E3, triples above.
fn default_unison_count(key: u8) -> u8 {
    match key {
        0..=35 => 1,           // .. B1
        36..=52 => 2,          // C2 .. E3
        _ => MAX_UNISON as u8, // F3 ..
    }
}

/// `Z = mu c = T / c`: the tension is roughly constant across the compass while
/// the wave speed rises, so the impedance falls steeply out of the bass and
/// then flattens out.
fn default_impedance(key: u8) -> f32 {
    interp_anchors(
        key_position(key),
        &[
            (key_position(21), 6.5f32.ln()),
            (key_position(36), 4.5f32.ln()),
            (key_position(60), 2.2f32.ln()),
            (key_position(84), 1.8f32.ln()),
            (key_position(108), 1.7f32.ln()),
        ],
    )
    .exp()
}

/// Fraction of a string's bridge force that becomes signal, in dB relative to
/// C4. A real soundboard is not an equally good radiator at every frequency:
/// its admittance peaks in the low-mid and falls away at both ends of the
/// compass, and the bass bridge is loaded by the long bass strings. Without
/// this the model's compass is tilted ~12 dB against the bass. Calibrated by
/// rendering mezzo-forte single notes and flattening their peak level.
fn default_bridge_gain(key: u8) -> f32 {
    db_to_amp(interp_anchors(
        key_position(key),
        &[
            (key_position(21), 6.5),
            (key_position(40), 3.0),
            (key_position(52), 2.8),
            (key_position(60), 2.6),
            (key_position(72), 2.0),
            (key_position(84), 0.4),
            (key_position(96), 0.4),
            (key_position(108), 2.0),
        ],
    ))
}

/// Body modes: (frequency Hz, Q, peak gain relative to the drive).
const DEFAULT_BODY_MODES: [(f32, f32, f32); 24] = [
    (42.0, 12.0, 0.55),
    (53.0, 13.0, 0.70),
    (63.0, 14.0, 0.85),
    (75.0, 15.0, 1.00),
    (85.0, 16.0, 0.90),
    (97.0, 17.0, 0.80),
    (114.0, 18.0, 0.95),
    (129.0, 19.0, 0.75),
    (142.0, 20.0, 0.85),
    (159.0, 21.0, 0.65),
    (168.0, 22.0, 0.80),
    (182.0, 22.0, 0.60),
    (200.0, 23.0, 0.70),
    (213.0, 24.0, 0.55),
    (230.0, 24.0, 0.65),
    (240.0, 25.0, 0.50),
    (258.0, 25.0, 0.60),
    (274.0, 26.0, 0.45),
    (295.0, 26.0, 0.50),
    (312.0, 27.0, 0.40),
    (327.0, 27.0, 0.45),
    (345.0, 28.0, 0.35),
    (370.0, 28.0, 0.40),
    (388.0, 28.0, 0.30),
];

// --------------------------------------------------------- float formatting

/// Serializers that write an `f32` as the shortest decimal that reads back as
/// the same `f32`, instead of the exact decimal expansion of its widening to
/// `f64` (`0.35` rather than `0.34999999403953552`).
///
/// Reading is the plain `f64 -> f32` conversion serde already does: a decimal
/// of at most nine significant digits — which is what the shortest form of an
/// `f32` is — converts through `f64` to the same `f32` it came from. The
/// round trip is pinned by `preset::tests::the_default_preset_round_trips`.
mod short {
    use serde::ser::{SerializeSeq, Serializer};

    fn widen(x: f32) -> f64 {
        // `f32`'s Display is the shortest decimal that round-trips through
        // `f32`; re-reading it as `f64` gives the nearest double to that
        // decimal, whose own shortest form is what lands in the file.
        x.to_string().parse().unwrap_or(x as f64)
    }

    pub fn scalar<S: Serializer>(x: &f32, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_f64(widen(*x))
    }

    pub fn list<S: Serializer>(v: &[f32], s: S) -> Result<S::Ok, S::Error> {
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
    use crate::types::{HIGHEST_KEY, LOWEST_KEY};

    /// The file in `presets/` next to the workspace root.
    pub(crate) fn default_preset_path() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../presets/default.toml")
    }

    #[test]
    fn the_default_preset_round_trips() {
        let preset = Preset::default();
        let text = preset.to_toml();
        let back = Preset::from_toml(&text).expect("round trip parses");
        // Bit-exact, not approximately: a preset that changed by an ulp when it
        // was written out would make every rendered comparison suspect.
        assert_eq!(preset, back);
        // ... and the shortest-decimal writer is doing its job.
        assert!(
            text.contains("excitation_scale = 0.4"),
            "scalars are not written in short form"
        );
    }

    /// The checked-in file must be exactly what `Preset::default` builds, which
    /// is what makes `--preset presets/default.toml` a no-op.
    #[test]
    fn the_checked_in_default_matches_the_built_in_one() {
        let path = default_preset_path();
        let loaded = Preset::load(&path).expect("presets/default.toml loads");
        assert_eq!(
            loaded,
            Preset::default(),
            "presets/default.toml is out of date; regenerate with `piano-emulator preset {}`",
            path.display()
        );
    }

    #[test]
    fn every_key_has_parameters() {
        let preset = Preset::default();
        for key in LOWEST_KEY..=HIGHEST_KEY {
            let s = preset.string_params(key);
            let h = preset.hammer_params(key);
            assert!(s.f0 > 0.0 && s.impedance > 0.0);
            assert_eq!(h.strings as usize, s.unison);
            assert!(h.reflection_seconds > 0.0 && h.stiffness > 0.0);
        }
        assert!(preset.validate().is_ok());
    }

    #[test]
    fn the_default_tuning_is_equal_temperament() {
        let preset = Preset::default();
        assert!((preset.f0(69) - 440.0).abs() < 1e-4);
        assert!((preset.f0(60) - 261.6256).abs() < 1e-3);
        for key in LOWEST_KEY..=HIGHEST_KEY {
            assert_eq!(preset.f0(key), note_to_freq(key));
        }
    }

    #[test]
    fn a_stretched_table_is_what_the_string_gets() {
        let mut preset = Preset::default();
        let i = key_index(108).unwrap();
        preset.notes.f0_hz[i] *= 1.01; // ~17 cents sharp, as a top octave is
        assert!(preset.validate().is_ok());
        let params = preset.string_params(108);
        assert!((params.f0 / note_to_freq(108) - 1.01).abs() < 1e-5);
        // ... and the hammer's agraffe round trip follows the tuning with it.
        assert!(
            (preset.hammer_params(108).reflection_seconds * params.f0 - params.strike_position)
                .abs()
                < 1e-9
        );
    }

    #[test]
    fn malformed_presets_are_rejected() {
        let short_table = {
            let mut p = Preset::default();
            p.notes.f0_hz.pop();
            p
        };
        assert!(short_table.validate().is_err());

        let bad_unison = {
            let mut p = Preset::default();
            p.notes.unison[0] = 4;
            p
        };
        assert!(bad_unison.validate().is_err());

        // Every one of these would reach the DSP as a divide by zero, a NaN,
        // or a resonator pole outside the unit circle.
        let breakages: [fn(&mut Preset); 23] = [
            |p: &mut Preset| p.notes.f0_hz[3] = 0.0,
            |p: &mut Preset| p.notes.sigma0[3] = -1.0,
            |p: &mut Preset| p.notes.inharmonicity_b[3] = -1e-4,
            |p: &mut Preset| p.notes.strike_position[3] = 1.0,
            |p: &mut Preset| p.notes.hammer_mass[3] = 0.0,
            |p: &mut Preset| p.notes.bridge_gain[3] = 0.0,
            |p: &mut Preset| p.voicing.excitation_scale = 0.0,
            |p: &mut Preset| p.hammer.velocity_min = 0.0,
            |p: &mut Preset| p.hammer.reflection_gain = 0.0,
            |p: &mut Preset| p.soundboard.fdn_t60_lf = 0.0,
            |p: &mut Preset| p.soundboard.body_modes[0].q = 0.0,
            |p: &mut Preset| {
                p.voicing.unison_layout[2].share.pop();
            },
            // The voicing's own numbers reach the same DSP: a NaN anywhere
            // here is a NaN in a mode frequency or an excitation, and it
            // spreads to the whole instrument within a block.
            |p: &mut Preset| p.voicing.horizontal_offset_hz[1] = f32::NAN,
            |p: &mut Preset| p.voicing.unison_layout[2].detune[1] = f32::NAN,
            |p: &mut Preset| p.voicing.unison_layout[2].share[1] = f32::NAN,
            |p: &mut Preset| p.voicing.unison_coupling = f32::NAN,
            |p: &mut Preset| p.voicing.resonance_coupling = f32::NAN,
            |p: &mut Preset| p.voicing.damper_weight[1].hz = f32::NAN,
            // A horizontal offset below the lowest fundamental would put that
            // partial at a negative frequency.
            |p: &mut Preset| p.voicing.horizontal_offset_hz[0] = -100.0,
            // Coupling past the point where the feedback loop sustains itself.
            |p: &mut Preset| p.voicing.unison_coupling = 1.0,
            |p: &mut Preset| p.voicing.resonance_coupling = -0.01,
            // A string outside its own group's detune width, and a hammer that
            // pulls one of the strings it strikes.
            |p: &mut Preset| p.voicing.unison_layout[2].detune[0] = -1.5,
            |p: &mut Preset| p.voicing.unison_layout[1].share[0] = -0.1,
        ];
        for break_it in breakages {
            let mut p = Preset::default();
            break_it(&mut p);
            assert!(p.validate().is_err(), "a broken preset validated");
        }

        assert!(Preset::from_toml("name = 'nope'").is_err());
        assert!(Preset::from_toml("this is not toml").is_err());
    }

    #[test]
    fn the_damper_grips_low_partials_hardest() {
        let v = Preset::default().voicing;
        assert!(v.damper_weight_at(100.0) > v.damper_weight_at(2_000.0));
        assert!(v.damper_weight_at(2_000.0) > v.damper_weight_at(10_000.0));
        // Clamped at both ends of the anchor list.
        assert_eq!(v.damper_weight_at(1.0), v.damper_weight_at(500.0));
        assert_eq!(v.damper_weight_at(40_000.0), v.damper_weight_at(12_000.0));
    }
}
