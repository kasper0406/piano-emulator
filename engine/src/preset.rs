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
//!
//! # Fields a preset may leave out
//!
//! A few fields describe a refinement of the model that has a neutral setting —
//! `notes.inharmonicity_b4` and `notes.contact_width` at zero,
//! `voicing.unison_sigma_scale` at one, `voicing.polarization_pan_spread` at
//! zero. Those are `#[serde(default)]` on the way in and skipped on the way out
//! while they hold the neutral value, so a file that predates the field keeps
//! playing the instrument it always described and the engine keeps writing that
//! same file back byte for byte — including `presets/default.toml`, which is
//! checked in and read by the tuner's own copy of this schema. A preset that
//! *uses* one of them writes it in full, and then every number the note is
//! played with is still in the file. Nothing else in the engine may rely on a
//! field being optional: [`Preset::validate`] checks the defaulted tables
//! exactly as it checks the mandatory ones.
//!
//! `[noise]` is the one such field whose default is not *neutral*: a preset
//! that omits it gets the mechanism levels `TUNING_REPORT.md` §5 measured, not
//! silence. It is written the same way — skipped while it equals the measured
//! table — for the same reason the others are: the file is the interface to the
//! tuner, and the engine emitting a section the tuner's copy of the schema does
//! not know would break every preset already written. Silence is available, and
//! has to be asked for, by writing the section with `level_db` far down.

use crate::hammer::HammerParams;
use crate::resonance::MAX_COUPLING;
use crate::soundboard::MAX_PAN_SPREAD;
use crate::string::{
    StringParams, MAX_CONTACT_WIDTH, MAX_SIGMA_SCALE, MAX_UNISON_COUPLING, MIN_SIGMA_SCALE,
};
use crate::types::{
    db_to_amp, index_to_note, interp_anchors, key_index, key_position, note_to_freq, LOWEST_KEY,
    MAX_UNISON, NUM_KEYS,
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
    /// The action's own sounds. Absent means the measured defaults — see
    /// [`NoiseTables`], and "Fields a preset may leave out" above.
    #[serde(default, skip_serializing_if = "is_default_noise")]
    pub noise: NoiseTables,
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
    /// How far apart the two polarizations sit in the stereo image, as a pan
    /// displacement either side of the key's own position.
    ///
    /// The horizontal polarization renders at `pan + spread * sign` and the
    /// vertical one at `pan - spread * sign`, with `sign` alternating by key
    /// parity so that spreading the image does not walk the whole instrument to
    /// one side. Because the two decay at very different rates, the balance of
    /// a single note then *moves* while it rings: 1.2–6.2 dB of drift between
    /// 0.3 s and 2 s in the recordings against 0.02–0.14 dB in the engine's own
    /// renders, which pans one mono voice per key and structurally cannot move
    /// at all (`TUNING_REPORT.md` §5).
    ///
    /// Zero — the default — keeps both polarizations at the key's pan and the
    /// single-buffer render path with them.
    #[serde(default, skip_serializing_if = "is_zero", serialize_with = "short::scalar")]
    pub polarization_pan_spread: f32,
    /// One entry per unison group size, 1 to [`MAX_UNISON`] strings.
    pub unison_layout: Vec<UnisonLayout>,
    /// Decay-rate multipliers for the individual strings of a unison, one row
    /// per group size exactly like [`Voicing::unison_layout`].
    ///
    /// A group whose strings are mistuned but share one damping law cannot move
    /// its own pitch as it decays; a real one does, by up to 32 cents over the
    /// fundamental's first 20 dB, because a mistuned string that outlives its
    /// neighbours takes the composite partial's pitch with it
    /// (`TUNING_REPORT.md` §6). All ones — the default — is the shared damping
    /// law, and the note's whole-note T60 is then exactly the one
    /// `notes.sigma0` asks for.
    #[serde(
        default = "unity_sigma_scale",
        skip_serializing_if = "is_unity_sigma_scale"
    )]
    pub unison_sigma_scale: Vec<UnisonSigmaScale>,
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

/// The four mechanism events, and what each of them sounds like.
///
/// `TUNING_REPORT.md` §5 is the parameter set: it measured Salamander's own
/// `rel*` and `pedal*` recordings, at the level the SFZ plays them, against a
/// velocity-90 strike of the same key. Nothing here needed fitting, which is
/// why the report ranks this the cheapest item on its backlog.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NoiseTables {
    /// The key and its action returning to rest, plus the damper landing on the
    /// string. One on every note-off, on every key: `rel76` is C7, three
    /// semitones above the damper break, so the sound does not stop where the
    /// dampers do.
    pub key_off: EventNoise,
    /// The damper felt leaving the string under a key pressed too slowly to
    /// reach escapement — the silent press of `PHYSICS.md` §6, where it is the
    /// *only* sound the note makes. Only on keys that have a damper, and
    /// deliberately not under a struck note-on: a lift under a hammer blow is
    /// inaudible, which is why no library records one, and putting one there
    /// would be a broadband burst on every single note that nothing can hear
    /// and the tuner's estimators would have to fit around.
    pub damper_lift: EventNoise,
    /// The sustain pedal's tray and the whole damper rail rising. Global, and
    /// scaled by how many dampers actually move.
    pub pedal_down: EventNoise,
    /// The same rail landing again.
    pub pedal_up: EventNoise,
}

/// One mechanism event: how loud, how long, and what colour.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventNoise {
    /// Spectral centre of the burst, Hz. Well under the ~2 kHz where the
    /// action's structure-borne spectrum ends (`PHYSICS.md` §5).
    #[serde(serialize_with = "short::scalar")]
    pub centroid_hz: f32,
    /// Time to fall 40 dB, seconds — the column `TUNING_REPORT.md` §5 reports.
    #[serde(serialize_with = "short::scalar")]
    pub decay_s: f32,
    /// How far the level travels, in dB, over the event's full drive range:
    /// release velocity for the key events, the fraction of the dampers that
    /// move for the pedal ones. The tabulated [`EventNoise::level_db`] is the
    /// level at the nominal drive, and this is the slope through it.
    #[serde(serialize_with = "short::scalar")]
    pub velocity_db: f32,
    /// Peak level, in dB relative to a velocity-90 strike of the same key,
    /// anchored at the keys it was measured at and interpolated across the
    /// compass. A global event carries a single anchor.
    pub level_db: Vec<NoiseAnchor>,
}

/// One measured point of a mechanism event's level across the compass.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NoiseAnchor {
    /// MIDI key this level belongs to.
    pub key: u8,
    #[serde(serialize_with = "short::scalar")]
    pub db: f32,
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
    /// Stiffness inharmonicity B in `f_k = k f0 sqrt(1 + B k^2 + B4 k^4)`.
    #[serde(serialize_with = "short::list")]
    pub inharmonicity_b: Vec<f32>,
    /// Fourth-order coefficient B4 of the same law. **Signed**: a wound bass
    /// string's series curves one way and the short wound tenor strings' the
    /// other (`TUNING_REPORT.md` §1). Absent means zero, which is the
    /// two-parameter law exactly.
    #[serde(
        default = "zero_table",
        skip_serializing_if = "is_zero_table",
        serialize_with = "short::list"
    )]
    pub inharmonicity_b4: Vec<f32>,
    /// Hammer strike point as a fraction of the speaking length.
    #[serde(serialize_with = "short::list")]
    pub strike_position: Vec<f32>,
    /// Width of the hammer's contact with the string, as a fraction of the
    /// speaking length: a real hammer averages the strike comb over 1–2 % of it
    /// instead of pinching a point. Absent means zero, which is the point force
    /// the comb alone describes.
    #[serde(
        default = "zero_table",
        skip_serializing_if = "is_zero_table",
        serialize_with = "short::list"
    )]
    pub contact_width: Vec<f32>,
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
        // The fourth-order inharmonicity is the one signed table: the sign is
        // the finding (`TUNING_REPORT.md` §1), so only finiteness can be
        // checked entry by entry. What the value has to *do* is checked below,
        // against the series it produces.
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
        // `f_k = k f0 sqrt(1 + B k^2 + B4 k^4)` is only a partial layout while
        // the radicand stays positive and the series it produces stays ordered.
        // With `B4` signed and `B4 k^4` growing four times as fast in the
        // exponent as `B k^2`, a coefficient that is harmless on the low
        // partials can fold the top of the series back down or take it under
        // the root — a NaN in a mode frequency, spread to the whole instrument
        // within a block. Checked over every partial the law could reach, up
        // to the Nyquist cap — NOT over `partial_count()`, whose `take_while`
        // stops at the first non-finite frequency, so bounding the check by it
        // would let a radicand that jumps straight negative truncate the
        // note's bank silently instead of being refused.
        for i in 0..NUM_KEYS {
            let p = self.string_params(index_to_note(i));
            let limit = crate::types::MAX_PARTIAL_RATIO * crate::types::SAMPLE_RATE;
            let mut previous = 0.0f32;
            for k in 1..=crate::types::MAX_PARTIALS {
                let radicand = p.partial_radicand(k);
                if !(radicand.is_finite() && radicand > 0.0) {
                    return Err(PresetError::invalid(format!(
                        "notes.inharmonicity_b[{i}] = {} with inharmonicity_b4[{i}] = {} \
                         puts partial {k} under a root of {radicand}",
                        p.inharmonicity_b, p.inharmonicity_b4
                    )));
                }
                let f = p.partial_freq(k);
                if !f.is_finite() || f <= previous {
                    return Err(PresetError::invalid(format!(
                        "notes.inharmonicity_b[{i}] = {} with inharmonicity_b4[{i}] = {} \
                         puts partial {k} at {f} Hz, not above the {previous} Hz before it",
                        p.inharmonicity_b, p.inharmonicity_b4
                    )));
                }
                // The legitimate end of the series: past the cap the banks are
                // never built, so nothing beyond it needs to be well-formed.
                if f >= limit {
                    break;
                }
                previous = f;
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
        // A displacement either side of a pan position that already reaches
        // `MAX_PAN`: the ceiling is what puts the outer polarization of the
        // outermost key exactly hard left or hard right, never past it.
        within(
            "voicing.polarization_pan_spread",
            v.polarization_pan_spread,
            0.0,
            MAX_PAN_SPREAD,
        )?;
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
        if v.unison_sigma_scale.len() != MAX_UNISON
            || v.unison_sigma_scale
                .iter()
                .enumerate()
                .any(|(i, row)| row.scale.len() != i + 1)
        {
            return Err(PresetError::invalid(format!(
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
                return Err(PresetError::invalid(format!(
                    "voicing.unison_sigma_scale[{row}].scale averages {mean}, expected 1"
                )));
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

        // The mechanism events reach a biquad's coefficients and an exponential
        // envelope on the audio path, so the same rule applies as everywhere
        // else here: a centroid at or past Nyquist is a filter with a pole
        // outside the unit circle, and a decay at zero is an event that never
        // ends.
        for (name, event) in self.noise.events() {
            positive(&format!("noise.{name}.centroid_hz"), event.centroid_hz)?;
            within(
                &format!("noise.{name}.centroid_hz"),
                event.centroid_hz,
                1.0,
                0.45 * crate::types::SAMPLE_RATE,
            )?;
            within(
                &format!("noise.{name}.decay_s"),
                event.decay_s,
                MIN_NOISE_DECAY_S,
                MAX_NOISE_DECAY_S,
            )?;
            finite(&format!("noise.{name}.velocity_db"), event.velocity_db)?;
            if event.level_db.is_empty() {
                return Err(PresetError::invalid(format!(
                    "noise.{name}.level_db is empty"
                )));
            }
            for (i, anchor) in event.level_db.iter().enumerate() {
                if key_index(anchor.key).is_none() {
                    return Err(PresetError::invalid(format!(
                        "noise.{name}.level_db[{i}].key is {}, which is not on the keyboard",
                        anchor.key
                    )));
                }
                // A mechanism event louder than the note it belongs to is not a
                // mechanism event; the measured range is -25 to -45 dB.
                if !anchor.db.is_finite() || anchor.db > 0.0 {
                    return Err(PresetError::invalid(format!(
                        "noise.{name}.level_db[{i}].db is {}, expected a finite level at or \
                         below 0 dB relative to a strike",
                        anchor.db
                    )));
                }
            }
            // The level is interpolated across the compass by `interp_anchors`,
            // which walks the anchors in order.
            if let Some(i) = event.level_db.windows(2).position(|w| w[0].key >= w[1].key) {
                return Err(PresetError::invalid(format!(
                    "noise.{name}.level_db[{}] is at key {}, not above the key {} before it",
                    i + 1,
                    event.level_db[i + 1].key,
                    event.level_db[i].key
                )));
            }
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
            inharmonicity_b4: n.inharmonicity_b4[i],
            strike_position: n.strike_position[i],
            contact_width: n.contact_width[i],
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

/// Shortest and longest a mechanism event may last. The measured events span
/// 0.165 s to 5.76 s; the bounds are a decade either side of that.
const MIN_NOISE_DECAY_S: f32 = 0.01;
const MAX_NOISE_DECAY_S: f32 = 10.0;

impl NoiseTables {
    /// The four events with their field names, for validation and for the
    /// engine's construction. In one place so neither can forget one.
    pub fn events(&self) -> [(&'static str, &EventNoise); 4] {
        [
            ("key_off", &self.key_off),
            ("damper_lift", &self.damper_lift),
            ("pedal_down", &self.pedal_down),
            ("pedal_up", &self.pedal_up),
        ]
    }
}

fn is_default_noise(noise: &NoiseTables) -> bool {
    *noise == NoiseTables::default()
}

/// The mechanism as `TUNING_REPORT.md` §5 measured it.
///
/// Levels are that table's "peak re strike" column, anchored at the keys the
/// samples belong to: `rel1` = A0, `rel37` = A3, `rel40` = C4, `rel52` = C5,
/// `rel76` = C7. They are not smooth — the recordings differ by 10 dB between
/// neighbouring octaves — and they are written as measured rather than
/// flattened, because a hand-drawn curve here would be a guess dressed as data.
///
/// The one event the report could **not** measure is the damper *lift*:
/// Salamander ships no such sample, and no library does, because a lift under a
/// strike is inaudible. It is written 6 dB under the same key's fall and much
/// shorter — the felt leaving the string is a lighter event than the key
/// arriving at its rest — and it is the least-supported number in this file.
/// It matters audibly in exactly one place, the silent key press of
/// `PHYSICS.md` §6, where it is the whole sound.
impl Default for NoiseTables {
    fn default() -> NoiseTables {
        // (key, peak dB re a velocity-90 strike of that key)
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
                // The measured centroids span 143-261 Hz and the decays
                // 0.165-0.285 s; one figure each, in the middle of both.
                centroid_hz: 190.0,
                decay_s: 0.24,
                // Pianoteq's Blüthner spans 12 dB over note-off velocity, and
                // Salamander's release group tracks velocity at 82/127.
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
                // pedalD1: -35.8 dB, 5.76 s to -40 dB, centroid 77 Hz — the
                // long rumble of the tray and the whole damper rail.
                centroid_hz: 77.0,
                decay_s: 5.76,
                velocity_db: 6.0,
                level_db: vec![NoiseAnchor {
                    key: LOWEST_KEY,
                    db: -35.8,
                }],
            },
            pedal_up: EventNoise {
                // pedalU1: -42.4 dB, 0.32 s, centroid 187 Hz.
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
fn unity_sigma_scale() -> Vec<UnisonSigmaScale> {
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

/// Field checks used by [`Preset::validate`].
fn table_length(name: &str, len: usize) -> Result<(), PresetError> {
    if len == NUM_KEYS {
        Ok(())
    } else {
        Err(PresetError::invalid(format!(
            "notes.{name} has {len} entries, expected {NUM_KEYS}"
        )))
    }
}

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

    /// Decay-rate multiplier of string `i` of `n`, applied to both of that
    /// string's polarizations. Exactly 1 in a preset that does not set the
    /// field, which leaves the string's sigmas untouched to the last bit.
    pub fn sigma_scale(&self, i: usize, n: usize) -> f32 {
        self.unison_sigma_scale[n.clamp(1, MAX_UNISON) - 1].scale[i]
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
                // The hand-tuned instrument is the point-force, one-`B`,
                // one-damping-law, one-pan-position piano v1 was: every field
                // added since Phase E sits at its neutral value here, and none
                // of them is written to the file.
                polarization_pan_spread: 0.0,
                unison_sigma_scale: unity_sigma_scale(),
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
                inharmonicity_b4: zero_table(),
                strike_position,
                contact_width: zero_table(),
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
            // The action as `TUNING_REPORT.md` §5 measured it. Unlike the other
            // fields a preset may leave out, this one's default is not silence:
            // the report's point is that the engine made *no* sound at a
            // release or a pedal move, and that was the model error.
            noise: NoiseTables::default(),
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

    /// A `B4` that takes the radicand negative *between* consecutive partials
    /// truncates `partial_count()` itself (`take_while` stops at the NaN), so a
    /// validation bounded by the count would never see it — C8 with `-0.3` used
    /// to validate cleanly while its bank silently shrank from 4 partials to 1.
    /// The check must run over the full reachable range instead.
    #[test]
    fn a_b4_that_jumps_the_radicand_negative_is_refused_not_truncated() {
        let mut p = Preset::default();
        *p.notes.inharmonicity_b4.last_mut().unwrap() = -0.3;
        assert!(
            p.validate().is_err(),
            "a bank-truncating B4 passed validation"
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
        let breakages: [fn(&mut Preset); 52] = [
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
            // The fields a preset may leave out are checked as hard as the ones
            // it may not. A fourth-order coefficient is signed, so what has to
            // be refused is not a sign but a series: one that goes under the
            // root (a NaN mode frequency) or turns over (partials in the wrong
            // order, which is not a string). -3e-8 reorders A0's eighty
            // partials, -1e-6 also takes them under the root partway up, and
            // -2 takes the fundamental itself under it.
            |p: &mut Preset| p.notes.inharmonicity_b4[0] = f32::NAN,
            |p: &mut Preset| p.notes.inharmonicity_b4[0] = -2.0,
            |p: &mut Preset| p.notes.inharmonicity_b4[0] = -1.0e-6,
            |p: &mut Preset| p.notes.inharmonicity_b4[0] = -3.0e-8,
            |p: &mut Preset| {
                p.notes.inharmonicity_b4.pop();
            },
            // A contact wider than the ceiling is not a hammer, and a negative
            // one is not a width.
            |p: &mut Preset| p.notes.contact_width[3] = 0.06,
            |p: &mut Preset| p.notes.contact_width[3] = -0.01,
            |p: &mut Preset| p.notes.contact_width[3] = f32::NAN,
            |p: &mut Preset| {
                p.notes.contact_width.pop();
            },
            // A decay-rate multiplier at zero is a string that never stops; a
            // row that does not average to 1 is a second decay control that
            // would silently retune the compass's T60 away from `sigma0`.
            |p: &mut Preset| p.voicing.unison_sigma_scale[2].scale[0] = 0.0,
            |p: &mut Preset| p.voicing.unison_sigma_scale[2].scale[0] = 3.0,
            |p: &mut Preset| p.voicing.unison_sigma_scale[2].scale[0] = f32::NAN,
            |p: &mut Preset| p.voicing.unison_sigma_scale[1].scale = vec![0.8, 0.8],
            |p: &mut Preset| {
                p.voicing.unison_sigma_scale[2].scale.pop();
            },
            // The polarizations may be spread across the stage, not past it.
            |p: &mut Preset| p.voicing.polarization_pan_spread = 0.5,
            |p: &mut Preset| p.voicing.polarization_pan_spread = -0.1,
            |p: &mut Preset| p.voicing.polarization_pan_spread = f32::NAN,
            // The mechanism's numbers reach a biquad and an envelope on the
            // audio path. A centroid at or past Nyquist is a pole outside the
            // unit circle; a decay at zero is an event that never ends; a level
            // above 0 dB is a thump louder than the note it belongs to.
            |p: &mut Preset| p.noise.key_off.centroid_hz = 0.0,
            |p: &mut Preset| p.noise.key_off.centroid_hz = 30_000.0,
            |p: &mut Preset| p.noise.key_off.centroid_hz = f32::NAN,
            |p: &mut Preset| p.noise.damper_lift.decay_s = 0.0,
            |p: &mut Preset| p.noise.pedal_down.decay_s = 60.0,
            |p: &mut Preset| p.noise.pedal_up.decay_s = f32::NAN,
            |p: &mut Preset| p.noise.key_off.velocity_db = f32::NAN,
            |p: &mut Preset| p.noise.key_off.level_db[0].db = 6.0,
            |p: &mut Preset| p.noise.key_off.level_db[0].db = f32::NAN,
            |p: &mut Preset| p.noise.key_off.level_db[0].key = 12,
            |p: &mut Preset| p.noise.key_off.level_db.clear(),
            // The level anchors are interpolated across the compass, so they
            // have to be in order.
            |p: &mut Preset| p.noise.key_off.level_db[1].key = 21,
        ];
        for break_it in breakages {
            let mut p = Preset::default();
            break_it(&mut p);
            assert!(p.validate().is_err(), "a broken preset validated");
        }

        assert!(Preset::from_toml("name = 'nope'").is_err());
        assert!(Preset::from_toml("this is not toml").is_err());
    }

    /// A preset that does not use the model's newer refinements does not
    /// mention them, and reads back as the instrument it describes. This is
    /// what keeps `presets/default.toml` — and every preset the tuner has
    /// already written — byte for byte what it was.
    #[test]
    fn neutral_refinements_are_absent_from_the_file_and_read_back_neutral() {
        let text = Preset::default().to_toml();
        for field in [
            "inharmonicity_b4",
            "contact_width",
            "unison_sigma_scale",
            "polarization_pan_spread",
            // The mechanism's default is the measured table rather than
            // silence, but it is skipped on the same terms and for the same
            // reason: the file is the tuner's interface.
            "[noise]",
        ] {
            assert!(!text.contains(field), "a neutral preset wrote {field}");
        }
        let back = Preset::from_toml(&text).expect("a preset without them still loads");
        assert_eq!(back, Preset::default());
        assert!(back.notes.inharmonicity_b4.iter().all(|&b| b == 0.0));
        assert!(back.notes.contact_width.iter().all(|&w| w == 0.0));
        assert_eq!(back.voicing.polarization_pan_spread, 0.0);
        for n in 1..=MAX_UNISON {
            for i in 0..n {
                assert_eq!(back.voicing.sigma_scale(i, n), 1.0);
            }
        }
    }

    /// A preset may voice the action differently — including switching it off,
    /// which is what a file that wants the pre-mechanism instrument has to say
    /// out loud.
    #[test]
    fn an_action_that_is_not_the_measured_one_is_written_in_full() {
        let mut preset = Preset::default();
        for event in [
            &mut preset.noise.key_off,
            &mut preset.noise.damper_lift,
            &mut preset.noise.pedal_down,
            &mut preset.noise.pedal_up,
        ] {
            for anchor in &mut event.level_db {
                anchor.db = -200.0;
            }
        }
        assert!(preset.validate().is_ok());
        let text = preset.to_toml();
        assert!(text.contains("[noise.key_off]"), "the silenced action was skipped");
        assert!(text.contains("[[noise.pedal_up.level_db]]"));
        assert_eq!(Preset::from_toml(&text).expect("round trip parses"), preset);
        // ... and the default is still the default, byte for byte.
        assert!(!Preset::default().to_toml().contains("[noise]"));
    }

    /// ... and a preset that does use them writes every one of its numbers.
    #[test]
    fn a_preset_that_uses_the_refinements_round_trips() {
        let mut preset = Preset::default();
        preset.notes.inharmonicity_b4[0] = -1.0e-8;
        preset.notes.inharmonicity_b4[87] = 3.5e-5;
        preset.notes.contact_width[60] = 0.0125;
        preset.voicing.polarization_pan_spread = 0.22;
        preset.voicing.unison_sigma_scale[2].scale = vec![0.85, 1.0, 1.15];
        preset.voicing.unison_sigma_scale[1].scale = vec![0.9, 1.1];
        assert!(preset.validate().is_ok());

        let text = preset.to_toml();
        assert!(text.contains("polarization_pan_spread = 0.22"));
        assert_eq!(text.matches("[[voicing.unison_sigma_scale]]").count(), MAX_UNISON);
        assert!(text.contains("0.0125"));
        // Bit-exact, like every other number in a preset.
        assert_eq!(Preset::from_toml(&text).expect("round trip parses"), preset);
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
