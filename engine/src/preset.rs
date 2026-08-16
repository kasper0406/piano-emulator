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
//! zero, `[voicing.bridge]` absent (the flat sympathetic bus), `notes.duplex`
//! absent (no aliquot segments), `notes.pan_spread` absent (the one global
//! spread applies to the whole compass), `notes.comb_floor` at zero (the bare
//! strike comb, nulls and all), `notes.partial_gains` and
//! `notes.partial_sigma_scale` absent (one on every partial of every key), and
//! `[noise.strike]` at its default (the hammer makes no noise of its own).
//! The three **inert** fields of `DECISIONS.md` 225 are written the same way
//! for a different reason — `voicing.unison_coupling` at zero,
//! `voicing.horizontal_offset_hz` all zero and `voicing.unison_sigma_scale` at
//! one are not neutral *settings*, they are the values at which nothing is
//! being asserted, since the coupled string construction reads none of the
//! three at any value ([`Preset::inert_fields`]). A file that still carries
//! them loads, validates and is written back with them; nothing this crate
//! produces has them.
//! Those are `#[serde(default)]` on the way in and skipped on the way out
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
//! that omits it gets the mechanism levels `docs/history/TUNING_REPORT.md` §5 measured, not
//! silence. It is written the same way — skipped while it equals the measured
//! table — for the same reason the others are: the file is the interface to the
//! tuner, and the engine emitting a section the tuner's copy of the schema does
//! not know would break every preset already written. Silence is available, and
//! has to be asked for, by writing the section with `level_db` far down. The
//! *fifth* event in that section, `[noise.strike]`, is the other way round: its
//! default is silence, because no recording in the library isolates a hammer
//! and a level nobody measured has no business being on by default.

use crate::duplex::MAX_DUPLEX_LOOP_GAIN;
use crate::hammer::HammerParams;
use crate::resonance::{BridgeFilter, MAX_BRIDGE_LOOP_GAIN, MAX_COUPLING};
use crate::soundboard::MAX_PAN_SPREAD;
use crate::string::{
    PartialShaping, StringParams, MAX_COMB_FLOOR, MAX_CONTACT_WIDTH, MAX_PARTIAL_GAIN,
    MAX_PARTIAL_SIGMA_SCALE, MAX_SIGMA_SCALE, MIN_MODE_SIGMA, MIN_PARTIAL_GAIN,
    MIN_PARTIAL_SIGMA_SCALE, MIN_SIGMA_SCALE,
};
use crate::types::{
    amp_to_db, db_to_amp, index_to_note, interp_anchors, key_index, key_position, note_to_freq,
    HIGHEST_KEY, LOWEST_KEY, MAX_UNISON, NUM_KEYS,
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

/// Largest value the inert `voicing.unison_coupling` may still carry.
///
/// **Kept for compatibility, and for nothing else.** It used to be a loop-gain
/// ceiling: the coupling passed a fraction of the neighbours' bridge force into
/// a string one block later, so a unison group was a feedback loop and a value
/// near 0.1 could sustain itself. The loop is gone — the coupling is a
/// construction-time property of the partial and lives in the eigenproblem — so
/// this is now only a schema bound, kept at the number every preset already
/// written was validated against so that all of them still load. Neither
/// checked-in preset sets the field any more (`DECISIONS.md` 324), and nothing
/// this workspace writes ever will; the bound exists for files written
/// elsewhere, before the split. Removing it means removing the field, which
/// means refusing to load those files.
pub const LEGACY_MAX_UNISON_COUPLING: f32 = 0.05;

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
    /// **Inert.** Frequency offset of the horizontal polarization, one per
    /// unison string.
    ///
    /// A fixed number of *hertz*, so the three values were beat rates of every
    /// partial of every key at once — an instrument-wide, note-independent,
    /// velocity-independent pulse at 0.270 / 0.350 / 0.520 Hz, which
    /// `renders/jitter/JITTER.md` measured in every row of every table it
    /// printed and `docs/history/FUNDAMENTALS.md` §2.2 showed to be 35x the physically
    /// derivable split and the one shape with no mechanism behind it. The
    /// polarization split is now the bridge's reactive anisotropy times the
    /// partial's own frequency (`engine::string`), which is a few hundredths of
    /// a hertz and different in every partial of every key.
    ///
    /// Still parsed, still validated, never read. [`Preset::inert_fields`]
    /// names it on load. Absent — the default, and what both checked-in presets
    /// now have — is all zeros, which is the value that makes it silent.
    #[serde(
        default = "no_horizontal_offset",
        skip_serializing_if = "is_no_horizontal_offset",
        serialize_with = "short::list"
    )]
    pub horizontal_offset_hz: Vec<f32>,
    /// **Inert.** Bridge coupling within a unison group, as a fraction of the
    /// string's wave impedance.
    ///
    /// It was applied one `BLOCK` late through the excitation, which is both
    /// ~25x too weak and phase-scrambled — 2.667 ms is 251 degrees at C4's
    /// fundamental and 11.4 periods at C6's fourth partial, so the sign of both
    /// the frequency pull and the damping split was randomised per partial. The
    /// forensics measured the whole of it end to end at **0.07 cents**. The
    /// coupling is not a free parameter at all: it is `radiated_share * sigma_k`,
    /// the same coefficient as the radiation damping the preset has already
    /// fitted (`docs/history/FUNDAMENTALS.md` §1.1), and it now lives in the eigenproblem.
    ///
    /// Still parsed, still validated, never read. Absent — the default, and
    /// what both checked-in presets now have — is zero.
    #[serde(
        default,
        skip_serializing_if = "is_zero",
        serialize_with = "short::scalar"
    )]
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
    /// at all (`docs/history/TUNING_REPORT.md` §5).
    ///
    /// Zero — the default — keeps both polarizations at the key's pan and the
    /// single-buffer render path with them.
    #[serde(default, skip_serializing_if = "is_zero", serialize_with = "short::scalar")]
    pub polarization_pan_spread: f32,
    /// The bridge admittance `B(f)` on the sympathetic bus's drive path.
    ///
    /// Absent — the default — is the unity filter, which is the flat bus the
    /// engine has always had, bit for bit. See [`BridgeVoicing`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bridge: Option<BridgeVoicing>,
    /// One entry per unison group size, 1 to [`MAX_UNISON`] strings.
    pub unison_layout: Vec<UnisonLayout>,
    /// **Inert.** Decay-rate multipliers for the individual strings of a
    /// unison, one row per group size exactly like [`Voicing::unison_layout`].
    ///
    /// It existed to write in by hand the one thing a group of *independent*
    /// oscillators cannot have: strings of one unison decaying at different
    /// rates, which is what moves a composite partial's pitch as the survivor
    /// takes over (`docs/history/TUNING_REPORT.md` §6). Under the coupled construction that
    /// split is an **output** — the bridge pushes the group's decay rates apart
    /// by construction, by a factor of 4.6 at C4's fundamental where the
    /// free-running banks were identical to the bit — so writing it in as well
    /// would be counting the same physics twice.
    ///
    /// Still parsed, still validated, never read.
    #[serde(
        default = "unity_sigma_scale",
        skip_serializing_if = "is_unity_sigma_scale"
    )]
    pub unison_sigma_scale: Vec<UnisonSigmaScale>,
    /// How firmly the damper felt grips a partial, as anchors interpolated in
    /// log frequency. Dampers hold low partials tightly and the top ones barely
    /// at all, which is the brief metallic zing on release.
    pub damper_weight: Vec<DamperAnchor>,
    /// How the hammer's blow changes *direction* with velocity.
    ///
    /// Absent — the default — is the fitted, velocity-independent strike vector,
    /// bit for bit. See [`StrikeDirection`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strike_direction: Option<StrikeDirection>,
}

/// The one place velocity can enter a linear model: the *direction* of the
/// hammer's excitation vector.
///
/// `docs/history/FUNDAMENTALS.md` §7.3 refuted the hope that the coupled construction would
/// bring velocity dependence with it, and the refutation is structural, not a
/// tolerance: the strike vector `u = s_j g_k` scales **uniformly** with
/// velocity, so `c = V^-1 u` scales uniformly too and every ratio in the mode
/// mixture is a constant. The eigenmodes are velocity-invariant by eigenvalue
/// and the mixture is velocity-invariant by linearity. Measured, that is the
/// engine's 0.006 cents and 0.054 dB of beat structure across velocities 40 /
/// 90 / 120 against the recording's **0.787 cents and 1.90 dB**
/// (`renders/jitter/EIGENMODE.md`; `docs/history/FUNDAMENTALS.md` §7.4, third fingerprint).
///
/// What *can* move with velocity is where the blow points. A hammer that
/// arrives faster arrives with its felt more compressed and its face at a
/// slightly different angle, so two ratios inside `u` move while its length
/// goes on scaling with the blow:
///
/// * the **vertical/horizontal** ratio — how much of the force goes into the
///   plane perpendicular to the soundboard and how much into the plane parallel
///   to it ([`StrikeDirection::vh_db_at_pp`], [`StrikeDirection::vh_db_at_ff`]);
/// * the **per-string share** asymmetry — the hammer is not square to the
///   strings, and how far out of square it is depends on how hard it is
///   travelling ([`StrikeDirection::share_tilt`]).
///
/// Both are ratios *inside* `u`, so neither changes the length of the strike
/// vector and neither is a second velocity law on the note's loudness: the
/// share tilt is applied about the group's mean, which stays exactly 1, and the
/// v/h ratio moves a component that starts 27.6 dB down. What they change is
/// the **mixture** `V^-1 u` — the one thing §7.3 proved a velocity-independent
/// direction cannot change.
///
/// All three fields zero is the fitted, velocity-independent strike vector at
/// every velocity, bit for bit; that is the neutrality contract, and it is what
/// `presets/default.toml` and `presets/salamander-c5.toml` both get by leaving
/// the section out.
///
/// Nothing here can be fitted yet — `docs/history/FUNDAMENTALS.md` §7.7's last row is the
/// estimator that would do it, and it is the next milestone.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StrikeDirection {
    /// Offset on [`Voicing::horizontal_gain_db`] at MIDI velocity 1, in dB.
    ///
    /// The v/h ratio is `horizontal_gain_db + lerp(pp, ff)` and the
    /// interpolation is linear in `(vel - 1) / 126`, so this end is reached
    /// exactly at the softest blow the protocol has.
    #[serde(serialize_with = "short::scalar")]
    pub vh_db_at_pp: f32,
    /// The same offset at MIDI velocity 127.
    ///
    /// A *positive* difference `ff - pp` is the physical direction: a faster
    /// hammer with more compressed felt puts more of the blow out of the
    /// vertical plane, which is why a fortissimo note has more aftersound
    /// relative to its prompt than a pianissimo one does.
    #[serde(serialize_with = "short::scalar")]
    pub vh_db_at_ff: f32,
    /// Full pianissimo-to-fortissimo swing of the group's share asymmetry, as a
    /// fraction of that asymmetry.
    ///
    /// `voicing.unison_layout.share` is a row averaging 1; this tilts how far
    /// each string sits from that mean, **about the mean and about
    /// mid-velocity**:
    ///
    /// ```text
    ///     s_j(v) = 1 + (1 + share_tilt (2 t - 1)) (s_j - 1),   t = (vel - 1)/126
    /// ```
    ///
    /// so the row still averages exactly 1 at every velocity (no loudness law
    /// smuggled in), the fitted shares are the shares of a mezzo blow, and a
    /// positive value is a hammer that meets the group *more* out of square the
    /// harder it is thrown. Zero leaves `unison_layout.share` alone at every
    /// velocity, to the last bit.
    #[serde(serialize_with = "short::scalar")]
    pub share_tilt: f32,
}

/// Largest v/h offset a [`StrikeDirection`] may declare at either end, in dB.
///
/// The quantity it offsets is `horizontal_gain_db`, fitted at −27.6 dB, and the
/// measurements that motivate a velocity dependence at all put the recording's
/// beat-depth swing at 1.90 dB over an 80-point velocity span
/// (`docs/history/FUNDAMENTALS.md` §7.4). Twelve decibels either way is six times the swing
/// that has to be explained and still leaves the horizontal plane a leak rather
/// than a second note; it is a bound on a knob nobody has fitted yet, not a
/// measurement.
pub const MAX_STRIKE_DIRECTION_DB: f32 = 12.0;
/// Largest share tilt, as a fraction of the group's fitted share asymmetry.
///
/// The shares are within a few percent of each other, so a fifth of that
/// asymmetry is already a hammer whose squareness visibly depends on how hard
/// it is thrown, and it keeps every share comfortably positive: a hammer cannot
/// pull.
pub const MAX_SHARE_TILT: f32 = 0.2;

/// Widest unison spread a key may declare, in cents (`notes.detune_cents`).
///
/// A unison is a group of strings a tuner has failed to make identical, and the
/// spread that survives that failure is single-figure cents: the shipped tables
/// run 0.44 to 3.89 and the fitted one's widest key is under four. A whole
/// semitone is four hundred times the beat rate the mechanism exists to
/// produce and is no longer one note — but the bound is here for an arithmetic
/// reason as well as a musical one. The eigenproblem's diagonal is
/// `2 pi f_k · detune_j`, so the spread multiplies **the top of the series**,
/// and the band between [`MAX_PARTIAL_RATIO`](crate::types::MAX_PARTIAL_RATIO)
/// and Nyquist is the only headroom there is: at this ceiling the widest string
/// of a three-string group sits 0.29 % sharp, which moves a 21.6 kHz partial to
/// 22.2 kHz and still leaves 1.8 kHz of it.
pub const MAX_DETUNE_CENTS: f32 = 100.0;

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

/// The bridge's driving-point admittance, as the shape of one filter.
///
/// A string does not terminate on a node: it terminates on a bridge whose
/// mobility `Y(f)` decides how fast each partial leaves into the board and, via
/// the board the whole instrument shares, how strongly one note's partials
/// reach another's (`PHYSICS.md` §4). The engine's sympathetic bus used to be
/// spectrally flat, so its halo was the same colour everywhere and no partial's
/// decay depended on where it sat. This section is that filter's shape.
///
/// It is written in two parts because the board *is* two things, split at Ege &
/// Boutillon's transition frequency `f_lim ≈ 1.1 kHz` (half a wavelength = the
/// mean rib spacing): below it a homogeneous plate with discrete, well
/// separated modes, above it waves that localise between the ribs and leave
/// only a smooth characteristic mobility. So:
///
/// * [`BridgeVoicing::backbone`] is the mean mobility — measured at roughly
///   `1.3e-3 s/kg` over 100–1000 Hz, falling in the treble — as anchors
///   interpolated in **log frequency**, which is the smooth half;
/// * [`BridgeVoicing::peaks`] are the discrete resonances, sharp and separated
///   below ~500 Hz, which is the modal half.
///
/// The whole thing is one shared filter on one mono signal, so its cost does
/// not scale with polyphony ([`crate::resonance::BridgeFilter`]).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BridgeVoicing {
    /// Mean mobility, as gains at frequencies interpolated in log `f`.
    /// 2 to [`MAX_BRIDGE_ANCHORS`] anchors, strictly ascending in `hz`.
    pub backbone: Vec<BridgeAnchor>,
    /// Discrete bridge resonances, at most [`MAX_BRIDGE_PEAKS`] of them.
    /// Empty — the default — leaves the backbone alone.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub peaks: Vec<BridgePeak>,
    /// Share of each partial's decay rate that is loss **into the board**, and
    /// therefore follows the admittance's own fluctuation: `Re Y` in the
    /// string's damping, which is the half of `PHYSICS.md` §4 that the bus
    /// cannot produce.
    ///
    /// # Why this is a share and not a rate
    ///
    /// `notes.sigma0` and `notes.sigma1` are *measured* — fitted to recorded
    /// decays — so the mean loss into the board is already inside them, once.
    /// Adding a damping proportional to `|B(f)|` would count it twice and
    /// retune the whole compass's T60. What the smooth fitted law cannot carry
    /// is the mode-to-mode *fluctuation* of the board's mobility, and that is
    /// exactly [`BridgeVoicing::peaks`]. So the partial's decay rate becomes
    ///
    /// ```text
    /// sigma_k <- sigma_k * (1 + share * (|P(f_k)| - 1))
    /// ```
    ///
    /// with `P` the peaks alone — the backbone is the mean mobility and stays
    /// where it is, in the fit. A partial on a +6 dB board mode then decays
    /// `1 + share` times faster and one in a −6 dB trough about `1 − share/2`
    /// times slower, which is the double-decay asymmetry Weinreich and
    /// Woodhouse describe, and `share` is the physical quantity Woodhouse
    /// quotes: the body-loss/air-loss ratio, above 0 dB (i.e. a share above
    /// one half) everywhere over ~160 Hz.
    ///
    /// Absent, or zero, the factor is exactly `1.0` for every partial and every
    /// string in the instrument is built bit for bit as it was before this
    /// field existed.
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
    /// `f / bandwidth`. The board's low modes are sharp; the ceiling is what
    /// keeps a resonance from becoming an oscillator in `f32`.
    #[serde(serialize_with = "short::scalar")]
    pub q: f32,
    /// Peak gain, in dB, over the backbone. Negative is an anti-resonance.
    #[serde(serialize_with = "short::scalar")]
    pub gain_db: f32,
}

/// Bounds on a `[voicing.bridge]` section. They are the schema, so the tuner's
/// copy states the same numbers; none of them is what makes the filter *safe*,
/// which is [`Preset::validate`]'s loop-gain check against the realised
/// response.
pub const MAX_BRIDGE_ANCHORS: usize = 24;
pub const MAX_BRIDGE_PEAKS: usize = 40;
/// Lowest and highest frequency an anchor or a peak may sit at. The bottom is
/// under the lowest partial the instrument has (A0 at 27.5 Hz); the top is
/// where the board has stopped being a radiator and a `Q`-50 resonator at
/// 48 kHz is still comfortably resolved.
pub const MIN_BRIDGE_HZ: f32 = 20.0;
pub const MAX_BRIDGE_HZ: f32 = 16_000.0;
/// Range of any bridge gain. The measured mobility fluctuates by ±10–15 dB
/// over the midrange, so ±20 dB of headroom is generous and −40 dB is a
/// through-going null.
pub const MIN_BRIDGE_GAIN_DB: f32 = -40.0;
pub const MAX_BRIDGE_GAIN_DB: f32 = 20.0;
pub const MIN_BRIDGE_Q: f32 = 0.5;
pub const MAX_BRIDGE_Q: f32 = 50.0;
/// Largest share of a partial's decay the admittance may be given
/// ([`BridgeVoicing::radiated_share`]). A share of 1 would let a deep enough
/// anti-resonance cancel a partial's damping altogether; nine tenths keeps the
/// factor positive whatever the peaks say, and the factor is clamped besides
/// ([`RADIATED_FACTOR_RANGE`](crate::string::RADIATED_FACTOR_RANGE)).
pub const MAX_RADIATED_SHARE: f32 = 0.9;

/// One duplex or aliquot segment of a key, as a resonance.
///
/// A string does not end at the bridge or at the agraffe: the front segment
/// (capo bar to tuning pin) and the rear segment (bridge to hitch pin) are
/// short, high-pitched, and — this is the point — **have no dampers**. They are
/// driven only through the bridge, and they ring on after the speaking length
/// has been stopped (`PHYSICS.md` §3).
///
/// Each entry is one measured segment resonance, not a ratio. Öberg &
/// Askenfelt measured every main and duplex string over D4–C8 on a
/// concert-condition grand and found real rear-duplex tuning generally *sharp*
/// of the nominal partial, with average and median deviations approaching
/// +50 cents (single keys at +190 and −100) and a spread *within* one trichord
/// averaging ~25 cents. That scatter is the sound, so the schema stores
/// frequencies and a preset must not derive them from `k·f0`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DuplexMode {
    /// Frequency of the segment, Hz.
    #[serde(serialize_with = "short::scalar")]
    pub hz: f32,
    /// Level of the segment at its own frequency, in dB relative to the bridge
    /// force driving it — i.e. the *resonant* gain, normalised so that
    /// lengthening `t60_s` makes the segment ring longer without making it
    /// louder. See [`crate::duplex`].
    #[serde(serialize_with = "short::scalar")]
    pub gain_db: f32,
    /// How long the segment rings, seconds to −60 dB. Deliberately short:
    /// nothing ever damps these banks, so their own decay is the only thing
    /// that lets a voice go back to sleep.
    #[serde(serialize_with = "short::scalar")]
    pub t60_s: f32,
}

/// Bounds on a `notes.duplex` row. As with the bridge, these are the schema and
/// the tuner's copy states the same numbers; what makes the segments *safe* is
/// [`Preset::validate`]'s loop-gain check against their realised response.
pub const MAX_DUPLEX_MODES: usize = 6;
/// Frequency range of a segment. The lowest duplex Öberg & Askenfelt measured
/// is a few hundred hertz (the segments are a small fraction of the speaking
/// length even at D4); the top is inside the band the engine still renders.
pub const MIN_DUPLEX_HZ: f32 = 200.0;
pub const MAX_DUPLEX_HZ: f32 = 18_000.0;
/// Level range. `+6 dB` is a segment that answers its own frequency twice as
/// hard as the bridge drives it, which is past any measured duplex; −60 dB is
/// inaudible under any playing.
pub const MIN_DUPLEX_GAIN_DB: f32 = -60.0;
pub const MAX_DUPLEX_GAIN_DB: f32 = 6.0;
/// Decay range. The ceiling is the real constraint of the feature: an undamped
/// bank that rings for 3 s keeps its voice awake for 3 s after every note that
/// touches it, and `PHYSICS.md` §3 asks for 0.5–2 s for exactly that reason.
pub const MIN_DUPLEX_T60_S: f32 = 0.05;
pub const MAX_DUPLEX_T60_S: f32 = 3.0;

/// One false beat: a within-string split on one partial of one key.
///
/// # What a false beat is, and why it is not any of the three things it looks
/// like
///
/// `renders/jitter/EIGENMODE.md` inverted the recording's measured beat depth
/// for the two-component pair that would produce it, partial by partial
/// (`D = 20 log10((1+r)/(1-r))`), and got the same answer at C4 and at A2: each
/// mid and low partial carries **a second component 4–7 dB down, 0.7–1.5 Hz
/// away**, at a spacing that does **not** scale with the partial number
/// (C4: 1.11, 1.48, 0.74, 0.74 Hz on k = 1..4). `docs/history/FUNDAMENTALS.md` §7.4 rules
/// out every mechanism the model already has:
///
/// * not the **unison** — a mistuning is a frequency *ratio*, so its beat rate
///   must be proportional to `k`, and C4's whole fitted detune is 0.149 Hz at
///   k = 1, seven to twenty times too narrow;
/// * not the **bridge's polarization split** — §2.2 derives 0.010 Hz from
///   Weinreich's measured admittance, a hundred times smaller, and it too grows
///   with the partial;
/// * not `voicing.horizontal_offset_hz` — the right order of magnitude in rate,
///   but 22 dB out in level (−27.6 dB against the implied −6) and the *same*
///   number on every partial of every key, which is the metronome the coupled
///   construction deleted.
///
/// What is left is Capleton's own subject ("False beats in coupled piano string
/// unisons", JASA 115(2), 2004): the two transverse planes of **one wire** at
/// genuinely different frequencies, from the wire's own geometry — non-uniform
/// diameter, an out-of-round or twisted cross-section, an asymmetric bridge-pin
/// termination. A property of one string and one partial, so it is uncorrelated
/// across `k` and across notes, which is exactly what the measurement says and
/// exactly what no note-independent constant can imitate.
///
/// # How it enters
///
/// It is an **input to the eigensolve**, not a detune applied to its output: the
/// offset goes on the diagonal of `Omega_k` for the split string's horizontal
/// entry, so the companion comes back as one of the group's own `2N`
/// eigenvalues, with the decay rate the coupled system gives it
/// (`docs/history/FUNDAMENTALS.md` §7.5 step 2, `omega_(j,h) = omega_(j,v)(1 + delta_j(k))`).
/// The mode count does not change: the false beat **moves and lifts** the
/// horizontal mode that was already there — the aftersound — rather than adding
/// anything to the bank.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FalseBeat {
    /// Which partial of this key is split, 1-based. At most one entry per
    /// partial: two splits of one wire's one partial are one split.
    pub k: u16,
    /// The split, in **hertz** — and hertz is right here where it was wrong for
    /// `horizontal_offset_hz`, because a wire's diameter irregularity is a
    /// property of that partial's own mode shape and not a ratio the whole
    /// series shares. That is the falsifiable half of the claim: fitted per
    /// string and per partial, `delta` must come back *uncorrelated* across `k`,
    /// or it is not a false beat.
    #[serde(serialize_with = "short::scalar")]
    pub hz: f32,
    /// How loud the companion stands, in dB relative to the loudest mode of the
    /// same partial — the quantity the measured beat depth inverts to, and
    /// therefore the quantity a fit can write. Solved for exactly by
    /// [`crate::string`]: the extra excitation of the split plane is chosen so
    /// that the companion eigenmode's radiated gain lands here.
    #[serde(serialize_with = "short::scalar")]
    pub db: f32,
}

/// Bounds on a `notes.false_beat` row.
///
/// At most eight splits per key, because the mechanism is a defect of one wire
/// and a note whose every partial is defective is a broken string, not a voiced
/// one. The rate band is the measurement's: the recording's implied companions
/// sit at 0.74–1.48 Hz at C4 and A2 and 2.22–5.19 Hz at C6, and 0.2–3.0 Hz
/// covers the register the mechanism was measured in with room either side —
/// under 0.2 Hz the split is a slow tilt of the decay rather than a beat, and
/// over 3.0 Hz it has left the band the ear hears as one moving partial. The
/// level band runs up to two planes of equal strength and may not go above,
/// because a companion louder than the mode it beats against is a second note.
///
/// The floor was −20 dB (1.6 dB of beat depth) while the level was inverted
/// open-loop from the recording's own depth, and it cut the mechanism off
/// exactly where the fault is loudest: the recording's bass and lower-midrange
/// *fundamentals* move 1.1–1.4 cents on beat depths of 0.8–1.4 dB, i.e.
/// companions at −27 dB, so the two cells `docs/history/FUNDAMENTALS.md` §II.5 names first —
/// A2 k=1 and k=2, 29x and 22x too still — were unwritable by construction.
/// The floor is now −40 dB (0.17 dB of depth), which is under the level at
/// which a companion still moves a resolved partial's *frequency* by a cent,
/// which is the quantity Column A measures (`DECISIONS.md` 249).
pub const MAX_FALSE_BEATS_PER_KEY: usize = 8;
pub const MIN_FALSE_BEAT_HZ: f32 = 0.2;
pub const MAX_FALSE_BEAT_HZ: f32 = 3.0;
pub const MIN_FALSE_BEAT_DB: f32 = -40.0;
pub const MAX_FALSE_BEAT_DB: f32 = 0.0;

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
/// `docs/history/TUNING_REPORT.md` §5 is the parameter set: it measured Salamander's own
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
    /// The hammer arriving, and the key and the action with it: the one
    /// mechanism event that happens *under* a note rather than beside one.
    ///
    /// Absent — the default — is silence, and silence is the neutral value here
    /// where it is not for the other four: the strike's level was never measured
    /// on its own (a `rel*` recording isolates a release, and no library
    /// isolates a blow), so the honest default is an event the engine does not
    /// play until a preset says what it sounds like. See [`StrikeNoise`].
    #[serde(default, skip_serializing_if = "is_silent_strike")]
    pub strike: StrikeNoise,
}

/// The hammer's own noise, which is the only mechanism event that is broadband
/// well past the action's 2 kHz ceiling — hence its own [`StrikeNoise::bandwidth_hz`].
///
/// Two measurements ask for it, from opposite ends. The realism benchmark finds
/// the engine's attacks **+5.2 dB more tonal** than the recordings' across six
/// phrases, worst on `staccato` (+11.3 dB) and `chords_pedal` (+9.1 dB)
/// (`renders/realism/REALISM.md`). The timbre ladder finds the first 30 ms with
/// every tracked partial subtracted 11.1 to 12.7 dB more tonal in the engine
/// than in the recording, at all three keys, and — the part that makes it a
/// missing *sound* rather than a missing partial — mixing the recording's own
/// residual back in closes it at every key and on both hosts
/// (`renders/timbre-ladder/ANALYSIS.md` §8.3). That residual sits at −20 dB
/// (C4), −16 (A2) and −10 (C6) relative to the source over the first 150 ms.
///
/// It is *not* the transient `docs/history/TUNING_REPORT.md` §4 refuted: that measurement was
/// broadband energy between the partials over the first 85 ms, and found the
/// engine within ~7 dB. What is missing is the residual's **spectrum** — its
/// flatness — which is compatible with that refutation and is what a burst
/// reaching 8 kHz supplies.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StrikeNoise {
    /// Spectral centre of the burst, Hz.
    #[serde(serialize_with = "short::scalar")]
    pub centroid_hz: f32,
    /// Time to fall 40 dB, seconds. Short: the residual the ladder measures is
    /// an attack, and the window it is measured in is 30 to 150 ms.
    #[serde(serialize_with = "short::scalar")]
    pub decay_s: f32,
    /// Upper band limit of the burst, Hz — the field the other four events do
    /// not have.
    ///
    /// Askenfelt's dummy-mass measurement puts the *structure-borne* spectrum of
    /// the action below about 2 kHz, and that is what a key-off or a pedal tray
    /// reaches the ear as ([`crate::noise::BANDWIDTH_HZ`], fixed for those four).
    /// A hammer striking a string is not structure-borne: the felt's contact
    /// noise and the string's own broadband onset radiate directly, and the
    /// residual the ladder measures at C6 is centred *above* 2 kHz. So this one
    /// event carries its own ceiling.
    #[serde(serialize_with = "short::scalar")]
    pub bandwidth_hz: f32,
    /// How far the level travels, in dB, over the full velocity range, through
    /// the tabulated level at velocity [`NOMINAL_STRIKE_VELOCITY`].
    #[serde(serialize_with = "short::scalar")]
    pub velocity_db: f32,
    /// Peak level, in dB relative to a velocity-90 strike of the same key,
    /// anchored at the keys it was measured at and interpolated across the
    /// compass — the same convention, and the same output-referenced
    /// calibration, as the other four events ([`crate::calibrate`]).
    pub level_db: Vec<NoiseAnchor>,
}

/// Velocity at which `[noise.strike]`'s tabulated level is the level played:
/// the same velocity-90 strike every mechanism level in `docs/history/TUNING_REPORT.md` §5 is
/// quoted against, so a level of −20 dB means −20 dB under *that* note.
pub const NOMINAL_STRIKE_VELOCITY: u8 = 90;

/// Bounds on `[noise.strike]`. The bandwidth's ceiling is where the felt's own
/// contact noise has stopped and the master shelf has taken over; the decay's
/// range is the attack window the ladder measures the residual in (30–150 ms),
/// a little either side.
pub const MAX_STRIKE_BANDWIDTH_HZ: f32 = 8_000.0;
pub const MIN_STRIKE_BANDWIDTH_HZ: f32 = 200.0;
pub const MIN_STRIKE_DECAY_S: f32 = 0.02;
pub const MAX_STRIKE_DECAY_S: f32 = 0.3;

/// Silence, which is what a preset that does not describe the hammer's noise
/// asks for. −200 dB is 190 dB under the quietest event anything else in this
/// file plays and is refused by nothing; the engine's own gate
/// (`noise::SILENT_AMPLITUDE`) stops the burst from being rendered at all, so
/// the neutral value is *bit-exact* silence and not a very quiet thump.
impl Default for StrikeNoise {
    fn default() -> StrikeNoise {
        StrikeNoise {
            // A plausible shape, so that a preset which only writes `level_db`
            // gets a hammer rather than a sine: the ladder's attack residual is
            // broadband and centred well above the action's events.
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

/// The level at which an event is not played at all. Not a threshold — the
/// engine's own is 160 dB below a strike — but the value a preset writes to mean
/// "this instrument does not have this sound".
pub const SILENT_LEVEL_DB: f32 = -200.0;

fn is_silent_strike(strike: &StrikeNoise) -> bool {
    *strike == StrikeNoise::default()
}

/// One mechanism event: how loud, how long, and what colour.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventNoise {
    /// Spectral centre of the burst, Hz. Well under the ~2 kHz where the
    /// action's structure-borne spectrum ends (`PHYSICS.md` §5).
    #[serde(serialize_with = "short::scalar")]
    pub centroid_hz: f32,
    /// Time to fall 40 dB, seconds — the column `docs/history/TUNING_REPORT.md` §5 reports.
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
    /// other (`docs/history/TUNING_REPORT.md` §1). Absent means zero, which is the
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
    /// Soft floor under the strike comb's nulls, one per key, as a fraction of
    /// the comb's crest: the excitation magnitude of partial `k` becomes
    /// `sqrt(sin^2(k pi x) + floor^2)` before the contact taper and the
    /// per-partial gain.
    ///
    /// `sin(k pi x)` has exact zeros; a hammer with width striking a stiff
    /// string terminated on a bridge does not. The engine's worst partial is
    /// measurably *at* those zeros — 42 dB down where the recording's deepest
    /// partial anywhere is 9.3 to 17.7 dB down and never at that index
    /// (`renders/timbre-ladder/ANALYSIS.md` §4a). Absent means zero, which is
    /// the bare comb.
    #[serde(
        default = "zero_table",
        skip_serializing_if = "is_zero_table",
        serialize_with = "short::list"
    )]
    pub comb_floor: Vec<f32>,
    /// Per-partial linear gain multipliers on the excitation comb, one row per
    /// key, 1-based in the partial index.
    ///
    /// # What the number means
    ///
    /// **The full measured ratio**: partial `k`'s amplitude at the strike in the
    /// recording, over what the engine's own excitation model predicts for it —
    /// `a_k(0) measured / a_k(0) rendered`, at one key and one velocity, through
    /// one tracker. Not the *roughness residual* it used to be, and the
    /// difference is the whole of `DECISIONS.md` 231.
    ///
    /// The old semantics fitted a smooth polynomial in `ln k` to each layer's
    /// spectrum and wrote only what was left over, on the reasoning that
    /// everything smooth in `ln k` is the hammer, the bridge and the microphone
    /// and that the engine has models for all three. Half right: the envelope
    /// really does contain the part of the blow that changes with velocity,
    /// which a velocity-independent table must not carry — but it *also*
    /// contains the engine's own error in that envelope, which does **not**
    /// change with velocity and which nothing else in the schema can hold. That
    /// error has been unfitted since the felt fits landed in Phase E, and it is
    /// audible: measured at C4 against the shipped preset it is
    /// **−6.53 / −2.55 / −0.91 / −0.05 dB** on k = 1..4, a 7.5 dB tilt, which is
    /// why the engine leads with k = 1 where the recording at velocity 90 leads
    /// with k = 2 — a note the ear places an octave differently at the attack.
    ///
    /// The tilt is still measured **on the engine** and not on the recording
    /// alone (`estimate::shaping::envelope_tilt` fits the same polynomial twice,
    /// once to each, and writes the difference), which is what divides the
    /// tracker's bias, the window and the microphone's comb out of the answer
    /// instead of modelling them; and the difference is offset so the table's
    /// geometric mean is 1, which is what stops it from becoming a second level
    /// control. It is a property of the excitation and not of the blow:
    /// velocity-independent by design, so a fit cannot smuggle a velocity law in
    /// here — that is [`StrikeDirection`]'s job, and it is a direction, not a
    /// gain. The roughness the field used to carry is still in it; it is now
    /// carried *with* the envelope error rather than instead of it.
    ///
    /// The roughness itself is unchanged and still needs a per-note, per-partial
    /// table: the measured excitation spectrum is 5–10 dB rougher than any
    /// smooth envelope times `sin(k pi x)` (engine control 2–5 dB) and the
    /// roughness is **not** shared between notes at the same frequency, so it
    /// cannot be a bridge curve (`docs/history/TUNING_REPORT.md` §3, backlog item 6).
    ///
    /// Absent — the default — is one everywhere. A row may be shorter than that
    /// key's partial count (the estimator tracks as far as it can) or empty, and
    /// every partial past its end is exactly 1.0; a row *longer* than the key's
    /// partial count is refused, because it is a table written for a different
    /// instrument.
    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        serialize_with = "short::table"
    )]
    pub partial_gains: Vec<Vec<f32>>,
    /// Per-partial multipliers on `partial_sigma(k)`, one row per key, with the
    /// same shape rules as [`NoteTables::partial_gains`].
    ///
    /// The two-exponential-plus-two-beats envelope law describes a real
    /// partial's decay to about 4 dB whatever produced it (`docs/history/TUNING_REPORT.md`
    /// §2), and the residual is per partial rather than per note: three strings
    /// and two polarizations make six components and fifteen beat rates, and the
    /// model fits two. This is the per-partial correction to the *rate*.
    ///
    /// Applied before the polarization split, so the vertical bank, the
    /// horizontal bank and the damper profile all follow it.
    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        serialize_with = "short::table"
    )]
    pub partial_sigma_scale: Vec<Vec<f32>>,
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
    /// The key's duplex and aliquot segments: the undamped lengths of string
    /// beyond the bridge and the agraffe, up to [`MAX_DUPLEX_MODES`] of them
    /// per key ([`DuplexMode`]).
    ///
    /// Empty — the default, and absent from the file — is the instrument with
    /// no segments at all, which is the engine as it was. A table that is
    /// present has one row per key, and a row may be empty: the bottom of the
    /// compass has no duplex worth the name, and Öberg & Askenfelt's survey
    /// starts at D4.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub duplex: Vec<Vec<DuplexMode>>,
    /// Within-string splits: which partials of which key beat against
    /// themselves, and how hard ([`FalseBeat`]).
    ///
    /// Empty — the default, and absent from the file — is the instrument with no
    /// false beats at all, which is the engine as it was and the only thing the
    /// equivalence contract of `DECISIONS.md` 229 pins. A table that is present
    /// has one row per key, and a row may be empty: a well-drawn wire in good
    /// condition has no measurable split, and most of them are.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub false_beat: Vec<Vec<FalseBeat>>,
    /// MIDI key numbers whose [`NoteTables::partial_gains`] and
    /// [`NoteTables::false_beat`] rows were **synthesized** from the fitted
    /// keys' own distributions rather than measured from a recording of this
    /// key.
    ///
    /// The engine does not read this: a synthesized row is played exactly like a
    /// measured one, which is the point of synthesizing it. It is carried in the
    /// preset because a number's *provenance* is part of the instrument's
    /// description, and because the alternative — a comment — does not survive
    /// the round trip through `serde` that every emitted preset makes. A library
    /// that later samples one of these keys can replace its rows and strike it
    /// from this list without having to guess which rows were measured, and
    /// `DECISIONS.md` 284's own re-fit is idempotent because it clears exactly
    /// the keys named here before drawing again.
    ///
    /// A key appears here only if a drawn row was actually *written* for it:
    /// the field says which rows are drawn, and a key whose draw was refused
    /// carries no drawn number and so has nothing to declare.
    ///
    /// Empty — the default, and absent from the file — is a preset every row of
    /// which is a measurement. Entries are strictly ascending and inside
    /// [`LOWEST_KEY`](crate::types::LOWEST_KEY)..=[`HIGHEST_KEY`](crate::types::HIGHEST_KEY),
    /// so the list cannot name a key twice or a key that does not exist.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub synthesized_texture: Vec<u8>,
    /// The keys whose [`NoteTables::partial_sigma_scale`] row was **drawn**
    /// rather than measured (`DECISIONS.md` 304).
    ///
    /// A sibling of [`NoteTables::synthesized_texture`] and deliberately not the
    /// same list, because the two do not cover the same keys and each stage has
    /// to clear exactly its own to stay idempotent. The texture stage draws for
    /// the keys the library never sampled; the decay stage draws **per band**
    /// rather than per key — a key the library did sample still has its 6-12 kHz
    /// drawn for where its own recording resolves too few partials there to read
    /// a band off — and it refuses any band whose correction its measurements
    /// cannot resolve, so its list is neither a subset nor a superset of the
    /// other. Folding both into one list would say "this key's rows are drawn"
    /// of a key with a measured gain row and a drawn decay row, which is what
    /// the field exists to distinguish.
    ///
    /// The decay stage may only declare a key here if the row is **its own**:
    /// the list is also the clearing list, and a sampled key's row can carry
    /// cells from the shaping stage that this one cannot reproduce
    /// (`DECISIONS.md` 321). On `presets/salamander-c5.toml` that leaves the six
    /// sampled keys of the top octave with no decay row at all, and the reason
    /// is measured rather than structural: A6 and C7 do resolve their 6-12 kHz
    /// band and it says the engine already decays faster than the recording
    /// there, and D#7 upward have too few partials standing over the render's
    /// own floor for anything to be closed on.
    ///
    /// Same rules as its sibling: the engine does not read it, a key appears
    /// only if a drawn row was written for it, entries are strictly ascending
    /// and inside `LOWEST_KEY..=HIGHEST_KEY`, and empty is absent from the file.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub synthesized_decay: Vec<u8>,
    /// Per-key override of [`Voicing::polarization_pan_spread`].
    ///
    /// The global scalar is one number for the whole compass, and the compass
    /// does not want one number: at the engine's ceiling of 0.4 the drift it
    /// produces is 0.24 dB at A0 and 8.67 dB at C5 against the recordings'
    /// 1.24 and 5.33 (`docs/history/TUNING_REPORT.md` §5, Milestone A update), so a spread
    /// that fits the bass overshoots the treble by 3 dB. Empty — the default,
    /// and absent from the file — means the global scalar applies to every
    /// key, which is the engine as it was.
    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        serialize_with = "short::list"
    )]
    pub pan_spread: Vec<f32>,
}

impl Preset {
    /// Reads and validates a preset file.
    pub fn load(path: &Path) -> Result<Preset, PresetError> {
        let text = std::fs::read_to_string(path).map_err(PresetError::Io)?;
        let preset = Preset::from_toml(&text)?;
        // On the *file* path only, and deliberately not in `from_toml`: a probe
        // preset built in memory by the tuner is not a user asking for a field,
        // and a warning printed once per render is not a warning.
        for field in preset.inert_fields() {
            eprintln!(
                "warning: {}: `{field}` is accepted but no longer read - the \
                 coupled-eigenmode unison derives what it used to assert \
                 (`docs/history/FUNDAMENTALS.md` §5, `DECISIONS.md` 225)",
                path.display()
            );
        }
        Ok(preset)
    }

    pub fn from_toml(text: &str) -> Result<Preset, PresetError> {
        let preset: Preset = toml::from_str(text).map_err(PresetError::Parse)?;
        preset.validate()?;
        Ok(preset)
    }

    /// The fields this preset sets that the string construction no longer
    /// reads, by name — empty for a preset that leaves all three at the value
    /// the free-running unison called neutral.
    ///
    /// They are kept in the schema rather than removed so that every preset
    /// already written still loads, and so that the tuner's copy of the schema
    /// (`tuner/src/preset.rs`, deliberately independent) does not have to move
    /// in the same commit. What they used to do is now derived:
    /// `unison_coupling` is `radiated_share * sigma_k` (it is the *same
    /// coefficient* as the radiation damping, `docs/history/FUNDAMENTALS.md` §1.1),
    /// `horizontal_offset_hz` is the bridge's reactive anisotropy times the
    /// partial's own frequency, and `unison_sigma_scale` is the eigenproblem's
    /// own output. See `DECISIONS.md` 225.
    pub fn inert_fields(&self) -> Vec<&'static str> {
        let v = &self.voicing;
        let mut inert = Vec::new();
        if v.unison_coupling != 0.0 {
            inert.push("voicing.unison_coupling");
        }
        if v.horizontal_offset_hz.iter().any(|&o| o != 0.0) {
            inert.push("voicing.horizontal_offset_hz");
        }
        if !is_unity_sigma_scale(&v.unison_sigma_scale) {
            inert.push("voicing.unison_sigma_scale");
        }
        inert
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
        // The two tables the loop above could only check the sign of, and both
        // are bounds the coupled construction needs rather than tidiness. The
        // spread multiplies every partial's frequency, so it spends the band
        // headroom above the partial cap; and `sigma0` is the floor of the whole
        // loss budget — `sigma_int = (1 - share) · scale · sigma_k` — which is
        // what `string::stable_sigma` asserts the sign of, on the `f64` the
        // Durand-Kerner solve returns. A strictly positive denormal passes every
        // check in the loop above and comes back out of the solve *negative*,
        // at the size of the solver's own rounding, because there is no signal
        // left in it: at `1e-44` the fitted T60 is `7e44` seconds. The floor is
        // [`MIN_MODE_SIGMA`], which is the rate under which the resonator's own
        // pole radius rounds to one in `f32` — a note that decays slower than
        // that is not a note this engine can hold, and 0.02 is still a T60 of
        // 345 s.
        for (i, &c) in n.detune_cents.iter().enumerate() {
            within(&format!("notes.detune_cents[{i}]"), c, 0.0, MAX_DETUNE_CENTS)?;
        }
        for (i, &s) in n.sigma0.iter().enumerate() {
            if s < MIN_MODE_SIGMA {
                return Err(PresetError::invalid(format!(
                    "notes.sigma0[{i}] is {s}, expected at least {MIN_MODE_SIGMA} \
                     (a T60 of 345 s)"
                )));
            }
        }
        // The fourth-order inharmonicity is the one signed table: the sign is
        // the finding (`docs/history/TUNING_REPORT.md` §1), so only finiteness can be
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
        // Checked here rather than with the per-partial tables below, because
        // `string_params` reads it and the radicand loop calls `string_params`.
        table_length("comb_floor", n.comb_floor.len())?;
        for (i, &floor) in n.comb_floor.iter().enumerate() {
            within(&format!("notes.comb_floor[{i}]"), floor, 0.0, MAX_COMB_FLOOR)?;
        }
        // The per-key stereo spread is either absent — the global scalar
        // applies — or a whole compass of them, each inside the same range the
        // scalar is held to, because each one reaches `Soundboard::add_voice`
        // as a pan displacement in exactly the same way.
        if !n.pan_spread.is_empty() {
            table_length("pan_spread", n.pan_spread.len())?;
            for (i, &s) in n.pan_spread.iter().enumerate() {
                within(&format!("notes.pan_spread[{i}]"), s, 0.0, MAX_PAN_SPREAD)?;
            }
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
                // Except at `k = 1`, which is not an end at all:
                // `StringParams::partial_count` floors its `take_while` at one,
                // so a key whose *fundamental* is already past the cap still
                // gets a bank — of one partial, out of the band the resonator is
                // defined in. The floor is right (a key with no partials is not
                // a key); what is wrong is reaching it, so the tuning is refused
                // here instead.
                if f >= limit {
                    if k == 1 {
                        return Err(PresetError::invalid(format!(
                            "notes.f0_hz[{i}] = {} puts the key's own fundamental at {f} Hz, \
                             at or past the {limit} Hz cap the partial series stops at",
                            p.f0
                        )));
                    }
                    break;
                }
                previous = f;
            }
        }
        // After the series check, so that the partial counts the row lengths
        // are measured against are counts of a bank the engine will really
        // build.
        self.validate_partial_tables()?;
        self.validate_false_beats()?;
        self.validate_synthesized_texture()?;
        self.validate_provenance("notes.synthesized_decay", &self.notes.synthesized_decay)?;

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
        within(
            "voicing.unison_coupling",
            v.unison_coupling,
            0.0,
            LEGACY_MAX_UNISON_COUPLING,
        )?;
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
            // Signed: the physical direction is a hammer that leaks *more*
            // sideways the harder it is thrown, but the sign of the share tilt
            // is a property of one action and nothing here knows which way it
            // goes, so only the magnitude is bounded.
            within(
                "voicing.strike_direction.share_tilt",
                d.share_tilt,
                -MAX_SHARE_TILT,
                MAX_SHARE_TILT,
            )?;
        }
        let max_b = self.validate_bridge()?;
        self.validate_duplex(max_b)?;
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
        // A band limit under the centroid is a burst whose energy is outside
        // its own band, and the two are then both describing the same thing
        // badly.
        if strike.centroid_hz > strike.bandwidth_hz {
            return Err(PresetError::invalid(format!(
                "noise.strike.centroid_hz is {} but its bandwidth_hz is {}, so the burst is \
                 centred outside its own band",
                strike.centroid_hz, strike.bandwidth_hz
            )));
        }
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
    /// different tuning, a different sample rate, or a different partial cap)
    /// and the entries the engine would silently drop are exactly the ones that
    /// say so.
    fn validate_partial_tables(&self) -> Result<(), PresetError> {
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
                return Err(PresetError::invalid(format!(
                    "notes.{name} has {} rows, expected {NUM_KEYS} (or none at all)",
                    table.len()
                )));
            }
            for (i, row) in table.iter().enumerate() {
                let key = index_to_note(i);
                let partials = self.string_params(key).partial_count();
                if row.len() > partials {
                    return Err(PresetError::invalid(format!(
                        "notes.{name}[{i}] (key {key}) has {} entries, but that key has only \
                         {partials} partials",
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

    /// Checks the within-string splits.
    ///
    /// Same all-or-nothing shape rule as the two per-partial tables — a preset
    /// either has the table or does not — and inside a key the row is a list and
    /// not a series, so what is checked is that every entry names a partial the
    /// key really has, that no partial is named twice (two splits of one wire's
    /// one partial are one split, and the second would silently overwrite the
    /// first), and that the rate and the level are inside the band the mechanism
    /// was measured in.
    fn validate_false_beats(&self) -> Result<(), PresetError> {
        let table = &self.notes.false_beat;
        if table.is_empty() {
            return Ok(());
        }
        if table.len() != NUM_KEYS {
            return Err(PresetError::invalid(format!(
                "notes.false_beat has {} rows, expected {NUM_KEYS} (or none at all)",
                table.len()
            )));
        }
        for (i, row) in table.iter().enumerate() {
            let key = index_to_note(i);
            if row.len() > MAX_FALSE_BEATS_PER_KEY {
                return Err(PresetError::invalid(format!(
                    "notes.false_beat[{i}] (key {key}) has {} entries, expected at most \
                     {MAX_FALSE_BEATS_PER_KEY}",
                    row.len()
                )));
            }
            let partials = self.string_params(key).partial_count();
            for (e, entry) in row.iter().enumerate() {
                let at = format!("notes.false_beat[{i}][{e}]");
                if entry.k == 0 || entry.k as usize > partials {
                    return Err(PresetError::invalid(format!(
                        "{at}.k is {}, but key {key} has partials 1..={partials}",
                        entry.k
                    )));
                }
                within(&format!("{at}.hz"), entry.hz, MIN_FALSE_BEAT_HZ, MAX_FALSE_BEAT_HZ)?;
                within(&format!("{at}.db"), entry.db, MIN_FALSE_BEAT_DB, MAX_FALSE_BEAT_DB)?;
                if row[..e].iter().any(|other| other.k == entry.k) {
                    return Err(PresetError::invalid(format!(
                        "{at} splits partial {} of key {key} a second time",
                        entry.k
                    )));
                }
            }
        }
        Ok(())
    }

    /// Checks the provenance list: real keys, in order, each named once.
    ///
    /// Strictly ascending does both jobs at once, and it is checked rather than
    /// sorted on load because a list that names a key twice is a list somebody
    /// built by appending, and the second entry is the one that would be lost.
    fn validate_synthesized_texture(&self) -> Result<(), PresetError> {
        self.validate_provenance("notes.synthesized_texture", &self.notes.synthesized_texture)
    }

    /// One provenance list, named so that the message says which one.
    fn validate_provenance(&self, name: &str, list: &[u8]) -> Result<(), PresetError> {
        let mut previous: Option<u8> = None;
        for &key in list {
            if !(LOWEST_KEY..=HIGHEST_KEY).contains(&key) {
                return Err(PresetError::invalid(format!(
                    "{name} names key {key}, outside {LOWEST_KEY}..={HIGHEST_KEY}"
                )));
            }
            if let Some(last) = previous {
                if key <= last {
                    return Err(PresetError::invalid(format!(
                        "{name} is not strictly ascending: {key} after {last}"
                    )));
                }
            }
            previous = Some(key);
        }
        Ok(())
    }

    /// Checks the bridge admittance's shape *and* what it does to the coupling
    /// loop.
    ///
    /// The shape checks are the usual ones — a resonator at or past Nyquist is
    /// a pole outside the unit circle, a `Q` of zero divides by zero, anchors
    /// out of order read the wrong interpolation pair. The loop check is the
    /// new one, and it is the reason this filter can exist at all.
    ///
    /// # Why a loop gain has to be computed rather than assumed
    ///
    /// `resonance.rs`'s stability argument was written for a *flat* bus: a
    /// string answers a steady drive at one of its own partials with at most
    /// about one signal unit per unit drive, so the tightest loop
    /// string → bus → string has gain `≈ coupling`, and bounding `coupling`
    /// bounded the loop. With `B` in the path the same loop has gain
    /// `≈ coupling · |B(f)|` at the frequency where it closes, and `B` is
    /// allowed gain well over one at its resonances — a preset could put +20 dB
    /// on every one of forty cascaded peaks and multiply the loop by a
    /// thousand. So the quantity that has to be bounded is not `coupling` but
    /// the **effective** coupling `coupling · max|B(f)|`, and `max|B|` is a
    /// property of the *realised* filter (the fitted shelf cascade and the
    /// peaking sections that were actually built), not of the numbers in the
    /// file. It is therefore measured, by evaluating the realised transfer
    /// function on a 512-point log grid from 20 Hz to 20 kHz **and scanning
    /// finely through every resonance** — a cascade *adds* decibels, so the
    /// maximum of two overlapping peaks lies between their centres, where
    /// neither the grid nor the centres look; `BridgeFilter::max_magnitude`
    /// documents the construction, entirely inside this schema, that hides
    /// 15.6 dB from a grid and a list of centres.
    ///
    /// The bound itself: with the worst-case string admittance of ~1 unit per
    /// unit drive, a loop closes when `coupling · max|B| · (coincident
    /// partials) ≥ 1`. [`MAX_BRIDGE_LOOP_GAIN`] is a quarter of that with a
    /// single coincidence, i.e. 12 dB of margin against the worst string in the
    /// instrument and 4× more against any realistic cluster. `MAX_COUPLING`
    /// still bounds `coupling` on its own, so a unity `B` is exactly as
    /// constrained as it was, and `DRIVE_CEILING` remains the hard backstop
    /// that holds whatever the tables say.
    ///
    /// Returns the measured `max|B|`, so that the duplex check and
    /// [`Preset::max_safe_coupling`] read the same number this refused on
    /// rather than measuring the filter a second time.
    fn validate_bridge(&self) -> Result<f32, PresetError> {
        let Some(bridge) = &self.voicing.bridge else {
            return Ok(1.0);
        };
        let n = bridge.backbone.len();
        if !(2..=MAX_BRIDGE_ANCHORS).contains(&n) {
            return Err(PresetError::invalid(format!(
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
        // The backbone is interpolated in log f between neighbours, so the
        // anchors have to be strictly ascending: out of order it reads the
        // wrong pair, and two at one frequency divide by a zero span.
        if let Some(i) = bridge.backbone.windows(2).position(|w| w[0].hz >= w[1].hz) {
            return Err(PresetError::invalid(format!(
                "voicing.bridge.backbone[{}] is at {} Hz, not above the {} Hz before it",
                i + 1,
                bridge.backbone[i + 1].hz,
                bridge.backbone[i].hz
            )));
        }
        if bridge.peaks.len() > MAX_BRIDGE_PEAKS {
            return Err(PresetError::invalid(format!(
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

        // Everything above is well-formed; this is whether it is *safe*.
        let filter = BridgeFilter::new(bridge);
        let max_b = filter.max_magnitude();
        if !max_b.is_finite() {
            return Err(PresetError::invalid(format!(
                "voicing.bridge has a response of {max_b} somewhere in the audio band"
            )));
        }
        let loop_gain = self.voicing.resonance_coupling * max_b;
        if loop_gain > MAX_BRIDGE_LOOP_GAIN {
            return Err(PresetError::invalid(format!(
                "voicing.bridge peaks at {:.1} dB, which with resonance_coupling = {} makes a \
                 sympathetic loop gain of {loop_gain}, past the {MAX_BRIDGE_LOOP_GAIN} the bus \
                 is stable under",
                amp_to_db(max_b),
                self.voicing.resonance_coupling
            )));
        }
        Ok(max_b)
    }

    /// Checks the duplex segments' shape *and* what 88 permanently undamped
    /// banks do to the coupling loop.
    ///
    /// The shape checks are the usual ones. The loop check is the one that
    /// matters, and it is a different loop from the bridge's: a duplex bank is
    /// never damped by anything, so a marginal loop through it has forever to
    /// grow, and there are 88 of them all reading and writing the same bus.
    ///
    /// # The bound
    ///
    /// Segment `j` puts `D_j(f)` of signal on the bus per unit of drive at `f`
    /// (its realised response, [`crate::duplex::magnitude`]), and gets back
    /// `coupling · B(f)` of every other segment's output one block later, plus
    /// `coupling · (B(f) − own_gain_j)` of its own — the bus subtracts a voice's
    /// own contribution, exactly when the bridge is flat and to within the
    /// admittance's own tilt when it is not (`resonance.rs`). So the tightest
    /// loop any frequency can close is
    ///
    /// ```text
    /// coupling · max|B| · ( sum_j |D_j(f)| + max_j |D_j(f)| )
    /// ```
    ///
    /// — the sum being every segment in the instrument that answers at that
    /// frequency and the extra term being the self-path's worst case. It is
    /// evaluated at every segment's own centre frequency, which is where a sum
    /// of resonances peaks, and bounded by
    /// [`MAX_DUPLEX_LOOP_GAIN`](crate::duplex::MAX_DUPLEX_LOOP_GAIN) — a
    /// quarter of unity, the same margin the bridge is held to.
    ///
    /// What this refuses is the preset that tunes every key's segments to the
    /// *same* frequency, which is 88 undamped resonators in one loop and is
    /// also, per Öberg & Askenfelt, not what a piano does: real duplex tuning
    /// scatters by tens of cents, and two Q-of-several-thousand resonators tens
    /// of cents apart contribute nothing to each other's loop. A preset whose
    /// segments are measured passes this by two orders of magnitude.
    fn validate_duplex(&self, max_b: f32) -> Result<(), PresetError> {
        let table = &self.notes.duplex;
        if table.is_empty() {
            return Ok(());
        }
        if table.len() != NUM_KEYS {
            return Err(PresetError::invalid(format!(
                "notes.duplex has {} rows, expected {NUM_KEYS} (or none at all)",
                table.len()
            )));
        }
        for (i, row) in table.iter().enumerate() {
            if row.len() > MAX_DUPLEX_MODES {
                return Err(PresetError::invalid(format!(
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

        // Everything above is well-formed; this is whether it is *safe*.
        let worst = self.duplex_response();
        let loop_gain = self.voicing.resonance_coupling * max_b * worst;
        if loop_gain > MAX_DUPLEX_LOOP_GAIN {
            return Err(PresetError::invalid(format!(
                "notes.duplex answers {worst} per unit of drive where its segments crowd \
                 together, which with resonance_coupling = {} and a bridge peaking at \
                 {:.1} dB makes an undamped loop gain of {loop_gain}, past the \
                 {MAX_DUPLEX_LOOP_GAIN} the bus is stable under",
                self.voicing.resonance_coupling,
                amp_to_db(max_b)
            )));
        }
        Ok(())
    }

    /// The worst `sum_j |D_j(f)| + max_j |D_j(f)|` over the frequencies the
    /// duplex table can close a loop at, i.e. the bracketed factor of
    /// [`Preset::validate_duplex`]'s bound. Zero when there is no table.
    fn duplex_response(&self) -> f32 {
        let table = &self.notes.duplex;
        let mut worst = 0.0f32;
        for probe in table.iter().flatten() {
            let (mut total, mut largest) = (0.0f32, 0.0f32);
            for row in table {
                let d = crate::duplex::magnitude(row, probe.hz);
                total += d;
                largest = largest.max(d);
            }
            worst = worst.max(total + largest);
        }
        worst
    }

    /// The largest `voicing.resonance_coupling` this preset may run at without
    /// breaking either loop bound — the *whole* contract in one number.
    ///
    /// It is the smallest of three things: [`MAX_COUPLING`], which bounds the
    /// scalar on its own and is what a flat bus has always been held to;
    /// `MAX_BRIDGE_LOOP_GAIN / max|B|`, the string → bus → string loop
    /// ([`Preset::validate_bridge`]); and, when the preset has segments,
    /// `MAX_DUPLEX_LOOP_GAIN / (max|B| · duplex response)`, the undamped loop
    /// through them ([`Preset::validate_duplex`]).
    ///
    /// `validate` refuses anything above this; [`ResonanceBus`] clamps to it, so
    /// that a *live* change to the coupling cannot walk a loaded preset past
    /// the bound its bridge was validated against. Measuring the realised
    /// filter costs a few milliseconds, so this is a construction-time call —
    /// never one the audio path makes.
    ///
    /// [`ResonanceBus`]: crate::resonance::ResonanceBus
    /// [`MAX_COUPLING`]: crate::resonance::MAX_COUPLING
    /// [`MAX_BRIDGE_LOOP_GAIN`]: crate::resonance::MAX_BRIDGE_LOOP_GAIN
    /// [`MAX_DUPLEX_LOOP_GAIN`]: crate::duplex::MAX_DUPLEX_LOOP_GAIN
    pub fn max_safe_coupling(&self) -> f32 {
        let max_b = match &self.voicing.bridge {
            Some(bridge) => BridgeFilter::new(bridge).max_magnitude(),
            None => 1.0,
        };
        self.coupling_ceiling(max_b)
    }

    /// [`Preset::max_safe_coupling`] with `max|B|` already measured.
    pub(crate) fn coupling_ceiling(&self, max_b: f32) -> f32 {
        let mut ceiling = MAX_COUPLING.min(MAX_BRIDGE_LOOP_GAIN / max_b);
        let duplex = self.duplex_response();
        if duplex > 0.0 {
            ceiling = ceiling.min(MAX_DUPLEX_LOOP_GAIN / (max_b * duplex));
        }
        ceiling.max(0.0)
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
            comb_floor: n.comb_floor[i],
            sigma0: n.sigma0[i],
            sigma1: n.sigma1[i],
            unison: n.unison[i] as usize,
            detune_cents: n.detune_cents[i],
            impedance: n.impedance[i],
            damper_sigma: n.damper_sigma[i],
            bridge_gain: n.bridge_gain[i],
        }
    }

    /// This key's per-partial excitation and decay tables, empty where the
    /// preset has none.
    ///
    /// A preset with no table at all, one whose table has an empty row for this
    /// key, and one whose row stops short of the key's partial count are the
    /// same instrument from here up: everything past the end of a row is 1.0
    /// ([`PartialShaping`]).
    pub fn partial_shaping(&self, key: u8) -> PartialShaping<'_> {
        let i = self.index(key);
        PartialShaping {
            gains: row(&self.notes.partial_gains, i),
            sigma_scale: row(&self.notes.partial_sigma_scale, i),
            false_beat: match self.notes.false_beat.get(i) {
                Some(row) => row,
                None => &[],
            },
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

    /// This key's duplex and aliquot segments, empty when the preset has none.
    ///
    /// A preset with no `notes.duplex` table at all and one whose table has an
    /// empty row for this key are the same instrument here, which is what lets
    /// the table be absent from the file.
    pub fn duplex_modes(&self, key: u8) -> &[DuplexMode] {
        match self.notes.duplex.get(self.index(key)) {
            Some(row) => row,
            None => &[],
        }
    }

    /// How far apart this key's two polarizations sit in the stereo image.
    ///
    /// `notes.pan_spread` when the preset has one, and
    /// `voicing.polarization_pan_spread` when it does not — so a preset written
    /// before the table existed behaves exactly as it did.
    pub fn pan_spread(&self, key: u8) -> f32 {
        match self.notes.pan_spread.get(self.index(key)) {
            Some(&spread) => spread,
            None => self.voicing.polarization_pan_spread,
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

/// The mechanism as `docs/history/TUNING_REPORT.md` §5 measured it.
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
            // Silent, and the only one of the five that is: nothing in
            // `docs/history/TUNING_REPORT.md` §5 measured a hammer on its own, so the level
            // belongs in the preset that was fitted with one.
            strike: StrikeNoise::default(),
        }
    }
}

/// One key's row of a ragged per-partial table, empty when the table is absent.
fn row(table: &[Vec<f32>], i: usize) -> &[f32] {
    match table.get(i) {
        Some(row) => row,
        None => &[],
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

/// The neutral [`Voicing::horizontal_offset_hz`]: no fixed hertz split on any
/// string of a unison. The field is inert (`DECISIONS.md` 225), so this is not
/// a *setting* — it is the one value that also switches off the warning.
fn no_horizontal_offset() -> Vec<f32> {
    vec![0.0; MAX_UNISON]
}

fn is_no_horizontal_offset(offsets: &[f32]) -> bool {
    offsets.len() == MAX_UNISON && offsets.iter().all(|&o| o == 0.0)
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

/// The checks every mechanism event shares, whatever struct it lives in.
///
/// All of it reaches a biquad's coefficients and an exponential envelope on the
/// audio path: a centroid at or past Nyquist is a pole outside the unit circle,
/// a decay at zero is an event that never ends, and an unordered anchor list is
/// read by `interp_anchors` in the wrong pairs.
fn validate_event(
    name: &str,
    centroid_hz: f32,
    decay_s: f32,
    decay_range: (f32, f32),
    velocity_db: f32,
    level_db: &[NoiseAnchor],
) -> Result<(), PresetError> {
    positive(&format!("noise.{name}.centroid_hz"), centroid_hz)?;
    within(
        &format!("noise.{name}.centroid_hz"),
        centroid_hz,
        1.0,
        0.45 * crate::types::SAMPLE_RATE,
    )?;
    within(
        &format!("noise.{name}.decay_s"),
        decay_s,
        decay_range.0,
        decay_range.1,
    )?;
    finite(&format!("noise.{name}.velocity_db"), velocity_db)?;
    if level_db.is_empty() {
        return Err(PresetError::invalid(format!(
            "noise.{name}.level_db is empty"
        )));
    }
    for (i, anchor) in level_db.iter().enumerate() {
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
    if let Some(i) = level_db.windows(2).position(|w| w[0].key >= w[1].key) {
        return Err(PresetError::invalid(format!(
            "noise.{name}.level_db[{}] is at key {}, not above the key {} before it",
            i + 1,
            level_db[i + 1].key,
            level_db[i].key
        )));
    }
    Ok(())
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
                // The two inert fields of `DECISIONS.md` 225, at the value that
                // is both neutral and silent. They used to carry 0.35 / 0.52 /
                // 0.27 Hz and 0.02 — the free-running unison's metronome and
                // its one-block-late cross-feed — and the coupled construction
                // reads neither. Carrying them would put two numbers nothing
                // computes into every preset written from this one.
                horizontal_offset_hz: no_horizontal_offset(),
                unison_coupling: 0.0,
                resonance_coupling: 0.012,
                // The hand-tuned instrument is the point-force, one-`B`,
                // one-damping-law, one-pan-position piano v1 was: every field
                // added since Phase E sits at its neutral value here, and none
                // of them is written to the file.
                polarization_pan_spread: 0.0,
                // ... and the bus it couples through is still the flat one.
                bridge: None,
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
                // The fitted strike vector, at every velocity. Nothing in the
                // tuner can fit a velocity dependence yet (`docs/history/FUNDAMENTALS.md`
                // §7.7's last row), and the default is the instrument as it was.
                strike_direction: None,
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
                comb_floor: zero_table(),
                partial_gains: Vec::new(),
                partial_sigma_scale: Vec::new(),
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
                // No duplex segments: the instrument as it was before they
                // existed. A preset that has them writes every one of them.
                duplex: Vec::new(),
                // No within-string splits: a false beat is a defect of one
                // wire, and a synthetic instrument has none until one is
                // measured on it.
                false_beat: Vec::new(),
                // Every row of the default preset is a law or a measurement,
                // and none of it is drawn.
                synthesized_texture: Vec::new(),
                synthesized_decay: Vec::new(),
                // No per-key override: `voicing.polarization_pan_spread`
                // applies to the whole compass, as it always did.
                pan_spread: Vec::new(),
            },
            // The action as `docs/history/TUNING_REPORT.md` §5 measured it. Unlike the other
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

    /// The same, one row per key: the per-partial tables are ragged, so they
    /// are a list of lists rather than a list.
    pub fn table<S: Serializer>(rows: &[Vec<f32>], s: S) -> Result<S::Ok, S::Error> {
        let mut seq = s.serialize_seq(Some(rows.len()))?;
        for row in rows {
            let widened: Vec<f64> = row.iter().map(|&x| widen(x)).collect();
            seq.serialize_element(&widened)?;
        }
        seq.end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{HIGHEST_KEY, LOWEST_KEY, MAX_PARTIAL_RATIO, SAMPLE_RATE};

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

    fn peak(hz: f32, q: f32, gain_db: f32) -> BridgePeak {
        BridgePeak { hz, q, gain_db }
    }

    /// A bridge that does nothing: two anchors at 0 dB, and whatever peaks the
    /// caller wants on top.
    fn flat_bridge(peaks: Vec<BridgePeak>) -> BridgeVoicing {
        BridgeVoicing {
            backbone: vec![
                BridgeAnchor {
                    hz: MIN_BRIDGE_HZ,
                    gain_db: 0.0,
                },
                BridgeAnchor {
                    hz: MAX_BRIDGE_HZ,
                    gain_db: 0.0,
                },
            ],
            peaks,
            radiated_share: 0.0,
        }
    }

    fn flat_bridge_with(
        peaks: Vec<BridgePeak>,
        break_it: impl Fn(&mut Vec<BridgeAnchor>),
    ) -> BridgeVoicing {
        let mut bridge = flat_bridge(peaks);
        break_it(&mut bridge.backbone);
        bridge
    }

    fn segment(hz: f32) -> DuplexMode {
        DuplexMode {
            hz,
            gain_db: -30.0,
            t60_s: 1.0,
        }
    }

    /// A legal one-segment-per-key duplex table — scattered, as a real one is —
    /// with one field of one segment broken by the caller.
    fn duplex_with(break_it: impl Fn(&mut DuplexMode)) -> Vec<Vec<DuplexMode>> {
        let mut table: Vec<Vec<DuplexMode>> = (0..NUM_KEYS)
            .map(|i| vec![segment(2_000.0 + 13.0 * i as f32)])
            .collect();
        break_it(&mut table[7][0]);
        table
    }

    /// One anchor is not a curve: the backbone needs two to interpolate.
    /// A whole compass of splits, one on E3's second partial, broken by `f`.
    fn false_beat_with(break_it: impl Fn(&mut FalseBeat)) -> Vec<Vec<FalseBeat>> {
        let mut entry = FalseBeat {
            k: 2,
            hz: 1.0,
            db: -6.0,
        };
        break_it(&mut entry);
        let mut rows = vec![Vec::new(); NUM_KEYS];
        rows[39] = vec![entry];
        rows
    }

    /// The neutral velocity law, broken by `f`.
    fn direction_with(break_it: impl Fn(&mut StrikeDirection)) -> StrikeDirection {
        let mut d = StrikeDirection {
            vh_db_at_pp: 0.0,
            vh_db_at_ff: 0.0,
            share_tilt: 0.0,
        };
        break_it(&mut d);
        d
    }

    fn one_anchor_bridge() -> BridgeVoicing {
        let mut bridge = flat_bridge(Vec::new());
        bridge.backbone.truncate(1);
        bridge
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
        let breakages: [fn(&mut Preset); 146] = [
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
            // A bridge admittance reaches a cascade of biquads on the audio
            // path and, unlike everything else here, is allowed gain over one:
            // what has to be refused is a shape that is malformed *and* a
            // shape that is well formed but takes the coupling loop past the
            // point where it sustains itself.
            |p: &mut Preset| p.voicing.bridge = Some(one_anchor_bridge()),
            |p: &mut Preset| {
                p.voicing.bridge = Some(BridgeVoicing {
                    backbone: (0..MAX_BRIDGE_ANCHORS + 1)
                        .map(|i| BridgeAnchor {
                            hz: 20.0 + i as f32,
                            gain_db: 0.0,
                        })
                        .collect(),
                    peaks: Vec::new(),
                    radiated_share: 0.0,
                })
            },
            |p: &mut Preset| p.voicing.bridge = Some(flat_bridge_with(Vec::new(), |b| b[1].hz = 10.0)),
            |p: &mut Preset| {
                p.voicing.bridge = Some(flat_bridge_with(Vec::new(), |b| b[1].hz = 20_000.0))
            },
            |p: &mut Preset| p.voicing.bridge = Some(flat_bridge_with(Vec::new(), |b| b[1].hz = 20.0)),
            |p: &mut Preset| {
                p.voicing.bridge = Some(flat_bridge_with(Vec::new(), |b| {
                    b[0].hz = MAX_BRIDGE_HZ;
                    b[1].hz = MIN_BRIDGE_HZ;
                }))
            },
            |p: &mut Preset| {
                p.voicing.bridge = Some(flat_bridge_with(Vec::new(), |b| b[0].hz = f32::NAN))
            },
            |p: &mut Preset| {
                p.voicing.bridge = Some(flat_bridge_with(Vec::new(), |b| b[0].gain_db = 30.0))
            },
            |p: &mut Preset| {
                p.voicing.bridge = Some(flat_bridge_with(Vec::new(), |b| b[0].gain_db = -60.0))
            },
            |p: &mut Preset| {
                p.voicing.bridge = Some(flat_bridge_with(Vec::new(), |b| b[0].gain_db = f32::NAN))
            },
            |p: &mut Preset| {
                p.voicing.bridge = Some(flat_bridge(
                    (0..MAX_BRIDGE_PEAKS + 1)
                        .map(|i| BridgePeak {
                            hz: 100.0 + i as f32,
                            q: 5.0,
                            gain_db: 0.0,
                        })
                        .collect(),
                ))
            },
            |p: &mut Preset| p.voicing.bridge = Some(flat_bridge(vec![peak(19.0, 5.0, 0.0)])),
            |p: &mut Preset| p.voicing.bridge = Some(flat_bridge(vec![peak(17_000.0, 5.0, 0.0)])),
            |p: &mut Preset| p.voicing.bridge = Some(flat_bridge(vec![peak(200.0, 0.2, 0.0)])),
            |p: &mut Preset| p.voicing.bridge = Some(flat_bridge(vec![peak(200.0, 80.0, 0.0)])),
            |p: &mut Preset| p.voicing.bridge = Some(flat_bridge(vec![peak(200.0, f32::NAN, 0.0)])),
            |p: &mut Preset| p.voicing.bridge = Some(flat_bridge(vec![peak(200.0, 5.0, 25.0)])),
            |p: &mut Preset| p.voicing.bridge = Some(flat_bridge(vec![peak(200.0, 5.0, f32::NAN)])),
            // Well formed, in range, and past the loop bound: `MAX_COUPLING`
            // with a +20 dB resonance is an effective coupling of 0.5.
            |p: &mut Preset| {
                p.voicing.resonance_coupling = MAX_COUPLING;
                p.voicing.bridge = Some(flat_bridge(vec![peak(200.0, 5.0, 20.0)]));
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
            // The duplex table is per key, so it is all 88 keys or none: a
            // short one would silently give the top of the compass no segments.
            |p: &mut Preset| p.notes.duplex = vec![Vec::new(); NUM_KEYS - 1],
            |p: &mut Preset| {
                p.notes.duplex = duplex_with(|_| {});
                p.notes.duplex[7] = (0..=MAX_DUPLEX_MODES)
                    .map(|k| segment(4_000.0 + 11.0 * k as f32))
                    .collect();
            },
            |p: &mut Preset| p.notes.duplex = duplex_with(|m| m.hz = MIN_DUPLEX_HZ - 1.0),
            |p: &mut Preset| p.notes.duplex = duplex_with(|m| m.hz = MAX_DUPLEX_HZ + 1.0),
            |p: &mut Preset| p.notes.duplex = duplex_with(|m| m.hz = f32::NAN),
            |p: &mut Preset| p.notes.duplex = duplex_with(|m| m.gain_db = 12.0),
            |p: &mut Preset| p.notes.duplex = duplex_with(|m| m.gain_db = -80.0),
            |p: &mut Preset| p.notes.duplex = duplex_with(|m| m.gain_db = f32::NAN),
            |p: &mut Preset| p.notes.duplex = duplex_with(|m| m.t60_s = 0.0),
            // A segment that outlives the note keeps its voice awake for as
            // long as it rings, which is why the ceiling is a hard one.
            |p: &mut Preset| p.notes.duplex = duplex_with(|m| m.t60_s = MAX_DUPLEX_T60_S + 1.0),
            |p: &mut Preset| p.notes.duplex = duplex_with(|m| m.t60_s = f32::NAN),
            // Well formed, in range, and past the loop bound: six segments on
            // one frequency on every key of the instrument, never damped.
            |p: &mut Preset| {
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
            // The per-key stereo spread is the same all-or-nothing table, and
            // each entry reaches a pan position on the audio thread.
            |p: &mut Preset| p.notes.pan_spread = vec![0.1; NUM_KEYS - 1],
            |p: &mut Preset| p.notes.pan_spread = vec![0.1; NUM_KEYS + 1],
            |p: &mut Preset| {
                p.notes.pan_spread = vec![0.1; NUM_KEYS];
                p.notes.pan_spread[9] = MAX_PAN_SPREAD + 0.01;
            },
            |p: &mut Preset| {
                p.notes.pan_spread = vec![0.1; NUM_KEYS];
                p.notes.pan_spread[9] = -0.01;
            },
            |p: &mut Preset| {
                p.notes.pan_spread = vec![0.1; NUM_KEYS];
                p.notes.pan_spread[9] = f32::NAN;
            },
            // The comb's floor is a fraction of its own crest, and it reaches
            // every excitation gain in the bank.
            |p: &mut Preset| p.notes.comb_floor[3] = MAX_COMB_FLOOR + 0.01,
            |p: &mut Preset| p.notes.comb_floor[3] = -0.01,
            |p: &mut Preset| p.notes.comb_floor[3] = f32::NAN,
            |p: &mut Preset| {
                p.notes.comb_floor.pop();
            },
            // The two ragged per-partial tables. All or nothing across the
            // compass, in range entry by entry, and never longer than the bank
            // they are describing.
            |p: &mut Preset| p.notes.partial_gains = vec![Vec::new(); NUM_KEYS - 1],
            |p: &mut Preset| p.notes.partial_gains = vec![vec![1.0]; NUM_KEYS + 1],
            |p: &mut Preset| {
                p.notes.partial_gains = vec![Vec::new(); NUM_KEYS];
                // C8 has two partials; a row of eight was measured somewhere
                // else.
                p.notes.partial_gains[NUM_KEYS - 1] = vec![1.0; 8];
            },
            |p: &mut Preset| {
                p.notes.partial_gains = vec![Vec::new(); NUM_KEYS];
                p.notes.partial_gains[9] = vec![1.0, MAX_PARTIAL_GAIN + 1.0];
            },
            |p: &mut Preset| {
                p.notes.partial_gains = vec![Vec::new(); NUM_KEYS];
                p.notes.partial_gains[9] = vec![1.0, MIN_PARTIAL_GAIN * 0.5];
            },
            |p: &mut Preset| {
                p.notes.partial_gains = vec![Vec::new(); NUM_KEYS];
                p.notes.partial_gains[9] = vec![f32::NAN];
            },
            |p: &mut Preset| p.notes.partial_sigma_scale = vec![Vec::new(); NUM_KEYS - 1],
            |p: &mut Preset| {
                p.notes.partial_sigma_scale = vec![Vec::new(); NUM_KEYS];
                p.notes.partial_sigma_scale[NUM_KEYS - 1] = vec![1.0; 8];
            },
            |p: &mut Preset| {
                p.notes.partial_sigma_scale = vec![Vec::new(); NUM_KEYS];
                p.notes.partial_sigma_scale[9] = vec![MAX_PARTIAL_SIGMA_SCALE + 1.0];
            },
            |p: &mut Preset| {
                p.notes.partial_sigma_scale = vec![Vec::new(); NUM_KEYS];
                // Zero is a pole on the unit circle: a partial that never stops.
                p.notes.partial_sigma_scale[9] = vec![0.0];
            },
            |p: &mut Preset| {
                p.notes.partial_sigma_scale = vec![Vec::new(); NUM_KEYS];
                p.notes.partial_sigma_scale[9] = vec![f32::NAN];
            },
            // The fifth mechanism event is checked exactly as the other four
            // are, plus the band limit only it has.
            |p: &mut Preset| p.notes.comb_floor[0] = f32::INFINITY,
            |p: &mut Preset| p.noise.strike.bandwidth_hz = MAX_STRIKE_BANDWIDTH_HZ + 1.0,
            |p: &mut Preset| p.noise.strike.bandwidth_hz = MIN_STRIKE_BANDWIDTH_HZ - 1.0,
            |p: &mut Preset| p.noise.strike.bandwidth_hz = f32::NAN,
            // A burst centred outside its own band.
            |p: &mut Preset| {
                p.noise.strike.centroid_hz = 4_000.0;
                p.noise.strike.bandwidth_hz = 2_000.0;
            },
            |p: &mut Preset| p.noise.strike.centroid_hz = 0.0,
            |p: &mut Preset| p.noise.strike.centroid_hz = f32::NAN,
            |p: &mut Preset| p.noise.strike.decay_s = MAX_STRIKE_DECAY_S + 0.1,
            |p: &mut Preset| p.noise.strike.decay_s = MIN_STRIKE_DECAY_S * 0.5,
            |p: &mut Preset| p.noise.strike.velocity_db = f32::NAN,
            |p: &mut Preset| p.noise.strike.level_db.clear(),
            |p: &mut Preset| p.noise.strike.level_db[0].db = 1.0,
            |p: &mut Preset| p.noise.strike.level_db[0].key = 120,
            |p: &mut Preset| {
                p.noise.strike.level_db = vec![
                    NoiseAnchor { key: 72, db: -20.0 },
                    NoiseAnchor { key: 60, db: -20.0 },
                ]
            },
            // The within-string splits. Same all-or-nothing shape rule as the
            // ragged tables, and every entry has to name a partial the key
            // really has: an offset on the diagonal of a mode that is not in the
            // bank is a table measured on a different instrument.
            |p: &mut Preset| p.notes.false_beat = vec![Vec::new(); NUM_KEYS - 1],
            |p: &mut Preset| p.notes.false_beat = false_beat_with(|e| e.k = 0),
            |p: &mut Preset| {
                // C8 has two partials.
                let mut rows = vec![Vec::new(); NUM_KEYS];
                rows[NUM_KEYS - 1] = vec![FalseBeat {
                    k: 9,
                    hz: 1.0,
                    db: -6.0,
                }];
                p.notes.false_beat = rows;
            },
            |p: &mut Preset| p.notes.false_beat = false_beat_with(|e| e.hz = MIN_FALSE_BEAT_HZ * 0.5),
            |p: &mut Preset| p.notes.false_beat = false_beat_with(|e| e.hz = MAX_FALSE_BEAT_HZ + 0.1),
            |p: &mut Preset| p.notes.false_beat = false_beat_with(|e| e.hz = f32::NAN),
            |p: &mut Preset| p.notes.false_beat = false_beat_with(|e| e.db = MAX_FALSE_BEAT_DB + 0.1),
            |p: &mut Preset| p.notes.false_beat = false_beat_with(|e| e.db = MIN_FALSE_BEAT_DB - 0.1),
            |p: &mut Preset| p.notes.false_beat = false_beat_with(|e| e.db = f32::NAN),
            // One wire's one partial has one split.
            |p: &mut Preset| {
                let mut rows = vec![Vec::new(); NUM_KEYS];
                rows[39] = vec![
                    FalseBeat {
                        k: 2,
                        hz: 1.0,
                        db: -6.0,
                    },
                    FalseBeat {
                        k: 2,
                        hz: 1.4,
                        db: -8.0,
                    },
                ];
                p.notes.false_beat = rows;
            },
            |p: &mut Preset| {
                let mut rows = vec![Vec::new(); NUM_KEYS];
                rows[39] = (1..=MAX_FALSE_BEATS_PER_KEY + 1)
                    .map(|k| FalseBeat {
                        k: k as u16,
                        hz: 1.0,
                        db: -6.0,
                    })
                    .collect();
                p.notes.false_beat = rows;
            },
            // The provenance list names real keys, in order, once each: a list
            // that names one twice is a list somebody built by appending, and
            // the second entry is the one that would be lost.
            |p: &mut Preset| p.notes.synthesized_texture = vec![LOWEST_KEY - 1],
            |p: &mut Preset| p.notes.synthesized_texture = vec![HIGHEST_KEY + 1],
            |p: &mut Preset| p.notes.synthesized_texture = vec![61, 60],
            |p: &mut Preset| p.notes.synthesized_texture = vec![60, 60],
            |p: &mut Preset| p.notes.synthesized_decay = vec![LOWEST_KEY - 1],
            |p: &mut Preset| p.notes.synthesized_decay = vec![HIGHEST_KEY + 1],
            |p: &mut Preset| p.notes.synthesized_decay = vec![61, 60],
            |p: &mut Preset| p.notes.synthesized_decay = vec![60, 60],
            // The velocity law for the strike vector's direction reaches the
            // mode gains at note-on, so its two ends and its tilt are bounded
            // exactly like everything else that does.
            |p: &mut Preset| {
                p.voicing.strike_direction = Some(direction_with(|d| {
                    d.vh_db_at_pp = -MAX_STRIKE_DIRECTION_DB - 0.1
                }))
            },
            |p: &mut Preset| {
                p.voicing.strike_direction = Some(direction_with(|d| {
                    d.vh_db_at_ff = MAX_STRIKE_DIRECTION_DB + 0.1
                }))
            },
            |p: &mut Preset| {
                p.voicing.strike_direction = Some(direction_with(|d| d.vh_db_at_ff = f32::NAN))
            },
            |p: &mut Preset| {
                p.voicing.strike_direction =
                    Some(direction_with(|d| d.share_tilt = MAX_SHARE_TILT + 0.01))
            },
            |p: &mut Preset| {
                p.voicing.strike_direction =
                    Some(direction_with(|d| d.share_tilt = -MAX_SHARE_TILT - 0.01))
            },
            |p: &mut Preset| {
                p.voicing.strike_direction = Some(direction_with(|d| d.share_tilt = f32::NAN))
            },
            // The three tables whose only bound used to be a sign, and which
            // `PianoString::new` rather than this function was left to discover
            // (`DECISIONS.md` 257). A fundamental past the partial cap, which
            // `StringParams::partial_count`'s `.max(1)` builds anyway; a unison
            // spread that is not a unison; and a fitted decay rate so far under
            // the arithmetic floor that the eigensolve's own rounding decides
            // its sign.
            |p: &mut Preset| p.notes.f0_hz[NUM_KEYS - 1] = 25_000.0,
            |p: &mut Preset| p.notes.detune_cents[0] = MAX_DETUNE_CENTS + 0.1,
            |p: &mut Preset| p.notes.sigma0[3] = 1.0e-44,
            |p: &mut Preset| p.notes.sigma0[3] = MIN_MODE_SIGMA * 0.5,
        ];
        for break_it in breakages {
            let mut p = Preset::default();
            break_it(&mut p);
            assert!(p.validate().is_err(), "a broken preset validated");
        }

        assert!(Preset::from_toml("name = 'nope'").is_err());
        assert!(Preset::from_toml("this is not toml").is_err());
    }

    /// Every preset `validate` accepts builds its whole compass without
    /// panicking, and the three that used to panic are named by field.
    ///
    /// `validate`'s own doc states the policy — untrusted input "has to be
    /// refused here, with a message naming the field, rather than reached at
    /// note-on" — and `string::stable_sigma`'s doc leans on it, asserting rather
    /// than clamping because "no preset a user could write can produce one".
    /// Three could (`DECISIONS.md` 257), and the assertion that they cannot is
    /// only worth what this test is worth: the refusals are checked *and* the
    /// legal rails are built, because a bound that refuses the corner by
    /// refusing everything near it would pass the first half alone.
    #[test]
    fn a_preset_that_would_panic_the_eigensolve_is_refused_by_field_name() {
        for (field, break_it) in [
            (
                "notes.f0_hz",
                Box::new(|p: &mut Preset| p.notes.f0_hz[NUM_KEYS - 1] = 25_000.0)
                    as Box<dyn Fn(&mut Preset)>,
            ),
            (
                "notes.detune_cents",
                Box::new(|p: &mut Preset| {
                    p.notes.detune_cents.iter_mut().for_each(|d| *d = 20_000.0)
                }),
            ),
            (
                "notes.sigma0",
                Box::new(|p: &mut Preset| p.notes.sigma0.iter_mut().for_each(|s| *s = 1.0e-44)),
            ),
        ] {
            let mut p = Preset::default();
            break_it(&mut p);
            let message = match p.validate() {
                Ok(()) => panic!("{field} out of range validated"),
                Err(e) => e.to_string(),
            };
            assert!(
                message.contains(field),
                "the refusal does not name {field}: {message}"
            );
        }

        // And the legal ends of the same three bounds are instruments: the
        // whole compass is built, so nothing here is refused by being made
        // unreachable.
        for break_it in [
            Box::new(|p: &mut Preset| {
                p.notes
                    .detune_cents
                    .iter_mut()
                    .for_each(|d| *d = MAX_DETUNE_CENTS)
            }) as Box<dyn Fn(&mut Preset)>,
            Box::new(|p: &mut Preset| {
                p.notes.sigma0.iter_mut().for_each(|s| *s = MIN_MODE_SIGMA);
                p.notes.sigma1.iter_mut().for_each(|s| *s = 0.0);
            }),
            // The top key's fundamental one hair under the cap, which is the
            // rail the `.max(1)` floor sits on.
            Box::new(|p: &mut Preset| {
                p.notes.f0_hz[NUM_KEYS - 1] = 0.9 * MAX_PARTIAL_RATIO * SAMPLE_RATE
            }),
        ] {
            let mut p = Preset::default();
            break_it(&mut p);
            p.validate().expect("a preset at the rails is still a preset");
            for key in LOWEST_KEY..=HIGHEST_KEY {
                let params = p.string_params(key);
                let string =
                    crate::string::PianoString::new(params, &p.voicing, PartialShaping::default());
                assert!(
                    string.partial_count() >= 1,
                    "key {key} came out of a validated preset with no partials"
                );
            }
        }
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
            "voicing.bridge",
            "duplex",
            "pan_spread",
            "comb_floor",
            "partial_gains",
            "partial_sigma_scale",
            "false_beat",
            "strike_direction",
            "[noise.strike]",
            // The mechanism's default is the measured table rather than
            // silence, but it is skipped on the same terms and for the same
            // reason: the file is the tuner's interface.
            "[noise]",
        ] {
            assert!(!text.contains(field), "a neutral preset wrote {field}");
        }
        let back = Preset::from_toml(&text).expect("a preset without them still loads");
        assert_eq!(back, Preset::default());
        assert_eq!(back.voicing.bridge, None);
        assert!(back.notes.duplex.is_empty());
        assert!(back.notes.pan_spread.is_empty());
        for key in LOWEST_KEY..=HIGHEST_KEY {
            assert!(back.duplex_modes(key).is_empty());
            assert_eq!(back.pan_spread(key), back.voicing.polarization_pan_spread);
        }
        assert!(back.notes.inharmonicity_b4.iter().all(|&b| b == 0.0));
        assert!(back.notes.contact_width.iter().all(|&w| w == 0.0));
        assert!(back.notes.comb_floor.iter().all(|&f| f == 0.0));
        assert!(back.notes.partial_gains.is_empty());
        assert!(back.notes.partial_sigma_scale.is_empty());
        assert!(back.notes.false_beat.is_empty());
        assert_eq!(back.voicing.strike_direction, None);
        // ... and every key reads the neutral shaping, whatever it asks for.
        for key in LOWEST_KEY..=HIGHEST_KEY {
            let shaping = back.partial_shaping(key);
            assert!(shaping.gains.is_empty() && shaping.sigma_scale.is_empty());
            assert!(shaping.false_beat.is_empty());
            for k in 1..=back.string_params(key).partial_count() {
                assert_eq!(shaping.gain_at(k), 1.0);
                assert_eq!(shaping.sigma_scale_at(k), 1.0);
                assert_eq!(shaping.false_beat_at(k), None);
            }
        }
        // The hammer's noise is the one event whose default is silence, and
        // silence is what a preset that does not describe it plays.
        assert_eq!(back.noise.strike, StrikeNoise::default());
        assert_eq!(back.noise.strike.level_db[0].db, SILENT_LEVEL_DB);
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
        preset.notes.pan_spread = (0..NUM_KEYS)
            .map(|i| 0.05 + 0.003 * i as f32)
            .collect();
        assert!(preset.validate().is_ok());

        let text = preset.to_toml();
        assert!(text.contains("polarization_pan_spread = 0.22"));
        assert!(text.contains("pan_spread = ["));
        // The per-key table wins over the global scalar wherever it exists.
        assert_eq!(preset.pan_spread(LOWEST_KEY), 0.05);
        assert_eq!(preset.pan_spread(HIGHEST_KEY), preset.notes.pan_spread[87]);
        assert_eq!(text.matches("[[voicing.unison_sigma_scale]]").count(), MAX_UNISON);
        assert!(text.contains("0.0125"));
        // Bit-exact, like every other number in a preset.
        assert_eq!(Preset::from_toml(&text).expect("round trip parses"), preset);
    }

    /// The two mechanisms the recording asks for and the model did not have —
    /// the within-string split and the velocity-dependent strike direction —
    /// write in full, read back bit for bit, and reach the key they describe.
    ///
    /// They are checked together because they are one finding: `docs/history/FUNDAMENTALS.md`
    /// §7.4 says the recording's mid and low partials each carry a companion
    /// 4–7 dB down and 0.7–1.5 Hz away (the split), and that *how much of each
    /// plane the hammer excites depends on how the hammer meets the string*,
    /// which is the one thing that changes with velocity (the direction).
    #[test]
    fn the_two_motion_mechanisms_round_trip_and_reach_the_key() {
        let mut preset = Preset::default();
        let c4 = key_index(60).unwrap();
        preset.notes.false_beat = vec![Vec::new(); NUM_KEYS];
        preset.notes.false_beat[c4] = vec![
            // C4's measured companions: −6.1 dB at 1.11 Hz on the fundamental,
            // −3.6 at 1.48 on the second (`renders/jitter/EIGENMODE.md`).
            FalseBeat {
                k: 1,
                hz: 1.11,
                db: -6.1,
            },
            FalseBeat {
                k: 2,
                hz: 1.48,
                db: -3.6,
            },
        ];
        preset.voicing.strike_direction = Some(StrikeDirection {
            vh_db_at_pp: -3.0,
            vh_db_at_ff: 4.5,
            share_tilt: 0.08,
        });
        assert!(
            preset.validate().is_ok(),
            "the two mechanisms did not validate: {:?}",
            preset.validate().err()
        );

        let text = preset.to_toml();
        assert!(text.contains("[[notes.false_beat]]") || text.contains("false_beat"));
        assert!(text.contains("[voicing.strike_direction]"));
        assert_eq!(Preset::from_toml(&text).expect("round trip parses"), preset);

        // The split reaches the key it names and no other, through the same
        // handle the string reads it by.
        let shaping = preset.partial_shaping(60);
        assert_eq!(shaping.false_beat.len(), 2);
        assert_eq!(shaping.false_beat_at(1).map(|e| e.hz), Some(1.11));
        assert_eq!(shaping.false_beat_at(2).map(|e| e.db), Some(-3.6));
        assert_eq!(shaping.false_beat_at(3), None);
        assert!(preset.partial_shaping(59).false_beat.is_empty());
    }

    /// The measured per-partial tables and the hammer's own noise write in
    /// full, read back bit for bit, and reach the string and the event they
    /// describe.
    #[test]
    fn the_per_partial_tables_and_the_hammers_noise_round_trip() {
        let mut preset = Preset::default();
        // A2's comb null is k = 17 and its measured roughness is the fault
        // `renders/timbre-ladder/ANALYSIS.md` §4a and `docs/history/TUNING_REPORT.md` §3
        // report; this is the shape of the table that answers them, on one key.
        let a2 = key_index(45).unwrap();
        let c8 = NUM_KEYS - 1;
        preset.notes.comb_floor[a2] = 0.08;
        preset.notes.comb_floor[c8] = 0.5;
        preset.notes.partial_gains = vec![Vec::new(); NUM_KEYS];
        preset.notes.partial_gains[a2] = vec![1.0, 0.75, 1.4, 0.5, 2.0];
        // A row exactly as long as the key's bank is legal; one entry longer is
        // not, which `malformed_presets_are_rejected` pins.
        let c8_partials = preset.string_params(HIGHEST_KEY).partial_count();
        preset.notes.partial_gains[c8] = vec![1.1; c8_partials];
        preset.notes.partial_sigma_scale = vec![Vec::new(); NUM_KEYS];
        preset.notes.partial_sigma_scale[a2] = vec![0.6, 1.0, 1.9];
        preset.noise.strike = StrikeNoise {
            centroid_hz: 1_800.0,
            decay_s: 0.045,
            bandwidth_hz: 7_000.0,
            velocity_db: 18.0,
            level_db: vec![
                NoiseAnchor { key: 21, db: -24.0 },
                NoiseAnchor { key: 60, db: -20.0 },
                NoiseAnchor { key: 96, db: -14.0 },
            ],
        };
        assert!(
            preset.validate().is_ok(),
            "a measured shaping did not validate: {:?}",
            preset.validate().err()
        );

        let text = preset.to_toml();
        assert!(text.contains("comb_floor = ["));
        assert!(text.contains("partial_gains = ["));
        assert!(text.contains("partial_sigma_scale = ["));
        assert!(text.contains("[noise.strike]"));
        assert!(text.contains("bandwidth_hz = 7000"));
        assert_eq!(text.matches("[[noise.strike.level_db]]").count(), 3);
        assert_eq!(Preset::from_toml(&text).expect("round trip parses"), preset);

        // The tables reach the key they were written for, and nothing else.
        let shaping = preset.partial_shaping(45);
        assert_eq!(shaping.gain_at(2), 0.75);
        assert_eq!(shaping.sigma_scale_at(3), 1.9);
        // Past the end of the row, and on a key with no row at all.
        assert_eq!(shaping.gain_at(6), 1.0);
        assert_eq!(shaping.sigma_scale_at(4), 1.0);
        let untouched = preset.partial_shaping(60);
        assert!(untouched.gains.is_empty() && untouched.sigma_scale.is_empty());
        assert_eq!(preset.string_params(45).comb_floor, 0.08);
        assert_eq!(preset.string_params(60).comb_floor, 0.0);
    }

    /// A row longer than the bank it describes is refused with the key in the
    /// message, because a table measured on a different instrument is exactly
    /// what that looks like and the entries the engine would drop are the ones
    /// that say so.
    #[test]
    fn an_over_long_per_partial_row_names_the_key_it_refuses() {
        for field in ["partial_gains", "partial_sigma_scale"] {
            let mut preset = Preset::default();
            let table = vec![Vec::new(); NUM_KEYS];
            if field == "partial_gains" {
                preset.notes.partial_gains = table;
                preset.notes.partial_gains[NUM_KEYS - 1] = vec![1.0; 9];
            } else {
                preset.notes.partial_sigma_scale = table;
                preset.notes.partial_sigma_scale[NUM_KEYS - 1] = vec![1.0; 9];
            }
            let message = preset.validate().expect_err("an over-long row is refused");
            let message = message.to_string();
            assert!(message.contains(field), "{message}");
            assert!(
                message.contains(&format!("key {HIGHEST_KEY}")),
                "the message does not name the key: {message}"
            );
        }
    }

    /// A voiced bridge is written in full, reads back bit for bit, and — the
    /// point of the section — describes the shape `PHYSICS.md` §4 asks for:
    /// sharp separated peaks below ~500 Hz on a mean mobility that fluctuates
    /// over the midrange and falls in the treble.
    #[test]
    fn a_voiced_bridge_round_trips_and_is_the_shape_the_physics_asks_for() {
        let mut preset = Preset::default();
        preset.voicing.bridge = Some(BridgeVoicing {
            backbone: [
                (30.0, -14.0),
                (100.0, -1.5),
                (250.0, 2.0),
                (600.0, -3.0),
                (1_100.0, 0.0),
                (2_000.0, -5.0),
                (4_000.0, -2.0),
                (10_000.0, -14.0),
            ]
            .into_iter()
            .map(|(hz, gain_db)| BridgeAnchor { hz, gain_db })
            .collect(),
            peaks: [
                (58.0, 22.0, 9.0),
                (91.0, 26.0, 7.0),
                (133.0, 24.0, -5.0),
                (188.0, 30.0, 8.0),
                (247.0, 28.0, 6.0),
                (331.0, 32.0, -6.0),
                (426.0, 30.0, 5.0),
                (1_450.0, 6.0, -4.0),
                (3_800.0, 4.0, 3.0),
            ]
            .into_iter()
            .map(|(hz, q, gain_db)| BridgePeak { hz, q, gain_db })
            .collect(),
            // Half of each partial's decay follows the board's own modes,
            // which also puts the field through the round trip below.
            radiated_share: 0.5,
        });
        assert!(
            preset.validate().is_ok(),
            "a realistic bridge did not validate: {:?}",
            preset.validate().err()
        );

        let text = preset.to_toml();
        assert_eq!(text.matches("[[voicing.bridge.backbone]]").count(), 8);
        assert_eq!(text.matches("[[voicing.bridge.peaks]]").count(), 9);
        assert_eq!(Preset::from_toml(&text).expect("round trip parses"), preset);

        // The realised filter, not the anchors: a mean mobility that stays
        // inside a few dB of unity through the midrange, discrete resonances
        // standing well clear of it below 500 Hz, and a treble that falls away.
        let filter = BridgeFilter::new(preset.voicing.bridge.as_ref().unwrap());
        let db = |hz| amp_to_db(filter.magnitude(hz));
        assert!(db(188.0) - db(160.0) > 5.0, "the 188 Hz mode is not a peak");
        assert!(db(247.0) - db(290.0) > 4.0, "the 247 Hz mode is not a peak");
        assert!(db(331.0) < db(290.0) - 3.0, "the 331 Hz anti-resonance is not a dip");
        assert!(db(10_000.0) < db(600.0) - 8.0, "the treble does not fall");
        // ... and it is a bridge, not an amplifier: the loop bound is what the
        // whole design rests on, so check there is real margin left at the
        // default coupling.
        let max_b = filter.max_magnitude();
        assert!(amp_to_db(max_b) < 14.0, "peaks at {:.1} dB", amp_to_db(max_b));
        assert!(preset.voicing.resonance_coupling * max_b < MAX_BRIDGE_LOOP_GAIN / 2.0);
    }

    /// A duplex layout of the shape the measurements describe: segments only
    /// from D4 up (where Öberg & Askenfelt's survey starts), two per key, rising
    /// smoothly across the compass and scattered by tens of cents — deliberately
    /// *not* at `k f0`, which is the paper's central negative finding.
    pub(crate) fn measured_duplex() -> Vec<Vec<DuplexMode>> {
        let mut state = 0x2545_f491u32;
        (0..NUM_KEYS)
            .map(|i| {
                let key = index_to_note(i);
                if key < 62 {
                    return Vec::new();
                }
                let position = key_position(key);
                [(2_200.0f32, -26.0f32), (3_600.0, -32.0)]
                    .into_iter()
                    .map(|(base, gain_db)| {
                        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        // ±60 cents, which is the spread the paper reports
                        // within a single trichord at its widest.
                        let cents = (state >> 16) as f32 / 65_535.0 * 120.0 - 60.0;
                        DuplexMode {
                            hz: base * 2.0f32.powf(1.5 * position + cents / 1200.0),
                            gain_db,
                            t60_s: 1.2,
                        }
                    })
                    .collect()
            })
            .collect()
    }

    /// A voiced duplex is written in full, reads back bit for bit, and leaves
    /// the loop bound a wide margin — which is what a *measured* layout does,
    /// because sharp resonances scattered by tens of cents never crowd.
    #[test]
    fn a_voiced_duplex_round_trips_and_stays_far_inside_the_loop_bound() {
        let mut preset = Preset::default();
        preset.notes.duplex = measured_duplex();
        assert!(
            preset.validate().is_ok(),
            "a measured duplex layout did not validate: {:?}",
            preset.validate().err()
        );
        // Present from D4 up and nowhere below it, and never on a partial.
        assert!(preset.duplex_modes(60).is_empty());
        assert_eq!(preset.duplex_modes(62).len(), 2);
        assert_eq!(preset.duplex_modes(108).len(), 2);
        for key in 62..=108u8 {
            let f0 = preset.f0(key);
            for m in preset.duplex_modes(key) {
                let k = (m.hz / f0).round().max(1.0);
                let cents = 1200.0 * (m.hz / (k * f0)).log2();
                assert!(cents.abs() > 1.0, "key {key} put a segment on partial {k}");
            }
        }

        let text = preset.to_toml();
        assert!(text.contains("duplex = "), "the segments were not written");
        assert_eq!(Preset::from_toml(&text).expect("round trip parses"), preset);

        // The margin, stated rather than assumed: a measured layout is two
        // orders of magnitude under the bound, so the check is a guard against
        // the pathological preset and not a constraint on a real one.
        let worst = preset
            .notes
            .duplex
            .iter()
            .flatten()
            .map(|probe| {
                let d: f32 = preset
                    .notes
                    .duplex
                    .iter()
                    .map(|row| crate::duplex::magnitude(row, probe.hz))
                    .sum();
                d
            })
            .fold(0.0f32, f32::max);
        let loop_gain = preset.voicing.resonance_coupling * 2.0 * worst;
        assert!(
            loop_gain < MAX_DUPLEX_LOOP_GAIN / 50.0,
            "a measured duplex sits at a loop gain of {loop_gain}"
        );
    }

    /// ... and the pathological one is refused: 88 undamped banks tuned to a
    /// single frequency is the loop the bound exists for.
    #[test]
    fn a_duplex_tuned_to_one_frequency_across_the_compass_is_refused() {
        let mut preset = Preset::default();
        preset.notes.duplex = vec![
            vec![DuplexMode {
                hz: 4_000.0,
                gain_db: MAX_DUPLEX_GAIN_DB,
                t60_s: MAX_DUPLEX_T60_S,
            }];
            NUM_KEYS
        ];
        assert!(preset.validate().is_err(), "88 co-tuned segments validated");
        // The same segments scattered by a quarter of a semitone per key — far
        // less than the paper's own spread — are perfectly legal.
        for (i, row) in preset.notes.duplex.iter_mut().enumerate() {
            row[0].hz = 4_000.0 * 2.0f32.powf(i as f32 * 25.0 / 1200.0);
        }
        assert!(
            preset.validate().is_ok(),
            "a scattered duplex was refused: {:?}",
            preset.validate().err()
        );
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
