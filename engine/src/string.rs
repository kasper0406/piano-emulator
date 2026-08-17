//! Piano string: partial layout, frequency-dependent damping, and the coupled
//! eigenmodes of a unison group of 1-3 strings in two polarizations.
//!
//! # What a partial is
//!
//! A key's sound is a group of `N` strings, each with two transverse
//! polarizations, all terminated on one point of one bridge. That is `2N`
//! degrees of freedom per partial, and they are **not** independent: the bridge
//! moves, so every string feels every other one through it. Weinreich (JASA 62,
//! 1474, 1977) measured the consequence and Woodhouse (JASA 150, 4375, 2021)
//! restates it — the group's normal modes have their frequencies pulled
//! together and their decay rates pushed apart, the mode that radiates most
//! dies first, and the one that radiates least is the aftersound.
//!
//! Per partial `k`, with `omega_j` the detuned frequency of string `j`:
//!
//! ```text
//!     a' = A_k a,   A_k = i Omega_k - sigma_int I - C_k
//!
//!     Omega_k = diag(omega_j)        (polarization does not change omega)
//!     C_k     = [ c_v J_N,   0     ]  J_N = all-ones N x N
//!               [   0,   c_h J_N   ]  c_p = gamma_v (g_p + i beta_p)
//! ```
//!
//! `C_k` is block diagonal — the off-diagonal of the 2x2 bridge admittance is
//! second order, `docs/history/FUNDAMENTALS.md` §2.5 — and each block is `c_p` times an
//! all-ones matrix, i.e. **rank one**. So the `2N x 2N` eigenproblem is two
//! rank-one updates of a diagonal, its characteristic equation factorises into
//! two degree-`N` complex polynomials ([`block_solve`]), and `D - cJ` is complex
//! **symmetric**, so the row of `V^-1` the strike projection needs is
//! `v_m / (v_m . v_m)` — one division per mode. No LAPACK, no matrix inversion,
//! and nothing in it depends on velocity, so the whole solve is construction
//! time and the note-on path is untouched.
//!
//! The `2N` eigenmodes go straight into two [`ModalBank`]s — the vertical
//! block's into one and the horizontal block's into the other, which is what
//! keeps the polarization stereo spread — at **the same mode count the
//! free-running construction used**. What they need that free modes do not is a
//! *complex* input gain: the strike projects onto a non-orthogonal eigenbasis,
//! so each mode starts at its own phase.
//!
//! # The two coefficients, and the two normalisations
//!
//! The coupling is not a free parameter. `gamma = Z omega G / pi` is
//! simultaneously the rate a partial loses energy into the board and the
//! strength with which the board couples this string to its neighbours
//! (Capleton, JASA 115(2), 2004, Eq. 2) — the same number. So:
//!
//! * [`radiated_share`] `= 1 - horizontal_decay_ratio`. The slowest mode of the
//!   coupled group radiates nothing and decays at `(1 - share) sigma_k`, the
//!   loudest decays at `sigma_k`, so `1 - share` **is** the fitted
//!   aftersound/prompt decay ratio, and `voicing.horizontal_decay_ratio` is the
//!   field that was fitted to recordings. This resolves `docs/history/FUNDAMENTALS.md`
//!   §2.6's contradiction with `voicing.bridge.radiated_share = 0.5` in favour
//!   of the measurement, and on Woodhouse's side of it.
//! * [`decay_scale`] solves, per partial, for the one factor on the loss budget
//!   that puts the composite of the `2N` modes at -60 dB on the anchor
//!   `6.91 / sigma_k` that `notes.sigma0` / `sigma1` were fitted to. This is the
//!   generalisation of the closed-form `vertical_decay_factor` the free-running
//!   construction used, which was exact only for its own two-exponential sum.
//!
//! # The two things the recording asks for that the eigenmodes alone cannot do
//!
//! The construction above is the physically right one and it does not reproduce
//! the recording (`docs/history/FUNDAMENTALS.md` §7.1). Two mechanisms sit on top of it, both
//! from §7.5's build order, and neither is a new solver:
//!
//! * **The within-string split** ([`FalseBeat`], `notes.false_beat`). Each
//!   recorded mid and low partial carries a companion 4-7 dB down and 0.7-1.5 Hz
//!   away at a spacing that does *not* scale with the partial number — not the
//!   unison (7-20x too narrow, and proportional to `k`), not the bridge's
//!   polarization split (a hundred times narrower), not `horizontal_offset_hz`
//!   (22 dB out, and note-independent). It is Capleton's false beat: the two
//!   planes of **one wire** at genuinely different frequencies. It enters as one
//!   more term on the diagonal of `Omega_k`, before the block is solved
//!   ([`partial_modes`]), so its companion is one of the group's own
//!   eigenvalues.
//! * **The velocity-dependent strike direction**
//!   ([`StrikeDirection`], `voicing.strike_direction`). `u` scales uniformly
//!   with velocity, so `V^-1 u` does too and every ratio in the mixture is a
//!   constant — which is why the shipped engine's beat structure holds to
//!   0.054 dB across an 80-point velocity span where the recording's moves
//!   1.90. The one thing that can move is where the blow *points*, and it moves
//!   two ratios inside `u` without changing its length
//!   ([`PianoString::set_strike`]).
//!
//! Both are absent from both shipped presets and neutral when absent, bit for
//! bit. See `DECISIONS.md` 233-238.
//!
//! # What this construction deleted
//!
//! `voicing.unison_coupling` (a one-block-late excitation cross-feed, measured
//! to move the result by 0.07 cents), `voicing.horizontal_offset_hz` (a *fixed
//! number of hertz*, so 0.270 / 0.350 / 0.520 Hz were beat rates of every
//! partial of every key — the instrument-wide pulse `renders/jitter/JITTER.md`
//! convicted) and `voicing.unison_sigma_scale` (the per-string decay split,
//! which is now an output). All three are still accepted by the schema and warn
//! on load; none is read. See `DECISIONS.md` 223-228.
//!
//! Every number here comes from the [`Preset`](crate::preset::Preset): the
//! per-note ones through [`StringParams`], the rest through
//! [`Voicing`].

use crate::modal::ModalBank;
use crate::preset::{FalseBeat, StrikeDirection, Voicing, NOMINAL_STRIKE_VELOCITY};
use crate::resonance::BridgeFilter;
use crate::types::{
    db_to_amp, velocity_from_midi, BLOCK, MAX_PARTIALS, MAX_PARTIAL_RATIO, MAX_UNISON, SAMPLE_RATE,
};

/// Note the per-note output gains are normalised against, so the preset's
/// `excitation_scale` stays a plain level control: mode k's force on the bridge
/// is proportional to f0, and this is the f0 that scale was calibrated at (C4).
const REFERENCE_F0: f32 = 261.6256;

/// Widest hammer contact a preset may declare, as a fraction of the speaking
/// length. A real hammer touches 1–2 % of it (`PHYSICS.md` §7); 5 % is already
/// past any measured felt and is where the raised-cosine taper below has nulled
/// the twentieth partial outright, so nothing above it describes a hammer.
pub const MAX_CONTACT_WIDTH: f32 = 0.05;

/// Bounds on a per-string decay-rate multiplier. The rows average to 1, so a
/// factor of two either way is already a group whose fastest string dies four
/// times sooner than its slowest — wider than any voicing, and wide enough that
/// the value stops being a multiplier and becomes a different note.
pub const MIN_SIGMA_SCALE: f32 = 0.5;
pub const MAX_SIGMA_SCALE: f32 = 2.0;

/// Bounds on a per-partial excitation gain (`notes.partial_gains`): −26 dB to
/// +26 dB.
///
/// The quantity is *measured*, not a knob — and since `DECISIONS.md` 231 it is
/// the **full** ratio of the recording's partial to the engine's own prediction
/// of it, not the roughness left over after a smooth envelope has been divided
/// out. That is what widened the range. The roughness alone justified ±20 dB:
/// `docs/history/TUNING_REPORT.md` §3 puts the recordings' scatter around the fitted comb at
/// 5–10 dB RMS with worst partials 12–29 dB out, and
/// `renders/timbre-ladder/ANALYSIS.md` §4a puts the deepest partial anywhere at
/// 9.3–17.7 dB below a smooth envelope. The envelope error rides on top of that
/// and is itself worth 7.5 dB of tilt over C4's first four partials
/// (−6.53 / −2.55 / −0.91 / −0.05 dB), so the two together can leave a partial
/// well past a factor of ten where either alone would not.
///
/// A factor of twenty either way is the widest a table can go and still be one
/// hammer striking one string: past it the fit has stopped correcting the
/// excitation model and started replacing it, which is a different repair and
/// belongs in the model it is replacing.
pub const MIN_PARTIAL_GAIN: f32 = 0.05;
pub const MAX_PARTIAL_GAIN: f32 = 20.0;

/// Bounds on a per-partial decay-rate multiplier (`notes.partial_sigma_scale`).
///
/// Narrower than [`RADIATED_FACTOR_RANGE`] is wide for the same reason that one
/// is clamped: `notes.sigma0`/`sigma1` are fitted to recorded decays, so this is
/// a correction to a measurement and not a second decay law. A factor of four
/// either way is already 12 dB of T60 either side of the fit.
pub const MIN_PARTIAL_SIGMA_SCALE: f32 = 0.25;
pub const MAX_PARTIAL_SIGMA_SCALE: f32 = 4.0;

/// Deepest a preset may fill in the strike comb's nulls
/// ([`StringParams::comb_floor`]).
///
/// A floor of 0.5 puts the null partial 6 dB under a partial at the comb's
/// crest, which is not a comb at all; the measured instrument's deepest partial
/// anywhere is 9.3–17.7 dB down, i.e. a floor of 0.13–0.34.
pub const MAX_COMB_FLOOR: f32 = 0.5;

/// Excitation taper of a hammer that touches a patch of the string rather than
/// a point, for partial `k` and a contact width of `width` speaking lengths.
///
/// The point-force comb `sin(k pi x)` is convolved with the contact profile, so
/// each partial is scaled by the profile's transform at that partial's
/// wavenumber; for a raised-cosine patch that is `cos^2(k pi w / 2)` down to its
/// first zero (`PHYSICS.md` §7, after Hall & Askenfelt). Past that zero the
/// analytic form turns back up, which a widening contact patch does not do, so
/// it is clamped: once a partial has a whole contact patch inside one of its
/// half-periods the hammer cannot drive it at all.
///
/// `contact_taper(k, 0.0)` is exactly 1.0, which is what makes a preset without
/// the field the point-force instrument bit for bit.
pub fn contact_taper(k: usize, width: f32) -> f32 {
    let phase = 0.5 * k as f32 * std::f32::consts::PI * width;
    if phase >= std::f32::consts::FRAC_PI_2 {
        0.0
    } else {
        let c = phase.cos();
        c * c
    }
}

/// Per-note string parameters. Every field is a starting point that automated
/// tuning is expected to overwrite later.
#[derive(Clone, Copy, Debug)]
pub struct StringParams {
    /// Fundamental frequency in Hz.
    pub f0: f32,
    /// Stiffness inharmonicity coefficient B in
    /// `f_k = k f0 sqrt(1 + B k^2 + B4 k^4)`.
    pub inharmonicity_b: f32,
    /// Fourth-order coefficient B4 of the same law, **signed**.
    ///
    /// One `B` is not enough at the bottom of the compass: fitted to a wound
    /// bass string's partials 1–8 and again to its partials 14–26, `B` comes
    /// back 25–37 % *smaller* on the upper band (A0 0.75, C1 0.66, D#1 0.63)
    /// and 24–45 % *larger* on the short wound tenor strings (F#1 1.24, A1
    /// 1.40, C2 1.45) — up to 78 cents of misplaced partial against a single
    /// coefficient (`docs/history/TUNING_REPORT.md` §1). The sign flips across that break,
    /// so the correction has to be signed. Zero everywhere reduces the law to
    /// the two-parameter one exactly.
    pub inharmonicity_b4: f32,
    /// Hammer strike point as a fraction of string length.
    pub strike_position: f32,
    /// Width of the hammer's contact with the string, as a fraction of the
    /// speaking length. Zero is the point force the comb `sin(k pi x)` assumes;
    /// see [`contact_taper`].
    pub contact_width: f32,
    /// Soft floor under the strike comb's nulls, as a fraction of the comb's
    /// crest: the excitation magnitude of partial `k` becomes
    /// `sqrt(sin^2(k pi x) + floor^2)`.
    ///
    /// `sin(k pi x)` has exact zeros and a real hammer on a real string does
    /// not: the contact patch has width, the string has stiffness, and the
    /// termination is not a node. Measured, the engine's worst partial is
    /// exactly where the comb crosses zero — 42 dB down at A2's k = 17 and at
    /// C6's k = 8 — while the recording's deepest partial anywhere is 9.3 to
    /// 17.7 dB below a smooth envelope and never at those indices
    /// (`renders/timbre-ladder/ANALYSIS.md` §4a). The contact taper cannot fill
    /// a null: it is a low-pass in `k` and multiplies the zero by something
    /// smaller.
    ///
    /// Zero — the default — is the bare comb, sign and all, bit for bit.
    pub comb_floor: f32,
    /// Frequency-independent part of the decay rate, 1/s.
    pub sigma0: f32,
    /// Coefficient of `(f_k/1000)^2` in the decay rate, 1/s.
    pub sigma1: f32,
    /// Number of unison strings for this note (1, 2 or 3).
    pub unison: usize,
    /// Full width of the unison detuning spread, in cents.
    ///
    /// Two strings of a unison differ in tension, and `f_k ∝ sqrt(T)` for every
    /// partial at once, so the mistuning is a ratio and not a number of hertz:
    /// the spec's "±0.1–0.5 Hz, slightly wider in treble" is the same few cents
    /// read at the two ends of the compass, and taken literally in the bass it
    /// would leave A2's unison 6.6 cents wide — audibly sour, and wider than
    /// the ±3 cents the spec's own tuning test allows.
    pub detune_cents: f32,
    /// Transverse wave impedance, kg/s. Sets how hard the hammer is loaded and
    /// how much string velocity a given force impulse produces.
    pub impedance: f32,
    /// Extra decay rate applied by a fully engaged damper, 1/s.
    pub damper_sigma: f32,
    /// Fraction of this note's bridge force that becomes signal — the
    /// soundboard's coupling to this part of the compass.
    pub bridge_gain: f32,
}

impl StringParams {
    /// The number under the root of the partial law: `1 + B k^2 + B4 k^4`.
    ///
    /// A preset is refused unless this stays positive and the series it
    /// produces stays ordered over the partials the note actually uses — a
    /// negative radicand is a NaN in a mode frequency, and a series that turns
    /// over is not a string.
    pub fn partial_radicand(&self, k: usize) -> f32 {
        let k = k as f32;
        let k2 = k * k;
        // `B k^2` keeps the association the two-parameter law used, so a preset
        // with `B4 = 0` lays its partials out bit for bit as before: the extra
        // term is then exactly `+ 0.0`.
        1.0 + self.inharmonicity_b * k * k + self.inharmonicity_b4 * k2 * k2
    }

    /// Frequency of partial `k` (1-based) including stiffness inharmonicity.
    pub fn partial_freq(&self, k: usize) -> f32 {
        k as f32 * self.f0 * self.partial_radicand(k).sqrt()
    }

    /// Decay rate of partial `k` for the note as a whole, 1/s: `6.91 / sigma`
    /// is the time that partial takes to fall 60 dB counting both
    /// polarizations. The vertical bank decays faster than this, the horizontal
    /// one slower, and the individual strings of a unison faster or slower
    /// again by [`Voicing::sigma_scale`](crate::preset::Voicing::sigma_scale) —
    /// see
    /// [`Voicing::vertical_decay_factor`](crate::preset::Voicing::vertical_decay_factor).
    pub fn partial_sigma(&self, k: usize) -> f32 {
        self.sigma0 + self.sigma1 * (self.partial_freq(k) / 1000.0).powi(2)
    }

    /// Number of partials that fit below `MAX_PARTIAL_RATIO * SAMPLE_RATE`,
    /// capped at `MAX_PARTIALS`.
    pub fn partial_count(&self) -> usize {
        let limit = MAX_PARTIAL_RATIO * SAMPLE_RATE;
        (1..=MAX_PARTIALS)
            .take_while(|&k| self.partial_freq(k) < limit)
            .count()
            .max(1)
    }
}

/// The per-partial tables of one key, as borrowed slices of the preset.
///
/// All three are *measurements* the smooth per-note laws cannot carry, and all
/// three are deliberately velocity-independent: the excitation spectrum of a
/// note is not shared with the note beside it (`docs/history/TUNING_REPORT.md` §3 refutes a
/// global admittance curve by measurement), neither is the way its individual
/// partials depart from the fitted decay law, and a wire's own geometry does not
/// know how hard it is being struck. Velocity enters this model in exactly one
/// place, and it is the *direction* of the strike vector
/// ([`StrikeDirection`](crate::preset::StrikeDirection)) rather than any table
/// here.
///
/// The two ragged rows may be shorter than the key's partial count — the
/// estimator measures as far up the series as it can track — and every partial
/// past the end is exactly 1.0. [`PartialShaping::default`] is that everywhere
/// with no splits, which is the instrument as it was built before these tables
/// existed.
#[derive(Clone, Copy, Debug, Default)]
pub struct PartialShaping<'a> {
    /// Linear multiplier on partial `k`'s excitation gain, 1-based: the full
    /// measured ratio of the recording's partial to the engine's own prediction
    /// of it (`NoteTables::partial_gains`, `DECISIONS.md` 231).
    pub gains: &'a [f32],
    /// Multiplier on partial `k`'s decay rate, 1-based.
    pub sigma_scale: &'a [f32],
    /// This key's within-string splits, in no particular order and at most one
    /// per partial ([`FalseBeat`]).
    pub false_beat: &'a [FalseBeat],
}

impl PartialShaping<'_> {
    /// Excitation multiplier of partial `k` (1-based); 1.0 past the table.
    #[inline]
    pub fn gain_at(&self, k: usize) -> f32 {
        self.gains.get(k - 1).copied().unwrap_or(1.0)
    }

    /// Decay-rate multiplier of partial `k` (1-based); 1.0 past the table.
    #[inline]
    pub fn sigma_scale_at(&self, k: usize) -> f32 {
        self.sigma_scale.get(k - 1).copied().unwrap_or(1.0)
    }

    /// The within-string split of partial `k` (1-based), if this key has one.
    ///
    /// A linear scan of at most [`MAX_FALSE_BEATS_PER_KEY`] entries, at
    /// construction time only: the table is a *list* rather than a series
    /// because a false beat is a defect of one wire at one partial and a key
    /// with eighty partials has at most a handful.
    ///
    /// [`MAX_FALSE_BEATS_PER_KEY`]: crate::preset::MAX_FALSE_BEATS_PER_KEY
    #[inline]
    pub fn false_beat_at(&self, k: usize) -> Option<FalseBeat> {
        self.false_beat.iter().copied().find(|e| e.k as usize == k)
    }
}

/// Excitation magnitude of partial `k` before the contact taper: the strike
/// comb, with a soft floor under its nulls.
///
/// `sqrt(sin^2 + floor^2)` never reaches zero and is within `floor^2/2|sin|` of
/// the bare comb wherever the comb is strong, so a floor lifts the nulls and
/// leaves everything else where it was. The comb's **sign** is kept: it is the
/// phase partial `k` starts at, and preserving it is what makes a zero floor the
/// old instrument to the last bit rather than to a rounding.
fn comb_magnitude(k: usize, strike_position: f32, floor: f32) -> f32 {
    let comb = (k as f32 * std::f32::consts::PI * strike_position).sin();
    if floor > 0.0 {
        // `signum` of a zero comb is ±1, so a partial with an exact node gets
        // the floor rather than nothing, which is the whole point.
        comb.signum() * (comb * comb + floor * floor).sqrt()
    } else {
        comb
    }
}


// ---------------------------------------------------- the bridge, as physics

/// `Im Y / Re Y` at the bridge: how reactive the termination is compared with
/// how lossy it is.
///
/// Weinreich's measured admittances make the two parts comparable and Capleton
/// §III.B works his example at a ratio of order one. It decides whether the
/// coupling **attracts** the group's frequencies (Woodhouse's anti-veering,
/// resistive-dominated, "with anti-veering there will be no beats") or repels
/// them. The literature does not pin it, so the prototype re-solved and
/// re-rendered the whole construction at 0, 0.25, 1 and 3 and reported every
/// statistic at each: the verdict is a claim about all four
/// (`renders/jitter/EIGENMODE.md`, "Sensitivity to Im Y / Re Y").
///
/// Deliberately a constant and not a preset field. `docs/history/FUNDAMENTALS.md` §7.6
/// suggested one; three fields are going inert in the same change, and nothing
/// in the tuner can yet fit this one — there is no measurement that separates it
/// from [`REACTIVE_ANISOTROPY`] until the strike vector gets a velocity
/// dependence (§7.5 step 3). See `DECISIONS.md` 226.
const REACTIVE_RATIO: f64 = 1.0;

/// How much less reactive the bridge is horizontally than vertically.
///
/// This one number replaces `voicing.horizontal_offset_hz` entirely. Capleton,
/// summarising Weinreich: "the angular variation of the reactive part of the
/// bridge admittance is at least a factor of 10 smaller than the variation in
/// the resistive part", and his worked example uses a reactive ratio of
/// 1 : 0.925. The polarization split it produces is `N gamma_v beta eps / 2 pi`
/// — a few hundredths of a hertz where the shipped field asserted 0.35, and a
/// function of the partial's own frequency and decay rate where the shipped
/// field was the same number on every partial of every key.
const REACTIVE_ANISOTROPY: f64 = 0.075;

/// The bridge's resistive anisotropy `Re Y_h / Re Y_v`, from
/// `voicing.horizontal_decay_ratio`.
///
/// `docs/history/FUNDAMENTALS.md` §5.4 says to re-read the field as this ratio directly —
/// "same number, now a property of the bridge instead of a decay law" — and that
/// is a decibel too far, because the horizontal plane **still radiates**. Take
/// one string, whose two polarizations are the whole of the eigenproblem:
///
/// ```text
///     sigma_v = sigma_int + gamma_v          = sigma
///     sigma_h = sigma_int + rho gamma_v      = rho (2 - rho) sigma
/// ```
///
/// with `sigma_int = (1 - share) sigma`, `gamma_v = share sigma` and
/// `share = 1 - rho`. So the *decay* ratio the field was fitted to is
/// `rho (2 - rho)`, not `rho`, and reading the field as `rho` makes every note's
/// aftersound decay `(2 - rho)` times too fast — 0.496 against the fitted 0.29 on
/// `presets/default.toml`. Inverting it instead,
///
/// ```text
///     rho = 1 - sqrt(1 - horizontal_decay_ratio)
/// ```
///
/// puts the single-string aftersound back exactly on the fitted rate (0.157 for
/// the default voicing, 0.090 for the measured one), and it is what takes the
/// rendered note's peak and attack level from +0.4 dB / +1.1 dB of the
/// free-running construction to **+0.4 dB / +0.5 dB** over the compass.
///
/// Clamped off both ends: a ratio of 0 is a horizontal plane that never stops
/// and a ratio of 1 is a bridge with no anisotropy at all, and neither is a
/// piano. `Preset::validate` already refuses both, so this is a floor under
/// arithmetic rather than a policy.
fn resistive_anisotropy(voicing: &Voicing) -> f64 {
    let fitted = f64::from(voicing.horizontal_decay_ratio).clamp(0.02, 0.98);
    1.0 - (1.0 - fitted).sqrt()
}

/// Fraction of a partial's decay rate that is loss **into the board**: the
/// coupling constant, derived rather than read.
///
/// The slowest mode of the coupled group radiates nothing and decays at
/// `(1 - share) sigma_k`, the loudest decays at `sigma_k`, so `1 - share` *is*
/// the fitted aftersound/prompt decay ratio — which is exactly what
/// `voicing.horizontal_decay_ratio` was fitted to be. `docs/history/FUNDAMENTALS.md` §2.6's
/// contradiction with `voicing.bridge.radiated_share = 0.5` is resolved here, in
/// favour of the field that was fitted to recordings; the shipped 0.5 would cap
/// the aftersound at half the prompt rate and delete the double decay.
///
/// `voicing.bridge.radiated_share` keeps its own, narrower job: the *fluctuation*
/// of the board's mobility on each partial's decay ([`radiated_damping`]).
fn radiated_share(voicing: &Voicing) -> f64 {
    1.0 - resistive_anisotropy(voicing)
}

/// The hammer's horizontal leak times the horizontal plane's radiation
/// efficiency, in amplitude.
///
/// Because `C` is block diagonal the two factors are never separable — every
/// horizontal mode's gain carries both as one common scalar — so the product is
/// what the model has, and it is taken from the field that was fitted to it.
/// At zero coupling the construction reduces to the free-running one exactly.
fn horizontal_leak(voicing: &Voicing) -> f64 {
    f64::from(db_to_amp(voicing.horizontal_gain_db))
}

/// Where a MIDI velocity sits between the softest and the loudest blow, 0 to 1.
///
/// Velocity 0 is a note-off in the protocol and never reaches a string, so the
/// span is 1 to 127 and both ends of a [`StrikeDirection`] are really reached.
fn strike_position_in_velocity(vel: u16) -> f32 {
    (velocity_from_midi(vel).max(1.0) - 1.0) / 126.0
}

/// What one velocity does to the *direction* of the strike vector: a scale on
/// the horizontal plane's share of the blow, and a tilt on how far the group's
/// per-string shares sit from their mean.
///
/// Both are ratios inside `u`, so neither changes its length — the loudness law
/// stays entirely in the hammer. See [`StrikeDirection`].
#[derive(Clone, Copy, Debug)]
struct StrikeMix {
    /// Multiplier on [`horizontal_leak`], in amplitude.
    leak_scale: f32,
    /// Multiplier on `(s_j - 1)`, the group's share asymmetry.
    share_tilt: f32,
}

impl StrikeMix {
    /// The mix a preset with no `[voicing.strike_direction]` has at every
    /// velocity: the fitted strike vector, untouched.
    const NEUTRAL: StrikeMix = StrikeMix {
        leak_scale: 1.0,
        share_tilt: 0.0,
    };

    fn at(direction: &StrikeDirection, vel: u16) -> StrikeMix {
        let t = strike_position_in_velocity(vel);
        let db = direction.vh_db_at_pp + (direction.vh_db_at_ff - direction.vh_db_at_pp) * t;
        StrikeMix {
            leak_scale: db_to_amp(db),
            // About mid-velocity, so `unison_layout.share` stays the shares of
            // a mezzo blow and a fit that writes this field does not silently
            // move the note it was fitted on.
            share_tilt: direction.share_tilt * (2.0 * t - 1.0),
        }
    }
}

// ------------------------------------------------------------ complex scalars

/// A complex number, `f64`, for the construction only.
///
/// Twenty lines rather than a dependency: the eigensolve needs four operators
/// and `exp`, it runs once per partial at preset load, and nothing on the audio
/// path ever sees one.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Cx {
    re: f64,
    im: f64,
}

impl Cx {
    const ZERO: Cx = Cx { re: 0.0, im: 0.0 };

    fn new(re: f64, im: f64) -> Cx {
        Cx { re, im }
    }

    fn from_polar(r: f64, theta: f64) -> Cx {
        Cx::new(r * theta.cos(), r * theta.sin())
    }

    fn norm(self) -> f64 {
        self.re.hypot(self.im)
    }

}

impl std::ops::Add for Cx {
    type Output = Cx;
    fn add(self, o: Cx) -> Cx {
        Cx::new(self.re + o.re, self.im + o.im)
    }
}

impl std::ops::Sub for Cx {
    type Output = Cx;
    fn sub(self, o: Cx) -> Cx {
        Cx::new(self.re - o.re, self.im - o.im)
    }
}

impl std::ops::Mul for Cx {
    type Output = Cx;
    fn mul(self, o: Cx) -> Cx {
        Cx::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
}

impl std::ops::Mul<f64> for Cx {
    type Output = Cx;
    fn mul(self, k: f64) -> Cx {
        Cx::new(self.re * k, self.im * k)
    }
}

impl std::ops::Div for Cx {
    type Output = Cx;
    fn div(self, o: Cx) -> Cx {
        let d = o.re * o.re + o.im * o.im;
        Cx::new(
            (self.re * o.re + self.im * o.im) / d,
            (self.im * o.re - self.re * o.im) / d,
        )
    }
}

impl std::ops::AddAssign for Cx {
    fn add_assign(&mut self, o: Cx) {
        *self = *self + o;
    }
}

impl std::ops::SubAssign for Cx {
    fn sub_assign(&mut self, o: Cx) {
        *self = *self - o;
    }
}

// ------------------------------------------------------------ the eigenproblem

/// Modes per partial: `N` strings x 2 polarizations, at most.
const MAX_MODES_PER_PARTIAL: usize = 2 * MAX_UNISON;

/// One eigenmode of one partial: a complex pole and a complex radiated gain.
#[derive(Clone, Copy, Debug, Default)]
pub struct StringMode {
    /// Frequency, Hz — the imaginary part of `lambda`, over `2 pi`.
    pub hz: f32,
    /// Decay rate, 1/s — minus the real part of `lambda`.
    pub sigma: f32,
    /// Real part of the mode's complex input gain.
    pub gain_re: f32,
    /// Imaginary part of it. Zero only by accident: the strike projects onto a
    /// non-orthogonal eigenbasis, so every mode starts at its own phase.
    pub gain_im: f32,
    /// Which polarization block the mode came out of, which decides where it
    /// radiates from when the preset spreads the polarizations.
    pub horizontal: bool,
}

impl StringMode {
    /// Magnitude of the complex input gain.
    pub fn gain(&self) -> f32 {
        self.gain_re.hypot(self.gain_im)
    }
}

/// Coefficients of `prod_j (z - r_j)`, low order first, monic.
fn poly_from_roots(roots: &[Cx], out: &mut [Cx]) -> usize {
    out[0] = Cx::new(1.0, 0.0);
    let mut len = 1;
    for &r in roots {
        out[len] = Cx::ZERO;
        len += 1;
        for i in (0..len).rev() {
            let c = out[i];
            out[i] = if i > 0 { out[i - 1] } else { Cx::ZERO } - c * r;
        }
    }
    len
}

/// All roots of a monic complex polynomial, by Durand-Kerner.
///
/// Degree is at most [`MAX_UNISON`] and the caller has already shifted the
/// variable so that every root is `O(radius)`, which is what makes a plain
/// Weierstrass iteration converge in a few dozen steps without any scaling care.
fn roots_of(coeff: &[Cx], radius: f64, out: &mut [Cx]) -> usize {
    let n = coeff.len() - 1;
    if n == 0 {
        return 0;
    }
    if n == 1 {
        out[0] = Cx::ZERO - coeff[0] / coeff[1];
        return 1;
    }
    let seed = Cx::new(0.4, 0.9);
    let scale = radius.max(1e-12);
    let mut power = Cx::new(1.0, 0.0);
    for slot in out.iter_mut().take(n) {
        *slot = power * scale;
        power = power * seed;
    }
    let eval = |x: Cx| {
        // Horner, high order first.
        coeff.iter().rev().fold(Cx::ZERO, |acc, &c| acc * x + c)
    };
    for _ in 0..500 {
        let mut moved = 0.0f64;
        for i in 0..n {
            let mut denom = coeff[n];
            for j in 0..n {
                if j != i {
                    denom = denom * (out[i] - out[j]);
                }
            }
            if denom.norm() < 1e-300 {
                continue;
            }
            let step = eval(out[i]) / denom;
            out[i] -= step;
            moved = moved.max(step.norm());
        }
        if moved < 1e-13 * scale {
            break;
        }
    }
    n
}

/// One eigenmode of one polarization block, before the strike is projected onto
/// it: the pole, the eigenvector, its complex-symmetric norm, and what it puts
/// on the bridge.
#[derive(Clone, Copy, Debug, Default)]
struct BlockMode {
    lambda: Cx,
    v: [Cx; MAX_UNISON],
    norm: Cx,
    radiated: Cx,
}

/// The `N` eigenmodes of one polarization block: `D - c J_N` with
/// `D = diag(i omega_j - sigma_int)`.
///
/// The rank-one structure gives both halves in closed form. Eigenvalues are the
/// roots of `prod_j (lambda - d_j) + c sum_i prod_{j != i} (lambda - d_j)` —
/// the second term is `c` times the derivative of the first — and eigenvectors
/// are `v_jm = 1/(d_j - lambda_m)`. Because `D - cJ` is complex **symmetric**
/// its left eigenvectors are its right ones transposed (not conjugated), so the
/// row of `V^-1` the strike projection needs is `v_m / (v_m . v_m)`; using
/// `v_m^H` would be the wrong basis for a non-normal matrix
/// (`docs/history/FUNDAMENTALS.md` §5.1).
fn block_solve(omegas: &[f64], sigma_int: f64, c: Cx, out: &mut [BlockMode]) -> usize {
    let n = omegas.len();
    debug_assert!(n <= MAX_UNISON);
    // Degenerate poles make `1/(d_j - lambda)` singular in a way that does not
    // cancel, so a group whose strings are tuned to the same number is nudged
    // apart by an amount far below anything audible or measurable (1e-6 rad/s
    // is 1.6e-7 Hz). A real unison is never exactly in tune, and
    // `notes.detune_cents = 0` is a bisection rung, not an instrument.
    let mut w = [0.0f64; MAX_UNISON];
    w[..n].copy_from_slice(omegas);
    for i in 0..n {
        for j in 0..i {
            if (w[i] - w[j]).abs() < 1e-6 {
                w[i] += 1e-6 * (i - j) as f64;
            }
        }
    }
    let mut d = [Cx::ZERO; MAX_UNISON];
    for (slot, &omega) in d[..n].iter_mut().zip(&w[..n]) {
        *slot = Cx::new(-sigma_int, omega);
    }
    // Shift to the block's centre before forming the polynomial: the roots are
    // then `O(detune spread + N|c|)` instead of `O(omega)`, which is the
    // difference between a well conditioned degree-3 solve and a hopeless one.
    let mut centre = Cx::ZERO;
    for &x in &d[..n] {
        centre += x;
    }
    centre = centre * (1.0 / n as f64);
    let mut e = [Cx::ZERO; MAX_UNISON];
    for (slot, &x) in e[..n].iter_mut().zip(&d[..n]) {
        *slot = x - centre;
    }
    let mut coeff = [Cx::ZERO; MAX_UNISON + 1];
    let len = poly_from_roots(&e[..n], &mut coeff);
    for i in 1..len {
        // Read before write: step `i` writes index `i-1` and reads index `i`,
        // which the previous steps have not touched.
        let above = coeff[i];
        coeff[i - 1] += c * above * i as f64;
    }
    let radius = e[..n].iter().map(|x| x.norm()).fold(0.0, f64::max) + n as f64 * c.norm();
    let mut roots = [Cx::ZERO; MAX_UNISON];
    let found = roots_of(&coeff[..len], radius.max(1e-9), &mut roots);
    let mut made = 0usize;
    for &z in &roots[..found] {
        let lambda = centre + z;
        let mut mode = BlockMode {
            lambda,
            ..BlockMode::default()
        };
        let mut norm = Cx::ZERO;
        let mut radiated = Cx::ZERO;
        for (slot, &dj) in mode.v[..n].iter_mut().zip(&d[..n]) {
            let v = Cx::new(1.0, 0.0) / (dj - lambda);
            *slot = v;
            norm += v * v;
            radiated += v;
        }
        if norm.norm() < 1e-300 {
            continue;
        }
        mode.norm = norm;
        mode.radiated = radiated;
        out[made] = mode;
        made += 1;
    }
    // Durand-Kerner returns its roots in whatever order the iteration left them,
    // which is not an order at all. Sorting by frequency makes the mode list a
    // stable, meaningful thing: mode `j` of the vertical block and mode `j` of
    // the horizontal one are then the same eigenvector at two admittances, which
    // is what makes the polarization split readable and what keeps two solves of
    // the same key comparable term by term.
    out[..made].sort_by(|a, b| a.lambda.im.partial_cmp(&b.lambda.im).expect("finite"));
    made
}

/// The radiated gain of one block mode for a strike vector `u`:
/// `G_m = (w . v_m) (v_m . u) / (v_m . v_m)`, with `w = 1` inside a block.
fn project(mode: &BlockMode, u: &[Cx]) -> Cx {
    let mut dot = Cx::ZERO;
    for (v, x) in mode.v.iter().zip(u) {
        dot += *v * *x;
    }
    mode.radiated * (dot / mode.norm)
}

/// One partial's `2N` eigenmodes, with the strike projections the instrument
/// needs.
///
/// Four gain tables rather than one, because two things about the *hammer* — not
/// about the strings — move at run time and neither may rebuild the eigenmodes:
///
/// * **which strings it reaches**, whole hammer or una corda (`gain` /
///   `una_gain`);
/// * **how far out of square it meets them**, which moves with velocity
///   ([`StrikeDirection`]). Because `project` is linear in `u` and the share
///   tilt enters `u` linearly, the mixture at any velocity is
///   `gain + tau gain_tilt` — so the derivative is cached beside the value and
///   a note-on is two fused multiply-adds per mode, with no solve and no
///   allocation.
///
/// The tilt tables are built only when the preset has a `[strike_direction]`;
/// without one they stay empty and a note-on is exactly the table swap it was.
#[derive(Clone, Copy, Debug)]
struct PartialSolution {
    lambda: [Cx; MAX_MODES_PER_PARTIAL],
    gain: [Cx; MAX_MODES_PER_PARTIAL],
    gain_tilt: [Cx; MAX_MODES_PER_PARTIAL],
    una_gain: [Cx; MAX_MODES_PER_PARTIAL],
    una_tilt: [Cx; MAX_MODES_PER_PARTIAL],
    /// `2N`. The first `N` entries are the vertical block.
    len: usize,
}

impl PartialSolution {
    fn empty() -> PartialSolution {
        PartialSolution {
            lambda: [Cx::ZERO; MAX_MODES_PER_PARTIAL],
            gain: [Cx::ZERO; MAX_MODES_PER_PARTIAL],
            gain_tilt: [Cx::ZERO; MAX_MODES_PER_PARTIAL],
            una_gain: [Cx::ZERO; MAX_MODES_PER_PARTIAL],
            una_tilt: [Cx::ZERO; MAX_MODES_PER_PARTIAL],
            len: 0,
        }
    }
}

/// The per-string strike vector of one partial, and its derivative with respect
/// to the share tilt.
///
/// `struck` is how many of the group the hammer actually reaches — every string
/// normally, one fewer under una corda, which is the *direction* of the strike
/// vector changing and therefore a different mixture of the same eigenmodes.
///
/// `tilt` is the other direction change, and it is written as a derivative
/// rather than evaluated: `s_j(tau) = 1 + (1 + tau)(s_j - 1)`, so
/// `d s_j / d tau = s_j - 1` and the mixture is affine in `tau`. The tilt is
/// taken about the row's mean, which is exactly 1, so the group's total
/// excitation is the same at every velocity and nothing here is a second
/// loudness law.
fn strike_vector(
    voicing: &Voicing,
    n: usize,
    gain_k: f64,
    struck: usize,
    out: &mut [Cx; MAX_UNISON],
    tilt: &mut [Cx; MAX_UNISON],
) {
    for (j, (slot, dslot)) in out[..n].iter_mut().zip(&mut tilt[..n]).enumerate() {
        if j < struck {
            let share = f64::from(voicing.strike_share(j, n));
            *slot = Cx::new(share * gain_k, 0.0);
            *dslot = Cx::new((share - 1.0) * gain_k, 0.0);
        } else {
            *slot = Cx::ZERO;
            *dslot = Cx::ZERO;
        }
    }
}

/// Which string of a group carries a within-string split.
///
/// A false beat is a defect of one wire, and `notes.false_beat` names a key and
/// a partial rather than a string, so the convention has to live somewhere: the
/// group's **first** string carries it. Nothing distinguishes the strings of a
/// unison but their detuning and their share of the blow, both of which are
/// symmetric under relabelling, and the first string is the one the una-corda
/// hammer always reaches — so a split written here is heard under both pedals.
const SPLIT_STRING: usize = 0;

/// The `2N` modes of partial `k`, at a given scale on the whole loss budget.
///
/// `scale` multiplies `sigma_int` and `gamma_v` together, which is the only free
/// parameter left once the physics has fixed their ratio; [`decay_scale`] picks
/// it so that the composite reaches -60 dB on the fitted anchor.
///
/// `split` is this partial's within-string false beat, if it has one. It enters
/// on the **diagonal of `Omega_k`** for the split string's horizontal entry,
/// before the block is solved, so its companion is one of the group's own
/// eigenvalues and carries a decay rate the coupled system chose — not a
/// sinusoid pasted onto the answer. The mode count does not move: the split
/// takes the horizontal mode that was already there, 27.6 dB down and a
/// hundredth of a hertz away, and puts it where the measurement says it is.
fn partial_modes(
    params: &StringParams,
    voicing: &Voicing,
    k: usize,
    sigma_hat: f64,
    gain_k: f64,
    scale: f64,
    split: Option<FalseBeat>,
) -> PartialSolution {
    let n = params.unison.clamp(1, MAX_UNISON);
    let share = radiated_share(voicing);
    let rho = resistive_anisotropy(voicing);
    let beta = REACTIVE_RATIO;
    let sigma = scale * sigma_hat;
    let sigma_int = (1.0 - share) * sigma;
    let gamma_v = share * sigma / n as f64;
    let c_v = Cx::new(gamma_v, gamma_v * beta);
    let c_h = Cx::new(
        gamma_v * rho,
        gamma_v * beta * (1.0 - REACTIVE_ANISOTROPY),
    );
    // The pull common to the whole partial is a tuning offset, not a beat: the
    // symmetric vertical mode would sit `N gamma_v beta / 2 pi` flat (0.9 cents
    // at C4 k=1), and `notes.f0` is fitted to recordings that already contain
    // whatever pull the real bridge applies. Adding it back keeps the pitch
    // exactly where the free-running construction put it, which is what makes
    // the change inaudible as a *tuning* change. What survives is the
    // difference between the modes — the anti-veering — and the difference
    // between the two blocks, which is the polarization split.
    let compensation = n as f64 * gamma_v * beta;
    let mut omegas = [0.0f64; MAX_UNISON];
    for (j, slot) in omegas[..n].iter_mut().enumerate() {
        let detune = f64::from(voicing.detune_ratio(j, n, params.detune_cents));
        *slot = f64::from(std::f32::consts::TAU) * f64::from(params.partial_freq(k)) * detune
            + compensation;
    }
    let mut u_full = [Cx::ZERO; MAX_UNISON];
    let mut d_full = [Cx::ZERO; MAX_UNISON];
    let mut u_una = [Cx::ZERO; MAX_UNISON];
    let mut d_una = [Cx::ZERO; MAX_UNISON];
    strike_vector(voicing, n, gain_k, n, &mut u_full, &mut d_full);
    strike_vector(voicing, n, gain_k, (n - 1).max(1), &mut u_una, &mut d_una);
    let leak = horizontal_leak(voicing);

    let mut solution = PartialSolution::empty();
    let mut block = [BlockMode::default(); MAX_UNISON];
    for (plane, &c) in [c_v, c_h].iter().enumerate() {
        let horizontal = plane == 1;
        // The false beat, on the diagonal and only in the other plane: it is a
        // split *within* one string, so the vertical block never sees it.
        let mut plane_omegas = omegas;
        if horizontal {
            if let Some(fb) = split {
                plane_omegas[SPLIT_STRING] +=
                    f64::from(std::f32::consts::TAU) * f64::from(fb.hz);
            }
        }
        let found = block_solve(&plane_omegas[..n], sigma_int, c, &mut block);
        let plane_gain = if horizontal { leak } else { 1.0 };
        // The false beat's own drive is a **second term beside the leak**, not a
        // factor inside it: the whole point of the mechanism is that the split
        // plane is driven 20 dB harder than the leak drives the other two. It is
        // therefore projected separately and added, which is also what keeps a
        // preset with no splits arithmetically identical to the construction
        // before this one — the expression it evaluates is unchanged, term for
        // term and rounding for rounding.
        let mut extra = [Cx::ZERO; MAX_UNISON];
        if horizontal {
            if let Some(fb) = split {
                // How hard the split plane has to be driven for its companion to
                // stand where the measurement says it does. `reference` is the
                // loudest mode of the vertical block, which the split does not
                // touch, so the level the table asks for is a level against a
                // fixed thing.
                let reference = solution.gain[..solution.len]
                    .iter()
                    .fold(0.0f64, |m, g| m.max(g.norm()));
                extra[SPLIT_STRING] = false_beat_drive(
                    &block[..found],
                    &u_full[..n],
                    leak,
                    gain_k,
                    reference * f64::from(db_to_amp(fb.db)),
                );
            }
        }
        let split_drive = extra[SPLIT_STRING] != Cx::ZERO;
        for mode in &block[..found] {
            let i = solution.len;
            solution.lambda[i] = mode.lambda;
            solution.gain[i] = project(mode, &u_full[..n]) * plane_gain;
            solution.gain_tilt[i] = project(mode, &d_full[..n]) * plane_gain;
            solution.una_gain[i] = project(mode, &u_una[..n]) * plane_gain;
            solution.una_tilt[i] = project(mode, &d_una[..n]) * plane_gain;
            if split_drive {
                let from_split = project(mode, &extra[..n]);
                solution.gain[i] += from_split;
                solution.una_gain[i] += from_split;
            }
            solution.len += 1;
        }
        // A block that lost a root to a degenerate norm would misalign the two
        // halves; pad it back out so the vertical block is always the first `n`.
        while solution.len < (plane + 1) * n {
            let i = solution.len;
            solution.lambda[i] = Cx::new(-sigma_int, omegas[0]);
            solution.gain[i] = Cx::ZERO;
            solution.gain_tilt[i] = Cx::ZERO;
            solution.una_gain[i] = Cx::ZERO;
            solution.una_tilt[i] = Cx::ZERO;
            solution.len += 1;
        }
    }
    solution
}

/// How much extra excitation the split string's horizontal plane needs for the
/// companion to stand at `target`.
///
/// # What "the companion" is, once the bridge has had its say
///
/// Not one mode. A force on **one** string of a group standing on **one** bridge
/// point reaches all `N` of the block's normal modes, and it reaches them at
/// comparable strength: measured on the default preset at C4's second partial,
/// a unit drive on the split string's horizontal plane comes back on the three
/// modes in the ratio 0.45 : 0.27 : 0.29. That is not a defect of the solve, it
/// is the coupling — `v_jm = 1/(d_j - lambda_m)` is only as localised as the
/// detuning is wide against `|c|`, and in the horizontal plane it is not very.
///
/// So the quantity normalised is the **whole horizontal block's contribution to
/// the partial at the strike**, `|sum_m G_m|` — the amplitude that beats against
/// the mode carrying the note, which is what a measured beat depth inverts to
/// and therefore the only thing a fit could write. `reference` is the loudest
/// mode of the *vertical* block, which the split does not touch, so the level
/// the table asks for is a level against something fixed.
///
/// # Why it is one quadratic and not a search
///
/// [`project`] is linear in `u`, so the block's coherent sum is affine in the
/// extra drive: `S(a) = A + a B`, with `A` what the leak already gives it and
/// `B` what one unit of drive on the split string would. Then
///
/// ```text
///     |B|^2 a^2 + 2 Re(A conj B) a + (|A|^2 - target^2) = 0
/// ```
///
/// and the positive root is the answer. `target` is above `|A|` by construction
/// — `notes.false_beat.db` runs from -20 dB of the loudest mode up to it, and
/// the leak sits 27.6 dB under — so the constant term is negative, the
/// discriminant is positive, and exactly one root is positive.
fn false_beat_drive(
    block: &[BlockMode],
    u: &[Cx],
    plane_gain: f64,
    gain_k: f64,
    target: f64,
) -> Cx {
    let n = u.len();
    let mut unit = [Cx::ZERO; MAX_UNISON];
    unit[SPLIT_STRING] = Cx::new(gain_k, 0.0);
    let mut a = Cx::ZERO;
    let mut b = Cx::ZERO;
    for mode in block {
        a += project(mode, u) * plane_gain;
        b += project(mode, &unit[..n]);
    }
    let bb = b.re * b.re + b.im * b.im;
    if bb < 1e-300 {
        return Cx::ZERO;
    }
    let half_b = a.re * b.re + a.im * b.im;
    let c = a.re * a.re + a.im * a.im - target * target;
    let disc = half_b * half_b - bb * c;
    if disc < 0.0 {
        // The leak alone already stands over the target, which the bounds on
        // `db` make unreachable; nothing to add.
        return Cx::ZERO;
    }
    Cx::new(((-half_b + disc.sqrt()) / bb).max(0.0) * gain_k, 0.0)
}

/// Grid the composite envelope's -60 dB crossing is found on, and how far past
/// the anchor it reaches.
///
/// The prototype used 4000 points over ten anchors and measured 67-82 ms per
/// key, which is 1.3 s for the compass; that is a second of preset load, and
/// `render_to_buffer` builds the engine inside the window the performance
/// acceptance tests measure, so it is also ten points of the budget. 256 points
/// over three anchors is 1.2 % of the target between grid lines and the crossing
/// is interpolated in log amplitude between the two either side of it, so what
/// the grid sets is the cost and not the answer. See [`decay_scale`] for the
/// other three quarters of the saving.
const T60_GRID: usize = 192;
const T60_SPAN: f64 = 3.0;

/// The composite's -60 dB time for a mode set, in seconds.
///
/// This is the **coherent** modulus `|sum_m G_m e^{lambda_m t}|` — beats and all
/// — because that is the signal, and the last time it is above the threshold is
/// what "the note has gone" means: a double-decay envelope with a beat on it can
/// dip below and come back. Two costs come with that choice and both are paid
/// below rather than avoided:
///
/// * it is a **grid** statistic (the modulus has no closed-form crossing), which
///   is why the mode set is advanced by a fixed complex ratio — one complex
///   multiply per mode per step instead of an `exp`;
/// * it is **not continuous in the mode set**: move the damping infinitesimally
///   and a beat trough that was just above the threshold drops just below it, and
///   the crossing jumps by a whole beat period. [`decay_scale`] therefore
///   bisects rather than iterating, because a Newton step on a staircase lands
///   anywhere inside a tread — measured at up to 48 % of the anchor before the
///   solve was made a bisection.
///
/// Returns [`T60_SPAN`] times `t_max / T60_SPAN` when the composite never gets
/// there, which keeps the solve's objective monotone instead of letting it fall
/// off a cliff.
fn composite_t60(solution: &PartialSolution, t_max: f64) -> f64 {
    let dt = t_max / T60_GRID as f64;
    let mut state = [Cx::ZERO; MAX_MODES_PER_PARTIAL];
    let mut step = [Cx::ZERO; MAX_MODES_PER_PARTIAL];
    let mut peak = 0.0f64;
    for i in 0..solution.len {
        state[i] = solution.gain[i];
        step[i] = Cx::from_polar(
            (solution.lambda[i].re * dt).exp(),
            solution.lambda[i].im * dt,
        );
    }
    // The strike's own amplitude, which is what -60 dB is counted from.
    let mut coherent = Cx::ZERO;
    for x in &state[..solution.len] {
        coherent += *x;
    }
    peak = peak.max(coherent.norm());
    if peak <= 0.0 {
        return 0.0;
    }
    let threshold = 1e-3 * peak;
    let mut last = 0usize;
    let mut above = 0.0f64;
    let mut below = 0.0f64;
    let mut previous = coherent.norm();
    for i in 1..=T60_GRID {
        for (x, z) in state[..solution.len].iter_mut().zip(&step[..solution.len]) {
            *x = *x * *z;
        }
        let mut sum = Cx::ZERO;
        for x in &state[..solution.len] {
            sum += *x;
        }
        let a = sum.norm();
        if previous > threshold && a <= threshold {
            last = i;
            above = previous;
            below = a;
        }
        previous = a;
    }
    if last == 0 {
        return t_max;
    }
    let frac = if above > below && below > 0.0 {
        ((above / threshold).ln() / (above / below).ln()).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (last as f64 - 1.0 + frac) * dt
}

/// The one scale on `sigma_int` and `gamma_v` that puts the composite's -60 dB
/// crossing on the fitted anchor `6.91 / sigma_hat`.
///
/// This is `Voicing::vertical_decay_factor` generalised: that closed form
/// assumes a two-exponential construction and is exact for it, there is no
/// closed form for `2N` coupled modes, so it is solved.
///
/// In two stages, because [`composite_t60`] is a staircase with jumps of one
/// beat period — a Newton step on it lands anywhere inside a tread, measured at
/// up to 48 % of the anchor — and a bisection wide enough to be safe is eight
/// times more grid work than the answer needs:
///
/// 1. **The beat-free envelope**, `sum_m |G_m| exp(-sigma_m t)`, which is
///    strictly decreasing and needs no grid at all. Newton on `ln t60` against
///    `ln scale`, whose slope is very nearly `-1` because scaling the whole loss
///    budget scales every decay rate, converges in three or four steps and lands
///    within about 20 % of the answer — the two statistics differ by however
///    much the tail's modes cancel.
/// 2. **A bisection of the real statistic** inside a factor of 2.5 either side
///    of that, eleven steps, widening to the full bracket in the rare cell where
///    the beat-free estimate was further out than that.
///
/// `refine` turns stage 2 off, and the caller turns it off for the partials that
/// are more than 50 dB under the note's loudest — where the whole partial is
/// inaudible, the difference between the two statistics is inaudible twice over,
/// and a bass note has sixty such partials out of eighty. Stage 2 is also
/// skipped outright for a single string, whose two modes come out with real,
/// positive gains a few thousandths of a hertz apart: the beat-free envelope
/// *is* the coherent one over any window a T60 fits in.
fn decay_scale(
    params: &StringParams,
    voicing: &Voicing,
    k: usize,
    sigma_hat: f64,
    gain_k: f64,
    split: Option<FalseBeat>,
    refine: bool,
) -> f64 {
    let target = 6.91 / sigma_hat.max(1e-6);
    let span = T60_SPAN * target;
    let modes_at = |x: f64| partial_modes(params, voicing, k, sigma_hat, gain_k, x.exp(), split);
    let residual = |x: f64| {
        let t60 = composite_t60(&modes_at(x), span);
        if t60 <= 0.0 {
            return f64::NEG_INFINITY;
        }
        t60 - target
    };

    // Stage 1: the beat-free envelope, gridless.
    let mut x = 0.0f64;
    for _ in 0..5 {
        let t60 = beatless_t60(&modes_at(x));
        if t60 <= 0.0 {
            break;
        }
        let step = (t60 / target).ln();
        if step.abs() < 1e-3 {
            break;
        }
        x += step.clamp(-2.0, 2.0);
    }

    // A single string has nothing to beat against: its two modes come out with
    // real, positive gains at frequencies a few thousandths of a hertz apart, so
    // the beat-free envelope *is* the coherent one over any window a T60 fits
    // in, and the grid below would only cost. Fifteen keys of the compass and a
    // fifth of every solve. A false beat is precisely the exception — it puts
    // the two planes of one wire a whole hertz apart at comparable amplitude,
    // which is a beat by construction — so a split partial takes the grid
    // however many strings the key has.
    if (params.unison.clamp(1, MAX_UNISON) == 1 && split.is_none()) || !refine {
        return x.exp();
    }

    // Stage 2: bisect the crossing that is actually rendered.
    //
    // The bisection converges on the staircase's *jump* rather than on a root,
    // because the objective has none: a trough that dips under the threshold and
    // comes back moves the answer by a whole beat period, and there are scales
    // either side of that jump and none in between. So the point this returns is
    // a whole tread from the anchor whenever the last halving lands on the far
    // side of it, which is worth 5-23 % on twelve of the default preset's 302
    // tracked cells. Returning the **best residual the search visited** instead —
    // the same discipline `estimate::motion::FalseBeatLoop` needed for the same
    // landscape (`DECISIONS.md` 250) — costs nothing and takes the worst error
    // from 22.9 % to 18.4 % and the p90 from 8.0 % to 6.3 % (4.2 % with
    // `T60_GRID` at 768). It is **not** shipped, and `DECISIONS.md` 259 records
    // why: it moves the render by -61.8 dB RMS, which is an order of magnitude
    // under item 221's click fix, and that is still enough to take two of
    // `tuner/tests/calibration.rs`'s round trips red — the estimators are
    // calibrated against this forward model, and re-deriving them is the
    // milestone `docs/history/FUNDAMENTALS.md` §7.7 describes and item 232 lists.
    let (mut lo, mut hi) = (x - 0.9, x + 0.9);
    let mut steps = 11;
    if residual(lo) <= 0.0 || residual(hi) >= 0.0 {
        lo = -4.0;
        hi = 4.0;
        steps = 18;
        if residual(lo) <= 0.0 {
            return lo.exp();
        }
        if residual(hi) >= 0.0 {
            return hi.exp();
        }
    }
    for _ in 0..steps {
        let mid = 0.5 * (lo + hi);
        if residual(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (0.5 * (lo + hi)).exp()
}

/// The composite's -60 dB time with the beats taken out of it:
/// `E(t) = sum_m |G_m| exp(-sigma_m t)`, `E(T60) = 1e-3 E(0)`.
///
/// Strictly decreasing, so it is a plain bisection in `t` with no grid, and it
/// is exactly the closed form it replaces — with one string and two
/// polarizations `E` is `1 + g_h` at the strike and `g_h exp(-sigma_h t)` in the
/// tail, which is `Voicing::vertical_decay_factor`'s equation term for term.
/// What it is *not* is the signal: the coherent modulus beats, and the beats are
/// worth up to 20 % of the crossing, which is why it only starts the solve.
fn beatless_t60(solution: &PartialSolution) -> f64 {
    let mut amp = [0.0f64; MAX_MODES_PER_PARTIAL];
    let mut sigma = [0.0f64; MAX_MODES_PER_PARTIAL];
    let mut total = 0.0f64;
    let mut slowest = f64::INFINITY;
    for i in 0..solution.len {
        amp[i] = solution.gain[i].norm();
        sigma[i] = (-solution.lambda[i].re).max(1e-9);
        total += amp[i];
        if amp[i] > 0.0 {
            slowest = slowest.min(sigma[i]);
        }
    }
    if total <= 0.0 || !slowest.is_finite() {
        return 0.0;
    }
    let level = |t: f64| -> f64 {
        let mut sum = 0.0;
        for i in 0..solution.len {
            sum += amp[i] * (-sigma[i] * t).exp();
        }
        sum
    };
    let threshold = 1e-3 * total;
    let mut hi = 2.0 * (total / threshold).ln() / slowest;
    if !hi.is_finite() || hi <= 0.0 {
        return 0.0;
    }
    let mut lo = 0.0f64;
    for _ in 0..32 {
        let mid = 0.5 * (lo + hi);
        if level(mid) > threshold {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// The unison group belonging to one key: the coupled eigenmodes of every
/// partial, in two banks split by polarization.
pub struct PianoString {
    params: StringParams,
    /// The vertical block's `N` modes of every partial, partial-major.
    vertical: ModalBank,
    /// The horizontal block's, in the same order.
    horizontal: ModalBank,
    /// One excitation buffer for the whole group. The per-string shares and the
    /// per-string timing skew are inside the modes' complex gains, which is the
    /// point of the eigen construction: `N` input buffers collapse into `2N`
    /// complex scalars.
    excitation: [f32; BLOCK],
    /// Extra damping per mode at full damper engagement, 1/s.
    damper_profile: Vec<f32>,
    /// Scratch for the current damper engagement, kept to avoid allocating.
    damper_extra: Vec<f32>,
    /// Mode gains for the whole hammer at the fitted strike direction, vertical
    /// block then horizontal, partial-major.
    strike_gain: Vec<[f32; 2]>,
    /// ... and for the una-corda hammer, which misses one string of the group
    /// and therefore hands the same eigenmodes a different mixture.
    una_corda_gain: Vec<[f32; 2]>,
    /// Derivative of each of those with respect to the share tilt, so that a
    /// velocity-dependent strike *direction* is two fused multiply-adds per mode
    /// at note-on instead of a second solve. Empty when the preset has no
    /// `[voicing.strike_direction]`, which is also when they are never read.
    strike_tilt: Vec<[f32; 2]>,
    una_corda_tilt: Vec<[f32; 2]>,
    /// The preset's velocity law for the strike vector's direction, if it has
    /// one. `None` is the fitted direction at every velocity, bit for bit.
    direction: Option<StrikeDirection>,
    partials: usize,
    /// Modes per partial per block, i.e. the number of unison strings.
    strings: usize,
    una_corda: bool,
    /// Velocity the mode gains currently hold the mixture for. Nominal until
    /// something strikes the key, which is what a sympathetically driven string
    /// keeps.
    velocity: u16,
    damper: f32,
}

/// How far the admittance may move a partial's fitted decay rate, as a
/// multiplier on it.
///
/// `sigma_k * (1 + share * (|P| - 1))` is unbounded above — forty +20 dB peaks
/// on one frequency multiply `|P|` by ten thousand — and can approach zero
/// below. The fitted `sigma(f)` is a *measurement* and the admittance's
/// fluctuation is a correction to it, so the correction is held to a factor of
/// four either way: a partial may ring twice as long or die four times as fast
/// as the recordings say, and no more, whatever the bridge asks for. Both ends
/// are far outside the ±10–15 dB of real board fluctuation, so this clamps
/// pathological presets and nothing else.
pub const RADIATED_FACTOR_RANGE: (f32, f32) = (0.25, 4.0);

/// The per-partial multiplier `voicing.bridge.radiated_share` implies, one
/// entry per partial. All ones — and built as ones, not computed — when the
/// preset has no bridge or asks for no share of it, which is what keeps every
/// existing preset's strings bit for bit what they were.
fn radiated_damping(params: &StringParams, voicing: &Voicing, partials: usize) -> Vec<f32> {
    let share = match &voicing.bridge {
        Some(bridge) if bridge.radiated_share > 0.0 => bridge.radiated_share,
        _ => return vec![1.0; partials],
    };
    // The *fluctuation* of the board's mobility, not its mean: the mean is
    // already in the fitted `sigma(f)` and adding it again would retune the
    // whole compass. See `BridgeVoicing::radiated_share`.
    let modes = BridgeFilter::peaks_only(voicing.bridge.as_ref().expect("checked above"));
    (1..=partials)
        .map(|k| {
            let excess = modes.magnitude(params.partial_freq(k)) - 1.0;
            (1.0 + share * excess).clamp(RADIATED_FACTOR_RANGE.0, RADIATED_FACTOR_RANGE.1)
        })
        .collect()
}

/// The smallest decay rate a mode may be given, 1/s.
///
/// Not a musical bound — `T60 = 6.91/0.02 = 345 s` is longer than any piano
/// string rings — but an arithmetic one. The resonator's pole radius is
/// `exp(-sigma/48000)` **in `f32`**, and every `sigma` under about `5.7e-3`
/// rounds to exactly `1.0`: a mode that is mathematically decaying but
/// numerically eternal, which accumulates in a bank that is never culled and is
/// the one failure mode a modal engine cannot recover from. At this floor the
/// radius is `1 - 4.2e-7`, six times the `f32` spacing below one.
pub const MIN_MODE_SIGMA: f32 = 0.02;

/// The stability invariant of the eigensolve, checked where the modes are made.
///
/// `A_k = i Omega_k - sigma_int I - C_k` is **dissipative**: `sigma_int > 0` by
/// the fitted decay law and `C_k = c_p J_N` has `Re c_p >= 0` because it is a
/// radiation loss, so the numerical range of `A_k` lies in `Re z <= -sigma_int`
/// and every eigenvalue must have `Re lambda <= -sigma_int < 0`. A root that
/// does not is a defect in [`block_solve`] — a sign error, a mis-shifted
/// Durand-Kerner, a `c` built from the wrong plane — not a preset a user could
/// write, because [`Preset::validate`](crate::preset::Preset::validate) floors
/// `notes.sigma0` at [`MIN_MODE_SIGMA`] and `sigma_int` is that floor times two
/// bounded ratios. So it is an `assert!` and not a clamp: the modes go into a
/// bank that no note-off can silence, and a growing one is unbounded output.
///
/// **The assert is taken on the `f64` the solver produced**, not on its `f32`
/// image. The two are not the same claim: a legal `sigma` of `1e-44` is
/// strictly positive in `f64` and rounds to `0.0` on the cast, so asserting
/// after the cast turns an `f32` rounding — which is what the
/// [`MIN_MODE_SIGMA`] floor below exists to absorb — into a panic. Rounding is
/// not a defect, and the floor is applied to the rounded value.
#[inline]
fn stable_sigma(sigma: f64, f0: f32, k: usize) -> f32 {
    assert!(
        sigma > 0.0 && sigma.is_finite(),
        "eigenmode of partial {k} at f0 {f0} Hz is not decaying: sigma = {sigma}"
    );
    (sigma as f32).max(MIN_MODE_SIGMA)
}

/// Whether a solved eigenmode sits inside the band the resonator is defined in.
///
/// This is the *same rule* [`StringParams::partial_count`] applies to the
/// undetuned series, applied to the modes that are really built: a partial is
/// worth building while it is below the Nyquist band, and past it a resonator
/// pole is an alias of something else rather than the partial it was made for.
/// The two rules differ only in where they are checked — `partial_count` reads
/// `k f0 sqrt(...)` before the solve, this reads `Im lambda / 2 pi` after it —
/// and the gap between [`MAX_PARTIAL_RATIO`] and one half is the headroom the
/// unison detuning and the bridge's frequency pull are allowed to spend.
///
/// It is deliberately **not** an assert. Nothing here is a defect: an
/// out-of-band mode is a preset whose tuning or whose fitted decay law asks for
/// a partial the sample rate cannot hold, and the model's answer to that has
/// always been to stop the series rather than to refuse the instrument. The
/// series is truncated at the first partial that fails, so a bank never holds a
/// mode above Nyquist and `PianoString::new` cannot panic on a validated preset.
#[inline]
fn mode_in_band(hz: f32) -> bool {
    hz.is_finite() && hz > 0.0 && hz < 0.5 * SAMPLE_RATE
}

impl PianoString {
    /// Builds the key's unison group by solving one `2N x 2N` eigenproblem per
    /// partial.
    ///
    /// `shaping` carries the two per-partial tables (`notes.partial_gains` and
    /// `notes.partial_sigma_scale`), and both still enter exactly where they
    /// did: the gains multiply the strike vector, the sigma scales multiply the
    /// fitted rate the whole loss budget is derived from.
    ///
    /// This is the expensive call in the instrument — about 14 ms for a bass key
    /// with 80 partials, ~1.2 s for the compass — and it is all preset-load
    /// work: nothing in `A_k` depends on velocity, so a note-on costs what it
    /// always did.
    pub fn new(params: StringParams, voicing: &Voicing, shaping: PartialShaping<'_>) -> Self {
        let partials = params.partial_count();
        let n = params.unison.clamp(1, MAX_UNISON);
        // `Re Y` in the per-partial damping: a partial that sits on a board
        // mode loses energy into the board faster than the smooth fitted decay
        // law says, and one in a trough slower. This is the half of
        // `PHYSICS.md` §4 the resonance bus cannot produce — the bus subtracts
        // each string's own contribution, so nothing in it is proportional to
        // the string's own motion and it can only ever *add* drive.
        let radiated = radiated_damping(&params, voicing, partials);
        let mut vertical = ModalBank::with_capacity(partials * n);
        let mut horizontal = ModalBank::with_capacity(partials * n);
        let mut strike_gain = Vec::with_capacity(2 * partials * n);
        let mut una_corda_gain = Vec::with_capacity(2 * partials * n);
        let direction = voicing.strike_direction;
        let tilt_capacity = if direction.is_some() {
            2 * partials * n
        } else {
            0
        };
        let mut strike_tilt = Vec::with_capacity(tilt_capacity);
        let mut una_corda_tilt = Vec::with_capacity(tilt_capacity);
        let mut damper_profile = Vec::with_capacity(partials * n);
        // The per-partial excitation, up front: the loudest of them is what
        // decides which partials are worth solving exactly (see `decay_scale`).
        let gains: Vec<f64> = (1..=partials)
            .map(|k| partial_gain(&params, voicing, &shaping, k))
            .collect();
        let loudest = gains.iter().fold(0.0f64, |m, &g| m.max(g.abs()));
        // How many partials survived the band check below. It is `partials` for
        // every preset either shipped file resembles — the whole compass builds
        // its full series, pinned by
        // `the_shipped_presets_build_every_partial_their_series_asks_for` — and
        // it is what makes the construction total rather than panicking on the
        // corner where a tuning or a fitted decay rate asks for more band than
        // the sample rate has.
        let mut built = 0usize;
        for k in 1..=partials {
            // The fitted whole-note rate of this partial, with the per-partial
            // table and the bridge's `Re Y` fluctuation on it — the same
            // `sigma_k` the free-running construction started from, before its
            // own vertical / horizontal / per-string factors, all three of which
            // the eigenproblem replaces.
            let sigma_hat =
                f64::from(params.partial_sigma(k) * shaping.sigma_scale_at(k) * radiated[k - 1]);
            let gain_k = gains[k - 1];
            let split = shaping.false_beat_at(k);
            let refine = gain_k.abs() > 10f64.powf(-50.0 / 20.0) * loudest;
            let scale = decay_scale(&params, voicing, k, sigma_hat, gain_k, split, refine);
            let solution = partial_modes(&params, voicing, k, sigma_hat, gain_k, scale, split);
            // Checked over the whole partial before any of it is pushed: the
            // `2N` modes of one partial are one entry in every array here, and a
            // series that stopped halfway through a partial would leave the two
            // banks holding different numbers of strings.
            let in_band = (0..solution.len).all(|i| {
                mode_in_band((solution.lambda[i].im / f64::from(std::f32::consts::TAU)) as f32)
            });
            if !in_band {
                break;
            }
            built = k;
            for i in 0..solution.len {
                let hz = (solution.lambda[i].im / f64::from(std::f32::consts::TAU)) as f32;
                let sigma = stable_sigma(-solution.lambda[i].re, params.f0, k);
                let g = [solution.gain[i].re as f32, solution.gain[i].im as f32];
                let bank = if i < n { &mut vertical } else { &mut horizontal };
                bank.push_mode_complex(hz, sigma, g[0], g[1]);
                strike_gain.push(g);
                una_corda_gain.push([
                    solution.una_gain[i].re as f32,
                    solution.una_gain[i].im as f32,
                ]);
                if direction.is_some() {
                    strike_tilt.push([
                        solution.gain_tilt[i].re as f32,
                        solution.gain_tilt[i].im as f32,
                    ]);
                    una_corda_tilt.push([
                        solution.una_tilt[i].re as f32,
                        solution.una_tilt[i].im as f32,
                    ]);
                }
            }
            // The damper's grip follows the same per-partial scale: it is a
            // decay rate on the same pole, and a partial the preset says is more
            // lossy than the fitted law is more lossy with the felt on it too.
            // Every mode of one partial gets the same entry — the felt is one
            // piece of cloth across the whole group.
            let extra = params.damper_sigma
                * voicing.damper_weight_at(params.partial_freq(k))
                * shaping.sigma_scale_at(k);
            for _ in 0..n {
                damper_profile.push(extra);
            }
        }
        let partials = built;
        let mut string = PianoString {
            vertical,
            horizontal,
            excitation: [0.0; BLOCK],
            damper_extra: vec![0.0; partials * n],
            damper_profile,
            strike_gain,
            una_corda_gain,
            strike_tilt,
            una_corda_tilt,
            direction,
            params,
            partials,
            strings: n,
            una_corda: false,
            velocity: u16::from(NOMINAL_STRIKE_VELOCITY),
            damper: 0.0,
        };
        // The banks were pushed with the fitted direction's gains, which is what
        // a preset without a `[strike_direction]` keeps for ever. One with a
        // direction has to start somewhere, and it starts where a string that is
        // never struck by a hammer stays: at the nominal velocity, which is what
        // the sympathetic bus drives it through.
        if string.direction.is_some() {
            string.refresh_gains();
        }
        string
    }

    pub fn params(&self) -> &StringParams {
        &self.params
    }

    /// Number of unison strings, i.e. modes per partial per polarization block.
    pub fn string_count(&self) -> usize {
        self.strings
    }

    pub fn partial_count(&self) -> usize {
        self.partials
    }

    /// The `2N` eigenmodes of partial `k` (1-based): the vertical block first.
    pub fn partial_modes(&self, k: usize) -> Vec<StringMode> {
        let base = (k - 1) * self.strings;
        (0..2 * self.strings)
            .map(|i| {
                let horizontal = i >= self.strings;
                let (bank, at) = if horizontal {
                    (&self.horizontal, base + i - self.strings)
                } else {
                    (&self.vertical, base + i)
                };
                StringMode {
                    hz: bank.mode_freq(at),
                    sigma: bank.mode_sigma(at),
                    gain_re: bank.mode_gain(at),
                    gain_im: bank.mode_gain_im(at),
                    horizontal,
                }
            })
            .collect()
    }

    /// Present state magnitudes `|s_m|` of the `2N` eigenmodes of partial `k`,
    /// in the same order [`PianoString::partial_modes`] returns them.
    ///
    /// This is what the culling floor is compared against, so it is what any
    /// claim about the floor has to be measured on.
    pub fn partial_amplitudes(&self, k: usize) -> Vec<f32> {
        let base = (k - 1) * self.strings;
        (0..2 * self.strings)
            .map(|i| {
                if i >= self.strings {
                    self.horizontal.mode_amplitude(base + i - self.strings)
                } else {
                    self.vertical.mode_amplitude(base + i)
                }
            })
            .collect()
    }

    /// Radiation-weighted centre frequency of partial `k` — where the partial
    /// sits, as opposed to where any one of its `2N` modes does.
    pub fn partial_freq(&self, k: usize) -> f32 {
        let modes = self.partial_modes(k);
        let total: f32 = modes.iter().map(|m| m.gain()).sum();
        if total <= 0.0 {
            return self.params.partial_freq(k);
        }
        modes.iter().map(|m| m.hz * m.gain()).sum::<f32>() / total
    }

    /// What partial `k` contributes to the note at the instant of the strike:
    /// `|sum_m G_m|`, which is what `notes.partial_gains` and the strike comb
    /// scale and what a spectrum of the attack measures.
    pub fn partial_gain(&self, k: usize) -> f32 {
        let modes = self.partial_modes(k);
        let re: f32 = modes.iter().map(|m| m.gain_re).sum();
        let im: f32 = modes.iter().map(|m| m.gain_im).sum();
        re.hypot(im)
    }

    /// Whether the hammer is currently missing one string of the group.
    pub fn una_corda(&self) -> bool {
        self.una_corda
    }

    /// The velocity the mode mixture currently stands at.
    pub fn strike_velocity(&self) -> u16 {
        self.velocity
    }

    /// Points the group's strike projection at the una-corda hammer, or back at
    /// the whole one. Only the mode *gains* move: the eigenmodes are a property
    /// of the strings and the bridge and do not know what struck them, which is
    /// why this is a table swap and not a rebuild.
    pub fn set_una_corda(&mut self, on: bool) {
        self.set_strike(on, self.velocity);
    }

    /// Points the strike projection at one blow: which strings the hammer
    /// reaches, and how hard it is travelling.
    ///
    /// Velocity reaches a *linear* model in exactly one way — through the
    /// direction of the excitation vector, never through the poles
    /// (`docs/history/FUNDAMENTALS.md` §7.3, §7.5 step 3) — so this is the whole of the
    /// engine's velocity dependence inside a string, and it is a remix of modes
    /// that were solved once at preset load. Bounded, allocation-free work
    /// proportional to the key's mode count, on the same path
    /// [`PianoString::set_una_corda`] has always been on.
    ///
    /// A preset with no `[voicing.strike_direction]` ignores the velocity
    /// entirely and this is the table swap it was.
    pub fn set_strike(&mut self, una_corda: bool, vel: u16) {
        if self.direction.is_none() {
            self.velocity = vel;
            if una_corda == self.una_corda {
                return;
            }
        } else if una_corda == self.una_corda && vel == self.velocity {
            return;
        }
        self.una_corda = una_corda;
        self.velocity = vel;
        self.refresh_gains();
    }

    /// Writes the mode gains the current hammer and velocity imply.
    ///
    /// `g = (base + tau tilt)` in the vertical block and `leak_scale` times the
    /// same in the horizontal one, which is [`StrikeMix`] evaluated once and
    /// applied by two fused multiply-adds per mode. Exact, not approximate: the
    /// strike projection is linear in `u` and the share tilt enters `u`
    /// linearly, so the derivative cached beside the value *is* the answer.
    fn refresh_gains(&mut self) {
        let mix = match &self.direction {
            Some(d) => StrikeMix::at(d, self.velocity),
            None => StrikeMix::NEUTRAL,
        };
        let (table, tilt) = if self.una_corda {
            (&self.una_corda_gain, &self.una_corda_tilt)
        } else {
            (&self.strike_gain, &self.strike_tilt)
        };
        let tau = mix.share_tilt;
        let scale = mix.leak_scale;
        for k in 0..self.partials {
            for j in 0..self.strings {
                let i = 2 * k * self.strings + j;
                let h = i + self.strings;
                let at = k * self.strings + j;
                // Neutral is bit-exact rather than nearly so: with no
                // `[strike_direction]` the tilt tables are empty, `tau` is 0 and
                // `scale` is 1, and this writes back exactly what was solved.
                let (mut vr, mut vi) = (table[i][0], table[i][1]);
                let (mut hr, mut hi) = (table[h][0], table[h][1]);
                if !tilt.is_empty() {
                    vr += tau * tilt[i][0];
                    vi += tau * tilt[i][1];
                    hr += tau * tilt[h][0];
                    hi += tau * tilt[h][1];
                }
                self.vertical.set_mode_gain_complex(at, vr, vi);
                self.horizontal
                    .set_mode_gain_complex(at, scale * hr, scale * hi);
            }
        }
    }

    /// Excitation buffer for the group, to be filled before `process`. Exactly
    /// `BLOCK` samples; cleared automatically after each `process`.
    ///
    /// One buffer, not one per string: the hammer's per-string shares and its
    /// timing skew are carried by the modes' complex gains.
    pub fn excitation_mut(&mut self) -> &mut [f32] {
        &mut self.excitation
    }

    /// Adds a common signal to the group's excitation — used for the
    /// sympathetic resonance drive, which reaches all strings of an undamped
    /// note.
    ///
    /// It enters on the strike vector rather than on a bridge vector of its own.
    /// The two differ only by the group's few-percent share asymmetry and the
    /// sub-sample timing skew; separating them would need a second gain table
    /// per key for a drive that is already 40 dB down.
    pub fn add_excitation_all(&mut self, signal: &[f32], gain: f32) {
        debug_assert_eq!(signal.len(), BLOCK);
        for (e, &x) in self.excitation.iter_mut().zip(signal) {
            *e += gain * x;
        }
    }

    /// Renders one block, **adding** the summed output of every eigenmode of
    /// the group into `out` (exactly `BLOCK` samples).
    pub fn process(&mut self, out: &mut [f32]) {
        debug_assert_eq!(out.len(), BLOCK);
        self.vertical.process_add(&self.excitation, out);
        self.horizontal.process_add(&self.excitation, out);
        self.excitation.fill(0.0);
    }

    /// Renders one block with the two polarization blocks kept apart: the
    /// vertical block's modes are **added** into `out_v` and the horizontal
    /// one's into `out_h` (exactly `BLOCK` samples each).
    ///
    /// It exists because the two blocks decay at very different rates, so
    /// giving them different stereo positions is what makes a note's image move
    /// as it dies (`docs/history/TUNING_REPORT.md` §5). Unlike the free-running construction
    /// this is now exactly [`PianoString::process`] split in two — same
    /// excitation, same modes, same accumulation order — so the two paths agree
    /// to the bit.
    pub fn process_split(&mut self, out_v: &mut [f32], out_h: &mut [f32]) {
        debug_assert_eq!(out_v.len(), BLOCK);
        debug_assert_eq!(out_h.len(), BLOCK);
        self.vertical.process_add(&self.excitation, out_v);
        self.horizontal.process_add(&self.excitation, out_h);
        self.excitation.fill(0.0);
    }

    /// Sets damper engagement, 0.0 = lifted, 1.0 = fully damped. Cheap enough
    /// to call every block, which is how the ~10 ms damper ramp is driven.
    pub fn set_damper(&mut self, amount: f32) {
        let amount = amount.clamp(0.0, 1.0);
        if amount == self.damper {
            return;
        }
        self.damper = amount;
        for (extra, &profile) in self.damper_extra.iter_mut().zip(&self.damper_profile) {
            *extra = amount * profile;
        }
        self.vertical.set_damping_profile(&self.damper_extra);
        self.horizontal.set_damping_profile(&self.damper_extra);
    }

    pub fn damper(&self) -> f32 {
        self.damper
    }

    pub fn energy(&self) -> f32 {
        self.vertical.energy() + self.horizontal.energy()
    }

    pub fn is_idle(&self) -> bool {
        self.vertical.is_idle() && self.horizontal.is_idle()
    }

    /// Silences the string immediately (used by `AllOff`).
    pub fn reset(&mut self) {
        self.vertical.reset_state();
        self.horizontal.reset_state();
        self.excitation.fill(0.0);
    }
}

/// The per-partial input gain `g_k`: the strike comb with its floor, the
/// contact taper, `notes.partial_gains`, and the gain staging that turns the
/// per-sample accumulation of the excitation into an integral over the hammer's
/// force pulse.
///
/// The per-string share is *not* here — it is in the strike vector, which is
/// where the eigen construction reads it.
fn partial_gain(
    params: &StringParams,
    voicing: &Voicing,
    shaping: &PartialShaping<'_>,
    k: usize,
) -> f64 {
    // Mode k's force on the bridge for a hammer impulse J is
    // `4 f0 J sin(k pi x_strike)`: the modal mass of the string is Z / (2 f0),
    // and turning the mode's displacement back into bridge force cancels the
    // wave impedance exactly.
    let output_scale = voicing.excitation_scale * params.bridge_gain * params.f0 / REFERENCE_F0;
    f64::from(
        output_scale
            * comb_magnitude(k, params.strike_position, params.comb_floor)
            * contact_taper(k, params.contact_width)
            * shaping.gain_at(k)
            / SAMPLE_RATE,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::LN_10;
    use crate::hammer::Hammer;
    use crate::preset::{
        BridgeAnchor, BridgePeak, BridgeVoicing, Preset, MAX_DETUNE_CENTS, MAX_FALSE_BEAT_DB,
        MAX_FALSE_BEAT_HZ, MAX_RADIATED_SHARE, MIN_FALSE_BEAT_DB, MIN_FALSE_BEAT_HZ,
    };

    fn preset() -> Preset {
        Preset::default()
    }

    /// Strikes a key for real — the hammer's pulse into the group's excitation —
    /// and returns `blocks` blocks of its output. Using the hammer rather than a
    /// unit impulse keeps the signal at the level the culling thresholds and the
    /// rest of the instrument are calibrated for.
    fn strike(key: u8, vel: u16, blocks: usize) -> Vec<f32> {
        let preset = preset();
        let mut string = PianoString::new(
            preset.string_params(key),
            &preset.voicing,
            PartialShaping::default(),
        );
        let mut hammer = Hammer::new(preset.hammer_params(key));
        hammer.strike_midi(vel);
        let mut out = vec![0.0f32; blocks * BLOCK];
        for chunk in out.chunks_mut(BLOCK) {
            hammer.add_pulse(string.excitation_mut(), 0, 1.0);
            hammer.advance(BLOCK);
            string.process(chunk);
        }
        out
    }

    fn rms(v: &[f32]) -> f32 {
        (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt()
    }

    /// The composite envelope of one partial, `t` seconds after the strike, as a
    /// fraction of its value at the strike. The modulus of the coherent sum, i.e.
    /// the signal, beats and all — the same quantity [`composite_t60`] solves on.
    fn envelope(modes: &[StringMode], t: f64) -> f64 {
        let (mut re, mut im) = (0.0f64, 0.0f64);
        for m in modes {
            let a = (-f64::from(m.sigma) * t).exp();
            let phase = std::f64::consts::TAU * f64::from(m.hz) * t;
            let (gr, gi) = (f64::from(m.gain_re), f64::from(m.gain_im));
            re += a * (gr * phase.cos() - gi * phase.sin());
            im += a * (gr * phase.sin() + gi * phase.cos());
        }
        let strike: f64 = {
            let (r, i) = modes.iter().fold((0.0f64, 0.0f64), |(r, i), m| {
                (r + f64::from(m.gain_re), i + f64::from(m.gain_im))
            });
            r.hypot(i)
        };
        re.hypot(im) / strike.max(f64::MIN_POSITIVE)
    }

    /// Where that envelope last passes -60 dB, found on a fine grid.
    fn measured_t60(modes: &[StringMode], t_max: f64) -> f64 {
        let steps = 20_000;
        let mut last = 0.0;
        for i in 0..=steps {
            let t = t_max * i as f64 / steps as f64;
            if envelope(modes, t) > 1e-3 {
                last = t;
            }
        }
        last
    }

    #[test]
    fn partial_layout_follows_the_inharmonicity_formula() {
        for key in [21u8, 48, 60, 84, 108] {
            let p = preset().string_params(key);
            let b = p.inharmonicity_b;
            for k in 1..=p.partial_count() {
                let want = k as f32 * p.f0 * (1.0 + b * (k * k) as f32).sqrt();
                assert!(
                    (p.partial_freq(k) - want).abs() < 1e-3 * want,
                    "key {key} partial {k}"
                );
                // Stretched, never harmonic: partial k sits above k * f0.
                assert!(p.partial_freq(k) > k as f32 * p.f0 || k == 1);
            }
            // The eighth partial of C4 must be stretched by more than 5 cents,
            // or the note is indistinguishable from a harmonic series.
            if key == 60 {
                let cents = 1200.0 * (p.partial_freq(8) / (8.0 * p.f0)).log2();
                assert!(cents > 5.0, "C4 partial 8 stretched only {cents} cents");
            }
        }
    }

    /// The two-parameter law, written out here so the tests below compare the
    /// engine's layout against the formula rather than against itself.
    fn two_parameter_freq(p: &StringParams, k: usize) -> f32 {
        let k = k as f32;
        k * p.f0 * (1.0 + p.inharmonicity_b * k * k).sqrt()
    }

    #[test]
    fn a_zero_fourth_order_coefficient_is_the_two_parameter_law_to_the_bit() {
        for key in 21..=108u8 {
            let p = preset().string_params(key);
            assert_eq!(p.inharmonicity_b4, 0.0);
            for k in 1..=p.partial_count() {
                assert_eq!(p.partial_freq(k), two_parameter_freq(&p, k), "key {key} k {k}");
            }
        }
    }

    #[test]
    fn a_fourth_order_coefficient_moves_the_high_partials_and_not_the_low_ones() {
        let base = preset().string_params(21); // A0, the note §1 measures worst
        // A wound bass string's series behaves as if `B` fell along it: fitted
        // to partials 14-26 it comes back 25-37 % below the fit to partials 1-8
        // (`docs/history/TUNING_REPORT.md` §1). `B + B4 k^2` is that falling coefficient.
        // Only part of that shape fits under one k^4 term: A0 is built with the
        // full 80 partials, and a coefficient that takes more than ~7.5 % off
        // `B` by the twentieth partial has turned the top of that series over
        // by the eightieth, which `Preset::validate` refuses. This is a third
        // of the way to the limit.
        let mut p = base;
        p.inharmonicity_b4 = -0.025 * base.inharmonicity_b / 400.0;

        let cents = |a: f32, b: f32| 1200.0 * (a / b).log2();
        // The fundamental cannot move: `B4 k^4` is 400^2 times smaller there
        // than at k = 20, which is the whole point of a second coefficient.
        assert!(
            cents(p.partial_freq(1), base.partial_freq(1)).abs() < 0.001,
            "the fundamental moved"
        );
        let radicand = 1.0 + p.inharmonicity_b * 400.0 + p.inharmonicity_b4 * 160_000.0;
        let want = 20.0 * p.f0 * radicand.sqrt();
        assert!((p.partial_freq(20) - want).abs() < 1e-4 * want);
        let moved = cents(p.partial_freq(20), base.partial_freq(20));
        assert!((-1.0..-0.1).contains(&moved), "partial 20 moved {moved} cents");
        let top = p.partial_count();
        assert!(cents(p.partial_freq(top), base.partial_freq(top)) < -20.0);
        for k in 2..=top {
            assert!(p.partial_freq(k) > p.partial_freq(k - 1), "partial {k}");
        }

        // The other sign is the short wound tenor string, whose high partials
        // come back *sharper* than one coefficient predicts (ratio 1.24-1.45).
        let mut sharp = base;
        sharp.inharmonicity_b4 = 0.025 * base.inharmonicity_b / 400.0;
        assert!(cents(sharp.partial_freq(20), base.partial_freq(20)) > 0.1);
        assert!(cents(sharp.partial_freq(top), base.partial_freq(top)) > 20.0);
        assert!(cents(sharp.partial_freq(1), base.partial_freq(1)).abs() < 0.001);
    }

    #[test]
    fn the_fourth_order_term_reaches_the_modes_and_the_partial_count() {
        let mut preset = preset();
        let i = crate::types::key_index(60).unwrap();
        // Enough curvature to pull the top of C4's series below the cap, which
        // is the one place the coefficient changes how many modes are built.
        preset.notes.inharmonicity_b4[i] = 4.0e-6;
        assert!(preset.validate().is_ok());
        let params = preset.string_params(60);
        let plain = Preset::default().string_params(60);
        assert!(params.partial_count() < plain.partial_count());

        let s = PianoString::new(params, &preset.voicing, PartialShaping::default());
        assert_eq!(s.partial_count(), params.partial_count());
        for k in 1..=s.partial_count() {
            let want = params.partial_freq(k);
            let cents = 1200.0 * (s.partial_freq(k) / want).log2();
            assert!(cents.abs() < 0.5, "partial {k} sits {cents} cents off nominal");
            assert!(want > two_parameter_freq(&params, k), "partial {k} not stretched");
        }
    }

    // ------------------------------------------- the coupled construction

    /// The whole point of the construction, stated as an assertion: the bridge
    /// splits one partial's decay rates apart, and the mode that radiates most
    /// is the one that dies first.
    ///
    /// The free-running construction it replaces gave every string of a unison
    /// the same decay rate to the last bit, which is what made its beat run at
    /// constant depth forever (`docs/history/FUNDAMENTALS.md` §3.3). Weinreich's normal modes
    /// cannot: the symmetric mode drives the bridge hard, so it loses energy
    /// fastest *and* radiates most, and the antisymmetric ones are the
    /// aftersound.
    #[test]
    fn the_bridge_splits_a_groups_decay_rates_and_the_loudest_mode_dies_first() {
        let preset = preset();
        for key in [45u8, 60, 84] {
            let s = PianoString::new(
                preset.string_params(key),
                &preset.voicing,
                PartialShaping::default(),
            );
            assert!(s.string_count() > 1, "key {key} is not a unison");
            let mut modes = s.partial_modes(1);
            assert_eq!(modes.len(), 2 * s.string_count());
            modes.sort_by(|a, b| b.gain().partial_cmp(&a.gain()).expect("finite"));

            let fastest = modes.iter().fold(0.0f32, |m, x| m.max(x.sigma));
            let slowest = modes.iter().fold(f32::MAX, |m, x| m.min(x.sigma));
            assert!(
                fastest / slowest > 2.0,
                "key {key}: decay rates {slowest}..{fastest} are not split"
            );
            // The loudest mode is the fastest: prompt sound and aftersound, out
            // of one coupling constant rather than out of `horizontal_gain_db`.
            assert_eq!(
                modes[0].sigma, fastest,
                "key {key}: the loudest mode is not the one that dies first"
            );
            // ... and the aftersound is quieter than the prompt, which is what
            // makes it an aftersound and not a second note.
            assert!(modes[1].gain() < modes[0].gain());
            // The modes are the group's, not the strings': every one of them
            // carries a phase, because the strike projects onto a
            // non-orthogonal eigenbasis.
            assert!(
                modes.iter().any(|m| m.gain_im.abs() > 1e-3 * m.gain()),
                "key {key}: every mode came back with a real gain"
            );
        }
    }

    /// The metronome, gone by construction.
    ///
    /// `voicing.horizontal_offset_hz` was a fixed number of *hertz*, so the
    /// three numbers 0.270 / 0.350 / 0.520 were beat rates of every partial of
    /// every key — the instrument-wide pulse `renders/jitter/JITTER.md` measured
    /// in every row it printed. Under the eigen construction a partial's beat
    /// rates come out of its own frequency, its own detuning and its own decay
    /// rate, so no two cells can share one. Both halves are asserted: the three
    /// shipped constants are gone, and nothing has replaced them.
    #[test]
    fn no_beat_rate_is_shared_across_the_compass() {
        let preset = preset();
        // Every (key, partial) cell's pairwise beat rates.
        let mut cells: Vec<(u8, usize, Vec<f64>)> = Vec::new();
        for key in 21..=108u8 {
            let s = PianoString::new(
                preset.string_params(key),
                &preset.voicing,
                PartialShaping::default(),
            );
            for k in 1..=s.partial_count() {
                let modes = s.partial_modes(k);
                let mut rates = Vec::new();
                for (i, a) in modes.iter().enumerate() {
                    for b in &modes[i + 1..] {
                        rates.push(f64::from((a.hz - b.hz).abs()));
                    }
                }
                cells.push((key, k, rates));
            }
        }
        assert!(cells.len() > 3000);

        // Half a millihertz is a beat period of half an hour: two rates that
        // close are the same rate by any measure a listener has.
        const SAME_HZ: f64 = 5.0e-4;
        for &shipped in &[0.270f64, 0.350, 0.520] {
            let hits = cells
                .iter()
                .filter(|(_, _, r)| r.iter().any(|x| (x - shipped).abs() < SAME_HZ))
                .count();
            assert!(
                hits * 100 < cells.len(),
                "{shipped} Hz is still a beat rate of {hits} of {} cells",
                cells.len()
            );
        }

        // ... and no rate at all is note-independent. Under the shipped
        // construction each of the three above was a rate of *every* cell of
        // every key with that many strings — 100 % — so what is asserted is that
        // no rate at all is now shared by more than a coincidence's worth of
        // them. Counted in one-millihertz bins over the band a listener reads as
        // a pulse, over cells rather than over rates, so a partial that happens
        // to have two close rates of its own cannot inflate the count.
        let mut bins: std::collections::HashMap<i64, std::collections::HashSet<(u8, usize)>> =
            std::collections::HashMap::new();
        for (key, k, rates) in &cells {
            for r in rates {
                if !(0.05..5.0).contains(r) {
                    continue;
                }
                for bin in [(r / SAME_HZ).floor() as i64, (r / SAME_HZ).ceil() as i64] {
                    bins.entry(bin).or_default().insert((*key, *k));
                }
            }
        }
        let worst = bins.values().map(|c| c.len()).max().unwrap_or(0);
        assert!(
            worst * 50 < cells.len(),
            "one beat rate is shared by {worst} of {} cells",
            cells.len()
        );
    }

    /// The polarization split is a property of the partial, not a constant.
    ///
    /// It is now `N gamma_v beta eps / 2 pi`, so it follows the partial's own
    /// decay rate and is a few *hundredths* of a hertz where the shipped field
    /// asserted 0.35 flat.
    #[test]
    fn the_polarization_split_scales_with_the_partial() {
        let preset = preset();
        let key = 60u8;
        let s = PianoString::new(
            preset.string_params(key),
            &preset.voicing,
            PartialShaping::default(),
        );
        let split = |k: usize| -> f32 {
            let modes = s.partial_modes(k);
            let n = s.string_count();
            // Mode `j` of the vertical block and mode `j` of the horizontal one
            // are the same eigenvector at two admittances, in the same order.
            (0..n)
                .map(|j| (modes[j].hz - modes[n + j].hz).abs())
                .fold(0.0f32, f32::max)
        };
        let low = split(1);
        let high = split(8);
        assert!(
            low < 0.05,
            "C4's fundamental splits by {low} Hz, against the physically \
             derivable hundredths and the shipped 0.35"
        );
        // Partial 8 decays about ten times faster than the fundamental, and the
        // split follows the loss it comes out of.
        assert!(high > 3.0 * low, "the split {low} -> {high} Hz did not scale");
    }

    /// The horizontal block's contribution to partial `k` at the strike, in dB
    /// under the mode that carries the note.
    ///
    /// This is the quantity `notes.false_beat.db` names and the quantity a
    /// measured beat depth inverts to: `|sum_h G|` against the loudest vertical
    /// mode. Not a per-mode level — a force on one string of a group standing on
    /// one bridge point reaches every mode of its block, so "the companion" is
    /// the block's coherent sum and not any one eigenvalue.
    fn companion_db(modes: &[StringMode]) -> f32 {
        let (re, im) = modes
            .iter()
            .filter(|m| m.horizontal)
            .fold((0.0f32, 0.0f32), |(r, i), m| (r + m.gain_re, i + m.gain_im));
        let reference = modes
            .iter()
            .filter(|m| !m.horizontal)
            .fold(0.0f32, |x, m| x.max(m.gain()));
        crate::types::amp_to_db(re.hypot(im) / reference)
    }

    /// A within-string split beats on the partial it names, at the level it
    /// names, and nowhere else.
    ///
    /// `docs/history/FUNDAMENTALS.md` §7.4's finding, built: the recording's mid and low
    /// partials each carry a companion 4-7 dB down and 0.7-1.5 Hz away, at a
    /// spacing uncorrelated with the partial number, which is neither the unison
    /// (7-20x too narrow, and proportional to `k`) nor the bridge's polarization
    /// split (a hundred times narrower) nor `horizontal_offset_hz` (22 dB out,
    /// and the same on every partial of every key). It is Capleton's false beat:
    /// the two planes of **one wire** at genuinely different frequencies.
    ///
    /// Three claims, and the third is the one that says it is a mechanism rather
    /// than a decoration: the level lands where the table asked, the partials
    /// beside it are untouched **to the bit**, and the split partial's *other*
    /// horizontal modes moved too — because the offset went in on the diagonal
    /// of `Omega_k` before the block was solved, so the bridge redistributed it.
    #[test]
    fn a_false_beat_splits_the_partial_it_names_and_nothing_beside_it() {
        const KEY: u8 = 60;
        const SPLIT: usize = 2;
        let preset = preset();
        let params = preset.string_params(KEY);
        let table = [FalseBeat {
            k: SPLIT as u16,
            hz: 1.0,
            db: -3.0,
        }];
        let plain = PianoString::new(params, &preset.voicing, PartialShaping::default());
        let split = PianoString::new(
            params,
            &preset.voicing,
            PartialShaping {
                false_beat: &table,
                ..PartialShaping::default()
            },
        );
        assert_eq!(split.partial_count(), plain.partial_count());

        // The level the table asked for, on the partial it named.
        let before = companion_db(&plain.partial_modes(SPLIT));
        let after = companion_db(&split.partial_modes(SPLIT));
        assert!(
            (after - (-3.0)).abs() < 0.1,
            "the companion stands at {after:+.2} dB where the table asked for -3.0 \
             (the leak alone gives {before:+.2})"
        );
        assert!(
            after > before + 1.0,
            "the split did not lift the plane it splits: {before:+.2} -> {after:+.2} dB"
        );

        // Every other partial is the instrument as it was, to the bit: a false
        // beat is a defect of one wire at one partial and nothing else in the
        // key may feel it.
        for k in 1..=plain.partial_count() {
            if k == SPLIT {
                continue;
            }
            for (a, b) in plain.partial_modes(k).iter().zip(&split.partial_modes(k)) {
                assert_eq!(a.hz, b.hz, "partial {k} moved");
                assert_eq!(a.sigma, b.sigma, "partial {k} changed decay");
                assert_eq!(a.gain_re, b.gain_re, "partial {k} changed gain");
                assert_eq!(a.gain_im, b.gain_im, "partial {k} changed gain");
            }
        }

        // ... and the mode count did not move. The split takes the horizontal
        // mode that was already there and puts it where the measurement says it
        // is; it does not add an oscillator.
        assert_eq!(
            split.partial_modes(SPLIT).len(),
            plain.partial_modes(SPLIT).len()
        );

        // The split is exactly as wide as it was asked to be — and the statistic
        // that says so is the one that only a *diagonal* offset can move. The
        // eigenvalues of a matrix sum to its trace, and `-c J_N` contributes the
        // same trace to both blocks, so
        //
        //     sum(horizontal f) - sum(vertical f)  =  offset - N (Im c_h - Im c_v)
        //
        // and the second term is the polarization split, hundredths of a hertz.
        // Whatever the solve does to the individual modes — and it moves all
        // three, and re-solves the loss budget besides — that difference has to
        // come back as the number in the table. A detune applied to the output
        // could not produce it; only a term on the diagonal of `Omega_k` can.
        let trace_split = |s: &PianoString| -> f32 {
            let modes = s.partial_modes(SPLIT);
            let sum = |horizontal: bool| -> f32 {
                modes
                    .iter()
                    .filter(|m| m.horizontal == horizontal)
                    .map(|m| m.hz)
                    .sum()
            };
            sum(true) - sum(false)
        };
        let widened = trace_split(&split) - trace_split(&plain);
        assert!(
            (widened - 1.0).abs() < 0.05,
            "the block's trace moved {widened:.4} Hz for a 1.0 Hz split"
        );

        // It went through the eigensolve rather than being pasted onto its
        // output: the modes the split did *not* name moved too, because one
        // string's diagonal entry is coupled to the others through the bridge.
        let lowest = |s: &PianoString| {
            s.partial_modes(SPLIT)
                .iter()
                .filter(|m| m.horizontal)
                .fold(f32::MAX, |x, m| x.min(m.hz))
        };
        let (a, b) = (lowest(&plain), lowest(&split));
        assert!(
            (a - b).abs() > 1.0e-3,
            "the lowest horizontal mode did not move at all ({a:.4} -> {b:.4}), so the \
             offset did not go through the solve"
        );
    }

    /// A split partial still reaches -60 dB where `notes.sigma0`/`sigma1` say it
    /// should.
    ///
    /// The composite of a split partial is a different sum of a different set of
    /// modes, so the T60 normalisation has to be re-solved *with the split in
    /// it* — which is why [`decay_scale`] takes the split and why the grid stage
    /// is no longer skipped for a single string when there is one: a false beat
    /// is a beat by construction, and the beat-free envelope is not the signal.
    #[test]
    fn a_split_partial_still_lands_on_its_own_decay_anchor() {
        let preset = preset();
        for key in [45u8, 60, 84] {
            let params = preset.string_params(key);
            for k in [1usize, 2] {
                let table = [FalseBeat {
                    k: k as u16,
                    hz: 1.2,
                    db: -4.0,
                }];
                let split = PianoString::new(
                    params,
                    &preset.voicing,
                    PartialShaping {
                        false_beat: &table,
                        ..PartialShaping::default()
                    },
                );
                let anchor = f64::from(6.91 / params.partial_sigma(k));
                let got = measured_t60(&split.partial_modes(k), 3.0 * anchor);
                assert!(
                    (got - anchor).abs() < 0.25 * anchor,
                    "key {key} partial {k} split: T60 {got:.2} s against the anchor's {anchor:.2}"
                );
            }
        }
    }

    /// Una corda changes the strike vector's *direction*, so the same eigenmodes
    /// come back in a different mixture. Nothing about the strings moves.
    #[test]
    fn una_corda_remixes_the_modes_without_rebuilding_them() {
        let preset = preset();
        let mut s = PianoString::new(
            preset.string_params(60),
            &preset.voicing,
            PartialShaping::default(),
        );
        assert_eq!(s.string_count(), 3);
        let full = s.partial_modes(1);
        s.set_una_corda(true);
        assert!(s.una_corda());
        let soft = s.partial_modes(1);
        for (a, b) in full.iter().zip(&soft) {
            assert_eq!(a.hz, b.hz, "una corda moved a mode's frequency");
            assert_eq!(a.sigma, b.sigma, "una corda moved a mode's decay");
        }
        // One string fewer under the hammer is a quieter note ...
        assert!(s.partial_gain(1) < 0.8 * {
            let mut plain = PianoString::new(
                preset.string_params(60),
                &preset.voicing,
                PartialShaping::default(),
            );
            plain.set_una_corda(false);
            plain.partial_gain(1)
        });
        // ... and putting the pedal back restores the mixture exactly.
        s.set_una_corda(false);
        for (a, b) in full.iter().zip(&s.partial_modes(1)) {
            assert_eq!(a.gain_re, b.gain_re);
            assert_eq!(a.gain_im, b.gain_im);
        }
    }

    /// Velocity moves the *mixture* and not the modes.
    ///
    /// `docs/history/FUNDAMENTALS.md` §7.3's second refutation, answered. The eigenmodes are
    /// velocity-invariant by eigenvalue and, with a velocity-independent strike
    /// vector, the mixture `V^-1 u` is velocity-invariant by linearity — which
    /// is why the engine's beat structure held to **0.006 cents and 0.054 dB**
    /// across velocities 40 / 90 / 120 against the recording's 0.787 and 1.90.
    /// The only handle that can move it is the strike vector's *direction*
    /// (§7.5 step 3), and this is that handle working: the poles do not move at
    /// all, the polarization balance moves monotonically over the whole velocity
    /// span, and the composite's beat depth moves with it.
    #[test]
    fn the_strike_direction_moves_the_mixture_monotonically_with_velocity() {
        const KEY: u8 = 60;
        let mut preset = preset();
        preset.voicing.strike_direction = Some(StrikeDirection {
            vh_db_at_pp: -6.0,
            vh_db_at_ff: 6.0,
            share_tilt: 0.15,
        });
        let mut s = PianoString::new(
            preset.string_params(KEY),
            &preset.voicing,
            preset.partial_shaping(KEY),
        );

        let nominal = s.partial_modes(1);
        let mut previous_mix = 0.0f32;
        let mut previous_depth = 0.0f64;
        let mut depths = Vec::new();
        for (step, vel) in [1u16, 40, 64, 90, 127].into_iter().enumerate() {
            s.set_strike(false, vel);
            assert_eq!(s.strike_velocity(), vel);
            let modes = s.partial_modes(1);

            // Not one pole moved. The strings and the bridge do not know how
            // hard the hammer is travelling, and this is a remix of modes solved
            // once at preset load — not a second solve.
            for (a, b) in nominal.iter().zip(&modes) {
                assert_eq!(a.hz, b.hz, "velocity {vel} moved a mode's frequency");
                assert_eq!(a.sigma, b.sigma, "velocity {vel} moved a mode's decay");
            }

            // The polarization balance, monotone over the whole span.
            let plane = |horizontal: bool| -> f32 {
                modes
                    .iter()
                    .filter(|m| m.horizontal == horizontal)
                    .map(|m| m.gain())
                    .sum()
            };
            let mix = plane(true) / plane(false);
            assert!(
                mix > previous_mix,
                "velocity {vel}: the v/h mix went {previous_mix:.4} -> {mix:.4}"
            );
            previous_mix = mix;

            // ... and the beat structure with it. The composite's envelope over
            // the seconds `renders/jitter` measures, with the note's own decay
            // divided out and the 5th and 95th percentiles taken — which is
            // `JITTER.md`'s beat depth, on a mode set instead of on a render.
            let carrier = modes
                .iter()
                .max_by(|a, b| a.gain().partial_cmp(&b.gain()).expect("finite"))
                .expect("a partial has modes")
                .sigma;
            let mut level: Vec<f64> = (0..=3_000)
                .map(|i| {
                    let t = 0.3 + 2.7 * f64::from(i) / 3_000.0;
                    20.0 * envelope(&modes, t).log10() + 20.0 * f64::from(carrier) * t / LN_10
                })
                .collect();
            level.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
            let at = |q: f64| level[((level.len() - 1) as f64 * q) as usize];
            let depth = at(0.95) - at(0.05);
            if step > 0 {
                assert!(
                    (depth - previous_depth).abs() > 0.05,
                    "velocity {vel}: the beat depth stayed at {depth:.3} dB"
                );
            }
            previous_depth = depth;
            depths.push(depth);
        }
        // Over the whole span it is a real change and not a rounding. This is
        // the statistic `renders/jitter`'s velocity rows read **0.054 dB** on
        // for the shipped engine and **1.90 dB** on for the recording; a
        // moderate direction (-6 dB to +6 dB, 0.15 of tilt) measures 0.93 dB
        // here, seventeen times the engine's and inside the band the ear reads
        // as a different blow rather than a louder one.
        let (lo, hi) = depths
            .iter()
            .fold((f64::MAX, 0.0f64), |(l, h), &d| (l.min(d), h.max(d)));
        assert!(
            hi - lo > 0.5,
            "the beat depth moved {:.2} dB over the whole velocity span {depths:?}",
            hi - lo
        );
    }

    /// A `[strike_direction]` of zeros is the fitted strike vector, at every
    /// velocity, to the bit.
    ///
    /// The neutrality contract for this mechanism, and it is stronger than the
    /// construction milestone's could be: nothing about the eigenproblem
    /// changes, so a preset that declares the section and asks it for nothing
    /// renders the same numbers as one that does not declare it at all.
    #[test]
    fn a_neutral_strike_direction_is_the_fitted_one_at_every_velocity() {
        const KEY: u8 = 60;
        let plain = preset();
        let mut voiced = preset();
        voiced.voicing.strike_direction = Some(StrikeDirection {
            vh_db_at_pp: 0.0,
            vh_db_at_ff: 0.0,
            share_tilt: 0.0,
        });
        let reference = PianoString::new(
            plain.string_params(KEY),
            &plain.voicing,
            plain.partial_shaping(KEY),
        );
        let mut s = PianoString::new(
            voiced.string_params(KEY),
            &voiced.voicing,
            voiced.partial_shaping(KEY),
        );
        for vel in [1u16, 40, 90, 127] {
            for una in [false, true] {
                s.set_strike(una, vel);
                let mut want = PianoString::new(
                    plain.string_params(KEY),
                    &plain.voicing,
                    plain.partial_shaping(KEY),
                );
                want.set_una_corda(una);
                for k in 1..=reference.partial_count() {
                    for (a, b) in want.partial_modes(k).iter().zip(&s.partial_modes(k)) {
                        assert_eq!(a.hz, b.hz);
                        assert_eq!(a.sigma, b.sigma);
                        assert_eq!(a.gain_re, b.gain_re, "velocity {vel} una {una} partial {k}");
                        assert_eq!(a.gain_im, b.gain_im, "velocity {vel} una {una} partial {k}");
                    }
                }
            }
        }
    }

    /// `Re Y` in the string's own damping: the half of `PHYSICS.md` §4 that
    /// the resonance bus cannot produce.
    ///
    /// A partial that sits on a board mode has to die faster than the smooth
    /// fitted decay law says, because that is where the board takes energy
    /// fastest. The whole loss budget of that partial is proportional to the
    /// fitted rate, so a factor of `1 + share (|P| - 1)` on the rate is that
    /// factor on every one of the partial's modes — exactly, since the T60
    /// normalisation is scale-invariant.
    #[test]
    fn a_partial_on_a_board_mode_decays_faster_than_the_fitted_law() {
        const KEY: u8 = 84;
        const PEAK_DB: f32 = 6.0;
        let base = preset();
        let f0 = base.string_params(KEY).partial_freq(1);

        let modes_at = |share: f32| {
            let mut preset = base.clone();
            preset.voicing.bridge = Some(BridgeVoicing {
                backbone: vec![
                    BridgeAnchor { hz: 20.0, gain_db: 0.0 },
                    BridgeAnchor { hz: 16_000.0, gain_db: 0.0 },
                ],
                peaks: vec![BridgePeak { hz: f0, q: 30.0, gain_db: PEAK_DB }],
                radiated_share: share,
            });
            preset.voicing.resonance_coupling = 0.0;
            assert!(preset.validate().is_ok(), "the probe preset is not legal");
            let params = preset.string_params(KEY);
            let s = PianoString::new(params, &preset.voicing, PartialShaping::default());
            s.partial_modes(1)
        };

        // A share of zero is the instrument as it was, to the last bit of every
        // pole: the factor is exactly 1.0 and nothing is recomputed.
        let plain = modes_at(0.0);
        let stock = PianoString::new(
            base.string_params(KEY),
            &base.voicing,
            PartialShaping::default(),
        );
        for (a, b) in plain.iter().zip(&stock.partial_modes(1)) {
            assert_eq!(a.sigma, b.sigma);
            assert_eq!(a.hz, b.hz);
        }

        // Half of the loss is into the board, and the board is 6 dB livelier
        // right here, so the partial must lose it 1 + 0.5 * (2 - 1) = 1.5 times
        // as fast. Every mode of it moves, and by nearly that factor rather than
        // exactly: scaling the loss budget scales the coupling too, so the ratio
        // of coupling to detuning — which is what decides how the group's modes
        // are arranged — moves with it, and the arrangement is not similar to
        // itself. Measured: 1.58 against 1.50.
        // Read off the *sum* of the partial's decay rates rather than any one
        // of them: that sum is the trace of `A_k`, so it is exactly
        // `N (2 sigma_int + gamma_v (1 + rho))` and therefore exactly
        // proportional to the fitted rate the whole budget is derived from,
        // while the individual modes are not — scaling the loss scales the
        // coupling too, and the group rearranges itself around a different ratio
        // of coupling to detuning.
        let share = 0.5f32;
        let want = 1.0 + share * (db_to_amp(PEAK_DB) - 1.0);
        let total = |m: &[StringMode]| m.iter().map(|x| x.sigma).sum::<f32>();
        let faster = total(&modes_at(share)) / total(&plain);
        assert!(
            (faster / want - 1.0).abs() < 0.05,
            "a partial on a {PEAK_DB} dB board mode decayed {faster:.4} times \
             faster, expected {want:.4}"
        );
    }

    #[test]
    fn partials_sit_where_the_formula_asks_for_them() {
        let preset = preset();
        for key in 21..=108u8 {
            let params = preset.string_params(key);
            let s = PianoString::new(params, &preset.voicing, PartialShaping::default());
            assert_eq!(s.partial_count(), params.partial_count());
            assert_eq!(
                s.partial_modes(1).len(),
                2 * params.unison.clamp(1, MAX_UNISON)
            );
            for k in 1..=s.partial_count() {
                // The radiated centre of the partial is the pitch it is heard
                // at, and the coupling must not move it: the frequency pull
                // common to the whole partial is compensated away, because
                // `notes.f0` is fitted to recordings that already contain
                // whatever pull the real bridge applies.
                let cents = 1200.0 * (s.partial_freq(k) / params.partial_freq(k)).log2();
                assert!(
                    cents.abs() < 0.5,
                    "key {key} partial {k} sits {cents:+.3} cents off nominal"
                );
            }
        }
    }

    /// The whole-note T60 anchors of `notes.sigma0` / `sigma1` survive the
    /// change of construction.
    ///
    /// This is the statement `Voicing::vertical_decay_factor` used to make in
    /// closed form and [`decay_scale`] now makes by solving: the composite of a
    /// partial's `2N` coupled modes reaches -60 dB at `6.91 / sigma_k`. The
    /// tolerance is what a beat costs — the crossing is the *last* one, so a
    /// trough that dips under the threshold and comes back moves it by a beat
    /// period, and 800 grid points resolve that to a few percent.
    #[test]
    fn fundamental_t60_matches_the_spec_anchors() {
        let preset = preset();
        // The contract's 5 %, and the one key of the four that needs the beat
        // period beside it. C8's anchor is 0.6 s and its six modes are 8.5 Hz
        // apart, so one trough of its own beat is 0.118 s — a fifth of the whole
        // note — and the crossing has nowhere else to land. The other three land
        // to a tenth of a percent, which is the solve and not a tolerance.
        for (key, want, tol) in [
            (21u8, 25.0f64, 0.002f64),
            (60, 12.0, 0.002),
            (84, 3.0, 0.002),
            (108, 0.6, 0.05),
        ] {
            let params = preset.string_params(key);
            let s = PianoString::new(params, &preset.voicing, PartialShaping::default());
            let anchor = 6.91 / f64::from(params.partial_sigma(1));
            assert!((anchor / want - 1.0).abs() < 0.01, "key {key}: the anchor moved");
            let modes = s.partial_modes(1);
            let got = measured_t60(&modes, 3.0 * want);
            let mut widest = 0.0f64;
            for (i, a) in modes.iter().enumerate() {
                for b in &modes[i + 1..] {
                    widest = widest.max(f64::from((a.hz - b.hz).abs()));
                }
            }
            let beat = if widest > 0.0 { 1.0 / widest } else { 0.0 };
            println!(
                "key {key}: T60 {got:.4} s against the {want} s anchor ({:+.2} %), \
                 beat period {beat:.3} s",
                100.0 * (got / want - 1.0)
            );
            assert!(
                (got - want).abs() < tol * want + if tol > 0.01 { beat } else { 0.0 },
                "key {key}: T60 {got:.3} s, expected {want} s"
            );
        }
    }

    /// ... and across the whole compass, on the partials a listener follows.
    ///
    /// **This is the equivalence contract's decay half, and the contract is
    /// "within 5 %".** It is met outright by the median and missed by the tail,
    /// and the reason is a property of the statistic rather than of the decay
    /// law: [`composite_t60`] is the *last* time a beating envelope passes
    /// -60 dB, so a trough that dips under the threshold and comes back moves
    /// the answer by a whole beat period. The 5 % clause is therefore asserted
    /// as four separate statements, none of which is the loose worst case alone:
    ///
    /// * the **median** is under 1 % — the solve lands, and what is left is not
    ///   a bias;
    /// * the **p90** is under 8.5 %, measured at 8.0;
    /// * the **worst** is under 25 %, measured at 22.9 at G7's fundamental;
    /// * and — the one that says what the other three mean — **at most one cell
    ///   in twenty misses its anchor by more than 5 % for a reason that is not
    ///   the crossing's own ambiguity**. For each cell the envelope's *first*
    ///   and *last* passes through -60 dB bracket every answer the question
    ///   "when did this partial reach -60 dB" has; a cell whose anchor lies
    ///   inside that bracket has not missed at all. **12 of 302 do not**, and
    ///   those twelve are the bisection landing on the far side of a tread —
    ///   `decay_scale` returning its best visited point instead of its last
    ///   takes them to 2 and the worst case to 18.4 %, which is measured in
    ///   `DECISIONS.md` 259 and not shipped, because it moves the render enough
    ///   to take two of the tuner's calibration round trips red.
    #[test]
    fn every_partials_t60_lands_on_its_own_anchor() {
        let preset = preset();
        let mut errors: Vec<f64> = Vec::new();
        let mut outside = 0usize;
        let mut worst_outside = (0.0f64, 0u8, 0usize);
        for key in 21..=108u8 {
            let params = preset.string_params(key);
            let s = PianoString::new(params, &preset.voicing, PartialShaping::default());
            for k in 1..=4.min(s.partial_count()) {
                let anchor = 6.91 / f64::from(params.partial_sigma(k));
                if anchor < 0.5 {
                    continue;
                }
                let modes = s.partial_modes(k);
                let span = 3.0 * anchor;
                let got = measured_t60(&modes, span);
                errors.push((got / anchor - 1.0).abs());
                // Every answer the question has: from the first time the
                // envelope goes under -60 dB to the last time it is over.
                let (first, last) = crossing_window(&modes, span);
                let margin = if anchor < first {
                    (first - anchor) / anchor
                } else if anchor > last {
                    (anchor - last) / anchor
                } else {
                    0.0
                };
                if margin > 0.05 {
                    outside += 1;
                    if margin > worst_outside.0 {
                        worst_outside = (margin, key, k);
                    }
                }
            }
        }
        assert!(errors.len() > 250);
        errors.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        let median = errors[errors.len() / 2];
        let p90 = errors[(errors.len() * 9) / 10];
        let worst = errors[errors.len() - 1];
        println!(
            "T60 against anchor over {} cells: median {:.1} %, p90 {:.1} %, worst {:.1} %; \
             {outside} cells miss by more than 5 % *and* land outside their own crossing \
             window, worst key {} k {} at {:.1} %",
            errors.len(),
            100.0 * median,
            100.0 * p90,
            100.0 * worst,
            worst_outside.1,
            worst_outside.2,
            100.0 * worst_outside.0
        );
        assert!(median < 0.01, "median T60 error {median}");
        assert!(p90 < 0.085, "p90 T60 error {p90}");
        assert!(worst < 0.25, "worst T60 error {worst}");
        assert!(
            outside * 20 <= errors.len(),
            "{outside} of {} cells miss their anchor by more than 5 % for a reason that is \
             not the crossing's own ambiguity",
            errors.len()
        );
    }

    /// The first and last times a partial's composite passes -60 dB.
    ///
    /// Between them the beating envelope is crossing back and forth, so every
    /// instant in the interval is a defensible answer to "when did this partial
    /// reach -60 dB" and the difference between them is the tread
    /// [`composite_t60`] is quantised by.
    fn crossing_window(modes: &[StringMode], t_max: f64) -> (f64, f64) {
        let steps = 20_000;
        let (mut first, mut last) = (t_max, 0.0);
        let mut seen_below = false;
        for i in 0..=steps {
            let t = t_max * i as f64 / steps as f64;
            if envelope(modes, t) > 1e-3 {
                last = t;
            } else if !seen_below {
                seen_below = true;
                first = t;
            }
        }
        (first, last)
    }

    #[test]
    fn a_rendered_note_decays_at_the_designed_rate() {
        // C6, whose anchor is 3 s, so the whole decay fits in a short render.
        let key = 84u8;
        let preset = preset();
        let params = preset.string_params(key);
        let anchor = 6.91 / params.partial_sigma(1);
        let blocks = (1.2 * anchor * SAMPLE_RATE / BLOCK as f32) as usize;
        let y = strike(key, 100, blocks);
        let win = (0.05 * SAMPLE_RATE) as usize;
        let env: Vec<f32> = y
            .chunks(win)
            .map(|c| c.iter().fold(0.0f32, |m, v| m.max(v.abs())))
            .collect();
        let peak = env.iter().fold(0.0f32, |m, &v| m.max(v));
        let mut last = 0.0f32;
        for (i, &v) in env.iter().enumerate() {
            if v > 1e-3 * peak {
                last = (i + 1) as f32 * win as f32 / SAMPLE_RATE;
            }
        }
        // A whole rendered note, culling and all, against the anchor its
        // `notes.sigma0` was fitted to.
        assert!(
            (last / anchor - 1.0).abs() < 0.3,
            "key {key}: the render died at {last} s against the {anchor} s anchor"
        );
    }

    // ------------------------------------------- the per-partial tables

    /// `notes.partial_gains` is a gain on one partial's excitation and on
    /// nothing else: doubling entry `k` doubles what partial `k` contributes to
    /// the note, every mode of it, and leaves every other partial where it was.
    #[test]
    fn a_doubled_partial_gain_doubles_that_partials_output_amplitude() {
        const KEY: u8 = 60;
        const K: usize = 5;
        let preset = preset();
        let params = preset.string_params(KEY);
        let mut gains = vec![1.0f32; params.partial_count()];
        gains[K - 1] = 2.0;
        let shaping = PartialShaping {
            gains: &gains,
            sigma_scale: &[],
            false_beat: &[],
        };

        let plain = PianoString::new(params, &preset.voicing, PartialShaping::default());
        let loud = PianoString::new(params, &preset.voicing, shaping);
        for k in 1..=plain.partial_count() {
            let want = if k == K { 2.0 } else { 1.0 };
            let got = loud.partial_gain(k) / plain.partial_gain(k);
            assert!(
                (got / want - 1.0).abs() < 1e-4,
                "partial {k} moved by {got}, expected {want}"
            );
            // The excitation table is not a damping table.
            for (a, b) in plain.partial_modes(k).iter().zip(&loud.partial_modes(k)) {
                assert_eq!(a.sigma, b.sigma, "partial {k} changed decay");
                assert_eq!(a.hz, b.hz, "partial {k} changed frequency");
            }
        }
    }

    /// `notes.partial_sigma_scale` reaches every mode of the partial and the
    /// damper profile with it.
    #[test]
    fn a_partial_sigma_scale_changes_that_partials_decay_and_its_damper() {
        const KEY: u8 = 84;
        let preset = preset();
        let params = preset.string_params(KEY);
        // Half the fitted rate on the fundamental and twice it on the second
        // partial: both directions.
        let mut scale = vec![1.0f32; params.partial_count()];
        scale[0] = 0.5;
        scale[1] = 2.0;
        let shaping = PartialShaping {
            gains: &[],
            sigma_scale: &scale,
            false_beat: &[],
        };
        let plain = PianoString::new(params, &preset.voicing, PartialShaping::default());
        let scaled = PianoString::new(params, &preset.voicing, shaping);
        let n = plain.string_count();
        for (k, &want) in scale.iter().enumerate().take(plain.partial_count()) {
            // What the table sets is the partial's *whole-note* rate, which is
            // the quantity the T60 normalisation solves against, so that is what
            // is asserted rather than each mode's own sigma: scaling the loss
            // budget scales the coupling with it, and the group's modes are not
            // similar to themselves under that.
            let anchor = 6.91 / f64::from(params.partial_sigma(k + 1));
            let plain_t60 = measured_t60(&plain.partial_modes(k + 1), 3.0 * anchor);
            let scaled_t60 =
                measured_t60(&scaled.partial_modes(k + 1), 3.0 * anchor / f64::from(want));
            // A third, not a tenth: the -60 dB crossing of a beating composite
            // is the *last* one, so a trough that dips under the threshold and
            // comes back moves it by a whole beat period, and a group whose
            // damping has just been halved beats at a different rate.
            assert!(
                (scaled_t60 * f64::from(want) / plain_t60 - 1.0).abs() < 0.3,
                "partial {}: T60 {plain_t60} -> {scaled_t60}, expected a factor of {}",
                k + 1,
                1.0 / want
            );
            // The partial's whole loss budget — the trace of `A_k`, i.e. the
            // sum of its modes' decay rates — moves by the table's entry, and
            // every partial the table leaves at 1 is untouched to the bit.
            let total = |s: &PianoString| {
                s.partial_modes(k + 1).iter().map(|m| m.sigma).sum::<f32>()
            };
            if want == 1.0 {
                assert_eq!(total(&scaled), total(&plain), "partial {} moved", k + 1);
            } else {
                let moved = total(&scaled) / total(&plain);
                // To a tenth, not exactly: the normalisation re-solves for a
                // group whose ratio of coupling to detuning has moved with the
                // budget, so the modes are not similar to themselves.
                assert!(
                    (moved / want - 1.0).abs() < 0.16,
                    "partial {}: the loss budget moved by {moved}, expected {want}",
                    k + 1
                );
            }
            // The damper is a decay rate on the same pole and follows too.
            assert!(
                (scaled.damper_profile[k * n] / (plain.damper_profile[k * n] * want) - 1.0).abs()
                    < 1e-6,
                "damper profile of partial {}",
                k + 1
            );
        }
    }

    /// `notes.comb_floor` lifts the partials the strike comb nulls and leaves
    /// every other partial where it was — the whole difference between a hammer
    /// with width and a hammer that is a point.
    #[test]
    fn the_comb_floor_lifts_the_null_partials_and_leaves_the_others_alone() {
        const KEY: u8 = 45; // A2
        const FLOOR: f32 = 0.05;
        let preset = preset();
        let params = preset.string_params(KEY);
        let comb = |k: usize| (k as f32 * std::f32::consts::PI * params.strike_position).sin();
        let null = (1..=params.partial_count())
            .min_by(|&a, &b| comb(a).abs().partial_cmp(&comb(b).abs()).unwrap())
            .expect("the key has partials");
        assert!(comb(null).abs() < 0.02, "A2's comb has no null to fill");

        let mut floored_params = params;
        floored_params.comb_floor = FLOOR;
        let plain = PianoString::new(params, &preset.voicing, PartialShaping::default());
        let floored = PianoString::new(floored_params, &preset.voicing, PartialShaping::default());

        // A zero floor is the bare comb to the last bit.
        let mut zero_params = params;
        zero_params.comb_floor = 0.0;
        let zero = PianoString::new(zero_params, &preset.voicing, PartialShaping::default());
        for k in 1..=plain.partial_count() {
            for (a, b) in zero.partial_modes(k).iter().zip(&plain.partial_modes(k)) {
                assert_eq!(a.gain_re, b.gain_re);
                assert_eq!(a.gain_im, b.gain_im);
            }
        }

        for k in 1..=plain.partial_count() {
            let c = comb(k);
            let want = (c * c + FLOOR * FLOOR).sqrt() / c.abs();
            let ratio = floored.partial_gain(k) / plain.partial_gain(k);
            assert!(
                (ratio / want - 1.0).abs() < 1e-3,
                "partial {k} moved by {ratio}, expected {want}"
            );
            let db = 20.0 * ratio.log10();
            if k == null {
                assert!(db > 12.0, "the null at {k} only rose {db:.1} dB");
            } else if c.abs() > 0.3 {
                assert!(db < 0.1, "partial {k} rose {db:.3} dB and is not a null");
                if c.abs() > 0.9 {
                    assert!(db < 0.02, "partial {k} rose {db:.3} dB at the comb's crest");
                }
            }
        }

        let depth = |s: &PianoString, k: usize| 20.0 * (s.partial_gain(k) / s.partial_gain(1)).log10();
        assert!(depth(&plain, null) < -30.0, "the control's null is not deep");
        assert!(
            depth(&floored, null) > -30.0,
            "the floor left the null at {:.1} dB",
            depth(&floored, null)
        );
    }

    /// A key whose row runs out — or has none at all — is the string the engine
    /// built before either table existed, to the bit.
    #[test]
    fn a_short_or_missing_per_partial_row_is_the_unshaped_string() {
        let preset = preset();
        for key in [21u8, 45, 60, 96] {
            let params = preset.string_params(key);
            let plain = PianoString::new(params, &preset.voicing, PartialShaping::default());
            let short = PianoString::new(
                params,
                &preset.voicing,
                PartialShaping {
                    gains: &[1.0, 1.0, 1.0],
                    sigma_scale: &[1.0],
                    false_beat: &[],
                },
            );
            for k in 1..=plain.partial_count() {
                for (a, b) in short.partial_modes(k).iter().zip(&plain.partial_modes(k)) {
                    assert_eq!(a.gain_re, b.gain_re, "key {key} partial {k}");
                    assert_eq!(a.gain_im, b.gain_im);
                    assert_eq!(a.sigma, b.sigma);
                    assert_eq!(a.hz, b.hz);
                }
            }
            assert_eq!(short.damper_profile, plain.damper_profile);
        }
    }

    #[test]
    fn contact_width_tapers_the_top_of_the_comb_monotonically() {
        // Zero width is exactly the point force, so a preset that does not
        // mention the field builds exactly the comb it always did.
        for k in 1..=MAX_PARTIALS {
            assert_eq!(contact_taper(k, 0.0), 1.0);
        }
        let widths = [0.0f32, 0.005, 0.01, 0.02, 0.03, 0.04, MAX_CONTACT_WIDTH];
        for k in [1usize, 4, 12, 30, 60] {
            for w in widths.windows(2) {
                let (wide, narrow) = (contact_taper(k, w[1]), contact_taper(k, w[0]));
                assert!(wide <= narrow, "partial {k}: {narrow} rose to {wide}");
                if narrow > 0.0 {
                    assert!(wide < narrow, "partial {k} did not move between {w:?}");
                }
            }
            for &w in &widths[1..] {
                assert!(contact_taper(k + 1, w) <= contact_taper(k, w));
            }
        }
        // Past its first null the taper stays at zero instead of turning back
        // up: a contact patch that spans a whole half-period cannot drive that
        // partial, and cannot start driving it again by getting wider.
        assert_eq!(contact_taper(40, MAX_CONTACT_WIDTH), 0.0); // k w = 2
        assert_eq!(contact_taper(70, MAX_CONTACT_WIDTH), 0.0); // k w = 3.5
        assert!(contact_taper(20, MAX_CONTACT_WIDTH) < 1e-6); // k w = 1, the null

        let width = 0.015;
        let mut preset = preset();
        for w in &mut preset.notes.contact_width {
            *w = width;
        }
        assert!(preset.validate().is_ok());
        let key = 96u8; // C7, where the contact is the largest fraction of a string
        let stock = Preset::default();
        let plain = PianoString::new(
            stock.string_params(key),
            &stock.voicing,
            PartialShaping::default(),
        );
        let tapered = PianoString::new(
            preset.string_params(key),
            &preset.voicing,
            PartialShaping::default(),
        );
        for k in 1..=tapered.partial_count() {
            let want = plain.partial_gain(k) * contact_taper(k, width);
            let got = tapered.partial_gain(k);
            assert!(
                (got - want).abs() <= 1e-4 * want.abs().max(1e-20),
                "partial {k} gain {got}, expected {want}"
            );
        }
    }

    #[test]
    fn high_partials_decay_faster_than_the_fundamental() {
        let p = preset().string_params(60);
        let mut previous = 0.0;
        for k in 1..=p.partial_count() {
            let sigma = p.partial_sigma(k);
            assert!(sigma > previous, "partial {k} decays no faster than {}", k - 1);
            previous = sigma;
        }
    }

    #[test]
    fn partials_stay_below_nyquist_and_the_cap() {
        for key in 21..=108u8 {
            let p = preset().string_params(key);
            let n = p.partial_count();
            assert!((1..=MAX_PARTIALS).contains(&n));
            assert!(p.partial_freq(n) < MAX_PARTIAL_RATIO * SAMPLE_RATE);
        }
    }

    #[test]
    fn strike_position_nulls_the_partial_it_sits_on() {
        // x_strike ~ 1/8 in the bass, so partial 8 must be far weaker than its
        // neighbours.
        let preset = preset();
        let params = preset.string_params(21);
        let s = PianoString::new(params, &preset.voicing, PartialShaping::default());
        let gain = |k: usize| {
            (k as f32 * std::f32::consts::PI * params.strike_position)
                .sin()
                .abs()
        };
        let node = (1.0 / params.strike_position).round() as usize;
        assert!(gain(node) < 0.3, "partial {node} is not nulled: {}", gain(node));
        assert!(gain(node) < 0.4 * gain(node - 1));
        assert!((1..node).any(|k| gain(k) > 0.95));
        assert!(s.partial_count() > node);
        // ... and the built partial follows the comb it was built from.
        assert!(s.partial_gain(node) < 0.4 * s.partial_gain(node - 1));
    }

    #[test]
    fn a_unison_group_beats() {
        // The coupled group still moves its own envelope — much less than the
        // free-running one did, which is the point, but not to a straight line.
        let y = strike(60, 100, (6.0 * SAMPLE_RATE / BLOCK as f32) as usize);
        let win = 4800;
        let env: Vec<f32> = y
            .chunks(win)
            .map(|c| c.iter().fold(0.0f32, |m, v| m.max(v.abs())))
            .collect();
        assert!(
            env.windows(2).any(|w| w[1] > w[0] * 1.001),
            "envelope decays monotonically: {env:?}"
        );
    }

    #[test]
    fn the_damper_kills_the_note() {
        let preset = preset();
        let mut s = PianoString::new(
            preset.string_params(60),
            &preset.voicing,
            PartialShaping::default(),
        );
        let mut hammer = Hammer::new(preset.hammer_params(60));
        hammer.strike_midi(100);
        let mut warm = vec![0.0f32; 40 * BLOCK];
        for chunk in warm.chunks_mut(BLOCK) {
            hammer.add_pulse(s.excitation_mut(), 0, 1.0);
            hammer.advance(BLOCK);
            s.process(chunk);
        }
        let loud = rms(&warm);
        s.set_damper(1.0);
        let blocks = (0.5 * SAMPLE_RATE / BLOCK as f32) as usize;
        let mut out = vec![0.0f32; blocks * BLOCK];
        for chunk in out.chunks_mut(BLOCK) {
            s.process(chunk);
        }
        let quiet = rms(&out[out.len() - BLOCK..]);
        assert!(quiet < loud * 0.01, "damped {quiet} vs struck {loud}");
    }

    #[test]
    fn the_damper_grips_low_partials_hardest() {
        let preset = preset();
        let p = preset.string_params(48);
        let s = PianoString::new(p, &preset.voicing, PartialShaping::default());
        let n = s.string_count();
        let first = s.damper_profile[0];
        let last = s.damper_profile[(s.partials - 1) * n];
        assert!(first > last * 2.0, "damper profile {first} .. {last}");
        assert!((first - p.damper_sigma).abs() < 0.01 * p.damper_sigma);
        // The felt is one piece of cloth across the group: every mode of a
        // partial gets the same grip.
        for k in 0..s.partials {
            for j in 1..n {
                assert_eq!(s.damper_profile[k * n + j], s.damper_profile[k * n]);
            }
        }
    }

    /// The split path is a different way of *reading* the group, not a different
    /// group — and now exactly so, to the bit, because the two polarization
    /// blocks are two banks driven by one buffer.
    #[test]
    fn splitting_the_polarizations_renders_the_same_string() {
        let (key, preset) = (60u8, preset());
        let mut summed = PianoString::new(
            preset.string_params(key),
            &preset.voicing,
            PartialShaping::default(),
        );
        let mut split = PianoString::new(
            preset.string_params(key),
            &preset.voicing,
            PartialShaping::default(),
        );
        let mut hammer = Hammer::new(preset.hammer_params(key));
        let mut hammer_split = Hammer::new(preset.hammer_params(key));
        hammer.strike_midi(100);
        hammer_split.strike_midi(100);

        let (mut a, mut v, mut h) = ([0.0f32; BLOCK], [0.0f32; BLOCK], [0.0f32; BLOCK]);
        let mut peak = 0.0f32;
        let mut worst = 0.0f32;
        for _ in 0..200 {
            hammer.add_pulse(summed.excitation_mut(), 0, 1.0);
            hammer_split.add_pulse(split.excitation_mut(), 0, 1.0);
            hammer.advance(BLOCK);
            hammer_split.advance(BLOCK);
            a.fill(0.0);
            v.fill(0.0);
            h.fill(0.0);
            summed.process(&mut a);
            split.process_split(&mut v, &mut h);
            for i in 0..BLOCK {
                peak = peak.max(a[i].abs());
                // Not bit-exact and cannot be: the two paths add the same
                // numbers in a different order. The modes are identical.
                worst = worst.max((a[i] - (v[i] + h[i])).abs());
            }
        }
        assert!(peak > 0.0);
        // Against the note's own peak, not against wherever the running maximum
        // had got to: three hundred modes cancelling into a small sample give a
        // large *relative* error on a number that is not there.
        assert!(
            worst <= 1e-6 * peak,
            "the two paths differ by {worst:e} on a peak of {peak:e}"
        );
        assert!((split.energy() / summed.energy() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn excitation_produces_output_and_is_consumed() {
        let preset = preset();
        let mut s = PianoString::new(
            preset.string_params(60),
            &preset.voicing,
            PartialShaping::default(),
        );
        s.excitation_mut()[0] = 1.0;
        let mut out = [0.0f32; BLOCK];
        s.process(&mut out);
        assert!(out.iter().any(|v| v.abs() > 0.0));
        assert!(s.excitation_mut().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn every_key_is_stable_and_finite() {
        for key in 21..=108u8 {
            let y = strike(key, 127, 200);
            assert!(y.iter().all(|v| v.is_finite()), "key {key} went non-finite");
            let head = rms(&y[..BLOCK]);
            let tail = rms(&y[y.len() - BLOCK..]);
            assert!(tail <= head, "key {key} gained energy: {head} -> {tail}");
        }
    }

    /// The eigensolve, against the closed form it has one for: two strings, no
    /// reactive part, so the `2 x 2` block is `[[d1 - c, -c], [-c, d2 - c]]` and
    /// `lambda± = d̄ - c ± sqrt(c² + (Δd/2)²)`.
    #[test]
    fn the_block_solver_agrees_with_the_two_by_two_closed_form() {
        let (w1, w2) = (2000.0f64, 2003.0);
        let sigma_int = 0.7;
        let c = Cx::new(0.9, 0.0);
        let mut out = [BlockMode::default(); MAX_UNISON];
        let found = block_solve(&[w1, w2], sigma_int, c, &mut out);
        assert_eq!(found, 2);
        let mean = Cx::new(-sigma_int, 0.5 * (w1 + w2));
        let half = Cx::new(0.0, 0.5 * (w1 - w2));
        // sqrt(c^2 + half^2), both real/imaginary here so it stays elementary.
        let disc = c * c + half * half;
        let r = disc.norm().sqrt();
        let theta = 0.5 * disc.im.atan2(disc.re);
        let root = Cx::from_polar(r, theta);
        let want = [mean - c + root, mean - c - root];
        for w in want {
            assert!(
                out[..found].iter().any(|m| (m.lambda - w).norm() < 1e-9),
                "{:?} is not among {:?}",
                w,
                &out[..found]
            );
        }
    }

    // ------------------------------------------------- the stability fuzz

    /// Xorshift64*, so the fuzz is reproducible and carries no dependency.
    fn fuzz_rng(seed: u64) -> impl FnMut() -> f64 {
        let mut state = seed | 1;
        move || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let x = state.wrapping_mul(0x2545_f491_4f6c_dd1d);
            (x >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// **The one failure a modal engine cannot recover from.**
    ///
    /// Every mode of every partial is a resonator whose pole is
    /// `r e^{i w}` with `r = exp(-sigma / 48000)`, stepped forever: nothing
    /// culls a bank that is still ringing, and no note-off can silence a pole
    /// at or outside the unit circle. Under the free-running construction the
    /// decay rates came straight off a fitted, positive law and could not be
    /// anything else. Under the coupled one they are the **roots of a
    /// polynomial** that `block_solve` finds by Durand-Kerner, from a matrix
    /// built out of six preset fields — so "every root decays" stopped being an
    /// assumption of the construction and became a property to be tested.
    ///
    /// Two conditions, and they are not the same condition at 48 kHz:
    ///
    /// * `sigma > 0` — the mathematical one. `A_k = i Omega_k - sigma_int I -
    ///   C_k` is dissipative, so it holds for every legal preset or
    ///   [`block_solve`] is wrong.
    /// * `r < 1` strictly — the arithmetic one. `f32` rounds `exp(-sigma/48000)`
    ///   to exactly one for every `sigma` under about `5.7e-3`, which is a mode
    ///   that is mathematically decaying and numerically eternal.
    ///   [`MIN_MODE_SIGMA`] is the floor that separates them.
    ///
    /// The fuzz perturbs every field that enters the eigenproblem — the two
    /// bridge anisotropies, the horizontal leak, the radiated share, the
    /// tuning, the per-string shares, the per-partial sigma scale, and a false
    /// beat at both rails — across the whole compass, and only ever feeds
    /// `PianoString::new` presets that `validate` has passed, because an
    /// invalid preset never reaches it in the instrument either.
    #[test]
    fn every_eigenmode_of_a_fuzzed_preset_is_strictly_inside_the_unit_circle() {
        let mut rng = fuzz_rng(0x5eed_1234_9abc_def0);
        let mut checked = 0usize;
        for trial in 0..12 {
            let mut preset = preset();
            let pick = |lo: f64, hi: f64, u: f64| (lo + (hi - lo) * u) as f32;
            preset.voicing.horizontal_gain_db = pick(-60.0, 0.0, rng());
            preset.voicing.horizontal_decay_ratio = pick(0.01, 1.5, rng());
            preset.voicing.excitation_scale = pick(0.1, 4.0, rng());
            if let Some(bridge) = preset.voicing.bridge.as_mut() {
                bridge.radiated_share = pick(0.0, MAX_RADIATED_SHARE.into(), rng());
            }
            for layout in &mut preset.voicing.unison_layout {
                let n = layout.share.len();
                for j in 0..n {
                    layout.share[j] = pick(0.2, 1.8, rng());
                    layout.detune[j] = pick(-1.0, 1.0, rng());
                }
                // The schema requires the shares to mean 1 and the detunes to
                // be ordered; a fuzz that cannot pass `validate` tests nothing.
                let mean: f32 = layout.share.iter().sum::<f32>() / n as f32;
                for j in 0..n {
                    layout.share[j] /= mean;
                }
                layout.detune.sort_by(f32::total_cmp);
            }
            // The three tables that decide where a mode *lands* rather than how
            // it is mixed, fuzzed over their whole legal range and not over the
            // neighbourhood of the fitted one. A spread of six cents and the
            // shipped decay law cannot reach either edge of the band, which is
            // how three panics in `PianoString::new` survived this fuzz
            // (`DECISIONS.md` 257): the tuning multiplies the top of the series
            // and the loss budget sets how far the bridge pulls it.
            for d in &mut preset.notes.detune_cents {
                *d = pick(0.0, f64::from(MAX_DETUNE_CENTS), rng());
            }
            let tuning = pick(0.25, 1.6, rng());
            for f0 in &mut preset.notes.f0_hz {
                *f0 *= tuning;
            }
            for s in &mut preset.notes.sigma0 {
                *s = pick(f64::from(MIN_MODE_SIGMA), 8.0, rng());
            }
            for s in &mut preset.notes.sigma1 {
                *s = pick(0.0, 4.0, rng());
            }
            // A false beat at both rails on every key: the diagonal term the
            // solver sees furthest from the vertical block's.
            preset.notes.false_beat = (0..crate::types::NUM_KEYS)
                .map(|_| {
                    vec![
                        FalseBeat {
                            k: 1,
                            hz: MAX_FALSE_BEAT_HZ,
                            db: MAX_FALSE_BEAT_DB,
                        },
                        FalseBeat {
                            k: 2,
                            hz: MIN_FALSE_BEAT_HZ,
                            db: MIN_FALSE_BEAT_DB,
                        },
                    ]
                })
                .collect();
            preset
                .validate()
                .unwrap_or_else(|e| panic!("fuzz trial {trial} built an illegal preset: {e:?}"));

            // Six keys per trial, walking the compass over the twelve trials:
            // bass, tenor, midrange and treble every time, and every unison
            // size somewhere in the sweep.
            for step in 0..6 {
                let key = 21 + ((trial * 7 + step * 15) % 88) as u8;
                let string = PianoString::new(
                    preset.string_params(key),
                    &preset.voicing,
                    preset.partial_shaping(key),
                );
                // A validated preset is an instrument at every key: the band
                // truncation is a backstop for arithmetic, not a way for a
                // legal tuning to lose its note.
                assert!(
                    string.partial_count() >= 1,
                    "trial {trial} key {key} came out with no partials"
                );
                for bank in [&string.vertical, &string.horizontal] {
                    for i in 0..bank.len() {
                        let sigma = bank.mode_sigma(i);
                        let r = bank.pole_radius(i);
                        let hz = bank.mode_freq(i);
                        assert!(
                            sigma >= MIN_MODE_SIGMA && sigma.is_finite(),
                            "trial {trial} key {key} mode {i}: sigma {sigma}"
                        );
                        assert!(
                            r < 1.0 && r > 0.0,
                            "trial {trial} key {key} mode {i}: pole radius {r} (sigma {sigma})"
                        );
                        assert!(
                            mode_in_band(hz),
                            "trial {trial} key {key} mode {i}: {hz} Hz is outside the band"
                        );
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 10_000, "the fuzz only reached {checked} modes");
    }
}
