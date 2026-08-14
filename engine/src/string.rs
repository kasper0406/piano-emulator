//! Piano string: partial layout, frequency-dependent damping, two
//! polarizations, and the unison group of 1-3 slightly detuned strings.
//!
//! A key's sound is a group of 1-3 strings; each string is two modal banks, one
//! per polarization. The vertical polarization couples strongly to the bridge
//! and decays fast, the horizontal one is excited ~12 dB less but outlives it by
//! more than three times. Their sum is the double decay a piano is recognised by.
//!
//! Every number here comes from the [`Preset`](crate::preset::Preset): the
//! per-note ones through [`StringParams`], the rest through
//! [`Voicing`].

use crate::modal::ModalBank;
use crate::preset::Voicing;
use crate::resonance::BridgeFilter;
use crate::types::{db_to_amp, BLOCK, MAX_PARTIALS, MAX_PARTIAL_RATIO, SAMPLE_RATE};

/// Largest unison bridge coupling a preset may ask for.
///
/// The coupling passes a fraction of the neighbours' bridge force into a
/// string one block later, so a group is a feedback loop: driven at one of its
/// partials a modal bank answers with `sin(k pi x) / sigma_k` times the force
/// it is given (the `output_scale` the coupling is divided by cancels), and a
/// hop around the loop is that, summed over both polarizations and over the
/// `n - 1` neighbours. The slowest partial in the instrument sets the worst
/// case: at the bottom of the compass the vertical bank contributes ~1.4 and
/// the horizontal one ~1.2 per neighbour, so a triple's loop gain is about
/// `5 * coupling` for a bass note ringing 25 s and about `9 * coupling` for
/// one ringing 40 s. Self-sustainment is therefore somewhere above 0.1, this
/// is a factor of two below the closer of the two, and — unlike the resonance
/// bus, which has a hard ceiling on its drive under it — nothing else bounds
/// this loop.
pub const MAX_UNISON_COUPLING: f32 = 0.05;

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

/// Bounds on a per-partial excitation gain (`notes.partial_gains`), ±20 dB.
///
/// The quantity is *measured* roughness, not a knob: `TUNING_REPORT.md` §3 puts
/// the recordings' scatter around the fitted comb at 5–10 dB RMS with worst
/// partials 12–29 dB out, and `renders/timbre-ladder/ANALYSIS.md` §4a puts the
/// deepest partial anywhere at 9.3–17.7 dB below a smooth envelope. ±20 dB
/// covers every one of those with room, and refuses a table that has stopped
/// describing one hammer striking one string.
pub const MIN_PARTIAL_GAIN: f32 = 0.1;
pub const MAX_PARTIAL_GAIN: f32 = 10.0;

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
    /// coefficient (`TUNING_REPORT.md` §1). The sign flips across that break,
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

/// The two per-partial tables of one key, as borrowed slices of the preset.
///
/// Both are *measurements* the smooth per-note laws cannot carry, and both are
/// deliberately velocity-independent: the roughness of a note's excitation
/// spectrum is not shared with the note beside it (`TUNING_REPORT.md` §3
/// refutes a global admittance curve by measurement), and neither is the way its
/// individual partials depart from the fitted decay law.
///
/// Either may be shorter than the key's partial count — the estimator measures
/// as far up the series as it can track — and every partial past the end is
/// exactly 1.0. [`PartialShaping::default`] is that everywhere, which is the
/// instrument as it was built before these tables existed.
#[derive(Clone, Copy, Debug, Default)]
pub struct PartialShaping<'a> {
    /// Linear multiplier on partial `k`'s excitation gain, 1-based.
    pub gains: &'a [f32],
    /// Multiplier on partial `k`'s decay rate, 1-based.
    pub sigma_scale: &'a [f32],
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

/// One physical string: two polarizations sharing an excitation input.
struct Polarizations {
    vertical: ModalBank,
    horizontal: ModalBank,
    excitation: [f32; BLOCK],
    /// This string's own output during the previous block, so the bridge
    /// coupling can drive its neighbours with everything but itself.
    previous: [f32; BLOCK],
}

/// The unison group belonging to one key.
pub struct PianoString {
    params: StringParams,
    strings: Vec<Polarizations>,
    /// Extra damping per partial at full damper engagement, 1/s.
    damper_profile: Vec<f32>,
    /// Scratch for the current damper engagement, kept to avoid allocating.
    damper_extra: Vec<f32>,
    /// Sum of every unison string's previous block.
    group_previous: [f32; BLOCK],
    /// Force per unit of neighbour output for the bridge coupling.
    coupling: f32,
    /// Share of the hammer's force each string of the group takes.
    shares: Vec<f32>,
    partials: usize,
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

impl PianoString {
    /// Builds the key's unison group.
    ///
    /// `shaping` carries the two per-partial tables (`notes.partial_gains` and
    /// `notes.partial_sigma_scale`); [`PartialShaping::default`] is the
    /// instrument as it was before they existed, to the bit.
    pub fn new(params: StringParams, voicing: &Voicing, shaping: PartialShaping<'_>) -> Self {
        let partials = params.partial_count();
        // `Re Y` in the per-partial damping: a partial that sits on a board
        // mode loses energy into the board faster than the smooth fitted decay
        // law says, and one in a trough slower. This is the half of
        // `PHYSICS.md` §4 the resonance bus cannot produce — the bus subtracts
        // each string's own contribution, so nothing in it is proportional to
        // the string's own motion and it can only ever *add* drive.
        let radiated = radiated_damping(&params, voicing, partials);
        // Mode k's force on the bridge for a hammer impulse J is
        // `4 f0 J sin(k pi x_strike)`: the modal mass of the string is
        // Z / (2 f0), and turning the mode's displacement back into bridge
        // force cancels the wave impedance exactly.
        let output_scale =
            voicing.excitation_scale * params.bridge_gain * params.f0 / REFERENCE_F0;
        let vertical_factor = voicing.vertical_decay_factor();
        let horizontal_gain = db_to_amp(voicing.horizontal_gain_db);
        let mut strings = Vec::with_capacity(params.unison);
        for (i, &polarization_offset) in voicing
            .horizontal_offset_hz
            .iter()
            .take(params.unison)
            .enumerate()
        {
            let detune = voicing.detune_ratio(i, params.unison, params.detune_cents);
            // The strings of a unison do not share one damping law: a group
            // whose strings are mistuned *and* decay at different rates is what
            // moves a composite partial's measured frequency as the survivor
            // takes over, which the recordings do by up to 32 cents and one
            // damping law cannot do at all (`TUNING_REPORT.md` §6).
            let sigma_scale = voicing.sigma_scale(i, params.unison);
            let mut vertical = ModalBank::with_capacity(partials);
            let mut horizontal = ModalBank::with_capacity(partials);
            for k in 1..=partials {
                let f = params.partial_freq(k) * detune;
                // The per-partial scale is applied to the note's own fitted
                // rate, before the polarization factor and before the per-string
                // one, so both banks — and the damper profile below — follow it.
                let sigma = params.partial_sigma(k)
                    * shaping.sigma_scale_at(k)
                    * vertical_factor
                    * sigma_scale
                    * radiated[k - 1];
                // g_k ∝ sin(k pi x_strike) nulls the partials with a node at
                // the strike point, `comb_floor` is how far a real hammer on a
                // real string misses that null, and the contact taper is what a
                // hammer wide enough to average over the comb does to the top of
                // it. `partial_gains` is the measured roughness the smooth comb
                // cannot carry. The 1/SAMPLE_RATE turns the per-sample
                // accumulation of the excitation into an integral over the
                // hammer's force pulse.
                let g = output_scale
                    * comb_magnitude(k, params.strike_position, params.comb_floor)
                    * contact_taper(k, params.contact_width)
                    * shaping.gain_at(k)
                    / SAMPLE_RATE;
                vertical.push_mode(f, sigma, g);
                horizontal.push_mode(
                    f + polarization_offset,
                    sigma * voicing.horizontal_decay_ratio,
                    g * horizontal_gain,
                );
            }
            strings.push(Polarizations {
                vertical,
                horizontal,
                excitation: [0.0; BLOCK],
                previous: [0.0; BLOCK],
            });
        }
        // The damper's grip follows the same per-partial scale: it is a decay
        // rate on the same pole, and a partial the preset says is more lossy
        // than the fitted law is more lossy with the felt on it too.
        let damper_profile: Vec<f32> = (1..=partials)
            .map(|k| {
                params.damper_sigma
                    * voicing.damper_weight_at(params.partial_freq(k))
                    * shaping.sigma_scale_at(k)
            })
            .collect();
        PianoString {
            strings,
            damper_extra: vec![0.0; partials],
            damper_profile,
            group_previous: [0.0; BLOCK],
            // Undo `output_scale` so the neighbour's output is a force in
            // newtons again before a fraction of it is passed on.
            coupling: voicing.unison_coupling / output_scale,
            shares: (0..params.unison)
                .map(|i| voicing.strike_share(i, params.unison))
                .collect(),
            params,
            partials,
            damper: 0.0,
        }
    }

    pub fn params(&self) -> &StringParams {
        &self.params
    }

    pub fn string_count(&self) -> usize {
        self.strings.len()
    }

    pub fn partial_count(&self) -> usize {
        self.partials
    }

    /// Frequency of partial `k` (1-based) on unison string `string`.
    pub fn partial_freq(&self, string: usize, k: usize) -> f32 {
        self.strings[string].vertical.mode_freq(k - 1)
    }

    /// Share of the hammer's force this string of the group takes. The hammer
    /// is not perfectly square to the strings — the same fact that gives the
    /// group its timing skew — so the shares differ by a few percent, which is
    /// what stops the group's summed fundamental cancelling exactly.
    pub fn strike_share(&self, string: usize) -> f32 {
        self.shares[string]
    }

    /// Excitation buffer for one unison string, to be filled before `process`.
    /// Exactly `BLOCK` samples; cleared automatically after each `process`.
    pub fn excitation_mut(&mut self, string: usize) -> &mut [f32] {
        &mut self.strings[string].excitation
    }

    /// Adds a common signal to every unison string's excitation — used for the
    /// sympathetic resonance drive, which reaches all strings of an undamped note.
    pub fn add_excitation_all(&mut self, signal: &[f32], gain: f32) {
        debug_assert_eq!(signal.len(), BLOCK);
        for s in &mut self.strings {
            for (e, &x) in s.excitation.iter_mut().zip(signal) {
                *e += gain * x;
            }
        }
    }

    /// Bridge coupling, one block late for the same reason the resonance bus
    /// is: it breaks the circular dependency between summing and driving. Each
    /// string is driven by its neighbours' previous block, its own removed.
    fn couple(&mut self) {
        for s in &mut self.strings {
            for ((e, &sum), &own) in s
                .excitation
                .iter_mut()
                .zip(&self.group_previous)
                .zip(&s.previous)
            {
                *e += self.coupling * (sum - own);
            }
        }
    }

    /// Renders one block, **adding** the summed output of every unison string
    /// and both polarizations into `out` (exactly `BLOCK` samples).
    pub fn process(&mut self, out: &mut [f32]) {
        debug_assert_eq!(out.len(), BLOCK);
        if self.strings.len() == 1 {
            let s = &mut self.strings[0];
            s.vertical.process_add(&s.excitation, out);
            s.horizontal.process_add(&s.excitation, out);
            s.excitation.fill(0.0);
            return;
        }

        self.couple();
        self.group_previous.fill(0.0);
        for s in &mut self.strings {
            s.previous.fill(0.0);
            s.vertical.process_add(&s.excitation, &mut s.previous);
            s.horizontal.process_add(&s.excitation, &mut s.previous);
            s.excitation.fill(0.0);
            for ((o, g), &v) in out
                .iter_mut()
                .zip(self.group_previous.iter_mut())
                .zip(&s.previous)
            {
                *o += v;
                *g += v;
            }
        }
    }

    /// Renders one block with the two polarizations kept apart: the vertical
    /// bank of every unison string is **added** into `out_v` and the horizontal
    /// one into `out_h` (exactly `BLOCK` samples each).
    ///
    /// The strings advance exactly as they do in [`PianoString::process`] —
    /// same excitation, same coupling, same state — so this is a different way
    /// of *reading* the group, not a different group. It exists because the two
    /// polarizations decay at very different rates, so giving them different
    /// stereo positions is what makes a note's image move as it dies
    /// (`TUNING_REPORT.md` §5). `process` stays the path for the common case:
    /// summing the polarizations per string, in string order, is the
    /// accumulation the instrument's renders are pinned against, and the split
    /// path cannot reproduce that order to the last bit.
    pub fn process_split(&mut self, out_v: &mut [f32], out_h: &mut [f32]) {
        debug_assert_eq!(out_v.len(), BLOCK);
        debug_assert_eq!(out_h.len(), BLOCK);
        if self.strings.len() == 1 {
            let s = &mut self.strings[0];
            s.vertical.process_add(&s.excitation, out_v);
            s.horizontal.process_add(&s.excitation, out_h);
            s.excitation.fill(0.0);
            return;
        }

        self.couple();
        self.group_previous.fill(0.0);
        // Stack scratch, not an allocation: the vertical bank needs a buffer of
        // its own before the string's two polarizations are added back together
        // for the bridge coupling.
        let mut vertical = [0.0f32; BLOCK];
        for s in &mut self.strings {
            vertical.fill(0.0);
            s.previous.fill(0.0);
            s.vertical.process_add(&s.excitation, &mut vertical);
            s.horizontal.process_add(&s.excitation, &mut s.previous);
            s.excitation.fill(0.0);
            for i in 0..BLOCK {
                out_v[i] += vertical[i];
                out_h[i] += s.previous[i];
                // The bridge sees one string, not two planes of one: the
                // coupling drives on the whole of it.
                s.previous[i] += vertical[i];
                self.group_previous[i] += s.previous[i];
            }
        }
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
        for s in &mut self.strings {
            s.vertical.set_damping_profile(&self.damper_extra);
            s.horizontal.set_damping_profile(&self.damper_extra);
        }
    }

    pub fn damper(&self) -> f32 {
        self.damper
    }

    pub fn energy(&self) -> f32 {
        self.strings
            .iter()
            .map(|s| s.vertical.energy() + s.horizontal.energy())
            .sum()
    }

    pub fn is_idle(&self) -> bool {
        self.strings
            .iter()
            .all(|s| s.vertical.is_idle() && s.horizontal.is_idle())
    }

    /// Silences the string immediately (used by `AllOff`).
    pub fn reset(&mut self) {
        for s in &mut self.strings {
            s.vertical.reset_state();
            s.horizontal.reset_state();
            s.excitation.fill(0.0);
            s.previous.fill(0.0);
        }
        self.group_previous.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hammer::Hammer;
    use crate::preset::{BridgeAnchor, BridgePeak, BridgeVoicing, Preset};

    fn preset() -> Preset {
        Preset::default()
    }

    /// Strikes a key for real — hammer pulse into every unison string — and
    /// returns `blocks` blocks of its output. Using the hammer rather than a
    /// unit impulse keeps the signal at the level the culling thresholds and
    /// the rest of the instrument are calibrated for.
    fn strike(key: u8, vel: u8, blocks: usize) -> Vec<f32> {
        let preset = preset();
        let mut string = PianoString::new(preset.string_params(key), &preset.voicing, PartialShaping::default());
        let mut hammer = Hammer::new(preset.hammer_params(key));
        hammer.strike_midi(vel);
        let mut out = vec![0.0f32; blocks * BLOCK];
        for chunk in out.chunks_mut(BLOCK) {
            for i in 0..string.string_count() {
                let share = string.strike_share(i);
                hammer.add_pulse(string.excitation_mut(i), 0, share);
            }
            hammer.advance(BLOCK);
            string.process(chunk);
        }
        out
    }

    fn rms(v: &[f32]) -> f32 {
        (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt()
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
        // (`TUNING_REPORT.md` §1). `B + B4 k^2` is that falling coefficient.
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
        // Partial 20 flattens by what the closed form says, and by an amount
        // the ear resolves: A0's partial 20 sits at ~570 Hz.
        let radicand = 1.0 + p.inharmonicity_b * 400.0 + p.inharmonicity_b4 * 160_000.0;
        let want = 20.0 * p.f0 * radicand.sqrt();
        assert!((p.partial_freq(20) - want).abs() < 1e-4 * want);
        let moved = cents(p.partial_freq(20), base.partial_freq(20));
        assert!((-1.0..-0.1).contains(&moved), "partial 20 moved {moved} cents");
        // The top of the series is where a k^4 term does its work: tens of
        // cents, on partials A0 puts at 2-3 kHz.
        let top = p.partial_count();
        assert!(cents(p.partial_freq(top), base.partial_freq(top)) < -20.0);
        // ... and the series is still a series.
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
    fn the_fourth_order_term_reaches_the_bank_and_the_partial_count() {
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
            assert!((s.partial_freq(1, k) - want).abs() < 1e-3 * want);
            assert!(want > two_parameter_freq(&params, k), "partial {k} not stretched");
        }
    }

    /// `Re Y` in the string's own damping: the half of `PHYSICS.md` §4 that
    /// the resonance bus cannot produce.
    ///
    /// The bus subtracts each string's own contribution, so nothing in it is
    /// proportional to the string's own motion and it can only ever *add*
    /// drive — `resonance.rs` pins that. A partial that sits on a board mode
    /// nevertheless has to die faster than the smooth fitted decay law says,
    /// because that is where the board takes energy fastest, and this is where
    /// it happens: at the partial's own pole, when the instrument is built.
    ///
    /// Measured on the rendered note, not on the coefficient: a single string,
    /// a bridge resonance sitting on its fundamental, and the vertical bank's
    /// stored energy read at two times, which decays at exactly twice the
    /// polarization's rate once only the fundamental is left.
    #[test]
    fn a_partial_on_a_board_mode_decays_faster_than_the_fitted_law() {
        const KEY: u8 = 84;
        const PEAK_DB: f32 = 6.0;
        let base = preset();
        let f0 = base.string_params(KEY).partial_freq(1);

        let rate_at = |share: f32| {
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
            let mut params = preset.string_params(KEY);
            params.unison = 1;
            let mut string = PianoString::new(params, &preset.voicing, PartialShaping::default());
            let mut hammer = Hammer::new(preset.hammer_params(KEY));
            hammer.strike_midi(100);
            let mut out = [0.0f32; BLOCK];
            let probes = [0.35f32, 0.6];
            let mut energy = Vec::new();
            for block in 0..(SAMPLE_RATE / BLOCK as f32) as usize {
                let t = (block * BLOCK) as f32 / SAMPLE_RATE;
                if probes.iter().any(|p| (t - p).abs() < BLOCK as f32 / SAMPLE_RATE * 0.5) {
                    energy.push(string.strings[0].vertical.energy());
                }
                hammer.add_pulse(string.excitation_mut(0), 0, 1.0);
                hammer.advance(BLOCK);
                string.process(&mut out);
            }
            assert_eq!(energy.len(), probes.len());
            (energy[0] / energy[1]).ln() / (2.0 * (probes[1] - probes[0]))
        };

        // A share of zero is the instrument as it was, to the last bit of the
        // pole: the factor is exactly 1.0 and nothing is recomputed.
        let plain = rate_at(0.0);
        let sigma_v = base.string_params(KEY).partial_sigma(1) * base.voicing.vertical_decay_factor();
        assert!(
            (plain / sigma_v - 1.0).abs() < 0.05,
            "a zero share moved the decay: {plain} against the designed {sigma_v}"
        );

        // Half of the loss is into the board, and the board is 6 dB livelier
        // right here, so the partial must lose it 1 + 0.5 * (2 - 1) = 1.5 times
        // as fast.
        let share = 0.5f32;
        let want = 1.0 + share * (db_to_amp(PEAK_DB) - 1.0);
        let faster = rate_at(share) / plain;
        assert!(
            (faster / want - 1.0).abs() < 0.05,
            "a partial on a {PEAK_DB} dB board mode decayed {faster:.3} times faster, \
             expected {want:.3}"
        );
    }

    #[test]
    fn banks_are_laid_out_from_the_formula() {
        let preset = preset();
        let params = preset.string_params(60);
        let s = PianoString::new(params, &preset.voicing, PartialShaping::default());
        assert_eq!(s.partial_count(), params.partial_count());
        assert!(s.partial_count() > 40);
        for k in 1..=s.partial_count() {
            // The middle string sits within a tenth of a Hz of nominal pitch.
            let want = params.partial_freq(k);
            assert!((s.partial_freq(1, k) - want).abs() < 1e-3 * want);
        }
    }

    /// Composite envelope of one partial: the two polarizations summed, as a
    /// fraction of their value at the moment of the strike.
    fn composite_envelope(sigma: f32, t: f32) -> f32 {
        let v = preset().voicing;
        let gain = db_to_amp(v.horizontal_gain_db);
        let sigma_v = sigma * v.vertical_decay_factor();
        ((-sigma_v * t).exp() + gain * (-v.horizontal_decay_ratio * sigma_v * t).exp())
            / (1.0 + gain)
    }

    #[test]
    fn fundamental_t60_matches_the_spec_anchors() {
        // The anchors are whole-note figures: the quiet, slow horizontal
        // polarization is what is still there at -60 dB, so it alone decides
        // when the note gets there.
        for (key, want) in [(21u8, 25.0f32), (60, 12.0), (84, 3.0), (108, 0.6)] {
            let sigma = preset().string_params(key).partial_sigma(1);
            let (mut lo, mut hi) = (0.0f32, 10.0 * want);
            for _ in 0..50 {
                let mid = 0.5 * (lo + hi);
                if composite_envelope(sigma, mid) > 1.0e-3 {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            assert!(
                (lo / want - 1.0).abs() < 0.02,
                "key {key}: T60 {lo} s, expected {want} s"
            );
        }
    }

    #[test]
    fn a_rendered_note_decays_at_the_designed_rate() {
        // One string, so nothing beats and nothing couples into it: each bank's
        // stored energy is then a clean exponential at twice its polarization's
        // decay rate, measured late enough that only the fundamental is left.
        let key = 84u8;
        let preset = preset();
        let mut params = preset.string_params(key);
        params.unison = 1;
        let sigma_v = params.partial_sigma(1) * preset.voicing.vertical_decay_factor();

        let mut string = PianoString::new(params, &preset.voicing, PartialShaping::default());
        let mut hammer = Hammer::new(preset.hammer_params(key));
        hammer.strike_midi(100);
        let mut out = [0.0f32; BLOCK];
        let mut samples: Vec<(f32, f32)> = Vec::new();
        let probes = [0.35f32, 0.6, 1.0, 2.0];
        for block in 0..(2.0 * SAMPLE_RATE / BLOCK as f32) as usize + 1 {
            let t = (block * BLOCK) as f32 / SAMPLE_RATE;
            if probes.iter().any(|p| (t - p).abs() < BLOCK as f32 / SAMPLE_RATE * 0.5) {
                samples.push((
                    string.strings[0].vertical.energy(),
                    string.strings[0].horizontal.energy(),
                ));
            }
            hammer.add_pulse(string.excitation_mut(0), 0, 1.0);
            hammer.advance(BLOCK);
            string.process(&mut out);
        }
        assert_eq!(samples.len(), probes.len());

        let rate = |e0: f32, e1: f32, dt: f32| (e0 / e1).ln() / (2.0 * dt);
        let vertical = rate(samples[0].0, samples[1].0, probes[1] - probes[0]);
        let horizontal = rate(samples[2].1, samples[3].1, probes[3] - probes[2]);
        assert!(
            (vertical / sigma_v - 1.0).abs() < 0.05,
            "vertical sigma {vertical}, expected {sigma_v}"
        );
        let want = sigma_v * preset.voicing.horizontal_decay_ratio;
        assert!(
            (horizontal / want - 1.0).abs() < 0.05,
            "horizontal sigma {horizontal}, expected {want}"
        );
    }

    /// Decay rate of one unison string's vertical bank, measured on the note
    /// the engine actually renders: the bank's stored energy is a clean
    /// exponential at twice its polarization's rate once the high partials have
    /// gone, so the fundamental alone is what the two probes see.
    fn per_string_vertical_sigma(preset: &Preset, key: u8, probes: [f32; 2]) -> Vec<f32> {
        let params = preset.string_params(key);
        let mut string = PianoString::new(params, &preset.voicing, PartialShaping::default());
        let mut hammer = Hammer::new(preset.hammer_params(key));
        hammer.strike_midi(100);
        let mut out = [0.0f32; BLOCK];
        let mut energy = vec![[0.0f32; 2]; params.unison];
        let last = probes[1] + 0.05;
        for block in 0..(last * SAMPLE_RATE / BLOCK as f32) as usize {
            let t = (block * BLOCK) as f32 / SAMPLE_RATE;
            for (p, &probe) in probes.iter().enumerate() {
                if (t - probe).abs() < BLOCK as f32 / SAMPLE_RATE * 0.5 {
                    for (s, e) in energy.iter_mut().enumerate() {
                        e[p] = string.strings[s].vertical.energy();
                    }
                }
            }
            for i in 0..string.string_count() {
                let share = string.strike_share(i);
                hammer.add_pulse(string.excitation_mut(i), 0, share);
            }
            hammer.advance(BLOCK);
            string.process(&mut out);
        }
        let dt = probes[1] - probes[0];
        energy
            .iter()
            .map(|e| (e[0] / e[1]).ln() / (2.0 * dt))
            .collect()
    }

    /// The strings of a unison do not share one damping law once the preset
    /// says they do not — and they do share it, exactly, when it does not.
    #[test]
    fn unison_sigma_scale_sets_each_string_of_a_group_going_at_its_own_rate() {
        let key = 84u8;
        for scales in [[1.0f32, 1.0, 1.0], [0.7, 1.0, 1.3]] {
            let mut preset = preset();
            preset.voicing.unison_sigma_scale[2].scale = scales.to_vec();
            // The bridge coupling is a second decay law on top of the one being
            // measured — it moves energy between the strings — and this test is
            // about the per-string sigma alone.
            preset.voicing.unison_coupling = 0.0;
            assert!(preset.validate().is_ok(), "{scales:?} is a legal voicing");

            let params = preset.string_params(key);
            assert_eq!(params.unison, 3);
            let designed = params.partial_sigma(1) * preset.voicing.vertical_decay_factor();
            let string = PianoString::new(params, &preset.voicing, PartialShaping::default());
            for (s, &scale) in scales.iter().enumerate() {
                // The layout says it ...
                assert!(
                    (string.strings[s].vertical.mode_sigma(0) / (designed * scale) - 1.0).abs()
                        < 1e-6,
                    "string {s} vertical sigma"
                );
                // ... for both polarizations.
                let horizontal = designed * scale * preset.voicing.horizontal_decay_ratio;
                assert!(
                    (string.strings[s].horizontal.mode_sigma(0) / horizontal - 1.0).abs() < 1e-6,
                    "string {s} horizontal sigma"
                );
            }
            // ... and so does the note it renders. With every scale at 1 this
            // is the whole-note T60 anchor of `notes.sigma0` being reproduced
            // string by string: a row that averages to 1 redistributes the
            // note's damping and does not retune it.
            for (s, (&scale, measured)) in scales
                .iter()
                .zip(per_string_vertical_sigma(&preset, key, [0.35, 0.6]))
                .enumerate()
            {
                let want = designed * scale;
                assert!(
                    (measured / want - 1.0).abs() < 0.05,
                    "{scales:?}: string {s} decays at {measured}, expected {want}"
                );
            }
        }
    }

    // ------------------------------------------- the per-partial tables

    /// Strikes one unison string of `key` and returns the output.
    ///
    /// One string, so nothing beats and nothing couples: the render is a plain
    /// sum of decaying sinusoids at the partials the bank was built with, which
    /// is what makes projecting onto one of them mean something.
    fn strike_single(preset: &Preset, key: u8, shaping: PartialShaping<'_>, blocks: usize) -> Vec<f32> {
        let mut params = preset.string_params(key);
        params.unison = 1;
        let mut string = PianoString::new(params, &preset.voicing, shaping);
        let mut hammer = Hammer::new(preset.hammer_params(key));
        hammer.strike_midi(100);
        let mut out = vec![0.0f32; blocks * BLOCK];
        for chunk in out.chunks_mut(BLOCK) {
            hammer.add_pulse(string.excitation_mut(0), 0, 1.0);
            hammer.advance(BLOCK);
            string.process(chunk);
        }
        out
    }

    /// Magnitude of `y` at `hz`, over a Hann-windowed span — enough to separate
    /// one partial of a single string from its neighbours.
    fn partial_magnitude(y: &[f32], hz: f32, from: usize, len: usize) -> f64 {
        let (mut re, mut im) = (0.0f64, 0.0f64);
        for (i, &v) in y[from..(from + len).min(y.len())].iter().enumerate() {
            let w = 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / len as f64).cos();
            let phase = std::f64::consts::TAU * hz as f64 * i as f64 / SAMPLE_RATE as f64;
            re += w * v as f64 * phase.cos();
            im -= w * v as f64 * phase.sin();
        }
        (re * re + im * im).sqrt() * 2.0 / len as f64
    }

    /// `notes.partial_gains` is a gain on one partial's excitation and on
    /// nothing else: doubling entry `k` doubles what partial `k` contributes to
    /// the rendered note, and leaves every other partial where it was.
    ///
    /// Measured on the render by subtraction rather than on the coefficient: the
    /// difference between the two renders is, by linearity, exactly the extra
    /// copy of partial `k`, so it must project to that partial's own amplitude
    /// at its own frequency and to nothing anywhere else.
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
        };

        // The layout first: exactly doubled there, exactly untouched elsewhere.
        let mut plain_params = params;
        plain_params.unison = 1;
        let plain_string = PianoString::new(plain_params, &preset.voicing, PartialShaping::default());
        let loud = PianoString::new(plain_params, &preset.voicing, shaping);
        for k in 0..plain_string.partial_count() {
            let want = if k + 1 == K { 2.0 } else { 1.0 } * plain_string.strings[0].vertical.mode_gain(k);
            assert_eq!(loud.strings[0].vertical.mode_gain(k), want, "partial {}", k + 1);
            let want_h = if k + 1 == K { 2.0 } else { 1.0 } * plain_string.strings[0].horizontal.mode_gain(k);
            assert_eq!(loud.strings[0].horizontal.mode_gain(k), want_h);
        }

        // ... and the sound. 0.1 s from 0.1 s in, past the hammer pulse.
        let blocks = (0.4 * SAMPLE_RATE / BLOCK as f32) as usize;
        let base = strike_single(&preset, KEY, PartialShaping::default(), blocks);
        let doubled = strike_single(&preset, KEY, shaping, blocks);
        let difference: Vec<f32> = doubled.iter().zip(&base).map(|(a, b)| a - b).collect();
        let (from, len) = ((0.1 * SAMPLE_RATE) as usize, (0.1 * SAMPLE_RATE) as usize);
        let at = |y: &[f32], k: usize| partial_magnitude(y, params.partial_freq(k), from, len);

        let extra = at(&difference, K);
        let original = at(&base, K);
        assert!(original > 0.0, "the probe partial never sounded");
        assert!(
            (extra / original - 1.0).abs() < 0.02,
            "the extra copy of partial {K} is {:.4} of the partial itself",
            extra / original
        );
        // Nothing else moved: the difference is silent at the neighbours.
        for k in [K - 1, K + 1, K + 3] {
            let leak = at(&difference, k) / at(&base, k);
            assert!(
                leak < 0.01,
                "doubling partial {K} moved partial {k} by a factor of {leak:e}"
            );
        }
    }

    /// `notes.partial_sigma_scale` reaches both polarizations, the damper
    /// profile, and the decay the note actually renders at.
    #[test]
    fn a_partial_sigma_scale_changes_that_partials_decay_and_its_damper() {
        const KEY: u8 = 84;
        let preset = preset();
        let params = preset.string_params(KEY);
        // Half the fitted rate on the fundamental and twice it on the second
        // partial: both directions, and the fundamental is the one the render
        // below can see on its own.
        let mut scale = vec![1.0f32; params.partial_count()];
        scale[0] = 0.5;
        scale[1] = 2.0;
        let shaping = PartialShaping {
            gains: &[],
            sigma_scale: &scale,
        };

        let mut single = params;
        single.unison = 1;
        let plain = PianoString::new(single, &preset.voicing, PartialShaping::default());
        let scaled = PianoString::new(single, &preset.voicing, shaping);
        for (k, &want) in scale.iter().enumerate().take(plain.partial_count()) {
            for bank in 0..2 {
                let (a, b) = if bank == 0 {
                    (
                        plain.strings[0].vertical.mode_sigma(k),
                        scaled.strings[0].vertical.mode_sigma(k),
                    )
                } else {
                    (
                        plain.strings[0].horizontal.mode_sigma(k),
                        scaled.strings[0].horizontal.mode_sigma(k),
                    )
                };
                assert!(
                    (b / (a * want) - 1.0).abs() < 1e-6,
                    "partial {} of bank {bank}: {b} against {}",
                    k + 1,
                    a * want
                );
            }
            // The damper is a decay rate on the same pole and follows too.
            assert!(
                (scaled.damper_profile[k] / (plain.damper_profile[k] * want) - 1.0).abs()
                    < 1e-6,
                "damper profile of partial {}",
                k + 1
            );
        }

        // And the note it renders: the vertical bank's stored energy decays at
        // twice its polarization's rate once only the fundamental is left, and
        // a fundamental told to lose energy half as fast has to take twice as
        // long to get there.
        let designed = single.partial_sigma(1) * preset.voicing.vertical_decay_factor();
        let measured = |shaping: PartialShaping<'_>| {
            let mut string = PianoString::new(single, &preset.voicing, shaping);
            let mut hammer = Hammer::new(preset.hammer_params(KEY));
            hammer.strike_midi(100);
            let mut out = [0.0f32; BLOCK];
            let probes = [0.35f32, 0.6];
            let mut energy = Vec::new();
            for block in 0..(0.7 * SAMPLE_RATE / BLOCK as f32) as usize {
                let t = (block * BLOCK) as f32 / SAMPLE_RATE;
                if probes.iter().any(|p| (t - p).abs() < BLOCK as f32 / SAMPLE_RATE * 0.5) {
                    energy.push(string.strings[0].vertical.energy());
                }
                hammer.add_pulse(string.excitation_mut(0), 0, 1.0);
                hammer.advance(BLOCK);
                string.process(&mut out);
            }
            assert_eq!(energy.len(), probes.len());
            (energy[0] / energy[1]).ln() / (2.0 * (probes[1] - probes[0]))
        };
        let plain_rate = measured(PartialShaping::default());
        let slow_rate = measured(shaping);
        assert!(
            (plain_rate / designed - 1.0).abs() < 0.05,
            "the control decayed at {plain_rate} against the designed {designed}"
        );
        assert!(
            (slow_rate / (designed * 0.5) - 1.0).abs() < 0.05,
            "a fundamental scaled by 0.5 decayed at {slow_rate}, expected {}",
            designed * 0.5
        );
    }

    /// `notes.comb_floor` lifts the partials the strike comb nulls and leaves
    /// every other partial where it was — which is the whole difference between
    /// a hammer with width and a hammer that is a point.
    ///
    /// A2 is the key `renders/timbre-ladder/ANALYSIS.md` §4a measures: the
    /// engine's worst partial there is k = 17, exactly where `sin(k pi x)`
    /// crosses zero, 42 dB down, while the recording's deepest partial anywhere
    /// is 9.3–17.7 dB down and never at that index.
    #[test]
    fn the_comb_floor_lifts_the_null_partials_and_leaves_the_others_alone() {
        const KEY: u8 = 45; // A2
        const FLOOR: f32 = 0.05;
        let preset = preset();
        let params = preset.string_params(KEY);
        let comb = |k: usize| (k as f32 * std::f32::consts::PI * params.strike_position).sin();
        // The null the strike position puts in this key's series, found rather
        // than written down: whichever partial the comb is quietest at.
        let null = (1..=params.partial_count())
            .min_by(|&a, &b| comb(a).abs().partial_cmp(&comb(b).abs()).unwrap())
            .expect("the key has partials");
        assert!(comb(null).abs() < 0.02, "A2's comb has no null to fill");

        let mut floored_params = params;
        floored_params.comb_floor = FLOOR;
        let plain = PianoString::new(params, &preset.voicing, PartialShaping::default());
        let floored = PianoString::new(floored_params, &preset.voicing, PartialShaping::default());

        // A zero floor is the bare comb to the last bit, on every string of the
        // group and both of its planes.
        let mut zero_params = params;
        zero_params.comb_floor = 0.0;
        let zero = PianoString::new(zero_params, &preset.voicing, PartialShaping::default());
        for s in 0..plain.string_count() {
            for k in 0..plain.partial_count() {
                assert_eq!(
                    zero.strings[s].vertical.mode_gain(k),
                    plain.strings[s].vertical.mode_gain(k)
                );
                assert_eq!(
                    zero.strings[s].horizontal.mode_gain(k),
                    plain.strings[s].horizontal.mode_gain(k)
                );
            }
        }

        for k in 1..=plain.partial_count() {
            let before = plain.strings[0].vertical.mode_gain(k - 1);
            let after = floored.strings[0].vertical.mode_gain(k - 1);
            // The sign is the partial's starting phase and does not move.
            if before != 0.0 {
                assert_eq!(after.signum(), before.signum(), "partial {k} changed sign");
            }
            // The magnitude is exactly the formula.
            let c = comb(k);
            let want = (c * c + FLOOR * FLOOR).sqrt() / c.abs();
            let ratio = after.abs() / before.abs();
            assert!(
                (ratio / want - 1.0).abs() < 1e-4,
                "partial {k} moved by {ratio}, expected {want}"
            );
            let db = 20.0 * ratio.log10();
            if k == null {
                assert!(db > 12.0, "the null at {k} only rose {db:.1} dB");
            } else if c.abs() > 0.3 {
                // Everything the comb actually excites moves by `floor^2/2c^2`,
                // which at a third of the crest is already a tenth of a decibel
                // and at the crest is a hundredth: the same floor is 12 dB at
                // the null and inaudible everywhere else, which is the whole
                // claim.
                assert!(db < 0.1, "partial {k} rose {db:.3} dB and is not a null");
                if c.abs() > 0.9 {
                    assert!(db < 0.02, "partial {k} rose {db:.3} dB at the comb's crest");
                }
            }
        }

        // Stated as the fault it fixes: the deepest partial in the comb goes
        // from far below the ladder's measured floor to inside it.
        let depth = |s: &PianoString, k: usize| {
            20.0 * (s.strings[0].vertical.mode_gain(k - 1).abs()
                / s.strings[0].vertical.mode_gain(0).abs())
            .log10()
        };
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
            // Three entries of ones and nothing else: the rest of the series
            // has to fall back to 1.0 rather than to 0.0.
            let short = PianoString::new(
                params,
                &preset.voicing,
                PartialShaping {
                    gains: &[1.0, 1.0, 1.0],
                    sigma_scale: &[1.0],
                },
            );
            for s in 0..plain.string_count() {
                for k in 0..plain.partial_count() {
                    assert_eq!(
                        short.strings[s].vertical.mode_gain(k),
                        plain.strings[s].vertical.mode_gain(k),
                        "key {key} partial {}",
                        k + 1
                    );
                    assert_eq!(
                        short.strings[s].vertical.mode_sigma(k),
                        plain.strings[s].vertical.mode_sigma(k)
                    );
                    assert_eq!(
                        short.strings[s].horizontal.mode_sigma(k),
                        plain.strings[s].horizontal.mode_sigma(k)
                    );
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
                assert!(
                    wide <= narrow,
                    "partial {k}: {narrow} at {} rose to {wide} at {}",
                    w[0],
                    w[1]
                );
                // Strictly, while there is anything left to take away.
                if narrow > 0.0 {
                    assert!(wide < narrow, "partial {k} did not move between {w:?}");
                }
            }
            // ... and always harder on the higher partial.
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

        // The gains the bank is built with are the comb times that taper.
        let width = 0.015;
        let mut preset = preset();
        for w in &mut preset.notes.contact_width {
            *w = width;
        }
        assert!(preset.validate().is_ok());
        let key = 96u8; // C7, where the contact is the largest fraction of a string
        let stock = Preset::default();
        let plain = PianoString::new(stock.string_params(key), &stock.voicing, PartialShaping::default());
        let tapered = PianoString::new(preset.string_params(key), &preset.voicing, PartialShaping::default());
        for k in 0..tapered.partial_count() {
            let want = plain.strings[0].vertical.mode_gain(k) * contact_taper(k + 1, width);
            let got = tapered.strings[0].vertical.mode_gain(k);
            assert!(
                (got - want).abs() <= 1e-6 * want.abs().max(1e-20),
                "partial {} gain {got}, expected {want}",
                k + 1
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
        // 1 / 0.12 is not an integer, so the null falls between two partials
        // and neither of them vanishes entirely.
        let node = (1.0 / params.strike_position).round() as usize;
        assert!(gain(node) < 0.3, "partial {node} is not nulled: {}", gain(node));
        assert!(gain(node) < 0.4 * gain(node - 1));
        assert!((1..node).any(|k| gain(k) > 0.95));
        assert!(s.partial_count() > node);
    }

    #[test]
    fn a_unison_group_beats() {
        // Three strings detuned by a fraction of a Hz make the fundamental's
        // envelope wobble instead of decaying monotonically.
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
        let mut s = PianoString::new(preset.string_params(60), &preset.voicing, PartialShaping::default());
        let mut hammer = Hammer::new(preset.hammer_params(60));
        hammer.strike_midi(100);
        let mut warm = vec![0.0f32; 40 * BLOCK];
        for chunk in warm.chunks_mut(BLOCK) {
            for i in 0..s.string_count() {
                hammer.add_pulse(s.excitation_mut(i), 0, 1.0);
            }
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
        let first = s.damper_profile[0];
        let last = s.damper_profile[s.partials - 1];
        assert!(first > last * 2.0, "damper profile {first} .. {last}");
        assert!((first - p.damper_sigma).abs() < 0.01 * p.damper_sigma);
    }

    #[test]
    fn bridge_coupling_rings_an_unstruck_sibling() {
        let preset = preset();
        let mut s = PianoString::new(preset.string_params(60), &preset.voicing, PartialShaping::default());
        assert_eq!(s.string_count(), 3);
        let mut out = [0.0f32; BLOCK];
        s.excitation_mut(0)[0] = 1.0;
        for _ in 0..400 {
            s.process(&mut out);
        }
        let sibling = s.strings[2].vertical.energy();
        assert!(sibling > 0.0, "unstruck sibling never picked anything up");
        // ... but stays far below the struck string: this is a weak coupling.
        assert!(sibling < s.strings[0].vertical.energy() * 0.25);
    }

    /// The split path is a different way of reading the group, not a different
    /// group: same excitation, same coupling, same state.
    #[test]
    fn splitting_the_polarizations_renders_the_same_string() {
        let (key, preset) = (60u8, preset());
        let mut summed = PianoString::new(preset.string_params(key), &preset.voicing, PartialShaping::default());
        let mut split = PianoString::new(preset.string_params(key), &preset.voicing, PartialShaping::default());
        let mut hammer = Hammer::new(preset.hammer_params(key));
        let mut hammer_split = Hammer::new(preset.hammer_params(key));
        hammer.strike_midi(100);
        hammer_split.strike_midi(100);

        let (mut a, mut v, mut h) = ([0.0f32; BLOCK], [0.0f32; BLOCK], [0.0f32; BLOCK]);
        let mut peak = 0.0f32;
        for _ in 0..200 {
            for i in 0..summed.string_count() {
                let share = summed.strike_share(i);
                hammer.add_pulse(summed.excitation_mut(i), 0, share);
                hammer_split.add_pulse(split.excitation_mut(i), 0, share);
            }
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
                // numbers in a different order.
                assert!((a[i] - (v[i] + h[i])).abs() <= 1e-6 * peak.max(1e-12));
            }
        }
        assert!(peak > 0.0);
        assert!((split.energy() / summed.energy() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn excitation_produces_output_and_is_consumed() {
        let preset = preset();
        let mut s = PianoString::new(preset.string_params(60), &preset.voicing, PartialShaping::default());
        s.excitation_mut(0)[0] = 1.0;
        let mut out = [0.0f32; BLOCK];
        s.process(&mut out);
        assert!(out.iter().any(|v| v.abs() > 0.0));
        assert!(s.excitation_mut(0).iter().all(|&v| v == 0.0));
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
}


