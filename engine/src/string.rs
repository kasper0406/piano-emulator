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

impl PianoString {
    pub fn new(params: StringParams, voicing: &Voicing) -> Self {
        let partials = params.partial_count();
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
                let sigma = params.partial_sigma(k) * vertical_factor * sigma_scale;
                // g_k ∝ sin(k pi x_strike) nulls the partials with a node at
                // the strike point, and the contact taper is what a hammer wide
                // enough to average over that comb does to the top of it; the
                // 1/SAMPLE_RATE turns the per-sample accumulation of the
                // excitation into an integral over the hammer's force pulse.
                let g = output_scale
                    * (k as f32 * std::f32::consts::PI * params.strike_position).sin()
                    * contact_taper(k, params.contact_width)
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
        let damper_profile: Vec<f32> = (1..=partials)
            .map(|k| params.damper_sigma * voicing.damper_weight_at(params.partial_freq(k)))
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
    use crate::preset::Preset;

    fn preset() -> Preset {
        Preset::default()
    }

    /// Strikes a key for real — hammer pulse into every unison string — and
    /// returns `blocks` blocks of its output. Using the hammer rather than a
    /// unit impulse keeps the signal at the level the culling thresholds and
    /// the rest of the instrument are calibrated for.
    fn strike(key: u8, vel: u8, blocks: usize) -> Vec<f32> {
        let preset = preset();
        let mut string = PianoString::new(preset.string_params(key), &preset.voicing);
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

        let s = PianoString::new(params, &preset.voicing);
        assert_eq!(s.partial_count(), params.partial_count());
        for k in 1..=s.partial_count() {
            let want = params.partial_freq(k);
            assert!((s.partial_freq(1, k) - want).abs() < 1e-3 * want);
            assert!(want > two_parameter_freq(&params, k), "partial {k} not stretched");
        }
    }

    #[test]
    fn banks_are_laid_out_from_the_formula() {
        let preset = preset();
        let params = preset.string_params(60);
        let s = PianoString::new(params, &preset.voicing);
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

        let mut string = PianoString::new(params, &preset.voicing);
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
        let mut string = PianoString::new(params, &preset.voicing);
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
            let string = PianoString::new(params, &preset.voicing);
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
        let plain = PianoString::new(stock.string_params(key), &stock.voicing);
        let tapered = PianoString::new(preset.string_params(key), &preset.voicing);
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
        let s = PianoString::new(params, &preset.voicing);
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
        let mut s = PianoString::new(preset.string_params(60), &preset.voicing);
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
        let s = PianoString::new(p, &preset.voicing);
        let first = s.damper_profile[0];
        let last = s.damper_profile[s.partials - 1];
        assert!(first > last * 2.0, "damper profile {first} .. {last}");
        assert!((first - p.damper_sigma).abs() < 0.01 * p.damper_sigma);
    }

    #[test]
    fn bridge_coupling_rings_an_unstruck_sibling() {
        let preset = preset();
        let mut s = PianoString::new(preset.string_params(60), &preset.voicing);
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
        let mut summed = PianoString::new(preset.string_params(key), &preset.voicing);
        let mut split = PianoString::new(preset.string_params(key), &preset.voicing);
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
        let mut s = PianoString::new(preset.string_params(60), &preset.voicing);
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


