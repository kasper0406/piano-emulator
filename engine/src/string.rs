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

/// Per-note string parameters. Every field is a starting point that automated
/// tuning is expected to overwrite later.
#[derive(Clone, Copy, Debug)]
pub struct StringParams {
    /// Fundamental frequency in Hz.
    pub f0: f32,
    /// Stiffness inharmonicity coefficient B in `f_k = k f0 sqrt(1 + B k^2)`.
    pub inharmonicity_b: f32,
    /// Hammer strike point as a fraction of string length.
    pub strike_position: f32,
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
    /// Frequency of partial `k` (1-based) including stiffness inharmonicity.
    pub fn partial_freq(&self, k: usize) -> f32 {
        let k = k as f32;
        k * self.f0 * (1.0 + self.inharmonicity_b * k * k).sqrt()
    }

    /// Decay rate of partial `k` for the note as a whole, 1/s: `6.91 / sigma`
    /// is the time that partial takes to fall 60 dB counting both
    /// polarizations. The vertical bank decays faster than this and the
    /// horizontal one slower — see
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
            let mut vertical = ModalBank::with_capacity(partials);
            let mut horizontal = ModalBank::with_capacity(partials);
            for k in 1..=partials {
                let f = params.partial_freq(k) * detune;
                let sigma = params.partial_sigma(k) * vertical_factor;
                // g_k ∝ sin(k pi x_strike) nulls the partials with a node at
                // the strike point; the 1/SAMPLE_RATE turns the per-sample
                // accumulation of the excitation into an integral over the
                // hammer's force pulse.
                let g = output_scale
                    * (k as f32 * std::f32::consts::PI * params.strike_position).sin()
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

        // Bridge coupling, one block late for the same reason the resonance bus
        // is: it breaks the circular dependency between summing and driving.
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


