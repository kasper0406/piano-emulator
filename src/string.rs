//! Piano string: partial layout, frequency-dependent damping, two
//! polarizations, and the unison group of 1-3 slightly detuned strings.
//!
//! A key's sound is a group of 1-3 strings; each string is two modal banks, one
//! per polarization. The vertical polarization couples strongly to the bridge
//! and decays fast, the horizontal one is excited ~12 dB less but outlives it by
//! more than three times. Their sum is the double decay a piano is recognised by.

use crate::modal::ModalBank;
use crate::types::{
    db_to_amp, interp_anchors, key_position, note_to_freq, BLOCK, MAX_PARTIALS, MAX_PARTIAL_RATIO,
    MAX_UNISON, SAMPLE_RATE,
};

/// Horizontal polarization: quieter input, much slower decay, and a small
/// frequency offset from the vertical one. The offset comes from the string's
/// transverse stiffness differing between the two planes, which no two strings
/// of a unison group share exactly — hence one value per string of the group.
const HORIZONTAL_GAIN_DB: f32 = -12.0;
const HORIZONTAL_DECAY_RATIO: f32 = 0.29;
const HORIZONTAL_OFFSET_HZ: [f32; MAX_UNISON] = [0.35, 0.52, 0.27];

/// Coefficient of the `(f/1000)^2` term in the per-partial decay rate. Air and
/// internal friction losses grow with frequency, so high partials die first.
const SIGMA_FREQ_COEFF: f32 = 0.5;

/// Bridge coupling within a unison group, as a fraction of the string's wave
/// impedance: the bridge is not a rigid termination, so each string feels a
/// force proportional to its neighbours' velocity. This is what keeps the
/// string the una corda hammer misses ringing, and it makes a unison group's
/// decay uneven rather than a plain sum of three exponentials.
const UNISON_COUPLING: f32 = 0.02;

/// Turns the string's force on the bridge, in newtons, into the engine's
/// internal signal unit. Purely gain staging: calibrated by rendering so a
/// mezzo-forte C4 lands near -20 dBFS after `OUTPUT_GAIN`, leaving the safety
/// limiter idle on a fortissimo chord.
const EXCITATION_SCALE: f32 = 0.40;

/// Note the per-note gains below are normalised against, so `EXCITATION_SCALE`
/// stays a plain level control.
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
    /// soundboard's coupling to this part of the compass. See [`bridge_gain_for`].
    pub bridge_gain: f32,
}

impl StringParams {
    pub fn for_key(key: u8) -> Self {
        let t = key_position(key);
        let f0 = note_to_freq(key);

        // Fundamental T60 anchors from the spec: 25 s at A0, 12 s at C4,
        // 3 s at C6, 0.6 s at C8. Interpolated in log-T60 so the curve is smooth.
        let t60 = interp_anchors(
            t,
            &[
                (key_position(21), 25.0f32.ln()),
                (key_position(60), 12.0f32.ln()),
                (key_position(84), 3.0f32.ln()),
                (key_position(108), 0.6f32.ln()),
            ],
        )
        .exp();
        let sigma_fundamental = 6.91 / t60;
        let sigma1 = SIGMA_FREQ_COEFF;
        let sigma0 = (sigma_fundamental - sigma1 * (f0 / 1000.0).powi(2)).max(0.01);

        StringParams {
            f0,
            inharmonicity_b: inharmonicity_for(key),
            strike_position: interp_anchors(t, &[(0.0, 0.12), (0.55, 0.115), (1.0, 0.14)]),
            sigma0,
            sigma1,
            unison: unison_count(key),
            // 0.28 Hz at C2 through 0.45 at C4 to 2.4 Hz at C8 — the spec's
            // range, and a beat period of 3-6 s where a pianist would hear it.
            detune_cents: interp_anchors(t, &[(0.0, 3.5), (1.0, 2.0)]),
            // Z = mu c = T / c: the tension is roughly constant across the
            // compass while the wave speed rises, so the impedance falls
            // steeply out of the bass and then flattens out.
            impedance: interp_anchors(
                t,
                &[
                    (key_position(21), 6.5f32.ln()),
                    (key_position(36), 4.5f32.ln()),
                    (key_position(60), 2.2f32.ln()),
                    (key_position(84), 1.8f32.ln()),
                    (key_position(108), 1.7f32.ln()),
                ],
            )
            .exp(),
            // Release T60 0.3 s in the bass falling to 0.1 s in the treble.
            damper_sigma: 6.91 / interp_anchors(t, &[(0.0, 0.3), (1.0, 0.1)]),
            bridge_gain: bridge_gain_for(key),
        }
    }

    /// Frequency of partial `k` (1-based) including stiffness inharmonicity.
    pub fn partial_freq(&self, k: usize) -> f32 {
        let k = k as f32;
        k * self.f0 * (1.0 + self.inharmonicity_b * k * k).sqrt()
    }

    /// Decay rate of partial `k` for the note as a whole, 1/s: `6.91 / sigma`
    /// is the time that partial takes to fall 60 dB counting both
    /// polarizations. The vertical bank decays faster than this and the
    /// horizontal one slower — see [`vertical_decay_factor`].
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

/// Inharmonicity B: ~1e-4 for the wound bass strings, dipping around C3,
/// then rising steeply through the short thick treble strings to ~1e-2 at C8.
fn inharmonicity_for(key: u8) -> f32 {
    let t = key_position(key);
    interp_anchors(
        t,
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
pub fn unison_count(key: u8) -> usize {
    match key {
        0..=35 => 1,     // .. B1
        36..=52 => 2,    // C2 .. E3
        _ => MAX_UNISON, // F3 ..
    }
}

/// How much faster the vertical polarization decays than the note as a whole.
///
/// The horizontal polarization starts `HORIZONTAL_GAIN_DB` down but decays
/// `HORIZONTAL_DECAY_RATIO` times as fast, so it is what is left at the end and
/// it alone sets when the note reaches -60 dB. Solving
/// `g_h exp(-rho sigma_v T60) = 1e-3 (1 + g_h)` for `sigma_v` gives the factor
/// between the spec's T60 anchors and the vertical bank's decay rate.
fn vertical_decay_factor() -> f32 {
    let gain = db_to_amp(HORIZONTAL_GAIN_DB);
    (gain / (1.0e-3 * (1.0 + gain))).ln() / (HORIZONTAL_DECAY_RATIO * 6.91)
}

/// How firmly the damper felt grips a partial at `f_hz`. Dampers hold the low
/// partials tightly and the top ones barely at all, which is why a released
/// note keeps a brief metallic zing.
fn damper_weight(f_hz: f32) -> f32 {
    interp_anchors(
        f_hz.max(1.0).ln(),
        &[
            (500.0f32.ln(), 1.0),
            (2000.0f32.ln(), 0.9),
            (6000.0f32.ln(), 0.35),
            (12000.0f32.ln(), 0.2),
        ],
    )
}

/// Fraction of a string's bridge force that becomes signal, in dB relative to
/// C4. A real soundboard is not an equally good radiator at every frequency:
/// its admittance peaks in the low-mid and falls away at both ends of the
/// compass, and the bass bridge is loaded by the long bass strings. Without
/// this the model's compass is tilted ~12 dB against the bass. Calibrated by
/// rendering fortissimo single notes and flattening their peak level; it is a
/// voicing table, and the obvious thing for automated tuning to replace.
fn bridge_gain_for(key: u8) -> f32 {
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

/// Frequency ratio of unison string `i` of `n` against nominal pitch, given the
/// group's full spread in cents.
///
/// The three strings are deliberately *not* evenly spaced. Evenly spaced
/// detunings make the three fundamentals coincide in antiphase at a fixed
/// point of every beat cycle and cancel to nothing — the note is heard pumping
/// to silence and back, which no piano does. With uneven spacing the two beat
/// rates are incommensurate and the cancellation never lines up.
fn detune_ratio(i: usize, n: usize, width_cents: f32) -> f32 {
    const PATTERN: [&[f32]; MAX_UNISON] = [&[0.0], &[-0.47, 0.53], &[-0.5, 0.11, 0.5]];
    let cents = width_cents * PATTERN[n.clamp(1, MAX_UNISON) - 1][i];
    (cents / 1200.0 * std::f32::consts::LN_2).exp()
}

/// Share of the hammer's force that string `i` of `n` receives. The hammer is
/// not perfectly square to the strings — the same fact that gives the group its
/// timing skew — so the shares differ by a few percent, and the group's summed
/// fundamental can no longer cancel exactly. Each row averages to 1, so the
/// group's total excitation does not depend on how the shares are spread, and
/// each is paired with a detuning pattern whose amplitude-weighted centre is
/// nominal pitch, so the group as a whole is in tune.
fn strike_share(i: usize, n: usize) -> f32 {
    const PATTERN: [&[f32]; MAX_UNISON] = [&[1.0], &[1.06, 0.94], &[1.09, 1.0, 0.91]];
    PATTERN[n.clamp(1, MAX_UNISON) - 1][i]
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
    partials: usize,
    damper: f32,
}

impl PianoString {
    pub fn new(params: StringParams) -> Self {
        let partials = params.partial_count();
        // Mode k's force on the bridge for a hammer impulse J is
        // `4 f0 J sin(k pi x_strike)`: the modal mass of the string is
        // Z / (2 f0), and turning the mode's displacement back into bridge
        // force cancels the wave impedance exactly.
        let output_scale = EXCITATION_SCALE * params.bridge_gain * params.f0 / REFERENCE_F0;
        let vertical_factor = vertical_decay_factor();
        let horizontal_gain = db_to_amp(HORIZONTAL_GAIN_DB);
        let mut strings = Vec::with_capacity(params.unison);
        for (i, &polarization_offset) in HORIZONTAL_OFFSET_HZ.iter().take(params.unison).enumerate()
        {
            let detune = detune_ratio(i, params.unison, params.detune_cents);
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
                    sigma * HORIZONTAL_DECAY_RATIO,
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
            .map(|k| params.damper_sigma * damper_weight(params.partial_freq(k)))
            .collect();
        PianoString {
            strings,
            damper_extra: vec![0.0; partials],
            damper_profile,
            group_previous: [0.0; BLOCK],
            // Undo `output_scale` so the neighbour's output is a force in
            // newtons again before a fraction of it is passed on.
            coupling: UNISON_COUPLING / output_scale,
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

    /// Share of the hammer's force this string of the group takes; see
    /// [`strike_share`].
    pub fn strike_share(&self, string: usize) -> f32 {
        strike_share(string, self.strings.len())
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
    use crate::hammer::{velocity_from_midi, Hammer, HammerParams};

    /// Strikes a key for real — hammer pulse into every unison string — and
    /// returns `blocks` blocks of its output. Using the hammer rather than a
    /// unit impulse keeps the signal at the level the culling thresholds and
    /// the rest of the instrument are calibrated for.
    fn strike(key: u8, vel: u8, blocks: usize) -> Vec<f32> {
        let params = StringParams::for_key(key);
        let mut string = PianoString::new(params);
        let mut hammer = Hammer::new(HammerParams::for_key(key, params.impedance));
        hammer.strike(velocity_from_midi(vel));
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
    fn unison_counts_follow_the_compass() {
        assert_eq!(unison_count(35), 1); // B1
        assert_eq!(unison_count(36), 2); // C2
        assert_eq!(unison_count(52), 2); // E3
        assert_eq!(unison_count(53), 3); // F3
    }

    #[test]
    fn inharmonicity_rises_towards_the_treble() {
        assert!(inharmonicity_for(108) > inharmonicity_for(60));
        assert!(inharmonicity_for(60) > inharmonicity_for(21));
        assert!(inharmonicity_for(21) > 0.0);
    }

    #[test]
    fn partial_layout_follows_the_inharmonicity_formula() {
        for key in [21u8, 48, 60, 84, 108] {
            let p = StringParams::for_key(key);
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
        let params = StringParams::for_key(60);
        let s = PianoString::new(params);
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
        let gain = db_to_amp(HORIZONTAL_GAIN_DB);
        let sigma_v = sigma * vertical_decay_factor();
        ((-sigma_v * t).exp() + gain * (-HORIZONTAL_DECAY_RATIO * sigma_v * t).exp()) / (1.0 + gain)
    }

    #[test]
    fn fundamental_t60_matches_the_spec_anchors() {
        // The anchors are whole-note figures: the quiet, slow horizontal
        // polarization is what is still there at -60 dB, so it alone decides
        // when the note gets there.
        for (key, want) in [(21u8, 25.0f32), (60, 12.0), (84, 3.0), (108, 0.6)] {
            let sigma = StringParams::for_key(key).partial_sigma(1);
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
        let mut params = StringParams::for_key(key);
        params.unison = 1;
        let sigma_v = params.partial_sigma(1) * vertical_decay_factor();

        let mut string = PianoString::new(params);
        let mut hammer = Hammer::new(HammerParams::for_key(key, params.impedance));
        hammer.strike(velocity_from_midi(100));
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
        let want = sigma_v * HORIZONTAL_DECAY_RATIO;
        assert!(
            (horizontal / want - 1.0).abs() < 0.05,
            "horizontal sigma {horizontal}, expected {want}"
        );
    }

    #[test]
    fn high_partials_decay_faster_than_the_fundamental() {
        let p = StringParams::for_key(60);
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
            let p = StringParams::for_key(key);
            let n = p.partial_count();
            assert!((1..=MAX_PARTIALS).contains(&n));
            assert!(p.partial_freq(n) < MAX_PARTIAL_RATIO * SAMPLE_RATE);
        }
    }

    #[test]
    fn strike_position_nulls_the_partial_it_sits_on() {
        // x_strike ~ 1/8 in the bass, so partial 8 must be far weaker than its
        // neighbours.
        let params = StringParams::for_key(21);
        let s = PianoString::new(params);
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

    /// Detuning in cents of string `i` of `n` for a group `width` cents wide.
    fn detune_cents(i: usize, n: usize, width: f32) -> f32 {
        1200.0 * detune_ratio(i, n, width).log2()
    }

    #[test]
    fn detune_spans_the_width_unevenly() {
        assert_eq!(detune_ratio(0, 1, 3.0), 1.0);
        for n in 1..=MAX_UNISON {
            let d: Vec<f32> = (0..n).map(|i| detune_cents(i, n, 3.0)).collect();
            assert!(d.windows(2).all(|w| w[1] > w[0]), "{d:?} is not ordered");
            let want = if n > 1 { 3.0 } else { 0.0 };
            assert!((d[n - 1] - d[0] - want).abs() < 1e-3, "{d:?} spans wrong");
        }
        // Evenly spaced detunings cancel exactly; these must not be even.
        let gaps = [
            detune_cents(1, 3, 3.0) - detune_cents(0, 3, 3.0),
            detune_cents(2, 3, 3.0) - detune_cents(1, 3, 3.0),
        ];
        assert!(gaps[0] / gaps[1] > 1.3 || gaps[1] / gaps[0] > 1.3, "{gaps:?}");
    }

    #[test]
    fn a_unison_group_is_in_tune_as_a_whole() {
        for n in 1..=MAX_UNISON {
            let shares: Vec<f32> = (0..n).map(|i| strike_share(i, n)).collect();
            let mean = shares.iter().sum::<f32>() / n as f32;
            assert!((mean - 1.0).abs() < 1e-6, "{shares:?} averages {mean}");
            if n > 1 {
                // No two strings may share an amplitude, or that pair cancels
                // exactly; and the loudness-weighted pitch must be nominal, or
                // the group as a whole plays flat or sharp.
                assert!(shares.windows(2).all(|w| (w[0] - w[1]).abs() > 0.02));
                let centre: f32 = (0..n).map(|i| shares[i] * detune_cents(i, n, 3.0)).sum();
                assert!(centre.abs() < 0.1, "group centre is {centre} cents off");
            }
        }
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
        let params = StringParams::for_key(60);
        let mut s = PianoString::new(params);
        let mut hammer = Hammer::new(HammerParams::for_key(60, params.impedance));
        hammer.strike(velocity_from_midi(100));
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
        let p = StringParams::for_key(48);
        let s = PianoString::new(p);
        let first = s.damper_profile[0];
        let last = s.damper_profile[s.partials - 1];
        assert!(first > last * 2.0, "damper profile {first} .. {last}");
        assert!((first - p.damper_sigma).abs() < 0.01 * p.damper_sigma);
    }

    #[test]
    fn bridge_coupling_rings_an_unstruck_sibling() {
        let mut s = PianoString::new(StringParams::for_key(60));
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
        let mut s = PianoString::new(StringParams::for_key(60));
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


