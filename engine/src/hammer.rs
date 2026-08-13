//! Lumped nonlinear felt hammer.
//!
//! At note-on the hammer/string collision is integrated forward into a short
//! force pulse, which the voice then streams into the string's modal banks.
//! Precomputing keeps the per-sample audio loop free of branches and lets the
//! unison strings read the same pulse at slightly different offsets.
//!
//! ```text
//! felt:   F = K c^p (1 + lambda dc/dt)   for compression c > 0
//! hammer: m v' = -F,                     c = x_hammer - y_contact
//! string: y_contact' = [F(t) - beta F(t - t_ref) - k_string y_contact] / (2 Z)
//! ```
//!
//! The string surrogate is the whole physics of the collision. A force applied
//! to a string launches velocity waves `F/(2Z)` both ways; the one travelling
//! to the agraffe comes back inverted after the round trip `t_ref` and stops
//! the contact point from running away from the hammer. That delayed term is
//! what makes a treble note (agraffe 30 us away, so the string stiffens almost
//! at once) have a short felt-dominated contact while a bass note (round trip
//! 4 ms, longer than the contact itself) rides on a purely resistive string for
//! milliseconds — and its arrivals *during* contact put the ripple in the force
//! pulse that gives a real piano tone its upper partials. The reflection from
//! the bridge side arrives long after any contact ends and is not modelled.
//!
//! The reflection is a delayed positive feedback path of gain
//! `k_felt t_ref / 2Z`, and beyond ~1 the lossless lumped model diverges — a
//! property of the model, not of the integrator. `beta` therefore carries as
//! much of it as is safe (all of it up to a hard mezzo-forte anywhere below the
//! top octave) and hands the rest to `k_string = (1 - beta) 2Z / t_ref`, the
//! spring the reflection is equivalent to once several round trips have passed.
//! The two have the same static effect on the contact; only the ripple differs.

use crate::types::SAMPLE_RATE;

/// Longest force pulse the scratch buffer can hold: 20 ms, far above the
/// 0.4-6 ms a real contact lasts, so the integration can never overrun.
pub const MAX_PULSE_SAMPLES: usize = 960;

/// Longest per-unison-string timing skew, 0.3 ms.
pub const MAX_SKEW_SAMPLES: usize = 15;

/// Substeps per audio sample used when integrating the collision. The felt
/// spring is the fastest thing in the instrument (~20 krad/s at a fortissimo
/// C8) and needs several steps per cycle; it also sets the resolution of the
/// reflection delay line, which is only 1.6 audio samples long at the top of
/// the compass. A note-on costs 6-60 us of the audio thread's budget.
const OVERSAMPLE: usize = 8;

/// Largest delayed-reflection loop gain `k_felt t_ref / 2Z` the contact model
/// carries literally. The reflection is a delayed positive feedback path, and
/// once its gain passes ~1 the lossless lumped model diverges — which is a
/// property of the model, not of the integrator. Above the limit the surplus is
/// handed to the reflection's quasi-static equivalent, the string spring
/// `2Z / t_ref`, which has the same effect on the contact but no delay.
const MAX_REFLECTION_GAIN_MARGIN: f32 = 1.0;

/// Ceiling on the Hunt-Crossley factor. The felt cannot stiffen without bound,
/// and the returning reflection can make the compression rate jump.
const MAX_HYSTERESIS_FACTOR: f32 = 2.0;

#[derive(Clone, Copy, Debug)]
pub struct HammerParams {
    /// Hammer head mass, kg.
    pub mass: f32,
    /// Felt stiffness K, in N/m^p.
    pub stiffness: f32,
    /// Felt nonlinearity exponent p.
    pub exponent: f32,
    /// Transverse wave impedance of one string, kg/s.
    pub impedance: f32,
    /// Strings the hammer meets at once. They load it in parallel, so the
    /// driving-point impedance and stiffness it works against are this many
    /// times a single string's, and each string receives this fraction of the
    /// force. It is the main reason a bass note's contact lasts three times a
    /// treble note's.
    pub strings: f32,
    /// Round trip from the strike point to the agraffe and back, in seconds.
    /// Sets where the string stops looking resistive and starts looking stiff.
    pub reflection_seconds: f32,
    /// Hunt-Crossley hysteresis coefficient, s/m: the felt is stiffer while
    /// being compressed than while relaxing, so it returns less energy than it
    /// stored. The loss grows with impact speed, which is the measured
    /// behaviour of felt (restitution ~0.9 at 1 m/s falling to ~0.6 at 4 m/s).
    pub hysteresis: f32,
    /// Stiffness multiplier applied under una corda.
    pub una_corda_stiffness: f32,
    /// Velocity reflection coefficient of the agraffe end of the speaking
    /// length. Below one because the termination is not perfectly rigid and,
    /// more importantly, string stiffness disperses the returning pulse.
    pub reflection_gain: f32,
    /// Hammer speed at MIDI velocity 1 and at 127, m/s.
    pub velocity_min: f32,
    pub velocity_max: f32,
}

impl HammerParams {
    /// Length of the reflection delay line, in integration substeps.
    fn reflection_substeps(&self) -> usize {
        let steps = (self.reflection_seconds * SAMPLE_RATE * OVERSAMPLE as f32).round() as usize;
        steps.clamp(1, MAX_PULSE_SAMPLES * OVERSAMPLE)
    }

    /// MIDI velocity 1..127 to hammer velocity in m/s. Exponential, so each
    /// MIDI step is a constant ratio — the mapping the ear reads as even.
    pub fn hammer_velocity(&self, vel: u8) -> f32 {
        let v = vel.clamp(1, 127) as f32;
        self.velocity_min * (self.velocity_max / self.velocity_min).powf((v - 1.0) / 126.0)
    }
}

pub struct Hammer {
    params: HammerParams,
    una_corda: bool,
    pulse: Vec<f32>,
    /// Force applied over the last round trip to the agraffe, one entry per
    /// integration substep: the delay line the reflection comes back through.
    history: Vec<f32>,
    len: usize,
    cursor: usize,
}

impl Hammer {
    pub fn new(params: HammerParams) -> Self {
        Hammer {
            una_corda: false,
            pulse: vec![0.0; MAX_PULSE_SAMPLES],
            history: vec![0.0; params.reflection_substeps()],
            params,
            len: 0,
            cursor: 0,
        }
    }

    pub fn params(&self) -> &HammerParams {
        &self.params
    }

    pub fn set_una_corda(&mut self, on: bool) {
        self.una_corda = on;
    }

    pub fn una_corda(&self) -> bool {
        self.una_corda
    }

    /// Integrates a strike at MIDI velocity `vel` through this hammer's own
    /// velocity mapping.
    pub fn strike_midi(&mut self, vel: u8) {
        self.strike(self.params.hammer_velocity(vel));
    }

    /// Integrates a strike at `velocity` m/s into the pulse buffer and rewinds
    /// the read cursor. Allocation-free: both buffers are sized at construction.
    pub fn strike(&mut self, velocity: f32) {
        self.len = 0;
        self.cursor = 0;
        if velocity <= 0.0 {
            return;
        }
        let k = if self.una_corda {
            self.params.stiffness * self.params.una_corda_stiffness
        } else {
            self.params.stiffness
        };
        let m = self.params.mass;
        let p = self.params.exponent;
        let two_z = 2.0 * self.params.impedance * self.params.strings;
        let dt = 1.0 / (SAMPLE_RATE * OVERSAMPLE as f32);

        // How much of the reflection the model can carry as a delay: estimate
        // the felt stiffness at the deepest compression this strike will reach
        // (all the hammer's energy stored in the felt) and cap the loop gain.
        let c_max = ((p + 1.0) * m * velocity * velocity / (2.0 * k)).powf(1.0 / (p + 1.0));
        let felt_stiffness_max = p * k * c_max.powf(p - 1.0);
        let loop_gain = felt_stiffness_max * self.params.reflection_seconds / two_z;
        // Fraction of the reflection the delay line may carry; the rest goes to
        // the spring it is equivalent to once several round trips have passed.
        let carried = (MAX_REFLECTION_GAIN_MARGIN / loop_gain).min(1.0);
        let delayed = self.params.reflection_gain * carried;
        let string_stiffness = (1.0 - carried) * two_z / self.params.reflection_seconds;

        // Hammer and contact-point displacement, both zero at first touch.
        let mut x = 0.0f32;
        let mut y = 0.0f32;
        let mut v = velocity;
        let mut compression_rate = v;
        let mut touched = false;

        self.history.fill(0.0);
        let mut read = 0usize;
        for n in 0..MAX_PULSE_SAMPLES {
            let mut acc = 0.0f32;
            let mut separated = false;
            for _ in 0..OVERSAMPLE {
                let c = x - y;
                // Hysteresis may not pull the felt into tension, hence the max.
                let f = if c > 0.0 {
                    let hysteresis =
                        (1.0 + self.params.hysteresis * compression_rate)
                            .clamp(0.0, MAX_HYSTERESIS_FACTOR);
                    k * c.powf(p) * hysteresis
                } else {
                    0.0
                };
                touched |= f > 0.0;
                acc += f;

                // What the string pushes back with: the inverted reflection on
                // its way home, plus the stiffness standing in for the rest of it.
                let string_reaction = delayed * self.history[read] + string_stiffness * y;
                self.history[read] = f;
                read += 1;
                if read == self.history.len() {
                    read = 0;
                }

                // Semi-implicit step for the contact point: the felt's local
                // stiffness dF/dc = p F / c passes 1e6 N/m at a hard treble
                // strike, so solving the step for y instead of stepping it
                // explicitly is what keeps the integration bounded without
                // paying for 100x oversampling.
                let felt_stiffness = if c > 0.0 { p * f / c } else { 0.0 };
                let contact_velocity = (f - string_reaction) / (two_z + felt_stiffness * dt);
                compression_rate = v - contact_velocity;
                v -= f / m * dt;
                x += v * dt;
                y += contact_velocity * dt;
                separated |= touched && x - y <= 0.0;
            }
            // The banks are driven per string, so the force is shared out.
            self.pulse[n] = acc / (OVERSAMPLE as f32 * self.params.strings);
            self.len = n + 1;
            if separated {
                break;
            }
        }
    }

    /// Force pulse computed by the last `strike`, in newtons.
    pub fn pulse(&self) -> &[f32] {
        &self.pulse[..self.len]
    }

    /// True while any unison string still has pulse left to read.
    pub fn is_active(&self) -> bool {
        self.len > 0 && self.cursor < self.len + MAX_SKEW_SAMPLES
    }

    /// Adds this block's slice of the force pulse into `out`, delayed by `skew`
    /// samples. Does not move the cursor, so every unison string can call it.
    pub fn add_pulse(&self, out: &mut [f32], skew: usize, gain: f32) {
        for (i, o) in out.iter_mut().enumerate() {
            let idx = self.cursor + i;
            if idx >= skew {
                if let Some(&f) = self.pulse().get(idx - skew) {
                    *o += gain * f;
                }
            }
        }
    }

    /// Advances the read cursor once all strings have been fed this block.
    pub fn advance(&mut self, frames: usize) {
        self.cursor += frames;
    }

    /// Abandons any pulse still in flight.
    pub fn reset(&mut self) {
        self.len = 0;
        self.cursor = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preset::Preset;
    use crate::types::{HIGHEST_KEY, LOWEST_KEY};

    fn hammer_for(key: u8) -> Hammer {
        Hammer::new(Preset::default().hammer_params(key))
    }

    fn struck(key: u8, vel: u8) -> Hammer {
        let mut h = hammer_for(key);
        h.strike_midi(vel);
        h
    }

    fn peak(h: &Hammer) -> f32 {
        h.pulse().iter().fold(0.0f32, |m, &v| m.max(v))
    }

    fn duration_ms(h: &Hammer) -> f32 {
        h.pulse().len() as f32 / SAMPLE_RATE * 1000.0
    }

    /// Fraction of the pulse's energy above `edge` Hz, by direct DFT of the
    /// force pulse — the measure of how bright the strike is.
    fn high_energy_fraction(h: &Hammer, edge: f32) -> f32 {
        let pulse = h.pulse();
        let bin = |f: f32| -> f32 {
            let w = std::f32::consts::TAU * f / SAMPLE_RATE;
            let (mut re, mut im) = (0.0f32, 0.0f32);
            for (n, &s) in pulse.iter().enumerate() {
                let phase = w * n as f32;
                re += s * phase.cos();
                im -= s * phase.sin();
            }
            re * re + im * im
        };
        let mut total = 0.0;
        let mut high = 0.0;
        // 100 Hz .. 8 kHz in 100 Hz steps: the band the strike's colour lives in.
        for i in 1..=80 {
            let f = 100.0 * i as f32;
            let e = bin(f);
            total += e;
            if f >= edge {
                high += e;
            }
        }
        high / total
    }

    #[test]
    fn velocity_mapping_is_monotonic_and_bounded() {
        let params = Preset::default().hammer_params(60);
        assert!((params.hammer_velocity(1) - params.velocity_min).abs() < 1e-6);
        assert!((params.hammer_velocity(127) - params.velocity_max).abs() < 1e-4);
        let mut prev = 0.0;
        for vel in 1..=127u8 {
            let v = params.hammer_velocity(vel);
            assert!(v > prev);
            prev = v;
        }
    }

    #[test]
    fn contact_time_is_physical_across_the_compass() {
        for key in [21u8, 40, 60, 84, 108] {
            for vel in [20u8, 80, 127] {
                let h = struck(key, vel);
                let ms = duration_ms(&h);
                // The spec's 0.5-3 ms is the typical range; a pianissimo bass
                // strike genuinely stays in contact for twice that.
                assert!(
                    (0.35..6.0).contains(&ms),
                    "key {key} vel {vel}: contact {ms} ms"
                );
            }
        }
    }

    #[test]
    fn contact_shortens_towards_the_treble() {
        let bass = duration_ms(&struck(21, 80));
        let middle = duration_ms(&struck(60, 80));
        let treble = duration_ms(&struck(96, 80));
        assert!(bass > middle, "bass {bass} ms vs middle {middle} ms");
        assert!(middle > treble, "middle {middle} ms vs treble {treble} ms");
    }

    #[test]
    fn no_strike_anywhere_runs_away() {
        // The delayed reflection is a positive feedback path, so this is the
        // load-bearing test of the whole contact model: over the entire
        // key/velocity plane the pulse must stay a passive collision. Impulse
        // is `m v (1 + e)`, and the restitution `e` of a felt hammer against a
        // string that carries energy away cannot reach 1.
        for key in LOWEST_KEY..=HIGHEST_KEY {
            for vel in [1u8, 20, 48, 80, 110, 127] {
                let h = struck(key, vel);
                let m = h.params().mass;
                let v = h.params().hammer_velocity(vel);
                let impulse: f32 =
                    h.pulse().iter().sum::<f32>() * h.params().strings / SAMPLE_RATE;
                assert!(
                    h.pulse().iter().all(|f| f.is_finite() && *f >= 0.0),
                    "key {key} vel {vel}: pulse is not a positive finite force"
                );
                assert!(
                    impulse > m * v * 0.9 && impulse < m * v * 2.0,
                    "key {key} vel {vel}: impulse {impulse} vs m v {}",
                    m * v
                );
            }
        }
    }

    #[test]
    fn harder_strikes_are_shorter_stronger_and_brighter() {
        let soft = struck(60, 30);
        let hard = struck(60, 120);
        assert!(peak(&hard) > peak(&soft) * 4.0);
        assert!(
            hard.pulse().len() < soft.pulse().len(),
            "hard {} ms vs soft {} ms",
            duration_ms(&hard),
            duration_ms(&soft)
        );
        let (b_soft, b_hard) = (
            high_energy_fraction(&soft, 1000.0),
            high_energy_fraction(&hard, 1000.0),
        );
        assert!(b_hard > b_soft * 1.2, "brightness {b_soft} -> {b_hard}");
    }

    #[test]
    fn una_corda_softens_the_blow() {
        let mut normal = hammer_for(60);
        let mut soft = hammer_for(60);
        soft.set_una_corda(true);
        normal.strike(2.0);
        soft.strike(2.0);
        assert!(peak(&soft) < peak(&normal));
        assert!(soft.pulse().len() > normal.pulse().len());
        assert!(high_energy_fraction(&soft, 1000.0) < high_energy_fraction(&normal, 1000.0));
    }

    #[test]
    fn skew_delays_the_pulse() {
        let mut h = hammer_for(60);
        h.strike(2.0);
        let mut a = [0.0f32; 64];
        let mut b = [0.0f32; 64];
        h.add_pulse(&mut a, 0, 1.0);
        h.add_pulse(&mut b, 10, 1.0);
        assert_eq!(&a[..54], &b[10..]);
        assert!(b[..10].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn restriking_reuses_the_buffers() {
        let mut h = hammer_for(60);
        let pulse_ptr = h.pulse.as_ptr();
        for vel in [1u8, 64, 127] {
            h.strike_midi(vel);
            assert!(h.is_active());
        }
        assert_eq!(h.pulse.as_ptr(), pulse_ptr);
    }
}


