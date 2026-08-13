//! Pedal state: sustain (continuous), sostenuto (captured set), una corda.
//!
//! This module only tracks state and answers "how much damping should key i
//! get". Applying it is `voice.rs`'s and `string.rs`'s job: they scale the
//! note's `damper_sigma` by [`PedalState::damper_amount`] and, per partial, by
//! the frequency-dependent damper weight that lives with the string.

use crate::types::{index_to_note, FIRST_UNDAMPED_KEY, NUM_KEYS};

/// Keys from G6 up have no dampers on a grand — they always ring.
pub fn has_damper(key: u8) -> bool {
    key < FIRST_UNDAMPED_KEY
}

pub struct PedalState {
    sustain: f32,
    sostenuto: bool,
    una_corda: bool,
    captured: [bool; NUM_KEYS],
}

impl PedalState {
    pub fn new() -> Self {
        PedalState {
            sustain: 0.0,
            sostenuto: false,
            una_corda: false,
            captured: [false; NUM_KEYS],
        }
    }

    pub fn sustain(&self) -> f32 {
        self.sustain
    }

    pub fn set_sustain(&mut self, value: f32) {
        self.sustain = value.clamp(0.0, 1.0);
    }

    pub fn sostenuto(&self) -> bool {
        self.sostenuto
    }

    /// Engaging sostenuto captures exactly the keys held at that instant — the
    /// pedal's tab rail can only catch damper levers that are already raised,
    /// so keys pressed afterwards are not held. Releasing it drops the whole
    /// captured set at once.
    pub fn set_sostenuto(&mut self, on: bool, held: &[bool; NUM_KEYS]) {
        self.sostenuto = on;
        if on {
            self.captured = *held;
        } else {
            self.captured = [false; NUM_KEYS];
        }
    }

    pub fn is_captured(&self, index: usize) -> bool {
        self.sostenuto && self.captured[index]
    }

    pub fn una_corda(&self) -> bool {
        self.una_corda
    }

    pub fn set_una_corda(&mut self, on: bool) {
        self.una_corda = on;
    }

    /// How much damping key `index` should receive right now: 0.0 = damper
    /// fully lifted, 1.0 = damper fully down. Half-pedal falls out of the
    /// continuous sustain value as the `(1 - pedal)` multiplier from the spec.
    pub fn damper_amount(&self, index: usize, key_held: bool) -> f32 {
        if !has_damper(index_to_note(index)) || key_held || self.is_captured(index) {
            0.0
        } else {
            1.0 - self.sustain
        }
    }

    pub fn reset(&mut self) {
        *self = PedalState::new();
    }
}

impl Default for PedalState {
    fn default() -> Self {
        PedalState::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::key_index;

    fn held_set(keys: &[u8]) -> [bool; NUM_KEYS] {
        let mut held = [false; NUM_KEYS];
        for &k in keys {
            held[key_index(k).unwrap()] = true;
        }
        held
    }

    #[test]
    fn top_octave_has_no_dampers() {
        assert!(has_damper(90));
        assert!(!has_damper(91));
        assert!(!has_damper(108));
    }

    #[test]
    fn sustain_scales_damping_continuously() {
        let mut p = PedalState::new();
        let c4 = key_index(60).unwrap();
        assert_eq!(p.damper_amount(c4, false), 1.0);
        p.set_sustain(0.5);
        assert_eq!(p.damper_amount(c4, false), 0.5);
        p.set_sustain(0.25);
        assert_eq!(p.damper_amount(c4, false), 0.75);
        p.set_sustain(1.0);
        assert_eq!(p.damper_amount(c4, false), 0.0);
        // Out-of-range pedal values must not invert the damper.
        p.set_sustain(2.0);
        assert_eq!(p.damper_amount(c4, false), 0.0);
        p.set_sustain(-1.0);
        assert_eq!(p.damper_amount(c4, false), 1.0);
    }

    #[test]
    fn held_keys_are_never_damped() {
        let p = PedalState::new();
        assert_eq!(p.damper_amount(key_index(60).unwrap(), true), 0.0);
    }

    #[test]
    fn sostenuto_holds_exactly_the_set_held_at_capture() {
        let mut p = PedalState::new();
        let held = held_set(&[48, 55, 60]);
        p.set_sostenuto(true, &held);
        for (index, &want) in held.iter().enumerate() {
            assert_eq!(
                p.is_captured(index),
                want,
                "key {} captured {}, expected {want}",
                index_to_note(index),
                p.is_captured(index)
            );
            // Captured keys stay lifted after release; every other key that
            // has a damper at all takes it.
            let expect_damping = if want || !has_damper(index_to_note(index)) {
                0.0
            } else {
                1.0
            };
            assert_eq!(p.damper_amount(index, false), expect_damping);
        }
    }

    #[test]
    fn sostenuto_ignores_keys_pressed_after_capture() {
        let mut p = PedalState::new();
        p.set_sostenuto(true, &held_set(&[48]));
        let (c3, c4) = (key_index(48).unwrap(), key_index(60).unwrap());
        // C4 is struck now: held, so undamped, but not captured — releasing it
        // must damp it while C3 keeps ringing.
        assert_eq!(p.damper_amount(c4, true), 0.0);
        assert!(!p.is_captured(c4));
        assert_eq!(p.damper_amount(c4, false), 1.0);
        assert_eq!(p.damper_amount(c3, false), 0.0);
    }

    #[test]
    fn releasing_sostenuto_drops_the_whole_set() {
        let mut p = PedalState::new();
        p.set_sostenuto(true, &held_set(&[48, 55]));
        p.set_sostenuto(false, &held_set(&[]));
        assert!(!p.sostenuto());
        for index in 0..NUM_KEYS {
            assert!(!p.is_captured(index));
        }
        assert_eq!(p.damper_amount(key_index(48).unwrap(), false), 1.0);
    }

    #[test]
    fn re_engaging_sostenuto_recaptures() {
        let mut p = PedalState::new();
        p.set_sostenuto(true, &held_set(&[48]));
        p.set_sostenuto(true, &held_set(&[60]));
        assert!(!p.is_captured(key_index(48).unwrap()));
        assert!(p.is_captured(key_index(60).unwrap()));
    }

    #[test]
    fn una_corda_is_a_plain_flag() {
        let mut p = PedalState::new();
        assert!(!p.una_corda());
        p.set_una_corda(true);
        assert!(p.una_corda());
        p.reset();
        assert!(!p.una_corda());
    }
}
