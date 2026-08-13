//! One voice per key: the unison string group, its hammer, and the damper
//! lifecycle that ties them to the pedals.
//!
//! A voice is never stolen. Re-striking a ringing key must not reset the modal
//! banks — the new hammer pulse adds into the state that is still there, which
//! is what happens physically and what makes repeated notes under the sustain
//! pedal sound right.

use crate::hammer::{Hammer, MAX_SKEW_SAMPLES};
use crate::pedal::PedalState;
use crate::preset::Preset;
use crate::resonance::ResonanceBus;
use crate::soundboard::pan_for_key;
use crate::string::PianoString;
use crate::types::{key_index, BLOCK};

/// Time constant of the damper engage/release ramp. The felt takes a few
/// milliseconds to settle onto the string; stepping the damping would click.
const DAMPER_RAMP_SECONDS: f32 = 0.010;

pub struct Voice {
    key: u8,
    index: usize,
    pan: f32,
    string: PianoString,
    hammer: Hammer,
    held: bool,
    damper_current: f32,
    damper_target: f32,
    damper_step: f32,
    /// This voice's output during the block the resonance bus was summed from.
    previous_out: [f32; BLOCK],
    previous_silent: bool,
}

impl Voice {
    pub fn new(key: u8, preset: &Preset) -> Self {
        let hammer = Hammer::new(preset.hammer_params(key));
        let mut voice = Voice {
            key,
            index: key_index(key).expect("voice key must be within A0..C8"),
            pan: pan_for_key(key),
            string: PianoString::new(preset.string_params(key), &preset.voicing),
            hammer,
            held: false,
            damper_current: 0.0,
            damper_target: 0.0,
            damper_step: BLOCK as f32 / (DAMPER_RAMP_SECONDS * crate::types::SAMPLE_RATE),
            previous_out: [0.0; BLOCK],
            previous_silent: true,
        };
        // Idle keys below G6 rest with their dampers down.
        voice.damper_current = if crate::pedal::has_damper(key) { 1.0 } else { 0.0 };
        voice.damper_target = voice.damper_current;
        voice.string.set_damper(voice.damper_current);
        voice
    }

    pub fn key(&self) -> u8 {
        self.key
    }

    pub fn index(&self) -> usize {
        self.index
    }

    pub fn pan(&self) -> f32 {
        self.pan
    }

    pub fn is_held(&self) -> bool {
        self.held
    }

    /// A voice can be skipped entirely when nothing is ringing and no hammer
    /// pulse is in flight.
    pub fn is_idle(&self) -> bool {
        !self.hammer.is_active() && self.string.is_idle()
    }

    pub fn string(&self) -> &PianoString {
        &self.string
    }

    pub fn note_on(&mut self, vel: u8, pedals: &PedalState) {
        self.held = true;
        self.hammer.set_una_corda(pedals.una_corda());
        self.hammer.strike_midi(vel);
        self.update_dampers(pedals);
    }

    pub fn note_off(&mut self, pedals: &PedalState) {
        self.held = false;
        self.update_dampers(pedals);
    }

    /// Recomputes the damper target from the current pedal state. Cheap; call
    /// it on every pedal or key change.
    pub fn update_dampers(&mut self, pedals: &PedalState) {
        self.damper_target = pedals.damper_amount(self.index, self.held);
    }

    /// Renders one block into `out` (overwritten), reading the sympathetic
    /// resonance bus and leaving `out` ready to be fed back into it.
    ///
    /// Returns false when the voice had nothing to render and `out` was not
    /// touched. A voice that is silent still has to run whenever its dampers
    /// are off the strings and the bus carries something — that is the whole
    /// mechanism of sympathetic resonance, and skipping it would mean a piano
    /// whose undamped strings never answer the ones being played.
    pub fn process(&mut self, out: &mut [f32], bus: &ResonanceBus) -> bool {
        debug_assert_eq!(out.len(), BLOCK);
        if self.is_idle() && !(bus.is_active() && self.damper_target < 1.0) {
            if !self.previous_silent {
                self.previous_out.fill(0.0);
                self.previous_silent = true;
            }
            return false;
        }
        out.fill(0.0);

        if self.damper_current != self.damper_target {
            let delta = self.damper_target - self.damper_current;
            self.damper_current += delta.clamp(-self.damper_step, self.damper_step);
            self.string.set_damper(self.damper_current);
        }

        // Under una corda the hammer misses one string of the group; the missed
        // string keeps ringing from whatever is already in its banks.
        let struck = if self.hammer.una_corda() {
            (self.string.string_count() - 1).max(1)
        } else {
            self.string.string_count()
        };
        if self.hammer.is_active() {
            for s in 0..struck {
                // Small timing skew across the group: the hammer is not
                // perfectly square to the strings.
                let skew = s * MAX_SKEW_SAMPLES / self.string.string_count().max(1);
                let share = self.string.strike_share(s);
                self.hammer
                    .add_pulse(self.string.excitation_mut(s), skew, share);
            }
            self.hammer.advance(BLOCK);
        }

        // Undamped strings pick up the rest of the instrument.
        if self.damper_current < 1.0 {
            let mut drive = [0.0f32; BLOCK];
            bus.drive(&self.previous_out, &mut drive);
            let gain = 1.0 - self.damper_current;
            self.string.add_excitation_all(&drive, gain);
        }

        self.string.process(out);
        self.previous_out.copy_from_slice(out);
        self.previous_silent = false;
        true
    }

    /// Immediate silence, used by `AllOff`.
    pub fn reset(&mut self) {
        self.held = false;
        self.hammer.reset();
        self.string.reset();
        self.previous_out.fill(0.0);
        self.previous_silent = true;
        self.damper_current = if crate::pedal::has_damper(self.key) { 1.0 } else { 0.0 };
        self.damper_target = self.damper_current;
        self.string.set_damper(self.damper_current);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn voice(key: u8) -> Voice {
        Voice::new(key, &Preset::default())
    }

    fn bus() -> ResonanceBus {
        ResonanceBus::new(Preset::default().voicing.resonance_coupling)
    }

    fn rms(v: &[f32]) -> f32 {
        (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt()
    }

    /// Renders `blocks` blocks and returns the RMS of the last one.
    fn render(voice: &mut Voice, blocks: usize) -> f32 {
        let bus = bus();
        let mut out = [0.0f32; BLOCK];
        for _ in 0..blocks {
            if !voice.process(&mut out, &bus) {
                out.fill(0.0);
            }
        }
        rms(&out)
    }

    #[test]
    fn a_fresh_voice_is_idle_and_silent() {
        let mut v = voice(60);
        assert!(v.is_idle());
        assert_eq!(render(&mut v, 10), 0.0);
    }

    #[test]
    fn note_on_makes_sound_and_note_off_stops_it() {
        let pedals = PedalState::new();
        let mut v = voice(60);
        v.note_on(90, &pedals);
        let struck = render(&mut v, 200);
        assert!(struck > 0.0);
        v.note_off(&pedals);
        // 0.5 s of damped decay must lose more than 40 dB.
        let after = render(&mut v, (0.5 * crate::types::SAMPLE_RATE / BLOCK as f32) as usize);
        assert!(
            after < struck * 0.005,
            "after note-off {after} vs struck {struck}"
        );
    }

    #[test]
    fn harder_strikes_are_louder() {
        let pedals = PedalState::new();
        let mut soft = voice(60);
        let mut hard = voice(60);
        soft.note_on(40, &pedals);
        hard.note_on(110, &pedals);
        assert!(render(&mut hard, 100) > render(&mut soft, 100) * 2.0);
    }

    #[test]
    fn undamped_treble_keeps_ringing_after_release() {
        let pedals = PedalState::new();
        let mut v = voice(96); // C7, above the damper break
        v.note_on(100, &pedals);
        render(&mut v, 50);
        v.note_off(&pedals);
        assert!(render(&mut v, 10) > 0.0);
    }

    /// A silent string must still run when its dampers are up and the bus has
    /// something to give it — that is what sympathetic resonance is — and must
    /// be skipped otherwise, which is what keeps the 88 voices affordable.
    #[test]
    fn a_silent_voice_runs_only_when_the_bus_can_reach_it() {
        let mut pedals = PedalState::new();
        let mut v = voice(60);
        let mut out = [0.0f32; BLOCK];

        let mut quiet = bus();
        quiet.begin_block();
        assert!(!quiet.is_active());
        assert!(!v.process(&mut out, &quiet));

        let mut loud = bus();
        loud.contribute(&[0.01; BLOCK]);
        loud.begin_block();
        assert!(loud.is_active());
        // Dampers still down: the bus cannot reach the string.
        assert!(!v.process(&mut out, &loud));

        pedals.set_sustain(1.0);
        v.update_dampers(&pedals);
        assert!(v.process(&mut out, &loud));
        assert!(out.iter().any(|&x| x != 0.0), "no sympathetic response");
    }

    #[test]
    fn restrike_does_not_reset_the_ringing_string() {
        let pedals = PedalState::new();
        let mut v = voice(60);
        v.note_on(100, &pedals);
        render(&mut v, 100);
        let before = v.string().energy();
        v.note_on(1, &pedals);
        assert!(v.string().energy() >= before * 0.5);
    }
}
