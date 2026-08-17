//! The engine: 88 voices, the event queue, and the block loop.
//!
//! `Engine::process` is the single rendering path — the cpal callback and the
//! offline renderer both go through it, which is what makes offline spectral
//! tests say something about what you actually hear.

use crate::calibrate::{MechanismCalibration, PEDAL_REFERENCE_KEY};
use crate::noise::{self, Burst, EventModel, NoiseShapes};
use crate::pedal::{has_damper, PedalState};
use crate::preset::Preset;
use crate::resonance::ResonanceBus;
use crate::soundboard::Soundboard;
use crate::types::{
    index_to_note, key_index, Event, PedalEvent, BLOCK, ESCAPEMENT_VELOCITY, NUM_KEYS,
};
use crate::voice::Voice;

/// Sustain-pedal position at which the damper rail is taken to move.
///
/// The pedal is continuous and the dampers follow it continuously, but the
/// *tray* only makes its noise once per gesture: the rumble belongs to the
/// crossing, not to the position. Half-way is where the felt leaves the strings
/// in earnest, so that is where the crossing is detected.
const PEDAL_NOISE_THRESHOLD: f32 = 0.5;

/// Events the UI thread may queue ahead of the audio thread. Two blocks of
/// slack at any realistic event rate.
pub const EVENT_QUEUE_CAPACITY: usize = 1024;

/// UI-thread handle for pushing events to a running engine.
pub struct EventSender {
    producer: rtrb::Producer<Event>,
}

impl EventSender {
    /// Returns false if the queue is full; never blocks and never allocates.
    pub fn send(&mut self, event: Event) -> bool {
        self.producer.push(event).is_ok()
    }
}

pub struct Engine {
    voices: Vec<Voice>,
    pedals: PedalState,
    resonance: ResonanceBus,
    soundboard: Soundboard,
    events: rtrb::Consumer<Event>,
    voice_out: [f32; BLOCK],
    held: [bool; NUM_KEYS],
    /// The two pedal-tray events. Two bursts and not one: the pedal-down rumble
    /// runs for nearly six seconds, and a pianist who lifts the pedal inside
    /// that has made two sounds, not replaced one.
    pedal_down: (EventModel, Burst),
    pedal_up: (EventModel, Burst),
    pedal_out: [f32; BLOCK],
    /// Frames rendered since the engine was built. Seeds the mechanism noise,
    /// which is why the same event list renders the same samples every time.
    frame: u64,
    /// Frames of the last rendered block that did not fit in the caller's
    /// buffer. The engine's state always advances a whole `BLOCK` at a time, so
    /// a request that is not a multiple of `BLOCK` leaves a remainder that the
    /// next call must emit first — dropping it would both click and slip time.
    spill_l: [f32; BLOCK],
    spill_r: [f32; BLOCK],
    /// Read position in the spill; `BLOCK` means empty.
    spill_pos: usize,
}

impl Engine {
    /// Builds the engine and its event queue from a preset. Everything the
    /// audio thread will ever touch is allocated here; the preset is read only
    /// during construction and is not retained.
    pub fn new(preset: &Preset) -> (Engine, EventSender) {
        let (producer, consumer) = rtrb::RingBuffer::new(EVENT_QUEUE_CAPACITY);
        // Built once and copied into the voices: normalising an event's filter
        // chain and measuring what this preset's strike puts out are the two
        // expensive things in this construction.
        let shapes = NoiseShapes::new(&preset.noise);
        let calibration = MechanismCalibration::new(preset, &shapes);
        let voices = (0..NUM_KEYS)
            .map(|i| Voice::new(index_to_note(i), preset, &shapes, &calibration))
            .collect();
        let engine = Engine {
            voices,
            pedals: PedalState::new(),
            resonance: ResonanceBus::from_preset(preset),
            soundboard: Soundboard::with_mics(&preset.soundboard, preset.voicing.mics.as_ref()),
            events: consumer,
            voice_out: [0.0; BLOCK],
            held: [false; NUM_KEYS],
            // The pedal belongs to no key, so its level is quoted against the
            // strike the tuner quotes it against: C4's (`calibrate.rs`).
            pedal_down: (
                EventModel::new(
                    &preset.noise.pedal_down,
                    shapes.pedal_down,
                    PEDAL_REFERENCE_KEY,
                    noise::NOMINAL_PEDAL_DRIVE,
                    calibration.pedal_down(),
                ),
                Burst::new(),
            ),
            pedal_up: (
                EventModel::new(
                    &preset.noise.pedal_up,
                    shapes.pedal_up,
                    PEDAL_REFERENCE_KEY,
                    noise::NOMINAL_PEDAL_DRIVE,
                    calibration.pedal_up(),
                ),
                Burst::new(),
            ),
            pedal_out: [0.0; BLOCK],
            frame: 0,
            spill_l: [0.0; BLOCK],
            spill_r: [0.0; BLOCK],
            spill_pos: BLOCK,
        };
        (engine, EventSender { producer })
    }

    pub fn pedals(&self) -> &PedalState {
        &self.pedals
    }

    pub fn voice(&self, index: usize) -> &Voice {
        &self.voices[index]
    }

    /// Number of voices currently producing sound.
    pub fn active_voices(&self) -> usize {
        self.voices.iter().filter(|v| !v.is_idle()).count()
    }

    /// Applies an event immediately. The audio thread reaches this through the
    /// queue; the offline renderer calls it directly at its scheduled times.
    pub fn handle_event(&mut self, event: Event) {
        match event {
            // Any nonzero velocity throws the hammer: velocity 1 is a real
            // pianissimo note (MAESTRO performances contain them at 1-3), so no
            // sounding velocity is ever reinterpreted as a silent press. The
            // silent press is spelled explicitly — KeyDown, or velocity 0 —
            // and the key still counts as held either way, so it is still what
            // sostenuto captures and still what the pedal state answers about.
            Event::NoteOn { key, vel } if vel > 0 => {
                if let Some(i) = key_index(key) {
                    self.held[i] = true;
                    self.voices[i].note_on(vel, &self.pedals, self.frame);
                }
            }
            Event::NoteOn { key, vel } => self.key_down(key, vel),
            Event::KeyDown { key } => self.key_down(key, ESCAPEMENT_VELOCITY),
            Event::NoteOff { key, vel } => {
                if let Some(i) = key_index(key) {
                    self.held[i] = false;
                    self.voices[i].note_off(vel, &self.pedals, self.frame);
                }
            }
            Event::Pedal(p) => self.handle_pedal(p),
            Event::AllOff => {
                self.pedals.reset();
                self.resonance.reset();
                self.soundboard.reset();
                self.held = [false; NUM_KEYS];
                self.pedal_down.1.reset();
                self.pedal_up.1.reset();
                self.pedal_out.fill(0.0);
                for v in &mut self.voices {
                    v.reset();
                }
            }
        }
    }

    fn key_down(&mut self, key: u8, vel: u16) {
        if let Some(i) = key_index(key) {
            self.held[i] = true;
            self.voices[i].key_down(vel, &self.pedals, self.frame);
        }
    }

    fn handle_pedal(&mut self, pedal: PedalEvent) {
        match pedal {
            PedalEvent::Sustain(value) => {
                let before = self.pedals.sustain();
                self.pedals.set_sustain(value);
                self.tray_noise(before, self.pedals.sustain());
            }
            PedalEvent::Sostenuto(on) => {
                let held = self.held;
                self.pedals.set_sostenuto(on, &held);
            }
            PedalEvent::UnaCorda(on) => self.pedals.set_una_corda(on),
        }
        for v in &mut self.voices {
            v.update_dampers(&self.pedals);
        }
    }

    /// Fires the pedal tray's noise when the sustain pedal crosses the middle,
    /// scaled by how much of the damper rail actually moves.
    ///
    /// A pedal pressed under a fully held chord lifts nothing and is nearly
    /// silent; one pressed over an empty keyboard moves all seventy dampers.
    /// That is the same "one shared event scaled by how many dampers actually
    /// moved" `PHYSICS.md` §5 asks for, and it is why a repeated pedal on a
    /// sustained chord does not machine-gun.
    fn tray_noise(&mut self, before: f32, after: f32) {
        let crossed_down = before < PEDAL_NOISE_THRESHOLD && after >= PEDAL_NOISE_THRESHOLD;
        let crossed_up = before >= PEDAL_NOISE_THRESHOLD && after < PEDAL_NOISE_THRESHOLD;
        if !(crossed_down || crossed_up) {
            return;
        }
        let drive = self.moving_dampers();
        let seed = noise::seed_of(0, self.frame);
        if crossed_down {
            let (model, burst) = &mut self.pedal_down;
            burst.trigger(model, drive, seed);
        } else {
            let (model, burst) = &mut self.pedal_up;
            burst.trigger(model, drive, seed);
        }
    }

    /// Fraction of the instrument's dampers that the sustain pedal can move:
    /// the ones that are neither held down by a key nor caught by sostenuto.
    fn moving_dampers(&self) -> f32 {
        let mut movable = 0usize;
        let mut moving = 0usize;
        for (i, &held) in self.held.iter().enumerate() {
            if !has_damper(index_to_note(i)) {
                continue;
            }
            movable += 1;
            if !held && !self.pedals.is_captured(i) {
                moving += 1;
            }
        }
        moving as f32 / movable.max(1) as f32
    }

    /// Renders `out_l.len()` frames, draining pending events before each block
    /// it renders. Any request length is accepted: the output is the same
    /// sample stream however it is cut up. Allocation-free and lock-free: safe
    /// to call from the audio callback.
    pub fn process(&mut self, out_l: &mut [f32], out_r: &mut [f32]) {
        debug_assert_eq!(out_l.len(), out_r.len());
        let frames = out_l.len();
        let mut done = self.drain_spill(out_l, out_r);
        while frames - done >= BLOCK {
            self.drain_events();
            let end = done + BLOCK;
            self.process_block(&mut out_l[done..end], &mut out_r[done..end]);
            done = end;
        }
        if done < frames {
            self.drain_events();
            let (mut block_l, mut block_r) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
            self.process_block(&mut block_l, &mut block_r);
            self.spill_l = block_l;
            self.spill_r = block_r;
            self.spill_pos = 0;
            done += self.drain_spill(&mut out_l[done..], &mut out_r[done..]);
        }
        debug_assert_eq!(done, frames);
    }

    fn drain_events(&mut self) {
        while let Ok(event) = self.events.pop() {
            self.handle_event(event);
        }
    }

    /// Copies as much of the pending remainder as fits, returning how many
    /// frames it wrote.
    fn drain_spill(&mut self, out_l: &mut [f32], out_r: &mut [f32]) -> usize {
        let n = (BLOCK - self.spill_pos).min(out_l.len());
        let end = self.spill_pos + n;
        out_l[..n].copy_from_slice(&self.spill_l[self.spill_pos..end]);
        out_r[..n].copy_from_slice(&self.spill_r[self.spill_pos..end]);
        self.spill_pos = end;
        n
    }

    fn process_block(&mut self, out_l: &mut [f32], out_r: &mut [f32]) {
        self.resonance.begin_block();
        self.soundboard.begin_block();
        for v in &mut self.voices {
            // The voice decides whether it has anything to render: silent
            // strings still run when their dampers are up and the resonance
            // bus has something for them to pick up. It places itself on the
            // board — a voice whose polarizations are panned apart arrives
            // there as two signals, and only the voice knows where they go —
            // and hands back its mono sum for the bus.
            if v.process(&mut self.voice_out, &self.resonance, &mut self.soundboard) {
                self.resonance.contribute(&self.voice_out);
            }
        }
        // The pedal tray is not a key: its noise arrives at the centre of the
        // instrument and, like the per-key mechanism noise, reaches the board
        // without passing through the sympathetic bus.
        if self.pedal_down.1.is_active() || self.pedal_up.1.is_active() {
            self.pedal_out.fill(0.0);
            self.pedal_down.1.add(&mut self.pedal_out);
            self.pedal_up.1.add(&mut self.pedal_out);
            self.soundboard.add_voice(&self.pedal_out, 0.0);
        }
        self.soundboard.process(out_l, out_r);
        self.frame += BLOCK as u64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> (Engine, EventSender) {
        Engine::new(&Preset::default())
    }

    #[test]
    fn an_idle_engine_renders_exact_silence() {
        let (mut engine, _tx) = engine();
        let (mut l, mut r) = ([1.0f32; 1000], [1.0f32; 1000]);
        engine.process(&mut l, &mut r);
        assert!(l.iter().chain(r.iter()).all(|&v| v == 0.0));
        assert_eq!(engine.active_voices(), 0);
    }

    #[test]
    fn queued_events_reach_the_voices() {
        let (mut engine, mut tx) = engine();
        assert!(tx.send(Event::NoteOn { key: 60, vel: 90 }));
        let (mut l, mut r) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
        engine.process(&mut l, &mut r);
        assert_eq!(engine.active_voices(), 1);
        assert!(l.iter().any(|v| v.abs() > 0.0));
        assert!(r.iter().any(|v| v.abs() > 0.0));
    }

    /// The stream a caller hears must not depend on how it is cut into calls:
    /// a device that asks for 100 frames at a time has to get exactly what a
    /// device asking for 128 gets, with no dropped samples at the seams.
    #[test]
    fn the_output_does_not_depend_on_the_request_length() {
        let render = |chunk: usize| {
            let (mut engine, _tx) = engine();
            engine.handle_event(Event::NoteOn { key: 60, vel: 90 });
            let frames = 20 * BLOCK;
            let (mut l, mut r) = (vec![0.0f32; frames], vec![0.0f32; frames]);
            let mut start = 0;
            while start < frames {
                let end = (start + chunk).min(frames);
                engine.process(&mut l[start..end], &mut r[start..end]);
                start = end;
            }
            (l, r)
        };
        let (ref_l, ref_r) = render(BLOCK);
        assert!(ref_l.iter().any(|v| v.abs() > 1e-6));
        for chunk in [1, 70, 100, 129, 333] {
            let (l, r) = render(chunk);
            assert_eq!(l, ref_l, "left channel differs at chunk {chunk}");
            assert_eq!(r, ref_r, "right channel differs at chunk {chunk}");
        }
    }

    /// Everything means everything, including the mechanism: a panic in the
    /// middle of a pedal-down rumble has to leave digital silence, not six
    /// seconds of it.
    #[test]
    fn all_off_silences_everything() {
        let (mut engine, _tx) = engine();
        for key in [48u8, 55, 60, 64] {
            engine.handle_event(Event::NoteOn { key, vel: 100 });
        }
        engine.handle_event(Event::Pedal(PedalEvent::Sustain(1.0)));
        let (mut l, mut r) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
        engine.process(&mut l, &mut r);
        for key in [48u8, 55, 60, 64] {
            engine.handle_event(Event::NoteOff { key, vel: 100 });
        }
        engine.process(&mut l, &mut r);
        engine.handle_event(Event::AllOff);
        engine.process(&mut l, &mut r);
        assert_eq!(engine.active_voices(), 0);
        assert!(l.iter().chain(r.iter()).all(|&v| v == 0.0));
    }

    /// The reason the silent press exists: sostenuto's tab rail catches damper
    /// levers that are already raised, and a key held down without striking has
    /// raised one. `pedal.rs` already reads "physically held", so this is a
    /// check that the new event reaches that state rather than a new mechanism.
    #[test]
    fn sostenuto_captures_a_key_that_was_pressed_without_striking() {
        for prepare in [
            Event::KeyDown { key: 48 },
            // ... and the same gesture expressed as a velocity-zero note-on.
            Event::NoteOn { key: 48, vel: 0 },
        ] {
            let (mut engine, _tx) = engine();
            engine.handle_event(prepare);
            engine.handle_event(Event::Pedal(PedalEvent::Sostenuto(true)));
            engine.handle_event(Event::NoteOff { key: 48, vel: 64 });
            assert!(
                engine.pedals().is_captured(key_index(48).unwrap()),
                "{prepare:?} was not captured"
            );
            assert_eq!(engine.active_voices(), 0, "{prepare:?} struck the string");
        }
    }

    /// The tray noise belongs to the crossing, not to the position: a pedal
    /// worked continuously above or below the threshold does not re-fire it,
    /// which is what stops a Disklavier recording's continuous CC 64 from
    /// machine-gunning.
    #[test]
    fn the_pedal_tray_sounds_once_per_crossing() {
        let render = |values: &[f32]| {
            let (mut engine, _tx) = engine();
            let (mut l, mut r) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
            let mut peak = 0.0f32;
            for &v in values {
                engine.handle_event(Event::Pedal(PedalEvent::Sustain(v)));
                engine.process(&mut l, &mut r);
                for i in 0..BLOCK {
                    peak = peak.max(l[i].abs()).max(r[i].abs());
                }
            }
            peak
        };
        // Creeping up to the threshold and no further: the dampers move, the
        // tray has not gone over.
        assert_eq!(render(&[0.1, 0.2, 0.3, 0.49]), 0.0);
        assert!(render(&[0.1, 0.2, 0.3, 0.5]) > 0.0);
        // Going over once and then working the pedal above the threshold is
        // one sound: bit for bit the same render as holding it there.
        let once = render(&[1.0, 1.0, 1.0, 1.0, 1.0]);
        let worked = render(&[1.0, 0.9, 1.0, 0.8, 1.0]);
        assert_eq!(
            once, worked,
            "working the pedal above the threshold re-fired the tray"
        );
        // ... and going back under it and over again is two.
        assert!(render(&[1.0, 0.0, 1.0, 1.0, 1.0]) > once);
    }

    #[test]
    fn sostenuto_captures_only_keys_held_at_pedal_down() {
        let (mut engine, _tx) = engine();
        engine.handle_event(Event::NoteOn { key: 48, vel: 80 });
        engine.handle_event(Event::Pedal(PedalEvent::Sostenuto(true)));
        engine.handle_event(Event::NoteOn { key: 60, vel: 80 });
        assert!(engine.pedals().is_captured(key_index(48).unwrap()));
        assert!(!engine.pedals().is_captured(key_index(60).unwrap()));
    }
}
