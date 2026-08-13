//! The engine: 88 voices, the event queue, and the block loop.
//!
//! `Engine::process` is the single rendering path — the cpal callback and the
//! offline renderer both go through it, which is what makes offline spectral
//! tests say something about what you actually hear.

use crate::pedal::PedalState;
use crate::preset::Preset;
use crate::resonance::ResonanceBus;
use crate::soundboard::Soundboard;
use crate::types::{index_to_note, key_index, Event, PedalEvent, BLOCK, NUM_KEYS};
use crate::voice::Voice;

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
        let voices = (0..NUM_KEYS)
            .map(|i| Voice::new(index_to_note(i), preset))
            .collect();
        let engine = Engine {
            voices,
            pedals: PedalState::new(),
            resonance: ResonanceBus::new(preset.voicing.resonance_coupling),
            soundboard: Soundboard::new(&preset.soundboard),
            events: consumer,
            voice_out: [0.0; BLOCK],
            held: [false; NUM_KEYS],
            spill_l: [0.0; BLOCK],
            spill_r: [0.0; BLOCK],
            spill_pos: BLOCK,
        };
        (engine, EventSender { producer })
    }

    pub fn pedals(&self) -> &PedalState {
        &self.pedals
    }

    pub fn resonance_mut(&mut self) -> &mut ResonanceBus {
        &mut self.resonance
    }

    pub fn soundboard_mut(&mut self) -> &mut Soundboard {
        &mut self.soundboard
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
            Event::NoteOn { key, vel } => {
                if let Some(i) = key_index(key) {
                    self.held[i] = true;
                    self.voices[i].note_on(vel, &self.pedals);
                }
            }
            Event::NoteOff { key } => {
                if let Some(i) = key_index(key) {
                    self.held[i] = false;
                    self.voices[i].note_off(&self.pedals);
                }
            }
            Event::Pedal(p) => self.handle_pedal(p),
            Event::AllOff => {
                self.pedals.reset();
                self.resonance.reset();
                self.soundboard.reset();
                self.held = [false; NUM_KEYS];
                for v in &mut self.voices {
                    v.reset();
                }
            }
        }
    }

    fn handle_pedal(&mut self, pedal: PedalEvent) {
        match pedal {
            PedalEvent::Sustain(value) => self.pedals.set_sustain(value),
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
            // bus has something for them to pick up.
            if v.process(&mut self.voice_out, &self.resonance) {
                self.resonance.contribute(&self.voice_out);
                self.soundboard.add_voice(&self.voice_out, v.pan());
            }
        }
        self.soundboard.process(out_l, out_r);
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

    #[test]
    fn all_off_silences_everything() {
        let (mut engine, _tx) = engine();
        for key in [48u8, 55, 60, 64] {
            engine.handle_event(Event::NoteOn { key, vel: 100 });
        }
        let (mut l, mut r) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
        engine.process(&mut l, &mut r);
        engine.handle_event(Event::AllOff);
        engine.process(&mut l, &mut r);
        assert_eq!(engine.active_voices(), 0);
        assert!(l.iter().chain(r.iter()).all(|&v| v == 0.0));
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
