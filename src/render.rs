//! Offline rendering.
//!
//! Runs a timed event list through exactly the same `Engine::process` the audio
//! callback uses, so a WAV rendered here is what the device would have played.

use crate::engine::Engine;
use crate::types::{Event, PedalEvent, BLOCK, HIGHEST_KEY, LOWEST_KEY, SAMPLE_RATE};
use std::path::Path;

/// An event with the time, in seconds from the start of the render, at which
/// it should be applied.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderEvent {
    pub time_s: f32,
    pub event: Event,
}

impl RenderEvent {
    pub fn new(time_s: f32, event: Event) -> Self {
        RenderEvent { time_s, event }
    }
}

/// Renders `duration_s` of audio, applying each event at the block boundary
/// that contains its timestamp.
pub fn render_to_buffer(events: &[RenderEvent], duration_s: f32) -> (Vec<f32>, Vec<f32>) {
    let mut schedule: Vec<RenderEvent> = events.to_vec();
    schedule.sort_by(|a, b| a.time_s.total_cmp(&b.time_s));

    let frames = (duration_s * SAMPLE_RATE).max(0.0) as usize;
    let mut left = vec![0.0f32; frames];
    let mut right = vec![0.0f32; frames];

    let (mut engine, _sender) = Engine::new();
    let mut next = 0usize;
    let mut start = 0usize;
    while start < frames {
        let end = (start + BLOCK).min(frames);
        let block_end_s = end as f32 / SAMPLE_RATE;
        while next < schedule.len() && schedule[next].time_s < block_end_s {
            engine.handle_event(schedule[next].event);
            next += 1;
        }
        engine.process(&mut left[start..end], &mut right[start..end]);
        start = end;
    }
    (left, right)
}

/// Renders to a 32-bit float stereo WAV at the engine's sample rate.
pub fn render_to_wav(
    path: &Path,
    events: &[RenderEvent],
    duration_s: f32,
) -> Result<(), hound::Error> {
    let (left, right) = render_to_buffer(events, duration_s);
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: SAMPLE_RATE as u32,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for (&l, &r) in left.iter().zip(&right) {
        writer.write_sample(l)?;
        writer.write_sample(r)?;
    }
    writer.finalize()
}

/// Small builder that keeps the sequences below readable.
struct Sequence {
    events: Vec<RenderEvent>,
}

impl Sequence {
    fn new() -> Self {
        Sequence { events: Vec::new() }
    }

    fn note(&mut self, at: f32, key: u8, vel: u8, dur: f32) -> &mut Self {
        self.events.push(RenderEvent::new(at, Event::NoteOn { key, vel }));
        self.events
            .push(RenderEvent::new(at + dur, Event::NoteOff { key }));
        self
    }

    fn chord(&mut self, at: f32, keys: &[u8], vel: u8, dur: f32) -> &mut Self {
        for &key in keys {
            self.note(at, key, vel, dur);
        }
        self
    }

    fn pedal(&mut self, at: f32, pedal: PedalEvent) -> &mut Self {
        self.events.push(RenderEvent::new(at, Event::Pedal(pedal)));
        self
    }

    fn finish(&mut self) -> Vec<RenderEvent> {
        self.events
            .sort_by(|a, b| a.time_s.total_cmp(&b.time_s));
        std::mem::take(&mut self.events)
    }
}

/// Length of [`demo_sequence`] including the final decay.
pub const DEMO_DURATION_S: f32 = 15.0;

/// Length of [`default_sequence`] including the final decay.
pub const DEFAULT_DURATION_S: f32 = 12.0;

/// A ~15 s piece in A minor that puts every part of the model on show: a
/// pedalled legato phrase, dry staccato chords, a crescendo from pianissimo to
/// fortissimo across three octaves, an una corda passage, a half-pedal, and a
/// sostenuto-held bass under detached chords.
pub fn demo_sequence() -> Vec<RenderEvent> {
    let mut s = Sequence::new();

    // 1. Legato melody over a pedalled bass: overlapping notes, the pedal
    //    changing with the harmony, mezzo-piano.
    s.pedal(0.0, PedalEvent::Sustain(1.0));
    s.chord(0.05, &[45, 52], 62, 1.8);
    for &(at, key, vel, dur) in &[
        (0.15f32, 69u8, 64u8, 0.55f32),
        (0.60, 72, 66, 0.50),
        (1.00, 76, 70, 0.55),
        (1.50, 74, 68, 0.50),
        (1.95, 72, 64, 0.75),
    ] {
        s.note(at, key, vel, dur);
    }
    s.pedal(1.90, PedalEvent::Sustain(0.0));
    s.pedal(1.95, PedalEvent::Sustain(1.0));
    s.chord(2.00, &[41, 48], 60, 1.8);
    s.note(2.60, 71, 66, 0.45);
    s.note(3.00, 69, 70, 0.95);
    s.pedal(3.90, PedalEvent::Sustain(0.0));

    // 2. Staccato, pedal up: short chords with clean damper cut-offs.
    for (i, keys) in [
        [45u8, 57, 60, 64],
        [44, 56, 59, 62],
        [45, 57, 60, 64],
        [44, 56, 59, 62],
    ]
    .iter()
    .enumerate()
    {
        s.chord(4.00 + i as f32 * 0.35, keys, if i % 2 == 0 { 88 } else { 80 }, 0.09);
    }
    s.chord(5.40, &[45, 57, 60, 64], 96, 0.30);

    // 3. Crescendo pianissimo to fortissimo, three octaves of A minor under
    //    the pedal, landing on a fortissimo chord.
    s.pedal(5.85, PedalEvent::Sustain(1.0));
    for (i, &key) in [45u8, 52, 57, 60, 64, 69, 72, 76, 81, 84, 88, 93]
        .iter()
        .enumerate()
    {
        let vel = 26 + (i as f32 * 8.4) as u8;
        s.note(5.90 + i as f32 * 0.13, key, vel, 0.5);
    }
    s.chord(7.60, &[69, 72, 76, 81], 118, 1.0);
    s.pedal(8.70, PedalEvent::Sustain(0.0));

    // 4. Una corda, pianissimo: two strings per note, softer and darker felt.
    s.pedal(8.75, PedalEvent::UnaCorda(true));
    s.pedal(8.80, PedalEvent::Sustain(1.0));
    s.chord(8.85, &[40, 52], 40, 0.80);
    for (i, &key) in [69u8, 67, 65, 64].iter().enumerate() {
        s.note(9.00 + i as f32 * 0.40, key, 34 - i as u8 * 2, 0.45);
    }
    s.chord(10.50, &[36, 48], 36, 1.00);
    // Half pedal thins the wash without cutting it off.
    s.pedal(11.15, PedalEvent::Sustain(0.45));
    s.pedal(11.50, PedalEvent::Sustain(0.0));
    s.pedal(11.55, PedalEvent::UnaCorda(false));

    // 5. Sostenuto holds the bass octave while the hands play detached chords
    //    over it with the sustain pedal up, then a fortissimo close.
    s.chord(11.60, &[33, 45], 100, 0.55);
    s.pedal(11.75, PedalEvent::Sostenuto(true));
    for (i, keys) in [[60u8, 64, 67], [62, 65, 69], [64, 67, 72]].iter().enumerate() {
        s.chord(12.20 + i as f32 * 0.35, keys, 75 + i as u8 * 4, 0.12);
    }
    s.pedal(13.25, PedalEvent::Sostenuto(false));
    s.pedal(13.30, PedalEvent::Sustain(1.0));
    s.chord(13.35, &[33, 45, 57, 60, 64, 69], 112, 1.20);

    s.finish()
}

/// Default sequence for `render <file.wav>`: a sweep of the whole compass in
/// minor thirds followed by a chromatic octave, which is what you want to hear
/// when checking that the instrument is even across the keyboard.
pub fn default_sequence() -> Vec<RenderEvent> {
    let mut s = Sequence::new();
    let mut t = 0.05;
    for key in (LOWEST_KEY..=HIGHEST_KEY).step_by(3) {
        s.note(t, key, 80, 0.45);
        t += 0.22;
    }
    // Chromatic octave down through the middle of the keyboard, quicker.
    t += 0.3;
    for key in (60u8..=72).rev() {
        s.note(t, key, 92, 0.24);
        t += 0.11;
    }
    // ... and a pedalled chord to hear the board and the sympathetic halo.
    s.pedal(t + 0.1, PedalEvent::Sustain(1.0));
    s.chord(t + 0.15, &[36, 48, 55, 60, 64, 67], 96, 0.8);
    s.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_sequence_renders_exact_silence() {
        let (l, r) = render_to_buffer(&[], 0.5);
        assert_eq!(l.len(), (0.5 * SAMPLE_RATE) as usize);
        assert!(l.iter().chain(r.iter()).all(|&v| v == 0.0));
    }

    #[test]
    fn a_struck_note_produces_bounded_audio() {
        let events = [RenderEvent::new(0.0, Event::NoteOn { key: 60, vel: 90 })];
        let (l, r) = render_to_buffer(&events, 1.0);
        let peak = l
            .iter()
            .chain(r.iter())
            .fold(0.0f32, |m, &v| m.max(v.abs()));
        assert!(peak > 0.01, "too quiet: peak {peak}");
        assert!(peak <= 1.0, "clipped: peak {peak}");
        assert!(l.iter().chain(r.iter()).all(|v| v.is_finite()));
    }

    #[test]
    fn demo_sequence_fits_its_declared_duration() {
        let events = demo_sequence();
        assert!(events.windows(2).all(|w| w[0].time_s <= w[1].time_s));
        assert!(events.last().unwrap().time_s < DEMO_DURATION_S);
        assert!(default_sequence().last().unwrap().time_s < DEFAULT_DURATION_S);
    }
}
