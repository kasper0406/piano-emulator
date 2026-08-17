//! The live wire format: Universal MIDI Packets, and MIDI 1.0 bytes.
//!
//! This is the only place that knows what a keyboard actually sends. It has no
//! platform in it — [`crate::midi::live`] owns Core MIDI, and this owns the
//! bytes and words — so the whole protocol surface is testable without a cable,
//! and the same parser serves the standalone today and the AUv3's
//! `AUMIDIEventListBlock` when `DISTRIBUTION.md` M2 lands (host-delivered UMP is
//! the same words).
//!
//! **What is read** — the same set `midi.rs` reads out of a file, because the
//! instrument is the same instrument:
//!
//! - note on / note off on every channel and every group, merged;
//! - release velocity, which the engine plays as how fast the damper falls;
//! - CC 64 as a *continuous* sustain pedal (the live path slews it: see
//!   [`crate::midi::live::Slew`]);
//! - CC 66 sostenuto and CC 67 una corda as switches;
//! - MIDI 2.0's 16-bit velocity and 32-bit controller values, at full
//!   resolution, into the fine lane of [`velocity_from_midi`].
//!
//! Everything else — pitch bend, aftertouch, program change, per-note
//! controllers, sysex, the whole utility and stream layer — is ignored, exactly
//! as in the file reader. `DISTRIBUTION.md`'s M9 is where the Piano Profile's
//! registered controllers arrive; there is nothing here to change first.
//!
//! ## The two protocols say "velocity 0" differently
//!
//! In MIDI 1.0 a note on at velocity 0 **is a note off** — the running-status
//! trick — and it is read as one, with the nominal release velocity, exactly as
//! `midi.rs` reads it in a file. In MIDI 2.0 that trick is gone: the protocol
//! has a real note off, so a note on at velocity 0 means what it says, and the
//! engine already has a gesture for it — the silent press of `PHYSICS.md` §6,
//! spelled `Event::NoteOn { vel: 0 }`. A UMP source can therefore prepare a
//! string from the keyboard, which no MIDI 1.0 controller can express.
//!
//! (Core MIDI's own 1.0 → 2.0 translation agrees: a `0x90 k 0` on the wire
//! arrives as a **note off** UMP with release velocity `0x8000`, not as a
//! 2.0 note on at zero. The fixtures in this module's tests are captured from
//! it.)

use crate::types::{
    hires_velocity, Event, PedalEvent, DEFAULT_RELEASE_VELOCITY, HIGHEST_KEY, LOWEST_KEY,
    MIDI1_MAX_VELOCITY,
};

/// MIDI controller numbers for the three pedals. Same three as the file reader.
const CC_SUSTAIN: u8 = 64;
const CC_SOSTENUTO: u8 = 66;
const CC_UNA_CORDA: u8 = 67;

/// UMP message types we look at. The rest are skipped by length.
const MT_MIDI1_CHANNEL_VOICE: u8 = 0x2;
const MT_MIDI2_CHANNEL_VOICE: u8 = 0x4;

/// Channel-voice status nibbles.
const STATUS_NOTE_OFF: u8 = 0x8;
const STATUS_NOTE_ON: u8 = 0x9;
const STATUS_CONTROL_CHANGE: u8 = 0xB;

/// Words in a UMP message, by message type. The table is fixed by the spec
/// (M2-104-UM §2.1.4) and is how an unknown message is skipped rather than
/// misread: length comes from the type, never from the content.
fn words_in_message(message_type: u8) -> usize {
    match message_type {
        0x0 | 0x1 | 0x2 | 0x6 | 0x7 => 1,
        0x3 | 0x4 | 0x8 | 0x9 | 0xA => 2,
        0xB | 0xC => 3,
        _ => 4,
    }
}

/// MIDI 2.0's scaling of a 7-bit value to 16 bits (M2-104-UM appendix A.2,
/// "min-center-max"): the bottom half is a plain shift, and the top half
/// repeats the source's low bits into the new ones so that 127 reaches full
/// scale and 64 lands exactly on centre.
///
/// This is not decoration: it is what Core MIDI applies when it up-translates
/// a MIDI 1.0 keyboard for a MIDI 2.0 port, so it is the function
/// [`velocity_from_ump`] has to invert if a 7-bit note is to play the note it
/// would have played on the 1.0 path.
pub fn upscale_7_to_16(value: u8) -> u16 {
    let value = value & 0x7F;
    let mut scaled = u16::from(value) << 9;
    if value <= 0x40 {
        return scaled;
    }
    // Repeat the low 6 bits upward until the new bits are filled.
    let mut repeat = u16::from(value & 0x3F) << 3;
    while repeat != 0 {
        scaled |= repeat;
        repeat >>= 6;
    }
    scaled
}

/// The continuous MIDI 1.0 velocity a 16-bit one stands for: the piecewise
/// linear inverse of [`upscale_7_to_16`], exact at all 128 of its points.
///
/// Exactness there is the whole property. A MIDI 1.0 keyboard reaching us
/// through Core MIDI's translation sends velocity 90 and arrives as 46 291; a
/// nearby scale — `v16 * 127 / 65535`, say — would turn that into 89.7 and make
/// the live path play a *different note* from the same file rendered offline.
/// This way the two paths agree exactly, and the fine resolution is the
/// interpolation between the points rather than a redefinition of them.
pub fn midi1_velocity_of(v16: u16) -> f32 {
    if v16 >= upscale_7_to_16(MIDI1_MAX_VELOCITY as u8) {
        return MIDI1_MAX_VELOCITY as f32;
    }
    // `upscale` never moves a value below `v * 512`, so the shift is an upper
    // bound on the bracket's index and at most one step of correction is
    // needed in either direction.
    let mut i = (v16 >> 9).min(MIDI1_MAX_VELOCITY) as u8;
    while i > 0 && upscale_7_to_16(i) > v16 {
        i -= 1;
    }
    while i < 127 && upscale_7_to_16(i + 1) <= v16 {
        i += 1;
    }
    let (low, high) = (upscale_7_to_16(i), upscale_7_to_16(i + 1));
    f32::from(i) + f32::from(v16 - low) / f32::from(high - low)
}

/// A MIDI 2.0 16-bit velocity as an [`Event`] velocity.
///
/// Zero stays zero — that is the silent press, and the one velocity the two
/// lanes share. Everything else lands in the fine lane, where a 7-bit velocity
/// `v` sits at exactly `v * VELOCITY_STEPS`.
pub fn velocity_from_ump(v16: u16) -> u16 {
    if v16 == 0 {
        0
    } else {
        hires_velocity(midi1_velocity_of(v16))
    }
}

/// A MIDI 2.0 32-bit controller value as the pedal position `0.0..=1.0`.
fn pedal_from_ump(v32: u32) -> f32 {
    v32 as f32 / u32::MAX as f32
}

/// A 7-bit controller value as the pedal position, as the file reader reads it.
fn pedal_from_midi1(value: u8) -> f32 {
    f32::from(value & 0x7F) / 127.0
}

/// Keys outside the 88 are dropped here rather than deeper in, so what reaches
/// the queue is exactly what the instrument will play.
fn playable(key: u8) -> Option<u8> {
    let key = key & 0x7F;
    (LOWEST_KEY..=HIGHEST_KEY).contains(&key).then_some(key)
}

/// One controller change, either protocol, as an [`Event`].
///
/// `position` is already `0.0..=1.0`; the switched pedals are down from half
/// travel, which is the 7-bit convention (64 of 127) carried over exactly.
fn controller(index: u8, position: f32) -> Option<Event> {
    let down = position >= f32::from(SWITCH_THRESHOLD) / 127.0;
    match index & 0x7F {
        CC_SUSTAIN => Some(Event::Pedal(PedalEvent::Sustain(position.clamp(0.0, 1.0)))),
        CC_SOSTENUTO => Some(Event::Pedal(PedalEvent::Sostenuto(down))),
        CC_UNA_CORDA => Some(Event::Pedal(PedalEvent::UnaCorda(down))),
        _ => None,
    }
}

/// A switched controller is down at 64 and up below it — the convention every
/// sustain-capable controller follows, and the one the file reader uses.
const SWITCH_THRESHOLD: u8 = 64;

/// One MIDI 1.0 channel-voice UMP (message type 2), as an [`Event`].
fn from_midi1_words(word: u32) -> Option<Event> {
    let status = ((word >> 20) & 0x0F) as u8;
    let d1 = ((word >> 8) & 0x7F) as u8;
    let d2 = (word & 0x7F) as u8;
    from_midi1_message(status, d1, d2)
}

/// One MIDI 1.0 channel-voice message from its status nibble and data bytes.
/// Shared by the UMP path and the byte-stream path, so there is one set of
/// rules and one place they are tested.
fn from_midi1_message(status: u8, d1: u8, d2: u8) -> Option<Event> {
    match status {
        STATUS_NOTE_ON if d2 > 0 => playable(d1).map(|key| Event::NoteOn {
            key,
            // A 7-bit source stays in the legacy lane: the number on the wire
            // is the number in the event, as it is when the same note comes
            // out of a file.
            vel: u16::from(d2),
        }),
        // Velocity 0 is the note-off half of a running-status note on.
        STATUS_NOTE_ON => playable(d1).map(|key| Event::NoteOff {
            key,
            vel: DEFAULT_RELEASE_VELOCITY,
        }),
        STATUS_NOTE_OFF => playable(d1).map(|key| Event::NoteOff {
            key,
            // Zero is "this keyboard does not measure release velocity", not
            // "released infinitely slowly".
            vel: match d2 {
                0 => DEFAULT_RELEASE_VELOCITY,
                v => u16::from(v),
            },
        }),
        STATUS_CONTROL_CHANGE => controller(d1, pedal_from_midi1(d2)),
        _ => None,
    }
}

/// One MIDI 2.0 channel-voice UMP (message type 4), as an [`Event`].
fn from_midi2_words(w0: u32, w1: u32) -> Option<Event> {
    let status = ((w0 >> 20) & 0x0F) as u8;
    let index = ((w0 >> 8) & 0x7F) as u8;
    match status {
        // The attribute (the low 16 bits of `w1`, typed by the low byte of
        // `w0`) is ignored: the only defined types are a MIDI 1.0 articulation,
        // a Profile-specific value and pitch 7.9, and none of them is something
        // this instrument models yet.
        STATUS_NOTE_ON => {
            let v16 = (w1 >> 16) as u16;
            playable(index).map(|key| Event::NoteOn {
                key,
                // Velocity 0 is a *silent press* here, not a note off: MIDI 2.0
                // has a real note off, so nothing is overloaded (module docs).
                vel: velocity_from_ump(v16),
            })
        }
        STATUS_NOTE_OFF => {
            let v16 = (w1 >> 16) as u16;
            playable(index).map(|key| Event::NoteOff {
                key,
                vel: match v16 {
                    0 => DEFAULT_RELEASE_VELOCITY,
                    v => velocity_from_ump(v),
                },
            })
        }
        STATUS_CONTROL_CHANGE => controller(index, pedal_from_ump(w1)),
        _ => None,
    }
}

/// Walks a run of Universal MIDI Packets and hands every [`Event`] in it to
/// `sink`, in order.
///
/// A packet's length comes from its message type, so an unknown or unhandled
/// message is *skipped whole* rather than misparsed — which is what makes it
/// safe to point this at a stream carrying sysex, jitter-reduction timestamps
/// or anything else the endpoint feels like sending. A truncated final message
/// ends the walk.
pub fn parse_ump(words: &[u32], mut sink: impl FnMut(Event)) {
    let mut i = 0;
    while i < words.len() {
        let message_type = (words[i] >> 28) as u8;
        let len = words_in_message(message_type);
        if i + len > words.len() {
            return;
        }
        let event = match message_type {
            MT_MIDI1_CHANNEL_VOICE => from_midi1_words(words[i]),
            MT_MIDI2_CHANNEL_VOICE => from_midi2_words(words[i], words[i + 1]),
            _ => None,
        };
        if let Some(event) = event {
            sink(event);
        }
        i += len;
    }
}

/// A MIDI 1.0 byte stream, with running status.
///
/// Core MIDI hands us UMP whenever the port asks for MIDI 2.0, so this is the
/// fallback path (`live.rs` uses it if `MIDIInputPortCreateWithProtocol` is
/// unavailable) — and it is also what a serial port, a `.syx` capture or a
/// future non-Apple backend would deliver. Running status is not optional:
/// keyboards use it for every note in a fast passage.
#[derive(Clone, Copy, Debug, Default)]
pub struct Midi1Stream {
    /// Current running status byte, or 0 when there is none.
    status: u8,
    /// Data bytes collected for the current message.
    data: [u8; 2],
    collected: usize,
}

impl Midi1Stream {
    pub fn new() -> Midi1Stream {
        Midi1Stream::default()
    }

    /// Feeds one byte and emits an [`Event`] if it completed a message we play.
    pub fn push(&mut self, byte: u8, mut sink: impl FnMut(Event)) {
        if byte >= 0xF8 {
            // System real time is interleaved anywhere, including inside a
            // message, and never disturbs running status.
            return;
        }
        if byte >= 0xF0 {
            // System common cancels running status; sysex content is not
            // reassembled, and its bytes look like data until the next status.
            self.status = 0;
            self.collected = 0;
            return;
        }
        if byte >= 0x80 {
            self.status = byte;
            self.collected = 0;
            return;
        }
        if self.status == 0 {
            return;
        }
        self.data[self.collected] = byte;
        self.collected += 1;
        let status = self.status >> 4;
        let wanted = if matches!(status, 0xC | 0xD) { 1 } else { 2 };
        if self.collected < wanted {
            return;
        }
        self.collected = 0;
        if let Some(event) = from_midi1_message(status, self.data[0], self.data[1]) {
            sink(event);
        }
    }

    /// Feeds a whole packet.
    pub fn push_bytes(&mut self, bytes: &[u8], mut sink: impl FnMut(Event)) {
        for &byte in bytes {
            self.push(byte, &mut sink);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{velocity_from_midi, VELOCITY_STEPS};

    fn events_of(words: &[u32]) -> Vec<Event> {
        let mut out = Vec::new();
        parse_ump(words, |e| out.push(e));
        out
    }

    fn bytes_of(bytes: &[u8]) -> Vec<Event> {
        let mut out = Vec::new();
        let mut stream = Midi1Stream::new();
        stream.push_bytes(bytes, |e| out.push(e));
        out
    }

    /// The 7-bit-to-16-bit scaling, against values **captured from Core MIDI's
    /// own translator** (`MIDIInputPortCreateWithProtocol(kMIDIProtocol_2_0)`
    /// fed `0x90 0x3C v` from a virtual source): if this table is wrong, every
    /// MIDI 1.0 keyboard plays the wrong velocity through the live path.
    #[test]
    fn the_upscale_is_core_midis_own() {
        for (v7, v16) in [
            (0u8, 0x0000u16),
            (1, 0x0200),
            (40, 0x5000),
            (64, 0x8000),
            (90, 0xB4D3),
            (100, 0xC924),
            (126, 0xFDF7),
            (127, 0xFFFF),
        ] {
            assert_eq!(upscale_7_to_16(v7), v16, "velocity {v7}");
        }
        // Monotone over the whole domain, which is what makes the inverse a
        // function at all.
        for v in 1..=127u8 {
            assert!(upscale_7_to_16(v) > upscale_7_to_16(v - 1));
        }
    }

    /// **The velocity map's proof.** Every old point comes back exactly, in
    /// both lanes and from both protocols, and the map is strictly increasing
    /// through the fine lane so nothing between the old points is a step.
    #[test]
    fn the_velocity_map_reproduces_every_old_point() {
        for v in 1..=127u16 {
            // Legacy lane: the number the field has always held.
            assert_eq!(velocity_from_midi(v), v as f32, "legacy lane at {v}");
            // Fine lane: the same velocity, 512 times as finely spelled.
            assert_eq!(
                velocity_from_midi(v * VELOCITY_STEPS),
                v as f32,
                "fine lane at {v}"
            );
            // And that is where a MIDI 2.0 source's rendering of the *same*
            // 7-bit velocity lands, which is the property that makes a
            // translated keyboard play the file's note and not a neighbour.
            let ump = upscale_7_to_16(v as u8);
            assert_eq!(velocity_from_ump(ump), v * VELOCITY_STEPS, "ump at {v}");
            assert_eq!(velocity_from_midi(velocity_from_ump(ump)), v as f32);
        }
        // Silent press: zero in, zero out, in every spelling.
        assert_eq!(velocity_from_ump(0), 0);
        assert_eq!(velocity_from_midi(0), 0.0);
    }

    /// **Nothing a `u8` could hold changed meaning.** The lane boundary is 255
    /// and not 127 for exactly this: a velocity above 127 is not legal MIDI,
    /// but the field could hold one and this repository *did* hold one — a
    /// velocity sweep that walks `layer * 8` to 128 — and the engine clamped it
    /// downstream. Anything that was clamped is still clamped, at the same
    /// place, to the same value.
    #[test]
    fn every_value_the_old_u8_field_could_hold_still_means_what_it_meant() {
        for v in 0..=u8::MAX {
            assert_eq!(
                velocity_from_midi(u16::from(v)),
                f32::from(v),
                "the legacy lane moved at {v}"
            );
        }
        // ... which includes the ones only the clamp ever saw.
        assert_eq!(velocity_from_midi(128), 128.0);
        assert_eq!(velocity_from_midi(255), 255.0);
        // The fine lane starts on the far side of them and never collides.
        assert!(velocity_from_ump(1) > crate::types::LEGACY_VELOCITY_MAX);
        assert_eq!(velocity_from_midi(velocity_from_ump(1)), 0.5);
    }

    /// The fine lane is genuinely finer: sixteen-bit velocities one step apart
    /// come out as distinct hammer inputs, which is the whole point of the
    /// widening (`SHIPPING.md` §4's SL88 MK2 clause).
    #[test]
    fn neighbouring_sixteen_bit_velocities_stay_distinct() {
        let base = upscale_7_to_16(90);
        let mut seen: Vec<f32> = Vec::new();
        for step in 0..64u16 {
            let v = velocity_from_midi(velocity_from_ump(base + step * 8));
            if let Some(&last) = seen.last() {
                assert!(v > last, "velocity map went flat or backwards at {step}");
            }
            seen.push(v);
        }
        // Sixty-four of them inside one MIDI 1.0 step.
        assert!(seen[63] - seen[0] < 1.0);
        assert!(seen[63] - seen[0] > 0.5);
    }

    /// Captured from the same loopback: a MIDI 1.0 keyboard, translated by the
    /// OS, must arrive as the events the file reader would have produced.
    #[test]
    fn a_translated_midi1_keyboard_plays_what_the_file_reader_plays() {
        // 0x90 0x3C 0x5A, 0x80 0x3C 0x64, 0x90 0x3C 0x00, as Core MIDI
        // re-spelled them for a MIDI 2.0 port.
        let events = events_of(&[
            0x4090_3C00,
            0xB4D3_0000,
            0x4080_3C00,
            0xC924_0000,
            0x4080_3C00,
            0x8000_0000,
        ]);
        assert_eq!(
            events,
            vec![
                Event::NoteOn {
                    key: 60,
                    vel: 90 * VELOCITY_STEPS
                },
                Event::NoteOff {
                    key: 60,
                    vel: 100 * VELOCITY_STEPS
                },
                // The note-on-at-zero the OS turned into a note off at centre.
                Event::NoteOff {
                    key: 60,
                    vel: 64 * VELOCITY_STEPS
                },
            ]
        );
        // ... and the velocities they carry are the file reader's numbers.
        for (event, want) in events.iter().zip([90.0, 100.0, 64.0]) {
            let vel = match event {
                Event::NoteOn { vel, .. } | Event::NoteOff { vel, .. } => *vel,
                _ => unreachable!(),
            };
            assert_eq!(velocity_from_midi(vel), want);
        }
    }

    /// The pedals, in both protocols. Captured 32-bit values again: Core MIDI
    /// scales CC 64 = 0/1/64/127 to exactly these.
    #[test]
    fn the_three_pedals_arrive_at_the_positions_they_were_played_at() {
        let events = events_of(&[
            0x40B0_4000,
            0x0000_0000,
            0x40B0_4000,
            0x0200_0000,
            0x40B0_4000,
            0x8000_0000,
            0x40B0_4000,
            0xFFFF_FFFF,
            0x40B0_4200,
            0xFFFF_FFFF,
            0x40B0_4300,
            0x1400_0000,
        ]);
        assert_eq!(events.len(), 6);
        assert_eq!(events[0], Event::Pedal(PedalEvent::Sustain(0.0)));
        assert_eq!(events[3], Event::Pedal(PedalEvent::Sustain(1.0)));
        assert_eq!(events[4], Event::Pedal(PedalEvent::Sostenuto(true)));
        assert_eq!(events[5], Event::Pedal(PedalEvent::UnaCorda(false)));
        // Half pedal is continuous and lands within a 7-bit step of centre.
        let Event::Pedal(PedalEvent::Sustain(half)) = events[2] else {
            panic!("not a sustain event");
        };
        assert!((half - 0.5).abs() < 0.5 / 127.0);
        // A 32-bit CC really is finer than a 7-bit one: two values a single
        // 32-bit step apart do not collapse.
        let fine = events_of(&[0x40B0_4000, 0x8000_0000, 0x40B0_4000, 0x8000_0100]);
        assert_ne!(fine[0], fine[1]);
    }

    /// MIDI 2.0's note on at velocity zero is the silent press, and nothing
    /// else in either protocol is.
    #[test]
    fn a_midi2_note_on_at_zero_is_the_silent_press() {
        assert_eq!(
            events_of(&[0x4090_3C00, 0x0000_0000]),
            vec![Event::NoteOn { key: 60, vel: 0 }]
        );
        // The same bytes in MIDI 1.0 are a note off, in both spellings.
        assert_eq!(
            events_of(&[0x2090_3C00]),
            vec![Event::NoteOff {
                key: 60,
                vel: DEFAULT_RELEASE_VELOCITY
            }]
        );
        assert_eq!(
            bytes_of(&[0x90, 60, 0]),
            vec![Event::NoteOff {
                key: 60,
                vel: DEFAULT_RELEASE_VELOCITY
            }]
        );
    }

    /// Message types we do not play are skipped by *length*, so a stream that
    /// carries them still parses the notes around them.
    #[test]
    fn unknown_messages_are_skipped_whole() {
        let words = events_of(&[
            0x0000_0000, // utility, 1 word
            0x40E0_0000, // pitch bend, 2 words
            0x8000_0000, //   (its second word)
            0x3000_0000, // sysex7, 2 words
            0x0000_0000, //   (its second word)
            0x5000_0000, // sysex8, 4 words
            0x0000_0000,
            0x0000_0000,
            0x0000_0000,
            0x4090_3C00, // and the note survives all of it
            0xFFFF_0000,
        ]);
        assert_eq!(
            words,
            vec![Event::NoteOn {
                key: 60,
                vel: 127 * VELOCITY_STEPS
            }]
        );
        // A truncated trailing message is dropped rather than read short.
        assert!(events_of(&[0x4090_3C00]).is_empty());
    }

    /// Keys outside A0..C8, and channels other than the first, are handled the
    /// way the file reader handles them: dropped, and merged, respectively.
    #[test]
    fn the_keyboard_is_the_88_keys_and_every_channel_is_the_same_piano() {
        assert!(events_of(&[0x4090_0C00, 0xFFFF_0000]).is_empty()); // C0
        assert!(events_of(&[0x4090_7800, 0xFFFF_0000]).is_empty()); // above C8
        let ch9 = events_of(&[0x4098_3C00, 0xB4D3_0000]);
        assert_eq!(
            ch9,
            vec![Event::NoteOn {
                key: 60,
                vel: 90 * VELOCITY_STEPS
            }]
        );
        // Group 5 is the same piano too.
        let group = events_of(&[0x4590_3C00, 0xB4D3_0000]);
        assert_eq!(group, ch9);
    }

    /// The byte-stream fallback: running status, release velocity, the pedals,
    /// and real-time bytes interleaved mid-message.
    #[test]
    fn the_byte_stream_fallback_reads_a_real_keyboards_habits() {
        // Running status: one 0x90, then note pairs.
        assert_eq!(
            bytes_of(&[0x90, 60, 90, 62, 80, 64, 70]),
            vec![
                Event::NoteOn { key: 60, vel: 90 },
                Event::NoteOn { key: 62, vel: 80 },
                Event::NoteOn { key: 64, vel: 70 },
            ]
        );
        // A clock byte in the middle of a message disturbs nothing.
        assert_eq!(
            bytes_of(&[0x90, 60, 0xF8, 90]),
            vec![Event::NoteOn { key: 60, vel: 90 }]
        );
        // Release velocity, and the "no measurement" zero.
        assert_eq!(
            bytes_of(&[0x80, 60, 120, 0x80, 62, 0]),
            vec![
                Event::NoteOff { key: 60, vel: 120 },
                Event::NoteOff {
                    key: 62,
                    vel: DEFAULT_RELEASE_VELOCITY
                },
            ]
        );
        // Pedals, continuous and switched, and an ignored controller.
        assert_eq!(
            bytes_of(&[0xB0, 64, 40, 0xB0, 66, 127, 0xB0, 67, 10, 0xB0, 7, 100]),
            vec![
                Event::Pedal(PedalEvent::Sustain(40.0 / 127.0)),
                Event::Pedal(PedalEvent::Sostenuto(true)),
                Event::Pedal(PedalEvent::UnaCorda(false)),
            ]
        );
        // Sysex cancels running status rather than being played as notes.
        assert!(bytes_of(&[0xF0, 0x7E, 0x7F, 0x06, 0x01, 0xF7, 60, 90]).is_empty());
        // Two-byte and one-byte messages do not desynchronise the stream.
        assert_eq!(
            bytes_of(&[0xC0, 5, 0xD0, 64, 0x90, 60, 90]),
            vec![Event::NoteOn { key: 60, vel: 90 }]
        );
    }

    /// Both lanes reach the same instrument: a note played at 7-bit velocity 90
    /// and the same note played at 16-bit velocity produce the same event
    /// velocity *through `velocity_from_midi`*, which is the only thing the
    /// engine reads.
    #[test]
    fn the_two_lanes_are_one_velocity() {
        let legacy = bytes_of(&[0x90, 60, 90]);
        let fine = events_of(&[0x4090_3C00, 0xB4D3_0000]);
        let velocity = |e: &Event| match e {
            Event::NoteOn { vel, .. } => velocity_from_midi(*vel),
            _ => unreachable!(),
        };
        assert_eq!(velocity(&legacy[0]), velocity(&fine[0]));
        assert_ne!(legacy[0], fine[0], "the two spellings are not the same u16");
    }
}
