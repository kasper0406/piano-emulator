//  MIDITranslation.swift — wire bytes and words to `pe_event_t`.
//
//  A mirror of `engine/src/midi/ump.rs`, function for function, because the C
//  ABI takes events and not MIDI: `pe_event` is the plugin's entry point and
//  there is no `pe_parse_ump`. The mirror is deliberate and it is *pinned*, not
//  trusted — `app/ParityHarness` renders the benchmark phrase through this file
//  and compares the md5 against the C harness's, so a divergence in any of the
//  velocity maps is a red parity run and not a subtle difference in the sound.
//
//  What is read is what the engine reads, no more: note on / note off on every
//  channel and every group, release velocity, CC 64 as a continuous sustain,
//  CC 66 and CC 67 as switches. Pitch bend, aftertouch, program change,
//  per-note controllers, sysex and the whole utility layer are ignored — the
//  Piano Profile's registered controllers are `DISTRIBUTION.md` M9.
//
//  SPDX-License-Identifier: MIT

import CPianoEmulator
import Foundation

public enum MIDITranslation {
    // MARK: - the numbers, from `engine/src/types.rs` and `midi/ump.rs`

    static let ccSustain: UInt8 = 64
    static let ccSostenuto: UInt8 = 66
    static let ccUnaCorda: UInt8 = 67
    /// A switched controller is down at 64 and up below it.
    static let switchThreshold: UInt8 = 64
    /// "This keyboard does not measure release velocity", not "released
    /// infinitely slowly".
    static let defaultReleaseVelocity: UInt32 = 64
    static let lowestKey: UInt8 = 21
    static let highestKey: UInt8 = 108
    static let midi1MaxVelocity: UInt16 = 127
    /// Every value the event's velocity field could hold when it was a `u8`.
    static let legacyVelocityMax: UInt16 = 255
    static let velocitySteps: UInt16 = 512

    // MARK: - velocity

    /// MIDI 2.0's min-center-max scaling of a 7-bit value to 16 bits
    /// (M2-104-UM appendix A.2) — what Core MIDI applies when it up-translates
    /// a MIDI 1.0 keyboard onto a MIDI 2.0 port, and therefore what
    /// `midi1Velocity(of:)` has to invert.
    public static func upscale7to16(_ value: UInt8) -> UInt16 {
        let value = value & 0x7F
        var scaled = UInt16(value) << 9
        if value <= 0x40 {
            return scaled
        }
        var repeated = UInt16(value & 0x3F) << 3
        while repeated != 0 {
            scaled |= repeated
            repeated >>= 6
        }
        return scaled
    }

    /// The continuous MIDI 1.0 velocity a 16-bit one stands for: the piecewise
    /// linear inverse of `upscale7to16`, exact at all 128 of its points.
    public static func midi1Velocity(of v16: UInt16) -> Float {
        if v16 >= upscale7to16(UInt8(midi1MaxVelocity)) {
            return Float(midi1MaxVelocity)
        }
        var i = UInt8(min(UInt16(v16 >> 9), midi1MaxVelocity))
        while i > 0 && upscale7to16(i) > v16 {
            i -= 1
        }
        while i < 127 && upscale7to16(i + 1) <= v16 {
            i += 1
        }
        let low = upscale7to16(i)
        let high = upscale7to16(i + 1)
        return Float(i) + Float(v16 - low) / Float(high - low)
    }

    /// A continuous MIDI velocity in the event field's **fine lane**.
    public static func hiresVelocity(_ velocity: Float) -> UInt16 {
        if velocity <= 0.0 {
            return 0
        }
        let steps = (velocity * Float(velocitySteps)).rounded()
        let top = Float(midi1MaxVelocity) * Float(velocitySteps)
        return UInt16(min(max(steps, Float(legacyVelocityMax) + 1.0), top))
    }

    /// A MIDI 2.0 16-bit velocity as an event velocity. Zero stays zero — that
    /// is the silent press, and the one velocity the two lanes share.
    public static func velocityFromUMP(_ v16: UInt16) -> UInt16 {
        v16 == 0 ? 0 : hiresVelocity(midi1Velocity(of: v16))
    }

    // MARK: - messages

    static func playable(_ key: UInt8) -> UInt8? {
        let key = key & 0x7F
        return (lowestKey...highestKey).contains(key) ? key : nil
    }

    /// The event kinds, as plain numbers.
    ///
    /// `pe_event_kind` cannot be *named* from Swift — cbindgen emits both an
    /// `enum pe_event_kind` and a `typedef uint32_t pe_event_kind`, which is
    /// legal C (tags and typedefs are different namespaces) and ambiguous to
    /// the Clang importer — so the constants are unwrapped once, here, and the
    /// rest of the Swift uses these.
    public enum Kind {
        public static let noteOn = PE_EVENT_NOTE_ON.rawValue
        public static let noteOff = PE_EVENT_NOTE_OFF.rawValue
        public static let keyDown = PE_EVENT_KEY_DOWN.rawValue
        public static let sustain = PE_EVENT_SUSTAIN.rawValue
        public static let sostenuto = PE_EVENT_SOSTENUTO.rawValue
        public static let unaCorda = PE_EVENT_UNA_CORDA.rawValue
        public static let allOff = PE_EVENT_ALL_OFF.rawValue
    }

    public static func event(
        _ kind: UInt32, key: UInt32 = 0, vel: UInt32 = 0, value: Float = 0
    ) -> pe_event_t {
        pe_event_t(kind: kind, key: key, vel: vel, value: value)
    }

    /// One controller change, either protocol. `position` is already 0…1.
    static func controller(_ index: UInt8, _ position: Float) -> pe_event_t? {
        let down = position >= Float(switchThreshold) / 127.0
        switch index & 0x7F {
        case ccSustain:
            return event(Kind.sustain, value: min(max(position, 0.0), 1.0))
        case ccSostenuto:
            return event(Kind.sostenuto, value: down ? 1.0 : 0.0)
        case ccUnaCorda:
            return event(Kind.unaCorda, value: down ? 1.0 : 0.0)
        default:
            return nil
        }
    }

    /// One MIDI 1.0 channel-voice message, from its status nibble and data
    /// bytes. Shared by the byte path and the message-type-2 UMP path, so there
    /// is one set of rules.
    public static func fromMIDI1(statusNibble: UInt8, d1: UInt8, d2: UInt8) -> pe_event_t? {
        switch statusNibble & 0x0F {
        case 0x9 where (d2 & 0x7F) > 0:
            guard let key = playable(d1) else { return nil }
            // A 7-bit source stays in the legacy lane: the number on the wire
            // is the number in the event.
            return event(Kind.noteOn, key: UInt32(key), vel: UInt32(d2 & 0x7F))
        case 0x9:
            // Velocity 0 is the note-off half of a running-status note on.
            guard let key = playable(d1) else { return nil }
            return event(Kind.noteOff, key: UInt32(key), vel: defaultReleaseVelocity)
        case 0x8:
            guard let key = playable(d1) else { return nil }
            let released = d2 & 0x7F
            return event(
                Kind.noteOff, key: UInt32(key),
                vel: released == 0 ? defaultReleaseVelocity : UInt32(released))
        case 0xB:
            return controller(d1, Float(d2 & 0x7F) / 127.0)
        default:
            return nil
        }
    }

    /// One MIDI 1.0 channel-voice message from a whole status byte. System
    /// messages are not ours to read.
    public static func fromMIDI1(statusByte: UInt8, d1: UInt8, d2: UInt8) -> pe_event_t? {
        guard statusByte >= 0x80, statusByte < 0xF0 else { return nil }
        return fromMIDI1(statusNibble: statusByte >> 4, d1: d1, d2: d2)
    }

    /// One MIDI 2.0 channel-voice UMP (message type 4).
    ///
    /// The note attribute is ignored: the only defined types are a MIDI 1.0
    /// articulation, a Profile-specific value and pitch 7.9, and none of them
    /// is something this instrument models yet.
    public static func fromMIDI2(w0: UInt32, w1: UInt32) -> pe_event_t? {
        let status = UInt8((w0 >> 20) & 0x0F)
        let index = UInt8((w0 >> 8) & 0x7F)
        switch status {
        case 0x9:
            let v16 = UInt16(truncatingIfNeeded: w1 >> 16)
            guard let key = playable(index) else { return nil }
            // Velocity 0 is a *silent press* here, not a note off: MIDI 2.0
            // has a real note off, so nothing is overloaded.
            return event(Kind.noteOn, key: UInt32(key), vel: UInt32(velocityFromUMP(v16)))
        case 0x8:
            let v16 = UInt16(truncatingIfNeeded: w1 >> 16)
            guard let key = playable(index) else { return nil }
            return event(
                Kind.noteOff, key: UInt32(key),
                vel: v16 == 0 ? defaultReleaseVelocity : UInt32(velocityFromUMP(v16)))
        case 0xB:
            return controller(index, Float(w1) / Float(UInt32.max))
        default:
            return nil
        }
    }

    /// Words in a UMP message, by message type (M2-104-UM §2.1.4). Length comes
    /// from the type, never from the content, which is what makes it safe to
    /// walk a stream carrying sysex or jitter-reduction timestamps.
    public static func wordsInMessage(_ messageType: UInt8) -> Int {
        switch messageType {
        case 0x0, 0x1, 0x2, 0x6, 0x7: return 1
        case 0x3, 0x4, 0x8, 0x9, 0xA: return 2
        case 0xB, 0xC: return 3
        default: return 4
        }
    }

    /// Walks a run of Universal MIDI Packets and hands every event in it to
    /// `sink`, in order. A truncated final message ends the walk.
    ///
    /// Allocation-free: it is called from the render block.
    public static func parseUMP(
        _ words: UnsafePointer<UInt32>, count: Int, sink: (pe_event_t) -> Void
    ) {
        var i = 0
        while i < count {
            let messageType = UInt8(words[i] >> 28)
            let length = wordsInMessage(messageType)
            if i + length > count {
                return
            }
            var parsed: pe_event_t?
            switch messageType {
            case 0x2: parsed = fromMIDI1Word(words[i])
            case 0x4: parsed = fromMIDI2(w0: words[i], w1: words[i + 1])
            default: parsed = nil
            }
            if let parsed {
                sink(parsed)
            }
            i += length
        }
    }

    /// One MIDI 1.0 channel-voice UMP (message type 2): `[mt:4][group:4]
    /// [status:4][channel:4][d1:8][d2:8]`.
    static func fromMIDI1Word(_ word: UInt32) -> pe_event_t? {
        fromMIDI1(
            statusNibble: UInt8((word >> 20) & 0x0F),
            d1: UInt8((word >> 8) & 0x7F),
            d2: UInt8(word & 0x7F))
    }

    /// A run of MIDI 1.0 bytes with no running status — which is what an
    /// `AURenderEventMIDI` carries, since the host has already reassembled the
    /// message for us.
    public static func fromMIDI1Bytes(_ bytes: UnsafePointer<UInt8>, count: Int) -> pe_event_t? {
        guard count >= 2 else { return nil }
        return fromMIDI1(
            statusByte: bytes[0], d1: bytes[1], d2: count >= 3 ? bytes[2] : 0)
    }
}
