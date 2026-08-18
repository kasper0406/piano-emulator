//  SMFReader.swift — a standard MIDI file, read exactly as `render.c` reads it.
//
//  This is a third mirror of the same thing (`engine/src/midi.rs` is the first,
//  `ffi/harness/render.c` the second), and it exists for one reason: the parity
//  claim is *sample-exact*, and a sample-exact claim needs the events to land
//  on the same frames. Where the arithmetic could differ it says which line of
//  the C it is mirroring, because that is where a divergence would hide:
//
//   - the tick→second map is evaluated in `Double` and *then* narrowed to
//     `Float`, as `midly`'s is;
//   - an event's frame is `RenderEvent::frame` — round-half-away-from-zero of
//     `time_s * 48000` in **Float**, not Double;
//   - events are stable-sorted by tick and *then* stable-sorted by frame, in
//     that order, with the position after the first sort as the second's
//     tiebreak;
//   - the render is `last_event + 4 s`, truncated to whole frames in Float.
//
//  If any of those is wrong the md5 will say so, which is the whole point of
//  measuring parity against a hash rather than against a tolerance.
//
//  SPDX-License-Identifier: MIT

import CPianoEmulator
import Foundation

struct TimedEvent {
    var tick: UInt64
    var order: UInt32
    var event: pe_event_t
    var timeSeconds: Float = 0
    var frame: Int = 0
}

struct Performance {
    var events: [TimedEvent]
    var lastEventSeconds: Float
    /// `MidiPerformance::duration_s`: the last event plus a four-second tail.
    var durationSeconds: Float { lastEventSeconds + 4.0 }
    /// The engine-rate frame count, truncated in `Float` as the Rust does.
    var engineFrames: Int {
        let frames = durationSeconds * Float(PE_ENGINE_SAMPLE_RATE)
        return frames > 0 ? Int(frames) : 0
    }
}

enum SMFReader {
    enum Failure: Error, CustomStringConvertible {
        case malformed(String)
        var description: String {
            switch self {
            case .malformed(let what): return "not a MIDI file we can read: \(what)"
            }
        }
    }

    private static let defaultMicrosecondsPerBeat = 500_000.0
    private static let ccSustain: UInt8 = 64
    private static let ccSostenuto: UInt8 = 66
    private static let ccUnaCorda: UInt8 = 67

    static func load(_ url: URL) throws -> Performance {
        let bytes = [UInt8](try Data(contentsOf: url))
        return try parse(bytes)
    }

    static func parse(_ bytes: [UInt8]) throws -> Performance {
        guard bytes.count >= 14, bytes[0...3].elementsEqual("MThd".utf8) else {
            throw Failure.malformed("no MThd")
        }
        var cursor = 8
        func readBE(_ count: Int) throws -> UInt32 {
            guard cursor + count <= bytes.count else { throw Failure.malformed("truncated") }
            var value: UInt32 = 0
            for _ in 0..<count {
                value = (value << 8) | UInt32(bytes[cursor])
                cursor += 1
            }
            return value
        }
        cursor = 4
        let headerLength = try readBE(4)
        let format = try readBE(2)
        let trackCount = try readBE(2)
        let division = try readBE(2)
        guard format != 2 else { throw Failure.malformed("format 2") }
        guard division & 0x8000 == 0, division != 0 else {
            throw Failure.malformed("SMPTE or zero division")
        }
        cursor = 8 + Int(headerLength)

        var events: [TimedEvent] = []
        var tempos: [(tick: UInt64, microsecondsPerBeat: Double)] = []

        func readVarLen() throws -> UInt32 {
            var value: UInt32 = 0
            for _ in 0..<4 {
                guard cursor < bytes.count else { throw Failure.malformed("truncated varlen") }
                let byte = bytes[cursor]
                cursor += 1
                value = (value << 7) | UInt32(byte & 0x7F)
                if byte & 0x80 == 0 { return value }
            }
            throw Failure.malformed("over-long varlen")
        }

        for _ in 0..<trackCount {
            guard cursor + 8 <= bytes.count else { break }
            guard bytes[cursor..<(cursor + 4)].elementsEqual("MTrk".utf8) else {
                throw Failure.malformed("expected MTrk")
            }
            cursor += 4
            let chunkLength = try readBE(4)
            let trackEnd = cursor + Int(chunkLength)
            guard trackEnd <= bytes.count else { throw Failure.malformed("truncated track") }
            var tick: UInt64 = 0
            var running: UInt8 = 0
            while cursor < trackEnd {
                tick += UInt64(try readVarLen())
                var status = bytes[cursor]
                if status & 0x80 != 0 {
                    cursor += 1
                    if status < 0xF0 { running = status }
                } else {
                    guard running != 0 else { throw Failure.malformed("running status with none") }
                    status = running
                }
                if status == 0xFF {
                    let meta = bytes[cursor]
                    cursor += 1
                    let length = Int(try readVarLen())
                    if meta == 0x51, length == 3 {
                        let value =
                            (Double(bytes[cursor]) * 65536) + (Double(bytes[cursor + 1]) * 256)
                            + Double(bytes[cursor + 2])
                        tempos.append((tick, value))
                    }
                    cursor += length
                    continue
                }
                if status == 0xF0 || status == 0xF7 {
                    cursor += Int(try readVarLen())
                    continue
                }
                let high = status & 0xF0
                let dataBytes = (high == 0xC0 || high == 0xD0) ? 1 : 2
                let d1 = bytes[cursor]
                cursor += 1
                var d2: UInt8 = 0
                if dataBytes == 2 {
                    d2 = bytes[cursor]
                    cursor += 1
                }
                if let event = translate(status: status, d1: d1, d2: d2) {
                    events.append(TimedEvent(tick: tick, order: 0, event: event))
                }
            }
            cursor = trackEnd
        }

        // `midi.rs::Clock::new`, in Double throughout. Two tempo events on the
        // same tick mean the last in file order wins.
        tempos = stableSortedByTick(tempos)
        let ticksPerBeat = Double(division)
        var segments: [(tick: UInt64, seconds: Double, rate: Double)] = [
            (0, 0, defaultMicrosecondsPerBeat / 1.0e6 / ticksPerBeat)
        ]
        for tempo in tempos {
            let rate = tempo.microsecondsPerBeat / 1.0e6 / ticksPerBeat
            let last = segments[segments.count - 1]
            let seconds = last.seconds + Double(tempo.tick - last.tick) * last.rate
            if tempo.tick == last.tick {
                segments[segments.count - 1] = (tempo.tick, seconds, rate)
            } else {
                segments.append((tempo.tick, seconds, rate))
            }
        }

        stableSort(&events) { $0.tick < $1.tick }
        for index in events.indices {
            var segment = 0
            while segment + 1 < segments.count, segments[segment + 1].tick <= events[index].tick {
                segment += 1
            }
            let seconds =
                segments[segment].seconds
                + Double(events[index].tick - segments[segment].tick) * segments[segment].rate
            // `as f32` in `midi.rs`, then `RenderEvent::frame`'s Float round.
            let timeSeconds = Float(seconds)
            events[index].timeSeconds = timeSeconds
            let t = timeSeconds > 0 ? timeSeconds : 0
            events[index].frame = Int((t * Float(PE_ENGINE_SAMPLE_RATE)).rounded())
        }
        let lastEventSeconds = events.last?.timeSeconds ?? 0
        stableSort(&events) { $0.frame < $1.frame }
        return Performance(events: events, lastEventSeconds: lastEventSeconds)
    }

    /// `midi.rs::translate`, byte for byte with `render.c`'s copy of it.
    private static func translate(status: UInt8, d1: UInt8, d2: UInt8) -> pe_event_t? {
        MIDITranslation.fromMIDI1(statusByte: status, d1: d1, d2: d2)
    }

    /// `render.c`'s `stable_sort`: the index *after the previous sort* is
    /// written into `order` and used as the tiebreak, so an unstable sort
    /// reproduces a Rust stable one.
    private static func stableSort(
        _ events: inout [TimedEvent], by less: (TimedEvent, TimedEvent) -> Bool
    ) {
        for index in events.indices { events[index].order = UInt32(index) }
        events.sort { a, b in
            if less(a, b) { return true }
            if less(b, a) { return false }
            return a.order < b.order
        }
    }

    private static func stableSortedByTick(_ tempos: [(tick: UInt64, microsecondsPerBeat: Double)])
        -> [(tick: UInt64, microsecondsPerBeat: Double)]
    {
        // Insertion sort, as the C does: there are never many, and it must be
        // stable so the later of two on one tick wins.
        var sorted = tempos
        for i in 1..<max(sorted.count, 1) {
            let key = sorted[i]
            var j = i
            while j > 0, sorted[j - 1].tick > key.tick {
                sorted[j] = sorted[j - 1]
                j -= 1
            }
            sorted[j] = key
        }
        return sorted
    }
}
