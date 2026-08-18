//  AudioHost.swift — the standalone app hosting its own AUv3.
//
//  `DISTRIBUTION.md` §Architecture: "The standalone app should host its own
//  AUv3 through `AVAudioEngine` rather than linking the engine a second way.
//  One code path gets exercised twice, and any bug that only shows up under AU
//  semantics shows up in our own app first."
//
//  That is what this does, and it is worth being exact about *which* path is
//  running, because there are two and they are not equally good:
//
//  1. **The appex.** `AVAudioUnitComponentManager` finds the instrument because
//     PluginKit registered the extension inside this very app bundle, and
//     `AVAudioUnit.instantiate` loads it — out of process, exactly as Logic
//     would. This is the path the milestone is about, and the one the window
//     reports when it is live.
//  2. **In process.** PluginKit has not registered the appex — a fresh build
//     that has never been launched from `/Applications`, or a machine where
//     registration has not caught up — so the app registers the *same*
//     `AUAudioUnit` subclass under the *same* component description with
//     `AUAudioUnit.registerSubclass` and instantiates that instead.
//
//  The fallback is not a second implementation: `AVAudioEngine`, `AVAudioUnit`,
//  the parameter tree, the preset list and the render block are identical, and
//  only the provider differs. It exists because App Review 4.2.3(i) asks the
//  app to work on its own, and "the plugin is not registered yet" is not an
//  acceptable reason for a piano to be silent. Which one is live is on screen.
//
//  SPDX-License-Identifier: MIT

import AVFoundation
import AudioToolbox
import CPianoEmulator
import CoreMIDI
import Foundation
import SwiftUI
import os

@MainActor
public final class AudioHost: ObservableObject {
    public enum Hosting: String {
        case appex = "AUv3 app extension, out of process"
        case inProcess = "registered in process — the appex is not registered yet"
    }

    @Published public private(set) var controller: PianoController?
    @Published public private(set) var hosting: Hosting?
    @Published public private(set) var statusLine = "starting the audio engine…"
    @Published public private(set) var midiPath = ""
    @Published public private(set) var failure: String?
    @Published public private(set) var midiSources: [String] = []

    private let engine = AVAudioEngine()
    private var audioUnit: AVAudioUnit?
    private var sender: AUMIDISender?
    private var midi: MIDIInput?

    private static let log = Logger(subsystem: "dev.pianoemulator.app", category: "AudioHost")

    public init() {}

    public func start() {
        let description = PianoIdentity.componentDescription
        let existing = AVAudioUnitComponentManager.shared().components(matching: description)
        if existing.isEmpty {
            AUAudioUnit.registerSubclass(
                PianoAudioUnit.self, as: description, name: PianoIdentity.displayName,
                version: PianoIdentity.version)
            hosting = .inProcess
        } else {
            hosting = .appex
        }
        statusLine = "loading the instrument…"

        AVAudioUnit.instantiate(with: description, options: []) { [weak self] unit, error in
            DispatchQueue.main.async {
                guard let self else { return }
                guard let unit else {
                    self.failure =
                        error?.localizedDescription ?? "the audio unit could not be instantiated"
                    self.statusLine = "no instrument"
                    return
                }
                self.attach(unit)
            }
        }
    }

    private func attach(_ unit: AVAudioUnit) {
        audioUnit = unit
        engine.attach(unit)
        engine.connect(unit, to: engine.mainMixerNode, format: nil)
        engine.prepare()

        // `pe_create` builds 88 voices and every coefficient — hundreds of
        // milliseconds — and it happens inside `engine.start()`, through
        // `allocateRenderResources`. Off the main thread, or the window is
        // frozen while the piano is strung.
        statusLine = "stringing the piano…"
        let engine = self.engine
        DispatchQueue.global(qos: .userInitiated).async {
            var startError: Error?
            do {
                try engine.start()
            } catch {
                startError = error
            }
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                if let startError {
                    self.failure = startError.localizedDescription
                    self.statusLine = "the audio engine did not start"
                    return
                }
                self.engineDidStart(unit)
            }
        }
    }

    private func engineDidStart(_ unit: AVAudioUnit) {
        let sender = AUMIDISender(unit: unit)
        self.sender = sender
        let controller = PianoController(audioUnit: unit.auAudioUnit)
        controller.noteSink = sender
        self.controller = controller

        let rate = engine.outputNode.outputFormat(forBus: 0).sampleRate
        let bypassed = rate == Double(PE_ENGINE_SAMPLE_RATE)
        statusLine = String(
            format: "%.0f Hz — %@", rate,
            bypassed
                ? "the boundary resampler is bypassed: this is the offline render, sample for sample"
                : "boundary-resampled from the engine's own 48 kHz")
        midiPath = sender.pathDescription

        let midi = MIDIInput()
        midi.onEventList = { [weak sender] list in sender?.send(eventList: list) }
        midi.onSourcesChanged = { names in
            DispatchQueue.main.async { [weak self] in self?.midiSources = names }
        }
        midi.start()
        self.midi = midi
    }

    public func stop() {
        midi?.stop()
        engine.stop()
    }
}

/// Everything the app has to say to the audio unit that is not a parameter.
///
/// Three routes, tried in the order that keeps the most resolution: the UMP
/// event-list block (a 16-bit velocity survives it), the byte-oriented block,
/// and `AVAudioUnitMIDIInstrument`, which is what `AVAudioUnit.instantiate`
/// hands back for an `aumu` component and is always there.
///
/// **Not main-actor-isolated on purpose.** `send(eventList:)` is called on Core
/// MIDI's own receive thread and must not hop: every stored property is a `let`
/// written once in `init`, and the three routes are the AU's own thread-safe
/// blocks.
final class AUMIDISender: PianoNoteSink, @unchecked Sendable {
    private let instrument: AVAudioUnitMIDIInstrument?
    private let eventListBlock: AUMIDIEventListBlock?
    private let eventBlock: AUScheduleMIDIEventBlock?
    /// One `MIDIEventList` of scratch, allocated once, so that building a
    /// one-message list does not allocate on the MIDI thread.
    private let scratch: UnsafeMutablePointer<MIDIEventList>

    let pathDescription: String

    init(unit: AVAudioUnit) {
        let auAudioUnit = unit.auAudioUnit
        instrument = unit as? AVAudioUnitMIDIInstrument
        eventListBlock = auAudioUnit.scheduleMIDIEventListBlock
        eventBlock = auAudioUnit.scheduleMIDIEventBlock
        scratch = UnsafeMutablePointer<MIDIEventList>.allocate(capacity: 1)
        scratch.initialize(to: MIDIEventList())
        if eventListBlock != nil {
            pathDescription = "MIDI 2.0 event lists"
        } else if eventBlock != nil {
            pathDescription = "MIDI 1.0 event block"
        } else if instrument != nil {
            pathDescription = "MIDI 1.0 through AVAudioUnitMIDIInstrument"
        } else {
            pathDescription = "no MIDI path — this is a bug"
        }
    }

    deinit {
        scratch.deallocate()
    }

    /// A whole `MIDIEventList` straight from Core MIDI, unopened where it can
    /// be. Called on the Core MIDI receive thread.
    func send(eventList: UnsafePointer<MIDIEventList>) {
        if let eventListBlock {
            _ = eventListBlock(AUEventSampleTimeImmediate, 0, eventList)
            return
        }
        // No event-list block: unpack the UMP into MIDI 1.0 bytes, which is the
        // only shape the older routes take.
        for packet in eventList.unsafeSequence() {
            let words = UnsafeRawPointer(packet)
                .advanced(by: MemoryLayout<MIDIEventPacket>.offset(of: \.words)!)
                .assumingMemoryBound(to: UInt32.self)
            MIDITranslation.parseUMP(words, count: Int(packet.pointee.wordCount)) { event in
                sendTranslated(event)
            }
        }
    }

    /// A `pe_event_t` that has already been translated, put back onto the wire
    /// as MIDI 1.0. Lossy for a fine-lane velocity, which is exactly why the
    /// event-list route above is tried first.
    private func sendTranslated(_ event: pe_event_t) {
        switch event.kind {
        case MIDITranslation.Kind.noteOn:
            send(status: 0x90, d1: UInt8(event.key & 0x7F), d2: UInt8(min(event.vel, 127)))
        case MIDITranslation.Kind.noteOff:
            send(status: 0x80, d1: UInt8(event.key & 0x7F), d2: UInt8(min(event.vel, 127)))
        case MIDITranslation.Kind.sustain:
            send(status: 0xB0, d1: 64, d2: UInt8(max(0, min(127, event.value * 127))))
        case MIDITranslation.Kind.sostenuto:
            send(status: 0xB0, d1: 66, d2: event.value != 0 ? 127 : 0)
        case MIDITranslation.Kind.unaCorda:
            send(status: 0xB0, d1: 67, d2: event.value != 0 ? 127 : 0)
        default:
            break
        }
    }

    func send(status: UInt8, d1: UInt8, d2: UInt8) {
        if let eventListBlock {
            // One MIDI 1.0 channel-voice message as a single-word UMP on group
            // 0: `[mt=2][group=0][status|channel][d1][d2]`.
            var word: UInt32 = (0x2 << 28) | (UInt32(status) << 16) | (UInt32(d1) << 8) | UInt32(d2)
            var packet = MIDIEventListInit(scratch, ._1_0)
            packet = MIDIEventListAdd(
                scratch, MemoryLayout<MIDIEventList>.size, packet, 0, 1, &word)
            _ = eventListBlock(AUEventSampleTimeImmediate, 0, UnsafePointer(scratch))
            _ = packet
            return
        }
        if let eventBlock {
            var bytes: (UInt8, UInt8, UInt8) = (status, d1, d2)
            withUnsafeBytes(of: &bytes) { raw in
                eventBlock(
                    AUEventSampleTimeImmediate, 0, 3,
                    raw.baseAddress!.assumingMemoryBound(to: UInt8.self))
            }
            return
        }
        instrument?.sendMIDIEvent(status, data1: d1, data2: d2)
    }

    func noteOn(key: UInt8, velocity: UInt8) {
        send(status: 0x90, d1: key, d2: velocity)
    }

    func noteOff(key: UInt8, velocity: UInt8) {
        send(status: 0x80, d1: key, d2: velocity)
    }

    func allNotesOff() {
        // CC 123 is not one of the three controllers the engine reads, so the
        // panic is spelled out: every key released, every pedal up.
        for key in 21...108 {
            send(status: 0x80, d1: UInt8(key), d2: 64)
        }
        send(status: 0xB0, d1: 64, d2: 0)
        send(status: 0xB0, d1: 66, d2: 0)
        send(status: 0xB0, d1: 67, d2: 0)
    }
}
