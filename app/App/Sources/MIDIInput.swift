//  MIDIInput.swift — Core MIDI, straight into the audio unit.
//
//  `DISTRIBUTION.md` §MIDI: "Standalone: Core MIDI directly.
//  `MIDIInputPortCreateWithProtocol` plus a virtual destination so other apps
//  can play us." Both are here, and both ask for MIDI 2.0, which means Core
//  MIDI up-translates a MIDI 1.0 keyboard for us and hands a UMP source its own
//  sixteen bits — the engine's fine velocity lane is reachable from a
//  controller that has one (`SHIPPING.md` §4, the SL88 MK2).
//
//  No parsing happens here. The `MIDIEventList` goes to the audio unit whole,
//  through `AUMIDIEventListBlock`, and is parsed once — in the render block, by
//  `MIDITranslation`, which is the same code the plugin uses when a DAW is the
//  one sending. One parser, one set of rules.
//
//  The block-based Core MIDI API is used throughout (`MIDIClientCreateWithBlock`,
//  `MIDIInputPortCreateWithProtocol` with a receive block), so there is no
//  `@convention(c)` callback and no hand-boxed context.
//
//  SPDX-License-Identifier: MIT

import CoreMIDI
import Foundation
import os

final class MIDIInput: @unchecked Sendable {
    /// Called on Core MIDI's receive thread, once per arriving list.
    var onEventList: ((UnsafePointer<MIDIEventList>) -> Void)?
    /// Called whenever the set of connected sources changes.
    var onSourcesChanged: (([String]) -> Void)?

    private var client = MIDIClientRef()
    private var port = MIDIPortRef()
    private var virtualDestination = MIDIEndpointRef()
    private var connected: Set<MIDIEndpointRef> = []

    private static let log = Logger(subsystem: "dev.pianoemulator.app", category: "MIDIInput")

    func start() {
        var status = MIDIClientCreateWithBlock("Piano Emulator" as CFString, &client) {
            [weak self] notification in
            // `kMIDIMsgSetupChanged` covers a keyboard being plugged in or
            // pulled out; rescanning is cheap and idempotent.
            if notification.pointee.messageID == .msgSetupChanged {
                self?.connectAllSources()
            }
        }
        guard status == noErr else {
            MIDIInput.log.error("MIDIClientCreateWithBlock failed: \(status)")
            return
        }

        status = MIDIInputPortCreateWithProtocol(
            client, "Input" as CFString, ._2_0, &port
        ) { [weak self] eventList, _ in
            self?.onEventList?(eventList)
        }
        if status != noErr {
            MIDIInput.log.error("MIDIInputPortCreateWithProtocol failed: \(status)")
        }

        // A virtual destination, so any other app on the machine can play this
        // piano without a cable.
        status = MIDIDestinationCreateWithProtocol(
            client, "Piano Emulator" as CFString, ._2_0, &virtualDestination
        ) { [weak self] eventList, _ in
            self?.onEventList?(eventList)
        }
        if status != noErr {
            MIDIInput.log.error("MIDIDestinationCreateWithProtocol failed: \(status)")
        }

        connectAllSources()
    }

    func stop() {
        if virtualDestination != 0 { MIDIEndpointDispose(virtualDestination) }
        if port != 0 { MIDIPortDispose(port) }
        if client != 0 { MIDIClientDispose(client) }
        virtualDestination = 0
        port = 0
        client = 0
    }

    private func connectAllSources() {
        guard port != 0 else { return }
        var names: [String] = []
        for index in 0..<MIDIGetNumberOfSources() {
            let source = MIDIGetSource(index)
            guard source != 0 else { continue }
            names.append(MIDIInput.name(of: source))
            if !connected.contains(source) {
                if MIDIPortConnectSource(port, source, nil) == noErr {
                    connected.insert(source)
                }
            }
        }
        onSourcesChanged?(names)
    }

    private static func name(of endpoint: MIDIEndpointRef) -> String {
        var unmanaged: Unmanaged<CFString>?
        guard MIDIObjectGetStringProperty(endpoint, kMIDIPropertyDisplayName, &unmanaged) == noErr,
            let value = unmanaged?.takeRetainedValue()
        else {
            return "unnamed source"
        }
        return value as String
    }
}
