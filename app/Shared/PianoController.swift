//  PianoController.swift — one view model for both UIs.
//
//  The plugin's own view (inside the appex) and the standalone app drive the
//  *same* instrument through the *same* surface: `AUParameterTree` for the
//  pedals and the trim, `factoryPresets`/`currentPreset` for the instrument,
//  two read-only meter parameters for the level and the voice count. All of
//  that crosses an XPC boundary unchanged, so this class works whether the AU
//  is our own object in this process or a proxy for an appex in another —
//  which is the point of the standalone hosting its own appex.
//
//  Notes are the one thing that does not go through the parameter tree, because
//  they are MIDI. The two hosts supply a `PianoNoteSink` each.
//
//  SPDX-License-Identifier: MIT

import AudioToolbox
import CPianoEmulator
import Combine
import Foundation
import SwiftUI

/// Where the on-screen keyboard's notes go. The app sends them into the AU as
/// MIDI; the plugin's own view posts them into the engine's queue.
public protocol PianoNoteSink: AnyObject {
    func noteOn(key: UInt8, velocity: UInt8)
    func noteOff(key: UInt8, velocity: UInt8)
    func allNotesOff()
}

public enum PianoParameter {
    public static let sustain = AUParameterAddress(PE_PARAM_SUSTAIN)
    public static let sostenuto = AUParameterAddress(PE_PARAM_SOSTENUTO)
    public static let unaCorda = AUParameterAddress(PE_PARAM_UNA_CORDA)
    public static let outputTrim = AUParameterAddress(PE_PARAM_OUTPUT_TRIM)

    /// Meters. Read-only, out of the automation range on purpose: a host that
    /// lists parameters shows the four above and these two as meters.
    public static let voices = AUParameterAddress(100)
    public static let peak = AUParameterAddress(101)
}

/// The two meter parameters, on their own object.
///
/// Not merged into `PianoController`: SwiftUI invalidates a view's whole body
/// when *any* `@Published` property of an observed object changes, so a 30 Hz
/// meter on the same object as the pedals redraws the keyboard thirty times a
/// second. Measured, on the appex hosting its own view: **36.7 % of one core
/// with nothing playing**, against 2.5 % of one core for the engine rendering
/// silence. The split, the change threshold below and
/// `startMetering`/`stopMetering` are the three halves of that fix.
@MainActor
public final class PianoMeters: ObservableObject {
    @Published public private(set) var peakDb: Float = -120
    @Published public private(set) var activeVoices: Int = 0

    /// Only publishes a change a person could see. A meter that republishes an
    /// unchanged −120 dB thirty times a second is thirty redraws of nothing.
    func update(peakDb newPeak: Float, activeVoices newVoices: Int) {
        if abs(newPeak - peakDb) > 0.25 { peakDb = newPeak }
        if newVoices != activeVoices { activeVoices = newVoices }
    }
}

@MainActor
public final class PianoController: ObservableObject {
    public let audioUnit: AUAudioUnit
    public weak var noteSink: PianoNoteSink?

    @Published public var sustain: Float = 0 { didSet { push(PianoParameter.sustain, sustain) } }
    @Published public var sostenuto = false {
        didSet { push(PianoParameter.sostenuto, sostenuto ? 1 : 0) }
    }
    @Published public var unaCorda = false {
        didSet { push(PianoParameter.unaCorda, unaCorda ? 1 : 0) }
    }
    @Published public var outputTrim: Float = 0 {
        didSet { push(PianoParameter.outputTrim, outputTrim) }
    }
    @Published public private(set) var presetNames: [String] = []
    @Published public var presetNumber: Int = 0 { didSet { selectPreset(presetNumber) } }

    public let meters = PianoMeters()
    /// Set while a preset is being built, because `pe_create` and
    /// `pe_load_preset_toml` take hundreds of milliseconds and the UI should
    /// say so rather than freeze silently.
    @Published public private(set) var busy = false

    private var timer: Timer?
    private var pushing = false

    public init(audioUnit: AUAudioUnit) {
        self.audioUnit = audioUnit
        presetNames = (audioUnit.factoryPresets ?? []).map(\.name)
        if let current = audioUnit.currentPreset, current.number >= 0 {
            presetNumber = current.number
        }
        readParameters()
    }

    deinit {
        timer?.invalidate()
    }

    /// Starts polling the meters. Off until something is on screen to show
    /// them: the plugin's own view is instantiated by PlugInKit whether or not
    /// a host ever displays it, and a timer that redraws an invisible window is
    /// pure cost in the *appex's* process.
    public func startMetering() {
        guard timer == nil else { return }
        let timer = Timer(timeInterval: 1.0 / 24.0, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { self?.readMeters() }
        }
        RunLoop.main.add(timer, forMode: .common)
        self.timer = timer
    }

    public func stopMetering() {
        timer?.invalidate()
        timer = nil
    }

    // MARK: - notes

    public func noteOn(key: UInt8, velocity: UInt8) {
        noteSink?.noteOn(key: key, velocity: velocity)
    }

    public func noteOff(key: UInt8) {
        // 64 is "this keyboard does not measure release velocity" — a mouse
        // does not, and the engine plays that as the nominal damper landing.
        noteSink?.noteOff(key: key, velocity: 64)
    }

    public func panic() {
        noteSink?.allNotesOff()
    }

    // MARK: - parameters

    private func parameter(_ address: AUParameterAddress) -> AUParameter? {
        audioUnit.parameterTree?.parameter(withAddress: address)
    }

    private func push(_ address: AUParameterAddress, _ value: Float) {
        guard !pushing else { return }
        parameter(address)?.setValue(value, originator: nil)
    }

    /// Reads the host's idea of every parameter back into the published
    /// properties without echoing it out again.
    public func readParameters() {
        pushing = true
        defer { pushing = false }
        sustain = parameter(PianoParameter.sustain)?.value ?? 0
        sostenuto = (parameter(PianoParameter.sostenuto)?.value ?? 0) >= 0.5
        unaCorda = (parameter(PianoParameter.unaCorda)?.value ?? 0) >= 0.5
        outputTrim = parameter(PianoParameter.outputTrim)?.value ?? 0
    }

    private func readMeters() {
        let peak = parameter(PianoParameter.peak)?.value ?? -120
        let voices = parameter(PianoParameter.voices)?.value ?? 0
        meters.update(
            peakDb: peak.isFinite ? peak : -120,
            activeVoices: voices.isFinite ? Int(voices) : 0)
    }

    // MARK: - presets

    private func selectPreset(_ number: Int) {
        guard let presets = audioUnit.factoryPresets, presets.indices.contains(number) else {
            return
        }
        guard audioUnit.currentPreset?.number != number else { return }
        busy = true
        let unit = audioUnit
        let preset = presets[number]
        // `pe_load_preset_toml` is hundreds of milliseconds of eigen-solving.
        // It is a main-thread call *of the AU's*, not of the UI's, and the AU
        // is on the other side of an XPC boundary when the appex is hosted, so
        // hopping off this thread is both allowed and necessary.
        DispatchQueue.global(qos: .userInitiated).async {
            unit.currentPreset = preset
            DispatchQueue.main.async { [weak self] in
                self?.busy = false
                self?.readParameters()
            }
        }
    }
}
