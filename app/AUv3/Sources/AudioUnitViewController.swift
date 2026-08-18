//  AudioUnitViewController.swift — the appex's principal class.
//
//  PluginKit instantiates this (`NSExtensionPrincipalClass`), the host asks it
//  for an audio unit (`AUAudioUnitFactory`), and it also *is* the plugin's
//  view. Both halves are required by the `com.apple.AudioUnit-UI` extension
//  point; there is no separate "audio only" entry.
//
//  SPDX-License-Identifier: MIT

import AVFoundation
import CoreAudioKit
import SwiftUI

public final class AudioUnitViewController: AUViewController, AUAudioUnitFactory {
    private var piano: PianoAudioUnit?
    private var controller: PianoController?
    private var hosted: NSViewController?
    private var noteSink: PluginNoteSink?

    public override func loadView() {
        // No storyboard and no xib: the appex ships one Swift file's worth of
        // view, built here.
        view = NSView(frame: NSRect(x: 0, y: 0, width: 680, height: 420))
    }

    public override func viewDidLoad() {
        super.viewDidLoad()
        installPanelIfPossible()
    }

    // PlugInKit instantiates this class to get at `createAudioUnit`, whether or
    // not a host ever puts the view on screen — so the meter polling is tied to
    // the view actually appearing. Without this the appex runs an AppKit
    // display cycle for an invisible window: 36.7 % of one core with nothing
    // playing, measured. (`PianoMeters` carries the rest of that story.)
    public override func viewDidAppear() {
        super.viewDidAppear()
        controller?.startMetering()
    }

    public override func viewDidDisappear() {
        super.viewDidDisappear()
        controller?.stopMetering()
    }

    public func createAudioUnit(with componentDescription: AudioComponentDescription) throws
        -> AUAudioUnit
    {
        let unit = try PianoAudioUnit(componentDescription: componentDescription, options: [])
        piano = unit
        DispatchQueue.main.async { [weak self] in self?.installPanelIfPossible() }
        return unit
    }

    @MainActor
    private func installPanelIfPossible() {
        guard isViewLoaded, hosted == nil, let piano else { return }
        let sink = PluginNoteSink(audioUnit: piano)
        let controller = PianoController(audioUnit: piano)
        controller.noteSink = sink
        self.controller = controller
        self.noteSink = sink

        let hosting = NSHostingController(rootView: PianoPanelView(controller: controller))
        addChild(hosting)
        hosting.view.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(hosting.view)
        NSLayoutConstraint.activate([
            hosting.view.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            hosting.view.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            hosting.view.topAnchor.constraint(equalTo: view.topAnchor),
            hosting.view.bottomAnchor.constraint(equalTo: view.bottomAnchor),
        ])
        hosted = hosting
    }
}

/// The plugin view's own keyboard, straight into the engine's SPSC queue.
///
/// The host's MIDI does not come this way — it arrives on the audio thread in
/// the render block's event list — and the two never race: `pe_post_event` is
/// drained by `pe_render` before the block it belongs to, which is exactly what
/// the queue is for (`piano_emulator.h`).
final class PluginNoteSink: PianoNoteSink {
    private let audioUnit: PianoAudioUnit

    init(audioUnit: PianoAudioUnit) {
        self.audioUnit = audioUnit
    }

    func noteOn(key: UInt8, velocity: UInt8) {
        audioUnit.post(
            MIDITranslation.event(
                MIDITranslation.Kind.noteOn, key: UInt32(key), vel: UInt32(velocity)))
    }

    func noteOff(key: UInt8, velocity: UInt8) {
        audioUnit.post(
            MIDITranslation.event(
                MIDITranslation.Kind.noteOff, key: UInt32(key), vel: UInt32(velocity)))
    }

    func allNotesOff() {
        audioUnit.post(MIDITranslation.event(MIDITranslation.Kind.allOff))
    }
}
