//  PianoPanelView.swift — the controls, shared by the plugin's view and the app.
//
//  Everything here talks to `PianoController`, which talks to the
//  `AUParameterTree`. Nothing in this file knows whether the audio unit is in
//  this process or in an appex on the other side of an XPC boundary.
//
//  SPDX-License-Identifier: MIT

import SwiftUI

public struct PianoPanelView: View {
    @ObservedObject public var controller: PianoController
    /// The app shows a MIDI section and a wider keyboard; the plugin's own
    /// view does not, because the host owns its MIDI.
    public var extraContent: AnyView?

    @State private var lowestKey: Int = 36

    public init(controller: PianoController, extraContent: AnyView? = nil) {
        self.controller = controller
        self.extraContent = extraContent
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            header
            Divider()
            HStack(alignment: .top, spacing: 24) {
                pedals
                Divider().frame(height: 96)
                meters
            }
            if let extraContent {
                Divider()
                extraContent
            }
            Divider()
            keyboard
        }
        .padding(16)
        .frame(minWidth: 620, minHeight: 380)
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Piano Emulator").font(.title2.weight(.semibold))
                Text("Physically modelled grand — modal synthesis, no samples")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 4) {
                Picker("Instrument", selection: $controller.presetNumber) {
                    ForEach(Array(controller.presetNames.enumerated()), id: \.offset) {
                        index, name in
                        Text(name).tag(index)
                    }
                }
                .labelsHidden()
                .frame(width: 240)
                .disabled(controller.busy)
                if controller.busy {
                    Text("building the instrument…")
                        .font(.caption2).foregroundStyle(.secondary)
                }
            }
        }
    }

    private var pedals: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Pedals").font(.headline)
            HStack(spacing: 8) {
                Text("Sustain").frame(width: 66, alignment: .leading)
                Slider(value: $controller.sustain, in: 0...1)
                    .frame(width: 190)
                Text(String(format: "%3.0f %%", controller.sustain * 100))
                    .font(.system(.caption, design: .monospaced))
                    .frame(width: 46, alignment: .trailing)
            }
            Text("Continuous: half-pedalling reaches the dampers as the fraction it was played at.")
                .font(.caption2).foregroundStyle(.secondary)
            HStack(spacing: 16) {
                Toggle("Sostenuto", isOn: $controller.sostenuto)
                Toggle("Una corda", isOn: $controller.unaCorda)
                Button("All notes off") { controller.panic() }
            }
            HStack(spacing: 8) {
                Text("Trim").frame(width: 66, alignment: .leading)
                Slider(value: $controller.outputTrim, in: -24...12)
                    .frame(width: 190)
                Text(String(format: "%+5.1f dB", controller.outputTrim))
                    .font(.system(.caption, design: .monospaced))
                    .frame(width: 60, alignment: .trailing)
            }
        }
    }

    private var meters: some View {
        MetersView(meters: controller.meters)
    }

    private var keyboard: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Keyboard").font(.headline)
                Spacer()
                Button {
                    lowestKey = max(21, lowestKey - 12)
                } label: {
                    Image(systemName: "chevron.left")
                }
                .disabled(lowestKey <= 21)
                Text(noteName(lowestKey)).font(.system(.caption, design: .monospaced))
                Button {
                    lowestKey = min(84, lowestKey + 12)
                } label: {
                    Image(systemName: "chevron.right")
                }
                .disabled(lowestKey >= 84)
            }
            KeyboardView(
                lowestKey: lowestKey, octaves: 4,
                onNoteOn: { key, velocity in controller.noteOn(key: key, velocity: velocity) },
                onNoteOff: { key in controller.noteOff(key: key) }
            )
            .frame(height: 110)
            Text("Click low on a key for a loud note, high for a soft one; drag to gliss.")
                .font(.caption2).foregroundStyle(.secondary)
        }
    }

    private func noteName(_ key: Int) -> String {
        let names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        return "\(names[key % 12])\(key / 12 - 1)"
    }
}

/// Its own view over its own object, so that a moving meter does not invalidate
/// the keyboard twenty-four times a second. See `PianoMeters`.
public struct MetersView: View {
    @ObservedObject public var meters: PianoMeters
    public init(meters: PianoMeters) { self.meters = meters }

    public var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Output").font(.headline)
            LevelMeterView(db: meters.peakDb)
                .frame(width: 180, height: 14)
            Text(String(format: "peak %.1f dBFS", meters.peakDb))
                .font(.system(.caption, design: .monospaced))
            Text("\(meters.activeVoices) voices ringing")
                .font(.caption).foregroundStyle(.secondary)
            Text(
                """
                A struck note and every string resonating with it: one note can \
                report seventeen.
                """
            )
            .font(.caption2).foregroundStyle(.secondary).frame(width: 200, alignment: .leading)
        }
    }
}

public struct LevelMeterView: View {
    public var db: Float
    public init(db: Float) { self.db = db }

    public var body: some View {
        GeometryReader { geometry in
            let fraction = CGFloat(max(0, min(1, (db + 60) / 66)))
            ZStack(alignment: .leading) {
                RoundedRectangle(cornerRadius: 3).fill(Color.secondary.opacity(0.18))
                RoundedRectangle(cornerRadius: 3)
                    .fill(db > -1 ? Color.red : (db > -6 ? Color.orange : Color.green))
                    .frame(width: geometry.size.width * fraction)
            }
        }
    }
}
