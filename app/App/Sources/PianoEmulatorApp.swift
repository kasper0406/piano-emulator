//  PianoEmulatorApp.swift — the container app, which is also the instrument.
//
//  It is a real instrument on its own: an on-screen keyboard with velocity, the
//  three pedals, both factory presets, a level meter, and Core MIDI input for
//  anyone who has a keyboard. That is App Review 4.2.3(i)'s requirement and it
//  is also just what you want when you are not in a DAW.
//
//  Launching it is also what registers the AUv3 with PluginKit
//  (`DISTRIBUTION.md` §Plugin formats), which is why the window says so.
//
//  SPDX-License-Identifier: MIT

import SwiftUI

@main
struct PianoEmulatorApp: App {
    @StateObject private var host = AudioHost()

    var body: some Scene {
        WindowGroup("Piano Emulator") {
            ContentView(host: host)
                .onAppear { host.start() }
                .onDisappear { host.stop() }
        }
        .windowResizability(.contentSize)
    }
}

struct ContentView: View {
    @ObservedObject var host: AudioHost

    var body: some View {
        if let controller = host.controller {
            PianoPanelView(controller: controller, extraContent: AnyView(hostingSection))
                .onAppear { controller.startMetering() }
                .onDisappear { controller.stopMetering() }
        } else {
            VStack(spacing: 12) {
                ProgressView()
                Text(host.statusLine).font(.callout)
                if let failure = host.failure {
                    Text(failure)
                        .font(.caption)
                        .foregroundStyle(.red)
                        .multilineTextAlignment(.center)
                        .frame(maxWidth: 420)
                }
            }
            .frame(width: 620, height: 380)
        }
    }

    private var hostingSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Image(
                    systemName: host.hosting == .appex
                        ? "checkmark.seal.fill" : "exclamationmark.triangle.fill"
                )
                .foregroundStyle(host.hosting == .appex ? Color.green : Color.orange)
                Text(host.hosting?.rawValue ?? "not hosted").font(.callout.weight(.medium))
            }
            Text(host.statusLine).font(.caption).foregroundStyle(.secondary)
            Text("MIDI out to the instrument: \(host.midiPath)")
                .font(.caption).foregroundStyle(.secondary)
            Text(
                host.midiSources.isEmpty
                    ? "No MIDI sources connected — play the keys below, or plug in a controller."
                    : "MIDI in: " + host.midiSources.joined(separator: ", ")
            )
            .font(.caption).foregroundStyle(.secondary)
            if host.hosting == .inProcess {
                Text(
                    """
                    Copy the app to /Applications and launch it once to register the \
                    Audio Unit with PluginKit; hosts will then find it, and so will this window.
                    """
                )
                .font(.caption2).foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
