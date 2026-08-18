//  PresetLibrary.swift — the two shipped instruments, read out of the bundle.
//
//  `presets/*.toml` are copied into the appex (and into the container app, for
//  the in-process fallback of `AudioHost`) by the Xcode target's resource
//  phase; `app/build.sh` fails the build if either is missing, because a
//  factory preset that is not in the bundle is a preset the plugin advertises
//  and cannot load.
//
//  SPDX-License-Identifier: MIT

import Foundation

public struct PianoFactoryPreset: Sendable {
    /// What the host's preset menu shows.
    public let name: String
    /// The resource, without the `.toml`.
    public let resource: String
}

public enum PresetLibrary {
    /// Order is the factory-preset numbering and is part of the saved state:
    /// a project stores the *number*, so these never get reordered. New presets
    /// are appended.
    public static let presets: [PianoFactoryPreset] = [
        PianoFactoryPreset(name: "Concert Grand", resource: "default"),
        PianoFactoryPreset(name: "Salamander C5 (measured)", resource: "salamander-c5"),
    ]

    /// The bundle the AU implementation lives in — the appex when hosted, the
    /// container app when the app is running the class in process. `Bundle(for:)`
    /// answers both without either having to know which it is.
    public static var bundle: Bundle {
        Bundle(for: BundleToken.self)
    }

    public static func toml(forPresetNumber number: Int) -> String? {
        guard presets.indices.contains(number) else { return nil }
        return toml(named: presets[number].resource)
    }

    public static func toml(named resource: String) -> String? {
        guard let url = bundle.url(forResource: resource, withExtension: "toml") else {
            return nil
        }
        return try? String(contentsOf: url, encoding: .utf8)
    }

    private final class BundleToken {}
}
