//  PianoIdentity.swift — the four-character codes, in one place.
//
//  These four bytes are the plugin's name to the whole of Core Audio: the host
//  writes them into a project file and expects to find the same instrument
//  years later, so they are chosen once and never changed. They are repeated in
//  `app/AUv3/Info.plist` (`NSExtensionAttributes` -> `AudioComponents`), which
//  is what PluginKit reads, and `app/build.sh` checks the two agree rather than
//  trusting that they do.
//
//  SPDX-License-Identifier: MIT

import AudioToolbox

public enum PianoIdentity {
    /// `aumu` — a music device, i.e. an instrument that makes sound from MIDI
    /// rather than from an input bus. Fixed by Apple; the only choice we make
    /// is the two below.
    public static let type: OSType = kAudioUnitType_MusicDevice

    /// `Pemu` — Piano EMUlator. Uppercase in the first position because Apple
    /// reserves all-lowercase four-character codes for itself.
    public static let subtype: OSType = fourCharCode("Pemu")

    /// `KsNi` — the author's initials. **This is the one identifier that must
    /// be registered with Apple before the plugin ships to anyone**
    /// (developer.apple.com's manufacturer-code registration); until then it is
    /// unique enough for a machine and honest about being provisional.
    public static let manufacturer: OSType = fourCharCode("KsNi")

    /// 1.0.0, in Core Audio's packed form: `0xMMMMmmbb`.
    public static let version: UInt32 = 0x0001_0000

    public static var componentDescription: AudioComponentDescription {
        AudioComponentDescription(
            componentType: type,
            componentSubType: subtype,
            componentManufacturer: manufacturer,
            componentFlags: 0,
            componentFlagsMask: 0
        )
    }

    /// What the AU calls itself. The colon is Core Audio's own convention:
    /// everything before it is the manufacturer, everything after the product.
    public static let displayName = "Kasper Nielsen: Piano Emulator"

    static func fourCharCode(_ string: StaticString) -> OSType {
        precondition(string.utf8CodeUnitCount == 4, "a four-character code has four characters")
        var code: OSType = 0
        for index in 0..<4 {
            code = (code << 8) | OSType(string.utf8Start[index])
        }
        return code
    }
}
