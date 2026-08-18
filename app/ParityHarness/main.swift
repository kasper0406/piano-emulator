//  main.swift — the AUv3, tested without a window, a host or a hand on a mouse.
//
//  `DISTRIBUTION.md` M2's "done when" is `auval` green and Logic. Neither can be
//  run from a build script, and neither is a *measurement*: `auval` says the
//  plugin is well-formed, and Logic says it makes a sound. What this harness
//  says is stronger and is the thing the milestone actually risks — that the
//  Swift plugin plays **the same instrument, sample for sample**, as the C
//  harness of `DECISIONS.md` 383 and the offline renderer behind it.
//
//  It does that by driving the real `internalRenderBlock` with real
//  `AURenderEvent` lists, at the host's own block cadence, with no host in the
//  room. Four things are checked:
//
//    parity     the benchmark phrase at 48 kHz, md5 of the payload, at three
//               host block sizes including one that is not a multiple of the
//               engine's 128 — because the event grid must survive that
//    rates      44.1 and 96 kHz through the boundary resampler: length, finite
//               samples, and a signal that is actually there
//    lifecycle  allocate/deallocate cycles, a rate change between them, reset
//    state      `fullState` round-trip: save, restore into a fresh AU, and
//               render the same md5 again
//
//  usage: parity-harness <preset.toml|-> <phrase.mid> [--component] [--out DIR]
//
//  SPDX-License-Identifier: MIT

import AVFoundation
import AudioToolbox
import CPianoEmulator
import CryptoKit
import Foundation

// MARK: - the recorded identity

/// The md5 of the WAV payload `ffi/harness/render.c` — and therefore
/// `cargo run -p piano-emulator -- render` — writes for `ffi/harness/phrase.mid`
/// at 48 kHz, per preset.
///
/// The first is `DECISIONS.md` 383's own number. The second is the measured
/// preset's, taken the same way; both are re-derivable in one line from a shell:
///
///     sh ffi/harness/build.sh target/release /tmp/render
///     /tmp/render presets/default.toml ffi/harness/phrase.mid /tmp/c.wav
///     tail -c +45 /tmp/c.wav | md5     # the C writer's header is 44 bytes
let expectedPhraseMD5: [String: String] = [
    "default": "f0fcb07999c00ca60110cd537de8f09e",
    "salamander-c5": "e13cd0ac9d367126ca7bf2b64b147e04",
]

// MARK: - arguments

var arguments = Array(CommandLine.arguments.dropFirst())
var useComponent = false
var outputDirectory: URL?
var positional: [String] = []
var index = 0
while index < arguments.count {
    switch arguments[index] {
    case "--component":
        useComponent = true
    case "--out":
        index += 1
        if index < arguments.count { outputDirectory = URL(fileURLWithPath: arguments[index]) }
    default:
        positional.append(arguments[index])
    }
    index += 1
}
guard positional.count == 2 else {
    FileHandle.standardError.write(
        Data("usage: parity-harness <preset.toml|-> <phrase.mid> [--component] [--out DIR]\n".utf8))
    exit(2)
}
let presetArgument = positional[0]
let midiURL = URL(fileURLWithPath: positional[1])
/// The recorded hash for *this* preset, if there is one. A preset with no
/// recorded hash is still checked for internal agreement across block sizes and
/// across a state round-trip; it just has nothing to be equal to.
let expected: String? =
    presetArgument == "-"
    ? expectedPhraseMD5["default"]
    : expectedPhraseMD5[URL(fileURLWithPath: presetArgument).deletingPathExtension().lastPathComponent]

// MARK: - helpers

func fail(_ message: String) -> Never {
    print("FAIL  \(message)")
    exit(1)
}

var failures = 0
func check(_ condition: Bool, _ message: String) {
    if condition {
        print("  ok    \(message)")
    } else {
        print("  FAIL  \(message)")
        failures += 1
    }
}

func md5(ofInterleaved samples: [Float]) -> String {
    var hasher = Insecure.MD5()
    samples.withUnsafeBufferPointer { buffer in
        hasher.update(
            bufferPointer: UnsafeRawBufferPointer(
                start: buffer.baseAddress, count: buffer.count * MemoryLayout<Float>.size))
    }
    return hasher.finalize().map { String(format: "%02x", $0) }.joined()
}

func writeWav(_ url: URL, left: [Float], right: [Float], rate: Double) throws {
    var data = Data()
    func u32(_ value: UInt32) { withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) } }
    func u16(_ value: UInt16) { withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) } }
    let frames = left.count
    let payloadBytes = UInt32(frames * 2 * MemoryLayout<Float>.size)
    data.append(contentsOf: Array("RIFF".utf8))
    u32(36 + payloadBytes)
    data.append(contentsOf: Array("WAVE".utf8))
    data.append(contentsOf: Array("fmt ".utf8))
    u32(16)
    u16(3)
    u16(2)
    u32(UInt32(rate))
    u32(UInt32(rate) * 8)
    u16(8)
    u16(32)
    data.append(contentsOf: Array("data".utf8))
    u32(payloadBytes)
    for frame in 0..<frames {
        withUnsafeBytes(of: left[frame].bitPattern.littleEndian) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: right[frame].bitPattern.littleEndian) { data.append(contentsOf: $0) }
    }
    try data.write(to: url)
}

// MARK: - the audio unit under test

/// Either our class directly — which needs no registration and is what runs in
/// a build script — or the registered component, which is the appex if
/// PluginKit has seen it.
///
/// The synchronous `AUAudioUnit(componentDescription:options:)` is deliberately
/// not used for the component: an appex loads **out of process**, and the
/// synchronous initializer answers `-10863`
/// (`kAudioUnitErr_CannotDoInCurrentContext`) for one. The asynchronous
/// instantiate is the only door, and a command-line tool has to turn its own
/// run loop for the completion to arrive.
func makeAudioUnit(viaComponent: Bool = useComponent) throws -> AUAudioUnit {
    let description = PianoIdentity.componentDescription
    guard viaComponent else {
        return try PianoAudioUnit(componentDescription: description, options: [])
    }
    var outcome: Result<AUAudioUnit, Error>?
    AUAudioUnit.instantiate(with: description, options: []) { unit, error in
        if let unit {
            outcome = .success(unit)
        } else {
            outcome = .failure(
                error
                    ?? NSError(
                        domain: NSOSStatusErrorDomain, code: -1,
                        userInfo: [NSLocalizedDescriptionKey: "no audio unit and no error"]))
        }
    }
    let deadline = Date().addingTimeInterval(30)
    while outcome == nil, Date() < deadline {
        RunLoop.current.run(mode: .default, before: Date().addingTimeInterval(0.05))
    }
    switch outcome {
    case .success(let unit): return unit
    case .failure(let error): throw error
    case nil: fail("the component did not instantiate within 30 s")
    }
}

func presetTOML() throws -> String {
    presetArgument == "-" ? "" : try String(contentsOfFile: presetArgument, encoding: .utf8)
}

/// One offline render: the AU's own `internalRenderBlock`, driven at
/// `blockSize` frames a call, with the performance's events delivered in
/// `AURenderEvent` lists at their true sample offsets.
func render(
    audioUnit: AUAudioUnit, performance: Performance, sampleRate: Double, blockSize: Int,
    presetText: String?, viaComponent: Bool = useComponent
) throws -> (left: [Float], right: [Float]) {
    if let presetText, !presetText.isEmpty {
        audioUnit.fullState = [
            PianoAudioUnit.StateKey.schemaVersion: PianoAudioUnit.stateSchemaVersion,
            PianoAudioUnit.StateKey.presetTOML: presetText,
        ]
    }
    guard let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 2) else {
        fail("no format at \(sampleRate)")
    }
    try audioUnit.outputBusses[0].setFormat(format)
    audioUnit.maximumFramesToRender = AUAudioFrameCount(blockSize)
    try audioUnit.allocateRenderResources()
    defer { audioUnit.deallocateRenderResources() }
    // In process we drive `internalRenderBlock` and build the `AURenderEvent`
    // list ourselves, which is the plugin's own path with no host in it. Out of
    // process there is no such list to build — the events cross the XPC
    // boundary through `scheduleMIDIEventBlock` and the *remote* base class
    // assembles them — so the component mode drives `renderBlock` and schedules,
    // exactly as `AVAudioEngine` would.
    let internalBlock: AUInternalRenderBlock? =
        viaComponent ? nil : audioUnit.internalRenderBlock
    let hostBlock: AURenderBlock? = viaComponent ? audioUnit.renderBlock : nil
    let schedule: AUScheduleMIDIEventBlock? =
        viaComponent ? audioUnit.scheduleMIDIEventBlock : nil
    if viaComponent, schedule == nil {
        fail("the component offers no scheduleMIDIEventBlock, so it cannot be played")
    }

    // The engine's clock is the only clock the instrument has, so the length in
    // *seconds* is fixed and the host's frame count follows from its rate —
    // exactly as `render.c` computes it.
    let engineFrames = performance.engineFrames
    let outputFrames = Int(Double(engineFrames) * sampleRate / Double(PE_ENGINE_SAMPLE_RATE))
    var left = [Float](repeating: 0, count: max(outputFrames, 1))
    var right = [Float](repeating: 0, count: max(outputFrames, 1))

    // Preallocated, because the render block is entitled to assume a host
    // behaves like one.
    let bufferList = AudioBufferList.allocate(maximumBuffers: 2)
    defer { free(bufferList.unsafeMutablePointer) }
    let eventStorage = UnsafeMutablePointer<AURenderEvent>.allocate(
        capacity: max(performance.events.count, 1))
    defer { eventStorage.deallocate() }

    var flags = AudioUnitRenderActionFlags()
    var nextEvent = 0
    var position = 0

    try left.withUnsafeMutableBufferPointer { leftBuffer in
        try right.withUnsafeMutableBufferPointer { rightBuffer in
            while position < outputFrames {
                let frames = min(blockSize, outputFrames - position)

                // Every event whose engine frame falls inside this host block.
                // At the engine's own rate the two clocks are the same clock.
                var head: UnsafeMutablePointer<AURenderEvent>?
                var tail: UnsafeMutablePointer<AURenderEvent>?
                var count = 0
                let blockEndEngineFrame = Int(
                    Double(position + frames) * Double(PE_ENGINE_SAMPLE_RATE) / sampleRate)
                while nextEvent < performance.events.count,
                    performance.events[nextEvent].frame < blockEndEngineFrame
                {
                    let timed = performance.events[nextEvent]
                    let hostFrame = Int(
                        Double(timed.frame) * sampleRate / Double(PE_ENGINE_SAMPLE_RATE))
                    let offset = max(0, min(frames - 1, hostFrame - position))
                    let slot = eventStorage.advanced(by: count)
                    var midi = AUMIDIEvent()
                    midi.next = nil
                    midi.eventSampleTime = AUEventSampleTime(position + offset)
                    midi.eventType = .MIDI
                    midi.reserved = 0
                    midi.length = 3
                    midi.cable = 0
                    midi.data = bytes(for: timed.event)
                    if let schedule {
                        var data = midi.data
                        withUnsafeBytes(of: &data) { raw in
                            schedule(
                                AUEventSampleTime(position + offset), 0, 3,
                                raw.baseAddress!.assumingMemoryBound(to: UInt8.self))
                        }
                    } else {
                        slot.pointee.MIDI = midi
                        if head == nil { head = slot } else { tail?.pointee.head.next = slot }
                        tail = slot
                    }
                    count += 1
                    nextEvent += 1
                }

                bufferList[0] = AudioBuffer(
                    mNumberChannels: 1, mDataByteSize: UInt32(frames * 4),
                    mData: UnsafeMutableRawPointer(leftBuffer.baseAddress! + position))
                bufferList[1] = AudioBuffer(
                    mNumberChannels: 1, mDataByteSize: UInt32(frames * 4),
                    mData: UnsafeMutableRawPointer(rightBuffer.baseAddress! + position))

                var timestamp = AudioTimeStamp()
                timestamp.mSampleTime = Float64(position)
                timestamp.mFlags = .sampleTimeValid

                let status: AUAudioUnitStatus
                if let internalBlock {
                    status = internalBlock(
                        &flags, &timestamp, AUAudioFrameCount(frames), 0,
                        bufferList.unsafeMutablePointer, UnsafePointer(head), nil)
                } else {
                    // The events were scheduled above; the host block only
                    // renders.
                    status = hostBlock!(
                        &flags, &timestamp, AUAudioFrameCount(frames), 0,
                        bufferList.unsafeMutablePointer, nil)
                }
                if status != noErr {
                    fail("the render block returned \(status)")
                }
                position += frames
            }
        }
    }
    return (left, right)
}

/// A `pe_event_t` back on the wire as the three MIDI 1.0 bytes a host would
/// have delivered. The round trip is deliberate: it is the *plugin's* parser
/// that the parity run has to exercise, not the harness's.
func bytes(for event: pe_event_t) -> (UInt8, UInt8, UInt8) {
    switch event.kind {
    case MIDITranslation.Kind.noteOn:
        return (0x90, UInt8(event.key & 0x7F), UInt8(min(event.vel, 127)))
    case MIDITranslation.Kind.noteOff:
        return (0x80, UInt8(event.key & 0x7F), UInt8(min(event.vel, 127)))
    case MIDITranslation.Kind.sustain:
        return (0xB0, 64, UInt8((event.value * 127).rounded()))
    case MIDITranslation.Kind.sostenuto:
        return (0xB0, 66, event.value != 0 ? 127 : 0)
    case MIDITranslation.Kind.unaCorda:
        return (0xB0, 67, event.value != 0 ? 127 : 0)
    default:
        return (0, 0, 0)
    }
}

func interleave(_ left: [Float], _ right: [Float]) -> [Float] {
    var out = [Float](repeating: 0, count: left.count * 2)
    for index in 0..<left.count {
        out[index * 2] = left[index]
        out[index * 2 + 1] = right[index]
    }
    return out
}

// MARK: - the runs

let performance = try SMFReader.load(midiURL)
let toml = try presetTOML()
print(
    """
    piano-emulator AUv3 offline harness
      audio unit   \(useComponent ? "registered component (appex if PluginKit has it)" : "PianoAudioUnit, in process")
      preset       \(presetArgument == "-" ? "the library's built-in default" : presetArgument)
      phrase       \(midiURL.lastPathComponent) — \(performance.events.count) events, \
    \(String(format: "%.3f", performance.durationSeconds)) s, \(performance.engineFrames) frames
    """)

print("\nparity — 48 kHz, md5 of the interleaved payload")
var parityHashes: [Int: String] = [:]
var reference: (left: [Float], right: [Float])?
for blockSize in [128, 256, 512, 1024] {
    let unit = try makeAudioUnit()
    let rendered = try render(
        audioUnit: unit, performance: performance, sampleRate: 48000, blockSize: blockSize,
        presetText: toml)
    let hash = md5(ofInterleaved: interleave(rendered.left, rendered.right))
    parityHashes[blockSize] = hash
    if reference == nil { reference = rendered }
    if let expected, !useComponent {
        check(
            hash == expected,
            "host block \(String(format: "%4d", blockSize)) frames → \(hash)"
                + (hash == expected ? "" : " (expected \(expected))"))
    } else {
        print("  note  host block \(String(format: "%4d", blockSize)) frames → \(hash)")
    }
    if let outputDirectory {
        try? FileManager.default.createDirectory(
            at: outputDirectory, withIntermediateDirectories: true)
        try writeWav(
            outputDirectory.appendingPathComponent("au-\(blockSize).wav"), left: rendered.left,
            right: rendered.right, rate: 48000)
    }
}
check(
    Set(parityHashes.values).count == 1,
    "every buffer size a multiple of the engine's 128 frames renders the same bytes")

// A buffer that is *not* a multiple of 128 cannot be sample-exact, and the
// reason is arithmetic rather than a defect: an event at frame 500 belongs to
// the engine block starting at 384, and a host with a 480-frame buffer only
// tells us about it after frame 480 has already been rendered. The event then
// lands on the next grid point — one 128-frame block late, 2.7 ms, which is
// exactly the grain `DECISIONS.md` 55 defines. What is measured here is that it
// is *one* block and not more.
// Out of process there is no `AURenderEvent` list to hand over: the events go
// through `scheduleMIDIEventBlock`, and because this audio unit advertises
// `audioUnitMIDIProtocol = ._2_0` the base class **up-translates** the MIDI 1.0
// bytes into UMP on the way in. Note velocities survive that exactly, by
// construction (`DECISIONS.md` 387: a 7-bit velocity `v` is `v * 512` in the
// fine lane and reads back as `v`). A 7-bit *controller* does not: MIDI 2.0
// defines 64 as centre, so CC 64 at 40 arrives as 40 << 25 over 2^32-1 =
// 0.31250 where the file reader has 40 / 127 = 0.31496. This section measures
// that difference instead of asserting it away.
if useComponent {
    print("\nprotocol — MIDI 1.0 bytes against the same phrase up-translated to UMP")
    let inProcess = try render(
        audioUnit: try makeAudioUnit(viaComponent: false), performance: performance,
        sampleRate: 48000, blockSize: 128, presetText: toml, viaComponent: false)
    let hash = md5(ofInterleaved: interleave(inProcess.left, inProcess.right))
    if let expected {
        check(hash == expected, "the same build, in process, is still the recorded render")
    }
    guard let reference else { fail("no reference render") }
    var firstDifference = -1
    var largest: Float = 0
    var sumSquares: Double = 0
    for index in 0..<min(reference.left.count, inProcess.left.count) {
        let delta = max(
            abs(reference.left[index] - inProcess.left[index]),
            abs(reference.right[index] - inProcess.right[index]))
        if delta != 0, firstDifference < 0 { firstDifference = index }
        largest = max(largest, delta)
        sumSquares += Double(delta) * Double(delta)
    }
    let rms = (sumSquares / Double(inProcess.left.count)).squareRoot()
    if firstDifference < 0 {
        check(true, "the two protocols render the same bytes")
    } else {
        print(
            String(
                format:
                    "  note  identical for the first %d frames (%.3f s), then %.1f dBFS peak / "
                    + "%.1f dBFS rms apart — the up-translated CC 64",
                firstDifference, Double(firstDifference) / 48000, 20 * log10(Double(largest)),
                20 * log10(rms)))
        check(
            largest < 1e-3,
            String(format: "and the difference stays under -60 dBFS (peak %.2e)", largest))
    }
}

print("\ngrid — host buffers that are not a multiple of 128")
for blockSize in [96, 480] {
    let unit = try makeAudioUnit()
    let rendered = try render(
        audioUnit: unit, performance: performance, sampleRate: 48000, blockSize: blockSize,
        presetText: toml)
    guard let reference else { fail("no reference render") }
    var firstDifference = -1
    var largest: Float = 0
    for index in 0..<min(rendered.left.count, reference.left.count) {
        let delta = max(
            abs(rendered.left[index] - reference.left[index]),
            abs(rendered.right[index] - reference.right[index]))
        if delta != 0, firstDifference < 0 { firstDifference = index }
        largest = max(largest, delta)
    }
    let finite = rendered.left.allSatisfy(\.isFinite) && rendered.right.allSatisfy(\.isFinite)
    check(finite, "\(blockSize)-frame buffers still produce finite samples")
    if firstDifference < 0 {
        print("  note  \(blockSize)-frame buffers happen to be sample-exact here too")
    } else {
        print(
            "  note  \(blockSize)-frame buffers: first difference at frame \(firstDifference), "
                + "largest \(String(format: "%.4f", largest)) — onsets one 128-frame block late")
    }
}

print("\nrates — the boundary resampler")
for rate in [44100.0, 96000.0] {
    let unit = try makeAudioUnit()
    let rendered = try render(
        audioUnit: unit, performance: performance, sampleRate: rate, blockSize: 512,
        presetText: toml)
    let expected = Int(Double(performance.engineFrames) * rate / Double(PE_ENGINE_SAMPLE_RATE))
    let peak = rendered.left.map(abs).max() ?? 0
    let finite = rendered.left.allSatisfy(\.isFinite) && rendered.right.allSatisfy(\.isFinite)
    check(rendered.left.count == expected, "\(Int(rate)) Hz renders \(expected) frames")
    check(finite, "\(Int(rate)) Hz produces finite samples throughout")
    check(peak > 0.001, "\(Int(rate)) Hz produces a signal (peak \(String(format: "%.4f", peak)))")
}

print("\nlifecycle — allocate, deallocate, change rate, reset")
do {
    let unit = try makeAudioUnit()
    guard let format48 = AVAudioFormat(standardFormatWithSampleRate: 48000, channels: 2),
        let format44 = AVAudioFormat(standardFormatWithSampleRate: 44100, channels: 2)
    else { fail("no formats") }
    for cycle in 0..<3 {
        try unit.outputBusses[0].setFormat(cycle % 2 == 0 ? format48 : format44)
        unit.maximumFramesToRender = cycle % 2 == 0 ? 512 : 1024
        try unit.allocateRenderResources()
        check(unit.renderResourcesAllocated, "cycle \(cycle): render resources allocated")
        unit.reset()
        unit.deallocateRenderResources()
        check(!unit.renderResourcesAllocated, "cycle \(cycle): render resources released")
    }
    // And it still plays afterwards.
    let rendered = try render(
        audioUnit: unit, performance: performance, sampleRate: 48000, blockSize: 128,
        presetText: toml)
    let hash = md5(ofInterleaved: interleave(rendered.left, rendered.right))
    check(
        hash == parityHashes[128], "after three cycles the instrument is unchanged → \(hash)")
}

print("\nstate — fullState round-trip")
do {
    let saver = try makeAudioUnit()
    if !toml.isEmpty {
        saver.fullState = [
            PianoAudioUnit.StateKey.schemaVersion: PianoAudioUnit.stateSchemaVersion,
            PianoAudioUnit.StateKey.presetTOML: toml,
        ]
    }
    guard let format = AVAudioFormat(standardFormatWithSampleRate: 48000, channels: 2) else {
        fail("no format")
    }
    try saver.outputBusses[0].setFormat(format)
    saver.maximumFramesToRender = 512
    try saver.allocateRenderResources()
    saver.parameterTree?.parameter(withAddress: PianoParameter.sustain)?.value = 0.42
    saver.parameterTree?.parameter(withAddress: PianoParameter.outputTrim)?.value = -3.5
    guard let saved = saver.fullState else { fail("fullState is nil") }
    saver.deallocateRenderResources()

    check(
        saved[PianoAudioUnit.StateKey.schemaVersion] as? Int == PianoAudioUnit.stateSchemaVersion,
        "the state carries a schema version")
    check(
        (saved[PianoAudioUnit.StateKey.presetTOML] as? String)?.isEmpty == false,
        "the state carries the whole preset as TOML "
            + "(\((saved[PianoAudioUnit.StateKey.presetTOML] as? String)?.utf8.count ?? 0) bytes)")
    let plist = try PropertyListSerialization.data(
        fromPropertyList: saved, format: .binary, options: 0)
    check(!plist.isEmpty, "the state is a property list a host can write (\(plist.count) bytes)")

    let restored = try makeAudioUnit()
    restored.fullState = saved
    check(
        restored.parameterTree?.parameter(withAddress: PianoParameter.sustain)?.value == 0.42,
        "the sustain pedal came back at 0.42")
    check(
        restored.parameterTree?.parameter(withAddress: PianoParameter.outputTrim)?.value == -3.5,
        "the trim came back at -3.5 dB")

    // And the restored instrument renders the phrase identically. The pedals
    // are put back where the parity run had them first, because a restored
    // sustain of 0.42 is a different performance.
    restored.parameterTree?.parameter(withAddress: PianoParameter.sustain)?.value = 0
    restored.parameterTree?.parameter(withAddress: PianoParameter.outputTrim)?.value = 0
    let rendered = try render(
        audioUnit: restored, performance: performance, sampleRate: 48000, blockSize: 128,
        presetText: nil)
    let hash = md5(ofInterleaved: interleave(rendered.left, rendered.right))
    check(
        hash == parityHashes[128], "the restored instrument is the same instrument → \(hash)")

    // A state from the future is refused rather than half-read.
    let future = try makeAudioUnit()
    var newer = saved
    newer[PianoAudioUnit.StateKey.schemaVersion] = PianoAudioUnit.stateSchemaVersion + 1
    future.fullState = newer
    check(
        future.parameterTree?.parameter(withAddress: PianoParameter.sustain)?.value == 0,
        "a state written by a newer schema is refused, not half-read")
}

// The falsification test for the one defect `auval` found: a host reaching the
// plugin through the AUv2 bridge schedules parameter events under the hashed
// `AudioUnitParameterID`, not under the tree's address, and narrowing that to
// an `Int32` before comparing it is a Swift trap — an appex that dies mid-render
// with `EXC_BREAKPOINT`. On the code before the fix, this section crashes the
// process.
if !useComponent {
    print("\nrobustness — a parameter event under an address no parameter has")
    let unit = try PianoAudioUnit(
        componentDescription: PianoIdentity.componentDescription, options: [])
    guard let format = AVAudioFormat(standardFormatWithSampleRate: 48000, channels: 2) else {
        fail("no format")
    }
    try unit.outputBusses[0].setFormat(format)
    unit.maximumFramesToRender = 128
    try unit.allocateRenderResources()
    let block = unit.internalRenderBlock
    let storage = UnsafeMutablePointer<AURenderEvent>.allocate(capacity: 2)
    defer { storage.deallocate() }

    var hashed = AUParameterEvent()
    hashed.next = storage.advanced(by: 1)
    hashed.eventSampleTime = 0
    hashed.eventType = .parameter
    hashed.rampDurationSampleFrames = 0
    // What `auval` actually scheduled for "Sustain": the AUv2 bridge's hash of
    // the parameter identifier, printed by `auval` as -1317617108.
    hashed.parameterAddress = AUParameterAddress(UInt32(bitPattern: -1_317_617_108))
    hashed.value = 1
    storage.pointee.parameter = hashed

    var real = AUParameterEvent()
    real.next = nil
    real.eventSampleTime = 0
    real.eventType = .parameter
    real.rampDurationSampleFrames = 0
    real.parameterAddress = PianoParameter.sustain
    real.value = 0.75
    storage.advanced(by: 1).pointee.parameter = real

    var left = [Float](repeating: 0, count: 128)
    var right = [Float](repeating: 0, count: 128)
    let bufferList = AudioBufferList.allocate(maximumBuffers: 2)
    defer { free(bufferList.unsafeMutablePointer) }
    var flags = AudioUnitRenderActionFlags()
    var timestamp = AudioTimeStamp()
    timestamp.mSampleTime = 0
    timestamp.mFlags = .sampleTimeValid
    left.withUnsafeMutableBufferPointer { l in
        right.withUnsafeMutableBufferPointer { r in
            bufferList[0] = AudioBuffer(
                mNumberChannels: 1, mDataByteSize: 512,
                mData: UnsafeMutableRawPointer(l.baseAddress!))
            bufferList[1] = AudioBuffer(
                mNumberChannels: 1, mDataByteSize: 512,
                mData: UnsafeMutableRawPointer(r.baseAddress!))
            let status = block(
                &flags, &timestamp, 128, 0, bufferList.unsafeMutablePointer,
                UnsafePointer(storage), nil)
            check(status == noErr, "the render survives it and returns noErr")
        }
    }
    check(
        unit.appliedParameters[Int(PE_PARAM_SUSTAIN)] == 0.75,
        "the sustain event beside it reached the engine (0.75)")
    check(
        unit.appliedParameters[Int(PE_PARAM_OUTPUT_TRIM)] == 0,
        "and nothing wrote through the address that has no parameter")
    unit.deallocateRenderResources()
}

print("")
if failures == 0 {
    print("all green")
    exit(0)
} else {
    print("\(failures) failed")
    exit(1)
}
