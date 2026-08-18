//  PianoAudioUnit.swift — the AUv3, over the C ABI of `ffi/`.
//
//  The whole plugin is this class plus `MIDITranslation`. It owns a `pe_engine`
//  and nothing else: every voicing number is in the preset, every sample comes
//  out of `pe_render`, and the resampler behind the ABI makes a host at 44.1 or
//  96 kHz the engine's problem rather than ours (`DISTRIBUTION.md` M0).
//
//  ## The thread contract, which is the reason this file is written the way it is
//
//  `piano_emulator.h` states one per function. Three of them shape the code:
//
//  - `pe_render` and `pe_event` are **audio thread**, allocation-free,
//    lock-free, syscall-free. So the render block captures a raw
//    `pe_render_state *` (see `pe_shim.h`) and touches no Swift object: no ARC,
//    no dictionaries, no `Array`, no `String`, no lazy `static let`.
//  - `pe_create` and `pe_load_preset_toml` are **main thread** and take
//    hundreds of milliseconds. A preset change therefore builds a *new* engine
//    off the audio thread and publishes the pointer with a release store; the
//    replaced engine is not destroyed, it is **retired** and freed in
//    `deallocateRenderResources`. Nothing is ever freed under a live render, so
//    there is no lock in the render path and no wait on the main one.
//  - `pe_post_event` is the standalone's queue and is not used here at all: a
//    host hands an AUv3 its MIDI on the audio thread, in the render block's
//    event list, so this file calls `pe_event` directly.
//
//  ## Event timing
//
//  An event takes effect at the start of the 128-frame block that contains its
//  sample (`DECISIONS.md` 55). The render block honours that *across host
//  buffer boundaries*: it keeps a running engine-frame count, and splits its
//  own rendering at the 128-frame grid points the events fall on, so a host
//  running 512- or 1024-frame buffers gets the same 2.7 ms quantisation as one
//  running 128 — and not the buffer-sized quantisation that applying everything
//  at the top of the block would give. That equality is what
//  `app/ParityHarness` measures.
//
//  SPDX-License-Identifier: MIT

import AVFoundation
import AudioToolbox
import CPianoEmulator
import CoreMIDI
import Foundation
import os

public final class PianoAudioUnit: AUAudioUnit {
    // MARK: - state

    /// Everything the render block touches. One allocation, made in `init` and
    /// freed in `deinit`; the render block only ever sees this pointer.
    private let state: UnsafeMutablePointer<pe_render_state>

    /// The engine the render block is currently using, as the main thread sees
    /// it. `nil` between `deallocateRenderResources` and the next allocate.
    private var liveEngine: OpaquePointer?

    /// Engines a preset change replaced. They are still reachable by a render
    /// that was in flight at the moment of the swap, so they are freed at
    /// `deallocateRenderResources` and not before.
    private var retiredEngines: [OpaquePointer] = []

    private var scratchLeft: UnsafeMutablePointer<Float>?
    private var scratchRight: UnsafeMutablePointer<Float>?

    /// The preset the engine is built from, as TOML. Never `nil`: the AU has an
    /// instrument from the moment it exists.
    private var presetText: String

    private var _currentPreset: AUAudioUnitPreset?
    private let _factoryPresets: [AUAudioUnitPreset]
    private let outputBus: AUAudioUnitBus
    private var _inputBusses: AUAudioUnitBusArray!
    private var _outputBusses: AUAudioUnitBusArray!

    private static let log = Logger(subsystem: "dev.pianoemulator.au", category: "PianoAudioUnit")

    // MARK: - state serialization

    /// Bumped when the *meaning* of a key changes. A state carrying a higher
    /// number is refused rather than half-read (`DISTRIBUTION.md` §Presets and
    /// state).
    public static let stateSchemaVersion = 1

    public enum StateKey {
        public static let schemaVersion = "pianoEmulator.schemaVersion"
        public static let abiVersion = "pianoEmulator.abiVersion"
        public static let presetTOML = "pianoEmulator.presetTOML"
        public static let factoryPresetNumber = "pianoEmulator.factoryPresetNumber"
    }

    // MARK: - construction

    public override init(
        componentDescription: AudioComponentDescription,
        options: AudioComponentInstantiationOptions = []
    ) throws {
        guard pe_abi_version() == PE_ABI_VERSION else {
            throw NSError(
                domain: NSOSStatusErrorDomain, code: Int(kAudioUnitErr_FailedInitialization),
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "the linked piano-emulator library and this header disagree about the ABI"
                ])
        }

        // 48 kHz is only the *default* the bus advertises; the host sets its own
        // rate on the bus before `allocateRenderResources`, and the boundary
        // resampler behind `pe_create` takes it from there.
        guard
            let format = AVAudioFormat(
                standardFormatWithSampleRate: Double(PE_ENGINE_SAMPLE_RATE), channels: 2)
        else {
            throw NSError(
                domain: NSOSStatusErrorDomain, code: Int(kAudioUnitErr_FormatNotSupported))
        }
        outputBus = try AUAudioUnitBus(format: format)
        outputBus.maximumChannelCount = 2

        state = UnsafeMutablePointer<pe_render_state>.allocate(capacity: 1)
        pe_state_init(state)

        _factoryPresets = PresetLibrary.presets.enumerated().map { index, preset in
            let factory = AUAudioUnitPreset()
            factory.number = index
            factory.name = preset.name
            return factory
        }
        // The bundle is the source of the instrument. If a build has somehow
        // shipped without its presets the AU still works — `pe_create` starts on
        // the library's own built-in default, which `presets/default.toml`
        // round-trips to float for float (`DECISIONS.md` 383) — so the fallback
        // is an empty string meaning "leave the built-in alone".
        presetText = PresetLibrary.toml(forPresetNumber: 0) ?? ""

        try super.init(componentDescription: componentDescription, options: options)

        _inputBusses = AUAudioUnitBusArray(audioUnit: self, busType: .input, busses: [])
        _outputBusses = AUAudioUnitBusArray(audioUnit: self, busType: .output, busses: [outputBus])
        maximumFramesToRender = 512
        buildParameterTree()
        _currentPreset = _factoryPresets.first
    }

    deinit {
        if let liveEngine { pe_destroy(liveEngine) }
        for engine in retiredEngines { pe_destroy(engine) }
        scratchLeft?.deallocate()
        scratchRight?.deallocate()
        state.deallocate()
    }

    // MARK: - busses and capabilities

    public override var inputBusses: AUAudioUnitBusArray { _inputBusses }
    public override var outputBusses: AUAudioUnitBusArray { _outputBusses }

    /// No inputs, two outputs. An instrument.
    public override var channelCapabilities: [NSNumber]? { [0, 2] }

    public override var canProcessInPlace: Bool { false }

    /// Parse UMP from day one (`DISTRIBUTION.md` §MIDI 2.0). Core MIDI does the
    /// 1.0 ↔ 2.0 translation, so advertising 2.0 costs nothing and the 16-bit
    /// velocities of a controller that sends them survive the trip — and the
    /// byte-oriented `AURenderEventMIDI` path is still implemented below, for
    /// hosts that deliver it.
    public override var audioUnitMIDIProtocol: MIDIProtocolID { ._2_0 }

    /// The host may store user presets for us; the base class does it through
    /// `fullState`, which carries the whole preset text.
    public override var supportsUserPresets: Bool { true }

    // MARK: - parameters

    private func buildParameterTree() {
        let sustain = AUParameterTree.createParameter(
            withIdentifier: "sustain", name: "Sustain", address: AUParameterAddress(PE_PARAM_SUSTAIN),
            min: 0, max: 1, unit: .generic, unitName: nil,
            flags: [.flag_IsReadable, .flag_IsWritable], valueStrings: nil,
            dependentParameters: nil)
        let sostenuto = AUParameterTree.createParameter(
            withIdentifier: "sostenuto", name: "Sostenuto",
            address: AUParameterAddress(PE_PARAM_SOSTENUTO),
            min: 0, max: 1, unit: .boolean, unitName: nil,
            flags: [.flag_IsReadable, .flag_IsWritable], valueStrings: nil,
            dependentParameters: nil)
        let unaCorda = AUParameterTree.createParameter(
            withIdentifier: "unaCorda", name: "Una Corda",
            address: AUParameterAddress(PE_PARAM_UNA_CORDA),
            min: 0, max: 1, unit: .boolean, unitName: nil,
            flags: [.flag_IsReadable, .flag_IsWritable], valueStrings: nil,
            dependentParameters: nil)
        let outputTrim = AUParameterTree.createParameter(
            withIdentifier: "outputTrim", name: "Output Trim",
            address: AUParameterAddress(PE_PARAM_OUTPUT_TRIM),
            min: -24, max: 12, unit: .decibels, unitName: nil,
            flags: [.flag_IsReadable, .flag_IsWritable], valueStrings: nil,
            dependentParameters: nil)

        sustain.value = 0
        sostenuto.value = 0
        unaCorda.value = 0
        outputTrim.value = 0

        // Two meters. They are parameters because that is the one channel an
        // AUv3 has that crosses the XPC boundary on demand: the standalone app
        // hosting its own appex reads the same two numbers a DAW would, and the
        // plugin's own view reads them the same way in process.
        let voices = AUParameterTree.createParameter(
            withIdentifier: "activeVoices", name: "Active Voices",
            address: PianoParameter.voices,
            min: 0, max: 88, unit: .generic, unitName: nil,
            flags: [.flag_IsReadable, .flag_MeterReadOnly], valueStrings: nil,
            dependentParameters: nil)
        let peak = AUParameterTree.createParameter(
            withIdentifier: "peakLevel", name: "Peak Level", address: PianoParameter.peak,
            min: -120, max: 6, unit: .decibels, unitName: nil,
            flags: [.flag_IsReadable, .flag_MeterReadOnly], valueStrings: nil,
            dependentParameters: nil)

        let tree = AUParameterTree.createTree(withChildren: [
            sustain, sostenuto, unaCorda, outputTrim, voices, peak,
        ])

        // The observers capture the raw state pointer and nothing else: a host
        // may call them from any thread, including the audio one, and a
        // captured `self` would put ARC traffic there.
        let stateRef = state
        tree.implementorValueObserver = { parameter, value in
            guard parameter.address < AUParameterAddress(PE_PARAM_COUNT) else { return }
            pe_state_set_param(stateRef, Int32(parameter.address), value)
        }
        tree.implementorValueProvider = { parameter in
            switch parameter.address {
            case PianoParameter.voices:
                return AUValue(pe_state_voices(stateRef))
            case PianoParameter.peak:
                let peak = max(pe_state_peak_left(stateRef), pe_state_peak_right(stateRef))
                return peak > 0 ? 20 * log10f(peak) : -120
            default:
                guard parameter.address < AUParameterAddress(PE_PARAM_COUNT) else { return 0 }
                return pe_state_param(stateRef, Int32(parameter.address))
            }
        }
        tree.implementorStringFromValueCallback = { parameter, valuePointer in
            let value = valuePointer?.pointee ?? parameter.value
            switch parameter.address {
            case PianoParameter.sustain:
                return String(format: "%.0f %%", value * 100)
            case PianoParameter.sostenuto, PianoParameter.unaCorda:
                return value >= 0.5 ? "Down" : "Up"
            case PianoParameter.voices:
                return String(format: "%.0f", value)
            default:
                return String(format: "%+.1f dB", value)
            }
        }

        parameterTree = tree
        // Push the initial values through the observer so the render block's
        // first comparison has something true to compare against.
        for parameter in tree.allParameters where parameter.address < AUParameterAddress(PE_PARAM_COUNT) {
            pe_state_set_param(state, Int32(parameter.address), parameter.value)
        }
    }

    // MARK: - presets

    public override var factoryPresets: [AUAudioUnitPreset]? { _factoryPresets }

    public override var currentPreset: AUAudioUnitPreset? {
        get { _currentPreset }
        set {
            guard let newValue else {
                _currentPreset = nil
                return
            }
            if newValue.number >= 0 {
                guard let toml = PresetLibrary.toml(forPresetNumber: newValue.number) else {
                    PianoAudioUnit.log.error(
                        "factory preset \(newValue.number) is not in the bundle")
                    return
                }
                _currentPreset = newValue
                replacePreset(with: toml)
            } else {
                // A user preset: the base class hands its saved `fullState`
                // back through `presetState`.
                if let saved = try? presetState(for: newValue) {
                    _currentPreset = newValue
                    fullState = saved
                }
            }
        }
    }

    // MARK: - state

    public override var fullState: [String: Any]? {
        get {
            var result = super.fullState ?? [:]
            result[StateKey.schemaVersion] = PianoAudioUnit.stateSchemaVersion
            result[StateKey.abiVersion] = Int(PE_ABI_VERSION)
            result[StateKey.presetTOML] = currentPresetText()
            if let number = _currentPreset?.number, number >= 0 {
                result[StateKey.factoryPresetNumber] = number
            }
            return result
        }
        set {
            guard let newValue else { return }
            if let version = newValue[StateKey.schemaVersion] as? Int,
                version > PianoAudioUnit.stateSchemaVersion
            {
                PianoAudioUnit.log.error(
                    """
                    refusing a state written by a newer version of this plugin \
                    (schema \(version) against \(PianoAudioUnit.stateSchemaVersion))
                    """)
                return
            }
            // The parameters first, then the instrument: setting the preset
            // rebuilds the engine, and the pedal positions are pushed into
            // whichever engine the render block picks up next.
            super.fullState = newValue
            if let toml = newValue[StateKey.presetTOML] as? String, !toml.isEmpty {
                replacePreset(with: toml)
            }
            if let number = newValue[StateKey.factoryPresetNumber] as? Int,
                _factoryPresets.indices.contains(number)
            {
                _currentPreset = _factoryPresets[number]
            } else {
                _currentPreset = nil
            }
        }
    }

    public override var fullStateForDocument: [String: Any]? {
        get { fullState }
        set { fullState = newValue }
    }

    /// The preset the instrument is actually playing, as TOML: from the engine
    /// when there is one (`pe_save_state` is the authority), from the text we
    /// were last given when there is not.
    private func currentPresetText() -> String {
        guard let engine = liveEngine else { return presetText }
        let needed = pe_save_state(engine, nil, 0)
        guard needed > 0 else { return presetText }
        var buffer = [UInt8](repeating: 0, count: needed)
        let written = buffer.withUnsafeMutableBufferPointer { pointer in
            pe_save_state(engine, pointer.baseAddress, needed)
        }
        guard written > 0, written <= needed else { return presetText }
        return String(decoding: buffer[0..<written], as: UTF8.self)
    }

    /// Builds a new engine on this thread and publishes it. The old one is
    /// retired, never freed here: see the file comment.
    private func replacePreset(with toml: String) {
        presetText = toml
        guard renderResourcesAllocated, let previous = liveEngine else { return }
        guard let replacement = makeEngine(rate: pe_host_sample_rate(previous),
                                           maxFrames: UInt32(pe_max_frames(previous)))
        else { return }
        liveEngine = replacement
        retiredEngines.append(previous)
        pe_state_set_engine(state, UnsafeMutableRawPointer(replacement))
    }

    private func makeEngine(rate: Double, maxFrames: UInt32) -> OpaquePointer? {
        guard let engine = pe_create(rate, maxFrames) else {
            PianoAudioUnit.log.error("pe_create refused \(rate) Hz / \(maxFrames) frames")
            return nil
        }
        if !presetText.isEmpty {
            var bytes = Array(presetText.utf8)
            // `pe_status` cannot be named from Swift either (see
            // `MIDITranslation.Kind`), so the comparison is on the raw value.
            let status = bytes.withUnsafeMutableBufferPointer { buffer -> Int32 in
                buffer.baseAddress!.withMemoryRebound(to: CChar.self, capacity: buffer.count) {
                    pe_load_preset_toml(engine, $0, buffer.count)
                }
            }
            if status != PE_OK.rawValue {
                let message = String(cString: pe_last_error(engine))
                PianoAudioUnit.log.error("the preset was refused: \(message, privacy: .public)")
                // The engine is still the built-in instrument and still
                // playable; that is exactly what `pe_load_preset_toml`
                // guarantees on failure.
            }
        }
        return engine
    }

    // MARK: - render resources

    /// Two channels or nothing. `channelCapabilities` already says `[0, 2]`, but
    /// a bus with `maximumChannelCount = 2` will *accept* one channel unless the
    /// AU says otherwise, and `auval` notices: "Can Initialize Unit to
    /// un-supported num channels: InputChan:0, OutputChan:1". The instrument's
    /// output is a virtual-mic pair in mid/side form (`PHYSICS.md` §8) — the
    /// mono fold-down is a *property* of the pair, not a second output format —
    /// so a mono bus is refused here rather than silently half-rendered.
    public override func shouldChange(to format: AVAudioFormat, for bus: AUAudioUnitBus) -> Bool {
        guard format.channelCount == 2 else { return false }
        return super.shouldChange(to: format, for: bus)
    }

    public override func allocateRenderResources() throws {
        guard outputBus.format.channelCount == 2 else {
            throw NSError(
                domain: NSOSStatusErrorDomain, code: Int(kAudioUnitErr_FormatNotSupported),
                userInfo: [
                    NSLocalizedDescriptionKey: "the piano's output is a stereo microphone pair"
                ])
        }
        try super.allocateRenderResources()

        let frames = Int(maximumFramesToRender)
        let left = UnsafeMutablePointer<Float>.allocate(capacity: frames)
        let right = UnsafeMutablePointer<Float>.allocate(capacity: frames)
        left.initialize(repeating: 0, count: frames)
        right.initialize(repeating: 0, count: frames)
        scratchLeft?.deallocate()
        scratchRight?.deallocate()
        scratchLeft = left
        scratchRight = right
        pe_state_set_scratch(state, left, right, UInt32(frames))

        guard
            let engine = makeEngine(
                rate: outputBus.format.sampleRate, maxFrames: maximumFramesToRender)
        else {
            super.deallocateRenderResources()
            throw NSError(
                domain: NSOSStatusErrorDomain, code: Int(kAudioUnitErr_FailedInitialization),
                userInfo: [NSLocalizedDescriptionKey: "the piano engine could not be built"])
        }
        if let previous = liveEngine { retiredEngines.append(previous) }
        liveEngine = engine
        pe_state_set_frames(state, 0)
        // NaN compares unequal to everything, so the first render block pushes
        // every pedal position into the fresh engine.
        for index in 0..<PE_PARAM_COUNT {
            pe_state_set_applied(state, index, Float.nan)
        }
        pe_state_publish_meter(state, 0, 0, 0)
        pe_state_set_engine(state, UnsafeMutableRawPointer(engine))
    }

    public override func deallocateRenderResources() {
        pe_state_set_engine(state, nil)
        super.deallocateRenderResources()
        if let engine = liveEngine {
            pe_destroy(engine)
            liveEngine = nil
        }
        for engine in retiredEngines { pe_destroy(engine) }
        retiredEngines.removeAll()
        pe_state_set_scratch(state, nil, nil, 0)
        scratchLeft?.deallocate()
        scratchRight?.deallocate()
        scratchLeft = nil
        scratchRight = nil
    }

    public override func reset() {
        // `pe_reset` is a main-thread call and must not race a render, which is
        // what the host guarantees when it calls `reset`.
        if let engine = liveEngine {
            pe_reset(engine)
        }
        pe_state_set_frames(state, 0)
        pe_state_publish_meter(state, 0, 0, 0)
    }

    // MARK: - the in-process note path

    /// Queues an event for the audio thread through the engine's SPSC queue.
    ///
    /// **This is not the host's path** — a host hands an AUv3 its MIDI in the
    /// render block's event list, which `internalRenderBlock` reads. It is the
    /// path the plugin's *own* view uses for its on-screen keyboard, which runs
    /// in this process and would otherwise have no way in. `pe_post_event` is
    /// single-producer, so exactly one thread may call this: the main one.
    @discardableResult
    public func post(_ event: pe_event_t) -> Bool {
        guard let liveEngine else { return false }
        return pe_post_event(liveEngine, event)
    }

    // MARK: - metering, for a UI that wants one

    public var activeVoices: Int { Int(pe_state_voices(state)) }

    /// The pedal positions the render block last pushed into the engine, in
    /// `PE_PARAM_*` order.
    ///
    /// For tests: it is the audio thread's own copy, and reading it while a
    /// render is in flight tells you about a block that may already be over.
    /// It is here because the parameter *tree*'s value is not evidence — a
    /// change that arrives in the render event list never passes through the
    /// tree — and "the event reached the engine" is the thing worth asserting.
    public var appliedParameters: [Float] {
        (0..<PE_PARAM_COUNT).map { pe_state_applied(state, $0) }
    }
    public var peakLevels: (left: Float, right: Float) {
        (pe_state_peak_left(state), pe_state_peak_right(state))
    }
    /// True when the host runs at the engine's own rate and the output is
    /// bit-for-bit what the offline renderer writes.
    public var resamplerBypassed: Bool {
        guard let liveEngine else { return false }
        return pe_is_bypassed(liveEngine)
    }
    public var engineHostSampleRate: Double {
        guard let liveEngine else { return outputBus.format.sampleRate }
        return pe_host_sample_rate(liveEngine)
    }

    // MARK: - the render block

    public override var internalRenderBlock: AUInternalRenderBlock {
        let state = self.state
        // Read once, on the main thread: the render block must not touch a
        // lazy `static let`, and the offset of a C struct field is a constant.
        let eventListOffset = MemoryLayout<AUMIDIEventList>.offset(of: \.eventList)!
        let packetWordsOffset = MemoryLayout<MIDIEventPacket>.offset(of: \.words)!

        return {
            actionFlags, timestamp, frameCount, outputBusNumber, outputData, events, pullInput in
            _ = actionFlags
            _ = outputBusNumber
            _ = pullInput
            return PianoAudioUnit.render(
                state: state,
                timestamp: timestamp,
                frameCount: frameCount,
                outputData: outputData,
                events: events,
                eventListOffset: eventListOffset,
                packetWordsOffset: packetWordsOffset)
        }
    }

    /// The whole audio thread. Everything it calls is `@inline(__always)` or a
    /// C function; nothing it touches allocates.
    private static func render(
        state: UnsafeMutablePointer<pe_render_state>,
        timestamp: UnsafePointer<AudioTimeStamp>,
        frameCount: AUAudioFrameCount,
        outputData: UnsafeMutablePointer<AudioBufferList>,
        events: UnsafePointer<AURenderEvent>?,
        eventListOffset: Int,
        packetWordsOffset: Int
    ) -> AUAudioUnitStatus {
        let buffers = UnsafeMutableAudioBufferListPointer(outputData)
        guard buffers.count >= 2 else { return kAudioUnitErr_FormatNotSupported }

        let byteCount = UInt32(frameCount) * UInt32(MemoryLayout<Float>.size)
        var left: UnsafeMutablePointer<Float>
        var right: UnsafeMutablePointer<Float>
        if let l = buffers[0].mData, let r = buffers[1].mData {
            left = l.assumingMemoryBound(to: Float.self)
            right = r.assumingMemoryBound(to: Float.self)
        } else {
            // A host may hand us a null buffer list and expect us to point it
            // at our own memory. That is what the scratch is for.
            guard let l = pe_state_scratch_left(state), let r = pe_state_scratch_right(state),
                frameCount <= pe_state_scratch_frames(state)
            else {
                return kAudioUnitErr_TooManyFramesToProcess
            }
            left = l
            right = r
            buffers[0].mData = UnsafeMutableRawPointer(l)
            buffers[1].mData = UnsafeMutableRawPointer(r)
        }
        buffers[0].mNumberChannels = 1
        buffers[1].mNumberChannels = 1
        buffers[0].mDataByteSize = byteCount
        buffers[1].mDataByteSize = byteCount

        guard let rawEngine = pe_state_engine(state) else {
            left.update(repeating: 0, count: Int(frameCount))
            right.update(repeating: 0, count: Int(frameCount))
            return noErr
        }
        let engine = OpaquePointer(rawEngine)

        // Pedal positions the host moved between render calls.
        applyParameterChanges(state: state, engine: engine)

        let base = pe_state_frames(state)
        let now = AUEventSampleTime(timestamp.pointee.mSampleTime)
        var rendered: AUAudioFrameCount = 0
        var cursor: UnsafePointer<AURenderEvent>? = events

        while let event = cursor {
            let head = event.pointee.head
            let offset = eventOffset(head.eventSampleTime, now: now, frameCount: frameCount)

            // The 128-frame block this event's sample belongs to. Render up to
            // it — but never backwards: an event whose block has already gone
            // by lands on the next one, which is the same 2.7 ms grain.
            let absolute = base &+ UInt64(offset)
            let blockStart = absolute - (absolute % UInt64(PE_BLOCK))
            if blockStart > base &+ UInt64(rendered) {
                let target = AUAudioFrameCount(min(blockStart &- base, UInt64(frameCount)))
                if target > rendered {
                    pe_render(engine, left + Int(rendered), right + Int(rendered), target - rendered)
                    rendered = target
                    pe_state_set_frames(state, base &+ UInt64(rendered))
                }
            }

            apply(
                event: event, head: head, state: state, engine: engine,
                eventListOffset: eventListOffset, packetWordsOffset: packetWordsOffset)
            cursor = UnsafePointer(head.next)
        }

        if rendered < frameCount {
            pe_render(
                engine, left + Int(rendered), right + Int(rendered), frameCount - rendered)
            rendered = frameCount
            pe_state_set_frames(state, base &+ UInt64(rendered))
        }

        // Output trim and the meter, in one pass. The trim is a plain gain on
        // the way out and is deliberately not an engine parameter: the
        // instrument's own level is calibrated (`OUTPUT_GAIN`, the limiter
        // budget) and this is the host's fader, not the piano's.
        let trimDb = pe_state_param(state, PE_PARAM_OUTPUT_TRIM)
        let gain: Float = trimDb == 0 ? 1 : powf(10, trimDb / 20)
        var peakLeft: Float = 0
        var peakRight: Float = 0
        for index in 0..<Int(frameCount) {
            let l = left[index] * gain
            let r = right[index] * gain
            left[index] = l
            right[index] = r
            let al = abs(l)
            let ar = abs(r)
            if al > peakLeft { peakLeft = al }
            if ar > peakRight { peakRight = ar }
        }
        pe_state_publish_meter(state, pe_active_voices(engine), peakLeft, peakRight)
        return noErr
    }

    @inline(__always)
    private static func eventOffset(
        _ eventSampleTime: AUEventSampleTime, now: AUEventSampleTime,
        frameCount: AUAudioFrameCount
    ) -> AUAudioFrameCount {
        // `AUEventSampleTimeImmediate` is hugely negative, so "before now"
        // covers it without a special case.
        guard eventSampleTime > now else { return 0 }
        let delta = eventSampleTime - now
        return delta >= AUEventSampleTime(frameCount) ? frameCount : AUAudioFrameCount(delta)
    }

    @inline(__always)
    private static func applyParameterChanges(
        state: UnsafeMutablePointer<pe_render_state>, engine: OpaquePointer
    ) {
        for index in 0..<PE_PARAM_COUNT {
            let value = pe_state_param(state, index)
            guard value != pe_state_applied(state, index) else { continue }
            pe_state_set_applied(state, index, value)
            if let event = pedalEvent(index: index, value: value) {
                pe_event(engine, event)
            }
        }
    }

    @inline(__always)
    private static func pedalEvent(index: Int32, value: Float) -> pe_event_t? {
        switch index {
        case PE_PARAM_SUSTAIN:
            return pe_event_t(
                kind: MIDITranslation.Kind.sustain, key: 0, vel: 0,
                value: min(max(value, 0), 1))
        case PE_PARAM_SOSTENUTO:
            return pe_event_t(
                kind: MIDITranslation.Kind.sostenuto, key: 0, vel: 0, value: value >= 0.5 ? 1 : 0)
        case PE_PARAM_UNA_CORDA:
            return pe_event_t(
                kind: MIDITranslation.Kind.unaCorda, key: 0, vel: 0, value: value >= 0.5 ? 1 : 0)
        default:
            // The trim is not an engine event.
            return nil
        }
    }

    @inline(__always)
    private static func apply(
        event: UnsafePointer<AURenderEvent>,
        head: AURenderEventHeader,
        state: UnsafeMutablePointer<pe_render_state>,
        engine: OpaquePointer,
        eventListOffset: Int,
        packetWordsOffset: Int
    ) {
        switch head.eventType {
        case .parameter, .parameterRamp:
            let parameter = event.pointee.parameter
            // The address is compared *before* it is narrowed. A host reaching
            // us through the AUv2 bridge schedules parameter events under the
            // hashed `AudioUnitParameterID` rather than under the tree's
            // address, and `Int32(someUInt64)` traps on those — which is
            // precisely what `auval`'s render tests found, and what the
            // `an_address_no_parameter_has_is_ignored_rather_than_fatal` case
            // in the harness now pins.
            guard parameter.parameterAddress < AUParameterAddress(PE_PARAM_COUNT) else { return }
            let index = Int32(parameter.parameterAddress)
            pe_state_set_param(state, index, parameter.value)
            pe_state_set_applied(state, index, parameter.value)
            if let pedal = pedalEvent(index: index, value: parameter.value) {
                pe_event(engine, pedal)
            }

        case .MIDI:
            let midi = event.pointee.MIDI
            guard midi.length >= 2 else { return }
            if let translated = MIDITranslation.fromMIDI1(
                statusByte: midi.data.0, d1: midi.data.1,
                d2: midi.length >= 3 ? midi.data.2 : 0)
            {
                pe_event(engine, translated)
            }

        case .midiEventList:
            let listPointer = UnsafeRawPointer(event)
                .advanced(by: eventListOffset)
                .assumingMemoryBound(to: MIDIEventList.self)
            for packet in listPointer.unsafeSequence() {
                let words = UnsafeRawPointer(packet)
                    .advanced(by: packetWordsOffset)
                    .assumingMemoryBound(to: UInt32.self)
                MIDITranslation.parseUMP(words, count: Int(packet.pointee.wordCount)) {
                    pe_event(engine, $0)
                }
            }

        default:
            // Sysex and anything a later SDK adds: ignored, exactly as the
            // engine's own reader ignores them.
            break
        }
    }
}
