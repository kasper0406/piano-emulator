# DISTRIBUTION.md — from engine to plugin to product

The engine is a self-contained Rust library with the properties a plugin host wants: `Engine::process(&mut l, &mut r)` accepts any request length and is bit-exact regardless of how the stream is cut up (decision 47), the audio path allocates nothing and locks nothing, events arrive through a pre-allocated SPSC queue, and every voicing number is data in a `Preset` (decision 50). Two things stand between that and a plugin: the sample rate is a compile-time constant (decision 17), and there is no host adapter. This document is the plan for both, and for selling the result.

Scope and target: macOS on Apple silicon, Logic Pro first. That is not a limitation for long — macOS 27 is Apple-silicon-only and the last release with full Rosetta 2 ([TechRadar, WWDC 2026](https://www.techradar.com/computing/mac-os/macos-27-golden-gate-announced-at-wwdc-2026-heres-everything-you-need-to-know)), so an arm64-only product is where the market is going anyway.

## M0 — the sample rate, which blocks everything else

A plugin must run at the host's rate. AUv3, CLAP and VST3 all *let* you refuse one (`allocateRenderResources()` throws, `clap_plugin.activate()` returns false, `setupProcessing` returns `kResultFalse`), but refusing is commercially not an option: hosts respond by dropping the plugin or showing an error, and `auvaltool` fails you. There are two real answers.

**(a) Runtime-parameterized rate.** `SAMPLE_RATE` appears in ~50 places across six DSP modules. Almost all of them are coefficient computation at construction time or in the damper update, so the hot loops do not change; but the work is not mechanical:

- `MAX_PARTIAL_RATIO * SAMPLE_RATE` decides how many partials a string gets. At 96 kHz every note above ~C6 gains partials above 21.6 kHz — inaudible, costly, and a change in bridge force and therefore in level. The cap would have to become an absolute frequency (~20 kHz) at every rate, which is a voicing change at 48 kHz too unless it is set exactly where 0.45·48000 already puts it.
- `soundboard::FDN_DELAYS` is eight mutually prime *sample* counts; they would have to be re-primed per rate, and `BOARD_LEVEL` (decision 43) re-measured, and the FDN T60 test re-derived.
- The hammer integrates its contact ODE at `1/(SAMPLE_RATE * 8)` against an agraffe reflection quantised to that grid, with a passivity guarantee asserted over the whole key × velocity plane (decision 29). `MAX_PULSE_SAMPLES = 960` is 20 ms at 48 kHz and 10 ms at 96 kHz.
- Every calibrated constant — `OUTPUT_GAIN`, `CULL_AMPLITUDE`, `bridge_gain_for`, the master shelf — was measured at 48 kHz, and the whole acceptance suite asserts numbers measured there. The tuner's self-calibration gate (decision 83) renders through the engine and would need the same treatment.

Cost: 2–3 weeks, and it permanently doubles the surface every future measurement has to cover. Benefit: nothing audible. A piano that voices differently at 44.1 and 96 kHz is a bug, not a feature.

**(b) Boundary resampling — recommended.** Keep the engine at 48 kHz forever. Put a high-quality asynchronous SRC between it and the host, and **bypass it entirely when the host runs at 48 kHz**, which is the common case and stays bit-identical to everything measured so far.

Use `rubato`'s `SincFixedOut` — fixed *output* block, variable input pull. That is exactly the shape our API wants: the resampler asks for N engine frames, `Engine::process` produces N of anything, remainder FIFO included. `rubato` is real-time-safe through `process_into_buffer` ([docs.rs/rubato](https://docs.rs/rubato)), and we have already measured its alignment behaviour to 1.8 µs and pinned it in a test (decision 64). Cost is two channels of a 128–256-tap polyphase sinc, well under 1 % of a core against the engine's 31–40 %.

Two honest consequences. First, because the source is a generator rather than a stream, the resampler's look-ahead is **not** latency — it just pulls more frames — so we report zero added latency to the host. It does mean the engine runs roughly `sinc_len/2` frames (~1.3 ms) ahead of the host clock, which shows up as event timing, not as delay. Second, at 44.1 kHz the SRC must filter below 22.05 kHz; cap partials at ~19.5 kHz so nothing folds. Both are measurable and both belong in the M0 tests: a null test at 48 kHz (bit-exact bypass), a swept-sine alias floor at 44.1 and 96 kHz, and a transient-position test like the one decision 64 already has.

This does not overturn decision 17. That decision refused to *silently detune* the instrument by reinterpreting its samples at another rate. A proper asynchronous SRC is the opposite of that, and it should be logged as a new decision saying so.

## Plugin formats — what to build, in what order

| Format | Logic | Third-party DAWs | App Store | Verdict |
|---|---|---|---|---|
| **AUv3** | yes | Live 11.2+, Reaper, not Pro Tools | **only legal format** | ship first |
| AUv2 | yes | everything on desktop except Pro Tools | forbidden (2.4.5(ii)) | later, nearly free |
| CLAP | no | Bitwig, Reaper 7, FL Studio, Studio One 7 | no | second |
| VST3 | no | everything | no | via CLAP wrapper |

**AUv3 is required and sufficient to start.** It is an app extension (`.appex`) inside a container `.app` — there is no standalone AUv3 ([Apple: Audio Unit v3 plug-ins](https://developer.apple.com/documentation/audiotoolbox/audio-unit-v3-plug-ins)). The extension declares itself through `NSExtension` → `NSExtensionAttributes` → `AudioComponents` with `type` = `aumu` (music device), a four-character `subtype` and `manufacturer` (at least one uppercase), and `sandboxSafe = true`, which we can honestly claim: an instrument that reads its presets from its own bundle needs no filesystem, network or IOKit access ([App Extension Programming Guide](https://developer.apple.com/library/archive/documentation/General/Conceptual/ExtensibilityPG/AudioUnit.html), [TN2247](https://developer.apple.com/library/archive/technotes/tn2247/_index.html)). App extensions must be sandboxed regardless of how they are distributed — PluginKit rejects a non-sandboxed appex outright. Hardened Runtime and notarization apply to the direct-download build.

Registration is via PluginKit, not a plugin folder, and **the container app must be launched once** from `/Applications` before hosts see the AU ([JUCE forum](https://forum.juce.com/t/are-auv3s-on-macos-supported/62101), [eclecticlight.co on PlugInKit](https://eclecticlight.co/2025/04/16/how-pluginkit-enables-app-extensions/)). Document that as a first-run step; it is also where the purchase/unlock flow lives.

**AUv2 is not deprecated** — Apple's own docs say only that "AUv2 is in maintenance mode" and new development should use AUv3 ([Hosting Audio Unit Extensions Using the AUv2 API](https://developer.apple.com/documentation/audiotoolbox/hosting-audio-unit-extensions-using-the-auv2-api)) — and it is still what most desktop vendors ship, because a `.component` drops into a folder with no container app and no launch-to-register dance. We do not need it on day one; we get it later for nearly nothing (below).

**VST3 licensing changed and the old objection is gone.** Steinberg relicensed the VST 3 SDK (3.8.0) to **MIT** on 29 October 2025 — no agreement to sign, no registration, branding guidelines now optional ([Steinberg licensing FAQ](https://steinbergmedia.github.io/vst3_dev_portal/pages/FAQ/Licensing.html), [KVR](https://www.kvraudio.com/news/steinberg-moves-vst-3-sdk-to-mit-open-source-license-asio-now-gplv3-65179)). Use the MIT/Apache `vst3` crate (coupler-rs), not the GPLv3 `vst3-sys`.

**Rust frameworks: check the maintenance status before committing.** `nih-plug` is explicitly in maintenance mode and points at a community fork (`nice-plug`, Codeberg, active); both export CLAP + VST3 + standalone and **neither supports AU in any form**. The interesting piece is `free-audio/clap-wrapper`, which projects one CLAP into VST3, AUv2, AUv3, AAX and a standalone; VST3 and AUv2 have been feature-complete since 2024, **AUv3 landed only in v0.15.1 (July 2026)** and needs the CMake Xcode generator. `blepfx/clap-wrapper-rs` vendors the wrapper and builds it with `cc` — no CMake — but exports VST3 and AUv2 only.

**Recommendation.** Write the AUv3 by hand in Swift over our own C ABI, and do not route the primary product through a wrapper. Reasons: AUv3 is the format that matters (Logic + App Store), it is the one place we want direct control of `AUParameterTree`, factory presets, `fullState` and the MIDI 2.0 event path; clap-wrapper's AUv3 is a month old; and a wrapped-CLAP AUv3 is a wrapper inside an app extension inside a container app. Then write a CLAP (thin — the same C ABI underneath) and take **VST3 and AUv2 for free** from `clap-wrapper-rs`. That covers every DAW except Pro Tools, which needs AAX and an Avid agreement and is not worth it.

## Architecture

```
engine/                  unchanged, 48 kHz, no knowledge of hosts
ffi/          (Rust)     cdylib+staticlib, C ABI, cbindgen header, SRC lives here
  ├── PianoAU.appex      Swift AUAudioUnit + SwiftUI view      ─┐ one container
  ├── Piano.app          SwiftUI standalone, hosts the .appex   ─┘ app, one target set
  └── clap/     (Rust)   CLAP entry point → VST3/AUv2 via clap-wrapper-rs
```

The `ffi` crate is the only new Rust code the plugin needs, and it owns three things: the C ABI, the boundary resampler, and the host-rate bookkeeping. Sketch:

```c
pe_engine *pe_create(double host_sample_rate, uint32_t max_frames);   // main thread
void       pe_destroy(pe_engine *);                                   // main thread
void       pe_reset(pe_engine *);                                     // main thread
int32_t    pe_load_preset_toml(pe_engine *, const char *utf8, size_t); // main thread
void       pe_render(pe_engine *, float *l, float *r, uint32_t n);     // audio thread
void       pe_event(pe_engine *, pe_event_t);                          // audio thread
bool       pe_post_event(pe_engine *, pe_event_t);                     // any thread (SPSC)
size_t     pe_save_state(pe_engine *, uint8_t *buf, size_t cap);       // main thread
```

The header states the thread contract per function; that is the whole point of writing it by hand rather than generating a flat surface. `cbindgen` produces the header from annotated types so the Swift side never drifts. Two Rust-specific hazards: a panic must not unwind across the FFI boundary (`catch_unwind` on the fallible entry points, `panic = "abort"` for the cdylib profile), and `Preset::validate` (decision 52) must run on the main thread, before the audio thread ever sees the coefficients.

**Events.** A host delivers MIDI to an AUv3 *on the audio thread*, in the render block's `AURenderEvent` list, so the plugin path calls `pe_event` directly and the SPSC queue is only used by the standalone app's CoreMIDI thread and its UI. Both already exist in the engine — `EventSender` and `handle_event` are separate entry points.

**Timing.** Host events carry sample offsets; our engine applies an event at the start of the next 128-frame block it renders, so onsets quantise to 2.7 ms (decision 55). Splitting the render call at event offsets does **not** fix this — the engine advances state only in whole blocks and spills the remainder — so sub-block scheduling means what decision 55 said it means: start a note's hammer pulse `n` samples into the block. That is a contained change (the pulse is already a precomputed buffer streamed into the banks) and it should land with the plugin, because a DAW's grid makes chord spread audible in a way the REPL never did.

**Parameters.** `AUParameterTree` with stable integer addresses: `sustain` (0…1 continuous — this is the instrument's headline feature, and hosts should be able to automate it as a curve), `sostenuto`, `unaCorda`, `outputTrim`, and later a voice-budget/quality control if CPU forces one. Parameter changes arrive in the render block as `AURenderEventParameter`; map them onto `Event::Pedal`. Ignore ramp events — the damper already ramps over 10 ms internally.

**Presets and state.** `presets/*.toml` ship inside the appex and are exposed as `AUAudioUnit.factoryPresets`. The default preset is 20 KB of TOML, 5.7 KB gzipped, so `fullState` should carry the **whole preset text**, not a reference to it: a project saved today must still open when the preset files have moved on, and 6 KB in a Logic project is free. Add a schema-version key and refuse to load a newer one. User-supplied presets are imported by the *container app* (which can show an `NSOpenPanel`) into an **App Group** container shared with the appex — that is the clean sandbox answer, and it avoids giving the extension `files.user-selected.read-write` at all.

**The standalone app should host its own AUv3** through `AVAudioEngine` rather than linking the engine a second way. One code path gets exercised twice, and any bug that only shows up under AU semantics shows up in our own app first. It is also what makes the container app a real product rather than the "empty wrapper" App Review punishes.

## MIDI

**Standalone:** CoreMIDI directly. `MIDIInputPortCreateWithProtocol` plus a virtual destination so other apps can play us. Core MIDI is still a C API with a variable-length `MIDIEventList` and a non-capturing `@convention(c)` callback, so either box the context by hand ([worked example](https://furnacecreek.org/blog/2024-04-06-modern-coremidi-event-handling-with-swift)) or take the dependency on [MIDIKit](https://github.com/orchetect/MIDIKit). We already parse note on/off, continuous CC64, CC66 and CC67 in `midi.rs`; the live path needs the same mapping and nothing more.

**In-plugin:** host-delivered, via `AUMIDIEventListBlock` / `scheduleMIDIEventListBlock` rather than the older byte-oriented block. One parser handles both protocols.

### MIDI 2.0 — verdict

The API is complete and has been for years: UMP shipped in macOS 11 / iOS 14, `AUAudioUnit.audioUnitMIDIProtocol` and `hostMIDIProtocol` in macOS 12, and macOS 15 added the `MIDIUMPEndpoint` / Function Block layer ([Apple: Incorporating MIDI 2 into your apps](https://developer.apple.com/documentation/coremidi/incorporating-midi-2-into-your-apps), [hostMIDIProtocol](https://developer.apple.com/documentation/audiotoolbox/auaudiounit/hostmidiprotocol)). Logic has a MIDI 2.0 switch in settings and records at MIDI 2.0 resolution ([Logic Pro MIDI preferences](https://support.apple.com/guide/logicpro/general-midi-preferences-lgcpb839d947/10.7/mac/11.0)). Because Core MIDI does 1.0↔2.0 translation and UMP endpoint discovery for us, **we do not implement MIDI-CI**; we set a protocol and parse what arrives.

What it would buy *this* instrument, honestly:

- **Velocity.** We are not velocity-layer-quantized — MIDI velocity lands in a continuous hammer speed of 0.2–6 m/s — so extra bits are genuinely usable, unlike in a sampler. But over a ~45 dB dynamic range, 128 steps is about 0.35 dB per step, under the loudness JND. The real thinness is at the soft end, where expressive playing crowds into velocities ~5–35. And no hardware sends 16 bits: Yamaha, the most candid vendor, states plainly that its MONTAGE M controllers sense **10 bits** and transmit them in the 16-bit field ([Yamaha MIDI 2.0](https://europe.yamaha.com/en/musical-instruments/keyboards/explore/midi-2-0/)). Realistic gain: 3 extra bits, concentrated exactly where we want them.
- **Pedal.** This is the better argument. There is no MSB/LSB partner for CC#64 (14-bit CC pairs only exist for CC 0–31), so half-pedaling in MIDI 1.0 is 7-bit, full stop — even Disklaviers, which sense damper position on a fine internal grid, emit 7-bit CC64 ([Yamaha on Disklavier pedals](https://hub.yamaha.com/pianos/p-acoustic/the-disklavier-piano-pedals/)). Our damper model is continuous, so it will expose that stepping on slow pedal moves in a way a gated sampler never does. **Mitigation available today, for free: slew-limit incoming CC64 over ~15 ms in the adapter.** Do that before waiting for anyone's 32-bit CC.
- **The Piano Profile is the actual news.** MIDI Association + AMEI released **M2-126-UM v1.0** (plus an implementation guide) just before NAMM 2026: a default velocity curve keyed to notated dynamics, **a formally defined continuous CC#64 with a specified half-pedal range**, registered controllers for temperament, hammer hardness, soundboard and sympathetic resonance, lid position, and mechanical noise — and optional per-note controllers for **hammer, damper and key position** ([MIDI Association announcement](https://midi.org/midi-association-and-amei-release-the-piano-profile-and-implementation-guide), [NAMM 2026 demo with Roland A-88MKII driving Ivory](https://midi.org/the-piano-profile-at-namm-2026)). Those registered controllers map almost one-to-one onto our `Preset` fields. This is the first spec that describes the instrument we actually built.

Against that: essentially nothing sends it. Confirmed UMP sources are the Roland A-88MKII (firmware 2.00+), Yamaha MONTAGE M, Studiologic SL mk2, Waldorf Quantum. **No digital or stage piano ships UMP output** — not Yamaha Clavinova, not Kawai, not Roland FP/RD, not Nord — which is the hardware our buyers own. Outside Logic and Cubase 15 (whose support users describe as not yet working), DAW support is absent: Ableton, Bitwig, Studio One, Reaper have none. And wrappers still down-convert to 1.0 on the way to plugins ([atsushieno, April 2026](https://atsushieno.github.io/2026/04/28/uapmd-aap-integration.html)).

**Plan:** parse UMP from day one, because it costs almost nothing — advertise `audioUnitMIDIProtocol = kMIDIProtocol_2_0`, implement `AUMIDIEventListBlock`, and widen the internal `Event` to carry `vel: u16` and `sustain: f32` (it already is `f32`). Use the `midi2` Rust crate (v0.11.1, July 2026, zero-copy, `no_std`-friendly) if we parse UMP in Rust; `midir` still has no UMP support. Then support **CC#88 high-resolution velocity prefix** in the MIDI 1.0 path — 14-bit velocity, adopted in 2010, and Pianoteq already receives it — which is a day's work and reaches more hardware than UMP does. Defer the Piano Profile's registered controllers and MIDI-CI Profile advertisement until the plugin ships; revisit when a hammer-action keyboard actually sends them.

## Distribution

**Mac App Store.** AUv3 is the only legal plugin format there — guideline 2.4.5(ii) requires a single self-contained bundle with no installers and no writes to shared locations, which rules out `.component`, `.vst3` and `.clap`. Sandbox both the app and the appex, share unlock state and user presets through an App Group and a keychain access group. Three review realities worth planning around:

- **4.2.3(i): "your app should work on its own without requiring installation of another app."** There is no plug-in carve-out, and container apps have been rejected under it repeatedly and released on appeal ([Loopy Pro forum thread with several developers' accounts](https://forum.loopypro.com/discussion/31760/audio-unit-container-app-usefulness)). Our standalone app must be a real instrument — plays from a MIDI keyboard, has an on-screen keyboard, presets, settings — and the metadata must never say "use this in your DAW."
- **3.1.1 / 2.4.5(vi): no license keys, no launch-time license screen, no self-updater** in the MAS build. If the product is paid, it is paid through the store or unlocked by IAP; purchase UI must live in the container app, because guideline 4.4 forbids IAP inside an extension.
- Commission is 30 %, or **15 % under the Small Business Program** if prior-year proceeds were ≤ $1M ([Apple](https://developer.apple.com/app-store/small-business-program/)). The $99/yr Developer Program covers both channels, and selling on the store *and* direct is allowed — the builds just differ.

**Direct distribution, as a complement not an alternative.** Developer ID signing, Hardened Runtime (`--options=runtime --timestamp`), notarization with `notarytool` (`altool` was decommissioned in November 2023), staple, ship a signed `.pkg` that installs AUv2/VST3/CLAP into `/Library/Audio/Plug-Ins/*` and the container app into `/Applications` ([Resolving common notarization issues](https://developer.apple.com/documentation/security/resolving-common-notarization-issues)). Sign inside-out, one component at a time; `codesign --deep` has been deprecated for signing since macOS 13 despite what most audio-plugin CI guides still say. Note that **entitlements are per-process: a plugin inherits the host's**, so `disable-library-validation` is the DAW's problem, not ours — nothing we put in the plugin's entitlements takes effect at load time. Test with `spctl -a -vvv -t install`, not just `pkgutil --check-signature`; there is a live macOS 26.3 issue where a correctly notarized pkg still fails `spctl` ([Apple forums](https://developer.apple.com/forums/thread/817887)).

Do the direct build first. It is the shorter path to real users, it exercises the same signing chain, and the App Store submission then only adds the store-specific build differences.

## Two risks worth pricing in now

**CPU.** The worst case is measured at 31–40 % of one M4 Pro performance core (decisions 25b, 37), with everything else on the machine idle. In a real session the piano shares the box with a mix, and Logic will run the AUv3 out of process at whatever buffer size the project uses. Before M6 we need a quality/voice-budget control that trades partial count or resonance-bus participation for headroom, chosen by measurement rather than by feel, and a number in the marketing copy that survives a busy project. This is the product's main competitive exposure against sample libraries, which are cheap on CPU and expensive on disk.

**AUv3 host friction.** Registration through PluginKit is not the plugin folder that DAW users know; hosts that scan folders rather than querying `AVAudioUnitComponentManager` will not see us at all, and there is an open report of a *sandboxed* host failing to instantiate an `.appex` AU with `-10863` while a non-sandboxed one succeeds ([Apple forums](https://developer.apple.com/forums/thread/774322)). Test in Logic, GarageBand, Live and Reaper before the first release; M8's AUv2 build is the escape hatch if a host we care about turns out to be one of the folder scanners.

## Milestones

| | Milestone | Effort | Done when |
|---|---|---|---|
| **M0** | Boundary resampler in `ffi`, 48 kHz bypass, partial cap at ~19.5 kHz, alias/null/transient tests | 1 wk | Bit-exact at 48 kHz; alias floor < −100 dB at 44.1/96 |
| **M1** | `ffi` crate: C ABI + cbindgen header + thread contract, `catch_unwind`, a C harness that renders a WAV | 1 wk | C harness output matches `cargo run -- render` sample-exactly |
| **M2** | AUv3 skeleton in Swift: `AUAudioUnit`, render block, `AURenderEvent` MIDI + UMP, sub-block note-on offsets in the engine | 2–3 wk | `auval -v aumu … -oop` green; plays in Logic |
| **M3** | Standalone SwiftUI app hosting the appex via `AVAudioEngine`, CoreMIDI input, on-screen keyboard | 2 wk | Playable from a hardware keyboard, no DAW |
| **M4** | `AUParameterTree`, factory presets, `fullState` save/restore, App Group preset import | 1–2 wk | Logic project reopens with pedal automation and preset intact |
| **M5** | Direct distribution: signing, notarization, `.pkg`, CI | 1 wk | Notarized, stapled, installs and loads on a clean Mac |
| **M6** | UI worth charging for (SwiftUI: keyboard, pedals, preset browser, meters) | 3–4 wk | — |
| **M7** | Mac App Store submission: sandbox audit, IAP if paid, review iterations | 2–3 wk | Approved |
| **M8** | CLAP entry point + VST3/AUv2 via `clap-wrapper-rs` | 1–2 wk | Loads in Reaper, Bitwig, Live |
| **M9** | MIDI 2.0 depth: Piano Profile registered controllers, CC#88 hi-res velocity, MPE | 1–2 wk | Roland A-88MKII drives it at 10-bit velocity |

Roughly four months of focused work to a notarized direct-download product, five to the App Store.

## What not to do yet

- **Do not runtime-parameterize the sample rate.** Option (a) above. Revisit only if a paying user demonstrates an audible difference the SRC caused.
- **Do not ship AUv2 first**, tempting as the drop-in `.component` is. It is App Store-illegal, and once M8 exists it costs one macro.
- **Do not build the AUv3 through `clap-wrapper`.** One month of field exposure on the format that carries the whole product, plus a CMake/Xcode-generator constraint on our build.
- **Do not implement MIDI-CI.** Core MIDI negotiates and translates; we would be reimplementing the OS. Profiles are the only reason to revisit, and that is M9 at the earliest.
- **Do not target iPad yet.** AUv3 is the same format there, but 88 always-on voices at 31–40 % of an M4 Pro performance core is a different proposition on an A-series, and it needs its own performance work and its own UI.
- **Do not build an iCloud/account/preset-sync layer.** Presets are 6 KB of gzipped TOML; a folder is the feature.
- **Do not chase Pro Tools.** AAX needs an Avid agreement and returns a niche of a niche for a piano.
- **Do not block the plugin on `TUNING.md` stage 2.** The default preset is a shippable instrument; measured presets are a product update, and one that gives the App Store listing something to say later.
