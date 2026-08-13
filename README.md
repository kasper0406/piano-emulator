# piano-emulator

A physically modeled (simulated) grand piano — no samples. Every note is synthesized in real time from a model of the instrument: stiff strings, felt hammers, dampers, pedals, and a soundboard. The design goal is sound quality first, with all model parameters exposed so they can later be fitted automatically to recordings of real pianos (see `TUNING.md`).

## What is modeled

- **Strings** — modal synthesis with stiffness inharmonicity (`f_k = k·f0·√(1+Bk²)`), frequency-dependent decay, and two polarizations per string for the characteristic fast-attack / slow-aftersound double decay.
- **Unison groups** — 1–3 strings per key, unevenly detuned and unevenly struck, coupled through the bridge: real beating and uneven decay rather than envelope tricks.
- **Hammer** — nonlinear felt (Hunt–Crossley hysteresis) integrated against an explicit agraffe reflection; contact time and brightness vary with key and velocity the way measured grands do.
- **Pedals** — sustain as *continuous* damper lift (half-pedaling works), sostenuto with correct capture semantics, and una corda (softer felt, one string of the group unstruck). Keys from G6 up have no dampers, as on a real grand.
- **Sympathetic resonance** — undamped strings pick up energy from everything else that rings; strike-and-release with the pedal down leaves the authentic halo.
- **Soundboard** — body resonances plus a short diffuse-field reverberator, per-key stereo placement, and a master chain with a safety limiter.

Full 88-key polyphony with the sustain pedal down runs at roughly a third of one performance core on an M4 Pro. The engine's offline renderer and the live audio path are the same code, so everything measurable in a rendered WAV is what you hear live.

## System requirements

- **macOS on Apple Silicon** (M-series) is the supported target; developed and tuned on an M4 Pro. The audio thread uses aarch64-specific code (FPCR flush-to-zero) and the modal loops are laid out for NEON; other platforms are untested.
- An output device that can run at **48 kHz** (the engine refuses to resample).
- **Rust 1.84+** (`rustup` default toolchain is fine).

## Build & run

```sh
cargo run -p piano-emulator
```

The repository is a cargo workspace: `engine/` is the instrument, `tuner/` is
the offline analysis and parameter-estimation crate `TUNING.md` describes, and
`presets/` holds the instrument's parameters as data. `cargo test --release` at
the root runs both crates, including the self-calibration gate, which puts the
tuner's whole estimation pipeline over notes the engine rendered from a known
preset and checks that the parameters come back.

The default (dev) profile is built with `opt-level = 3` — the DSP is unusable unoptimized — so plain `cargo run`/`cargo test` are real-time capable while keeping fast incremental builds. `--release` additionally enables thin LTO and disables debug assertions; it is the profile performance is measured on, and the performance acceptance test only runs there. The binary opens the default output device and drops you into a REPL:

```
n C4 90            strike C4 at velocity 90 (names like F#3, Bb2; A4 = 440 Hz)
off C4             release the key
chord C3 E3 G3 95  strike together
ped sus 0.5        sustain pedal (0..1, continuous) — also: sos 0|1, uc 0|1
demo               ~15 s musical demo
render out.wav     render offline to WAV — add `demo`, or a `.mid` file to replay
panic              everything off
quit
```

The same work can be done without an audio device at all:

```sh
cargo run -p piano-emulator -- render out.wav song.mid --preset presets/default.toml
cargo run -p piano-emulator -- preset my-piano.toml   # write out a preset to edit
```

Tests run with plain `cargo test`; add `--release` to include the performance-budget test:

```sh
cargo test --release
```

## Presets

Every number that voices the instrument — the per-note tables (tuning, inharmonicity, decay, unison detuning, strike position, impedance, damper and hammer parameters) and the global constants (polarization balance, couplings, hammer felt, soundboard) — lives in a preset file. `--preset <file.toml>` voices both the live engine and offline renders from it; without it the built-in default is used, which is exactly `presets/default.toml`.

The `f0` table is the tuning, so a stretch-tuned (Railsback) instrument is a preset like any other. The default table is equal temperament.

`presets/salamander-c5.toml` is the first preset *measured* rather than tuned by hand: the tuning, inharmonicity, damping and unison detuning of the Yamaha C5 recorded as the [Salamander Grand Piano](https://freepats.zenvoid.org/Piano/acoustic-grand-piano.html) (Alexander Holm, CC-BY 3.0), estimated from 480 recordings by the tuner. Everything those recordings cannot identify — strike position, felt, soundboard, coupling, dampers — is inherited from the default. To reproduce it:

```sh
data/fetch_salamander.sh                       # 707 MiB, checksummed, into the gitignored data/
cargo run --release -p piano-tuner -- survey \
    data/salamander/SalamanderGrandPiano-V3+20200602.sfz \
    --preset presets/default.toml --cache data/cache/salamander \
    --name salamander-c5 --out presets/salamander-c5.toml \
    --credit 'Salamander Grand Piano V3 (Yamaha C5) by Alexander Holm, CC-BY 3.0'
cargo run --release -p piano-tuner --example salamander_ab   # A/B renders into renders/
```

## MIDI replay

`render` accepts a standard MIDI file: note on/off on every channel, CC 64 as a *continuous* sustain pedal (half-pedalling survives), CC 66 sostenuto, CC 67 una corda, and the file's tempo map. Events are scheduled against the same event path the keyboard uses, so a replay is a performance the instrument plays rather than a special mode.

## Documentation

- `SPEC.md` — the model specification and acceptance tests.
- `DECISIONS.md` — the running log of every design decision and deviation.
- `TUNING.md` — the plan for estimating parameters automatically from recordings of real pianos (in progress; stage 1, its self-calibration gate and the first measured preset are built, in `tuner/`).
- `presets/default.toml` — the hand-tuned v1 instrument, written out in full.
- `presets/salamander-c5.toml` — the same instrument with everything stage 1 could measure off a real Yamaha C5 written into it.
