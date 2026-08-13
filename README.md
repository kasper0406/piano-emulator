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
cargo build --release
cargo run --release -p piano-emulator
```

Use `--release`: the DSP is far too slow in debug builds. The binary opens the default output device and drops you into a REPL:

```
n C4 90            strike C4 at velocity 90 (names like F#3, Bb2; A4 = 440 Hz)
off C4             release the key
chord C3 E3 G3 95  strike together
ped sus 0.5        sustain pedal (0..1, continuous) — also: sos 0|1, uc 0|1
demo               ~15 s musical demo
render out.wav     render offline to WAV
panic              everything off
quit
```

Tests (including the spectral acceptance suite and the performance budget) run with:

```sh
cargo test --release
```

## Documentation

- `SPEC.md` — the model specification and acceptance tests.
- `DECISIONS.md` — the running log of every design decision and deviation.
- `TUNING.md` — the plan for estimating parameters automatically from recordings of real pianos (in progress).
