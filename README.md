# piano-emulator

A physically modeled (simulated) grand piano — no samples. Every note is synthesized in real time from a model of the instrument: stiff strings, felt hammers, dampers, pedals, and a soundboard. The design goal is sound quality first, with all model parameters exposed so they can later be fitted automatically to recordings of real pianos (see `TUNING.md`).

## What is modeled

- **Strings** — modal synthesis with stiffness inharmonicity (`f_k = k·f0·√(1+Bk²)`), frequency-dependent decay, and two polarizations per string for the characteristic fast-attack / slow-aftersound double decay.
- **Unison groups** — 1–3 strings per key, unevenly detuned and unevenly struck, coupled through the bridge: real beating and uneven decay rather than envelope tricks.
- **Hammer** — nonlinear felt (Hunt–Crossley hysteresis) integrated against an explicit agraffe reflection; contact time and brightness vary with key and velocity the way measured grands do.
- **Pedals** — sustain as *continuous* damper lift (half-pedaling works), sostenuto with correct capture semantics, and una corda (softer felt, one string of the group unstruck). Keys from G6 up have no dampers, as on a real grand. A damper that is touching but not seated limits the string nonlinearly, so a half-pedal buzzes rather than merely decaying faster.
- **The action** — the piano's own noises, at the levels they were measured at on a real instrument: the key-off thump on every release (scaled by how fast the key is let go), the damper lifting under a silently pressed key, and the pedal tray going down and coming up, scaled by how many dampers actually move. Panned per key, band-limited like the structure-borne path it travels, and deterministic — the same performance renders the same samples. The levels are ratios to a strike of the same key as a microphone hears it, so the strike they are quoted against is measured through the finished chain when the instrument is built: a preset that voices the piano more quietly takes its action down with it.
- **Touch** — a key pressed too gently to reach escapement lifts its damper and strikes nothing, which is how a pianist prepares sympathetic resonance without the pedal; release velocity sets how fast the damper falls.
- **Sympathetic resonance** — undamped strings pick up energy from everything else that rings; strike-and-release with the pedal down leaves the halo behind. A preset can give the bridge its measured admittance — a mean mobility curve with the board's discrete modes on it — and the halo is then coloured by the board the strings actually share instead of being spectrally flat, while a partial that sits on one of those modes loses energy into the board faster than the smooth fitted decay law says (`radiated_share`: T60 11.3 s against 14.6 s with a flat bridge). `render out.wav halo` is a phrase written to show it. Measured honestly the treble aftersound is still 21 dB short of the instrument it was fitted to — and that gap is on the board's late field rather than on this coupling, which has 0.1 dB of authority over it even at the largest value the stability contract will certify (`DECISIONS.md` 182–184).
- **Duplex and aliquot segments** — the lengths of string beyond the bridge and the agraffe, which have no dampers at all. A preset can give a key up to six of them, at measured frequencies rather than harmonic ratios; they are driven by that key's own bridge force and by the rest of the instrument through the bridge, and neither the key, nor the sustain pedal, nor sostenuto, nor una corda can stop them. Play a treble note staccato and the shimmer stays behind — at the top of the schema's range. At the level the *measured* table currently ships them they are 148 dB below the note and nobody can hear them; the mechanism is right and the drive is wrong — a segment is normalised to answer its own frequency, and a struck string's bridge force has almost nothing there (`DECISIONS.md` 162, 163, 170, `PHYSICS.md` §3).
- **Soundboard** — body resonances plus a short diffuse-field reverberator, per-key stereo placement, and a master chain with a safety limiter. The two polarizations of a key can be panned apart, so a note's stereo image *moves* as the fast plane dies; a preset may set that spread per key, because one number for the whole compass overshoots the treble by three decibels and undershoots the bass by five.

Full 88-key polyphony with the sustain pedal down runs at roughly a third of one performance core on an M4 Pro (41.6 % on the measured preset, whose duplex segments are never damped). The engine's offline renderer and the live audio path are the same code, so everything measurable in a rendered WAV is what you hear live.

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
hold C4            press the key silently: the damper lifts, nothing sounds
off C4 [rel]       release the key, optionally at a release velocity
chord C3 E3 G3 95  strike together
ped sus 0.5        sustain pedal (0..1, continuous) — also: sos 0|1, uc 0|1
demo               ~15 s musical demo
render out.wav     render offline to WAV — add `demo`, `halo` (the
                   sympathetic-resonance phrase), or a `.mid` file to replay
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

Every number that voices the instrument — the per-note tables (tuning, inharmonicity, decay, unison detuning, strike position, impedance, damper and hammer parameters), the global constants (polarization balance, couplings, the bridge admittance, hammer felt, soundboard) and the action's own noises — lives in a preset file. `--preset <file.toml>` voices both the live engine and offline renders from it; without it the built-in default is used, which is exactly `presets/default.toml`.

The `f0` table is the tuning, so a stretch-tuned (Railsback) instrument is a preset like any other. The default table is equal temperament.

`presets/salamander-c5.toml` is the first preset *measured* rather than tuned by hand: the tuning, inharmonicity (including the signed fourth-order term the wound bass needs), damping, unison detuning, per-string decay spread, stereo directivity and action noise of the Yamaha C5 recorded as the [Salamander Grand Piano](https://freepats.zenvoid.org/Piano/acoustic-grand-piano.html) (Alexander Holm, CC-BY 3.0), estimated from 480 recordings by the tuner. It also carries what only a *second* stage can measure, because it depends on the whole instrument at once rather than on one note: the bridge admittance, the sympathetic coupling, a per-key stereo spread, and 100 duplex segments over 23 keys taken from the library's release-resonance recordings at their measured frequencies — a median of 27 cents off the nearest partial of their own note, which is the scatter that makes a duplex sound like one. Everything the recordings cannot identify — strike position, hammer contact width, felt, soundboard, dampers — is inherited from the default. To reproduce it:

```sh
data/fetch_salamander.sh                       # 707 MiB, checksummed, into the gitignored data/
cargo run --release -p piano-tuner -- survey \
    data/salamander/SalamanderGrandPiano-V3+20200602.sfz \
    --preset presets/default.toml --cache data/cache/salamander \
    --name salamander-c5 --out presets/salamander-c5.toml \
    --credit 'Salamander Grand Piano V3 (Yamaha C5) by Alexander Holm, CC-BY 3.0'
cargo run --release -p piano-tuner --example fit_sympathetic -- \
    data/salamander/SalamanderGrandPiano-V3+20200602.sfz \
    --preset presets/salamander-c5.toml --out presets/salamander-c5.toml
cargo run --release -p piano-tuner --example salamander_ab   # A/B renders into renders/
cargo run --release -p piano-tuner --example verify_milestone_b -- [old-preset.toml]
```

`verify_milestone_b` re-measures the sympathetic milestone from rendered audio
with the same code the recordings are measured with — the spectrum census, the
halo isolated by subtraction, render health, neutrality, cost, and what the
between-partial statistic is actually made of. Given the preset as it stood
before a change it prints both columns.

`survey` is stage 1: everything an isolated recorded note can identify.
`fit_sympathetic` is stage 2, which is render-and-measure — it fits the duplex
segments from the library's release-resonance recordings, the sympathetic
coupling and bridge admittance by rendering the engine and measuring it against
`TUNING_REPORT.md`'s own numbers, and the per-key stereo spread by inverting
each key's drift on a line measured on the engine. Run it without `--out` to
measure and print without writing anything.

## MIDI replay

`render` accepts a standard MIDI file: note on/off on every channel, CC 64 as a *continuous* sustain pedal (half-pedalling survives), CC 66 sostenuto, CC 67 una corda, and the file's tempo map. Events are scheduled against the same event path the keyboard uses, so a replay is a performance the instrument plays rather than a special mode.

## Documentation

- `SPEC.md` — the model specification and acceptance tests.
- `DECISIONS.md` — the running log of every design decision and deviation.
- `TUNING.md` — the plan for estimating parameters automatically from recordings of real pianos (in progress; stage 1, its self-calibration gate and the first measured preset are built, in `tuner/`).
- `presets/default.toml` — the hand-tuned v1 instrument, written out in full.
- `presets/salamander-c5.toml` — the same instrument with everything stage 1 could measure off a real Yamaha C5 written into it.
