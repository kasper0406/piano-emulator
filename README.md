# piano-emulator

A physically modeled (simulated) grand piano — no samples. Every note is synthesized in real time from a model of the instrument: stiff strings, felt hammers, dampers, pedals, and a soundboard. The design goal is sound quality first, with all model parameters exposed so they can later be fitted automatically to recordings of real pianos (see `TUNING.md`).

## What is modeled

- **Strings** — modal synthesis with stiffness inharmonicity (`f_k = k·f0·√(1+Bk²)`) and frequency-dependent decay. A key's 1–3 strings and their two polarizations are not independent oscillators: they terminate on one bridge point, so each partial is a `2N × 2N` coupled system, solved at preset load into `2N` eigenmodes whose frequencies the bridge pulls together and whose decay rates it pushes apart. The mode that radiates most dies first, which is the fast-attack / slow-aftersound double decay — arriving out of one coupling constant rather than out of a hand-set balance.
- **Unison groups** — 1–3 strings per key, unevenly detuned and unevenly struck, coupled through the bridge: real beating and uneven decay rather than envelope tricks.
- **Hammer** — nonlinear felt (Hunt–Crossley hysteresis) integrated against an explicit agraffe reflection; contact time and brightness vary with key and velocity the way measured grands do.
- **Pedals** — sustain as *continuous* damper lift (half-pedaling works), sostenuto with correct capture semantics, and una corda (softer felt, one string of the group unstruck). Keys from G6 up have no dampers, as on a real grand. A damper that is touching but not seated limits the string nonlinearly, so a half-pedal buzzes rather than merely decaying faster.
- **The action** — the piano's own noises, at the levels they were measured at on a real instrument: the key-off thump on every release (scaled by how fast the key is let go), the damper lifting under a silently pressed key, and the pedal tray going down and coming up, scaled by how many dampers actually move. Panned per key, band-limited like the structure-borne path it travels, and deterministic — the same performance renders the same samples. The levels are ratios to a strike of the same key as a microphone hears it, so the strike they are quoted against is measured through the finished chain when the instrument is built: a preset that voices the piano more quietly takes its action down with it.
- **Touch** — a key pressed too gently to reach escapement lifts its damper and strikes nothing, which is how a pianist prepares sympathetic resonance without the pedal; release velocity sets how fast the damper falls.
- **Sympathetic resonance** — undamped strings pick up energy from everything else that rings; strike-and-release with the pedal down leaves the halo behind. A preset can give the bridge its measured admittance — a mean mobility curve with the board's discrete modes on it — and the halo is then coloured by the board the strings actually share instead of being spectrally flat, while a partial that sits on one of those modes loses energy into the board faster than the smooth fitted decay law says (`radiated_share`: T60 11.3 s against 14.6 s with a flat bridge). `render out.wav halo` is a phrase written to show it. Measured honestly the treble aftersound is still 21 dB short of the instrument it was fitted to — and that gap is on the board's late field rather than on this coupling, which has 0.1 dB of authority over it even at the largest value the stability contract will certify (`DECISIONS.md` 182–184).
- **Duplex and aliquot segments** — the lengths of string beyond the bridge and the agraffe, which have no dampers at all. A preset can give a key up to six of them, at measured frequencies rather than harmonic ratios; they are driven by that key's own bridge force and by the rest of the instrument through the bridge, and neither the key, nor the sustain pedal, nor sostenuto, nor una corda can stop them. Play a treble note staccato and the shimmer stays behind — at the top of the schema's range. At the level the *measured* table currently ships them they are 148 dB below the note and nobody can hear them; the mechanism is right and the drive is wrong — a segment is normalised to answer a **steady** drive at its own frequency, receives only an **impulsive** one, and the factor between those is `1 − r`, a part in ten thousand. Measured at the schema's own ceiling: both segments at +6 dB leave a halo 81.7 dB under their strike, which is under the level the modal culling zeroes, so *no* legal `gain_db` makes one audible. Fixing it re-decides what the field means and is a milestone of its own (`DECISIONS.md` 162, 163, 170, 260, `PHYSICS.md` §3).
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

**Two of those gates are red, on purpose and by name.**

`a_known_duplex_comes_back_from_the_engines_own_render_of_it` fails and has
since the unison became a coupled eigenproblem. It is not a tolerance and it is
not the estimator — with the modal culling switched off the injected segment
comes back at −0.05 cents having rung 1.38 s of the 1.4 s it was given — it is
the duplex bullet below (`DECISIONS.md` 260).

`no_note_of_the_line_wobbles_unlike_the_rest` fails because one note of a melody
does not belong. `tuner/tests/melody.rs` renders the Ode to Joy melody line
solo through the engine and through the recordings of the same piano and asks
whether any note stands further off the line's own register trend than the
piano's own worst note stands off its. F4 stands **1.28 dB** off in beating
where the recordings' worst note stands 0.21 dB off theirs, and clearing the
`notes.false_beat` splits that key was *drawn* takes it to 0.56: split depth is
the one thing `DECISIONS.md` 284 drew without closing it on the render, and
until it is closed the gate says so (`DECISIONS.md` 296-298).

Both are left failing rather than skipped because that is the only honest way to
carry a defect nobody has fixed.

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

**Iteration convention:** iterate with plain `cargo test` (the dev profile is
already `opt-level = 3`, and the release-only perf/calibration gates skip
themselves) and run `cargo test --release --workspace` once before calling work
done — the dev cycle is several times faster because release carries thin LTO
and a single codegen unit for honest perf numbers. The one-shot forensic
examples in `tuner/examples/` are behind the `diagnostics` feature so routine
builds skip them; run them with
`cargo run --release -p piano-tuner --features diagnostics --example <name>`.

**The measurement tools are parallel and cache what is not under test.** The
batch drivers render tens of independent notes per run — each render builds its
own engine and shares nothing — so they run across the cores, and every parallel
loop collects into an ordered container: `COMPASS.md`, `REALISM.md` and every
rendered file are the same bytes at any thread count. On top of that, the
*reference* side of a comparison is cached to disk under `data/cache/`, keyed by
content, because it does not move when the engine does:

| cache | holds | keyed on |
|---|---|---|
| `data/cache/reference/` | the Salamander recordings played by `piano_tuner::sampler`, as f32 WAV | `sampler::SAMPLER_VERSION`, the SFZ file's bytes, and the phrase (or key and velocity), duration and sample rate asked for |
| `data/cache/calibration/` | the self-calibration gate's tracked notes | a fingerprint of the engine's own audio and the tracker's own output on a probe note, plus the preset TOML, the note and the tracker settings |

Nothing is invalidated by a timestamp or a `--refresh` flag: a changed input
simply hashes to a different name and misses, so an entry is either the answer
to exactly this question or it is not read at all. The one thing hashing cannot
see is a change to the sampler's own code, which is what `SAMPLER_VERSION` is
for — **bump it in the same commit as any change that moves a rendered sample**;
its doc comment says exactly which changes those are. A cache hit is
bit-identical to a fresh render, not merely close, and
`tuner/tests/reference_cache.rs` is the test that says so. The caches are pure
speed: deleting `data/cache/` changes no number anywhere, and `data/` is
gitignored, so a fresh checkout simply starts cold.

Measured on an M4 Pro (`DECISIONS.md` 284): `compass_scan` 39 s -> 4.2 s cold and
3.1 s warm, `realism_bench` 59 s -> 14.3 s and 6.2 s, and
`cargo test --release -p piano-tuner --test calibration` 161 s -> 41 s and 36 s.
The calibration gate's own subsets are named in that file's header.

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
cargo run --release -p piano-tuner --example fit_motion -- \
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

`fit_motion` is stage 2's *motion* half: the within-string false beat (`notes.false_beat`) and the
strike vector's velocity law (`[voicing.strike_direction]`) inverted from the recordings' own beat
depth and rate, `notes.detune_cents` re-fitted where the coupled unison still lets a beat identify
it, and `notes.partial_gains` as the full measured ratio of the recording's time-zero spectrum to
the engine's own render of the same note. Unlike `fit_partials` it is re-entrant: every fit clears
the field it writes from the probe before rendering it. `DECISIONS.md` 239-248.

Its fifth stage, `--stage texture`, is the one that reaches the 58 keys the library never sampled.
A per-partial row and a within-string split are *measurements* and cannot be invented for a key
nobody recorded — but the **distributions** the 28 measured keys carry are statements about the
instrument, and those can be drawn from: how much roughness a row has (register-free, 4.4 dB of
robust spread), how tied neighbouring partials are (lag-1 +0.11), how far up the series a row
reaches, how many splits a wire has and at what rate and depth. Every cell is a draw seeded from
the key number and one named constant, so a re-emitted preset is the same preset; no cell is ever
copied from a neighbour, because the recordings say the roughness is *not* shared between notes at
the same frequency; nothing is drawn that the fitted rows cannot separate from a *colour*, which is
why the tilt is not drawn and a row shorter than four cells is not written; and the drawn rows go
through the same rails, the same power pin and the same close-on-the-render the measured ones do. Which rows were drawn is written down in
`notes.synthesized_texture` so a library that later samples one of those keys can replace them
without guessing. `DECISIONS.md` 284-291.

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
- `renders/realism/REALISM.md` — the standing realism scoreboard: six fixed phrases rendered from one event list through both the engine and the Salamander recordings, with `TUNING.md`'s stage-2 losses measured over each pair *and* the same measurement between two recordings of the same piano, which is the noise floor that makes the first number readable. It also carries **Columns A and B** (`FUNDAMENTALS.md` §II.3): four per-cell measurements of how a single partial *moves* — instantaneous-frequency mismatch and placement, beat-depth error and velocity coherence — over sixteen key × partial cells at three velocities, each with a gate, because every other column on the board is a functional of energy and the artefact those were built to catch is not. All four gates pass on the measured preset (`DECISIONS.md` 253); `cargo run --release -p piano-tuner --example motion_score` is the same four numbers with every cell printed, in seven seconds, for iterating a fit against them. Written by `cargo run --release -p piano-tuner --example realism_bench` (needs `data/fetch_salamander.sh`); the metrics themselves live in `tuner/src/realism.rs` so the scoreboard and the loss an optimizer minimises are one piece of code.
- **The melody gate** is the listener's own test, made permanent
  (`DECISIONS.md` 296-298): `cargo test -p piano-tuner --test melody` plays the
  Ode to Joy melody line alone — the soprano of the `excerpt` phrase, the same
  notes from the same `realism::ODE_MELODY` — through the engine and through the
  recordings, measures each note's roughness, beating and 2-6 kHz share, and
  asserts that no note stands further off the line's own register trend than the
  recordings' worst note stands off theirs. It exists because every other
  standing number here is either a *compass* statistic (88 keys struck alone) or
  a mean over a phrase, and neither of those is a tune with one note wrong in it.
  `cargo run --release -p piano-tuner --features diagnostics --example melody_line`
  is the same measurement printed in full, with flags that undo one table at a
  time so a failure can be attributed to the table that causes it; it writes
  `renders/melody/MELODY.md` and the two rendered lines beside it, because the
  complaint this gate exists for was made by listening to them.
- **Brilliance** has no standing report, because the audit that measured it (`DECISIONS.md` 292-295) moved nothing: `cargo run --release -p piano-tuner --features diagnostics --example brilliance` prints, per key and per phrase, how much 2-6 kHz and 6-12 kHz energy the engine carries against the recording of the same note at 0.1 s and at 1 s, each against the reference's own velocity-layer spread. It exists because `COMPASS.md`'s `centroid` is a mean *partial index* and the ear's brightness is absolute. It refused the top octave's decay (the recording's late energy there is its room, 20-30 dB over the note's own partial), acquitted the master shelf on its measured leverage, and convicted the partial envelope above the fitted rows — an error in partial *number* rather than in frequency, and one whose fix is a decay re-fit rather than a filter. The measurements are in `tuner/src/estimate/brilliance.rs`.
- `presets/default.toml` — the hand-tuned v1 instrument, written out in full.
- `presets/salamander-c5.toml` — the same instrument with everything stage 1 could measure off a real Yamaha C5 written into it. Its `notes.partial_gains` and `notes.false_beat` tables now cover the whole compass: 28 keys measured against their own recordings and 50 **drawn** from those keys' distributions, named in `notes.synthesized_texture` (`DECISIONS.md` 284-291, 300). Both halves of a drawn key are closed on the **render** — the row against the recordings' own roughness of that register, the splits against their own beat depth — which is what a fitted key's row and a fitted key's splits are each closed against too.
