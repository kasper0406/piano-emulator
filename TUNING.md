# TUNING.md — automated parameter estimation plan

> **Status (refreshed 2026-08-16). Live document; Phases A–E are done and
> stage 2 is most of the way built, which is more than the execution plan at the
> bottom describes.**
>
> **Implemented.** Phase A (preset extraction, stretch tuning, MIDI replay),
> Phase B (the `tuner/` crate, its STFT partial tracker and every estimator),
> Phase C (the self-calibration gate, `tuner/tests/calibration.rs`), Phase D
> (`presets/salamander-c5.toml`, estimated from 480 Salamander recordings) and
> Phase E (the residual report, now `docs/history/TUNING_REPORT.md`). Stage 1 is
> the `survey` subcommand. What this document calls "stage 2 (MAESTRO replay
> fitting)" was **not** built the way it is described below: rather than fitting
> interaction parameters against a Disklavier performance corpus, stage 2 is
> render-and-measure against the *same* Salamander library — `sympathetic`
> (duplex segments, halo coupling, bridge admittance, per-key stereo spread),
> `fit` (false beat, strike direction, detune, partial gains, drawn texture) and
> `tail` (the upper partials' decay). MAESTRO has never been downloaded. Four
> standing boards were added that this plan does not mention: `bench`
> (`REALISM.md`, the fitness function), `compass`, `melody` and `chain`.
>
> **Planned and not built.** CMA-ES or any other global optimiser over the
> preset: every stage above is a direct inversion or a closed-loop bisection,
> and nothing yet minimises `REALISM.md`'s loss as an objective. Stage 2 over
> MAESTRO, and with it everything that needs a *performance* rather than an
> isolated note or a rendered phrase. `DECISIONS.md` 317 ranks what is left.
>
> **Superseded.** The "Engine prerequisites" list below is history — all three
> landed. Where this file and `DECISIONS.md` disagree, the log wins.

Goal: replace hand-tuned per-note tables with parameters estimated from recordings of real pianos, so the instrument converges toward a specific, real piano rather than a plausible one. Two stages: (1) direct per-note estimation from isolated-note recordings, (2) interaction-parameter fitting by replaying ground-truth performances and comparing against the recording. Stage 1 does most of the work because the modal model's parameters map almost one-to-one onto measurable quantities.

## Data sources

| Dataset | What | Use |
|---|---|---|
| **Salamander Grand Piano v3** | Yamaha C5, 48 kHz/24-bit lossless, 16 velocity layers, notes sampled at minor-third spacing (A0, C1, D#1, …), plus release samples, string-resonance samples, pedal noises. CC-BY. | Stage 1: per-note estimation. Primary target piano for the first calibrated preset. |
| **MAESTRO v3** | ~200 h Yamaha Disklavier competition performances, 44.1 kHz audio with note-accurate aligned MIDI incl. velocities and pedal CC. | Stage 2: interaction parameters (coupling, soundboard, dampers, pedal curves). Use a small curated subset (a few solo pieces with clear pedal usage), not the full 120 GB. |
| YouTube / user-annotated recordings | Lossy, unknown chain; user can annotate room/instrument. | Validation and "chase a specific famous piano" experiments only — never parameter estimation. Lossy codecs destroy exactly the evidence we fit (high partials, decay tails, resonance halo). |

Sample-rate policy: engine stays at 48 kHz. Salamander is already 48 kHz. MAESTRO audio gets resampled 44.1 → 48 kHz offline with a high-quality resampler (`rubato`); resampling artifacts are far below the loss floor of stage 2.

## Architecture

Convert the package into a cargo workspace:

```
engine/        # existing crate (piano-emulator), unchanged role
tuner/         # new crate: analysis + estimation pipeline (offline only, no RT constraints)
presets/       # output: versioned preset files (self-describing TOML or JSON)
data/          # downloaded datasets (gitignored), fetch scripts with checksums
```

### Engine prerequisites (must land before estimation is useful)

1. **Preset extraction.** All hardcoded per-note tables (B curve, sigma0/sigma1, T60 anchors, unison count/detune, strike position, hammer K/p/mass, damper D, soundboard mix/decays, coupling) move into a serde-serializable `Preset` struct; engine constructs from a `Preset`, binary gains `--preset <file>`. Current hardcoded values become `presets/default.toml`.
2. **Stretch tuning.** Real pianos are stretch-tuned (Railsback curve); the engine currently assumes equal temperament. `Preset` gains a per-note `f0` table; `note_to_freq` becomes a preset lookup with the ET formula as fallback.
3. **MIDI replay rendering.** `render` learns to take a standard MIDI file (`midly` crate): schedule NoteOn/NoteOff/pedal CC64 (continuous, for half-pedal), CC66, CC67 through the existing event queue into an offline render. This is both stage 2's forward model and a generally useful feature.

### Tuner crate components

- **Audio I/O:** WAV + FLAC decode (`symphonia` or `claxon` + `hound`).
- **Partial tracker:** STFT (large windows, ≥ 2^16, hop ~10 ms) → per-frame peak picking with parabolic interpolation → track association across frames seeded by the inharmonic model's predicted `f_k`. Output per note: `[(k, f_k(t), a_k(t))]` trajectories.
- **Estimators** (each a pure function of trajectories, unit-tested on synthetic input):
  - `f0`, `B`: robust least-squares fit of `f_k = k f0 sqrt(1 + B k²)` over detected partials (median-of-ratios initialization; reject outlier partials > 20 cents off).
  - Per-partial decay: fit `log a_k(t)` with a **two-exponential** model → fast/slow sigma per partial → engine's polarization split (amplitude ratio + decay ratio) and smooth `sigma(f)` curves.
  - Unison detune: dominant modulation frequency of each partial envelope (autocorrelation of `a_k(t)` residual after decay removal) → beat rates → detune per note.
  - Strike position: fit the `sin(k π x)` comb to time-zero partial amplitudes (minima locations are the strong signal).
  - Hammer K/p/mass + velocity map: time-zero partial amplitudes across the 16 velocity layers, divided by mode input gains, give the excitation spectrum per layer; fit the felt model's pulse spectrum to it. The layer→hammer-velocity mapping is unknown, so fit it jointly as a monotonic map (16 free values, monotonicity-constrained).
- **Interpolation:** Salamander samples every 3 semitones → fit smooth curves (monotone cubic in log-f) across the compass for every estimated quantity; all 88 notes read from the curves. Estimated notes also keep their direct values.
- **Loss functions (stage 2):** multi-resolution log-magnitude STFT loss (window sizes ~ {256, 1024, 4096}) with mel weighting, plus feature losses: partial decay-rate error, spectral-centroid-vs-velocity error, onset transient energy envelope error.
- **Recording-chain absorber (stage 2):** a static linear filter applied to the engine output before loss — parameterized as a ~40-band cepstrally-smooth log-magnitude EQ (optionally + one short early-reflection IR later). Jointly optimized so room/mic/mastering coloration lands here instead of bending piano parameters. User room annotations seed/constrain it.
- **Optimizer (stage 2):** CMA-ES over the ~15–25 interaction parameters (coupling, damper D and weight-curve, pedal response curve, soundboard mix, FDN decays, body-mode gains, chain-EQ). Batch candidate evaluation is embarrassingly parallel → rayon across cores now; the SME/batched-matmul formulation (DECISIONS.md #11) is the upgrade path if evaluation becomes the bottleneck.

## The self-calibration gate (do this before touching any real data)

Render isolated notes **from our own engine** with a known preset, run the full stage-1 pipeline on those WAVs, and require that it recovers the known parameters: `B` within 2 %, per-partial T60 within 5 %, detune within 0.05 Hz, strike position within 5 %, hammer params within 10 %. This closes the loop on estimator correctness with exact ground truth and zero confounds. Any estimator that can't pass this on synthetic data has no business reading real data. Keep it as a permanent `cargo test --release` integration test in `tuner/`.

## Execution plan (next workflow, after v1 build completes)

Phase A — **Engine prerequisites**: preset extraction + stretch tuning + MIDI replay (touches engine; must keep all v1 acceptance tests green; `presets/default.toml` reproduces current sound bit-exactly or within measurement tolerance).

Phase B — **Tuner foundation** (parallel with A where files are disjoint): tuner crate, audio I/O, STFT/partial tracker, estimators — each unit-tested on synthetic trajectories.

Phase C — **Self-calibration gate**: wire A+B together; the recovery test above must pass before proceeding.

Phase D — **Salamander stage 1**: fetch script (+ checksums, gitignored data dir), run estimation across all sampled notes × velocity layers, fit compass curves, emit `presets/salamander-c5.toml`. Gate: v1 acceptance tests pass with the new preset (tolerances may need loosening where the real piano genuinely differs — e.g., its actual T60s); A/B renders + spectrogram-diff images for human review.

Phase E — **Report**: measured-vs-model residuals per note — this tells us what the *model* is missing (expected suspects: longitudinal/phantom partials, soundboard directivity, pedal noises) and drives the next modeling iteration.

Stage 2 (MAESTRO replay fitting) becomes its own workflow after Phase E, informed by its residuals.

## Risks / open points

- **Salamander minor-third spacing** means 2 of every 3 notes are interpolated — fine for smooth quantities (B, decay), possibly visible for idiosyncratic ones (individual unison detunes). Acceptable for v1 presets.
- **Release/pedal noise samples** in Salamander are a gift for later (key-off thump, damper noise) — out of scope for this pass, noted in Phase E report.
- **Sympathetic resonance can't be estimated from isolated notes** — that's inherently stage 2 (MAESTRO includes rich pedal usage).
- The felt model may not be able to match the measured excitation spectra at all velocities — if residuals are structured, that's model insufficiency, not estimator failure; report, don't force.
