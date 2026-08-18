# CONTEXT.md — the onboarding pack

Read this FIRST, instead of spelunking the repo. It is the maintained summary of
what exists, what is guaranteed, and where things live. `DECISIONS.md` (the
numbered log, 390+ items) is the authority when anything disagrees; this file
tells you which items matter for your task. Update this file when a milestone
changes what it states.

## What this is

A physically modeled grand piano in Rust: modal synthesis, no samples. Every
parameter is physical and lives in a preset (TOML). The measured preset
(`presets/salamander-c5.toml`) was estimated from recordings of a real Yamaha C5
(Salamander library, CC-BY); `presets/default.toml` is the hand-tuned v1
instrument. MIT, free, open source; App Store via AUv3 planned
(`DISTRIBUTION.md`, `SHIPPING.md`).

## Workspace map

- `engine/` — the instrument. Real-time path: no alloc/locks/syscalls after
  construction; 48 kHz fixed; 128-frame blocks with a remainder FIFO so any
  request length renders identically. Strings are coupled eigenmodes solved at
  preset load (2N×2N per partial: unisons × two polarizations, bridge-coupled).
  Mechanism noises, silent press, continuous half-pedal, sympathetic bus with a
  validated stability contract, virtual-mic stereo pair in mid/side form.
  Live MIDI in (`--midi-in`, CoreMIDI/UMP, u16 velocity, two value-preserving
  lanes: 0–255 = MIDI-1 numbers, 256+ = fine 1/512 steps).
- `ffi/` — C ABI (`pe_*`, header committed) + boundary resampler, bypassed
  bit-exactly at 48 kHz. The C harness renders sample-exactly what the CLI does.
- `tuner/` — offline analysis + the preset factory + the boards, all
  subcommands of `piano-tuner`: `track/estimate`, `survey`, `fit --stage`,
  `sympathetic`, `tail`, `noise`, `mics`, `radiation`, and the boards
  `bench/compass/melody/chain` (each writes its own document into `renders/`),
  audits `score/brilliance/residuals/ab`. Fit loops are batched (rayon);
  reference renders and the calibration corpus are content-cached under
  `data/cache/` (no refresh flags — caches key on content and cannot lie).
- `forensics/` — workspace member EXCLUDED from default-members: one-shot
  instruments behind numbered DECISIONS items. Build with `cargo build -p
  forensics`. Its README indexes them. Verifiers: REUSE these instead of
  writing new measurement code.
- `presets/`, `data/` (gitignored, fetch scripts checked in), `renders/`
  (gitignored), `docs/history/` (superseded investigation records).

## Invariants and contracts (each is pinned by tests)

- Determinism: identical inputs render identical bytes, including noise
  (seeded per event) and across process runs.
- Default-preset neutrality: new preset fields are absent-means-old; when a
  construction change makes bit-exactness impossible, the contract is measured
  equivalence with pins, and the break is a numbered decision.
- Stability: every eigenmode strictly inside the unit circle (asserted at
  construction, fuzzed at schema rails); sympathetic loop gain validated
  against the realized bridge filter; DRIVE_CEILING backstop.
- Provenance: measured vs synthesized preset values are marked
  (`notes.synthesized_texture` / `synthesized_decay`); fitting uses only
  genuinely recorded reference keys; scoring does too (transposed reference
  notes are listening material — the library samples minor thirds).
- Mono discipline: the mic pair's mono fold-down equals the pan-pot render
  (bound ~-120 dBFS); every mono board is computed on mono sums.

## The gates (run `piano-tuner <board>`; all seconds-fast, warm)

`bench` (REALISM.md: mel vs floor, modulation, attack, release, stereo
coherence + per-channel columns), `compass` (88 keys vs strung-alike
neighbors + recordings), `melody` (the Ode line: roughness/wobble/hf/strike
/channel, head+tail windows, recorded-key bars), motion columns A1/A2/B1/B2
(FM axes), limiter budget, release-click, stability fuzz, perf (<50% of one
core; currently ~30%).

**Documented reds (deliberate, by name):**
1. `a_known_duplex_comes_back_from_the_engines_own_render_of_it` — duplex
   segments need broadband drive; field semantics to be re-decided (D260).
2. `each_loudspeaker_has_the_recordings_spectrum_where_the_mic_pair_acts` —
   the modal lobe's per-channel defect (D392-396). Two mechanism milestones
   have been reverted rather than shipped (D404, D406) and a third was not
   started (D407-411): the mechanism is understood, the *source* cannot pay
   for it, and that is the open item below. The gate prints a sixth-octave
   board (D405) before it asserts — that is the readable half of a red.
   **This red is the second half of a two-milestone repair and is gated on
   the first (D411): do not accept a mandate that asks for it on its own.**
   The first half was built and measured in D412-416 and does **not** land;
   read those before proposing either half again.

## Conventions (hard rules for agents)

- Iterate with plain `cargo test` (dev profile is opt-level 3; release-only
  gates self-skip). Full `cargo test --release --workspace` at most twice per
  agent, at phase ends. Suite baseline: 708 green / 2 documented reds.
- Any command trending past ~5 minutes: parallelize or split the tool; never
  wrap it in a sleep/poll loop. Time-box closed-on-render fit loops; report
  budgets; report-and-stop beats converge-at-any-cost.
- Fixes land in the fit/draw/construction, never as hand-edits to preset
  values or widened bars. Falsification tests: a fixed defect gets a test
  that reproduces it on the old code.
- DECISIONS.md is append-only with continuous numbering; parallel workflows
  get reserved ranges. Renders are gitignored; nothing in renders/ is ever
  the only copy of evidence.
- Do not commit; the session owner reviews and commits.

## Current open items (beyond the two reds)

**The blocker under the third red** (D407-416). The direct path had per-partial
gains and *no radiated response between the partials*. That stage now exists —
`[soundboard.radiation]`, a fitted sixth-octave minimum-phase colouration of the
drive, absent-means-old, railed, `f64`, +0.5 points of one core (D412) — and
`piano-tuner radiation` fits it against D410's own target, reproducing D408's
table to ±0.01 dB. **It is not in the shipped preset and the reason is
measured** (D414): with the partials free the fit converges (deficit column flat
to 0.202 dB over nineteen bands, 180 Hz +0.04 → +6.91) and **four gates go red**
— a 16.7 dB curve moves individual keys' fundamentals, F#4 reads +9.75 dB bright
on the melody board, and `bench`'s 125-250 Hz `r0` goes 0.057 → 0.340; with the
partials held it cannot reach the target at all, because 60 % of the band the
target is written on *is* partials. **What was learned and is worth starting
from** (D413): D408's "+0.04 against +8.97 required" is two questions read as
one — **8.93 dB of shape**, which a source colouration can produce and the
instrument refuses, and **1.99 dB of uniform**, which is arithmetically
unreachable by any source (a share has no uniform component) and is the
rotation's own bill; the span's pair average reads **−0.19 dB** against the
recording, so the source's level over 100-810 Hz is already right. **The nodal
rotation is still not to be built** (D411's ordering rule, unchanged): the first
half has not landed. A fourth attempt needs a statistic that separates the floor
between partials from the partials themselves — that is the missing instrument,
not another filter design. D411's median-vs-pooled item is settled and printed
both ways (D415): `mono_balance` (median, still the gate) reads **+7.93** at
252 Hz where `mono_pooled` (energy-weighted) reads **+2.78**, and eight of
nineteen bands change sign between them.

Treble sympathetic halo ~21 dB short (board late field); per-key brightness
tilt not drawn for unsampled keys (needs more recorded keys by policy);
phantom partials deferred (-60 dB); AUv3/standalone Swift app not started
(SHIPPING.md sequences it); SL88 MK2 hardware smoke test pending hardware.
