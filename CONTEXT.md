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
  `sympathetic`, `tail`, `noise`, `mics`, and the boards
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

## Conventions (hard rules for agents)

- Iterate with plain `cargo test` (dev profile is opt-level 3; release-only
  gates self-skip). Full `cargo test --release --workspace` at most twice per
  agent, at phase ends. Suite baseline: 701 green / 2 documented reds.
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

**The blocker under the third red** (D407-411): the direct path has per-partial
gains and *no radiated response between the partials*, so the engine's 100-800
Hz mono shape cannot be moved where a key has no partial. An energy-conserving
nodal mechanism needs the source to stand up to **+8.97 dB** above the
recording's own mono at 180 Hz before it is applied; it stands at +0.04, and
`body_modes` x8 reaches +1.7 while `partial_gains` +9 dB reaches +2.9. Next
milestone: a fitted sixth-octave colouration of the mono drive, fitted to the
recording's mono **divided by the pair's own mono transfer** — measure both
with `forensics/.../mono_mechanism keys`. It moves every mono board. Its
acceptance is D408's *standing* column rising to its *required* column, not
the per-channel gate; **the nodal rotation is not to be built until it lands**
(D411), in either the Givens or the `C=(A+B)/2, S=(A−B)/2` form, because an
energy-conserving mechanism on today's source lands the fold-down 4-6 dB under
the recording across the nodal band. Settle one thing before that fit is
written: `mono_balance` is a *median over keys*, so it weights a key carrying
1% of a band like one carrying 42% (D411) — the fit's target is pooled and
level-matched, and the two agree only if the band's energy is spread evenly.

Treble sympathetic halo ~21 dB short (board late field); per-key brightness
tilt not drawn for unsampled keys (needs more recorded keys by policy);
phantom partials deferred (-60 dB); AUv3/standalone Swift app not started
(SHIPPING.md sequences it); SL88 MK2 hardware smoke test pending hardware.
