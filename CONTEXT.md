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
- `app/` — **Swift, not Rust, and the only non-cargo directory that ships.** The
  AUv3 (`aumu`/`Pemu`/`KsNi`, sandboxSafe, `AUParameterTree` for the pedals plus
  two read-only meters, `fullState` = the whole preset TOML with a schema
  version, factory presets from the bundle) and a SwiftUI standalone app that
  **hosts that same appex** through `AVAudioEngine`, with Core MIDI in, an
  on-screen keyboard and a meter. Both link `target/dist/libpiano_emulator_ffi.a`
  through a module map over the committed header. `app/Shared/` is the code both
  use; `app/ParityHarness/` renders the benchmark phrase through the AU's own
  render block with no GUI and hashes it. `app/build.sh` is the whole build
  (cargo → XcodeGen → xcodebuild → harness); `app/PianoEmulator.xcodeproj`,
  `app/.build/` and `app/build/` are generated and gitignored. D426-432.
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
- **Unscored dimensions are how this repository fails.** Five times now a
  defect a listener heard sat under a fully green board because no column was a
  function of it: the mechanism's loudness against its own note (D341), the
  pair's energy against the note's mono fold-down (D394), the two capsules'
  per-band level asymmetry (D417, accepted as unscored rather than modelled),
  **which loudspeaker a note's fundamental comes out of** (D446), and — the
  fifth and the plainest — **how loud the note is** (D453/D456: every melody
  column was a shape, a ratio or a position, so a C4 nine decibels under the
  piano's own moved none of them). The first four have one pattern, every board
  being a *mono fold-down* or a *symmetric* function of the pair; the fifth has
  another and it is now its own rule: **every per-note verdict on the melody
  board was a *median* over the recorded keys, and a per-key error cancels out
  of a median.** Each column had carried a `seam` — the worst departure from
  that median — since D288 and not one was gated on it. Before adding a
  mechanism, ask which statistic would move if it were wrong; if the answer is
  "none", that is the column to write first, and ask it of the *order statistic*
  as well as of the quantity. The shape recurs one level up: D457's level fit
  also moved the whole melodic register 3-6 dB against the compass line, which
  a seam taken against the register's own median cannot see (D457's caveat ii).
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
/channel/**balance**/`splitting`/**`loudness`**, head+tail windows,
recorded-key bars; **every window is
counted from an onset found in the 3 kHz-and-up band** — D452, because below
that a 1 ms envelope of a bass note is its carrier and the old detector landed
+73 ms past C4's own hammer on the engine and +42 on the recording;
**`loudness` is D456's, the first column on any board here that is a function
of a level, and the first whose verdict is a *seam* — the worst per-key
departure from the register's median error, where every balance verdict is
that median and cannot see a per-key error at all**), motion columns A1/A2/B1/B2
(FM axes), limiter budget, release-click, stability fuzz, perf (<50% of one
core; currently ~30%).

**Documented reds (deliberate, by name):**
1. `a_known_duplex_comes_back_from_the_engines_own_render_of_it` — duplex
   segments need broadband drive; field semantics to be re-decided (D260).
2. `each_loudspeaker_has_the_recordings_spectrum_where_the_mic_pair_acts` and
   3. `the_engines_stereo_image_is_the_recordings_in_every_band` — **one
   shortfall, read in two units, and it is the price of D418's rail.**
   `[voicing.mics.modal].lift` is now railed at **one**, the null, because
   above one the lobe inverts one loudspeaker against the other — the left
   over 232.0-272.3 Hz and the right over 316.0-357.4, with the flip landing
   mid-tune — and manufactures up to +6.18 dB of pair energy the mono sum
   does not contain (D392, D417; the "outright nulls at two frequencies" of
   D392 do not exist and D423 is the correction). The recording's
   own nodal band asks for `E_side/E_mid` of **1.26 at 125-250 Hz and 1.58 at
   250-500** — more difference than sum, which for this construction *is* a
   lift above the null — and the refit under the rail reaches **0.80 and
   1.03**. `r0` reads that as 0.224/0.120 and 0.214/0.059 bars out on the
   coherence gate; `pair_db` reads the same thing as −0.88/0.49 and
   −0.66/0.38 on the per-channel gate. Both were green *on the artifact*.
   The per-channel gate's *shape* half additionally excludes the
   capsule-placement asymmetry D417 accepted as unscored
   (`ChannelColumn::asymmetry`, printed per band by the gate itself) and is
   still red at 2.38/1.91 and 2.47/1.74 — and **red against the unexcluded
   bar too** (1.15 and 1.09), which is why the exclusion changes no verdict
   this gate asserts. It is a *policy* sized by an arithmetic floor and not
   the floor itself: the floor holds for a model whose two channels depart
   symmetrically from their own mono, and a nodal-line lobe is not one
   (D424). **Do not close either by moving a bar**; D418's frontier map is
   the three swept conflicts that bound them, D423 adds the fourth axis
   (lift below the rail, which is worse), and
   `the_acceptance_still_fails_on_the_lobe_it_was_re_barred_against` is the
   falsification that keeps the exclusion narrower than the defect.
   D404/D406/D411/D414 are the four mechanism attempts that did not land and
   are what a fifth must start from.
5. `no_beat_rate_is_shared_across_the_measured_presets_compass` — **new at
   D458 and a histogram edge rather than a defect**, so it is a red that should
   be closed by re-basing a statistic and not by any preset. It counts, over
   4082 (key, partial) cells, the largest set sharing an eigenmode beat rate to
   within one millihertz, and requires it under one cell in fifty. The shipped
   preset read **80 of 4082 — 1.96 % against a bar of 2.00 %, a margin of 1.6
   cells** — and D455's decay refit reads **85**. `forensics/beat_census`
   prints what those cells are: **78 % of all cells have a mode split under
   0.05 Hz** and are excluded outright, another 907 sit between 0.05 and
   0.10 Hz, and the "worst bin" is the densest millihertz of that pile-up **at
   the bottom edge of the counted range** — a beat period of eighteen seconds.
   78 of the 85 are *different cells* from the 80 before, over 40 keys and 46
   partial indices, with no pattern; the clusters the test names by hand
   (`notes.false_beat`'s 3.0 Hz ceiling in 29 cells, its 0.37 Hz floor in 52)
   are both under the bar. Any refit of `notes.partial_sigma_scale` reshuffles
   it. Two fit-side attempts to close it were measured and one was reverted for
   costing the melody board's head `roughness` (D458).
6. `the_lines_pitches_come_out_of_the_loudspeaker_the_recordings_do` — the
   melody board's `balance` column, **new and red on the instrument that
   shipped** (D446-448). `10 log10(E_L / E_R)` at each note's own
   fundamental, median over the recorded ladder: **+8.84 dB against a bar of
   1.94**, where `channel` on the same renders is −0.49 against 0.91 and
   **green**. The two do not disagree — `channel` is `E_L + E_R`, symmetric
   under swapping the loudspeakers, so it cannot see a lean at all.
   `[voicing.mics.modal]` is `L = m(1 + B)`, `R = m(1 − B)`, so wherever
   `Re B > 0` it is **not a widener but a pan**, and its 174.3-456.5 Hz span
   contains every fundamental of the Ode line. **Do not close it by moving a
   bar** (`1.4826·MAD/√9` off the reference's own takes, no engine in it) and
   do not close it with the frontier: D448's table has four points green on
   both melody stereo columns and every one buys the register median by
   parking the band's upper edge between two notes of the tune, taking D4 to
   **+23 dB** and the line's swing from 13.1 to 27.9. What would close it is a
   **per-channel gain** — the reference leans −5.73 dB over the ladder and the
   feasible set reaches −2.0 — which is exactly the capsule placement D417
   accepted as unscored.

## Conventions (hard rules for agents)

- Iterate with plain `cargo test` (dev profile is opt-level 3; release-only
  gates self-skip). Full `cargo test --release --workspace` at most twice per
  agent, at phase ends. Measured in this tree at D458: **722 green / 6 reds**,
  every red named (D260's duplex, D418's two, D446's `balance`, D451's
  `splitting`, and D458's `no_beat_rate_is_shared_across_the_measured_presets_compass`
  — a histogram edge rather than a defect, see the reds list). Historical baseline: **712 green / 4 documented reds**
  (D418, verified and left standing by D423-425; the fourth is D446's
  `balance` column, which adds four green tests and one red one).
- Any command trending past ~5 minutes: parallelize or split the tool; never
  wrap it in a sleep/poll loop. Time-box closed-on-render fit loops; report
  budgets; report-and-stop beats converge-at-any-cost.
- Fixes land in the fit/draw/construction, never as hand-edits to preset
  values or widened bars. Falsification tests: a fixed defect gets a test
  that reproduces it on the old code.
- DECISIONS.md is append-only with continuous numbering; parallel workflows
  get reserved ranges. **A range you are handed is not a range you may write
  in until you have checked it against the ones already declared** — they live
  in the items themselves (`grep 'reserved range' DECISIONS.md`), a live
  workstream's claim can be wider than the numbers it has used yet, and two
  uncommitted hunks in one working tree cannot see each other. D446 is the
  overlap that happened (440-450 handed out inside the integration track's
  426-445) and the fix is to renumber before committing, never to write into
  the other track's range. Renders are gitignored; nothing in renders/ is ever
  the only copy of evidence.
- Do not commit; the session owner reviews and commits.

## Current open items (beyond the reds)

**The stereo line is re-opened by one column** (D446-448) and the entry above
it is the whole of it: `melody`'s `balance`. Nothing in `presets/` moved — `git
diff` over it is empty and the sounding path is byte-identical — and what is in
the tree is the column, its falsification, its reference-against-itself control,
the melody term inside `piano-tuner mics`' closed loop (D416's lesson: close on
what the gates read), and D448's forty-one-point frontier. Two things a
successor should not have to re-derive. **(i) The register and the tune want
opposite bands.** The column's median is taken over the recorded ladder, D#3 to
D#5, and the tune is C4 to G4; a *wide* band (170-1000 Hz x0.99) gives the best
line anyone has measured — swing 8.8 dB against the shipped 13.1, median line
error +0.78 against +7.8, and the coherence board's 125-250 improves from 0.224
to 0.175 — while its ladder median is +7.78 and red. **(ii) The reference's own
per-note image is not a piano's.** Its C4 puts the fundamental **16.86 dB into
the right capsule** (verified outside the tuner, stable from 4 to 32 heterodyne
cycles and by plain FFT), against −1.0 at the neighbours, and its median lean
over the ladder is −5.73 dB where nothing the schema can build passes −2.0.

**The rest of the stereo line is closed** (D417-425). The stopgap shipped: the lift is
railed at the null, `presets/salamander-c5.toml`'s `[voicing.mics]` is refitted
under it (`width` 1.632, `diffuse_coherence` 4.099, band **174.3-456.5 Hz at
0.99**), and the capsule-placement asymmetry is excluded from the per-channel
target and printed. **What that removes is a mechanism, not a level, and the
difference is measured** (D423, D425 — D417's "the audible artifacts are gone"
and D418's account of the nulls are both wrong and are corrected there). The
lobe's `B` is a **complex** filter response and the `R = m(1 − g)` shorthand two
milestones reasoned from is wrong by up to 13 dB: there are **no unity crossings
and never were** (`|1 − B| = 2|sin(arg B/2)|` at `|B| = 1`, not zero), and the
inverted spans on the old preset are 232.0-272.3 Hz in L and 316.0-357.4 in R —
0.18-0.23 octaves, not 0.76. Gone by construction under the rail, unbuildable by
any preset the schema accepts: **channel inversion in either channel**, the
pitch-dependent flip of which loudspeaker carries it, and pair energy above
+3.01 dB. **Not gone, and worse than before: the per-channel level loss.** `1 ± B`
is smallest where `|B|` is closest to *one*, so railing the lift at one deepens
it — the old lobe's worst was **−20.5 dB at 349.8 Hz** with either channel >10 dB
down over 0.105 octaves, and the refit reaches **−33.1 dB at 221.4 Hz in the
LEFT channel** (A3's fundamental) over 0.286 octaves in two zones. That notch is
inside the 125-250 band where the per-channel gate reads `dev_L −0.98` against
the recording's `+1.40`, which D423 names as the lead for a successor rather
than a proven cause. Of D392's three listener
complaints re-measured on the shipped instrument: C4's prominence is **gone**
(+6.42 → +4.07 dB with E4 now above it, the lobe's own prediction flat across
the line), the hammer/fundamental complaint has **lost its mechanism but not
its level** (F#4 R −18.33 → −6.54, F4 R −9.61 → **−10.09**; the worst loss on
the line improves 8 dB and moves note, and what fills a nulled channel is the
pair's geometric side rather than the lobe), and the chords' brilliance tilt is
**half gone** (the
lobe's share L −4.95 → −4.23, R −2.64 → +0.18; the −8.15 dB `dmono` deficit is
pre-existing and not the mic section's). A smaller lift does not buy the level
back either: 0.75 and 0.50 on the same band take the coherence board from
0.224/0.214 bars out to 0.477/0.427 and 0.738/0.602 (D423). What it bought
beside that: **melody all five green** (the `channel` balance −0.49 against a
bar of 0.91, where the merely-clamped instrument reads −1.06 and fails),
spacing readback **+6 / +1 / −7 %** against 20, and D395's third constraint
dissolved — the estimator's band boundary moves from 225 Hz to **170** under a
weaker lobe (`where_the_bands_lower_edge_starts_biasing_the_reading`, an
`#[ignore]`d instrument in `tuner/tests/mics.rs`). The next mechanism, if one is
ever attempted, is not a filter design: it is the mono fold-down paying for a
nodal line, and D411's ordering rule and D414's refutation still stand.

**The blocker under the mono-source milestone** (D407-416). The direct path had per-partial
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

**C4's two defects are closed, and what is left of each is named**
(D452-458). The listener's "that C sounds off" was **not** pitch (+0.17 cents
relative) and **not** the mic stage: both defects reproduce on a single note
struck alone, mono, at velocity 88 (`forensics/c4_ledger`), so the pair, the
phrase, the limiter and the master gain were exonerated by construction.

**(a) The level — D272 is re-opened and has a written home** (D457). A key's
own loudness against the recording of the same key now lives in
`notes.partial_gains`, written through that field's own pinning by a new
re-entrant stage, `piano-tuner level`. D272's objection was right and is the
design: the removed level's spread is 4.82 dB, a smooth curve explains a
quarter of it, the residual is white, and carried in full it puts F#5 17.9 dB
over its neighbours — so what is written is each key's own measurement
**shrunk per key toward the compass line by `1 − (take_sigma / departure)²`**,
with the noise measured on the library as the distance between two takes of one
key (1.40 dB here), capped at the piano's *own* worst key-to-key level residual
(6.40 dB). A-weighted head energy against the recording of the same key,
normalised to the nine-key ladder median: **C4 −8.96 → −0.96, D#3 −8.27 →
−1.20**, worst key on the ladder **8.96 → 1.85 dB**, every key improved or
held. The tune's own contour (A-weighted 125 ms steps into and out of every
C4, `forensics/melody_contour`) went from **−1.75 in / +1.10 out** — a *dip*
into the note the recording *lifts* into, at +2.44 / −1.41 — to **+1.03 /
−0.37**: the sign is the piano's now. **The gate this class was invisible to
is the melody board's `loudness` column** (D456), gated on its **seam**, and
the shipped-before preset fails it at C4 (8.87 against a bar of 5.21) where
this one passes (1.90 at A3).

**(b) The held octave — the sub-2 kHz band has an owner** (D454-455). The span
convention was decided first and by measurement (D454): a partial's fall is now
the difference of two **mean-power readings over a 0.45 s window** at the same
two instants, slid rather than narrowed at the strike, which is the only
convention of seven whose answer survives moving the span's edges (1.18 / 1.46
against the old reading-at-an-instant's 1.50 / 1.94, and least squares' 1.69 /
6.67). The band below 2 kHz then went to **`tail`, per partial, closed on the
render** (D455) — not to `shaping` normalising the low band separately, and
that is measured rather than argued: renormalising C4's low half half-fixes
`k = 1` (0.53 → 0.71 where the render asks 0.97) and *breaks* `k = 2`
(0.87 → 1.16 where it asks 0.93). The seam was never the piano: the correction
the render asks for below 2 kHz is the reciprocal of what shaping's whole-row
normalisation divided out, rising with the key exactly as the share of a row's
partials above 2 kHz does (median ×1.43 over the recorded keys, ×2.70 at C5).
After the refit the octave-against-fundamental decay relationship's mean
`|error|` over the ladder is **8.30 → 3.09 dB** with **C4 +13.50 → −0.60**, and
C4's held `k2 − k1` at 1.5 s is **−20.2 → −5.1** against the recording's +0.1.
D335's step is **gone at its source** — the recorded keys' sub-2 kHz geometric
means come back to one, `LowDecay`'s line goes flat (`exp(+0.5488 −
0.01411·key)`, r −0.797 → `exp(+0.0745 + 0.00031·key)`, r +0.034) — so its
falsification changed material rather than retiring. Melody tail `hf`
3.75 → 3.06; band decay gap tenor 6-12k +5.27 → +7.92, treble 2-6k
+11.52 → +10.93.

**What is left, in order of size** (D458): the beat-census red above; **D#5's
octave**, the one ladder key whose decay relationship regressed, whose `k = 2`
renders 24.5 dB under the recording's at 0.10 s and is therefore a
`partial_gains` hole rather than a decay; the **5.1 dB left of C4's held
octave**, which is entirely the 5.8 dB the two partials already differ by at
0.10 s and is a gain-row question a per-key scalar cannot reach; the low band's
converged residual sitting at 1.10 rather than 1.00 because a per-cell deadband
approached from one side stops early; and the fact that `fit --stage
partial_gains` re-pins every row it re-fits, so **`level` must be re-run after
any `fit`**.

Treble sympathetic halo ~21 dB short (board late field); per-key brightness
tilt not drawn for unsampled keys (needs more recorded keys by policy);
phantom partials deferred (-60 dB); SL88 MK2 hardware smoke test pending
hardware.

**The plugin is built and proven, and what is left is named** (D426-432). The
AUv3 and the standalone both build headlessly from `app/build.sh`; `auval -v
aumu Pemu KsNi` is green with **no warnings**, out of process, 11 kHz to
192 kHz; and the AU renders the benchmark phrase **sample for sample** with the
C harness of D383 — md5 `f0fcb07999c00ca60110cd537de8f09e` on
`presets/default.toml`, `e13cd0ac9d367126ca7bf2b64b147e04` on the measured one,
at every host buffer that is a multiple of the engine's 128-frame block. Still
open: **sub-block note-on offsets in `engine/`**, which is the other half of
M2's own scope and the reason a DAW's grid still meets D55's 2.7 ms
quantisation; the four host smoke tests (Logic, GarageBand, Live, Reaper) that
need a person at a machine; signing/notarization (M5); the App Group preset
importer (M4's remainder); the App Store (M7); CLAP/VST3/AUv2 (M8). Two
measurements worth carrying forward: advertising MIDI 2.0 makes a 7-bit CC 64
mean `v<<25 / 2^32-1` rather than `v/127`, worth −94 dBFS peak on the benchmark
phrase and **not** a defect (D431a); and a 30 Hz meter on the same
`ObservableObject` as the rest of a SwiftUI panel cost 36.7 % of one core in the
appex with nothing playing (D431b).
