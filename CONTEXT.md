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
  validated stability contract, undamped duplex/aliquot segments launched by the
  hammer's own broadband knock across the bridge (D481-484), virtual-mic stereo
  pair in mid/side form.
  Live MIDI in (`--midi-in`, CoreMIDI/UMP, u16 velocity, two value-preserving
  lanes: 0–255 = MIDI-1 numbers, 256+ = fine 1/512 steps).
- `ffi/` — C ABI (`pe_*`, header committed) + boundary resampler, bypassed
  bit-exactly at 48 kHz. The C harness renders sample-exactly what the CLI does.
- `tuner/` — offline analysis + the preset factory + the boards, all
  subcommands of `piano-tuner`: `adapt` (the library-description layer: writes
  the instrument definition a library does not ship and resamples a tree onto
  the engine's clock, once, offline), `listen` (per-preset listening material
  against **that preset's own** library), `track/estimate`, `survey`,
  `fit --stage`,
  `sympathetic` (`--only duplex` runs its first stage alone), `tail`, `noise`
  (`--stage mechanism` writes the four mechanism events through D531's
  plausibility gate; the default stage is the hammer's own balance),
  `mics`, `radiation`, and the boards
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

## The preset range and the library adapter (D516-530)

**Three measured pianos, one factory.** `salamander-c5` (Yamaha C5, CC BY 3.0),
`concert-grand-d` (Steinway D concert grand, bitKlavier Grand "Piano Bar"
image, Princeton, CC BY 4.0) and `upright-parlour` (Knight upright, VCSL,
CC0 1.0). **Preset names are descriptive, never brand names**; the instrument,
library, author and licence live in `ATTRIBUTION.md` and in each preset's own
`description` field, and the standing rule — licence + URL recorded in
`ATTRIBUTION.md` *and* in the fetch script *before* a parameter ships — is now
**tested rather than remembered** (`tuner/tests/adapter.rs`, three tests).

**What a sample library is, as data**: `tuner/src/adapter.rs`, one
`LibrarySpec` per library, holding the five things the factory used to assume.
(i) `Layout` — which keys are *genuinely recorded*, the set the evaluation
policy fits and scores against; (ii) `Bands` — how many velocity layers and
where their bands sit, which **is** the abscissa of every velocity fit because
`Sample::midi_velocity` is the band midpoint (D519); (iii) the rate, published
against delivered, so a preset whose material passed through a resampler says
so; (iv) `FilePattern` — how a file names its key and layer, including VCSL's
note names sitting an octave below the standard spelling; (v) `MechanismFiles`
— the key-off thumps, the pedal tray and the pitched release resonances, which
are the whole input of `noise` and half of `halo`.

`piano-tuner adapt <id>` does two things, once per library rather than once per
fit: it **writes the instrument definition** a library does not ship, and it
**resamples a tree published at another rate onto the engine's clock** in one
offline pass of `audio::resample` to float WAV, so the boundary resampler is
not inside every subsequent measurement of that preset. A generated map asserts
`amp_veltrack=0`, no `volume`, no `tune`, no `offset`: it is a measurement
input, not a performance instrument.

**Salamander is the first instance and the one nothing is generated for.** It
ships its own map, every bar in this repository was measured through it, and
`adapt salamander` refuses. Its description exists to be *falsified* against
the shipped file, and the render path is proved unmoved rather than asserted:
`the_salamander_reference_render_is_bit_exact` pins an FNV-1a hash of the
sampler's render of all six benchmark phrases, and the same hash was measured
on a tree with the adapter removed (D517).

**The remaining Salamander-shaped line**: nine drivers joined Salamander's own
filename to their data directory. Six now call `adapter::instrument_path`
(`bench`, `compass`, `noise`, `level`, `stereo`, `ab`). Three do not, because
they belong to other workstreams' files — `tools/melody.rs`, `tools/mics.rs`
and `tools/tail.rs` — and until they adopt it a generated library gets a
symlink at the legacy name (`adapt --legacy-alias`). **That scaffold is meant
to be deleted**; it is a one-line change in each (D521).

**Per-library bars, permanently.** Each preset is scored against **its own**
recordings and never against another piano's: every board's floor is the
reference's own take-to-take disagreement, so pointing a board at a different
data directory re-measures the bar as well as the target. `piano-tuner listen
<data> <preset>` writes the per-preset listening material — the melody line and
a pedalled chord phrase, engine and that library's own recordings, each
normalised separately — into `renders/<preset>/` with a `README.md` naming
which of the tune's keys are genuine takes.

## Invariants and contracts (each is pinned by tests)

- Determinism: identical inputs render identical bytes, including noise
  (seeded per event) and across process runs.
- Default-preset neutrality: new preset fields are absent-means-old; when a
  construction change makes bit-exactness impossible, the contract is measured
  equivalence with pins, and the break is a numbered decision.
- Stability: every eigenmode strictly inside the unit circle (asserted at
  construction, fuzzed at schema rails); sympathetic loop gain validated
  against the realized bridge filter; DRIVE_CEILING backstop.
- **Unscored dimensions are how this repository fails.** Seven times now a
  defect a listener heard sat under a fully green board because no column was a
  function of it: the mechanism's loudness against its own note (D341), the
  pair's energy against the note's mono fold-down (D394), the two capsules'
  per-band level asymmetry (D417, accepted as unscored rather than modelled),
  **which loudspeaker a note's fundamental comes out of** (D446), **how loud the
  note is** (D453/D456: every melody column was a shape, a ratio or a position,
  so a C4 nine decibels under the piano's own moved none of them), and the two
  newest, which are the **fourth and fifth unscored dimensions of the image**
  (D459/D460): **where the note's own overtones sit** and **when the note
  arrives at each loudspeaker**. Three patterns, and the third is new. The first
  four have one: every board is a *mono fold-down* or a *symmetric* function of
  the pair. The fifth has another and it is now its own rule: **every per-note
  verdict on the melody board was a *median* over the recorded keys, and a
  per-key error cancels out of a median** — each column had carried a `seam`
  since D288 and not one was gated on it. **The sixth and seventh have a third
  and it is the sharpest yet: a column can be a difference of two quantities
  while neither quantity is itself scored.** `splitting` is `balance − comb`
  *exactly* (it is one function), so a mechanism that moved a note's
  fundamental and that same note's overtones together cancelled out of it while
  `balance` read one frequency per note and `channel` was a sum over the pair —
  and the pair *geometry* is exactly such a mechanism. Its companion is a
  question of units rather than of statistics: every column of every board here
  was a function of two **magnitudes**, and half of where a listener puts a note
  is the interchannel **phase**. Before adding a mechanism, ask which statistic
  would move if it were wrong; if the answer is "none", that is the column to
  write first — and ask it of the *order statistic* as well as of the quantity,
  of each **term** of a difference as well as of the difference, and of the
  *phase* as well as of the level. **And of the target** (D466): a column can be
  a perfectly good statistic pointed at the wrong answer. **And of whether the
  statistic is monotone in the mechanism it is meant to bound** (D485, the
  eighth entry and a new pattern): the owner's complaint was a per-key
  bass-left/treble-right lean, and the obvious reading of it — the slope of
  `balance` against key — moves ±0.2 dB per semitone with nothing but a band's
  comb phase and is **not monotone in `width` at all**, so it would have passed
  every instrument the ladder contained. The reading that works is the note's
  whole *broadband* channel ratio, where the comb averages out and the pan law
  is what is left. Before writing a gate, render the mechanism at four settings
  and check the column moves with it. **And: a defect can hide another one.**
  `voicing.polarization_pan_spread` alternated its sign with the key parity, so
  a Theil-Sen line through the ladder was pulled toward flat and the shipped
  instrument passed `gradient` at −0.192 where the same instrument with the
  spread out reads −0.580 and fails. **The eighth is that
  same rule with the knife turned round** (D500): the treble halo was not
  unscored for want of a statistic — `estimate::halo` had *five*, and three of
  them were §4's between-partial census, which has a **floor**, and the floor is
  the struck note's own partials smeared outside the guard band. Taking the bus
  **and** the segments out of the instrument entirely moves that census by
  **0.08 dB at C6 and 0.87 at C7**, where the recordings stand 20-28 dB above
  the engine. So the question to ask of a target is not only "is it the right
  answer" but "**would the mechanism move it at all**" — and the way to find out
  is to remove the mechanism and re-read, which is a one-line experiment that
  eight milestones did not run. Four image columns
  were scored against the recording's own image for three milestones while
  D417's own entry in this list said that image is a microphone stand — so the
  only ways to pass were a lean the schema cannot build and a band edge parked
  between two notes of the tune, and both were measured and refused before
  anyone re-read the target. The order-statistic rule arrives with it and in a
  new shape: under the neutral target a *signed* median passes a line that
  alternates ±20 dB (the shipped line's splits read −0.4 signed and +7.1 by
  magnitude), so a target of zero has to be paired with a magnitude. The shape recurs one level up: D457's level
  fit also moved the whole melodic register 3-6 dB against the compass line,
  which a seam taken against the register's own median cannot see (D457's caveat
  ii); and D459's verdicts are a **slope** and a **swing** because a ramp across
  a tune cancels out of a median and is subtracted by a residual about that
  line's own trend.
- Provenance: measured vs synthesized preset values are marked
  (`notes.synthesized_texture` / `synthesized_decay`); fitting uses only
  genuinely recorded reference keys; scoring does too (transposed reference
  notes are listening material — the library samples minor thirds).
- Mono discipline: the mic pair's mono fold-down equals the pan-pot render
  (bound ~-120 dBFS); every mono board is computed on mono sums.

## The gates (run `piano-tuner <board>`; all seconds-fast, warm)

`halo` (`tuner/tests/halo.rs`, D500-506: the sympathetic halo of one struck key,
recorded alone against the engine's own isolated by subtraction, ten keys C4 to
D#6, verdict a **seam**, bar the reference's own take floor — it is a *test
file* rather than a board because its reference is fourteen files in the library
and its render is two per key),
`bench` (REALISM.md: mel vs floor, modulation, attack, release, stereo
coherence + per-channel columns), `compass` (88 keys vs strung-alike
neighbors + recordings), `melody` (the Ode line: roughness/wobble/hf/strike
/channel/**balance**/`splitting`/**`comb`**/**`cue`**/**`loudness`**/**`gradient`**, head+tail
windows, recorded-key bars; **`gradient` is D485's and it is the owner's own
complaint written as a statistic — the Theil-Sen slope of the note's whole
*broadband* `10 log10(E_L/E_R)` against key **over the recorded ladder**, target
zero, bar the recording's own gradient (−0.377 dB/semitone) plus its take
floor, x1.25 = 0.490. `comb` and `balance` read the tune; this reads the
compass, and nothing else could: `channel` is a sum, `balance` is a median
magnitude and a ramp about the ladder's centre has median nothing, `comb`'s
slope is over five semitones. It is broadband and not at the fundamental
because the same slope on `balance` is **not monotone in `width`** (−0.161,
−0.184, −0.286, −0.112, −0.075, −0.083 at widths 1.632 down to 0.1) — a gate on
a mechanism has to be monotone in it. The owner's number is a **schema rail**,
not a bar: `soundboard::MIC_WIDTH` ceilings `width` at 0.3 and `Preset::validate`
refuses more by name. Five of those columns are scored against a NEUTRAL image
since D466 — flat with pitch, cues in agreement, neither loudspeaker favoured —
and not against the recording's own image, which is a mic-placement accident
D417 measured and accepted as unscored (its C4: 16.86 dB into one capsule and
−949 µs with it). The targets moved; no bar did, every bar is still the
recording's own take-to-take floor, and all four columns stayed red. `balance`
and `splitting` are the **median magnitude** of the engine's own image over the
notes they score — a signed median passes a line that alternates ±20 dB —
`comb`'s slope is against flat, and `cue`'s bound is the head's own 660 µs at
**every** note (D469)**; **every window is
counted from an onset found in the 3 kHz-and-up band** — D452, because below
that a 1 ms envelope of a bass note is its carrier and the old detector landed
+73 ms past C4's own hammer on the engine and +42 on the recording;
**`loudness` is D456's, the first column on any board here that is a function
of a level, and the first whose verdict is a *seam* — the worst per-key
departure from the register's median error, where every balance verdict is
that median and cannot see a per-key error at all**), motion columns A1/A2/B1/B2
(FM axes), limiter budget, release-click, stability fuzz, perf (<50% of one
core; currently ~30%).

**Known gaps (D463: `#[ignore]`d tests, run with `cargo test -- --ignored`;
formerly "documented reds" — same inventory, but the default suite now runs
green and a failing test always means something is actually wrong). Six remain
and the numbering below keeps the closed ones' places. Four closed at D485-488
— the whole image half of this list — and three opened, all three of them one
number: the side energy the owner's verdict spent.**
1. *(closed, D481-484)* the duplex gap — **the oldest gap this repo had, and
   it is closed.** `a_known_duplex_comes_back_from_the_engines_own_render_of_it`
   is green and its `#[ignore]` is off, with
   `the_round_trip_reader_finds_nothing_when_the_duplex_is_silenced` beside it
   as the falsification it never had. What D260 got right was that a segment was
   normalised for a steady drive it never received; the drive it should receive
   is the **hammer's own force pulse** — a rear duplex is the same wire,
   continuous over the bridge. Measured (`forensics/duplex_drive`): at a segment
   52 cents off C5's fifth partial the pulse carries **+48.1 dB** more than the
   note's own bridge force, and across those 52 cents the pulse falls 0.7 dB
   where the bridge force falls 29.6. `DUPLEX_LEVEL_OFFSET_DB` **93.7 → 56.68**.
2. `each_loudspeaker_has_the_recordings_spectrum_where_the_mic_pair_acts` and
   3. `the_engines_stereo_image_is_the_recordings_in_every_band` — **one
   shortfall, read in two units, and since D485-488 it is the bill for the
   owner's verdict rather than the price of D418's rail.** Both are **re-barred
   at D486** to the neutral policy's own ceiling on the side energy: `r0 = 0` is
   `E_side = E_mid` is `|T| = 1`, which is D418's lift rail written in the
   coherence board's units — above it `|1 − T|` can vanish and one loudspeaker
   inverts against the other. The statistic and the bar are unmoved (the
   recording against its own second take, or the material's uncertainty); the
   **target** is `max(reference_r0, 0)` and `min(reference_pair_db, +3.0103 dB)`,
   both printed per band with the exclusion beside them before either gate
   asserts. Four of six bands are untouched. On the point D487 installs both go
   to **6 red**: `r0` +0.969/+0.981/+0.936/+0.843/+0.811/+0.868 against targets
   of +0.953/0/0/0/0/+0.041, and `pair_db` balances −0.16 to −2.95 — the pair
   carries **0.04-0.64 dB of side where the recording carries 2.8-3.9**. **Do
   not close either by moving a bar**; D418's frontier map still bounds them,
   D423 adds the lift axis, and
   `the_acceptance_still_fails_on_the_lobe_it_was_re_barred_against` and
   `the_recordings_own_line_is_the_image_the_neutral_policy_excludes` (which now
   asserts the recording is exactly zero against itself, exactly the exclusion
   against neutral, and passes iff the exclusion is inside its band's bar) keep
   both exclusions narrower than the defects.
4. *(closed, D485-488)* `no_note_of_the_line_arrives_from_two_places_at_once` —
   D451's `splitting`, **7.06 → 2.34 against a bar of 3.26**, its `#[ignore]`
   off. What closed it is `[voicing.mics.modal]` being **deleted** (D487): the
   band was the mechanism D451 convicted, its edges bracketed every fundamental
   of the tune and none of their overtones, and D461's corner measured absence
   at 33.44 bars against the *recording's* image — under D466's neutral target
   and D485's verdict the sign flips.
5. *(closed by deletion, D464)* the measured-preset beat census.
6. *(closed, D485-488)* `the_lines_pitches_come_out_of_the_loudspeaker_the_recordings_do`
   — D446's `balance`, **3.98 → 1.34 against a bar of 1.73**, its `#[ignore]`
   off. The band that was "not a widener but a pan" over 174.3-456.5 Hz is gone.
7. *(closed, D485-488)* `the_tunes_overtones_stay_where_the_recordings_do` —
   D459's `comb`, **slope 2.725 → 0.043 of 0.643 and swing 18.52 → 0.40 of
   5.15**, its `#[ignore]` off. Two mechanisms: D467's alternating polarization
   spread **retired** (D487) and the width rail (D485), which suppresses the
   geometric comb by a factor of five — measured, because *both* of `comb`'s
   falsifications stopped convicting at width 0.3 and now build their pair past
   the schema at `WIDTH_BEFORE_D485`.
8. *(closed, D485-488)* `the_lines_two_localisation_cues_agree_as_the_recordings_do`
   — D460/D469's `cue`, **worst note 1102 → 293 µs against the head's own 660,
   and `corr(ILD, ITD)` −0.539 → +0.169 where the policy asks only for
   positive**, its `#[ignore]` off. D471 said the lift was the only knob this
   column was monotone in; deleting the band is the limit of lowering the lift.
9. `the_engines_halo_is_as_loud_as_the_recordings_own` — D500-505, the treble
   sympathetic halo, 21.2 dB short.
10. `the_two_loudspeakers_play_this_line_as_the_recording_does` — the melody
   board's `channel` column, **new at D488 and it is items 2-3 in a third unit**.
   `10 log10((E_L + E_R) / 2 E_M)` over the recorded ladder reads **−3.82
   against a bar of 0.94** where it read −0.54 before the install. D470's
   arithmetic is why it is not a separate defect: `E_L + E_R = 2(1 + |T|²)|M|²`,
   so `channel` **is** a measurement of the side energy and so is `r@0`. **Do
   not close it by widening the pair** — D485's rail is an owner's verdict — and
   the mechanism that would close it without one is D470's named missing
   **incoherent early board field in 125-500 Hz, present from the first sample**,
   which does not exist.
11. `the_estimator_reads_back_a_spacing_the_engine_was_given` and
   12. `the_shipped_pair_is_visible_in_the_shipped_instruments_own_renders` —
   **new at D488, and neither is a statement about the piano.** The readback is
   **−87 / −89 / −92 %** against a 20 % tolerance where the same code read
   −4 / −0 / −9 on the preset D487 replaces, and the shipped geometry explains
   0.283 ms of its own renders' delays against a no-pair null of 0.080.
   `ENGINE_LAG_PER_ITD` is a calibration of the **tuner against a pair**,
   re-measured at D465 on presets whose geometric difference ran at `width`
   1.632; D485's rail is 0.3, so the constant is read on a pair a fifth as
   visible (and the retired spread was propping it up too — with the spread out
   and nothing else changed the readback is already −87 %). **Do not close it by
   widening the bar**: D465 named the fix and it is not a new constant — a
   forward model of what the estimator *reads*, predicted median lag as a
   function of spacing, aspect, width and the known band, inverted numerically.
   The geometry the fit writes is still inverted from the **recording's** own
   delays, and the delay residual on the installed point is **0.303 ms,
   unchanged and equal to the delay inversion's own best**.


## Conventions (hard rules for agents)

- Iterate with plain `cargo test` (dev profile is opt-level 3; release-only
  gates self-skip). Full `cargo test --release --workspace` at most twice per
  agent, at phase ends. Since D463 the suite is **green / 0 failed**:
  **771 passed / 0 failed / 9 ignored** at D488, against 735 / 0 / 9 before it.
  The remaining known gaps are `#[ignore]`d with their decision number in the
  reason string and `cargo test -- --ignored` runs the inventory on demand.
  **Six, and the inventory turned over at D485-488**: D446's `balance`, D451's
  `splitting`, D459's `comb` and D460's `cue` all went green and lost their
  attribute (the attribute coming off IS the close of the gap's decision item),
  and three opened — the melody board's `channel` and the tuner's own two
  spacing-readback gates — of which the first is D418's two coherence gates in a
  third unit and the other two are a calibration of the tuner against a pair,
  not a statement about the piano. A red test still always means something is
  actually wrong. Historical baselines: 735/0/9 at D484, 727/8 at D462,
  722/6 at D458, 712/4 at D446.
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

**The range's own queue** (D528-530). `concert-grand-d` and `upright-parlour`
ship with **five of the seven factory stages landed** — survey, sympathetic,
fit, tail, level — and two deliberately not forced. **`noise` refuses on both**
(`0 of 150` readings inverted on the grand, `0 of 225` on the upright: every
one rails, because these libraries' mechanism recordings sit far louder against
their own note than Salamander's −37 dB key-off group does), so `[noise.strike]`
is inherited from `presets/default.toml` and is **not** a measurement of either
piano. **And the same section had a second writer with no gate on it** (D531):
the *survey* wrote `[noise.key_off]` and `[noise.damper_lift]` from those same
implausible recordings, so `concert-grand-d` shipped a key-off table at −1 to
−9 dB with one anchor at **exactly 0.0**, the rail. `estimate::noise` now
screens every mechanism reading against `MAX_MECHANISM_LEVEL_DB` (**−21.0 dB**
against the group's own note: Salamander's own 88 readings run −39.0 to −24.64,
Askenfelt's structure-borne path is ~40 dB under the partials, and the gate is
the hottest genuine reading plus three take-to-take sigmas) and refuses a group
whose plausible readings are not a strict majority. A refused table is **not
written** — the preset inherits `default.toml`'s and says so in its
`description`. The grand's whole `[noise]` section is now absent from the file;
the upright loses its `[noise.pedal_up]` (−16.71 dB, 1 of 4 takes plausible)
and keeps its pedal-down (4 of 4). Salamander is unmoved and it is *proved*
unmoved: `salamanders_own_mechanism_is_written_bit_identically`
(`tuner/tests/noise.rs`) re-measures it off the corpus and asserts the four
tables field for field against the shipped preset. `piano-tuner noise --stage
mechanism` is the re-entrant way to run that stage on a finished preset without
the whole factory. **`mics` cannot move on either** — 115.757 bars out at the seed and at
every step down to 0.0039, because `presets/default.toml` carries no
`[voicing.mics]` section and a fit whose knobs are out of the signal path has a
flat gradient — and seeding one has to wait for the stereo-install track's
`width` rail to settle. Both are why `bench`'s STEREO columns are red in six
bands of six on the grand: **one un-run stage, seen eleven times.**

**Two per-library hazards a successor must not trip over.** (i) **A two-layer
library's take floor is not a take floor.** Every bar here is the reference
played out of its *neighbouring velocity layer*; on sixteen layers that is a
second take a decibel away, on two it is the other dynamic, measured 10.5 dB
away. The upright's `bench` mel floors are 3.34-7.90 dB against the grand's
1.10-2.14, **two** of its six phrases score *below* their own floor
(`alberti_fast` 5.68 of 5.92 and `arpeggio_dynamics` 6.91 of 7.90; `scale_mf`
is the third-closest and is *above* its floor, by 0.14 dB — corrected at
D531), and its
melody board shows four reds to the grand's ten mostly because its bars are
three to ten times wider. (ii) **`estimate::level::MAX_LEVEL_DB` is 6.40 dB and
that number is Salamander's**, not the library's — the one borrowed bar left in
the factory, and it binds: the grand's C7 renders 27.5 dB under its own
recording, the cap turns the fit into a no-op, and the worst recorded key goes
12.09 → 12.68 dB where Salamander's goes 8.96 → 1.85. The fix is to measure the
cap on the library being fitted; it needs its own item because the measurement
must reproduce 6.40 exactly for Salamander or it silently re-bars the shipped
preset.


**The owner ruled on the stereo image, the neutral point is installed, and what
it cost is one number** (D485-488, on D446-472). The verdict, on a width ladder
rendered from the neutral base: *"It should be 0.3 or less for sure. This effect
shouldn't be dominating at all."* The effect is the per-key bass-left/
treble-right lean; `voicing.mics.width` is the gain on it; it shipped at 1.632.
Three things landed and the third is the install. **(a) The gate**: `melody`'s
eleventh column `gradient`, the Theil-Sen slope of the note's whole broadband
`10 log10(E_L/E_R)` against key over the recorded **ladder** — no column read the
compass, only the tune — with the owner's number as a **schema rail**
(`MIC_WIDTH` 2.0 → 0.3, `Preset::validate` refusing more by name, `Knob::Width`
bounded there) rather than as a bar. **(b) The re-bar** (D486): the coherence
boards stop asking for more difference than sum — `r0 = 0` and
`pair_db = +3.01 dB` are D418's own lift rail in those boards' units — with the
exclusion printed per band and the control asserting both halves. **(c) The
install** (D487): `polarization_pan_spread` **retired by its own fit stage**
(`sympathetic --only pan-spread`, which still renders, measures and prints the
drift and writes the null; 2.86 dB of drift given up, and D467's mic-stage gain
trim is what buys it back), `[voicing.mics.modal]` **deleted**, `width` 0.3,
`diffuse_coherence` 4.099 → 0.739, `source_extent_m` 0.161.

**What it bought and what it cost, in one table** (D488). Bought: `balance`
3.98 → **1.34** (bar 1.73), `splitting` 7.06 → **2.34** (3.26), `comb` slope
2.725 → **0.043** (0.643) and swing 18.52 → **0.40** (5.15), `cue` 1102 →
**293 µs** (660) and corr −0.539 → **+0.169**, `gradient` −0.580 → **−0.119**
(0.490) — **four standing known gaps closed at once**, and the per-channel board
3 red / 3.49 bars → 1 red / 2.65. Cost: `channel` −0.54 → **−3.82** (0.94), the
thirty recorded keys' coherence 2 red / 5.16 → **6 red / 49.08**, the phrases
4 red / 15.84 → 6 red / 68.65, and the tuner's own spacing readback −4/−0/−9 %
→ **−87/−89/−92 %**. **It is one number, not seven gates**: D470's
`E_L + E_R = 2(1 + |T|²)|M|²` makes `channel` and `r@0` both measurements of the
side energy, and the pair now carries 0.04-0.64 dB of side where the recording
carries 2.8-3.9. Mono fold-down 0.0000 dB worst band; perf 28.4 % of one core;
delay residual 0.303 ms, unchanged.

**Two things a successor should not have to re-derive, and one warning.** **(i)
The constrained fit is degenerate under a neutral target and D470's theorem is
why**: with `width` railed, every coherence band is red at every candidate, so
the ten-bar penalty is a constant that ranks nothing and the only term with a
gradient is an image whose target is zero — which deleting the pair satisfies.
The band stage walked `width` to its 0.05 search floor; the point that ships was
chosen on a printed four-point grid at the rail (the fit's own band, D418's
band, a low-lift band, and **no band**, which is 1 red melody column against 4).
`mics --stage extent` did the same thing on its own axis — 1.500 m, its search
bound, at which `doubling_the_spacing_doubles_the_delay_the_renders_carry` reads
1.39x instead of 2x — and was refused on that gate, so the extent that ships is
D468's own compass answer. **(ii) `ENGINE_LAG_PER_ITD` is a calibration of the
*tuner against a pair*** and it was propped up by two things this milestone
removed: the width and the pan spread. D465 named the fix — a forward model of
what the estimator reads, inverted numerically — and it is what retires gaps 11
and 12. **(iii) The warning**: the boards say this image is neutral and they
cannot say it is *there*. The one check no gate here performs is a listening
pass on the installed instrument by the person who gave the verdict.

**Why that price is not a bad trade but the same budget twice** (D470):
`L = M(1+T)`, `R = M(1−T)`, mono discipline *is* `L + R = 2M`, so `T` is the
whole design freedom and `E_L + E_R = 2(1 + |T|²)|M|²` — `channel` and `r@0`
**are** measurements of the side energy, and every image fix is paid out of
D418's two known gaps. Three refusals bound it, and the first is a theorem: a
mono-exact, zero-latency pair with no interchannel phase is a **pan-pot**
(causality forces it), the linear-phase version costs **21-43 ms** of latency,
and for a purely imaginary `T` a 660 µs per-note bound caps the reachable
coherence at `cos(2π f τ)` — **+0.47 at C4** against a target of −0.226. The
recording's 125-500 Hz coherence and a head-sized ITD bound are arithmetically
incompatible for any mono-exact **coherent** pair, so the named missing
mechanism is an **incoherent early board field in 125-500 Hz present from the
first sample**, and it does not exist. D485-488 is the decision to spend that
budget rather than the mechanism that would have made it free.

**The rest of the stereo line is closed** (D417-425). **Read what follows as
history: the band it describes is deleted and the width it quotes is illegal
since D485-488** — what survives is the account of what a nodal-line lobe does,
which is why it is kept. The stopgap shipped: the lift was
railed at the null, `presets/salamander-c5.toml`'s `[voicing.mics]` was refitted
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

**Treble sympathetic halo: it now has a column, the column is red by 21.2 dB,
and the fit that would close it has been measured and cannot** (D500-506, and
known gap 9 above). It is not the duplex (D484) and it is not the coupling
either: at the ceiling `Preset::validate` certifies, the coupling buys 0.85 dB.
The census D484 quoted **cannot see the mechanism at all** — removing the bus
and the segments moves it 0.08 dB at C6 — so `salamander_targets` is re-decided
(D501) onto the halo the library recorded alone, with `rt_decay` paid, and the
fit's objective and the gate's verdict are now one quantity. `presets/` and
`engine/` are untouched by that milestone. Also from it: per-key brightness
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
