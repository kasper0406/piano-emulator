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
  subcommands of `piano-tuner`: `track/estimate`, `survey`, `fit --stage`,
  `sympathetic` (`--only duplex` runs its first stage alone), `tail`, `noise`,
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
  a perfectly good statistic pointed at the wrong answer. Four image columns
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

`bench` (REALISM.md: mel vs floor, modulation, attack, release, stereo
coherence + per-channel columns), `compass` (88 keys vs strung-alike
neighbors + recordings), `melody` (the Ode line: roughness/wobble/hf/strike
/channel/**balance**/`splitting`/**`comb`**/**`cue`**/**`loudness`**, head+tail windows,
recorded-key bars; **four of those columns are scored against a NEUTRAL image
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
and the numbering below keeps the closed ones' places:**
1. *(closed, D481-484)* the duplex gap — **the oldest gap this repo had, and
   it is closed.** `a_known_duplex_comes_back_from_the_engines_own_render_of_it`
   is green and its `#[ignore]` is off, with
   `the_round_trip_reader_finds_nothing_when_the_duplex_is_silenced` beside it
   as the falsification it never had. What D260 got right was that a segment was
   normalised for a steady drive it never received; what it did not say is which
   drive it *should* receive, and the answer is the **hammer's own force pulse**
   — a rear duplex is the same wire, continuous over the bridge, so what
   launches it is the travelling knock and not the line spectrum the speaking
   length settles into. Measured (`forensics/duplex_drive`): at a segment 52
   cents off C5's fifth partial the pulse carries **+48.1 dB** more than the
   note's own bridge force, and across those 52 cents the pulse falls 0.7 dB
   where the bridge force falls 29.6. `gain_db` follows, as the same impulse
   normalisation `string.rs` uses for the note's own partials — *how hard this
   segment answers the knock, relative to the key's own speaking length* — so
   level and length are separate measurements at last (under the old convention
   a segment asked to ring twice as long came out 6 dB quieter).
   `DUPLEX_LEVEL_OFFSET_DB` **93.7 → 56.68** and the 57 that remain are a
   stated convention. On one held-and-released C5 the segment comes back at
   **+0.00 cents** and rings **1.40 s of the 1.4 s** it was given, against
   −1.84 cents and 0.18 s before.
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
4. `no_note_of_the_line_arrives_from_two_places_at_once` — the melody board's
   `splitting` column (D451), red since it was written and **still red after
   D461's refit**. `image(f1) − Σ w_k image(f_k) / Σ w_k` over partials 2-4,
   median over the *line's own five pitches* rather than over the ladder,
   because the defect is a band and the ladder spans three regimes of its
   edges. It is the one column of the image that no point of D462's frontier
   brings inside its bar without emptying the pair: it reads **+7.39** on the
   instrument that shipped and **+7.94** after the refit, against a bar of
   3.26, and its two neighbours behave oppositely across the same frontier —
   widening the pair takes `splitting` *green* while taking `comb` worse
   (`the_comb_gate_fails_on_a_pair_that_stands_twice_as_wide`), which is the
   arithmetic of `splitting = balance − comb` and not a coincidence.
5. *(closed by deletion, D464)* the measured-preset beat census — a histogram
   edge rather than a defect (D458's 5-cells-over-bar reading reshuffled across
   refits), deleted by the owner's direction. The construction-level census
   (`string::tests::no_beat_rate_is_shared_across_the_compass`) still gates the
   metronome; `forensics/beat_census` remains the instrument if a listener ever
   reports a coherent pulse.
6. `the_lines_pitches_come_out_of_the_loudspeaker_the_recordings_do` — the
   melody board's `balance` column (D446-448), **re-barred to the neutral
   target at D466 and still red**: the median *magnitude* of the engine's own
   image over the recorded ladder is **3.98 dB against a bar of 1.73** (it read
   +8.61 against the recording's own image before the re-bar, of 1.94). The
   statistic is `10 log10(E_L / E_R)` at each note's own
   fundamental, and `channel` on the same renders is −0.54 against 0.94 and
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
7. `the_tunes_overtones_stay_where_the_recordings_do` — the melody board's
   `comb` column, **D459, red on the instrument that ships, its slope re-barred
   against flat at D466 (2.725 of 0.643, swing 18.52 of 5.15), and now with a
   named dominant term and a mechanism that reaches both bars** — D467's
   alternating polarization spread (zeroing it alone: slope 2.725 → 1.107,
   swing 18.52 → 9.09) and D468's line source (`source_extent_m`, built, fitted
   by `mics --stage extent`, **not installed**: with the spread zeroed it reads
   slope 0.191 and swing 2.15, both green, and costs 20 bars of the recorded
   keys' coherence board). It is the
   fourth unscored dimension of the image. It is the energy-weighted mean of
   `10 log10(E_L/E_R)` over each note's own partials **2-4** — where the note's
   *colour* sits, as against `balance`'s *pitch* — and the reason nothing read
   it is one line of arithmetic: **`splitting` is `balance − comb` exactly**, so
   a mechanism that moves a note's fundamental and that note's own overtones
   together and moves the next note's somewhere else cancels out of `splitting`
   and never enters `balance`. The **pair geometry** is exactly such a
   mechanism. Its verdict is the line's **slope** and its **swing** and never a
   median or a residual, because the defect is a *ramp* and a median cancels a
   ramp: shipped it reads **−2.724 dB per semitone against the recording's
   +0.504** (error 3.228 of a 0.643 bar) and **18.52 dB of swing against 4.11**
   (bar 5.15), with the tune's colour crossing the whole image between E4 and
   F4. **Do not close it by moving a bar** and do not expect a geometry to close
   it: D462's two frontier tables are the map, and the only point that brings it
   inside its bars is `width` 0.1 — the side signal deleted — which takes the
   recorded keys' gate from 2 red bands to 6 and the phrases' from 4 to 6. The
   one direction left is D448(d)'s per-channel gain.
8. `the_lines_two_localisation_cues_agree_as_the_recordings_do` — the melody
   board's `cue` column, **D460, red on the instrument that ships and red on
   both halves since D469's re-bar** (the bound is the head's own 660 µs at
   every note, with the recording's anomalous C4 no longer inflating it to
   1186: the engine reads **1102 µs at C4**, where it used to pass by 7 %; and
   the agreement half is now `corr(ILD, ITD) > 0`, gated on the line's own ILD
   swing clearing the take floor, and the engine reads **−0.539**, cues on
   opposite sides). It is
   the fifth unscored dimension of the image, and the **first column on any
   board here that is a function of an interchannel phase**. It is the
   interchannel time at each note's own fundamental, read off the phase of the
   same heterodyne `balance` reads a level with; positive means the left channel
   leads. Two verdicts. The **bound** is physics — a head is 0.18 m across, so
   nothing in a room hands the ears more than about 660 µs — and it *passes*, by
   7 %, and only because the bar is the larger of that and the recording's own
   worst note, which is the anomalous C4 D448(ii) measured at 16.86 dB and which
   carries **−949 µs** of its own. The **agreement** is what fails: `corr(ILD,
   ITD)` over the line reads **−0.539 where the recording reads +0.831** and its
   own second take +0.825, short by **1.369 against a bar of 0.193** — the
   engine's time cue runs the opposite way down the tune from the recording's
   while its level cue runs the same way, so the ear is handed a note whose two
   halves are on two different sides. Neither cue alone reports it: `balance` is
   the level, this is the time, and only their product is wrong. **Do not close
   it by moving a bar**; D462's refusal is the map of what closing it costs
   today (the refit that took it to +0.193 of 0.193 broke the estimator's own
   spacing readback at −25 %).


## Conventions (hard rules for agents)

- Iterate with plain `cargo test` (dev profile is opt-level 3; release-only
  gates self-skip). Full `cargo test --release --workspace` at most twice per
  agent, at phase ends. Since D463 the suite is **green / 0 failed** — the
  remaining known gaps (D418's two, D446's `balance`, D451's `splitting` and
  D459/D460's `comb` and `cue`, two new columns rather than two new defects:
  the instrument did not move, the board learned to read two more things about
  it — and since D466 the last four carry their **re-barred** distance in the
  reason string, against the neutral target rather than against the recording's
  own image; none of them turned green under the re-bar, so none of them lost
  its attribute) are `#[ignore]`d with their decision number in the reason
  string, and
  `cargo test -- --ignored` runs the gap inventory on demand. **Six, not
  eight**: D458's census was closed by deletion (D464) and D260's duplex by its
  mechanism landing (D481-484), so
  `a_known_duplex_comes_back_from_the_engines_own_render_of_it` and its
  falsification `the_round_trip_reader_finds_nothing_when_the_duplex_is_silenced`
  now run in the default release suite. A red test now
  always means something is actually wrong. Historical red-suite baselines:
  727/8 at D462, 722/6 at D458, 712/4 at D446.
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

**The stereo image is scored against a neutral target, the estimator's wall is
gone, and the one mechanism that reaches the bars is built and not installed**
(D446-448, D451, D459-472). `melody`'s `balance`, `splitting`, `comb` and `cue`
are four readings of one pair of channels — the fundamental's place, that place
against the note's own overtones', the overtones' own ramp across the tune, and
the interchannel **time**, the only column on any board here that is a function
of a phase — and `splitting = balance − comb` exactly. All four are red on the
instrument that ships and **all four are now scored against a NEUTRAL image**
(D466): flat with pitch, cues in agreement, neither loudspeaker favoured. That
is a re-bar of three targets plus `cue`'s bound and of **no bar** — every bar is
still the recording's own take-to-take floor — and it closed nothing: 3.98 of
1.73, 7.06 of 3.26, slope 2.725 of 0.643 with swing 18.52 of 5.15, corr −0.539
where the policy asks only for positive and 1102 µs at C4 against the head's own
660. What it removed is a target that was itself a defect: the recording's image
is D417's microphone stand (its C4 16.86 dB into one capsule and −949 µs with
it, its ladder median −5.73 dB), and the control test now asserts **both** halves
— zero against the recording's own image, and *failing* against neutral by 5.69
dB and 949 µs, which is the size of the exclusion written down.

**Three things a successor should not have to re-derive.** **(i) The eighteen
hertz are gone** (D465). The lobe's own interchannel phase is computable from
the preset, `estimate::mics` subtracts it before the delay vote, and
`ENGINE_LAG_PER_ITD` — which carried the shipped band's group delay inside it —
is re-measured at **1.36**: the implied constant moves ±11 % with a band's
*width* raw and ±4 % corrected, the band D462 refused reads +3/+4/+2 % where it
read −24/−8/−11, and `Knob::ModalHi`'s floor is back at 200 Hz. **(ii) The
dominant term of `comb` is `polarization_pan_spread`** (D467): it buys a
*directivity* with a *position* (C4's two polarizations at pan −0.42 and +0.30,
the sign alternating with key parity — which is exactly why the tune's colour
crosses the image between E4 and F4), and zeroing it alone takes the slope
2.725 → 1.107 and the swing 18.52 → 9.09. The replacement to build is a
per-polarization interchannel **gain** trim at the mic stage: same drift, ~3 dB
of image instead of 11. **(iii) The source is not a point** (D468).
`[voicing.mics].source_extent_m` averages the two capsule pressures over a line
metres long, absent-means-old, `Mics::taps` only, tabulated at preset load
(25.4 % of one core against the point source's 25.2, where the naive quadrature
costs 33.6), mono-exact by construction, no added latency, with its own fit
stage (`mics --stage extent`). With the spread zeroed it takes **all four image
columns inside their bars at once** — slope 0.224, swing 1.66, worst 146 µs,
corr +0.437, balance 1.71, splitting 0.43 — and **it is not installed**, because
the price is 20 bars on the thirty recorded keys' coherence board, 31 on the six
phrases, and `channel` −0.54 → −2.49 against 0.94.

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
first sample**, and it does not exist. The lift is the one knob `cue` is
monotone in (D471: 590 µs and corr +0.68 at 0.20 against 1102 and −0.54 at
0.99) and its rail stays at the inversion boundary because moving it to the
image bar's 0.25 would make the shipped preset illegal and commit the next fit
to 15.7 bars nobody has agreed to.

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

Treble sympathetic halo still short and **it is not the duplex** (D484: on the
between-partial census the segments narrow C6's 29.5 dB shortfall by 0.08 dB and
C7's by nothing — a census is a floor across a band and six segments are six
lines, so whatever fills that floor is broadband and per-note); per-key brightness
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
