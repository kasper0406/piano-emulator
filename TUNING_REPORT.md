# TUNING_REPORT.md — Phase E: measured-vs-model residuals

What `TUNING.md`'s stage 1 could **not** fit on the Salamander Grand Piano (Yamaha C5, 30 sampled
keys × 16 velocity layers), why, and what it would cost to fix each one. Every number here is a
measurement taken by `tuner/examples/residuals.rs` over the 450 analysed recordings and over the
engine's own renders of the same notes; nothing is estimated by eye or taken from the literature.

```
cargo run --release --example residuals            # ~2.5 min against the warm cache
```

---

## Executive summary

Stage 1 fits the model well where the model is right, and the places it cannot fit are specific,
measurable and — for the most part — cheap to fix.

**The control makes the claims falsifiable.** Every measurement was repeated on notes the engine
rendered from `presets/salamander-c5.toml`, where the model is true by construction. The estimator
returns 0.06–0.6 cents of inharmonic residual on those renders against 1–16 cents on the
recordings, so the frequency findings are the piano's. It returns 0.8–4.9 dB of envelope residual on
the renders against 2.3–3.5 dB on the recordings — more on the synthetic material than on the piano
at four of the eight keys — so the envelope findings are *not*: that metric measures the estimator,
and it is reported as unusable rather than as evidence.

**Ranked by audibility, what the model is missing:**

1. **The whole instrument rings and the engine's does not — in the treble it dominates.** One
   second after a fortissimo C7, the energy *between* the struck note's partials is 3.5 dB below
   the energy *in* them (soft layer: 16 dB below); the engine's render of the same note is 48 dB
   below. At C6 it is −22 dB against −48. The peaks responsible repeat at the same frequencies
   across all velocity layers (C7: 2067, 1603, 1650, 1691 Hz), so they are resonances of the
   instrument, not noise. Salamander's own string-resonance samples put the same halo at −31 to
   −43 dB relative to a mezzo-forte strike. It is the mechanism behind the level difference
   `DECISIONS.md` 96 measured one second after a C7 strike (source −61 dB, estimated preset −65,
   hand-tuned default −87), and it is what forced the recorded-floor detector of item 89.
2. **Nothing models the action.** The key-off recordings play at −25 to −39 dB relative to a
   velocity-90 strike of the same key, last 165–285 ms and are centred at 143–261 Hz; the pedal-down
   sample plays at −36 dB with a six-second 70 Hz rumble, pedal-up at −42 dB over 0.3 s. The engine
   makes no sound at all at a note-off or a pedal move — only a change of damping. Cheapest fix on
   this list, and the most obviously missing sound in playing.
3. **A wound bass string does not have one inharmonicity coefficient.** At A0, `B` fitted to
   partials 1–8 is 3.07e-4 and to partials 14–26 is 2.30e-4 — a ratio of 0.75; C1 gives 0.66, D#1
   0.63. Against the coefficient the preset actually writes, A0's measured partials stand up to
   **78 cents** off (15.7 cents RMS), C1 36, D#1 41. Between F#1 and C2 the ratio inverts (1.24–1.45)
   and from D#2 to C6 the two-parameter law holds to 1–5 cents.
4. **The stereo balance of a note moves while it decays; the engine's cannot move at all.** Median
   per-partial left-minus-right drift between 0.3 s and 2 s: 1.2–6.2 dB on the recordings, 0.02–0.14
   dB on the engine's renders of the same notes.
5. **The excitation spectrum is 5–10 dB rougher than any smooth envelope × `sin(kπx)`** (engine
   control: 2–5 dB) — and the roughness is *not* shared between notes at the same frequency, so the
   cheap fix (one global bridge-admittance curve) is refuted by measurement before it was tried.
6. **Phantom partials are real, quadratic, and 60–95 dB down.** At combination frequencies that
   stand clear of every transverse partial, the recordings carry energy 3–21 dB above the local
   floor growing as the *square* of the note's level (slope 1.8–2.6 dB per dB); the engine's renders
   sit 13–28 dB *below* their local floor there. Confirmed — and last on the list, because −60 dB.

**Refuted:** pitch glide. Beyond 200 ms the fundamental holds to a few cents, and inside 100 ms the
measurement is worthless (the engine's control, which cannot glide, returns −17 to +108 cents at
50 ms). The fixed-pole model is not measurably wrong here. **Also refuted:** a missing attack
transient — broadband energy between the partials during the first 85 ms is within ~7 dB of the
engine's.

**Cost:** items 2, 3, 4 and the unison-decay finding (§6) are each 1–3 days, need no new fitting
machinery, and cost nothing per sample. Item 1 is stage-2 work by construction (isolated notes
cannot separate coupling from radiation). Items 5 and 6 are the expensive ones and the measurements
argue for deferring both.

---

## Method, and why the control matters

Two families of measurement:

* **Trajectory-domain** (all 480 recordings, through the survey's trajectory cache): per-partial
  deviation from the fitted stiff-string law, per-partial frequency slide against its own decay,
  envelope-fit residual, and scatter of the time-zero amplitudes around the strike comb.
* **Audio-domain** (8 keys × 3 layers, plus every layer of 6 keys for the phantom test): a spectrum
  census of one sustained frame, the broadband energy between the partials, per-partial stereo
  balance, and the level/decay/colour of the mechanism recordings.

Every one of them is run a second time on notes the engine renders from the estimated preset. That
is the whole design: an estimator's residual on material the model generated is the estimator's
noise floor, and only the excess over it is evidence about the piano. Three of the six candidate
findings changed status because of the control — two survived it, one (the envelope residual) did
not, and one (pitch glide) was refuted outright.

New code: `tuner/src/residual.rs` (the measurements, 10 unit tests) and
`tuner/examples/residuals.rs` (the driver). Both read the same cached trajectories the survey used,
so nothing here re-derives a number Phase D already measured.

---

## 1. Frequencies: one `B` per note is not enough in the bottom octave and a half

The engine lays a string's partials out as `f_k = k f0 sqrt(1 + B k²)` with one `B` per note. To
test the law rather than the fit, `B` was fitted twice per recording — once to partials 1–8 alone
and once to partials 14–26 alone — and the two compared. Over a narrow band of `k` the pair
`(f0, B)` is correlated, so neither number means much alone; what the two bands compare is the
*curvature* of the measured series, which is exactly what one `B` has to reproduce.

Median over the 16 layers of each key. "preset RMS/worst" is the deviation of the measured partials
from the layout `presets/salamander-c5.toml` actually writes for that key — what the engine will
put on the bridge, against what was recorded.

| key | note | partials | preset RMS | worst (k) | B (k≤8) | B (14≤k≤26) | ratio |
|---:|:--|---:|---:|---:|---:|---:|---:|
| 21 | A0 | 56 | 15.7 c | 78.6 c (1) | 3.07e-4 | 2.30e-4 | **0.75** |
| 24 | C1 | 48 | 13.1 c | 35.7 c (26) | 2.64e-4 | 1.76e-4 | **0.66** |
| 27 | D#1 | 51 | 15.4 c | 40.9 c (26) | 2.36e-4 | 1.48e-4 | **0.63** |
| 30 | F#1 | 49 | 6.2 c | 28.4 c (1) | 7.56e-5 | 9.36e-5 | **1.24** |
| 33 | A1 | 45 | 11.5 c | 50.4 c (21) | 6.71e-5 | 9.40e-5 | **1.40** |
| 36 | C2 | 36 | 5.5 c | 32.5 c (25) | 6.64e-5 | 9.62e-5 | **1.45** |
| 39 | D#2 | 36 | 3.6 c | 13.5 c (36) | 7.76e-5 | 7.55e-5 | 0.97 |
| 42 | F#2 | 28 | 1.6 c | 5.1 c (29) | 6.73e-5 | 7.28e-5 | 1.08 |
| 45 | A2 | 27 | 2.7 c | 9.3 c (24) | 7.86e-5 | 7.61e-5 | 0.97 |
| 48 | C3 | 24 | 1.1 c | 4.5 c (1) | 1.20e-4 | 1.12e-4 | 0.93 |
| 54 | F#3 | 22 | 5.1 c | 18.5 c (17) | 1.69e-4 | 1.67e-4 | 0.99 |
| 60 | C4 | 16 | 1.4 c | 4.8 c (15) | 2.93e-4 | — | — |
| 72 | C5 | 11 | 4.2 c | 11.6 c (10) | 8.05e-4 | — | — |
| 84 | C6 | 6 | 2.5 c | 4.4 c (6) | 2.70e-3 | — | — |
| 96 | C7 | 3 | 9.3 c | 16.0 c (3) | — | — | — |
| 105 | A7 | 2 | 18.5 c | 27.8 c (2) | — | — | — |
| 108 | C8 | 2 | 36.5 c | 50.1 c (1) | — | — | — |

Control, same estimator, engine renders: **0.06 c** RMS at A0, 0.08 at A1, 0.26 at A2, 0.06 at
A3/C4, 0.10 at C5, 0.09 at C6, 0.60 at C7. The measured deviations are the piano's by two orders of
magnitude.

> **Update (Milestone A).** Most of the bass rows of this table are the *tracker*, not the
> piano. Above partial 24–25 at A0, C1 and D#1 the partial tracker loses one partial and every
> index above it names the partial above itself: A0's spectrum has peaks at 696.7 and 729.5 Hz
> where tracks 24 and 25 report 696.0 and 750.8, and from track 26 up the numbering is one out.
> Fitting that top is what gave those keys the coefficient the preset wrote, and with it the
> 78-cent fundamental. `estimate::inharmonic::trusted_prefix` now truncates the series at the
> first skip; over the partials that survive it, the layout the preset writes stands 8.8 c (A0),
> 6.3 c (C1) and 4.6 c (D#1) from the recordings, and the two-band diagnostic below is reproduced
> by the estimator (0.80/0.68/0.65 against 0.75/0.66/0.63) where before it read 1.01–1.03. See
> `DECISIONS.md` 131–134, 140–141. The `ratio` column, which was computed over partials 14–26,
> is *not* affected — those indices are correct — and it is the column that turned out to be
> right.

Three separate things are in that table.

* **Wound bass strings (A0–D#1): `B` falls 25–37 % along the series.** Refitting A0's partials 2–24
  with a single `B` brings the residual back to ±4 cents, so the *shape* is describable — it is the
  single coefficient anchored on the strong low partials that misplaces the high ones. Partials 14–26
  of A0 sit at 390–760 Hz, where the ear resolves them individually.
* **The low tenor (F#1–C2): the ratio inverts, 1.24–1.45.** These are the shortest wound strings;
  their high partials are *sharper* than a constant `B` predicts. Same size of error (worst 28–50
  cents), opposite sign — so a fix has to be a signed second coefficient, not a one-sided
  correction.
* **The top octave is an interpolation error, not a physics error.** A7's 18.5 c and C8's 36.5 c are
  the price of `DECISIONS.md` 92: C8 was refused, so the compass curve reaches three keys past its
  last measurement. C7's 9.3 c preset residual against its own 3.8 c fit is the same effect one
  octave down.

## 2. Envelopes: the metric measures the estimator, not the piano

| | A0 | A1 | A2 | A3 | C4 | C5 | C6 | C7 |
|:--|---:|---:|---:|---:|---:|---:|---:|---:|
| envelope residual, recordings | 2.31 | 2.88 | 3.37 | 3.42 | 3.03 | 3.18 | 3.08 | 2.42 |
| envelope residual, engine renders | 0.81 | 2.46 | 4.87 | 4.42 | 4.14 | 4.08 | 3.83 | 2.39 |
| late trend, recordings | +0.95 | +1.43 | +1.89 | +1.48 | +0.71 | +0.13 | +1.04 | +0.19 |
| late trend, engine renders | +0.10 | +1.48 | +3.22 | +5.10 | +4.24 | +0.63 | −0.22 | −0.25 |

(dB; the trend is measured-minus-modelled level at the end of the fitted span.)

The engine's own renders leave as much residual as the piano does, and at A2–C4 more. The
two-exponential-plus-two-beats envelope model (`DECISIONS.md` 81, 82, 84) simply does not describe a
three-string unison's envelope to better than ~4 dB, whatever produced it — a three-string group
beats at three rates and the model fits one.

**Consequence for stage 2:** a loss function built on per-partial envelope error would spend most of
its gradient on this. Use the prompt decay rate (`DECISIONS.md` 90) and the beat *rate*, not the
envelope residual.

## 3. Excitation spectra: rough, and rough per note

The engine's input gain for mode `k` is one smooth per-note scale times `sin(k π x)`. Scatter of the
measured time-zero amplitudes around that (after fitting both) is unreachable at any parameter
setting:

| | A0 | C1 | A1 | C2 | A2 | C3 | C4 | C5 | C6 |
|:--|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| recordings, RMS | 9.8 | 6.7 | 7.6 | 8.1 | 5.9 | 5.1 | 6.8 | 6.2 | 2.8 |
| recordings, worst partial | 28.8 | 23.1 | 21.8 | 23.7 | 14.0 | 11.9 | 16.6 | 12.3 | 4.9 |
| engine renders, RMS | 4.9 | — | 2.2 | — | 2.0 | — | 3.2 | 2.1 | — |

(dB.) Roughly twice the control in dB, and the excess grows toward the bass. It also grows with
velocity (A0: 7.9 dB at the softest layer, 11.3 dB at the loudest), which is what a hammer that
flattens against the string does to the high partials.

**Where does the roughness live?** If it were the bridge and the soundboard, it would be a function
of frequency and shared by neighbouring notes — one admittance curve for the whole instrument, a
table with no per-sample cost. Every partial of every note was therefore binned in thirds of an
octave and the mean across notes compared with the spread across notes:

| band | notes | mean across notes | spread across notes |
|:--|---:|---:|---:|
| 307–376 Hz | 13 | −0.36 dB | 3.19 dB |
| 376–461 Hz | 11 | +0.07 dB | 3.51 dB |
| 461–565 Hz | 14 | −1.54 dB | 6.62 dB |
| 565–693 Hz | 14 | −0.16 dB | 5.10 dB |
| 693–849 Hz | 15 | −1.40 dB | 2.59 dB |
| 849–1042 Hz | 16 | +0.62 dB | 4.07 dB |
| 1042–1277 Hz | 19 | −1.70 dB | 3.20 dB |
| 1277–1566 Hz | 18 | +0.69 dB | 2.69 dB |
| 1566–1920 Hz | 20 | +1.06 dB | 2.68 dB |
| 1920–2354 Hz | 20 | +0.39 dB | 4.56 dB |
| 2354–2887 Hz | 20 | −0.13 dB | 2.97 dB |
| 2887–3539 Hz | 21 | +0.82 dB | 3.48 dB |
| 3539–4340 Hz | 21 | −0.48 dB | 3.12 dB |
| 4340–5321 Hz | 18 | −1.81 dB | 3.58 dB |
| 5321–6525 Hz | 15 | −0.68 dB | 2.11 dB |
| 6525–8000 Hz | 14 | −1.21 dB | 3.21 dB |

Above 300 Hz the notes agree on nothing: the shared part is within ±1.8 dB of zero in every band
while the spread between notes is 2.1–6.6 dB. (Below 300 Hz the bins hold only 2–9 notes, each
contributing its own low partials, so "shared" and "per-note" are not separable there, and the means
there are large: +14 dB at 74–90 Hz, +8.7 at 90–111, −5.6 at 204–250, with spreads of 4–8 dB.) The
strike fit's smooth envelope absorbs any *smooth* frequency trend by construction, so what this
rules out is precisely a non-smooth global admittance curve — the cheap fix. What is left is per-note, per-partial, and
`DECISIONS.md` 93's warning applies to measuring it: a close microphone's own comb is inseparable
from the string's.

## 4. The spectrum census: what radiates that is not the struck string

One sustained frame (0.5 s after the strike), every peak classified against the note's measured
transverse partials, against sums and differences of them, and against the other keys' pitches.
Levels are relative to the loudest transverse partial in the same frame. "between" is the broadband
energy between the partials on an 85 ms window, at the strike and one second later.

| source | key | vel | peaks | transverse | unexplained | loudest | between @0 | between @1 s |
|:--|---:|---:|---:|---:|---:|---:|---:|---:|
| salamander | A0 | 40 | 182 | 129 | 31 | −33.7 | −49.7 | −55.9 |
| salamander | A0 | 108 | 300 | 229 | 43 | −27.1 | −44.0 | −56.6 |
| engine | A0 | 90 | 168 | 130 | 13 | −49.1 | −54.2 | −67.2 |
| salamander | C4 | 108 | 51 | 34 | 13 | −47.9 | −23.5 | −44.3 |
| engine | C4 | 90 | 49 | 22 | 18 | −48.2 | −25.6 | −47.0 |
| salamander | C6 | 40 | 80 | 10 | 51 | −46.0 | −16.3 | **−22.1** |
| salamander | C6 | 108 | 66 | 18 | 19 | −40.1 | −22.9 | **−26.4** |
| engine | C6 | 90 | 26 | 17 | 0 | — | −21.3 | −47.7 |
| salamander | C7 | 40 | 122 | 6 | 95 | −32.3 | −23.4 | **−15.9** |
| salamander | C7 | 68 | 73 | 7 | 47 | −28.1 | −25.4 | **−13.0** |
| salamander | C7 | 108 | 82 | 11 | 47 | −31.7 | −26.5 | **−3.5** |
| engine | C7 | 90 | 21 | 10 | 6 | −48.7 | −21.5 | −48.2 |

Three readings:

* **In the middle of the compass the engine is right.** C4: 13 unexplained peaks at −48 dB against
  the engine's 18 at −48 dB; between-partial energy −44 dB against −47 dB. Nothing to fix.
* **In the top two octaves the recordings are mostly *not* the struck string one second on.** C7
  fortissimo: −3.5 dB. The engine: −48 dB. The unexplained peaks repeat at the same frequencies
  across velocity layers (2067, 1603, 1650, 1691 Hz at C7; 803–1004 and 1364 Hz at C6), which a
  noise floor does not do, and the ratio gets *worse* with velocity (−15.9 → −13.0 → −3.5 dB), which
  a fixed room floor also does not do. It is the instrument: the undamped strings above G6, the rest
  of the compass ringing sympathetically through the bridge, and the board.
* **At the strike the engine's broadband energy is already in the right place** (−16…−50 dB
  recorded, −21…−54 dB rendered). There is no large missing hammer-noise transient inside the
  struck-note samples. The action noise is a separate recording — §5.

> **Update (sympathetic milestone).** The table above was re-measured with the same code on the engine as it
> stands now, at C4/C6/C7 and velocities 40/68/90/108, before and after the bridge admittance, the duplex segments
> and the re-fitted coupling. **`between@1s` did not move by more than a tenth of a decibel anywhere**: C4 −47.0,
> C6 −47.6, C7 −48.1, before and after alike, against the recordings' −44.3, −26.4 and −3.5 re-measured beside
> them. Section 5's `harm*` half *did* move — the release-resonance halo rose from −87.1 to −50.9 dB at C3 and
> from −137.4 to −72.8 at C5, against targets of −31 and −39 — so the milestone raised the halo without raising
> this column. The reason is that this column has a floor: with the sympathetic coupling and the duplex removed
> altogether, and with the board path, the body modes and the diffuse field removed one at a time, it stays at
> −47.2 / −47.6 / −48.0, while changing only the analysis window from 43 ms to 1365 ms moves it from −45.9 to
> −10.5 at C7. It is the leakage of the note's own decaying partials outside the guard band, and at 85 ms it sits
> at about −48 dB. The recordings are 3 to 44 dB above that floor, so the finding this section reports stands
> unchanged; what does not stand is using the number as a fitting target, which is what `estimate::halo`'s
> `between C6` and `between C7` rows do. `DECISIONS.md` 167–169.
>
> **Update (review pass): the gap is real above the floor, and it is not on the coupling.** On a 341 ms window,
> where the leakage floor lifts, C7 fortissimo one second in reads **−17.0 dB recorded against the engine's
> −38.5**, and C6 −20.5 against −49.0 — so 21.5 and 28.5 dB of genuine deficit, measurable. Taking each engine
> path to its limit and re-measuring (`independent_audit.rs` section 10) says which one owns it: the sympathetic
> coupling **removed entirely** gives −38.6 at C7 and raised to the largest value the stability contract will ever
> certify gives −38.5 — 0.1 dB of authority over this statistic, in either direction, which is also what the loop
> bound predicts (no legal preset can put more than 0.25 of effective coupling anywhere, and this one already runs
> 0.06–0.17). The board's **diffuse field** owns it: `soundboard.fdn_t60` ×4 gives −21.5 at C7 and −20.9 at C6,
> within 4.5 and 0.4 dB of the recordings. Backlog item 5's cost line ("the level is a coupling parameter,
> `resonance.rs`") is therefore wrong and is corrected below. `DECISIONS.md` 184.

### Phantom partials: confirmed, quadratic, and quiet

`f_i + f_j` stands flat of transverse partial `i+j` by only `3 f0 B i j (i+j) / 2` hertz, which for
low `i, j` is narrower than the analysis window's main lobe — probing there measures partial `i+j`
and nothing else, which is what a first attempt did. The probes below are the pairs whose sum stands
at least two main lobes clear of *every* measured partial, measured across all 16 velocity layers:

| source | key | pair | clear of partials | slope vs product | slope vs note | margin over floor | level re loudest |
|:--|---:|:--|---:|---:|---:|---:|---:|
| salamander | A0 | f8+f9 | 12.4 Hz | +1.06 | **+2.35** | 9.9 dB | −59.8 dB |
| engine | A0 | f8+f9 | 12.4 Hz | +0.60 | +1.27 | −13.0 dB | −89.2 dB |
| salamander | C4 | f8+f9 | 148.5 Hz | +0.32 | **+2.19** | 16.4 dB | −81.7 dB |
| salamander | C4 | f8+f8 | 114.4 Hz | +0.32 | +1.82 | 8.6 dB | −86.2 dB |
| engine | C4 | f8+f9 | 148.5 Hz | +0.48 | +2.25 | −20.9 dB | −123.2 dB |
| salamander | C5 | f5+f8 | 276.4 Hz | +0.53 | **+2.60** | 21.2 dB | −76.8 dB |
| engine | C5 | f5+f8 | 276.4 Hz | +0.32 | +1.51 | −19.8 dB | −158.5 dB |

A quadratic mechanism predicts one dB of phantom per dB of the *product* of the two partials, hence
two dB per dB of the note. Every Salamander probe that stands 8 dB or more above its local floor
returns a slope against the note's level of **1.8 to 2.6**; the probes with margins under 7 dB
return 1.2–1.5, which is what a floor does. The engine's probes are all *below* their local floor
(negative margin) — it has no such mechanism, as designed. Confirmed, at −60 to −95 dB relative to
the loudest partial.

## 5. Directivity, and the mechanism the engine does not have

**Stereo balance per partial**, loudest layer, left minus right:

| source | key | partials | median @0.3 s | spread @0.3 s | median @2 s | **drift** |
|:--|---:|---:|---:|---:|---:|---:|
| salamander | A0 | 62 | +1.93 | 16.19 | +1.93 | **1.24** |
| engine | A0 | 64 | +6.30 | 14.60 | +6.20 | 0.04 |
| salamander | A2 | 35 | +2.66 | 16.18 | +1.44 | **4.73** |
| engine | A2 | 34 | +3.81 | 11.76 | +3.79 | 0.09 |
| salamander | C4 | 24 | +2.19 | 15.31 | +1.11 | **3.96** |
| engine | C4 | 20 | +1.12 | 6.53 | +1.09 | 0.09 |
| salamander | C6 | 8 | −6.00 | 10.09 | +1.96 | **6.19** |
| engine | C6 | 7 | −1.78 | 1.80 | −2.09 | 0.45 |

(dB; drift is the median over partials of |Δ(2 s) − Δ(0.3 s)|.)

> **Update (Milestone A).** `voicing.polarization_pan_spread` gives the two polarizations
> different pan positions, and `estimate::directivity` fits it by inverting a line measured on the
> engine itself. Salamander's median drift over 28 sampled keys is 4.40 dB, which asks for a
> spread of 0.49 against the engine's ceiling of 0.40; at the ceiling the same measurement on the
> engine's renders returns 0.24 (A0), 1.26 (A2), 3.33 (C4), 8.67 (C5) and 5.59 (C7) against the
> recordings' 1.24, 4.73, 3.96, 5.33 and 5.85 — the direction of the compass is right, the bass
> still barely moves. `DECISIONS.md` 137–138.

The *spread* is not diagnostic — the engine's diffuse field already scatters partials between the
channels by 2–21 dB. The *drift* is: 1.2–6.2 dB in the recordings against 0.02–0.14 dB in the
engine's, and the engine's is zero for a structural reason. It pans one mono voice per key, so the
balance it produces cannot change as the note decays, whatever the pan. In the recordings it does,
which means the two microphones hear different decay rates — different radiation for the fast and
slow parts of the note.

**The mechanism recordings**, at the level the SFZ plays them (the key-off group is attenuated 37 dB,
the pedal groups 19–20 dB; comparing the raw files would say a damper landing is as loud as the
note), against a velocity-90 strike of the same key:

| recording | key | peak re strike | rms re strike | decay to −40 dB | centroid |
|:--|---:|---:|---:|---:|---:|
| rel1 | A0 | −37.3 dB | −44.4 dB | 0.165 s | 166 Hz |
| rel37 | A3 | −30.2 dB | −31.7 dB | 0.245 s | 187 Hz |
| rel40 | C4 | −35.4 dB | −37.5 dB | 0.265 s | 192 Hz |
| rel52 | C5 | −25.4 dB | −26.7 dB | 0.210 s | 143 Hz |
| rel76 | C7 | −33.5 dB | −29.1 dB | 0.285 s | 255 Hz |
| pedalD1 | — | −35.8 dB | −38.7 dB | 5.76 s | 77 Hz |
| pedalU1 | — | −42.4 dB | −45.1 dB | 0.320 s | 187 Hz |
| harmLC3 | C3 | −30.7 dB | −40.4 dB | 1.01 s | 314 Hz |
| harmLC5 | C5 | −39.0 dB | −46.7 dB | 2.13 s | 507 Hz |

The engine produces none of these sounds. The key-off thump alone is a −25 to −39 dB event on every
note release; the pedal-down rumble is a −36 dB, six-second, 70 Hz event on every pedal press. The
`harm*` samples are a direct measurement of the halo §4 found missing in the treble, and give a
target for the sympathetic-coupling fit: −31 dB at C3, −39 dB at C5, ringing for 1–2 s.

> **Correction (review pass): the `harm*` rows are not a sympathetic-coupling target.** Split into the
> struck key's own partials and everything else, **80 % of `harmLC3`'s energy is at C3's own partials** — a
> damper takes a few tenths of a second to stop a wound string and the recording contains that decay. The
> engine's *whole* post-release signal, damper working and nothing subtracted, reads **−24.1 dB with a 76 %
> own-partial share against the recording's −30.7 dB and 80 %**: like for like it is 6.6 dB louder than this
> target, not 29 dB quieter as a coupling-only residual makes it look. At C5 the same measurement leaves
> **20 dB of genuine deficit** (−54.6 against −34.5). The *duration* half survives at both keys and belongs to
> the damper rather than to the coupling: measured identically on both, the release tail falls 20 dB in
> **0.50 s / 0.60 s** recorded and **0.15 s / 0.10 s** rendered. `DECISIONS.md` 183.

> **Update (Milestone A).** The engine plays all of them, and the column above is the parameter
> set (`DECISIONS.md` 108–115). Two corrections since: the levels in this table are ratios measured
> at the *microphone*, so the engine's reference — a velocity-90 strike of the same key — is now
> measured at its output rather than at the soundboard's input, per preset and per key
> (`engine/src/calibrate.rs`); and a mechanism burst no longer inherits the filter state of a
> finished one. Rendered against what each preset asks for, over nine keys and 16 events each, the
> key-off moved from +2.3 dB (default) and +3.9 dB (salamander-c5) to −0.33 and −0.36 — the two
> presets now agree to 0.03 dB — and the pedal events from +1.8/+3.8 (down) and +0.1/+1.6 (up) to
> −0.3/−0.1 and −1.1/−1.1. `DECISIONS.md` 144–145.

## 6. Pitch: one refutation and one unexpected finding

**Refuted — nonlinear pitch glide.** Re-tracked at four periods per window (43–85 ms), the
fundamental's frequency through the first 1.6 s, in cents relative to its value at 1.6 s:

| source | key | vel | 50 ms | 100 ms | 200 ms | 400 ms | 800 ms |
|:--|---:|---:|---:|---:|---:|---:|---:|
| salamander | C3 | 108 | −7.99 | −10.57 | −6.69 | −1.30 | −1.96 |
| engine | C3 | 90 | **+107.73** | +17.94 | +1.89 | +0.97 | +0.66 |
| salamander | C4 | 108 | +0.89 | −0.40 | +2.09 | −0.41 | −0.17 |
| engine | C4 | 90 | −6.60 | +2.17 | +0.12 | +0.70 | +0.53 |
| salamander | C5 | 108 | −6.89 | +2.44 | −0.67 | +0.84 | +4.62 |
| engine | C5 | 90 | −2.77 | −2.37 | −2.14 | −1.54 | −0.91 |

The engine's modal poles do not move, so its column *is* the measurement's error bar: ±108 cents at
50 ms, ±18 at 100 ms, a few cents from 200 ms on. Salamander's excursions are no larger. There is no
evidence for a pitch glide the engine cannot produce, and no measurement at this window length could
find one smaller than about 10 cents in the first 100 ms.

**Unexpected — the *measured* fundamental of some tenor notes drifts as it decays, and the engine's
does not.** Median over 16 layers of the frequency change over the fundamental's first 20 dB:

| key | A2 | C3 | D#3 | **F#3** | A3 | C4 | D#4 | F#4 | A4 | C5 |
|:--|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cents | −2.7 | −7.7 | +0.6 | **−31.9** | −0.4 | −1.7 | −1.0 | +2.4 | +3.7 | −1.5 |

Engine control on the same measurement: −2.0 to +0.7 cents. A single string cannot do this and
neither can the engine's unison, whose strings share one damping law: what moves a composite
partial's measured frequency is a unison whose strings are mistuned *and* decay at different rates,
so that the survivor's pitch takes over. F#3's 32 cents is the extreme case — and F#3 belongs to the
same set of F# keys that came back with anomalous T60s in Phase D (`DECISIONS.md` 94: F#5, F#6, F#7
at roughly twice their neighbours'). Whether that is this piano's voicing or something systematic in
the library is the one open question this report leaves.

---

## Ranked backlog

Audibility is judged on the measured level and on how often the sound occurs in playing. Cost is
engineering time plus the real-time budget (worst case is currently 39.5 % of one core against
`SPEC.md`'s 50 % goal).

| # | Insufficiency | Evidence | Audibility | Cost | Notes |
|---:|:--|:--|:--|:--|:--|
| 1 | No action, key-off or pedal noise | §5: −25 to −39 dB key-off, −36 dB pedal-down, on every release and pedal move | **High** | **S** — new `noise.rs`: one filtered noise burst per event, per-register gain/decay/centroid tables in the preset (measured values above); no cost when idle, ~1 % of a core while sounding | Nothing needs fitting: §5 *is* the parameter set |
| 2 | One `B` per note | §1: ratio 0.63–0.75 in the wound bass, 1.24–1.45 in the low tenor; up to 78 cents of misplaced partial | **High** in the bottom 1.5 octaves | **S** — one extra per-note coefficient in `StringParams::partial_freq`, a preset field, and the two-band fit that already exists in `residual.rs`; setup-time only | Sign flips across the break, so the parameter must be signed |
| 3 | Unison strings share one damping law | §6: measured fundamental drifts up to 32 cents (F#3), 0.2 cents in the engine | **Medium**, note-specific | **S** — per-string sigma scale in `PianoString::new`, one preset field; setup-time only | Estimable from the beat envelope's decay asymmetry |
| 4 | Stereo image cannot move | §5: 1.2–6.2 dB drift against 0.02–0.14 dB | **Medium** on held chords | **S** for the drift — the two polarizations are already separate `process_add` calls; give them separate pan positions (one extra block buffer per voice). **L** for the full per-partial pattern: per-mode L/R gains is a second accumulation in the modal loop, ~+20–30 % of the string cost | Take the S; the spread is not diagnostic (engine 2–21 dB already) |
| 5 | Treble/aftersound halo far too quiet | §4: at a 341 ms window C7 at 1 s is −17.0 dB recorded against the engine's −38.5, C6 −20.5 against −49.0 | **High** in the top two octaves | **M** — ~~a coupling parameter~~ **the board's late field**: the coupling has 0.1 dB of authority over this number even at the stability contract's ceiling, `soundboard.fdn_t60` has 17–28 dB (`DECISIONS.md` 184). A fit against the recordings, after §9's question — how much of their late field is the instrument and how much the room | Also the reason Phase D's treble decays needed a floor detector |
| 6 | Excitation spectrum smoother than the piano's | §3: 5–10 dB scatter against 2–5 dB control; not shared between notes | **Medium** (note-to-note character) | **L** — per-note per-partial gain table, ~40 numbers × 88 keys, and the microphone-comb confound of `DECISIONS.md` 93 | The cheap global-admittance version is refuted by measurement (§3) |
| 7 | No phantom partials | §4: confirmed quadratic, −60 to −95 dB | **Low** | **L** — a per-sample nonlinearity in the string or an explicit combination bank | Defer. The measurement's value is that it closes the question |

Two things this report deliberately does not recommend:

* **Do not chase the envelope residual** (§2). The engine's own renders produce as much of it.
* **Do not build a global bridge-admittance curve** (§3). Measured, refuted, before it was built.

## What stage 1 cannot say, and stage 2 must

Isolated-note recordings contain no information about the coupling *between* strings (the halo of §4
is a mixture of coupling and radiation that one note cannot separate), about pedal-down behaviour,
or about repetition and re-strike into a ringing string. Those are `TUNING.md` stage 2, and this
report leaves them three things to start from: the halo levels of §5 as a target, the warning of §2
about which loss terms are informative, and the finding of §3 that the excitation residual is
per-note — which means stage 2 should not try to absorb it into the global recording-chain EQ.
