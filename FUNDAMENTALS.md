# FUNDAMENTALS.md — the string model against the physics it claims to be

First half: a physics-lens review of `engine/src/string.rs`, written after M3 against the
literature and against the measurements already in this repository. The perception reviewer
appends the second half.

The trigger is a listening complaint the scoreboard did not catch: after M3 the instrument still
sounds "sinusoidal and artificial", and now audibly *jitters inside the note frequency* — "mini
frequency oscillations" — while `renders/realism/REALISM.md` improved (mel 5.34 → 4.91, modulation
5.76 → 3.84). `renders/jitter/JITTER.md` has since measured the artefact and confirmed its
mechanism. This file asks the prior question: **is the model the artefact comes out of the right
model at all**, judged against Weinreich (1977, 1990), Capleton (2004), Woodhouse (2021 / Euphonics
§7.3) and Aramaki, Bensa *et al.* (2001) — and if not, what is the right one and what does it cost.

Nothing here was written by re-rendering. Every engine number is either read out of
`presets/salamander-c5.toml` (the preset the forensics used) or computed in closed form from it;
every measured number is cited to the file that measured it. `engine/` is untouched.
`cargo test --release` in `tuner/`: **238 passed, 0 failed**.

---

## 0. Verdicts

| # | choice in `string.rs` | verdict |
|:--|:--|:--|
| **a1** | two polarization banks per string, vertical loud-and-fast / horizontal quiet-and-slow | **SOUND** — it is the correct eigenbasis, and Weinreich's measurement is what makes it correct |
| **a2** | the split between them is a **fixed number of hertz** (`horizontal_offset_hz = [0.35, 0.52, 0.27]`), the same at every partial of every key | **WRONG** — 35× the physically derivable value, and constant where the physics is proportional to ω. This is the metronome. |
| **a3** | the two polarizations never exchange energy | **DEFENSIBLE-BUT-AUDIBLE** — the off-diagonal admittance really is small; what it costs is the slow, non-periodic wander that would break up the beat |
| **b1** | unison strings as independent banks at fixed detune ratios | **WRONG** — it is the `μ → ∞` limit of a system whose actual `μ` is 0.27–0.8 over the whole bass and midrange |
| **b2** | `unison_coupling`, applied one block late through the excitation | **WRONG** — ~25× too weak *and* phase-scrambled by the 2.67 ms delay; measured to change the result by −0.07 cents, i.e. it does nothing |
| **c** | the fitted detune / polarization / per-string-σ parameters as a description of what a recorded composite partial does | **WRONG as a description** — the parameters are fitted to a model that cannot produce the target's defining property (velocity dependence), so fitting them harder moves the artefact rather than removing it |
| **d** | — | the correct construction is a `2N × 2N` complex eigenproblem solved at preset-load time. It stays inside `ModalBank`, at **the same mode count**, for **one extra FMA per mode-sample**, and it *deletes* three preset fields. |

The one-line summary: **`string.rs` builds a piano unison in the wrong physical regime.** It models
2N free-running oscillators whose frequencies are set by the tuning and whose decay rates are set by
hand; the instrument it is imitating has 2N *coupled* oscillators whose frequencies are pulled
together by the bridge and whose decay rates are split apart by it. Those are not neighbouring
approximations. They are opposite limits of the same 2N × 2N matrix, and the piano is on the other
side.

---

## 1. The physical baseline, and the coefficient the preset already contains

### 1.1 One string on a bridge that moves

A string terminated on a rigid support has real mode frequencies. Terminated on a bridge of complex
admittance (mobility) `Y`, each mode's frequency acquires a complex shift. Weinreich's result, as
Capleton writes it (JASA **115**(2), 2004, Eq. 2):

```
    δω_n = i Z_n Y_n ω_n / π
```

with `Z` the string's transverse wave impedance and `Y = G + iB` the driving-point admittance at the
attachment. Splitting it:

```
    Re δω = − (Z ω / π) B        the frequency pull   (reactive part of Y)
    Im δω = + (Z ω / π) G        the decay rate       (resistive part of Y)
```

So one number, `γ ≡ Z ω G / π`, is simultaneously *the rate at which the partial loses energy into
the board* and *the strength with which the board couples this string to everything else attached at
the same place*. **They are the same coefficient.** That is the single most important structural fact
in this review, because it means the coupling in a unison group is not a free parameter to be tuned:
it is fixed, to within the reactive/resistive ratio, by the decay rate the preset has already fitted
to the recordings.

### 1.2 Three routes to the same number

For a middle-octave string, Weinreich's measurements give `Z ≈ 2 kg/s` and `|Y| ≈ 10⁻³ s/kg`
(quoted by Capleton §III.B). At C4's fundamental:

```
    γ = Z G ω / π = 2 × 10⁻³ × 2π × 261.5 / π  =  1.05 s⁻¹
```

Cross-checks:

| route | value at ≈ C4 | source |
|:--|--:|:--|
| `Z G ω / π` from Weinreich's measured `Z`, `Y` | **1.05 s⁻¹** | Capleton §III.B |
| Weinreich's measured *prompt* decay of Eb3, 8 dB/s | **0.92 s⁻¹** | Five Lectures, Fig. 1 |
| the shipped preset's own `radiated_share × σ_v(C4, k=1)` = 0.5 × 1.675 | **0.84 s⁻¹** | `presets/salamander-c5.toml` |

Three independent routes inside 25 %. The preset's `[voicing.bridge].radiated_share = 0.5` — whose
documented meaning is "share of each partial's decay rate that is loss **into the board**" — is
already the measurement the coupled model needs. There is no new parameter to fit.

### 1.3 What that implies about the two polarizations

The bridge gives more readily perpendicular to the soundboard than parallel to it, so `G_v > G_h`;
that is why the vertical polarization is louder and dies sooner and the horizontal one is the
aftersound (Weinreich, *Five Lectures*, §"Prompt sound and aftersound"; Capleton: "it is reasonable
to assume that the decay rate in the horizontal axis will be about a quarter of that in the vertical
axis"). The engine's `horizontal_decay_ratio` is 0.29 in `default.toml` and 0.172 in the fitted
preset — the right mechanism, roughly the right size.

The *frequency* split between the two polarizations is a different quantity: it comes only from the
anisotropy of the **reactive** part `B`, and Weinreich measured that part to be nearly isotropic —
Capleton, summarising him: *"the reactive part is approximately the same for both directions … the
angular variation of the reactive part of the bridge admittance is at least a factor of 10 smaller
than the variation in the resistive part."* This is the fact §2 turns into a refutation.

---

## 2. (a) Two polarization banks at fixed hertz offsets, with independent decays

### 2.1 The structure — SOUND

Two banks per string, one per polarization, with the vertical excited harder and damped harder, is
not an approximation of the eigenmode solution: **when the bridge's principal admittance axes are
vertical and horizontal, it *is* the eigenmode solution.** The 2 × 2 admittance matrix is then
diagonal, its eigenvectors are the two linear polarizations, and the complex eigenvalues are exactly
`iω − σ_int − (Zω/π)(G_p + iB_p)` for `p ∈ {v, h}`. Capleton is explicit that the general case is
elliptical and "will resolve into two modes linearly polarized with directions and phases determined
by the bridge admittance matrix" — and Weinreich's measurement is what says the resolved directions
are close to vertical and horizontal. `string.rs` chose the right basis. Keep it.

The double decay that falls out of it is the documented, expected consequence: Woodhouse (JASA
**150**, 4375, 2021) shows the two polarizations of a *single* string suffice to produce a
double-decay envelope, and predicts for the piano that "loss factor due to body coupling exceeds air
damping by ~20 dB across the entire midrange (above ~160 Hz)". The engine's whole double-decay
architecture is on firm ground.

### 2.2 The value 0.35 Hz — WRONG, by a factor of about 35

Invert §1.1 for the split between the two polarizations of one string:

```
    Δf = f · Z · ΔB / π       (ΔB = B_v − B_h, the reactive anisotropy)
```

Capleton's worked example: a reactive ratio of 1 : 0.925 (`ΔB = 7.5 × 10⁻⁵ s/kg`) gives, at middle
C's fundamental, **"less than 1 beat per 10 seconds"** — 0.012 Hz by the formula. To get 1 Hz you
need `ΔB ≈ 6 × 10⁻³ s/kg`, which is "in the order of 6:1" anisotropy, and Weinreich's results
"showed a variation of this magnitude not to be present."

Now solve for what the shipped preset asserts. `horizontal_offset_hz[0] = 0.35 Hz` at C4's
fundamental requires

```
    ΔB = 0.35 × π / (261.5 × 2)  =  2.1 × 10⁻³ s/kg
```

**a reactive anisotropy about twice the entire measured magnitude of `Y`.** The vertical and
horizontal reactive admittances would have to differ by more than either of them is. That is not a
mis-set parameter; it is outside the space of admittance matrices.

The same number can be derived from the preset without going near Weinreich's `Z` and `Y` at all,
which is the check worth doing because it uses only the engine's own fitted quantities. With
`N·γ_v = radiated_share × σ_v = 0.837 s⁻¹` at C4 k=1, a reactive part comparable to the resistive
one (`β ≈ 1`, Weinreich), and Capleton's measured anisotropy (`ε = 0.075`):

```
    Δf = N γ_v β ε / 2π  =  0.837 × 0.075 / 2π  =  0.010 Hz
```

0.010 Hz against 0.35 Hz. **A factor of 35, and a period of 100 s against 2.9 s.** The physically
derivable split is inaudible as a beat — it tilts the decay and nothing more. The shipped one sits
squarely inside the band the ear reads as pulsing.

### 2.3 The *shape* — constant hertz — is the metronome

No mechanism produces a polarization split that is constant in hertz across a partial series:

* through the admittance, `δω ∝ ω` (§1.1) — the split grows with partial number;
* through a difference in effective speaking length in the two planes, `Δf/f` is constant — the
  split grows with partial number;
* through a difference in effective tension, `f ∝ √T` — the split grows with partial number.

Constant in hertz is the one law with no mechanism behind it, and it has a specific, catastrophic
consequence. Because vertical mode `k` of string `j` and horizontal mode `k` of the *same* string
differ by exactly `horizontal_offset_hz[j]` **independently of `k` and independently of the key**,
the three numbers 0.27, 0.35 and 0.52 Hz are beat rates of *every partial of every note in the
instrument*. `JITTER.md`'s component tables confirm it without exception:

| key | partial | beat rates the preset builds (Hz) |
|:--|--:|:--|
| C4 | k = 1 | 0.058, 0.069, 0.091, 0.149, 0.192, 0.201, 0.259, 0.261, **0.270**, 0.328, **0.350**, 0.419, 0.462, **0.520**, 0.611 |
| C4 | k = 4 | 0.013, 0.018, 0.232, 0.246, **0.270**, 0.288, **0.350**, 0.363, 0.502, 0.516, **0.520**, 0.533, 0.596, 0.866, 0.883 |
| A2 | k = 1 | 0.048, 0.218, 0.302, **0.350**, **0.520**, 0.568 |
| C6 | k = 4 | **0.270**, **0.350**, **0.520**, 1.357, 1.627, … |
| A4 | k = 3 | 0.111, 0.159, **0.270**, 0.291, **0.350**, 0.409, **0.520**, 0.641, … |

Every row. Every key. So a held chord modulates *coherently* at 0.27, 0.35 and 0.52 Hz across all of
its notes and all of their partials — an instrument-wide, note-independent, velocity-independent
pulse. That is precisely the percept "jittering inside the note frequency… mini frequency
oscillations", and it is not a rendering artefact but an exact consequence of the parameterisation.

It also shows up in the aggregate scoreboard, unlabelled: in `renders/realism/REALISM.md`, three of
the six phrases (`arpeggio_dynamics`, `chords_pedal`, `alberti_fast`) have their **worst modulation
rate at 0.6 Hz**, 4.1–6.9 dB out — the lowest bin of the modulation metric, where 0.27–0.52 Hz lands
at that resolution. The metric has been reporting this fault all along; nothing told it what it was
looking at.

### 2.4 The guaranteed null, in closed form

Two components with a fixed frequency offset, a fixed initial amplitude ratio `g = 10^(H/20)` (from
`horizontal_gain_db`) and a fixed decay ratio have exactly **one** amplitude-crossing time, and it is
a closed form:

```
    t_× = ln(1/g) / (σ_v − σ_h)
```

Because the amplitude ratio is monotone, the deepest null the pair can produce — and therefore the
largest instantaneous-frequency excursion — is *guaranteed* to occur, once, at `t_×`, at a time no
property of the strike can move. Evaluated on the shipped preset (`H = −27.61 dB`, `ρ = 0.172`):

| key | σ_v (k=1) | σ_h | **t_× bare** | measured `equal at` (`JITTER.md`) | `under dB` |
|:--|--:|--:|--:|--:|--:|
| A2 | 0.966 | 0.166 | 3.98 s | — (outside the window) | — |
| C4 | 1.675 | 0.288 | 2.29 s | 2.51 / 2.58 s at k = 2, 4 | −4.6 |
| A4 | 2.729 | 0.470 | 1.41 s | **2.73 s at k = 1** | **0.0** |
| C6 | 4.277 | 0.736 | 0.90 s | **2.43 / 1.10 / 1.03 / 0.96 s at k = 1…4** | **0.0** |

(The per-partial tables `notes.partial_sigma_scale` and `voicing.unison_sigma_scale` scale `t_×` by
`1/s`, which is why the measured times sit above the bare ones; this is the whole of M3's
contribution to the artefact, and it is a *shift* of the crossing, not a creation of it — matching
the forensics' −0.27 c attribution to `partial_sigma`, in both directions.)

At C6 the `v0/h0` pair is the **loudest** thing in the partial at the crossing (`under dB = 0.0`) at
every one of k = 1…4, all at 0.350 Hz, all around 1 s. Every partial of a C6 therefore passes through
a full null at the same rate at nearly the same moment. That is the worst single cell in the model.

### 2.5 No cross-polarization coupling — DEFENSIBLE-BUT-AUDIBLE

Neglecting the off-diagonal of the 2 × 2 admittance is defensible: Weinreich measured the principal
axes to be close to vertical/horizontal, so the off-diagonal is a second-order term. What it costs is
that the polarization plane never *rotates*. In the real string the two polarizations exchange energy
slowly and the motion is elliptical with a drifting orientation, which modulates the radiated
amplitude at rates that are neither fixed nor periodic. Removing it is what leaves the beat as a
clean sinusoidal envelope with nothing on top of it — `JITTER.md`'s envelope flatness for the engine
at C4 k = 1 is −63.2 dB (one line) against the recording's −27.6 dB.

### 2.6 An internal contradiction between two shipped fields

`radiated_share = 0.5` says half of each partial's decay rate is loss into the board, i.e.
`σ_int = 0.5 σ_v`. A polarization that couples to the board *less* can then decay no more slowly
than `σ_int`:

```
    σ_h ≥ σ_int = 0.5 σ_v      ⟹      horizontal_decay_ratio ≥ 0.5
```

The shipped preset has `horizontal_decay_ratio = 0.172`. The two fields make incompatible claims
about the same physical quantity, by a factor of about three. One of them is wrong — most likely
`radiated_share`, since 0.172 is close to Capleton's "about a quarter" — and the eigen-construction
of §5 cannot be built until they are reconciled, because it uses both.

(In the code as written, `radiated_share` multiplies only the *fluctuation* of the board's mobility,
`1 + share·(|P(f_k)| − 1)` — deliberately, so the mean is not counted twice; see
`radiated_damping`. The contradiction is between the two fields' stated physical *meanings*, which is
what matters here, because §5 needs the quantity the doc comment describes and there is nowhere else
in the preset to get it from.)

---

## 3. (b) The unison group: independent banks plus a one-block-late coupling

### 3.1 What Weinreich's normal modes are

Two nominally identical strings on one bridge point form a single dynamical system. Woodhouse (2021),
restating Weinreich:

> The antisymmetric mode has the two strings moving in opposite directions, so that the net force
> exerted on the bridge is zero and no energy is lost to the soundboard. The symmetric mode has the
> two strings moving together, exerting a large combined force on the bridge. The bridge moves in
> response to this force, so that the effective length of the string is changed and the frequency is
> shifted a little, and also some energy is lost to the soundboard so that the loss factor is higher.

And the consequence:

> The mode with the faster decay is automatically associated with more soundboard motion and
> therefore more sound radiation, so that it usually dominates the early sound. However, sooner or
> later its amplitude will fall below that of the slower-decaying mode, which will then take over and
> dominate the "after-sound".

Note what is *not* in that description: perpetual beating. The aftersound is **one surviving mode**,
not two components continuing to beat. Beating in the coupled system is a transient, present only
while both modes are comparable in amplitude.

The synthesis literature reached the same structure from the other end: Aramaki, Bensa *et al.*
(2001), resynthesising two coupled piano strings from laser-velocimetry measurements, describe the
mechanical system as producing "doublets of components, thus generating beats **and** double decays
on the amplitudes of the partials" — the doublet, its frequency split and its two decay rates are all
one output of one coupling, not three independent parameters as they are in `string.rs`.

### 3.2 The regime parameter, and which side the piano is on

Write each string's partial as a complex amplitude and put the coupling in as §1.1 requires. For two
strings, mean frequency `ω̄`, mistuning `Δω`, complex coupling `c = γ(1 + iβ)`:

```
    ȧ = M a ,     M = [ iω₁ − σ_int − c        −c          ]
                      [      −c           iω₂ − σ_int − c  ]

    λ± = iω̄ − σ_int − c ± √( c² − (Δω/2)² )
```

Everything is decided by the dimensionless ratio

```
    μ ≡ (Δω/2) / |c| = π Δf / γ
```

* **`μ → ∞`** (mistuning ≫ coupling): `λ± → iω_{1,2} − σ_int − c`. Two independent lines at their own
  frequencies, with **the same, averaged, damping**, beating forever at exactly `Δf`. This is
  `string.rs`, exactly, with `c` set to zero and the damping written in by hand.
* **`μ = 0`** (perfect unison): `λ = iω̄ − σ_int` (antisymmetric — no radiation loss, no frequency
  pull) and `λ = iω̄ − σ_int − 2c` (symmetric — twice the loss, twice the pull). Two modes at the
  *same* frequency with **decay rates split by a factor of `(σ_int + 2γ)/σ_int`**. No beat at all;
  pure double decay.
* **`μ < 1`**: `√(c² − (Δω/2)²)` is (nearly) real, so the split is in the **real** part — the strings'
  frequencies are pulled together and their decay rates pushed apart. Woodhouse calls this
  *anti-veering* for the piano case and states it plainly: with a resistive-dominated bridge
  admittance the frequencies **attract and can momentarily merge**, and *"with anti-veering there
  will be no beats"* — only a change in the decay envelope.

Now evaluate `μ` on the shipped preset, using `γ = radiated_share × σ_v` (§1.2) and the widest pair
of each group:

| key | unison | detune spread at k=1 | γ (s⁻¹) | **μ** | regime |
|:--|--:|--:|--:|--:|:--|
| A2 | 2 | 0.048 Hz | 0.483 | **0.31** | locked |
| C4 | 3 | 0.149 Hz | 0.837 | **0.56** | locked |
| A4 | 3 | 0.349 Hz | 1.365 | **0.80** | locked |
| C6 | 3 | 1.181 Hz | 2.138 | **1.74** | veering, beat slowed ~18 % |

Across all 73 keys with more than one string, `μ` runs from **0.27** (key 50, D3) to **5.5** (key 96,
C7), and **35 of 73 are below 1** — the entire bass and midrange. The engine models all of them at
`μ = ∞`.

This lines up exactly with the forensics' distribution finding: the engine is "too dead where the ear
tracks pitch (fundamentals of the mid/low register) and too spiky where it does not". The mid and low
register is precisely where `μ` is smallest and where discarding the coupling changes the answer
most.

### 3.3 What the correct solution actually produces — C4, computed

Solving the 3 × 3 (one polarization, purely resistive `Y`, `σ_int = (1−share)σ_v`,
`γ = share·σ_v/N` so that the loud mode decays at exactly the fitted `σ_v`):

| | frequency − nominal | σ (s⁻¹) | T60 | radiated weight |
|:--|--:|--:|--:|--:|
| coupled mode 1 | **+0.0499 Hz** | 0.898 | 7.70 s | 0.47 |
| coupled mode 2 | **−0.0451 Hz** | 0.968 | 7.14 s | 0.68 |
| coupled mode 3 | **+0.0115 Hz** | **1.484** | 4.66 s | **1.52** |
| *what `string.rs` builds* | −0.091 / 0.000 / +0.058 Hz | **1.675, all three** | 4.13 s | 1.09 / 1.00 / 0.91 |

Read off three things the engine cannot do:

1. **The frequency spread contracts** from 0.149 Hz to 0.095 Hz — 36 % — and the individual beat
   rates change from {0.058, 0.091, 0.149} to {0.038, 0.056, 0.095}. The tuning no longer maps
   one-to-one onto the beat rate. This is the anti-veering.
2. **The decay rates split by a factor of 1.65** where the engine's are identical to the last bit.
   Identical decay rates are what make the engine's unison beat *at constant depth forever*; split
   rates are what make a real one beat once and stop.
3. **The radiated weights differ by 3.3×**, and the mode that radiates most is the one that dies
   first. That is Weinreich's prompt sound / aftersound, arriving free from the tuning.

The full 6 × 6 (three strings × two polarizations, `Re Y_h/Re Y_v = 0.172`, reactive anisotropy
1 : 0.925, hammer driving the vertical plane with a 3 % horizontal leak, strike shares
`[1.09, 1.0, 0.91]`) sorted by radiated amplitude:

| | Δf (Hz) | σ (s⁻¹) | T60 | |G| (dB re loudest) |
|:--|--:|--:|--:|--:|
| 1 | −0.143 | 1.582 | 4.4 s | **0.0** |
| 2 | −0.027 | 0.907 | 7.6 s | **−21.7** |
| 3 | +0.053 | 0.861 | 8.0 s | −33.0 |
| 4 | −0.163 | 0.959 | 7.2 s | −43.8 |
| 5 | −0.019 | 0.854 | 8.1 s | −62.4 |
| 6 | +0.056 | 0.843 | 8.2 s | −73.3 |

One mode carries the note; the next is **21.7 dB down**. Two components 21.7 dB apart can modulate
the envelope by at most **1.4 dB peak-to-trough**. So the correct construction predicts: *a C4 does
not beat measurably while it is loud*; the modulation appears only in the tail, once, around the time
the loud mode has fallen 21.7 dB — about 4.6 s — as a single crossover, not as a train. Beside that,
`JITTER.md` measures the engine's C4 k = 3 and k = 4 beating at **15.2 and 17.3 dB** depth,
periodically, from 0.3 s on.

Note also that `−21.7 dB` is the aftersound level, *derived*. The engine hand-sets the same quantity
as `horizontal_gain_db = −27.6 dB`. In the correct model it is a consequence of the detuning and the
strike-share asymmetry — which is exactly the handle a piano tuner uses, and exactly the handle the
backlog's item 5 ("treble/aftersound halo far too quiet", 20 dB out) has been missing.

### 3.4 `unison_coupling` as implemented — WRONG on two counts

`PianoString::couple` adds `coupling · (group_previous − own_previous)` into each string's
excitation, one `BLOCK` late.

**Magnitude.** The physics needs a coupling of the order of `γ = 0.84 s⁻¹` at C4 — i.e. of the order
of the string's own decay rate, because it *is* the string's own decay rate (§1.1). The preset's
`unison_coupling = 0.02` produces, by `string.rs`'s own loop-gain estimate in the module doc
(≈ `5 × coupling` for the slowest bass note, less everywhere else), an effect of order 0.1 on a decay
rate in the bass and far less at C4. The forensics measured it end to end: turning
`unison_coupling` off changes the frequency jitter by **−0.07 cents** and the beat depth by
**−0.06 dB**. The engine has a unison-coupling parameter that does not couple — and because
`MAX_UNISON_COUPLING = 0.05` is only 2.5× the shipped value, no legal preset can reach the physical
strength either. (The two are in different units — the preset's is a fraction of the wave impedance,
the physics' is a rate — so the comparison worth trusting is the measured one: the physics needs the
coupled and uncoupled solutions to differ by the whole of §3.3's table, and the shipped coupling
moves the result by seven hundredths of a cent.)

**Phase.** One block is 128/48000 = **2.667 ms**. At C4's fundamental that is 0.70 of a period —
**251° of phase error**; at C4 k = 4, 2.79 periods; at C6 k = 4, 11.4 periods. The phase of the
coupling term therefore wraps repeatedly across a single note's partial series, so the *sign* of both
the frequency pull and the damping split is effectively randomised per partial. `DECISIONS.md` 151
already established exactly this for the resonance bus — "a string's response at its own partial is
90° out of phase with its drive, so self-feedback shifts the partial's *frequency* rather than its
damping, and what decides the sign of the small damping component is the bus's one-block delay". The
same argument applies here and is fatal for the same reason: **a bridge admittance is a
frequency-dependent complex number, and a fixed sample delay is not one.**

The fix is not to make the delay shorter. It is to stop expressing the coupling in the time domain at
all: the coupling has no state, it is a construction-time property of the partial, and it belongs in
the eigenproblem (§5). That deletes `couple()`, `group_previous`, `Polarizations::previous`,
`MAX_UNISON_COUPLING` and the loop-gain worry with it.

---

## 4. (c) What a fixed-offset sum must do, and what the recording actually does

### 4.1 What the construction mathematically must do

For `M` components `A_i e^{−σ_i t} e^{i(2π f_i t + φ_i)}`, the composite's instantaneous frequency is

```
                Σ_{i,j} A_i A_j f_i cos(φ_i − φ_j)
    f_inst(t) = ──────────────────────────────────
                    | Σ_i A_i e^{iφ_i} |²
```

Four properties follow, none of them adjustable:

1. **The denominator can reach zero**, so the excursion is *unbounded*, and it is largest exactly
   where the partial's amplitude is smallest. Any measure of "where the jitter sits" must therefore
   come back near zero. `JITTER.md`'s two-equal-sinusoids control: 21.7 cents raw, **wRMS 0.01**.
2. **The beat rates are `|f_i − f_j|` forever**, unchanging, since the `f_i` are preset constants.
   `M = 2N` gives `N(2N−1)` rates — 15 for a triple, 6 for a pair, which is precisely the length of
   every list in `JITTER.md`'s component tables.
3. **There is exactly one amplitude-crossing per pair**, at a time fixed by the preset (§2.4), so the
   deepest null of a note is scheduled at construction.
4. **Nothing about the strike can move any of it.** Velocity scales every `A_i` together; the ratios,
   the rates and the crossing times are untouched.

### 4.2 What the recording does

| property | recording | engine | source |
|:--|--:|--:|:--|
| mean jitter over 16 cells (cents) | 1.50 | 1.27 | `JITTER.md` |
| **where the jitter sits** (`wRMS/raw`, median) | **0.58** | **0.16** | `JITTER.md` |
| … worst cell (A4 k = 1) | **1.15** | **0.03** | `JITTER.md` |
| **spread across vel 40/90/120** (cents, mean / max) | **0.80 / 3.45** | **0.008 / 0.03** | `JITTER.md` |
| **beat-depth spread across velocity** (dB, mean / max) | **2.27 / 7.30** | **0.048 / 0.29** | `JITTER.md` |
| linewidth excess over a same-decay control (cents) | −0.03 / +0.14 / −0.20 | +0.22 / −0.14 / −0.30 | `ANALYSIS.md`, C4/A2/C6 |
| envelope modulation, C4 k = 1 (dB RMS, 0.1–20 Hz) | 3.05 | 0.18 | `ANALYSIS.md` |
| envelope flatness, C4 k = 1 (dB) | −27.6 | −63.2 | `JITTER.md` |

Read together these say something quite precise, and it is not "the engine jitters too much":

* **The level is right; the *distribution* is wrong.** 1.27 against 1.50 cents on average, but 33×
  too still at C4 k = 1 (0.08 vs 2.66) and 4.5× too much at A4 k = 1 (3.91 vs 0.87). Which cell gets
  which is a lottery on whether the scheduled crossing of §2.4 happens to fall inside the analysis
  window. That is not a model producing a distribution; it is a model producing one event per pair
  and letting the calendar decide.
* **The recording's wobble rides the loud part of the partial** (`wRMS/raw` 0.58, up to 1.48). A
  free-running pair *cannot* do that — property (1) above forces the opposite, and the engine duly
  reads 0.03–0.16. This is the single cleanest discriminator in the whole dataset, and it says the
  recording's frequency movement is not a beat null.
* **The recording's beat structure moves with velocity by 100× more than the engine's** (0.80 vs
  0.008 cents; 2.27 vs 0.048 dB). By property (4), *no* setting of `detune_cents`,
  `horizontal_offset_hz`, `unison_sigma_scale`, `partial_gains` or `partial_sigma_scale` can produce
  this. It is a structural impossibility, not a fitting failure. In the coupled model it is
  immediate: the eigenmodes are fixed but the *mixture* `c = V⁻¹u` is set by the strike vector, and
  Weinreich's own abstract lists the three things the aftersound depends on as "bridge admittance,
  **hammer irregularities**, and the exact state in which the piano is tuned."
* **The recording has modulation without a resolvable doublet.** C4 k = 1 shows 3.05 dB RMS of
  envelope modulation at a centroid of 0.93 Hz, yet a linewidth excess of −0.03 cents over a
  same-decay single-sinusoid control — where `ANALYSIS.md`'s own control says two equal components
  0.5 Hz apart read **+2.1 cents** of excess. A persistent equal-amplitude pair at ~0.9 Hz would have
  been seen. It was not. The modulation is there and the doublet is not: the components are unequal
  by the time the linewidth window opens at 1.5 s, which is the eigenmode picture (one survivor) and
  not the free-running one (two equals).

Taken together: **the recording's composite partial beats early, once, while it is loud, by an amount
that depends on how it was struck, and settles into a single line.** The engine's beats late,
forever, at a null, by an amount fixed at preset-load time. Verdict on the fitted parameters as a
description: **WRONG**.

### 4.3 Why M3 improved the scoreboard and made the sound worse

`REALISM.md`'s modulation metric measures *how much* the band envelopes move, not *how* they move.
The instrument was short of envelope movement; M3's per-partial tables moved the amplitude-crossing
times (`t_× ∝ 1/s`, §2.4) so that more of the scheduled nulls landed inside the analysis windows.
That deepens the metronome and raises the score at the same time. The forensics' attribution is
consistent with this and with nothing else: `partial_gains` is **innocent** (+0.08 cents, +0.08 dB) —
which is what one expects, since no per-partial excitation table can change a frequency-structure
error — while `partial_sigma_scale` moves the crossings by −0.27 cents / −0.98 dB **in both
directions** (A4 k = 1: 3.91 → 1.55 when removed; C4 k = 1: 0.08 → 0.34). M3 is a contributor at the
margin. The cause predates it and is `notes.detune_cents` + `voicing.horizontal_offset_hz` on an
uncoupled unison.

The general lesson for the scoreboard: **an aggregate that rewards the presence of modulation will
reward a metronome.** The metrics that separate them already exist and are cheap — `wRMS/raw`, and
the velocity spread of the beat structure — and neither is on the scoreboard.

---

## 5. (d) The construction the physics asks for

### 5.1 The eigenproblem

Per partial `k` of one key. Degrees of freedom: `N` strings × 2 polarizations = `2N` complex modal
amplitudes `a ∈ ℂ^{2N}`. Write

```
    ȧ = A_k a ,     A_k = i Ω_k − σ_int,k I − C_k

    Ω_k = diag(ω_k · detune_j)                       (2N × 2N, polarization does not change ω)
    C_k = (Z_k ω_k / π) · Y(ω_k)                     the bridge, in s⁻¹
```

`Y(ω_k)` is the `2N × 2N` mobility matrix at the attachments: entry `((j,p),(j′,p′))` is the velocity
of string `j` in plane `p` per unit force from string `j′` in plane `p′`. Its diagonal is the
self-admittance (radiation damping and the frequency pull of §1.1); its off-diagonal is the mutual
admittance (the coupling). For strings of one unison landing on one bridge point, mutual ≈ self, and

```
    C_k  ≈  [ c_v · J_N      0       ]        J_N = all-ones N × N
            [    0        c_h · J_N  ]        c_p = γ_v (g_p + i β_p)
```

Eigendecompose `A_k = V Λ V⁻¹`. Each eigenvalue `λ_m = −σ_m + i 2π f_m` **is** one mode of the group
at this partial: a frequency and a decay rate. There are `2N` of them — the same count the engine
already builds.

**Excitation.** The hammer delivers a common pulse shape `h(t)` with per-string share `s_j`, the
existing timing skew `d_j`, and a small horizontal leak `ε_j` (the hammer is not square to the
strings — the same fact `strike_share` already encodes):

```
    u_{(j,v)} = s_j g_k e^{−i ω_k d_j} ,      u_{(j,h)} = ε_j s_j g_k e^{−i ω_k d_j}
    c = V⁻¹ u                                  ← V⁻¹, not Vᴴ: A is non-normal
```

`g_k` is the existing `comb_magnitude × contact_taper × partial_gains / SAMPLE_RATE`. Nothing in the
excitation model changes; it is merely projected onto a different basis. The delay `d_j` becomes a
phase rotation at `ω_k`, which is why it collapses into one complex scalar per mode instead of
needing `N` input buffers.

**Radiation.** The bridge force from mode `m` is `w · v_m` with `w_{(j,v)} = 1`, `w_{(j,h)} = η` (the
horizontal plane radiates less). So mode `m`'s output gain is

```
    G_m = (w · v_m) · c_m         ∈ ℂ
```

At zero detuning the antisymmetric eigenvectors give `w · v_m = 0` exactly — they radiate nothing.
Detuning breaks the symmetry and `|w · v_m|` grows with it. **The aftersound level becomes a function
of the tuning**, which is what tuners have always claimed and what the model has never been able to
express.

**Normalisation, so the fitted T60 anchors survive.** `notes.sigma0`/`sigma1` are fitted to recorded
decays and already contain the coupled system's *heard* rate. Adding `γ` on top would double-count,
exactly as `BridgeVoicing::radiated_share`'s own doc warns for the mean mobility. Set

```
    σ_int,k = (1 − share) · σ_k ,     γ_v = share · σ_k / N
```

so that the loud (symmetric, vertical) mode decays at exactly `σ_k` — the number the anchors were
calibrated to — and everything slower is aftersound underneath it. This is the same discipline as
`unison_sigma_scale`'s mean-of-1 constraint (`DECISIONS.md` 105), applied to a mechanism instead of a
multiplier.

### 5.2 It stays inside `ModalBank`

| | today | proposed |
|:--|:--|:--|
| modes per partial per key | `2N` (N banks × 2 polarizations) | `2N` (eigenmodes) — **identical** |
| bank state | `re, im` per mode | unchanged |
| pole | `a = e^{−σ/fs} e^{iω/fs}` | unchanged |
| input gain | **real** `g` | **complex** `g = g_re + i g_im` |
| runtime coupling | `couple()` every block, 3 extra buffers | **deleted** |
| per-note construction | 2N × K `push_mode` calls | 2N × K `push_mode` calls, from a per-key cache |

The only change to the hot loop is that `Chunk::step` gains one FMA:

```
    re ← a_re·re − a_im·im + g_re·x
    im ← a_re·im + a_im·re + g_im·x      ← the new term
```

That takes the per-lane per-sample cost from 5 mul + 4 add to 6 mul + 5 add — call it **~+20 % on the
modal inner loop**, and the
string is not the whole instrument, so plausibly +3–5 % of a core against the current 39.9 % worst
case. It must be measured, not assumed. Two facts partly pay for it: `couple()` and its three `BLOCK`
buffers per group disappear, and so does the `MAX_UNISON_COUPLING` loop-gain contract.

(An exactly equivalent alternative, if touching `step` is unattractive: keep `g` real and make the
*output* projection complex, `y = Σ_k (α_k·Im s_k + β_k·Re s_k)`. Same one extra FMA, same
mathematics — a single-input linear system cannot tell the two apart. A third option, kicking the
mode state at strike time and leaving `g` real, costs nothing in the loop but is exact only for an
impulsive hammer and would break the sympathetic-drive and re-strike paths. Not recommended.)

### 5.3 Where the eigensolve happens, and what it costs

Nothing in `A_k` depends on velocity — only `u` does, and `u` enters only through `V⁻¹u`. So cache,
per key and per partial, at **preset load**: `(f_m, σ_m, w·v_m, row m of V⁻¹)` for `m = 1…2N`. That is
88 keys × ≤ 80 partials × 6 modes × ~14 floats ≈ **2.4 MB**, built once. Per note-on the cost is one
`2N`-vector complex product per partial — the same order as the per-mode gain arithmetic
`PianoString::new` already does.

The eigensolve itself need not be a dense `6 × 6`. With all strings on one bridge point, `C_k` is
rank 2 (only the two bridge directions carry it), so the characteristic equation collapses to a `2 × 2`
determinant condition in `λ`:

```
    det( I − C₂ · M(λ) ) = 0 ,     M(λ)_{pp′} = δ_{pp′} Σ_j 1 / (λ − d_{jp})
```

— `2N` roots of a scalar-structured rational equation, a few Newton steps each, no LAPACK. (If the
strings' attachment points are made distinct enough that mutual ≠ self, fall back to a dense `6 × 6`;
still trivial at load time.)

### 5.4 The parameter map

| field today | becomes | note |
|:--|:--|:--|
| `notes.detune_cents`, `unison_layout.detune` | **unchanged** — the diagonal `Ω_k` | still the tuning, which is what a tuner sets |
| `unison_layout.share` | **unchanged** — enters `u` | now also sets the aftersound level, via the antisymmetric projection |
| `voicing.unison_coupling` | **deleted** | it is `share · σ_k`, already fitted (§1.2) |
| `voicing.horizontal_offset_hz` | **deleted**, replaced by one dimensionless reactive anisotropy `ε` | split becomes `∝ ω`; 0.010 Hz at C4 k=1 for Weinreich's measured `ε` |
| `voicing.horizontal_decay_ratio` | re-read as `Re Y_h / Re Y_v` | same number, now a property of the bridge instead of a decay law; reconcile with `radiated_share` first (§2.6) |
| `voicing.horizontal_gain_db` | **derived** from `w·v_m` and `ε_j` | −21.7 dB computed at C4 against −27.6 dB hand-set |
| `voicing.unison_sigma_scale` | **largely redundant** | the per-string decay split it was built for (`DECISIONS.md` 105, `TUNING_REPORT.md` §6) now falls out of the eigenproblem |
| `voicing.bridge.radiated_share` | **promoted** — it is now the coupling constant | the one number the whole construction turns on |
| `notes.partial_gains`, `partial_sigma_scale` | **unchanged** | forensically innocent (+0.08 c) and −0.27 c respectively |

Three fields deleted, one derived, one redundant. The model gets *more* constrained, not less — which
is the usual sign of having found the right one.

### 5.5 Falsifiable predictions

If the construction of §5.1 is right, then on the same forensics harness, with no re-fitting:

1. `wRMS/raw` for the engine rises from 0.16 towards the recording's 0.58, because the amplitude
   crossing moves early — into the loud part of the note — as soon as the decay rates split.
2. The velocity spread of the beat structure rises from 0.008 cents / 0.048 dB towards the
   recording's 0.80 cents / 2.27 dB, driven entirely by `ε_j` and `s_j` in `u`. If it does not, the
   strike vector is still too rigid and the hammer model is the next suspect.
3. The lines at **0.270 / 0.350 / 0.520 Hz vanish from every partial of every key**, and
   `REALISM.md`'s "worst modulation rate" stops being 0.6 Hz on three of six phrases.
4. C4's fundamental beat depth falls from a periodic 15–17 dB (k = 3, 4) to ≤ 1.5 dB until the tail,
   and the C6 full nulls at `under dB = 0.0` disappear.
5. The treble aftersound gets louder without touching `soundboard.fdn_t60` — the halo of backlog
   item 5 — because the antisymmetric modes now radiate in proportion to the treble's wider detuning.
6. `string::tests::a_unison_group_beats` should still pass, but only just: the envelope will still be
   non-monotone, by much less.

And two things that would falsify it:

* If `μ` computed from a *re-fitted* `radiated_share` comes back ≫ 1 across the midrange, the
  free-running model was right after all and the artefact is only the polarization offset of §2.
* If the recording's `wRMS/raw` is an artefact of its own noise floor rather than of where its
  frequency movement sits, the strongest single discriminator in §4.2 evaporates. Worth one control:
  the same statistic on a resynthesis of the recording's own tracked partials (`renders/timbre-ladder`
  rung `01`), which has the recording's amplitudes and none of its noise.

---

## 6. Ranked, with the evidence

| # | finding | verdict | evidence | cost |
|--:|:--|:--|:--|:--|
| 1 | `horizontal_offset_hz` is a fixed hertz offset, so 0.27/0.35/0.52 Hz are beat rates of every partial of every key | **WRONG** | `JITTER.md` component tables (every row); `REALISM.md` worst modulation rate 0.6 Hz on 3/6 phrases | **S** — the value is 35× out and the *shape* is wrong; even the interim fix (make it a ratio, shrink it to `~0.01 Hz` at C4 k=1) is one line |
| 2 | the unison is built at `μ = ∞` where the instrument is at `μ = 0.27–0.8` over bass and midrange | **WRONG** | §3.2–3.3; Weinreich 1977; Woodhouse 2021 (anti-veering ⇒ no beats); 35/73 keys below `μ = 1` | **M** — §5, cached at preset load, `+1` FMA in the modal loop |
| 3 | the fixed-offset construction cannot be velocity-dependent at all | **WRONG** | recording 0.80 c / 2.27 dB of velocity spread against the engine's 0.008 / 0.048 | falls out of 2 |
| 4 | `unison_coupling` is ~25× too weak and phase-scrambled by the 2.667 ms block delay | **WRONG** | forensics: −0.07 c, −0.06 dB; `DECISIONS.md` 151 makes the same phase argument for the bus | deleted by 2 |
| 5 | `radiated_share = 0.5` and `horizontal_decay_ratio = 0.172` are mutually inconsistent by ~3× | **WRONG** | §2.6, arithmetic on the shipped preset | **S**, but blocks 2 |
| 6 | the scoreboard rewards the presence of modulation, not its kind | — | M3 improved modulation 5.76 → 3.84 by deepening a metronome | **S** — add `wRMS/raw` and the velocity spread of beat depth to `REALISM.md` |
| 7 | two banks per string, vertical fast / horizontal slow | **SOUND** | Weinreich (principal axes); Capleton ("about a quarter"); Woodhouse (double decay from polarizations alone) | keep |
| 8 | no cross-polarization coupling | **DEFENSIBLE-BUT-AUDIBLE** | Weinreich: reactive part near-isotropic, off-diagonal second order | defer until 1–5 are done |

**Build order:** 5 (reconcile the two decay claims), then 1 as an interim (it is a one-line change
that removes the instrument-wide 0.35 Hz pulse on its own, and is worth listening to *before*
building 2, because it isolates how much of the complaint is the polarization metronome and how much
is the unison), then 2 + 3 + 4 together as one change in `tuner/` prototype form, then 6.

---

## Sources

* Gabriel Weinreich, "Coupled piano strings", *J. Acoust. Soc. Am.* **62**(6), 1474–1484 (1977) —
  [abstract](https://pubs.aip.org/asa/jasa/article-abstract/62/6/1474/765090/Coupled-piano-strings)
* Gabriel Weinreich, "The coupled motion of piano strings", in *Five Lectures on the Acoustics of the
  Piano* (Royal Swedish Academy of Music, 1990) —
  [online](https://www.speech.kth.se/music/5_lectures/weinreic/motion.html)
* Brian Capleton, "False beats in coupled piano string unisons", *J. Acoust. Soc. Am.* **115**(2),
  885–892 (2004) —
  [PDF](http://persianney.com/misc/False%20beats%20in%20coupled%20piano%20string%20unisons.pdf)
* Jim Woodhouse, "A necessary condition for double-decay envelopes in stringed instruments",
  *J. Acoust. Soc. Am.* **150**(6), 4375 (2021) —
  [open version](https://api.repository.cam.ac.uk/server/api/core/bitstreams/ab932038-6866-48af-b7ab-0d31eef9c394/content)
* Jim Woodhouse, *Euphonics* §7.3, "Multiple strings and double decays" —
  [online](https://euphonics.org/7-3-multiple-strings-and-double-decays/)
* J. Bensa, S. Bilbao, R. Kronland-Martinet, J. O. Smith III, "The simulation of piano string
  vibration: from physical models to finite difference schemes and digital waveguides",
  *J. Acoust. Soc. Am.* **114**(2), 1095–1107 (2003) — [PDF](https://hal.science/hal-00088329v1/document)
* M. Aramaki, J. Bensa, L. Daudet, P. Guillemain, R. Kronland-Martinet, "Resynthesis of coupled piano
  string vibrations based on physical modeling", *J. New Music Research* **30**(3), 213–226 (2001) —
  [abstract](https://www.tandfonline.com/doi/abs/10.1076/jnmr.30.3.213.7472)

In-repo evidence: `renders/jitter/JITTER.md`, `renders/timbre-ladder/ANALYSIS.md`,
`renders/realism/REALISM.md`, `TUNING_REPORT.md` §§3, 5, 6, `PHYSICS.md` §4,
`DECISIONS.md` 38, 103–107, 148–152, `presets/salamander-c5.toml`.

---

<!-- Part II of FUNDAMENTALS.md — the perception-lens review. Part I (the physical-model review) precedes it when merged. -->

---

# Part II — The perception lens: why 4.9 dB still sounds sinusoidal

The scoreboard improved at M3 (mel 5.34 → 4.91 dB against a 1.59 floor, modulation 5.76 → 3.84 against 0.96, `renders/realism/REALISM.md`; DECISIONS 205, 220) and the careful listener's verdict got *worse*: still "sinusoidal and artificial", and now "jittering inside the note — mini frequency oscillations". Both are correct. This half of the review shows they are measurements of different things, audits the benchmark against the percept, audits the listening chain that let the divergence run for three milestones, and says what "sinusoidal" most plausibly still is. Every number below is in a file in this repository or is specified precisely enough to be put in one.

## II.1 Salience is not energy: what the ear foregrounds at levels the metrics cannot see

The artifacts the jitter forensics measured (`renders/jitter/JITTER.md`) are, in energy terms, nothing. In perceptual terms they are the foreground, because the auditory system detects *coherent modulation of a resolved partial* at thresholds two orders of magnitude below anything a spectral-energy distance responds to.

**Coherent slow FM.** Frequency difference limens measured with slow FM (the classic Shower & Biddulph curve, and modern low-rate FM detection data, e.g. Moore & Sek) sit at 0.2–0.3 % of the carrier for rates of 1–6 Hz — **3–5 cents on a steady mid-register tone**, lower for trained listeners, and lower again when the FM is the *only* thing moving (the engine's partials are otherwise stationary to 0.01 c, JITTER.md §0 controls). Against that threshold:

- A4 k=1, engine: **3.91 cents RMS** of instantaneous-frequency deviation in 0.1–20 Hz, against the recording's 0.87 (JITTER.md A4 table). This is not marginal; it is several times detection threshold, and it is the register where the complaint says "jittering inside the note".
- C4 k=1, engine: **0.08 cents** against the recording's **2.66**. The recording's own wobble is at or above threshold — its *absence* is equally audible, as purity. A fundamental that holds 0.08 c for three seconds is, within the measurement's floor (the `all_off` one-sinusoid-per-partial control reads 0.01), literally a decaying sinusoid, and "sinusoidal" is the technically correct word for it.

In spectral-energy terms both numbers are invisible: FM at 0.35 Hz puts its sidebands 0.35 Hz from the carrier — inside a single bin of the realism benchmark's longest window (4096 samples at 48 kHz = 11.7 Hz bins), and four orders of magnitude inside one mel band (64 bands over 20 Hz–16 kHz ≈ 59 mel each ≈ **150–230 cents wide** in the mid-register, against a 3-cent excursion).

**Regular deep AM.** Amplitude-modulation detection at low rates is a fraction of a dB (intensity JNDs ~0.5–1 dB). The engine's upper mid-register partials beat **15–17 dB deep** (C4 k=3: 15.19 dB, k=4: 17.31 dB, against the recording's 8.80 and 2.51 — JITTER.md C4 beat-depth table): fifteen to thirty times threshold, at metronomically fixed rates. Regularity is itself the cue: listeners detect deviations from isochrony at a few percent, so a beat train that repeats to **0.008 cents and 0.048 dB across velocities 40/90/120** (engine, against the recording's 0.80 c and 2.27 dB — JITTER.md velocity rows) reads as a machine even when a single period of it, heard once, would pass.

**Where the movement sits.** The recording's wobble rides on the partial while it is loud (power-weighted over plain RMS, `wRMS/raw`, median **0.58**); the engine's is a spike at the amplitude null of a beat (median **0.16**, and 0.03 at A4 k=1). The ear tracks the pitch of a note through its loud, resolved low partials (the dominance region, roughly partials 2–5 in this register — Plomp; Moore): movement there reads as a living string, stillness there reads as an oscillator. Movement concentrated at nulls reads instead as a flutter/dropout event — which is the reported percept, "mini frequency oscillations", at the reported location, "inside the note".

**The salience inversion, stated once.** The remaining mel distance is carried by per-partial and per-key *level* errors of ±5 dB and more (DECISIONS 215, 216) — energetically enormous, perceptually "a differently voiced piano", i.e. background. The FM/AM structure above is energetically sub-bin and perceptually foreground. A scoreboard built entirely from energy functionals will therefore improve while the percept worsens, which is exactly what happened at M3.

## II.2 The scoreboard, audited against the percept

Each metric, against what the complaint is made of:

- **Mel distance is FM-blind by construction, twice over.** It discards phase (magnitude spectrogram), and its bands are ~2 semitones wide in the mid-register against a 1–4 cent artifact. Any frequency modulation that keeps a partial's energy inside one mel band — which 4 cents always does — changes the feature vector by *exactly nothing*. Mel can fall forever while the wobble stands.
- **The realism modulation metric excludes the artifact's band.** `REALISM.md` defines it over **0.5–50 Hz**. The engine's beat-rate table (JITTER.md, "what the shipped preset builds") puts the dominant lines at **0.27, 0.35 and 0.52 Hz** — the fixed `horizontal_offset_hz` values of DECISIONS 38 — plus detune beats at 0.01–0.9 Hz. Most of the metronome sits *below the metric's low edge*. What moved the column 5.77 → 3.84 at M3 was the felt-limiter click fix (DECISIONS 218–220): broadband AM at the note rate, a genuine defect, removed — but nothing about the interior of a note. The column's improvement was honestly earned and honestly irrelevant to the complaint; worse, removing 252 clicks unmasked the quieter clockwork underneath, which is one reason the user hears the jitter *now*.
- **It also aggregates over bands and phrases.** Even inside 0.5–50 Hz, one partial's beat is averaged with 63 other bands over six phrases whose median note is short (144 notes in 14 s of `alberti_fast`). The ladder's per-partial version of the same measurement (0.1–20 Hz, `ANALYSIS.md` metric 3) *did* see the problem — engine 4.64 vs 2.58 dB at C4, 0.77 vs 2.38 at A2, flagged in §8 as "the most diagnostic single number" and again at M3's close as *not closed* (DECISIONS 213) — but it never became a scoreboard column, and the scoreboard is what the milestones were gated on.
- **Linewidth certified the wrong conclusion, for a knowable reason.** `ANALYSIS.md` §2 measures the −6 dB width of a 2-s FFT — a *power-weighted* statistic. The engine's IF spikes happen where the partial has no power (wRMS/raw 0.03–0.16), so they widen nothing; and the recording's genuine 2.66-cent wobble at C4 k=1 sits **under the 7.55-cent linewidth floor** of that partial's own decay rate (the floor table at the head of each key's section). Both signals therefore read "zero excess width" — which is true — and the file concluded "linewidth — refuted, nothing to close, do not build it" (§ "Which ingredients…", item 4). The refutation was right about rung 05's random-walk detune and wrong as a generalization: *no metric in the ladder measures instantaneous frequency at all*. Five metric families — roughness, linewidth, envelope modulation, attack level, attack flatness — and four of them are amplitude functionals; the fifth is the power-weighted FFT above. The f_k(t) axis of the ladder was never instrumented.
- **The mean hides a distribution error.** Over the forensics' 16 cells the engine's jitter *mean* (1.27 c) is close to the recording's (1.50 c). Any scoreboard mean over cells would have passed it. The percept is the distribution: 33× too still at C4 k=1, all of A2 dead (0.11–0.36 vs 1.15–1.70), 4.5× too much at A4 k=1, and everything concentrated at nulls. Scoreboard columns must be built as per-cell mismatch ratios, not pooled means — that is the design error to not repeat.

## II.3 Two scoreboard columns that would have caught this

Both are promotions of measurements that already exist in `tuner/examples/jitter_forensics.rs`; both satisfy `ANALYSIS.md` §7's own certification rule (far for the engine, near-zero for an exact resynthesis: rung 01 plays the recording's *measured* f_k(t) and a_k(t), so it inherits the recording's jitter and beat structure by construction, while the forensics' controls put the metric floor at 0.00–0.05 c). Both should be validated exactly as §7 validates: reference-vs-neighbouring-velocity-layer is the floor, rung 01-vs-00 must read inside it.

**Column A — per-partial instantaneous-frequency stability (`IF mismatch`, unitless; and `IF placement`, ratio).**

- *Cells:* keys {A2, C4, A4, C6} × partials k = 1..4, velocity 90, single held note ≥ 3.2 s, mono sum.
- *Measurement (as implemented in `jitter_forensics.rs`):* complex demodulation at the partial's own spectral peak through a Gaussian band-pass of 5 ms time constant (31.8 Hz, capped at carrier/4); phase derivative on a 1 kHz grid; deviation track over 0.3–3.0 s; J = RMS of the track restricted to 0.1–20 Hz, in cents. L = wRMS/raw: the same RMS power-weighted by the partial's instantaneous power, divided by the unweighted RMS.
- *Columns:* `IF mismatch` = geometric mean over the 16 cells of max(J_eng, J_ref)/min(J_eng, J_ref) — symmetric, so "too dead" fails as loudly as "too spiky"; a cell where both are under 0.1 c (both at floor) is scored 1. `IF placement` = median over cells of L_eng / L_ref.
- *Current values (computable from JITTER.md today):* `IF mismatch` ≈ **4.5** (driven by C4 k=1's 33×, A2's 5–10×, A4 k=1's 4.5×); `IF placement` ≈ **0.16/0.58 ≈ 0.3**.
- *Gate:* mismatch ≤ 2.0, placement ≥ 0.5. Either gate alone fails the shipped preset; both would have failed it at every milestone since the unison existed, i.e. this column catches the defect at M0, not M3.

**Column B — envelope-modulation determinism (`beat-depth error`, dB; and `velocity coherence`, ratio).**

- *Cells:* same 16, plus the same 16 at velocities 40 and 120 (both renders and reference layers — the reference velocity layers already exist and the forensics already read them).
- *Measurement:* band-limit the partial's log envelope to 0.1–20 Hz over 0.3–3.0 s; D = p95 − p5 in dB (beat depth). Per cell, S_J = max − min of J across the three velocities, S_D = max − min of D.
- *Columns:* `beat-depth error` = mean over cells of |D_eng − D_ref| (current value from JITTER.md's table: **6.8 dB**, with per-cell errors spanning −9.1 to +14.8 — the mean's small signed value, +2.0, is exactly what must *not* be reported). `velocity coherence` = (mean S_eng)/(mean S_ref), pooled over J and D. Current value: **0.008 c/0.80 c ≈ 0.01** on frequency, **0.048 dB/2.27 dB ≈ 0.02** on depth.
- *Gate:* beat-depth error ≤ 3 dB; velocity coherence ≥ 0.25.
- *One deliberate exclusion:* modulation-spectrum *flatness* is not a column. The forensics established that a deep periodic beat spikes the track at every null and a spike train is broadband however regular it is (JITTER.md, flatness caveat), and that autocorrelation-based periodicity scores fail on 2.7 s of a 3–20 s beat period (a statistic that scored 0.78–0.89 on the one-sinusoid control was measured and removed). Velocity invariance carries the "metronome" claim instead, and carries it harder: nothing stochastic or amplitude-coupled can hold 0.008 c across an 80-point velocity span.

`velocity coherence` is the column with the physics in it: Weinreich coupling makes the unison's eigenmode structure depend on the relative amplitudes and phases the hammer hands the strings, so a coupled unison *cannot* be velocity-invariant, and a free-running one cannot be anything else.

## II.4 The listening evidence chain, audited end to end

What the repository's auditable record says was available to hear, milestone by milestone, and where each A/B could not have exposed the defect:

1. **`renders/salamander-ab/` (DECISIONS 96, Phase D/E).** Eight single notes (A0–A3, C4–C7), demo, pedal phrase — all velocity 90, engine-both-presets plus source. Level-matched on first-second RMS (correctly, per item 96's own reasoning). *Gap:* one velocity; no repeated-note material; single notes but the listener was steered at preset-vs-preset differences (decay tracking), not engine-vs-recording note interiors.
2. **`renders/timbre-ladder/` (the instrument that steered M2/M3).** Three keys — C4, A2, C6 — ten rungs, **4-second files**, velocity 90, RMS-matched over 0.2–2 s. Three structural problems, none noted in the README's "How to listen":
   - *Duration.* The engine's guaranteed amplitude-equality crossings sit at **2.43–2.86 s** (JITTER.md component tables: C6 k=1 at 2.43 s, A4 k=1 at 2.73 s, C4 k=2 at 2.51 s) and the beat periods are 2–20 s. A 4-s cut contains at most one null, partly under the fade. One dip heard once is a piano; the same dip at the same millisecond on every audition is a machine — and only the second thing is the artifact. No file the user was given was long enough to state the regularity, and every re-listen of a deterministic render *is* the repetition that would state it, except that listeners attribute sameness-across-replays to the file, not the instrument.
   - *Register.* The ladder never rendered anything between C4 and C6. The forensics put the wobble's maximum at **A4 k=1 (3.91 c, full-equality v0/h0 crossing at 0.35 Hz, `under dB = 0.0`)**. Every M3 fitting decision generalized from three keys that bracket but do not contain the worst register.
   - *Level window.* The 0.2–2 s RMS window ends before the 2.4–2.9 s crossings, so rungs are matched on their pre-null seconds. Minor, but it means the one audible event in the beat cycle is also the one event outside the loudness match.
3. **`renders/realism/` (the scoreboard's own audio).** Six phrases, engine vs `piano_tuner::sampler`. Two deeper problems:
   - *Phrase choice.* No phrase holds a single exposed mid-register note for the 3+ seconds the artifact needs. `alberti_fast` is 144 notes in 14 s; `staccato` is 80 ms notes; `scale_mf` walks on; `chords_pedal` holds chords, whose dense partial fields mask a single partial's beat. The one gesture the complaint describes — one note, middle of the keyboard, held and listened to — is in no phrase, so the scoreboard's mean is a mean over material that structurally underweights the defect.
   - *The reference is itself deterministic.* The ground truth is a sampler replaying fixed recordings. Play the same event list twice and the reference is bit-identical too. Therefore **no engine-vs-reference comparison in this repository, aural or numerical, can ever expose determinism** — the property the velocity-invariance measurement shows is the engine's most un-piano-like one. Determinism is only visible across velocities (recording layers differ: spread 0.80 c / 2.27 dB) or against a live instrument. Until the jitter set, no cross-velocity render pair existed.
4. **Rung 05's refutation over-propagated.** `ANALYSIS.md` correctly convicted the 2-cent per-partial random walk (independent per partial, adds >5 Hz energy the recording lacks, six of six metrics away — "do not build it"). The recorded conclusion, "linewidth/FM: nothing to close", then stood while M3 was steered entirely at amplitude tables (`partial_gains`, `partial_sigma_scale`, `comb_floor`, strike noise). The ladder had **no rung between 01 and 07 that varied frequency behavior only** — no "engine amplitudes + measured f_k(t)" rung, no "coupled unison" rung — so the one axis on which 01 and 07 still differ was neither instrumented (II.2) nor auditioned in isolation. That is the hole in the chain: 01 was correctly identified as "sounds like the recording", 07 as not, and every *measured* difference between them was chased while the unmeasured one survived untouched.

No A/B was mis-leveled; item 96 and the ladder both got the level discipline right, and the jitter set verifies RMS match to four decimals. The chain's failures are duration, register, velocity, phrase choice, and a reference that cannot testify about determinism.

## II.5 Triage: what "sinusoidal" most plausibly is, after M3

Rung 01 closes roughness (94–103 % of the gap), linewidth (to 0.01 c), and modulation energy (to 0.009–0.040 dB) — `ANALYSIS.md` §"Does rung 01 close the gap". What 07 still differs from 01/00 in, post-M3 (the ladder was re-measured at DECISIONS 213 and is bit-identical after 220), ranked by the perceptual weight established in II.1:

1. **Frozen fundamentals in the pitch-carrying register — the "sinusoidal" itself.** Engine C4 k=1: envelope modulation **0.18 dB** vs the recording's **3.05**; IF jitter **0.08 c** vs **2.66**; beat depth **0.41 dB** vs **9.46**. All of A2 k=1–4: 0.24/0.26/0.53/1.02 dB vs 0.48/1.33/3.36/3.19, and 0.11–0.36 c vs 1.15–1.70. The partials the ear takes pitch and "aliveness" from are stationary to the measurement floor. This is the largest single 07-vs-01 difference on the most diagnostic metric and it sits exactly where pitch dominance sits. M3 did not touch it because no amplitude table can: the recording's fundamental movement is the *composite* of coupled strings trading energy, and the engine's C4 fundamental components decay so uniformly that their beats stay shallow (the detune pattern's incommensurate rates, DECISIONS 38, spread the nulls away — at the price of spreading away the movement too).
2. **The clockwork where movement does exist — the "jittering".** C4 k=3/4 and A4 k=1: beats 15–17 dB deep at preset-fixed rates, IF spikes at the nulls (wRMS/raw 0.03–0.16), the whole structure identical at every velocity, and **one rate — 0.350 Hz — present in every partial of every key** because `horizontal_offset_hz` is a fixed number of hertz (0.35/0.52/0.27, DECISIONS 38): the entire compass shares one metronome, so any chord beats in unison with itself. M3 *did* touch this, in both directions: `partial_sigma_scale` moved the equality crossings into the 0.3–3 s window at A4 (3.91 → 1.55 c when the table is removed) while improving C4 k=1 slightly, and the felt-click fix removed the broadband bed that had partially masked it. Hence "NOW hears": M3 sharpened and unmasked a defect that predates it — the bisection convicts `detune_cents` + `horizontal_offset_hz` + `unison_sigma_scale` on an uncoupled unison (−1.65 c / −12.1 dB for `no_detune`, matched by `single_string`), and clears M3's `partial_gains` outright (+0.08 c).
3. **Deterministic line-structure even in the movement.** Where the engine's envelopes do move, they move as one or two discrete modulation lines (flatness −34.7 dB at C4, −60.1 at A2, vs the recording's −25.4 and −30.9): the movement it has is *more periodic* than the piano's even before its depth is wrong.
4. **The late-treble continuum (C6 >5 Hz: 1.80 vs 2.38 dB, and rung 01 itself only reaches 1.12)** — real, smaller, and partly a tracker ceiling rather than an engine defect; and **the attack, now closed** (flatness gap 0.4–3.4 dB, level −5.0 vs −5.1; DECISIONS 213) — no longer a plausible carrier of "sinusoidal", whatever it contributed before M3.

The through-line of 1–3 is a single structural fact, confirmed by the forensics and unreachable by any preset table: every partial is 2–6 **free-running** sinusoids at preset-fixed offsets with independent decays, where the real unison is a **coupled** system (Weinreich, JASA 1977) whose eigenmodes shift in frequency, split in decay rate, and depend on the strike — and the engine's `unison_coupling` (a one-block-late excitation cross-feed of bridge force, `string.rs::couple`; DECISIONS 33) measurably does not produce any of that: zeroing it moves the jitter by **−0.07 c**, i.e. the coupling parameter does not couple, consistent with DECISIONS 151's finding that nothing proportional to a string's own motion feeds back anywhere in the engine. The fix therefore lives in `engine/src/string.rs`'s eigenstructure (out of scope for this review's read-only phase; the model half of this document carries it), and its acceptance test is already written: Column A and Column B of II.3, gated per cell, at three velocities.

**Falsifiable summary.** If the mechanism above is the percept, then: (a) `renders/jitter/A4/02_no_detune.wav` vs `01_engine.wav` removes the "jittering inside the note" (3.91 → 0.15 c) at the cost of a deader tone; (b) `renders/jitter/C4/01_engine.wav` vs `00_recording.wav` exposes "sinusoidal" as the frozen fundamental (0.08 vs 2.66 c) independent of any beat; and (c) no listening comparison confined to velocity 90 and 4-second files can distinguish the shipped engine from one with a genuinely coupled unison whose eigenmodes happen to match at that one velocity — which is precisely why the scoreboard needs II.3's columns and not another audition.

---

*Status: review conducted read-only over `engine/src/string.rs`, `TUNING_REPORT.md`, `renders/timbre-ladder/ANALYSIS.md`, `renders/realism/REALISM.md`, `renders/jitter/JITTER.md`, and DECISIONS 33, 38, 96, 103–107, 148–152, 187–222. `engine/` untouched. `cargo test --release` in `tuner/`: 238 passed, 0 failed (202 lib + 19 calibration + 3 decoding + 6 estimators + 7 tracking + 1 doc).*


## 7. PROTOTYPE VERDICT — §5 built, rendered and measured

`tuner/examples/eigenmode_prototype.rs` builds the construction of §5.1 offline, renders C4, A2
and C6 through it, and runs `JITTER.md`'s own measurement code on the result.
`renders/jitter/EIGENMODE.md` is its output; `renders/jitter/eigenmode/<note>/NN_*.wav` and
`renders/jitter/eigenmode_{C4,A2,C6}.wav` are the level-matched listening set. `engine/` is
untouched. `cargo test --release` in `tuner/`: 238 passed, 0 failed.

The rig has one control the earlier files did not: **`02_modal_shipped`** is the *same* offline
renderer — same hammer, same soundboard, same panning, same block grid — carrying
`PianoString::new`'s construction. Its numbers track the shipped engine's cell for cell — C4 k=1 jitter 0.08
against 0.08, A2 beat depth 0.38/0.53/1.27/3.13 against 0.38/0.53/1.30/2.95, C4 placement
0.58/0.06/0.03/0.06 against 0.59/0.07/0.03/0.06 — so `03_eigenmode − 02_modal_shipped` is the
eigenproblem and nothing else, and the residual difference between rows `01` and `02` is the size of
everything the offline rig leaves out (the duplex, the sympathetic bus, the mechanism noise).

### 7.1 The verdict, in one line

**The construction is correct, it is cheap, it deletes the artefact's mechanism — and it does not
close the gap to the recording. It gets three of §5.5's six predictions, refutes two of them outright,
and in doing so proves that the recording's frequency movement is not a unison beat at all.**

### 7.2 What it does

| | shipped engine | eigenmode | recording |
|:--|--:|--:|--:|
| C4 k=2 jitter, cents | 1.38 | **0.44** | 2.75 |
| C4 k=2 beat depth, dB | 8.88 | **3.86** | 13.89 |
| C4 k=4 beat depth, dB | 17.30 | **12.36** | 2.51 |
| C6 k=1 beat depth, dB | 17.32 | **12.63** | 11.37 |
| B1 beat-depth error, mean abs dB over 12 cells | 6.74 | **5.91** | 0 |
| A2 placement `wRMS/raw`, mean | 0.42 | **0.43** | 0.55 |

Three things it gets, all structural:

1. **The 0.270 / 0.350 / 0.520 Hz metronome is gone by construction** (§5.5 prediction 3). The
   polarization split is now `N γ_v β ε / 2π`, proportional to ω: at C4 k=1 the three
   vertical modes come out at −0.0167 / +0.0761 / +0.1528 Hz and their horizontal partners at
   −0.0239 / +0.0870 / +0.1565 Hz, i.e. **0.004–0.011 Hz apart** against the shipped 0.35 Hz flat,
   and it scales with the partial instead of being the same number on every partial of every key. Nothing in the instrument can beat at a note-independent rate any more.
2. **The guaranteed full null is gone** (§2.4, §5.5 prediction 4). At C4 k=1 the loudest eigenmode
   stands **16.8 dB** over the next and the six modes' decay rates run 0.62 → 0.14 s⁻¹, a **4.6×
   split**, where the engine's are identical to the bit. Measured beat depth 0.17 dB. At C6 the
   `under dB = 0.0` crossings `JITTER.md` found at k = 1…4 do not occur.
3. **The decay-rate split, the frequency contraction and the radiated weights all fall out of one
   coupling constant.** `unison_sigma_scale` is not read, `unison_coupling` is not read,
   `horizontal_offset_hz` is not read, `vertical_decay_factor` is not read. Three preset fields
   deleted and one derived, exactly as §5.4 said.

**Cost, measured.** The per-key solve is 67.4 ms (C4, 58 partials × 6 modes), 82.0 ms (A2, 80 × 4),
19.7 ms (C6, 16 × 6) — dominated 5:1 by the T60 normalisation's envelope search, which at 800 grid
points instead of 4000 costs 14.4 ms and moves the answer by 2.5 % (scales 2.912/2.868/2.427/2.621
→ 2.892/2.830/2.401/2.687). So the whole compass is ~1.3 s at preset load as written and well under
100 ms with a secant solve. The eigensolve itself is a degree-`N` complex polynomial per
polarization block — no LAPACK, no matrix inversion — because `C_k` is block diagonal with a
rank-one block, and `D − cJ` is complex symmetric so the row of `V⁻¹` the strike projection needs is
`v_m / (v_m · v_m)`.

### 7.3 What it does not do — the honest negative

**Refuted (1): §5.5 prediction 1 and 2, and with them Column A.** The
instantaneous-frequency mismatch against the recording goes **3.39 → 4.37** — *worse*. The
construction makes the mid and low register **stiller**, which is the direction the forensics
already said was wrong:

| cents, 0.1–20 Hz | recording | engine | eigenmode |
|:--|--:|--:|--:|
| C4 k=1 | **2.66** | 0.08 | **0.11** |
| A2 k=1 | **1.44** | 0.22 | **0.04** |
| A2 k=2 | **1.15** | 0.11 | **0.05** |
| A2 k=1 beat depth, dB | 0.81 | 0.38 | **0.01** |

This is not a failure of the prototype; it is the coupled model working. Woodhouse's anti-veering is
explicit — "with anti-veering there will be no beats" — and 35 of 73 multi-string keys are below
`μ = 1`. The coupling locks the bass and midrange unison, and the recording goes on moving 1.4–2.7
cents anyway. **Therefore the recording's frequency movement in the bass and midrange is not a
unison beat, coupled or free.** §3.2's regime argument was right about the physics and wrong about
what it would buy.

**Refuted (2): §5.5 prediction 2 in particular — velocity dependence is structurally unreachable
here too.** Engine velocity spread 0.006 cents, eigenmode **0.010**, recording **0.787** (mean over
12 cells; C4 k=1 alone: 1.61 cents, C6 k=4: 3.45). §5.1 predicted this would come from `ε_j` and
`s_j` in the strike vector — but `u = s_j g_k e^{−iω d_j}` scales *uniformly* with velocity, so
`c = V⁻¹u` scales uniformly too and every ratio in the mode mixture is a constant. The eigenmodes
are velocity-invariant by eigenvalue and the mixture is velocity-invariant by linearity. Nothing in
§5 can produce velocity-dependent beating. The only handles that could are a strike *vector* whose
**direction** moves with velocity — per-string shares, the timing skew `d_j`, or the horizontal leak
— and all three are constants in the preset today.

**Robustness.** The one constant the literature does not pin is `β = Im Y / Re Y`, which decides
whether the coupling attracts the group's frequencies (Woodhouse's resistive-dominated anti-veering)
or repels them. The whole construction was re-solved and re-rendered at `β ∈ {0, 0.25, 1, 3}`:

| `Im Y / Re Y` | A1 mismatch | C4 k=1 c | C4 k=2 c | A2 k=1 c | C6 k=1 c | C4 k=2 depth dB |
|--:|--:|--:|--:|--:|--:|--:|
| 0.00 | 4.46 | 0.01 | 2.73 | 0.04 | 1.32 | 13.32 |
| 0.25 | 4.15 | 0.10 | 1.36 | 0.04 | 1.47 | 10.70 |
| **1.00** | **4.37** | 0.11 | 0.44 | 0.04 | 2.26 | 3.86 |
| 3.00 | 6.28 | 0.07 | 0.22 | 0.04 | 1.13 | 1.73 |

Every value is worse than the engine's 3.39, and the C4 and A2 fundamentals are dead (0.01–0.11 and
0.04 cents against 2.66 and 1.44) at every one of them. The negative result is not a choice of `β`.

**And one thing it breaks.** The double decay survives at C4 and A2 — mean error in the aftersound
level (where the tail's straight line extrapolates back to at the strike, relative to the prompt's)
goes 19.1 → 15.4 dB at C4 and 13.2 → 9.8 dB at A2 — and **breaks at C6**, 4.9 → 21.2 dB. C6's
fitted `detune_cents` is 1.947, `μ = 1.74`, so its three vertical modes come out within 7.3 dB of
each other and there is no quiet slow survivor left to be an aftersound. That is not an argument
against the construction; it is the first concrete consequence of the fact that `detune_cents` was
fitted *through the free-running forward model* and cannot be carried across unchanged.

### 7.4 What the recording's composite partials are actually doing

The prototype's real contribution is that, by removing the unison beat completely, it isolates what
is left. Two statistics added here say what that is.

**The companion each partial implies.** Inverting the measured beat depth for the amplitude ratio of
a two-component pair (`D = 20 log₁₀((1+r)/(1−r))`), beside the rate the envelope's own sign changes
imply — how loud the second component is, and how far away, in the units the preset is written in:

| implied companion, dB / Hz | k=1 | k=2 | k=3 | k=4 |
|:--|--:|--:|--:|--:|
| C4 recording | **−6.1 / 1.11** | **−3.6 / 1.48** | −6.6 / 0.74 | −16.9 / 0.74 |
| C4 engine | −32.5 / 1.48 | −6.5 / 0.93 | −3.1 / 0.74 | −2.4 / 0.74 |
| C4 eigenmode | −40.0 / 0.74 | −13.2 / 0.74 | −4.6 / 0.74 | −4.3 / 1.11 |
| A2 recording | −26.6 / 1.11 | −21.7 / 1.48 | **−5.5 / 0.74** | **−5.6 / 0.74** |
| A2 engine | −33.1 / 1.11 | −30.4 / 0.74 | −22.6 / 0.74 | −15.5 / 0.74 |
| A2 eigenmode | −65.2 / 2.96 | −58.0 / 0.74 | −33.1 / 0.74 | −19.6 / 0.74 |
| C6 recording | −4.8 / 2.22 | −6.1 / 4.07 | −2.6 / 4.44 | −8.1 / 5.19 |
| C6 eigenmode | −4.1 / 1.48 | −3.5 / 1.85 | −3.2 / 2.59 | −2.2 / 4.07 |

Read the recording's rows against `k`. A beat from a **unison mistuning** is a frequency *ratio*, so
its rate must be proportional to `k`; a fixed-hertz offset gives the same rate on every partial. The
recording's C4 rates are 1.11, 1.48, 0.74, 0.74 Hz and A2's are 1.11, 1.48, 0.74, 0.74 Hz —
**flat in `k`, not proportional to it**, and 7–20× wider than the fitted unison spread (C4's whole
detune is 0.984 cents = 0.149 Hz at k=1; A2's is 0.048 Hz). Only C6 rises with `k` (2.22 → 5.19),
and there the fitted detune is genuinely wide.

So the recording's mid and low partials each contain **a second component 4–7 dB down sitting
0.7–1.5 Hz away, at a spacing that does not scale with the partial number**. That is:

* **not the unison** — wrong size by 7–20×, and wrong `k` dependence;
* **not the bridge's polarization split** — §2.2 derives 0.010 Hz from the measured admittance, and
  this is a hundred times larger;
* **not the engine's `horizontal_offset_hz` either** — right order of magnitude in *rate* (0.35 vs
  ~1 Hz), but 22 dB out in *level* (−27.6 dB against the implied −6 dB), and constant where the
  measurement is merely uncorrelated with `k`.

What it looks like is a **within-string** split at near-equal amplitude — the two transverse planes
of one string at genuinely different frequencies, which is Capleton's actual subject ("**False
beats** in coupled piano string unisons", JASA 115(2), 2004) and which the review above used only
for its bridge-admittance numbers. A false beat comes from the wire's own geometry — non-uniform
diameter, an out-of-round or twisted cross-section, an asymmetric bridge-pin termination — not from
the bridge's mobility, so §2.2's refutation of 0.35 Hz *as a bridge effect* stands and is simply
about the wrong mechanism. Three of the recording's fingerprints follow from it and from nothing
else in the model:

* the two components are **comparable in amplitude early**, which is why the recording's wobble
  rides the loud part of the partial (`wRMS/raw` 0.55 mean, 0.98 at A2 k=1) instead of spiking at a
  null;
* the split is a property of the **individual string and the individual partial**, so it is
  uncorrelated across `k` and across notes — no metronome, and an envelope-modulation spectrum that
  reads as a continuum (recording −27.6 / −16.4 / −33.1 dB flatness at C4 against the engine's
  −63.1 / −38.6);
* how much of each plane the hammer excites depends on **how the hammer meets the string**, which
  is the one thing that changes with velocity — over these 12 cells the recording's velocity spread
  is **0.787 cents and 1.90 dB** of beat depth against **0.006 / 0.054** for the engine and
  **0.010 / 0.023** for this prototype.

The two candidates this does *not* rule out, and the controls that would: **(i)** the recording's
own room and the sampler's stereo pair — test by running the same statistics on
`renders/timbre-ladder` rung `01`, which carries the recording's measured per-partial amplitudes and
none of its background (§5.5's second falsifier, still unrun); **(ii)** the string's geometric
nonlinearity, i.e. pitch following amplitude — argued against here by the AM–FM regression, which
comes back **negative** on 7 of 12 recording cells (C4 k=1 r = −0.77 at −0.77 cents/dB) where
tension modulation would force it positive, and positive on 9 of 12 engine cells.

### 7.5 Does this replace `string.rs`'s construction?

**Yes, but it is not the fix for the complaint, and shipping it alone would make the instrument
audibly stiller in exactly the register the user is listening to.** The three claims separate:

| claim | verdict |
|:--|:--|
| the eigen construction is the physically right one | **upheld** — and it is cheaper than what it replaces: same mode count, one extra FMA, three preset fields deleted, `couple()` and its three per-group `BLOCK` buffers gone |
| it removes the artefact the user reported | **upheld for the metronome** — the note-independent 0.27/0.35/0.52 Hz pulse and the scheduled full nulls cannot exist under it |
| it reproduces the recording | **refuted for the construction alone** — A1 3.39 → 4.37, velocity spread 0.010 against 0.787, and the C4/A2 fundamentals go from too still to stiller. Steps 2 and 3 below are what closes it, and they close it: all four columns pass at A1 1.39 / A2 0.92 / B1 1.67 / B2 1.03 (`DECISIONS.md` 253), with A2 k=1 at 1.35 cents against the recording's 1.44 where the construction alone read **0.04** |

The build order that follows is therefore **not** "ship §5 and re-measure". It is:

1. **Ship §5's construction** — for the metronome, for the deleted fields, and because every later
   experiment needs a forward model that is not wrong. Accept that it *reduces* movement.
2. **Add the within-string split as a real mechanism**, on top of it — **built and then solved on
   the render** (`DECISIONS.md` 233–234, 249–252): a per-string, per-partial
   frequency offset between the two polarization blocks of the *same* string, of order 1 Hz in the
   bass and midrange, with the two planes at comparable amplitude (implied −4 to −7 dB, against
   `horizontal_gain_db`'s −27.6). In the eigenproblem this is one more term on the diagonal of
   `Ω_k`, not a new solver: `ω_(j,h) = ω_(j,v)(1 + δ_j(k))`. It is the only candidate the
   measurements support, and it is falsifiable — if `δ` fitted per string and per partial does not
   come back uncorrelated across `k`, it is not a false beat. (It comes back uncorrelated at 24 of
   30 sampled keys and the other six are refused. What did **not** survive contact with the render
   is the *open-loop* inversion of the level from the recording's own depth: the asked level is
   quoted against one polarization block and the depth is measured on all `2N` modes, so the two
   disagree by up to 16.4 dB and `B1` was a mean of that. The level is now bisected, and the rate
   stepped, against the engine's own render — `estimate::motion::FalseBeatLoop`.)
3. **Make the strike vector's *direction* velocity-dependent**, which is the only place velocity
   dependence can enter a linear model — **built**, `voicing.strike_direction` (`DECISIONS.md`
   235–236): `s_j(v)`, `d_j(v)`, and the split of energy between the two
   planes as a function of hammer speed. Without this, nothing in this family of models can produce
   the recording's 0.787 cents of velocity spread, and no amount of fitting will change that.
   (`d_j(v)` is not among the three fields: the timing skew is not a phase and was dropped outright,
   `DECISIONS.md` 227. What was built is the v/h ratio and the per-string share tilt, both
   interpolated in velocity, both leaving `|u|` alone.)
4. **Re-fit `detune_cents` under the new forward model** — the shipped tables were fitted through
   the free-running one, and C6's aftersound breaking (4.9 → 21.2 dB) is the first place that bites.
   **Attempted and largely refuted**, `DECISIONS.md` 242: at 28 of the library's 30 sampled keys the
   recording's own companions come back *flat in `k`*, so the beat that is there is the wire's and
   not the tuning's and there is no beat rate left to invert; where the rates do track `k` the
   inversion runs and moves one key. The aftersound was tried as the objective in its place and does
   not carry a fit — swept over the tuning it runs over ranges of 13 to 96 dB with no monotone shape
   — so it is *reported* against the recording's own, key by key, which is this step's check
   delivered as a measurement rather than as a fit.

### 7.6 Migration sketch — construction time only

Nothing below touches the audio thread's structure.

* **`Preset::load`** gains a per-key cache: for each partial, `2N` entries of
  `(f_m, σ_m, G_m)` where `G_m = (w·v_m)(v_m·u)/(v_m·v_m)` is complex. Built by
  `partial_modes` + `decay_scale` as prototyped — 88 keys × ≤80 partials × ≤6 modes × 6 floats
  ≈ 1 MB, ~0.1–1.3 s depending on the T60 solver, measured above.
* **`PianoString::new`** stops computing frequencies and sigmas. It reads the cache and calls
  `push_mode` the same `2N × K` times it does now. `strike_share` moves out of the caller and into
  `u`; the timing skew moves out of `Hammer::add_pulse`'s `skew` argument and into `u` as a phase.
* **`ModalBank`** gains a complex input gain — `g_im` beside `g_re`, one extra FMA in
  `Chunk::step` (`im += a_re·im + a_im·re + g_im·x`). The prototype's `EigenBank` is that
  recurrence in `f64`; the arithmetic is identical.
* **Deleted**: `PianoString::couple`, `Polarizations::previous`, `group_previous`,
  `MAX_UNISON_COUPLING`, `Voicing::vertical_decay_factor`, `voicing.unison_coupling`,
  `voicing.horizontal_offset_hz`, `voicing.unison_sigma_scale`. The unison group stops being a
  feedback loop, so the loop-gain contract in `string.rs`'s module doc goes with it.
* **Unchanged**: the hammer, the soundboard, the resonance bus, the duplex, the damper profile, the
  strike comb, the contact taper, `partial_gains`, `partial_sigma_scale`, the panning. The
  prototype drives the real `Hammer` and the real `Soundboard` and changes neither.
* **One new preset field** where three were removed: the bridge's reactive-to-resistive ratio `β`
  (dimensionless, order 1) and its anisotropy `ε` (0.075). `radiated_share` is promoted to the
  coupling constant and must be **re-fitted or derived**: this prototype derives it as
  `1 − horizontal_decay_ratio = 0.828`, because the slowest mode radiates nothing and therefore
  decays at `(1 − share)σ_k`, which makes `1 − share` the fitted aftersound/prompt ratio. That is
  §2.6's contradiction resolved in favour of the field that was fitted to recordings, and it is on
  Woodhouse's side of it (body coupling ≫ air damping across the midrange).

### 7.7 What the tuner would fit under the new construction

| today | under §5 |
|:--|:--|
| `notes.detune_cents` — fitted to make the free-running beat rates match | **still fitted, but re-fitted**: the beat rate is no longer `Δf`, it is `Δf` after anti-veering, so the map from tuning to beat is now a solve and not an identity |
| `voicing.unison_layout.share` | **fitted harder** — it now sets the aftersound *level* through the antisymmetric projection, so it is identifiable from the tail and not only from the attack |
| `voicing.unison_sigma_scale` | **gone** — the per-string decay split it existed to write in is an output |
| `voicing.horizontal_offset_hz` | **gone**, replaced by the bridge's `ε` (one global number) **plus**, if step 2 above survives its own test, a per-string per-partial `δ_j(k)` fitted from the measured beat rates — which is where the tuner's effort should go, because that is where the recording's energy is |
| `voicing.horizontal_gain_db` | **gone as a global** — the vertical/horizontal amplitude ratio becomes the hammer's leak, and the measurements say it is near −5 dB in the midrange, not −27.6 |
| `voicing.unison_coupling` | **gone** — it is `radiated_share × σ_k` |
| `voicing.bridge.radiated_share` | **the one number the construction turns on**; fit it against the *aftersound/prompt decay ratio*, which is directly measurable and is what pins it |
| `notes.sigma0`, `sigma1`, `partial_sigma_scale`, `partial_gains` | **unchanged**, and now anchored by the per-partial T60 normalisation rather than by one global `vertical_decay_factor` |
| — | **new, and the one worth building the estimator for**: the velocity dependence of the strike vector. Measure `wRMS/raw` and the beat depth at three velocities per key, which the forensics harness already does, and fit `s_j(v)` / `d_j(v)` to it. This is the only parameter family in the model that can move the number that is currently 0.010 against 0.787. — **built**, `estimate::motion::fit_strike_direction` + `SwingLine` (`DECISIONS.md` 241): the recording gives the sign, the engine gives the size, because a beat depth saturates and the column the field exists to move is a *spread*. `B2` reads **1.011** against the 0.25 gate. |

**Acceptance tests for any of this**: the perception review's Column A (`A1` IF mismatch, gate < 2.0;
`A2` placement, gate > 0.5) and Column B (`B1` beat-depth error, gate < 3 dB; `B2` velocity
coherence, gate > 0.25× the reference). Current standings over 12 cells — engine 3.39 / 0.42 / 6.74 /
0.006, eigenmode 4.37 / 0.43 / 5.91 / 0.010, reference 1.00 / 0.55 / 0 / 0.787. Not one of the four
passes for either construction, and `B2` is the one that says why.

**Built, and all four pass**, over Part II's own 16 cells (`tuner/src/realism.rs::motion_columns`,
`renders/realism/REALISM.md`, `DECISIONS.md` 244, 249–253). `presets/salamander-c5.toml` at three
stages:

| | `A1` ≤ 2.0 | `A2` ≥ 0.5 | `B1` ≤ 3 dB | `B2` ≥ 0.25 |
|:--|--:|--:|--:|--:|
| the eigen construction alone | 4.34 | 1.31 | 5.64 | 0.127 |
| + the motion mechanisms, level inverted open-loop from the recording | 2.93 | 1.32 | 3.88 | 1.011 |
| **+ the level and rate solved on the render (`FalseBeatLoop`)** | **1.39** | **0.92** | **1.67** | **1.03** |
| the recording against itself | 1.00 | 1.00 | 0.00 | 1.00 |

The third row is that milestone's reading and is left at it. On the shipped preset as it stands after
the disciplined gains refit and the master-gain calibration (`DECISIONS.md` 273-274, 277-278) the same
four read **1.375 / 0.866 / 1.643 / 1.050** — all four still pass, and the master gain cannot move them
at all, since every column here is a ratio between the engine and the recording.

Every one of the four is the two mechanisms; the gains move no column and the mechanisms move no
energy metric. What the open-loop inversion could not do was *arrive*: the level is quoted against
one block's coherent sum and the depth is measured on the whole partial, so A4 k=3 was asked for the
companion its recorded 3.19 dB implies and rendered 19.58 (`DECISIONS.md` 250). Two things had to
change with it — the schema's −20 dB level floor, which made the dead fundamentals unwritable by
construction (item 249), and the objective the *rate* is solved against, because at A2 k=1 the
recording's depth and its frequency deviation imply rates 4.8x apart and a two-component partial has
only one (item 251).

`B1`'s residue is now one thing and it is not this mechanism: **57 of 128 candidate partials get no
row at all** because the coupled unison already beats deeper than the piano does, and the four of
them in the cell set carry 1.14 of the column's 1.67 dB. That is `voicing.unison_layout.share`, one
row above.


---

# Verification errata

An independent verifier reproduced every load-bearing number in this file (jitter tables, eigenvalues against the closed-form 2x2 solution, preset-derived quantities, literature quotes) and judged the chain trustworthy. Four corrections it recorded, kept here rather than silently edited in:

- **Column A/B cell sets:** Part II defines its baselines over 16 key x partial cells (including A4); section 7 quotes them over 12 (C4/A2/C6), which is why the two A1 baselines differ (~4.5 vs 3.39). An implementation must pin the cell set, and clamp per-cell values at the measured 0.05-cent floor before any max/min ratio.
- **`radiated_share`:** the mu census and the 0.010 Hz derived polarization split in sections 3-4 use the shipped 0.5; section 7 resolves the field to 0.828, under which mu shrinks x0.60 (>= 35 of 73 keys below 1) and C6's quoted mu = 1.74 becomes ~1.05 — softening, not overturning, section 7.3's explanation of the C6 aftersound break.
- **Prediction bookkeeping:** section 7's "three of six predictions won" is strictly: predictions 3 and 4 confirmed, 1 and 2 refuted, 5 and 6 not evaluated; the third win (C4 k=2) was not a numbered prediction.
- **Orphan numbers:** section 4.2's "wRMS/raw up to 1.48" matches no published cell (max 1.15 at vel 90); section 3.3's 3.3x radiated-weight ratio is sensitive to unstated skew-phase assumptions (a zero-skew rederivation gives ~14x; the qualitative claim stands).

A later review of the *implementation* added two corrections of its own, both recorded with their measurements in `DECISIONS.md` rather than edited in here:

- **The equivalence contract** the shipped construction is held to is 0.5 cents of pitch, 5 % of whole-note T60 and 0.5 dB of strike level on `presets/default.toml`. Pitch and level meet it as stated (worst 0.073 cents at construction, +0.49 dB from A0 to C6, with C7 and C8 pinned at the +1.0 / +3.0 they cost). **T60 does not meet it cell by cell and cannot**, because the statistic is the last crossing of a beating envelope and jumps by a whole beat period: median 0.0 %, p90 8.0 %, worst 22.9 %, and 12 of 302 cells outside the bracket their own crossing ambiguity allows. `DECISIONS.md` 259 states the four assertions that replace the single loose bound, and measures the `decay_scale` change that would halve the residual and why it is not taken.
- **`notes.duplex` is inert at every legal value**, which is why one workspace test is red rather than green: `DECISIONS.md` 260. Section 7.7's list of what the tuner would fit under the new construction does not name it, and it should — it is the next thing the estimator side is blocked on.
