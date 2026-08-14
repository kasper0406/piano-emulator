# PHYSICS.md — the next modelling iterations

What the v1 instrument does not contain, ranked by truth bought per unit of work. Everything here is **additive**:
nothing replaces modal strings, inharmonicity, frequency-dependent decay, two polarizations, unison groups with bridge
coupling, the Hunt–Crossley hammer with its agraffe reflection, continuous dampers / sostenuto / una corda, the
sympathetic bus, or the body modes + FDN soundboard.

**Cost yardstick.** ~20 000 live resonators measure 31–40 % of one M4 Pro core (`DECISIONS.md` 25b, 37), so one extra
resonator ≈ **0.002 % of a core** and a per-sample scalar filter on one signal ≈ ten resonators. Everything below
totals ~4–6 %. The budget is not the constraint; model risk is.

**Evidence rule.** Nothing here gets built until `TUNING.md` Phase E shows the residual it is meant to explain —
*stated in advance* under each entry. That is why Phase E comes before this work.

---

## 1. Longitudinal string modes and phantom partials

**Physics.** Transverse motion stretches the string, so tension carries a term ∝ `(∂y/∂x)²` that drives longitudinal
waves along it (Bank & Sujbert, [JASA 117(4) 2268, 2005](https://home.mit.bme.hu/~bank/publist/jasa05.pdf), Eqs. 6–7).
The *forced* response is the **phantom partials** at sums/differences `f_m ± f_n`, dominated by adjacent parents;
since even phantoms sit near `2f_n` the series carries inharmonicity `B/4` (ibid. Eq. 20 — Nakamura & Naganuma's
"lower series"). The *free* response is the longitudinal modes at `f'_k = k·c_L/2L`, `c_L = √(E/ρ) ≈ 5050 m/s` in
steel — 1.2–2 kHz for bass speaking lengths — decaying in ~0.15 s and acting as formants over the phantoms.
Transverse→longitudinal coupling far exceeds the reverse, so a **one-way** model is defensible at playing dynamics.

**Audible when.** The metallic growl of a hard-struck bass note; the largest timbral gap between our bass and a real
one. Bank & Lehtonen's ABX tests ([JASA 128(3) EL117, 2010](https://pubs.aip.org/asa/jasa/article/128/3/EL117/598974))
found longitudinal components audible up to **C5** but judged modelling up to **A3** sufficient. The drive is
quadratic in transverse amplitude, so it appears only at forte and above — exactly where our bass is thin.

**Build.** New `engine/src/longitudinal.rs`: one `ModalBank` of 4–8 modes per key below ~A3, owned by `PianoString`.
Bank's "resonator-based string model" (§V C) *is* our architecture. One simplification: rather than forming
`F_{t→l,k}` per longitudinal mode from individual modal amplitudes (`ModalBank` sums and does not expose them), take
the string's existing bridge-force output, apply a first-order +6 dB/oct tilt (the `∝ k` weighting of `∂y/∂x`), square
it, high-pass ~200 Hz against DC and slow difference terms, and use that as the bank's common excitation. Bank
sanctions a *common* excitation for all longitudinal resonators (§V B); what we additionally lose is which `(m,n)`
pairs feed which `k`, and the module must say so. Output reaches the soundboard through its own gain and high-shelf
(the board answers a longitudinal bridge force differently — Bank's `H_l(z)`), never back into the transverse banks.
~240 resonators (0.5 %) plus three scalar ops/sample per bass string. The risk is not CPU: a squared signal is an easy
way to build something that sounds like distortion rather than like a piano.

**Evidence.** Phase E must show, on the loudest Salamander bass layers, peaks *between* the transverse partials that
(i) grow ~quadratically with layer level where transverse partials grow ~linearly, (ii) fit `f_p = f₀ p √(1 + Bp²/4)`
with the `B` the inharmonicity estimator already returns, and (iii) show broad emphasis at 1–2 kHz. Fitting needs an
estimator that is *not* a trajectory consumer: it reads the residual spectrum after the tracked transverse partials
are removed and fits `f'_1` and the formant envelope, with `c_L/2L` as a hard prior on `f'_1`. Fit the level rather
than derive it — phantoms also arise from nonlinearity in the *structural* parts (Bilbao et al., [JASA 142(4) EL344,
2017](https://pubs.aip.org/asa/jasa/article/142/4/EL344/853105)).

## 2. Tension-modulation pitch glide — **refuted by measurement** (`TUNING_REPORT.md` §6)

> **Status:** the check this entry prescribed was run in Phase E: beyond 200 ms the recorded fundamental holds to a
> few cents, and inside 100 ms the measurement's own error bar (±17–108 cents on the engine control) swamps any
> plausible glide. Per this entry's own rule — "no drift → delete the entry" — do not build this. Kept for the record;
> revisit only with a measurement technique that can see inside the first 100 ms.

**Physics.** The same nonlinearity seen by the transverse modes: under the uniform-tension approximation `T̄(t) = T₀ +
(ESπ²/4L²)·Σ n² y_n²(t)` (Bank & Sujbert Eq. A2; Legge & Fletcher, [JASA 76, 5,
1984](https://www.phys.unsw.edu.au/music/people/publications/Leggeetal1984.pdf)) and `f_k ∝ √T`, so every partial
starts sharp after a hard strike and glides down as the amplitude decays.

**Audible when.** A few cents over the first 100–300 ms, fortissimo, bottom two octaves — heard not as pitch but as
the attack settling, and as a struck note briefly out of tune with its own ringing neighbours. Small; listed because
it is nearly free *once §1 exists* and near-worthless alone.

**Build & evidence.** Per block, form `ratio = √(1 + α·E)` from `ModalBank::energy()` (already public) and apply it to
every pole angle via a new `ModalBank::scale_frequencies`; the angle change is tiny, so a two-term rotation update
avoids 80 sin/cos per block. Preset `tension_alpha[88]`, zero above ~C3, active only above an energy threshold and for
~0.5 s: < 0.5 % of a core, but fiddly — a frequency-varying resonator bank is the one place here where a coefficient
update can click. The confirmation is the cheapest check in this document and runnable today, since the tracker
already outputs `f_k(t)`: plot the tracked fundamental over the first 300 ms of Salamander's top and bottom velocity
layers at A0/C2/C3. Drift at layer 15 and none at layer 3 is the effect, and `α` fits from its magnitude. No drift →
delete the entry.

## 3. Duplex scaling and aliquot segments

> **Built (sympathetic milestone), and it does not yet sound.** `engine/src/duplex.rs` is the bank, held by
> `Voice` rather than by `PianoString` because everything in `string.rs` is damped and these are the one part of the
> instrument that never is (`DECISIONS.md` 153). Both drives are there — the key's own bridge force and the
> sympathetic bus — the schema takes measured frequencies as this entry demands, and the estimator recovers 100
> segments over 23 keys of Salamander at a median +27 cents off the nearest partial, which is the scatter Öberg &
> Askenfelt describe (161). Three findings qualify it. (a) `gain_db` is normalised to the segment's *steady*
> response so that level and length are independent measurements, and `ModalBank`'s cull then zeroes the state
> before that steady response can build: a segment asked to ring 1.4 s rings 0.21 s, and the gap between what the
> file says and what the render does is 93.7 dB (162). (b) 88 permanently undamped banks close a loop that never
> dies, so the level the measured table ships at was set by the loop budget rather than by the measurement (163).
> (c) End to end, the segments of `presets/salamander-c5.toml` contribute **−148.6 dB** relative to the note that
> drives them (170). The prediction of this entry — treble shimmer surviving a staccato release — is reachable at
> the schema's ceiling and unreachable at the measured setting, and the fix is the cull or the normalisation, not
> the estimator. Re-measured in the review pass and unchanged (−84.6 dB during the note, −244 dB at the segment
> frequencies 0.4 s after the release, −57.7 dB during at the largest boost the validator accepts). The reason is
> item 157 read the other way round: `gain_db` is normalised to the segment's response *at its own frequency*, and
> a struck string's bridge force is a sum of decaying sinusoids at its own partials, so a resonator a hertz wide
> sitting between them is handed almost nothing to answer. Raising the table cannot fix that and the loop bound
> would refuse it anyway; what the segments need is a drive with energy where they are — the hammer's own broadband
> knock through the bridge — which is a build, not a number.

**Physics.** The string does not end at the bridge or the agraffe. The front segment (capo bar/agraffe to tuning pin)
and the rear segment (bridge to hitch pin) are short, undamped, high-pitched strings sharing the bridge with the
speaking length — on aliquot grands the rear one is nominally tuned to a partial of it. They are driven only through
the bridge, and ring on after the speaking length is damped.

**Audible when.** Treble shimmer, and the halo that survives a staccato release in the top three octaves. Öberg &
Askenfelt measured every main and duplex string over **D4–C8** on a concert-condition grand, saw both segments in
bridge motion and radiated sound, and ran an ABX test ([JASA 131(1) 856,
2012](https://pubs.aip.org/asa/jasa/article-abstract/131/1/856/823090)): damping the **front** duplex was *clearly*
perceptible to musicians **and** to naive listeners; the rear duplex perceptible but less pronounced.

**Build.** The cheapest structural addition here. A per-key `ModalBank` of 2–6 modes at 1.5–8 kHz, **never damped**,
in `PianoString` beside the polarizations. Input: the key's own bridge force (already computed as `group_previous`)
*plus* the sympathetic bus — that second path is what makes the duplex answer other notes, which is most of what it is
for. Being permanently undamped it must respect `ModalBank`'s culling or it will keep 88 voices awake, so give it a
shorter T60 (0.5–2 s) than intuition suggests. ≈ 350 resonators ≈ 0.7 %; a day. **Do not tune the segments to nominal
harmonic ratios** — the same paper found real rear-duplex tuning generally sharp, average and median deviations
approaching **+50 cents** (single keys at +190 and −100), with spread *within* one trichord averaging ~25 cents and
occasionally 60. That scatter is the sound: store measured frequencies, not ratios.

**Evidence.** Phase E should show sustained treble energy at frequencies *not* in `f_k = k f₀√(1+Bk²)` outliving the
tracked partials — most clearly in Salamander's release samples (`harmL*`, `harmS*`, `harmV3*`: three velocity tiers
of "release string resonances", 2–3 s each), which are literally a recording of what still rings when the dampers
land. Estimation is the existing tracker with the inharmonic seed removed: peak-pick the residual, keep peaks with
T60 > 0.3 s, write the strongest 2–6 per key.

## 4. Bridge admittance: two-way string ↔ soundboard coupling

> **Phase E update** (`TUNING_REPORT.md` §3, §4): the *coupling* half of this entry is strengthened — the treble
> sympathetic halo is the report's #1 finding (C7 fortissimo: between-partial energy −3.5 dB vs the engine's −48 dB)
> and its level is explicitly a stage-2 coupling parameter. But the report *refutes* using one global admittance
> curve to explain excitation-spectrum roughness: that residual is per-note, not shared across notes at the same
> frequency. Build the admittance filter for coupling and decay shaping; do not expect it to fix excitation spectra.
>
> **Built (sympathetic milestone), with one prediction confirmed and one refuted.** `B(f)` is a fitted shelf
> backbone plus RBJ peaking sections on the resonance bus (`DECISIONS.md` 148–150), stability is a validated
> contract rather than a hope (149), and it costs 0.2 % of a core. *Frequency-selective excitation is exactly
> true*: a ±12 dB resonance moves the energy one string delivers to another by +11.8 / −11.6 dB, at A0, C4 and C6
> alike (151). *The decay-rate coupling does not fall out free* and cannot, because with `own` subtracted nothing
> in the loop is proportional to the string's own motion — the struck string's own level moves under 0.4 dB at the
> loop ceiling while its halo moves 12 dB, and the residual self-term shifts frequency rather than damping. `Re Y`
> belongs in `string.rs`'s `partial_sigma`, and a test pins its absence until it lands (151). What the fitted
> filter bought, end to end: the release-resonance halo rose 36 dB at C3 and 65 dB at C5 and stopped dying inside
> a second (169), and this section's own headline statistic — between-partial energy at one second — did not move
> at all, because it is sitting on the analysis window's leakage floor at about −48 dB and no engine path moves it
> (168). The treble halo this entry was written about is still 21 dB (C6) to 44 dB (C7) short of the recordings.
>
> **Both halves resolved (review pass).** *The decay-rate coupling is built*, and where this entry says it belongs:
> `Re Y` is in `string.rs`'s per-partial damping as `[voicing.bridge].radiated_share`, the share of a partial's
> fitted decay that is loss into the board, modulated by the *peaks* alone — the backbone is the mean mobility and
> is already inside the fitted `sigma(f)`, so putting it in again would count it twice. A partial 2.7 cents off a
> board mode now comes back at T60 11.33 s against 14.60 s with a unity bridge, and 11.42 s with the coupling
> switched off, i.e. it is the string's own damping and not loop feedback (`DECISIONS.md` 182). *And the treble
> halo this entry was written about is not on this path at all*: with the sympathetic coupling raised to the
> largest value the stability contract will ever certify, C7's between-partial energy one second in moves **0.0 dB**
> (−38.5 dB, against the recording's −17.0); with the board's diffuse field `T60` multiplied by four it moves
> **17 dB** at C7 and **28 dB** at C6, to within 4.5 and 0.4 dB of the recordings. The missing aftersound is the
> late field of §8/§9, not the coupling of §4 — and how much of the recordings' late field is the instrument and
> how much is the room is exactly §9's question (`DECISIONS.md` 184).
>
> The stability contract also had to be repaired to be one: measuring `max|B|` on a log grid plus the peaks' own
> centres is evadable, because a cascade *adds* decibels and two overlapping resonances put their maximum between
> their centres — a preset inside this schema hid 15.6 dB there and realised a loop gain of 1.5 while measuring
> 0.25 (`DECISIONS.md` 179).

**Physics.** A string terminates on a bridge with complex admittance `Y(f)`, not on a node. `Re Y` sets how fast each
partial loses energy into the board, and because all strings share the board, partials of *different* notes that fall
close in frequency couple through it. Within one unison that is Weinreich ([JASA 62(6) 1474,
1977](https://pubs.aip.org/asa/jasa/article-pdf/62/6/1474/11470322/1474_1_online.pdf)): symmetric mode strongly
coupled and fast (prompt sound) against antisymmetric mode weakly coupled and slow (aftersound). *Across* notes it is
Cartling ([JASA 117(4) 2259, 2005](https://pubs.aip.org/asa/jasa/article-abstract/117/4/2259/540260)): weak
coexcitation of an adjacent tone through the bridge gives frequency *and* amplitude modulation, two coexcited tones
give beating modulations of opposite phase, and unison detuning significantly increases the frequency deviation — so
the two effects interact.

**Audible when.** This is what a held chord *does*. Our halo is a scalar coupling through a flat bus: spectrally
uniform, and it never changes anyone's decay rate. Woodhouse notes that for the piano the body-loss/air-loss ratio
exceeds 0 dB above ~160 Hz, so coupled double decay is predicted for *every* multi-strung note, not only near isolated
body resonances as on a guitar ([Euphonics §7.3](https://euphonics.org/7-3-multiple-strings-and-double-decays/)).
Loudest with the pedal down, mid-register, slow sustained playing — where the model is most obviously a synthesizer.

**Build.** Both halves exist. `resonance.rs` feeds strings `coupling·(bus − own)`; insert a bridge-admittance filter
`B`, giving `coupling·B(bus − own)` — one shared filter on one mono signal, ~24–40 resonators (0.06 %). Its target
shape is documented: mean driving-point mobility ≈ **1.3×10⁻³ s/kg** (impedance ≈ 800–1000 kg/s) over 100–1000 Hz with
**±10–15 dB** fluctuation, falling to a few hundred kg/s in the treble; sharp well-separated peaks below ~500 Hz, a
slight dip to 2 kHz, a slight rise to 4 kHz. Build it as *modal peaks + smooth backbone*, not one long modal bank,
because of Ege & Boutillon's transition frequency **f_lim ≈ 1.1 kHz** (half-wavelength = mean inter-rib spacing):
below it the board is a homogeneous plate with discrete modes (~0.05 modes/Hz); above it waves localize between ribs,
apparent modal density collapses to ~0.01 modes/Hz, and the Skudrzyk characteristic mobility `Y_c = n(f)/(4M)` is the
right object ([arXiv:1210.5688](https://arxiv.org/abs/1210.5688), [arXiv:1210.5109](https://arxiv.org/abs/1210.5109)).
Two things fall out free: sympathetic excitation becomes frequency-selective, and the coupling becomes two-way — the
decay-rate coupling. `DECISIONS.md` 23's stability argument must be redone, since `B` has gain > 1 at its resonances:
re-derive `DRIVE_CEILING` and `MAX_COUPLING` against `max|B(f)|`, with the long-render boundedness test as the gate.
That, plus re-voicing `resonance_coupling`, is the work.

**Evidence.** Two signatures. First, per-partial decay residuals: `DECISIONS.md` 84(b) reports 20–35 % T60 error on
beating unisons and blames the envelope model, but a partial sitting on a board resonance should *also* decay
systematically faster than the fitted `σ(f)` curve — a residual correlated with **frequency across notes**, not with
note, is the admittance showing through. Second, the release samples again, where what remains is shaped by what the
board hands back. Fitting the admittance from isolated notes is hard; honestly this is a **stage-2** parameter
(`TUNING.md`'s CMA-ES on MAESTRO's pedalled playing), seeded from the existing body modes.

## 5. Action, damper and pedal noise

**Physics.** Askenfelt measured the whole structure-borne path directly, by removing C4's strings and replacing them
with a 4 kg dummy mass (*Observations on the transient components of the piano tone*, [STL-QPSR 34(4) 15,
1993](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=624e7d25054fb6f5e9ab864e48f432d7dc5871fd)). The
results are a specification: the structure-borne spectrum **extends only to ~2 kHz** and sits **~40 dB below the
string partials**; a **touch precursor** starts **20–30 ms before** hammer–string contact on a struck/staccato touch,
~30–40 dB below the first transversal wave, dominated by key resonances at **290 and 440 Hz**; after contact the key
rings at **~900 Hz** until the rebounding hammer is caught by the back check 10–20 ms later; and the structure
contributes keybed **95 / 330 Hz**, soundboard **100 Hz**, rim **250 Hz**, whole body **38 Hz**. Key-bottom timing
spans far more than intuition suggests: Goebl, Bresin & Galembo measured key bottom up to **35 ms after** hammer–string
contact at very soft dynamics and **4 ms before** it at very strong, crossing over at 1.4–3.8 m/s depending on the
instrument ([JASA 118(2) 1154, 2005](https://iwk.mdw.ac.at/goebl/papers/Goebl-Bresin-Galembo_JASA2005_PianoAction.pdf)).

**Audible when.** Constantly, and the most obvious "this is synthetic" tell left. It dominates at *pianissimo* —
Askenfelt's calibration is that a mezzo-forte blow on the dummy reaches bridge levels "compatible with pianissimo
level in normal playing" — and in fast repeated notes, staccato, and every pedal move. Goebl et al. showed listeners
identify touch type above chance from a recording, and at chance once the **first 250 ms is replaced by silence**
([ISMA 2004](https://ofai.at/papers/oefai-tr-2004-02.pdf)): the cue *is* the action noise. Fontana et al. found expert
musicians correctly localize which key was played **from the mechanical noise alone** ([JASA 156(1) 164,
2024](https://pubs.aip.org/asa/jasa/article/156/1/164/3302118/)) — so it must be panned per note, not centred. Bank &
Chabassier's survey of model-based pianos ([IEEE SPM 36(1), 2019](https://inria.hal.science/hal-01894219/document))
contains no treatment of these noises: open ground, not a solved problem we are late to.

**Build.** New `engine/src/mechanism.rs`, outside the string path: short excitations summed into the **soundboard
input** — correct, since the thump reaches the ear through keybed and board — at the key's own pan, band-limited to
~2 kHz and ~40 dB under the partials. Each event is a shaped noise burst (white through 2–4 biquads) or a 3–5 mode
`ModalBank` ping at the frequencies above. Five events: **touch precursor** 20–30 ms *before* the strike, gated on
touch type (Askenfelt found it "very much reduced" on a strained touch and **absent entirely** on a pressed/legato
one, which is exactly §6's silent press, free); **key-bottom thump** offset from the strike on Goebl's +35 ms…−4 ms
curve; **damper fall** at note-off, scaled by release velocity; **damper lift**; **pedal tray up/down**, one shared
event scaled by how many dampers actually moved. Under 1 % of a core; only note-on/off need new plumbing.

**Evidence.** No Phase E gate needed — the data is on disk and unusually direct. Salamander ships **88 per-key
release-noise samples** (`rel1…rel88`, 0.46 s, 48 kHz/24-bit, filed in the SFZ under `//HammerNoise` at `volume=-37`,
`amp_veltrack=82`) and **four pedal samples** (`pedalD1/D2`, 6.4 s; `pedalU1/U2`, 0.6–0.7 s). A new tuner stage fits a
filtered-noise or few-mode model per key to `rel*` (LPC or a modal fit over the first 50 ms is plenty) and reads level
and velocity tracking from the SFZ's own `volume`/`amp_veltrack`/`rt_decay`. For *absolute* levels, which Salamander
cannot give, add **BiVib** (Papetti, Avanzini & Fontana, [Appl. Sci. 9(5) 914,
2019](https://www.mdpi.com/2076-3417/9/5/914), CC BY-NC-SA): all 88 notes × 10 velocities on Disklavier grand and
upright, binaural plus keyboard accelerometer, full calibration chain published, three lid configurations (§8 wants
those too). Sanity target for the release-noise velocity law: Pianoteq's Blüthner uses a **12 dB** range over note-off
velocity, additionally modulated by note-on→note-off duration.

## 6. Silent key press, release velocity, nonlinear damper contact

**Physics.** Depressing a key slowly enough that the jack escapes without the hammer reaching the string lifts that
note's damper and nothing else — the standard way of preparing sympathetic resonance, and written into repertoire.
Release speed sets how fast the damper falls, so it controls the damping *ramp*, not just its endpoint. And a
partially-engaged damper is not merely extra `σ`: Lehtonen, Askenfelt & Välimäki measured the acoustic signal *and*
damper acceleration through a part-pedalled note ([JASA-EL 126(2) EL49,
2009](https://pubs.aip.org/asa/jasa/article/126/2/EL49/903951/)) and found three intervals — free vibration,
damper–string interaction, free vibration again — where the middle one decays fast *and alters the timbre*, because
the felt **nonlinearly limits the string's deflection** at its position. That is the buzz of a half pedal and a slow
release, and our linear damper cannot make it.

**Audible when.** The silent press is categorical: pieces that call for it are unplayable without it. Release velocity
is subtler and continuous — a released chord that stops versus one that is *let* go. The nonlinear contact matters
only while the damper is touching but not seated: exactly the half-pedal and slow-release gestures the engine already
exposes but under-serves.

**Build.** Mostly structural. `Event::NoteOn{key, vel}` splits into a key-down that always lifts the damper and a
strike that happens only above the escapement threshold; `NoteOff` carries a release velocity setting
`Voice::damper_step` (today a constant ~10 ms ramp). `pedal.rs`'s sostenuto capture already reads "physically held",
so a silently held key is captured correctly with no further change — precisely the mechanism the effect exists to
exploit. `midi.rs` passes note-on velocity 0 and note-off velocity through; the REPL gains `hold <note>` and
`off <note> [vel]`. The nonlinear part is one soft clipper on the string's summed output, fed back as a correction
while `0 < damper_current < 1`, on a handful of voices at a time. A day, plus two acceptance tests: a silently held C3
answers a struck G4 *with the pedal up*, and a slow release decays slower than a fast one. Little to fit from
Salamander — the ramp shape is the existing `damper_weight(f_k)` table; release-velocity scaling and the clipper
threshold are stage-2 parameters, and MAESTRO carries note-off velocities and continuous pedal.

## 7. Hammer contact width in the excitation comb

**Physics.** Our mode input gain is `g_k ∝ sin(kπ·x_strike)` — a point force. A real hammer contacts 1–2 % of the
speaking length, so the excitation is that comb convolved with the contact profile: nulls fill in and high partials
fall off faster than the comb alone predicts. Measured excitation spectra for real hammers and strings depart from the
ideal comb in exactly this way (Hall & Askenfelt, *Piano string excitation V*, JASA 83, 1627, 1988).

**Audible when & build.** Modest and everywhere: a few dB of high-partial content and the depth of the strike nulls —
the brightness balance of the whole instrument — largest in the treble where contact width is a bigger fraction of a
short string. One line in `string.rs`: multiply `g_k` by a width taper (for a raised-cosine patch of relative width
`w`, `cos²(kπw/2)` tapering to zero is close enough for `kw < 1`), evaluated once at construction. Preset
`contact_width[88]`, initialised 0.01–0.02. Zero runtime cost, half a day. Listed because it is a *known error* with a
one-line fix, sitting upstream of the felt estimator.

**Evidence.** Directly measurable with what exists. `estimate/strike.rs` already fits the comb to time-zero partial
amplitudes, and `DECISIONS.md` 75/84(d) already reports the comb-corrected spectrum being off by "the same fixed
pattern of up to 6 dB, identical at every velocity". If Phase E shows that pattern is a monotone high-frequency droop
and that measured nulls are shallower than `sin(kπx)` predicts, that is this effect; `w` becomes a second free
parameter in the existing strike estimator, which should also improve the felt stiffness fit since `K` reads the same
spectrum.

## 8. Radiation, lid, microphones and the listener

**Physics & audibility.** The soundboard is a large radiator with frequency-dependent directivity, strongly modified
by the lid; what a listener or microphone receives is a position-dependent filtered and delayed mixture, not a
pan-potted mono sum. `soundboard::pan_for_key` (±0.6) stands in for bass and treble bridges being in different places
and models neither distance, lid, nor microphone. Lid effects are large: closing it attenuates above ~200 Hz but
*adds* ~5 dB at low frequencies; open, the principal lobes sit ~15–35° above horizontal with off-axis upper harmonics
up to 10 dB down; at 250 Hz the level is ~5 dB higher behind the instrument than in front (Meyer; Bork & Meyer, via
[SOS](https://www.soundonsound.com/techniques/recording-real-pianos)). Radiation efficiency itself transitions at
1–1.6 kHz and is mode-controlled below ~500 Hz (Suzuki, JASA 80(6) 1573, 1986). This is realism of the *recording*,
not of the instrument — which is why it is last, and why it nevertheless decides whether an A/B against a real
recording means anything.

**Build.** A new stage between `Soundboard` and the master chain, changing neither. Two or more virtual mic positions;
per mic, a per-key delay and gain from geometry (bass and treble bridge coordinates), a first-order distance/air
filter, and one delayed, filtered lid reflection. Binaural is the same stage with an HRTF pair instead of mics, behind
a preset switch rather than in the default path. Keep the FDN as the soundboard's diffuse field (`DECISIONS.md` 19). A
few delay lines and biquads, < 1 % — but without measured directivity this becomes a plausible-sounding invention,
worse than the honest pan-pot it replaces, and there is no shortcut: the TU Berlin instrument-directivity database
([arXiv:2307.02110](https://arxiv.org/abs/2307.02110)) covers 41 instruments and contains **no piano**. BiVib's three
lid configurations (§5) are the nearest usable measurement.

**Evidence.** This is where `TUNING.md`'s stage-2 recording-chain absorber lands. Today that absorber is a 40-band
static EQ that will happily swallow mic placement, lid and room in one curve. Splitting it into *mic geometry* +
*room* (§9) and letting CMA-ES fit the geometry is strictly better: geometry has priors, an anonymous EQ has none.

## 9. Room ambience as a separate stage

**Physics & audibility.** A room is not a soundboard. The board's diffuse field is dense, short (T60 ≈ 0.4 s falling
to 0.1 s at 8 kHz) and part of the instrument; its loss factor is only ~2 % (Ege & Boutillon), so its impulse response
is far shorter than any room's. A room adds discrete early reflections and a longer, differently coloured tail, and
sits *after* the microphone — the separation Pianoteq also makes, its instrument model being anechoic with the room a
third, convolution stage. Not available here at any setting: the FDN cannot be lengthened into a room without changing
the instrument's own colour, because it lives inside the board path.

**Build & evidence.** Optional stage after §8, off by default: a short early-reflection tap delay plus a longer FDN,
or convolution with a supplied IR — the offline renderer can afford convolution even if the live path cannot. < 1 %
for the FDN form. Every recording the tuner will read has a room in it, and giving the loss a room to put the room in
is what stops room decay being fitted as string decay — the most likely way for stage 2 to produce a preset that is
wrong in an interesting way. Cheap insurance.

---

## Considered and not proposed

- **A separate cabinet/rim resonance stage.** Keane's modal analysis of combined case + soundboard motion found
  soundboard deformation exceeding case deformation by **at least 10 dB and typically 20–30 dB** in an upright and by
  more in a grand, case impedance ~5×10⁵ kg/s against the board's ~10³, and case-panel effects on soundboard modes
  *"less than 5 %"* ([ACOUSTICS 2006](https://www.acoustics.asn.au/conference_proceedings/AASNZ2006/papers/p88.pdf)).
  The rim matters as the board's *boundary condition*, which our 24 body modes already absorb (`DECISIONS.md` 21).
- **Deriving `σ(f)` from a bridge admittance instead of fitting it.** Purer, practically worse: the tuner measures
  per-partial decay directly to 1.5–4.5 % on a single-strung note (`DECISIONS.md` 81). §4 adds the *coupling* the
  fitted curve cannot express, and leaves the curve alone.
- **"Blooming"** (energy migrating from low to high partials during the note — a Pianoteq parameter). No mechanism in
  the literature we could fit rather than invent. Revisit only if Phase E shows partial envelopes that *rise*.
- **A full N×N inter-string coupling matrix.** §4 buys the audible part — frequency-shaped, two-way coupling — at O(N)
  through the shared bus, which was `DECISIONS.md` 9's bet and is still right.

---

## Recommended ordering

Four milestones, ranked strictly by audibility per unit of effort. Phase E runs first and can kill or reorder §1, §2,
§3 and §7; it cannot kill milestone A's noise work, already justified by data on disk.

**A — the mechanism (§5, §6, §7).** *Why first:* the highest audibility-per-effort here by a wide margin and the least
risky — §5's parameters come out of Salamander's `rel*`/`pedal*` and BiVib rather than out of a new estimator, §6 is
mostly event plumbing, §7 is one line that fixes a known model error and improves the felt fit at the same time.
Together they are the difference between an instrument that sounds right in isolation and one that sounds *played*.
Do it first even though none of it is string physics.

**B — the bridge (§4, then §3).** *Why second:* both reuse structures that already exist (body bank, resonance bus,
`ModalBank`) for well under 1 % of a core, and together they turn the sympathetic halo from a diffuse wash into the
instrument answering itself. §4 before §3, because §3's banks should be driven through the same admittance path — the
other order builds §3's input twice. Budget real time for §4's stability re-derivation; it is the only change here
that can make the engine unbounded.

**C — the nonlinear string (§1, then §2).** *Why third:* the largest remaining *timbral* gap and the most defensible
physics in this document — but register-limited (A3 and below, per Bank & Lehtonen), dynamics-limited (forte and
above), and the one entry where a wrong implementation sounds like distortion rather than like a slightly wrong piano.
A and B improve every note; this improves thirty. §2 rides along at almost no extra cost, never alone.

**D — the listener (§8, §9).** *Why last:* it changes nothing about the instrument, it is the easiest thing here to
fake convincingly and therefore the easiest to get wrong invisibly, and its real payoff is downstream — it gives
`TUNING.md` stage 2 somewhere honest to put the room and the microphone instead of bending string parameters to absorb
them. Build it when stage 2 needs it, not before.
