# piano-emulator — v1 specification

Physical-model piano synthesizer in Rust. Target: Apple Silicon (M-series) MacBooks, real-time latency, sound quality first. This spec defines the architecture, module boundaries, DSP math, and acceptance tests. Follow it; if you must deviate, note the deviation in your report and append to DECISIONS.md.

## Project layout

Single cargo package `piano-emulator` (lib `piano_emulator` + bin `piano-emulator`) at the repo root.

```
src/
  lib.rs        # module declarations + public re-exports (owned by scaffold; DSP agents do NOT edit)
  types.rs      # shared constants & small types
  modal.rs      # SIMD-friendly resonator bank (core DSP primitive)
  string.rs     # PianoString: partial layout, damping, polarizations, unison group
  hammer.rs     # nonlinear felt hammer -> excitation force pulse
  soundboard.rs # body/soundboard model, stereo output
  resonance.rs  # sympathetic resonance bus
  pedal.rs      # sustain / sostenuto / una corda state
  voice.rs      # per-key voice: strings + hammer + damper lifecycle
  engine.rs     # Engine: owns 88 voices, event queue, process(block)
  audio.rs      # cpal output stream (real-time)
  render.rs     # offline rendering to WAV (hound)
  repl.rs       # terminal REPL parsing/commands
  main.rs       # bin entry: start engine + audio + REPL
tests/          # integration tests incl. spectral analysis (rustfft as dev-dep)
```

Dependencies (keep minimal): `cpal`, `hound`, `rtrb` (or a hand-rolled SPSC ring) for the event queue. Dev-deps: `rustfft`. No async runtimes, no heavy frameworks.

## Global conventions

- `SAMPLE_RATE = 48_000.0` f32, internal `BLOCK = 128` frames, stereo out.
- All audio math in f32, SoA layouts, block-based processing.
- **Real-time thread rules (audio callback):** no allocation, no locks, no syscalls, no logging. Events arrive via a pre-allocated SPSC ring buffer. All voices/buffers pre-allocated at startup.
- Denormal protection: flush-to-zero (set FPCR FZ bit on aarch64 in the audio thread, plus amplitude culling below).
- MIDI note numbering for keys: 21 (A0) … 108 (C8). `note_to_freq(n) = 440 * 2^((n-69)/12)`.

## DSP model

### Modal resonator bank (`modal.rs`)

The primitive everything builds on. A bank of N damped resonators excited by a common per-sample input `x[n]` (the force signal), each with input gain `g_k`:

Per mode k, complex one-pole (phasor) form:
```
s_k[n] = a_k * s_k[n-1] + g_k * x[n]      where a_k = r_k * e^(i*w_k)
y[n]   = sum_k Im(s_k[n])                  (or Re; pick one consistently)
r_k    = exp(-sigma_k / SAMPLE_RATE)       sigma_k = decay rate (1/s), T60_k = 6.91 / sigma_k
w_k    = 2*pi*f_k / SAMPLE_RATE
```
Store as SoA: `re[], im[], a_re[], a_im[], g[]`. Process a block with a tight loop over modes outer / samples inner (or vice versa — benchmark; modes-outer with per-mode scalar recurrence over 128 samples autovectorizes poorly across the sample axis but vectorizes across modes if you restructure; acceptable either way as long as perf budget is met).

Requirements:
- `set_damping_scale(k_range, extra_sigma)` cheap runtime adjustment (for dampers/pedal) — implement by storing base `sigma_k` and recomputing `r_k` only when damping state changes (not per sample). Smooth damping changes over ~1 block to avoid clicks.
- **Amplitude culling:** modes whose envelope |s| has decayed below −90 dBFS contribution get skipped (track per-mode or per-bank coarse energy). A bank whose total energy is below −100 dBFS reports itself idle so the voice can sleep.

### String (`string.rs`)

Partial frequencies with stiffness inharmonicity:
```
f_k = k * f0 * sqrt(1 + B * k^2),  k = 1..K
K capped so f_k < 0.45 * SAMPLE_RATE, and K <= 80
```
Inharmonicity coefficient B per note (approximate a concert grand; make it a per-note table that later automated tuning can overwrite): from ~1e-4 at A0 for wound strings, dipping to ~3e-4 around C3, ~4e-4 at C4, rising smoothly to ~1e-2 at C8. Use a smooth interpolated curve with these anchor points; exact values are a starting point, not gospel.

Per-partial decay rate, frequency dependent:
```
sigma_k = sigma0 + sigma1 * (f_k / 1000)^2
```
Choose `sigma0`, `sigma1` per note so that T60 of the fundamental is ≈ 25 s at A0, ≈ 12 s at C4, ≈ 3 s at C6, ≈ 0.6 s at C8 (undamped, sustain phase), and high partials die faster than low ones.

**Two polarizations:** each string is TWO modal banks: vertical (full amplitude, faster decay) and horizontal (≈ −12 dB input gain, ≈ 3–4× slower decay, partial frequencies offset by a fraction of a Hz). Their sum produces the characteristic fast-attack/slow-tail double decay.

**Unison groups:** notes ≤ B1 have 1 string, C2..E3 have 2, ≥ F3 have 3. Unison strings are detuned by ±0.1–0.5 Hz (scale with note; slightly wider in treble) → audible beating and additional decay complexity. Una corda reduces the number of struck strings by one (and softens the hammer, see below); *unstruck* unison strings still resonate sympathetically via their banks receiving the in-voice coupling (cheap: feed a small fraction of struck-string output into unstruck siblings).

**Excitation input gain per mode:** `g_k ∝ sin(k * pi * x_strike)` with strike position `x_strike ≈ 0.12` (fraction of string length; per-note table, slightly varying 0.115–0.14 across the compass). This nulls partials near k = 1/x_strike as on a real piano.

### Hammer (`hammer.rs`)

Lumped nonlinear felt model producing a force pulse F[n] over the first few ms, which is fed as the excitation input `x[n]` to the string's modal banks (both polarizations, all unison strings struck, with per-string small timing skew < 0.3 ms).

Model: hammer mass m colliding with string; felt compression force `F = K * c^p` with hysteresis optional for v1. Integrate the ODE at audio rate (or 2× oversampled for stability) against the string's driving-point response approximated by its wave impedance Z (constant per note is acceptable for v1):
```
hammer:  m * v' = -F(c),  c = x_hammer - y_string_contact
string surrogate: y_contact' ≈ F / (2Z)
```
p ≈ 2.3–3.0 (harder in treble), K and m per note (mass ~11 g bass → ~4 g treble). Contact ends when c ≤ 0; typical pulse 0.5–3 ms, shorter and spectrally brighter at high velocity. Velocity mapping: MIDI vel 1..127 → hammer velocity ~0.2..6 m/s, perceptually reasonable dynamic curve.

Precomputing the pulse into a short buffer at note-on (outside the per-sample audio loop, but still in the audio thread — must be alloc-free: use a fixed max-length scratch buffer per voice) and then streaming it into the banks is the recommended structure.

Una corda: multiply K by ~0.7 and reduce struck-string count by one → softer, darker, different unison balance.

### Dampers & pedals (`pedal.rs`, damper logic in `voice.rs`)

- Damper = extra damping added to a string's modes: `sigma_damped_k = sigma_k + D * damper_weight(f_k)` where D gives release T60 ≈ 0.1–0.3 s (faster in treble), and `damper_weight` decreases for very high partials (dampers grip fundamentals better than very high partials — brief metallic "zing" on release is realistic).
- Keys from G6 (MIDI 91) up have **no dampers** — always resonating.
- **Sustain pedal:** continuous value 0..1. Effective damping multiplier `(1 - pedal)` on D (so 0.5 = half-pedal = partial damping). Applies to all keys not currently held.
- **Sostenuto:** captures the set of keys physically held at pedal-down; those keys' dampers stay lifted until pedal-up.
- **Una corda:** boolean; affects hammer as above.
- Note-off with pedal up: engage damper (smoothly, over ~10 ms). Re-strike of a ringing string must NOT reset the banks — the new hammer pulse adds into the still-ringing state (this is physically what happens and audibly matters with the pedal down).

### Sympathetic resonance (`resonance.rs`)

Global mono resonance bus each block:
```
bus[n] = sum over all strings of y_string[n]   (already computed — reuse, don't recompute)
```
Each *undamped* string additionally receives `coupling * (bus[n] - own_contribution[n])` as excitation input. `coupling` ≈ 0.005–0.03 (tune by ear/test: strike-and-release C3 with pedal down must produce an audible halo; pedal-up must not). Subtracting own contribution exactly is required to avoid self-reinforcement instability; verify stability in a test (long render, bounded output).

### Soundboard / body (`soundboard.rs`)

Input: per-voice outputs with a per-key stereo pan position (bass keys slightly left, treble right, ±0.6 max). Output: stereo.

Two components, both cheap:
1. **Low-frequency body modes:** ~24 fixed resonators (another modal bank) 40–400 Hz, Q 5–30, mixed in lightly — gives the "cabinet" color.
2. **Diffuse board reverberation:** small 8-line FDN (Householder or Hadamard feedback), delays 3–15 ms (mutually prime sample counts), frequency-dependent decay T60 ≈ 0.4 s at LF falling to ~0.1 s at 8 kHz, stereo-decorrelated output taps. This is the soundboard's diffuse field, NOT a room reverb — keep it short and dense.

Mix: `out = 0.65 * panned_direct + 0.35 * board(direct_mono)` (make the ratio a parameter). A gentle global high-shelf and DC blocker (first-order HP @ 10 Hz) on the master. Soft-clip safety limiter (tanh-style) at the very end, engaged only above −1 dBFS.

### Voice lifecycle (`voice.rs`, `engine.rs`)

- 88 voices, statically allocated, one per key (no stealing needed — a key restrike reuses its voice's still-ringing banks, see above).
- Voice states: Idle (culled, skipped entirely), Ringing. Damper state orthogonal.
- Engine events (SPSC from UI thread): `NoteOn{key, vel}`, `NoteOff{key}`, `Pedal{Sustain(f32)|Sostenuto(bool)|UnaCorda(bool)}`, `AllOff`.
- `Engine::process(&mut [f32], &mut [f32])` renders one block; used by both cpal callback and offline renderer (identical code path — this is what makes offline tests meaningful).

## Performance budget

Worst case: sustain pedal down, glissando sweep so ALL 88 keys ring (≈ 230 strings × 2 polarizations, avg ≈ 45 live partials each after culling). This must consume **< 50 % of one core** on an M-series (measure with a `--bench`-style timing harness in a test or example: render 10 s offline, report ratio of render time to audio time). If over budget: more aggressive culling first, then restructure loops. Do not degrade the model to hit budget without noting it.

## Terminal REPL (`repl.rs`, `main.rs`)

Line commands (case-insensitive, note names like `C4`, `F#3`, `Bb2`; A4 = 440):
```
n <note> [vel=80]        strike a note
off <note>               release a key
chord <note> <note> ...  strike together (vel optional last arg as number)
ped sus <0..1> | sos <0|1> | uc <0|1>
demo                     built-in ~15 s musical demo exercising pedals/dynamics
render <file.wav> [demo] render the demo (or a default sequence) offline
panic                    all notes + pedals off
quit
```
Print a short help on start. Unknown input → help line, never crash.

## Acceptance tests (`tests/`)

All run offline through the engine's process path.

1. **Tuning/inharmonicity:** render 3 s of A2, C4, A4 single strikes; FFT (rustfft, ≥ 2^18 window with zero-padding + parabolic peak interpolation); detected partials 1–8 must match `f_k = k f0 sqrt(1+Bk^2)` within ±3 cents, and must NOT match the harmonic series for notes where B makes partial 8 deviate > 5 cents.
2. **Decay sanity:** T60 of C4 fundamental (band-filtered energy) in 8–20 s; C7 in 0.3–2 s; sound after note-off (pedal up) decays > 40 dB within 0.5 s.
3. **Beating:** C4 (3 unison strings) amplitude envelope of the fundamental band shows modulation (beating) with period in 0.5–10 s — i.e., non-monotonic envelope.
4. **Pedal:** strike C3, release with sustain pedal 1.0 → energy 2 s after release within 12 dB of pedal-less sustain at same time point; with pedal 0 → at least 40 dB lower.
5. **Sympathetic resonance:** hold C3 silently (strike vel 1 then immediately damp — or add a test-only "lift dampers" hook), strike & damp G4 hard: C3 string bank energy must rise above its noise floor. Simpler variant acceptable: pedal down, strike-and-release one note, verify broadband halo energy exists 1 s later, and pedal up → doesn't.
6. **Stability/safety:** 30 s render of dense random playing with pedal: no NaN/Inf, peak ≤ 0 dBFS, DC < −60 dBFS, and idle engine renders exact silence.
7. **Performance:** the budget test above, asserted at < 80 % of one core in CI-ish conditions (< 50 % is the design goal; the assert has headroom for debug machines) — run in `--release` only (`#[ignore]` by default with a note, or cfg on `debug_assertions`).

Run `cargo test --release` for DSP tests. Keep unit tests next to modules for math-level checks (e.g., resonator frequency accuracy: a single mode at 1 kHz must peak within 0.1 Hz).
