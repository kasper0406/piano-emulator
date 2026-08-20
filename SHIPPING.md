# SHIPPING.md — from engine to product

What stands between the current instrument and something a person installs and
plays. Written 2026-08-16, against suite state 605 green / 1 documented red.
`DISTRIBUTION.md` holds the plugin-architecture detail this file sequences;
where they disagree, this file is newer.

## Posture (decided)

- **Free and open source.** MIT (`LICENSE`), source on GitHub, no payment
  anywhere. This deletes `DISTRIBUTION.md`'s IAP/receipt/unlock sections and
  most App Review risk.
- **Channels:** Mac App Store (easy install; AUv3 is the only store-legal
  plugin format) **and** notarized direct downloads via GitHub Releases plus a
  Homebrew cask. One signing pipeline behind both; the $99/yr Apple Developer
  Program is the only unavoidable cost.
- **No room modeling, ever.** The instrument ships anechoic-plus-board; DAW
  reverb (Space Designer, ChromaVerb, anything) provides the space. Measured
  basis: a fitted recording-chain absorbs ≤ 6 % of the remaining benchmark
  distance (DECISIONS 311–318).
- **Data licensing:** every preset's source library is recorded in
  `ATTRIBUTION.md` **and in its own fetch script** before its parameters ship,
  and since `DECISIONS.md` 516 that rule is enforced by tests rather than
  remembered. Shipped: Salamander CC BY 3.0, bitKlavier Grand CC BY 4.0, VCSL
  CC0 1.0 — all permissive, none NC. MAESTRO (unused) is CC BY-NC-SA;
  **BiVib** (Zenodo, every key x 10 velocities, calibrated to physical SPL) is
  the one genuinely attractive NC candidate and is **deferred to the owner**:
  the NC half this file already accepts, but the **ShareAlike** half is a
  copyleft on derivatives sitting against an MIT repository. Candidates
  examined and refused are listed in `ATTRIBUTION.md` so the ground is not
  walked twice.

## Engine: what remains, honestly

Quality remainder — named, measured, none blocking usability:

| item | state |
|---|---|
| duplex drive | the one deliberate red test; needs the field's meaning re-decided (DECISIONS 260) |
| treble sympathetic halo | ~21 dB short; lives in the board's late field (instrument, not room) |
| stereo structure | engine's interchannel correlation is inverted vs the recording (bass decorrelated / treble coherent, should be the opposite); largest measured unscored gap — needs a stereo loss term first, then PHYSICS §8 virtual-mic rendering |
| per-key brightness individuality | tilt deliberately not drawn for unsampled keys (would invent ~6.6 dB of centroid); only more recorded keys can close it |
| phantom partials | confirmed at −60 to −95 dB; deferred on audibility |

Integration essentials (the actual gap to "playable"):

1. **Live MIDI input.** The engine has none — REPL and MIDI-file render only.
   CoreMIDI arrives with `DISTRIBUTION.md` M3 (standalone app).
2. **Host sample rates.** Engine is fixed 48 kHz by design; the boundary
   resampler (M0) bridges 44.1/96 kHz hosts.
3. **Real-keyboard calibration:** CC64 slew (~15 ms) so 7-bit half-pedal does
   not step; velocity-curve preference per controller.
4. **MIDI 2.0 / SL88 MK2** (the owner's intended controller, one of the few
   shipping UMP sources): widen `Event` velocity `u8 → u16`, generalize
   `velocity_from_midi` to a continuous map, parse UMP per `DISTRIBUTION.md`.
   Engine internals are already continuous (`f32` hammer velocity — 24-bit
   mantissa against 16-bit velocity), so nothing downstream changes.
   **Acceptance test: the SL88 MK2 end-to-end, distinct fine-grained
   velocities landing distinct hammer speeds.**

Sequence to "usable in Logic and standalone": M0 (resampler) → M1 (C ABI) →
M2 (AUv3, `auval` green) → M3 (standalone + CoreMIDI + MIDI 2.0 input) →
M4 (params/presets/state) → M5 (signing, notarization, cask, Releases) →
M7 (App Store). Engine quality work continues in parallel; the two streams do
not share files.

## Presets: the factory and the range

A preset is built, not authored: `piano-tuner survey → fit → sympathetic →
tail → noise`, gated by `bench / compass / melody / score / brilliance`, every
measured-vs-synthesized value provenance-marked. `presets/salamander-c5.toml`
is the proof it works end to end.

- **New measured piano** = a sample library + one factory run + listening QA.
  Licence recorded first, always. **The adapter constraint this file used to
  state is closed** (`DECISIONS.md` 516-530): what a library *is* — recorded
  keys, velocity layers and their bands, sample rate, key map, mechanism
  samples — is a `LibrarySpec` in `tuner/src/adapter.rs`, and `piano-tuner
  adapt` writes the instrument definition a library does not ship and resamples
  a tree published at another rate onto the engine's clock, once, offline. Two
  more pianos have been through it: `concert-grand-d` (Steinway D, bitKlavier
  Grand, CC BY 4.0) and `upright-parlour` (Knight upright, VCSL, CC0 1.0).
  Salamander's own render path is unchanged and bit-exactness is pinned by
  test, because every bar here was measured through it.
- **Designed variants** (range without new data): physical parameter
  transforms — hammer hardness, string scaling, board decays — validated by the
  same gates. Cheap because every parameter is physical and exposed; maps
  directly onto the MIDI Piano Profile's registered controllers. Not built yet;
  the schema makes it a small milestone.
- **Hygiene before presets multiply:** a schema-version key in the preset
  format (planned for host state in `DISTRIBUTION.md`, not yet in the TOML).

## Evaluation (decided, permanent)

Fitting uses only genuinely recorded reference keys (always has). Scoring
does too, now: transposed reference notes are listening material, unscored
(DECISIONS 328–333; the measured transposition ambiguity is 2.67 dB of mel,
53 % of the engine's whole current distance). The engine renders all 88 keys
as first-class instruments — on per-key evenness the model is judged against
recordings, never against the reference's resampling seams.
