# Attribution

Every measured preset in this repository is a set of **parameters estimated
from recordings of a real piano**. The recordings themselves are never
redistributed here: `data/fetch_*.sh` downloads each library (checksummed,
into the gitignored `data/`) for anyone re-running the estimation pipeline,
and the reference side of the realism benchmark plays it back for comparison.
The scripts are checked in and the data they fetch is not — the licence below
travels with the recordings, and what this repository ships is MIT-licensed
numbers derived from them.

**The standing rule, and it is tested rather than remembered** (`tuner/tests/
adapter.rs`): a library's licence, its URL and its source are recorded *here*
and *in its fetch script* **before** any parameter estimated from it ships in
a preset. `every_described_library_has_a_checked_in_fetch_script_carrying_its_licence`
and `attribution_names_every_library_a_preset_ships_from` fail if either is
missing, and `every_measured_preset_names_its_library` fails if a preset file
does not carry its own provenance.

**Preset names are descriptive, never brand names.** `concert-grand-d`,
`upright-parlour`, `salamander-c5`. The instrument, the library, the author
and the licence live here; a trademark does not become a product name because
a recording of one was measured.

---

## `presets/salamander-c5.toml` — Yamaha C5 grand

Its tuning, inharmonicity, decay, unison detuning, excitation texture,
micro-motion, stereo directivity and action-noise tables were estimated from
the **Salamander Grand Piano** (V3), a Yamaha C5 recorded at 48 kHz / 24 bit
with two AKG C414 in AB about 12 cm above the strings.

| | |
|---|---|
| instrument | Yamaha C5 grand |
| library | Salamander Grand Piano V3 |
| author | **Alexander Holm** (`axeldenstore [at] gmail [dot] com`); FLAC release assembled for FreePats by roberto@zenvoid.org |
| licence | **CC BY 3.0** — <http://creativecommons.org/licenses/by/3.0/> |
| source | <https://freepats.zenvoid.org/Piano/acoustic-grand-piano.html> |
| shape | 30 keys at minor thirds A0–C8, 16 velocity layers, 48 kHz native |
| fetch | `data/fetch_salamander.sh` |
| ships | estimated parameters only |
| does not ship | the recordings, in any form, whole or excerpted |

## `presets/concert-grand-d.toml` — Steinway D concert grand

Estimated from the **bitKlavier Grand Sample Library**, "Piano Bar" mic image,
48 kHz / 24 bit: a Steinway D concert grand recorded in Taplin Auditorium,
Princeton University, between 13 and 20 January 2021. The Piano Bar image is a
spaced pair of Earthworks omnis on a bar laid across the harp near the
hammers — a close image without room or lid interaction, and the same geometry
class as Salamander's AKG C414 AB pair, so `[voicing.mics]`'s mid/side model
transfers unchanged.

| | |
|---|---|
| instrument | Steinway D concert grand, Taplin Auditorium, Princeton University |
| library | bitKlavier Grand Sample Library, Piano Bar mic image, 48k/24b |
| author | **Daniel Trueman**, Princeton University Department of Music (`dtrueman [at] princeton [dot] edu`) |
| licence | **CC BY 4.0** — <https://creativecommons.org/licenses/by/4.0/> |
| licence provenance | carried in the archive item's own metadata, field `licenseurl` of <https://archive.org/metadata/bitKlavierGrand_PianoBar_48k24b> — not merely asserted by a third party |
| source | <https://archive.org/details/bitKlavierGrand_PianoBar_48k24b>, <https://bitklavier.com/the-bitklavier-grand/> |
| shape | 30 keys at minor thirds A0–C8, 16 velocity layers, 48 kHz native; 88 chromatic key-off, 90 release resonances, 4 pedal |
| fetch | `data/fetch_bitklavier.sh` (md5, sha1 and sha256 all checked) |
| ships | estimated parameters only |
| does not ship | the recordings, in any form, whole or excerpted |

**Four editorial processes were applied to this library before it was
published, and each biases a stage of the factory.** They are recorded here
because they are part of the provenance of every number in the preset, and
they are repeated in `data/fetch_bitklavier.sh`, in the generated instrument
definition's own header, and in `LibrarySpec::caveats`:

1. **Per-sample gain rebalancing** — the author's methodology states "small
   adjustments (usually <2 dB) to the gains of all the samples so that they
   were evenly distributed, soft to loud, and so they matched as well as
   possible across the keyboard". The `level` stage measures exactly that, so
   on this library it fits *the editor's* balance and not the piano's.
2. **RX8 spectral denoise on every sample**, to remove room and mic noise.
   That removes the low-level broadband floor the between-partial halo census
   reads: the treble-halo shortfall is likely unmeasurable here, and the halo
   work must not be re-baselined against this library.
3. **5 ms attack fade and trimmed leading silence** on every pitched sample,
   plus a 100 ms release fade at file end. Hammer-hardness and strike
   estimates are biased soft.
4. **The mechanism samples carry far more hall** than Salamander's tightly
   edited ones (−17.5 dB of late energy at 0.3 s against −41.6), which is a
   real cost to the `noise` stage.

The library ships **no SFZ**. `piano-tuner adapt bitklavier-piano-bar` writes
one over the tree from the library's description; it is a generated
measurement input and is regenerated rather than edited.

## `presets/upright-parlour.toml` — Knight upright

Estimated from the **Versilian Community Sample Library**'s "Upright Piano,
Knight": a Knight upright (Alfred Knight Ltd, London), recorded mid-close and
acoustically neutral per VCSL's own contribution standard.

| | |
|---|---|
| instrument | Knight upright piano |
| library | VCSL — "Upright Piano, Knight" (from the VSCO 2 Pro sample set) |
| author | **Versilian Studios LLC**; the Knight upright was sampled by **Simon Dalzell** of Ivy Audio |
| licence | **CC0 1.0** — <https://creativecommons.org/publicdomain/zero/1.0/> |
| licence provenance | the repository's `LICENSE` is the CC0 1.0 legal code verbatim; the README states "no royalties, no credit, no special terms" |
| source | <https://github.com/sgossner/VCSL>, bundle at <https://versilian-studios.com/vcsl-keys/> |
| shape | 45 keys — whole tones A0 upward plus C8 — over the full 88-key compass, 2 velocity layers, published at 44.1 kHz / 24 bit |
| fetch | `data/fetch_vcsl_knight_upright.sh` |
| ships | estimated parameters only |
| does not ship | the recordings, in any form, whole or excerpted |

CC0 asks for nothing. The attribution above is recorded anyway: knowing which
piano a preset is a measurement *of* is a property of the measurement, not a
licence obligation.

**Three properties of this library that are part of every number estimated
from it:**

1. **Two velocity layers.** The survey reduces a key to the median over its
   layers — sixteen measurements on Salamander, two here — and the hammer fit
   gets 2 × 45 = 90 (velocity, value) pairs against Salamander's 16 × 30 = 480.
   Per-key brightness against velocity is out of scope for this preset by
   construction, not by omission.
2. **Resampled once, offline, 44 100 → 48 000 Hz**, by the crate's own
   band-limited sinc resampler, written to float WAV so that no dither
   decision enters the material. The tree the estimators read is not the tree
   that was published, and that is a stated systematic of this preset.
3. **The shipped SFZ is not used as a measurement input.** It is auto-generated
   and carries per-region `volume` that compresses the piano's real 10.5 dB
   layer difference to about 5, per-sample `tune` of up to −47 cents from the
   generator's bass pitch-detection failures, and `offset` trims of the attack.
   It is kept in the tree as `shipped-VCSL-generated.sfz` for reference and
   nothing reads it.

---

## Candidates examined and refused

Recorded here so the same ground is not walked twice, and so that "we looked"
is a checkable claim rather than a memory.

| library | instrument | why not |
|---|---|---|
| **Piano in 162** (Ivy Audio) | Steinway Model B | The best-shaped grand in the survey — 88 chromatic keys, 5 layers, 2 round robins, a close pair inside the piano — and **refused**: the product page states "Copyright © 2015 by Ivy Audio … public redistribution of the library is prohibited", there is no licence URL and no derivative grant, and the only distribution today is BitTorrent. Worth noting: its author, **Simon Dalzell**, is the same person who recorded the Knight upright above, is named and contactable, and an explicit written permission would unlock it. |
| **Splendid Grand Piano** | claimed Steinway | "Public Domain by AKAI" — **unverifiable**; no rightsholder statement anywhere. Unclear provenance is refused. |
| **Maestro Concert Grand v2** (Mats Helgesson) | Yamaha CFIII | "All rights reserved… You may not modify and spread this soundfont without the author's written permission." |
| **Church Steinway** (Pianobook, Richard Luke) | Steinway D | Pianobook terms, not an open licence. Also wet by design — a Glasgow church acoustic — against `SHIPPING.md`'s "no room modelling, ever". |
| **BiVib** (Papetti/Avanzini, Zenodo) | Yamaha DC3 M4 grand + DU1A upright | By far the best density (every key × 10 velocities, calibrated to physical SPL) and **deferred, not refused**: it is **binaural dummy-head audio**, which is an HRTF rather than a mic pair `[voicing.mics]` can represent; it is 33 GB; and it is **CC BY-NC-SA 4.0**. `SHIPPING.md` accepts NC for this free product, but the **ShareAlike** half is a copyleft on derivatives sitting against an MIT repository, and that is an owner decision. |
| **University of Iowa MIS piano** | Steinway model B, 2 × Neumann KM 84 | **Deferred to an owner decision, and the one worth making.** It is the only free candidate with **88 genuinely recorded keys**, which is exactly the material the "per-key brightness tilt not drawn for unsampled keys" gap says only more recorded keys can close. Its terms are an explicit unrestricted grant from a named rightsholder — "these recordings have been freely available on this website and may be downloaded and used for any projects, without restrictions" (<https://theremin.music.uiowa.edu/MIS.html>) — but **not a formal licence instrument**, and it needs an AIFF decoder, a generated map and 44.1 → 48 resampling. |
| **City Piano** (bigcat Instruments) | Baldwin baby grand | Deliberately dry and well shaped, but the packager's public-domain claim has **no upstream provenance for the raw recordings**, and distribution is MediaFire only. Flagged, not taken. |
| **Upright Piano KW** (FreePats) | Kawai upright | CC0 and clean, but **disqualified by its material rather than its licence**: the map is `loop_mode=loop_continuous` and the files are truncated to the loop — A4 fortissimo is **2.96 s long**. There is no decay to measure, so `tail`, `decay`, `sympathetic` and `halo` all read a cut tape. |
| **VCSL "Grand Piano, S Model B 1895"** | 1895 Steinway B | CC0, 42 keys whole-tone — but its own folder carries `NORMALIZED.txt`: "The sustain and non-sustain articulations for this instrument have been normalized." Usable for tuning, inharmonicity and decay; useless for `level` and the velocity law. A candidate for a later voice, with that written down. |
| **Headroom Piano** (Bengt Nilsson) | Yamaha C3 | CC BY 4.0 and usable; adds little next to Salamander's C5. |
| **FreePats honky-tonk** (Francis Bacon player piano) | player piano | CC0; a period voice for a later range, too small for a factory run today. |
| **FreePats YDP Grand** | Yamaha Disklavier Pro | CC BY 3.0, but **SF2 only** and 36 MiB — too little material to be worth a factory run. |

---

Action-noise levels and structure-borne bandwidth were cross-checked against
figures published in the piano-acoustics literature (Askenfelt; Goebl, Bresin
& Galembo; Lehtonen, Askenfelt & Välimäki), cited where used in `PHYSICS.md`
and `DECISIONS.md`.
