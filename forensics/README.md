# forensics — the lab instruments behind `DECISIONS.md`

These are **one-shot instruments**, not tools. Each one was built to settle a
single question, was run, printed a table into a numbered item of
`DECISIONS.md`, and then had nothing left to do. They are kept because an item
that quotes a measurement is only as good as the ability to take that
measurement again, and deleting the instrument would leave the claim with
nothing under it.

They are therefore **not part of the build**. `forensics` is a workspace member
so that it shares the lockfile, the profiles and the path dependencies, but it
is excluded from the workspace's `default-members`: `cargo build`, `cargo test`
and `cargo clippy` at the root never compile it. Build it on purpose:

```sh
cargo build -p forensics                             # all of them
cargo run --release -p forensics --bin <name> -- ... # one of them
```

What is **not** here is anything that gets run again when the instrument moves.
The preset factory and the standing boards are subcommands of the
`piano-tuner` binary — `piano-tuner --help` lists them — and they are covered
by the workspace's tests and lints like any other shipped code. The dividing
line is re-runnability, not size or age: a program that answers a question once
lives here; a program that answers the same question about every new preset
lives there.

Each file's own header says what it measured, how, and which items it produced;
the one-line summaries below are only an index.

| instrument | what it settled |
|---|---|
| `jitter_forensics` | The instantaneous frequency of a composite partial, on the engine and on the recording, attributed to the mechanism that produces it. The measurement it found was promoted into `piano_tuner::motion`. |
| `eigenmode_prototype` | The coupled-eigenmode unison, built offline and measured against both the shipped engine and the recording, before it became the engine's own construction. |
| `timbre_ladder` | Ten level-matched renderings of one note, from the Salamander recording to the engine, one rung per hypothesis about what is missing. |
| `ladder_analysis` | The objective measurements over every rung of that ladder, per key. |
| `verify_ladder` | Seven independent checks that the ladder in `renders/timbre-ladder/` is what it claims to be, before anyone listens to it. |
| `verify_milestone_a` | Independent re-measurement, from rendered audio, of everything the Milestone A engine changes claimed. |
| `verify_milestone_b` | The same for the sympathetic-resonance milestone: bridge admittance, duplex segments, per-key stereo spread. |
| `verify_scan` | A cross-check harness with its own metric implementations on purpose — different window, different peak search, different presence rule — against the compass and fit reports. |
| `independent_audit` | The sympathetic milestone again, measured with its own DSP rather than the tuner's. |
| `audit_render` | The renderer that produces the audio `independent_audit` measures. |
| `limiter_probe` | Where `tuner/tests/limiter.rs`'s numbers come from: the six benchmark phrases rendered raw, with the limiter's budget measured on them. |
| `output_gain` | `DECISIONS.md` 42's calibration run as a procedure rather than quoted as a claim. |
| `gain_level` | What a `notes.partial_gains` row does to a key's *level*, measured on the render rather than argued from the row. |
| `key_probe` | One key struck alone with one fitted table removed at a time — what in the preset is responsible for a key the compass flags. |
| `top_octave` | Where the top octave's notes end and what ends them (`DECISIONS.md` 275-276). |
| `analysis_render` | The 65-WAV audio-quality corpus behind `DECISIONS.md` 46's independent audit — T60 holds, velocity ladders, a compass sweep, pedal and halo phrases, headroom chords. |
| `drift_line` | The line `estimate::directivity` inverts `voicing.polarization_pan_spread` through, measured on the engine as it stands. |
| `tail_seam` | Where a note's 0.5-2.0 s brightness comes from, partial by partial, engine against the recording of the same key — the instrument behind `DECISIONS.md` 334-336. |
| `line_noise` | Which of the engine's two mechanism events the melody render's noise is, both measured whole by silencing them one at a time, and the sampler's own lead-in at the recorded keys — `DECISIONS.md` 338-339. |
| `stereo_channels` | What the three listening complaints about `renders/stereo/` had in common: the pair geometry exonerated, the side lift convicted, per channel and per note — `DECISIONS.md` 392. |
| `channel_fidelity` | The recording's own interchannel behaviour at a sixth of an octave over thirty keys, which is what the per-channel board's bars are made of — `DECISIONS.md` 393. |
| `channel_verify` | The per-channel repair re-measured with its own DSP, against the board that graded it. |
| `mono_mechanism` | Whether the recording's own mono sum pays for its nodal band, how much of that the engine's mono already inherited, and — sections added for `DECISIONS.md` 407-411 — which keys carry the difference, which stage of the engine carries it (seven one-stage ablations), and the headroom an energy-conserving mechanism needs against what every fitted knob reaches. |
| `side_injection` | Whether an **antisymmetric** (side-only) radiator can buy the recording's nodal band for free: it fits a per-sixth-octave decorrelated side source over 151-339 Hz until the treated pair-over-mono is the recording's, proves the mono fold-down unmoved, and re-reads every gate statistic on the treated renders — `DECISIONS.md` 417. |
| `onset_probe` | Where the melody board's per-note window actually starts, per side and per envelope block length — the table that turned `DECISIONS.md` 452's onset miss into a number. |
| `onset_truth` | The referee `onset_probe` is graded against: the hammer found in the band it is broadband in, plus a dump of the broadband envelope so an attack's *shape* can be read and not only its time. |
| `onset_choice` | The sweep that picked the shipped detector — block length x high-pass x filter order x forward window, over both of the melody board's lines and both of their sides, scored on worst and median miss. |
| `span_convention` | Seven ways of fitting a partial's decay slope x six spans x nine recorded keys, scored on how much the *ratio* the fit writes moves when the span does — the measurement that picked `estimate::tail::READING_S` (`DECISIONS.md` 454). |
| `melody_contour` | The fast Ode line read in A-weighted 125 ms steps: how much louder C4 is than the note before it and the note after it is than C4, engine against the recording — the owner's percept of `DECISIONS.md` 453 as arithmetic, and item 457's before/after. |
| `duplex_drive` | What actually reaches an undamped segment, in the three units it can arrive in — the hammer's force pulse, the note's own bridge force and the resonance bus — scored not by their peaks but by their transform *at the segment's own centre frequency*, which is the only thing a sub-hertz resonator integrates. The instrument behind `DECISIONS.md` 481. |
| `duplex_ab` | The duplex milestone's own evidence in one run: the C4-repair renders hashed with the segments stripped (they must not move), the segments' own share of the melody, the ladder and a treble phrase, a per-key head-and-tail census, the between-partial targets with the segments in and out, and the A/B WAVs in `renders/duplex/`. `DECISIONS.md` 483-484. |
| `beat_census` | Which (key, partial) cells share an eigenmode beat rate, by key and by partial, on any two presets — the census `engine/tests/partials.rs` reduces to one number, and the evidence that its worst bin is the bottom edge of its own counted range (`DECISIONS.md` 458). |
| `c4_ledger` | A key's **level** and its **octave partial's decay**, engine against the recording of the same key, per fitted-table ablation — the instrument behind `DECISIONS.md` 453. The question `key_probe` deliberately cannot answer, because it removes the common offset before it scores. |
| `decay_probe` | Whether a fitted decay rate is a property of the piano or of the span it was fitted over: the recording's own per-partial slope over four spans, its spread, and the beat depth doing the biasing. |
| `treble_halo` | The treble sympathetic halo measured three ways on the same renders — §4's between-partial census against its own leakage floor, the sub-fundamental band where a struck string has no mode at all, and §5's `harm*` release resonances, which is the halo recorded alone — plus the ablation that charges the shortfall to a path, the frontier of what the halo fit can still reach, and the A/B in `renders/halo/` — whose two sides are each divided by their **own** strike, so the gap a listener hears is the column's own digit and the instrument prints the two side by side (item 506; written raw it played C6 11.5 dB too wide). `DECISIONS.md` 500-506. |
| `preset_variant` | Writes a copy of a preset with one thing changed, so that a standing board can be run on the variant without a hand-edited file. The only entry here that is a *helper* rather than a measurement, and it is here because the experiments in `DECISIONS.md` 338-341 are only reproducible with it. |

Most of them write into `renders/` or into `target/`, both gitignored, and
several need the Salamander library — `data/fetch_salamander.sh` first, 707 MiB
into the gitignored `data/`.

## The documents cited elsewhere that are not in the repository

`renders/` is gitignored, so three reports that the engine's source, its tests
and `docs/history/FUNDAMENTALS.md` cite by name are **not in a fresh checkout**.
They are not lost — each is the output of an instrument here, and that is what
this crate is for. Run the instrument to get the document back:

| cited as | regenerate with |
|---|---|
| `renders/jitter/JITTER.md` | `cargo run --release -p forensics --bin jitter_forensics` |
| `renders/jitter/EIGENMODE.md` | `cargo run --release -p forensics --bin eigenmode_prototype` |
| `renders/timbre-ladder/ANALYSIS.md` | `cargo run --release -p forensics --bin timbre_ladder`, then `--bin ladder_analysis` |

Each is a measurement of the engine **as it stood when the item that quotes it
was written**, so a re-run on today's engine answers the same question about a
different instrument and will not reproduce the quoted numbers — `DECISIONS.md`
281 and 322 list `renders/analysis/`, `renders/jitter/` and
`renders/timbre-ladder/` as deliberately stale for exactly that reason. The
numbers in the numbered items stand as the record; these commands are how the
*measurement* is reproduced, not the value.
