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
