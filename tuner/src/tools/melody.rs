//! The melody-evenness gate as a driver: the same measurement
//! `tests/melody.rs` gates on, printed in full and with the instrument
//! modifiable, so that a failure can be attributed to the table that causes it.
//!
//! ```sh
//! cargo run --release -p piano-tuner -- melody \
//!     [data/salamander] [renders/melody] [presets/salamander-c5.toml] [flags]
//! ```
//!
//! Flags, each of which edits the preset **in memory** before it is rendered:
//!
//! | flag | what it makes |
//! |---|---|
//! | `--before` | the instrument at the head of `DECISIONS.md` 284: every key named in `notes.synthesized_texture` loses its drawn row and its drawn splits |
//! | `--no-drawn-splits` | the drawn `false_beat` rows only, removed; the drawn gain rows stay |
//! | `--no-drawn-gains` | the drawn `partial_gains` rows only, removed |
//! | `--no-splits` | every `false_beat` row on the compass, drawn or measured |
//! | `--no-drawn-decay` | the drawn `partial_sigma_scale` rows — every key named in `notes.synthesized_decay` — which is the other side of the C4 tail seam |
//! | `--no-drawn-low` | the same rows' cells **under 2 kHz** only, returned to 1.0: the instrument of `DECISIONS.md` 331, and the falsification `tests/melody.rs` carries |
//! | `--clear-key K` | one key's row and splits |
//! | `--clear-decay K` | one key's `partial_sigma_scale` row: `--clear-decay 60` is the C4 experiment |
//! | `--clear-low K` | one key's cells under 2 kHz: `--clear-low 60` is item 334's own row, worth 5.33 dB of C4's tail |
//! | `--before-noise` | `[noise.strike]` at the level and velocity law it carried before `DECISIONS.md` 340 refit it: the instrument the `strike` column fails on |
//! | `--no-strike` | `[noise.strike]` silenced outright |
//!
//! It also writes the four rendered lines into the output directory, because the
//! complaint this gate exists for was made by listening to them.

use std::path::{Path, PathBuf};

use piano_emulator::preset::Preset;
use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_tuner::audio::Audio;
use piano_tuner::estimate::melody::{self, Column, LineNote, NoteTexture, Window};
use piano_tuner::realism::{Phrase, RecordedKeys, VelocityLayers};
use piano_tuner::sampler::engine_events;
use piano_tuner::{SampleLibrary, SAMPLE_RATE};

pub fn run(args: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
    let positional: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    let data = PathBuf::from(
        positional
            .first()
            .map(|s| s.as_str())
            .unwrap_or("data/salamander"),
    );
    let out = PathBuf::from(
        positional
            .get(1)
            .map(|s| s.as_str())
            .unwrap_or("renders/melody"),
    );
    let preset_path = PathBuf::from(
        positional
            .get(2)
            .map(|s| s.as_str())
            .unwrap_or("presets/salamander-c5.toml"),
    );

    let sfz = data.join("SalamanderGrandPiano-V3+20200602.sfz");
    if !sfz.exists() {
        eprintln!(
            "the reference piano is not here: {}\nrun data/fetch_salamander.sh first (707 MiB).",
            sfz.display()
        );
        std::process::exit(2);
    }
    std::fs::create_dir_all(&out)?;

    let mut preset = Preset::load(&preset_path)?;
    let mut what = String::from("as shipped");
    let drawn = preset.notes.synthesized_texture.clone();
    let drawn_decay = preset.notes.synthesized_decay.clone();
    let clear = |preset: &mut Preset, key: u8, gains: bool, splits: bool, decay: bool| {
        let i = usize::from(key.saturating_sub(21));
        if gains {
            if let Some(row) = preset.notes.partial_gains.get_mut(i) {
                row.clear();
            }
        }
        if splits {
            if let Some(row) = preset.notes.false_beat.get_mut(i) {
                row.clear();
            }
        }
        if decay {
            if let Some(row) = preset.notes.partial_sigma_scale.get_mut(i) {
                row.clear();
            }
        }
    };
    // The band `TailCorrection::at` holds at one, put back the way item 331
    // found it: the cells alone, with the two bands above them untouched, so a
    // column that moves has moved for one reason.
    let clear_low = |preset: &mut Preset, key: u8| {
        let params = preset.string_params(key);
        let hz: Vec<f64> = (1..=params.partial_count())
            .map(|k| f64::from(params.partial_freq(k)))
            .collect();
        let i = usize::from(key.saturating_sub(21));
        if let Some(row) = preset.notes.partial_sigma_scale.get_mut(i) {
            for (cell, &f) in row.iter_mut().zip(&hz) {
                if f < piano_tuner::estimate::tail::LOW_BAND.1 {
                    *cell = 1.0;
                }
            }
            while row.last() == Some(&1.0) {
                row.pop();
            }
        }
    };
    for (i, arg) in args.iter().enumerate() {
        match arg.as_str() {
            "--before" => {
                for &key in &drawn {
                    clear(&mut preset, key, true, true, false);
                }
                preset.notes.synthesized_texture.clear();
                what = "before (item 284 undone)".into();
            }
            "--no-drawn-splits" => {
                for &key in &drawn {
                    clear(&mut preset, key, false, true, false);
                }
                what = "drawn splits removed".into();
            }
            "--no-drawn-gains" => {
                for &key in &drawn {
                    clear(&mut preset, key, true, false, false);
                }
                what = "drawn gain rows removed".into();
            }
            "--no-splits" => {
                for row in preset.notes.false_beat.iter_mut() {
                    row.clear();
                }
                what = "every false_beat row removed".into();
            }
            "--no-drawn-decay" => {
                for &key in &drawn_decay {
                    clear(&mut preset, key, false, false, true);
                }
                preset.notes.synthesized_decay.clear();
                what = "drawn partial_sigma_scale rows removed".into();
            }
            "--clear-key" => {
                let key: u8 = args[i + 1].parse()?;
                clear(&mut preset, key, true, true, false);
                what = format!("key {key} cleared");
            }
            "--no-drawn-low" => {
                for &key in &drawn_decay {
                    clear_low(&mut preset, key);
                }
                what = "the drawn rows' cells under 2 kHz returned to 1.0".into();
            }
            "--clear-low" => {
                let key: u8 = args[i + 1].parse()?;
                clear_low(&mut preset, key);
                what = format!("key {key}'s cells under 2 kHz returned to 1.0");
            }
            "--before-noise" => {
                for anchor in preset.noise.strike.level_db.iter_mut() {
                    anchor.db += melody::STRIKE_REFIT_LEVEL_DB;
                }
                preset.noise.strike.velocity_db = melody::STRIKE_VELOCITY_DB_BEFORE;
                what = "[noise.strike] as DECISIONS 340 found it".into();
            }
            "--no-strike" => {
                preset.noise.strike.level_db = vec![piano_emulator::preset::NoiseAnchor {
                    key: 21,
                    db: piano_emulator::preset::SILENT_LEVEL_DB,
                }];
                what = "[noise.strike] silenced".into();
            }
            "--clear-decay" => {
                let key: u8 = args[i + 1].parse()?;
                clear(&mut preset, key, false, false, true);
                preset.notes.synthesized_decay.retain(|&k| k != key);
                what = format!("key {key}'s partial_sigma_scale row cleared");
            }
            _ => {}
        }
    }
    preset.validate()?;

    let library = SampleLibrary::from_sfz(&sfz)?;
    let layers = VelocityLayers::from_library(&library)?;
    let recorded = RecordedKeys::from_library(&library)?;
    let ladder_keys = melody::ladder_keys(&recorded, &melody::line_keys());

    println!(
        "melody line: {} pitches, engine on {} ({what}), reference {}",
        melody::line_keys().len(),
        preset_path.display(),
        sfz.display()
    );
    println!(
        "  scored reference keys on the line: {}   |   transposed and unscored: {}",
        named(&melody::line_keys()
            .into_iter()
            .filter(|&k| recorded.is_recorded(k))
            .collect::<Vec<u8>>()),
        named(&melody::line_keys()
            .into_iter()
            .filter(|&k| !recorded.is_recorded(k))
            .collect::<Vec<u8>>()),
    );
    println!("  bars measured off the recorded ladder: {}\n", named(&ladder_keys));

    let mut columns: Vec<Column> = Vec::new();
    let mut head: Option<(Vec<NoteTexture>, Vec<NoteTexture>)> = None;
    let mut tail: Option<(Vec<NoteTexture>, Vec<NoteTexture>)> = None;
    for window in [Window::Head, Window::Tail] {
        let line_phrase = melody::line_for(window);
        let line_notes = melody::line_notes_for(window);
        let ladder_phrase = melody::ladder(&ladder_keys, window);
        let ladder_notes = melody::ladder_notes(&ladder_keys, window);
        let (line, engine_notes, reference_notes, engine_audio, reference_audio) = measure_phrase(
            &preset,
            &sfz,
            &data,
            &layers,
            &line_phrase,
            &line_notes,
            window,
        )?;
        let (population, _, _, ladder_engine, ladder_reference) = measure_phrase(
            &preset,
            &sfz,
            &data,
            &layers,
            &ladder_phrase,
            &ladder_notes,
            window,
        )?;
        engine_audio.write_wav(out.join(format!("{}_engine.wav", line_phrase.name)))?;
        reference_audio.write_wav(out.join(format!("{}_reference.wav", line_phrase.name)))?;
        // The ladder itself, which is where **every bar on this board comes
        // from** and which was measured and thrown away until item 456. A
        // reader who wants to know whether the engine's D#3 is as loud as the
        // piano's has to be able to listen to the two, and the `loudness`
        // column's whole subject is a thing one hears before one measures it.
        ladder_engine.write_wav(out.join(format!("{}_engine.wav", ladder_phrase.name)))?;
        ladder_reference.write_wav(out.join(format!("{}_reference.wav", ladder_phrase.name)))?;
        columns.extend(melody::compare(window, &line, &population, &recorded));
        let pair = (engine_notes, reference_notes);
        match window {
            Window::Head => head = Some(pair),
            Window::Tail => tail = Some(pair),
        }
    }

    println!("{}", melody::report(&columns));

    // Every occurrence, so that a pitch whose median hides a spread shows it.
    if let Some((engine_notes, reference_notes)) = &head {
        println!("every note of the head window, in time order:");
        println!(
            "  {:>6}  {:<4}  {:>9}  {:>7}  {:>7}  {:>7}",
            "at", "key", "roughness", "wobble", "hf", "strike"
        );
        for (e, r) in engine_notes.iter().zip(reference_notes) {
            println!(
                "  {:6.2}  {:<4}  {:5.2}/{:<5.2}  {:3.2}/{:<3.2}  {:6.1}/{:<6.1}  {:5.1}/{:<5.1}",
                e.onset_s,
                melody::note_name(e.key),
                e.roughness_db,
                r.roughness_db,
                e.wobble_db,
                r.wobble_db,
                e.hf_db,
                r.hf_db,
                e.strike_db,
                r.strike_db
            );
        }
    }
    let failed: Vec<String> = columns
        .iter()
        .filter(|c| !c.pass)
        .map(|c| format!("{} ({}) at {}", c.metric, c.window.name(), melody::note_name(c.standout_key)))
        .collect();
    println!(
        "\n{}",
        if failed.is_empty() {
            "every column inside the recorded register's own scatter".to_string()
        } else {
            format!("uneven in {failed:?}")
        }
    );

    let report = out.join("MELODY.md");
    std::fs::write(
        &report,
        melody_report(
            &columns,
            head.as_ref(),
            tail.as_ref(),
            &recorded,
            &ladder_keys,
            &preset_path,
            &sfz,
            &what,
        ),
    )?;
    println!("{}", report.display());
    Ok(())
}

fn named(keys: &[u8]) -> String {
    if keys.is_empty() {
        return "none".to_string();
    }
    keys.iter()
        .map(|&k| melody::note_name(k))
        .collect::<Vec<String>>()
        .join(" ")
}

#[allow(clippy::type_complexity)]
fn measure_phrase(
    preset: &Preset,
    sfz: &Path,
    data: &Path,
    layers: &VelocityLayers,
    phrase: &Phrase,
    notes: &[LineNote],
    window: Window,
) -> Result<
    (
        melody::Lines,
        Vec<NoteTexture>,
        Vec<NoteTexture>,
        Audio,
        Audio,
    ),
    piano_tuner::Error,
> {
    let sr = f64::from(SAMPLE_RATE);
    let partial_hz = |key: u8| -> Vec<f64> {
        let params = preset.string_params(key);
        (1..=piano_tuner::series::PARTIALS)
            .map(|k| f64::from(params.partial_freq(k)))
            .collect()
    };
    let engine = render_engine(preset, phrase);
    let reference = render_reference(sfz, data, phrase, "reference", &phrase.events)?;
    let alt = render_reference(sfz, data, phrase, "alt-layer", &layers.shift(&phrase.events))?;
    let engine_notes = melody::measure_line(&engine, sr, notes, &partial_hz, window);
    let reference_notes = melody::measure_line(&reference, sr, notes, &partial_hz, window);
    let layer_notes = melody::measure_line(&alt, sr, notes, &partial_hz, window);
    Ok((
        melody::Lines::new(
            melody::per_key(&engine_notes),
            melody::per_key(&reference_notes),
            melody::per_key(&layer_notes),
        ),
        engine_notes,
        reference_notes,
        engine,
        reference,
    ))
}

fn render_engine(preset: &Preset, phrase: &Phrase) -> Audio {
    let events: Vec<RenderEvent> = engine_events::to_render_events(&phrase.events);
    let (left, right) = render_to_buffer(preset, &events, phrase.duration_s as f32);
    Audio::new(SAMPLE_RATE, vec![left, right]).expect("the engine renders stereo")
}

fn render_reference(
    sfz: &Path,
    data: &Path,
    phrase: &Phrase,
    name: &str,
    events: &[piano_tuner::TimedEvent],
) -> Result<Audio, piano_tuner::Error> {
    melody::reference_line(sfz, data, phrase, name, events)
}

/// `MELODY.md`: the standing report for the gate in `tuner/tests/melody.rs`.
#[allow(clippy::too_many_arguments)]
fn melody_report(
    columns: &[Column],
    head: Option<&(Vec<NoteTexture>, Vec<NoteTexture>)>,
    tail: Option<&(Vec<NoteTexture>, Vec<NoteTexture>)>,
    recorded: &RecordedKeys,
    ladder_keys: &[u8],
    preset: &Path,
    sfz: &Path,
    what: &str,
) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let _ = write!(
        out,
        "# The melody, note by note\n\n\
The Ode to Joy melody line of `realism::excerpt` played **alone** — no harmony — \
through the engine on `{}` ({what}) and through the recordings of the \
Yamaha C5 at `{}`. Thirty notes over five pitches; the two half-beat passing \
notes are too short to measure and the other 28 are.\n\n\
This is the listener's own statistic. `COMPASS.md` strikes 88 keys alone and \
scores each against its neighbours; `REALISM.md` averages six phrases into one \
number per column. Neither of them is a tune with one note wrong in it, and a \
tune with one note wrong in it is what opened `DECISIONS.md` 284.\n\n\
## What is measured, per note\n\n\
| metric | definition | what writes it |\n|---|---|---|\n\
| `roughness` | mean absolute step between adjacent partial levels, dB | `notes.partial_gains` |\n\
| `wobble` | median over partials of the RMS of that partial's dB envelope about its own straight-line decay | `notes.false_beat` |\n\
| `hf` | 2-6 kHz share of the note's power, dB | brilliance, at absolute frequency (`DECISIONS.md` 292) |\n\
| `strike` | attack tonality of the first 30 ms from the note's own strike, dB — large is a line spectrum, zero is a continuum | `[noise.strike]`, against the tonal attack (`DECISIONS.md` 341) |\n\
| `channel` | `10 log10((E_L + E_R) / 2 E_M)` over the note's window, dB | `[voicing.mics]`, against the note's own mono fold-down (`DECISIONS.md` 394) |\n\
| `balance` | `10 log10(E_L / E_R)` at the note's own `f0`, heterodyned, dB — positive is a left lean | `[voicing.mics]`, against the recording's own lean at the same note (`DECISIONS.md` 446) |\n\
| `splitting` | that same image position at `f0` **minus** its energy-weighted mean over the note's own partials 2-4, dB — positive means the pitch is left of the note's own colour | `[voicing.mics]`, against the recording's own split at the same note (`DECISIONS.md` 451) |\n\
| `loudness` | A-weighted energy of the note's own window on the **mono fold-down**, dB | `notes.partial_gains`' per-key level, against the recording of the same key (`DECISIONS.md` 456-457) |\n\n\
The first three are measured in **two windows** of the same note, found again \
inside the phrase by the largest rise in a 1 ms envelope — the sampler plays \
every recording from its own start, so the offset between a note-on and its \
hammer is a property of one sample file and differs from note to note.\n\n\
| window | seconds from the strike | what lives there | line |\n|---|---|---|---|\n\
| `head` | {:.2}-{:.2} | `partial_gains`, the hammer's colour | the tune at tempo, unpedalled |\n\
| `tail` | {:.2}-{:.2} | `partial_sigma_scale`, the decay | the tune's own pitches, slowly and legato |\n\n\
`strike` has one window and it is neither of those: the head starts at 0.03 s \
*because* that is past the hammer's noise, so the one span the three texture \
metrics deliberately exclude is the only span this one is about. It is also the \
only column of the four that is a **balance** rather than an evenness — see \
*The balance*, below.\n\n\
The tail columns are `DECISIONS.md` 330. A window that closes at 0.40 s cannot \
see a decay row at all, and the regression that came back at C4 is a decay row. \
They cannot be read at the melody's own tempo, and that is arithmetic rather \
than preference: with the dampers working the melody's 0.45 s notes are over \
before 0.5 s, and with the pedal held instead they are not over but they are not \
alone — three later strikes land inside the window, two of which on this tune \
are the note's own third and fifth harmonics. Measured, on the pedalled line the \
five pitches' tail `hf` spans 2.7 dB where the same five pitches with the tail \
to themselves span 8.3. So the tail is read off the line's own pitches, at the \
line's own velocity, in the order the tune introduces them, one strike every \
2.5 s and each held 2.2 s: every note is let go before the next is struck and \
the window is entirely inside the held note.\n\n\
## Which reference notes are scored\n\n\
`DECISIONS.md` 328, and it is permanent. The library records one key every minor \
third. Of this line's five pitches **exactly one is a recording**:\n\n\
| pitch | the reference note is |\n|---|---|\n",
        preset.display(),
        sfz.display(),
        melody::NOTE_WINDOW_S.0,
        melody::NOTE_WINDOW_S.1,
        melody::TAIL_WINDOW_S.0,
        melody::TAIL_WINDOW_S.1,
    );
    for key in melody::line_keys() {
        let _ = writeln!(
            out,
            "| {} | {} |",
            melody::note_name(key),
            recorded.provenance(key)
        );
    }
    let _ = write!(
        out,
        "\nA transposed note stays in every render and is exactly what a player of \
this library hears. It carries **no per-note score**: its inharmonicity, its \
unison beat, its decay and its brightness are the neighbour's, resampled, and \
scoring the engine against it measures a resampler. It is marked \
*transposed — unscored* wherever a per-note number would otherwise be.\n\n\
That also takes the **bar** off this line: four of its five reference notes are \
two recordings cloned, so their scatter about their own trend is far smaller \
than a piano's and a bar set on it would be a bar set on a resampler. Every bar \
below is measured instead on the **recorded keys of the melody's register**, \
played as the same music through the same window — {} — and is never smaller \
than the **per-take scatter**, the distance between one recorded key and its own \
neighbouring velocity layer, which is the same noise floor `REALISM.md` uses, \
per note instead of per phrase.\n\n\
## The verdict\n\n\
`stands out` is the largest departure of any note from the engine line's own \
Theil-Sen register trend — the engine renders all five pitches from its own \
tables, so all five are its own work. `bar` is the larger of `register` and \
`take`, times {:.2}. `register` is how far the recorded keys' own notes go from \
their own trend, read at **the same order statistic the gate is** — the median \
of the largest of five draws, not the largest of nine, because the largest of \
nine is systematically bigger and a bar set on it would forgive the engine an \
amount that grows with how many keys the library happened to sample. `worst` is \
that population's single worst key, printed for the reader and not used. `take` \
is the per-take scatter. `seam` is `engine - recording` over the recorded keys \
and its departure from their median error, **reported and not gated**: item \
297's objection is half answered — both sides are now the same note — and half \
not, because the engine's absolute distance from the piano moves by several dB \
across a register for reasons no per-note floor covers. `floor` beside it is the \
same statistic with the neighbouring velocity layer standing in for the \
engine.\n\n\
| metric | window | stands out | at | bar | register | worst | take | verdict | seam | at | floor |\n\
|---|---|--:|---|--:|--:|--:|--:|---|--:|---|--:|\n",
        named(ladder_keys),
        melody::ALLOWANCE,
    );
    for c in columns.iter().filter(|c| c.gated_on_spread) {
        let _ = writeln!(
            out,
            "| `{}` | `{}` | **{:.2}** | {} | {:.2} | {:.2} | {:.2} | {:.2} | {} | {:.2} | {} | {:.2} |",
            c.metric,
            c.window.name(),
            c.standout,
            melody::note_name(c.standout_key),
            c.bar,
            c.population_bar,
            c.population_standout,
            c.take_scatter,
            if c.spread_pass { "pass" } else { "**FAIL**" },
            c.seam,
            melody::note_name(c.seam_key),
            c.seam_floor,
        );
    }
    let _ = write!(
        out,
        "\n## The balances\n\n\
`DECISIONS.md` 341, 394, 446, 451. `strike`, `channel`, `balance` and \
`splitting` ask a different question from the three above them, so they are \
gated on a different number. The three ask *does one note of the line stand out \
from the rest*, which the engine answers on its own and which needs no recording \
of the note. These four ask *is the mechanism as loud against the note as the \
piano's is*, *do the two loudspeakers play this note as the piano's two channels \
do*, *does this note's own fundamental come out of the loudspeaker the piano's \
comes out of* and *does the note arrive from one place in the image at all* — \
all comparisons with a recording, and therefore only meaningful where the \
reference note is a real measurement of the piano, as the median of \
`engine - recording`.\n\n\
Three of the four take that median over the **recorded ladder** and score the \
one note of the line that is a recording, which is item 328's rule: a transposed \
reference note's inharmonicity, decay and brightness are a neighbour's run \
through a resampler, and scoring against them measures the resampler. \
**`splitting` takes it over the line's own five pitches**, and the exception is \
arithmetic rather than a concession (`DECISIONS.md` 451): resampling multiplies \
every frequency of a take by one factor and touches neither channel's amplitude, \
so a transposed note's `E_L/E_R` at its own k-th partial *is* the take's, exactly \
— and a split is a difference of two such ratios. It has to be scored there, \
because the defect it was written for is a **band**: over the recorded ladder \
D#3 has its fundamental below the band and its second partial inside it, and C5 \
and D#5 are clear of the band altogether, so a median over the ladder mixes three \
regimes and reports the smallest. The melodic register is one regime.\n\n\
`channel` is `10 log10((E_L + E_R) / 2 E_M)`: what the two loudspeakers put in \
the room against what this note's own mono fold-down says they do. **It is the \
only column on this board, or on any board in this repository, that is not a \
function of that fold-down** — which is why a stereo stage could make C4 four \
decibels louder in the room than its neighbours with every gate green \
(`DECISIONS.md` 392).\n\n\
`balance` is `10 log10(E_L / E_R)` at the note's **own fundamental**, \
heterodyned — *which* loudspeaker the pitch comes out of. It is `channel`'s \
missing half and the reason it is missing is arithmetic: `E_L + E_R` is \
symmetric under swapping the two channels, so an instrument that puts every \
fundamental of a tune seven decibels into the left loudspeaker, against a \
recording that leans one and a half decibels right, moves `channel` by nothing \
(`DECISIONS.md` 446).\n\n\
`splitting` is that same reading at `f0` **minus** the energy-weighted mean of \
it over the note's own partials 2-4 — how far the pitch sits from the note's \
own colour in the image. It is `balance`'s missing half for a reason of the \
same shape one more time: `balance` reads **one** frequency per note, and a \
stage that is a *band* does not move a note, it moves the part of a note that \
lies inside its edges. The band item 422 shipped spans 174.3-456.5 Hz, which \
contains every fundamental of this line (261.6 to 392.0 Hz) and **none** of \
those notes' second partials (523 Hz and up), so every note of the tune had its \
pitch panned one way and its colour left where it was. A listener is handed a \
note arriving from two places at once, with zero cents of tuning error anywhere \
(`DECISIONS.md` 451; the *that C sounds off* percept itself is item 453's \
mono-domain pair — this column is the image half of the session's findings). The \
weights are the pair energy each reading was taken from, so a partial that is \
not sounding — a ratio of two noise floors — weighs nothing.\n\n\
`balance` and `splitting` are the **two columns of this board that carry both \
verdicts**: the balance half convicts a whole line pulled the same way, which a \
median can see and the line's own trend cannot, and the spread half convicts \
note-to-note jumps, which the line answers on its own and a median cannot see \
because they cancel in it. Their spread verdicts are in the evenness table above \
and their balance verdicts are here.\n\n\
The bar is the larger of two things, times {:.2}: the median distance between \
**two takes of one recorded key** (the same key out of the neighbouring velocity \
layer), which is the same velocity-layer floor `REALISM.md` quotes beside every \
cell; and `1.4826·MAD / sqrt(n)` over the ladder, which is how well nine keys \
pin a median that moves across them. The second term is there because the first \
is not a floor for every statistic: `strike` reads 1.64 dB between two layers of \
one key and `channel` reads **0.03**, because the two layers are the same two \
microphones on the same key, and a bar of 0.04 dB would be a bar on the \
recording's dither. The evenness numbers are printed beside both and are not \
gated.\n\n\
| metric | window | balance | bar | one key's two takes | verdict | evenness | at | of |\n\
|---|---|--:|--:|--:|---|--:|---|--:|\n",
        melody::ALLOWANCE,
    );
    for c in columns.iter().filter(|c| c.gated_on_balance) {
        let _ = writeln!(
            out,
            "| `{}` | `{}` | **{:+.2}** | {:.2} | {:.2} | {} | {:.2} | {} | {:.2} |",
            c.metric,
            c.window.name(),
            c.balance,
            c.balance_bar,
            c.balance_bar / melody::ALLOWANCE,
            if c.balance_pass { "pass" } else { "**FAIL**" },
            c.standout,
            melody::note_name(c.standout_key),
            c.bar,
        );
    }
    let _ = write!(
        out,
        "\n## The seam\n\n\
`DECISIONS.md` 456. Every column above carries a `seam` — how far the engine's \
distance from one recorded key departs from the register's **median** distance \
— and until this milestone not one of them was gated on it. `loudness` is the \
column where that is not a detail but the whole defect, and it is the one \
column of this board whose verdict is a seam.\n\n\
It is the A-weighted energy of the note's own head on the **mono fold-down** \
— the only column here that is a function of how *loud* the note is at all. \
`roughness`, `wobble` and `hf` are shapes, `strike`, `channel` and `splitting` \
are ratios and `balance` is a position; a note eight decibels under the piano's \
own at the same key moves none of them, which is exactly what `DECISIONS.md` \
453 found on C4 and what item 272 decided not to write anywhere. A-weighted \
because that is what makes a level comparable across keys — 261.6 and 392.0 Hz \
are 2.3 dB apart on that curve before anything about the piano is considered — \
and mono because items 417 and 451 forbid solving a level in the pair.\n\n\
Its **median** is printed and never gated: for this column that median is the \
engine's master gain against the library's mastering, about 15 dB, and it is \
nobody's error. What is gated is the departure from it, against the same \
statistic with the neighbouring velocity layer standing in for the engine — two \
takes of one piano, or how far a recorded key's own level stands from the \
recorded register's trend, whichever is larger — times {:.2}.\n\n\
| metric | window | seam | at | bar | two layers of one key | the register's own | median (ungated) | verdict |\n\
|---|---|--:|---|--:|--:|--:|--:|---|\n",
        melody::ALLOWANCE,
    );
    for c in columns.iter().filter(|c| c.gated_on_seam) {
        let _ = writeln!(
            out,
            "| `{}` | `{}` | **{:.2}** | {} | {:.2} | {:.2} | {:.2} | {:+.2} | {} |",
            c.metric,
            c.window.name(),
            c.seam,
            melody::note_name(c.seam_key),
            c.seam_bar,
            c.seam_floor,
            c.seam_bar / melody::ALLOWANCE,
            c.balance,
            if c.seam_pass { "pass" } else { "**FAIL**" },
        );
    }
    let _ = write!(
        out,
        "\n`clone bar` is what the bar **would** have been under the retired rule — \
the same statistic on the line's own reference notes and on their neighbouring \
velocity layer — kept so the size of the change is visible rather than asserted:\n\n\
| metric | window | bar (recorded register) | clone line | clone layer |\n|---|---|--:|--:|--:|\n"
    );
    for c in columns.iter().filter(|c| c.gated_on_spread) {
        let _ = writeln!(
            out,
            "| `{}` | `{}` | {:.2} | {:.2} | {:.2} |",
            c.metric,
            c.window.name(),
            c.bar,
            c.clone_standout * melody::ALLOWANCE,
            c.clone_layer_standout * melody::ALLOWANCE,
        );
    }
    // **The two image tables, side by side** — the session's own measurement
    // in the form it was made in (`DECISIONS.md` 449, 451). Everything in them
    // is in the per-column tables below as well; the point of putting them
    // here is that *which loudspeaker a note comes out of* and *whether it
    // comes out of one at all* are two readings of one pair of channels, and a
    // reader comparing the engine's line with the recording's should not have
    // to hold two pages open.
    let image_table = |metric: &str, what: &str| -> String {
        use std::fmt::Write as _;
        let mut table = String::new();
        let Some(c) = columns
            .iter()
            .find(|c| c.metric == metric && c.window == melody::Window::Head)
        else {
            return table;
        };
        let _ = write!(table, "\n\n{what}\n\n|  |");
        for n in &c.notes {
            let _ = write!(table, " {} |", melody::note_name(n.key));
        }
        let _ = write!(table, " median |\n|---|");
        for _ in &c.notes {
            let _ = write!(table, "--:|");
        }
        let _ = writeln!(table, "--:|");
        let median = |pick: &dyn Fn(&melody::LineNoteScore) -> f64| -> f64 {
            let mut v: Vec<f64> = c.notes.iter().map(pick).filter(|x| x.is_finite()).collect();
            v.sort_by(f64::total_cmp);
            if v.is_empty() {
                f64::NAN
            } else {
                v[v.len() / 2]
            }
        };
        let rows: [(&str, fn(&melody::LineNoteScore) -> f64); 3] = [
            ("engine", |n| n.engine),
            ("recordings", |n| n.reference),
            ("error", |n| n.error),
        ];
        for (label, pick) in rows {
            let _ = write!(table, "| {label} |");
            for n in &c.notes {
                let v = pick(n);
                if v.is_finite() {
                    let _ = write!(table, " {v:+.2} |");
                } else {
                    let _ = write!(table, " *unscored* |");
                }
            }
            let _ = writeln!(table, " **{:+.2}** |", median(&pick));
        }
        let _ = writeln!(
            table,
            "\nbalance **{:+.2}** against a bar of {:.2} — {}; the line's own spread \
{:.2} at {} against {:.2} — {}.",
            c.balance,
            c.balance_bar,
            if c.balance_pass { "pass" } else { "**FAIL**" },
            c.standout,
            melody::note_name(c.standout_key),
            c.bar,
            if c.spread_pass { "pass" } else { "**FAIL**" },
        );
        table
    };
    let _ = write!(
        out,
        "\n## The image, note by note\n\n\
Two readings of one pair of channels, over the head window, positive is a left \
lean. `balance` says **which loudspeaker** the pitch comes out of; `splitting` \
says whether the pitch and the note's own colour come out of the same one. \
Neither is a number the mono fold-down every other board in this repository is \
computed on can carry (`DECISIONS.md` 446, 451).{}{}\n",
        image_table(
            "balance",
            "**`balance`** — `10 log10(E_L / E_R)` at each pitch's own fundamental. \
Scored at the one note of this line the library recorded; the median row is the \
line's, and the column's verdict is taken over the recorded ladder."
        ),
        image_table(
            "splitting",
            "**`splitting`** — the same reading at the fundamental minus its \
energy-weighted mean over that note's own partials 2-4. Every note of the line is \
scored, because a resampler carries an image ratio through unchanged \
(`melody::METRIC_ON_LINE`), and the median row **is** the column's verdict."
        ),
    );
    let _ = write!(
        out,
        "\n## The five pitches\n\n\
`e` is the engine, `r` the recording, `l` the recording's neighbouring velocity \
layer. `resid` is the departure from that line's own trend. `error` is present \
only where the reference note is a recording of that key.\n\n"
    );
    for c in columns {
        let _ = write!(
            out,
            "### `{}`, `{}` window\n\n| key | source | e | r | l | e resid | error |\n\
|---|---|--:|--:|--:|--:|--:|\n",
            c.metric,
            c.window.name()
        );
        for n in &c.notes {
            let _ = writeln!(
                out,
                "| {} | {} | {:.2} | {:.2} | {:.2} | {:+.2} | {} |",
                melody::note_name(n.key),
                if n.recorded { "recorded" } else { "transposed" },
                n.engine,
                n.reference,
                n.layer,
                n.engine_residual,
                if n.scored {
                    format!("{:+.2}", n.error)
                } else {
                    "*transposed — unscored*".to_string()
                },
            );
        }
        let _ = write!(
            out,
            "\nThe recorded keys of the register, which is where this column's bar comes from:\n\n\
| key | e | r | r resid | l | take | error | seam |\n|---|--:|--:|--:|--:|--:|--:|--:|\n"
        );
        for p in &c.population {
            let _ = writeln!(
                out,
                "| {} | {:.2} | {:.2} | {:+.2} | {:.2} | {:.2} | {:+.2} | {:+.2} |",
                melody::note_name(p.key),
                p.engine,
                p.reference,
                p.reference_residual,
                p.layer,
                p.take_delta,
                p.error,
                p.seam
            );
        }
        let _ = writeln!(out);
    }
    for (title, pair) in [("head", head), ("tail", tail)] {
        let Some((engine, reference)) = pair else {
            continue;
        };
        let _ = write!(
            out,
            "## Every note of the `{title}` window, in time order\n\n\
| at | key | source | roughness e/r | wobble e/r | hf e/r | strike e/r | balance e/r | splitting e/r |\n\
|--:|---|---|--:|--:|--:|--:|--:|--:|\n"
        );
        for (e, r) in engine.iter().zip(reference) {
            let _ = writeln!(
                out,
                "| {:.2} | {} | {} | {:.2}/{:.2} | {:.2}/{:.2} | {:.1}/{:.1} | {:.1}/{:.1} | \
{:+.1}/{:+.1} | {:+.1}/{:+.1} |",
                e.onset_s,
                melody::note_name(e.key),
                if recorded.is_recorded(e.key) {
                    "recorded"
                } else {
                    "transposed"
                },
                e.roughness_db,
                r.roughness_db,
                e.wobble_db,
                r.wobble_db,
                e.hf_db,
                r.hf_db,
                e.strike_db,
                r.strike_db,
                e.balance_db,
                r.balance_db,
                e.split_db,
                r.split_db
            );
        }
        let _ = writeln!(out);
    }
    let _ = write!(
        out,
        "```sh\ncargo run --release -p piano-tuner -- melody \\\n\
    data/salamander renders/melody presets/salamander-c5.toml\n```\n\n\
The gate itself is `cargo test -p piano-tuner --test melody`.\n"
    );
    out
}
