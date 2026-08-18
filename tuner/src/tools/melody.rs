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
        let (population, _, _, _, _) = measure_phrase(
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
| `balance` | `10 log10(E_L / E_R)` at the note's own `f0`, heterodyned, dB — positive is a left lean | `[voicing.mics]`, against the recording's own lean at the same note (`DECISIONS.md` 446) |\n\n\
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
`DECISIONS.md` 341, 394, 446. `strike`, `channel` and `balance` ask a different \
question from the three above them, so they are gated on a different number. The \
three ask *does one note of the line stand out from the rest*, which the engine \
answers on its own and which needs no recording of the note. These three ask *is \
the mechanism as loud against the note as the piano's is*, *do the two \
loudspeakers play this note as the piano's two channels do* and *does this \
note's own fundamental come out of the loudspeaker the piano's comes out of* — \
all comparisons with a recording, and therefore only meaningful at a key the \
library recorded, so all are scored on the **recorded ladder** and not on the \
line at all, as the median over those keys of `engine - recording`.\n\n\
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
(`DECISIONS.md` 446). It is also the **one column of this board that carries \
both verdicts**: the balance half convicts a uniform lean, which a median over \
nine recorded keys can see and the line's own trend cannot, and the spread half \
convicts note-to-note jumps, which the line answers on its own and a median \
cannot see because they cancel in it. Its spread verdict is in the evenness \
table above and its balance verdict is here.\n\n\
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
                if n.recorded {
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
| at | key | source | roughness e/r | wobble e/r | hf e/r | strike e/r |\n|--:|---|---|--:|--:|--:|--:|\n"
        );
        for (e, r) in engine.iter().zip(reference) {
            let _ = writeln!(
                out,
                "| {:.2} | {} | {} | {:.2}/{:.2} | {:.2}/{:.2} | {:.1}/{:.1} | {:.1}/{:.1} |",
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
                r.strike_db
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
