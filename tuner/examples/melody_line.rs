//! The melody-evenness gate as a driver: the same measurement
//! `tests/melody.rs` gates on, printed in full and with the instrument
//! modifiable, so that a failure can be attributed to the table that causes it.
//!
//! ```sh
//! cargo run --release -p piano-tuner --features diagnostics --example melody_line \
//!     -- [data/salamander] [renders/melody] [presets/salamander-c5.toml] [flags]
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
//! | `--clear-key K` | one key's row and splits |
//!
//! It also writes the two rendered lines into the output directory, because the
//! complaint this gate exists for was made by listening to them.

use std::path::{Path, PathBuf};

use piano_emulator::preset::Preset;
use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_emulator::types::Event;
use piano_tuner::audio::Audio;
use piano_tuner::cache;
use piano_tuner::estimate::melody;
use piano_tuner::realism::{Phrase, VelocityLayers};
use piano_tuner::sampler::{Sampler, SamplerEvent, SAMPLER_VERSION};
use piano_tuner::{SampleLibrary, SAMPLE_RATE};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
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
    let clear = |preset: &mut Preset, key: u8, gains: bool, splits: bool| {
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
    };
    for (i, arg) in args.iter().enumerate() {
        match arg.as_str() {
            "--before" => {
                for &key in &drawn {
                    clear(&mut preset, key, true, true);
                }
                preset.notes.synthesized_texture.clear();
                what = "before (item 284 undone)".into();
            }
            "--no-drawn-splits" => {
                for &key in &drawn {
                    clear(&mut preset, key, false, true);
                }
                what = "drawn splits removed".into();
            }
            "--no-drawn-gains" => {
                for &key in &drawn {
                    clear(&mut preset, key, true, false);
                }
                what = "drawn gain rows removed".into();
            }
            "--no-splits" => {
                for row in preset.notes.false_beat.iter_mut() {
                    row.clear();
                }
                what = "every false_beat row removed".into();
            }
            "--clear-key" => {
                let key: u8 = args[i + 1].parse()?;
                clear(&mut preset, key, true, true);
                what = format!("key {key} cleared");
            }
            _ => {}
        }
    }
    preset.validate()?;

    let phrase = melody::soprano();
    let sr = f64::from(SAMPLE_RATE);
    let notes = melody::line_notes();
    let partial_hz = |key: u8| -> Vec<f64> {
        let params = preset.string_params(key);
        (1..=piano_tuner::series::PARTIALS)
            .map(|k| f64::from(params.partial_freq(k)))
            .collect()
    };

    let layers = VelocityLayers::from_library(&SampleLibrary::from_sfz(&sfz)?)?;
    let engine = render_engine(&preset, &phrase);
    let reference = render_reference(&sfz, &data, &phrase, "reference", &phrase.events)?;
    let alt = render_reference(
        &sfz,
        &data,
        &phrase,
        "alt-layer",
        &layers.shift(&phrase.events),
    )?;
    engine.write_wav(out.join("ode_soprano_engine.wav"))?;
    reference.write_wav(out.join("ode_soprano_reference.wav"))?;

    let engine_notes = melody::measure_line(&engine.mono(), sr, &notes, &partial_hz);
    let reference_notes = melody::measure_line(&reference.mono(), sr, &notes, &partial_hz);
    let layer_notes = melody::measure_line(&alt.mono(), sr, &notes, &partial_hz);
    let engine_keys = melody::per_key(&engine_notes);
    let reference_keys = melody::per_key(&reference_notes);
    let layer_keys = melody::per_key(&layer_notes);
    let verdicts = melody::compare(&engine_keys, &reference_keys, &layer_keys);

    println!(
        "melody line: {} notes over {} pitches, engine on {} ({what}), reference {}\n",
        engine_notes.len(),
        engine_keys.len(),
        preset_path.display(),
        sfz.display()
    );
    println!("{}", melody::report(&verdicts));

    // Every occurrence, so that a pitch whose median hides a spread shows it.
    println!("every note, in time order:");
    println!("  {:>6}  {:<4}  {:>9}  {:>7}  {:>7}", "at", "key", "roughness", "wobble", "hf");
    for (e, r) in engine_notes.iter().zip(&reference_notes) {
        println!(
            "  {:6.2}  {:<4}  {:5.2}/{:<5.2}  {:3.2}/{:<3.2}  {:6.1}/{:<6.1}",
            e.onset_s,
            melody::note_name(e.key),
            e.roughness_db,
            r.roughness_db,
            e.wobble_db,
            r.wobble_db,
            e.hf_db,
            r.hf_db
        );
    }
    let failed: Vec<&str> = verdicts
        .iter()
        .filter(|v| !v.pass)
        .map(|v| v.metric)
        .collect();
    println!(
        "\n{}",
        if failed.is_empty() {
            "every column inside the piano's own line".to_string()
        } else {
            format!("uneven in {failed:?}")
        }
    );

    let report = out.join("MELODY.md");
    std::fs::write(
        &report,
        melody_report(&verdicts, &engine_notes, &reference_notes, &preset_path, &sfz, &what),
    )?;
    println!("{}", report.display());
    Ok(())
}

fn render_engine(preset: &Preset, phrase: &Phrase) -> Audio {
    let events: Vec<RenderEvent> = phrase
        .events
        .iter()
        .map(|e| {
            let event = match e.event {
                SamplerEvent::NoteOn { key, vel } => Event::NoteOn { key, vel },
                SamplerEvent::NoteOff { key, vel } => Event::NoteOff { key, vel },
                other => panic!("the soprano line has no {other:?} in it"),
            };
            RenderEvent::new(e.time_s as f32, event)
        })
        .collect();
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
    let mut key = cache::Fingerprint::new();
    key.str("tests/melody/reference")
        .u64(u64::from(SAMPLER_VERSION))
        .file(sfz)?
        .u64(u64::from(SAMPLE_RATE))
        .str(phrase.name)
        .str(name)
        .f64(phrase.duration_s);
    let path = cache::reference_dir(data)
        .join(format!("melody-{}-{name}-{}.wav", phrase.name, key.hex()));
    let rendered = cache::audio(&path, || {
        let mut sampler = Sampler::new(sfz)?;
        sampler.render(events, phrase.duration_s)
    })?;
    Ok(melody::align_reference(&rendered, phrase.events[0].time_s))
}

/// `MELODY.md`: the standing report for the gate in `tuner/tests/melody.rs`.
fn melody_report(
    columns: &[melody::Column],
    engine: &[melody::NoteTexture],
    reference: &[melody::NoteTexture],
    preset: &Path,
    sfz: &Path,
    what: &str,
) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let _ = write!(
        out,
        "# The melody, note by note\n\n\
The Ode to Joy melody line of `realism::excerpt` played **alone** — no harmony, \
no pedal — through the engine on `{}` ({what}) and through the recordings of the \
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
| `hf` | 2-6 kHz share of the note's power, dB | brilliance, at absolute frequency (`DECISIONS.md` 292) |\n\n\
Each note is measured over {:.2}-{:.2} s of its own strike, found again inside \
the phrase by the largest rise in a 1 ms envelope — the sampler plays every \
recording from its own start, so the offset between a note-on and its hammer is \
a property of one sample file and differs from note to note.\n\n\
## The verdict\n\n\
`stands out` is the largest departure of any note from the engine line's own \
Theil-Sen register trend. The bar is the same statistic on the recordings, and \
never under the same statistic on the recordings' neighbouring velocity layer. \
`seam` is `engine - recording` per note and its departure from the line's median \
error — item 288's S1 on the melody — reported and not gated.\n\n\
| metric | stands out | at | bar | piano | layer | verdict | seam | at |\n\
|---|--:|---|--:|--:|--:|---|--:|---|\n",
        preset.display(),
        sfz.display(),
        melody::NOTE_WINDOW_S.0,
        melody::NOTE_WINDOW_S.1,
    );
    for c in columns {
        let _ = writeln!(
            out,
            "| `{}` | **{:.2}** | {} | {:.2} | {:.2} | {:.2} | {} | {:.2} | {} |",
            c.metric,
            c.standout,
            melody::note_name(c.standout_key),
            c.bar,
            c.reference_standout,
            c.layer_standout,
            if c.pass { "pass" } else { "**FAIL**" },
            c.seam,
            melody::note_name(c.seam_key),
        );
    }
    let _ = write!(
        out,
        "\n## The five pitches\n\n\
`e` is the engine, `r` the recording, `l` the recording's neighbouring velocity \
layer. `resid` is the departure from that line's own trend.\n\n"
    );
    for c in columns {
        let _ = write!(
            out,
            "### `{}`\n\n| key | e | r | l | e resid | r resid | error | seam |\n\
|---|--:|--:|--:|--:|--:|--:|--:|\n",
            c.metric
        );
        for n in &c.notes {
            let _ = writeln!(
                out,
                "| {} | {:.2} | {:.2} | {:.2} | {:+.2} | {:+.2} | {:+.2} | {:+.2} |",
                melody::note_name(n.key),
                n.engine,
                n.reference,
                n.layer,
                n.engine_residual,
                n.reference_residual,
                n.error,
                n.seam
            );
        }
        let _ = writeln!(out);
    }
    let _ = write!(
        out,
        "## Every note, in time order\n\n\
| at | key | roughness e/r | wobble e/r | hf e/r |\n|--:|---|--:|--:|--:|\n"
    );
    for (e, r) in engine.iter().zip(reference) {
        let _ = writeln!(
            out,
            "| {:.2} | {} | {:.2}/{:.2} | {:.2}/{:.2} | {:.1}/{:.1} |",
            e.onset_s,
            melody::note_name(e.key),
            e.roughness_db,
            r.roughness_db,
            e.wobble_db,
            r.wobble_db,
            e.hf_db,
            r.hf_db
        );
    }
    let _ = write!(
        out,
        "\n```sh\ncargo run --release -p piano-tuner --features diagnostics \\\n\
    --example melody_line -- data/salamander renders/melody presets/salamander-c5.toml\n```\n\n\
The gate itself is `cargo test -p piano-tuner --test melody`.\n"
    );
    out
}
