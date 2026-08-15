//! The listener's test, made permanent.
//!
//! `DECISIONS.md` 284 opened on a complaint that no standing gate could have
//! caught: playing the Ode to Joy excerpt, one note of the melody — C4 — was
//! heard as textured unlike the notes either side of it. Everything the
//! milestone was then measured with is a *compass* statistic (88 keys struck
//! alone, each scored against its own neighbours) or a *phrase* statistic (a
//! mean log-mel distance over six pieces of music). Neither of them is a tune
//! with one note wrong in it: the compass never plays a melody, and a mean over
//! a phrase moves by hundredths when one of its thirty notes is sour.
//!
//! So this file plays the melody. `estimate::melody` renders the excerpt's
//! soprano line alone through the engine and through the recordings of the same
//! piano, measures three textures per note, removes the line's own register
//! trend with a Theil-Sen line, and asks whether any note stands out further
//! than the *recordings'* own worst note does. The bar is the piano, which is
//! the only bar that means anything: a real melody is not even either.
//!
//! One test per column, because which way a note fails to belong is the
//! attribution and a single verdict would throw it away — and one more, which is
//! what makes the other three worth having:
//! [`the_gate_fails_on_the_preset_this_milestone_started_from`] runs the same
//! measurement with item 284 **undone in memory**: every key named in
//! `notes.synthesized_texture` has its drawn `partial_gains` row and its drawn
//! `false_beat` splits removed, which is the preset as it stood at the head of
//! the milestone (the 28 fitted keys, C4 among them, were never touched by it).
//! A gate nobody has seen fail is not a gate.
//!
//! # One of these is red, on purpose and by name
//!
//! `no_note_of_the_line_wobbles_unlike_the_rest` fails, and `DECISIONS.md` 298
//! and 300 are why. It fails on **both** presets, and what is left in it after
//! item 300 is not the drawn texture. It used to be: the column stood at
//! **1.28 dB** against a bar of 0.26 because F4's *drawn* splits took that note
//! from 1.97 dB of wobble to 2.64, split depth being the one quantity item 284
//! drew without closing it on the render. Item 300 closes it — F4 comes back to
//! **1.54 dB against the piano's 1.98** and the column to **0.43 at D4** — and
//! the attribution is now measured from the other side: clearing every drawn
//! split from the shipped preset moves the column from 0.43 to **0.44**, so the
//! drawn splits are worth **nothing** in it, and clearing the *measured* ones
//! too takes it to 0.26. What remains is the engine's own coupled unison being
//! more uneven from key to key than the piano's — D4, which carries no drawn row
//! and no split of any kind, wobbles 2.21 dB against its recording's 1.95 — and
//! that predates every one of these milestones. The gate is left failing rather
//! than skipped or widened, because that is the only honest way to carry a
//! defect nobody has fixed.
//!
//! Every test here needs the Salamander library and skips itself without it,
//! the same way `tests/reference_cache.rs` does.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use piano_emulator::preset::Preset;
use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_emulator::types::Event;
use piano_tuner::audio::Audio;
use piano_tuner::cache;
use piano_tuner::estimate::melody::{self, Column, LineNote, NoteTexture};
use piano_tuner::realism::{Phrase, VelocityLayers};
use piano_tuner::sampler::{Sampler, SamplerEvent, SAMPLER_VERSION};
use piano_tuner::{SampleLibrary, SAMPLE_RATE};

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn sfz() -> Option<PathBuf> {
    let path = repo()
        .join("data/salamander")
        .join("SalamanderGrandPiano-V3+20200602.sfz");
    path.exists().then_some(path)
}

fn shipped_preset() -> Preset {
    Preset::load(&repo().join("presets/salamander-c5.toml")).expect("the measured preset loads")
}

/// The same preset with `DECISIONS.md` 284 taken back out of it: every key the
/// milestone drew for loses its row and its splits, and the keys it measured
/// keep theirs.
fn without_drawn_texture(mut preset: Preset) -> Preset {
    let drawn = preset.notes.synthesized_texture.clone();
    assert!(
        !drawn.is_empty(),
        "the shipped preset names no drawn keys; this test has nothing to undo"
    );
    for key in drawn {
        let i = usize::from(key - 21);
        if let Some(row) = preset.notes.partial_gains.get_mut(i) {
            row.clear();
        }
        if let Some(row) = preset.notes.false_beat.get_mut(i) {
            row.clear();
        }
    }
    preset.notes.synthesized_texture.clear();
    preset.validate().expect("undoing a milestone is still legal");
    preset
}

// ---------------------------------------------------------------------------
// Rendering the line
// ---------------------------------------------------------------------------

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

/// The recordings playing the same line, cached to disk the way every other
/// reference render in this repository is: it is a function of the sampler, the
/// library and the phrase, none of which move when the engine does.
fn render_reference(
    sfz: &Path,
    phrase: &Phrase,
    name: &str,
    events: &[piano_tuner::TimedEvent],
) -> Audio {
    let mut key = cache::Fingerprint::new();
    key.str("tests/melody/reference")
        .u64(u64::from(SAMPLER_VERSION))
        .file(sfz)
        .expect("the sfz is readable")
        .u64(u64::from(SAMPLE_RATE))
        .str(phrase.name)
        .str(name)
        .f64(phrase.duration_s);
    let dir = cache::reference_dir(&repo().join("data/salamander"));
    let path = dir.join(format!("melody-{}-{name}-{}.wav", phrase.name, key.hex()));
    let rendered = cache::audio(&path, || {
        let mut sampler = Sampler::new(sfz)?;
        sampler.render(events, phrase.duration_s)
    })
    .expect("the reference line renders");
    melody::align_reference(&rendered, phrase.events[0].time_s)
}

/// Both lines measured and scored, for one preset.
fn score(preset: &Preset, sfz: &Path) -> (Vec<Column>, Vec<NoteTexture>, Vec<NoteTexture>) {
    let phrase = melody::soprano();
    let notes: Vec<LineNote> = melody::line_notes();
    let sr = f64::from(SAMPLE_RATE);
    let partial_hz = |key: u8| -> Vec<f64> {
        let params = preset.string_params(key);
        (1..=piano_tuner::series::PARTIALS)
            .map(|k| f64::from(params.partial_freq(k)))
            .collect()
    };

    let layers = VelocityLayers::from_library(
        &SampleLibrary::from_sfz(sfz).expect("the library reads"),
    )
    .expect("the library has velocity layers");
    let engine = render_engine(preset, &phrase);
    let reference = render_reference(sfz, &phrase, "reference", &phrase.events);
    let alt = render_reference(sfz, &phrase, "alt-layer", &layers.shift(&phrase.events));
    let engine_notes = melody::measure_line(&engine.mono(), sr, &notes, &partial_hz);
    let reference_notes = melody::measure_line(&reference.mono(), sr, &notes, &partial_hz);
    let layer_notes = melody::measure_line(&alt.mono(), sr, &notes, &partial_hz);
    let columns = melody::compare(
        &melody::per_key(&engine_notes),
        &melody::per_key(&reference_notes),
        &melody::per_key(&layer_notes),
    );
    (columns, engine_notes, reference_notes)
}

// ---------------------------------------------------------------------------
// The gate
// ---------------------------------------------------------------------------

/// The shipped instrument's line, measured once for the whole file: three
/// renders and 28 windowed notes are not worth doing three times.
fn shipped() -> Option<&'static Vec<Column>> {
    static ONCE: OnceLock<Option<Vec<Column>>> = OnceLock::new();
    ONCE.get_or_init(|| {
        let sfz = sfz()?;
        let (columns, _, _) = score(&shipped_preset(), &sfz);
        print!("{}", melody::report(&columns));
        Some(columns)
    })
    .as_ref()
}

/// One column of the gate: no note of the engine's line may stand further off
/// the line's own register trend than the recordings' worst note stands off
/// theirs.
fn gate(metric: &str) {
    let Some(columns) = shipped() else {
        eprintln!("no data/salamander in this tree; skipping the melody gate");
        return;
    };
    let c = columns
        .iter()
        .find(|c| c.metric == metric)
        .expect("a named column");
    assert!(
        c.pass,
        "{} stands out {:.2} at {} against a bar of {:.2} \
         (the piano's own worst note {:.2}, its velocity layer {:.2})\n{}",
        c.metric,
        c.standout,
        melody::note_name(c.standout_key),
        c.bar,
        c.reference_standout,
        c.layer_standout,
        melody::report(std::slice::from_ref(c))
    );
}

/// The drawn `notes.partial_gains` rows, heard as a tune.
#[test]
fn no_note_of_the_line_is_rougher_than_the_rest() {
    gate("roughness");
}

/// **The documented red.** The `notes.false_beat` splits, heard as a tune: the
/// line's worst note wobbles 0.43 dB off a trend whose piano wobbles 0.21 dB off
/// its own. Since `DECISIONS.md` 300 that note is D4, which carries no drawn
/// row and no split at all, and the drawn splits are worth 0.01 dB of it. See
/// this file's header and `DECISIONS.md` 296, 298 and 300.
#[test]
fn no_note_of_the_line_wobbles_unlike_the_rest() {
    gate("wobble");
}

/// Brilliance at absolute frequency, heard as a tune.
#[test]
fn no_note_of_the_line_is_brighter_than_the_rest() {
    gate("hf");
}

/// The same three columns on the instrument this milestone started from: C4
/// with its measured row and its measured splits, and D4/E4/F4/G4 with nothing
/// at all.
///
/// It is not asserted *which* note fails — the melody's other four pitches are
/// all bare there, so the seam can be read from either side of it — only that
/// the line is measurably uneven, which is what makes the three gates above
/// statements about the instrument rather than about the metric.
#[test]
fn the_gate_fails_on_the_preset_this_milestone_started_from() {
    let Some(sfz) = sfz() else {
        eprintln!("no data/salamander in this tree; skipping the melody falsification");
        return;
    };
    let before = without_drawn_texture(shipped_preset());
    let (columns, _, _) = score(&before, &sfz);
    let text = melody::report(&columns);
    println!("{text}");
    assert!(
        columns.iter().any(|c| !c.pass),
        "the pre-texture instrument passes every column, so the gate does not \
         test what it was written for\n{text}"
    );
}

/// The two lines are the same music, so a note measurable on one is measurable
/// on the other, and the gate is never comparing 28 notes with 12.
#[test]
fn both_lines_measure_the_same_notes() {
    let Some(sfz) = sfz() else {
        eprintln!("no data/salamander in this tree; skipping");
        return;
    };
    let (_, engine, reference) = score(&shipped_preset(), &sfz);
    assert_eq!(engine.len(), 28);
    assert_eq!(reference.len(), 28);
    for (e, r) in engine.iter().zip(&reference) {
        assert_eq!(e.key, r.key);
        assert!(
            e.values().iter().chain(r.values().iter()).all(|v| v.is_finite()),
            "key {} at {:.2} s reads {:?} / {:?}",
            e.key,
            e.onset_s,
            e.values(),
            r.values()
        );
    }
}
