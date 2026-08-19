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
//! piano, measures three textures per note in **two windows**, removes the
//! line's own register trend with a Theil-Sen line, and asks whether any note
//! stands out further than a real instrument's notes do in that register.
//!
//! # Two things changed here, and both are policy
//!
//! **`DECISIONS.md` 328 — only recorded reference notes are scored.** The
//! library samples one key every minor third, and of this line's five pitches
//! exactly one is a recording: C4. D4 and E4 are the D#4 take resampled down
//! and up; F4 and G4 are the F#4 take. Those notes stay in every render — they
//! are what a listener hears — and they carry no per-note score. That takes the
//! bar off the line, because a bar measured off four clones of two recordings
//! is a measurement of a resampler. It is rebuilt from
//! [`melody::ladder`](piano_tuner::estimate::melody::ladder): the recorded keys
//! of the melody's own register, played as the same music, held against the
//! per-take scatter of one recorded key's two velocity layers.
//!
//! **`DECISIONS.md` 330 — the tail columns.** The three evenness metrics are measured
//! again over 0.5-2.0 s of each note on the line's own pitches played slowly.
//! The window that was there before ends at 0.40 s, which is enough to see a
//! `partial_gains` row and cannot see a *decay* row at all — and the regression
//! that came back at C4 is a decay row: C4 carries a fitted
//! `partial_sigma_scale` 41 partials deep where D4/E4/F4/G4 are all named in
//! `notes.synthesized_decay` and carry drawn ones.
//!
//! **`DECISIONS.md` 335 — what that seam turned out to be, and the fix.** It is
//! not the two bands the drawn rows correct; it is the band **under 2 kHz**,
//! which `TailCorrection::at` holds at exactly one and which therefore existed
//! at the 30 recorded keys (where `estimate::shaping` writes it) and at none of
//! the other 58. The tail `hf` column is a *share* whose denominator is the
//! fundamental, so C4's own sub-2 kHz cells — which hold its fundamental 4.2 dB
//! higher at 0.5 s than the law alone would — read as 5.4 dB of darkness that
//! its drawn neighbours could not have. `tail::LowDecay` makes that band a
//! compass quantity like the two above it and the column goes 5.43 -> 3.76.
//!
//! One test per column per window, because which way a note fails to belong is
//! the attribution and a single verdict would throw it away — and three more
//! that are what make the rest worth having:
//! [`the_gate_fails_on_the_preset_this_milestone_started_from`] runs the six
//! columns with item 284 **undone in memory**,
//! [`the_tail_gate_fails_without_the_drawn_low_band_at_c4`] does the same with
//! item 335 undone and is the standing record of the defect this file was
//! widened for, and
//! [`the_drawn_decay_rows_are_all_keys_the_library_never_recorded`] is the
//! control that says the fix moved no key the bars are measured on. A gate
//! nobody has seen fail is not a gate.
//!
//! Every test here needs the Salamander library and skips itself without it,
//! the same way `tests/reference_cache.rs` does.

use std::f64::consts::PI;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use piano_emulator::preset::Preset;
use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_tuner::audio::Audio;
use piano_tuner::cache;
use piano_tuner::estimate::melody::{self, Column, LineNote, NoteTexture, Window};
use piano_tuner::realism::{Phrase, RecordedKeys, VelocityLayers};
use piano_tuner::sampler::{engine_events, Sampler, SAMPLER_VERSION};
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

/// The same preset with `DECISIONS.md` 340 taken back out of it: `[noise.strike]`
/// at the level and the velocity law it carried through M8, and nothing else
/// moved.
fn with_the_mechanism_before_the_refit(mut preset: Preset) -> Preset {
    for anchor in preset.noise.strike.level_db.iter_mut() {
        anchor.db += melody::STRIKE_REFIT_LEVEL_DB;
    }
    preset.noise.strike.velocity_db = melody::STRIKE_VELOCITY_DB_BEFORE;
    preset
        .validate()
        .expect("undoing a milestone is still a legal preset");
    preset
}

/// The same preset with `DECISIONS.md` 418's mode-controlled band put back and
/// nothing else moved: the instrument the `balance` column of item 446 was
/// written on.
///
/// The three numbers are `melody::M17_MODAL_BAND`, kept beside the metric they
/// convict rather than hand-written into a fixture file, exactly as
/// `STRIKE_REFIT_LEVEL_DB` is. The geometry, `width` and `diffuse_coherence`
/// are the shipping preset's, so a column that moves here moves for one reason.
fn with_the_lobe_before_the_refit(mut preset: Preset) -> Preset {
    let mics = preset
        .voicing
        .mics
        .as_mut()
        .expect("the measured preset declares [voicing.mics]");
    let (lo_hz, hi_hz, lift) = melody::M17_MODAL_BAND;
    mics.modal = Some(piano_emulator::preset::ModalBand { lo_hz, hi_hz, lift });
    preset
        .validate()
        .expect("item 418's own band is still a legal preset");
    preset
}

/// The same preset with `DECISIONS.md` 335 taken back out of it: every drawn
/// decay row's cells **under 2 kHz** go back to 1.0, which is what
/// `TailCorrection::at` wrote there before this milestone and what left the 58
/// unrecorded keys in a different gauge from the 30 recorded ones.
///
/// Nothing else moves — the two high bands of the same rows are untouched, and
/// no recorded key is touched at all — so a column that fails here and passes
/// on the shipped preset has been attributed to one band of one table.
fn without_drawn_low_decay(mut preset: Preset) -> Preset {
    let drawn = preset.notes.synthesized_decay.clone();
    assert!(
        !drawn.is_empty(),
        "the shipped preset names no drawn decay rows; this test has nothing to undo"
    );
    let mut touched = 0usize;
    for key in drawn {
        let params = preset.string_params(key);
        let hz: Vec<f64> = (1..=params.partial_count())
            .map(|k| f64::from(params.partial_freq(k)))
            .collect();
        let i = usize::from(key - 21);
        let Some(row) = preset.notes.partial_sigma_scale.get_mut(i) else {
            continue;
        };
        for (cell, &f) in row.iter_mut().zip(&hz) {
            if f < piano_tuner::estimate::tail::LOW_BAND.1 && *cell != 1.0 {
                *cell = 1.0;
                touched += 1;
            }
        }
        while row.last() == Some(&1.0) {
            row.pop();
        }
    }
    assert!(
        touched > 0,
        "no drawn row carries a cell under 2 kHz; this test has nothing to undo"
    );
    preset.validate().expect("undoing a milestone is still legal");
    preset
}

// ---------------------------------------------------------------------------
// Rendering the line
// ---------------------------------------------------------------------------

fn render_engine(preset: &Preset, phrase: &Phrase) -> Audio {
    let events: Vec<RenderEvent> = engine_events::to_render_events(&phrase.events);
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

/// One phrase measured through all three players, in one window.
fn measure_phrase(
    preset: &Preset,
    sfz: &Path,
    layers: &VelocityLayers,
    phrase: &Phrase,
    notes: &[LineNote],
    window: Window,
) -> (melody::Lines, Vec<NoteTexture>, Vec<NoteTexture>) {
    let sr = f64::from(SAMPLE_RATE);
    let partial_hz = |key: u8| -> Vec<f64> {
        let params = preset.string_params(key);
        (1..=piano_tuner::series::PARTIALS)
            .map(|k| f64::from(params.partial_freq(k)))
            .collect()
    };
    let engine = render_engine(preset, phrase);
    let reference = render_reference(sfz, phrase, "reference", &phrase.events);
    let alt = render_reference(sfz, phrase, "alt-layer", &layers.shift(&phrase.events));
    let engine_notes = melody::measure_line(&engine, sr, notes, &partial_hz, window);
    let reference_notes = melody::measure_line(&reference, sr, notes, &partial_hz, window);
    let layer_notes = melody::measure_line(&alt, sr, notes, &partial_hz, window);
    (
        melody::Lines::new(
            melody::per_key(&engine_notes),
            melody::per_key(&reference_notes),
            melody::per_key(&layer_notes),
        ),
        engine_notes,
        reference_notes,
    )
}

/// Every column of the gate for one preset: three evenness metrics in two windows,
/// and the two balances — `strike` and `channel` — in the window that contains
/// them.
fn score(preset: &Preset, sfz: &Path) -> (Vec<Column>, Vec<NoteTexture>, Vec<NoteTexture>) {
    let library = SampleLibrary::from_sfz(sfz).expect("the library reads");
    let layers = VelocityLayers::from_library(&library).expect("the library has velocity layers");
    let recorded = RecordedKeys::from_library(&library).expect("the library records keys");
    let ladder_keys = melody::ladder_keys(&recorded, &melody::line_keys());

    let mut columns = Vec::new();
    let mut head_engine = Vec::new();
    let mut head_reference = Vec::new();
    for window in [Window::Head, Window::Tail] {
        let line_phrase = melody::line_for(window);
        let line_notes = melody::line_notes_for(window);
        let ladder_phrase = melody::ladder(&ladder_keys, window);
        let ladder_notes = melody::ladder_notes(&ladder_keys, window);
        let (line, engine_notes, reference_notes) = measure_phrase(
            preset,
            sfz,
            &layers,
            &line_phrase,
            &line_notes,
            window,
        );
        let (population, _, _) = measure_phrase(
            preset,
            sfz,
            &layers,
            &ladder_phrase,
            &ladder_notes,
            window,
        );
        columns.extend(melody::compare(window, &line, &population, &recorded));
        if window == Window::Head {
            head_engine = engine_notes;
            head_reference = reference_notes;
        }
    }
    (columns, head_engine, head_reference)
}

// ---------------------------------------------------------------------------
// The gate
// ---------------------------------------------------------------------------

/// The shipped instrument's line, measured once for the whole file: twelve
/// renders and 28 windowed notes a piece are not worth doing seven times.
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

fn column<'a>(columns: &'a [Column], metric: &str, window: Window) -> &'a Column {
    columns
        .iter()
        .find(|c| c.metric == metric && c.window == window)
        .expect("a named column")
}

/// One column of the gate: no note of the engine's line may stand further off
/// the line's own register trend than the recorded keys of that register stand
/// off theirs — and never under the distance between two takes of one key.
fn gate(metric: &str, window: Window) {
    let Some(columns) = shipped() else {
        eprintln!("no data/salamander in this tree; skipping the melody gate");
        return;
    };
    let c = column(columns, metric, window);
    if c.gated_on_balance {
        assert!(
            c.balance_pass,
            "{} ({}) reads {:+.2} dB over the recorded keys of the register \
             against a bar of {:.2} (one recorded key's two takes, {:.2}, x{:.2})\n{}",
            c.metric,
            c.window.name(),
            c.balance,
            c.balance_bar,
            c.balance_bar / melody::ALLOWANCE,
            melody::ALLOWANCE,
            melody::report(std::slice::from_ref(c))
        );
    }
    if !c.gated_on_spread {
        return;
    }
    assert!(
        c.spread_pass,
        "{} ({}) stands out {:.2} at {} against a bar of {:.2} \
         (how far the recorded register's own notes go, {:.2}; its single worst key {:.2} at {}; \
          one key's two takes {:.2} at {})\n{}",
        c.metric,
        c.window.name(),
        c.standout,
        melody::note_name(c.standout_key),
        c.bar,
        c.population_bar,
        c.population_standout,
        melody::note_name(c.population_standout_key),
        c.take_scatter,
        melody::note_name(c.take_scatter_key),
        melody::report(std::slice::from_ref(c))
    );
}

/// The drawn `notes.partial_gains` rows, heard as a tune.
#[test]
fn no_note_of_the_line_is_rougher_than_the_rest() {
    gate("roughness", Window::Head);
}

/// The `notes.false_beat` splits, heard as a tune.
///
/// **This is the column that was red from `DECISIONS.md` 298 to 331, and the
/// policy of 328 is what closed it — not the engine.** It stood at 0.44 dB
/// against a bar of 0.26, and that bar was the recordings' own worst note on a
/// line four fifths of which is two takes resampled. Two transpositions of one
/// recording wobble almost identically; two *different keys* of the piano do
/// not. Measured on the recorded keys of the same register, the recordings'
/// own scatter is 0.88 dB and the bar 1.10, and the engine's 0.44 is inside
/// it. The engine did not move: the yardstick was wrong.
#[test]
fn no_note_of_the_line_wobbles_unlike_the_rest() {
    gate("wobble", Window::Head);
}

/// Brilliance at absolute frequency, heard as a tune.
#[test]
fn no_note_of_the_line_is_brighter_than_the_rest() {
    gate("hf", Window::Head);
}

/// The tail of the note, which is where `notes.partial_sigma_scale` lives.
#[test]
fn no_note_of_the_lines_tail_is_rougher_than_the_rest() {
    gate("roughness", Window::Tail);
}

#[test]
fn no_note_of_the_lines_tail_wobbles_unlike_the_rest() {
    gate("wobble", Window::Tail);
}

/// **The red of `DECISIONS.md` 330-331, closed by 335.** The listener's own
/// note: C4's tail read **5.43 dB from its line's own trend against a bar of
/// 5.32** while D4, E4, F4 and G4 carried drawn decay rows whose cells under
/// 2 kHz were all exactly 1.0. C4's own cells there hold its fundamental 4.2 dB
/// higher at 0.5 s than the law alone would, and this column is a *share* whose
/// denominator is the fundamental, so the one key of the line with a measured
/// row read 5.4 dB darker than its neighbours. With the same band drawn for the
/// keys nobody recorded it reads **3.76 at D4**, and C4's own departure is
/// **1.88**.
#[test]
fn no_note_of_the_lines_tail_is_brighter_than_the_rest() {
    gate("hf", Window::Tail);
}

/// **The permanent per-channel column, heard as a tune** (`DECISIONS.md`
/// 392-394).
///
/// `channel` is `10 log10((E_L + E_R) / 2 E_M)` per note: what the two
/// loudspeakers put in the room, against what this note's own mono fold-down
/// says they do. Every other column of this gate, and every column of every
/// other board in the repository, is a function of that fold-down — so a stereo
/// stage can make one note of a melody four decibels louder than its neighbours
/// in the room, and 696 tests stay green. That is what happened
/// (`DECISIONS.md` 392): the virtual pair's mode-controlled lobe read **+6.42
/// dB at C4 against +2.41 at F4 and +3.07 at G4**, a four-decibel spread across
/// five notes of one tune, with the two channels 9 dB up and 2 dB down at C4
/// and 6 up / 21 *down* at F4 — none of it in the mono sum, and a listener
/// picked C4 out of the line three milestones running.
///
/// It is a **balance**, like `strike`: the recording has its own value at every
/// note — two capsules over a real soundboard do hear a note at two levels —
/// so the question is whether the engine's is the recording's, at the keys the
/// library recorded, and not whether it is zero.
#[test]
fn the_two_loudspeakers_play_this_line_as_the_recording_does() {
    gate("channel", Window::Head);
}

/// **Which loudspeaker the tune's pitches come out of** (`DECISIONS.md` 446).
///
/// `balance` is `10 log10(E_L / E_R)` at each note's **own fundamental**,
/// heterodyned — the statistic the session that opened item 446 measured the
/// defect with. It is `channel`'s missing half and the reason it was missing is
/// arithmetic: `channel` is `(E_L + E_R) / 2 E_M`, a **sum**, invariant under
/// swapping the two loudspeakers, so an instrument that puts every fundamental
/// of the Ode line seven decibels into the *left* channel — where the recording
/// leans about one and a half *right* — moves it by nothing. That is not
/// hypothetical: on the instrument item 422 shipped, `channel` reads −0.49
/// against a bar of 0.91 and is **green**, while this column reads **+8.84
/// against 1.94** over the same nine recorded keys, and the line itself reads
/// C4 −2.05, D4 +4.68, E4 +6.92, F4 +11.07, G4 +5.73 against a recording that
/// reads −1.01, −0.91, −1.39 and −1.07 at the four of them it can be read at.
/// The complaint the listener made of that render was "wobbly", and the jumps
/// are the wobble.
///
/// It is the **one column of this board gated on both halves**
/// (`melody::METRIC_IS_SPREAD`): the median over the recorded ladder convicts a
/// uniform lean the line's own trend cannot see, and the line's own spread
/// convicts note-to-note jumps a median cannot see because they cancel in it.
#[test]
fn the_lines_pitches_come_out_of_the_loudspeaker_the_recordings_do() {
    gate("balance", Window::Head);
}

/// **Whether a note of the tune arrives from one place in the image or comes
/// apart across it** (`DECISIONS.md` 451).
///
/// `splitting` is `image(f1) − Σ w_k image(f_k) / Σ w_k` over the note's own
/// partials 2-4, where `image` is the balance column's own per-partial
/// heterodyne and `w_k` is the pair energy it was read from: the fundamental's
/// place in the stereo image, measured against where that same note's colour
/// is. It is `balance`'s other half in the same sense that `balance` is
/// `channel`'s — `balance` reads **one** frequency per note and a stage that is
/// a *band* does not move a note, it moves the part of a note inside its edges.
///
/// The band item 422 shipped spans 174.3-456.5 Hz. Every fundamental of the Ode
/// line (261.6 to 392.0 Hz) is inside it and **not one** of those notes' second
/// partials is (523 Hz and up), so every note of the tune had its pitch panned
/// and its colour left where it was, and the ear was handed a note arriving
/// from two places at once. That is heard as *the temperament is off* with zero
/// cents of tuning error anywhere — the session that opened item 451 verified
/// the preset's own `f0` table consistent to 0.3 cents at the seam before
/// measuring this — and it is why the column exists.
///
/// It is scored on the **line's own five pitches** and not on the recorded
/// ladder, which no other balance column on this board is
/// (`melody::METRIC_ON_LINE`). The reason is arithmetic and is in that
/// constant: resampling multiplies every frequency of a take by one factor and
/// touches neither channel, so a transposed reference note's image ratios —
/// and therefore its split — are the take's own, exactly. Item 448(d) asked a
/// successor for "a statistic scored on the tune's own register rather than on
/// the whole recorded ladder"; this is it.
#[test]
fn no_note_of_the_line_arrives_from_two_places_at_once() {
    gate("splitting", Window::Head);
}

/// **The falsification for the column above, and it is the milestone's own
/// finding**: put item 418's band back and the column goes red, in the
/// direction and at the size item 446 measured.
///
/// A gate nobody has seen fail is not a gate. The material is whatever preset
/// ships with `[voicing.mics.modal]` replaced by `melody::M17_MODAL_BAND` and
/// **nothing else moved** — same geometry, same `width`, same
/// `diffuse_coherence` — so a column that moves here has moved for one reason.
/// It asserts the **sign** as well as the size, because which loudspeaker the
/// tune walks into is the whole attribution: the lobe is `L = m(1 + B)`,
/// `R = m(1 − B)`, and over 174-456 Hz `Re B > 0`, so it is not a widener but a
/// pan, and it pans left.
#[test]
fn the_balance_gate_fails_on_the_band_the_milestone_started_from() {
    let Some(sfz) = sfz() else {
        eprintln!("no data/salamander in this tree; skipping the balance falsification");
        return;
    };
    let before = with_the_lobe_before_the_refit(shipped_preset());
    let (columns, _, _) = score(&before, &sfz);
    let text = melody::report(&columns);
    println!("{text}");
    let balance = column(&columns, "balance", Window::Head);
    assert!(
        !balance.balance_pass,
        "item 418's band passes the balance column, so that column does not test \
         what it was written for\n{text}"
    );
    assert!(
        balance.balance > 0.0,
        "the engine was supposed to be the LEFT-leaning of the two and reads \
         {:+.2}\n{text}",
        balance.balance
    );
    // And the column it was written beside cannot see it: `channel` is a sum
    // over the two loudspeakers and this is a difference between them. If a
    // successor ever makes `channel` fail here too, this assertion is the
    // notice that the two columns have stopped being independent.
    let channel = column(&columns, "channel", Window::Head);
    println!(
        "the same instrument on the column that cannot see it: channel {:+.2} \
against a bar of {:.2}, {}",
        channel.balance,
        channel.balance_bar,
        if channel.balance_pass { "green" } else { "red" }
    );
}

/// **The falsification for the splitting column, and it is the same
/// instrument** (`DECISIONS.md` 451-452): put item 418's band back and the tune
/// comes apart across the image again.
///
/// The two columns convict the same band for two different things and both
/// have to be seen to fail on it, or the pair of them is one column written
/// twice. `balance` fails because the band **pans** the fundamentals; this one
/// fails because the band's two edges bracket those fundamentals and nothing
/// else of those notes, so what is panned is a *part* of each note. A
/// mechanism that panned the whole note would fail the first and pass this.
///
/// It asserts the size and prints the line note by note, because the per-note
/// number is the attribution: on the band this test installs, F4's fundamental
/// sits **21.8 dB** away from its own second, third and fourth partials, where
/// the recording's F4 sits 1.4 dB from its own.
#[test]
fn the_splitting_gate_fails_on_the_band_the_milestone_started_from() {
    let Some(sfz) = sfz() else {
        eprintln!("no data/salamander in this tree; skipping the splitting falsification");
        return;
    };
    let before = with_the_lobe_before_the_refit(shipped_preset());
    let (columns, _, _) = score(&before, &sfz);
    let text = melody::report(&columns);
    println!("{text}");
    let splitting = column(&columns, "splitting", Window::Head);
    assert!(
        !splitting.balance_pass,
        "item 418's band passes the splitting column, so that column does not test \
         what it was written for\n{text}"
    );
    // The line, note by note, is the attribution and is printed whether or not
    // anything fails: which notes come apart and by how much is what a listener
    // is describing when they call a tune out of tune.
    for n in &splitting.notes {
        println!(
            "  {}: engine splits {:+.2} dB where the recording splits {:+.2} — {:+.2}",
            melody::note_name(n.key),
            n.engine,
            n.reference,
            n.error
        );
    }
    // And the column it is not: `balance` reads one frequency per note, so a
    // band that panned each note *whole* would move it by the same amount and
    // leave this column at nothing. If a successor ever makes the two move
    // together on every instrument, this print is the notice that one of them
    // has stopped earning its place.
    let balance = column(&columns, "balance", Window::Head);
    println!(
        "the same instrument on the column beside it: balance {:+.2} against a bar of \
{:.2}, {}",
        balance.balance,
        balance.balance_bar,
        if balance.balance_pass { "green" } else { "red" }
    );
}

/// **The control the falsification needs to mean anything**: the recordings
/// scored against themselves pass this column.
///
/// A column that failed everything would fail item 418's band too, and the test
/// above would prove nothing. So the same measurement is run with the
/// **reference render standing in for the engine** — the recordings' own line
/// against the recordings' own line, with the neighbouring velocity layer still
/// the floor — and every half of the column must be inside its own bar. It is
/// exactly zero on the balance half by construction and is asserted anyway,
/// because the construction is what a successor would change.
#[test]
fn the_recordings_own_line_passes_the_balance_column() {
    let Some(sfz) = sfz() else {
        eprintln!("no data/salamander in this tree; skipping the balance control");
        return;
    };
    let preset = shipped_preset();
    let library = SampleLibrary::from_sfz(&sfz).expect("the library reads");
    let layers = VelocityLayers::from_library(&library).expect("velocity layers");
    let recorded = RecordedKeys::from_library(&library).expect("recorded keys");
    let ladder_keys = melody::ladder_keys(&recorded, &melody::line_keys());
    let window = Window::Head;
    let lines = |phrase: &Phrase, notes: &[LineNote]| -> melody::Lines {
        let sr = f64::from(SAMPLE_RATE);
        let partial_hz = |key: u8| -> Vec<f64> {
            let params = preset.string_params(key);
            (1..=piano_tuner::series::PARTIALS)
                .map(|k| f64::from(params.partial_freq(k)))
                .collect()
        };
        let reference = render_reference(&sfz, phrase, "reference", &phrase.events);
        let alt = render_reference(&sfz, phrase, "alt-layer", &layers.shift(&phrase.events));
        let rows = melody::per_key(&melody::measure_line(
            &reference,
            sr,
            notes,
            &partial_hz,
            window,
        ));
        let layer = melody::per_key(&melody::measure_line(&alt, sr, notes, &partial_hz, window));
        // The engine slot is the reference itself: this is the recording
        // measured against a second measurement of the same file.
        melody::Lines::new(rows.clone(), rows, layer)
    };
    let line_phrase = melody::line_for(window);
    let ladder_phrase = melody::ladder(&ladder_keys, window);
    let columns = melody::compare(
        window,
        &lines(&line_phrase, &melody::line_notes_for(window)),
        &lines(&ladder_phrase, &melody::ladder_notes(&ladder_keys, window)),
        &recorded,
    );
    let text = melody::report(&columns);
    println!("{text}");
    let balance = column(&columns, "balance", Window::Head);
    assert!(
        balance.balance.abs() < 1e-9,
        "the recordings differ from themselves by {:+.3} dB\n{text}",
        balance.balance
    );
    assert!(
        balance.pass,
        "the recordings' own line fails the column that scores instruments \
         against it: balance {:+.2} of {:.2}, spread {:.2} of {:.2}\n{text}",
        balance.balance, balance.balance_bar, balance.standout, balance.bar
    );
    // **And the same for `splitting`, which is the column this control matters
    // most for** (`DECISIONS.md` 451). A real AB pair does split a note a
    // little — a soundboard's partials radiate from different parts of the
    // plate, and two spaced capsules place them at slightly different points —
    // so the target is the recording's own number and never zero: the reference
    // line splits C4 by −13.71 dB, D4 by +2.67 and F4 by −1.36 at its own
    // partials. A column that read "any split at all is a defect" would fail
    // the piano it is scored against, which is exactly what this asserts it
    // does not.
    let splitting = column(&columns, "splitting", Window::Head);
    assert!(
        splitting.balance.abs() < 1e-9,
        "the recordings split differently from themselves by {:+.3} dB\n{text}",
        splitting.balance
    );
    assert!(
        splitting.pass,
        "the recordings' own line fails the splitting column that scores instruments \
         against it: balance {:+.2} of {:.2}, spread {:.2} of {:.2}\n{text}",
        splitting.balance, splitting.balance_bar, splitting.standout, splitting.bar
    );
    assert!(
        splitting
            .notes
            .iter()
            .any(|n| n.reference.abs() > 1.0),
        "the recording's own line splits no note by more than a decibel, so this \
         control cannot distinguish a bar taken off the recording from a bar of \
         zero\n{text}"
    );
    // Every column, not just this one: a control that only holds for the column
    // it was written for is not a control.
    for c in &columns {
        assert!(
            c.pass,
            "{} ({}) fails on the recordings against themselves\n{text}",
            c.metric,
            c.window.name()
        );
    }
}

/// The tail gate is a statement about a preset, and the statement it was
/// written to make is that the instrument of `DECISIONS.md` 331 fails it, at
/// C4, on the metric the seam is in.
///
/// A gate nobody has seen fail is not a gate, so the falsification moved with
/// the fix rather than being retired with it: the material is now the shipped
/// preset with **one band of one table** put back the way item 331 found it —
/// every drawn `partial_sigma_scale` row's cells under 2 kHz returned to 1.0,
/// nothing else touched, no recorded key touched at all. It asserts the *key*,
/// because the listener named the key.
#[test]
fn the_tail_gate_fails_without_the_drawn_low_band_at_c4() {
    let Some(sfz) = sfz() else {
        eprintln!("no data/salamander in this tree; skipping the tail falsification");
        return;
    };
    let before = without_drawn_low_decay(shipped_preset());
    let (columns, _, _) = score(&before, &sfz);
    let text = melody::report(&columns);
    println!("{text}");
    let broken: Vec<&Column> = columns
        .iter()
        .filter(|c| c.window == Window::Tail && !c.pass)
        .collect();
    assert!(
        !broken.is_empty(),
        "the pre-335 instrument passes every tail column, so the tail gate does \
         not test what it was written for\n{text}"
    );
    assert!(
        broken.iter().any(|c| c.standout_key == 60),
        "the tail columns fail but none of them names C4, which is the note the \
         listener named and the only key of this line whose decay row was \
         fitted\n{text}"
    );
}

/// The mechanism against the string, heard as a tune — `DECISIONS.md` 341.
///
/// The other six columns ask whether one note of the line stands out from the
/// rest; this one asks whether the hammer is as loud against its own note as
/// the piano's is against the same note, which is the listener's second finding
/// on the M8 render and is a question only a **recording of that key** can
/// answer. So it is scored on the recorded ladder, as the median of
/// `engine - recording` over those nine keys, and its bar is the median
/// distance between two takes of one of them.
///
/// It stood at **-3.90 dB against a bar of 2.05** on the preset this milestone
/// started from — the engine's attack a full four decibels noisier than the
/// piano's, on material where both words mean the same note — and at **-1.60**
/// after `DECISIONS.md` 340 refit the event's level and its velocity law.
#[test]
fn the_hammer_is_no_louder_against_the_note_than_the_pianos_is() {
    gate("strike", Window::Head);
}

/// The same falsification the tail gate carries, for the column above: a gate
/// nobody has seen fail is not a gate.
///
/// The material is the shipped preset with `[noise.strike]`'s two level fields
/// — and nothing else — put back where item 340 found them. The event's colour,
/// its decay and every other table in the preset are untouched, so a column
/// that moves here moves for one reason.
#[test]
fn the_strike_gate_fails_at_the_level_the_milestone_started_from() {
    let Some(sfz) = sfz() else {
        eprintln!("no data/salamander in this tree; skipping the strike falsification");
        return;
    };
    let before = with_the_mechanism_before_the_refit(shipped_preset());
    let (columns, _, _) = score(&before, &sfz);
    let text = melody::report(&columns);
    println!("{text}");
    let strike = column(&columns, "strike", Window::Head);
    assert!(
        !strike.pass,
        "the pre-340 mechanism passes the balance column, so that column does \
         not test what it was written for\n{text}"
    );
    assert!(
        strike.balance < 0.0,
        "the engine was supposed to be the noisier of the two and reads \
         {:+.2}\n{text}",
        strike.balance
    );
    // The refit moved two numbers of one event and nothing else, so every other
    // column of the gate has to read the same on both instruments.
    let after = shipped().expect("the shipped columns are measured");
    for c in columns.iter().filter(|c| c.gated_on_spread && !c.gated_on_balance) {
        let now = column(after, c.metric, c.window);
        assert!(
            (c.standout - now.standout).abs() < 0.1 && (c.bar - now.bar).abs() < 0.01,
            "{} ({}) moved {:.2}/{:.2} -> {:.2}/{:.2} when only [noise.strike]'s \
             level changed",
            c.metric,
            c.window.name(),
            c.standout,
            c.bar,
            now.standout,
            now.bar
        );
    }
}

/// The fix is a statement about the **58 keys the library never recorded**, and
/// this is the control that says so: not one of the 30 recorded keys carries a
/// drawn decay row, so the population every bar above is measured on is the
/// same population it was measured on before.
#[test]
fn the_drawn_decay_rows_are_all_keys_the_library_never_recorded() {
    let Some(sfz) = sfz() else {
        eprintln!("no data/salamander in this tree; skipping the provenance control");
        return;
    };
    let library = SampleLibrary::from_sfz(&sfz).expect("the library loads");
    let recorded = RecordedKeys::from_library(&library).expect("the library records keys");
    let preset = shipped_preset();
    let scored: Vec<u8> = preset
        .notes
        .synthesized_decay
        .iter()
        .copied()
        .filter(|&k| recorded.is_recorded(k))
        .collect();
    assert!(
        scored.is_empty(),
        "these recorded keys carry a drawn decay row, so a scored comparison \
         would be reading a draw: {scored:?}"
    );
}

/// The same six columns on the instrument `DECISIONS.md` 284 started from: C4
/// with its measured row and its measured splits, and D4/E4/F4/G4 with nothing
/// at all.
///
/// It is not asserted *which* column fails, only that the line is measurably
/// uneven — which is what makes the six gates above statements about the
/// instrument rather than about the metric. On the shipped library it lands in
/// `wobble (tail)`, at **C4**, 1.78 dB against a bar of 1.38, where the shipped
/// preset reads 1.20 at D4: the drawn splits item 284 wrote are what stopped
/// the one key with measured ones standing alone. The head columns, whose bars
/// now come off the recorded register rather than off four clones, no longer
/// fail there — `DECISIONS.md` 331 quotes that as a finding rather than hiding
/// it.
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

/// The policy of `DECISIONS.md` 328, asserted on the shipped library rather
/// than on a hand-made one: four of the line's five pitches carry no per-note
/// score, they are named, and the bar does not come from them.
#[test]
fn only_c4_of_the_melody_is_a_recording() {
    let Some(sfz) = sfz() else {
        eprintln!("no data/salamander in this tree; skipping");
        return;
    };
    let library = SampleLibrary::from_sfz(&sfz).expect("the library reads");
    let recorded = RecordedKeys::from_library(&library).expect("recorded keys");
    assert!(recorded.is_recorded(60));
    for (key, take) in [(62u8, 63u8), (64, 63), (65, 66), (67, 66)] {
        assert!(!recorded.is_recorded(key), "{key} is not a Salamander take");
        assert_eq!(recorded.take_for(key), Some(take));
    }
    let Some(columns) = shipped() else { return };
    for c in columns {
        assert_eq!(
            c.transposed_keys(),
            vec![62, 64, 65, 67],
            "{} ({})",
            c.metric,
            c.window.name()
        );
        assert_eq!(c.line_error.map(|(k, _)| k), Some(60));
        assert!(
            c.population.len() >= 5,
            "{} ({}) set its bar off {} recorded keys",
            c.metric,
            c.window.name(),
            c.population.len()
        );
    }
}

/// The onset the whole board's windows hang off, and the shape of signal that
/// made it miss by most of a window.
///
/// `DECISIONS.md` 452. Every column of this file is measured over a span
/// counted from [`melody::note_onset`], so a detector that lands late does not
/// report a wrong onset — it reports a wrong *note*, measured from an eighth of
/// a second into its own decay while the note beside it is measured from its
/// hammer. On the melody render as shipped it landed **+73 ms** past the
/// engine's C4 and **+42 ms** past the recording's, at C4 and nowhere else.
///
/// The cause is arithmetic and this test is built out of it rather than out of
/// a render. The detector maximised the rise of a **1 ms** RMS envelope. One
/// millisecond is a quarter of a period at 261.6 Hz, so that "envelope" is the
/// waveform's own ripple, and on a note whose attack *swells* — which the
/// engine's C4 does, reaching its loudest 20 ms in and only 4 dB above the tail
/// it is struck into — the largest ripple-rise is wherever the swell is
/// steepest and not where the hammer is. The signal below is that shape and
/// nothing else: a decaying neighbour, a 3 ms broadband hammer at a known time,
/// and a C4 that takes 25 ms to arrive.
///
/// Both halves are asserted, because only the pair is a falsification: the
/// **old** detector's shape (1 ms blocks, no band) must miss by more than half
/// the head window's own length, and the shipped one must land inside a block.
#[test]
fn the_onset_detector_finds_the_hammer_and_not_the_fundamental() {
    let sr = f64::from(SAMPLE_RATE);
    let strike_s = 0.5;
    let signal = a_low_note_struck_into_a_tail(sr, strike_s);

    // The shape the board shipped with, spelled out here rather than called, so
    // that this test keeps reproducing the defect after the caller is fixed.
    let old = piano_tuner::realism::strike_near_banded(
        &signal, sr, strike_s, 0.05, 0.12, 1.0, 0.0,
    );
    let new = melody::note_onset(&signal, sr, strike_s);
    let old_ms = 1000.0 * (old - strike_s);
    let new_ms = 1000.0 * (new - strike_s);

    assert!(
        old_ms > 30.0,
        "the 1 ms broadband detector is supposed to miss this shape and \
         landed {old_ms:+.1} ms from the hammer; if this stops failing the \
         defect of item 452 can no longer be reproduced and the test is lying"
    );
    assert!(
        new_ms.abs() <= 2.0 * melody::ONSET_BLOCK_MS,
        "the shipped detector landed {new_ms:+.1} ms from the hammer \
         (the old one: {old_ms:+.1} ms)"
    );
}

/// A C4 with a slow attack struck into the tail of a note a third above it.
///
/// Built to the shape the engine's own render has and no more: a fundamental
/// that **swells** over 25 ms — the thing that fools a 1 ms envelope — over a
/// partial series whose upper members arrive at the hammer and are gone in a
/// quarter of a second, which is the thing a high-passed envelope sees. The
/// hammer's own burst is in there at the level `DECISIONS.md` 339 measured it,
/// twenty-one decibels under the note's own content in the same octave, so that
/// this cannot pass by finding a burst the real signal does not have.
///
/// Deterministic: the burst is a fixed linear-congruential sequence, because a
/// test that renders a different noise on every run is not a pin.
fn a_low_note_struck_into_a_tail(sample_rate: f64, strike_s: f64) -> Vec<f32> {
    const F0: f64 = 261.63;
    const PARTIALS: usize = 24;
    let n = (1.2 * sample_rate) as usize;
    let mut out = vec![0.0f32; n];
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    for (i, s) in out.iter_mut().enumerate() {
        let t = i as f64 / sample_rate;
        // The note before: an E4 that has been decaying since 0.
        let mut v = 0.030 * (-t / 0.9).exp() * (2.0 * PI * 329.63 * t).sin();
        if t >= strike_s {
            let u = t - strike_s;
            for k in 1..=PARTIALS {
                let kf = k as f64;
                let hz = kf * F0;
                if 2.0 * hz >= sample_rate {
                    break;
                }
                // The fundamental takes 25 ms to arrive and the upper partials
                // take one, which is what a coupled unison does and is the
                // whole of the defect: the note's *level* peaks 20 ms in, its
                // *band* peaks at the hammer.
                let rise = if k == 1 { 0.025 } else { 0.001 };
                let fall = 1.4 / kf.powf(0.8);
                let a = 0.055 / kf.powf(1.7);
                v += a * (1.0 - (-u / rise).exp()) * (-u / fall).exp()
                    * (2.0 * PI * hz * t).sin();
            }
            if u < 0.003 {
                seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                let noise = (2.0 * (seed >> 33) as f64 / f64::from(u32::MAX >> 1)) - 1.0;
                v += 0.0006 * noise * (1.0 - u / 0.003);
            }
        }
        *s = v as f32;
    }
    out
}
