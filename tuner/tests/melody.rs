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
//! **`DECISIONS.md` 459-460 — the two columns that read the *image* rather than
//! the note.** `comb` is where a note's own overtones sit between the
//! loudspeakers and `cue` is when the note arrives at each of them. They are
//! here because `splitting` is `balance − comb` exactly, so this board scored a
//! difference of two image positions and one of the two positions and was blind
//! to any mechanism that moved both together — which is what the pair
//! *geometry* does — and because every column above them is a function of two
//! magnitudes, where half of where a listener puts a note is a phase. Neither
//! takes a median or a residual: the defect is a ramp across the tune, and a
//! median cancels a ramp while a residual about the line's own trend subtracts
//! it.
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
use piano_tuner::estimate::shaping::ShapingConfig;
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

// ---------------------------------------------------------------------------
// Rendering the line
// ---------------------------------------------------------------------------

/// The composition `DECISIONS.md` 455 removed, put back: every key's sub-2 kHz
/// `partial_sigma_scale` cells divided by the **whole** row's geometric mean.
///
/// Built from arithmetic on whatever preset ships, so it is never a copy of a
/// file. See the falsification that uses it for why this is the composition.
fn with_the_low_band_seam(mut preset: Preset) -> Preset {
    for key in 21..=108u8 {
        let i = usize::from(key - 21);
        let row = preset.notes.partial_sigma_scale[i].clone();
        if row.is_empty() {
            continue;
        }
        let params = preset.string_params(key);
        let hz: Vec<f64> = (1..=params.partial_count())
            .map(|k| f64::from(params.partial_freq(k)))
            .collect();
        let logs: Vec<f64> = row
            .iter()
            .map(|&g| f64::from(g))
            .filter(|g| *g > 0.0)
            .map(f64::ln)
            .collect();
        if logs.is_empty() {
            continue;
        }
        let geomean = (logs.iter().sum::<f64>() / logs.len() as f64).exp();
        preset.notes.partial_sigma_scale[i] =
            piano_tuner::estimate::tail::low_row(&row, &hz, geomean.recip());
    }
    preset
}

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
            "{} ({}) reads {:+.2} {} against a bar of {:.2} (one recorded key's two \
             takes, {:.2}, x{:.2}); against the recording's own image it reads \
             {:+.2}\n{}",
            c.metric,
            c.window.name(),
            c.balance,
            if c.neutral_target {
                "off a neutral image, median magnitude over the notes it scores"
            } else {
                "dB over the recorded keys of the register"
            },
            c.balance_bar,
            c.balance_bar / melody::ALLOWANCE,
            melody::ALLOWANCE,
            c.balance_vs_reference,
            melody::report(std::slice::from_ref(c))
        );
    }
    if c.gated_on_seam {
        assert!(
            c.seam_pass,
            "{} ({}) departs {:.2} dB from the register's median at {} against a bar of \
             {:.2} (the larger of one key's two takes, {:.2}, and the recorded register's \
             own spread, x{:.2}); the median itself is {:+.2} and is not gated\n{}",
            c.metric,
            c.window.name(),
            c.seam,
            melody::note_name(c.seam_key),
            c.seam_bar,
            c.seam_floor,
            melody::ALLOWANCE,
            c.balance,
            melody::report(std::slice::from_ref(c))
        );
    }
    // The four halves `DECISIONS.md` 459 adds. Each asserts only where the
    // column names it, and each prints the whole line, because on these two
    // columns *which* note is wrong is the attribution.
    if c.gated_on_slope {
        assert!(
            c.slope_pass,
            "{} ({}) tilts {:+.3} dB per semitone across the tune where a neutral image \
             is flat and the recording tilts {:+.3} — an error of {:.3} against a bar of \
             {:.3} (the recording's own slope plus how far it moves between two takes of \
             it, {:+.3}, x{:.2})\n{}",
            c.metric,
            c.window.name(),
            c.slope,
            c.reference_slope,
            c.slope_error,
            c.slope_bar,
            c.layer_slope,
            melody::ALLOWANCE,
            melody::report(std::slice::from_ref(c))
        );
    }
    if c.gated_on_swing {
        assert!(
            c.swing_pass,
            "{} ({}) swings {:.2} across the line's five pitches where the recording \
             swings {:.2}, against a bar of {:.2} (x{:.2})\n{}",
            c.metric,
            c.window.name(),
            c.swing,
            c.reference_swing,
            c.swing_bar,
            melody::ALLOWANCE,
            melody::report(std::slice::from_ref(c))
        );
    }
    if c.gated_on_bound {
        assert!(
            c.bound_pass,
            "{} ({}) puts {} {:.0} µs apart between the two loudspeakers, against the \
             head's own {:.0} — a bound every note carries, and no longer the larger of \
             it and the recording's own worst note ({:.0}, item 417's C4)\n{}",
            c.metric,
            c.window.name(),
            melody::note_name(c.bound_key),
            c.bound,
            c.bound_bar,
            c.reference_bound,
            melody::report(std::slice::from_ref(c))
        );
    }
    if c.gated_on_agreement {
        assert!(
            c.agreement_pass,
            "{} ({}): the engine's two localisation cues point opposite ways over the \
             line, r = {:+.2}, where the policy asks only that they point the same way \
             at all — the level cue swings {:.2} dB across the tune against a floor of \
             {:.2}, so it is a cue and not a rounding (the recording's two cues read \
             {:+.2}, its layer {:+.2})\n{}",
            c.metric,
            c.window.name(),
            c.engine_corr,
            c.ild_swing,
            c.ild_floor,
            c.reference_corr,
            c.layer_corr,
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
#[ignore = "D446/D466/D463 known gap, re-barred to the neutral target: the ladder's median fundamental sits 3.98 dB off centre against a bar of 1.73 (it read +8.61 against the recording's own image before D466); run with --ignored to read the current distance"]
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
#[ignore = "D451/D466/D463 known gap, re-barred to the neutral target: the line's median note splits its fundamental 7.06 dB from its own overtones against a bar of 3.26 (+7.39 against the recording before D466); run with --ignored to read the current distance"]
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

/// **Whether the tune's overtones stay in one place in the image as the melody
/// rises** (`DECISIONS.md` 459).
///
/// `comb` is the energy-weighted mean of `10 log10(E_L / E_R)` over each note's
/// own partials 2-4 — where the note's *colour* sits, as against `balance`,
/// which is where its pitch sits. The two columns beside it cannot see it and
/// the reason is one line of arithmetic: `splitting = balance − comb` exactly,
/// so a mechanism that moves a note's fundamental and its own overtones
/// **together**, and moves the next note's somewhere else, cancels out of
/// `splitting` completely and never enters `balance`, which reads one frequency
/// per note.
///
/// A spaced pair is exactly such a mechanism, by construction. Two capsules
/// `d` apart hear one source at an interchannel delay that depends on where
/// along the keyboard it is; `L = mid + side` and `R = mid − side` then comb
/// against frequency with a spacing set by that delay, so every partial of one
/// key moves together and the next key's partials sit somewhere else on the
/// comb. The result is a *ramp*, and a ramp is what the two verdicts here are
/// about: the **slope** of the engine's line against the recording's own, and
/// its **swing**. Neither is a median (a ramp cancels in one) and neither is a
/// residual about the line's own trend (a ramp is the trend).
///
/// It is scored at every note of the line, transposed or not, for item 451's
/// reason exactly: resampling multiplies every frequency of a take by one
/// factor and touches neither channel's amplitude, so a transposed note's
/// `E_L/E_R` at its `k`-th partial *is* the donor take's.
#[test]
#[ignore = "D459/D466/D463 known gap, re-barred to the neutral target: the pair geometry and the alternating polarization spread (D467) tilt the line's overtones -2.725 dB/semitone against a bar of 0.643 and swing them 18.52 against 5.15; D468's mechanism reaches both bars and is not installed; run with --ignored to read the current distance"]
fn the_tunes_overtones_stay_where_the_recordings_do() {
    gate("comb", Window::Head);
}

/// **Whether the two loudspeakers hand the ear a time difference a room could
/// have produced, and whether it agrees with the level** (`DECISIONS.md` 459).
///
/// `cue` is the interchannel time difference at each note's own fundamental,
/// read off the **phase** of the same heterodyne `balance` reads a level with.
/// Every other column of every board in this repository is a function of two
/// magnitudes; a listener localises from two cues, and this is the other one.
///
/// Two verdicts, and they are two different questions. The **bound** is
/// physics: a head is about 0.18 m across, so nothing in a room produces more
/// than about 660 µs of interchannel time, and a pair that hands the ear more
/// than that is not placing a source at the edge of the image. The bar is the
/// larger of that and the recording's own worst note, which on this library is
/// the larger of the two — the reference C4 whose capsule anomaly item 448(ii)
/// measured at 16.86 dB carries −949 µs of its own — so the engine is held to
/// the recording rather than to a listening theory.
///
/// The **agreement** is the defect that opened the milestone: `corr(ILD, ITD)`
/// over the line, the recording's minus the engine's. A real pair puts the
/// louder side and the earlier side on the same side of the image and the
/// recording reads **+0.83**; a filter that turns phase inside a band containing
/// every fundamental of the tune, over a pair whose geometry combs the same
/// notes, can make the two disagree — the level saying left while the time says
/// right — and the shipped instrument reads **−0.54**. Neither cue alone
/// reports that: `balance` is the level and this column's other half is the
/// time, and it is only their *product* that is wrong.
#[test]
#[ignore = "D460/D469/D463 known gap, re-barred: the line's two cues point opposite ways (corr -0.539, where the policy asks only for positive) and C4 carries 1102 us against the head's own 660, which is now a per-note bound with the recording's C4 no longer inflating it; run with --ignored to read the current distance"]
fn the_lines_two_localisation_cues_agree_as_the_recordings_do() {
    gate("cue", Window::Head);
}

/// **The falsification for `cue`, and it is the band this milestone inherited**:
/// put item 418's mode-controlled band back and the two cues disagree, because
/// what the band does inside its edges is turn a **phase**.
///
/// A gate nobody has seen fail is not a gate, and the pair of assertions is what
/// makes this one about the band rather than about the pair. `[voicing.mics.
/// modal]` is a twelfth-order cascade whose response `B` is complex, `L = m(1 +
/// B)` and `R = m(1 − B)`; over 174-456 Hz — which brackets every fundamental of
/// the Ode line and none of their overtones — that is not only the pan item 446
/// convicted but a per-note *time* offset, and it is not a time any pair of
/// capsules could have produced. So the band is asserted to fail this column
/// **and** to move the interchannel time of the line's own notes, which is the
/// attribution.
#[test]
fn the_cue_gate_fails_on_the_band_the_milestone_started_from() {
    let Some(sfz) = sfz() else {
        eprintln!("no data/salamander in this tree; skipping the cue falsification");
        return;
    };
    let before = with_the_lobe_before_the_refit(shipped_preset());
    let (columns, _, _) = score(&before, &sfz);
    let text = melody::report(&columns);
    println!("{text}");
    let cue = column(&columns, "cue", Window::Head);
    assert!(
        !cue.pass,
        "item 418's band passes the cue column, so that column does not test what it \
         was written for: |ITD| {:.0} µs of {:.0}, agreement {:+.2} of {:.2}\n{text}",
        cue.bound, cue.bound_bar, cue.agreement, cue.agreement_bar
    );
    // The line, note by note: which notes the band hands a time no pair could
    // have produced is what a listener is describing when they say the image
    // moves with the tune.
    for n in &cue.notes {
        println!(
            "  {}: engine {:+8.1} µs where the recording reads {:+8.1}",
            melody::note_name(n.key),
            n.engine,
            n.reference
        );
    }
    assert!(
        cue.notes.iter().any(|n| n.engine.abs() > melody::HEAD_ITD_US),
        "the band produces no note past the head's own 660 µs, so this test is not \
         about the mechanism it names\n{text}"
    );
}

/// **The second falsification for `comb`, and the diagnosis of
/// `DECISIONS.md` 467**: the two polarizations of one key are placed at two
/// *positions*, the sign of the displacement alternates with key parity, and
/// the image reads the position.
///
/// `voicing.polarization_pan_spread` exists to buy a **directivity** — the
/// measured drift of a single note's balance while it rings, 1.2-6.2 dB in the
/// recordings against 0.02-0.14 in a pan-potted engine — and it buys it by
/// moving where the two planes *are*: `voice.rs` renders the horizontal one at
/// `pan + spread·sign` and the vertical at `pan − spread·sign`, with `sign`
/// flipping on `key % 2` so that the instrument does not walk to one side. At
/// the shipped spread that is C4's two polarizations at pan **−0.42 and +0.30**
/// — over a metre of the string band apart — and D4's at the mirror image of
/// that.
///
/// Through a pan-pot that was a level trim and nothing more. Through a spaced
/// pair it is two *places*, so it is two interchannel delays and two comb
/// phases, and it alternates note by note: it is the dominant term of the ramp
/// `comb` reads, and the tune's colour crossing the whole image between E4 and
/// F4 (item 459) is exactly the parity of those five keys — C4, D4, E4 even and
/// F4, G4 odd.
///
/// This is the A/B: the shipped instrument, and the same instrument with the
/// spread at zero and nothing else changed. It asserts that `comb` **convicts
/// the alternation** — the shipped spread is worse in both of the column's own
/// verdicts — which is what item 451's pattern calls a tautology today and an
/// A/B the moment the mechanism is replaced by the per-polarization gain trim
/// item 467 names.
#[test]
fn the_comb_gate_fails_on_an_instrument_whose_polarization_spread_alternates_in_pan() {
    let Some(sfz) = sfz() else {
        eprintln!("no data/salamander in this tree; skipping the pan-spread falsification");
        return;
    };
    let mut flat = shipped_preset();
    // Both homes of the number: the per-key table when the preset draws one,
    // and the one global spread when it does not (`Preset::pan_spread`).
    let shipped_spread = melody::line_keys()
        .iter()
        .map(|&k| flat.pan_spread(k))
        .fold(0.0f32, f32::max);
    flat.voicing.polarization_pan_spread = 0.0;
    for cell in flat.notes.pan_spread.iter_mut() {
        *cell = 0.0;
    }
    flat.validate().expect("no spread is a legal preset");
    assert!(
        shipped_spread > 0.0,
        "the shipped preset spreads no polarization over the line, so this A/B is \
         of nothing"
    );
    let (columns, _, _) = score(&flat, &sfz);
    let text = melody::report(&columns);
    println!("{text}");
    let without = column(&columns, "comb", Window::Head);
    let Some(shipped_columns) = shipped() else {
        return;
    };
    let with = column(shipped_columns, "comb", Window::Head);
    println!(
        "the polarization spread, on the column that reads a position: slope {:+.3}/semitone with it and {:+.3} without, swing {:.2} and {:.2}, note by note {:?} against {:?}",
        with.slope,
        without.slope,
        with.swing,
        without.swing,
        with.notes.iter().map(|n| (n.key, (100.0 * n.engine).round() / 100.0)).collect::<Vec<_>>(),
        without.notes.iter().map(|n| (n.key, (100.0 * n.engine).round() / 100.0)).collect::<Vec<_>>(),
    );
    assert!(
        !with.pass,
        "the shipped instrument passes `comb`, so this A/B convicts nothing
{text}"
    );
    assert!(
        with.slope_error > without.slope_error && with.swing > without.swing,
        "zeroing the alternating polarization spread does not improve the tune's own          comb (slope error {:.3} -> {:.3}, swing {:.2} -> {:.2}), so `comb` is not          reading the position the spread puts the two planes at
{text}",
        with.slope_error,
        without.slope_error,
        with.swing,
        without.swing
    );
}

/// **The falsification for `comb`, and it is a different mechanism from every
/// other falsification on this board**: widen the pair and the tune's overtones
/// walk across the image.
///
/// Every other falsification here reinstalls a *band* or an old *table*. This
/// one moves the **geometry** and nothing else — the spacing is doubled, the
/// modal band, the width, the coherence and the height are the shipped
/// instrument's — because the defect `comb` was written for is a property of
/// where the two capsules stand. A pair twice as far apart has twice the
/// interchannel delay at every key, so its first interference null falls an
/// octave lower, right into the partials this column reads, and the ramp across
/// the tune steepens.
///
/// **And it asserts the pair, because the pair is the whole argument.** On this
/// instrument `splitting` — the column that was red on the shipped preset and
/// that a whole milestone was written for — goes *green* on its own verdict
/// (−2.36 dB against a bar of 3.26) while `comb` gets **worse** (slope −2.72 →
/// −3.47 per semitone, swing 18.5 → 26.6 dB). That is not a coincidence and it
/// is not luck: `splitting = balance − comb`, so widening the pair moves a
/// note's fundamental and that note's own overtones the same way and most of it
/// cancels. An instrument the scored column calls fine and the new column
/// convicts is exactly what "unscored dimension" means, and this test is it.
#[test]
fn the_comb_gate_fails_on_a_pair_that_stands_twice_as_wide() {
    let Some(sfz) = sfz() else {
        eprintln!("no data/salamander in this tree; skipping the comb falsification");
        return;
    };
    let mut wide = shipped_preset();
    let shipped_spacing = {
        let mics = wide
            .voicing
            .mics
            .as_mut()
            .expect("the measured preset declares [voicing.mics]");
        let was = mics.spacing_m;
        mics.spacing_m = 2.0 * was;
        was
    };
    wide.validate()
        .expect("twice the shipped spacing is still a legal preset");
    let (columns, _, _) = score(&wide, &sfz);
    let text = melody::report(&columns);
    println!("{text}");
    let comb = column(&columns, "comb", Window::Head);
    assert!(
        !comb.pass,
        "a pair standing {:.4} m apart instead of {shipped_spacing:.4} passes the comb \
         column, so that column does not test what it was written for: slope {:+.3}/sm \
         of {:.3}, swing {:.2} of {:.2}\n{text}",
        2.0 * shipped_spacing,
        comb.slope_error,
        comb.slope_bar,
        comb.swing,
        comb.swing_bar,
    );
    // **The arithmetic this test exists for, and `DECISIONS.md` 466 moved
    // where it is visible.** `splitting = balance − comb` exactly, so a
    // mechanism that moves a note's fundamental and that note's own overtones
    // together cancels out of `splitting` — and under item 459's target, which
    // was the recording's own image, the cancellation was a *verdict*: this
    // instrument took `splitting` green while `comb` got worse. Under the
    // neutral target `splitting`'s statistic is the median **magnitude** of the
    // line's own splits, which no longer cancels, so the same instrument now
    // fails both. What is asserted is therefore the cancellation itself rather
    // than a verdict built on it: on this instrument the column read the way
    // item 451 read it — the median of `engine − recording` — is **inside its
    // own bar**, and `comb` is five times outside its.
    let splitting = column(&columns, "splitting", Window::Head);
    println!(
        "the same instrument on the column that mostly cancels it: splitting swings \
{:.2} where comb swings {:.2}; splitting against the recording reads {:+.2} of {:.2} and \
its own median magnitude reads {:+.2} of the same bar",
        splitting.swing,
        comb.swing,
        splitting.balance_vs_reference,
        splitting.balance_bar,
        splitting.balance
    );
    assert!(
        splitting.balance_vs_reference.abs() <= splitting.balance_bar,
        "the pair-widening instrument moves `splitting` against the recording too, so \
         this test no longer shows what only `comb` can see: splitting {:+.2} of {:.2}\n{text}",
        splitting.balance_vs_reference,
        splitting.balance_bar
    );
    let shipped_comb = shipped().map_or(f64::NAN, |cs| column(cs, "comb", Window::Head).swing);
    assert!(
        comb.swing > shipped_comb,
        "twice the spacing does not comb the tune's overtones worse than the shipped \
         spacing does ({:.2} against {shipped_comb:.2}), so the column is not reading the \
         geometry\n{text}",
        comb.swing
    );
}

/// **The control the falsifications need to mean anything, and since
/// `DECISIONS.md` 466 it is two statements instead of one.**
///
/// A column that failed everything would fail item 418's band too and the tests
/// above would prove nothing, so the same measurement is run with the
/// **reference render standing in for the engine** — the recordings' own line
/// against the recordings' own line, with the neighbouring velocity layer still
/// the floor.
///
/// **(a) Against the recording's own image, every column is exactly zero and
/// passes.** That is the old control, unchanged, and it is what says the
/// machinery has no bias of its own: `Column::balance_vs_reference`,
/// `slope_error`, `agreement` are all identically zero here by construction and
/// are asserted anyway, because the construction is what a successor changes.
///
/// **(b) Against a *neutral* image, the recording fails — and by how much is
/// the size of the exclusion this milestone's policy rests on.** The three
/// columns of `melody::METRIC_IS_NEUTRAL` and the head-bound half of `cue` are
/// now scored against flat-and-centred rather than against this take, and this
/// take is **not** flat and centred: item 417 measured why (two capsules placed
/// asymmetrically across the board's nodal lines, one session's microphone
/// stand) and item 448(ii) and item 460 measured the size (C4's fundamental
/// 16.86 dB into the right capsule, carrying −949 µs with it, against −1.0 dB
/// at its neighbours). A policy that excludes something ought to be able to say
/// what it excluded and how large it was; this asserts that it is still there,
/// still that large, and still the C4 both other items named.
#[test]
fn the_recordings_own_line_is_the_image_the_neutral_policy_excludes() {
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
        balance.balance_vs_reference.abs() < 1e-9,
        "the recordings differ from themselves by {:+.3} dB\n{text}",
        balance.balance_vs_reference
    );
    // (b): and against neutral it is the recording that fails.
    assert!(
        !balance.balance_pass,
        "the recording's own image passes a neutral target, so the exclusion this \
         milestone's policy rests on is not a measurement of anything: own {:+.2} of \
         {:.2}\n{text}",
        balance.balance,
        balance.balance_bar
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
        splitting.balance_vs_reference.abs() < 1e-9,
        "the recordings split differently from themselves by {:+.3} dB\n{text}",
        splitting.balance_vs_reference
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
    // **And the same for `comb` and `cue`** (`DECISIONS.md` 459-460), where the
    // control matters for a third reason: both are scored against the
    // recording's *own* shape and not against a flat line, and a column that
    // demanded a flat overtone image or a zero interchannel time would fail the
    // piano it is scored against. The recording's line tilts and its capsules
    // are 12 cm apart; the assertions below are that the fixture actually
    // carries both, so a pass here is not a pass on nothing.
    let comb = column(&columns, "comb", Window::Head);
    // Under the neutral target `slope_error` is `|slope|` and is no longer zero
    // by construction; what is zero by construction is the distance between the
    // two lines, which is `slope − reference_slope` on the same two rows.
    assert!(
        (comb.slope - comb.reference_slope).abs() < 1e-9,
        "the recordings' own overtone image tilts differently from itself: {:+.4} \
         against {:+.4}\n{text}",
        comb.slope,
        comb.reference_slope
    );
    assert!(
        comb.swing_pass,
        "the recordings' own line fails the swing half, which is measured off it: \
         swing {:.2} of {:.2}\n{text}",
        comb.swing,
        comb.swing_bar
    );
    assert!(
        comb.reference_swing > 1.0,
        "the recording's own overtone image is flat to within a decibel across the \
         tune, so this control cannot distinguish a bar taken off the recording from a \
         bar of zero\n{text}"
    );
    let cue = column(&columns, "cue", Window::Head);
    assert!(
        cue.agreement.abs() < 1e-9,
        "the recordings' two cues agree differently with themselves: {:+.3}\n{text}",
        cue.agreement
    );
    // The agreement half is the one verdict of these four the recording keeps
    // under the neutral policy, and it keeps it *because* the policy asks only
    // for a positive correlation: the recording's two cues agree at +0.83.
    assert!(
        cue.agreement_pass && cue.engine_corr > 0.0,
        "the recording's own two localisation cues do not agree, so the half of \
         `cue` that survives item 469 is a bar on nothing: r = {:+.2}\n{text}",
        cue.engine_corr
    );
    // And the bound half is the one it fails, at the note item 417 named.
    assert!(
        !cue.bound_pass && cue.bound_key == 60,
        "the recording's own line is inside the head's own {:.0} µs at every note, so \
         `cue`'s per-note bound is not excluding anything: worst {:.0} µs at {}\n{text}",
        melody::HEAD_ITD_US,
        cue.bound,
        melody::note_name(cue.bound_key)
    );
    assert!(
        cue.notes.iter().any(|n| n.reference.abs() > 100.0),
        "the recording puts no note of the line more than 100 µs apart between its two \
         channels, so this control cannot distinguish a bar taken off a spaced pair \
         from a bar of zero\n{text}"
    );
    println!(
        "the recording's own line: comb slope {:+.3}/semitone, swing {:.2} dB; \
cue worst {:.0} µs at {}, and its two localisation cues agree at r = {:+.2} \
(the neighbouring velocity layer reads {:+.2})",
        comb.reference_slope,
        comb.reference_swing,
        cue.reference_bound,
        melody::note_name(cue.bound_key),
        cue.reference_corr,
        cue.layer_corr
    );
    // Every column, not just these: a control that only holds for the column it
    // was written for is not a control. **Against the recording's own image**,
    // which is what the columns outside `METRIC_IS_NEUTRAL` are still scored
    // against, and which is the whole of the old assertion.
    for c in &columns {
        assert!(
            c.balance_vs_reference.abs() < 1e-9,
            "{} ({}) differs from the recordings by {:+.3} on the recordings \
             themselves\n{text}",
            c.metric,
            c.window.name(),
            c.balance_vs_reference
        );
        if !c.neutral_target && c.metric != "cue" {
            assert!(
                c.pass,
                "{} ({}) fails on the recordings against themselves\n{text}",
                c.metric,
                c.window.name()
            );
        }
    }
    // **The size of the exclusion, printed rather than only asserted** — this is
    // the number `DECISIONS.md` 466 quotes and the one a successor re-opening
    // the policy has to argue with.
    println!(
        "the image this policy excludes: the recording's own ladder median magnitude is \
{:.2} dB against a neutral bar of {:.2}, its line splits by {:.2} dB, its overtone image \
tilts {:+.3}/semitone, and its worst note carries {:.0} µs at {} against the head's own {:.0}",
        balance.balance,
        balance.balance_bar,
        splitting.balance,
        comb.slope,
        cue.bound,
        melody::note_name(cue.bound_key),
        melody::HEAD_ITD_US
    );
}

/// The tail gate is a statement about a preset, and the statement it was
/// written to make is that a low band carrying the **whole row's** normaliser
/// fails it, at C4, on the metric the seam is in.
///
/// # Why the material changed with `DECISIONS.md` 455
///
/// Item 335's falsification put every *drawn* `partial_sigma_scale` row's
/// sub-2 kHz cells back to 1.0 and asserted the tail columns failed. It cannot
/// fail any more, and the reason is this milestone's own result rather than a
/// weakening: the step that draw existed to carry is **gone at its source**.
/// `estimate::tail::low_correct_row` closes every recorded key's sub-2 kHz
/// cells on the render, those keys' geometric means come back to about one, and
/// the line through them that `LowDecay` draws from is therefore flat at one —
/// so setting a drawn key's low cells to 1.0 now changes almost nothing.
/// `the_drawn_low_band_is_within_the_fits_own_resolution_of_one` asserts that
/// directly, on the preset and with no render in it.
///
/// What replaces it is the defect item 455 removed, rebuilt from whatever
/// preset ships and from arithmetic rather than from a stored table:
/// `estimate::shaping::partial_sigma_scale` normalises a row to geometric mean
/// one over **all** of its partials, `tail` then multiplies only the cells above
/// 2 kHz until the render matches, and what is left under 2 kHz is the row's own
/// low half divided by a normaliser the *high* half set. Dividing each key's
/// sub-2 kHz cells by its whole row's geometric mean puts that composition back
/// exactly, at every key at once, and nothing else is touched. It asserts the
/// *key*, because the listener named the key.
#[test]
fn the_tail_gate_fails_when_the_low_band_carries_the_whole_rows_normaliser() {
    let Some(sfz) = sfz() else {
        eprintln!("no data/salamander in this tree; skipping the tail falsification");
        return;
    };
    let before = with_the_low_band_seam(shipped_preset());
    let (columns, _, _) = score(&before, &sfz);
    let text = melody::report(&columns);
    println!("{text}");
    // **What the seam does is move the tail `hf` column onto C4**, which is
    // item 334's finding stated as a column: that column is a *share* whose
    // denominator is the fundamental, and the low band is what holds a
    // fundamental up. It no longer takes the column past its bar, and that is a
    // measurement rather than a weakening — item 452's onset fix took the
    // column's own worst note from 3.04 to 1.82 and this milestone's span
    // convention took the tail's from 3.75 to 3.06, so a defect that was worth
    // failing it in item 331's tree is worth naming the key in this one. What
    // is asserted is therefore the key and the size, both against the shipped
    // instrument measured the same way.
    let hf = column(&columns, "hf", Window::Tail);
    let shipped_hf = shipped().map(|cs| column(cs, "hf", Window::Tail));
    let at_c4 = |c: &Column| -> f64 {
        c.notes
            .iter()
            .find(|n| n.key == 60)
            .map_or(f64::NAN, |n| n.engine_residual.abs())
    };
    assert_eq!(
        hf.standout_key, 60,
        "the low band's seam does not put the tail `hf` column's worst note on C4, \
         which is the key item 334 attributed and the listener named\n{text}"
    );
    let (now, before) = (at_c4(hf), shipped_hf.map_or(f64::NAN, at_c4));
    assert!(
        now > before + 2.0,
        "C4 stands {now:.2} dB off the line's own tail `hf` trend with the seam back \
         against {before:.2} on the shipped instrument, so the seam is not what \
         moved it\n{text}"
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

/// **Is this note as loud as the piano's own note at the same key** —
/// `DECISIONS.md` 456, and the column every other one on this board is blind to
/// by construction.
///
/// `roughness`, `wobble` and `hf` are shapes; `strike`, `channel` and
/// `splitting` are ratios; `balance` is a position. A note eight decibels under
/// the piano's own moves none of them, which is why the C4 a listener picked
/// out of this very tune sat under a fully green board — the fourth time this
/// repository has failed that way (`CONTEXT.md`'s standing warning) and the
/// first time the missing column is a **level**.
///
/// It is gated on its **seam** and not on its median: the median is the
/// engine's master gain against the library's mastering, about 15 dB, and is
/// nobody's error, while the departure from it is the piano's own voicing plus
/// the engine's own mistakes. On the preset item 453 diagnosed it read
/// **8.87 dB at C4 against a bar of 5.21**; after item 457's per-key level it
/// reads **1.89 at A3**.
#[test]
fn every_note_of_the_line_is_as_loud_as_the_pianos_own() {
    gate("loudness", Window::Head);
}

/// The falsification for the column above, built the way this repository builds
/// them: out of the shipped preset and the **policy** the milestone replaced,
/// never out of a copy of a file or a stored table of numbers.
///
/// `DECISIONS.md` 272's decision was that a key's own level is measured,
/// reported and *written nowhere* — `estimate::shaping::energy_offset` pins
/// every `notes.partial_gains` row so that the power it puts through the
/// engine's own spectrum is the power that was already there.
/// `shaping::unlevel_row` is that policy as a function of a row, so applying it
/// to every key of the shipped preset reconstructs an instrument that carries
/// **no per-key level anywhere**, which is what every preset this repository
/// has ever shipped was.
///
/// It asserts the key as well as the failure, because the listener named the
/// key.
#[test]
fn the_loudness_gate_fails_on_the_policy_that_wrote_no_level_at_all() {
    let Some(sfz) = sfz() else {
        eprintln!("no data/salamander in this tree; skipping the loudness falsification");
        return;
    };
    let mut before = shipped_preset();
    let shaping = ShapingConfig::default();
    for row in &mut before.notes.partial_gains {
        *row = piano_tuner::estimate::shaping::unlevel_row(row, &shaping);
    }
    let (columns, _, _) = score(&before, &sfz);
    let text = melody::report(&columns);
    println!("{text}");
    let loudness = column(&columns, "loudness", Window::Head);
    assert!(
        !loudness.seam_pass,
        "an instrument carrying no per-key level at all passes the loudness column, \
         so that column does not test what it was written for: seam {:.2} against a \
         bar of {:.2}\n{text}",
        loudness.seam,
        loudness.seam_bar
    );
    // And it names the key, though not by rank: `unlevel_row` takes the level
    // out of **every** key, so which one ends up worst is a property of the
    // whole compass, while what item 453 measured is C4's own departure. So the
    // assertion is on C4's own seam, against the same key on the shipped
    // instrument.
    let c4 = |c: &Column| -> f64 {
        c.population
            .iter()
            .find(|p| p.key == 60)
            .map_or(f64::NAN, |p| p.seam.abs())
    };
    let now = c4(loudness);
    let shipped = shipped().map_or(f64::NAN, |cs| c4(column(cs, "loudness", Window::Head)));
    assert!(
        now > shipped + 2.0,
        "C4 departs {now:.2} dB from the register's median with no level written \
         anywhere against {shipped:.2} on the shipped instrument, so item 457's \
         per-key level is not what moved it\n{text}"
    );
}

/// **The step `DECISIONS.md` 335 drew is gone at its source**, which is what
/// makes the falsification above change its material rather than weaken.
///
/// Item 335 measured a fitted/unfitted step in one band: the geometric mean of
/// a row's cells under 2 kHz read 1.00 at A0, 0.94 at F#3, 0.75 at C4, 0.59 at
/// A4 and 0.39 at C5 over the *recorded* keys and exactly 1.000 at all 37 drawn
/// ones, because `TailCorrection::at` returns one below 2 kHz and nothing else
/// had ever written there. Item 455 gave the band to a render-closed per-partial
/// correction, so the recorded keys' own means came back to one and the line
/// `LowDecay` draws from went flat.
#[test]
fn the_drawn_low_band_is_within_the_fits_own_resolution_of_one() {
    let preset = shipped_preset();
    // Item 335's own statistic: the **group difference** between the keys the
    // library recorded and the keys it did not. Not every key's own mean — the
    // bass keys carry sixty cells under 2 kHz and their own measured spread,
    // and flattening that would be a different defect — but the step between
    // the two populations, which is what a draw can create and what item 335
    // measured at 1.17 dB of rate.
    // The provenance list is the population split, and it is the preset's own:
    // a key named in `notes.synthesized_decay` carries a drawn row.
    let drawn_keys = preset.notes.synthesized_decay.clone();
    let mut groups: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
    for key in 21..=108u8 {
        let i = usize::from(key - 21);
        let row = &preset.notes.partial_sigma_scale[i];
        let params = preset.string_params(key);
        let hz: Vec<f64> = (1..=params.partial_count())
            .map(|k| f64::from(params.partial_freq(k)))
            .collect();
        let Some((mean, _)) = piano_tuner::estimate::tail::low_mean(row, &hz) else {
            continue;
        };
        groups[usize::from(!drawn_keys.contains(&key))].push(mean.ln());
    }
    let median = |v: &mut Vec<f64>| -> f64 {
        v.sort_by(f64::total_cmp);
        if v.is_empty() {
            0.0
        } else {
            v[v.len() / 2]
        }
    };
    let (drawn, fitted) = (median(&mut groups[0]), median(&mut groups[1]));
    let step = (fitted - drawn).exp();
    let bar = piano_tuner::estimate::tail::LOW_DEADBAND;
    assert!(
        (bar.recip()..=bar).contains(&step),
        "the keys the library recorded carry a sub-2 kHz geometric mean of \
         {:.3} against {:.3} at the keys it did not — a step of x{step:.3}, outside \
         the fit's own resolution of x{bar:.2}",
        fitted.exp(),
        drawn.exp()
    );
}
