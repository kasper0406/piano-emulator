//! **The `halo` column**: how loud the rest of the instrument is when one key
//! is struck (`DECISIONS.md` 500-503).
//!
//! `CONTEXT.md`'s standing rule is that unscored dimensions are how this
//! repository fails, and the treble sympathetic halo has been a *named* gap
//! since `docs/history/TUNING_REPORT.md` §4 without ever being a column. It was
//! not unscored for want of trying — `estimate::halo::salamander_targets`
//! carries five rows and the fit closes on them — but three of the five are
//! §4's **between-partial census**, and that census has a floor, and the floor
//! is the struck note itself. On an 85 ms window a treble note's own decaying
//! partials smear outside the guard band at about −48 dB, which is where the
//! engine reads; and the census's own falsification here
//! ([`the_between_partial_census_cannot_see_the_halo_at_all`]) shows that
//! removing **every** sympathetic path from the instrument moves the C6 row by
//! a tenth of a decibel. A target no mechanism can move is not a target.
//!
//! So this file scores the halo the way the library recorded it: alone.
//! Salamander samples the string resonance a released key leaves behind
//! separately from the note and separately from the key-off thump, and the
//! engine's own halo is isolated the same way — by rendering the note twice and
//! subtracting, which removes the struck string, the hammer and the mechanism
//! by cancellation rather than by a window chosen after the fact.
//!
//! Four tests and only one of them is the verdict. The other three are what
//! make the verdict worth having: the reader must find **nothing** when there
//! is no halo, it must find **more** when there is more, and the column it
//! replaces must be shown unable to see either.
//!
//! Every test here needs the Salamander library and skips itself without it,
//! the same way `tests/melody.rs` does.

use std::path::PathBuf;

use piano_emulator::preset::Preset;
use piano_tuner::estimate::halo::{
    self, HaloColumn, HALO_BAR_DB, HALO_FIRST_KEY, HALO_HOLD_S, HALO_VELOCITY,
};
use piano_tuner::SampleLibrary;

/// Salamander's struck-note groups' own `amp_veltrack`. The one number the
/// reference side needs that `SampleLibrary` does not carry per sample; it is
/// asserted against the file by [`the_attack_groups_velocity_law_is_what_this_file_assumes`].
const ATTACK_VELTRACK: f64 = 73.0;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn sfz() -> Option<PathBuf> {
    let path = repo()
        .join("data/salamander")
        .join("SalamanderGrandPiano-V3+20200602.sfz");
    path.exists().then_some(path)
}

/// The shipped instrument, **with the microphone pair taken out**.
///
/// This column is a mono statistic and mono discipline is a contract: the
/// pair's fold-down equals the pan-pot's render to about −120 dBFS
/// (`CONTEXT.md`), so `[voicing.mics]` cannot move a ratio of two mono peaks.
/// Taking it out is therefore free, and it buys two things that are not. The
/// column stops being a function of a stereo refit it has nothing to do with —
/// which matters here, because `presets/salamander-c5.toml`'s `[voicing.mics]`
/// is owned and re-fitted by a different workstream — and the halo the reader
/// isolates is provably the instrument answering itself rather than anything a
/// capsule geometry does to it.
fn shipped_preset() -> Preset {
    let text = std::fs::read_to_string(repo().join("presets/salamander-c5.toml"))
        .expect("the measured preset is in the tree");
    let mut preset = piano_tuner::preset::Preset::from_toml(&text).expect("it parses");
    preset.voicing.mics = None;
    Preset::from_toml(&preset.to_toml()).expect("the mono instrument is a legal one")
}

/// The instrument with nothing sympathetic in it: no bus, no segments.
fn without_the_halo(mut preset: Preset) -> Preset {
    preset.voicing.resonance_coupling = 0.0;
    preset.notes.duplex = Vec::new();
    preset
        .validate()
        .expect("an instrument with no sympathetic path is still a legal one");
    preset
}

fn column() -> Option<(HaloColumn, SampleLibrary)> {
    let sfz = sfz()?;
    let library = SampleLibrary::from_sfz(&sfz).expect("the library reads");
    let preset = shipped_preset();
    Some((
        halo::halo_column(&preset, &library, ATTACK_VELTRACK),
        library,
    ))
}

/// The verdict, and the whole reason this file exists.
///
/// **Red on the instrument that ships, by 21.2 dB**, and `#[ignore]`d under
/// `DECISIONS.md` 463's policy with its own item in the reason string. The
/// diagnosis is 502 and the disposition 503: no legal setting of any knob in
/// the schema closes it (the coupling has 0.85 dB of authority left before the
/// stability contract refuses the preset, the segments have 0.4, and the
/// board's diffuse field bought 6.5 at a T60 ten times the one a soundboard
/// has), and what does close it is a mechanism the engine does not have.
///
/// The verdict is a **seam** — the worst per-key shortfall — and not a median,
/// because `CONTEXT.md`'s own rule from D453/D456/D459 is that a per-key error
/// cancels out of a median and a ramp cancels out of it twice, and this defect
/// is both: it runs from +3.9 dB at D#4 to +22.6 at D#6.
#[test]
#[cfg_attr(debug_assertions, ignore = "the gate is only meaningful in --release")]
#[cfg_attr(
    not(debug_assertions),
    ignore = "DECISIONS.md 502-503: the treble halo is 21.2 dB short and the mechanism that \
              would close it is not built"
)]
fn the_engines_halo_is_as_loud_as_the_recordings_own() {
    let Some((column, _)) = column() else {
        eprintln!("skipping: the Salamander library is not here");
        return;
    };
    assert!(
        column.rows.len() >= 8,
        "the column found only {} keys; the library should give ten from C4 up",
        column.rows.len()
    );
    println!("  key   recorded   engine    error");
    for row in &column.rows {
        println!(
            "  {:>3}   {:>+8.2} {:>+8.2} {:>+8.2}",
            row.key,
            row.recorded.peak_db,
            row.engine.peak_db,
            row.error_db()
        );
    }
    let (key, seam) = column.seam().expect("a populated column has a seam");
    println!(
        "  seam {seam:+.2} dB at key {key}, median {:+.2}, slope {:+.4} dB/semitone, \
         bar {HALO_BAR_DB:.2}",
        column.median_db(),
        column.slope_db_per_semitone()
    );
    assert!(
        column.passes(),
        "the halo is {seam:+.2} dB out at key {key} against a bar of {HALO_BAR_DB:.2}"
    );
}

/// **The falsification.** The reader must find nothing when there is nothing to
/// find.
///
/// This is the test the between-partial census never had and could not pass:
/// take every sympathetic path out of the instrument and the halo the reader
/// isolates is *identically zero samples*, because the two renders it subtracts
/// are the same render. A reader that still returned a level would be reading
/// the note, the window or its own arithmetic, and a green verdict from it
/// would mean nothing.
#[test]
#[cfg_attr(debug_assertions, ignore = "the gate is only meaningful in --release")]
fn the_halo_reader_finds_nothing_when_the_sympathetic_path_is_silenced() {
    let bare = without_the_halo(shipped_preset());
    for key in [60u8, 72, 84] {
        let level = halo::engine_halo_level(&bare, key, HALO_VELOCITY, HALO_HOLD_S);
        assert!(
            level.is_none(),
            "key {key}: the reader returned {level:?} from an instrument with no bus and no \
             segments"
        );
    }
    // ... and the control beside it, so that the null above is the mechanism
    // and not a broken reader.
    let shipped = shipped_preset();
    let level = halo::engine_halo_level(&shipped, 84, HALO_VELOCITY, HALO_HOLD_S)
        .expect("the shipped instrument has a halo to find");
    assert!(
        level.peak_db.is_finite() && level.peak_db < 0.0,
        "the shipped instrument's C6 halo reads {level:?}"
    );
}

/// **The control.** The reader must find *more* when there is more.
///
/// A tenth of the coupling is 20 dB less drive on the bus, and the column has
/// to move with it or it is not a function of the mechanism it is named after.
/// It moves about half a decibel per decibel of drive rather than one, for the
/// reason `calibration::the_halo_level_follows_the_coupling_the_fit_inverts_it_on`
/// records: a louder halo wakes voices that were culled.
#[test]
#[cfg_attr(debug_assertions, ignore = "the gate is only meaningful in --release")]
fn the_halo_column_rises_with_the_coupling_it_is_a_function_of() {
    let base = shipped_preset();
    let level = |coupling: f32| -> f64 {
        let mut preset = base.clone();
        preset.voicing.resonance_coupling = coupling;
        halo::engine_halo_level(&preset, 84, HALO_VELOCITY, HALO_HOLD_S)
            .map_or(f64::NAN, |l| l.peak_db)
    };
    let quiet = level(0.001);
    let loud = level(base.voicing.resonance_coupling);
    println!("C6 halo at coupling 0.001: {quiet:.1} dB; at the shipped one: {loud:.1} dB");
    assert!(
        loud - quiet > 5.0,
        "ten times the coupling moved the column {:+.1} dB",
        loud - quiet
    );
}

/// **The falsification of the column this one replaces**, and the reason
/// `salamander_targets`' two treble rows have to go.
///
/// §4's between-partial census is asked the one question that decides whether
/// it can be a fitting target: how much does it move when the whole mechanism
/// it is supposed to measure is taken out? On C6 and C7, on a window three
/// times longer than the shipped `HaloConfig`'s — which is the *best* case for
/// it, because a longer window is what lifts the leakage floor — the answer is
/// under a decibel, where the recordings stand 20 to 28 dB above the engine.
#[test]
#[cfg_attr(debug_assertions, ignore = "the gate is only meaningful in --release")]
fn the_between_partial_census_cannot_see_the_halo_at_all() {
    use piano_emulator::render::{render_to_buffer, RenderEvent};
    use piano_emulator::types::{Event, SAMPLE_RATE};

    let Some(sfz) = sfz() else {
        eprintln!("skipping: the Salamander library is not here");
        return;
    };
    let _ = sfz;
    let shipped = shipped_preset();
    let bare = without_the_halo(shipped.clone());
    let tuner_preset = piano_tuner::preset::Preset::from_toml(
        &std::fs::read_to_string(repo().join("presets/salamander-c5.toml")).expect("read"),
    )
    .expect("parse");
    let survey = piano_tuner::survey::SurveyConfig::default();

    for key in [84u8, 96] {
        let f0 = f64::from(tuner_preset.notes.f0_hz[usize::from(key - 21)]);
        let config = halo::HaloConfig {
            window: 16_384,
            at_s: 1.0,
            ..halo::HaloConfig::default()
        };
        let note_config = survey.note_config(f0).expect("a note config");
        let census = |preset: &Preset| -> f64 {
            let (l, r) = render_to_buffer(
                preset,
                &[RenderEvent::new(
                    0.0,
                    Event::NoteOn {
                        key,
                        vel: u16::from(HALO_VELOCITY),
                    },
                )],
                4.0,
            );
            let mono: Vec<f32> = l.iter().zip(&r).map(|(&a, &b)| 0.5 * (a + b)).collect();
            halo::between_partials(&mono, f64::from(SAMPLE_RATE), f0, &note_config, &config)
                .map(|b| b.at_late_db)
                .unwrap_or(f64::NAN)
        };
        let with = census(&shipped);
        let without = census(&bare);
        println!("  key {key}: census {with:+.2} with the halo, {without:+.2} with none");
        assert!(
            (with - without).abs() < 1.5,
            "key {key}: taking the whole sympathetic path out moved the census {:+.2} dB, so it \
             can see the mechanism after all and this test is the one that is wrong",
            with - without
        );
    }
}

/// The reference pays the hold it was recorded after, and the SFZ says how
/// much.
///
/// A release recording is what the strings still hold when the damper lands, so
/// its level is only comparable with a strike's once the hold is named — and
/// reading it without `rt_decay` reads the halo 6 to 9 dB too loud, which is
/// where `salamander_targets`' `harmLC3` = −31 and `harmLC5` = −39 came from.
#[test]
fn the_reference_pays_the_hold_it_was_recorded_after() {
    let Some(sfz) = sfz() else {
        eprintln!("skipping: the Salamander library is not here");
        return;
    };
    let library = SampleLibrary::from_sfz(&sfz).expect("the library reads");
    let resonances = library.mechanism_of(piano_tuner::library::MechanismKind::StringResonance);
    assert!(
        !resonances.is_empty(),
        "the library maps no string resonances at all"
    );
    assert!(
        resonances.iter().all(|s| s.rt_decay > 0.0),
        "some release resonance declares no rt_decay, so its level cannot be placed against a \
         strike"
    );
    let held = halo::recorded_halo_level(&library, 72, HALO_VELOCITY, 1.0, ATTACK_VELTRACK)
        .expect("C5 has a release resonance");
    let struck = halo::recorded_halo_level(&library, 72, HALO_VELOCITY, 0.0, ATTACK_VELTRACK)
        .expect("the same, read with no hold");
    let rt = resonances
        .iter()
        .find(|s| s.key == Some(72) && s.lovel <= HALO_VELOCITY)
        .map(|s| s.rt_decay)
        .expect("C5's own rt_decay");
    println!("C5's halo reads {held:.2?} after a 1 s hold and {struck:.2?} after none");
    assert!(
        (struck.peak_db - held.peak_db - rt).abs() < 1.0e-6,
        "the hold is worth {:+.3} dB where the SFZ declares {rt:+.3}",
        struck.peak_db - held.peak_db
    );
}

/// The one number the reference side needs that the library does not carry per
/// sample, checked against the file rather than trusted.
#[test]
fn the_attack_groups_velocity_law_is_what_this_file_assumes() {
    let Some(sfz) = sfz() else {
        eprintln!("skipping: the Salamander library is not here");
        return;
    };
    let text = std::fs::read_to_string(&sfz).expect("the SFZ reads");
    let attack_groups: Vec<f64> = text
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with("<group>") && !l.contains("trigger=release"))
        .filter_map(|l| {
            let at = l.find("amp_veltrack=")? + "amp_veltrack=".len();
            l[at..].split_whitespace().next()?.parse().ok()
        })
        .collect();
    assert!(
        !attack_groups.is_empty() && attack_groups.iter().all(|&v| v == ATTACK_VELTRACK),
        "the struck-note groups declare {attack_groups:?}, not {ATTACK_VELTRACK}"
    );
    assert_eq!(HALO_FIRST_KEY, 60);
}
