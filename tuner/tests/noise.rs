//! The mechanism-noise write gate, and the two things it must get right.
//!
//! `DECISIONS.md` 531. The survey writes `[noise.key_off]`,
//! `[noise.damper_lift]`, `[noise.pedal_down]` and `[noise.pedal_up]` straight
//! off a library's mechanism recordings, measured as a peak against a strike of
//! the same key. On Salamander that is a measurement of a piano. On the
//! bitKlavier grand it is a measurement of the *editor* — its key-off group is
//! published at the session gain with the hall on it, and reads from −14.9 dB
//! to **+10.87 dB** against the note it belongs to, which is a damper landing
//! louder than the chord. The old code clamped that to 0 dB and wrote it, and
//! `presets/concert-grand-d.toml` shipped a key-off table at −1 … −9 dB with one
//! key at exactly 0.0 — the rail, printed as a measurement.
//!
//! Two tests, and they pull in opposite directions on purpose:
//!
//! * [`an_editorially_hot_key_off_group_writes_no_table_at_all`] is the
//!   falsification. It is built from the bitKlavier readings as
//!   `survey::measure_mechanism` actually read them, and it asserts both halves
//!   — that the gated code refuses, **and** that the ungated arithmetic still on
//!   disk in [`compass_anchors`] reproduces the defect from the same input, so
//!   the test would have failed before the fix and passes after it.
//! * [`salamanders_own_mechanism_is_written_bit_identically`] is the other
//!   side. The gate must not cost the one library whose mechanism group is
//!   genuine anything at all: measured off the shipped corpus, its tables have
//!   to come back **equal to the ones `presets/salamander-c5.toml` already
//!   carries**, field for field and anchor for anchor, and to hash to a pinned
//!   constant. It needs the 707 MiB corpus and skips itself without it, as the
//!   other corpus tests here do.

use std::path::PathBuf;

use piano_tuner::estimate::noise::{
    compass_anchors, fit_noise_screened, EventMetrics, MechanismMeasurements, NoiseConfig,
    MAX_MECHANISM_LEVEL_DB,
};
use piano_tuner::preset::{EventNoise, NoiseTables, Preset};
use piano_tuner::survey::measure_mechanism;
use piano_tuner::SampleLibrary;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the tuner sits in the workspace")
        .to_path_buf()
}

fn reading(key: u8, level_db: f64) -> EventMetrics {
    EventMetrics {
        key: Some(key),
        level_db,
        decay_s: 0.36,
        centroid_hz: 211.0,
        reference_key: key,
    }
}

/// The bitKlavier grand's key-off group as `survey::measure_mechanism` reads it
/// off `data/bitklavier-piano-bar`, sampled every eighth key across the
/// compass. All 88 readings sit above `MAX_MECHANISM_LEVEL_DB`; 21 of them are
/// positive. The whole table is in `DECISIONS.md` 531.
fn an_editorially_hot_library() -> MechanismMeasurements {
    let levels: [(u8, f64); 12] = [
        (21, -4.1),
        (29, -9.6),
        (37, -3.4),
        (45, -9.1),
        (53, -2.7),
        (61, 8.8),
        (63, 8.8),
        (69, 1.6),
        (77, 1.5),
        (91, 10.9),
        (99, 7.5),
        (108, -7.6),
    ];
    MechanismMeasurements {
        key_off: levels.iter().map(|&(k, db)| reading(k, db)).collect(),
        key_off_veltrack: None,
        velocity_span: Some((4, 127)),
        ..MechanismMeasurements::default()
    }
}

/// **The falsification.** A library whose mechanism samples are editorially hot
/// writes no mechanism table, and the arithmetic that used to write one still
/// produces the preset that shipped.
#[test]
fn an_editorially_hot_key_off_group_writes_no_table_at_all() {
    let base = NoiseTables::default();
    let measurements = an_editorially_hot_library();
    let config = NoiseConfig::default();

    // (a) What the old code did with exactly this input, still on disk: the
    // per-anchor reduction with nothing screening it. This is the defect.
    let levels: Vec<(Option<u8>, f64)> = measurements
        .key_off
        .iter()
        .map(|m| (m.key, m.level_db))
        .collect();
    let ungated = compass_anchors(&levels, 0.0, &config).expect("the old rule wrote a table");
    assert!(
        ungated.iter().all(|a| f64::from(a.db) > MAX_MECHANISM_LEVEL_DB),
        "every anchor the old rule wrote is hotter than the gate: {ungated:?}"
    );
    assert!(
        ungated.iter().any(|a| a.db == 0.0),
        "and at least one of them is the rail itself, which is how \
         presets/concert-grand-d.toml came to carry `db = 0.0`: {ungated:?}"
    );

    // (b) What the gated code does with it: nothing. Not a quieter table, not
    // the surviving tail — the base preset's own, unchanged, which is the
    // convention that says nobody measured this on this piano.
    let (fitted, screening) = fit_noise_screened(&measurements, &base, &config);
    assert_eq!(fitted.key_off, base.key_off, "{screening:?}");
    assert_eq!(fitted.damper_lift, base.damper_lift, "{screening:?}");
    assert!(!screening.key_off.accepted());
    assert_eq!(screening.key_off.kept, 0);
    assert_eq!(screening.key_off.read, measurements.key_off.len());
    assert_eq!(
        screening.refused(),
        vec!["key_off", "damper_lift"],
        "the lift is derived from the fall and carries its verdict"
    );

    // (c) And the preset has to say so. A table left at the base's value is
    // indistinguishable from one that was measured and happened to agree,
    // unless the file records the refusal.
    let described = screening.describe("estimated by piano-tuner from somewhere");
    assert!(described.contains("[noise.key_off]"), "{described}");
    assert!(described.contains("[noise.damper_lift]"), "{described}");
    assert!(described.contains("DECISIONS.md 531"), "{described}");
    // Re-entrant: the mechanism stage may be run over its own output.
    assert_eq!(screening.describe(&described), described);
}

/// A group that is *mostly* honest still writes, because the per-anchor median
/// was always able to survive a broken file or two — and one that is half
/// honest does not, because a median of the survivors is a median of the tail.
#[test]
fn the_gate_is_a_strict_majority_and_not_a_veto_on_one_bad_file() {
    let base = NoiseTables::default();
    let config = NoiseConfig::default();
    let mostly = MechanismMeasurements {
        key_off: vec![
            reading(21, -37.0),
            reading(33, -35.0),
            reading(45, -33.0),
            reading(57, 4.0),
        ],
        ..MechanismMeasurements::default()
    };
    let (fitted, screening) = fit_noise_screened(&mostly, &base, &config);
    assert!(screening.key_off.accepted(), "{screening:?}");
    assert_ne!(fitted.key_off, base.key_off);
    assert!(
        fitted.key_off.level_db.iter().all(|a| f64::from(a.db) <= MAX_MECHANISM_LEVEL_DB),
        "and the one hot file is not in what was written: {:?}",
        fitted.key_off.level_db
    );

    // Two takes of one gesture, one of them implausible — the bitKlavier
    // grand's pedal-down group exactly — is not a majority.
    let split = MechanismMeasurements {
        pedal_down: vec![
            EventMetrics { key: None, ..reading(60, -33.4) },
            EventMetrics { key: None, ..reading(60, -20.2) },
        ],
        ..MechanismMeasurements::default()
    };
    let (fitted, screening) = fit_noise_screened(&split, &base, &config);
    assert!(!screening.pedal_down.accepted(), "{screening:?}");
    assert_eq!(fitted.pedal_down, base.pedal_down);
}

/// FNV-1a over a mechanism table's every field, in the order the preset writes
/// them. Not a cryptographic digest: what it has to do is change when any
/// number changes and be reproducible across machines and process runs, so it
/// eats the `f32` bit patterns rather than the values.
fn hash_noise(noise: &NoiseTables) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |byte: u8| {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    };
    let mut event = |event: &EventNoise| {
        for value in [event.centroid_hz, event.decay_s, event.velocity_db] {
            for byte in value.to_bits().to_le_bytes() {
                eat(byte);
            }
        }
        for anchor in &event.level_db {
            eat(anchor.key);
            for byte in anchor.db.to_bits().to_le_bytes() {
                eat(byte);
            }
        }
    };
    for (_, e) in noise.events() {
        event(e);
    }
    hash
}

/// **The bit-exactness pin**, the other half of the falsification.
///
/// Measured 2026-08-20 on `presets/salamander-c5.toml`'s own `[noise]` tables —
/// the four mechanism events as the survey wrote them before the gate existed —
/// and reproduced by the gated code from the corpus. **Do not update this
/// constant to make a test pass.** It moving means the mechanism of the one
/// measured piano this repository is barred against has moved.
const SALAMANDER_MECHANISM_HASH: u64 = 0xf250_ba06_a3d1_3131;

/// The gate costs the genuine library nothing: same readings, same tables, and
/// they are the tables already in the shipped preset.
#[test]
fn salamanders_own_mechanism_is_written_bit_identically() {
    let sfz = repo()
        .join("data/salamander")
        .join("SalamanderGrandPiano-V3+20200602.sfz");
    if !sfz.exists() {
        eprintln!("skipping: {} is not here (707 MiB, gitignored)", sfz.display());
        return;
    }
    let library = SampleLibrary::from_sfz(&sfz).expect("the shipped SFZ reads");
    let config = NoiseConfig::default();
    let measurements = measure_mechanism(&library, &config);
    let base = Preset::load(repo().join("presets/default.toml")).expect("the base preset loads");
    let (fitted, screening) = fit_noise_screened(&measurements, &base.noise, &config);

    // Every reading passes, and none of them is close to the gate: the
    // hottest of the 88 key-off recordings is -24.64 dB.
    assert!(screening.refused().is_empty(), "{screening:?}");
    assert_eq!(screening.key_off.kept, screening.key_off.read);
    assert_eq!(screening.key_off.read, 88);
    assert!(
        screening.key_off.hottest_db < MAX_MECHANISM_LEVEL_DB - 3.0,
        "the genuine library clears the gate by more than three decibels: {screening:?}"
    );
    assert!(screening.describe("x") == "x", "nothing to record");

    // And the tables are the shipped preset's, field for field.
    let shipped = Preset::load(repo().join("presets/salamander-c5.toml"))
        .expect("the measured preset loads");
    for ((name, ours), (_, theirs)) in fitted.events().into_iter().zip(shipped.noise.events()) {
        assert_eq!(ours, theirs, "[noise.{name}] moved");
    }
    assert_eq!(
        hash_noise(&fitted),
        SALAMANDER_MECHANISM_HASH,
        "the mechanism of the measured preset moved"
    );
}
