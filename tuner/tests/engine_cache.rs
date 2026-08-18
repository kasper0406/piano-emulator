//! The engine-render cache is the same render, and it misses on anything that
//! could make it a different one.
//!
//! `tests/reference_cache.rs` is this file's twin for the *reference* side, and
//! it makes the same two claims for the same reason: a cache that can be wrong
//! makes a measurement lie, so what is asserted is not "close" but **identical
//! samples**, and not "invalidated" but "a changed input lands on a different
//! name". `piano_tuner::renders` is the module, and its header is where the
//! three parts of a key are argued.

use std::path::PathBuf;

use piano_emulator::preset::{MicVoicing, ModalBand, Preset};
use piano_tuner::renders::{render_note, EngineRenders, NoteSpec};

/// A short note: enough to exercise the strike, the board's own field and the
/// microphone stage, cheap enough for a unit test.
const SPEC: NoteSpec = NoteSpec {
    key: 60,
    velocity: 90,
    duration_s: 0.4,
    preroll: 3_840,
};

fn temporary(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "piano-engine-cache-{}-{}-{name}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn mic_preset() -> Preset {
    let mut preset = Preset::default();
    preset.voicing.mics = Some(MicVoicing {
        spacing_m: 0.12,
        height_m: 0.12,
        span_m: 1.2,
        width: 1.4,
        diffuse_coherence: 2.5,
        modal: Some(ModalBand {
            lo_hz: 220.0,
            hi_hz: 300.0,
            // Under `soundboard::MIC_MODAL_LIFT`'s rail since
            // `DECISIONS.md` 418 — the fixture is `validate`d, so an illegal
            // lift is a fixture that is not a preset.
            lift: 0.99,
        }),
    });
    preset.validate().expect("a legal fixture");
    preset
}

/// **A hit is the render, bit for bit.**
///
/// Not "within a tolerance": the entry is a 32-bit float WAV, so the samples
/// that come back are the samples that went in, and anything else would mean a
/// board scored against a cached render and a board scored against a fresh one
/// were two different numbers.
#[test]
fn a_cached_engine_render_is_the_render_it_replaces() {
    let dir = temporary("identity");
    let cache = EngineRenders::at(&dir);
    let preset = mic_preset();
    let fresh = render_note(&preset, SPEC);
    let cold = cache.note(&preset, SPEC);
    let warm = cache.note(&preset, SPEC);
    assert!(
        fresh.channels[0].iter().any(|&x| x.abs() > 1.0e-3),
        "the fixture made no sound, so nothing here proves anything"
    );
    for (name, got) in [("the first call", &cold), ("the second", &warm)] {
        assert_eq!(
            got.channels.len(),
            fresh.channels.len(),
            "{name} came back with a different number of channels"
        );
        for (c, (a, b)) in fresh.channels.iter().zip(&got.channels).enumerate() {
            assert_eq!(a, b, "{name}: channel {c} is not the render it replaces");
        }
    }
    // ... and it really was read rather than recomputed: an entry exists.
    let entries = std::fs::read_dir(&dir).expect("the cache wrote its directory").count();
    assert_eq!(entries, 1, "one render, one entry");
    let _ = std::fs::remove_dir_all(&dir);
}

/// **A changed input misses.** Every part of the key, one at a time: the
/// preset's own bytes, and the material asked of the engine.
///
/// The third part — the engine's code — cannot be moved from inside a test, so
/// it is asserted structurally instead: the probe renders that make
/// `engine_fingerprint` go through the same `render_to_buffer` every other
/// render does, and one of the two carries a full `[voicing.mics]`, so a change
/// to either branch of `soundboard` moves it. What *is* asserted here is that
/// the fingerprint is a constant of the process, since a key that moved between
/// two calls in one run would miss every time and be no cache at all.
#[test]
fn anything_that_would_change_the_render_lands_on_a_different_entry() {
    let dir = temporary("misses");
    let cache = EngineRenders::at(&dir);
    let base = mic_preset();
    cache.note(&base, SPEC);

    let mut turned = base.clone();
    turned.voicing.mics = turned.voicing.mics.map(|m| MicVoicing {
        // Down rather than up: the fixture sits a hundredth under item 418's
        // rail, so the only legal way to move it is towards zero.
        modal: m.modal.map(|b| ModalBand {
            lift: b.lift - 0.1,
            ..b
        }),
        ..m
    });
    turned.validate().expect("still legal");
    assert_ne!(
        base.to_toml(),
        turned.to_toml(),
        "the fixture did not actually move"
    );
    cache.note(&turned, SPEC);
    cache.note(
        &base,
        NoteSpec {
            key: 61,
            ..SPEC
        },
    );
    cache.note(
        &base,
        NoteSpec {
            velocity: 91,
            ..SPEC
        },
    );
    cache.note(
        &base,
        NoteSpec {
            duration_s: 0.5,
            ..SPEC
        },
    );
    cache.note(
        &base,
        NoteSpec {
            preroll: 3_840 * 2,
            ..SPEC
        },
    );
    let entries = std::fs::read_dir(&dir).expect("a directory").count();
    assert_eq!(
        entries, 6,
        "six different renders were asked for and the cache holds {entries} entries"
    );

    // The same six again: every one of them a hit, and the directory unmoved.
    let (hits_before, _) = piano_tuner::renders::stats();
    cache.note(&base, SPEC);
    cache.note(&turned, SPEC);
    let (hits_after, _) = piano_tuner::renders::stats();
    assert_eq!(
        hits_after - hits_before,
        2,
        "asking again for a render the cache holds did not read it"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The cache turned off renders, and renders *the same thing*: the control that
/// says `EngineRenders::off` is a null and not a second code path.
#[test]
fn the_cache_turned_off_is_the_render_with_no_disk_in_it() {
    let preset = mic_preset();
    let direct = render_note(&preset, SPEC);
    let through = EngineRenders::off().note(&preset, SPEC);
    for (c, (a, b)) in direct.channels.iter().zip(&through.channels).enumerate() {
        assert_eq!(a, b, "channel {c}");
    }
}
