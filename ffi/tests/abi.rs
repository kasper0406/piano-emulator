//! The C ABI's contract, from the C caller's side: what every entry point does
//! with a null, with a lie, and with the ordinary case.
//!
//! These are the tests that stand in for the Swift that does not exist yet
//! (`DISTRIBUTION.md` M2/M3). A host is not a friendly caller — it hands you the
//! rate it feels like, it saves state at odd moments and it reloads a project
//! whose preset has moved on — so every one of those is spelled out here rather
//! than left to the header's prose.

use piano_emulator_ffi::*;
use std::ffi::CStr;

fn note_on(key: u32, vel: u32) -> pe_event_t {
    pe_event_t {
        kind: pe_event_kind::PE_EVENT_NOTE_ON as u32,
        key,
        vel,
        value: 0.0,
    }
}

/// Renders `frames` and returns both channels.
unsafe fn render(engine: *mut pe_engine, frames: usize) -> (Vec<f32>, Vec<f32>) {
    let mut l = vec![0.0f32; frames];
    let mut r = vec![0.0f32; frames];
    pe_render(engine, l.as_mut_ptr(), r.as_mut_ptr(), frames as u32);
    (l, r)
}

unsafe fn last_error(engine: *mut pe_engine) -> String {
    CStr::from_ptr(pe_last_error(engine))
        .to_string_lossy()
        .into_owned()
}

#[test]
fn the_library_and_the_header_agree_on_the_abi_version() {
    assert_eq!(pe_abi_version(), PE_ABI_VERSION);
    assert_eq!(PE_ENGINE_SAMPLE_RATE, 48_000);
}

/// A null handle is a no-op everywhere, because a host that failed to
/// instantiate us will still call `deallocateRenderResources` and a render
/// block that has been torn down still has a pointer.
#[test]
fn every_entry_point_survives_a_null_handle() {
    unsafe {
        pe_destroy(std::ptr::null_mut());
        pe_reset(std::ptr::null_mut());
        pe_event(std::ptr::null_mut(), note_on(60, 90));
        assert!(!pe_post_event(std::ptr::null_mut(), note_on(60, 90)));
        assert_eq!(
            pe_load_preset_toml(std::ptr::null_mut(), c"".as_ptr(), 0),
            pe_status::PE_ERR_NULL
        );
        assert_eq!(
            pe_save_state(std::ptr::null_mut(), std::ptr::null_mut(), 0),
            0
        );
        assert_eq!(pe_host_sample_rate(std::ptr::null()), 0.0);
        assert!(!pe_is_bypassed(std::ptr::null()));
        assert_eq!(pe_active_voices(std::ptr::null()), 0);
        assert_eq!(pe_max_frames(std::ptr::null()), 0);
        // Never null, so a caller can print it without checking.
        assert!(CStr::from_ptr(pe_last_error(std::ptr::null()))
            .to_bytes()
            .is_empty());
        // A null render target is refused rather than written through.
        let engine = pe_create(48_000.0, 128);
        pe_render(engine, std::ptr::null_mut(), std::ptr::null_mut(), 128);
        pe_render(engine, std::ptr::null_mut(), std::ptr::null_mut(), 0);
        pe_destroy(engine);
    }
}

/// Rates and block sizes a host cannot really mean are refused at construction,
/// which is the one place a plugin is allowed to say no.
#[test]
fn absurd_arguments_are_refused_rather_than_allocated_for() {
    for &(rate, frames) in &[
        (0.0, 128),
        (-48_000.0, 128),
        (f64::NAN, 128),
        (f64::INFINITY, 128),
        (1.0, 128),
        (1.0e9, 128),
        (48_000.0, 0),
        (48_000.0, u32::MAX),
    ] {
        let engine = pe_create(rate, frames);
        assert!(engine.is_null(), "{rate} Hz / {frames} frames was accepted");
    }
    // ... and the ones a host really does mean are not.
    for &(rate, frames) in &[
        (44_100.0, 1u32),
        (48_000.0, 32),
        (96_000.0, 4096),
        (192_000.0, 65536),
    ] {
        unsafe {
            let engine = pe_create(rate, frames);
            assert!(!engine.is_null(), "{rate} Hz / {frames} frames was refused");
            assert_eq!(pe_max_frames(engine), frames);
            pe_destroy(engine);
        }
    }
}

/// The default preset is a playable instrument before anything is loaded: a
/// host that never calls `pe_load_preset_toml` still gets a piano, which is
/// what makes the AUv3's first instantiation sound.
#[test]
fn a_fresh_engine_plays_the_built_in_preset() {
    unsafe {
        let engine = pe_create(48_000.0, 256);
        assert_eq!(pe_active_voices(engine), 0);
        let (l, r) = render(engine, 256);
        assert!(l.iter().chain(&r).all(|&v| v == 0.0), "idle is not silent");
        pe_event(engine, note_on(60, 90));
        let (l, r) = render(engine, 256);
        // One strike wakes more than one voice: every undamped string that can
        // answer it through the bridge is running too.
        assert!(pe_active_voices(engine) > 0);
        assert!(l.iter().any(|&v| v != 0.0) && r.iter().any(|&v| v != 0.0));
        assert!(last_error(engine).is_empty());
        pe_destroy(engine);
    }
}

/// A preset arrives as bytes and is parsed *and validated* on the main thread
/// (`DECISIONS.md` 52). All three ways it can be wrong are refused with the
/// instrument still playable, and the message names the field.
#[test]
fn a_bad_preset_is_refused_with_the_old_instrument_still_playing() {
    let good = std::fs::read_to_string("../presets/default.toml").expect("the shipped preset");
    // Parses as TOML, fails `Preset::validate`: A0 cannot have a negative
    // fundamental.
    let invalid = good.replacen("f0_hz = [\n    27.5,", "f0_hz = [\n    -27.5,", 1);
    assert_ne!(invalid, good, "the fixture edit did not apply");

    unsafe {
        let engine = pe_create(48_000.0, 256);

        assert_eq!(
            pe_load_preset_toml(engine, good.as_ptr() as *const i8, good.len()),
            pe_status::PE_OK
        );
        assert!(last_error(engine).is_empty());

        let cases: [(&str, pe_status, &str); 3] = [
            ("not a preset at all", pe_status::PE_ERR_PRESET, ""),
            (&invalid, pe_status::PE_ERR_PRESET, "f0_hz"),
            ("", pe_status::PE_ERR_PRESET, ""),
        ];
        for (text, expected, mentions) in cases {
            assert_eq!(
                pe_load_preset_toml(engine, text.as_ptr() as *const i8, text.len()),
                expected
            );
            let message = last_error(engine);
            assert!(!message.is_empty(), "a refusal with no message");
            assert!(
                message.contains(mentions),
                "{message:?} does not name {mentions:?}"
            );
        }

        // Not UTF-8 at all.
        let bytes = [0xffu8, 0xfe, 0x00, 0x01];
        assert_eq!(
            pe_load_preset_toml(engine, bytes.as_ptr() as *const i8, bytes.len()),
            pe_status::PE_ERR_UTF8
        );
        assert_eq!(
            pe_load_preset_toml(engine, std::ptr::null(), 0),
            pe_status::PE_ERR_NULL
        );

        // ... and after all of that the instrument the last *good* load built is
        // still there and still sounds.
        pe_event(engine, note_on(60, 90));
        let (l, _r) = render(engine, 512);
        assert!(l.iter().any(|&v| v != 0.0), "the engine went silent");
        pe_destroy(engine);
    }
}

/// `pe_save_state` is the AUv3's `fullState`: ask for the size, allocate, ask
/// again, and what comes back reloads into a different engine and renders the
/// same audio.
#[test]
fn state_is_the_preset_text_and_it_round_trips() {
    unsafe {
        let engine = pe_create(48_000.0, 256);
        let needed = pe_save_state(engine, std::ptr::null_mut(), 0);
        assert!(needed > 1000, "a preset is bigger than {needed} bytes");
        // Too small a buffer asks again rather than truncating.
        let mut small = vec![0u8; needed - 1];
        assert_eq!(
            pe_save_state(engine, small.as_mut_ptr(), small.len()),
            needed
        );
        assert!(
            small.iter().all(|&b| b == 0),
            "a short buffer was written to"
        );

        let mut state = vec![0u8; needed];
        assert_eq!(
            pe_save_state(engine, state.as_mut_ptr(), state.len()),
            needed
        );
        let text = String::from_utf8(state).expect("state is UTF-8 TOML");
        assert!(
            text.contains("name ="),
            "state is not a preset: {:?}",
            &text[..40]
        );

        // The other half of the round trip: a fresh engine, the saved bytes, the
        // same events, the same samples.
        let restored = pe_create(48_000.0, 256);
        assert_eq!(
            pe_load_preset_toml(restored, text.as_ptr() as *const i8, text.len()),
            pe_status::PE_OK
        );
        for e in [engine, restored] {
            pe_event(e, note_on(55, 88));
        }
        let (a_l, a_r) = render(engine, 4096);
        let (b_l, b_r) = render(restored, 4096);
        for (i, ((x, y), (p, q))) in a_l.iter().zip(&a_r).zip(b_l.iter().zip(&b_r)).enumerate() {
            assert_eq!(x.to_bits(), p.to_bits(), "left differs at {i}");
            assert_eq!(y.to_bits(), q.to_bits(), "right differs at {i}");
        }
        pe_destroy(engine);
        pe_destroy(restored);
    }
}

/// Loading a preset stops everything sounding — it is a new instrument, not a
/// modified one — and leaves the engine playable.
#[test]
fn loading_a_preset_replaces_the_instrument() {
    unsafe {
        let engine = pe_create(48_000.0, 256);
        pe_event(engine, note_on(60, 100));
        render(engine, 256);
        assert!(pe_active_voices(engine) > 0);
        let good = std::fs::read_to_string("../presets/default.toml").expect("the shipped preset");
        assert_eq!(
            pe_load_preset_toml(engine, good.as_ptr() as *const i8, good.len()),
            pe_status::PE_OK
        );
        assert_eq!(
            pe_active_voices(engine),
            0,
            "the old note survived the load"
        );
        let (l, _) = render(engine, 256);
        assert!(l.iter().all(|&v| v == 0.0));
        pe_event(engine, note_on(60, 100));
        let (l, _) = render(engine, 256);
        assert!(l.iter().any(|&v| v != 0.0), "the new instrument is mute");
        pe_destroy(engine);
    }
}

/// `pe_reset` is the transport-stop button: everything down, everything silent,
/// immediately, and playable again straight after.
#[test]
fn reset_silences_the_instrument_without_rebuilding_it() {
    unsafe {
        let engine = pe_create(44_100.0, 256);
        pe_event(
            engine,
            pe_event_t {
                kind: pe_event_kind::PE_EVENT_SUSTAIN as u32,
                key: 0,
                vel: 0,
                value: 1.0,
            },
        );
        for key in [48, 55, 60, 64] {
            pe_event(engine, note_on(key, 110));
        }
        render(engine, 4096);
        assert!(pe_active_voices(engine) > 0);
        pe_reset(engine);
        let (l, r) = render(engine, 4096);
        assert_eq!(pe_active_voices(engine), 0);
        assert!(
            l.iter().chain(&r).all(|&v| v == 0.0),
            "reset left {} nonzero samples",
            l.iter().chain(&r).filter(|&&v| v != 0.0).count()
        );
        pe_event(engine, note_on(60, 90));
        let (l, _) = render(engine, 4096);
        assert!(l.iter().any(|&v| v != 0.0), "the engine did not come back");
        pe_destroy(engine);
    }
}

/// Every way a C caller can hand us a value we cannot use, and what happens: a
/// key off the keyboard and an unknown kind are ignored, a velocity past the
/// engine's range is clamped rather than wrapped, and a pedal position that is
/// not a number is dropped rather than reaching a damper coefficient.
#[test]
fn nonsense_from_c_is_ignored_clamped_or_dropped_but_never_played() {
    unsafe {
        let engine = pe_create(48_000.0, 256);
        for event in [
            note_on(20, 90),        // below A0
            note_on(109, 90),       // above C8
            note_on(1_000_000, 90), // not a MIDI note at all
            pe_event_t {
                kind: 0xdead_beef,
                key: 60,
                vel: 90,
                value: 0.0,
            },
            pe_event_t {
                kind: pe_event_kind::PE_EVENT_SUSTAIN as u32,
                key: 0,
                vel: 0,
                value: f32::NAN,
            },
        ] {
            pe_event(engine, event);
            // The queue takes anything the engine can *represent* — a key
            // number that is a valid `u8` but not one of the 88 goes through
            // and is dropped at the voice lookup, exactly as a MIDI file's
            // out-of-compass note is (`midi.rs::playable`).
            pe_post_event(engine, event);
        }
        let (l, r) = render(engine, 4096);
        assert_eq!(pe_active_voices(engine), 0);
        assert!(
            l.iter().chain(&r).all(|&v| v == 0.0),
            "something out of range made a sound"
        );

        // A velocity past the top is clamped rather than dropped: whatever
        // else `65535` might be, it is not *silent*.
        let loud = pe_create(48_000.0, 256);
        pe_event(engine, note_on(60, PE_MAX_VELOCITY));
        pe_event(loud, note_on(60, 1_000_000));
        let (a, _) = render(engine, 4096);
        let (b, _) = render(loud, 4096);
        assert!(a.iter().any(|&v| v != 0.0));
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "clamping differs at frame {i}");
        }

        // The two lanes of the velocity field (`PE_MAX_VELOCITY`) are one
        // velocity: a MIDI 1.0 host sending 90 and a MIDI 2.0 host sending the
        // same note at full resolution render the same samples.
        let legacy = pe_create(48_000.0, 256);
        let fine = pe_create(48_000.0, 256);
        pe_event(legacy, note_on(60, 90));
        pe_event(fine, note_on(60, 90 * 512));
        let (c, _) = render(legacy, 4096);
        let (d, _) = render(fine, 4096);
        assert!(c.iter().any(|&v| v != 0.0));
        for (i, (x, y)) in c.iter().zip(&d).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "the two lanes differ at frame {i}"
            );
        }
        pe_destroy(legacy);
        pe_destroy(fine);
        pe_destroy(loud);
        pe_destroy(engine);
    }
}

/// The queue is bounded and says so. A UI thread that runs away cannot make the
/// audio thread allocate, and cannot silently lose events either — `false` is
/// the caller's problem to handle.
#[test]
fn the_event_queue_is_bounded_and_reports_its_limit() {
    unsafe {
        let engine = pe_create(48_000.0, 128);
        let mut accepted = 0;
        for _ in 0..5000 {
            if pe_post_event(engine, note_on(60, 1)) {
                accepted += 1;
            } else {
                break;
            }
        }
        assert!(
            (1000..=1024).contains(&accepted),
            "the queue took {accepted} events before refusing"
        );
        // Draining it makes room again.
        render(engine, 128);
        assert!(pe_post_event(engine, note_on(60, 1)));
        pe_destroy(engine);
    }
}
