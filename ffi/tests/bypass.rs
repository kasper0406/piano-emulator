//! The null test: at 48 kHz the C ABI is `Engine::process` and nothing else.
//!
//! This is the test `DISTRIBUTION.md` M0 opens with, and the reason the whole
//! boundary is a branch rather than a ratio. Everything the instrument has ever
//! been measured on — `PHYSICS.md`'s acceptance numbers, `DECISIONS.md`'s
//! render hashes, the tuner's self-calibration gate — was measured on the
//! engine's own samples at 48 kHz. If a plugin at 48 kHz produced anything else,
//! every one of those numbers would quietly stop describing what a player hears.
//!
//! So: same preset, same events, same request lengths, compared as **bit
//! patterns** rather than as floats, because `-0.0 == 0.0` and `NaN != NaN` are
//! both wrong answers to "is this the same audio".

use piano_emulator::engine::Engine;
use piano_emulator::preset::Preset;
use piano_emulator::types::{Event, PedalEvent};
use piano_emulator_ffi::*;

/// One (call index, event) script, played identically through both paths.
/// It covers every event the ABI carries: a struck note, a silent press, a
/// continuous half-pedal, both switched pedals, a release velocity, and the
/// panic button.
fn script() -> Vec<(usize, pe_event_t, Event)> {
    let ffi = |kind: pe_event_kind, key: u32, vel: u32, value: f32| pe_event_t {
        kind: kind as u32,
        key,
        vel,
        value,
    };
    vec![
        (
            0,
            ffi(pe_event_kind::PE_EVENT_SUSTAIN, 0, 0, 1.0),
            Event::Pedal(PedalEvent::Sustain(1.0)),
        ),
        (
            0,
            ffi(pe_event_kind::PE_EVENT_NOTE_ON, 48, 96, 0.0),
            Event::NoteOn { key: 48, vel: 96 },
        ),
        (
            2,
            ffi(pe_event_kind::PE_EVENT_KEY_DOWN, 64, 0, 0.0),
            Event::KeyDown { key: 64 },
        ),
        (
            3,
            ffi(pe_event_kind::PE_EVENT_NOTE_ON, 72, 104, 0.0),
            Event::NoteOn { key: 72, vel: 104 },
        ),
        (
            5,
            ffi(pe_event_kind::PE_EVENT_SUSTAIN, 0, 0, 0.45),
            Event::Pedal(PedalEvent::Sustain(0.45)),
        ),
        (
            6,
            ffi(pe_event_kind::PE_EVENT_SOSTENUTO, 0, 0, 1.0),
            Event::Pedal(PedalEvent::Sostenuto(true)),
        ),
        (
            7,
            ffi(pe_event_kind::PE_EVENT_UNA_CORDA, 0, 0, 1.0),
            Event::Pedal(PedalEvent::UnaCorda(true)),
        ),
        (
            9,
            ffi(pe_event_kind::PE_EVENT_NOTE_OFF, 48, 112, 0.0),
            Event::NoteOff { key: 48, vel: 112 },
        ),
        (
            11,
            ffi(pe_event_kind::PE_EVENT_NOTE_ON, 84, 1, 0.0),
            Event::NoteOn { key: 84, vel: 1 },
        ),
    ]
}

/// Renders `calls` blocks of `chunk` frames through the engine directly.
fn direct(chunk: usize, calls: usize) -> (Vec<f32>, Vec<f32>) {
    let (mut engine, _tx) = Engine::new(&Preset::default());
    let (mut l, mut r) = (vec![0.0f32; chunk * calls], vec![0.0f32; chunk * calls]);
    for call in 0..calls {
        for (at, _, event) in script() {
            if at == call {
                engine.handle_event(event);
            }
        }
        let range = call * chunk..(call + 1) * chunk;
        engine.process(&mut l[range.clone()], &mut r[range]);
    }
    (l, r)
}

/// ... and the same thing through the C ABI at 48 kHz.
fn through_ffi(chunk: usize, calls: usize, queue: bool) -> (Vec<f32>, Vec<f32>) {
    let (mut l, mut r) = (vec![0.0f32; chunk * calls], vec![0.0f32; chunk * calls]);
    unsafe {
        let engine = pe_create(48_000.0, chunk as u32);
        assert!(!engine.is_null());
        assert!(pe_is_bypassed(engine), "48 kHz did not take the bypass");
        for call in 0..calls {
            for (at, event, _) in script() {
                if at == call {
                    if queue {
                        assert!(pe_post_event(engine, event));
                    } else {
                        pe_event(engine, event);
                    }
                }
            }
            let offset = call * chunk;
            pe_render(
                engine,
                l[offset..].as_mut_ptr(),
                r[offset..].as_mut_ptr(),
                chunk as u32,
            );
        }
        pe_destroy(engine);
    }
    (l, r)
}

fn assert_bit_identical(a: &(Vec<f32>, Vec<f32>), b: &(Vec<f32>, Vec<f32>), what: &str) {
    for (channel, (x, y)) in [(&a.0, &b.0), (&a.1, &b.1)].iter().enumerate() {
        assert_eq!(x.len(), y.len(), "{what}: length");
        for (i, (&p, &q)) in x.iter().zip(y.iter()).enumerate() {
            assert_eq!(
                p.to_bits(),
                q.to_bits(),
                "{what}: channel {channel} differs at frame {i} ({p:e} vs {q:e})"
            );
        }
    }
}

/// The headline: `pe_render` at 48 kHz is byte-identical to `Engine::process`,
/// at every request length — including the ones that are not a multiple of the
/// engine's 128-frame block, where the remainder FIFO of `DECISIONS.md` 47 is
/// what keeps the streams together.
#[test]
fn pe_render_at_48_khz_is_byte_identical_to_the_engine() {
    for &chunk in &[1, 64, 128, 100, 333, 512, 4096] {
        let calls = (48_000 / chunk).clamp(13, 400);
        let reference = direct(chunk, calls);
        assert!(
            reference.0.iter().any(|&v| v != 0.0),
            "the reference render is silent, so this proves nothing"
        );
        assert_bit_identical(
            &reference,
            &through_ffi(chunk, calls, false),
            &format!("{chunk}-frame calls"),
        );
    }
}

/// The same, with the events taking the SPSC queue instead of the audio
/// thread's direct call — the standalone app's path against the plugin's. The
/// queue is drained at the start of every block the engine renders, so as long
/// as an event is posted before the block that should hear it, the two paths
/// are the same audio and not merely similar audio.
#[test]
fn the_queue_and_the_direct_call_are_the_same_audio() {
    let reference = direct(128, 40);
    assert_bit_identical(&reference, &through_ffi(128, 40, true), "queued events");
}

/// A host is allowed to render more than it said it would. It costs a second
/// resampler chunk (nothing at 48 kHz, where there is no resampler), and it is
/// still the same stream.
#[test]
fn rendering_more_than_max_frames_is_still_the_same_stream() {
    let reference = direct(1024, 8);
    let mut l = vec![0.0f32; 8192];
    let mut r = vec![0.0f32; 8192];
    unsafe {
        let engine = pe_create(48_000.0, 128);
        for call in 0..8 {
            for (at, event, _) in script() {
                if at == call {
                    pe_event(engine, event);
                }
            }
            let offset = call * 1024;
            pe_render(
                engine,
                l[offset..].as_mut_ptr(),
                r[offset..].as_mut_ptr(),
                1024,
            );
        }
        pe_destroy(engine);
    }
    assert_bit_identical(&reference, &(l, r), "1024 frames against max_frames 128");
}

/// Every other rate builds the resampler, and 48 kHz is the only one that does
/// not. Exact equality, not a tolerance: 48000.5 Hz is a host we resample for,
/// because reinterpreting our samples at a rate that is nearly right is the one
/// thing `DECISIONS.md` 17 refused.
#[test]
fn only_exactly_48_khz_takes_the_bypass() {
    unsafe {
        for &(rate, bypassed) in &[
            (48_000.0, true),
            (48_000.5, false),
            (44_100.0, false),
            (88_200.0, false),
            (96_000.0, false),
            (192_000.0, false),
        ] {
            let engine = pe_create(rate, 512);
            assert!(!engine.is_null(), "{rate} Hz was refused");
            assert_eq!(pe_is_bypassed(engine), bypassed, "{rate} Hz");
            assert_eq!(pe_host_sample_rate(engine), rate);
            pe_destroy(engine);
        }
    }
}

/// A render at another rate is the same *music*: same length in seconds, same
/// notes, nothing silent and nothing clipped. The spectral half of this is in
/// `resampler.rs`; this is the "does it still play" half.
#[test]
fn the_resampled_path_produces_the_same_performance() {
    let seconds = 2.0;
    unsafe {
        for &rate in &[44_100.0, 96_000.0] {
            let frames = (seconds * rate) as usize;
            let mut l = vec![0.0f32; frames];
            let mut r = vec![0.0f32; frames];
            let engine = pe_create(rate, 512);
            pe_event(
                engine,
                pe_event_t {
                    kind: pe_event_kind::PE_EVENT_NOTE_ON as u32,
                    key: 60,
                    vel: 96,
                    value: 0.0,
                },
            );
            let mut done = 0;
            while done < frames {
                let n = 512.min(frames - done);
                pe_render(
                    engine,
                    l[done..].as_mut_ptr(),
                    r[done..].as_mut_ptr(),
                    n as u32,
                );
                done += n;
            }
            pe_destroy(engine);
            let peak = l.iter().chain(&r).fold(0.0f32, |m, &v| m.max(v.abs()));
            assert!(peak > 0.01, "{rate} Hz rendered near-silence: peak {peak}");
            assert!(peak <= 1.0, "{rate} Hz clipped: peak {peak}");
            assert!(l.iter().chain(&r).all(|v| v.is_finite()));
            // The onset is where the note-on was, not a filter's worth later.
            let onset = l.iter().position(|&v| v.abs() > 1.0e-4).unwrap_or(frames);
            assert!(
                onset < (0.01 * rate) as usize,
                "{rate} Hz: the note started {onset} frames in"
            );
        }
    }
}
