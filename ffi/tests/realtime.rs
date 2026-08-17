//! The audio thread's half of the contract, checked rather than asserted in
//! prose: `pe_render` and `pe_event` allocate nothing.
//!
//! The engine has been allocation-free on the audio path since it was written;
//! what is new here is the boundary, which has a resampler with internal
//! buffers and an output remainder, and a `catch_unwind` around everything.
//! Each of those is a place a heap allocation could appear without anyone
//! noticing — until a Logic session at a 32-frame buffer starts clicking on
//! whatever the allocator was doing at the time.
//!
//! A counting global allocator is the only way to be sure. There is exactly one
//! test in this file because the counter is process-wide and any other test
//! running beside it would be counted too.

use piano_emulator_ffi::*;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

static ARMED: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

struct Counting;

// SAFETY: every method forwards to `System` unchanged; the counters are the
// only addition and they do not allocate.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if ARMED.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if ARMED.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

#[test]
fn the_audio_path_never_allocates() {
    // Every rate: the bypass, a downsampling host, an upsampling host, and one
    // whose block size is not the one the engine works in.
    for &(rate, max_frames) in &[
        (48_000.0, 128u32),
        (44_100.0, 512),
        (96_000.0, 64),
        (88_200.0, 480),
    ] {
        unsafe {
            let engine = pe_create(rate, max_frames);
            assert!(!engine.is_null());
            let mut l = vec![0.0f32; 4096];
            let mut r = vec![0.0f32; 4096];
            // Warm up: the first render on a thread sets the FPCR and the first
            // events walk into cold code. Anything lazy happens here.
            pe_event(
                engine,
                pe_event_t {
                    kind: pe_event_kind::PE_EVENT_NOTE_ON as u32,
                    key: 60,
                    vel: 90,
                    value: 0.0,
                },
            );
            pe_render(engine, l.as_mut_ptr(), r.as_mut_ptr(), max_frames);

            ALLOCATIONS.store(0, Ordering::Relaxed);
            ARMED.store(true, Ordering::Relaxed);
            for i in 0..200u32 {
                // Held notes, released notes, a moving pedal and a silent
                // press: every branch of `handle_event` that a session touches.
                let event = |kind: pe_event_kind, key: u32, vel: u32, value: f32| pe_event_t {
                    kind: kind as u32,
                    key,
                    vel,
                    value,
                };
                pe_event(
                    engine,
                    event(
                        pe_event_kind::PE_EVENT_NOTE_ON,
                        40 + i % 48,
                        1 + i % 126,
                        0.0,
                    ),
                );
                pe_event(
                    engine,
                    event(pe_event_kind::PE_EVENT_NOTE_OFF, 40 + i % 48, 64, 0.0),
                );
                pe_event(
                    engine,
                    event(
                        pe_event_kind::PE_EVENT_SUSTAIN,
                        0,
                        0,
                        (i % 100) as f32 / 100.0,
                    ),
                );
                pe_event(engine, event(pe_event_kind::PE_EVENT_KEY_DOWN, 30, 0, 0.0));
                pe_post_event(engine, event(pe_event_kind::PE_EVENT_UNA_CORDA, 0, 0, 1.0));
                // A host is allowed to vary its block size call to call.
                let frames = match i % 4 {
                    0 => max_frames,
                    1 => 1,
                    2 => max_frames / 2 + 1,
                    _ => max_frames * 3,
                };
                pe_render(
                    engine,
                    l.as_mut_ptr(),
                    r.as_mut_ptr(),
                    frames.clamp(1, 4096),
                );
            }
            ARMED.store(false, Ordering::Relaxed);
            let count = ALLOCATIONS.load(Ordering::Relaxed);
            pe_destroy(engine);
            assert_eq!(
                count, 0,
                "{rate} Hz / {max_frames} frames: the audio path allocated {count} times"
            );
        }
    }
}
