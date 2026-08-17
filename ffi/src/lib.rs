//! The C ABI the plugin and the standalone app are built on.
//!
//! This crate is the only new Rust the hosts need (`DISTRIBUTION.md`
//! §Architecture). It owns three things and no more:
//!
//! 1. **The C ABI** below, generated into `include/piano_emulator.h` by
//!    `cbindgen` (`./generate-header.sh`) so the Swift side cannot drift from
//!    it. Every entry point states its thread in the header, because that
//!    contract is the whole reason for writing this by hand rather than
//!    exporting a flat surface.
//! 2. **The host-rate boundary** (`resample`), bypassed bit-exactly at
//!    48 kHz.
//! 3. **The two Rust-specific hazards at an FFI boundary**: a panic must not
//!    unwind into C (every entry point is wrapped in `catch_unwind`, and the
//!    shipped library is built with `--profile dist`, which is `panic =
//!    "abort"`), and `Preset::validate` (`DECISIONS.md` 52) must run on the
//!    main thread before the audio thread ever sees a coefficient — which it
//!    does, inside `pe_load_preset_toml`, which is a main-thread call.
//!
//! ## The thread contract, in one place
//!
//! The handle is not `Sync` and nothing here locks. There are exactly two
//! threads in the contract:
//!
//! - **The audio thread** owns the engine and the resampler: `pe_render` and
//!   `pe_event` and nothing else. Both are allocation-free, lock-free and
//!   syscall-free.
//! - **The main thread** owns construction, the preset and the state blob:
//!   `pe_create`, `pe_destroy`, `pe_reset`, `pe_load_preset_toml`,
//!   `pe_save_state`, `pe_last_error`. These allocate, and they must not
//!   run while a render is in flight — in an AUv3 that means outside the
//!   window between `allocateRenderResources` and `deallocateRenderResources`,
//!   which is exactly where a host makes those calls anyway.
//! - `pe_post_event` is the one crossing: a single producer on *any one*
//!   thread (the standalone app's CoreMIDI callback, or its UI) hands events to
//!   the audio thread through the engine's pre-allocated SPSC queue. It touches
//!   no state the audio thread owns.
//!
//! The plugin path does not use the queue at all: a host delivers MIDI to an
//! AUv3 on the audio thread, in the render block's event list, so it calls
//! `pe_event` directly. Both entry points already existed in the engine —
//! `EventSender` and `handle_event` are separate on purpose.

pub mod resample;

use piano_emulator::engine::{Engine, EventSender};
use piano_emulator::preset::Preset;
use piano_emulator::types::enable_flush_to_zero;
use piano_emulator::{Event, PedalEvent, BLOCK};
use resample::Boundary;
use std::cell::{Cell, UnsafeCell};
use std::ffi::{c_char, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};

/// Version of this ABI. Bumped whenever a signature, a struct layout or the
/// meaning of a field changes; a host that loads the library dynamically should
/// compare it against `pe_abi_version` before calling anything else.
pub const PE_ABI_VERSION: u32 = 2;

/// The engine's own sample rate, in Hz. A host running at exactly this rate
/// gets the resampler bypassed and is sample-for-sample what the offline
/// renderer produces.
pub const PE_ENGINE_SAMPLE_RATE: u32 = 48_000;

/// Largest velocity the engine takes. Anything above it is clamped.
///
/// **The field has two lanes**, and they are the engine's own
/// (`piano_emulator::velocity_from_midi`), unchanged by the crossing:
///
/// - `0 ..= 255` is a **MIDI 1.0 velocity**, the 7-bit number a keyboard sends
///   (the lane is a byte wide so that no value the engine's own field could
///   hold before it widened changed meaning). 0 is the silent press — the
///   damper lifts and nothing is struck; anything over 127 is clamped to 127.
/// - `256 ..= 65535` is a **high-resolution velocity** in 1/512 of a MIDI step,
///   which is where MIDI 2.0's 16 bits go (`SHIPPING.md` §4: the SL88 MK2 sends
///   them). A 7-bit velocity `v` is exactly `v * 512` here, so both spellings
///   of the same note play the same note.
///
/// A host with a 7-bit source should use the first lane and can ignore the
/// second entirely. The ABI carried this field in 32 bits from version 1 so
/// that widening it would not move any struct offset; version 2 is the widening
/// itself, and it only added meaning to values that used to be clamped.
pub const PE_MAX_VELOCITY: u32 = 65_535;

/// Return codes. Zero is success and every failure is negative, so
/// `if (pe_load_preset_toml(...) != PE_OK)` is the whole of a caller's error
/// handling unless it wants to print `pe_last_error`.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum pe_status {
    PE_OK = 0,
    /// A required pointer was null.
    PE_ERR_NULL = -1,
    /// An argument was out of range: a sample rate that is not a positive
    /// finite number, a zero or absurd block size, a length with a null buffer.
    PE_ERR_INVALID_ARGUMENT = -2,
    /// The preset text is not valid UTF-8.
    PE_ERR_UTF8 = -3,
    /// The preset did not parse, or failed `Preset::validate`. The message —
    /// which names the offending field — is at `pe_last_error`.
    PE_ERR_PRESET = -4,
    /// The destination buffer is too small; the call that returns a size tells
    /// you how much it needs.
    PE_ERR_BUFFER_TOO_SMALL = -5,
    /// A panic was caught at the boundary. The library is still usable — no
    /// unwinding crossed into C — but whatever was being done did not happen,
    /// and this is a bug worth reporting.
    PE_ERR_PANIC = -6,
}

/// What a `pe_event_t` is. Values outside this list are ignored rather than
/// rejected, so a newer host cannot crash an older library.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum pe_event_kind {
    /// `key`, `vel`. Any nonzero velocity throws the hammer — velocity 1 is a
    /// genuine pianissimo note and is never reinterpreted as a silent press
    /// (`DECISIONS.md` 55). Velocity 0 means the same as `PE_EVENT_KEY_DOWN`.
    PE_EVENT_NOTE_ON = 0,
    /// `key`, `vel`, where `vel` is the **release** velocity: how fast the key
    /// returns sets how fast the damper lands and how loud the key-off thump
    /// is. A source with no measurement should send 64, not 0.
    PE_EVENT_NOTE_OFF = 1,
    /// `key`. The silent press: the damper lifts, nothing is struck.
    PE_EVENT_KEY_DOWN = 2,
    /// `value`, 0.0 up to 1.0 down, **continuous** — half-pedalling reaches the
    /// dampers as the fraction it was played at. Map CC 64 to `cc / 127.0`, and
    /// slew-limit it over ~15 ms if it came from a 7-bit controller
    /// (`SHIPPING.md` §3).
    PE_EVENT_SUSTAIN = 3,
    /// `value` != 0.0 is down. A lever that either catches the raised damper
    /// levers or does not; there is no half of it.
    PE_EVENT_SOSTENUTO = 4,
    /// `value` != 0.0 is down. Softens the hammer and drops one struck unison
    /// string.
    PE_EVENT_UNA_CORDA = 5,
    /// Everything up, everything silent, immediately. Not a reset of the
    /// instrument's clock — see `pe_reset`.
    PE_EVENT_ALL_OFF = 6,
}

/// One event, passed by value. 16 bytes, no padding, no pointers: it can be
/// built in a render callback, copied into a queue and sent over a wire.
///
/// Fields not named by the event's kind are ignored; zero them anyway.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[allow(non_camel_case_types)]
pub struct pe_event_t {
    /// One of `pe_event_kind`. An unrecognised value is ignored.
    pub kind: u32,
    /// MIDI note number. A0 is 21 and C8 is 108; anything outside the 88 keys
    /// is ignored rather than folded into the compass.
    pub key: u32,
    /// MIDI velocity in either of `PE_MAX_VELOCITY`'s two lanes; anything
    /// above it is clamped.
    pub vel: u32,
    /// Pedal position, 0.0..=1.0. Ignored by the key events.
    pub value: f32,
}

impl pe_event_t {
    /// The engine event this is, or `None` for a kind the engine has nothing to
    /// do with. Every value is range-checked here: this is untrusted input from
    /// C, and the engine's own `Event` is a set of invariants.
    fn to_event(self) -> Option<Event> {
        let key = u8::try_from(self.key).ok()?;
        let vel = u16::try_from(self.vel.min(PE_MAX_VELOCITY)).ok()?;
        let value = if self.value.is_finite() {
            self.value.clamp(0.0, 1.0)
        } else {
            return None;
        };
        Some(match self.kind {
            k if k == pe_event_kind::PE_EVENT_NOTE_ON as u32 => Event::NoteOn { key, vel },
            k if k == pe_event_kind::PE_EVENT_NOTE_OFF as u32 => Event::NoteOff { key, vel },
            k if k == pe_event_kind::PE_EVENT_KEY_DOWN as u32 => Event::KeyDown { key },
            k if k == pe_event_kind::PE_EVENT_SUSTAIN as u32 => {
                Event::Pedal(PedalEvent::Sustain(value))
            }
            k if k == pe_event_kind::PE_EVENT_SOSTENUTO as u32 => {
                Event::Pedal(PedalEvent::Sostenuto(value != 0.0))
            }
            k if k == pe_event_kind::PE_EVENT_UNA_CORDA as u32 => {
                Event::Pedal(PedalEvent::UnaCorda(value != 0.0))
            }
            k if k == pe_event_kind::PE_EVENT_ALL_OFF as u32 => Event::AllOff,
            _ => return None,
        })
    }
}

/// Everything the audio thread owns, in one place so that no other entry point
/// can reach it by accident.
struct Audio {
    engine: Engine,
    boundary: Boundary,
}

/// An engine, its boundary resampler and its preset. Opaque to C.
///
/// The fields are in `UnsafeCell`s and are handed out one at a time rather than
/// through a `&mut` to the whole struct, because `pe_post_event` runs on a
/// different thread from `pe_render` and the two must never alias: the audio
/// thread reaches `audio`, the producer reaches `sender`, and the main thread
/// reaches the rest.
#[allow(non_camel_case_types)]
pub struct pe_engine {
    audio: UnsafeCell<Audio>,
    sender: UnsafeCell<EventSender>,
    preset_toml: UnsafeCell<String>,
    last_error: UnsafeCell<CString>,
    host_sample_rate: f64,
    max_frames: u32,
}

thread_local! {
    /// ARM flush-to-zero is per-thread, so it has to be set on the thread that
    /// renders — which, in a plugin, is a thread the host made and we never see
    /// the start of. The first `pe_render` on a thread sets it.
    static FLUSH_TO_ZERO: Cell<bool> = const { Cell::new(false) };
}

/// Widest host block we will accept. Well past any real device; the point is
/// that a garbage value cannot ask for a gigabyte of scratch.
const MAX_BLOCK_FRAMES: u32 = 1 << 16;

/// Sample rates outside this are not a host, they are a mistake.
const MIN_HOST_RATE: f64 = 8_000.0;
const MAX_HOST_RATE: f64 = 768_000.0;

impl pe_engine {
    fn build(preset: Preset, toml: String, host_sample_rate: f64, max_frames: u32) -> Option<Self> {
        let boundary = Boundary::new(host_sample_rate, max_frames)?;
        let (engine, sender) = Engine::new(&preset);
        Some(pe_engine {
            audio: UnsafeCell::new(Audio { engine, boundary }),
            sender: UnsafeCell::new(sender),
            preset_toml: UnsafeCell::new(toml),
            last_error: UnsafeCell::new(CString::default()),
            host_sample_rate,
            max_frames,
        })
    }

    /// # Safety
    /// The caller must hold the audio-thread half of the contract.
    #[allow(clippy::mut_from_ref)]
    unsafe fn audio(&self) -> &mut Audio {
        &mut *self.audio.get()
    }

    /// # Safety
    /// The caller must be the single producer.
    #[allow(clippy::mut_from_ref)]
    unsafe fn sender(&self) -> &mut EventSender {
        &mut *self.sender.get()
    }

    /// # Safety
    /// Main thread only.
    unsafe fn set_error(&self, message: &str) {
        // A NUL inside the message would truncate it; nothing that reaches here
        // has one, and if it did the truncation is still a readable message.
        let sanitized: String = message.replace('\0', " ");
        *self.last_error.get() = CString::new(sanitized).unwrap_or_default();
    }
}

/// Runs `body`, turning a panic into a status rather than an unwind into C.
///
/// With `--profile dist` (`panic = "abort"`) no panic ever gets this far; this
/// is what makes the staticlib and the rlib safe to link into a binary that
/// unwinds, and what keeps the tests honest.
fn guard<T>(fallback: T, body: impl FnOnce() -> T) -> T {
    catch_unwind(AssertUnwindSafe(body)).unwrap_or(fallback)
}

/// Version of the ABI this library was built with; compare against
/// `PE_ABI_VERSION` from the header you compiled against.
///
/// **Thread: any.**
#[no_mangle]
pub extern "C" fn pe_abi_version() -> u32 {
    PE_ABI_VERSION
}

/// Builds an engine for a host running at `host_sample_rate` Hz and rendering
/// at most `max_frames` frames per call.
///
/// The engine starts on the built-in default preset — a complete, shippable
/// instrument — so a host that never calls `pe_load_preset_toml` still makes
/// a piano. `max_frames` sizes the resampler's chunk: a host that renders
/// exactly that many frames gets one resampler call per render. Rendering more
/// than `max_frames` in one call is still correct and still allocation-free.
///
/// At exactly 48000 Hz the resampler is not built at all and the host's buffers
/// go straight to the engine.
///
/// Returns null if the arguments are out of range, if the preset in the binary
/// is somehow unbuildable, or if the allocation fails. This is the only call
/// that allocates the instrument (88 voices, every coefficient), so it takes
/// hundreds of milliseconds and belongs nowhere near an audio thread.
///
/// **Thread: main.**
#[no_mangle]
pub extern "C" fn pe_create(host_sample_rate: f64, max_frames: u32) -> *mut pe_engine {
    guard(std::ptr::null_mut(), || {
        if !(MIN_HOST_RATE..=MAX_HOST_RATE).contains(&host_sample_rate)
            || max_frames == 0
            || max_frames > MAX_BLOCK_FRAMES
        {
            return std::ptr::null_mut();
        }
        let preset = Preset::default();
        let toml = preset.to_toml();
        match pe_engine::build(preset, toml, host_sample_rate, max_frames) {
            Some(engine) => Box::into_raw(Box::new(engine)),
            None => std::ptr::null_mut(),
        }
    })
}

/// Frees an engine. Null is accepted and does nothing. After this the pointer
/// is dangling: nothing else may be called with it, and no render may be in
/// flight.
///
/// **Thread: main.**
///
/// # Safety
/// `engine` must be a pointer from `pe_create` that has not been destroyed.
#[no_mangle]
pub unsafe extern "C" fn pe_destroy(engine: *mut pe_engine) {
    if engine.is_null() {
        return;
    }
    guard((), || drop(Box::from_raw(engine)));
}

/// Silences the instrument and clears the boundary filter's history: every
/// damper down, every string still, no partial resampler chunk left over. What
/// a host calls when the transport stops or the plugin is bypassed.
///
/// It advances the engine by one 128-frame block, which is what flushes the
/// remainder left by the last render; that block is silence and is discarded.
///
/// It deliberately does **not** reset the engine's frame counter, which seeds
/// the mechanism noise: two identical gestures a minute apart are not supposed
/// to make identical key noise, and a transport stop is not a new piano.
/// Rebuild the engine if you want that.
///
/// **Thread: main** (it must not race a render).
///
/// # Safety
/// `engine` must be a live pointer from `pe_create`.
#[no_mangle]
pub unsafe extern "C" fn pe_reset(engine: *mut pe_engine) {
    let Some(state) = engine.as_ref() else {
        return;
    };
    guard((), || {
        let audio = state.audio();
        audio.engine.handle_event(Event::AllOff);
        // `AllOff` silences everything the engine is *about to* render, but the
        // remainder FIFO of `DECISIONS.md` 47 still holds up to a block of audio
        // that was rendered before it — at a host rate that is not a multiple of
        // 128 there is nearly always some. One throwaway block flushes it: after
        // `AllOff` the block that replaces it is exact silence, so what is left
        // in the FIFO is silence too.
        let mut discard = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
        audio.engine.process(&mut discard.0, &mut discard.1);
        audio.boundary.reset();
    });
}

/// Loads a preset from TOML text, replacing the instrument.
///
/// `text` is `len` bytes of UTF-8; it need not be NUL-terminated. The preset is
/// parsed **and validated** here (`DECISIONS.md` 52) — a negative decay rate, a
/// zero hammer mass, a table of the wrong length are all refused with a message
/// naming the field, which `pe_last_error` returns — so the audio thread
/// never sees a coefficient that has not been checked.
///
/// On success the engine is rebuilt from scratch and everything sounding stops.
/// On failure the old instrument is untouched and still playable.
///
/// This allocates and takes hundreds of milliseconds. It also invalidates the
/// event queue, so no `pe_post_event` may be in flight.
///
/// **Thread: main.**
///
/// # Safety
/// `engine` must be live, and `text` must point to `len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn pe_load_preset_toml(
    engine: *mut pe_engine,
    text: *const c_char,
    len: usize,
) -> pe_status {
    let Some(state) = engine.as_ref() else {
        return pe_status::PE_ERR_NULL;
    };
    if text.is_null() {
        state.set_error("preset text is null");
        return pe_status::PE_ERR_NULL;
    }
    let bytes = std::slice::from_raw_parts(text as *const u8, len);
    guard(pe_status::PE_ERR_PANIC, || {
        let Ok(toml) = std::str::from_utf8(bytes) else {
            state.set_error("preset text is not valid UTF-8");
            return pe_status::PE_ERR_UTF8;
        };
        let preset = match Preset::from_toml(toml) {
            Ok(preset) => preset,
            Err(e) => {
                state.set_error(&e.to_string());
                return pe_status::PE_ERR_PRESET;
            }
        };
        let (new_engine, new_sender) = Engine::new(&preset);
        // Only now, with a validated preset and a built engine in hand, is
        // anything the audio thread reads touched.
        let audio = state.audio();
        audio.engine = new_engine;
        audio.boundary.reset();
        *state.sender.get() = new_sender;
        *state.preset_toml.get() = preset.to_toml();
        state.set_error("");
        pe_status::PE_OK
    })
}

/// The message from the last failed call on this engine, as a NUL-terminated
/// UTF-8 string, or an empty string if the last call succeeded. Never null.
///
/// The pointer is owned by the engine and is valid until the next call that can
/// fail, or until `pe_destroy`.
///
/// **Thread: main.**
///
/// # Safety
/// `engine` must be live.
#[no_mangle]
pub unsafe extern "C" fn pe_last_error(engine: *const pe_engine) -> *const c_char {
    match engine.as_ref() {
        Some(state) => (*state.last_error.get()).as_ptr(),
        None => c"".as_ptr(),
    }
}

/// Renders `frames` frames into two non-interleaved channel buffers.
///
/// `left` and `right` must each have room for `frames` floats. Any length is
/// accepted, including zero and lengths that are not a multiple of the engine's
/// 128-frame block: the stream is the same however it is cut up
/// (`DECISIONS.md` 47).
///
/// At 48 kHz this is `Engine::process` and nothing else — no copy, no filter,
/// bit-for-bit what the offline renderer writes. At any other rate the output
/// is pulled through the boundary sinc, which asks the engine for as many
/// frames as it needs.
///
/// Allocation-free, lock-free, syscall-free. The first call on a thread sets
/// that thread's ARM flush-to-zero bit, which is why it must be the audio
/// thread that calls it.
///
/// **Thread: audio.** Never call it concurrently with itself or with any
/// main-thread entry point.
///
/// # Safety
/// `engine` must be live, and both buffers must be writable for `frames`
/// floats.
#[no_mangle]
pub unsafe extern "C" fn pe_render(
    engine: *mut pe_engine,
    left: *mut f32,
    right: *mut f32,
    frames: u32,
) {
    let Some(state) = engine.as_ref() else {
        return;
    };
    if frames == 0 {
        return;
    }
    if left.is_null() || right.is_null() {
        return;
    }
    FLUSH_TO_ZERO.with(|set| {
        if !set.get() {
            enable_flush_to_zero();
            set.set(true);
        }
    });
    let n = frames as usize;
    let l = std::slice::from_raw_parts_mut(left, n);
    let r = std::slice::from_raw_parts_mut(right, n);
    guard((), || {
        let Audio { engine, boundary } = state.audio();
        boundary.render(
            &mut |l: &mut [f32], r: &mut [f32]| engine.process(l, r),
            l,
            r,
        );
    });
}

/// Applies an event immediately, at the start of the next block the engine
/// renders.
///
/// This is the plugin path: a host hands an AUv3 its MIDI on the audio thread,
/// in the render block's event list, so the events for a block are applied
/// before the block is rendered. Onsets therefore quantise to the engine's
/// 128-frame block, 2.7 ms (`DECISIONS.md` 55); sub-block placement is
/// `DISTRIBUTION.md` M2 and will not change this signature.
///
/// An unknown kind, a key outside the 88, a velocity or pedal position out of
/// range: all ignored or clamped, never a crash.
///
/// **Thread: audio.**
///
/// # Safety
/// `engine` must be live.
#[no_mangle]
pub unsafe extern "C" fn pe_event(engine: *mut pe_engine, event: pe_event_t) {
    let Some(state) = engine.as_ref() else {
        return;
    };
    let Some(event) = event.to_event() else {
        return;
    };
    guard((), || state.audio().engine.handle_event(event));
}

/// Queues an event for the audio thread through the engine's pre-allocated SPSC
/// queue. Returns false if the queue is full (1024 events) or the event is not
/// one the engine understands; never blocks, never allocates.
///
/// This is the standalone app's path: CoreMIDI callbacks and the UI post here,
/// and the audio thread drains the queue before every block it renders. A
/// plugin has no use for it — see `pe_event`.
///
/// **Thread: any one thread.** The queue is single-producer: two threads
/// posting at once is undefined. Serialise them, or give each its own engine.
///
/// # Safety
/// `engine` must be live.
#[no_mangle]
pub unsafe extern "C" fn pe_post_event(engine: *mut pe_engine, event: pe_event_t) -> bool {
    let Some(state) = engine.as_ref() else {
        return false;
    };
    let Some(event) = event.to_event() else {
        return false;
    };
    guard(false, || state.sender().send(event))
}

/// Writes the engine's state — the whole preset, as TOML — into `buf`, and
/// returns the number of bytes it needs.
///
/// Call it with a null `buf` (or too small a `cap`) to ask for the size; call
/// it again with room and it writes that many bytes, **without** a trailing
/// NUL. Returns 0 only if the engine pointer is null or a panic was caught.
///
/// This is what an AUv3's `fullState` should carry: the preset *text*, not a
/// reference to a file (`DISTRIBUTION.md` §Presets and state). Six kilobytes
/// gzipped in a project file is free, and a project saved today still opens
/// when the preset files have moved on. Restore it by handing the same bytes
/// back to `pe_load_preset_toml`.
///
/// **Thread: main.**
///
/// # Safety
/// `engine` must be live, and `buf` must be writable for `cap` bytes if it is
/// not null.
#[no_mangle]
pub unsafe extern "C" fn pe_save_state(engine: *mut pe_engine, buf: *mut u8, cap: usize) -> usize {
    let Some(state) = engine.as_ref() else {
        return 0;
    };
    guard(0, || {
        let toml = &*state.preset_toml.get();
        let needed = toml.len();
        if buf.is_null() || cap < needed {
            return needed;
        }
        std::ptr::copy_nonoverlapping(toml.as_ptr(), buf, needed);
        needed
    })
}

/// The host rate this engine was built for, as it was given to `pe_create`.
///
/// **Thread: any.**
///
/// # Safety
/// `engine` must be live.
#[no_mangle]
pub unsafe extern "C" fn pe_host_sample_rate(engine: *const pe_engine) -> f64 {
    engine.as_ref().map_or(0.0, |state| state.host_sample_rate)
}

/// Whether the boundary resampler is bypassed — true exactly when the host runs
/// at 48000 Hz, in which case the output is bit-identical to the engine's own.
///
/// **Thread: any.**
///
/// # Safety
/// `engine` must be live.
#[no_mangle]
pub unsafe extern "C" fn pe_is_bypassed(engine: *const pe_engine) -> bool {
    engine.as_ref().is_some_and(|state| {
        matches!(
            (*state.audio.get()).boundary,
            crate::resample::Boundary::Bypass
        )
    })
}

/// Voices that are not idle, 0..=88 — the struck notes *and* every string
/// still ringing sympathetically with them, which is why one note can report
/// seventeen. For a meter or a test; not needed to play the instrument.
///
/// **Thread: audio** (it reads the state the audio thread owns).
///
/// # Safety
/// `engine` must be live.
#[no_mangle]
pub unsafe extern "C" fn pe_active_voices(engine: *const pe_engine) -> u32 {
    engine.as_ref().map_or(0, |state| {
        (*state.audio.get()).engine.active_voices() as u32
    })
}

/// The block size this engine was built for, as it was given to `pe_create`.
///
/// **Thread: any.**
///
/// # Safety
/// `engine` must be live.
#[no_mangle]
pub unsafe extern "C" fn pe_max_frames(engine: *const pe_engine) -> u32 {
    engine.as_ref().map_or(0, |state| state.max_frames)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The guard exists to turn a panic into a status instead of an unwind
    /// across the C ABI, which is undefined behaviour. The shipped cdylib is
    /// built with `panic = "abort"` and never gets here; the staticlib and the
    /// rlib are linked into binaries that unwind, and they do.
    #[test]
    fn a_panic_becomes_a_status_rather_than_an_unwind() {
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let caught = guard(pe_status::PE_ERR_PANIC, || -> pe_status {
            panic!("a coefficient that should not have got this far")
        });
        let passed_through = guard(0usize, || 42usize);
        std::panic::set_hook(hook);
        assert_eq!(caught, pe_status::PE_ERR_PANIC);
        assert_eq!(passed_through, 42);
    }

    /// The range checks at the boundary, where C's word is not taken for
    /// anything.
    #[test]
    fn events_from_c_are_range_checked_before_they_become_engine_events() {
        let event = |kind: u32, key: u32, vel: u32, value: f32| pe_event_t {
            kind,
            key,
            vel,
            value,
        };
        let note_on = pe_event_kind::PE_EVENT_NOTE_ON as u32;
        assert_eq!(
            event(note_on, 60, 90, 0.0).to_event(),
            Some(Event::NoteOn { key: 60, vel: 90 })
        );
        // Wider than a MIDI note, and wider than the engine's velocity.
        assert_eq!(event(note_on, 300, 90, 0.0).to_event(), None);
        assert_eq!(
            event(note_on, 60, 1_000_000, 0.0).to_event(),
            Some(Event::NoteOn {
                key: 60,
                vel: PE_MAX_VELOCITY as u16
            })
        );
        // Both lanes cross unchanged: the ABI does not reinterpret a velocity,
        // it only refuses one the engine could not hold.
        assert_eq!(
            event(note_on, 60, 90 * 512, 0.0).to_event(),
            Some(Event::NoteOn {
                key: 60,
                vel: 90 * 512
            })
        );
        // Unknown kinds are ignored, not guessed at.
        assert_eq!(event(0xdead_beef, 60, 90, 0.0).to_event(), None);
        // A pedal position that is not a number never reaches a damper.
        let sustain = pe_event_kind::PE_EVENT_SUSTAIN as u32;
        assert_eq!(event(sustain, 0, 0, f32::NAN).to_event(), None);
        assert_eq!(
            event(sustain, 0, 0, 3.0).to_event(),
            Some(Event::Pedal(PedalEvent::Sustain(1.0)))
        );
        assert_eq!(
            event(sustain, 0, 0, -1.0).to_event(),
            Some(Event::Pedal(PedalEvent::Sustain(0.0)))
        );
    }
}
