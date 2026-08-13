//! Physical-model piano synthesizer.
//!
//! Signal flow, all of it block-based at [`types::BLOCK`] frames:
//!
//! ```text
//! REPL --(SPSC queue)--> Engine
//!                          |
//!                          +-- 88 Voice: Hammer pulse -> PianoString
//!                          |              (unison group x 2 polarizations
//!                          |               of ModalBank), damped per PedalState
//!                          |
//!                          +-- ResonanceBus: sum of all strings, fed back
//!                          |                 into the undamped ones
//!                          |
//!                          +-- Soundboard: pan, body, FDN, master -> stereo
//! ```
//!
//! `lib.rs` owns the module graph and the public surface; the DSP modules are
//! edited independently and must not need changes here.

pub mod audio;
pub mod engine;
pub mod hammer;
pub mod modal;
pub mod pedal;
pub mod render;
pub mod repl;
pub mod resonance;
pub mod soundboard;
pub mod string;
pub mod types;
pub mod voice;

pub use engine::{Engine, EventSender};
pub use hammer::{Hammer, HammerParams};
pub use modal::ModalBank;
pub use pedal::PedalState;
pub use render::{render_to_buffer, render_to_wav, RenderEvent};
pub use resonance::ResonanceBus;
pub use soundboard::Soundboard;
pub use string::{PianoString, StringParams};
pub use types::{
    index_to_note, key_index, note_to_freq, Event, PedalEvent, BLOCK, HIGHEST_KEY, LOWEST_KEY,
    NUM_KEYS, SAMPLE_RATE,
};
pub use voice::Voice;
