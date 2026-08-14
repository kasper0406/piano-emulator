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
//!                          |              + Burst: the key's own mechanism
//!                          |                noise, straight to the board
//!                          |
//!                          +-- ResonanceBus: sum of all strings, fed back
//!                          |                 into the undamped ones
//!                          |
//!                          +-- Burst x2: the sustain pedal's tray, centred
//!                          |
//!                          +-- Soundboard: pan, body, FDN, master -> stereo
//! ```
//!
//! Every tuning number the instrument uses comes from a [`Preset`], loaded from
//! a TOML file or built from [`Preset::default`]; the DSP modules hold no
//! parameter tables of their own. The one thing a preset cannot state is what
//! its own strike comes to at the output, which is what its `[noise]` levels are
//! quoted against — [`calibrate`] measures that when the engine is built, and
//! nothing measures anything after that.
//!
//! `lib.rs` owns the module graph and the public surface; the DSP modules are
//! edited independently and must not need changes here.

pub mod audio;
pub mod calibrate;
pub mod engine;
pub mod hammer;
pub mod midi;
pub mod modal;
pub mod noise;
pub mod pedal;
pub mod preset;
pub mod render;
pub mod repl;
pub mod resonance;
pub mod soundboard;
pub mod string;
pub mod types;
pub mod voice;

pub use engine::{Engine, EventSender};
pub use hammer::{Hammer, HammerParams};
pub use midi::MidiPerformance;
pub use modal::ModalBank;
pub use noise::Burst;
pub use pedal::PedalState;
pub use preset::Preset;
pub use render::{render_to_buffer, render_to_wav, RenderEvent};
pub use resonance::ResonanceBus;
pub use soundboard::Soundboard;
pub use string::{PianoString, StringParams};
pub use types::{
    index_to_note, key_index, note_to_freq, Event, PedalEvent, BLOCK, DEFAULT_RELEASE_VELOCITY,
    ESCAPEMENT_VELOCITY, HIGHEST_KEY, LOWEST_KEY, NUM_KEYS, SAMPLE_RATE,
};
pub use voice::Voice;
