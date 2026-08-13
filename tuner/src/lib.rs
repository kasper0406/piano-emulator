//! Offline analysis and parameter estimation for piano-emulator, as described
//! in `TUNING.md`.
//!
//! The chain is: decode a recording ([`audio`]), optionally resample it onto
//! the engine's 48 kHz clock, transform it ([`stft`]), and follow each partial
//! of the struck note across frames ([`tracker`]) to produce the
//! `[(k, f_k(t), a_k(t))]` trajectories ([`trajectory`]) that every estimator
//! reads. The [`estimate`] module then inverts one part of the instrument's
//! model per submodule — tuning and inharmonicity, decay rates and the
//! polarization split, unison detuning, strike position, the felt hammer — and
//! [`preset`] writes what they found into the file the engine plays.
//! [`pipeline`] wires all of that together for one note.
//!
//! Nothing here is real time and nothing here is shared with the engine crate:
//! the tuner analyses what the engine (or a real piano) produced, and the
//! preset file is the only interface between them.
//!
//! ```no_run
//! use piano_tuner::pipeline::{analyze_note, NoteConfig};
//! use piano_tuner::preset::{Preset, PresetBuilder};
//! use piano_tuner::{audio, InharmonicModel};
//!
//! let recording = audio::load_at("data/C4-v8.flac", 48_000)?;
//! let analysis = analyze_note(
//!     &recording.mono(),
//!     48_000.0,
//!     InharmonicModel::new(261.6, 4e-4),
//!     &NoteConfig::default(),
//! )?;
//! let preset = PresetBuilder::new(Preset::load("presets/default.toml")?)
//!     .note(analysis.estimate(60))
//!     .polarization(analysis.decays.polarization)
//!     .build()?;
//! preset.save("presets/measured.toml")?;
//! # Ok::<(), piano_tuner::Error>(())
//! ```

pub mod audio;
pub mod error;
pub mod estimate;
pub mod library;
pub mod numeric;
pub mod pipeline;
pub mod preset;
pub mod residual;
pub mod stft;
pub mod survey;
pub mod synth;
pub mod tracker;
pub mod trajectory;

pub use audio::Audio;
pub use library::{Sample, SampleLibrary};
pub use survey::{Survey, SurveyConfig};
pub use error::{Error, Result};
pub use estimate::decay::fit_decays;
pub use estimate::{
    estimate_unison, fit_hammer, fit_inharmonic, fit_strike_position, interpolate_keys,
    CompassCurve, DecayConfig, DecayReport, FitSpan, HammerConfig, HammerFit, InharmonicConfig,
    InharmonicFit, StrikeConfig, StrikeFit, UnisonConfig, UnisonEstimate,
};
pub use pipeline::{analyze_note, analyze_trajectories, NoteAnalysis, NoteConfig};
pub use preset::{NoteEstimate, Preset, PresetBuilder};
pub use stft::{find_peaks, Peak, Spectrogram, SpectrumFrame, Stft, StftConfig};
pub use tracker::{detect_onset, hann_decay_gain, PartialTracker, TrackerConfig};
pub use trajectory::{
    cents, InharmonicModel, NoteId, NoteTrajectories, PartialTrack, TrackPoint,
};

/// The engine's sample rate. Everything the tuner measures is quoted on this
/// clock, so recordings at any other rate are resampled on the way in.
pub const SAMPLE_RATE: u32 = 48_000;
