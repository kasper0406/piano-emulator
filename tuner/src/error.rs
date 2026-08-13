//! One error type for the whole crate. The tuner is an offline tool: every
//! failure ends up in front of a human, so the variants carry the message they
//! want to print rather than a machine-readable code.

use std::fmt;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Wav(hound::Error),
    Flac(claxon::Error),
    Json(serde_json::Error),
    /// The resampler refused the ratio or the buffer geometry.
    Resample(String),
    /// A file extension or an encoding the loader does not handle.
    Unsupported(String),
    /// An analysis parameter that cannot describe a valid transform.
    Config(String),
    /// An estimator was handed data it cannot fit: too few partials, an
    /// envelope with no decay in it, a spectrum with no null. Not a bug and not
    /// a broken file — a recording that does not answer the question asked of
    /// it — so it carries the reason for the human reading the pipeline log.
    Estimate(String),
    /// A preset that would not survive the engine's own validation.
    Preset(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "io error: {e}"),
            Error::Wav(e) => write!(f, "wav error: {e}"),
            Error::Flac(e) => write!(f, "flac error: {e}"),
            Error::Json(e) => write!(f, "json error: {e}"),
            Error::Resample(m) => write!(f, "resampling failed: {m}"),
            Error::Unsupported(m) => write!(f, "unsupported: {m}"),
            Error::Config(m) => write!(f, "invalid configuration: {m}"),
            Error::Estimate(m) => write!(f, "estimation failed: {m}"),
            Error::Preset(m) => write!(f, "invalid preset: {m}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            Error::Wav(e) => Some(e),
            Error::Flac(e) => Some(e),
            Error::Json(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

impl From<hound::Error> for Error {
    fn from(e: hound::Error) -> Self {
        Error::Wav(e)
    }
}

impl From<claxon::Error> for Error {
    fn from(e: claxon::Error) -> Self {
        Error::Flac(e)
    }
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::Json(e)
    }
}

impl From<toml::de::Error> for Error {
    fn from(e: toml::de::Error) -> Self {
        Error::Preset(e.to_string())
    }
}

impl From<rubato::ResampleError> for Error {
    fn from(e: rubato::ResampleError) -> Self {
        Error::Resample(e.to_string())
    }
}

impl From<rubato::ResamplerConstructionError> for Error {
    fn from(e: rubato::ResamplerConstructionError) -> Self {
        Error::Resample(e.to_string())
    }
}
