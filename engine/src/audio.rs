//! Real-time output through cpal.
//!
//! The callback does nothing but deinterleave and call `Engine::process`: no
//! allocation, no locks, no syscalls, no logging. Everything it needs is moved
//! in when the stream is built.

use crate::engine::Engine;
use crate::types::{enable_flush_to_zero, BLOCK, SAMPLE_RATE};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};
use std::error::Error;
use std::fmt;

/// Frames requested from the device. A multiple of `BLOCK`, low enough for a
/// playable latency (~5 ms) and high enough to survive scheduling jitter.
const PREFERRED_BUFFER_FRAMES: u32 = 256;

#[derive(Debug)]
pub enum AudioError {
    NoOutputDevice,
    /// The device cannot run at `SAMPLE_RATE`; retuning the engine is not
    /// supported, so this is fatal rather than silently detuning the piano.
    UnsupportedSampleRate,
    /// Only f32 output is supported (native on CoreAudio).
    UnsupportedFormat(SampleFormat),
    Cpal(String),
}

impl fmt::Display for AudioError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AudioError::NoOutputDevice => write!(f, "no default audio output device"),
            AudioError::UnsupportedSampleRate => {
                write!(
                    f,
                    "output device does not support {} Hz",
                    SAMPLE_RATE as u32
                )
            }
            AudioError::UnsupportedFormat(fmt) => {
                write!(f, "unsupported output sample format {fmt:?}")
            }
            AudioError::Cpal(msg) => write!(f, "audio device error: {msg}"),
        }
    }
}

impl Error for AudioError {}

/// A running output stream. Dropping it stops audio.
pub struct AudioOutput {
    stream: cpal::Stream,
    device_name: String,
    channels: usize,
}

impl AudioOutput {
    /// Opens the default output device and starts feeding it from `engine`.
    pub fn start(mut engine: Engine) -> Result<AudioOutput, AudioError> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or(AudioError::NoOutputDevice)?;
        let device_name = device.name().unwrap_or_else(|_| "unknown".to_string());

        let supported = device
            .supported_output_configs()
            .map_err(|e| AudioError::Cpal(e.to_string()))?
            .filter(|c| c.channels() >= 2)
            .find(|c| {
                c.min_sample_rate().0 <= SAMPLE_RATE as u32
                    && c.max_sample_rate().0 >= SAMPLE_RATE as u32
            })
            .ok_or(AudioError::UnsupportedSampleRate)?
            .with_sample_rate(cpal::SampleRate(SAMPLE_RATE as u32));

        if supported.sample_format() != SampleFormat::F32 {
            return Err(AudioError::UnsupportedFormat(supported.sample_format()));
        }

        let channels = supported.channels() as usize;
        // The buffer-size range must come from the config the stream is
        // actually built with: another entry of the iterator may be a
        // different format or channel count with a different valid range.
        let buffer_size = *supported.buffer_size();
        let mut config: StreamConfig = supported.into();
        config.buffer_size = match buffer_size {
            cpal::SupportedBufferSize::Range { min, max } => {
                cpal::BufferSize::Fixed(PREFERRED_BUFFER_FRAMES.clamp(min, max))
            }
            cpal::SupportedBufferSize::Unknown => cpal::BufferSize::Default,
        };

        // Scratch deinterleave buffers, sized for the largest callback we will
        // accept in one pass; longer callbacks are handled in several passes.
        let mut left = vec![0.0f32; BLOCK * 16];
        let mut right = vec![0.0f32; BLOCK * 16];
        let mut fpcr_set = false;

        let stream = device
            .build_output_stream(
                &config,
                move |data: &mut [f32], _| {
                    if !fpcr_set {
                        enable_flush_to_zero();
                        fpcr_set = true;
                    }
                    for chunk in data.chunks_mut(left.len() * channels) {
                        let frames = chunk.len() / channels;
                        let (l, r) = (&mut left[..frames], &mut right[..frames]);
                        engine.process(l, r);
                        for (i, frame) in chunk.chunks_mut(channels).enumerate() {
                            frame[0] = l[i];
                            frame[1] = r[i];
                            frame[2..].fill(0.0);
                        }
                    }
                },
                move |err| {
                    // Runs on cpal's error thread, not the audio thread.
                    eprintln!("audio stream error: {err}");
                },
                None,
            )
            .map_err(|e| AudioError::Cpal(e.to_string()))?;

        stream.play().map_err(|e| AudioError::Cpal(e.to_string()))?;

        Ok(AudioOutput {
            stream,
            device_name,
            channels,
        })
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn sample_rate(&self) -> f32 {
        SAMPLE_RATE
    }

    pub fn pause(&self) -> Result<(), AudioError> {
        self.stream
            .pause()
            .map_err(|e| AudioError::Cpal(e.to_string()))
    }
}
