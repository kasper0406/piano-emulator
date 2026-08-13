//! Decoding of the recordings the estimators read — WAV and FLAC, the two
//! lossless containers the datasets in `TUNING.md` ship in — plus the
//! resampling hook that brings anything not already at the engine's 48 kHz
//! onto the engine's clock.

use std::path::Path;

use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

use crate::error::{Error, Result};

/// Decoded audio, de-interleaved: one `Vec<f32>` per channel, all the same
/// length, samples normalised to ±1.
#[derive(Clone, Debug)]
pub struct Audio {
    pub sample_rate: u32,
    pub channels: Vec<Vec<f32>>,
}

impl Audio {
    pub fn new(sample_rate: u32, channels: Vec<Vec<f32>>) -> Result<Self> {
        if sample_rate == 0 {
            return Err(Error::Unsupported("sample rate of zero".into()));
        }
        if channels.is_empty() {
            return Err(Error::Unsupported("no channels".into()));
        }
        let len = channels[0].len();
        if channels.iter().any(|c| c.len() != len) {
            return Err(Error::Unsupported("channels of unequal length".into()));
        }
        Ok(Self { sample_rate, channels })
    }

    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    pub fn frames(&self) -> usize {
        self.channels[0].len()
    }

    pub fn duration_s(&self) -> f64 {
        self.frames() as f64 / f64::from(self.sample_rate)
    }

    /// Mean of the channels. The right reduction for partial tracking: a
    /// piano note is one physical source, so its partials are coherent across
    /// a stereo pair and averaging raises them relative to the (largely
    /// decorrelated) room and noise.
    pub fn mono(&self) -> Vec<f32> {
        if self.channels.len() == 1 {
            return self.channels[0].clone();
        }
        let scale = 1.0 / self.channels.len() as f32;
        let mut out = vec![0.0f32; self.frames()];
        for channel in &self.channels {
            for (o, &x) in out.iter_mut().zip(channel.iter()) {
                *o += x;
            }
        }
        for o in out.iter_mut() {
            *o *= scale;
        }
        out
    }

    /// Band-limited resampling to `target_hz`. A no-op clone when the rates
    /// already agree, so callers can apply it unconditionally.
    pub fn resampled(&self, target_hz: u32) -> Result<Audio> {
        if target_hz == self.sample_rate {
            return Ok(self.clone());
        }
        let channels = resample(&self.channels, self.sample_rate, target_hz)?;
        Audio::new(target_hz, channels)
    }

    /// Write the buffer back out as 32-bit float WAV. Used for fixtures and
    /// for eyeballing what the loader actually produced.
    pub fn write_wav(&self, path: impl AsRef<Path>) -> Result<()> {
        let spec = hound::WavSpec {
            channels: self.channel_count() as u16,
            sample_rate: self.sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut writer = hound::WavWriter::create(path, spec)?;
        for frame in 0..self.frames() {
            for channel in &self.channels {
                writer.write_sample(channel[frame])?;
            }
        }
        writer.finalize()?;
        Ok(())
    }
}

/// Decode a file, dispatching on its extension.
pub fn load(path: impl AsRef<Path>) -> Result<Audio> {
    let path = path.as_ref();
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    match extension.as_str() {
        "wav" | "wave" => load_wav(path),
        "flac" => load_flac(path),
        other => Err(Error::Unsupported(format!(
            "{}: unknown audio extension {:?}",
            path.display(),
            other
        ))),
    }
}

/// Decode a file and bring it to `target_hz`. This is the hook `TUNING.md`
/// asks for: MAESTRO is 44.1 kHz and the engine is 48 kHz, and every
/// measurement downstream is quoted on the engine's clock.
pub fn load_at(path: impl AsRef<Path>, target_hz: u32) -> Result<Audio> {
    load(path)?.resampled(target_hz)
}

pub fn load_wav(path: impl AsRef<Path>) -> Result<Audio> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let count = usize::from(spec.channels);
    if count == 0 {
        return Err(Error::Unsupported("wav with no channels".into()));
    }
    let mut channels = vec![Vec::with_capacity(reader.len() as usize / count); count];
    match spec.sample_format {
        hound::SampleFormat::Float => {
            for (i, sample) in reader.samples::<f32>().enumerate() {
                channels[i % count].push(sample?);
            }
        }
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1i64 << (spec.bits_per_sample - 1)) as f32;
            for (i, sample) in reader.samples::<i32>().enumerate() {
                channels[i % count].push(sample? as f32 * scale);
            }
        }
    }
    truncate_to_whole_frames(&mut channels);
    Audio::new(spec.sample_rate, channels)
}

pub fn load_flac(path: impl AsRef<Path>) -> Result<Audio> {
    let mut reader = claxon::FlacReader::open(path)?;
    let info = reader.streaminfo();
    let count = info.channels as usize;
    if count == 0 {
        return Err(Error::Unsupported("flac with no channels".into()));
    }
    if info.bits_per_sample == 0 || info.bits_per_sample > 32 {
        return Err(Error::Unsupported(format!(
            "flac with {} bits per sample",
            info.bits_per_sample
        )));
    }
    let scale = 1.0 / (1i64 << (info.bits_per_sample - 1)) as f32;
    let mut channels = vec![Vec::new(); count];
    for (i, sample) in reader.samples().enumerate() {
        channels[i % count].push(sample? as f32 * scale);
    }
    truncate_to_whole_frames(&mut channels);
    Audio::new(info.sample_rate, channels)
}

/// A file truncated mid-frame would otherwise leave channels of unequal
/// length, which `Audio::new` rejects; drop the partial frame instead.
fn truncate_to_whole_frames(channels: &mut [Vec<f32>]) {
    let shortest = channels.iter().map(|c| c.len()).min().unwrap_or(0);
    for channel in channels.iter_mut() {
        channel.truncate(shortest);
    }
}

/// Sinc resampling of de-interleaved channels.
///
/// The filter is long (256 taps at 256× oversampling, Blackman-Harris) because
/// this runs offline and the whole point of the exercise is that high partials
/// and decay tails survive intact.
///
/// The output is time-aligned with the input — output frame `j` is input time
/// `j * to_hz / from_hz`, with the filter reading implicit zeros before the
/// start of the signal. `SincFixedIn` compensates its own group delay
/// internally, so nothing is trimmed from the front; the tail is flushed with
/// silence until the output reaches the length the rate change implies.
/// `resampling_preserves_the_position_of_a_transient` pins that alignment,
/// because it is a property of the resampler rather than of this function.
pub fn resample(channels: &[Vec<f32>], from_hz: u32, to_hz: u32) -> Result<Vec<Vec<f32>>> {
    if from_hz == 0 || to_hz == 0 {
        return Err(Error::Resample("sample rate of zero".into()));
    }
    if from_hz == to_hz {
        return Ok(channels.to_vec());
    }
    let count = channels.len();
    if count == 0 {
        return Err(Error::Resample("no channels".into()));
    }
    let frames = channels[0].len();
    let ratio = f64::from(to_hz) / f64::from(from_hz);

    let parameters = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        oversampling_factor: 256,
        interpolation: SincInterpolationType::Cubic,
        window: WindowFunction::BlackmanHarris2,
    };
    const CHUNK: usize = 4096;
    let mut resampler = SincFixedIn::<f64>::new(ratio, 1.0, parameters, CHUNK, count)?;

    let input: Vec<Vec<f64>> = channels
        .iter()
        .map(|c| c.iter().map(|&x| f64::from(x)).collect())
        .collect();
    let target = (frames as f64 * ratio).round() as usize;
    let mut out: Vec<Vec<f64>> = vec![Vec::with_capacity(target); count];

    let mut position = 0;
    while position < frames {
        let wanted = resampler.input_frames_next();
        let produced = if frames - position >= wanted {
            let slices: Vec<&[f64]> = input.iter().map(|c| &c[position..position + wanted]).collect();
            position += wanted;
            resampler.process(&slices, None)?
        } else {
            let slices: Vec<&[f64]> = input.iter().map(|c| &c[position..]).collect();
            position = frames;
            resampler.process_partial(Some(&slices), None)?
        };
        append(&mut out, produced);
    }
    // Flush the filter's delay line with silence until the tail of the signal
    // has come out the far end.
    while out[0].len() < target {
        let produced = resampler.process_partial::<Vec<f64>>(None, None)?;
        if produced[0].is_empty() {
            break;
        }
        append(&mut out, produced);
    }

    Ok(out
        .into_iter()
        .map(|channel| channel.into_iter().take(target).map(|x| x as f32).collect())
        .collect())
}

fn append(out: &mut [Vec<f64>], produced: Vec<Vec<f64>>) {
    for (channel, block) in out.iter_mut().zip(produced) {
        channel.extend_from_slice(&block);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(freq: f64, sample_rate: f64, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                (0.5 * (2.0 * std::f64::consts::PI * freq * i as f64 / sample_rate).sin()) as f32
            })
            .collect()
    }

    #[test]
    fn mono_averages_the_channels() {
        let audio = Audio::new(48_000, vec![vec![1.0, -1.0], vec![0.0, 1.0]]).unwrap();
        assert_eq!(audio.mono(), vec![0.5, 0.0]);
        assert_eq!(audio.frames(), 2);
        assert_eq!(audio.channel_count(), 2);
    }

    #[test]
    fn unequal_channels_are_rejected() {
        assert!(Audio::new(48_000, vec![vec![0.0; 4], vec![0.0; 3]]).is_err());
        assert!(Audio::new(0, vec![vec![0.0; 4]]).is_err());
        assert!(Audio::new(48_000, vec![]).is_err());
    }

    #[test]
    fn a_written_wav_reads_back_unchanged() {
        let dir = std::env::temp_dir().join("piano-tuner-wav-roundtrip");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tone.wav");
        let audio = Audio::new(48_000, vec![tone(440.0, 48_000.0, 512), tone(660.0, 48_000.0, 512)])
            .unwrap();
        audio.write_wav(&path).unwrap();
        let back = load(&path).unwrap();
        assert_eq!(back.sample_rate, 48_000);
        assert_eq!(back.channel_count(), 2);
        assert_eq!(back.channels, audio.channels);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn integer_wavs_are_normalised_to_unit_scale() {
        let dir = std::env::temp_dir().join("piano-tuner-wav-int");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("int16.wav");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 44_100,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&path, spec).unwrap();
        for sample in [0i16, 16_384, -16_384, 32_767] {
            writer.write_sample(sample).unwrap();
        }
        writer.finalize().unwrap();
        let audio = load(&path).unwrap();
        assert_eq!(audio.sample_rate, 44_100);
        assert!((audio.channels[0][1] - 0.5).abs() < 1e-6);
        assert!((audio.channels[0][2] + 0.5).abs() < 1e-6);
        assert!((audio.channels[0][3] - 1.0).abs() < 1e-4);
        std::fs::remove_file(&path).ok();
    }

    /// Least-squares fit of `a cos(wt) + b sin(wt)` to `x[lo..hi]`, and the
    /// power of what is left over. The span must cover a whole number of
    /// cycles for the two basis functions to be orthogonal.
    fn fit_sinusoid(x: &[f32], freq: f64, sample_rate: f64, lo: usize, hi: usize) -> (f64, f64, f64) {
        let (mut re, mut im) = (0.0f64, 0.0f64);
        let omega = 2.0 * std::f64::consts::PI * freq / sample_rate;
        for (i, &sample) in x[lo..hi].iter().enumerate() {
            let phase = omega * (lo + i) as f64;
            re += f64::from(sample) * phase.cos();
            im += f64::from(sample) * phase.sin();
        }
        let n = (hi - lo) as f64;
        let (a, b) = (2.0 * re / n, 2.0 * im / n);
        let mut residual = 0.0f64;
        for (i, &sample) in x[lo..hi].iter().enumerate() {
            let phase = omega * (lo + i) as f64;
            let model = a * phase.cos() + b * phase.sin();
            residual += (f64::from(sample) - model).powi(2);
        }
        // `a cos + b sin` is `amplitude * sin(wt + psi)`; the source tone is
        // `amplitude * sin(wt)`, so `-psi/omega` is the delay in samples.
        let amplitude = (a * a + b * b).sqrt();
        let delay = -a.atan2(b) / omega;
        (amplitude, delay, residual / n)
    }

    #[test]
    fn resampling_preserves_a_tone_and_the_signals_duration() {
        // A pure tone is the worst case for a resampler in the sense that
        // every artefact it produces (aliases, images, passband ripple) is
        // isolated in the residual after the tone itself is fitted away.
        for &freq in &[100.0f64, 1_000.0, 15_000.0] {
            let source = vec![tone(freq, 44_100.0, 44_100)];
            let out = resample(&source, 44_100, 48_000).unwrap();
            assert_eq!(out[0].len(), 48_000);

            // Stay clear of both ends, where the filter is reading the zeros
            // outside the signal, and cover whole cycles.
            let period = 48_000.0 / freq;
            let lo = 2_400;
            let span = ((43_200.0 / period).floor() * period).round() as usize;
            let (amplitude, delay, residual) = fit_sinusoid(&out[0], freq, 48_000.0, lo, lo + span);

            assert!((amplitude - 0.5).abs() < 5e-4, "{freq} Hz: amplitude {amplitude}");
            let snr_db = 10.0 * ((amplitude * amplitude / 2.0) / residual).log10();
            assert!(snr_db > 120.0, "{freq} Hz: residual SNR {snr_db:.1} dB");
            // rubato leaves a constant sub-sample offset (0.084 frames at this
            // ratio, the same at every frequency — a pure delay, not
            // dispersion). Nothing the tuner measures can see 1.8 us, but a
            // change of alignment at the sample level would matter, so the
            // bound is tight enough to catch one.
            assert!(delay.abs() < 0.2, "{freq} Hz: delay {delay} frames");
        }
    }

    #[test]
    fn resampling_preserves_the_position_of_a_transient() {
        // Output frame j must be input time j / ratio. rubato's SincFixedIn
        // compensates its own group delay; this pins that, because if it ever
        // stops doing so every measurement the tuner makes shifts in time.
        let mut impulse = vec![0.0f32; 20_000];
        impulse[8_000] = 1.0;
        let out = resample(&[impulse], 44_100, 48_000).unwrap();
        let ratio = 48_000.0 / 44_100.0;
        let (peak, _) = out[0]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            .unwrap();
        let expected = 8_000.0 * ratio;
        assert!((peak as f64 - expected).abs() < 1.5, "peak at {peak}, expected {expected:.1}");
    }

    #[test]
    fn resampling_to_the_same_rate_is_a_copy() {
        let audio = Audio::new(48_000, vec![tone(440.0, 48_000.0, 64)]).unwrap();
        assert_eq!(audio.resampled(48_000).unwrap().channels, audio.channels);
    }
}
