//! Short-time Fourier front end: Hann window, configurable hop, zero-padded
//! FFT, and parabolic interpolation of spectral peaks.
//!
//! Two conventions the rest of the crate depends on:
//!
//! * **Magnitudes are sinusoid amplitudes.** Each frame's magnitude spectrum is
//!   divided by half the window's coherent gain, so a sinusoid of amplitude `A`
//!   produces a peak of height `A`. (The peak of the windowed transform of
//!   `A cos(w n + p)` is `A/2 * sum(w)`; dividing by `sum(w)/2` recovers `A`.)
//! * **A frame is timestamped at the centre of its window.** The window is
//!   symmetric, so this is the instant its measurement actually describes —
//!   and it is what makes the decay compensation in `tracker` exact.

use std::sync::Arc;

use rustfft::{num_complex::Complex32, Fft, FftPlanner};

use crate::error::{Error, Result};

/// Geometry of the transform.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StftConfig {
    /// Window length in samples. `TUNING.md` asks for >= 2^16 on real
    /// recordings: a piano's partials sit a few Hz apart in the bass and the
    /// unison beats are slower still.
    pub window: usize,
    /// Advance between frames, in samples.
    pub hop: usize,
    /// Transform length; `>= window`, the remainder zero-padded. Padding does
    /// not add information but it samples the main lobe densely enough that
    /// the parabolic peak fit is a local fit rather than an extrapolation.
    pub fft_size: usize,
}

impl StftConfig {
    pub fn new(window: usize, hop: usize, fft_size: usize) -> Result<Self> {
        let config = Self { window, hop, fft_size };
        config.validate()?;
        Ok(config)
    }

    /// `window` samples advanced by `hop`, transformed at `pad` times the
    /// window length.
    pub fn padded(window: usize, hop: usize, pad: usize) -> Result<Self> {
        Self::new(window, hop, window.saturating_mul(pad))
    }

    pub fn validate(&self) -> Result<()> {
        if self.window < 4 {
            return Err(Error::Config(format!("window {} is too short", self.window)));
        }
        if self.window % 2 != 0 {
            return Err(Error::Config("window length must be even".into()));
        }
        if self.hop == 0 {
            return Err(Error::Config("hop must be at least one sample".into()));
        }
        if self.fft_size < self.window {
            return Err(Error::Config(format!(
                "fft size {} is smaller than the window {}",
                self.fft_size, self.window
            )));
        }
        Ok(())
    }

    pub fn window_s(&self, sample_rate: f64) -> f64 {
        self.window as f64 / sample_rate
    }

    pub fn hop_s(&self, sample_rate: f64) -> f64 {
        self.hop as f64 / sample_rate
    }

    /// Number of frames a signal of `frames` samples yields. Only windows that
    /// are fully covered by the signal are analysed: a partially filled window
    /// is a windowed measurement of a shorter signal, and its amplitude is
    /// wrong by an amount nothing downstream can undo.
    pub fn frame_count(&self, frames: usize) -> usize {
        if frames < self.window {
            0
        } else {
            (frames - self.window) / self.hop + 1
        }
    }
}

impl Default for StftConfig {
    /// 2^16 samples (1.37 s at 48 kHz) advanced by 480 (10 ms), transformed at
    /// 2^17 — `TUNING.md`'s partial-tracker settings.
    fn default() -> Self {
        Self {
            window: 1 << 16,
            hop: 480,
            fft_size: 1 << 17,
        }
    }
}

/// A planned transform. Holds the window and the FFT plan; analysis itself
/// allocates only the output.
pub struct Stft {
    config: StftConfig,
    window: Vec<f32>,
    /// `2 / sum(w)`: converts a transform magnitude into a sinusoid amplitude.
    amplitude_scale: f32,
    fft: Arc<dyn Fft<f32>>,
}

impl Stft {
    pub fn new(config: StftConfig) -> Result<Self> {
        config.validate()?;
        let window = hann(config.window);
        let sum: f64 = window.iter().map(|&w| f64::from(w)).sum();
        let fft = FftPlanner::new().plan_fft_forward(config.fft_size);
        Ok(Self {
            config,
            window,
            amplitude_scale: (2.0 / sum) as f32,
            fft,
        })
    }

    pub fn config(&self) -> StftConfig {
        self.config
    }

    /// Number of magnitude bins per frame: the non-negative half of the
    /// spectrum, DC and Nyquist included.
    pub fn bins(&self) -> usize {
        self.config.fft_size / 2 + 1
    }

    /// Stream the frames of `signal`, calling `frame(time_s, magnitudes)` for
    /// each. The magnitude slice is reused between calls, so the whole
    /// spectrogram of a long recording never has to exist at once — at the
    /// default settings it would be hundreds of megabytes.
    pub fn for_each_frame(
        &self,
        signal: &[f32],
        sample_rate: f64,
        mut frame: impl FnMut(f64, &[f32]),
    ) {
        let n = self.config.frame_count(signal.len());
        let mut buffer = vec![Complex32::new(0.0, 0.0); self.config.fft_size];
        let mut scratch = vec![Complex32::new(0.0, 0.0); self.fft.get_inplace_scratch_len()];
        let mut magnitude = vec![0.0f32; self.bins()];
        let centre_offset = 0.5 * self.config.window as f64;

        for i in 0..n {
            let start = i * self.config.hop;
            let block = &signal[start..start + self.config.window];
            for (slot, (&x, &w)) in buffer.iter_mut().zip(block.iter().zip(self.window.iter())) {
                *slot = Complex32::new(x * w, 0.0);
            }
            for slot in buffer[self.config.window..].iter_mut() {
                *slot = Complex32::new(0.0, 0.0);
            }
            self.fft.process_with_scratch(&mut buffer, &mut scratch);
            for (m, slot) in magnitude.iter_mut().zip(buffer.iter()) {
                *m = slot.norm() * self.amplitude_scale;
            }
            let time_s = (start as f64 + centre_offset) / sample_rate;
            frame(time_s, &magnitude);
        }
    }

    /// Collect every frame into a `Spectrogram`. Convenient for tests and for
    /// short signals; prefer [`Stft::for_each_frame`] on real recordings.
    pub fn analyze(&self, signal: &[f32], sample_rate: f64) -> Spectrogram {
        let mut frames = Vec::with_capacity(self.config.frame_count(signal.len()));
        self.for_each_frame(signal, sample_rate, |time_s, magnitude| {
            frames.push(SpectrumFrame {
                time_s,
                magnitude: magnitude.to_vec(),
            });
        });
        Spectrogram {
            sample_rate,
            config: self.config,
            frames,
        }
    }
}

/// One analysed frame: amplitude-calibrated magnitudes, timestamped at the
/// centre of its window.
#[derive(Clone, Debug)]
pub struct SpectrumFrame {
    pub time_s: f64,
    pub magnitude: Vec<f32>,
}

/// Every frame of a signal. Deliberately not serializable: a cached
/// spectrogram is larger than the audio it came from, and the trajectories are
/// what downstream code wants anyway.
#[derive(Clone, Debug)]
pub struct Spectrogram {
    pub sample_rate: f64,
    pub config: StftConfig,
    pub frames: Vec<SpectrumFrame>,
}

impl Spectrogram {
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Peaks of one frame, loudest-first ordering not guaranteed (they come out
    /// in ascending frequency).
    pub fn peaks(&self, frame: usize, floor_db: f64) -> Vec<Peak> {
        let mut out = Vec::new();
        find_peaks(
            &self.frames[frame].magnitude,
            self.sample_rate,
            self.config.fft_size,
            floor_db,
            &mut out,
        );
        out
    }
}

/// An interpolated spectral peak.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Peak {
    /// Interpolated frequency, in Hz.
    pub frequency_hz: f64,
    /// Interpolated amplitude, in the units of the input signal.
    pub amplitude: f64,
    /// The bin the peak was found in, before interpolation.
    pub bin: usize,
}

/// Locate every local maximum of `magnitude` at least `floor_db` below the
/// frame's loudest bin, refining each by fitting a parabola to the log
/// magnitudes of the peak bin and its two neighbours.
///
/// The log domain is the right one for this: near its top a Hann main lobe is
/// very nearly Gaussian, so its logarithm is very nearly the parabola being
/// fitted. Fitting the linear magnitudes instead biases the frequency by a
/// few hundredths of a bin and the amplitude low by a fraction of a dB.
pub fn find_peaks(
    magnitude: &[f32],
    sample_rate: f64,
    fft_size: usize,
    floor_db: f64,
    out: &mut Vec<Peak>,
) {
    out.clear();
    if magnitude.len() < 3 {
        return;
    }
    let max = magnitude.iter().fold(0.0f32, |m, &x| m.max(x));
    if max <= 0.0 || !max.is_finite() {
        return;
    }
    let floor = f64::from(max) * 10f64.powf(floor_db / 20.0);
    let bin_hz = sample_rate / fft_size as f64;

    for bin in 1..magnitude.len() - 1 {
        let (a, b, c) = (
            f64::from(magnitude[bin - 1]),
            f64::from(magnitude[bin]),
            f64::from(magnitude[bin + 1]),
        );
        // `>` on the left and `>=` on the right reports a flat top once.
        if b < floor || b <= a || b < c {
            continue;
        }
        let (offset, amplitude) = if a > 0.0 && c > 0.0 {
            let (la, lb, lc) = (a.ln(), b.ln(), c.ln());
            let curvature = la - 2.0 * lb + lc;
            if curvature < 0.0 {
                let delta = (0.5 * (la - lc) / curvature).clamp(-0.5, 0.5);
                (delta, (lb - 0.25 * (la - lc) * delta).exp())
            } else {
                (0.0, b)
            }
        } else {
            (0.0, b)
        };
        out.push(Peak {
            frequency_hz: (bin as f64 + offset) * bin_hz,
            amplitude,
            bin,
        });
    }
}

/// Periodic Hann window, `w[n] = 0.5 - 0.5 cos(2 pi n / N)`. Periodic rather
/// than symmetric because the DFT treats the frame as one period of a periodic
/// signal, and the periodic form is the one whose transform has no
/// discontinuity at the wrap.
pub fn hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let phase = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
            (0.5 - 0.5 * phase.cos()) as f32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(freq: f64, amplitude: f64, phase: f64, sample_rate: f64, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = i as f64 / sample_rate;
                (amplitude * (2.0 * std::f64::consts::PI * freq * t + phase).cos()) as f32
            })
            .collect()
    }

    #[test]
    fn the_periodic_hann_window_has_the_expected_coherent_gain() {
        let w = hann(1024);
        let sum: f64 = w.iter().map(|&x| f64::from(x)).sum();
        assert!((sum - 512.0).abs() < 1e-3, "{sum}");
    }

    #[test]
    fn config_rejects_geometry_it_cannot_transform() {
        assert!(StftConfig::new(1024, 256, 512).is_err());
        assert!(StftConfig::new(1024, 0, 2048).is_err());
        assert!(StftConfig::new(2, 1, 2).is_err());
        assert!(StftConfig::new(1024, 256, 2048).is_ok());
    }

    #[test]
    fn a_peak_reports_the_sinusoids_frequency_and_amplitude() {
        let sr = 48_000.0;
        // Deliberately between bins, and at an amplitude that is not 1.
        let (f, a) = (1234.567, 0.317);
        let stft = Stft::new(StftConfig::padded(1 << 14, 1 << 12, 2).unwrap()).unwrap();
        let signal = sine(f, a, 0.7, sr, 1 << 15);
        let spec = stft.analyze(&signal, sr);
        assert!(spec.len() >= 3);
        for frame in 0..spec.len() {
            let peaks = spec.peaks(frame, -60.0);
            let top = peaks
                .iter()
                .copied()
                .max_by(|x, y| x.amplitude.total_cmp(&y.amplitude))
                .unwrap();
            assert!(
                (top.frequency_hz - f).abs() < 0.01,
                "frame {frame}: {} Hz",
                top.frequency_hz
            );
            assert!(
                (top.amplitude - a).abs() < 0.005 * a,
                "frame {frame}: amplitude {}",
                top.amplitude
            );
        }
    }

    #[test]
    fn frames_are_timestamped_at_the_centre_of_their_window() {
        let stft = Stft::new(StftConfig::new(1024, 512, 1024).unwrap()).unwrap();
        let spec = stft.analyze(&vec![0.0f32; 4096], 48_000.0);
        assert_eq!(spec.len(), 7);
        assert!((spec.frames[0].time_s - 512.0 / 48_000.0).abs() < 1e-12);
        assert!((spec.frames[1].time_s - 1024.0 / 48_000.0).abs() < 1e-12);
    }

    #[test]
    fn two_partials_are_resolved_independently() {
        let sr = 48_000.0;
        let n = 1 << 15;
        let mut signal = sine(440.0, 0.5, 0.0, sr, n);
        for (s, x) in signal.iter_mut().zip(sine(881.3, 0.25, 1.1, sr, n)) {
            *s += x;
        }
        let stft = Stft::new(StftConfig::padded(1 << 14, 1 << 13, 2).unwrap()).unwrap();
        let spec = stft.analyze(&signal, sr);
        let peaks = spec.peaks(0, -60.0);
        let near = |f: f64| {
            peaks
                .iter()
                .copied()
                .filter(|p| (p.frequency_hz - f).abs() < 1.0)
                .max_by(|a, b| a.amplitude.total_cmp(&b.amplitude))
                .unwrap()
        };
        let low = near(440.0);
        let high = near(881.3);
        assert!((low.frequency_hz - 440.0).abs() < 0.01, "{low:?}");
        assert!((high.frequency_hz - 881.3).abs() < 0.01, "{high:?}");
        assert!((low.amplitude - 0.5).abs() < 0.005, "{low:?}");
        assert!((high.amplitude - 0.25).abs() < 0.005, "{high:?}");
    }
}
