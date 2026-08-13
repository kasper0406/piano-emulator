//! Synthetic test signals: sums of exponentially decaying sinusoids with known
//! frequencies, amplitudes and decay rates, optionally buried in white noise.
//!
//! This is the ground truth every estimator in the crate is unit-tested
//! against. It is deliberately the *model* the estimators assume rather than
//! anything the engine renders — a test that passes here says "the estimator
//! inverts its own model correctly", which is the only thing a unit test can
//! honestly say. `TUNING.md`'s self-calibration gate, which runs the pipeline
//! over the engine's own output, is what tests the assumption itself.

use crate::trajectory::{InharmonicModel, PartialTrack, TrackPoint};

/// One exponentially decaying sinusoid: `a exp(-sigma t) sin(2 pi f t + phase)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Partial {
    /// Partial index, 1-based — carried so tests can match a rendered partial
    /// to a recovered track without re-deriving `k` from the frequency.
    pub k: u32,
    pub frequency_hz: f64,
    /// Amplitude at the onset.
    pub amplitude: f64,
    /// Decay rate in 1/s, the engine's `sigma_k`.
    pub sigma: f64,
    pub phase: f64,
}

impl Partial {
    pub fn new(k: u32, frequency_hz: f64, amplitude: f64, sigma: f64) -> Self {
        Self {
            k,
            frequency_hz,
            amplitude,
            sigma,
            phase: 0.0,
        }
    }

    pub fn with_phase(mut self, phase: f64) -> Self {
        self.phase = phase;
        self
    }

    /// True envelope `t` seconds after the onset.
    pub fn amplitude_at(&self, t: f64) -> f64 {
        if t < 0.0 {
            0.0
        } else {
            self.amplitude * (-self.sigma * t).exp()
        }
    }

    pub fn t60(&self) -> f64 {
        6.907_755 / self.sigma
    }
}

/// A synthetic note: a set of partials struck at `onset_s`.
#[derive(Clone, Debug)]
pub struct Tone {
    pub sample_rate: f64,
    pub duration_s: f64,
    pub onset_s: f64,
    pub partials: Vec<Partial>,
}

impl Tone {
    pub fn new(sample_rate: f64, duration_s: f64, partials: Vec<Partial>) -> Self {
        Self {
            sample_rate,
            duration_s,
            onset_s: 0.0,
            partials,
        }
    }

    pub fn with_onset(mut self, onset_s: f64) -> Self {
        self.onset_s = onset_s;
        self
    }

    /// A stiff string: partials laid out by `model`, amplitudes falling as
    /// `1/k`, and the engine's frequency-dependent damping law
    /// `sigma_k = sigma0 + sigma1 (f_k / 1000)^2`.
    pub fn from_model(
        model: InharmonicModel,
        count: u32,
        sigma0: f64,
        sigma1: f64,
        sample_rate: f64,
        duration_s: f64,
    ) -> Self {
        let partials = (1..=count)
            .map(|k| {
                let f = model.partial(k);
                let khz = f / 1000.0;
                // Irrational phase steps so the partials never line up into a
                // spuriously peaky waveform.
                Partial::new(k, f, 1.0 / f64::from(k), sigma0 + sigma1 * khz * khz)
                    .with_phase(f64::from(k) * 0.618_034 * std::f64::consts::PI)
            })
            .collect();
        Self::new(sample_rate, duration_s, partials)
    }

    pub fn partial(&self, k: u32) -> Option<&Partial> {
        self.partials.iter().find(|p| p.k == k)
    }

    pub fn frames(&self) -> usize {
        (self.duration_s * self.sample_rate).round() as usize
    }

    pub fn render(&self) -> Vec<f32> {
        (0..self.frames())
            .map(|i| {
                let t = i as f64 / self.sample_rate - self.onset_s;
                if t < 0.0 {
                    return 0.0;
                }
                let sum: f64 = self
                    .partials
                    .iter()
                    .map(|p| {
                        p.amplitude_at(t)
                            * (2.0 * std::f64::consts::PI * p.frequency_hz * t + p.phase).sin()
                    })
                    .sum();
                sum as f32
            })
            .collect()
    }

    /// Render with additive white Gaussian noise at `snr_db`, measured as the
    /// ratio of the whole signal's RMS to the noise RMS. `seed` makes the
    /// noise reproducible, so a test that fails does so every time.
    pub fn render_with_noise(&self, snr_db: f64, seed: u64) -> Vec<f32> {
        let mut signal = self.render();
        let rms = rms(&signal);
        let level = rms * 10f64.powf(-snr_db / 20.0);
        add_white_noise(&mut signal, level, seed);
        signal
    }
}

/// The trajectory a tracker would measure for a group of components that share
/// one analysis window: the magnitude of their complex sum, sampled every
/// `hop_s`.
///
/// This is how a unison group reaches the estimators. Two strings a fraction of
/// a hertz apart are one peak in any window long enough to resolve the note at
/// all, and what varies is that peak's amplitude — so the beat lives in the
/// trajectory rather than in a second track. Building it analytically here,
/// instead of rendering the sum and tracking it, keeps the beat estimator's
/// tests about the estimator: the envelope is exactly the one the model says a
/// beating pair has.
pub fn beating_track(
    k: u32,
    components: &[Partial],
    onset_s: f64,
    hop_s: f64,
    duration_s: f64,
) -> PartialTrack {
    let count = (duration_s / hop_s).floor() as usize + 1;
    let points = (0..count)
        .map(|i| {
            let t = i as f64 * hop_s;
            let (mut re, mut im, mut weight, mut frequency) = (0.0, 0.0, 0.0, 0.0);
            for partial in components {
                let a = partial.amplitude_at(t);
                let phase = 2.0 * std::f64::consts::PI * partial.frequency_hz * t + partial.phase;
                re += a * phase.cos();
                im += a * phase.sin();
                weight += a;
                frequency += a * partial.frequency_hz;
            }
            TrackPoint {
                time_s: onset_s + t,
                // The amplitude-weighted mean of the components' frequencies:
                // where a peak picker would land while they are unresolved.
                frequency_hz: if weight > 0.0 { frequency / weight } else { 0.0 },
                amplitude: (re * re + im * im).sqrt(),
            }
        })
        .collect();
    PartialTrack { k, points }
}

pub fn rms(signal: &[f32]) -> f64 {
    if signal.is_empty() {
        return 0.0;
    }
    (signal
        .iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        / signal.len() as f64)
        .sqrt()
}

/// Add zero-mean Gaussian noise of the given RMS level.
pub fn add_white_noise(signal: &mut [f32], level: f64, seed: u64) {
    let mut rng = SplitMix64::new(seed);
    for sample in signal.iter_mut() {
        *sample += (level * rng.normal()) as f32;
    }
}

/// SplitMix64 — a few lines, statistically fine for test noise, and it keeps
/// the crate free of an RNG dependency.
pub struct SplitMix64 {
    state: u64,
    spare: Option<f64>,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare: None,
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform on (0, 1].
    pub fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) + 1) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal, by Box-Muller; the second variate is kept for the
    /// next call rather than thrown away.
    pub fn normal(&mut self) -> f64 {
        if let Some(spare) = self.spare.take() {
            return spare;
        }
        let (u, v) = (self.next_f64(), self.next_f64());
        let radius = (-2.0 * u.ln()).sqrt();
        let angle = 2.0 * std::f64::consts::PI * v;
        self.spare = Some(radius * angle.sin());
        radius * angle.cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_rendered_partial_follows_its_envelope() {
        let tone = Tone::new(48_000.0, 1.0, vec![Partial::new(1, 1000.0, 0.5, 2.0)]);
        let signal = tone.render();
        // The peak of a 1 kHz sinusoid inside a 10 ms slice is its envelope.
        for &t in &[0.05f64, 0.4, 0.9] {
            let start = (t * 48_000.0) as usize;
            let peak = signal[start..start + 480]
                .iter()
                .fold(0.0f32, |m, &x| m.max(x.abs()));
            let expected = 0.5 * (-2.0 * t).exp();
            assert!(
                (f64::from(peak) - expected).abs() < 0.02 * expected,
                "t={t}: {peak} vs {expected}"
            );
        }
    }

    #[test]
    fn silence_before_the_onset() {
        let tone = Tone::new(48_000.0, 0.5, vec![Partial::new(1, 440.0, 1.0, 1.0)]).with_onset(0.1);
        let signal = tone.render();
        assert!(signal[..4_800].iter().all(|&x| x == 0.0));
        assert!(signal[4_800..].iter().any(|&x| x.abs() > 0.5));
    }

    #[test]
    fn noise_lands_at_the_requested_snr() {
        let tone = Tone::new(48_000.0, 1.0, vec![Partial::new(1, 1000.0, 1.0, 1.0)]);
        let clean = tone.render();
        let noisy = tone.render_with_noise(40.0, 7);
        let noise: Vec<f32> = noisy.iter().zip(&clean).map(|(&a, &b)| a - b).collect();
        let snr = 20.0 * (rms(&clean) / rms(&noise)).log10();
        assert!((snr - 40.0).abs() < 0.3, "{snr} dB");
    }

    #[test]
    fn the_generator_is_deterministic() {
        let tone = Tone::new(48_000.0, 0.1, vec![Partial::new(1, 1000.0, 1.0, 1.0)]);
        assert_eq!(tone.render_with_noise(40.0, 3), tone.render_with_noise(40.0, 3));
        assert_ne!(tone.render_with_noise(40.0, 3), tone.render_with_noise(40.0, 4));
    }
}
