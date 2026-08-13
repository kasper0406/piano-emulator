//! Partial tracker: STFT → per-frame peak picking → association across frames,
//! seeded by the inharmonic model's predicted `f_k`.
//!
//! Seeding is what makes this tracker simple enough to trust. A general
//! sinusoidal tracker has to decide which peaks belong together; here the
//! caller already knows, to within a few cents, where every partial of the
//! recorded note should be, so association reduces to "which peak is nearest
//! the prediction, inside a window narrow enough that no other partial can
//! reach into it". The measured frequency is then free to disagree with the
//! seed — that disagreement is exactly what the `f0`/`B` estimator fits.

use crate::error::Result;
use crate::stft::{find_peaks, Peak, Stft, StftConfig};
use crate::trajectory::{cents, InharmonicModel, NoteTrajectories, PartialTrack, TrackPoint};

/// Settings for [`PartialTracker`].
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TrackerConfig {
    pub stft: StftConfig,
    /// Highest partial index to look for. The engine synthesizes at most 80.
    pub max_partials: u32,
    /// Partials above `frequency_limit * sample_rate` are not searched for.
    /// 0.45 matches the engine's own cap (`SPEC.md`, "String").
    pub frequency_limit: f64,
    /// Peak-picking floor, in dB below the loudest bin of the same frame. It
    /// is relative to the frame rather than to the recording so that it
    /// follows the note down as it decays.
    pub peak_floor_db: f64,
    /// Half-width of the association window around a prediction, in cents,
    /// before the spacing cap below is applied.
    pub tolerance_cents: f64,
    /// Hard cap on the association window as a fraction of the distance to
    /// the neighbouring predicted partials. Below 0.5 no two partials can
    /// compete for the same peak; high in the compass, where partial spacing
    /// falls below `tolerance_cents`, this is the binding constraint.
    pub spacing_fraction: f64,
    /// A candidate more than this far below the loudest candidate in the same
    /// window is discarded. Hann sidelobes are 31 dB down, so this rejects a
    /// sidelobe of the true peak — which sits closer to the prediction than
    /// the peak itself whenever the partial has drifted.
    pub sidelobe_reject_db: f64,
    /// A candidate more than this far below the loudest measurement so far on
    /// the same track is discarded: past that point the partial has decayed
    /// into the noise and what is left in the window is not it.
    pub track_range_db: f64,
    /// Consecutive frames without a candidate before a started track is
    /// closed. A partial that dips into a beat null and comes back should not
    /// become two tracks, so this is generous.
    pub max_gap_frames: usize,
    /// Tracks shorter than this are dropped as spurious.
    pub min_track_frames: usize,
    /// Undo the windowing bias on decaying partials (see
    /// [`hann_decay_gain`]). Wanted for anything that fits envelopes.
    pub decay_compensation: bool,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            stft: StftConfig::default(),
            max_partials: 80,
            frequency_limit: 0.45,
            peak_floor_db: -100.0,
            tolerance_cents: 60.0,
            spacing_fraction: 0.45,
            sidelobe_reject_db: 20.0,
            track_range_db: 80.0,
            max_gap_frames: 8,
            min_track_frames: 3,
            decay_compensation: true,
        }
    }
}

pub struct PartialTracker {
    config: TrackerConfig,
    stft: Stft,
}

impl PartialTracker {
    pub fn new(config: TrackerConfig) -> Result<Self> {
        let stft = Stft::new(config.stft)?;
        Ok(Self { config, stft })
    }

    pub fn config(&self) -> &TrackerConfig {
        &self.config
    }

    /// Extract one trajectory per partial of `seed` from `signal`.
    pub fn track(
        &self,
        signal: &[f32],
        sample_rate: f64,
        seed: InharmonicModel,
    ) -> NoteTrajectories {
        let stft = self.stft.config();
        let limit = self.config.frequency_limit * sample_rate;
        let k_max = seed.partials_below(limit, self.config.max_partials);

        let mut live: Vec<LiveTrack> = (1..=k_max)
            .map(|k| LiveTrack::new(k, seed.partial(k), self.window_half_width(&seed, k)))
            .collect();

        let mut peaks: Vec<Peak> = Vec::new();
        self.stft
            .for_each_frame(signal, sample_rate, |time_s, magnitude| {
                find_peaks(
                    magnitude,
                    sample_rate,
                    stft.fft_size,
                    self.config.peak_floor_db,
                    &mut peaks,
                );
                self.associate(&mut live, &peaks, time_s);
            });

        let window_s = stft.window_s(sample_rate);
        let mut tracks: Vec<PartialTrack> = live
            .into_iter()
            .filter(|t| t.points.len() >= self.config.min_track_frames)
            .map(|t| PartialTrack { k: t.k, points: t.points })
            .collect();
        if self.config.decay_compensation {
            for track in tracks.iter_mut() {
                compensate_decay(&mut track.points, window_s);
            }
        }

        NoteTrajectories {
            source: String::new(),
            note: None,
            sample_rate,
            window_s,
            hop_s: stft.hop_s(sample_rate),
            seed,
            onset_s: detect_onset(signal, sample_rate),
            tracks,
        }
    }

    /// Half-width of partial `k`'s association window, in Hz: the smaller of
    /// `tolerance_cents` and a fraction of the gap to the nearer neighbouring
    /// partial. Fixed from the seed rather than recomputed as the prediction
    /// drifts, so a track that starts to wander cannot widen its own window.
    fn window_half_width(&self, seed: &InharmonicModel, k: u32) -> f64 {
        let f = seed.partial(k);
        let by_cents = f * (2f64.powf(self.config.tolerance_cents / 1200.0) - 1.0);
        let below = if k > 1 {
            f - seed.partial(k - 1)
        } else {
            seed.partial(2) - f
        };
        let above = seed.partial(k + 1) - f;
        by_cents.min(below.min(above) * self.config.spacing_fraction)
    }

    /// Match this frame's peaks to the live tracks. Proposals from every track
    /// are pooled and taken in order of increasing frequency error, so a peak
    /// goes to the track that predicted it best and no peak is claimed twice.
    fn associate(&self, live: &mut [LiveTrack], peaks: &[Peak], time_s: f64) {
        let sidelobe = 10f64.powf(-self.config.sidelobe_reject_db / 20.0);
        let range = 10f64.powf(-self.config.track_range_db / 20.0);

        let mut proposals: Vec<Proposal> = Vec::new();
        for (index, track) in live.iter().enumerate() {
            if track.closed {
                continue;
            }
            let lo = track.predicted_hz - track.half_width_hz;
            let hi = track.predicted_hz + track.half_width_hz;
            let first = peaks.partition_point(|p| p.frequency_hz < lo);
            let window = &peaks[first..];
            let mut loudest = 0.0f64;
            for peak in window.iter().take_while(|p| p.frequency_hz <= hi) {
                loudest = loudest.max(peak.amplitude);
            }
            if loudest <= 0.0 {
                continue;
            }
            let threshold = (loudest * sidelobe).max(track.peak_amplitude * range);
            for (offset, peak) in window
                .iter()
                .enumerate()
                .take_while(|(_, p)| p.frequency_hz <= hi)
            {
                if peak.amplitude < threshold {
                    continue;
                }
                proposals.push(Proposal {
                    cost: cents(track.predicted_hz, peak.frequency_hz).abs(),
                    track: index,
                    peak: first + offset,
                });
            }
        }
        proposals.sort_by(|a, b| a.cost.total_cmp(&b.cost));

        let mut peak_taken = vec![false; peaks.len()];
        let mut matched = vec![false; live.len()];
        for proposal in proposals {
            if matched[proposal.track] || peak_taken[proposal.peak] {
                continue;
            }
            matched[proposal.track] = true;
            peak_taken[proposal.peak] = true;
            live[proposal.track].extend(&peaks[proposal.peak], time_s);
        }
        for (track, matched) in live.iter_mut().zip(matched) {
            if !matched {
                track.miss(self.config.max_gap_frames);
            }
        }
    }
}

struct Proposal {
    cost: f64,
    track: usize,
    peak: usize,
}

struct LiveTrack {
    k: u32,
    predicted_hz: f64,
    half_width_hz: f64,
    peak_amplitude: f64,
    misses: usize,
    closed: bool,
    points: Vec<TrackPoint>,
}

impl LiveTrack {
    fn new(k: u32, predicted_hz: f64, half_width_hz: f64) -> Self {
        Self {
            k,
            predicted_hz,
            half_width_hz,
            peak_amplitude: 0.0,
            misses: 0,
            closed: false,
            points: Vec::new(),
        }
    }

    fn extend(&mut self, peak: &Peak, time_s: f64) {
        self.points.push(TrackPoint {
            time_s,
            frequency_hz: peak.frequency_hz,
            amplitude: peak.amplitude,
        });
        // The prediction follows the measurement: a real string's partials
        // drift (tension recovery after the strike, temperature) by more than
        // the tracker's own precision over the length of a note.
        self.predicted_hz = peak.frequency_hz;
        self.peak_amplitude = self.peak_amplitude.max(peak.amplitude);
        self.misses = 0;
    }

    fn miss(&mut self, max_gap: usize) {
        // A track that has not started yet keeps waiting: the analysis window
        // may still be straddling the strike.
        if self.points.is_empty() {
            return;
        }
        self.misses += 1;
        if self.misses > max_gap {
            self.closed = true;
        }
    }
}

/// Ratio between what a Hann-windowed frame measures for an exponentially
/// decaying partial and the partial's true amplitude at the centre of that
/// window.
///
/// The measured amplitude is the window-weighted mean of the envelope,
/// `sum(w a) / sum(w)`. For `a(t) = exp(-sigma t)` about the centre of a
/// window of length `T`, that integral is available in closed form:
///
/// ```text
///     G(x) = sinh(x)/x * pi^2 / (pi^2 + x^2),      x = sigma T / 2
/// ```
///
/// `G >= 1` always — the window sees more of the loud early part of the decay
/// than of the quiet late part — and `G` is even in `x`, so the same
/// correction applies to a rising envelope. At the default 1.37 s window a
/// partial with a 3.5 s T60 reads 60 % high, which no amount of care
/// downstream can undo, so the tracker divides it out.
///
/// Note that a constant `G` biases only the intercept of a `log a` fit and not
/// its slope: decay-rate estimation is unbiased with or without this. What the
/// correction buys is a correct absolute envelope, and an unbiased slope where
/// `sigma` itself varies with time (the two-exponential case).
pub fn hann_decay_gain(x: f64) -> f64 {
    const PI2: f64 = std::f64::consts::PI * std::f64::consts::PI;
    // Beyond a few decades of decay inside one window the exponential model of
    // the envelope has stopped being meaningful; clamp rather than blow up.
    let x = x.abs().min(5.0);
    let sinc = if x < 1e-4 {
        // sinh(x)/x to fourth order — the ratio is 0/0 at the origin.
        1.0 + x * x / 6.0
    } else {
        x.sinh() / x
    };
    sinc * PI2 / (PI2 + x * x)
}

/// Divide the windowing bias out of a track's amplitudes, using a decay rate
/// estimated locally from the track itself.
fn compensate_decay(points: &mut [TrackPoint], window_s: f64) {
    if points.len() < 3 || window_s <= 0.0 {
        return;
    }
    let raw: Vec<(f64, f64)> = points.iter().map(|p| (p.time_s, p.amplitude)).collect();
    let half = 0.5 * window_s;
    for (index, point) in points.iter_mut().enumerate() {
        let sigma = local_decay_rate(&raw, index, half);
        point.amplitude /= hann_decay_gain(0.5 * sigma * window_s);
    }
}

/// Least-squares slope of `ln a` against `t` over the points within `half`
/// seconds of `raw[index]`, negated: the local decay rate in 1/s.
fn local_decay_rate(raw: &[(f64, f64)], index: usize, half: f64) -> f64 {
    let centre = raw[index].0;
    let (mut n, mut st, mut sy, mut stt, mut sty) = (0.0f64, 0.0, 0.0, 0.0, 0.0);
    for &(t, amplitude) in raw {
        if (t - centre).abs() > half || amplitude <= 0.0 || !amplitude.is_finite() {
            continue;
        }
        let (dt, y) = (t - centre, amplitude.ln());
        n += 1.0;
        st += dt;
        sy += y;
        stt += dt * dt;
        sty += dt * y;
    }
    if n < 3.0 {
        return 0.0;
    }
    let denominator = n * stt - st * st;
    if denominator.abs() < 1e-18 {
        return 0.0;
    }
    -(n * sty - st * sy) / denominator
}

/// Strike time, as the first moment a short-term RMS envelope rises 40 dB
/// above the noise it started from — or, if the recording has no quiet head,
/// 40 dB below its own peak.
pub fn detect_onset(signal: &[f32], sample_rate: f64) -> f64 {
    let block = ((sample_rate * 0.001) as usize).max(1);
    if signal.len() < block {
        return 0.0;
    }
    let envelope: Vec<f64> = signal
        .chunks(block)
        .map(|c| (c.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>() / c.len() as f64).sqrt())
        .collect();
    let peak = envelope.iter().fold(0.0f64, |m, &x| m.max(x));
    if peak <= 0.0 {
        return 0.0;
    }
    let threshold = peak * 0.01;
    let index = envelope.iter().position(|&x| x >= threshold).unwrap_or(0);
    index as f64 * block as f64 / sample_rate
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_decay_gain_matches_numerical_integration() {
        let window_s = 1.0;
        for &sigma in &[0.0, 0.5, 2.0, 5.0] {
            let n = 200_001;
            let (mut num, mut den) = (0.0f64, 0.0f64);
            for i in 0..n {
                let u = i as f64 / (n - 1) as f64;
                let t = (u - 0.5) * window_s;
                let w = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * u).cos();
                num += w * (-sigma * t).exp();
                den += w;
            }
            let numeric = num / den;
            let closed = hann_decay_gain(0.5 * sigma * window_s);
            assert!(
                (numeric - closed).abs() < 1e-4 * closed,
                "sigma {sigma}: {numeric} vs {closed}"
            );
        }
    }

    #[test]
    fn the_decay_gain_is_even_and_at_least_one() {
        assert!((hann_decay_gain(0.0) - 1.0).abs() < 1e-12);
        assert!((hann_decay_gain(1.3) - hann_decay_gain(-1.3)).abs() < 1e-12);
        assert!(hann_decay_gain(2.0) > 1.0);
    }

    #[test]
    fn the_association_window_never_reaches_a_neighbouring_partial() {
        // C8-ish: 60 cents is wider than the spacing between high partials,
        // so the spacing cap has to take over.
        let seed = InharmonicModel::new(4186.0, 0.01);
        let tracker = PartialTracker::new(TrackerConfig {
            stft: StftConfig::padded(1 << 12, 1 << 10, 2).unwrap(),
            ..TrackerConfig::default()
        })
        .unwrap();
        for k in 1..6 {
            let width = tracker.window_half_width(&seed, k);
            let gap = seed.partial(k + 1) - seed.partial(k);
            assert!(width < 0.5 * gap, "k={k}: {width} vs gap {gap}");
        }
    }

    #[test]
    fn the_onset_lands_on_the_strike() {
        let sr = 48_000.0;
        let mut signal = vec![0.0f32; 4_800];
        signal.extend((0..48_000).map(|i| {
            let t = f64::from(i) / sr;
            ((-3.0 * t).exp() * (2.0 * std::f64::consts::PI * 440.0 * t).sin()) as f32
        }));
        let onset = detect_onset(&signal, sr);
        assert!((onset - 0.1).abs() < 0.005, "{onset}");
    }
}
