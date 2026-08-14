//! How close the engine sounds to a real piano, measured.
//!
//! `TUNING.md` stage 2 fits the interaction parameters by replaying a
//! performance and comparing the render against a recording. This module is
//! that comparison, written once so that the loss the optimizer minimises and
//! the scoreboard a human reads are the same code. Nothing here knows about
//! the engine or about the sampler: every function takes two signals that are
//! supposed to be the same performance and says how far apart they are.
//!
//! Five metrics, each answering a different question. They are deliberately
//! not combined into one number here — a single scalar hides which axis moved,
//! and the whole point of `renders/timbre-ladder/ANALYSIS.md` was that the
//! axes disagree.
//!
//! 1. **Multi-resolution log-mel distance** ([`multi_res_log_mel_distance`]) —
//!    `TUNING.md`'s own loss: windows {256, 1024, 4096}, 64 mel bands to
//!    16 kHz, mean absolute difference in dB. The short window sees attacks,
//!    the long one sees partials; the mean of the three is the headline.
//! 2. **Modulation-spectrum distance** ([`modulation_distance`]) — how each
//!    band's level *moves*, over 0.5–50 Hz. The ladder found this the most
//!    diagnostic axis: beating, uneven decay and the liveliness no envelope
//!    model reproduces all live here, and a spectral distance is nearly blind
//!    to it.
//! 3. **Attack tonality delta** ([`attack_tonality_delta`]) — spectral
//!    flatness of the first 30 ms of every detected onset, engine minus
//!    reference. Negative means the engine's attack is noisier than the
//!    piano's, positive means it is too pure.
//! 4. **Per-band energy-envelope correlation**
//!    ([`band_envelope_correlation`]) — does bass, mid and treble rise and
//!    fall *together* in the two renders? Insensitive to timbre, sensitive to
//!    decay rates, pedal behaviour and register balance over time.
//! 5. **Release-tail energy delta** ([`release_tail_delta`]) — level in the
//!    0.5 s after a note-off that nothing else interrupts. This is the damper,
//!    the key-off noise and whatever is left ringing, and it is the one metric
//!    the whole-phrase measures average away.
//!
//! ## What the numbers are not
//!
//! Every metric is computed on the **mono** sum. The engine places keys in the
//! stereo field by its own rule and a recording carries the microphone pair it
//! was made with; a stereo distance would mostly measure that disagreement,
//! which no preset parameter in scope can fix.
//!
//! Every metric assumes the two signals are **level-matched and aligned**.
//! [`level_match`] does the first (whole-phrase RMS, both scaled to a common
//! target so neither is privileged); alignment is the caller's job and is free
//! when both renders are driven by the same event list on the same block grain.
//!
//! A distance of zero means *identical*, which nothing real ever is. The
//! number that makes a distance readable is the **noise floor**: run the same
//! metric between two recordings of the same piano playing the same phrase
//! and see what it reads. [`VelocityLayers`] builds that comparison out of the
//! sample library itself — the neighbouring velocity layer of every note is a
//! second recording of the same instrument playing the same music.

use std::f64::consts::PI;

use rustfft::{num_complex::Complex32, FftPlanner};

use crate::audio::Audio;
use crate::error::{Error, Result};
use crate::library::SampleLibrary;
use crate::sampler::{SamplerEvent, TimedEvent};
use crate::stft::{Stft, StftConfig};

// ---------------------------------------------------------------------------
// Constants: the shape of the loss. Changing any of these changes every number
// in `renders/realism/REALISM.md`, so they are named and documented rather than
// buried in call sites.
// ---------------------------------------------------------------------------

/// Window lengths of the multi-resolution distance, in samples.
/// `TUNING.md`'s "~{256, 1024, 4096}", taken literally.
pub const MULTI_RES_WINDOWS: [usize; 3] = [256, 1024, 4096];

/// Hop as a fraction of the window: 75 % overlap at every resolution.
pub const HOP_DIVISOR: usize = 4;

/// Mel bands in the distance.
pub const MEL_BANDS: usize = 64;

/// Bottom of the mel scale, in Hz. Below A0's 27.5 Hz, so the lowest key's
/// fundamental is inside the first band rather than under it.
pub const MEL_F_MIN: f64 = 20.0;

/// Top of the mel scale, in Hz.
pub const MEL_F_MAX: f64 = 16_000.0;

/// Dynamic range kept below the loudest mel cell of the *pair*, in dB.
///
/// A log-magnitude distance without a floor is dominated by silence: two
/// renders that agree everywhere audible can differ by 60 dB where both are at
/// −140 dBFS, and that difference is numerical, not musical. The floor is
/// taken from the pair rather than from each signal so that the clamp is the
/// same operation on both and the distance stays symmetric.
pub const MEL_FLOOR_DB: f64 = -80.0;

/// Analysis window for every envelope-domain metric (modulation, band
/// correlation), in samples: 21 ms, short enough to follow an attack and long
/// enough to resolve the bass.
pub const ENVELOPE_WINDOW: usize = 1024;

/// Hop for the envelope-domain metrics, in samples: 5.33 ms, i.e. an envelope
/// sampled at 187.5 Hz, so modulation up to 93 Hz is unaliased.
pub const ENVELOPE_HOP: usize = 256;

/// Bands the modulation spectrum is measured in. Fewer than [`MEL_BANDS`]:
/// a modulation spectrum needs the whole phrase in one transform, and 16 bands
/// keeps each one wide enough that its envelope is a level rather than a
/// partial.
pub const MODULATION_BANDS: usize = 16;

/// Modulation frequencies compared, in Hz. Below 0.5 Hz is the note's own
/// decay (measured better by the band correlation); above 50 Hz is roughness
/// rather than movement, and the spectral distance already sees it.
pub const MODULATION_LO_HZ: f64 = 0.5;
/// Top of the compared modulation range, in Hz.
pub const MODULATION_HI_HZ: f64 = 50.0;

/// Log-spaced bins the modulation spectrum is reduced to before comparison.
/// Averaging inside a bin is what turns a noisy periodogram into a statistic.
pub const MODULATION_BINS: usize = 12;

/// Smallest level movement the modulation distance distinguishes, on the scale
/// the modulation spectrum is quoted in: `20·log10` of a magnitude in decibels,
/// so −40 means one hundredth of a decibel of movement at that rate.
///
/// An *absolute* floor rather than one relative to each band's own loudest
/// modulation. A band that does not move at all still produces a modulation
/// spectrum 270 dB down — the arithmetic of subtracting a mean from a constant
/// — and a relative floor would stretch that numerical dust across the full
/// comparison range and report it as a difference.
pub const MODULATION_FLOOR_DB: f64 = -40.0;

/// Dynamic range of the band *levels* the modulation spectrum is taken of, in
/// dB below the loudest cell of the whole spectrogram.
///
/// Tighter than [`MEL_FLOOR_DB`] on purpose. The spectral distance compares
/// levels, where 80 dB of range is still information; the modulation distance
/// compares how a level *moves*, and taking a logarithm of a band that is 80 dB
/// down magnifies the analysis window's own leakage into a large, entirely
/// artificial modulation. Sixty decibels under the loudest moment of the phrase
/// is below anything that contributes to how the sound is heard to move.
pub const MODULATION_LEVEL_FLOOR_DB: f64 = -60.0;

/// Length of the attack the tonality metric reads, in seconds.
pub const ATTACK_WINDOW_S: f64 = 0.030;

/// How far past an onset the rise time is looked for, in seconds. A grand's
/// hammer is in contact for at most a couple of milliseconds, but the note it
/// leaves behind takes rather longer to reach its loudest.
pub const ATTACK_RISE_WINDOW_S: f64 = 0.060;

/// Length of the release tail the energy delta reads, in seconds.
pub const RELEASE_WINDOW_S: f64 = 0.5;

/// The three registers the envelope correlation is reported in: name, low
/// edge, high edge in Hz. Bass ends where the lowest strings' second partial
/// does; treble starts where the piano has no fundamentals left.
pub const ENERGY_BANDS: [(&str, f64, f64); 3] = [
    ("bass", 20.0, 250.0),
    ("mid", 250.0, 2_000.0),
    ("treble", 2_000.0, 16_000.0),
];

/// RMS both signals of a pair are brought to before anything is measured.
/// −26 dBFS: loud enough to be well above any dither, quiet enough that a
/// phrase with a fortissimo chord in it does not need the peak guard.
pub const TARGET_RMS: f32 = 0.05;

/// Peak a level-matched pair is allowed to reach before both are scaled down
/// together.
pub const PEAK_CEILING: f32 = 0.98;

// ---------------------------------------------------------------------------
// Mel scale
// ---------------------------------------------------------------------------

/// HTK's mel scale — the one `TUNING.md`'s "mel weighting" means, and the one
/// every mel-STFT loss in the literature uses.
pub fn hz_to_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Inverse of [`hz_to_mel`].
pub fn mel_to_hz(mel: f64) -> f64 {
    700.0 * (10.0f64.powf(mel / 2595.0) - 1.0)
}

/// A triangular mel filterbank over the non-negative half of a transform.
#[derive(Clone, Debug)]
pub struct MelBank {
    bands: usize,
    bins: usize,
    /// Per band, the `(bin, weight)` pairs with non-zero weight.
    weights: Vec<Vec<(usize, f32)>>,
    centres_hz: Vec<f64>,
}

impl MelBank {
    /// Triangles whose apexes are equally spaced in mel between `f_min` and
    /// `f_max`, each peaking at 1.0.
    ///
    /// No area normalisation: a band's value is the energy that is actually in
    /// it, which is what makes the numbers readable as levels. Both signals go
    /// through the same bank, so the weighting cancels in the difference.
    ///
    /// **The narrow-band fallback.** At a 256-sample window the bins are
    /// 187.5 Hz apart and the bottom two dozen mel triangles are narrower than
    /// one bin, so they would collect nothing. A band that collects nothing is
    /// worse than useless in a distance: it reads the floor in *both* signals
    /// and contributes an exact zero, silently diluting the mean. Such a band
    /// is given unit weight on the bin nearest its apex instead, so all
    /// [`MEL_BANDS`] rows carry the level at their own frequency at every
    /// resolution — several low rows then repeat the same bin, which is an
    /// honest statement that the short window cannot separate them.
    pub fn new(bands: usize, fft_size: usize, sample_rate: f64, f_min: f64, f_max: f64) -> Result<Self> {
        if bands < 2 {
            return Err(Error::Config(format!("{bands} mel bands is not a filterbank")));
        }
        if fft_size < 4 || fft_size % 2 != 0 {
            return Err(Error::Config(format!("fft size {fft_size} cannot be a mel bank")));
        }
        if !(f_min >= 0.0 && f_max > f_min && f_max <= sample_rate / 2.0) {
            return Err(Error::Config(format!(
                "mel range {f_min}..{f_max} Hz does not fit inside a {sample_rate} Hz signal"
            )));
        }
        let bins = fft_size / 2 + 1;
        let bin_hz = sample_rate / fft_size as f64;
        let (mel_lo, mel_hi) = (hz_to_mel(f_min), hz_to_mel(f_max));
        let edges: Vec<f64> = (0..bands + 2)
            .map(|i| mel_to_hz(mel_lo + (mel_hi - mel_lo) * i as f64 / (bands + 1) as f64))
            .collect();

        let mut weights = vec![Vec::new(); bands];
        for (b, slot) in weights.iter_mut().enumerate() {
            let (lo, mid, hi) = (edges[b], edges[b + 1], edges[b + 2]);
            for k in 0..bins {
                let f = k as f64 * bin_hz;
                let w = if f > lo && f < mid {
                    (f - lo) / (mid - lo)
                } else if f >= mid && f < hi {
                    (hi - f) / (hi - mid)
                } else {
                    0.0
                };
                if w > 0.0 {
                    slot.push((k, w as f32));
                }
            }
            if slot.is_empty() {
                let k = (mid / bin_hz).round().clamp(0.0, (bins - 1) as f64) as usize;
                slot.push((k, 1.0));
            }
        }
        Ok(MelBank {
            bands,
            bins,
            weights,
            centres_hz: edges[1..=bands].to_vec(),
        })
    }

    pub fn bands(&self) -> usize {
        self.bands
    }

    /// Apex frequency of each band, in Hz.
    pub fn centres_hz(&self) -> &[f64] {
        &self.centres_hz
    }

    /// Sum of the weights of one band. Zero would mean a dead row; the
    /// constructor guarantees it never is.
    pub fn band_weight(&self, band: usize) -> f64 {
        self.weights[band].iter().map(|&(_, w)| f64::from(w)).sum()
    }

    /// Band energies of one power spectrum. `power` must be the `fft_size/2+1`
    /// non-negative bins.
    pub fn apply(&self, power: &[f32], out: &mut [f64]) {
        debug_assert_eq!(power.len(), self.bins);
        debug_assert_eq!(out.len(), self.bands);
        for (o, band) in out.iter_mut().zip(self.weights.iter()) {
            let mut sum = 0.0f64;
            for &(k, w) in band {
                sum += f64::from(w) * f64::from(power[k]);
            }
            *o = sum;
        }
    }
}

// ---------------------------------------------------------------------------
// Mel spectrogram
// ---------------------------------------------------------------------------

/// A mel spectrogram in linear energy, one row per frame.
#[derive(Clone, Debug)]
pub struct MelSpec {
    pub sample_rate: f64,
    pub window: usize,
    pub hop: usize,
    pub centres_hz: Vec<f64>,
    /// `frames[t][b]`, linear band energy.
    pub frames: Vec<Vec<f64>>,
}

impl MelSpec {
    pub fn bands(&self) -> usize {
        self.centres_hz.len()
    }

    /// Time the centre of frame `t` describes, in seconds.
    pub fn time_s(&self, frame: usize) -> f64 {
        (frame * self.hop) as f64 + 0.5 * self.window as f64
    }

    /// Frame time in seconds (the value [`MelSpec::time_s`] returns, divided
    /// by the sample rate).
    pub fn frame_time_s(&self, frame: usize) -> f64 {
        self.time_s(frame) / self.sample_rate
    }

    /// Loudest cell, in dB.
    pub fn peak_db(&self) -> f64 {
        let mut peak = f64::NEG_INFINITY;
        for row in &self.frames {
            for &e in row {
                if e > 0.0 {
                    let db = 10.0 * e.log10();
                    if db > peak {
                        peak = db;
                    }
                }
            }
        }
        peak
    }
}

/// Mel spectrogram of a mono signal.
///
/// Magnitudes come from [`Stft`], which calibrates a sinusoid of amplitude `A`
/// to a peak of `A` regardless of window length — so the three resolutions of
/// the multi-resolution distance are on one level scale and their dB
/// differences are directly comparable.
pub fn mel_spectrogram(
    signal: &[f32],
    sample_rate: f64,
    window: usize,
    hop: usize,
    bands: usize,
    f_min: f64,
    f_max: f64,
) -> Result<MelSpec> {
    let config = StftConfig::new(window, hop, window)?;
    if config.frame_count(signal.len()) == 0 {
        return Err(Error::Config(format!(
            "{} samples is shorter than the {window}-sample window",
            signal.len()
        )));
    }
    let stft = Stft::new(config)?;
    let bank = MelBank::new(bands, window, sample_rate, f_min, f_max)?;
    let mut frames = Vec::with_capacity(config.frame_count(signal.len()));
    let bins = stft.bins();
    let mut power = vec![0.0f32; bins];
    let mut row = vec![0.0f64; bands];
    stft.for_each_frame(signal, sample_rate, |_t, magnitude| {
        for (p, &m) in power.iter_mut().zip(magnitude.iter()) {
            *p = m * m;
        }
        bank.apply(&power, &mut row);
        frames.push(row.clone());
    });
    Ok(MelSpec {
        sample_rate,
        window,
        hop,
        centres_hz: bank.centres_hz().to_vec(),
        frames,
    })
}

// ---------------------------------------------------------------------------
// 1. Multi-resolution log-mel distance
// ---------------------------------------------------------------------------

/// One resolution of the log-mel distance, with the breakdown that says where
/// it comes from.
#[derive(Clone, Debug)]
pub struct MelDiff {
    pub window: usize,
    pub hop: usize,
    /// Mean absolute dB difference over every band and frame.
    pub mean: f64,
    /// Mean absolute dB difference of each band.
    pub per_band: Vec<f64>,
    /// Mean absolute dB difference of each frame.
    pub per_frame: Vec<f64>,
    /// Signed mean dB difference of each band (engine minus reference):
    /// positive means the engine has more energy there.
    pub signed_per_band: Vec<f64>,
    pub centres_hz: Vec<f64>,
    pub times_s: Vec<f64>,
}

impl MelDiff {
    /// Band with the largest mean absolute difference: `(centre Hz, dB)`.
    pub fn worst_band(&self) -> (f64, f64) {
        let mut best = (0.0, f64::NEG_INFINITY);
        for (i, &d) in self.per_band.iter().enumerate() {
            if d > best.1 {
                best = (self.centres_hz[i], d);
            }
        }
        best
    }

    /// Instant with the largest mean absolute difference: `(seconds, dB)`.
    ///
    /// Searched over a 30 ms moving average rather than frame by frame. One
    /// frame is five milliseconds and one frame's disagreement is noise; the
    /// question the report asks is which *moment* of the phrase disagrees, and
    /// thirty milliseconds is about the length of the shortest thing a piano
    /// does.
    pub fn worst_time(&self) -> (f64, f64) {
        if self.per_frame.is_empty() {
            return (0.0, 0.0);
        }
        let spacing = if self.times_s.len() > 1 {
            self.times_s[1] - self.times_s[0]
        } else {
            1.0
        };
        let half = ((0.015 / spacing).round() as usize).max(1);
        let n = self.per_frame.len();
        let mut best = (self.times_s[0], f64::NEG_INFINITY);
        for i in 0..n {
            let lo = i.saturating_sub(half);
            let hi = (i + half + 1).min(n);
            let mean = self.per_frame[lo..hi].iter().sum::<f64>() / (hi - lo) as f64;
            if mean > best.1 {
                best = (self.times_s[i], mean);
            }
        }
        best
    }
}

/// Log-mel difference at one resolution, in dB.
pub fn log_mel_diff(
    engine: &[f32],
    reference: &[f32],
    sample_rate: f64,
    window: usize,
    hop: usize,
    bands: usize,
) -> Result<MelDiff> {
    let a = mel_spectrogram(engine, sample_rate, window, hop, bands, MEL_F_MIN, MEL_F_MAX)?;
    let b = mel_spectrogram(reference, sample_rate, window, hop, bands, MEL_F_MIN, MEL_F_MAX)?;
    let n = a.frames.len().min(b.frames.len());
    if n == 0 {
        return Err(Error::Config("no frames in common".into()));
    }
    let floor = a.peak_db().max(b.peak_db()) + MEL_FLOOR_DB;
    let db = |e: f64| -> f64 {
        if e <= 0.0 {
            floor
        } else {
            (10.0 * e.log10()).max(floor)
        }
    };

    let mut per_band = vec![0.0f64; bands];
    let mut signed_per_band = vec![0.0f64; bands];
    let mut per_frame = vec![0.0f64; n];
    for (t, frame) in per_frame.iter_mut().enumerate() {
        let mut frame_sum = 0.0;
        for (k, (band, signed)) in per_band.iter_mut().zip(signed_per_band.iter_mut()).enumerate() {
            let d = db(a.frames[t][k]) - db(b.frames[t][k]);
            *band += d.abs();
            *signed += d;
            frame_sum += d.abs();
        }
        *frame = frame_sum / bands as f64;
    }
    for v in per_band.iter_mut() {
        *v /= n as f64;
    }
    for v in signed_per_band.iter_mut() {
        *v /= n as f64;
    }
    let mean = per_frame.iter().sum::<f64>() / n as f64;
    Ok(MelDiff {
        window,
        hop,
        mean,
        per_band,
        per_frame,
        signed_per_band,
        centres_hz: a.centres_hz.clone(),
        times_s: (0..n).map(|t| a.frame_time_s(t)).collect(),
        })
}

/// `TUNING.md`'s stage-2 spectral loss: the mean over three resolutions of the
/// mean absolute log-mel difference, in dB.
#[derive(Clone, Debug)]
pub struct MultiResDistance {
    /// One [`MelDiff`] per entry of [`MULTI_RES_WINDOWS`].
    pub resolutions: Vec<MelDiff>,
    /// Mean of the three, and the number the scoreboard quotes.
    pub mean: f64,
}

impl MultiResDistance {
    /// The middle resolution (1024), which is the one the images are drawn at
    /// and the one whose per-band and per-frame breakdowns are readable.
    pub fn detail(&self) -> &MelDiff {
        &self.resolutions[1]
    }
}

/// Multi-resolution log-mel distance between two mono signals, in dB.
pub fn multi_res_log_mel_distance(
    engine: &[f32],
    reference: &[f32],
    sample_rate: f64,
) -> Result<MultiResDistance> {
    let mut resolutions = Vec::with_capacity(MULTI_RES_WINDOWS.len());
    for &window in &MULTI_RES_WINDOWS {
        resolutions.push(log_mel_diff(
            engine,
            reference,
            sample_rate,
            window,
            window / HOP_DIVISOR,
            MEL_BANDS,
        )?);
    }
    let mean = resolutions.iter().map(|r| r.mean).sum::<f64>() / resolutions.len() as f64;
    Ok(MultiResDistance { resolutions, mean })
}

// ---------------------------------------------------------------------------
// 2. Modulation-spectrum distance
// ---------------------------------------------------------------------------

/// Distance between the modulation spectra of two signals' band envelopes.
#[derive(Clone, Debug)]
pub struct ModulationDistance {
    /// Mean absolute dB difference over every (band, modulation bin) cell.
    pub mean: f64,
    /// Mean absolute dB difference of each band.
    pub per_band: Vec<f64>,
    /// Mean absolute dB difference of each modulation bin.
    pub per_bin: Vec<f64>,
    /// Apex frequency of each band, in Hz.
    pub centres_hz: Vec<f64>,
    /// Geometric centre of each modulation bin, in Hz.
    pub mod_centres_hz: Vec<f64>,
}

impl ModulationDistance {
    /// Band with the largest difference: `(centre Hz, dB)`.
    pub fn worst_band(&self) -> (f64, f64) {
        let mut best = (0.0, f64::NEG_INFINITY);
        for (i, &d) in self.per_band.iter().enumerate() {
            if d > best.1 {
                best = (self.centres_hz[i], d);
            }
        }
        best
    }

    /// Modulation rate with the largest difference: `(Hz, dB)`.
    pub fn worst_rate(&self) -> (f64, f64) {
        let mut best = (0.0, f64::NEG_INFINITY);
        for (i, &d) in self.per_bin.iter().enumerate() {
            if d > best.1 {
                best = (self.mod_centres_hz[i], d);
            }
        }
        best
    }
}

/// A modulation spectrum: per band a row of dB values, the bands' apex
/// frequencies, and the modulation bins' centre frequencies.
type ModulationSpectrum = (Vec<Vec<f64>>, Vec<f64>, Vec<f64>);

/// Modulation spectrum of every band of one signal.
///
/// Per band: the level in dB over time, mean removed (so the metric is blind
/// to how loud the band is and sees only how it *moves*), Hann-windowed, and
/// transformed. The periodogram is then averaged inside [`MODULATION_BINS`]
/// log-spaced bins between [`MODULATION_LO_HZ`] and [`MODULATION_HI_HZ`],
/// which is what turns one noisy realisation into a statistic.
fn modulation_spectrum(signal: &[f32], sample_rate: f64) -> Result<ModulationSpectrum> {
    let spec = mel_spectrogram(
        signal,
        sample_rate,
        ENVELOPE_WINDOW,
        ENVELOPE_HOP,
        MODULATION_BANDS,
        MEL_F_MIN,
        MEL_F_MAX,
    )?;
    let n = spec.frames.len();
    if n < 16 {
        return Err(Error::Config(format!(
            "{n} envelope frames is too short for a modulation spectrum"
        )));
    }
    let fps = sample_rate / ENVELOPE_HOP as f64;
    let fft_size = n.next_power_of_two().max(64);
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(fft_size);
    let hann: Vec<f64> = (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / n as f64).cos())
        .collect();
    let coherent_gain: f64 = hann.iter().sum::<f64>() / 2.0;

    // Log-spaced modulation bins.
    let mut edges = Vec::with_capacity(MODULATION_BINS + 1);
    for i in 0..=MODULATION_BINS {
        let t = i as f64 / MODULATION_BINS as f64;
        edges.push(MODULATION_LO_HZ * (MODULATION_HI_HZ / MODULATION_LO_HZ).powf(t));
    }
    let mod_centres: Vec<f64> = (0..MODULATION_BINS)
        .map(|i| (edges[i] * edges[i + 1]).sqrt())
        .collect();

    // One floor for the whole spectrogram, not one per band: a band that never
    // rises above it is then a flat line rather than a magnified picture of the
    // analysis window's own leakage, and reads no modulation at all — the
    // honest answer for a band nothing audible is happening in.
    let global_peak = spec.peak_db();
    if !global_peak.is_finite() {
        return Err(Error::Config("a silent signal has no modulation spectrum".into()));
    }
    let floor = global_peak + MODULATION_LEVEL_FLOOR_DB;

    let mut out = Vec::with_capacity(MODULATION_BANDS);
    let mut buffer = vec![Complex32::new(0.0, 0.0); fft_size];
    for band in 0..MODULATION_BANDS {
        let level: Vec<f64> = (0..n)
            .map(|t| {
                let e = spec.frames[t][band];
                if e > 0.0 {
                    (10.0 * e.log10()).max(floor)
                } else {
                    floor
                }
            })
            .collect();
        let mean = level.iter().sum::<f64>() / n as f64;

        for (slot, (&x, &w)) in buffer.iter_mut().zip(level.iter().zip(hann.iter())) {
            *slot = Complex32::new(((x - mean) * w) as f32, 0.0);
        }
        for slot in buffer[n..].iter_mut() {
            *slot = Complex32::new(0.0, 0.0);
        }
        fft.process(&mut buffer);

        // Mean power inside each log-spaced bin; nearest FFT bin when a bin is
        // narrower than the transform's resolution.
        let bin_hz = fps / fft_size as f64;
        let mut row = Vec::with_capacity(MODULATION_BINS);
        for i in 0..MODULATION_BINS {
            let (lo, hi) = (edges[i], edges[i + 1]);
            let k_lo = (lo / bin_hz).ceil().max(1.0) as usize;
            let k_hi = ((hi / bin_hz).floor() as usize).min(fft_size / 2);
            let (mut sum, mut count) = (0.0f64, 0usize);
            for slot in buffer.iter().take(k_hi.min(fft_size / 2) + 1).skip(k_lo) {
                let m = f64::from(slot.norm()) / coherent_gain;
                sum += m * m;
                count += 1;
            }
            if count == 0 {
                let k = (mod_centres[i] / bin_hz)
                    .round()
                    .clamp(1.0, (fft_size / 2) as f64) as usize;
                let m = f64::from(buffer[k].norm()) / coherent_gain;
                sum = m * m;
                count = 1;
            }
            row.push((sum / count as f64).sqrt());
        }
        out.push(row.iter().map(|&m| if m > 0.0 { 20.0 * m.log10() } else { f64::NEG_INFINITY }).collect());
    }
    Ok((out, spec.centres_hz, mod_centres))
}

/// Distance between two signals' modulation spectra, in dB.
pub fn modulation_distance(
    engine: &[f32],
    reference: &[f32],
    sample_rate: f64,
) -> Result<ModulationDistance> {
    let (a, centres, mod_centres) = modulation_spectrum(engine, sample_rate)?;
    let (b, _, _) = modulation_spectrum(reference, sample_rate)?;

    let mut per_band = vec![0.0f64; MODULATION_BANDS];
    let mut per_bin = vec![0.0f64; MODULATION_BINS];
    let mut total = 0.0;
    let floor = MODULATION_FLOOR_DB;
    for (band, out) in per_band.iter_mut().enumerate() {
        for (bin, into) in per_bin.iter_mut().enumerate() {
            let d = (a[band][bin].max(floor) - b[band][bin].max(floor)).abs();
            *out += d;
            *into += d;
            total += d;
        }
        *out /= MODULATION_BINS as f64;
    }
    for v in per_bin.iter_mut() {
        *v /= MODULATION_BANDS as f64;
    }
    Ok(ModulationDistance {
        mean: total / (MODULATION_BANDS * MODULATION_BINS) as f64,
        per_band,
        per_bin,
        centres_hz: centres,
        mod_centres_hz: mod_centres,
    })
}

// ---------------------------------------------------------------------------
// 3. Onsets and attack tonality
// ---------------------------------------------------------------------------

/// Onset times, in seconds, from spectral flux over a mel spectrogram.
///
/// Detected rather than read off the event list on purpose: stage 2 compares
/// against recordings whose alignment is only as good as their MIDI, and a
/// metric that needs a perfect event list cannot be used there. The caller
/// detects on the *reference* and measures both signals at those positions, so
/// the two are read at the same instants.
pub fn detect_onsets(signal: &[f32], sample_rate: f64) -> Result<Vec<f64>> {
    let hop = 128usize;
    let window = ENVELOPE_WINDOW;
    let spec = mel_spectrogram(
        signal,
        sample_rate,
        window,
        hop,
        MEL_BANDS,
        MEL_F_MIN,
        MEL_F_MAX,
    )?;
    let n = spec.frames.len();
    if n < 4 {
        return Ok(Vec::new());
    }
    let floor = spec.peak_db() + MEL_FLOOR_DB;
    let db = |e: f64| if e > 0.0 { (10.0 * e.log10()).max(floor) } else { floor };

    let mut flux = vec![0.0f64; n];
    for (t, slot) in flux.iter_mut().enumerate().skip(1) {
        let mut sum = 0.0;
        for (now, before) in spec.frames[t].iter().zip(spec.frames[t - 1].iter()) {
            let d = db(*now) - db(*before);
            if d > 0.0 {
                sum += d;
            }
        }
        *slot = sum / MEL_BANDS as f64;
    }
    let global = flux.iter().cloned().fold(0.0f64, f64::max);
    if global <= 0.0 {
        return Ok(Vec::new());
    }

    let fps = sample_rate / hop as f64;
    let half = (0.15 * fps).round() as usize; // local statistics window
    let refractory = (0.06 * fps).round() as usize;
    let peak_span = 3usize;

    let mut onsets = Vec::new();
    let mut last: Option<usize> = None;
    for t in 1..n - 1 {
        let lo = t.saturating_sub(peak_span);
        let hi = (t + peak_span).min(n - 1);
        if flux[lo..=hi].iter().cloned().fold(0.0f64, f64::max) > flux[t] {
            continue;
        }
        if flux[t] < 0.10 * global {
            continue;
        }
        let wlo = t.saturating_sub(half);
        let whi = (t + half).min(n - 1);
        let window_slice = &flux[wlo..=whi];
        let mean = window_slice.iter().sum::<f64>() / window_slice.len() as f64;
        let var = window_slice.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>()
            / window_slice.len() as f64;
        if flux[t] < mean + 0.6 * var.sqrt() {
            continue;
        }
        if let Some(prev) = last {
            if t - prev < refractory {
                continue;
            }
        }
        last = Some(t);
        onsets.push(refine_onset(signal, sample_rate, t * hop));
    }
    Ok(onsets)
}

/// Move a coarse onset (the *start* sample of the flux frame that fired) onto
/// the steepest rise of a 1 ms energy envelope inside that frame. The flux
/// peak is a frame index; a 30 ms attack window placed a frame early or late
/// measures something else.
fn refine_onset(signal: &[f32], sample_rate: f64, frame_start: usize) -> f64 {
    let step = (0.001 * sample_rate).round().max(1.0) as usize;
    let span = ENVELOPE_WINDOW + step;
    let start = frame_start.saturating_sub(step * 2);
    let end = (start + span).min(signal.len());
    if end <= start + 2 * step {
        return frame_start as f64 / sample_rate;
    }
    let cells = (end - start) / step;
    let mut level = Vec::with_capacity(cells);
    for c in 0..cells {
        let s = start + c * step;
        let e = (s + step).min(end);
        let mean: f64 = signal[s..e].iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>()
            / (e - s).max(1) as f64;
        level.push(mean.sqrt());
    }
    let mut best = 0usize;
    let mut best_rise = f64::NEG_INFINITY;
    for c in 1..cells {
        let rise = level[c] - level[c - 1];
        if rise > best_rise {
            best_rise = rise;
            best = c - 1;
        }
    }
    (start + best * step) as f64 / sample_rate
}

/// Spectral flatness of a block, as a *tonality* in dB: the arithmetic mean of
/// the power spectrum over its geometric mean, so a sinusoid is a large
/// positive number and white noise is 0.
pub fn attack_tonality_db(block: &[f32], sample_rate: f64) -> f64 {
    let n = block.len().next_power_of_two().max(64);
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n);
    let mut buffer = vec![Complex32::new(0.0, 0.0); n];
    let len = block.len();
    for (i, slot) in buffer.iter_mut().take(len).enumerate() {
        let w = 0.5 - 0.5 * (2.0 * PI * i as f64 / len as f64).cos();
        *slot = Complex32::new(block[i] * w as f32, 0.0);
    }
    fft.process(&mut buffer);

    // 50 Hz to 10 kHz: below is the window's own leakage, above is where a
    // 30 ms block of a piano note has nothing but the noise floor.
    let bin_hz = sample_rate / n as f64;
    let k_lo = (50.0 / bin_hz).ceil().max(1.0) as usize;
    let k_hi = ((10_000.0 / bin_hz).floor() as usize).min(n / 2);
    if k_hi <= k_lo + 4 {
        return 0.0;
    }
    let mut sum = 0.0f64;
    let mut log_sum = 0.0f64;
    let mut count = 0usize;
    let mut peak = 0.0f64;
    for slot in buffer.iter().take(k_hi + 1).skip(k_lo) {
        let p = f64::from(slot.norm_sqr());
        peak = peak.max(p);
        count += 1;
        sum += p;
    }
    if count == 0 || peak <= 0.0 {
        return 0.0;
    }
    // A floor 120 dB under the block's own peak keeps the geometric mean from
    // being decided by a numerically empty bin.
    let floor = peak * 1e-12;
    for slot in buffer.iter().take(k_hi + 1).skip(k_lo) {
        let p = f64::from(slot.norm_sqr()).max(floor);
        log_sum += p.ln();
    }
    let arithmetic = sum / count as f64;
    let geometric = (log_sum / count as f64).exp();
    if arithmetic <= 0.0 || geometric <= 0.0 {
        return 0.0;
    }
    10.0 * (arithmetic / geometric).log10()
}

/// Attack tonality of the engine minus that of the reference, over the onsets
/// the reference has.
#[derive(Clone, Debug)]
pub struct AttackDelta {
    pub onsets: usize,
    /// Engine minus reference, mean over onsets, in dB. Positive: the engine's
    /// attacks are more tonal (less noisy) than the piano's.
    pub mean_signed_db: f64,
    /// Mean of the absolute per-onset differences, in dB.
    pub mean_abs_db: f64,
    pub engine_mean_db: f64,
    pub reference_mean_db: f64,
    /// Mean time from onset to the loudest millisecond of the attack, in
    /// seconds: `(engine, reference)`. Not a distance — the plainest
    /// description there is of how differently the two things start.
    pub rise_s: (f64, f64),
    /// Onset with the largest absolute difference: `(seconds, signed dB)`.
    pub worst: Option<(f64, f64)>,
}

/// Time from `onset_s` to the loudest millisecond within
/// [`ATTACK_RISE_WINDOW_S`] of it, in seconds.
pub fn attack_rise_s(signal: &[f32], sample_rate: f64, onset_s: f64) -> Option<f64> {
    let step = (0.001 * sample_rate).round().max(1.0) as usize;
    let start = (onset_s * sample_rate).round().max(0.0) as usize;
    let end = (start + (ATTACK_RISE_WINDOW_S * sample_rate) as usize).min(signal.len());
    if end <= start + 2 * step {
        return None;
    }
    let cells = (end - start) / step;
    let raw: Vec<f64> = (0..cells)
        .map(|c| {
            let s = start + c * step;
            let e = (s + step).min(end);
            signal[s..e].iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>()
                / (e - s) as f64
        })
        .collect();
    // A millisecond is a fraction of a cycle at the bottom of the compass, so
    // the raw cells ripple at the waveform's own rate. Five of them averaged is
    // an envelope; without this the "loudest millisecond" of a bass note is
    // whichever crest of the fundamental happened to land in a cell.
    let span = 2usize;
    let level: Vec<f64> = (0..cells)
        .map(|c| {
            let lo = c.saturating_sub(span);
            let hi = (c + span + 1).min(cells);
            raw[lo..hi].iter().sum::<f64>() / (hi - lo) as f64
        })
        .collect();
    let peak = level.iter().cloned().fold(0.0f64, f64::max);
    if peak <= 0.0 {
        return None;
    }
    // The *first* millisecond within half a decibel of the loudest, not the
    // argmax: an attack that arrives at once and then holds has a plateau, and
    // which cell of a plateau wins an argmax is decided by rounding.
    let threshold = peak * 10.0f64.powf(-0.05);
    let cell = level.iter().position(|&v| v >= threshold).unwrap_or(0);
    Some(cell as f64 * step as f64 / sample_rate)
}

/// Attack tonality delta over given onsets.
pub fn attack_tonality_delta(
    engine: &[f32],
    reference: &[f32],
    sample_rate: f64,
    onsets: &[f64],
) -> AttackDelta {
    let len = (ATTACK_WINDOW_S * sample_rate).round() as usize;
    let mut deltas = Vec::new();
    let mut engine_levels = Vec::new();
    let mut reference_levels = Vec::new();
    let mut rises = Vec::new();
    for &t in onsets {
        let start = (t * sample_rate).round().max(0.0) as usize;
        let end = start + len;
        if end > engine.len() || end > reference.len() {
            continue;
        }
        let ea = attack_tonality_db(&engine[start..end], sample_rate);
        let rb = attack_tonality_db(&reference[start..end], sample_rate);
        engine_levels.push(ea);
        reference_levels.push(rb);
        if let (Some(x), Some(y)) = (
            attack_rise_s(engine, sample_rate, t),
            attack_rise_s(reference, sample_rate, t),
        ) {
            rises.push((x, y));
        }
        deltas.push((t, ea - rb));
    }
    if deltas.is_empty() {
        return AttackDelta {
            onsets: 0,
            mean_signed_db: 0.0,
            mean_abs_db: 0.0,
            engine_mean_db: 0.0,
            reference_mean_db: 0.0,
            rise_s: (0.0, 0.0),
            worst: None,
        };
    }
    let rise_s = if rises.is_empty() {
        (0.0, 0.0)
    } else {
        let k = rises.len() as f64;
        (
            rises.iter().map(|r| r.0).sum::<f64>() / k,
            rises.iter().map(|r| r.1).sum::<f64>() / k,
        )
    };
    let n = deltas.len() as f64;
    let worst = deltas
        .iter()
        .cloned()
        .fold((0.0f64, 0.0f64), |acc, x| if x.1.abs() > acc.1.abs() { x } else { acc });
    AttackDelta {
        onsets: deltas.len(),
        mean_signed_db: deltas.iter().map(|d| d.1).sum::<f64>() / n,
        mean_abs_db: deltas.iter().map(|d| d.1.abs()).sum::<f64>() / n,
        engine_mean_db: engine_levels.iter().sum::<f64>() / n,
        reference_mean_db: reference_levels.iter().sum::<f64>() / n,
        rise_s,
        worst: Some(worst),
    }
}

// ---------------------------------------------------------------------------
// 4. Per-band energy-envelope correlation
// ---------------------------------------------------------------------------

/// Pearson correlation of the two signals' level-over-time in each register.
#[derive(Clone, Debug)]
pub struct BandCorrelation {
    pub names: [&'static str; 3],
    pub r: [f64; 3],
}

impl BandCorrelation {
    /// Register whose envelopes agree least: `(name, r)`.
    pub fn worst(&self) -> (&'static str, f64) {
        let mut best = (self.names[0], self.r[0]);
        for i in 1..3 {
            if self.r[i] < best.1 {
                best = (self.names[i], self.r[i]);
            }
        }
        best
    }
}

/// Level over time of one frequency range, in dB, floored 60 dB under its own
/// peak.
fn band_envelope(signal: &[f32], sample_rate: f64, lo_hz: f64, hi_hz: f64) -> Result<Vec<f64>> {
    let config = StftConfig::new(ENVELOPE_WINDOW, ENVELOPE_HOP, ENVELOPE_WINDOW)?;
    if config.frame_count(signal.len()) == 0 {
        return Err(Error::Config("signal shorter than the envelope window".into()));
    }
    let stft = Stft::new(config)?;
    let bin_hz = sample_rate / ENVELOPE_WINDOW as f64;
    let k_lo = (lo_hz / bin_hz).ceil().max(0.0) as usize;
    let k_hi = ((hi_hz / bin_hz).floor() as usize).min(stft.bins() - 1);
    let mut out = Vec::with_capacity(config.frame_count(signal.len()));
    stft.for_each_frame(signal, sample_rate, |_t, magnitude| {
        let mut sum = 0.0f64;
        for &bin in magnitude.iter().take(k_hi + 1).skip(k_lo) {
            let m = f64::from(bin);
            sum += m * m;
        }
        out.push(sum);
    });
    let peak = out.iter().cloned().fold(0.0f64, f64::max);
    let floor = if peak > 0.0 { 10.0 * peak.log10() - 60.0 } else { 0.0 };
    Ok(out
        .into_iter()
        .map(|e| if e > 0.0 { (10.0 * e.log10()).max(floor) } else { floor })
        .collect())
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n < 2 {
        return 0.0;
    }
    let ma = a[..n].iter().sum::<f64>() / n as f64;
    let mb = b[..n].iter().sum::<f64>() / n as f64;
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let (da, db) = (a[i] - ma, b[i] - mb);
        sab += da * db;
        saa += da * da;
        sbb += db * db;
    }
    if saa <= 0.0 || sbb <= 0.0 {
        return 0.0;
    }
    sab / (saa * sbb).sqrt()
}

/// Lag, in seconds, at which the two signals' broadband energy envelopes agree
/// best — positive when `engine` is *later* than `reference`.
///
/// Not a quality metric: an alignment check. Every distance in this module
/// assumes the two renders are the same performance at the same instants, and
/// a systematic offset of a few milliseconds would inflate all of them —
/// especially the short-window spectral distance and the attack tonality —
/// while looking exactly like a timbre difference. Searched over ±`max_lag_s`
/// on the envelope rather than the waveform, because two different pianos
/// playing the same note have no phase relationship to find.
pub fn envelope_lag_s(
    engine: &[f32],
    reference: &[f32],
    sample_rate: f64,
    max_lag_s: f64,
) -> Result<f64> {
    let a = band_envelope(engine, sample_rate, 20.0, sample_rate / 2.0 - 1.0)?;
    let b = band_envelope(reference, sample_rate, 20.0, sample_rate / 2.0 - 1.0)?;
    let n = a.len().min(b.len());
    if n < 8 {
        return Ok(0.0);
    }
    let frame_s = ENVELOPE_HOP as f64 / sample_rate;
    let span = ((max_lag_s / frame_s).round() as isize).max(1);
    let mut best = (0isize, f64::NEG_INFINITY);
    for lag in -span..=span {
        let lo = lag.max(0) as usize;
        let hi = (n as isize + lag.min(0)) as usize;
        if hi <= lo + 4 {
            continue;
        }
        let x: Vec<f64> = (lo..hi).map(|i| a[i]).collect();
        let y: Vec<f64> = (lo..hi).map(|i| b[(i as isize - lag) as usize]).collect();
        let r = pearson(&x, &y);
        if r > best.1 {
            best = (lag, r);
        }
    }
    Ok(best.0 as f64 * frame_s)
}

/// Correlation of the bass, mid and treble energy envelopes.
pub fn band_envelope_correlation(
    engine: &[f32],
    reference: &[f32],
    sample_rate: f64,
) -> Result<BandCorrelation> {
    let mut names = ["", "", ""];
    let mut r = [0.0f64; 3];
    for (i, &(name, lo, hi)) in ENERGY_BANDS.iter().enumerate() {
        let hi = hi.min(sample_rate / 2.0 - 1.0);
        let a = band_envelope(engine, sample_rate, lo, hi)?;
        let b = band_envelope(reference, sample_rate, lo, hi)?;
        names[i] = name;
        r[i] = pearson(&a, &b);
    }
    Ok(BandCorrelation { names, r })
}

// ---------------------------------------------------------------------------
// 5. Release-tail energy delta
// ---------------------------------------------------------------------------

/// Level in the 0.5 s after a note-off, engine minus reference.
#[derive(Clone, Debug)]
pub struct ReleaseDelta {
    /// Note-offs that had a clean window — nothing struck during it.
    pub windows: usize,
    pub mean_signed_db: f64,
    pub mean_abs_db: f64,
    /// Release with the largest absolute difference: `(seconds, signed dB)`.
    pub worst: Option<(f64, f64)>,
}

fn rms_db(signal: &[f32], sample_rate: f64, from_s: f64, to_s: f64) -> Option<f64> {
    let start = (from_s * sample_rate).round().max(0.0) as usize;
    let end = ((to_s * sample_rate).round().max(0.0) as usize).min(signal.len());
    if end <= start {
        return None;
    }
    let mean: f64 = signal[start..end]
        .iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        / (end - start) as f64;
    if mean <= 0.0 {
        return None;
    }
    Some(10.0 * mean.log10())
}

/// Energy delta over the [`RELEASE_WINDOW_S`] after each note-off that nothing
/// interrupts.
///
/// A note-off whose window contains another strike is skipped: what would be
/// measured there is the new note, not the tail of the old one. Phrases that
/// leave no clean window report zero windows, and the metric is then simply
/// not available for them — which is a fact about the phrase, not a failure.
pub fn release_tail_delta(
    engine: &[f32],
    reference: &[f32],
    sample_rate: f64,
    note_offs: &[f64],
    note_ons: &[f64],
) -> ReleaseDelta {
    let mut deltas = Vec::new();
    for &t in note_offs {
        let end = t + RELEASE_WINDOW_S;
        if note_ons.iter().any(|&on| on > t - 0.02 && on < end) {
            continue;
        }
        let (Some(a), Some(b)) = (
            rms_db(engine, sample_rate, t, end),
            rms_db(reference, sample_rate, t, end),
        ) else {
            continue;
        };
        deltas.push((t, a - b));
    }
    if deltas.is_empty() {
        return ReleaseDelta {
            windows: 0,
            mean_signed_db: 0.0,
            mean_abs_db: 0.0,
            worst: None,
        };
    }
    let n = deltas.len() as f64;
    let worst = deltas
        .iter()
        .cloned()
        .fold((0.0f64, 0.0f64), |acc, x| if x.1.abs() > acc.1.abs() { x } else { acc });
    ReleaseDelta {
        windows: deltas.len(),
        mean_signed_db: deltas.iter().map(|d| d.1).sum::<f64>() / n,
        mean_abs_db: deltas.iter().map(|d| d.1.abs()).sum::<f64>() / n,
        worst: Some(worst),
    }
}

// ---------------------------------------------------------------------------
// Level matching
// ---------------------------------------------------------------------------

/// RMS of a mono signal.
pub fn rms(signal: &[f32]) -> f64 {
    if signal.is_empty() {
        return 0.0;
    }
    (signal.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>() / signal.len() as f64).sqrt()
}

fn scale(audio: &Audio, gain: f32) -> Audio {
    Audio {
        sample_rate: audio.sample_rate,
        channels: audio
            .channels
            .iter()
            .map(|c| c.iter().map(|&x| x * gain).collect())
            .collect(),
    }
}

fn peak(audio: &Audio) -> f32 {
    audio
        .channels
        .iter()
        .flat_map(|c| c.iter())
        .fold(0.0f32, |m, &x| m.max(x.abs()))
}

/// Bring a pair to a common level: each scaled so that the RMS of its whole
/// mono sum is [`TARGET_RMS`], then both scaled together if either would clip.
///
/// Whole-phrase RMS rather than the first note's: the two renders are the same
/// music, so any difference in *where* the energy sits over the phrase is a
/// real difference and must survive the normalisation. Scaling both to a fixed
/// target rather than one onto the other keeps the operation symmetric — the
/// engine is not measured against a moved goalpost and the reference is not
/// bent to fit.
pub fn level_match(a: &Audio, b: &Audio) -> Result<(Audio, Audio)> {
    let (ra, rb) = (rms(&a.mono()), rms(&b.mono()));
    if ra <= 0.0 || rb <= 0.0 {
        return Err(Error::Config("a silent render cannot be level-matched".into()));
    }
    let mut a = scale(a, (f64::from(TARGET_RMS) / ra) as f32);
    let mut b = scale(b, (f64::from(TARGET_RMS) / rb) as f32);
    let loudest = peak(&a).max(peak(&b));
    if loudest > PEAK_CEILING {
        let guard = PEAK_CEILING / loudest;
        a = scale(&a, guard);
        b = scale(&b, guard);
    }
    Ok((a, b))
}

// ---------------------------------------------------------------------------
// The whole comparison
// ---------------------------------------------------------------------------

/// Every metric for one pair.
#[derive(Clone, Debug)]
pub struct RealismMetrics {
    pub mel: MultiResDistance,
    pub modulation: ModulationDistance,
    pub attack: AttackDelta,
    pub bands: BandCorrelation,
    pub release: ReleaseDelta,
    /// Alignment check, not a distance: see [`envelope_lag_s`].
    pub lag_s: f64,
}

/// Run every metric over a level-matched, aligned pair.
///
/// `note_ons` and `note_offs` are the performance's own event times, used only
/// to choose the release windows; the onsets the attack metric reads are
/// *detected* on `reference`.
pub fn compare(
    engine: &[f32],
    reference: &[f32],
    sample_rate: f64,
    note_ons: &[f64],
    note_offs: &[f64],
) -> Result<RealismMetrics> {
    let onsets = detect_onsets(reference, sample_rate)?;
    Ok(RealismMetrics {
        mel: multi_res_log_mel_distance(engine, reference, sample_rate)?,
        modulation: modulation_distance(engine, reference, sample_rate)?,
        attack: attack_tonality_delta(engine, reference, sample_rate, &onsets),
        bands: band_envelope_correlation(engine, reference, sample_rate)?,
        release: release_tail_delta(engine, reference, sample_rate, note_offs, note_ons),
        lag_s: envelope_lag_s(engine, reference, sample_rate, 0.05)?,
    })
}

// ---------------------------------------------------------------------------
// The noise floor: the neighbouring velocity layer
// ---------------------------------------------------------------------------

/// The velocity bands of a sampled instrument, in ascending order.
///
/// A sampled piano is a set of *recordings*, and the neighbouring velocity
/// layer of a note is a second, independent recording of the same piano
/// playing very nearly the same music. Rendering a phrase twice — once as
/// written and once with every note moved into the adjacent layer — and
/// measuring the two against each other gives each metric its noise floor:
/// the distance below which "the engine differs from the piano" says nothing,
/// because the piano differs from itself by that much.
#[derive(Clone, Debug)]
pub struct VelocityLayers {
    bands: Vec<(u8, u8)>,
}

impl VelocityLayers {
    /// The distinct velocity bands the library's attack samples declare.
    pub fn from_library(library: &SampleLibrary) -> Result<Self> {
        let mut bands: Vec<(u8, u8)> = library.samples().map(|s| (s.lovel, s.hivel)).collect();
        bands.sort_unstable();
        bands.dedup();
        if bands.len() < 2 {
            return Err(Error::Config(
                "the library has fewer than two velocity layers, so it has no noise floor".into(),
            ));
        }
        Ok(VelocityLayers { bands })
    }

    pub fn bands(&self) -> &[(u8, u8)] {
        &self.bands
    }

    /// Index of the band a velocity falls in.
    pub fn band_of(&self, vel: u8) -> Option<usize> {
        self.bands.iter().position(|&(lo, hi)| vel >= lo && vel <= hi)
    }

    /// A velocity in the band *next to* the one `vel` is in — one louder where
    /// there is one, otherwise one quieter. The value returned is the middle
    /// of that band, which is the most representative velocity in it.
    ///
    /// Velocity 0 (the silent press) is returned unchanged: it is a gesture,
    /// not a dynamic, and moving it would change the phrase.
    pub fn alternate(&self, vel: u8) -> u8 {
        if vel == 0 {
            return 0;
        }
        let Some(i) = self.band_of(vel) else {
            return vel;
        };
        let j = if i + 1 < self.bands.len() { i + 1 } else { i.saturating_sub(1) };
        let (lo, hi) = self.bands[j];
        (((u16::from(lo) + u16::from(hi)) / 2) as u8).max(1)
    }

    /// The same performance with every strike moved into the adjacent layer.
    /// Note-offs keep their release velocity: the reference player ignores it.
    pub fn shift(&self, events: &[TimedEvent]) -> Vec<TimedEvent> {
        events
            .iter()
            .map(|e| match e.event {
                SamplerEvent::NoteOn { key, vel } => TimedEvent::new(
                    e.time_s,
                    SamplerEvent::NoteOn { key, vel: self.alternate(vel) },
                ),
                _ => *e,
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// The phrase set
// ---------------------------------------------------------------------------

/// The phrase set. Bump when a phrase's notes change, so a `REALISM.md` from
/// an older run is recognisably not comparable.
pub const PHRASE_SET_VERSION: u32 = 1;

/// One benchmark phrase: a fixed event list and how long to render it.
#[derive(Clone, Debug)]
pub struct Phrase {
    pub name: &'static str,
    /// What this phrase is for — one line, printed into `REALISM.md`.
    pub description: &'static str,
    pub duration_s: f64,
    pub events: Vec<TimedEvent>,
}

impl Phrase {
    pub fn note_on_times(&self) -> Vec<f64> {
        self.events
            .iter()
            .filter(|e| matches!(e.event, SamplerEvent::NoteOn { vel, .. } if vel > 0))
            .map(|e| e.time_s)
            .collect()
    }

    pub fn note_off_times(&self) -> Vec<f64> {
        self.events
            .iter()
            .filter(|e| matches!(e.event, SamplerEvent::NoteOff { .. }))
            .map(|e| e.time_s)
            .collect()
    }

    pub fn note_count(&self) -> usize {
        self.note_on_times().len()
    }
}

fn note(out: &mut Vec<TimedEvent>, t: f64, key: u8, vel: u8, dur: f64) {
    out.extend(TimedEvent::note(t, key, vel, dur));
}

fn sustain(out: &mut Vec<TimedEvent>, t: f64, down: bool) {
    out.push(TimedEvent::new(
        t,
        SamplerEvent::Sustain(if down { 1.0 } else { 0.0 }),
    ));
}

/// A two-octave C major scale at a moderate dynamic, no pedal. The plainest
/// thing a piano does: if the register balance or the per-note level curve is
/// wrong, it is wrong here and nothing hides it.
pub fn scale_mf() -> Phrase {
    const UP: [u8; 15] = [48, 50, 52, 53, 55, 57, 59, 60, 62, 64, 65, 67, 69, 71, 72];
    let mut events = Vec::new();
    let step = 0.30;
    let mut t = 0.2;
    for &key in UP.iter() {
        note(&mut events, t, key, 72, 0.28);
        t += step;
    }
    for &key in UP.iter().rev().skip(1) {
        let last = key == UP[0];
        note(&mut events, t, key, 72, if last { 1.2 } else { 0.28 });
        t += step;
    }
    Phrase {
        name: "scale_mf",
        description: "two-octave C major scale, mezzo-forte, no pedal",
        duration_s: 12.0,
        events,
    }
}

/// The same three-octave C major arpeggio at five dynamics from pianissimo to
/// fortissimo. What moves between the sweeps is the hammer: contact time,
/// brightness, and the velocity-to-level law.
pub fn arpeggio_dynamics() -> Phrase {
    const UP: [u8; 10] = [48, 52, 55, 60, 64, 67, 72, 76, 79, 84];
    const VELOCITIES: [u8; 5] = [20, 45, 70, 95, 120];
    let mut events = Vec::new();
    let step = 0.14;
    let mut t = 0.2;
    for &vel in &VELOCITIES {
        for &key in UP.iter() {
            note(&mut events, t, key, vel, 0.13);
            t += step;
        }
        for (i, &key) in UP.iter().rev().skip(1).enumerate() {
            let last = i + 2 == UP.len();
            note(&mut events, t, key, vel, if last { 0.5 } else { 0.13 });
            t += step;
        }
        t += 0.30;
    }
    Phrase {
        name: "arpeggio_dynamics",
        description: "three-octave arpeggio swept pp / p / mf / f / ff",
        duration_s: 17.0,
        events,
    }
}

/// Five chords with syncopated pedalling: the pedal comes up on each new
/// harmony and goes back down just after it, so every chord is caught by the
/// damper lift rather than by the key. Sustain, damper timing and whatever
/// rings in sympathy are all in this one.
pub fn chords_pedal() -> Phrase {
    let chords: [(&[u8], u8); 5] = [
        (&[48, 52, 55, 60], 78),
        (&[45, 52, 57, 60], 70),
        (&[41, 48, 53, 57], 74),
        (&[43, 50, 53, 59], 82),
        (&[48, 52, 55, 60, 64], 66),
    ];
    let mut events = Vec::new();
    let mut t = 0.2;
    for (keys, vel) in chords {
        sustain(&mut events, t - 0.02, false);
        for &key in keys {
            note(&mut events, t, key, vel, 1.5);
        }
        sustain(&mut events, t + 0.12, true);
        t += 3.0;
    }
    sustain(&mut events, t + 1.2, false);
    Phrase {
        name: "chords_pedal",
        description: "I–vi–IV–V7–I with syncopated sustain-pedal changes",
        duration_s: 18.0,
        events,
    }
}

/// Short notes up the compass, twice, loud then soft. Every note is 80 ms
/// long and the next is 620 ms away, so each one leaves a clean half second
/// with nothing in it but its own release — the attack and the damper,
/// separated, thirteen times per pass.
pub fn staccato() -> Phrase {
    const KEYS: [u8; 13] = [28, 33, 40, 45, 52, 57, 64, 69, 76, 81, 88, 93, 100];
    let mut events = Vec::new();
    let step = 0.62;
    let mut t = 0.2;
    for &vel in &[95u8, 65] {
        for &key in &KEYS {
            note(&mut events, t, key, vel, 0.08);
            t += step;
        }
    }
    Phrase {
        name: "staccato",
        description: "80 ms notes across seven octaves, forte then piano, no pedal",
        duration_s: 18.0,
        events,
    }
}

/// Alberti bass in sixteenths under a scalar right hand. Everything here is a
/// repetition: the same four keys sixteen times a bar. A model whose repeated
/// strikes are too even, or whose re-strike into a still-ringing string is
/// wrong, has nowhere to hide.
pub fn alberti_fast() -> Phrase {
    // (bass, third, fifth) of the bar's harmony, and the right hand's eight
    // eighth-notes over it.
    let bars: [([u8; 3], [u8; 8]); 6] = [
        ([48, 52, 55], [72, 74, 76, 77, 79, 77, 76, 74]),
        ([47, 50, 55], [76, 74, 72, 74, 71, 74, 79, 74]),
        ([48, 52, 55], [72, 74, 76, 77, 79, 81, 83, 84]),
        ([47, 50, 55], [83, 81, 79, 77, 76, 74, 72, 71]),
        ([48, 52, 55], [72, 76, 79, 84, 79, 76, 72, 76]),
        ([47, 50, 55], [74, 71, 74, 79, 74, 71, 72, 72]),
    ];
    let sixteenth = 0.125;
    let mut events = Vec::new();
    let mut t = 0.2;
    for (harmony, melody) in bars {
        let bar_start = t;
        // Left hand: bass, fifth, third, fifth — four times a bar.
        let pattern = [harmony[0], harmony[2], harmony[1], harmony[2]];
        for i in 0..16 {
            note(&mut events, t, pattern[i % 4], 70, 0.115);
            t += sixteenth;
        }
        for (i, &key) in melody.iter().enumerate() {
            let at = bar_start + i as f64 * 2.0 * sixteenth;
            let last = i == 7;
            note(&mut events, at, key, 85, if last { 0.5 } else { 0.23 });
        }
    }
    Phrase {
        name: "alberti_fast",
        description: "sixteenth-note Alberti bass under an eighth-note melody, six bars",
        duration_s: 14.0,
        events,
    }
}

/// Eight bars of the Ode to Joy theme (Beethoven, Symphony No. 9 — public
/// domain), harmonised with a bass note, a left-hand chord and a pedal change
/// on every harmony. The only phrase with three simultaneous textures, and
/// therefore the only one where masking between them can go wrong.
pub fn excerpt() -> Phrase {
    const C: [u8; 4] = [36, 48, 52, 55];
    const G: [u8; 4] = [43, 47, 50, 55];
    const G7: [u8; 4] = [43, 47, 53, 55];
    // (bar-relative onset in beats, chord)
    let harmony: [(f64, [u8; 4]); 13] = [
        (0.0, C),
        (4.0, C),
        (6.0, G),
        (8.0, C),
        (10.0, G),
        (12.0, G7),
        (16.0, C),
        (20.0, C),
        (22.0, G),
        (24.0, C),
        (26.0, G),
        (28.0, G7),
        (30.0, C),
    ];
    // (onset in beats, key, length in beats)
    let melody: [(f64, u8, f64); 30] = [
        (0.0, 64, 1.0),
        (1.0, 64, 1.0),
        (2.0, 65, 1.0),
        (3.0, 67, 1.0),
        (4.0, 67, 1.0),
        (5.0, 65, 1.0),
        (6.0, 64, 1.0),
        (7.0, 62, 1.0),
        (8.0, 60, 1.0),
        (9.0, 60, 1.0),
        (10.0, 62, 1.0),
        (11.0, 64, 1.0),
        (12.0, 64, 1.5),
        (13.5, 62, 0.5),
        (14.0, 62, 2.0),
        (16.0, 64, 1.0),
        (17.0, 64, 1.0),
        (18.0, 65, 1.0),
        (19.0, 67, 1.0),
        (20.0, 67, 1.0),
        (21.0, 65, 1.0),
        (22.0, 64, 1.0),
        (23.0, 62, 1.0),
        (24.0, 60, 1.0),
        (25.0, 60, 1.0),
        (26.0, 62, 1.0),
        (27.0, 64, 1.0),
        (28.0, 62, 1.5),
        (29.5, 60, 0.5),
        (30.0, 60, 3.0),
    ];
    let beat = 0.5;
    let start = 0.2;
    let mut events = Vec::new();
    for (i, (at, chord)) in harmony.iter().enumerate() {
        let t = start + at * beat;
        let until = harmony
            .get(i + 1)
            .map(|(next, _)| start + next * beat)
            .unwrap_or(start + 33.0 * beat);
        sustain(&mut events, (t - 0.02).max(0.0), false);
        for (j, &key) in chord.iter().enumerate() {
            let vel = if j == 0 { 68 } else { 58 };
            note(&mut events, t, key, vel, (until - t - 0.05).max(0.1));
        }
        sustain(&mut events, t + 0.10, true);
    }
    for (at, key, len) in melody {
        note(&mut events, start + at * beat, key, 88, (len * beat - 0.05).max(0.08));
    }
    sustain(&mut events, start + 34.0 * beat, false);
    Phrase {
        name: "excerpt",
        description: "Ode to Joy (Beethoven, public domain), harmonised, pedalled",
        duration_s: 19.0,
        events,
    }
}

/// The whole phrase set, in the order `REALISM.md` reports it.
pub fn phrase_set() -> Vec<Phrase> {
    vec![
        scale_mf(),
        arpeggio_dynamics(),
        chords_pedal(),
        staccato(),
        alberti_fast(),
        excerpt(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f64 = 48_000.0;

    fn sine(freq: f64, amplitude: f64, sample_rate: f64, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = i as f64 / sample_rate;
                (amplitude * (2.0 * PI * freq * t).sin()) as f32
            })
            .collect()
    }

    /// A decaying tone struck every `period` seconds — a stand-in for a phrase
    /// with a rhythm, so the envelope-domain metrics have something to read.
    fn plucks(freq: f64, period: f64, t60: f64, seconds: f64, sample_rate: f64) -> Vec<f32> {
        let n = (seconds * sample_rate) as usize;
        let mut out = vec![0.0f32; n];
        let stride = (period * sample_rate) as usize;
        let decay = (10.0f64).powf(-3.0 / (t60 * sample_rate));
        let mut start = 0usize;
        while start < n {
            let mut a = 1.0f64;
            for (i, slot) in out.iter_mut().enumerate().skip(start) {
                *slot += (a * (2.0 * PI * freq * (i - start) as f64 / sample_rate).sin()) as f32;
                a *= decay;
                if a < 1e-6 {
                    break;
                }
            }
            start += stride;
        }
        out
    }

    fn audio(mono: &[f32]) -> Audio {
        Audio::new(48_000, vec![mono.to_vec(), mono.to_vec()]).unwrap()
    }

    #[test]
    fn the_mel_scale_is_its_own_inverse() {
        for &hz in &[20.0, 100.0, 440.0, 1000.0, 4186.0, 16_000.0] {
            let round = mel_to_hz(hz_to_mel(hz));
            assert!((round - hz).abs() < 1e-6, "{hz} -> {round}");
        }
        // Monotone, and the scale really is compressive above 1 kHz.
        assert!(hz_to_mel(200.0) - hz_to_mel(100.0) > hz_to_mel(8_200.0) - hz_to_mel(8_100.0));
    }

    #[test]
    fn every_mel_band_carries_energy_at_every_resolution() {
        for &window in &MULTI_RES_WINDOWS {
            let bank = MelBank::new(MEL_BANDS, window, SR, MEL_F_MIN, MEL_F_MAX).unwrap();
            for b in 0..MEL_BANDS {
                assert!(
                    bank.band_weight(b) > 0.0,
                    "band {b} of a {window}-sample window collects nothing"
                );
            }
            assert_eq!(bank.centres_hz().len(), MEL_BANDS);
            // Centres ascend and stay inside the declared range.
            for w in bank.centres_hz().windows(2) {
                assert!(w[1] > w[0]);
            }
            assert!(bank.centres_hz()[0] > MEL_F_MIN);
            assert!(*bank.centres_hz().last().unwrap() < MEL_F_MAX);
        }
    }

    #[test]
    fn a_tone_lands_in_the_band_that_contains_it() {
        let signal = sine(1000.0, 0.5, SR, 48_000);
        let spec = mel_spectrogram(&signal, SR, 4096, 1024, MEL_BANDS, MEL_F_MIN, MEL_F_MAX).unwrap();
        let mid = spec.frames.len() / 2;
        let (best, _) = spec.frames[mid]
            .iter()
            .enumerate()
            .fold((0usize, 0.0f64), |acc, (i, &e)| if e > acc.1 { (i, e) } else { acc });
        let centre = spec.centres_hz[best];
        assert!(
            (centre - 1000.0).abs() < 120.0,
            "1 kHz landed in the band centred at {centre} Hz"
        );
        // And the level is the amplitude the STFT calibration promises.
        let db = 10.0 * spec.frames[mid][best].log10();
        assert!((db - 20.0 * 0.5f64.log10()).abs() < 1.5, "{db} dB");
    }

    #[test]
    fn a_signal_against_itself_has_no_distance() {
        let x = plucks(220.0, 0.4, 1.5, 4.0, SR);
        let m = compare(&x, &x, SR, &[0.0, 0.4, 0.8], &[3.0]).unwrap();
        assert_eq!(m.mel.mean, 0.0);
        for r in &m.mel.resolutions {
            assert_eq!(r.mean, 0.0);
        }
        assert!(m.modulation.mean < 1e-9, "{}", m.modulation.mean);
        assert_eq!(m.attack.mean_abs_db, 0.0);
        for r in m.bands.r {
            assert!((r - 1.0).abs() < 1e-9, "{r}");
        }
        assert!(m.release.windows > 0);
        assert_eq!(m.release.mean_abs_db, 0.0);
        assert_eq!(m.lag_s, 0.0);
    }

    #[test]
    fn the_alignment_check_finds_a_shift_that_was_put_there() {
        let x = plucks(220.0, 0.4, 1.0, 4.0, SR);
        for &shift_ms in &[-20.0f64, -5.0, 0.0, 5.0, 20.0] {
            let shift = (shift_ms.abs() / 1000.0 * SR) as usize;
            let (late, early): (Vec<f32>, Vec<f32>) = if shift_ms >= 0.0 {
                // `late` starts with silence, so it lags.
                (
                    std::iter::repeat(0.0).take(shift).chain(x.iter().copied()).collect(),
                    x.clone(),
                )
            } else {
                (
                    x.clone(),
                    std::iter::repeat(0.0).take(shift).chain(x.iter().copied()).collect(),
                )
            };
            let lag = envelope_lag_s(&late, &early, SR, 0.05).unwrap();
            let frame = ENVELOPE_HOP as f64 / SR;
            assert!(
                (lag - shift_ms / 1000.0).abs() <= frame,
                "{shift_ms} ms read as {} ms",
                lag * 1000.0
            );
        }
    }

    #[test]
    fn a_level_change_alone_is_not_a_distance() {
        let x = plucks(220.0, 0.4, 1.5, 4.0, SR);
        let loud: Vec<f32> = x.iter().map(|&v| v * 2.0).collect();
        let (a, b) = level_match(&audio(&x), &audio(&loud)).unwrap();
        let (ma, mb) = (a.mono(), b.mono());
        // The match is exact, so every metric must read zero.
        assert!((rms(&ma) - rms(&mb)).abs() < 1e-9);
        let d = multi_res_log_mel_distance(&ma, &mb, SR).unwrap();
        assert!(d.mean < 1e-4, "{}", d.mean);
        // ...and it really did equalise them rather than leaving them alone.
        // Unmatched, the pair differs by 6 dB wherever both are above the
        // floor, which over a decaying phrase averages to rather less.
        let raw = multi_res_log_mel_distance(&x, &loud, SR).unwrap();
        assert!(raw.mean > 1.0, "{}", raw.mean);
    }

    #[test]
    fn the_mel_distance_grows_with_the_size_of_the_change() {
        let base = plucks(220.0, 0.5, 1.5, 4.0, SR);
        let mut previous = 0.0;
        for &extra_db in &[-40.0, -20.0, -10.0, 0.0f64] {
            let amp = 10.0f64.powf(extra_db / 20.0);
            let added = sine(3_000.0, amp * 0.2, SR, base.len());
            let perturbed: Vec<f32> = base.iter().zip(added.iter()).map(|(&a, &b)| a + b).collect();
            let d = multi_res_log_mel_distance(&perturbed, &base, SR).unwrap();
            assert!(
                d.mean > previous,
                "{extra_db} dB of added tone read {} against {previous}",
                d.mean
            );
            previous = d.mean;
        }
        assert!(previous > 1.0);
    }

    #[test]
    fn the_worst_band_is_the_one_that_was_changed() {
        let base = plucks(220.0, 0.5, 1.5, 4.0, SR);
        let added = sine(3_000.0, 0.15, SR, base.len());
        let perturbed: Vec<f32> = base.iter().zip(added.iter()).map(|(&a, &b)| a + b).collect();
        let d = log_mel_diff(&perturbed, &base, SR, 4096, 1024, MEL_BANDS).unwrap();
        let (hz, db) = d.worst_band();
        assert!((hz - 3_000.0).abs() < 400.0, "worst band at {hz} Hz");
        assert!(db > 5.0, "{db} dB");
        // And the engine side is the one with more energy there.
        let k = d.centres_hz.iter().position(|&c| c >= 2_800.0).unwrap();
        assert!(d.signed_per_band[k] > 0.0);
    }

    #[test]
    fn the_modulation_distance_sees_a_tremolo_that_is_not_there() {
        let n = (5.0 * SR) as usize;
        let plain = sine(1_000.0, 0.3, SR, n);
        let modulated: Vec<f32> = plain
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let t = i as f64 / SR;
                (f64::from(x) * (1.0 + 0.5 * (2.0 * PI * 4.0 * t).sin())) as f32
            })
            .collect();
        let same = modulation_distance(&modulated, &modulated, SR).unwrap();
        assert!(same.mean < 1e-9, "{}", same.mean);

        let d = modulation_distance(&modulated, &plain, SR).unwrap();
        // Fifteen of the sixteen bands are empty in both signals and read an
        // exact zero, so the mean is the one band's answer divided by sixteen.
        assert!(d.mean > 1.0, "{}", d.mean);
        // It is found at 4 Hz and in the band that holds the carrier.
        let (rate, _) = d.worst_rate();
        assert!((rate - 4.0).abs() < 2.5, "worst modulation rate {rate} Hz");
        let (hz, band_db) = d.worst_band();
        assert!((hz - 1_000.0).abs() < 600.0, "worst band {hz} Hz");
        assert!(band_db > 5.0, "the carrier band read only {band_db} dB");
        // Bands with nothing in them agree exactly rather than reading the
        // window's leakage: that is what the level floor is for.
        let quiet = d.per_band.iter().filter(|&&v| v == 0.0).count();
        assert!(quiet >= MODULATION_BANDS - 4, "{:?}", d.per_band);
    }

    #[test]
    fn an_onset_is_found_where_each_note_starts() {
        let expected: Vec<f64> = (0..6).map(|i| 0.5 + i as f64 * 0.6).collect();
        let n = (4.5 * SR) as usize;
        let mut signal = vec![0.0f32; n];
        let decay = (10.0f64).powf(-3.0 / (0.35 * SR));
        for &t in &expected {
            let start = (t * SR) as usize;
            let mut a = 1.0f64;
            for (i, slot) in signal.iter_mut().enumerate().skip(start) {
                *slot += (0.4 * a * (2.0 * PI * 440.0 * (i - start) as f64 / SR).sin()) as f32;
                a *= decay;
                if a < 1e-5 {
                    break;
                }
            }
        }
        let found = detect_onsets(&signal, SR).unwrap();
        assert_eq!(found.len(), expected.len(), "found {found:?}");
        for (f, e) in found.iter().zip(expected.iter()) {
            assert!((f - e).abs() < 0.012, "onset at {f} s, expected {e} s");
        }
    }

    #[test]
    fn the_attack_delta_reads_how_tonal_the_attack_is() {
        let n = (0.5 * SR) as usize;
        let tone = sine(440.0, 0.3, SR, n);
        // Same tone with a short noise burst on the attack.
        let mut noisy = tone.clone();
        let mut state = 0x1234_5678_9abc_def0u64;
        for (i, slot) in noisy.iter_mut().take((0.02 * SR) as usize).enumerate() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r = ((state >> 33) as f64 / f64::from(u32::MAX >> 1)) - 1.0;
            let fade = 1.0 - i as f64 / (0.02 * SR);
            *slot += (0.3 * r * fade) as f32;
        }
        let onsets = [0.0];
        let d = attack_tonality_delta(&tone, &noisy, SR, &onsets);
        assert_eq!(d.onsets, 1);
        assert!(
            d.mean_signed_db > 6.0,
            "a pure attack should read more tonal than a noisy one, got {}",
            d.mean_signed_db
        );
        // Symmetric.
        let back = attack_tonality_delta(&noisy, &tone, SR, &onsets);
        assert!((back.mean_signed_db + d.mean_signed_db).abs() < 1e-9);
    }

    #[test]
    fn the_rise_time_is_how_long_the_attack_takes_to_reach_its_loudest() {
        let n = (0.5 * SR) as usize;
        let instant = sine(440.0, 0.3, SR, n);
        // The same note ramped in over 20 ms.
        let ramped: Vec<f32> = instant
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let t = i as f64 / SR;
                (f64::from(x) * (t / 0.020).min(1.0)) as f32
            })
            .collect();
        assert!(attack_rise_s(&instant, SR, 0.0).unwrap() < 0.003);
        let slow = attack_rise_s(&ramped, SR, 0.0).unwrap();
        assert!((slow - 0.020).abs() < 0.003, "{slow} s");

        let d = attack_tonality_delta(&instant, &ramped, SR, &[0.0]);
        assert!(d.rise_s.0 < 0.003 && (d.rise_s.1 - 0.020).abs() < 0.003, "{:?}", d.rise_s);
    }

    #[test]
    fn the_band_correlation_falls_when_one_register_moves_differently() {
        let bass = plucks(80.0, 0.7, 1.2, 4.0, SR);
        let treble = plucks(3_000.0, 0.5, 0.6, 4.0, SR);
        let together: Vec<f32> = bass.iter().zip(treble.iter()).map(|(&a, &b)| a + b).collect();
        // Same bass, treble struck on a different grid.
        let other_treble = plucks(3_000.0, 0.23, 0.6, 4.0, SR);
        let scrambled: Vec<f32> = bass
            .iter()
            .zip(other_treble.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        let c = band_envelope_correlation(&scrambled, &together, SR).unwrap();
        assert!(c.r[0] > 0.95, "bass should still agree: {}", c.r[0]);
        assert!(c.r[2] < 0.7, "treble should not: {}", c.r[2]);
        assert_eq!(c.worst().0, "treble");
    }

    #[test]
    fn the_release_delta_reads_the_level_of_the_tail() {
        let n = (3.0 * SR) as usize;
        // A note that stops at 1.0 s, and the same note 6 dB louder in the
        // half second after it.
        let mut quiet = vec![0.0f32; n];
        let mut loud = vec![0.0f32; n];
        for i in 0..n {
            let t = i as f64 / SR;
            let v = 0.2 * (2.0 * PI * 500.0 * t).sin();
            quiet[i] = v as f32;
            loud[i] = if t >= 1.0 { (v * 2.0) as f32 } else { v as f32 };
        }
        let d = release_tail_delta(&loud, &quiet, SR, &[1.0], &[0.0]);
        assert_eq!(d.windows, 1);
        assert!((d.mean_signed_db - 6.02).abs() < 0.05, "{}", d.mean_signed_db);
        // A note-on inside the window disqualifies it.
        let blocked = release_tail_delta(&loud, &quiet, SR, &[1.0], &[0.0, 1.3]);
        assert_eq!(blocked.windows, 0);
    }

    #[test]
    fn the_phrase_set_is_what_it_claims_to_be() {
        let set = phrase_set();
        assert_eq!(set.len(), 6);
        let mut names: Vec<&str> = set.iter().map(|p| p.name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), set.len(), "phrase names must be unique");

        for phrase in &set {
            assert!(
                (8.0..=20.0).contains(&phrase.duration_s),
                "{} is {} s",
                phrase.name,
                phrase.duration_s
            );
            assert!(phrase.note_count() >= 8, "{} has {} notes", phrase.name, phrase.note_count());

            let mut held: Vec<u8> = Vec::new();
            let mut ordered = phrase.events.clone();
            ordered.sort_by(|a, b| a.time_s.partial_cmp(&b.time_s).unwrap());
            for e in &ordered {
                assert!(e.time_s >= 0.0);
                assert!(
                    e.time_s <= phrase.duration_s - 0.4,
                    "{}: an event at {} s in a {} s render",
                    phrase.name,
                    e.time_s,
                    phrase.duration_s
                );
                match e.event {
                    SamplerEvent::NoteOn { key, vel } => {
                        assert!((21..=108).contains(&key), "{}: key {key}", phrase.name);
                        assert!(vel > 0 && vel <= 127, "{}: velocity {vel}", phrase.name);
                        held.push(key);
                    }
                    SamplerEvent::NoteOff { key, .. } => {
                        let at = held.iter().position(|&k| k == key);
                        assert!(at.is_some(), "{}: key {key} released without a strike", phrase.name);
                        held.remove(at.unwrap());
                    }
                    SamplerEvent::Sustain(v) => {
                        // Half-pedal is not comparable: the reference player
                        // reads CC 64 as a switch. Only the stops are used.
                        assert!(v == 0.0 || v == 1.0, "{}: half pedal {v}", phrase.name);
                    }
                    other => panic!("{}: {other:?} is not comparable between the two players", phrase.name),
                }
            }
            assert!(held.is_empty(), "{}: {held:?} never released", phrase.name);
        }
    }

    #[test]
    fn the_pedalled_phrases_are_the_ones_that_use_the_pedal() {
        let pedalled: Vec<&str> = phrase_set()
            .iter()
            .filter(|p| p.events.iter().any(|e| matches!(e.event, SamplerEvent::Sustain(_))))
            .map(|p| p.name)
            .collect();
        assert_eq!(pedalled, vec!["chords_pedal", "excerpt"]);
    }

    #[test]
    fn the_staccato_phrase_leaves_a_clean_release_window_after_every_note() {
        let phrase = staccato();
        let ons = phrase.note_on_times();
        let offs = phrase.note_off_times();
        let clean = offs
            .iter()
            .filter(|&&t| !ons.iter().any(|&on| on > t - 0.02 && on < t + RELEASE_WINDOW_S))
            .count();
        assert_eq!(clean, offs.len(), "{clean} clean of {}", offs.len());
    }

    #[test]
    fn the_alternate_velocity_is_always_a_different_layer() {
        let layers = VelocityLayers {
            bands: vec![(1, 26), (27, 34), (35, 36), (37, 43), (44, 46), (121, 127)],
        };
        for vel in 1..=127u8 {
            let alt = layers.alternate(vel);
            match (layers.band_of(vel), layers.band_of(alt)) {
                (Some(a), Some(b)) => assert_ne!(a, b, "velocity {vel} -> {alt} stayed in band {a}"),
                (None, _) => assert_eq!(alt, vel, "an unmapped velocity must be left alone"),
                (Some(_), None) => panic!("velocity {vel} -> {alt}, which is in no layer"),
            }
        }
        // The silent press is a gesture, not a dynamic.
        assert_eq!(layers.alternate(0), 0);
        // The top layer has to borrow from below.
        assert!(layers.alternate(125) < 121);
    }

    #[test]
    fn shifting_a_phrase_moves_the_strikes_and_nothing_else() {
        let layers = VelocityLayers {
            bands: vec![(1, 63), (64, 127)],
        };
        let phrase = chords_pedal();
        let shifted = layers.shift(&phrase.events);
        assert_eq!(shifted.len(), phrase.events.len());
        for (a, b) in phrase.events.iter().zip(shifted.iter()) {
            assert_eq!(a.time_s, b.time_s);
            match (a.event, b.event) {
                (SamplerEvent::NoteOn { key: k1, vel: v1 }, SamplerEvent::NoteOn { key: k2, vel: v2 }) => {
                    assert_eq!(k1, k2);
                    assert_ne!(v1, v2);
                }
                (x, y) => assert_eq!(x, y),
            }
        }
    }
}


