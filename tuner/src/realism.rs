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
//! ## Columns A and B: what those five are blind to
//!
//! Every metric above is a functional of *energy*, and `docs/history/FUNDAMENTALS.md` Part II
//! is the argument that the percept the instrument is still failing on is not
//! one. A 4-cent frequency modulation of a resolved partial is several times
//! detection threshold and changes a mel feature vector by exactly nothing (the
//! bands are two semitones wide); a beat 15 dB deep at a fixed rate is thirty
//! times threshold and sits *below* the modulation metric's own 0.5 Hz low edge.
//! M3 is the demonstration: the modulation column improved 5.76 → 3.84 while the
//! listener's verdict got worse, because an aggregate that rewards the presence
//! of modulation rewards a metronome.
//!
//! [`motion_columns`] is Part II §II.3's answer, promoted from the forensics that
//! found the fault. Four numbers over sixteen key × partial cells at three
//! velocities ([`MOTION_KEYS`], [`MOTION_PARTIALS`], [`MOTION_VELOCITIES`]),
//! each a **per-cell mismatch against the recording** rather than a pooled mean,
//! because the mean is what hid this: over the same cells the engine's jitter
//! averaged 1.27 cents against the recording's 1.50 while being 33x too still at
//! C4 k=1 and 4.5x too spiky at A4 k=1.
//!
//! * **A1 `IF mismatch`** ([`A1_GATE`]) — symmetric, so too dead fails as loudly
//!   as too spiky.
//! * **A2 `IF placement`** ([`A2_GATE`]) — does the wobble ride the loud part of
//!   the partial, as the recording's does, or spike at the null of a beat, which
//!   is all a sum of free-running sinusoids can do.
//! * **B1 `beat-depth error`** ([`B1_GATE_DB`]).
//! * **B2 `velocity coherence`** ([`B2_GATE`]) — the column with the physics in
//!   it, and the only one of the four that anything in this repository can move.
//!
//! ## The stereo columns, and what this header used to say
//!
//! The five metrics above and columns A and B are all computed on the **mono**
//! sum, and this header used to give the reason: the engine places keys in the
//! stereo field by its own rule and a recording carries the microphone pair it
//! was made with, so a stereo distance would mostly measure that disagreement.
//! That was true and it was the wrong conclusion. `DECISIONS.md` 314 measured
//! the disagreement and it is the **largest single difference in the chain
//! experiment**: the recording's channels are +0.945 correlated below 125 Hz
//! and near zero above, the engine's are −0.577 in the bass and +0.964 in the
//! treble — exactly inverted, in every band — and nothing on the scoreboard
//! could see it. A difference nothing scores is a difference nothing can
//! regress, so item 317 (a) asks for the loss to get a stereo term *before* the
//! two-microphone geometry of `PHYSICS.md` §8 is built.
//!
//! [`stereo_image`] is that term: per band, the interchannel correlation at lag
//! zero, the peak |r| over ±5 ms and where it sits, and a mid/side energy
//! ratio. [`stereo_columns`] scores it the way everything else here is scored,
//! against a floor made of the reference disagreeing with itself. These columns
//! are marked **STEREO** wherever they are printed, because every other number
//! in this module is a mono sum and mixing the two would be a lie about what
//! moved.
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
use crate::motion::{partial_motion, Motion, Spectrum, IF_FLOOR_CENTS};
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

/// How far **back** from a detected onset each signal's own strike is looked
/// for before its attack window is placed, in seconds.
///
/// `DECISIONS.md` 338. The onsets this metric reads are detected on the
/// *reference*, and the reference is a sampler: it plays each recording from
/// the file's own start, so what reaches the ear is late by however much silence
/// there was between the engineer's trigger and the hammer. Measured over the
/// 30 recorded keys at five velocities, that lead-in is **+19 ms at the median,
/// +27 ms on average and +112 ms at the worst key** — so an engine read at the
/// reference's onset is read a fifth of the way through its own attack and, at
/// the tail of that distribution, entirely past it.
///
/// `estimate::melody` has always windowed each side on its own strike for
/// exactly this reason (`ONSET_SEARCH_S`, and its own note: "a gate whose two
/// sides are windowed by different rules is not comparing them"). This is the
/// same device on the phrase board. 120 ms covers the whole measured
/// distribution; the search is additionally clamped so that it can never reach
/// back past the midpoint to the previous onset, which is what keeps a fast
/// phrase from finding the note before.
pub const ATTACK_SEARCH_BACK_S: f64 = 0.120;

/// How far **forward** the same search looks. Small: a detected onset is
/// already at or after the strike, so the forward half only absorbs the flux
/// detector's own frame quantisation.
pub const ATTACK_SEARCH_FORWARD_S: f64 = 0.010;

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
    pub fn new(
        bands: usize,
        fft_size: usize,
        sample_rate: f64,
        f_min: f64,
        f_max: f64,
    ) -> Result<Self> {
        if bands < 2 {
            return Err(Error::Config(format!(
                "{bands} mel bands is not a filterbank"
            )));
        }
        if fft_size < 4 || fft_size % 2 != 0 {
            return Err(Error::Config(format!(
                "fft size {fft_size} cannot be a mel bank"
            )));
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
    let a = mel_spectrogram(
        engine,
        sample_rate,
        window,
        hop,
        bands,
        MEL_F_MIN,
        MEL_F_MAX,
    )?;
    let b = mel_spectrogram(
        reference,
        sample_rate,
        window,
        hop,
        bands,
        MEL_F_MIN,
        MEL_F_MAX,
    )?;
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
        for (k, (band, signed)) in per_band
            .iter_mut()
            .zip(signed_per_band.iter_mut())
            .enumerate()
        {
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
        return Err(Error::Config(
            "a silent signal has no modulation spectrum".into(),
        ));
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
        out.push(
            row.iter()
                .map(|&m| {
                    if m > 0.0 {
                        20.0 * m.log10()
                    } else {
                        f64::NEG_INFINITY
                    }
                })
                .collect(),
        );
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
    let db = |e: f64| {
        if e > 0.0 {
            (10.0 * e.log10()).max(floor)
        } else {
            floor
        }
    };

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
        let var = window_slice
            .iter()
            .map(|&x| (x - mean) * (x - mean))
            .sum::<f64>()
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
        let mean: f64 = signal[s..e]
            .iter()
            .map(|&x| f64::from(x) * f64::from(x))
            .sum::<f64>()
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

/// Where a signal's own strike is, near `near_s`: the largest rise in a 1 ms
/// RMS envelope over `[near_s - back_s, near_s + forward_s]`.
///
/// A rise rather than a level, because a note is struck into the tail of the
/// one before it and any threshold on level would fire on that tail; a piano
/// strike is the one thing in the window that goes up. The rise is taken over
/// three milliseconds because one is inside the hammer's own contact time and a
/// single block is noise.
///
/// This is the primitive both per-note windows in this repository are placed
/// with — [`attack_tonality_delta`] here and `estimate::melody::note_onset`,
/// which delegates to [`strike_near_banded`] — so that the two boards cannot
/// drift apart on the one question of *where a note starts*.
pub fn strike_near(
    signal: &[f32],
    sample_rate: f64,
    near_s: f64,
    back_s: f64,
    forward_s: f64,
) -> f64 {
    strike_near_banded(signal, sample_rate, near_s, back_s, forward_s, 1.0, 0.0)
}

/// [`strike_near`] with the two things that decide whether it finds a hammer
/// exposed: the envelope's block length and the band the envelope is taken
/// over.
///
/// **Both defaults are wrong for a low note and the reason is arithmetic**
/// (`DECISIONS.md` 452). An RMS over one millisecond of a 261.6 Hz tone is not
/// an envelope: the period is 3.8 ms, so the block covers a quarter of a cycle
/// and the "envelope" swings by whatever the waveform does inside it. The rise
/// this function maximises is then a rise of the *carrier*, not of the note,
/// and on a note whose attack is soft the largest such rise lands wherever the
/// ripple happens to be steepest — measured at up to **+73 ms** past C4's own
/// hammer on the engine's melody render and **+42 ms** on the recording's.
/// Lengthening the block alone does not fix it, because a longer block also
/// blurs the attack it is looking for: at 3 ms the engine's C4 still misses by
/// +16 and every other note acquires an 8-11 ms bias.
///
/// What fixes it is asking the *band the hammer is in*. A strike is the one
/// broadband event in a melody; a sounding note's tail has almost nothing above
/// 2 kHz, and neither does the low-frequency ripple that produces the miss. A
/// 2 ms envelope of the signal high-passed at 2 kHz places every note of both
/// sides of the melody render within **6 ms** of its grid time, worst case,
/// against 73 ms for the shipped detector — and the residual few milliseconds
/// are the search's own convention (it returns the *start* of the three-block
/// span that rose the most), which is identical on both sides and so cancels.
///
/// `highpass_hz <= 0` skips the filter, which is what keeps [`strike_near`]
/// bit-identical for the phrase board's `attack` column.
pub fn strike_near_banded(
    signal: &[f32],
    sample_rate: f64,
    near_s: f64,
    back_s: f64,
    forward_s: f64,
    block_ms: f64,
    highpass_hz: f64,
) -> f64 {
    let block = ((sample_rate * block_ms * 1e-3) as usize).max(1);
    let from = ((((near_s - back_s) * sample_rate) as isize).max(0)) as usize;
    let to = ((((near_s + forward_s) * sample_rate) as usize) + block).min(signal.len());
    if from + 4 * block >= to {
        return near_s;
    }
    // Filtered over the search span only, and with the filter's own settling
    // taken before the span starts, so that a per-note call costs a couple of
    // hundred milliseconds of biquad rather than a pass over the phrase.
    let settle = if highpass_hz > 0.0 {
        ((0.01 * sample_rate) as usize).min(from)
    } else {
        0
    };
    let span = &signal[from - settle..to];
    let filtered;
    let span: &[f32] = if highpass_hz > 0.0 {
        filtered = highpassed(span, sample_rate, highpass_hz);
        &filtered[settle..]
    } else {
        span
    };
    let envelope: Vec<f64> = span
        .chunks(block)
        .map(|c| {
            (c.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>() / c.len() as f64).sqrt()
        })
        .collect();
    let step = 3usize;
    let mut best = (0usize, f64::MIN);
    for i in 0..envelope.len().saturating_sub(step) {
        let rise = envelope[i + step] - envelope[i];
        if rise > best.1 {
            best = (i, rise);
        }
    }
    from as f64 / sample_rate + best.0 as f64 * block as f64 / sample_rate
}

/// Two second-order Butterworth high-pass sections in cascade, forward only.
///
/// Forward only on purpose: a zero-phase pass would smear the attack backwards
/// in time, which is the one thing an onset detector must not do. The phase
/// this adds at the cutoff is a fraction of a block.
///
/// **Two sections and not one**, because one is not enough for the job by a
/// margin this repository can state: a single section is 12 dB/octave, which
/// puts C4's fundamental only 35 dB down at a 2 kHz corner, and the note's own
/// 2-6 kHz content sits 26 dB under that fundamental — so a third of what the
/// "high band" envelope would be reading is the fundamental leaking through it,
/// swelling on the fundamental's own schedule. Cascading the section takes the
/// leak to 70 dB down and the band to what it says it is.
fn highpassed(x: &[f32], sample_rate: f64, cutoff: f64) -> Vec<f32> {
    let once = highpass_section(x, sample_rate, cutoff);
    highpass_section(&once, sample_rate, cutoff)
}

fn highpass_section(x: &[f32], sample_rate: f64, cutoff: f64) -> Vec<f32> {
    let w = (PI * cutoff / sample_rate).tan();
    let k = std::f64::consts::SQRT_2;
    let norm = 1.0 / (1.0 + k * w + w * w);
    let (b0, b1, b2) = (norm, -2.0 * norm, norm);
    let a1 = 2.0 * (w * w - 1.0) * norm;
    let a2 = (1.0 - k * w + w * w) * norm;
    let (mut x1, mut x2, mut y1, mut y2) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    x.iter()
        .map(|&s| {
            let x0 = f64::from(s);
            let y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
            x2 = x1;
            x1 = x0;
            y2 = y1;
            y1 = y0;
            y0 as f32
        })
        .collect()
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
            signal[s..e]
                .iter()
                .map(|&x| f64::from(x) * f64::from(x))
                .sum::<f64>()
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
    for (i, &t) in onsets.iter().enumerate() {
        // Each side is windowed on **its own** strike, not on the onset the
        // detector found in the reference (`DECISIONS.md` 338). The search
        // reaches back at most to the midpoint of the gap to the previous
        // onset, so a fast phrase cannot find the note before.
        let back = match i.checked_sub(1).and_then(|j| onsets.get(j)) {
            Some(&previous) => ATTACK_SEARCH_BACK_S.min(0.5 * (t - previous).max(0.0)),
            None => ATTACK_SEARCH_BACK_S,
        };
        let et = strike_near(engine, sample_rate, t, back, ATTACK_SEARCH_FORWARD_S);
        let rt = strike_near(reference, sample_rate, t, back, ATTACK_SEARCH_FORWARD_S);
        let window = |signal: &[f32], at: f64| -> Option<f64> {
            let start = (at * sample_rate).round().max(0.0) as usize;
            let end = start + len;
            (end <= signal.len()).then(|| attack_tonality_db(&signal[start..end], sample_rate))
        };
        let (Some(ea), Some(rb)) = (window(engine, et), window(reference, rt)) else {
            continue;
        };
        engine_levels.push(ea);
        reference_levels.push(rb);
        if let (Some(x), Some(y)) = (
            attack_rise_s(engine, sample_rate, et),
            attack_rise_s(reference, sample_rate, rt),
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
    let worst = deltas.iter().cloned().fold((0.0f64, 0.0f64), |acc, x| {
        if x.1.abs() > acc.1.abs() {
            x
        } else {
            acc
        }
    });
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
        return Err(Error::Config(
            "signal shorter than the envelope window".into(),
        ));
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
    let floor = if peak > 0.0 {
        10.0 * peak.log10() - 60.0
    } else {
        0.0
    };
    Ok(out
        .into_iter()
        .map(|e| {
            if e > 0.0 {
                (10.0 * e.log10()).max(floor)
            } else {
                floor
            }
        })
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
    let worst = deltas.iter().cloned().fold((0.0f64, 0.0f64), |acc, x| {
        if x.1.abs() > acc.1.abs() {
            x
        } else {
            acc
        }
    });
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
    (signal
        .iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        / signal.len() as f64)
        .sqrt()
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
        return Err(Error::Config(
            "a silent render cannot be level-matched".into(),
        ));
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
// The note-off nonlinearity budget
// ---------------------------------------------------------------------------

/// Bottom of the band [`note_off_hf`] reads. Above the tenth partial of every
/// key the phrase set plays below the treble, so what a string is *supposed* to
/// radiate there is already 40 dB down and what a waveshaper puts there is not.
pub const NOTE_OFF_HF_HZ: f64 = 10_000.0;

/// The window after a note-off the band is read in, in seconds. Starts two
/// milliseconds late so the damper's own arrival is inside it, and runs 33 ms,
/// which covers a nominal release's whole felt interval — 10 ms of arrival plus
/// the 11 ms it takes to retire — with room either side.
pub const NOTE_OFF_WINDOW_S: (f64, f64) = (0.002, 0.035);

/// The window before it the reading is referred to: the same length, ending
/// 5 ms before the key is let go, so it is the note itself and not the release.
pub const NOTE_OFF_REFERENCE_S: (f64, f64) = (-0.038, -0.005);

/// How much energy above [`NOTE_OFF_HF_HZ`] a damper's landing *adds*, in dB —
/// the band in [`NOTE_OFF_WINDOW_S`] over the same band in
/// [`NOTE_OFF_REFERENCE_S`], one reading per note-off given.
///
/// This is the statistic a soft limiter driven past its knee is visible in and
/// almost nothing else is. Folding a waveform puts harmonics of it all the way
/// to Nyquist; a string's own series is truncated at `MAX_PARTIAL_RATIO` and
/// rolled off long before this band, and a damper *arriving* can only take
/// energy out of it. So a real piano reads negative here — the Salamander
/// recording of the staccato phrase reads a mean of −3.2 dB and never more than
/// +0.7 — and anything strongly positive is the engine's own arithmetic.
///
/// A ratio of two windows of the same signal, so it is scale-free: an engine
/// render, a recording and a level-matched pair all read the same number. The
/// caller decides which note-offs are worth reading; a window with a strike or
/// a pedal move in it is measuring the strike or the pedal.
pub fn note_off_hf(mono: &[f32], sample_rate: f64, note_offs: &[f64]) -> Vec<f64> {
    let (from, to) = NOTE_OFF_WINDOW_S;
    let n = ((to - from) * sample_rate).round() as usize;
    let size = n.next_power_of_two();
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(size);
    let first = (NOTE_OFF_HF_HZ * size as f64 / sample_rate).ceil() as usize;
    let band = |at: f64| -> Option<f64> {
        let start = (at * sample_rate).round();
        if start < 0.0 || start as usize + n > mono.len() {
            return None;
        }
        let start = start as usize;
        let mut buffer = vec![Complex32::new(0.0, 0.0); size];
        for (i, slot) in buffer.iter_mut().take(n).enumerate() {
            // Hann, so a partial's own skirt does not leak into the band.
            let w = 0.5 - 0.5 * (2.0 * PI * i as f64 / n as f64).cos();
            *slot = Complex32::new((f64::from(mono[start + i]) * w) as f32, 0.0);
        }
        fft.process(&mut buffer);
        let power: f64 = buffer[first..=size / 2]
            .iter()
            .map(|c| f64::from(c.norm_sqr()))
            .sum();
        (power > 0.0).then_some(power)
    };
    note_offs
        .iter()
        .filter_map(|&t| {
            let after = band(t + from)?;
            let before = band(t + NOTE_OFF_REFERENCE_S.0)?;
            Some(10.0 * (after / before).log10())
        })
        .collect()
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
// The evaluation policy: which reference notes are the piano
// ---------------------------------------------------------------------------

/// Which keys the library actually **recorded**, and where every other key's
/// reference sound really comes from.
///
/// `DECISIONS.md` 328 makes this permanent and it is the one rule every scored
/// per-note comparison in this repository now obeys. A sampled piano records a
/// subset of the compass — Salamander takes one note every minor third, 30 of
/// 88 — and plays the other 58 keys by **transposing** the nearest take. That
/// transposed note is a perfectly good thing to listen to and it is what the
/// scoreboard's phrases are played on. It is not a measurement of the piano at
/// that key: its inharmonicity, its unison beat rate, its decay and its
/// brightness are all the *neighbour's*, shifted. Scoring the engine's D4
/// against a resampled D#4 measures the resampler.
///
/// So: **transposed reference notes stay in every render and carry no per-note
/// score.** Reports mark them `transposed — unscored` rather than dropping them,
/// because a reader has to be able to see that the note was played and why its
/// column is empty.
///
/// Two questions this answers, both of which a scoring surface needs:
///
/// * [`RecordedKeys::is_recorded`] — may this key carry a per-note score at all?
/// * [`RecordedKeys::take_for`] / [`RecordedKeys::alternate_take`] — which
///   recording is a transposed key actually made of, and which other recording
///   could have made it instead. The second is what
///   [`RecordedKeys::routing`] builds: a *second* legitimate reconstruction of
///   the same music, so that the cost of transposition can be measured rather
///   than asserted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordedKeys {
    keys: Vec<u8>,
}

impl RecordedKeys {
    /// The keys the library has an attack recording of, ascending.
    pub fn from_library(library: &SampleLibrary) -> Result<Self> {
        let mut keys: Vec<u8> = library.keys().collect();
        keys.sort_unstable();
        keys.dedup();
        if keys.len() < 2 {
            return Err(Error::Config(
                "the library records fewer than two keys, so nothing can be scored against it"
                    .into(),
            ));
        }
        Ok(RecordedKeys { keys })
    }

    /// For tests and for libraries built by hand.
    pub fn from_keys(keys: &[u8]) -> Self {
        let mut keys = keys.to_vec();
        keys.sort_unstable();
        keys.dedup();
        RecordedKeys { keys }
    }

    pub fn keys(&self) -> &[u8] {
        &self.keys
    }

    /// Whether this key is a take of its own — the only kind of note a scored
    /// per-note comparison may use.
    pub fn is_recorded(&self, key: u8) -> bool {
        self.keys.binary_search(&key).is_ok()
    }

    /// The recorded key whose take a player transposes onto `key`: the nearest
    /// one, which is what an SFZ that maps each recording over its immediate
    /// neighbours does. A recorded key is its own take.
    pub fn take_for(&self, key: u8) -> Option<u8> {
        self.keys
            .iter()
            .copied()
            .min_by_key(|&k| (k.abs_diff(key), k))
    }

    /// The *other* recording that could have been transposed onto `key`: the
    /// second-nearest take. `None` for a recorded key, which needs no
    /// transposition, and `None` where the library has only one take in reach.
    ///
    /// This is the substitution `bench` measures the transposition cost with.
    /// Both routes are equally defensible reconstructions of a note nobody
    /// recorded, so how far apart they land is how much of "the reference" at
    /// that key is the resampler rather than the piano.
    pub fn alternate_take(&self, key: u8) -> Option<u8> {
        if self.is_recorded(key) {
            return None;
        }
        let first = self.take_for(key)?;
        self.keys
            .iter()
            .copied()
            .filter(|&k| k != first)
            .min_by_key(|&k| (k.abs_diff(key), k))
    }

    /// The take every key should be played from when each transposed key is
    /// moved onto its [`alternate_take`](RecordedKeys::alternate_take):
    /// recorded keys keep their own recording, and everything else swaps route.
    ///
    /// The rendering of this map is `Instrument::rerouted`.
    pub fn routing(&self) -> impl Fn(u8) -> Option<u8> + '_ {
        move |key| {
            if self.is_recorded(key) {
                Some(key)
            } else {
                self.alternate_take(key).or_else(|| self.take_for(key))
            }
        }
    }

    /// `recorded` or `transposed from D#4 (-1)` — what a report prints in a
    /// provenance column.
    pub fn provenance(&self, key: u8) -> String {
        if self.is_recorded(key) {
            return "recorded".to_string();
        }
        match self.take_for(key) {
            Some(take) => format!(
                "transposed from {} ({:+})",
                note_name(take),
                i32::from(key) - i32::from(take)
            ),
            None => "unmapped".to_string(),
        }
    }

    /// The recorded keys inside `[lo, hi]`, which is the population a per-note
    /// bar in that register is measured off.
    pub fn in_range(&self, lo: u8, hi: u8) -> Vec<u8> {
        self.keys
            .iter()
            .copied()
            .filter(|&k| k >= lo && k <= hi)
            .collect()
    }
}

/// `C4` for 60. The same spelling every report in this crate uses.
pub fn note_name(key: u8) -> String {
    const NAMES: [&str; 12] = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    format!(
        "{}{}",
        NAMES[usize::from(key) % 12],
        i32::from(key) / 12 - 1
    )
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
        self.bands
            .iter()
            .position(|&(lo, hi)| vel >= lo && vel <= hi)
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
        let j = if i + 1 < self.bands.len() {
            i + 1
        } else {
            i.saturating_sub(1)
        };
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
                    SamplerEvent::NoteOn {
                        key,
                        vel: self.alternate(vel),
                    },
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

/// The Ode to Joy melody line of [`excerpt`]: `(onset in beats, key, length in
/// beats)`, thirty notes over five distinct pitches (C4, D4, E4, F4, G4).
///
/// A `const` rather than a local because a second tool plays exactly this line
/// on its own: `estimate::melody`'s evenness gate renders the soprano solo and
/// asks whether any one of its notes is textured unlike the rest. That question
/// is only about *this* phrase if the two lists cannot drift apart.
pub const ODE_MELODY: [(f64, u8, f64); 30] = [
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

/// Seconds per beat in [`excerpt`] and in the soprano line taken out of it.
pub const ODE_BEAT: f64 = 0.5;

/// Seconds of silence before [`excerpt`] begins.
pub const ODE_START: f64 = 0.2;

/// Velocity every melody note of [`excerpt`] is struck at.
pub const ODE_MELODY_VEL: u8 = 88;

/// Eight bars of the Ode to Joy theme (Beethoven, Symphony No. 9 — public
/// domain), harmonised with a bass note, a left-hand chord and a pedal change
/// on every harmony. The only phrase with three simultaneous textures, and
/// therefore the only one where masking between them can go wrong.
pub fn excerpt() -> Phrase {
    const C: [u8; 4] = [36, 48, 52, 55];
    const G: [u8; 4] = [43, 47, 50, 55];
    const G7: [u8; 4] = [43, 47, 53, 55];
    let melody = ODE_MELODY;
    let beat = ODE_BEAT;
    let start = ODE_START;
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
        note(
            &mut events,
            start + at * beat,
            key,
            ODE_MELODY_VEL,
            (len * beat - 0.05).max(0.08),
        );
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

// ---------------------------------------------------------------------------
// Columns A and B: what the six energy metrics above are blind to
// ---------------------------------------------------------------------------

/// The keys the motion columns are measured on, in the order they are reported.
/// `docs/history/FUNDAMENTALS.md` Part II §II.3 pins four, and the verification errata pin
/// that it is these four and not section 7's three: A4 is the cell the forensics
/// found the engine's worst frequency excursion on, and leaving it out is why
/// Part II's `A1` baseline (~4.5 over 16 cells) and section 7's (3.39 over 12)
/// differ.
pub const MOTION_KEYS: [(u8, &str); 4] = [(45, "A2"), (60, "C4"), (69, "A4"), (84, "C6")];

/// Partials per key. Low enough that every key has them and low enough that the
/// ear resolves them individually.
pub const MOTION_PARTIALS: u32 = 4;

/// The three velocities every cell is rendered and recorded at. Column B's
/// velocity coherence is the spread across them, and it is the column with the
/// physics in it: a coupled unison's mode *mixture* is set by the strike, so it
/// cannot be velocity-invariant, and a free-running one cannot be anything else.
pub const MOTION_VELOCITIES: [u8; 3] = [40, 90, 120];

/// The velocity Columns A and B1 are quoted at — the same one every per-note
/// table in the preset is fitted at.
pub const MOTION_REFERENCE_VELOCITY: u8 = 90;

/// `IF mismatch` must be at most this. Symmetric by construction, so "too dead"
/// fails as loudly as "too spiky".
pub const A1_GATE: f64 = 2.0;
/// `IF placement` must be at least this.
pub const A2_GATE: f64 = 0.5;
/// `beat-depth error` must be at most this many dB.
pub const B1_GATE_DB: f64 = 3.0;
/// `velocity coherence` must be at least this fraction of the reference's own.
pub const B2_GATE: f64 = 0.25;

/// One key × partial × velocity, measured on both signals.
#[derive(Clone, Copy, Debug)]
pub struct MotionCell {
    pub key: u8,
    pub k: u32,
    pub velocity: u8,
    /// `None` when the partial did not stand over its own neighbourhood, which
    /// is a cell that measured nothing rather than a cell that measured zero.
    pub engine: Option<Motion>,
    pub reference: Option<Motion>,
}

impl MotionCell {
    fn both(&self) -> Option<(Motion, Motion)> {
        Some((self.engine?, self.reference?))
    }
}

/// Measures one signal's first [`MOTION_PARTIALS`] partials.
///
/// `partial_hz` is the nominal frequency of each partial — the caller's, because
/// only the caller knows the preset's inharmonicity. The search half-width is a
/// fraction of the fundamental, so a partial that has been pulled by the bridge
/// is still found and a neighbouring one is not.
pub fn measure_partials(signal: &[f64], partial_hz: &[f64]) -> Vec<Option<Motion>> {
    let mut spectrum = Spectrum::new(signal);
    let half_width = partial_hz.first().copied().unwrap_or(100.0) * 0.35;
    partial_hz
        .iter()
        .map(|&hz| partial_motion(&mut spectrum, hz, half_width))
        .collect()
}

/// Columns A and B, and the pieces they are made of.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotionColumns {
    /// **A1, `IF mismatch`.** Geometric mean over the reference-velocity cells
    /// of `max(J_eng, J_ref) / min(J_eng, J_ref)`, both clamped at
    /// [`crate::motion::IF_FLOOR_CENTS`] first.
    pub if_mismatch: f64,
    /// **A2, `IF placement`.** Median over the same cells of `L_eng / L_ref`.
    pub if_placement: f64,
    /// **B1, `beat-depth error`.** Mean over the same cells of
    /// `|D_eng − D_ref|`, dB. Reported as a mean of absolute values
    /// deliberately: the signed mean of a table spanning −9.1 to +14.8 dB is
    /// +2.0, which is exactly what must not be reported.
    pub beat_depth_error_db: f64,
    /// **B2, `velocity coherence`.** The geometric mean of the two ratios
    /// below, which is how the column is "pooled over J and D".
    pub velocity_coherence: f64,
    /// The frequency half of B2: the engine's mean per-cell spread of `J` across
    /// the three velocities, over the reference's.
    pub velocity_coherence_freq: f64,
    /// The depth half: the same for `D`.
    pub velocity_coherence_depth: f64,
    /// The four spreads the two ratios are made of, so a report can say what
    /// moved rather than only how far: engine cents, reference cents, engine dB,
    /// reference dB.
    pub spread_cents: (f64, f64),
    pub spread_depth_db: (f64, f64),
    /// Cells that measured on both sides, at the reference velocity.
    pub cells: usize,
    /// Cells that had all three velocities on both sides.
    pub velocity_cells: usize,
}

impl MotionColumns {
    /// Whether all four gates pass. Nothing in this repository has ever passed
    /// one of them, which is the point of writing them down.
    pub fn passes(&self) -> bool {
        self.if_mismatch <= A1_GATE
            && self.if_placement >= A2_GATE
            && self.beat_depth_error_db <= B1_GATE_DB
            && self.velocity_coherence >= B2_GATE
    }
}

/// Reduces the measured cells to the four columns.
///
/// A1, A2 and B1 are taken over the [`MOTION_REFERENCE_VELOCITY`] cells; B2 is
/// taken over the same cells' spread across all of [`MOTION_VELOCITIES`]. That
/// split is pinned here rather than left to the caller because
/// `docs/history/FUNDAMENTALS.md`'s two halves quote the columns over different cell sets and
/// the errata require an implementation to choose one.
///
/// Every per-cell frequency deviation is clamped at
/// [`crate::motion::IF_FLOOR_CENTS`] before any ratio is taken — the errata's
/// own instruction, and it is what a ratio needs to mean anything: two cells
/// that are both at the measurement's floor are *the same*, and without the
/// clamp they read a mismatch of thirty.
pub fn motion_columns(cells: &[MotionCell]) -> MotionColumns {
    let floor = |c: f64| c.max(IF_FLOOR_CENTS);
    let at_reference: Vec<&MotionCell> = cells
        .iter()
        .filter(|c| c.velocity == MOTION_REFERENCE_VELOCITY)
        .collect();

    let mut log_mismatch = 0.0;
    let mut placements: Vec<f64> = Vec::new();
    let mut depth_errors: Vec<f64> = Vec::new();
    let mut counted = 0usize;
    for cell in &at_reference {
        let Some((engine, reference)) = cell.both() else {
            continue;
        };
        let (a, b) = (floor(engine.band_cents), floor(reference.band_cents));
        log_mismatch += (a.max(b) / a.min(b)).ln();
        if reference.placement() > 0.0 {
            placements.push(engine.placement() / reference.placement());
        }
        depth_errors.push((engine.beat_depth_db - reference.beat_depth_db).abs());
        counted += 1;
    }

    // B2: per cell, the spread of each statistic over the velocities that
    // measured on both sides — and only cells where *both* signals have all
    // three, so the ratio is not a comparison of different cell sets.
    let mut engine_cents: Vec<f64> = Vec::new();
    let mut reference_cents: Vec<f64> = Vec::new();
    let mut engine_depth: Vec<f64> = Vec::new();
    let mut reference_depth: Vec<f64> = Vec::new();
    for &(key, _) in &MOTION_KEYS {
        for k in 1..=MOTION_PARTIALS {
            let group: Vec<(Motion, Motion)> = MOTION_VELOCITIES
                .iter()
                .filter_map(|&velocity| {
                    cells
                        .iter()
                        .find(|c| c.key == key && c.k == k && c.velocity == velocity)
                        .and_then(MotionCell::both)
                })
                .collect();
            if group.len() < MOTION_VELOCITIES.len() {
                continue;
            }
            let spread = |values: Vec<f64>| {
                values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                    - values.iter().copied().fold(f64::INFINITY, f64::min)
            };
            engine_cents.push(spread(
                group.iter().map(|(e, _)| floor(e.band_cents)).collect(),
            ));
            reference_cents.push(spread(
                group.iter().map(|(_, r)| floor(r.band_cents)).collect(),
            ));
            engine_depth.push(spread(group.iter().map(|(e, _)| e.beat_depth_db).collect()));
            reference_depth.push(spread(group.iter().map(|(_, r)| r.beat_depth_db).collect()));
        }
    }
    let mean = |v: &[f64]| {
        if v.is_empty() {
            0.0
        } else {
            v.iter().sum::<f64>() / v.len() as f64
        }
    };
    let ratio = |a: f64, b: f64| if b > 0.0 { a / b } else { 0.0 };
    let freq = ratio(mean(&engine_cents), mean(&reference_cents));
    let depth = ratio(mean(&engine_depth), mean(&reference_depth));

    placements.sort_by(|a, b| a.partial_cmp(b).expect("finite placements"));
    MotionColumns {
        if_mismatch: if counted == 0 {
            f64::NAN
        } else {
            (log_mismatch / counted as f64).exp()
        },
        if_placement: if placements.is_empty() {
            f64::NAN
        } else {
            placements[placements.len() / 2]
        },
        beat_depth_error_db: mean(&depth_errors),
        // Pooled as a geometric mean: the two halves are in different units
        // (cents and decibels) and neither may dominate the other by being
        // numerically larger.
        velocity_coherence: (freq.max(0.0) * depth.max(0.0)).sqrt(),
        velocity_coherence_freq: freq,
        velocity_coherence_depth: depth,
        spread_cents: (mean(&engine_cents), mean(&reference_cents)),
        spread_depth_db: (mean(&engine_depth), mean(&reference_depth)),
        cells: counted,
        velocity_cells: engine_cents.len(),
    }
}

// ---------------------------------------------------------------------------
// The stereo columns: the one difference every metric above is blind to
// ---------------------------------------------------------------------------

//
// `DECISIONS.md` 314 measured the engine's stereo image against the
// recording's and found the largest single difference in the whole chain
// experiment — and the only one no column of `REALISM.md` could see, because
// every metric above is computed on the mono sum. The recording's two channels
// are **+0.945 correlated below 125 Hz** and fall to about zero through the
// mid and treble, with a peak |r| of 0.57-0.65 at lags of −0.23 to +1.98 ms:
// that is a spaced pair of microphones, two capsules well inside a wavelength
// of each other in the bass seeing one wavefront and seeing the same sound
// about 60 % coherent at a sub-millisecond delay above it. The engine is
// **exactly inverted** — a soundboard FDN whose two output taps carry opposite
// signs decorrelates the bass to −0.577, and `soundboard::pan_for_key` scales
// one mono voice into two channels, which is +0.964 at lag zero in the treble.
//
// Item 317 (a) is the instruction this section answers: *give the loss a
// stereo term first*, because a stage built to fix something nothing scores is
// a stage nobody can regress. Nothing here is a room — `PHYSICS.md` §9 is
// refused by measurement in item 315 and stays out of scope. This measures the
// **presentation**: an instrument and the pair of microphones in front of it,
// §8's subject, which is what the interchannel table actually points at.

/// The six bands the stereo image is read in.
///
/// `estimate::chain::SPATIAL_BANDS` exactly, so every number here is
/// comparable to the table item 314 published. Four of the six are octaves and
/// two are not — 500 Hz-2 kHz spans two and 2-6 kHz a little over one and a
/// half — because the useful statistic up there is a *register* and not a
/// band: above 2 kHz a piano note has forty partials in one octave and the
/// correlation of any one of them is the correlation of its own beating.
pub const STEREO_BANDS: [(&str, f64, f64); 6] = [
    ("63-125", 63.0, 125.0),
    ("125-250", 125.0, 250.0),
    ("250-500", 250.0, 500.0),
    ("500-2k", 500.0, 2_000.0),
    ("2k-6k", 2_000.0, 6_000.0),
    ("6k-12k", 6_000.0, 12_000.0),
];

/// **Sixth-octave bands over 100-800 Hz**: the resolution the mode-controlled
/// band's own shape lives at.
///
/// [`STEREO_BANDS`] is one octave wide and that is right for everything it was
/// built for — a correlation needs a band with several partials in it, and the
/// six of them are `estimate::chain`'s own set. It is **too coarse for one
/// thing**, and `DECISIONS.md` 403 is what that cost: a nodal band 0.96 octaves
/// wide sits inside two scoreboard bands that are an octave each, so an angle
/// that is right at the bottom of the band and eight decibels too large at the
/// top averages out to a column that reads `-0.09` against a bar of `0.49` and
/// calls itself green. The recording's own profile is not flat across an octave
/// there either — its pair-over-mono runs `+9.4 dB at 180 Hz` down to `+1.7 at
/// 320` — so an octave column is asking the engine to match an average of a
/// shape rather than the shape.
///
/// A sixth of an octave is the resolution `forensics/src/bin/mono_mechanism.rs`
/// measured the recording's profile at, and it is the coarsest one that
/// resolves the two features the profile has: the fall through zero between 127
/// and 180 Hz and the return between 254 and 320. The bands are geometric,
/// centred on `100 x 2^(k/6)`, and their edges meet.
///
/// [`STEREO_BAND_FLOOR_DB`] is unchanged for them. A sixth-octave band holds
/// about 8 dB less than an octave one on a flat spectrum, which is nowhere near
/// the 60 dB the floor is there to exclude; what the floor removes is a band a
/// signal has *nothing* in, and that is a property of the signal.
pub const STEREO_FINE_BANDS: [(&str, f64, f64); 19] = [
    ("100", 94.4, 106.0),
    ("112", 105.9, 118.9),
    ("126", 118.9, 133.5),
    ("141", 133.4, 149.8),
    ("159", 149.8, 168.2),
    ("178", 168.2, 188.8),
    ("200", 188.8, 211.9),
    ("224", 211.9, 237.8),
    ("252", 237.8, 266.9),
    ("283", 266.9, 299.7),
    ("317", 299.6, 336.3),
    ("356", 336.3, 377.6),
    ("400", 377.6, 423.8),
    ("449", 423.8, 475.7),
    ("504", 475.7, 533.9),
    ("566", 533.9, 599.3),
    ("635", 599.3, 672.7),
    ("713", 672.7, 755.0),
    ("800", 755.0, 847.4),
];

/// Silence before the strike in a single-key stereo render, in **samples**.
///
/// Where the window a stereo image is read over *starts* is not a free choice,
/// and the first version of this measurement made it one: it asked for 0.05 s
/// of preroll, which is 2400 samples, which is not a whole number of the
/// engine's 128-sample blocks. An event takes effect at the head of the block
/// that contains it (`piano_emulator::render`), so the note began at sample
/// 2304 and the window began at 2400 — **96 samples, two milliseconds, into a
/// note that was supposed to start at its first sample**. `DECISIONS.md` 378 is
/// what that cost, and it is two separate errors rather than one:
///
/// * A window that starts *inside* a note starts with a **step**, and a step is
///   broadband. Measured on the shipped preset, the engine's 6-12 kHz band went
///   from readable on 15 of the 30 recorded keys to readable on 29 and its
///   median `r0` from −0.08 to +0.30 — an entire column made of the window's
///   own edge. Fading that edge over the same 96 samples puts the band back
///   where the aligned window has it (A0: −56.3 dB with the step, −66.9 dB
///   faded, −72.1 dB aligned), which is the proof that it is the edge.
/// * A window that starts *after* the strike is **missing the strike**, and in
///   the engine's 125-500 Hz the strike is most of what a treble key has: the
///   two bands' medians over the same thirty keys flip from `+0.204/+0.218` to
///   `−0.221/−0.169` between the aligned window and the misaligned one. Fading
///   the edge does *not* undo that, so it is content and not splatter.
///
/// The recording is unmoved by the same thing — its two mid bands read
/// `−0.115/−0.226` aligned and `−0.105/−0.223` 96 samples in — which is what
/// licenses reading it from an onset *detector* while the engine is read from
/// the strike itself.
///
/// So the strike goes on the first sample of the window, and since the engine
/// can only start a note at the head of a block, the preroll is a whole number
/// of blocks: **3840 = 30 × 128**, 80 ms, long enough that an onset is never at
/// sample zero. The callers that render this material — `tuner/tests/stereo.rs`
/// and `tools::mics` — assert the block alignment at compile time.
pub const STEREO_PREROLL_SAMPLES: usize = 3_840;

/// Widest lag the interchannel correlation is searched over, seconds.
///
/// Five milliseconds is 1.7 m of air: wider than any mic pair and wider than
/// the first reflection off a lid, and narrow enough that a piano partial's own
/// period does not turn the search into a periodicity measurement above 200 Hz.
pub const STEREO_MAX_LAG_S: f64 = 0.005;

/// How far under the whole signal's energy a band may sit and still be read,
/// in dB.
///
/// A correlation is a ratio of two energies, and in a band that holds none it
/// is a ratio of two noise floors. A0's 6-12 kHz band is such a band; so is
/// C8's 63-125 Hz. Reading them would put the arithmetic of silence into the
/// median. The band is dropped from *every* signal of a comparison when it is
/// unreadable in any one of them, so the three sides are always the same set.
pub const STEREO_BAND_FLOOR_DB: f64 = -60.0;

/// Ceiling on the reported mid/side ratio, dB.
///
/// The engine before this milestone can produce two channels that differ by a
/// gain alone, whose side energy is exactly zero and whose ratio is therefore
/// infinite. A median of infinities is an infinity and a table of them says
/// nothing, so the ratio is clamped: ±60 dB is a side signal a millionth of the
/// mid's energy, which is past any microphone pair and past any recording.
pub const STEREO_MS_CLAMP_DB: f64 = 60.0;

/// How much further than the reference's own disagreement with itself the
/// engine is allowed to go before a stereo column fails.
///
/// `estimate::melody::ALLOWANCE`'s number and `estimate::melody`'s argument for
/// it: a bar set at exactly the reference's own scatter fails half of a
/// perfect instrument, and a quarter more is the smallest margin that does not.
pub const STEREO_ALLOWANCE: f64 = 1.25;

/// **The most side energy a mono-exact pair may carry, as this board's own
/// `r0`: zero** — `E_side = E_mid`, `|T| = 1`.
///
/// `DECISIONS.md` 486, and it is `MIC_MODAL_LIFT`'s rail of one written in the
/// coherence board's units rather than a number anybody chose.
/// `r0 = (E_mid − E_side)/(E_mid + E_side)`, so **`r0 < 0` *is*
/// `E_side > E_mid`**; and writing the pair as `L = M(1 + T)`, `R = M(1 − T)`
/// — which mono discipline forces (`DECISIONS.md` 470) — that band's
/// `E_side/E_mid` is `|T|²`. Above `|T| = 1` the denominator `|1 − T|` can
/// vanish: one loudspeaker inverts against the other and a partial's image
/// `20 log10 |1 + T| / |1 − T|` is unbounded. That is exactly what item 418
/// railed the lift at one to forbid, after a listener found the artifact three
/// separate ways while this board was green on it.
///
/// So under the neutral policy of item 466 — and under the owner's verdict of
/// item 485, which is what finally decides D470's budget — a target below zero
/// is a target that asks for a mechanism the schema refuses. The **statistic**
/// does not move, the **bar** does not move (it is still the recording against
/// its own second take, or the material's own uncertainty, whichever is
/// larger), and the **target** becomes `max(reference_r0, 0)`. Every band where
/// the recording's own `r0` is positive is untouched, which on this library is
/// four of six.
///
/// [`StereoColumn::excluded_r0`] is what that removes in each band and the gate
/// prints it before it asserts anything, for item 417's reason verbatim: an
/// acceptance nobody can read is indistinguishable from a widened bar.
pub const NEUTRAL_SIDE_CEILING_R0: f64 = 0.0;

/// The same ceiling in the per-channel board's units: `pair_db` is
/// `10 log10(1 + E_side/E_mid)`, so `E_side/E_mid = 1` is
/// **`10 log10 2 = 3.0103 dB`**.
///
/// `DECISIONS.md` 486. It is the number item 418 already quotes as "one is a
/// ceiling of +3.01 dB where the recording's own two nodal bands read +3.54 and
/// +3.88"; what item 486 does is stop asking for the 3.54 and the 3.88.
pub const NEUTRAL_PAIR_CEILING_DB: f64 = 3.010_299_956_639_812;

/// What two channels do in one band.
#[derive(Clone, Copy, Debug)]
pub struct StereoBand {
    pub name: &'static str,
    pub lo_hz: f64,
    pub hi_hz: f64,
    /// Normalised interchannel correlation at lag zero. This is the number a
    /// pan-pot pins at +1 and item 314 found inverted; it is signed, so an
    /// anti-correlated pair reads negative rather than merely small.
    pub r0: f64,
    /// The correlation at the lag of largest |r| over ±[`STEREO_MAX_LAG_S`],
    /// signed. A spaced pair shows a peak well away from zero lag; a pan-pot
    /// cannot.
    pub peak_r: f64,
    /// Where that peak is, in milliseconds.
    ///
    /// The correlation is `c[τ] = Σ_t L[t+τ]·R[t]`, so a right channel that is
    /// the left one delayed peaks at **negative** τ: positive `lag_ms` means
    /// the *right* channel leads and the source is nearer the right capsule.
    /// (`estimate::chain`'s own test pins this — a 2 ms delay into the right
    /// channel reads −2.00 ms — and the sentence its field carried said the
    /// opposite.)
    pub lag_ms: f64,
    /// `10·log10` of mid energy over side energy in this band, mid `(L+R)/2`
    /// and side `(L−R)/2`, clamped to ±[`STEREO_MS_CLAMP_DB`]. The same fact
    /// as `r0` in energy terms for a level-balanced pair, and *not* the same
    /// fact when the two channels differ in level, which is what makes it worth
    /// printing beside it.
    pub mid_side_db: f64,
    /// This band's share of the whole signal's energy, dB. Zero or negative;
    /// see [`STEREO_BAND_FLOOR_DB`].
    pub level_db: f64,
}

impl StereoBand {
    /// Is there enough in this band to make a ratio of two energies mean
    /// anything?
    pub fn readable(&self) -> bool {
        self.level_db.is_finite() && self.level_db > STEREO_BAND_FLOOR_DB && self.r0.is_finite()
    }
}

/// One signal's whole stereo image.
#[derive(Clone, Debug)]
pub struct StereoImage {
    pub broadband: StereoBand,
    pub bands: Vec<StereoBand>,
}

impl StereoImage {
    /// The band a frequency falls in, clamped at both ends.
    ///
    /// Clamped rather than optional because the caller that needs this is the
    /// compass, which asks for "the band of this key's fundamental" for all 88
    /// keys — and A0's 27.5 Hz is under the lowest band's 63 Hz. What it gets
    /// there is the band its *second* partial is in, which is the lowest band
    /// the material supports, and the report says so.
    pub fn band_for(hz: f64) -> usize {
        STEREO_BANDS
            .iter()
            .position(|&(_, _, hi)| hz < hi)
            .unwrap_or(STEREO_BANDS.len() - 1)
    }

    /// The band containing `hz`, by [`StereoImage::band_for`].
    pub fn at(&self, hz: f64) -> StereoBand {
        self.bands[Self::band_for(hz)]
    }
}

fn stereo_spectrum(signal: &[f32], n: usize, planner: &mut FftPlanner<f32>) -> Vec<Complex32> {
    let forward = planner.plan_fft_forward(n);
    let mut buffer: Vec<Complex32> = (0..n)
        .map(|i| Complex32::new(signal.get(i).copied().unwrap_or(0.0), 0.0))
        .collect();
    forward.process(&mut buffer);
    buffer
}

/// One bin's contribution to the cross-spectrum and to the four energies.
///
/// The energies are accumulated over the same bins the cross-spectrum is built
/// from rather than over a zeroed copy of the whole transform: Parseval gives
/// the identical number for a fraction of the memory, which matters because a
/// phrase is a million samples and there are seven bands.
#[inline]
fn stereo_accumulate(
    j: usize,
    a: &[Complex32],
    b: &[Complex32],
    cross: &mut [Complex32],
    acc: &mut [f64; 4],
) {
    let (x, y) = (a[j], b[j]);
    cross[j] = x * y.conj();
    acc[0] += f64::from(x.norm_sqr());
    acc[1] += f64::from(y.norm_sqr());
    acc[2] += f64::from(((x + y) * 0.5).norm_sqr());
    acc[3] += f64::from(((x - y) * 0.5).norm_sqr());
}

/// One band of the image, and the band's absolute energy so the next one can
/// be quoted as a share of the whole.
fn stereo_band(
    a: &[Complex32],
    b: &[Complex32],
    sample_rate: f64,
    band: Option<(&'static str, f64, f64)>,
    total_energy: Option<f64>,
    planner: &mut FftPlanner<f32>,
) -> (StereoBand, f64) {
    let n = a.len();
    let mut cross = vec![Complex32::new(0.0, 0.0); n];
    let mut acc = [0.0f64; 4];
    match band {
        None => {
            for j in 0..n {
                stereo_accumulate(j, a, b, &mut cross, &mut acc);
            }
        }
        Some((_, lo, hi)) => {
            let bin = |hz: f64| (hz * n as f64 / sample_rate).round() as usize;
            let (blo, bhi) = (bin(lo).max(1), bin(hi).min(n / 2));
            if bhi >= blo {
                for j in blo..=bhi {
                    stereo_accumulate(j, a, b, &mut cross, &mut acc);
                    // The negative-frequency twin, which carries the same
                    // energy and is what makes the inverse transform real.
                    if n - j != j {
                        stereo_accumulate(n - j, a, b, &mut cross, &mut acc);
                    }
                }
            }
        }
    }
    let scale = 1.0 / n as f64;
    let (ea, eb, em, es) = (
        acc[0] * scale,
        acc[1] * scale,
        acc[2] * scale,
        acc[3] * scale,
    );
    let (name, lo, hi) = band.unwrap_or(("broadband", 0.0, sample_rate / 2.0));
    let energy = ea + eb;
    let level_db = 10.0 * (energy / total_energy.unwrap_or(energy).max(1e-300)).log10();
    if ea <= 0.0 || eb <= 0.0 {
        return (
            StereoBand {
                name,
                lo_hz: lo,
                hi_hz: hi,
                r0: f64::NAN,
                peak_r: f64::NAN,
                lag_ms: f64::NAN,
                mid_side_db: f64::NAN,
                level_db: f64::NEG_INFINITY,
            },
            energy,
        );
    }
    let inverse = planner.plan_fft_inverse(n);
    inverse.process(&mut cross);
    let norm = (ea * eb).sqrt();
    let value = |lag: isize| -> f64 {
        let idx = if lag >= 0 {
            lag as usize
        } else {
            n - (-lag) as usize
        };
        f64::from(cross[idx].re) * scale / norm
    };
    let max_lag = ((STEREO_MAX_LAG_S * sample_rate).round() as usize)
        .min(n / 4)
        .max(1);
    let mut best = (0isize, f64::NEG_INFINITY);
    for lag in -(max_lag as isize)..=(max_lag as isize) {
        let v = value(lag).abs();
        if v > best.1 {
            best = (lag, v);
        }
    }
    (
        StereoBand {
            name,
            lo_hz: lo,
            hi_hz: hi,
            r0: value(0),
            peak_r: value(best.0),
            lag_ms: best.0 as f64 / sample_rate * 1000.0,
            mid_side_db: (10.0 * (em / es.max(1e-300)).log10())
                .clamp(-STEREO_MS_CLAMP_DB, STEREO_MS_CLAMP_DB),
            level_db,
        },
        energy,
    )
}

/// The interchannel image of one signal: broadband and per [`STEREO_BANDS`].
///
/// Correlation is normalised per channel, so it is invariant to what
/// [`level_match`] does and to any per-channel gain; the mid/side ratio is not,
/// and is the column that sees a level imbalance.
pub fn stereo_image(left: &[f32], right: &[f32], sample_rate: f64) -> Result<StereoImage> {
    if left.is_empty() || right.is_empty() {
        return Err(Error::Config("a stereo image needs two channels".into()));
    }
    // Twice the signal, rounded up: a circular correlation of a padded pair is
    // the linear one over every lag the search looks at.
    let n = (left.len().max(right.len()) * 2).next_power_of_two();
    let mut planner = FftPlanner::<f32>::new();
    let a = stereo_spectrum(left, n, &mut planner);
    let b = stereo_spectrum(right, n, &mut planner);
    let (broadband, total) = stereo_band(&a, &b, sample_rate, None, None, &mut planner);
    let bands = STEREO_BANDS
        .iter()
        .map(|&band| stereo_band(&a, &b, sample_rate, Some(band), Some(total), &mut planner).0)
        .collect();
    Ok(StereoImage { broadband, bands })
}

/// [`stereo_image`] of an [`Audio`]'s first two channels.
pub fn stereo_image_of(audio: &Audio) -> Result<StereoImage> {
    if audio.channel_count() < 2 {
        return Err(Error::Config("a stereo image needs two channels".into()));
    }
    stereo_image(
        &audio.channels[0],
        &audio.channels[1],
        f64::from(audio.sample_rate),
    )
}

/// One narrow band of the interchannel *profile*: the same two numbers
/// [`StereoBand`] carries, at a resolution fine enough to see a shape rather
/// than six values.
///
/// `STEREO_BANDS` is deliberately coarse — it is `estimate::chain`'s own band
/// set, chosen so that the scoreboard's columns are comparable with item 314.
/// Six numbers are enough to *score* an image and not enough to *model* one:
/// `DECISIONS.md` 357 named the 125-500 Hz shortfall and said the missing thing
/// was "measured directivity", and a measured directivity is a curve. This is
/// that curve's sample.
#[derive(Clone, Copy, Debug)]
pub struct StereoProfilePoint {
    /// Geometric centre of the band, Hz.
    pub hz: f64,
    pub lo_hz: f64,
    pub hi_hz: f64,
    /// Lag-zero interchannel correlation over this band alone.
    pub r0: f64,
    /// `10 log10` of the mid energy over the side energy, unclamped.
    ///
    /// This is the number the model is built on rather than `r0`, and the
    /// reason is an identity: when the mid and the side are uncorrelated —
    /// which is what a *difference* carried by an independent field means —
    /// `r0 = (M − S)/(M + S)` exactly. So a mid/side ratio is a coherence, and
    /// a filter that sets the ratio sets the coherence.
    pub mid_side_db: f64,
    /// The band's share of the whole signal's energy, dB.
    pub level_db: f64,
}

/// Resolution of [`stereo_profile`], bands per octave.
///
/// Sixth-octave: fine enough that the recording's 125-500 Hz dip is a dozen
/// points rather than two, coarse enough that one 3 s note holds hundreds of
/// bins in every band down to 40 Hz.
pub const STEREO_PROFILE_PER_OCTAVE: usize = 6;

/// The band the profile starts and stops at, Hz. Below 40 Hz no key in the
/// library has a fundamental and the recording's own noise floor takes over;
/// above 16 kHz the library's own content stops.
pub const STEREO_PROFILE_RANGE_HZ: (f64, f64) = (40.0, 16_000.0);

// ---------------------------------------------------------------------------
// The mono question, pooled over keys (`DECISIONS.md` 407-412)
// ---------------------------------------------------------------------------

/// The span the mono question is asked over, and the span every share is
/// normalised inside — so a band's number is a *local shape* and not the
/// engine's standing broadband tilt (`DECISIONS.md` 343, 407).
pub const MONO_SPAN_HZ: (f64, f64) = (100.0, 810.0);

/// [`stereo_profile`]'s own sixth-octave centres, from 40 Hz, restricted to
/// [`MONO_SPAN_HZ`].
///
/// It is deliberately that grid and not [`STEREO_FINE_BANDS`]: item 408's
/// headroom table and item 411(b)'s reconciliation are both printed on it, and
/// a fit whose bands were a different grid from the table it has to move would
/// be a fit nobody could check. The two grids overlap by 93 % and item 411(b)
/// measured what that is worth (0.8 % of a band 12.2 % wide).
pub fn mono_grid() -> Vec<f64> {
    let ratio = 2.0f64.powf(1.0 / STEREO_PROFILE_PER_OCTAVE as f64);
    let mut hz = STEREO_PROFILE_RANGE_HZ.0;
    let mut out = Vec::new();
    while hz <= MONO_SPAN_HZ.1 {
        if hz >= MONO_SPAN_HZ.0 {
            out.push(hz);
        }
        hz *= ratio;
    }
    out
}

/// One take's band energies on [`mono_grid`]: `[E_L, E_R, E_M]` per band, with
/// `M = (L + R)/2`.
///
/// In `f64` from the transform down, and with the same band edges, the same
/// rounding of a frequency to a bin and the same negative-frequency twin as
/// `forensics/src/bin/mono_mechanism.rs` — which is what lets the fit and the
/// forensic instrument that graded it be checked against each other to the
/// digit rather than to a tolerance.
#[derive(Clone, Debug)]
pub struct MonoBands {
    pub bands: Vec<[f64; 3]>,
    /// The **whole take's** `[E_L, E_R, E_M]`, every bin from DC to Nyquist.
    ///
    /// Not used by the level match, which is the span's own total on purpose
    /// (`MONO_SPAN_HZ`), but needed to ask the one question a within-span share
    /// cannot: where the span as a whole sits against the rest of the
    /// instrument. A share has no uniform component — `Σ share = 1` on both
    /// sides — so the fit's own statistic is structurally blind to it, and it
    /// has to be measured somewhere.
    pub total: [f64; 3],
}

impl MonoBands {
    pub fn of(left: &[f32], right: &[f32], sample_rate: f64, grid: &[f64]) -> MonoBands {
        let n = left.len().max(right.len()).next_power_of_two();
        let mut planner = FftPlanner::<f64>::new();
        let a = mono_spectrum(left, n, &mut planner);
        let b = mono_spectrum(right, n, &mut planner);
        let half = 2.0f64.powf(0.5 / STEREO_PROFILE_PER_OCTAVE as f64);
        let bin = |hz: f64| (hz * n as f64 / sample_rate).round() as usize;
        let bands = grid
            .iter()
            .map(|&hz| {
                let (blo, bhi) = (bin(hz / half).max(1), bin(hz * half).min(n / 2));
                let mut acc = [0.0f64; 3];
                if bhi >= blo {
                    for j in blo..=bhi {
                        for &(x, y) in &[(a[j], b[j]), (a[n - j], b[n - j])] {
                            acc[0] += x.norm_sqr();
                            acc[1] += y.norm_sqr();
                            acc[2] += ((x + y) * 0.5).norm_sqr();
                        }
                    }
                }
                let scale = 1.0 / n as f64;
                [acc[0] * scale, acc[1] * scale, acc[2] * scale]
            })
            .collect();
        let mut total = [0.0f64; 3];
        for (x, y) in a.iter().zip(&b) {
            total[0] += x.norm_sqr();
            total[1] += y.norm_sqr();
            total[2] += ((x + y) * 0.5).norm_sqr();
        }
        let scale = 1.0 / n as f64;
        for v in &mut total {
            *v *= scale;
        }
        MonoBands { bands, total }
    }

    /// The span's own energy against the whole take's, for `[E_L+E_R, E_M]`.
    pub fn span_over_take(&self) -> (f64, f64) {
        let mono: f64 = self.bands.iter().map(|e| e[2]).sum();
        let pair: f64 = self.bands.iter().map(|e| e[0] + e[1]).sum();
        (
            pair / (self.total[0] + self.total[1]),
            mono / self.total[2],
        )
    }

    /// This take's whole [`MONO_SPAN_HZ`] mono energy — the level match.
    pub fn mono_total(&self) -> f64 {
        self.bands.iter().map(|e| e[2]).sum()
    }

    /// This band's share of that total, linear.
    pub fn mono_share(&self, i: usize) -> f64 {
        self.bands[i][2] / self.mono_total()
    }
}

fn mono_spectrum(
    x: &[f32],
    n: usize,
    planner: &mut FftPlanner<f64>,
) -> Vec<rustfft::num_complex::Complex<f64>> {
    let fft = planner.plan_fft_forward(n);
    let mut buf: Vec<rustfft::num_complex::Complex<f64>> = (0..n)
        .map(|i| {
            rustfft::num_complex::Complex::new(f64::from(x.get(i).copied().unwrap_or(0.0)), 0.0)
        })
        .collect();
    fft.process(&mut buf);
    buf
}

/// One band of item 408's headroom table, pooled over the keys.
#[derive(Clone, Copy, Debug)]
pub struct MonoColumn {
    pub hz: f64,
    /// `10 log10((E_L + E_R) / 2 E_M)` for the recording, pooled: how much of
    /// what the two capsules carry its own mono sum does not.
    pub reference_pair_db: f64,
    /// The same for the engine.
    pub engine_pair_db: f64,
    /// **Required**: `reference_pair − engine_pair`. An energy-conserving nodal
    /// mechanism costs the fold-down exactly the pair energy it adds, so this
    /// is how far above the recording's own mono the *source* has to stand
    /// before one is applied (`DECISIONS.md` 408).
    pub required_db: f64,
    /// **Standing**: pooled, level-matched engine mono share less the
    /// recording's. What the source actually stands at.
    pub standing_db: f64,
    /// The engine's share of its own span energy in this band, pooled — the
    /// weight the level match gives the band, and what the fit's offset is
    /// taken against so a colouration moves the *shape* and not the level.
    pub engine_share: f64,
}

impl MonoColumn {
    /// What the colouration owes this band: `required − standing`, before the
    /// level-preserving offset. Item 409's "cost less headroom" column.
    pub fn deficit_db(&self) -> f64 {
        self.required_db - self.standing_db
    }
}

/// Item 408's table, pooled over `(reference, engine)` pairs of takes.
///
/// Pooled and not a median over keys, which is the estimator question item 411
/// left open and named: a sixth-octave band of *one* key is dominated by
/// whether a partial happens to land in it, and pooling the level-matched
/// energies weights each key by how much it actually has in the band. Every
/// key is pooled, with no readability filter, which is exactly how item 408's
/// own table was computed.
pub fn mono_columns(grid: &[f64], takes: &[(MonoBands, MonoBands)]) -> Vec<MonoColumn> {
    let mut engine_shares = Vec::with_capacity(grid.len());
    let mut columns = Vec::with_capacity(grid.len());
    for i in 0..grid.len() {
        let pair_over_mono = |pick: fn(&(MonoBands, MonoBands)) -> &MonoBands| {
            let (mut p, mut m) = (0.0f64, 0.0f64);
            for t in takes {
                let take = pick(t);
                let total = take.mono_total();
                p += (take.bands[i][0] + take.bands[i][1]) / total;
                m += 2.0 * take.bands[i][2] / total;
            }
            10.0 * (p / m).log10()
        };
        fn reference(t: &(MonoBands, MonoBands)) -> &MonoBands {
            &t.0
        }
        fn engine(t: &(MonoBands, MonoBands)) -> &MonoBands {
            &t.1
        }
        let (mut e, mut r) = (0.0f64, 0.0f64);
        for (reference, engine) in takes {
            e += engine.mono_share(i);
            r += reference.mono_share(i);
        }
        engine_shares.push(e);
        let reference_pair_db = pair_over_mono(reference);
        let engine_pair_db = pair_over_mono(engine);
        columns.push(MonoColumn {
            hz: grid[i],
            reference_pair_db,
            engine_pair_db,
            required_db: reference_pair_db - engine_pair_db,
            standing_db: 10.0 * (e / r).log10(),
            engine_share: e,
        });
    }
    let total: f64 = engine_shares.iter().sum();
    for c in &mut columns {
        c.engine_share /= total;
    }
    columns
}

/// The interchannel image of one signal as a *curve*: sixth-octave bands from
/// 40 Hz to 16 kHz.
///
/// Same arithmetic as [`stereo_image`] — one FFT pair, the cross-spectrum and
/// the four energies accumulated over each band's own bins — with the band list
/// generated rather than tabulated, and without the ±5 ms lag search, which is
/// meaningless in a sixth-octave band (one band of a periodic signal correlates
/// with itself at every period).
pub fn stereo_profile(left: &[f32], right: &[f32], sample_rate: f64) -> Result<Vec<StereoProfilePoint>> {
    if left.is_empty() || right.is_empty() {
        return Err(Error::Config("a stereo profile needs two channels".into()));
    }
    let n = (left.len().max(right.len())).next_power_of_two();
    let mut planner = FftPlanner::<f32>::new();
    let a = stereo_spectrum(left, n, &mut planner);
    let b = stereo_spectrum(right, n, &mut planner);
    let ratio = 2.0f64.powf(1.0 / STEREO_PROFILE_PER_OCTAVE as f64);
    let half = ratio.sqrt();
    let bin = |hz: f64| (hz * n as f64 / sample_rate).round() as usize;
    let mut total = 0.0f64;
    for (x, y) in a.iter().zip(&b) {
        total += f64::from(x.norm_sqr()) + f64::from(y.norm_sqr());
    }
    total /= n as f64;
    let mut points = Vec::new();
    let mut hz = STEREO_PROFILE_RANGE_HZ.0;
    while hz <= STEREO_PROFILE_RANGE_HZ.1 {
        let (lo, hi) = (hz / half, hz * half);
        let (blo, bhi) = (bin(lo).max(1), bin(hi).min(n / 2));
        if bhi >= blo {
            let mut acc = [0.0f64; 4];
            for j in blo..=bhi {
                let (x, y) = (a[j], b[j]);
                acc[0] += f64::from(x.norm_sqr());
                acc[1] += f64::from(y.norm_sqr());
                acc[2] += f64::from(((x + y) * 0.5).norm_sqr());
                acc[3] += f64::from(((x - y) * 0.5).norm_sqr());
                // The negative-frequency twin carries the same energy; the
                // real part of the cross-spectrum needs both.
                let (u, v) = (a[n - j], b[n - j]);
                acc[0] += f64::from(u.norm_sqr());
                acc[1] += f64::from(v.norm_sqr());
                acc[2] += f64::from(((u + v) * 0.5).norm_sqr());
                acc[3] += f64::from(((u - v) * 0.5).norm_sqr());
            }
            let scale = 1.0 / n as f64;
            let (ea, eb) = (acc[0] * scale, acc[1] * scale);
            let (em, es) = (acc[2] * scale, acc[3] * scale);
            // The zero-lag cross-energy *is* `M − S`: expand `(L±R)/2` and the
            // like terms cancel, leaving `<L,R>`. So no inverse transform is
            // needed for `r0`, and it is normalised by the same geometric mean
            // [`stereo_band`] uses so the two agree where the bands coincide.
            let r0 = if ea > 0.0 && eb > 0.0 {
                (em - es) / (ea * eb).sqrt()
            } else {
                f64::NAN
            };
            points.push(StereoProfilePoint {
                hz,
                lo_hz: lo,
                hi_hz: hi,
                r0,
                mid_side_db: 10.0 * (em / es.max(1e-300)).log10(),
                level_db: 10.0 * ((ea + eb) / total.max(1e-300)).log10(),
            });
        }
        hz *= ratio;
    }
    Ok(points)
}

/// [`stereo_profile`] of an [`Audio`]'s first two channels.
pub fn stereo_profile_of(audio: &Audio) -> Result<Vec<StereoProfilePoint>> {
    if audio.channel_count() < 2 {
        return Err(Error::Config("a stereo profile needs two channels".into()));
    }
    stereo_profile(
        &audio.channels[0],
        &audio.channels[1],
        f64::from(audio.sample_rate),
    )
}

/// One thing — a phrase, or a key — measured three ways.
///
/// `alternate` is the floor's other half and it is what makes the column
/// readable: a *second recording of the same piano playing the same music*,
/// which for a phrase is the neighbouring velocity layer ([`VelocityLayers`])
/// and for a single key is that key's other layer. Whatever the reference and
/// its alternate disagree about, the engine is not asked to agree about.
#[derive(Clone, Debug)]
pub struct StereoItem {
    pub label: String,
    pub engine: StereoImage,
    pub reference: StereoImage,
    pub alternate: StereoImage,
}

/// One band of the stereo scoreboard, pooled over the items.
#[derive(Clone, Debug)]
pub struct StereoColumn {
    pub band: usize,
    pub name: &'static str,
    pub lo_hz: f64,
    pub hi_hz: f64,
    /// Median over the items of each side's `r0`.
    pub engine_r0: f64,
    pub reference_r0: f64,
    pub alternate_r0: f64,
    /// Median |peak r| and the median lag it sits at, both sides.
    pub engine_peak_r: f64,
    pub reference_peak_r: f64,
    pub engine_lag_ms: f64,
    pub reference_lag_ms: f64,
    /// Median mid/side ratio, both sides.
    pub engine_mid_side_db: f64,
    pub reference_mid_side_db: f64,
    /// **The target the score is taken against**: `max(reference_r0, 0)` since
    /// `DECISIONS.md` 486 — see [`NEUTRAL_SIDE_CEILING_R0`]. Equal to
    /// [`Self::reference_r0`] in every band where the recording sees more sum
    /// than difference.
    pub target_r0: f64,
    /// `target_r0 − reference_r0`: **how much of the recording's own
    /// decorrelation this board has stopped asking for**, and it is zero in
    /// every band where the recording's `r0` is positive. Printed by the gate
    /// before it asserts.
    pub excluded_r0: f64,
    /// **The score**: |engine r0 − [`Self::target_r0`]|, both medians. What this
    /// band of the image *is*, engine against the target.
    pub error: f64,
    /// **The floor**: the identical statistic between the reference and its
    /// alternate take — two recordings of one piano, reduced the same way.
    pub floor: f64,
    /// Robust sigma (1.4826·MAD) of the reference's own per-item `r0`: how much
    /// this band's correlation moves across the material at all. Not a bar on
    /// its own — a real instrument is *supposed* to move across the compass,
    /// and being excused from a motion the recording makes is not a floor.
    pub scatter: f64,
    /// `scatter / sqrt(items)`: how well the material pins the reference's own
    /// median. This is the term that says how precisely the question can be
    /// asked at all.
    pub uncertainty: f64,
    /// `max(floor, uncertainty) · STEREO_ALLOWANCE`.
    pub bar: f64,
    /// The stricter statement, reported and **not gated**: the median over
    /// items of the per-item |engine r0 − reference r0|, beside the same median
    /// between the reference and its alternate take. A band whose median is
    /// right key by key has the same value here as in [`StereoColumn::error`];
    /// a band that is right on average and wrong at every key does not. This is
    /// what `PHYSICS.md` §8's *per-key* delay and gain would have to close, and
    /// the column exists so that closing the median alone cannot be mistaken
    /// for closing the image.
    pub per_key_error: f64,
    pub per_key_floor: f64,
    pub pass: bool,
    /// Items that were readable in all three signals.
    pub items: usize,
    /// The item furthest off, and by how much.
    pub worst: Option<(String, f64)>,
}

fn stereo_median(values: &[f64]) -> f64 {
    crate::numeric::median(values).unwrap_or(f64::NAN)
}

/// Robust sigma: `1.4826 · MAD`, the same estimator the compass scores with.
fn stereo_sigma(values: &[f64]) -> f64 {
    let centre = stereo_median(values);
    let spread: Vec<f64> = values.iter().map(|v| (v - centre).abs()).collect();
    1.4826 * stereo_median(&spread)
}

/// The stereo scoreboard: one row per band, engine against reference, with the
/// reference's own disagreement with itself beside it.
///
/// **The score is a median against a median.** Per band, the engine's r@0
/// pooled over the items against the reference's, pooled the same way. The bar
/// is `max(floor, uncertainty) · `[`STEREO_ALLOWANCE`], and both of its terms
/// are the reference disagreeing with itself: `floor` is the same median taken
/// on a *second recording* of the same material — the neighbouring velocity
/// layer — and `uncertainty` is `scatter / sqrt(n)`, how well n items pin a
/// median that moves by `scatter` across them. Nothing in the bar is a function
/// of the engine, which is the property that makes it a bar rather than a
/// description: a threshold fitted to what the engine happens to do cannot be
/// failed.
///
/// **Why the pooled scatter is not itself the bar**, though the melody gate's
/// population term is the analogous thing: a recording's r@0 moves across the
/// compass because the keys are in different places relative to the
/// microphones, and that motion is a fact the engine is meant to *reproduce*,
/// not to be excused from. Using it as the bar would let a band pass at +0.71
/// where the recording reads 0.00, purely because the recording's own keys
/// scatter by 0.45 about that zero. It enters as `sqrt(n)` smaller instead —
/// the precision of the question, not a licence to miss it — and the per-key
/// distance is reported beside the pooled one so that a model which fixes the
/// median without fixing the image is visible as such.
pub fn stereo_columns(items: &[StereoItem]) -> Vec<StereoColumn> {
    STEREO_BANDS
        .iter()
        .enumerate()
        .map(|(b, &(name, lo, hi))| {
            let readable: Vec<&StereoItem> = items
                .iter()
                .filter(|it| {
                    it.engine.bands[b].readable()
                        && it.reference.bands[b].readable()
                        && it.alternate.bands[b].readable()
                })
                .collect();
            let pick = |f: fn(&StereoBand) -> f64, side: fn(&StereoItem) -> &StereoImage| {
                readable
                    .iter()
                    .map(|it| f(&side(it).bands[b]))
                    .collect::<Vec<f64>>()
            };
            fn engine(it: &StereoItem) -> &StereoImage {
                &it.engine
            }
            fn reference(it: &StereoItem) -> &StereoImage {
                &it.reference
            }
            fn alternate(it: &StereoItem) -> &StereoImage {
                &it.alternate
            }
            let e_r0 = pick(|x| x.r0, engine);
            let r_r0 = pick(|x| x.r0, reference);
            let a_r0 = pick(|x| x.r0, alternate);
            // **The target is the recording's own `r0` or the neutral policy's
            // ceiling on the side energy, whichever asks for less side**
            // (`DECISIONS.md` 486, `NEUTRAL_SIDE_CEILING_R0`). Item by item as
            // well as on the median, because the per-item column is the one
            // that says whether a band is right key by key.
            let target = |r: f64| r.max(NEUTRAL_SIDE_CEILING_R0);
            let errors: Vec<f64> = e_r0
                .iter()
                .zip(&r_r0)
                .map(|(e, r)| (e - target(*r)).abs())
                .collect();
            let floors: Vec<f64> = a_r0
                .iter()
                .zip(&r_r0)
                .map(|(a, r)| (a - target(*r)).abs())
                .collect();
            let (median_e, median_r, median_a) = (
                stereo_median(&e_r0),
                stereo_median(&r_r0),
                stereo_median(&a_r0),
            );
            let target_r0 = target(median_r);
            let error = (median_e - target_r0).abs();
            // The floor stays what it always was — the recording against its own
            // second take — because how finely a thing can be resolved is the
            // recording's answer whatever the target is (`DECISIONS.md` 466).
            let floor = (median_a - median_r).abs();
            let scatter = stereo_sigma(&r_r0);
            let uncertainty = if readable.is_empty() {
                f64::NAN
            } else {
                scatter / (readable.len() as f64).sqrt()
            };
            let bar = floor.max(uncertainty) * STEREO_ALLOWANCE;
            let worst = errors
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, &d)| (readable[i].label.clone(), d));
            StereoColumn {
                band: b,
                name,
                lo_hz: lo,
                hi_hz: hi,
                engine_r0: median_e,
                reference_r0: median_r,
                alternate_r0: median_a,
                target_r0,
                excluded_r0: target_r0 - median_r,
                engine_peak_r: stereo_median(&pick(|x| x.peak_r.abs(), engine)),
                reference_peak_r: stereo_median(&pick(|x| x.peak_r.abs(), reference)),
                engine_lag_ms: stereo_median(&pick(|x| x.lag_ms, engine)),
                reference_lag_ms: stereo_median(&pick(|x| x.lag_ms, reference)),
                engine_mid_side_db: stereo_median(&pick(|x| x.mid_side_db, engine)),
                reference_mid_side_db: stereo_median(&pick(|x| x.mid_side_db, reference)),
                error,
                floor,
                scatter,
                uncertainty,
                bar,
                per_key_error: stereo_median(&errors),
                per_key_floor: stereo_median(&floors),
                pass: error.is_finite() && bar.is_finite() && error <= bar,
                items: readable.len(),
                worst,
            }
        })
        .collect()
}

/// The stereo table, as `REALISM.md` prints it and as the gate prints itself
/// when it fails. One writer so the scoreboard and the gate never disagree.
pub fn stereo_report(columns: &[StereoColumn]) -> String {
    use std::fmt::Write as _;
    let mut s = String::new();
    let _ = writeln!(
        s,
        "| band | engine r@0 | reference r@0 | target r@0 | excluded | \\|err\\| | bar | floor | scatter | per-item \\|err\\| / floor | \
engine peak \\|r\\| @ lag | reference peak \\|r\\| @ lag | engine M/S | reference M/S | n |"
    );
    let _ = writeln!(
        s,
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    );
    for c in columns {
        let _ = writeln!(
            s,
            "| `{}` | {:+.3} | {:+.3} | {:+.3} | {:+.3} | {:.3} | {:.3}{} | {:.3} | {:.3} | {:.3} / {:.3} | \
{:.3} @ {:+.2} ms | {:.3} @ {:+.2} ms | {:+.1} dB | {:+.1} dB | {} |",
            c.name,
            c.engine_r0,
            c.reference_r0,
            c.target_r0,
            c.excluded_r0,
            c.error,
            c.bar,
            if c.pass { "" } else { " **RED**" },
            c.floor,
            c.scatter,
            c.per_key_error,
            c.per_key_floor,
            c.engine_peak_r,
            c.engine_lag_ms,
            c.reference_peak_r,
            c.reference_lag_ms,
            c.engine_mid_side_db,
            c.reference_mid_side_db,
            c.items
        );
    }
    s
}

// ---------------------------------------------------------------------------
// Per-channel spectral fidelity
// ---------------------------------------------------------------------------

/// **What each loudspeaker plays, as a spectrum, against what the recording's
/// own two channels play** — the dimension every other board in this repository
/// is blind to (`DECISIONS.md` 392-393).
///
/// # Why it had to exist
///
/// Every scoreboard here scores a **mono sum**, deliberately, and
/// [`stereo_image`] — the one exception — scores a *correlation*: `r0`, the peak
/// |r| and its lag, and the mid-over-side ratio. None of the four is a
/// **spectrum of one channel**, and a correlation is blind to level by
/// construction (it is normalised per channel). So a stage that leaves the mono
/// sum bit-identical and matches `r0` band for band can still put one channel
/// 9 dB up and the other 20 dB down at a particular frequency, and *nothing*
/// scores it. That is not hypothetical: the virtual microphone pair's
/// mode-controlled lobe did exactly that, a listener heard it three separate
/// ways — a melody note standing out, the hammers too loud, the reference more
/// brilliant — and all three were invisible to 696 green tests.
///
/// # The statistic
///
/// Per band, per channel, **the channel's own share of its own broadband
/// energy, minus the same take's mono share of the same band**:
///
/// ```text
/// dev_L(b) = 10 log10( E_L(b) / E_L ) − 10 log10( E_M(b) / E_M )      M = (L+R)/2
/// ```
///
/// Three properties, and each one is load-bearing:
///
/// * **It is a shape, not a level.** A gain on the whole take cancels, and so
///   does a gain on one channel — both terms are referenced to their own
///   broadband. So it needs no level match, and it cannot be passed or failed
///   by the output gain, which is fitted elsewhere.
/// * **It is exactly zero for a pan-pot**, at every band, because a pan-potted
///   pair's two channels *are* the mono sum scaled. So the number is a pure
///   measure of what a stereo stage did to each channel's spectrum, referenced
///   to the thing every other board already scores.
/// * **The recording has its own, and it is not zero** — two capsules over a
///   real soundboard see different spectra. So the target is the recording's
///   value, and the bar is the recording disagreeing with itself, exactly as in
///   [`stereo_columns`]. Nothing here is fitted to the engine.
///
/// Measured on the chords phrase at 250-500 Hz the three takes read: engine
/// **+4.49**, recording **+1.34**, and a pan-pot **+0.70** — and the engine's
/// own mono is identical to the pan-pot's sample for sample, which is the whole
/// point.
#[derive(Clone, Copy, Debug)]
pub struct ChannelBand {
    pub name: &'static str,
    pub lo_hz: f64,
    pub hi_hz: f64,
    /// `dev_L`, dB: the left channel's spectral shape against the take's own
    /// mono shape, in this band.
    pub dev_left_db: f64,
    /// `dev_R`, the same for the right channel.
    pub dev_right_db: f64,
    /// The two channels' own level difference in this band, dB (`L − R`).
    ///
    /// Not scored — it is a *position*, and where a note sits between two
    /// capsules is the pan law's business — but printed, because it is the
    /// number that separates "both channels lifted" from "one channel nulled",
    /// and those are different repairs.
    pub balance_db: f64,
    /// **What the two loudspeakers put in the room against what this take's
    /// own mono fold-down says they do**, dB: `10 log10((E_L + E_R) / 2 E_M)`,
    /// in this band.
    ///
    /// It is [`estimate::melody::pair_over_mono_db`] per band rather than per
    /// note, and it is the *loudness* dimension `dev_L`/`dev_R` cannot see:
    /// both of those are shapes, referenced to the take's own mono spectrum, so
    /// a stage that doubles the pair's energy at one frequency while leaving
    /// the fold-down alone moves neither. Zero for any pan-potted signal, and
    /// **not** zero for the recording — two capsules over a real plate do carry
    /// more energy than their sum (`DECISIONS.md` 392, 395).
    pub pair_db: f64,
    /// **This band's share of the take's own mono fold-down**, dB:
    /// `10 log10(E_M(b) / E_M)` with `M = (L + R)/2`.
    ///
    /// The plain mono spectral shape — the statistic every other board in this
    /// repository is a form of — measured here on the same renders and against
    /// the same recording as the per-channel columns, so that a stereo stage's
    /// cost to the fold-down is visible *on the stereo board* rather than only
    /// three boards away.
    ///
    /// It exists because the mode-controlled band stopped being free. Until
    /// `DECISIONS.md` 392 the microphone section left `(L + R)/2` bit-identical
    /// at every setting, so no mono statistic could move and none was needed
    /// here; a nodal line that costs the mono sum what the pair gains
    /// (`soundboard::ModalRotation`) makes it the other half of the trade, and a
    /// fit that could see `pair_db` and not this one would buy the per-channel
    /// board with the fold-down and call it progress.
    pub mono_db: f64,
    /// This band's share of the whole take's energy, dB; see
    /// [`STEREO_BAND_FLOOR_DB`].
    pub level_db: f64,
}

impl ChannelBand {
    /// Is there enough in this band for the ratio of two energies to mean
    /// anything?
    pub fn readable(&self) -> bool {
        self.level_db.is_finite()
            && self.level_db > STEREO_BAND_FLOOR_DB
            && self.dev_left_db.is_finite()
            && self.dev_right_db.is_finite()
    }

    /// The worse of the two channels, which is what the column scores: a stage
    /// that ruins one loudspeaker and leaves the other alone has ruined the
    /// image, and an average over the two would forgive it by half.
    pub fn worse_db(&self) -> f64 {
        if self.dev_left_db.abs() >= self.dev_right_db.abs() {
            self.dev_left_db
        } else {
            self.dev_right_db
        }
    }
}

/// One take's per-channel spectral shape, band by band.
#[derive(Clone, Debug)]
pub struct ChannelShape {
    pub broadband: ChannelBand,
    /// Per [`STEREO_BANDS`].
    pub bands: Vec<ChannelBand>,
    /// Per [`STEREO_FINE_BANDS`] — the same statistics at the resolution the
    /// mode-controlled band's own shape lives at. Computed from the same two
    /// spectra and against the same broadband totals, so a fine row and a
    /// coarse row are the same measurement read at two widths.
    pub fine: Vec<ChannelBand>,
}

fn channel_band(
    a: &[Complex32],
    b: &[Complex32],
    sample_rate: f64,
    band: Option<(&'static str, f64, f64)>,
    totals: Option<([f64; 3], f64)>,
) -> (ChannelBand, [f64; 3], f64) {
    let n = a.len();
    let mut acc = [0.0f64; 3];
    let mut add = |j: usize| {
        let (x, y) = (a[j], b[j]);
        acc[0] += f64::from(x.norm_sqr());
        acc[1] += f64::from(y.norm_sqr());
        acc[2] += f64::from(((x + y) * 0.5).norm_sqr());
    };
    match band {
        None => {
            for j in 0..n {
                add(j);
            }
        }
        Some((_, lo, hi)) => {
            let bin = |hz: f64| (hz * n as f64 / sample_rate).round() as usize;
            let (blo, bhi) = (bin(lo).max(1), bin(hi).min(n / 2));
            if bhi >= blo {
                for j in blo..=bhi {
                    add(j);
                    if n - j != j {
                        add(n - j);
                    }
                }
            }
        }
    }
    let scale = 1.0 / n as f64;
    let e = [acc[0] * scale, acc[1] * scale, acc[2] * scale];
    let (name, lo, hi) = band.unwrap_or(("broadband", 0.0, sample_rate / 2.0));
    let pair = e[0] + e[1];
    let (whole, whole_pair) = totals.unwrap_or((e, pair));
    // Every one of these is a ratio of a band's energy to the *same signal's*
    // whole energy, so a gain on that signal — on the take, or on one channel
    // alone — cancels in both terms of the difference.
    let share = |i: usize| 10.0 * (e[i] / whole[i].max(1e-300)).log10();
    let (sl, sr, sm) = (share(0), share(1), share(2));
    (
        ChannelBand {
            name,
            lo_hz: lo,
            hi_hz: hi,
            dev_left_db: sl - sm,
            dev_right_db: sr - sm,
            pair_db: 10.0 * (pair / (2.0 * e[2]).max(1e-300)).log10(),
            mono_db: 10.0 * (e[2] / whole[2].max(1e-300)).log10(),
            balance_db: 10.0 * (e[0] / e[1].max(1e-300)).log10(),
            level_db: 10.0 * (pair / whole_pair.max(1e-300)).log10(),
        },
        e,
        pair,
    )
}

/// The per-channel spectral shape of one stereo signal: broadband and per
/// [`STEREO_BANDS`], the same band set the rest of the stereo work scores on.
pub fn channel_shape(left: &[f32], right: &[f32], sample_rate: f64) -> Result<ChannelShape> {
    if left.is_empty() || right.is_empty() {
        return Err(Error::Config("a channel shape needs two channels".into()));
    }
    // No lag search here and so no inverse transform: half the size
    // [`stereo_image`] needs, and one forward pair is the whole cost.
    let n = left.len().max(right.len()).next_power_of_two();
    let mut planner = FftPlanner::<f32>::new();
    let a = stereo_spectrum(left, n, &mut planner);
    let b = stereo_spectrum(right, n, &mut planner);
    let (broadband, totals, total_pair) = channel_band(&a, &b, sample_rate, None, None);
    let bands = STEREO_BANDS
        .iter()
        .map(|&band| channel_band(&a, &b, sample_rate, Some(band), Some((totals, total_pair))).0)
        .collect();
    let fine = STEREO_FINE_BANDS
        .iter()
        .map(|&band| channel_band(&a, &b, sample_rate, Some(band), Some((totals, total_pair))).0)
        .collect();
    Ok(ChannelShape {
        broadband,
        bands,
        fine,
    })
}

/// [`channel_shape`] of an [`Audio`]'s first two channels.
pub fn channel_shape_of(audio: &Audio) -> Result<ChannelShape> {
    if audio.channel_count() < 2 {
        return Err(Error::Config("a channel shape needs two channels".into()));
    }
    channel_shape(
        &audio.channels[0],
        &audio.channels[1],
        f64::from(audio.sample_rate),
    )
}

/// One thing — a phrase, or a key — measured three ways, as [`StereoItem`] is.
#[derive(Clone, Debug)]
pub struct ChannelItem {
    pub label: String,
    pub engine: ChannelShape,
    pub reference: ChannelShape,
    pub alternate: ChannelShape,
}

/// One band of the per-channel scoreboard, pooled over the items.
#[derive(Clone, Debug)]
pub struct ChannelColumn {
    pub band: usize,
    pub name: &'static str,
    pub lo_hz: f64,
    pub hi_hz: f64,
    /// Median `dev_L` and `dev_R` over the items, all three takes.
    pub engine_left_db: f64,
    pub engine_right_db: f64,
    pub reference_left_db: f64,
    pub reference_right_db: f64,
    pub alternate_left_db: f64,
    pub alternate_right_db: f64,
    /// **The score**: the worse of the two channels' `|engine − reference|`,
    /// on the medians.
    pub error: f64,
    /// **The floor**: the identical statistic between the recording and its own
    /// second take.
    pub floor: f64,
    /// Robust sigma of the reference's own worse-channel value across the items.
    pub scatter: f64,
    /// `scatter / sqrt(items)`.
    pub uncertainty: f64,
    /// `max(floor, uncertainty) · `[`STEREO_ALLOWANCE`].
    pub bar: f64,
    /// **The capsule-placement asymmetry, excluded from the target rather than
    /// chased** (`DECISIONS.md` 417, and item 328's rule applied to a second
    /// property of the reference).
    ///
    /// `|reference_left_db − reference_right_db| / 2`, in decibels, out of the
    /// recording alone.
    ///
    /// **What it is.** The recording's two capsules do not straddle the board's
    /// nodal lines symmetrically — they are one session's placement — so the
    /// recording's own `spread = dev_L − dev_R` runs to **+5.85 dB at 178 Hz**
    /// where an incoherent side source can move it by nothing at all and a
    /// coherent one only through uncontrollable cross terms
    /// (`renders/side-injection/SIDE_INJECTION.md` §5f, item 417's refutation).
    /// Modelling it would mean per-channel per-band gains fitted to where two
    /// microphones stood on one afternoon, which the standing no-room and
    /// no-mic-idiosyncrasy policy refuses: it is a property of the reference,
    /// not of a piano.
    ///
    /// **Why exactly half the spread.** [`Self::error`] is
    /// `max(|dev_L − ref_L|, |dev_R − ref_R|)`. An engine whose two channels
    /// depart *symmetrically* from their own mono has `dev_L = dev_R = x`, and
    /// `max(|x − rl|, |x − rr|)` is minimised at the midpoint, where it is
    /// `|rl − rr| / 2` **whatever `x` is chosen**. So half the reference spread
    /// is the floor this statistic puts under any symmetric model: such a model
    /// cannot read better than this in a band however well it is fitted.
    ///
    /// **This engine is not in that class, and the difference is measured**
    /// (`DECISIONS.md` 424). A nodal-line lobe is `L = m(1 + g)` and
    /// `R = m(1 − g)` — per-channel *by construction* — so `dev_L ≠ dev_R`
    /// here, and the diffuser's rotation moves which channel is up with
    /// frequency. On the shipped instrument the engine's own spread reads
    /// **−1.27 dB at 125-250 Hz and +3.64 at 250-500** against the reference's
    /// **+1.52 and −1.30**: opposite in sign in both bands, and larger than the
    /// reference's in the second. So for this model the exclusion is **not** an
    /// unreachable floor — the section could in principle put its spread on the
    /// reference's side of the ledger, and doing so would be fitting
    /// per-channel per-band gains to where two microphones stood on one
    /// afternoon, which is exactly what item 417 refused. **The floor argument
    /// sizes the exclusion; the policy is what justifies it**, and stating it
    /// the other way round would be claiming an arithmetic necessity this
    /// construction does not have.
    ///
    /// **Three things keep it from being a widened bar, and the third is the
    /// one that binds today.** (i) It is computed out of the reference's two
    /// medians and nothing else, so like [`Self::bar`] it cannot be fitted to
    /// and cannot move when the engine does. (ii)
    /// `the_acceptance_still_fails_on_the_lobe_it_was_re_barred_against`
    /// asserts that the pre-418 unclamped lobe is still red against
    /// [`Self::reachable`], so the exclusion is narrower than the defect it
    /// could be accused of hiding. (iii) On the shipped instrument it **changes
    /// no verdict the gate asserts**: the two bands asserted absolutely are red
    /// with it and without it (2.38 dB against an unexcluded bar of 1.15 and a
    /// reachable 1.91; 2.47 against 1.09 and 1.74), and the other four bands
    /// are asserted only against a pan-potted engine's own error with the
    /// *unexcluded* `bar` as the slack. Where the exclusion does change a
    /// target is the **fit** — `mics::channel_excess` reads
    /// [`Self::reachable`] — so that the fit and the gate close on one
    /// definition of what is being asked for.
    ///
    /// The gate prints it, and the engine's own spread beside it, before it
    /// asserts: an acceptance nobody can read is indistinguishable from a
    /// widened bar.
    pub asymmetry: f64,
    /// **The bar the verdict is actually taken against**: `bar + asymmetry`,
    /// the recording's own target with item 417's acceptance subtracted from
    /// it.
    ///
    /// `bar` alone is what the recording asks of a model that could place its
    /// capsules where the session did. [`Self::asymmetry`] is the part of that
    /// ask which item 417 accepted as unscored, and this is the remainder. Read
    /// [`Self::asymmetry`] for why the subtraction is a policy sized by an
    /// arithmetic floor rather than an arithmetic necessity, and for the three
    /// things that keep it from being a widened bar — on the shipped instrument
    /// it changes no verdict this gate asserts, and both bands it is largest in
    /// are red with it and without it.
    pub reachable: f64,
    /// **The per-item distance**: the same worse-of-two-channels rule applied
    /// key by key and then taken at the median, so a band that is right on
    /// average and wrong at every key is visible here and nowhere else. This is
    /// the column the melody's C4 lived in.
    ///
    /// It is gated — in the lobe's own two bands, and against
    /// [`ChannelColumn::scatter`] rather than against [`Self::per_key_floor`].
    /// The floor is the recording against its own second take, **0.11-0.19 dB**:
    /// the same key through the same two capsules twice, which repeats almost
    /// exactly. No microphone model with three fitted numbers can reproduce
    /// *which* plate mode falls where at every one of thirty keys, and a bar
    /// that demanded it would be a gate nobody could pass and therefore no gate
    /// at all. What the recording's own **key-to-key** sigma is, on the other
    /// hand, is a real ceiling and a measured one: it says the engine's
    /// per-key disagreement with the recording must not exceed the spread the
    /// recording itself has across keys. A lobe sitting half a cent from a
    /// melody note (`DECISIONS.md` 392) leaves that ceiling; a per-key lottery
    /// of the recording's own size does not. `DECISIONS.md` 395.
    pub per_key_error: f64,
    pub per_key_floor: f64,
    /// `scatter · `[`STEREO_ALLOWANCE`] — the bar [`Self::per_key_error`] is
    /// read against, and the reason is in that field's own comment.
    pub per_key_bar: f64,
    pub per_key_pass: bool,
    /// Median [`ChannelBand::pair_db`] over the items, all three takes.
    pub engine_pair_db: f64,
    pub reference_pair_db: f64,
    pub alternate_pair_db: f64,
    /// **The target [`Self::pair_balance`] is taken against**:
    /// `min(reference_pair_db, `[`NEUTRAL_PAIR_CEILING_DB`]`)` since
    /// `DECISIONS.md` 486. Equal to [`Self::reference_pair_db`] in every band
    /// where the recording's pair carries no more side than sum.
    pub target_pair_db: f64,
    /// `reference_pair_db − target_pair_db`: how much of the recording's own
    /// pair energy this board has stopped asking for, in decibels, and zero in
    /// every band under the ceiling. Printed by the gate before it asserts.
    pub excluded_pair_db: f64,
    /// **The loudness score**: the median over the items of the engine's
    /// `pair_db` less the recording's, *signed*, because the recording has its
    /// own value at every key and the question is whether the engine's is the
    /// recording's — the `strike`/`channel` balance construction of
    /// `estimate::melody`, on thirty keys instead of nine.
    pub pair_balance: f64,
    /// The same statistic between the recording and its own second take.
    pub pair_floor: f64,
    /// Robust sigma of the recording's own `pair_db` across the items.
    pub pair_scatter: f64,
    /// `max(pair_floor, pair_scatter / sqrt(items)) · `[`STEREO_ALLOWANCE`].
    pub pair_bar: f64,
    pub pair_pass: bool,
    /// Median [`ChannelBand::mono_db`] over the items, engine and recording.
    pub engine_mono_db: f64,
    pub reference_mono_db: f64,
    /// **The fold-down score**: the median over the items of the engine's mono
    /// share less the recording's, *signed*. Built exactly as
    /// [`Self::pair_balance`] is, on the same items, so the two halves of what a
    /// nodal line does — what the pair gains and what the sum loses — are read
    /// in the same units against the same kind of bar.
    pub mono_balance: f64,
    pub mono_floor: f64,
    pub mono_scatter: f64,
    pub mono_bar: f64,
    pub mono_pass: bool,
    /// **The same fold-down question, pooled by energy instead of by rank**
    /// (`DECISIONS.md` 411's open item, settled in 412).
    ///
    /// [`Self::mono_balance`] is `stereo_median` over the items of each item's
    /// own `mono_db` difference, so a key carrying **1 %** of a band's energy
    /// and a key carrying **42 %** move it equally. Item 407(a) measured what
    /// that costs: the 252 Hz row reads the treble keys' sub-fundamental floor
    /// (+4.7 to +16.8 dB on keys carrying 1-2 % each) while C3 and C4, who own
    /// 80 % of the band, read +1.4 and +4.2.
    ///
    /// This is the estimator the *fit* uses — [`mono_columns`] — read on the
    /// board's own items and bands: every item's level-matched band energy
    /// summed, engine over recording. Where a band's energy is spread evenly
    /// the two agree; where it is not, the difference between them is the
    /// measurement of that, and both are printed for exactly that reason.
    ///
    /// It is deliberately **not** what `mono_pass` is decided on. Item 411
    /// refused to move the number every item from 405 onward is anchored to
    /// inside the milestone whose finding is that nothing moved; this milestone
    /// moves the instrument instead, so re-anchoring the bar in the same breath
    /// would make the two changes impossible to tell apart. The median keeps
    /// the gate and the pooled column keeps it honest.
    pub mono_pooled: f64,
    /// The recording's own second take through [`Self::mono_pooled`] — the
    /// floor that column would be read against.
    pub mono_pooled_floor: f64,
    pub pass: bool,
    pub items: usize,
    pub worst: Option<(String, f64)>,
}

/// The per-channel scoreboard: one row per band, each loudspeaker's spectrum
/// against the recording's own, with the recording's disagreement with itself
/// beside it.
///
/// The construction is [`stereo_columns`]' exactly — median against median, bar
/// out of the reference and its alternate take alone — so the two boards are
/// read the same way and neither can be fitted to.
pub fn channel_columns(items: &[ChannelItem]) -> Vec<ChannelColumn> {
    channel_columns_over(items, &STEREO_BANDS, |s| &s.bands)
}

/// **The same board at sixth-octave resolution**, over [`STEREO_FINE_BANDS`].
///
/// Every column here is the identical statistic to [`channel_columns`]' — the
/// same medians, the same bars out of the recording's own second take and its
/// own key-to-key spread — read over a band a sixth of an octave wide instead
/// of a whole one. It exists because two of the three things the per-channel
/// board scores are *shapes over frequency* and an octave is wider than the
/// shape: `DECISIONS.md` 403 measures the mode-controlled band overshooting the
/// recording's pair-over-mono by up to 8 dB at the top of its own band while
/// the octave column that contains it reads inside its bar, and the fold-down's
/// cost hiding in the same average.
pub fn channel_fine_columns(items: &[ChannelItem]) -> Vec<ChannelColumn> {
    channel_columns_over(items, &STEREO_FINE_BANDS, |s| &s.fine)
}

/// One board over one band set. The two public boards differ only in which
/// bands they read and which of [`ChannelShape`]'s two lists they read them
/// from, so there is one implementation and no way for them to drift apart.
fn channel_columns_over(
    items: &[ChannelItem],
    table: &[(&'static str, f64, f64)],
    of: fn(&ChannelShape) -> &Vec<ChannelBand>,
) -> Vec<ChannelColumn> {
    table
        .iter()
        .enumerate()
        .map(|(b, &(name, lo, hi))| {
            let readable: Vec<&ChannelItem> = items
                .iter()
                .filter(|it| {
                    of(&it.engine)[b].readable()
                        && of(&it.reference)[b].readable()
                        && of(&it.alternate)[b].readable()
                })
                .collect();
            let pick = |f: fn(&ChannelBand) -> f64, side: fn(&ChannelItem) -> &ChannelShape| {
                readable
                    .iter()
                    .map(|it| f(&of(side(it))[b]))
                    .collect::<Vec<f64>>()
            };
            fn engine(it: &ChannelItem) -> &ChannelShape {
                &it.engine
            }
            fn reference(it: &ChannelItem) -> &ChannelShape {
                &it.reference
            }
            fn alternate(it: &ChannelItem) -> &ChannelShape {
                &it.alternate
            }
            let el = stereo_median(&pick(|x| x.dev_left_db, engine));
            let er = stereo_median(&pick(|x| x.dev_right_db, engine));
            let rl = stereo_median(&pick(|x| x.dev_left_db, reference));
            let rr = stereo_median(&pick(|x| x.dev_right_db, reference));
            let al = stereo_median(&pick(|x| x.dev_left_db, alternate));
            let ar = stereo_median(&pick(|x| x.dev_right_db, alternate));
            let error = (el - rl).abs().max((er - rr).abs());
            let floor = (al - rl).abs().max((ar - rr).abs());
            // The per-item distance uses the same "worse channel" rule item by
            // item, so a key that is wrong in the left and a key that is wrong
            // in the right both count.
            let per_item = |side: fn(&ChannelItem) -> &ChannelShape| -> Vec<f64> {
                readable
                    .iter()
                    .map(|it| {
                        let x = of(side(it))[b];
                        let r = of(&it.reference)[b];
                        (x.dev_left_db - r.dev_left_db)
                            .abs()
                            .max((x.dev_right_db - r.dev_right_db).abs())
                    })
                    .collect()
            };
            let errors = per_item(engine);
            let floors = per_item(alternate);
            // The loudness column: signed, per item, against the recording's
            // own value at that key.
            // **The target is the recording's own `pair_db` or the neutral
            // policy's ceiling on the side energy, whichever asks for less
            // side** (`DECISIONS.md` 486, `NEUTRAL_PAIR_CEILING_DB`). This is
            // the coherence board's own re-bar read in the other of the two
            // units item 418 says the shortfall lives in: `pair_db` is
            // `10 log10(1 + E_side/E_mid)` and `E_side/E_mid = 1` is +3.0103 dB.
            let pair_target = |r: f64| r.min(NEUTRAL_PAIR_CEILING_DB);
            let pair_of = |side: fn(&ChannelItem) -> &ChannelShape| -> Vec<f64> {
                readable
                    .iter()
                    .map(|it| of(side(it))[b].pair_db - pair_target(of(&it.reference)[b].pair_db))
                    .collect()
            };
            let mono_of = |side: fn(&ChannelItem) -> &ChannelShape| -> Vec<f64> {
                readable
                    .iter()
                    .map(|it| of(side(it))[b].mono_db - of(&it.reference)[b].mono_db)
                    .collect()
            };
            let mono_balance = stereo_median(&mono_of(engine));
            let mono_floor = stereo_median(&mono_of(alternate)).abs();
            // Pooled: the level-matched band energies summed over the items
            // and the two sums divided, which is `mono_columns`' estimator on
            // the board's own material.
            let mono_pooled_of = |side: fn(&ChannelItem) -> &ChannelShape| -> f64 {
                let (mut e, mut r) = (0.0f64, 0.0f64);
                for it in &readable {
                    e += 10.0f64.powf(of(side(it))[b].mono_db / 10.0);
                    r += 10.0f64.powf(of(&it.reference)[b].mono_db / 10.0);
                }
                10.0 * (e / r).log10()
            };
            let mono_pooled = mono_pooled_of(engine);
            let mono_pooled_floor = mono_pooled_of(alternate).abs();
            let mono_scatter = stereo_sigma(&pick(|x| x.mono_db, reference));
            let mono_bar = mono_floor.max(if readable.is_empty() {
                f64::NAN
            } else {
                mono_scatter / (readable.len() as f64).sqrt()
            }) * STEREO_ALLOWANCE;
            let pair_balance = stereo_median(&pair_of(engine));
            let pair_floor = stereo_median(&pair_of(alternate)).abs();
            let pair_scatter = stereo_sigma(&pick(|x| x.pair_db, reference));
            let pair_bar = pair_floor.max(if readable.is_empty() {
                f64::NAN
            } else {
                pair_scatter / (readable.len() as f64).sqrt()
            }) * STEREO_ALLOWANCE;
            let scatter = stereo_sigma(&pick(ChannelBand::worse_db, reference));
            let uncertainty = if readable.is_empty() {
                f64::NAN
            } else {
                scatter / (readable.len() as f64).sqrt()
            };
            let bar = floor.max(uncertainty) * STEREO_ALLOWANCE;
            // The capsule-placement asymmetry item 417 accepted as unscored,
            // out of the reference's own two medians and nothing else. See
            // `ChannelColumn::asymmetry` for why it is exactly half the spread.
            let asymmetry = (rl - rr).abs() / 2.0;
            let reachable = bar + asymmetry;
            let worst = errors
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, &d)| (readable[i].label.clone(), d));
            ChannelColumn {
                band: b,
                name,
                lo_hz: lo,
                hi_hz: hi,
                engine_left_db: el,
                engine_right_db: er,
                reference_left_db: rl,
                reference_right_db: rr,
                alternate_left_db: al,
                alternate_right_db: ar,
                error,
                floor,
                scatter,
                uncertainty,
                bar,
                asymmetry,
                reachable,
                per_key_error: stereo_median(&errors),
                per_key_floor: stereo_median(&floors),
                per_key_bar: scatter * STEREO_ALLOWANCE,
                per_key_pass: {
                    let e = stereo_median(&errors);
                    e.is_finite() && scatter.is_finite() && e <= scatter * STEREO_ALLOWANCE
                },
                engine_pair_db: stereo_median(&pick(|x| x.pair_db, engine)),
                reference_pair_db: stereo_median(&pick(|x| x.pair_db, reference)),
                alternate_pair_db: stereo_median(&pick(|x| x.pair_db, alternate)),
                target_pair_db: pair_target(stereo_median(&pick(|x| x.pair_db, reference))),
                excluded_pair_db: stereo_median(&pick(|x| x.pair_db, reference))
                    - pair_target(stereo_median(&pick(|x| x.pair_db, reference))),
                pair_balance,
                pair_floor,
                pair_scatter,
                pair_bar,
                pair_pass: pair_balance.is_finite()
                    && pair_bar.is_finite()
                    && pair_balance.abs() <= pair_bar,
                engine_mono_db: stereo_median(&pick(|x| x.mono_db, engine)),
                reference_mono_db: stereo_median(&pick(|x| x.mono_db, reference)),
                mono_balance,
                mono_floor,
                mono_scatter,
                mono_bar,
                mono_pass: mono_balance.is_finite()
                    && mono_bar.is_finite()
                    && mono_balance.abs() <= mono_bar,
                mono_pooled,
                mono_pooled_floor,
                // The verdict is taken against the *reachable* bar: item 417's
                // accepted exclusion is subtracted from the target here and
                // nowhere else, so there is one definition of it and the gate,
                // the boards and the fit all read the same one.
                pass: error.is_finite() && reachable.is_finite() && error <= reachable,
                items: readable.len(),
                worst,
            }
        })
        .collect()
}

/// The per-channel table, as the boards print it and as the gate prints itself
/// when it fails. One writer, for the reason [`stereo_report`] has one.
pub fn channel_report(columns: &[ChannelColumn]) -> String {
    use std::fmt::Write as _;
    let mut s = String::new();
    let _ = writeln!(
        s,
        "| band | engine L / R | reference L / R | alternate L / R | \\|err\\| | reachable | \
bar | excl. asym | floor | \
scatter | per-item \\|err\\| / floor | pair E / R | pair target | excl. side | balance | bar | mono E / R | balance | bar | \
pooled | pooled floor | worst | n |"
    );
    let _ = writeln!(
        s,
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|"
    );
    for c in columns {
        let _ = writeln!(
            s,
            "| `{}` | {:+.2} / {:+.2} | {:+.2} / {:+.2} | {:+.2} / {:+.2} | {:.2} | {:.2}{} | \
{:.2} | {:.2} | \
{:.2} | {:.2} | {:.2}{} / {:.2} | {:+.2} / {:+.2} | {:+.2} | {:.2} | {:+.2}{} | {:.2} | {:+.2} / {:+.2} | \
{:+.2}{} | {:.2} | {:+.2} | {:.2} | {} | {} |",
            c.name,
            c.engine_left_db,
            c.engine_right_db,
            c.reference_left_db,
            c.reference_right_db,
            c.alternate_left_db,
            c.alternate_right_db,
            c.error,
            c.reachable,
            if c.pass { "" } else { " **RED**" },
            c.bar,
            c.asymmetry,
            c.floor,
            c.scatter,
            c.per_key_error,
            if c.per_key_pass { "" } else { "!" },
            c.per_key_floor,
            c.engine_pair_db,
            c.reference_pair_db,
            c.target_pair_db,
            c.excluded_pair_db,
            c.pair_balance,
            if c.pair_pass { "" } else { " **RED**" },
            c.pair_bar,
            c.engine_mono_db,
            c.reference_mono_db,
            c.mono_balance,
            if c.mono_pass { "" } else { " **RED**" },
            c.mono_bar,
            c.mono_pooled,
            c.mono_pooled_floor,
            c.worst
                .as_ref()
                .map_or_else(|| "—".to_string(), |(k, d)| format!("{k} {d:.2}")),
            c.items
        );
    }
    s
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
        let spec =
            mel_spectrogram(&signal, SR, 4096, 1024, MEL_BANDS, MEL_F_MIN, MEL_F_MAX).unwrap();
        let mid = spec.frames.len() / 2;
        let (best, _) =
            spec.frames[mid]
                .iter()
                .enumerate()
                .fold(
                    (0usize, 0.0f64),
                    |acc, (i, &e)| if e > acc.1 { (i, e) } else { acc },
                );
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
                    std::iter::repeat(0.0)
                        .take(shift)
                        .chain(x.iter().copied())
                        .collect(),
                    x.clone(),
                )
            } else {
                (
                    x.clone(),
                    std::iter::repeat(0.0)
                        .take(shift)
                        .chain(x.iter().copied())
                        .collect(),
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
            let perturbed: Vec<f32> = base
                .iter()
                .zip(added.iter())
                .map(|(&a, &b)| a + b)
                .collect();
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
        let perturbed: Vec<f32> = base
            .iter()
            .zip(added.iter())
            .map(|(&a, &b)| a + b)
            .collect();
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

    /// `DECISIONS.md` 338: each side of the comparison is windowed on **its
    /// own** strike, so a player that leads the other by a couple of dozen
    /// milliseconds is still compared attack against attack.
    #[test]
    fn each_side_of_the_attack_is_windowed_on_its_own_strike() {
        // One note, twice: the same tone with the same noisy attack, but the
        // second one begins 25 ms later — the sampler's own lead-in, which is
        // +19 ms at the median over Salamander's recorded keys.
        let n = (0.6 * SR) as usize;
        let note = |delay_s: f64| -> Vec<f32> {
            let start = (delay_s * SR) as usize;
            let mut out = vec![0.0f32; n];
            let mut state = 0x1234_5678_9abc_def0u64;
            for i in 0..(n - start) {
                let t = i as f64 / SR;
                let mut x = 0.3 * (2.0 * PI * 440.0 * t).sin() * (-3.0 * t).exp();
                if t < 0.02 {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let r = ((state >> 33) as f64 / f64::from(u32::MAX >> 1)) - 1.0;
                    x += 0.3 * r * (1.0 - t / 0.02);
                }
                out[start + i] = x as f32;
            }
            out
        };
        let early = note(0.05);
        let late = note(0.075);
        // The onset the detector finds in the *late* signal, which is what the
        // board hands both sides.
        let onsets = detect_onsets(&late, SR).unwrap();
        assert_eq!(onsets.len(), 1, "{onsets:?}");
        let d = attack_tonality_delta(&early, &late, SR, &onsets);
        assert_eq!(d.onsets, 1);
        assert!(
            d.mean_abs_db < 1.0,
            "two identical attacks 25 ms apart read {:.2} dB apart",
            d.mean_abs_db
        );
        // And the search is what does it: measured at the onset itself, the
        // early signal is read 25 ms into its own decay and comes back far more
        // tonal, because the noise it began with is over.
        let len = (ATTACK_WINDOW_S * SR) as usize;
        let at = (onsets[0] * SR) as usize;
        let naive = attack_tonality_db(&early[at..at + len], SR)
            - attack_tonality_db(&late[at..at + len], SR);
        assert!(
            naive > 3.0,
            "the mis-placed window was supposed to be the defect, and it reads {naive:.2} dB"
        );
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
        assert!(
            d.rise_s.0 < 0.003 && (d.rise_s.1 - 0.020).abs() < 0.003,
            "{:?}",
            d.rise_s
        );
    }

    #[test]
    fn the_band_correlation_falls_when_one_register_moves_differently() {
        let bass = plucks(80.0, 0.7, 1.2, 4.0, SR);
        let treble = plucks(3_000.0, 0.5, 0.6, 4.0, SR);
        let together: Vec<f32> = bass
            .iter()
            .zip(treble.iter())
            .map(|(&a, &b)| a + b)
            .collect();
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
        assert!(
            (d.mean_signed_db - 6.02).abs() < 0.05,
            "{}",
            d.mean_signed_db
        );
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
            assert!(
                phrase.note_count() >= 8,
                "{} has {} notes",
                phrase.name,
                phrase.note_count()
            );

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
                        assert!(
                            at.is_some(),
                            "{}: key {key} released without a strike",
                            phrase.name
                        );
                        held.remove(at.unwrap());
                    }
                    SamplerEvent::Sustain(v) => {
                        // Half-pedal is not comparable: the reference player
                        // reads CC 64 as a switch. Only the stops are used.
                        assert!(v == 0.0 || v == 1.0, "{}: half pedal {v}", phrase.name);
                    }
                    other => panic!(
                        "{}: {other:?} is not comparable between the two players",
                        phrase.name
                    ),
                }
            }
            assert!(held.is_empty(), "{}: {held:?} never released", phrase.name);
        }
    }

    #[test]
    fn the_pedalled_phrases_are_the_ones_that_use_the_pedal() {
        let pedalled: Vec<&str> = phrase_set()
            .iter()
            .filter(|p| {
                p.events
                    .iter()
                    .any(|e| matches!(e.event, SamplerEvent::Sustain(_)))
            })
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
            .filter(|&&t| {
                !ons.iter()
                    .any(|&on| on > t - 0.02 && on < t + RELEASE_WINDOW_S)
            })
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
                (Some(a), Some(b)) => {
                    assert_ne!(a, b, "velocity {vel} -> {alt} stayed in band {a}")
                }
                (None, _) => assert_eq!(alt, vel, "an unmapped velocity must be left alone"),
                (Some(_), None) => panic!("velocity {vel} -> {alt}, which is in no layer"),
            }
        }
        // The silent press is a gesture, not a dynamic.
        assert_eq!(layers.alternate(0), 0);
        // The top layer has to borrow from below.
        assert!(layers.alternate(125) < 121);
    }

    // ---- the evaluation policy -------------------------------------------

    /// The Salamander mapping: one take every minor third, A0 to C8.
    fn minor_thirds() -> RecordedKeys {
        RecordedKeys::from_keys(&(21u8..=108).step_by(3).collect::<Vec<u8>>())
    }

    #[test]
    fn a_key_is_either_a_recording_or_the_nearest_one_transposed() {
        let keys = minor_thirds();
        assert_eq!(keys.keys().len(), 30);
        for key in 21u8..=108 {
            let take = keys.take_for(key).expect("every key is reachable");
            assert!(keys.is_recorded(take), "{take} is not a take");
            assert!(
                key.abs_diff(take) <= 1,
                "key {key} plays {take}, {} semitones away",
                key.abs_diff(take)
            );
            assert_eq!(keys.is_recorded(key), take == key);
        }
        // Named the way a report names it.
        assert_eq!(keys.provenance(60), "recorded");
        assert_eq!(keys.provenance(62), "transposed from D#4 (-1)");
        assert_eq!(keys.provenance(64), "transposed from D#4 (+1)");
    }

    #[test]
    fn the_second_route_onto_a_transposed_key_is_the_other_neighbour() {
        let keys = minor_thirds();
        // D4 is D#4 down a semitone, or C4 up two.
        assert_eq!(keys.take_for(62), Some(63));
        assert_eq!(keys.alternate_take(62), Some(60));
        // E4 is D#4 up, or F#4 down two.
        assert_eq!(keys.take_for(64), Some(63));
        assert_eq!(keys.alternate_take(64), Some(66));
        // A recorded key needs no route and is offered none.
        assert_eq!(keys.alternate_take(60), None);
        // The routing map: recorded keys keep themselves, everything else swaps.
        let route = keys.routing();
        for key in 21u8..=108 {
            let to = route(key).expect("every key routes somewhere");
            assert!(keys.is_recorded(to));
            if keys.is_recorded(key) {
                assert_eq!(to, key, "{key} is a take and must stay on it");
            } else {
                assert_ne!(
                    to,
                    keys.take_for(key).unwrap(),
                    "{key} was not rerouted at all"
                );
            }
        }
    }

    #[test]
    fn the_register_a_bar_is_measured_in_is_the_recorded_keys_of_that_register() {
        let keys = minor_thirds();
        assert_eq!(
            keys.in_range(51, 76),
            vec![51, 54, 57, 60, 63, 66, 69, 72, 75]
        );
        assert!(keys.in_range(61, 62).is_empty());
    }

    #[test]
    fn a_library_of_one_key_cannot_be_scored_against() {
        let one = RecordedKeys::from_keys(&[60]);
        assert!(one.is_recorded(60));
        assert_eq!(one.take_for(21), Some(60));
        assert_eq!(one.alternate_take(21), None);
        // With nothing else in reach, the routing leaves the key where it was.
        assert_eq!(one.routing()(21), Some(60));
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
                (
                    SamplerEvent::NoteOn { key: k1, vel: v1 },
                    SamplerEvent::NoteOn { key: k2, vel: v2 },
                ) => {
                    assert_eq!(k1, k2);
                    assert_ne!(v1, v2);
                }
                (x, y) => assert_eq!(x, y),
            }
        }
    }

    // ---- Columns A and B -------------------------------------------------

    fn cell(
        key: u8,
        k: u32,
        velocity: u8,
        engine: (f64, f64, f64),
        reference: (f64, f64, f64),
    ) -> MotionCell {
        let motion = |(band, placement, depth): (f64, f64, f64)| Motion {
            mean_hz: 440.0,
            peak_db: 40.0,
            band_cents: band,
            raw_cents: 1.0,
            weighted_cents: placement,
            beat_depth_db: depth,
            beat_rate_hz: 1.0,
            prompt_db_s: -12.0,
            tail_db_s: -6.0,
            aftersound_db: 10.0,
        };
        MotionCell {
            key,
            k,
            velocity,
            engine: Some(motion(engine)),
            reference: Some(motion(reference)),
        }
    }

    /// An exact resynthesis of the recording scores 1 / 1 / 0, which is
    /// `ANALYSIS.md` §7's certification rule for any new metric: near-zero for
    /// the thing that is the reference by construction.
    #[test]
    fn the_columns_score_a_perfect_copy_at_their_own_identity() {
        let mut cells = Vec::new();
        for (i, &(key, _)) in MOTION_KEYS.iter().enumerate() {
            for k in 1..=MOTION_PARTIALS {
                for (j, &velocity) in MOTION_VELOCITIES.iter().enumerate() {
                    let v = (1.0 + i as f64 + 0.4 * j as f64, 0.5, 3.0 + j as f64);
                    cells.push(cell(key, k, velocity, v, v));
                }
            }
        }
        let columns = motion_columns(&cells);
        assert_eq!(columns.cells, 16);
        assert_eq!(columns.velocity_cells, 16);
        assert!(
            (columns.if_mismatch - 1.0).abs() < 1e-12,
            "{}",
            columns.if_mismatch
        );
        assert!((columns.if_placement - 1.0).abs() < 1e-12);
        assert!(columns.beat_depth_error_db < 1e-12);
        assert!((columns.velocity_coherence - 1.0).abs() < 1e-12);
        assert!(columns.passes());
    }

    /// The errata's clamp, which is the difference between "both signals are at
    /// the measurement's floor" and "the engine is thirty times too still".
    #[test]
    fn two_cells_at_the_floor_are_not_a_mismatch() {
        let cells: Vec<MotionCell> = (1..=MOTION_PARTIALS)
            .map(|k| {
                cell(
                    60,
                    k,
                    MOTION_REFERENCE_VELOCITY,
                    (0.001, 0.5, 3.0),
                    (0.03, 0.5, 3.0),
                )
            })
            .collect();
        let columns = motion_columns(&cells);
        assert!(
            (columns.if_mismatch - 1.0).abs() < 1e-9,
            "mismatch {} without the clamp would be 30",
            columns.if_mismatch
        );
    }

    /// A1 is symmetric: a partial thirty times too still and one thirty times
    /// too spiky are the same failure.
    #[test]
    fn the_mismatch_column_punishes_stillness_and_spikiness_alike() {
        let still = motion_columns(&[cell(60, 1, 90, (0.1, 0.5, 3.0), (3.0, 0.5, 3.0))]);
        let spiky = motion_columns(&[cell(60, 1, 90, (3.0, 0.5, 3.0), (0.1, 0.5, 3.0))]);
        assert!((still.if_mismatch - spiky.if_mismatch).abs() < 1e-12);
        assert!(
            (still.if_mismatch - 30.0).abs() < 1e-9,
            "{}",
            still.if_mismatch
        );
        assert!(!still.passes());
    }

    /// B2 reads the *spread across velocity*, which is zero for anything the
    /// strike vector cannot move — and that is the whole claim of the column.
    #[test]
    fn a_velocity_invariant_engine_has_no_coherence_however_right_it_is() {
        let mut cells = Vec::new();
        for k in 1..=MOTION_PARTIALS {
            for (j, &velocity) in MOTION_VELOCITIES.iter().enumerate() {
                cells.push(cell(
                    60,
                    k,
                    velocity,
                    (1.0, 0.5, 6.0),
                    (1.0 + 0.8 * j as f64, 0.5, 6.0 + 2.0 * j as f64),
                ));
            }
        }
        let columns = motion_columns(&cells);
        assert!(
            columns.velocity_coherence < 1e-12,
            "{}",
            columns.velocity_coherence
        );
        assert!(columns.spread_cents.0 < 1e-12 && columns.spread_cents.1 > 1.0);
        assert!(!columns.passes());
    }

    /// A cell only one of the two signals measured is not a cell.
    #[test]
    fn a_partial_only_one_signal_resolved_is_not_scored() {
        let mut only_engine = cell(60, 1, 90, (1.0, 0.5, 3.0), (1.0, 0.5, 3.0));
        only_engine.reference = None;
        let columns = motion_columns(&[only_engine]);
        assert_eq!(columns.cells, 0);
        assert!(columns.if_mismatch.is_nan());
    }

    // -----------------------------------------------------------------------
    // The stereo columns
    // -----------------------------------------------------------------------

    /// Pink-ish broadband noise, so every band of the image has something in it.
    fn stereo_noise(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let mut lp = 0.0f64;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                let u = ((state >> 32) as f64 / (1u64 << 31) as f64) - 1.0;
                // One pole of integration lifts the bass so the 63-125 Hz band
                // clears the level floor; the raw white part keeps the treble.
                lp = 0.98 * lp + 0.02 * u;
                (0.5 * u + 6.0 * lp) as f32
            })
            .collect()
    }

    /// [`stereo_profile`] and [`stereo_image`] are one measurement at two
    /// resolutions, so they have to agree: the profile's points inside a
    /// scoreboard band, weighted by their own energy, are that band's `r0`.
    ///
    /// Not a tautology — the two take different transform lengths, different
    /// bin sets and different normalisations, and the profile skips the lag
    /// search entirely because it computes the zero-lag value in closed form
    /// (`M - S` *is* the cross-energy). If that shortcut were wrong this is
    /// where it would show.
    #[test]
    fn the_profile_and_the_six_bands_are_one_measurement_at_two_resolutions() {
        let x = stereo_noise(1 << 16, 23);
        // A delayed, partly independent right channel: something with real
        // structure across frequency rather than a constant.
        let other = stereo_noise(1 << 16, 24);
        let right: Vec<f32> = (0..x.len())
            .map(|i| 0.8 * x[i.saturating_sub(9)] + 0.4 * other[i])
            .collect();
        let image = stereo_image(&x, &right, SR).expect("two channels");
        let profile = stereo_profile(&x, &right, SR).expect("two channels");
        assert!(profile.len() > 40, "a sixth-octave profile is a curve");
        for (b, band) in image.bands.iter().enumerate() {
            if !band.readable() {
                continue;
            }
            let (mut num, mut den) = (0.0f64, 0.0f64);
            for point in &profile {
                if point.hz < band.lo_hz || point.hz >= band.hi_hz {
                    continue;
                }
                // The point's own energy, from its level share.
                let w = 10.0f64.powf(point.level_db / 10.0);
                num += w * point.r0;
                den += w;
            }
            if den <= 0.0 {
                continue;
            }
            let pooled = num / den;
            assert!(
                (pooled - band.r0).abs() < 0.06,
                "{}: the curve pools to {pooled:+.4} where the band reads {:+.4}",
                band.name,
                band.r0
            );
            assert_eq!(b, StereoImage::band_for(band.lo_hz + 1.0));
        }
    }

    /// The profile is what makes an *anti-phase* band visible as a shape, and
    /// the shape is the model: a side signal larger than the mid is a negative
    /// `r0`, and the mid/side ratio and the correlation say the same thing.
    ///
    /// Built here rather than measured: mid and side are made independent, and
    /// the side is band-limited and lifted. Below the band the pair is `+1`,
    /// inside it is negative, above it is `+1` again — which is exactly the
    /// three-regime curve `soundboard::ModalLobe` produces and the recording
    /// has (`DECISIONS.md` 369).
    #[test]
    fn a_side_larger_than_the_mid_reads_as_a_negative_lobe_in_the_profile() {
        let mid = stereo_noise(1 << 16, 31);
        let side_source = stereo_noise(1 << 16, 32);
        // A band-limited side — three one-poles on the lower edge and two on
        // the upper, so the skirts do not put the lobe's own energy into the
        // probes above and below it — at seven times the level, which is what
        // it takes for the side to beat the mid inside a band that steep.
        let (lo, hi) = (200.0f64, 500.0f64);
        let coeff = |hz: f64| 1.0 - (-std::f64::consts::TAU * hz / SR).exp();
        let (a, b) = (coeff(lo), coeff(hi));
        let (mut low, mut high) = ([0.0f64; 3], [0.0f64; 2]);
        let side: Vec<f32> = side_source
            .iter()
            .map(|&x| {
                let mut y = f64::from(x);
                for state in &mut low {
                    *state += a * (y - *state);
                    y -= *state;
                }
                for state in &mut high {
                    *state += b * (y - *state);
                    y = *state;
                }
                (7.0 * y) as f32
            })
            .collect();
        let left: Vec<f32> = mid.iter().zip(&side).map(|(&m, &s)| m + s).collect();
        let right: Vec<f32> = mid.iter().zip(&side).map(|(&m, &s)| m - s).collect();
        let profile = stereo_profile(&left, &right, SR).expect("two channels");
        let at = |hz: f64| -> StereoProfilePoint {
            *profile
                .iter()
                .min_by(|p, q| {
                    (p.hz / hz).ln().abs().total_cmp(&(q.hz / hz).ln().abs())
                })
                .expect("a point")
        };
        let (below, inside, above) = (at(63.0), at(320.0), at(4_000.0));
        assert!(below.r0 > 0.8, "under the lobe: {:+.3}", below.r0);
        assert!(inside.r0 < -0.3, "inside the lobe: {:+.3}", inside.r0);
        assert!(above.r0 > 0.8, "over the lobe: {:+.3}", above.r0);
        // `r0` and the mid/side ratio are the same statement when the two are
        // uncorrelated, which is what makes a mid/side filter a coherence.
        for point in [below, inside, above] {
            let from_ratio = {
                let ratio = 10.0f64.powf(point.mid_side_db / 10.0);
                (ratio - 1.0) / (ratio + 1.0)
            };
            assert!(
                (from_ratio - point.r0).abs() < 0.05,
                "{:.0} Hz: r0 {:+.3} against {:+.3} from the mid/side ratio",
                point.hz,
                point.r0,
                from_ratio
            );
        }
    }

    /// The engine's own answer before this milestone: one mono voice scaled
    /// into two channels. Every band correlates at +1 at lag zero and the side
    /// signal is exactly nothing — which is what [`STEREO_MS_CLAMP_DB`] is for.
    #[test]
    fn a_pan_pot_reads_plus_one_in_every_band_at_lag_zero() {
        let x = stereo_noise(1 << 15, 11);
        let right: Vec<f32> = x.iter().map(|&v| 0.6 * v).collect();
        let image = stereo_image(&x, &right, SR).expect("two channels");
        assert!(image.broadband.r0 > 0.999, "{:.4}", image.broadband.r0);
        for band in &image.bands {
            if !band.readable() {
                continue;
            }
            assert!(band.r0 > 0.99, "{} read {:.4}", band.name, band.r0);
            assert!(
                band.lag_ms.abs() < 1e-6,
                "{} peaked at {:.3} ms",
                band.name,
                band.lag_ms
            );
        }
    }

    /// Why the mid/side ratio is printed beside `r0` and is not the same number
    /// twice: a pan-pot is +1 correlated whatever it does to the levels, and
    /// the side energy it produces is a pure statement about the pan position.
    /// Two channels that are equal are all mid and the ratio clamps; the same
    /// pair panned to 0.6 is still +1 correlated and reads 12 dB.
    #[test]
    fn the_mid_side_ratio_sees_a_level_imbalance_that_the_correlation_cannot() {
        let x = stereo_noise(1 << 15, 11);
        let same = stereo_image(&x, &x, SR).expect("two channels");
        assert!(same.broadband.r0 > 0.9999);
        assert!(
            (same.broadband.mid_side_db - STEREO_MS_CLAMP_DB).abs() < 1e-9,
            "equal channels have no side signal at all and must clamp, read {:.1} dB",
            same.broadband.mid_side_db
        );
        let panned: Vec<f32> = x.iter().map(|&v| 0.6 * v).collect();
        let off = stereo_image(&x, &panned, SR).expect("two channels");
        assert!(
            off.broadband.r0 > 0.9999,
            "still a pan-pot: {:.4}",
            off.broadband.r0
        );
        // mid (1+0.6)/2, side (1-0.6)/2, so 20*log10(0.8/0.2).
        assert!(
            (off.broadband.mid_side_db - 12.04).abs() < 0.1,
            "read {:.2} dB",
            off.broadband.mid_side_db
        );
    }

    /// The soundboard's two opposite-sign taps, in the limit: an inverted
    /// channel is −1 at lag zero and all side. A metric that reported |r| would
    /// call this identical to a pan-pot, which is the whole reason `r0` is
    /// signed.
    #[test]
    fn an_inverted_channel_reads_minus_one_and_is_all_side() {
        let x = stereo_noise(1 << 15, 12);
        let right: Vec<f32> = x.iter().map(|&v| -v).collect();
        let image = stereo_image(&x, &right, SR).expect("two channels");
        assert!(image.broadband.r0 < -0.999, "{:.4}", image.broadband.r0);
        assert!(
            image.broadband.peak_r < -0.999,
            "{:.4}",
            image.broadband.peak_r
        );
        assert!(
            image.broadband.mid_side_db < -50.0,
            "{:.1} dB",
            image.broadband.mid_side_db
        );
    }

    /// A spaced pair: the same wavefront reaching two capsules a delay apart.
    /// Lag zero says almost nothing and the peak says everything, which is the
    /// shape item 314 measured on the recording.
    #[test]
    fn a_delayed_channel_shows_the_delay_as_its_peak_lag() {
        let x = stereo_noise(1 << 15, 13);
        let delay = 96; // 2 ms at 48 kHz
        let mut right = vec![0.0f32; x.len()];
        right[delay..].copy_from_slice(&x[..x.len() - delay]);
        let image = stereo_image(&x, &right, SR).expect("two channels");
        assert!(
            image.broadband.r0.abs() < 0.3,
            "a 2 ms shift must decorrelate at lag zero, read {:.3}",
            image.broadband.r0
        );
        assert!(
            (image.broadband.lag_ms + 2.0).abs() < 0.05,
            "the peak must sit at the delay, read {:+.3} ms",
            image.broadband.lag_ms
        );
        assert!(
            image.broadband.peak_r > 0.9,
            "and be a strong peak, read {:.3}",
            image.broadband.peak_r
        );
    }

    /// The spaced-pair *shape*, at the geometry item 314 is about: an AKG pair
    /// about 12 cm apart is 0.35 ms of air, which is a fifteenth of a
    /// wavelength at 100 Hz and three wavelengths at 10 kHz. So the bass stays
    /// correlated and the treble does not — the recording's +0.945 below 125 Hz
    /// falling to nothing above, out of a delay alone and no room at all.
    #[test]
    fn a_microphone_spacing_correlates_the_bass_and_not_the_treble() {
        let x = stereo_noise(1 << 16, 17);
        let delay = 17; // 0.354 ms at 48 kHz, i.e. 12 cm of air
        let mut right = vec![0.0f32; x.len()];
        right[delay..].copy_from_slice(&x[..x.len() - delay]);
        let image = stereo_image(&x, &right, SR).expect("two channels");
        let bass = image.bands[0];
        let treble = image.bands[5];
        assert!(
            bass.r0 > 0.9,
            "63-125 Hz should still be one wavefront, read {:.3}",
            bass.r0
        );
        assert!(
            treble.r0.abs() < 0.5,
            "6-12 kHz is many wavelengths out and should not be, read {:.3}",
            treble.r0
        );
        assert!(
            image.bands[0].r0 > image.bands[3].r0,
            "and the fall must be monotone through the bands: {:.3} then {:.3}",
            image.bands[0].r0,
            image.bands[3].r0
        );
    }

    /// Independent channels: zero at lag zero, zero at the peak, and 0 dB of
    /// mid over side.
    #[test]
    fn independent_channels_read_zero_and_split_their_energy_evenly() {
        let left = stereo_noise(1 << 15, 14);
        let right = stereo_noise(1 << 15, 15);
        let image = stereo_image(&left, &right, SR).expect("two channels");
        assert!(image.broadband.r0.abs() < 0.05, "{:.4}", image.broadband.r0);
        assert!(
            image.broadband.mid_side_db.abs() < 1.0,
            "{:.2} dB",
            image.broadband.mid_side_db
        );
    }

    /// A band with nothing in it is not read at all, rather than read as the
    /// ratio of two noise floors.
    #[test]
    fn a_band_with_nothing_in_it_is_not_readable() {
        let n = 1 << 15;
        // Windowed, because a rectangular cut of a sine leaks 50 dB of skirt
        // into every band and the thing being tested is a band that is empty.
        let tone: Vec<f32> = sine(1_000.0, 0.5, SR, n)
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let w = 0.5 - 0.5 * (2.0 * PI * i as f64 / n as f64).cos();
                (f64::from(v) * w) as f32
            })
            .collect();
        let image = stereo_image(&tone, &tone, SR).expect("two channels");
        let voiced = StereoImage::band_for(1_000.0);
        assert_eq!(voiced, 3, "1 kHz is the 500 Hz-2 kHz band");
        assert!(image.bands[voiced].readable());
        assert!(
            !image.bands[0].readable(),
            "63-125 Hz holds nothing and read {:.1} dB",
            image.bands[0].level_db
        );
    }

    /// The compass asks for "the band this key's fundamental is in", for all 88
    /// keys, and A0 is below the lowest band.
    #[test]
    fn the_fundamental_band_is_clamped_at_both_ends() {
        assert_eq!(
            StereoImage::band_for(27.5),
            0,
            "A0 clamps up into 63-125 Hz"
        );
        assert_eq!(StereoImage::band_for(100.0), 0);
        assert_eq!(StereoImage::band_for(261.6), 2);
        assert_eq!(StereoImage::band_for(4_186.0), 4);
        assert_eq!(
            StereoImage::band_for(20_000.0),
            5,
            "and clamps down at the top"
        );
    }

    /// Three pairs whose lag-zero correlation is a chosen number: a common part
    /// and an independent part, mixed to taste. `item` seeds both draws, so
    /// distinct items really are distinct material and the column's scatter is
    /// a scatter rather than zero.
    fn stereo_item(item: u64, engine: f64, reference: f64, alternate: f64) -> StereoItem {
        let n = 1 << 15;
        let build = |r: f64, seed: u64| {
            let common = stereo_noise(n, 100 + item);
            let own = stereo_noise(n, seed + 7 * item);
            let mix = ((1.0 - r * r).max(0.0)).sqrt();
            let right: Vec<f32> = common
                .iter()
                .zip(&own)
                .map(|(&c, &o)| (r as f32) * c + (mix as f32) * o)
                .collect();
            stereo_image(&common, &right, SR).expect("two channels")
        };
        StereoItem {
            label: format!("item{item}"),
            engine: build(engine, 31),
            reference: build(reference, 41),
            alternate: build(alternate, 51),
        }
    }

    /// The bar is made of the reference and of nothing else. Moving the engine
    /// moves the score and must not move the bar it is scored against — the
    /// property that separates a gate from a description.
    #[test]
    fn the_stereo_bar_is_built_out_of_the_reference_alone() {
        let good: Vec<StereoItem> = (0..6).map(|i| stereo_item(i, 0.90, 0.90, 0.80)).collect();
        let bad: Vec<StereoItem> = (0..6).map(|i| stereo_item(i, -0.60, 0.90, 0.80)).collect();
        let a = stereo_columns(&good);
        let b = stereo_columns(&bad);
        for (x, y) in a.iter().zip(&b) {
            if x.items == 0 || y.items == 0 {
                continue;
            }
            assert!(
                (x.bar - y.bar).abs() < 1e-12,
                "{}: the bar moved {:.4} -> {:.4} when only the engine changed",
                x.name,
                x.bar,
                y.bar
            );
            assert!((x.floor - y.floor).abs() < 1e-12);
            assert!(
                x.pass,
                "{} should pass at r 0.90 against 0.90: {:?}",
                x.name, x
            );
            assert!(!y.pass, "{} must fail at r −0.60 against +0.90", y.name);
            assert!(
                y.error > 1.0,
                "{}: an inverted channel is 1.5 of correlation away, read {:.3}",
                y.name,
                y.error
            );
        }
        assert!(a.iter().any(|c| c.items > 0), "some band must be readable");
    }

    /// The bar is `max(floor, scatter) · ALLOWANCE`, and the arithmetic of that
    /// is worth pinning because it is what every red in the gate is measured
    /// against.
    #[test]
    fn the_stereo_bar_takes_the_larger_of_the_two_disagreements() {
        let items: Vec<StereoItem> = (0..3).map(|i| stereo_item(i, 0.90, 0.90, 0.80)).collect();
        for c in stereo_columns(&items) {
            if c.items == 0 {
                continue;
            }
            assert!(
                (c.bar - c.floor.max(c.uncertainty) * STEREO_ALLOWANCE).abs() < 1e-12,
                "{}: {:.4} vs {:.4}",
                c.name,
                c.bar,
                c.floor.max(c.uncertainty) * STEREO_ALLOWANCE
            );
            assert!(c.bar >= c.floor, "{}: a bar under its own floor", c.name);
            assert!(
                (c.uncertainty - c.scatter / (c.items as f64).sqrt()).abs() < 1e-12,
                "{}: the uncertainty is the scatter over root n",
                c.name
            );
        }
    }
}
