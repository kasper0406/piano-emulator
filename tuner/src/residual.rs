//! What the model could not fit: the measurements `TUNING.md`'s Phase E is
//! written from.
//!
//! Every estimator in [`estimate`](crate::estimate) answers "what parameters
//! make the engine's model fit this recording best". None of them answers "and
//! how much of the recording is left over" — which is the question that decides
//! what to model next. This module measures the leftovers, in four places where
//! the engine's model makes a falsifiable claim about a recording:
//!
//! * **Partial frequencies are constant and follow `k f0 sqrt(1 + B k^2)`.** A
//!   modal bank's poles do not move, so a partial that drifts in frequency as
//!   it decays is something the engine cannot produce at any parameter setting
//!   ([`track_glide`]), and so is one that sits off the two-parameter law
//!   ([`partial_residuals`]).
//! * **A partial's envelope is two exponentials times two beats.** What is left
//!   over ([`PartialResidual::envelope_residual_db`]) is energy arriving at that
//!   frequency from somewhere the model does not have.
//! * **The excitation spectrum is a smooth envelope times `sin(k pi x)`.** The
//!   engine's per-mode input gains are exactly that, so scatter of the measured
//!   time-zero amplitudes around it ([`excitation_scatter`]) is unreachable.
//! * **Everything radiating is a transverse partial of a struck string.** A
//!   spectrum census ([`classify_peaks`], [`band_split`]) separates the peaks
//!   that are transverse partials from those at sums and differences of them —
//!   the phantom partials of the longitudinal nonlinearity — from those at
//!   *other keys'* pitches, and measures the broadband energy between all of
//!   them.
//!
//! Nothing here decides whether a leftover is audible; it reports levels in dB
//! relative to the note that produced them, and the report ranks them.
//!
//! The same measurements run on the engine's own renders, where the model is
//! true by construction, are the control: a residual the estimator returns on
//! synthetic material is the estimator's, not the piano's.

use crate::error::{Error, Result};
use crate::estimate::decay::DecayFit;
use crate::estimate::strike::StrikeFit;
use crate::pipeline::NoteAnalysis;
use crate::stft::{Peak, Stft, StftConfig};
use crate::trajectory::{cents, PartialTrack};

/// Settings for the residual measurements.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResidualConfig {
    /// How far below the loudest partial a track may be and still be measured.
    /// The same guard every estimator uses: below it the tracker is following
    /// the noise floor.
    pub level_db: f64,
    /// Amplitude drop over which [`track_glide`] measures a partial's frequency
    /// change, in dB below its own peak.
    pub glide_drop_db: f64,
    /// Half-width of the amplitude window each end of the glide is averaged
    /// over, in dB.
    pub glide_window_db: f64,
    /// How far a measured envelope point must stand above the track's own tail
    /// before it is compared with the model, in dB. Below that the recording's
    /// floor is what is being measured (`DECISIONS.md` item 89).
    pub floor_margin_db: f64,
    /// How close a peak must be to a predicted transverse partial to be called
    /// one, in cents.
    pub transverse_cents: f64,
    /// How close a peak must be to `f_i +- f_j` to be called a phantom, in
    /// cents. Tighter than [`Self::transverse_cents`] because a combination is
    /// a prediction with no free parameter in it, and because the transverse
    /// partial it must be told apart from is only tens of cents away.
    pub combination_cents: f64,
    /// How far a peak must stand from every transverse partial before it may be
    /// called a phantom, in cents.
    ///
    /// `f_i + f_j` and transverse partial `i + j` are the same frequency but
    /// for the inharmonicity, which separates them by
    /// `600 B ((i+j)^2 - (i^3 + j^3)/(i+j)) / ln 2` cents — 12 for `f_3 + f_4`
    /// of a `B = 4e-4` string, 30 for `f_5 + f_6` of the same one. Below this
    /// margin the two hypotheses are not distinguishable by a peak's frequency,
    /// so the transverse partial keeps the peak. Every phantom count is
    /// therefore a lower bound.
    pub min_separation_cents: f64,
    /// How close a peak must be to another key's pitch to be called that
    /// string ringing sympathetically, in cents. Tight, and still weak
    /// evidence: keys are a hundred cents apart, so a window of `2 c` catches
    /// `c/50` of all frequencies by chance. The control run says how much.
    pub neighbour_cents: f64,
    /// Highest transverse partial index combinations are formed from.
    ///
    /// Every pair is a candidate, so the set of predicted combination
    /// frequencies grows quadratically while their spacing shrinks: past about
    /// the sixth partial the predictions are dense enough that a peak of the
    /// noise floor lands on one, and a census taken there measures the density
    /// rather than the instrument. Six pairs give the twenty-one combinations
    /// that carry the audible phantoms.
    pub max_combination_k: u32,
}

impl Default for ResidualConfig {
    fn default() -> Self {
        Self {
            level_db: 60.0,
            glide_drop_db: 20.0,
            glide_window_db: 4.0,
            floor_margin_db: 8.0,
            transverse_cents: 35.0,
            combination_cents: 12.0,
            min_separation_cents: 25.0,
            neighbour_cents: 12.0,
            max_combination_k: 6,
        }
    }
}

/// What one partial of one recording did that the fitted model does not.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PartialResidual {
    pub k: u32,
    /// Amplitude-weighted measured frequency, Hz.
    pub frequency_hz: f64,
    /// Peak level of this partial relative to the note's loudest, dB.
    pub level_db: f64,
    /// Measured frequency minus the fitted `k f0 sqrt(1 + B k^2)`, in cents.
    pub model_cents: f64,
    /// How far the partial's own frequency fell while it decayed
    /// [`ResidualConfig::glide_drop_db`], in cents. Positive is falling — the
    /// direction a string whose tension modulation is dying away goes.
    pub glide_cents: Option<f64>,
    /// RMS of the envelope fit's residual, dB.
    pub envelope_residual_db: f64,
    /// Systematic part of that residual: measured minus modelled level at the
    /// end of the fitted span, in dB, from a straight line through the
    /// residual. Positive means the recording outlasts the model.
    pub envelope_trend_db: Option<f64>,
}

/// Per-partial residuals of one analysed recording.
///
/// The frequency is the track's amplitude-weighted mean, which is what the
/// inharmonicity fit reads, so `model_cents` is the same residual the fit
/// minimised — reported per partial rather than as one RMS.
pub fn partial_residuals(analysis: &NoteAnalysis, config: &ResidualConfig) -> Vec<PartialResidual> {
    let trajectories = &analysis.trajectories;
    let loudest = trajectories
        .tracks
        .iter()
        .filter_map(|track| track.peak())
        .map(|peak| peak.amplitude)
        .fold(0.0f64, f64::max);
    if loudest <= 0.0 {
        return Vec::new();
    }
    let floor = loudest * 10f64.powf(-config.level_db / 20.0);
    let model = analysis.inharmonic.model;
    let start_s = trajectories.onset_s + 0.5 * trajectories.window_s;

    trajectories
        .tracks
        .iter()
        .filter(|track| track.peak().is_some_and(|peak| peak.amplitude >= floor))
        .filter_map(|track| {
            let peak = track.peak()?;
            let frequency_hz = track.weighted_frequency()?;
            let fit = analysis.decays.fit(track.k);
            let deviation = fit.and_then(|fit| {
                envelope_deviation(track, fit, trajectories.onset_s, start_s, config)
            });
            Some(PartialResidual {
                k: track.k,
                frequency_hz,
                level_db: 20.0 * (peak.amplitude / loudest).log10(),
                model_cents: model.cents_from_partial(track.k, frequency_hz),
                glide_cents: track_glide(track, config),
                envelope_residual_db: fit.map_or(f64::NAN, |fit| fit.residual_db),
                envelope_trend_db: deviation,
            })
        })
        .collect()
}

/// How far a partial's frequency moves while it decays.
///
/// A struck string's tension rises with the square of its amplitude, so its
/// partials start sharp and fall as the note dies — an effect the engine cannot
/// produce, its poles being fixed. What is measured here is the size of that
/// slide: the frequency is regressed against the partial's *own level*, in
/// bins of [`ResidualConfig::glide_window_db`] over the top
/// [`ResidualConfig::glide_drop_db`] of the decay, and the reported number is
/// the interval between the ends of the fitted line, in cents. Positive is
/// falling.
///
/// Against level rather than against time, because the frequency of a
/// nonlinear string follows its amplitude and not the clock; by bin medians
/// rather than by points, because a beating envelope crosses the same level
/// several times and one mis-tracked frame would otherwise decide the answer.
///
/// A frame measures the average frequency over its whole window, so a window
/// longer than the slide reports part of it. The number is a lower bound on the
/// real excursion, and the bass — where the survey's windows are two thirds of
/// a second — understates it most.
pub fn track_glide(track: &PartialTrack, config: &ResidualConfig) -> Option<f64> {
    let peak = track.peak()?;
    if peak.amplitude <= 0.0 || config.glide_window_db <= 0.0 {
        return None;
    }
    let bins = (config.glide_drop_db / config.glide_window_db).ceil().max(2.0) as usize;
    let mut buckets: Vec<Vec<f64>> = vec![Vec::new(); bins];
    for point in &track.points {
        if point.amplitude <= 0.0 || point.frequency_hz <= 0.0 {
            continue;
        }
        let down = -20.0 * (point.amplitude / peak.amplitude).log10();
        if !(0.0..config.glide_drop_db).contains(&down) {
            continue;
        }
        buckets[(down / config.glide_window_db) as usize].push(point.frequency_hz);
    }
    let points: Vec<(f64, f64)> = buckets
        .iter()
        .enumerate()
        .filter(|(_, bucket)| bucket.len() >= 2)
        .map(|(bin, bucket)| {
            let mut sorted = bucket.clone();
            sorted.sort_by(f64::total_cmp);
            (
                (bin as f64 + 0.5) * config.glide_window_db,
                sorted[sorted.len() / 2],
            )
        })
        .collect();
    if points.len() < 3 {
        return None;
    }
    let n = points.len() as f64;
    let mean_x = points.iter().map(|&(x, _)| x).sum::<f64>() / n;
    let mean_y = points.iter().map(|&(_, y)| y).sum::<f64>() / n;
    let (num, den) = points.iter().fold((0.0, 0.0), |(num, den), &(x, y)| {
        (num + (x - mean_x) * (y - mean_y), den + (x - mean_x).powi(2))
    });
    if den <= 0.0 {
        return None;
    }
    let slope = num / den;
    let at = |level: f64| mean_y + slope * (level - mean_x);
    Some(cents(at(config.glide_drop_db), at(0.0)))
}

/// The systematic part of an envelope fit's residual: measured minus modelled
/// level at the end of the fitted span, in dB, from a least-squares line
/// through the per-frame residual.
///
/// A straight line and not the RMS, because the two say different things. The
/// RMS counts a beat the model missed and a tail it got wrong equally; the line
/// separates them, and it is the tail that decides how long the note is heard
/// to ring.
///
/// Points within [`ResidualConfig::floor_margin_db`] of the track's own tail
/// level are dropped: that is the recording's floor and the fit did not see
/// them either.
fn envelope_deviation(
    track: &PartialTrack,
    fit: &DecayFit,
    onset_s: f64,
    start_s: f64,
    config: &ResidualConfig,
) -> Option<f64> {
    let floor = tail_level(track) * 10f64.powf(config.floor_margin_db / 20.0);
    let points: Vec<(f64, f64)> = track
        .points
        .iter()
        .filter(|p| p.time_s >= start_s && p.amplitude > floor)
        .filter(|p| p.time_s - onset_s <= fit.span_s)
        .map(|p| {
            let t = p.time_s - onset_s;
            let modelled = fit.modulated_amplitude_at(t);
            (t, 20.0 * (p.amplitude / modelled).log10())
        })
        .filter(|&(_, db)| db.is_finite())
        .collect();
    if points.len() < 4 {
        return None;
    }
    let n = points.len() as f64;
    let mean_t = points.iter().map(|&(t, _)| t).sum::<f64>() / n;
    let mean_y = points.iter().map(|&(_, y)| y).sum::<f64>() / n;
    let (num, den) = points.iter().fold((0.0, 0.0), |(num, den), &(t, y)| {
        (num + (t - mean_t) * (y - mean_y), den + (t - mean_t).powi(2))
    });
    if den <= 0.0 {
        return None;
    }
    let slope = num / den;
    let last = points.last().expect("checked non-empty").0;
    Some(mean_y + slope * (last - mean_t))
}

/// Median amplitude of the last eighth of a track: what the partial decayed
/// *to*, which on a real recording is the room and the rest of the instrument
/// rather than silence.
fn tail_level(track: &PartialTrack) -> f64 {
    let n = track.points.len();
    if n < 8 {
        return 0.0;
    }
    let mut tail: Vec<f64> = track.points[n - n / 8..].iter().map(|p| p.amplitude).collect();
    tail.sort_by(f64::total_cmp);
    tail[tail.len() / 2]
}

/// Scatter of the measured excitation spectrum around the model the engine can
/// produce: a smooth envelope times the strike comb.
///
/// The engine's input gain for mode `k` is `sin(k pi x)` times one smooth
/// per-note scale, so any structure in the measured time-zero amplitudes that
/// [`StrikeFit`] cannot reproduce is unreachable at every parameter setting —
/// it is the bridge's and the soundboard's own impedance, partial by partial.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExcitationScatter {
    /// RMS of the measured-minus-modelled partial levels, dB.
    pub rms_db: f64,
    /// Largest single deviation, dB.
    pub worst_db: f64,
    /// Partial it happened at.
    pub worst_k: u32,
    pub partials: usize,
}

pub fn excitation_scatter(spectrum: &[(u32, f64)], fit: &StrikeFit) -> Option<ExcitationScatter> {
    let mut sum = 0.0;
    let mut worst = 0.0f64;
    let mut worst_k = 0;
    let mut used = 0usize;
    for &(k, amplitude) in spectrum {
        let modelled = fit.amplitude(k);
        if !(amplitude > 0.0 && modelled > 0.0) {
            continue;
        }
        let error = 20.0 * (amplitude / modelled).log10();
        sum += error * error;
        if error.abs() > worst {
            worst = error.abs();
            worst_k = k;
        }
        used += 1;
    }
    (used > 0).then(|| ExcitationScatter {
        rms_db: (sum / used as f64).sqrt(),
        worst_db: worst,
        worst_k,
        partials: used,
    })
}

// ------------------------------------------------------------- spectrum census

/// What a spectral peak turned out to be.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PeakClass {
    /// Partial `k` of the struck string's transverse series.
    Transverse { k: u32 },
    /// A phantom partial: `f_i + f_j` if `sum`, `|f_i - f_j|` otherwise.
    ///
    /// These are the signature of the longitudinal nonlinearity. A transverse
    /// motion of amplitude `a` stretches the string by an amount going as
    /// `a^2`, which drives the longitudinal (and the bridge) at every sum and
    /// difference of the transverse frequencies. `f_i + f_j` is *not* a
    /// transverse partial: partial `i + j` sits sharp of it by the
    /// inharmonicity, which is what tells the two apart.
    Combination { i: u32, j: u32, sum: bool },
    /// The pitch of a different key: a string the strike set going
    /// sympathetically, or the recording's own crosstalk.
    Neighbour { key: u8 },
    /// None of the above.
    Unexplained,
}

/// One peak of a census.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClassifiedPeak {
    pub frequency_hz: f64,
    /// Level relative to the reference amplitude the census was given, dB.
    pub level_db: f64,
    pub class: PeakClass,
}

/// Classifies each peak of a spectrum against what the model says can be there.
///
/// `partials` are the note's *measured* transverse partials (index and
/// frequency), `neighbours` the fundamentals of the instrument's other keys.
/// `reference` is the amplitude levels are quoted against — partial 1's,
/// normally. `skirt_hz` is the half-width of the analysis window's main lobe:
/// an unexplained peak closer than that to a louder classified one is that
/// peak's own skirt and is dropped rather than reported.
pub fn classify_peaks(
    peaks: &[Peak],
    partials: &[(u32, f64)],
    neighbours: &[(u8, f64)],
    reference: f64,
    skirt_hz: f64,
    config: &ResidualConfig,
) -> Vec<ClassifiedPeak> {
    let mut classified: Vec<ClassifiedPeak> = peaks
        .iter()
        .map(|peak| ClassifiedPeak {
            frequency_hz: peak.frequency_hz,
            level_db: 20.0 * (peak.amplitude / reference).log10(),
            class: classify_one(peak.frequency_hz, partials, neighbours, config),
        })
        .collect();
    let shadowed: Vec<bool> = classified
        .iter()
        .map(|peak| {
            peak.class == PeakClass::Unexplained
                && classified.iter().any(|other| {
                    other.class != PeakClass::Unexplained
                        && other.level_db > peak.level_db
                        && (other.frequency_hz - peak.frequency_hz).abs() < skirt_hz
                })
        })
        .collect();
    let mut keep = shadowed.iter();
    classified.retain(|_| !*keep.next().expect("one flag per peak"));
    classified
}

fn classify_one(
    frequency_hz: f64,
    partials: &[(u32, f64)],
    neighbours: &[(u8, f64)],
    config: &ResidualConfig,
) -> PeakClass {
    let nearest = |candidates: &mut dyn Iterator<Item = (f64, PeakClass)>| {
        candidates
            .filter(|&(f, _)| f > 0.0)
            .map(|(f, class)| (cents(f, frequency_hz).abs(), class))
            .min_by(|a, b| a.0.total_cmp(&b.0))
    };

    let transverse = nearest(
        &mut partials.iter().map(|&(k, f)| (f, PeakClass::Transverse { k })),
    );
    let low: Vec<(u32, f64)> = partials
        .iter()
        .copied()
        .filter(|&(k, _)| k <= config.max_combination_k)
        .collect();
    let combination = nearest(&mut low.iter().enumerate().flat_map(|(a, &(i, fi))| {
        low[a..].iter().flat_map(move |&(j, fj)| {
            [
                (fi + fj, PeakClass::Combination { i, j, sum: true }),
                ((fi - fj).abs(), PeakClass::Combination { i, j, sum: false }),
            ]
        })
    }));

    // A phantom is only claimed where the transverse series is far enough away
    // for the two to be told apart; otherwise the transverse partial keeps the
    // peak, whichever is marginally nearer.
    let separated = transverse.map_or(true, |(error, _)| error > config.min_separation_cents);
    if let Some((error, class)) = combination {
        if separated && error <= config.combination_cents {
            return class;
        }
    }
    if let Some((error, class)) = transverse {
        if error <= config.transverse_cents {
            return class;
        }
    }
    nearest(&mut neighbours.iter().map(|&(key, f)| (f, PeakClass::Neighbour { key })))
        .filter(|&(error, _)| error <= config.neighbour_cents)
        .map_or(PeakClass::Unexplained, |(_, class)| class)
}

/// How a band of one spectrum divides between the partials and everything else.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BandSplit {
    /// Power in bins within `guard_hz` of a partial.
    pub partial_power: f64,
    /// Power in every other bin of the band.
    pub between_power: f64,
    /// Median *amplitude* of those other bins — the level of what lies between
    /// the partials, which the total power overstates whenever one loud
    /// unmodelled peak sits in there.
    pub between_median: f64,
    pub between_bins: usize,
}

impl BandSplit {
    /// Power between the partials relative to power in them, dB.
    pub fn between_db(&self) -> f64 {
        if self.partial_power <= 0.0 || self.between_power <= 0.0 {
            return f64::NEG_INFINITY;
        }
        10.0 * (self.between_power / self.partial_power).log10()
    }
}

/// Splits a band of a magnitude spectrum into the partials and the rest.
///
/// `partials` are frequencies in Hz and `guard_hz` the half-width claimed
/// around each: at least the main lobe of the analysis window, or the partials'
/// own skirts count as the noise between them.
pub fn band_split(
    magnitude: &[f32],
    sample_rate: f64,
    fft_size: usize,
    partials: &[f64],
    guard_hz: f64,
    band: (f64, f64),
) -> BandSplit {
    let bin_hz = sample_rate / fft_size as f64;
    let mut split = BandSplit {
        partial_power: 0.0,
        between_power: 0.0,
        between_median: 0.0,
        between_bins: 0,
    };
    let mut between: Vec<f64> = Vec::new();
    for (bin, &value) in magnitude.iter().enumerate() {
        let f = bin as f64 * bin_hz;
        if f < band.0 || f > band.1 {
            continue;
        }
        let power = f64::from(value) * f64::from(value);
        if partials.iter().any(|&p| (p - f).abs() <= guard_hz) {
            split.partial_power += power;
        } else {
            split.between_power += power;
            between.push(f64::from(value));
        }
    }
    between.sort_by(f64::total_cmp);
    split.between_bins = between.len();
    if !between.is_empty() {
        split.between_median = between[between.len() / 2];
    }
    split
}

/// Amplitude of each of `partials` in one magnitude spectrum: the largest bin
/// within `guard_hz`, or `None` where the band is empty.
///
/// The largest bin rather than an interpolated peak: this is used to compare
/// one partial between two channels, where the bias of taking a maximum is the
/// same on both sides and cancels.
pub fn partial_levels(
    magnitude: &[f32],
    sample_rate: f64,
    fft_size: usize,
    partials: &[f64],
    guard_hz: f64,
) -> Vec<Option<f64>> {
    let bin_hz = sample_rate / fft_size as f64;
    partials
        .iter()
        .map(|&f| {
            let lo = (((f - guard_hz) / bin_hz).ceil().max(0.0)) as usize;
            let hi = (((f + guard_hz) / bin_hz).floor().max(0.0) as usize).min(magnitude.len() - 1);
            (lo <= hi)
                .then(|| {
                    magnitude[lo..=hi]
                        .iter()
                        .fold(0.0f64, |m, &v| m.max(f64::from(v)))
                })
                .filter(|&a| a > 0.0)
        })
        .collect()
}

/// How one note's partials divide between two channels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StereoBalance {
    /// Median of the per-partial level differences, dB. This is the note's
    /// position in the image — the one number the engine's per-key pan can
    /// reproduce.
    pub median_db: f64,
    /// Spread of those differences from the tenth to the ninetieth percentile,
    /// dB. The engine pans a mono voice, so everything here is unreachable.
    pub spread_db: f64,
    pub partials: usize,
}

/// Per-partial `20 log10(left / right)`, summarised. `None` unless at least
/// three partials were measured in both channels.
pub fn stereo_balance(left: &[Option<f64>], right: &[Option<f64>]) -> Option<StereoBalance> {
    let mut deltas: Vec<f64> = left
        .iter()
        .zip(right)
        .filter_map(|(l, r)| Some(20.0 * ((*l)? / (*r)?).log10()))
        .filter(|d| d.is_finite())
        .collect();
    if deltas.len() < 3 {
        return None;
    }
    deltas.sort_by(f64::total_cmp);
    let at = |q: f64| deltas[((deltas.len() - 1) as f64 * q).round() as usize];
    Some(StereoBalance {
        median_db: at(0.5),
        spread_db: at(0.9) - at(0.1),
        partials: deltas.len(),
    })
}

/// One Hann-windowed, amplitude-calibrated spectrum of `signal`, starting at
/// sample `start`.
///
/// The transform conventions are [`stft`](crate::stft)'s: a sinusoid of
/// amplitude `A` reads `A`.
pub fn frame_spectrum(signal: &[f32], start: usize, window: usize, pad: usize) -> Result<Vec<f32>> {
    let end = start
        .checked_add(window)
        .ok_or_else(|| Error::Config("frame does not fit in the signal".into()))?;
    if end > signal.len() {
        return Err(Error::Config(format!(
            "a {window}-sample frame at {start} does not fit in {} samples",
            signal.len()
        )));
    }
    let stft = Stft::new(StftConfig::padded(window, window, pad)?)?;
    let mut out = Vec::new();
    stft.for_each_frame(&signal[start..end], 1.0, |_, magnitude| {
        out.clear();
        out.extend_from_slice(magnitude);
    });
    Ok(out)
}

// --------------------------------------------------------------- action noise

/// What a recording of a mechanism rather than a string is made of: the key-off
/// samples, the pedal-action samples, and the attack transient of a struck note
/// before the partials take over.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransientMetrics {
    pub peak: f64,
    /// RMS over the whole recording.
    pub rms: f64,
    /// Time from the peak to 40 dB below it, on a 5 ms RMS envelope. Infinite
    /// if the recording never gets there.
    pub decay_s: f64,
    /// Power-weighted mean frequency of the first 100 ms, Hz.
    pub centroid_hz: f64,
    pub duration_s: f64,
}

/// Level, decay and colour of a transient recording.
pub fn transient_metrics(signal: &[f32], sample_rate: f64) -> Option<TransientMetrics> {
    if signal.is_empty() || sample_rate <= 0.0 {
        return None;
    }
    let block = ((0.005 * sample_rate) as usize).max(1);
    let envelope: Vec<f64> = signal
        .chunks(block)
        .map(|chunk| {
            let sum: f64 = chunk.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
            (sum / chunk.len() as f64).sqrt()
        })
        .collect();
    let (peak_block, &peak) = envelope
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))?;
    if peak <= 0.0 {
        return None;
    }
    let target = peak * 1e-2;
    let decay_s = envelope[peak_block..]
        .iter()
        .position(|&a| a <= target)
        .map_or(f64::INFINITY, |i| i as f64 * block as f64 / sample_rate);

    let total: f64 = signal.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
    let window = (0.1 * sample_rate) as usize;
    let window = window.min(signal.len()).next_power_of_two().min(1 << 14);
    let centroid_hz = if signal.len() >= window && window >= 4 {
        let spectrum = frame_spectrum(signal, 0, window, 1).ok()?;
        let bin_hz = sample_rate / window as f64;
        let (num, den) = spectrum.iter().enumerate().fold((0.0, 0.0), |(n, d), (bin, &m)| {
            let power = f64::from(m) * f64::from(m);
            (n + power * bin as f64 * bin_hz, d + power)
        });
        if den > 0.0 {
            num / den
        } else {
            f64::NAN
        }
    } else {
        f64::NAN
    };

    Some(TransientMetrics {
        peak: signal.iter().fold(0.0f64, |m, &x| m.max(f64::from(x).abs())),
        rms: (total / signal.len() as f64).sqrt(),
        decay_s,
        centroid_hz,
        duration_s: signal.len() as f64 / sample_rate,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trajectory::{InharmonicModel, TrackPoint};

    fn decaying_track(k: u32, f_start: f64, f_end: f64, points: usize) -> PartialTrack {
        // Amplitude falls 40 dB over the track and the frequency slides with
        // it, which is the shape a nonlinear string's partial has.
        PartialTrack {
            k,
            points: (0..points)
                .map(|i| {
                    let u = i as f64 / (points - 1) as f64;
                    TrackPoint {
                        time_s: u,
                        frequency_hz: f_start + (f_end - f_start) * u,
                        amplitude: 10f64.powf(-2.0 * u),
                    }
                })
                .collect(),
        }
    }

    #[test]
    fn a_partial_that_slides_reports_the_interval_it_slid() {
        // 20 dB down is half way through this track, where the frequency has
        // moved half of its 20 Hz: 440 to 430 is 39.8 cents.
        let track = decaying_track(1, 440.0, 420.0, 101);
        let glide = track_glide(&track, &ResidualConfig::default()).unwrap();
        assert!((glide - 39.8).abs() < 0.5, "{glide} cents");
    }

    #[test]
    fn a_partial_that_holds_its_pitch_reports_no_glide() {
        let track = decaying_track(1, 440.0, 440.0, 101);
        assert!(track_glide(&track, &ResidualConfig::default()).unwrap().abs() < 1e-9);
        // Nothing 20 dB down: no measurement, rather than a wrong one.
        let short = PartialTrack {
            points: track.points[..5].to_vec(),
            ..track
        };
        assert_eq!(track_glide(&short, &ResidualConfig::default()), None);
    }

    #[test]
    fn a_phantom_is_not_mistaken_for_the_transverse_partial_beside_it() {
        let model = InharmonicModel::new(110.0, 4e-4);
        let partials: Vec<(u32, f64)> = (1..=12).map(|k| (k, model.partial(k))).collect();
        let config = ResidualConfig::default();
        // f_5 + f_6 sits 30 cents flat of the eleventh transverse partial: far
        // enough for the classifier to tell them apart. f_3 + f_4 is only 12
        // cents from the seventh and stays with it, which is the conservative
        // direction.
        let phantom = model.partial(5) + model.partial(6);
        let gap = cents(phantom, model.partial(11));
        assert!((gap - 30.0).abs() < 2.0, "{gap} cents");
        let peaks = [
            Peak { frequency_hz: phantom, amplitude: 0.01, bin: 0 },
            Peak { frequency_hz: model.partial(11), amplitude: 0.1, bin: 0 },
            Peak { frequency_hz: model.partial(3) + model.partial(4), amplitude: 0.01, bin: 0 },
            Peak { frequency_hz: 261.63, amplitude: 0.005, bin: 0 },
            Peak { frequency_hz: 1600.0, amplitude: 0.002, bin: 0 },
        ];
        let census = classify_peaks(&peaks, &partials, &[(60, 261.63)], 1.0, 1.0, &config);
        assert_eq!(census[0].class, PeakClass::Combination { i: 5, j: 6, sum: true });
        assert_eq!(census[1].class, PeakClass::Transverse { k: 11 });
        assert_eq!(census[2].class, PeakClass::Transverse { k: 7 });
        assert_eq!(census[3].class, PeakClass::Neighbour { key: 60 });
        assert_eq!(census[4].class, PeakClass::Unexplained);
        assert!((census[1].level_db - -20.0).abs() < 0.1);
    }

    #[test]
    fn a_skirt_of_a_loud_partial_is_not_reported_as_unexplained() {
        let partials = [(1u32, 440.0)];
        let peaks = [
            Peak { frequency_hz: 440.0, amplitude: 1.0, bin: 0 },
            Peak { frequency_hz: 452.0, amplitude: 0.01, bin: 0 },
        ];
        let config = ResidualConfig::default();
        // 12 Hz away and 40 dB down, with a 20 Hz main lobe: the window's own
        // skirt, not a peak of the instrument.
        let census = classify_peaks(&peaks, &partials, &[], 1.0, 20.0, &config);
        assert_eq!(census.len(), 1);
        // With a narrower window the same peak stands on its own.
        let census = classify_peaks(&peaks, &partials, &[], 1.0, 5.0, &config);
        assert_eq!(census.len(), 2);
        assert_eq!(census[1].class, PeakClass::Unexplained);
    }

    #[test]
    fn the_band_split_separates_a_partial_from_the_noise_under_it() {
        let sample_rate = 48_000.0;
        let n = 1 << 13;
        // One partial at 1 kHz and a deterministic broadband floor 40 dB down.
        let mut signal: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f64 / sample_rate;
                (0.5 * (std::f64::consts::TAU * 1000.0 * t).sin()) as f32
            })
            .collect();
        let mut state = 12345u32;
        for x in signal.iter_mut() {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *x += 0.005 * ((state >> 8) as f32 / 8_388_608.0 - 1.0);
        }
        let spectrum = frame_spectrum(&signal, 0, n, 1).unwrap();
        let split = band_split(&spectrum, sample_rate, n, &[1000.0], 30.0, (100.0, 10_000.0));
        assert!(split.between_bins > 1000);
        // The partial holds essentially all the power; what is between the
        // partials is the floor that was put there.
        assert!(split.between_db() < -25.0, "{}", split.between_db());

        // And the median between the partials follows that floor rather than
        // the partial: with the noise removed it is the window's own skirt,
        // orders of magnitude quieter.
        let clean: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f64 / sample_rate;
                (0.5 * (std::f64::consts::TAU * 1000.0 * t).sin()) as f32
            })
            .collect();
        let quiet = band_split(
            &frame_spectrum(&clean, 0, n, 1).unwrap(),
            sample_rate,
            n,
            &[1000.0],
            30.0,
            (100.0, 10_000.0),
        );
        assert!(
            split.between_median > 10.0 * quiet.between_median,
            "{} vs {}",
            split.between_median,
            quiet.between_median
        );
    }

    #[test]
    fn the_stereo_balance_reads_a_pan_and_the_spread_around_it() {
        let left: Vec<Option<f64>> = vec![Some(1.0), Some(1.0), Some(1.0), Some(1.0), None];
        let right: Vec<Option<f64>> = vec![
            Some(0.5),                       // +6 dB
            Some(1.0),                       // 0
            Some(2.0),                       // -6 dB
            Some(1.0),                       // 0
            Some(1.0),
        ];
        let balance = stereo_balance(&left, &right).unwrap();
        assert_eq!(balance.partials, 4);
        assert!(balance.median_db.abs() < 1e-9);
        assert!((balance.spread_db - 12.04).abs() < 0.01, "{balance:?}");
        assert_eq!(stereo_balance(&left[..2], &right[..2]), None);
    }

    #[test]
    fn partial_levels_read_the_peak_inside_the_guard_and_nothing_outside_it() {
        let magnitude: Vec<f32> = (0..16).map(|i| if i == 5 { 1.0 } else { 0.1 }).collect();
        // 32-point transform at 32 Hz: one bin per hertz.
        let levels = partial_levels(&magnitude, 32.0, 32, &[5.0, 12.0], 1.5);
        assert!((levels[0].unwrap() - 1.0).abs() < 1e-6);
        assert!((levels[1].unwrap() - 0.1).abs() < 1e-6);
    }

    #[test]
    fn a_transient_reports_its_decay_and_its_colour() {
        let sample_rate = 48_000.0;
        // A 2 kHz tone decaying 40 dB in 200 ms.
        let sigma = 2.0 * std::f64::consts::LN_10 / 0.2;
        let signal: Vec<f32> = (0..(sample_rate as usize) / 2)
            .map(|i| {
                let t = i as f64 / sample_rate;
                ((-sigma * t).exp() * (std::f64::consts::TAU * 2000.0 * t).sin()) as f32
            })
            .collect();
        let metrics = transient_metrics(&signal, sample_rate).unwrap();
        assert!((metrics.decay_s - 0.2).abs() < 0.02, "{metrics:?}");
        assert!((metrics.centroid_hz - 2000.0).abs() < 200.0, "{metrics:?}");
        assert!((metrics.duration_s - 0.5).abs() < 1e-9);
    }

    #[test]
    fn a_frame_that_does_not_fit_is_an_error_rather_than_a_short_frame() {
        let signal = vec![0.0f32; 100];
        assert!(frame_spectrum(&signal, 0, 128, 1).is_err());
        assert!(frame_spectrum(&signal, 40, 64, 1).is_err());
        assert_eq!(frame_spectrum(&signal, 36, 64, 1).unwrap().len(), 33);
    }

    #[test]
    fn the_excitation_scatter_is_zero_on_a_spectrum_the_comb_explains() {
        let config = crate::estimate::strike::StrikeConfig::default();
        let x = 0.12;
        let spectrum: Vec<(u32, f64)> = (1..=20)
            .map(|k| {
                let kf = f64::from(k);
                (k, (kf * std::f64::consts::PI * x).sin().abs() / kf)
            })
            .collect();
        let fit = crate::estimate::strike::fit_strike_position(&spectrum, &config).unwrap();
        let scatter = excitation_scatter(&spectrum, &fit).unwrap();
        assert_eq!(scatter.partials, 20);
        assert!(scatter.rms_db < 1.5, "{scatter:?}");

        // One partial pushed 8 dB up — a bridge resonance, in the shape the
        // engine's smooth per-mode gains cannot take — shows up in the scatter.
        let mut bumped = spectrum.clone();
        bumped[6].1 *= 2.5;
        let fit = crate::estimate::strike::fit_strike_position(&bumped, &config).unwrap();
        let scatter = excitation_scatter(&bumped, &fit).unwrap();
        assert!(scatter.rms_db > 1.5, "{scatter:?}");
        assert!(scatter.worst_db > 5.0, "{scatter:?}");
    }
}
