//! Stage 1 over a whole sample library: every sampled note, every velocity
//! layer, into one preset.
//!
//! `pipeline` analyses one recording. This module is what turns a library of
//! them into an instrument, and the three things it has to decide are:
//!
//! * **How to transform each note.** A window that resolves the partials of A0
//!   is eight times longer than one C6 wants, and a window longer than a note
//!   needs smooths away the beats the estimators are there to measure
//!   (`DECISIONS.md` item 82). The geometry is therefore derived from the
//!   note's own pitch — a fixed number of periods of the fundamental — rather
//!   than fixed for the run.
//! * **What to do with sixteen answers per note.** Most of what stage 1
//!   measures does not depend on how hard the key was struck: the tuning, the
//!   inharmonicity, the strike point and the unison detuning are properties of
//!   the string. Each layer measures them independently, so the note's value is
//!   the **median** over its layers — sixteen noisy measurements of one number,
//!   reduced by the estimator that does not care what the two quietest layers
//!   did. What genuinely varies with velocity — the excitation spectrum — is
//!   kept per layer and is exactly what the felt fit reads.
//! * **What not to write.** A recording of somebody else's piano has no
//!   newtons-to-amplitude calibration in it, so the felt stiffness is not
//!   identifiable at all (`estimate::hammer`) and the hammer mass is degenerate
//!   with it. Neither is the strike position, for a different reason: a
//!   microphone a few centimetres above the string picks each partial up
//!   through that partial's own mode shape at the microphone's position, which
//!   is a `sin(k pi x)` comb of exactly the form the strike estimator inverts,
//!   and the two combs are not separable from one recording. What the fit
//!   returns on close-miked material is therefore the microphone's comb, not
//!   the hammer's — right to divide out of an excitation spectrum, wrong to
//!   write into a preset. Those tables are left as the base preset has them,
//!   and the survey reports what it saw instead of writing it.
//!
//! Tracking is cached to disk keyed by the recording and the transform
//! geometry, because it is the only expensive step: the fits that follow run in
//! milliseconds and get iterated on, the STFT pass does not.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::error::{Error, Result};
use crate::estimate::decay::{DecayConfig, DecayCurve, DecayFit, PolarizationSplit, LN_1000};
use crate::estimate::hammer::{
    fit_hammer, fit_velocity_map, FeltParams, HammerConfig, HammerFit, LayerSpectrum,
    SpectrumWeighting, VelocityMap,
};
use crate::estimate::inharmonic::InharmonicConfig;
use crate::estimate::noise::{EventMetrics, MechanismMeasurements};
use crate::estimate::spread::{note_spread_over, NoteSpread, SigmaSpread, SpreadConfig};
use crate::estimate::strike::StrikeFit;
use crate::library::{MechanismKind, Sample, SampleLibrary};
use crate::numeric::weighted_least_squares;
use crate::pipeline::{analyze_trajectories, track_refined, NoteAnalysis, NoteConfig};
use crate::preset::{
    equal_temperament, key_index, vertical_decay_factor, NoteEstimate, Preset, PresetBuilder,
};
use crate::stft::StftConfig;
use crate::tracker::TrackerConfig;
use crate::residual::transient_metrics;
use crate::trajectory::{InharmonicModel, NoteId, NoteTrajectories};
use crate::{audio, SAMPLE_RATE};

/// How a survey reads a library.
#[derive(Clone, Debug, PartialEq)]
pub struct SurveyConfig {
    /// The per-note settings. Everything except the transform geometry, which
    /// [`SurveyConfig::note_config`] fills in from the note's pitch.
    pub note: NoteConfig,
    /// Where tracked trajectories are cached. `None` re-transforms every run.
    pub cache_dir: Option<PathBuf>,
    /// Ignore a cached file even if it matches, and overwrite it.
    pub refresh_cache: bool,
    /// Seconds discarded from the end of every recording.
    ///
    /// Sample libraries fade their tails to silence so that a note can be
    /// looped or truncated without a click; Salamander's fade is about a second
    /// long and takes the last 25 dB with it. Fitting a decay through it would
    /// read the fade as the instrument.
    pub trim_tail_s: f64,
    /// Longest stretch of a recording to analyse, from the start of the file.
    pub max_duration_s: f64,
    /// Periods of the fundamental the analysis window must span. Four is what
    /// it takes to separate neighbouring partials at all (a Hann main lobe is
    /// `4/T` wide and partials are `f0` apart); the rest is margin for the
    /// stiffness that pushes them off the grid and for the noise between them.
    pub window_periods: f64,
    /// Bounds on the window, in samples. The floor is a time resolution the
    /// treble's decays need; the ceiling stops the bottom octave from asking
    /// for a window longer than the beats it must not smooth.
    pub min_window: usize,
    pub max_window: usize,
    /// Hop as a fraction of the window: the envelope's time resolution is the
    /// window itself, so anything finer is oversampling and only costs frames.
    pub hop_divisor: usize,
    /// Zero-padding factor of the transform.
    pub pad: usize,
    /// Worker threads. Zero asks the machine.
    pub threads: usize,
}

impl Default for SurveyConfig {
    fn default() -> Self {
        Self {
            note: NoteConfig {
                // Two partials, not four. The top two octaves of a real piano
                // have almost nothing above the fundamental — C7's second
                // partial is 44 dB down and its third 55 — so a fit that
                // insists on four gives up on the treble entirely, and the
                // treble is where the tuning is stretched most and where the
                // instrument most needs measuring. Two partials determine
                // `(f0, B)` exactly, with nothing left over to check them
                // with; what checks them is the median over sixteen layers and
                // [`PLAUSIBLE_B`].
                inharmonic: InharmonicConfig {
                    min_partials: 2,
                    ..NoteConfig::default().inharmonic
                },
                // A wider search than the tracker's own 60 cents. A real piano
                // is stretch-tuned, and by the top octave that is not a
                // rounding error: Salamander's A7 stands 38 cents above equal
                // temperament and its C8 — which the library's own "Retuned"
                // variant tries to patch — very nearly a semitone. Seeded at
                // equal temperament, a 60-cent window misses both, and what it
                // finds instead is whatever noise was inside it.
                tracker: TrackerConfig {
                    tolerance_cents: 120.0,
                    ..NoteConfig::default().tracker
                },
                ..NoteConfig::default()
            },
            cache_dir: None,
            refresh_cache: false,
            trim_tail_s: 1.2,
            max_duration_s: 30.0,
            window_periods: 12.0,
            min_window: 1 << 12,
            max_window: 1 << 15,
            hop_divisor: 16,
            pad: 2,
            threads: 0,
        }
    }
}

impl SurveyConfig {
    /// The transform geometry for a note of pitch `f0_hz`: the next power of
    /// two spanning [`window_periods`](Self::window_periods) periods of it,
    /// clamped to the configured bounds.
    ///
    /// Powers of two are not required by anything here — they are what keeps
    /// the FFT fast and the set of distinct cache files small.
    pub fn geometry(&self, f0_hz: f64) -> Result<StftConfig> {
        if f0_hz.is_nan() || f0_hz <= 0.0 || self.max_window < self.min_window {
            return Err(Error::Config(format!(
                "no window for a note at {f0_hz} Hz within {}..={}",
                self.min_window, self.max_window
            )));
        }
        let periods = (self.window_periods * f64::from(SAMPLE_RATE) / f0_hz).ceil();
        let wanted = (periods.clamp(1.0, self.max_window as f64) as usize).next_power_of_two();
        let window = wanted.clamp(self.min_window, self.max_window);
        StftConfig::padded(window, (window / self.hop_divisor.max(1)).max(1), self.pad)
    }

    /// The per-note settings for a note of pitch `f0_hz`.
    pub fn note_config(&self, f0_hz: f64) -> Result<NoteConfig> {
        Ok(NoteConfig {
            tracker: TrackerConfig {
                stft: self.geometry(f0_hz)?,
                ..self.note.tracker
            },
            ..self.note
        })
    }

    fn worker_count(&self) -> usize {
        if self.threads > 0 {
            self.threads
        } else {
            std::thread::available_parallelism().map_or(1, |n| n.get())
        }
    }
}

/// Inharmonicity coefficients a piano string can actually have, from the
/// longest wound bass string to the shortest treble one. Anything outside is a
/// fit that failed, not a string — and a fit from a treble note's two audible
/// partials has nothing but this to catch it.
const PLAUSIBLE_B: std::ops::Range<f64> = 1e-5..3e-2;

/// How far from equal temperament a note may be measured and still be taken as
/// a tuning rather than as a broken recording, in cents.
///
/// A Railsback curve is worth about 35 cents flat at the bottom of a small
/// grand and 40 sharp at the top, so this is a wide band and not a constraint
/// on how a piano may be tuned. What it excludes is Salamander's C8, whose
/// fundamental sits 99 cents — a semitone — above its own key: the whole note
/// is unusable, not just its pitch, so nothing is taken from it.
const PLAUSIBLE_STRETCH_CENTS: f64 = 60.0;

/// One analysed recording.
#[derive(Clone, Debug)]
pub struct LayerAnalysis {
    pub layer: u8,
    /// Middle of the layer's MIDI velocity band — where the velocity map is
    /// anchored.
    pub midi_velocity: u8,
    pub analysis: NoteAnalysis,
}

/// A recording that did not survive the pipeline, and why.
#[derive(Clone, Debug)]
pub struct Failure {
    pub key: u8,
    pub layer: u8,
    pub path: PathBuf,
    pub reason: String,
}

/// Every layer of one key.
#[derive(Clone, Debug)]
pub struct NoteSurvey {
    pub key: u8,
    pub layers: Vec<LayerAnalysis>,
}

impl NoteSurvey {
    /// Median over the layers of whatever `get` reads out of one of them.
    ///
    /// The median rather than a mean: a layer whose fit went wrong is wrong by
    /// orders of magnitude, not by a few percent, and there are sixteen of them
    /// per note.
    pub fn median<F: Fn(&NoteAnalysis) -> Option<f64>>(&self, get: F) -> Option<f64> {
        median(self.layers.iter().filter_map(|l| get(&l.analysis)))
    }

    /// The `f0` the `(f0, B)` fits agreed on. [`NoteSurvey::tuning`] is what
    /// gets written — it re-derives `f0` from partial 1, which is measured far
    /// better than the pair is at the top of the compass.
    pub fn f0_hz(&self) -> Option<f64> {
        self.median(|a| Some(a.inharmonic.model.f0_hz))
    }

    pub fn inharmonicity_b(&self) -> Option<f64> {
        self.median(|a| Some(a.inharmonic.model.b).filter(|b| PLAUSIBLE_B.contains(b)))
    }

    pub fn strike_position(&self) -> Option<f64> {
        self.median(|a| a.strike.as_ref().map(|s| s.position))
    }

    /// The signed fourth-order inharmonicity, by majority of the layers.
    ///
    /// Each layer decides for itself whether its partial series curves away
    /// from the two-parameter law by more than its own noise
    /// (`estimate::inharmonic`'s two-band guard). A note where most of them say
    /// it does is a note with a `B4`, and its value is the median of theirs; a
    /// note where most say it does not is measured to *have* no `B4`, which is
    /// a zero and not a missing measurement — that is what keeps one lucky
    /// layer out of the preset.
    pub fn inharmonicity_b4(&self) -> Option<f64> {
        if self.layers.is_empty() {
            return None;
        }
        let quartic: Vec<f64> = self
            .layers
            .iter()
            .map(|l| l.analysis.inharmonic.model.b4)
            .filter(|b4| b4.is_finite() && *b4 != 0.0)
            .collect();
        if quartic.len() * 2 <= self.layers.len() {
            return Some(0.0);
        }
        Some(median(quartic.into_iter()).unwrap_or(0.0))
    }

    /// What this note's beating partials say about its strings' damping.
    ///
    /// `detune_cents` is the group's **full** width — the interval between the
    /// outer strings, which is the pair the survivor argument is about — and
    /// not the dominant beat the unison estimator reports;
    /// [`Survey::spreads`] is where the two are converted.
    pub fn spread(
        &self,
        strings: usize,
        detune_cents: f64,
        config: &SpreadConfig,
    ) -> NoteSpread {
        note_spread_over(
            self.key,
            strings,
            detune_cents,
            self.layers.iter().map(|l| &l.analysis.trajectories),
            config,
        )
    }

    /// The two-band diagnostic, medianed over the layers: how far the ratio of
    /// `B(high k)` to `B(low k)` stands from 1, and by how many of its own
    /// standard deviations. This is what decides whether a fourth-order term is
    /// fitted at all, so a note that came back without one is read here.
    pub fn band_ratio(&self) -> Option<(f64, f64)> {
        let ratio = self.median(|a| a.inharmonic.bands.map(|b| b.ratio()))?;
        let sigmas = self.median(|a| a.inharmonic.bands.map(|b| b.sigmas_from_one()))?;
        Some((ratio, sigmas))
    }

    /// The hammer's contact width, as the strike fits saw it. Reported, not
    /// written: it comes out of the same comb as the strike position, so it
    /// carries the same microphone confound — see the module header.
    pub fn contact_width(&self) -> Option<f64> {
        self.median(|a| a.strike.as_ref().and_then(|s| s.contact_width))
    }

    pub fn detune_cents(&self) -> Option<f64> {
        self.median(|a| a.unison.as_ref().map(|u| u.detune_cents))
    }

    /// The note's damping law, from the **prompt** decay of each partial.
    ///
    /// Not from its T60. A T60 is defined 60 dB down and a recording of a real
    /// piano contains 35 or 40, so the last twenty decibels of it are always an
    /// extrapolation along the fitted slow component — the least determined
    /// number in the fit, and on this material the one that decides the answer.
    /// It is what made neighbouring minor thirds come back with T60s a factor
    /// of five apart. The *fast* component is the prompt sound, it is measured
    /// over the part of the record that is above the floor, and it is what the
    /// ear calls the note's decay.
    ///
    /// Converting it into the table's convention is what `factor` is for: the
    /// engine builds both polarizations from one table entry and its global
    /// split, so the entry that gives a vertical bank the measured prompt rate
    /// is that rate divided by
    /// [`vertical_decay_factor`](crate::preset::vertical_decay_factor). The
    /// aftersound then comes from the split, measured across the whole
    /// instrument, rather than from each note's own worst-conditioned
    /// parameter.
    pub fn decay_curve(&self, factor: f64, config: &DecayConfig) -> Option<DecayCurve> {
        let curves: Vec<DecayCurve> = self
            .layers
            .iter()
            .filter_map(|l| prompt_decay_curve(&l.analysis.decays.partials, factor, config))
            .collect();
        // The two coefficients are medianed separately: they are strongly
        // anticorrelated inside one fit, and the pair from any single layer is
        // noisier than the pair of medians.
        Some(DecayCurve {
            sigma0: median(curves.iter().map(|c| c.sigma0))?,
            sigma1: median(curves.iter().map(|c| c.sigma1))?,
            residual: median(curves.iter().map(|c| c.residual))?,
        })
    }

    pub fn polarization(&self) -> Option<PolarizationSplit> {
        let split = |get: fn(&PolarizationSplit) -> f64| {
            self.median(|a| {
                Some(a.decays.polarization)
                    .filter(|p| p.partials > 0)
                    .map(|p| get(&p))
            })
        };
        Some(PolarizationSplit {
            gain_db: split(|p| p.gain_db)?,
            decay_ratio: split(|p| p.decay_ratio)?,
            partials: self
                .layers
                .iter()
                .map(|l| l.analysis.decays.polarization.partials)
                .max()
                .unwrap_or(0),
        })
    }

    /// The note's strike point, as a fit rather than a number.
    ///
    /// The hammer sits where it sits whatever the key was struck at, so all
    /// sixteen layers measure one quantity; but [`LayerSpectrum::from_decays`]
    /// needs a whole [`StrikeFit`] (it divides by the fitted comb), so what is
    /// returned is the layer's fit nearest the note's median rather than a
    /// synthetic one.
    pub fn strike_fit(&self) -> Option<&StrikeFit> {
        let median = self.strike_position()?;
        self.layers
            .iter()
            .filter_map(|l| l.analysis.strike.as_ref())
            .min_by(|a, b| {
                (a.position - median)
                    .abs()
                    .total_cmp(&(b.position - median).abs())
            })
    }

    /// Where partial 1 of this note actually sits, in Hz.
    ///
    /// The preset's `f0` is the *string's* fundamental, which is partial 1
    /// divided by the stiffness factor `sqrt(1 + B)` — so a note whose `B` came
    /// back wrong has an `f0` wrong by half of that error, which at the top of
    /// the compass is tens of cents. Partial 1 itself is a strong isolated peak
    /// and is measured well everywhere, so the tuning is taken from it and
    /// converted with whatever `B` the note ended up with.
    pub fn partial_hz(&self, k: u32) -> Option<f64> {
        self.median(|a| Some(a.inharmonic.model.partial(k)))
    }

    /// This note's `f0`, or `None` if what was measured is not a tuning of this
    /// key at all — see [`PLAUSIBLE_STRETCH_CENTS`]. A note that fails this
    /// test contributes nothing: whatever the tracker followed, it was not this
    /// string, so its damping and its beats are no more usable than its pitch.
    pub fn tuning(&self, fallback_b: f64) -> Option<f64> {
        let b = self.inharmonicity_b().unwrap_or(fallback_b);
        let f0 = self.partial_hz(1)? / (1.0 + b).sqrt();
        let stretch = 1200.0 * (f0 / equal_temperament(self.key)).log2();
        (stretch.abs() <= PLAUSIBLE_STRETCH_CENTS).then_some(f0)
    }

    /// What stage 1 measured about this note, ready for the preset builder.
    ///
    /// The felt is not here, and neither is the strike position: see the module
    /// header. `fallback_b` is the inharmonicity to convert partial 1 with when
    /// the note's own could not be measured — the base preset's, normally.
    pub fn estimate(&self, factor: f64, fallback_b: f64, config: &DecayConfig) -> NoteEstimate {
        let Some(f0) = self.tuning(fallback_b) else {
            return NoteEstimate::new(self.key);
        };
        let curve = self.decay_curve(factor, config);
        NoteEstimate {
            key: self.key,
            f0_hz: Some(f0),
            inharmonicity_b: self.inharmonicity_b(),
            inharmonicity_b4: self.inharmonicity_b4(),
            strike_position: None,
            // The width comes out of the same comb as the strike position, and
            // is left out of the file for the same reason: a close microphone's
            // own `sin(k pi x)` comb is not separable from the hammer's, so
            // what the fit returns on library material is right to divide out
            // of an excitation spectrum and wrong to write into a preset.
            contact_width: None,
            // Both are fitted by their own estimators against material a single
            // note's trajectories do not carry — the comb floor needs the
            // deepest partial of the whole layer set, the damper needs the
            // release recordings — and are written through
            // `PresetBuilder::note` by `examples/fit_partials`.
            comb_floor: None,
            damper_sigma: None,
            sigma0: curve.map(|c| c.sigma0),
            sigma1: curve.map(|c| c.sigma1),
            detune_cents: self.detune_cents(),
            hammer_mass: None,
            hammer_stiffness: None,
            hammer_exponent: None,
        }
    }
}

/// What the felt fit found for one note, and the layer speeds it implies.
#[derive(Clone, Debug)]
pub struct HammerReport {
    pub key: u8,
    pub fit: HammerFit,
    /// `(MIDI velocity, hammer speed)` per layer, in the library's order.
    pub layers: Vec<(u8, f64)>,
}

/// Stage 1's answer for a whole library.
#[derive(Clone, Debug)]
pub struct Survey {
    pub notes: Vec<NoteSurvey>,
    pub failures: Vec<Failure>,
}

impl Survey {
    /// Analyses every recording the library maps.
    ///
    /// A recording that fails is recorded in [`Survey::failures`] and the run
    /// continues: one unreadable file out of five hundred must not cost the
    /// other four hundred and ninety-nine, and a survey that silently dropped
    /// it would be worse than one that did.
    pub fn run(
        library: &SampleLibrary,
        config: &SurveyConfig,
        mut progress: impl FnMut(&Sample, &Result<NoteAnalysis>),
    ) -> Survey {
        let samples: Vec<&Sample> = library.samples().collect();
        let next = AtomicUsize::new(0);
        let mut done: Vec<Option<Result<NoteAnalysis>>> = (0..samples.len()).map(|_| None).collect();

        // A hand-rolled work queue rather than a dependency: the unit of work
        // is one recording, they are independent, and they take between a
        // fraction of a second and several seconds each, so an atomic index is
        // both the simplest and the best-balanced way to hand them out.
        let workers = config.worker_count().min(samples.len().max(1));
        std::thread::scope(|scope| {
            let (tx, rx) = std::sync::mpsc::channel();
            for _ in 0..workers {
                let (next, samples, tx) = (&next, &samples, tx.clone());
                scope.spawn(move || loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    let Some(sample) = samples.get(index) else {
                        return;
                    };
                    let result = analyze_sample(sample, config);
                    if tx.send((index, result)).is_err() {
                        return;
                    }
                });
            }
            drop(tx);
            for (index, result) in rx {
                progress(samples[index], &result);
                done[index] = Some(result);
            }
        });

        let mut notes: Vec<NoteSurvey> = Vec::new();
        let mut failures = Vec::new();
        for (sample, result) in samples.iter().zip(done) {
            let result = result.expect("every queued recording is answered");
            match result {
                Ok(analysis) => {
                    let note = match notes.iter_mut().position(|n| n.key == sample.key) {
                        Some(i) => &mut notes[i],
                        None => {
                            notes.push(NoteSurvey {
                                key: sample.key,
                                layers: Vec::new(),
                            });
                            notes.last_mut().expect("just pushed")
                        }
                    };
                    note.layers.push(LayerAnalysis {
                        layer: sample.layer,
                        midi_velocity: sample.midi_velocity(),
                        analysis,
                    });
                }
                Err(error) => failures.push(Failure {
                    key: sample.key,
                    layer: sample.layer,
                    path: sample.path.clone(),
                    reason: error.to_string(),
                }),
            }
        }
        Survey { notes, failures }
    }

    pub fn note(&self, key: u8) -> Option<&NoteSurvey> {
        self.notes.iter().find(|n| n.key == key)
    }

    /// The instrument's polarization split: the median over every note that
    /// produced one. It is a single global pair in the engine, so this is the
    /// one place a per-note measurement has to be collapsed to one number.
    pub fn polarization(&self) -> Option<PolarizationSplit> {
        let notes: Vec<PolarizationSplit> = self.notes.iter().filter_map(|n| n.polarization()).collect();
        Some(PolarizationSplit {
            gain_db: median(notes.iter().map(|p| p.gain_db))?,
            decay_ratio: median(notes.iter().map(|p| p.decay_ratio))?,
            partials: notes.iter().map(|p| p.partials).max().unwrap_or(0),
        })
    }

    /// Every note's decay spread, in key order.
    ///
    /// The group's full detune width comes from the note's own beat where it
    /// measured one — widened through `base`'s unison layout exactly as
    /// [`PresetBuilder`] widens it before writing the table — and from `base`'s
    /// table where it did not. A note whose width is unknown is not measured:
    /// the drift alone says nothing without the interval it is a fraction of.
    pub fn spreads(&self, base: &Preset, config: &SpreadConfig) -> Vec<NoteSpread> {
        let mut spreads: Vec<NoteSpread> = self
            .notes
            .iter()
            .filter_map(|note| {
                let index = key_index(note.key)?;
                let strings = usize::from(base.notes.unison[index]);
                let fraction = base.voicing.dominant_beat_fraction(strings);
                let detune = match note.detune_cents() {
                    Some(beat) if fraction > 0.0 => beat / fraction,
                    _ => f64::from(base.notes.detune_cents[index]),
                };
                Some(note.spread(strings, detune, config))
            })
            .collect();
        spreads.sort_by_key(|note| note.key);
        spreads
    }

    /// The instrument's per-string decay spread: [`Survey::spreads`] pooled by
    /// group size.
    pub fn sigma_spread(&self, base: &Preset, config: &SpreadConfig) -> SigmaSpread {
        SigmaSpread::pooled(&self.spreads(base, config), config)
    }

    /// Fits the felt and the layer speeds for one note.
    ///
    /// The contact the hammer meets — impedance, string count, the reflection
    /// delay — comes from `base`: those are not measurable from a recording and
    /// the fit needs them to mean anything. The newtons-to-amplitude gain is
    /// left unknown, which is what a recording of somebody else's piano is; see
    /// [`HammerConfig::gain`].
    pub fn hammer(&self, key: u8, base: &Preset, config: &HammerConfig) -> Result<HammerReport> {
        let note = self
            .note(key)
            .ok_or_else(|| Error::Estimate(format!("key {key} was not surveyed")))?;
        let index = key_index(key)
            .ok_or_else(|| Error::Preset(format!("key {key} is not on the keyboard")))?;
        let strike = note
            .strike_fit()
            .ok_or_else(|| Error::Estimate(format!("key {key} has no strike position")))?;
        let f0 = note
            .f0_hz()
            .ok_or_else(|| Error::Estimate(format!("key {key} has no pitch")))?;
        let weighting = SpectrumWeighting::default();
        let layers: Vec<LayerSpectrum> = note
            .layers
            .iter()
            .map(|l| LayerSpectrum::from_decays(l.layer, &l.analysis.decays, strike, &weighting))
            .collect();
        let start = FeltParams {
            mass: f64::from(base.notes.hammer_mass[index]),
            stiffness: f64::from(base.notes.hammer_stiffness[index]),
            exponent: f64::from(base.notes.hammer_exponent[index]),
        };
        let fit = fit_hammer(
            &layers,
            &start,
            &HammerConfig {
                contact: config.contact.for_note(
                    f0,
                    strike.position,
                    f64::from(base.notes.unison[index]),
                    f64::from(base.notes.impedance[index]),
                ),
                ..*config
            },
        )?;
        let pairs = note
            .layers
            .iter()
            .map(|l| l.midi_velocity)
            .zip(fit.velocities.iter().copied())
            .collect();
        Ok(HammerReport { key, fit, layers: pairs })
    }

    /// How much faster the engine's vertical bank runs than the `sigma` tables
    /// say, for the split this survey measured — the conversion
    /// [`NoteSurvey::decay_curve`] needs. Falls back to the base preset's own
    /// split when nothing was measured.
    pub fn vertical_factor(&self, base: &Preset) -> f64 {
        match self.polarization() {
            Some(split) => vertical_decay_factor(split.gain_db, split.decay_ratio),
            None => base.voicing.vertical_decay_factor(),
        }
    }

    /// Everything measured, loaded into a builder over `base`.
    ///
    /// A builder rather than a preset: the caller names the instrument, writes
    /// its provenance, and decides whether the velocity map the felt fit
    /// produced is worth writing — which is a judgement about a degeneracy
    /// (`estimate::hammer`), not a measurement this module can make.
    pub fn builder(&self, base: Preset, config: &DecayConfig) -> PresetBuilder {
        let factor = self.vertical_factor(&base);
        let mut builder = PresetBuilder::new(base.clone());
        for note in &self.notes {
            let fallback_b = key_index(note.key)
                .map_or(0.0, |i| f64::from(base.notes.inharmonicity_b[i]));
            builder = builder.note(note.estimate(factor, fallback_b, config));
        }
        if let Some(split) = self.polarization() {
            builder = builder.polarization(split);
        }
        builder
    }
}

/// `sigma0` and `sigma1` from the prompt rates of one recording's partials.
///
/// The same weighted line as [`fit_decay_curve`](crate::estimate::decay), over
/// a different ordinate: `fast.sigma / factor` instead of the partial's T60.
/// See [`NoteSurvey::decay_curve`] for why. The guard is the same one that
/// keeps an unmeasured T60 out of the curve, applied to the prompt rate — a
/// partial whose prompt decay is slower than the record is long has not been
/// seen decay either.
fn prompt_decay_curve(
    partials: &[DecayFit],
    factor: f64,
    config: &DecayConfig,
) -> Option<DecayCurve> {
    if factor.is_nan() || factor <= 0.0 {
        return None;
    }
    let rates: Vec<(f64, f64)> = partials
        .iter()
        .filter(|fit| fit.frequency_hz > 0.0 && fit.fast.sigma > 0.0)
        .filter(|fit| LN_1000 / fit.fast.sigma <= config.max_t60_ratio * fit.span_s)
        .map(|fit| ((fit.frequency_hz / 1000.0).powi(2), fit.fast.sigma / factor))
        .collect();
    if rates.len() < 2 {
        return None;
    }
    let basis: Vec<f64> = rates.iter().flat_map(|&(x, _)| [1.0, x]).collect();
    let y: Vec<f64> = rates.iter().map(|&(_, s)| s).collect();
    let solution = weighted_least_squares(&basis, &y, &vec![1.0; rates.len()], 2)?;
    // Damping cannot be negative in either term: radiation and internal
    // friction only remove energy. A term the data wants negative is a term the
    // data does not support, so it is dropped and the rest refitted.
    let (mut sigma0, mut sigma1) = (solution[0], solution[1]);
    if sigma1 < 0.0 {
        sigma1 = 0.0;
        sigma0 = y.iter().sum::<f64>() / y.len() as f64;
    }
    if sigma0 < 0.0 {
        sigma0 = 0.0;
        let num: f64 = rates.iter().map(|&(x, s)| x * s).sum();
        let den: f64 = rates.iter().map(|&(x, _)| x * x).sum();
        sigma1 = if den > 0.0 { num / den } else { 0.0 };
    }
    let residual = (rates
        .iter()
        .map(|&(x, s)| (s - (sigma0 + sigma1 * x)).powi(2))
        .sum::<f64>()
        / rates.len() as f64)
        .sqrt();
    Some(DecayCurve {
        sigma0,
        sigma1,
        residual,
    })
}

/// Fits the engine's two-point velocity map to every layer speed in `reports`.
///
/// Pooled across notes on purpose: the map is global in the engine, and one
/// note's sixteen speeds carry the whole degeneracy between hammer mass, felt
/// stiffness and speed that `estimate::hammer` describes. What is common to
/// thirty notes is the library's velocity *scaling*; what is not averages down.
pub fn pooled_velocity_map(reports: &[HammerReport]) -> Result<VelocityMap> {
    let pairs: Vec<(u8, f64)> = reports.iter().flat_map(|r| r.layers.iter().copied()).collect();
    fit_velocity_map(&pairs)
}

/// Analyses one recording, through the trajectory cache.
pub fn analyze_sample(sample: &Sample, config: &SurveyConfig) -> Result<NoteAnalysis> {
    // Equal temperament is only where to start looking: the tracker's window is
    // 60 cents wide and no piano is tuned further out than that, and the first
    // fitting pass replaces the seed with the recording's own pitch.
    let seed = equal_temperament(sample.key);
    let note_config = config.note_config(seed)?;
    let trajectories = trajectories_for(sample, &note_config, config)?;
    analyze_trajectories(trajectories, &note_config)
}

/// The tracked trajectories of one recording: from the cache if the cache has
/// them at this geometry, otherwise from the audio, in which case they are
/// written back.
pub fn trajectories_for(
    sample: &Sample,
    note_config: &NoteConfig,
    config: &SurveyConfig,
) -> Result<NoteTrajectories> {
    let path = config
        .cache_dir
        .as_ref()
        .map(|dir| cache_path(dir, sample, &note_config.tracker.stft));
    if let Some(path) = &path {
        if !config.refresh_cache {
            // A cached file that does not describe this recording is a stale
            // one, not an answer: fall through and re-track rather than trust a
            // name.
            if let Ok(cached) = NoteTrajectories::read_json(path) {
                if cached.source == sample.path.to_string_lossy() {
                    return Ok(cached);
                }
            }
        }
    }

    let signal = load_signal(&sample.path, config)?;
    let (trajectories, _) = track_refined(
        &signal,
        f64::from(SAMPLE_RATE),
        InharmonicModel::harmonic(equal_temperament(sample.key)),
        note_config,
    )?;
    let trajectories = trajectories
        .with_source(sample.path.to_string_lossy().into_owned())
        .with_note(NoteId::layer(sample.key, sample.layer));
    if let Some(path) = &path {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        trajectories.write_json(path)?;
    }
    Ok(trajectories)
}

/// Decodes a recording onto the engine's clock and cuts it to the part that is
/// the instrument: mono, tail trimmed, length capped.
pub fn load_signal(path: &Path, config: &SurveyConfig) -> Result<Vec<f32>> {
    let recording = audio::load_at(path, SAMPLE_RATE)?;
    let rate = f64::from(recording.sample_rate);
    let mut signal = recording.mono();
    let keep = ((recording.duration_s() - config.trim_tail_s).min(config.max_duration_s) * rate)
        .max(0.0) as usize;
    if keep == 0 {
        return Err(Error::Estimate(format!(
            "{}: {:.2} s of audio is shorter than the {:.2} s trimmed from its tail",
            path.display(),
            recording.duration_s(),
            config.trim_tail_s
        )));
    }
    signal.truncate(keep);
    Ok(signal)
}

/// Where one recording's trajectories live.
///
/// The name carries the transform geometry, so changing it invalidates the
/// cache by missing it rather than by reading something that no longer applies.
/// Other tracker settings do not appear: `refresh_cache` is how those get
/// re-run.
fn cache_path(dir: &Path, sample: &Sample, stft: &StftConfig) -> PathBuf {
    dir.join(format!(
        "key{:03}-layer{:02}-w{}-h{}-n{}.json",
        sample.key, sample.layer, stft.window, stft.hop, stft.fft_size
    ))
}

/// Measures every mechanism recording a library maps.
///
/// The level of each is quoted against a strike of `reference_velocity` on the
/// same key, at the level the *instrument* plays both — the SFZ's own `volume`
/// on each side — which is the difference between "a damper landing is 37 dB
/// under the note" and "a damper landing is as loud as the note". A key-off
/// recording of a key the library never struck is measured against the nearest
/// key it did, and says which.
///
/// A recording that will not decode is skipped rather than fatal: this runs at
/// the end of a survey that took minutes, and one missing file must not cost
/// the other ninety.
pub fn measure_mechanism(
    library: &SampleLibrary,
    config: &crate::estimate::noise::NoiseConfig,
) -> MechanismMeasurements {
    // One decode per *file*, not per recording that refers to it: a library
    // that samples every third key answers three of its own key-off recordings
    // with the same strike, and these are multi-megabyte FLACs.
    let mut strikes: std::collections::HashMap<PathBuf, Option<f64>> =
        std::collections::HashMap::new();
    let mut reference = |key: u8| -> Option<(u8, f64)> {
        let sample = library.nearest_layer(key, config.reference_velocity)?;
        let level = *strikes.entry(sample.path.clone()).or_insert_with(|| {
            let recording = audio::load_at(&sample.path, SAMPLE_RATE).ok()?;
            let metrics = transient_metrics(&recording.mono(), f64::from(SAMPLE_RATE))?;
            Some(sample.volume_db + 20.0 * metrics.peak.log10())
        });
        Some((sample.key, level?))
    };
    let mut measure = |kind: MechanismKind| -> Vec<EventMetrics> {
        library
            .mechanism_of(kind)
            .into_iter()
            .filter_map(|sample| {
                let recording = audio::load_at(&sample.path, SAMPLE_RATE).ok()?;
                let metrics = transient_metrics(&recording.mono(), f64::from(SAMPLE_RATE))?;
                // A global event has no key of its own; the middle of the
                // keyboard is what §5 measured the pedal against.
                let (reference_key, strike_db) = reference(sample.key.unwrap_or(60))?;
                Some(EventMetrics {
                    key: sample.key,
                    level_db: sample.volume_db + 20.0 * metrics.peak.log10() - strike_db,
                    decay_s: metrics.decay_s,
                    centroid_hz: metrics.centroid_hz,
                    reference_key,
                })
            })
            .filter(|metrics| metrics.level_db.is_finite())
            .collect()
    };
    MechanismMeasurements {
        key_off: measure(MechanismKind::KeyOff),
        pedal_down: measure(MechanismKind::PedalDown),
        pedal_up: measure(MechanismKind::PedalUp),
        key_off_veltrack: library
            .mechanism_of(MechanismKind::KeyOff)
            .first()
            .and_then(|sample| sample.amp_veltrack),
        velocity_span: library.velocity_span(),
    }
}

fn median(values: impl Iterator<Item = f64>) -> Option<f64> {
    let mut v: Vec<f64> = values.filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return None;
    }
    v.sort_by(f64::total_cmp);
    let mid = v.len() / 2;
    Some(if v.len() % 2 == 0 {
        0.5 * (v[mid - 1] + v[mid])
    } else {
        v[mid]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_window_follows_the_note_and_stays_inside_its_bounds() {
        let config = SurveyConfig::default();
        // A0: twelve periods of 27.5 Hz is 21 000 samples, so the next power of
        // two, capped at the configured ceiling.
        assert_eq!(config.geometry(27.5).unwrap().window, 1 << 15);
        // C4: twelve periods is 2 200 samples, so 4 096 — the floor is not
        // binding yet.
        assert_eq!(config.geometry(261.6).unwrap().window, 1 << 12);
        // C8 wants 138 samples and gets the floor.
        assert_eq!(config.geometry(4186.0).unwrap().window, config.min_window);
        assert!(config.geometry(0.0).is_err());

        let stft = config.geometry(261.6).unwrap();
        assert_eq!(stft.hop, stft.window / config.hop_divisor);
        assert_eq!(stft.fft_size, stft.window * config.pad);
    }

    #[test]
    fn the_cache_name_changes_with_the_geometry() {
        let sample = Sample {
            path: PathBuf::from("/lib/C4v1.flac"),
            key: 60,
            layer: 0,
            lovel: 1,
            hivel: 26,
            volume_db: 0.0,
        };
        let config = SurveyConfig::default();
        let a = cache_path(Path::new("/cache"), &sample, &config.geometry(261.6).unwrap());
        let b = cache_path(Path::new("/cache"), &sample, &config.geometry(27.5).unwrap());
        assert_ne!(a, b);
        assert!(a.to_string_lossy().contains("key060-layer00"));
    }

    /// One `NoteAnalysis` per layer, differing only in what the fits returned,
    /// is all the aggregation needs to be exercised. Each layer's partials
    /// decay at `value` (its prompt rate) so that the medians can be checked
    /// through both the tuning and the damping.
    fn survey_of(values: &[f64]) -> NoteSurvey {
        use crate::estimate::decay::{DecayReport, Exponential, PolarizationSplit};
        use crate::estimate::inharmonic::InharmonicFit;
        let partials = |sigma: f64| {
            (1..=4)
                .map(|k| DecayFit {
                    k,
                    frequency_hz: 261.6 * f64::from(k),
                    fast: Exponential {
                        amplitude: 1.0,
                        sigma,
                    },
                    slow: Exponential {
                        amplitude: 0.0,
                        sigma,
                    },
                    beats: Default::default(),
                    residual_db: 0.0,
                    points: 100,
                    span_s: 60.0,
                })
                .collect()
        };
        NoteSurvey {
            key: 60,
            layers: values
                .iter()
                .enumerate()
                .map(|(i, &v)| LayerAnalysis {
                    layer: i as u8,
                    midi_velocity: 64,
                    analysis: NoteAnalysis {
                        trajectories: NoteTrajectories {
                            source: String::new(),
                            note: None,
                            sample_rate: 48_000.0,
                            window_s: 0.1,
                            hop_s: 0.01,
                            seed: InharmonicModel::harmonic(261.6),
                            onset_s: 0.0,
                            tracks: Vec::new(),
                        },
                        inharmonic: InharmonicFit {
                            model: InharmonicModel::new(v, 1e-4),
                            used: Vec::new(),
                            rejected: Vec::new(),
                            residual_cents: 0.0,
                            worst_cents: 0.0,
                            bands: None,
                            residual_cents_2: 0.0,
                        },
                        decays: DecayReport {
                            partials: partials(v),
                            curve: DecayCurve {
                                sigma0: v,
                                sigma1: 0.0,
                                residual: 0.0,
                            },
                            polarization: PolarizationSplit {
                                gain_db: -v,
                                decay_ratio: 0.3,
                                partials: 4,
                            },
                        },
                        unison: None,
                        strike: None,
                    },
                })
                .collect(),
        }
    }

    #[test]
    fn a_notes_value_is_the_median_of_its_layers() {
        // Two layers whose fits ran away, thirteen that agree: the median is
        // the answer, which a mean would not be.
        let config = DecayConfig::default();
        let mut values: Vec<f64> = (0..13).map(|i| 260.0 + f64::from(i) * 0.1).collect();
        values.push(1.0);
        values.push(9_000.0);
        let note = survey_of(&values);
        assert!((note.f0_hz().unwrap() - 260.6).abs() < 1e-9);
        // The partials of each layer decay at that layer's value, so the
        // damping law's constant term is the same median, divided by the
        // conversion into the table's convention.
        let curve = note.decay_curve(2.0, &config).unwrap();
        assert!((curve.sigma0 - 130.3).abs() < 1e-6, "{curve:?}");
        assert!(curve.sigma1.abs() < 1e-9, "{curve:?}");
        assert!((note.polarization().unwrap().gain_db + 260.6).abs() < 1e-9);
        assert_eq!(note.strike_position(), None);
        assert_eq!(note.detune_cents(), None);

        let estimate = note.estimate(2.0, 1e-4, &config);
        assert_eq!(estimate.key, 60);
        assert_eq!(estimate.hammer_stiffness, None);
        assert!(estimate.inharmonicity_b.unwrap() > 0.0);
        // The tuning is partial 1, not the fitted f0: they differ by the
        // stiffness factor and it is partial 1 that was measured.
        let f0 = estimate.f0_hz.unwrap();
        assert!((f0 * (1.0f64 + 1e-4).sqrt() - 260.6 * (1.0f64 + 1e-4).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn a_partial_whose_decay_outlasts_the_record_is_not_in_the_damping_law() {
        use crate::estimate::decay::Exponential;
        let config = DecayConfig::default();
        let slow = |k: u32, sigma: f64, span_s: f64| DecayFit {
            k,
            frequency_hz: 261.6 * f64::from(k),
            fast: Exponential {
                amplitude: 1.0,
                sigma,
            },
            slow: Exponential {
                amplitude: 0.0,
                sigma,
            },
            beats: Default::default(),
            residual_db: 0.0,
            points: 100,
            span_s,
        };
        // Two partials seen decay and one that is barely into its own T60 over
        // a three-second record: the third would halve the fitted `sigma0` if
        // it counted.
        let seen = vec![slow(1, 1.0, 30.0), slow(2, 1.0, 30.0), slow(3, 0.05, 3.0)];
        let curve = prompt_decay_curve(&seen, 1.0, &config).unwrap();
        assert!((curve.sigma0 - 1.0).abs() < 1e-9, "{curve:?}");
        assert!(prompt_decay_curve(&seen[2..], 1.0, &config).is_none());
    }

    #[test]
    fn an_empty_note_has_nothing_to_estimate() {
        let config = DecayConfig::default();
        let note = survey_of(&[]);
        assert_eq!(note.f0_hz(), None);
        assert!(note.decay_curve(2.0, &config).is_none());
        assert_eq!(note.estimate(2.0, 1e-4, &config).f0_hz, None);
    }
}
