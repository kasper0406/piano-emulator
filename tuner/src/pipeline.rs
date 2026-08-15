//! Stage 1 end to end: one recorded note in, one set of estimates out.
//!
//! The interesting part is the seed loop. The tracker is told where to look for
//! partial `k` and looks inside a window narrow enough that no two partials can
//! compete for one peak — which means a seed that is wrong by more than about
//! half a partial spacing at some `k` makes the tracker follow the *wrong*
//! partial from there up, and the estimators downstream have no way of knowing.
//! A harmonic seed is wrong in exactly that way: inharmonicity pushes partial
//! `k` sharp by a factor `sqrt(1 + B k^2)`, which for a typical `B` passes half
//! a partial spacing somewhere around the tenth partial.
//!
//! So the analysis is run twice. The first pass fits `(f0, B)` to the low
//! partials only, where a harmonic seed is still right; the second re-tracks
//! with that model, which is now good to a few cents everywhere, and fits
//! everything.
//!
//! That loop is also what bounds the fourth-order coefficient the last fit can
//! find. The seed the second pass tracks with is a *two-parameter* model, so a
//! `B4` displaces the high partials from where the tracker is looking for them:
//! at C2 a coefficient of `3e-8` puts partial 40 fifty cents off the seed,
//! inside the 60-cent association window, and one of `6e-8` puts it past —
//! whereupon the tracker follows the wrong peaks up there and the fit comes
//! back with a *worse* residual and no fourth-order term at all. A third pass
//! seeded with the full model would lift the ceiling; nothing measured so far
//! has needed it, and it would cost half again the only expensive step in the
//! pipeline.

use crate::error::{Error, Result};
use crate::estimate::decay::{fit_decays, DecayConfig, DecayReport};
use crate::estimate::inharmonic::{fit_inharmonic, InharmonicConfig, InharmonicFit};
use crate::estimate::strike::{fit_strike_position, StrikeConfig, StrikeFit};
use crate::estimate::unison::{estimate_unison, UnisonConfig, UnisonEstimate};
use crate::preset::NoteEstimate;
use crate::tracker::{PartialTracker, TrackerConfig};
use crate::trajectory::{InharmonicModel, NoteTrajectories};

/// How hard to work at getting the tracker's seed right.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SeedRefinement {
    /// Tracking passes. One trusts the caller's seed; two is the loop above;
    /// more buys nothing on any note tried so far.
    pub passes: usize,
    /// Partials the first pass is allowed to fit. Low enough that a harmonic
    /// seed still points at the right peaks: at `B = 1e-3` the eighth partial
    /// is 40 cents sharp of the eighth harmonic, and the tracker's window is
    /// 60 cents.
    pub first_pass_partials: u32,
}

impl Default for SeedRefinement {
    fn default() -> Self {
        Self {
            passes: 2,
            first_pass_partials: 8,
        }
    }
}

/// Every setting stage 1 has, in one place.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NoteConfig {
    pub tracker: TrackerConfig,
    pub inharmonic: InharmonicConfig,
    pub decay: DecayConfig,
    pub unison: UnisonConfig,
    pub strike: StrikeConfig,
    pub refinement: SeedRefinement,
}

/// What one note yielded.
///
/// The tuning and the decays are required — a recording that cannot give up its
/// partials or their envelopes is a failed analysis. The other two are
/// optional: a single-strung bass note has no unison to beat, and a note whose
/// partials stop before the first null of the strike comb has no strike
/// position in it. Both are ordinary facts about a recording, not errors.
#[derive(Clone, Debug)]
pub struct NoteAnalysis {
    pub trajectories: NoteTrajectories,
    pub inharmonic: InharmonicFit,
    pub decays: DecayReport,
    pub unison: Option<UnisonEstimate>,
    pub strike: Option<StrikeFit>,
}

impl NoteAnalysis {
    /// Packages the estimates for [`PresetBuilder`](crate::preset::PresetBuilder).
    pub fn estimate(&self, key: u8) -> NoteEstimate {
        let mut estimate = NoteEstimate::new(key)
            .with_inharmonic(&self.inharmonic)
            .with_decays(&self.decays);
        if let Some(unison) = &self.unison {
            estimate = estimate.with_unison(unison);
        }
        if let Some(strike) = &self.strike {
            estimate = estimate.with_strike(strike);
        }
        estimate
    }
}

/// Runs the whole per-note analysis on one mono signal.
pub fn analyze_note(
    signal: &[f32],
    sample_rate: f64,
    seed: InharmonicModel,
    config: &NoteConfig,
) -> Result<NoteAnalysis> {
    let (trajectories, _) = track_refined(signal, sample_rate, seed, config)?;
    analyze_trajectories(trajectories, config)
}

/// The estimators alone, on trajectories that have already been tracked.
///
/// Splitting this out is what makes a cached tracking pass useful: a survey of a
/// whole sample library transforms every recording once, writes the trajectories
/// to disk, and re-runs the fits against them as often as the estimators change.
///
/// **This half is the expensive one, and this comment used to say the
/// opposite** ("the half of `analyze_note` that costs milliseconds"). Measured
/// on A1 held for 26 s, the note the self-calibration gate leans on hardest:
/// rendering it costs 0.32 s, [`track_refined`] 0.20 s, and the fits here
/// **2.5-3.7 s** — `fit_decays` alone over eighty partials of a twenty-six
/// second record. The claim was presumably true when the decay fit read a
/// handful of partials of a short note; it is not true of the note this crate
/// actually runs, and a cache built on it buys a tenth of what it looks like it
/// should (`DECISIONS.md` 284).
pub fn analyze_trajectories(
    trajectories: NoteTrajectories,
    config: &NoteConfig,
) -> Result<NoteAnalysis> {
    let inharmonic = fit_inharmonic(&trajectories, &config.inharmonic)?;
    let decays = fit_decays(&trajectories, &config.decay)?;
    let unison = estimate_unison(&trajectories, &decays, &config.unison).ok();
    let strike = fit_strike_position(&decays.excitation_spectrum(), &config.strike).ok();
    Ok(NoteAnalysis {
        trajectories,
        inharmonic,
        decays,
        unison,
        strike,
    })
}

/// Tracks the note, refining the seed from the partials it finds.
pub fn track_refined(
    signal: &[f32],
    sample_rate: f64,
    seed: InharmonicModel,
    config: &NoteConfig,
) -> Result<(NoteTrajectories, InharmonicFit)> {
    let tracker = PartialTracker::new(config.tracker)?;
    let passes = config.refinement.passes.max(1);
    let mut model = seed;
    let mut result = None;
    for pass in 0..passes {
        let last = pass + 1 == passes;
        let trajectories = tracker.track(signal, sample_rate, model);
        let settings = if last {
            config.inharmonic
        } else {
            InharmonicConfig {
                max_partial: config
                    .refinement
                    .first_pass_partials
                    .min(config.inharmonic.max_partial),
                ..config.inharmonic
            }
        };
        let fit = fit_inharmonic(&trajectories, &settings)?;
        model = fit.model;
        result = Some((trajectories, fit));
    }
    result.ok_or_else(|| Error::Estimate("no tracking pass ran".into()))
}
