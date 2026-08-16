//! The sympathetic halo: how much of what a piano radiates is *not* the string
//! that was struck, and the two preset fields that decide it.
//!
//! `docs/history/TUNING_REPORT.md` §4 is the measurement this module exists to close. One
//! second after a fortissimo C7 the energy *between* the struck note's partials
//! is 3.5 dB below the energy *in* them; the engine's render of the same note
//! was 48 dB below. At C6 the recordings give −22 to −26 dB against the
//! engine's −48. In the middle of the compass the engine was already right —
//! C4 at −44 dB recorded against −47 rendered — and §4's first reading is that
//! *nothing there needs fixing*, which makes the mid-range a constraint on the
//! fit rather than a target for it.
//!
//! §5 adds an independent check that does not depend on any struck note:
//! Salamander's `harmL*` release-resonance recordings put the halo at
//! **−31 dB** at C3 and **−39 dB** at C5 relative to a strike of the same key,
//! ringing for 1–2 s.
//!
//! # Why this cannot be fitted the way stage 1 fits anything
//!
//! Every stage-1 estimator inverts a model: a decay rate comes out of a line
//! through log amplitudes, a strike position out of the nulls of a comb. There
//! is no such inversion here, because the halo of a note is the whole rest of
//! the instrument answering it, and what it is *worth* depends on the coupling,
//! the admittance, the duplex segments and the dampers at once. `TUNING.md`
//! says as much — the coupling is stage 2, and stage 2 is render-and-measure.
//!
//! So this module holds the two halves that a driver (`tuner/src/tools/sympathetic.rs`)
//! puts in a loop: the *measurements*, which are §4's and §5's own and are
//! computed here so that the engine's renders and the recordings go through
//! identical code, and the *step*, which turns a set of errors in dB into the
//! next [`HaloVoicing`] to render with.
//!
//! # What is actually free
//!
//! Two numbers and a shape, and the shape is not free:
//!
//! * `voicing.resonance_coupling` — one scalar, the overall level of the halo.
//! * `[voicing.bridge]`'s backbone — its **overall gain** and its **treble
//!   tilt**, and nothing else. The rest of the curve is `PHYSICS.md` §4's
//!   published driving-point mobility ([`MOBILITY_SHAPE`]): mean mobility
//!   ≈ 1.3e-3 s/kg over 100–1000 Hz, a slight dip to 2 kHz, a slight rise to
//!   4 kHz, and Ege & Boutillon's plate/rib transition at `f_lim ≈ 1.1 kHz`.
//!   Fitting 24 anchors to eight measurements would fit the measurements'
//!   noise; the published curve is data and is treated as such.
//! * `[voicing.bridge]`'s peaks — the board's discrete modes below ~500 Hz,
//!   seeded from the preset's own `soundboard.body_modes` exactly as §4 says
//!   to, with the ±10–15 dB of fluctuation §4 measured. Not fitted at all.
//!
//! There is deliberately **no per-register coupling field**. A register's share
//! of the halo is what the backbone's shape *is*: the drive a string receives
//! is `coupling · B(f)` at the frequency it answers on, so "more halo in the
//! treble" and "the backbone rises in the treble" are the same statement, and
//! adding a second control for it would be a parameter the model already has.

use crate::error::{Error, Result};
use crate::pipeline::{track_refined, NoteConfig};
use crate::preset::{
    BridgeAnchor, BridgePeak, BridgeVoicing, MAX_BRIDGE_GAIN_DB, MAX_BRIDGE_PEAKS, MAX_BRIDGE_Q,
    MIN_BRIDGE_GAIN_DB, MIN_BRIDGE_Q, Preset,
};
use crate::residual::{band_split, frame_spectrum, transient_metrics};
use crate::response::BridgeResponse;
use crate::trajectory::InharmonicModel;

/// The engine's ceiling on `voicing.resonance_coupling`
/// (`engine::resonance::MAX_COUPLING`), and on the loop the bridge closes
/// (`engine::resonance::MAX_BRIDGE_LOOP_GAIN`).
pub const MAX_RESONANCE_COUPLING: f32 = 0.05;
pub const MAX_BRIDGE_LOOP_GAIN: f32 = 0.25;

/// Share of the loop bound a *fit* is allowed to ask for.
///
/// The bound is a refusal threshold, not a target. A fit that walks its scalar
/// up until the validator is about to refuse writes a file with 0.000 dB of
/// headroom, and then any tightening of the `max|B|` measurement — one extra
/// grid point, a different rounding, the fine scan that closed the
/// grid-evasion hole — turns the shipped preset into one the engine refuses to
/// load. Ten percent is 0.9 dB of sound and the difference between a file that
/// is legal and a file that is *provably* legal.
pub const LOOP_GAIN_FIT_MARGIN: f32 = 0.9;

/// `PHYSICS.md` §4's published driving-point mobility, in dB relative to its
/// own value at the plate/rib transition frequency.
///
/// Read off the shape §4 states rather than fitted: mean mobility over
/// 100–1000 Hz (the flat middle), falling away below the lowest board modes, a
/// slight dip to 2 kHz and a slight rise to 4 kHz, and above `f_lim ≈ 1.1 kHz`
/// the regime where waves localise between the ribs and the apparent modal
/// density collapses. The ±10–15 dB of fluctuation §4 also quotes is *not*
/// here: that is the discrete modes, and they are the peaks.
pub const MOBILITY_SHAPE: [(f32, f32); 11] = [
    (20.0, -16.0),
    (50.0, -6.0),
    (100.0, 0.0),
    (200.0, 2.0),
    (400.0, -2.0),
    (700.0, 1.0),
    (1_100.0, 0.0),
    (2_000.0, -4.0),
    (4_000.0, -1.0),
    (8_000.0, -8.0),
    (16_000.0, -16.0),
];

/// The transition frequency the tilt pivots about: Ege & Boutillon's `f_lim`,
/// where a half wavelength equals the mean inter-rib spacing.
pub const TRANSITION_HZ: f64 = 1_100.0;

/// Peak-to-mean fluctuation of the discrete board modes, dB — the middle of
/// `PHYSICS.md` §4's measured ±10–15 dB.
const PEAK_FLUCTUATION_DB: f64 = 12.0;

/// Share of a partial's decay rate that is loss into the board
/// (`voicing.bridge.radiated_share`).
///
/// Read off the literature `PHYSICS.md` §4 already cites rather than fitted,
/// for the same reason [`MOBILITY_SHAPE`] is: Woodhouse's body-loss/air-loss
/// ratio for the piano exceeds 0 dB above ~160 Hz, i.e. more than half of what
/// a partial loses up there goes into the board. One half is that statement at
/// its own threshold, and it is the conservative end of it — the number the
/// board's *fluctuation* is allowed to modulate, with the mean already inside
/// the fitted `sigma(f)`.
pub const RADIATED_SHARE: f32 = 0.5;

/// What a driver is allowed to move, and where the fit is between renders.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HaloVoicing {
    /// `voicing.resonance_coupling`.
    pub coupling: f32,
    /// Overall gain of the admittance backbone, dB. Degenerate with `coupling`
    /// as far as the sound goes; both exist because `coupling` has a hard
    /// ceiling of 0.05 and the loop bound allows `coupling · max|B|` up to
    /// 0.25, so the gain is the headroom the coupling ceiling alone denies.
    pub backbone_gain_db: f64,
    /// Tilt of the backbone above [`TRANSITION_HZ`], in dB at 16 kHz. Positive
    /// lifts the treble, which is where §4 found the engine 26 dB short.
    pub treble_tilt_db: f64,
}

impl Default for HaloVoicing {
    fn default() -> Self {
        Self {
            coupling: 0.012,
            backbone_gain_db: 0.0,
            treble_tilt_db: 0.0,
        }
    }
}

impl HaloVoicing {
    /// The backbone anchors this voicing describes: [`MOBILITY_SHAPE`] lifted
    /// by `backbone_gain_db` and tilted above [`TRANSITION_HZ`], with every
    /// gain clamped into the schema's range.
    pub fn backbone(&self) -> Vec<BridgeAnchor> {
        let top = (f64::from(MOBILITY_SHAPE[MOBILITY_SHAPE.len() - 1].0) / TRANSITION_HZ).log2();
        MOBILITY_SHAPE
            .iter()
            .map(|&(hz, gain_db)| {
                let octaves = (f64::from(hz) / TRANSITION_HZ).log2().max(0.0);
                let tilt = self.treble_tilt_db * octaves / top;
                BridgeAnchor {
                    hz,
                    gain_db: (f64::from(gain_db) + self.backbone_gain_db + tilt).clamp(
                        f64::from(MIN_BRIDGE_GAIN_DB),
                        f64::from(MAX_BRIDGE_GAIN_DB),
                    ) as f32,
                }
            })
            .collect()
    }

    /// The whole `[voicing.bridge]` section: this backbone and the board's own
    /// modes as peaks.
    pub fn bridge(&self, peaks: Vec<BridgePeak>) -> BridgeVoicing {
        BridgeVoicing {
            backbone: self.backbone(),
            peaks,
            radiated_share: RADIATED_SHARE,
        }
    }

    /// Writes the voicing into a preset, leaving everything else alone, and
    /// checks it: a voicing past the loop bound is refused here rather than by
    /// the engine three minutes into a render.
    pub fn apply(&self, preset: &mut Preset, peaks: Vec<BridgePeak>) -> Result<()> {
        preset.voicing.resonance_coupling = self.coupling;
        preset.voicing.bridge = Some(self.bridge(peaks));
        preset.validate()
    }

    /// `coupling · max|B|` — the quantity both crates bound, so that a fit can
    /// see how much room it has left before it asks for it.
    pub fn loop_gain(&self, peaks: &[BridgePeak]) -> f32 {
        self.coupling * BridgeResponse::new(&self.bridge(peaks.to_vec())).max_magnitude()
    }

    /// The largest `coupling` this backbone leaves room for, given how much of
    /// the loop the duplex segments already occupy.
    ///
    /// `duplex_factor` is `Preset::duplex_response_factor` — zero for an
    /// instrument without segments, in which case only the bridge's own bound
    /// applies. Both loops are bounded at a quarter of unity and both are
    /// proportional to the coupling, so the ceiling is the tighter of the two —
    /// times [`LOOP_GAIN_FIT_MARGIN`], because a fit that servos its scalar
    /// onto the exact edge of legality ships a file that the next tightening of
    /// the measurement makes illegal.
    pub fn coupling_ceiling(&self, peaks: &[BridgePeak], duplex_factor: f32) -> f32 {
        let max_b = BridgeResponse::new(&self.bridge(peaks.to_vec())).max_magnitude();
        (LOOP_GAIN_FIT_MARGIN * MAX_BRIDGE_LOOP_GAIN / (max_b * duplex_factor.max(1.0)))
            .min(MAX_RESONANCE_COUPLING)
    }
}

/// The board's discrete modes as bridge resonances, seeded from the preset's
/// own `soundboard.body_modes`.
///
/// `PHYSICS.md` §4: "sharp well-separated peaks below ~500 Hz", and "seeded
/// from the existing body modes". The frequencies and Qs are the board's; only
/// the gains are converted, from the modes' own relative amplitudes to a
/// fluctuation whose peak-to-mean is 12 dB — §4's measured
/// ±10–15 dB. Nothing here is fitted.
pub fn peaks_from_body_modes(preset: &Preset) -> Vec<BridgePeak> {
    let modes: Vec<(f32, f32, f32)> = preset
        .soundboard
        .body_modes
        .iter()
        .filter(|m| m.hz > 0.0 && m.gain > 0.0 && m.q > 0.0)
        .map(|m| (m.hz, m.q, m.gain))
        .take(MAX_BRIDGE_PEAKS)
        .collect();
    if modes.is_empty() {
        return Vec::new();
    }
    let mut gains: Vec<f64> = modes.iter().map(|&(_, _, g)| f64::from(g).ln()).collect();
    gains.sort_by(f64::total_cmp);
    let centre = gains[gains.len() / 2];
    let widest = modes
        .iter()
        .map(|&(_, _, g)| (f64::from(g).ln() - centre).abs())
        .fold(0.0f64, f64::max)
        .max(1.0e-12);
    modes
        .iter()
        .map(|&(hz, q, gain)| BridgePeak {
            hz,
            q: q.clamp(MIN_BRIDGE_Q, MAX_BRIDGE_Q),
            gain_db: (PEAK_FLUCTUATION_DB * (f64::from(gain).ln() - centre) / widest) as f32,
        })
        .collect()
}

/// Settings for the between-partial census.
///
/// The defaults are `docs/history/TUNING_REPORT.md` §4's own, so a number measured here is
/// comparable with the table there rather than merely similar to it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HaloConfig {
    /// Window the between-partial energy is measured on, samples. 4096 at
    /// 48 kHz is 85 ms: short enough to keep a treble note's partials apart
    /// from what lies between them.
    pub window: usize,
    /// A partial is only counted if its peak stands within this of the note's
    /// loudest.
    pub level_db: f64,
    /// Top of the band the census looks at, Hz. The bottom is 0.75 of the
    /// note's own fundamental, below which a recording holds room rumble and
    /// the microphone's offset.
    pub band_high_hz: f64,
    /// When the halo is read, in seconds after the onset.
    pub at_s: f64,
}

impl Default for HaloConfig {
    fn default() -> Self {
        Self {
            window: 4_096,
            level_db: 70.0,
            band_high_hz: 12_000.0,
            at_s: 1.0,
        }
    }
}

/// One note's census: how much of what it radiates is not its own partials.
#[derive(Clone, Debug, PartialEq)]
pub struct BetweenPartials {
    /// Power between the partials relative to power in them, dB, at the strike.
    pub at_strike_db: f64,
    /// The same, `HaloConfig::at_s` later. This is §4's headline column.
    pub at_late_db: f64,
    /// Partials the census was taken against.
    pub partials: usize,
}

/// `docs/history/TUNING_REPORT.md` §4's between-partial measurement, on any signal that
/// contains one struck note.
///
/// The partials are the ones the tracker *found*, not the ones a model
/// predicts: a census asks what is left over once everything the note actually
/// put in the spectrum is accounted for.
pub fn between_partials(
    signal: &[f32],
    sample_rate: f64,
    f0_hz: f64,
    note_config: &NoteConfig,
    config: &HaloConfig,
) -> Result<BetweenPartials> {
    let (trajectories, fit) = track_refined(
        signal,
        sample_rate,
        InharmonicModel::harmonic(f0_hz),
        note_config,
    )?;
    let loudest = trajectories
        .tracks
        .iter()
        .filter_map(|t| t.peak())
        .map(|p| p.amplitude)
        .fold(0.0f64, f64::max);
    let floor = loudest * 10f64.powf(-config.level_db / 20.0);
    let frequencies: Vec<f64> = trajectories
        .tracks
        .iter()
        .filter(|t| t.peak().is_some_and(|p| p.amplitude >= floor))
        .filter_map(|t| t.weighted_frequency())
        .collect();
    if frequencies.is_empty() {
        return Err(Error::Estimate("no partials to take a census against".into()));
    }
    let band = (0.75 * fit.model.partial(1), config.band_high_hz);
    let guard = (4.0 * sample_rate / config.window as f64).max(3.0);
    let onset = (trajectories.onset_s * sample_rate) as usize;
    let at = |start: usize| -> Result<f64> {
        let frame = frame_spectrum(signal, start, config.window, 1)?;
        Ok(band_split(
            &frame,
            sample_rate,
            config.window,
            &frequencies,
            guard,
            band,
        )
        .between_db())
    };
    Ok(BetweenPartials {
        at_strike_db: at(onset)?,
        at_late_db: at(onset + (config.at_s * sample_rate) as usize)?,
        partials: frequencies.len(),
    })
}

/// What a release-resonance recording is worth, relative to a strike of the
/// same key: `docs/history/TUNING_REPORT.md` §5's `harm*` rows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResonanceLevel {
    /// Peak of the resonance relative to the strike's peak, dB, both at the
    /// level the instrument plays them.
    pub peak_db: f64,
    /// Time to fall 40 dB, seconds. §5 measured 1.01 s at C3 and 2.13 s at C5.
    pub decay_s: f64,
    /// Power-weighted centroid of the first 100 ms, Hz.
    pub centroid_hz: f64,
}

/// Measures one release resonance against one strike.
///
/// Both signals are on whatever scale the caller supplies; only the ratio is
/// used, which is what makes the same routine work on Salamander's files (with
/// the SFZ's own `volume` applied to each side) and on the engine's renders
/// (where there is no such attenuation and the ratio is the engine's own gain
/// staging).
pub fn resonance_level(
    resonance: &[f32],
    resonance_gain_db: f64,
    strike: &[f32],
    strike_gain_db: f64,
    sample_rate: f64,
) -> Option<ResonanceLevel> {
    let halo = transient_metrics(resonance, sample_rate)?;
    let note = transient_metrics(strike, sample_rate)?;
    if !(halo.peak > 0.0 && note.peak > 0.0) {
        return None;
    }
    Some(ResonanceLevel {
        peak_db: resonance_gain_db + 20.0 * halo.peak.log10()
            - (strike_gain_db + 20.0 * note.peak.log10()),
        decay_s: halo.decay_s,
        centroid_hz: halo.centroid_hz,
    })
}

/// One thing the fit is trying to hit: a measured target with a tolerance, and
/// the frequency it lives at.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HaloTarget {
    /// What it is called in `docs/history/TUNING_REPORT.md`.
    pub name: &'static str,
    /// The key the measurement is of, which is what puts it on the backbone.
    pub key: u8,
    /// The band the measurement is a level *in*, Hz — a note's between-partial
    /// energy sits around its own partials, and a release resonance around its
    /// own centroid. This is what makes a target an argument about a *part* of
    /// the backbone rather than about all of it.
    pub hz: f64,
    /// The level the recordings gave, dB, and how far from it is acceptable.
    pub target_db: f64,
    pub tolerance_db: f64,
}

/// `docs/history/TUNING_REPORT.md` §4 and §5's targets, as the fit reads them.
///
/// The C4 row is not a target but a *constraint*: §4's first reading is that
/// the mid-range was already right, so the fit must be able to fail on it.
pub fn salamander_targets() -> Vec<HaloTarget> {
    vec![
        HaloTarget {
            name: "harmLC3",
            key: 48,
            hz: 314.0,
            target_db: -31.0,
            tolerance_db: 3.0,
        },
        HaloTarget {
            name: "harmLC5",
            key: 72,
            hz: 507.0,
            target_db: -39.0,
            tolerance_db: 3.0,
        },
        HaloTarget {
            name: "between C4",
            key: 60,
            hz: 1_000.0,
            // §4: −44.3 dB recorded against −47.0 rendered. The engine is
            // inside the band already and the fit's job here is not to move.
            target_db: -45.5,
            tolerance_db: 1.5,
        },
        HaloTarget {
            name: "between C6",
            key: 84,
            hz: 2_500.0,
            // §4: −22.1 dB soft, −26.4 dB loud.
            target_db: -24.0,
            tolerance_db: 2.0,
        },
        HaloTarget {
            name: "between C7",
            key: 96,
            hz: 5_000.0,
            // §4: −15.9, −13.0 and −3.5 dB over the velocity layers. The band
            // is wide because the quantity is wide: it moves 12 dB with
            // velocity, which is the finding and not the error bar.
            target_db: -9.75,
            tolerance_db: 6.25,
        },
    ]
}

/// How far one render is from one target.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HaloError {
    pub target: HaloTarget,
    pub measured_db: f64,
}

impl HaloError {
    /// Signed error in dB: positive means the engine is too quiet, which is
    /// the direction §4 found it in everywhere above C5.
    pub fn error_db(&self) -> f64 {
        self.target.target_db - self.measured_db
    }

    pub fn inside_tolerance(&self) -> bool {
        self.error_db().abs() <= self.target.tolerance_db
    }
}

/// One step of the fit: errors in dB in, the next voicing out.
///
/// The update is deliberately the simplest thing that can work, because every
/// evaluation costs a render:
///
/// * the **overall** level moves by the error at the pivot frequency — the
///   weighted mean of the errors, weighted by each target's tolerance, so a
///   target the report gives a 1.5 dB band to pulls six times as hard as one
///   it gives a 6.25 dB band to;
/// * the **tilt** moves by the slope of the errors against `log2(f / f_lim)`,
///   which is the one degree of freedom the backbone has left.
///
/// `rate` damps both, because the halo's level is not exactly one dB per dB of
/// coupling — a louder halo wakes more voices, which makes it louder still —
/// and a full-step update oscillates. 0.6 converges in five or six renders on
/// everything tried.
///
/// The coupling takes as much of the overall move as its own ceiling allows and
/// the backbone gain takes the rest, so a fit is never silently clipped by
/// `MAX_COUPLING` without the curve being given the chance to carry it.
pub fn refine(
    voicing: HaloVoicing,
    errors: &[HaloError],
    peaks: &[BridgePeak],
    duplex_factor: f32,
    rate: f64,
) -> HaloVoicing {
    let usable: Vec<&HaloError> = errors
        .iter()
        .filter(|e| e.measured_db.is_finite() && e.error_db().is_finite())
        .collect();
    if usable.is_empty() {
        return voicing;
    }
    let weight = |e: &HaloError| 1.0 / e.target.tolerance_db.max(0.25);
    let total: f64 = usable.iter().map(|e| weight(e)).sum();
    let mean_error: f64 = usable.iter().map(|e| weight(e) * e.error_db()).sum::<f64>() / total;

    // Slope of the error against log frequency, about the transition: a
    // weighted least-squares line, which with one target degenerates to no
    // tilt at all rather than to a division by zero.
    let x = |e: &HaloError| (e.target.hz / TRANSITION_HZ).log2();
    let mean_x: f64 = usable.iter().map(|e| weight(e) * x(e)).sum::<f64>() / total;
    let variance: f64 = usable
        .iter()
        .map(|e| weight(e) * (x(e) - mean_x).powi(2))
        .sum();
    let covariance: f64 = usable
        .iter()
        .map(|e| weight(e) * (x(e) - mean_x) * (e.error_db() - mean_error))
        .sum();
    let slope = if variance > 1.0e-6 {
        covariance / variance
    } else {
        0.0
    };
    let top = (f64::from(MOBILITY_SHAPE[MOBILITY_SHAPE.len() - 1].0) / TRANSITION_HZ).log2();

    let mut next = HaloVoicing {
        treble_tilt_db: voicing.treble_tilt_db + rate * slope * top,
        ..voicing
    };
    // Split the overall move: as much into the coupling as its ceiling allows,
    // the remainder into the backbone's gain.
    //
    // The two are not independent — lifting the backbone lifts `max|B|`, which
    // *lowers* the coupling ceiling — so the split is settled and the ceiling
    // is then applied once more against the backbone that was actually chosen.
    // Without that second pass a step can leave the pair past the loop bound,
    // and the preset it writes is refused by both crates.
    let wanted_db = rate * mean_error;
    let ceiling = next.coupling_ceiling(peaks, duplex_factor);
    let coupling = (f64::from(voicing.coupling) * 10f64.powf(wanted_db / 20.0))
        .clamp(1.0e-4, f64::from(ceiling));
    let taken_db = 20.0 * (coupling / f64::from(voicing.coupling).max(1.0e-9)).log10();
    next.backbone_gain_db = (voicing.backbone_gain_db + wanted_db - taken_db).clamp(-20.0, 20.0);
    next.coupling = (coupling as f32).min(next.coupling_ceiling(peaks, duplex_factor));
    next
}

#[cfg(test)]
mod tests {
    use super::*;

    fn peaks() -> Vec<BridgePeak> {
        peaks_from_body_modes(&crate::preset::Preset::from_toml(
            &std::fs::read_to_string(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../presets/default.toml"),
            )
            .expect("read"),
        )
        .expect("parse"))
    }

    #[test]
    fn the_neutral_voicing_is_the_published_curve_and_nothing_else() {
        let voicing = HaloVoicing::default();
        let backbone = voicing.backbone();
        assert_eq!(backbone.len(), MOBILITY_SHAPE.len());
        for (anchor, &(hz, gain_db)) in backbone.iter().zip(&MOBILITY_SHAPE) {
            assert_eq!(anchor.hz, hz);
            assert_eq!(anchor.gain_db, gain_db);
        }
        // Ascending, which the schema requires and the engine interpolates on.
        assert!(backbone.windows(2).all(|w| w[0].hz < w[1].hz));
    }

    #[test]
    fn the_tilt_pivots_at_the_transition_and_leaves_the_bass_alone() {
        let tilted = HaloVoicing {
            treble_tilt_db: 12.0,
            ..HaloVoicing::default()
        };
        let flat = HaloVoicing::default().backbone();
        let lifted = tilted.backbone();
        for (a, b) in flat.iter().zip(&lifted) {
            let moved = f64::from(b.gain_db) - f64::from(a.gain_db);
            if a.hz <= TRANSITION_HZ as f32 {
                assert!(moved.abs() < 1.0e-4, "{} Hz moved {moved}", a.hz);
            } else {
                assert!(moved > 0.0, "{} Hz did not move", a.hz);
            }
        }
        // The tilt is quoted at the top of the curve, and that is where it is.
        assert!((f64::from(lifted[10].gain_db - flat[10].gain_db) - 12.0).abs() < 1.0e-4);
    }

    /// The peaks are the board's own modes, at the fluctuation §4 measured —
    /// not fitted, and inside the schema on both sides.
    #[test]
    fn the_bridge_peaks_are_the_boards_modes_at_the_measured_fluctuation() {
        let peaks = peaks();
        assert_eq!(peaks.len(), 24);
        assert!(peaks.iter().all(|p| p.hz < 500.0), "§4 asks for peaks below 500 Hz");
        assert!(peaks.iter().all(|p| (MIN_BRIDGE_Q..=MAX_BRIDGE_Q).contains(&p.q)));
        let widest = peaks
            .iter()
            .map(|p| f64::from(p.gain_db).abs())
            .fold(0.0, f64::max);
        assert!((widest - PEAK_FLUCTUATION_DB).abs() < 0.01, "{widest}");
        assert!(peaks.iter().any(|p| p.gain_db > 0.0) && peaks.iter().any(|p| p.gain_db < 0.0));
    }

    /// The whole point of `coupling_ceiling`: a backbone with gain in it eats
    /// the coupling's headroom, and the fit has to know that before it renders.
    #[test]
    fn a_louder_backbone_leaves_less_room_for_the_coupling() {
        let peaks = peaks();
        let quiet = HaloVoicing::default();
        let loud = HaloVoicing {
            backbone_gain_db: 12.0,
            ..quiet
        };
        assert!(loud.coupling_ceiling(&peaks, 0.0) < quiet.coupling_ceiling(&peaks, 0.0));
        // And the bound both crates enforce is respected at the ceiling.
        let at_ceiling = HaloVoicing {
            coupling: loud.coupling_ceiling(&peaks, 0.0),
            ..loud
        };
        assert!(at_ceiling.loop_gain(&peaks) <= MAX_BRIDGE_LOOP_GAIN + 1.0e-6);
        // Segments in the loop take the room away from the coupling: they are
        // never damped, so their share of it never goes back.
        assert!(loud.coupling_ceiling(&peaks, 4.0) < 0.26 * loud.coupling_ceiling(&peaks, 1.0));
    }

    /// A step that is too quiet everywhere lifts the level and does not invent
    /// a tilt; a step that is too quiet only in the treble does the opposite.
    #[test]
    fn a_uniform_error_moves_the_level_and_a_sloped_one_moves_the_tilt() {
        let peaks = peaks();
        let start = HaloVoicing::default();
        let target = |hz: f64, target_db: f64| HaloTarget {
            name: "t",
            key: 60,
            hz,
            target_db,
            tolerance_db: 2.0,
        };
        let uniform: Vec<HaloError> = [300.0, 1_100.0, 4_000.0]
            .into_iter()
            .map(|hz| HaloError {
                target: target(hz, -30.0),
                measured_db: -36.0,
            })
            .collect();
        let level = refine(start, &uniform, &peaks, 0.0, 1.0);
        assert!(level.treble_tilt_db.abs() < 0.1, "{level:?}");
        // Six dB up, however it was split between the two controls.
        let moved = 20.0 * f64::from(level.coupling / start.coupling).log10()
            + level.backbone_gain_db
            - start.backbone_gain_db;
        assert!((moved - 6.0).abs() < 0.1, "{moved}");

        let sloped: Vec<HaloError> = [(300.0, -30.0), (1_100.0, -30.0), (4_400.0, -30.0)]
            .into_iter()
            .zip([-30.0, -30.0, -42.0])
            .map(|((hz, target_db), measured_db)| HaloError {
                target: target(hz, target_db),
                measured_db,
            })
            .collect();
        let tilted = refine(start, &sloped, &peaks, 0.0, 1.0);
        assert!(tilted.treble_tilt_db > 5.0, "{tilted:?}");
    }

    /// A between-partial census on a signal that is *only* partials returns a
    /// very low number, and one with broadband energy added returns a high one.
    /// This is the metric §4 tabulates, so it has to mean what §4 says.
    #[test]
    fn the_census_separates_a_pure_note_from_one_with_a_halo_around_it() {
        let sample_rate = 48_000.0;
        let n = (sample_rate * 3.0) as usize;
        let f0 = 261.63;
        let mut clean = vec![0.0f32; n];
        let mut haloed = vec![0.0f32; n];
        // A deterministic "everything else in the instrument": a comb of
        // slowly decaying tones that are not partials of this note.
        let others: Vec<f64> = (1..40).map(|i| 173.0 * f64::from(i) + 41.0).collect();
        for i in 0..n {
            let t = i as f64 / sample_rate;
            let mut note = 0.0;
            for k in 1..=12 {
                let f = f0 * f64::from(k);
                note += (0.5 / f64::from(k)) * (-1.2 * t).exp()
                    * (std::f64::consts::TAU * f * t).sin();
            }
            let mut halo = 0.0;
            for (j, &f) in others.iter().enumerate() {
                halo += 0.004 * (-0.35 * t).exp()
                    * (std::f64::consts::TAU * f * t + 0.37 * j as f64).sin();
            }
            clean[i] = note as f32;
            haloed[i] = (note + halo) as f32;
        }
        let survey = crate::survey::SurveyConfig::default();
        let note_config = survey.note_config(f0).unwrap();
        let config = HaloConfig::default();
        let a = between_partials(&clean, sample_rate, f0, &note_config, &config).unwrap();
        let b = between_partials(&haloed, sample_rate, f0, &note_config, &config).unwrap();
        assert!(b.at_late_db > a.at_late_db + 20.0, "{a:?} against {b:?}");
        // And the halo grows relative to the note, because it decays slower —
        // which is the shape of §4's own columns.
        assert!(b.at_late_db > b.at_strike_db, "{b:?}");
    }
}
