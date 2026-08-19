//! The realised responses of the two filters the preset schema describes,
//! mirrored from the engine.
//!
//! `preset.rs` is the tuner's copy of the *file*; this is the tuner's copy of
//! what two of its sections mean. Both are duplicated rather than shared for
//! the reason `DECISIONS.md` item 57 gives — the tuner does not depend on the
//! engine, and the preset file is the whole interface — and both are kept
//! honest the same way, by a test that holds the copies against each other on
//! material both can read.
//!
//! Why a *magnitude* mirror is needed at all, when the tuner never filters
//! anything with these: because the two stability bounds the schema carries are
//! computed from the filter that gets **built**, not from the numbers in the
//! file. `voicing.bridge.backbone` is a curve, and the curve is fitted by a
//! cascade of shelves that only approximates it (`engine::resonance`), so
//! `max|B|` is a property of the fit. A tuner that guessed at it would write
//! presets the engine refuses, or — worse — refuse presets the engine would
//! have played. Every routine below is therefore the engine's own arithmetic,
//! term for term and in the same order and precision, so that the two answers
//! agree to the last bit.

use crate::preset::{BridgeVoicing, DuplexMode};

const SAMPLE_RATE: f64 = crate::SAMPLE_RATE as f64;

/// One second-order section, in `f64` as the engine builds it: the bridge's low
/// modes are the sharpest filters in the instrument and an `f32` coefficient
/// has about four significant digits of pole position left down there.
#[derive(Clone, Copy, Debug, Default)]
struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
}

/// `cos(-w)`, `sin(-w)`, `cos(-2w)`, `sin(-2w)` for one frequency, computed
/// once per frequency and reused across the sections — the engine's own
/// arrangement, including the double-angle identity, because the maximum search
/// below has to agree with the engine's to the last bit.
#[derive(Clone, Copy, Debug)]
struct Trig {
    c1: f64,
    s1: f64,
    c2: f64,
    s2: f64,
}

impl Trig {
    fn at_hz(hz: f64) -> Trig {
        Trig::at_radians(std::f64::consts::TAU * hz / SAMPLE_RATE)
    }

    fn at_radians(w: f64) -> Trig {
        let (s1, c1) = (-w).sin_cos();
        Trig {
            c1,
            s1,
            c2: c1 * c1 - s1 * s1,
            s2: 2.0 * s1 * c1,
        }
    }
}

impl Biquad {
    /// `|H(e^{jw})|`, `w` in radians per sample.
    fn magnitude(&self, w: f64) -> f64 {
        self.magnitude_at(&Trig::at_radians(w))
    }

    /// `|H|` from the four already-computed trigonometric terms.
    fn magnitude_at(&self, t: &Trig) -> f64 {
        let nr = self.b0 + self.b1 * t.c1 + self.b2 * t.c2;
        let ni = self.b1 * t.s1 + self.b2 * t.s2;
        let dr = 1.0 + self.a1 * t.c1 + self.a2 * t.c2;
        let di = self.a1 * t.s1 + self.a2 * t.s2;
        ((nr * nr + ni * ni) / (dr * dr + di * di)).sqrt()
    }

    /// The RBJ high shelf at `S = 1`.
    fn high_shelf(hz: f64, gain_db: f64) -> Biquad {
        let a = 10.0f64.powf(gain_db / 40.0);
        let w0 = std::f64::consts::TAU * hz / SAMPLE_RATE;
        let (sin, cos) = w0.sin_cos();
        let alpha = sin * std::f64::consts::FRAC_1_SQRT_2;
        let two_root = 2.0 * a.sqrt() * alpha;
        let a0 = (a + 1.0) - (a - 1.0) * cos + two_root;
        Biquad {
            b0: a * ((a + 1.0) + (a - 1.0) * cos + two_root) / a0,
            b1: -2.0 * a * ((a - 1.0) + (a + 1.0) * cos) / a0,
            b2: a * ((a + 1.0) + (a - 1.0) * cos - two_root) / a0,
            a1: 2.0 * ((a - 1.0) - (a + 1.0) * cos) / a0,
            a2: ((a + 1.0) - (a - 1.0) * cos - two_root) / a0,
        }
    }

    /// The RBJ peaking (bell) section.
    fn peaking(hz: f64, q: f64, gain_db: f64) -> Biquad {
        let a = 10.0f64.powf(gain_db / 40.0);
        let w0 = std::f64::consts::TAU * hz / SAMPLE_RATE;
        let (sin, cos) = w0.sin_cos();
        let alpha = sin / (2.0 * q);
        let a0 = 1.0 + alpha / a;
        Biquad {
            b0: (1.0 + alpha * a) / a0,
            b1: (-2.0 * cos) / a0,
            b2: (1.0 - alpha * a) / a0,
            a1: (-2.0 * cos) / a0,
            a2: (1.0 - alpha / a) / a0,
        }
    }
}

/// Refinement passes the backbone fit is allowed (`engine::resonance`).
const BACKBONE_FIT_PASSES: usize = 32;
/// Largest step one shelf of the fit may be pushed to, dB.
const MAX_BACKBONE_STEP_DB: f64 = 80.0;
/// The coarse log grid the realised response is measured on, and its ends.
const RESPONSE_GRID: usize = 512;
const RESPONSE_LOW_HZ: f64 = 20.0;
const RESPONSE_HIGH_HZ: f64 = 20_000.0;
/// The fine scan around every resonance, and the refinement that follows it
/// (`engine::resonance`). A cascade adds decibels, so two overlapping peaks put
/// their maximum *between* their centres, where neither the grid nor the centres
/// look; the engine documents the construction, inside the schema, that hides
/// 15.6 dB from a measurement that only samples those.
const PEAK_WINDOW_BANDWIDTHS: f64 = 8.0;
const SAMPLES_PER_BANDWIDTH: f64 = 64.0;
const REFINE_STEPS: usize = 60;

/// The bridge admittance as the engine realises it: a broadband gain, one
/// second-order high shelf per backbone interval fitted to the anchors, and one
/// peaking section per bridge resonance.
pub struct BridgeResponse {
    gain: f64,
    sections: Vec<Biquad>,
    /// Centre frequency and -3 dB bandwidth of every peak.
    peaks: Vec<(f64, f64)>,
}

impl BridgeResponse {
    /// The flat bus, which is what a preset without a `[voicing.bridge]`
    /// section asks for.
    pub fn unity() -> BridgeResponse {
        BridgeResponse {
            gain: 1.0,
            sections: Vec::new(),
            peaks: Vec::new(),
        }
    }

    pub fn new(bridge: &BridgeVoicing) -> BridgeResponse {
        let anchors: Vec<(f64, f64)> = bridge
            .backbone
            .iter()
            .map(|a| (f64::from(a.hz), f64::from(a.gain_db)))
            .collect();
        if anchors.len() < 2 {
            // Refused by `validate` before it can reach a filter; the guard is
            // here so that a caller measuring an unvalidated draft cannot index
            // past the end of the anchor list.
            return BridgeResponse::unity();
        }
        let (gain_db, steps) = fit_backbone(&anchors);
        let mut sections = Vec::with_capacity(steps.len() + bridge.peaks.len());
        for (i, &step) in steps.iter().enumerate() {
            let corner = (anchors[i].0 * anchors[i + 1].0).sqrt();
            sections.push(Biquad::high_shelf(corner, step));
        }
        for peak in &bridge.peaks {
            sections.push(Biquad::peaking(
                f64::from(peak.hz),
                f64::from(peak.q),
                f64::from(peak.gain_db),
            ));
        }
        BridgeResponse {
            gain: 10.0f64.powf(gain_db / 20.0),
            sections,
            peaks: bridge
                .peaks
                .iter()
                .map(|p| (f64::from(p.hz), f64::from(p.hz / p.q.max(1.0e-6))))
                .collect(),
        }
    }

    /// Either the filter a `[voicing.bridge]` section describes or the flat bus
    /// a preset without one plays.
    pub fn of(bridge: Option<&BridgeVoicing>) -> BridgeResponse {
        match bridge {
            Some(bridge) => BridgeResponse::new(bridge),
            None => BridgeResponse::unity(),
        }
    }

    fn is_unity(&self) -> bool {
        self.sections.is_empty() && self.gain == 1.0
    }

    /// `|B(f)|` of the filter as realised.
    pub fn magnitude(&self, hz: f64) -> f64 {
        self.magnitude_trig(&Trig::at_hz(hz))
    }

    fn magnitude_trig(&self, t: &Trig) -> f64 {
        self.sections
            .iter()
            .fold(self.gain, |m, s| m * s.magnitude_at(t))
    }

    /// The largest `|B(f)|` in the audio band — what the coupling loop is
    /// bounded against.
    ///
    /// The coarse log grid covers the smooth half of the filter; the fine scan
    /// through every resonance covers the modal half, where the cascade's
    /// decibels add and the maximum of two overlapping peaks sits between their
    /// centres rather than at either of them; the golden-section refinement
    /// turns the best sample of each into a local maximum. The engine does
    /// exactly this and returns the result as an `f32`; the cast is part of the
    /// mirror, because a bound that differed in the last bit would let a preset
    /// through one crate and not the other.
    pub fn max_magnitude(&self) -> f32 {
        if self.is_unity() {
            return 1.0;
        }
        let mut max = 0.0f64;

        let ratio = (RESPONSE_HIGH_HZ / RESPONSE_LOW_HZ).ln() / (RESPONSE_GRID - 1) as f64;
        let mut best = (0.0f64, RESPONSE_LOW_HZ);
        for i in 0..RESPONSE_GRID {
            let hz = RESPONSE_LOW_HZ * (i as f64 * ratio).exp();
            let m = self.magnitude(hz);
            if m > best.0 {
                best = (m, hz);
            }
        }
        max = max.max(self.refine(best.1, best.1 * ratio));

        for &(hz, bandwidth) in &self.peaks {
            let step = bandwidth / SAMPLES_PER_BANDWIDTH;
            if !step.is_finite() || step <= 0.0 {
                continue;
            }
            let lo = (hz - PEAK_WINDOW_BANDWIDTHS * bandwidth).max(RESPONSE_LOW_HZ);
            let hi = (hz + PEAK_WINDOW_BANDWIDTHS * bandwidth).min(RESPONSE_HIGH_HZ);
            let mut best = (self.magnitude(hz), hz);
            let points = ((hi - lo) / step).ceil() as usize;
            for i in 0..=points {
                let f = (lo + i as f64 * step).min(hi);
                let m = self.magnitude(f);
                if m > best.0 {
                    best = (m, f);
                }
            }
            max = max.max(self.refine(best.1, step));
        }
        max as f32
    }

    /// Golden-section maximisation of `|B|` over `centre +- half_width`.
    fn refine(&self, centre: f64, half_width: f64) -> f64 {
        const INV_PHI: f64 = 0.618_033_988_749_894_9;
        let mut lo = (centre - half_width).max(RESPONSE_LOW_HZ);
        let mut hi = (centre + half_width).min(RESPONSE_HIGH_HZ);
        let mut best = self.magnitude(centre);
        let (mut c, mut d) = (hi - INV_PHI * (hi - lo), lo + INV_PHI * (hi - lo));
        let (mut fc, mut fd) = (self.magnitude(c), self.magnitude(d));
        for _ in 0..REFINE_STEPS {
            best = best.max(fc).max(fd);
            if fc > fd {
                hi = d;
                d = c;
                fd = fc;
                c = hi - INV_PHI * (hi - lo);
                fc = self.magnitude(c);
            } else {
                lo = c;
                c = d;
                fc = fd;
                d = lo + INV_PHI * (hi - lo);
                fd = self.magnitude(d);
            }
        }
        best.max(fc).max(fd)
    }
}

/// Fits a shelf cascade to the backbone anchors: the broadband gain in dB and
/// one step in dB per anchor interval (`engine::resonance::fit_backbone`).
fn fit_backbone(anchors: &[(f64, f64)]) -> (f64, Vec<f64>) {
    let n = anchors.len();
    let corners: Vec<f64> = (0..n - 1)
        .map(|i| (anchors[i].0 * anchors[i + 1].0).sqrt())
        .collect();

    let mut gain_db = anchors[0].1;
    let mut steps: Vec<f64> = (0..n - 1).map(|i| anchors[i + 1].1 - anchors[i].1).collect();
    let mut best = (gain_db, steps.clone(), f64::INFINITY);

    for _ in 0..BACKBONE_FIT_PASSES {
        let shelves: Vec<Biquad> = corners
            .iter()
            .zip(&steps)
            .map(|(&hz, &step)| Biquad::high_shelf(hz, step))
            .collect();
        let errors: Vec<f64> = anchors
            .iter()
            .map(|&(hz, target)| {
                let w = std::f64::consts::TAU * hz / SAMPLE_RATE;
                let realised = shelves.iter().fold(1.0, |m, s| m * s.magnitude(w));
                target - (gain_db + 20.0 * realised.max(1.0e-300).log10())
            })
            .collect();

        let worst = errors.iter().fold(0.0f64, |m, e| m.max(e.abs()));
        if worst < best.2 {
            best = (gain_db, steps.clone(), worst);
        }
        if worst < 1.0e-3 || !worst.is_finite() {
            break;
        }
        gain_db += errors[0];
        for i in 0..n - 1 {
            steps[i] = (steps[i] + errors[i + 1] - errors[i])
                .clamp(-MAX_BACKBONE_STEP_DB, MAX_BACKBONE_STEP_DB);
        }
    }
    (best.0, best.1)
}

/// `sigma · T60`, i.e. `ln(1000)` (`engine::duplex`).
const T60_DECADES: f32 = 3.0 * std::f32::consts::LN_10;

/// Pole radius, pole angle and input gain of one segment.
///
/// Since `DECISIONS.md` 481 `gain_db` is an **impulse** normalisation stated
/// against the key's own bridge scale — the same one `string.rs` builds a
/// partial's gain from — so the input gain is `G * scale / SAMPLE_RATE` and the
/// steady response at resonance is the derived quantity. Computed in `f32`
/// because the engine computes it in `f32`.
fn resonator(mode: &DuplexMode, scale: f32) -> (f32, f32, f32) {
    let sigma = T60_DECADES / mode.t60_s;
    let r = (-sigma / crate::SAMPLE_RATE as f32).exp();
    let w = std::f32::consts::TAU * mode.hz / crate::SAMPLE_RATE as f32;
    (
        r,
        w,
        10f32.powf(mode.gain_db / 20.0) * scale * mode.hz / crate::SAMPLE_RATE as f32,
    )
}

/// The key's bridge-force scale per hertz of the resonator's own fundamental,
/// mirroring `engine::string`'s `bridge_excitation_scale_per_hz`.
pub fn bridge_excitation_scale_per_hz(excitation_scale: f32, bridge_gain: f32) -> f32 {
    excitation_scale * bridge_gain / REFERENCE_F0
}

/// Note the per-note output gains are normalised against (`engine::string`).
pub const REFERENCE_F0: f32 = 261.6256;

/// `|D(f)|` of a whole row of segments as realised: signal out per unit of
/// **steady** drive at `hz` (`engine::duplex::magnitude`).
///
/// Magnitudes are summed rather than complex responses — the conservative
/// reading, and the right one for a stability bound, because the relative phase
/// of two segments that land on one frequency is not something a preset
/// controls.
pub fn duplex_magnitude(modes: &[DuplexMode], scale: f32, hz: f32) -> f32 {
    let w = std::f32::consts::TAU * hz / crate::SAMPLE_RATE as f32;
    modes
        .iter()
        .map(|mode| {
            let (r, wk, g) = resonator(mode, scale);
            let delta = wk - w;
            let (re, im) = (1.0 - r * delta.cos(), -r * delta.sin());
            0.5 * g / (re * re + im * im).sqrt()
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preset::{BridgeAnchor, BridgePeak};

    fn bridge(backbone: &[(f32, f32)], peaks: &[(f32, f32, f32)]) -> BridgeVoicing {
        BridgeVoicing {
            backbone: backbone
                .iter()
                .map(|&(hz, gain_db)| BridgeAnchor { hz, gain_db })
                .collect(),
            peaks: peaks
                .iter()
                .map(|&(hz, q, gain_db)| BridgePeak { hz, q, gain_db })
                .collect(),
            radiated_share: 0.0,
        }
    }

    #[test]
    fn a_bridge_with_no_section_is_the_flat_bus() {
        let unity = BridgeResponse::of(None);
        assert_eq!(unity.max_magnitude(), 1.0);
        for hz in [20.0, 440.0, 1_100.0, 19_000.0] {
            assert_eq!(unity.magnitude(hz), 1.0);
        }
    }

    /// The fit is what makes the mirror necessary: the realised cascade passes
    /// through the anchors, so `max|B|` is neither the largest anchor nor the
    /// sum of the peaks.
    #[test]
    fn the_fitted_backbone_passes_through_its_anchors() {
        // `engine::resonance`'s own anchor list, so this is a comparison of
        // the two fits and not of two different curves.
        let voicing = bridge(
            &[
                (30.0, -12.0),
                (100.0, -2.0),
                (300.0, 1.5),
                (1_100.0, 0.0),
                (2_000.0, -4.0),
                (4_000.0, -1.0),
                (8_000.0, -9.0),
                (14_000.0, -18.0),
            ],
            &[],
        );
        let response = BridgeResponse::new(&voicing);
        for anchor in &voicing.backbone {
            let realised = 20.0 * response.magnitude(f64::from(anchor.hz)).log10();
            assert!(
                (realised - f64::from(anchor.gain_db)).abs() < 1.0,
                "{} Hz realised {realised:.2} dB against {} asked for",
                anchor.hz,
                anchor.gain_db
            );
        }
    }

    #[test]
    fn a_peak_shows_up_in_the_measured_maximum() {
        // A Q-50 resonance is narrower than the grid's 1.4 % step, so this
        // only passes because the peak centres are probed as well.
        let response = BridgeResponse::new(&bridge(&[(20.0, 0.0), (16_000.0, 0.0)], &[(
            3_137.0, 50.0, 18.0,
        )]));
        let max_db = 20.0 * f64::from(response.max_magnitude()).log10();
        assert!((max_db - 18.0).abs() < 0.5, "{max_db:.2} dB");
    }

    /// Since `DECISIONS.md` 481 the realised response at a segment's own
    /// frequency is a *derived* quantity — the gain over the mode's own
    /// bandwidth — so it scales with `t60_s` as a Q does, and the loop bound
    /// this mirror exists to compute sees that.
    #[test]
    fn a_segments_realised_response_is_its_gain_over_its_own_bandwidth() {
        for t60_s in [0.1f32, 0.5, 1.5, 3.0] {
            for gain_db in [-30.0f32, -12.0, 0.0] {
                let mode = DuplexMode {
                    hz: 4_400.0,
                    gain_db,
                    t60_s,
                };
                let sigma = T60_DECADES / t60_s;
                let predicted = 10f32.powf(gain_db / 20.0) * mode.hz / (2.0 * sigma);
                let realised = duplex_magnitude(&[mode], 1.0, mode.hz);
                assert!(
                    (realised / predicted - 1.0).abs() < 0.01,
                    "t60 {t60_s} gain {gain_db}: {realised:.3e} against {predicted:.3e}"
                );
            }
        }
    }

    #[test]
    fn segments_on_one_frequency_add_up_and_ones_far_apart_do_not() {
        let together = [
            DuplexMode { hz: 4_400.0, gain_db: 0.0, t60_s: 1.0 },
            DuplexMode { hz: 4_400.0, gain_db: 0.0, t60_s: 1.0 },
        ];
        let apart = [
            DuplexMode { hz: 4_400.0, gain_db: 0.0, t60_s: 1.0 },
            DuplexMode { hz: 9_100.0, gain_db: 0.0, t60_s: 1.0 },
        ];
        let one = 4_400.0 / (2.0 * T60_DECADES);
        assert!((duplex_magnitude(&together, 1.0, 4_400.0) / (2.0 * one) - 1.0).abs() < 0.02);
        assert!((duplex_magnitude(&apart, 1.0, 4_400.0) / one - 1.0).abs() < 0.02);
    }
}
