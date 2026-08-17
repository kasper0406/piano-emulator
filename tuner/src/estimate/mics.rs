//! Where the microphone pair was, measured off the two channels it recorded.
//!
//! `PHYSICS.md` §8 gives the engine a pair of virtual capsules over the string
//! band (`voicing.mics`, `DECISIONS.md` 351-358), and at that milestone the
//! section's five numbers came out of a sweep over the gate's own renders, with
//! nothing about the *recording* entering except the range of lags item 314
//! published. This module is the inversion that reads the pair out of the
//! recording itself — the same thing `estimate::inharmonic` is to a partial
//! ladder, applied to a stereo image.
//!
//! # What is measurable, and what it measures
//!
//! Two capsules `d` apart above a source at lateral offset `x` and height `h`
//! hear the same sound at two times, and the *difference* of those two times is
//! the only part of them a recording carries — a common delay is a latency and
//! every take in the library is trimmed to its own onset (`DECISIONS.md` 315).
//! So the observable is the **interchannel time difference**
//!
//! ```text
//! ITD(x) = ( sqrt((x + d/2)^2 + h^2) - sqrt((x - d/2)^2 + h^2) ) / c
//! ```
//!
//! and the free variables it can be inverted for are the three lengths of
//! [`MicGeometry`]: the spacing `d`, the height `h`, and the `span` that turns
//! a key's normalised pan position into the metres `x` it sits at. One curve,
//! three unknowns, and they are **not** three independent directions of it:
//!
//! * the curve **saturates** at `±d/c` for a source far off to one side, so its
//!   asymptote is the spacing and nothing else — this is the well-conditioned
//!   direction, and it is why a pair 12 cm apart and a pair 50 cm apart are
//!   never confusable;
//! * how *fast* it saturates is set by `span/h` — a key at the end of the
//!   compass is `0.6·span` from the centre line and `h` above it, and only the
//!   ratio of those two turns into an angle;
//! * `span` and `h` separately move the curve only through the near-field term
//!   `(d/2)^2` under the roots, which is a few per cent of the radius at any
//!   plausible geometry. They are therefore fitted **as their ratio**, with the
//!   height held at whatever the caller believes, and [`GeometryFit::conditioning`]
//!   reports how flat that direction was so the claim is never larger than the
//!   measurement.
//!
//! A **level** difference is measurable too and is what pins the absolute
//! scale — inverse distance is not scale-free where a delay difference nearly
//! is — but the recording's interchannel level difference is the mic pair's
//! geometry *plus the instrument's own directivity*, which `estimate::directivity`
//! already measures as a separate fact and which the model has no term for. It
//! is reported ([`KeyLag::ild_db`]) and deliberately not fitted: a scale read
//! out of a term the model does not contain would be a number about the piano's
//! radiation pattern wearing a microphone's name.
//!
//! # How the time difference is measured
//!
//! [`interchannel_lag`] is GCC-PHAT (Knapp & Carter 1976): the cross-spectrum
//! divided by its own magnitude before the inverse transform, so every bin in
//! the band votes once regardless of how loud it is. On a piano note that
//! matters more than usual — the fundamental carries most of the energy and the
//! least of the timing, since one period of A0 is 36 ms and the whole search is
//! ±3 — and the alternative, a plain cross-correlation, returns the period of
//! the loudest partial about as often as it returns the delay.
//!
//! Which band it is read in, and over how much of the note, are both measured
//! rather than assumed — see [`LagConfig`], which is where the usual practice
//! (a mid-band onset window) turned out to return noise on this library and the
//! bass over the whole note turned out to be where the two channels agree at
//! all.
//!
//! # The self-calibration this supports
//!
//! `TUNING.md`'s standing rule is that an estimator is worth its output only if
//! it can recover a parameter the engine was *told* to have. The engine's mic
//! stage is not a literal delay pair — it is mid plus side, and the mid is
//! delay-free by construction so that no scoreboard moves (item 352) — so it is
//! not obvious that a TDOA estimator can read it at all. Write the two channels
//! out for a single source at pan `p`, with `δ` the geometric delay of the
//! farther capsule:
//!
//! ```text
//! L = (m - w·u_R/2)·x(t)  +  (w·u_L/2)·x(t-δ)
//! R = (m + w·u_R/2)·x(t)  -  (w·u_L/2)·x(t-δ)
//! ```
//!
//! — two taps, at zero and at the geometric delay, and the cross-correlation of
//! that pair carries a lobe at each. It turns out that the estimator does read
//! the geometry back, but **not at unit scale**: the far channel is
//! maximum-phase at any usable width and carries a further `δ` of group delay of
//! its own, so a phase-transform estimator sees close to twice the geometric
//! delay. That factor is [`ENGINE_LAG_PER_ITD`], it is measured rather than
//! assumed, and `tuner/tests/mics.rs` is where the recovery it enables is
//! checked against the engine rather than asserted here.

use rustfft::{num_complex::Complex32, FftPlanner};

use crate::error::{Error, Result};
use crate::numeric::{parabolic_offset, NelderMead};

/// Speed of sound in air, m/s. Mirrors `soundboard::SPEED_OF_SOUND`: this crate
/// does not link the engine, and a metre of geometry has to mean the same
/// number of samples on both sides of the fit or the inversion is of a
/// different instrument.
pub const SPEED_OF_SOUND: f64 = 343.0;

/// The pair, in metres.
///
/// `span` is not a microphone dimension: it is how far `pan = 1` is from the
/// centre line of the string band, i.e. the instrument's own width as the
/// engine's pan axis measures it. It is fitted here because the ITD curve
/// cannot tell an instrument twice as wide from a pair twice as high, and
/// pretending otherwise would put that ambiguity into the spacing instead.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MicGeometry {
    pub spacing_m: f64,
    pub height_m: f64,
    pub span_m: f64,
}

impl MicGeometry {
    pub fn new(spacing_m: f64, height_m: f64, span_m: f64) -> Self {
        Self {
            spacing_m,
            height_m,
            span_m,
        }
    }

    /// Distance from a source at pan `pan` to the left and right capsules.
    pub fn distances(&self, pan: f64) -> (f64, f64) {
        let x = pan.clamp(-1.0, 1.0) * self.span_m;
        let half = 0.5 * self.spacing_m;
        let h2 = self.height_m * self.height_m;
        (
            ((x + half).powi(2) + h2).sqrt(),
            ((x - half).powi(2) + h2).sqrt(),
        )
    }

    /// Interchannel time difference, seconds, **positive when the right capsule
    /// hears the source first** — the sign convention
    /// `realism::StereoBand::lag_ms` reports and the one
    /// `soundboard::Mics::taps` delays with.
    pub fn itd_s(&self, pan: f64) -> f64 {
        let (dl, dr) = self.distances(pan);
        (dl - dr) / SPEED_OF_SOUND
    }

    /// Interchannel level difference under a 1/distance law, dB, left minus
    /// right. Reported for comparison, never fitted; see the module header.
    pub fn ild_db(&self, pan: f64) -> f64 {
        let (dl, dr) = self.distances(pan);
        20.0 * (dr / dl).log10()
    }

    /// The largest time difference this pair can produce, seconds: a source on
    /// the line through both capsules, which is `spacing / c` exactly.
    pub fn max_itd_s(&self) -> f64 {
        self.spacing_m / SPEED_OF_SOUND
    }
}

/// What one item of material contributes to the fit.
#[derive(Clone, Copy, Debug)]
pub struct KeyLag {
    /// Where the source sits on the engine's own pan axis, -1..1.
    pub pan: f64,
    /// Measured interchannel time difference, seconds, right-leads-positive.
    pub lag_s: f64,
    /// Height of the GCC-PHAT peak, 0..1. Used as the fit's weight: a note
    /// whose two channels agree about nothing should not move the geometry.
    pub confidence: f64,
    /// Measured interchannel level difference, dB, left minus right. Reported
    /// beside the fit and not part of it.
    pub ild_db: f64,
}

/// Which part of a note, and which band of it, the delay is read from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LagConfig {
    /// Band the cross-spectrum is whitened and summed over, Hz.
    ///
    /// **The default is the bass, and that is a measurement rather than a
    /// preference.** A delay estimator needs the two channels to be looking at
    /// one wavefront, and on this library they only are down there: over the
    /// thirty recorded keys the PHAT peak is **0.85-1.00 over 40-160 Hz** and
    /// **0.11-0.44 over 200-4000 Hz**, which is the same fact
    /// `realism::stereo_image` reports as a peak |r| of 0.97 in 63-125 Hz
    /// against 0.50-0.62 in every band above it. Above a few hundred hertz an
    /// AB pair 12 cm over the strings is in the near field of an extended
    /// radiator and there is no single delay to find; the search returns the
    /// largest of a field of similar peaks, and over the compass those land
    /// uniformly across the whole ±3 ms window. The tool that drives this
    /// prints both bands so the claim is visible rather than asserted.
    ///
    /// The band also has to be narrow enough at the top that half a period is
    /// wider than the search: 160 Hz is 3.1 ms, so a delay and its alias are
    /// never the same answer inside ±3.
    pub band_hz: (f64, f64),
    /// Seconds of note the delay is read over.
    ///
    /// A whole note rather than an onset window, which is the opposite of what
    /// a direct-path measurement usually wants — and it is the bass that makes
    /// it right. One period of A0's second partial is 18 ms, so an onset window
    /// short enough to exclude the board's later field holds no cycles of the
    /// thing being measured. Measured both ways over the recorded keys, the
    /// answers agree to a tenth of a millisecond wherever the peak is
    /// confident, and the long window is the one whose peak is confident more
    /// often.
    pub window_s: f64,
    /// Widest delay searched, seconds. `MAX_MIC_SPACING_M / c` is 2.9 ms, so
    /// three is every geometry the engine can be asked for and nothing else.
    pub max_lag_s: f64,
}

impl Default for LagConfig {
    fn default() -> Self {
        Self {
            band_hz: (40.0, 160.0),
            window_s: 3.0,
            max_lag_s: 0.003,
        }
    }
}

/// One delay measurement: where the whitened cross-correlation peaks, how high
/// the peak is, and the level difference over the same window.
#[derive(Clone, Copy, Debug)]
pub struct Lag {
    pub lag_s: f64,
    /// Peak of the PHAT correlation, normalised so that a pure delay of a
    /// common signal reads 1 and two independent channels read about
    /// `1/sqrt(bins)`.
    pub confidence: f64,
    pub ild_db: f64,
}

/// GCC-PHAT delay between two channels, over `config`'s band and window.
///
/// The sign is `realism`'s: the correlation is `c[τ] = Σ L[t+τ]·R[t]`, so a
/// **positive** answer means the right channel leads.
pub fn interchannel_lag(
    left: &[f32],
    right: &[f32],
    sample_rate: f64,
    config: &LagConfig,
) -> Result<Lag> {
    let window = ((config.window_s * sample_rate).round() as usize).max(64);
    let take = window.min(left.len()).min(right.len());
    if take < 64 {
        return Err(Error::Config(
            "a delay needs at least a few milliseconds of two channels".into(),
        ));
    }
    let (l, r) = (&left[..take], &right[..take]);
    // Linear, not circular: a padded pair correlates over every lag searched.
    let n = (2 * take).next_power_of_two();
    let mut planner = FftPlanner::<f32>::new();
    let forward = planner.plan_fft_forward(n);
    let mut a: Vec<Complex32> = (0..n)
        .map(|i| Complex32::new(l.get(i).copied().unwrap_or(0.0), 0.0))
        .collect();
    let mut b: Vec<Complex32> = (0..n)
        .map(|i| Complex32::new(r.get(i).copied().unwrap_or(0.0), 0.0))
        .collect();
    forward.process(&mut a);
    forward.process(&mut b);

    let bin = |hz: f64| ((hz * n as f64 / sample_rate).round() as usize).max(1);
    let (lo, hi) = (bin(config.band_hz.0), bin(config.band_hz.1).min(n / 2));
    if hi <= lo {
        return Err(Error::Config("the delay band holds no bins".into()));
    }
    // Phase transform: every bin in the band contributes a unit phasor, so the
    // answer is a vote over frequencies rather than over energies.
    let mut cross = vec![Complex32::new(0.0, 0.0); n];
    let mut bins = 0.0f64;
    for j in lo..=hi {
        let x = a[j] * b[j].conj();
        let m = x.norm();
        if m <= 0.0 {
            continue;
        }
        let unit = x / m;
        cross[j] = unit;
        if n - j != j {
            cross[n - j] = unit.conj();
        }
        bins += 1.0;
    }
    if bins < 8.0 {
        return Err(Error::Config("the delay band holds no signal".into()));
    }
    let inverse = planner.plan_fft_inverse(n);
    inverse.process(&mut cross);
    // Two conjugate halves of `bins` unit phasors sum to `2·bins` at a perfect
    // delay, which is what makes the peak read one there.
    let scale = 1.0 / (2.0 * bins);
    let value = |lag: isize| -> f64 {
        let idx = if lag >= 0 {
            lag as usize
        } else {
            n - (-lag) as usize
        };
        f64::from(cross[idx].re) * scale
    };
    let max_lag = ((config.max_lag_s * sample_rate).round() as isize).max(1);
    let mut best = (0isize, f64::NEG_INFINITY);
    for lag in -max_lag..=max_lag {
        let v = value(lag);
        if v > best.1 {
            best = (lag, v);
        }
    }
    // Sub-sample: at 48 kHz one sample is 7.1 mm of path difference and the
    // geometry is being read to the centimetre.
    let offset = parabolic_offset(value(best.0 - 1), value(best.0), value(best.0 + 1));
    let energy = |c: &[f32]| c.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>();
    let (el, er) = (energy(l), energy(r));
    Ok(Lag {
        lag_s: (best.0 as f64 + offset) / sample_rate,
        confidence: best.1.clamp(0.0, 1.0),
        ild_db: if el > 0.0 && er > 0.0 {
            10.0 * (el / er).log10()
        } else {
            f64::NAN
        },
    })
}

/// What the fit is allowed to move, and where it starts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeometryConfig {
    /// Bounds on the spacing, metres. The engine's own `MAX_MIC_SPACING_M` is
    /// the upper one; the lower keeps the fit off a degenerate coincident pair.
    pub spacing_bounds: (f64, f64),
    /// Bounds on `span / height`, the only shape the ITD curve constrains.
    pub aspect_bounds: (f64, f64),
    /// The height the pair is *asserted* to be at, metres. The fit moves `span`
    /// against it; see the module header on why one of the two has to be held.
    pub height_m: f64,
    /// Weight floor: an item whose PHAT peak is under this is not a delay
    /// measurement and is dropped rather than down-weighted.
    pub confidence_floor: f64,
}

impl Default for GeometryConfig {
    fn default() -> Self {
        Self {
            spacing_bounds: (0.03, 1.0),
            aspect_bounds: (0.2, 12.0),
            height_m: 0.45,
            confidence_floor: 0.05,
        }
    }
}

/// Tukey biweight cut, in robust sigmas.
///
/// The material has real outliers and they are not noise: a key whose two
/// channels happen to cancel at lag zero can put the search on the *other*
/// lobe of the same correlation and return `+0.7 ms` where its neighbours
/// return `-0.6`. On the engine's own renders that happens to about one key in
/// four, and on the recording it happens too (C3 reads -0.10 ms among a run of
/// -0.6). A least-squares fit reads a handful of sign-flipped points as a much
/// wider pair and walks to the bound; 4.685 sigmas is the biweight's own
/// standard tuning, which is 95 % efficient on clean Gaussian scatter and
/// gives a point five sigmas out no vote at all.
const BIWEIGHT_CUT: f64 = 4.685;

/// How much of its **geometric** interchannel delay the engine's own
/// microphone stage shows a delay estimator, on a source far enough off-axis
/// that the geometry has saturated.
///
/// Not a property of this estimator and not an error in it: a property of the
/// mid/side construction `soundboard::Mics` is written as, and the reason the
/// engine's *stated* spacing and the spacing its output behaves like are two
/// different numbers. A real spaced pair gives `L = g_L·x(t-T_L)`,
/// `R = g_R·x(t-T_R)`, and an estimator returns `T_L - T_R`. The engine gives,
/// for a source left of centre,
///
/// ```text
/// L = (m + w·u_L/2)·x(t) - (w·u_R/2)·x(t-δ)
/// R = (m - w·u_L/2)·x(t) + (w·u_R/2)·x(t-δ)
/// ```
///
/// — two *two-tap* channels rather than one delayed one. Past a width of about
/// `2m/u_L` the right channel's second tap is the larger of its two, which
/// makes that channel **maximum-phase**: a maximum-phase two-tap filter carries
/// a further `δ` of group delay of its own, on top of the `δ` the geometry put
/// there. The cross-spectrum's phase slope — which is all a phase-transform
/// estimator measures — is then twice the geometric one, and the readout is
/// `2δ`.
///
/// **1.58, measured, not 2.** Over the eleven bass keys, at three spacings an
/// octave apart (0.12, 0.24, 0.48 m) through the whole instrument, the median
/// readout is 1.455, 1.760 and 1.530 times the geometry's own saturated delay;
/// this is their geometric mean. It falls short of the algebra's 2 because the
/// board's diffuse field arrives with no delay at all and pulls every readout
/// towards zero, and because about one key in four lands on the *other* lobe of
/// its own correlation and reads the opposite sign — both visible in
/// `tuner/tests/mics.rs`'s own printout.
///
/// This is `estimate::directivity::DRIFT_PER_SPREAD_DB`'s pattern exactly: what
/// a parameter does to the engine's output is *measured* on the engine rather
/// than predicted from a model of it, encoded as one constant, and re-checked
/// against the engine by a test that fails when the constant stops being true.
pub const ENGINE_LAG_PER_ITD: f64 = 1.58;

/// The inverted geometry and what the inversion is worth.
#[derive(Clone, Debug)]
pub struct GeometryFit {
    pub geometry: MicGeometry,
    /// Weighted RMS of `measured − modelled` over the items, milliseconds.
    pub residual_ms: f64,
    /// The same for a pair that is not there at all (every ITD zero): the
    /// null model this fit has to beat to have measured anything.
    pub null_ms: f64,
    /// Items that carried a delay worth fitting.
    pub items: usize,
    /// Items the robust pass gave a weight under a tenth: the sign-flipped
    /// ones, counted rather than hidden.
    pub rejected: usize,
    /// How curved the objective is along `span/height` relative to along the
    /// spacing: the factor by which the aspect is worse determined than the
    /// spacing. A large number is not an error — it is the module header's
    /// second bullet, quantified on this material.
    pub conditioning: f64,
    pub converged: bool,
}

fn weighted_rms(items: &[KeyLag], f: impl Fn(&KeyLag) -> f64) -> f64 {
    let (mut num, mut den) = (0.0, 0.0);
    for it in items {
        num += it.confidence * f(it).powi(2);
        den += it.confidence;
    }
    if den <= 0.0 {
        f64::NAN
    } else {
        (num / den).sqrt()
    }
}

/// One least-squares inversion at the weights it is handed: a coarse sweep to
/// pick the basin, then a simplex in log coordinates.
///
/// The sweep is not decoration. The objective is smooth but not convex — a
/// pair twice as wide seen half as far off-axis is a local minimum of the
/// curve's *shape*, distinguishable only by its asymptote — and a simplex
/// dropped anywhere finds whichever basin it started in.
fn invert(items: &[KeyLag], config: &GeometryConfig) -> (f64, f64, bool) {
    let cost = |spacing: f64, aspect: f64| -> f64 {
        let g = MicGeometry::new(spacing, config.height_m, aspect * config.height_m);
        weighted_rms(items, |it| it.lag_s - g.itd_s(it.pan))
    };
    let clamped = |p: &[f64]| -> (f64, f64) {
        (
            p[0].exp()
                .clamp(config.spacing_bounds.0, config.spacing_bounds.1),
            p[1].exp()
                .clamp(config.aspect_bounds.0, config.aspect_bounds.1),
        )
    };
    const SWEEP: usize = 24;
    let mut start = (0.15f64, 1.5f64);
    let mut best = f64::INFINITY;
    for si in 0..SWEEP {
        let spacing = config.spacing_bounds.0
            * (config.spacing_bounds.1 / config.spacing_bounds.0)
                .powf(si as f64 / (SWEEP - 1) as f64);
        for ai in 0..SWEEP {
            let aspect = config.aspect_bounds.0
                * (config.aspect_bounds.1 / config.aspect_bounds.0)
                    .powf(ai as f64 / (SWEEP - 1) as f64);
            let v = cost(spacing, aspect);
            if v < best {
                best = v;
                start = (spacing, aspect);
            }
        }
    }
    let simplex = NelderMead {
        max_evaluations: 600,
        tolerance: 1e-9,
        initial_step: 0.1,
    };
    let minimum = simplex.minimize(&[start.0.ln(), start.1.ln()], |p| {
        let (spacing, aspect) = clamped(p);
        cost(spacing, aspect)
    });
    let (spacing, aspect) = clamped(&minimum.point);
    (spacing, aspect, minimum.converged)
}

/// Inverts a set of per-key delays for the pair that produced them.
///
/// Weighted least squares in the delays themselves, over `(spacing, span/height)`
/// in log coordinates — positive by construction, and one step size meaningful
/// for both — and then **once more with a Tukey biweight** on the first pass's
/// residuals ([`BIWEIGHT_CUT`]). The second pass is not polish: the material
/// carries sign-flipped points, one key in four on the engine's own renders,
/// and a single least-squares pass reads a handful of them as a much wider pair
/// and walks to the bound.
pub fn fit_geometry(items: &[KeyLag], config: &GeometryConfig) -> Result<GeometryFit> {
    let mut used: Vec<KeyLag> = items
        .iter()
        .copied()
        .filter(|it| it.lag_s.is_finite() && it.confidence >= config.confidence_floor)
        .collect();
    if used.len() < 4 {
        return Err(Error::Config(
            "a pair cannot be inverted from fewer than four delays".into(),
        ));
    }
    let items_used = used.len();
    let null_ms = weighted_rms(&used, |it| it.lag_s) * 1e3;

    let (mut spacing, mut aspect, mut converged) = invert(&used, config);
    let mut rejected = 0;
    {
        let g = MicGeometry::new(spacing, config.height_m, aspect * config.height_m);
        let residuals: Vec<f64> = used
            .iter()
            .map(|it| (it.lag_s - g.itd_s(it.pan)).abs())
            .collect();
        let sigma = 1.4826 * crate::numeric::median(&residuals).unwrap_or(0.0);
        if sigma > 0.0 {
            for (it, r) in used.iter_mut().zip(&residuals) {
                let u = r / (BIWEIGHT_CUT * sigma);
                let weight = if u >= 1.0 { 0.0 } else { (1.0 - u * u).powi(2) };
                if weight < 0.1 {
                    rejected += 1;
                }
                it.confidence *= weight;
            }
            if used.iter().map(|it| it.confidence).sum::<f64>() > 0.0 {
                let (s, a, c) = invert(&used, config);
                spacing = s;
                aspect = a;
                converged = c;
            }
        }
    }

    let geometry = MicGeometry::new(spacing, config.height_m, aspect * config.height_m);
    let cost = |spacing: f64, aspect: f64| -> f64 {
        let g = MicGeometry::new(spacing, config.height_m, aspect * config.height_m);
        weighted_rms(&used, |it| it.lag_s - g.itd_s(it.pan))
    };
    // Conditioning: how far each parameter has to move, in relative terms, to
    // double the residual. The ratio of the two is the shape of the valley.
    let base = cost(spacing, aspect).max(1e-12);
    let reach = |along_spacing: bool| -> f64 {
        let mut factor = 1.0;
        for _ in 0..64 {
            factor *= 1.02;
            let v = if along_spacing {
                cost(spacing * factor, aspect)
            } else {
                cost(spacing, aspect * factor)
            };
            if v >= 2.0 * base {
                break;
            }
        }
        factor - 1.0
    };
    let (ds, da) = (reach(true), reach(false));

    // The residual is reported over **every** item, robust weights included in
    // the fit but not in the score: a fit that threw a third of the compass
    // away should have to say so in its own number.
    let all: Vec<KeyLag> = items
        .iter()
        .copied()
        .filter(|it| it.lag_s.is_finite() && it.confidence >= config.confidence_floor)
        .collect();
    Ok(GeometryFit {
        geometry,
        residual_ms: weighted_rms(&all, |it| it.lag_s - geometry.itd_s(it.pan)) * 1e3,
        null_ms,
        items: items_used,
        rejected,
        conditioning: if ds > 0.0 { da / ds } else { f64::INFINITY },
        converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f64 = 48_000.0;

    /// A deterministic broadband source: the only thing a delay estimator can
    /// be tested on, since a periodic one has a delay for every period.
    fn noise(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed | 1;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0
            })
            .collect()
    }

    /// `signal` delayed by `samples`, linearly interpolated, as a channel.
    fn delayed(signal: &[f32], samples: f64) -> Vec<f32> {
        let whole = samples.floor() as isize;
        let frac = (samples - samples.floor()) as f32;
        (0..signal.len())
            .map(|i| {
                let j = i as isize - whole;
                let a = if j >= 0 && (j as usize) < signal.len() {
                    signal[j as usize]
                } else {
                    0.0
                };
                let b = if j > 0 && ((j - 1) as usize) < signal.len() {
                    signal[(j - 1) as usize]
                } else {
                    0.0
                };
                (1.0 - frac) * a + frac * b
            })
            .collect()
    }

    #[test]
    fn a_delay_between_two_channels_is_read_back_with_its_sign() {
        let source = noise(8_192, 7);
        for &samples in &[-31.0f64, -12.5, 0.0, 5.25, 40.0] {
            // `left` delayed by `+samples` means the right channel leads, which
            // is the positive direction of the convention.
            let left = delayed(&source, samples.max(0.0));
            let right = delayed(&source, (-samples).max(0.0));
            let lag = interchannel_lag(
                &left,
                &right,
                SR,
                &LagConfig {
                    // A wide band on white noise: the default is the narrow
                    // bass window the *recording* has to be read in, and 120 Hz
                    // of it resolves a delay to about half a sample.
                    band_hz: (200.0, 8_000.0),
                    window_s: 8_192.0 / SR,
                    ..LagConfig::default()
                },
            )
            .expect("two channels of noise");
            assert!(
                (lag.lag_s * SR - samples).abs() < 0.2,
                "delay of {samples} samples read as {}",
                lag.lag_s * SR
            );
            assert!(lag.confidence > 0.8, "a pure delay is a confident peak");
        }
    }

    #[test]
    fn two_independent_channels_are_not_a_delay() {
        let lag = interchannel_lag(
            &noise(8_192, 1),
            &noise(8_192, 2),
            SR,
            &LagConfig {
                band_hz: (200.0, 8_000.0),
                ..LagConfig::default()
            },
        )
        .expect("two channels");
        assert!(
            lag.confidence < 0.2,
            "independent noise read as a delay at confidence {}",
            lag.confidence
        );
    }

    #[test]
    fn the_time_difference_saturates_at_the_spacing() {
        let g = MicGeometry::new(0.4, 0.5, 1.0);
        assert!((g.itd_s(0.0)).abs() < 1e-12, "a centred source has no ITD");
        assert!(g.itd_s(1.0) > 0.0, "a source to the right leads on the right");
        assert!(g.itd_s(1.0) < g.max_itd_s(), "and never by more than d/c");
        let far = MicGeometry::new(0.4, 0.01, 5.0);
        assert!(
            (far.itd_s(1.0) - far.max_itd_s()).abs() < 1e-4,
            "a source on the capsule line attains the bound"
        );
    }

    #[test]
    fn the_inversion_recovers_a_geometry_it_was_given() {
        let truth = MicGeometry::new(0.32, 0.40, 0.90);
        let items: Vec<KeyLag> = (0..30)
            .map(|i| {
                let pan = -0.6 + 1.2 * i as f64 / 29.0;
                KeyLag {
                    pan,
                    lag_s: truth.itd_s(pan),
                    confidence: 1.0,
                    ild_db: truth.ild_db(pan),
                }
            })
            .collect();
        let fit = fit_geometry(
            &items,
            &GeometryConfig {
                height_m: truth.height_m,
                ..GeometryConfig::default()
            },
        )
        .expect("thirty delays");
        assert!(
            (fit.geometry.spacing_m - truth.spacing_m).abs() < 0.005,
            "spacing came back as {:.3} m",
            fit.geometry.spacing_m
        );
        assert!(
            (fit.geometry.span_m - truth.span_m).abs() < 0.05,
            "span came back as {:.3} m",
            fit.geometry.span_m
        );
        assert!(fit.residual_ms < 0.002, "an exact curve fits exactly");
    }

    #[test]
    fn a_noisy_curve_still_pins_the_spacing() {
        let truth = MicGeometry::new(0.5, 0.45, 1.0);
        let jitter = noise(64, 99);
        let items: Vec<KeyLag> = (0..30)
            .map(|i| {
                let pan = -0.6 + 1.2 * i as f64 / 29.0;
                KeyLag {
                    pan,
                    // A twentieth of a millisecond of scatter — two and a half
                    // samples at 48 kHz, and about what one recorded key's two
                    // velocity layers disagree by.
                    lag_s: truth.itd_s(pan) + 5e-5 * f64::from(jitter[i]),
                    confidence: 1.0,
                    ild_db: 0.0,
                }
            })
            .collect();
        let fit = fit_geometry(&items, &GeometryConfig::default()).expect("thirty delays");
        assert!(
            (fit.geometry.spacing_m - truth.spacing_m).abs() < 0.05,
            "spacing came back as {:.3} m against 0.500",
            fit.geometry.spacing_m
        );
        assert!(
            fit.residual_ms < fit.null_ms,
            "the fit must beat the no-pair null"
        );
    }
}
