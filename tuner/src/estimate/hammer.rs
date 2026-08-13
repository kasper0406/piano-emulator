//! Felt hammer parameters, and the velocity map of a sample library, from the
//! excitation spectra of one note played at every velocity layer.
//!
//! # What is being inverted
//!
//! The collision between hammer and string is a two-state ODE — the felt is a
//! nonlinear spring `F = K c^p`, the hammer is a mass, and the string end is a
//! resistance `2 Z n` with the agraffe's reflection arriving back after
//! `t_ref`. Integrating it forward gives a force pulse a millisecond or two
//! long, and the magnitude spectrum of that pulse is the excitation each
//! partial receives. So the measured time-zero partial amplitudes, divided by
//! the strike comb `sin(k pi x)`, *are* the felt model's pulse spectrum, up to
//! one constant that turns newtons into recorded amplitude.
//!
//! The pulse is reimplemented here in `f64` rather than borrowed from the
//! engine, deliberately: an estimator has to invert exactly the forward model
//! its output will be played through, so this is a copy of `engine/src/hammer.rs`'s
//! contact integration with the same semi-implicit step, the same
//! Hunt-Crossley hysteresis and the same reflection loop-gain limit. If that
//! model changes, this must follow it.
//!
//! # What is identifiable, and what is not
//!
//! A layer's spectrum is a *shape* and a *level*. With the layer velocities
//! unknown — the whole point of fitting them — the level of each layer is used
//! up by its own velocity, so what constrains the felt is how the shapes vary
//! from layer to layer. Working through the scalings of the contact equations:
//! the pulse duration is `m / 2Z` times a dimensionless function of `p` and the
//! scaled velocity, so the **mass** is fixed by how long the pulses are, and
//! the **exponent** by how the shape moves as the level does. The **stiffness**
//! only ever appears multiplied by the unknown newtons-to-amplitude gain — it
//! is identifiable exactly when that gain is known. [`HammerConfig::gain`]
//! therefore takes it as a given when the recording chain is calibrated (which
//! it is when the "recording" is our own engine's render), and fits it
//! otherwise, in which case `K` should be read as "`K` for this gain".
//!
//! # The velocity map
//!
//! A sample library's layers are an unknown monotone function of hammer speed.
//! Each layer's velocity is fitted freely and the sequence is then projected
//! onto the monotone cone by isotonic regression, which is exact rather than
//! penalized: the fit can put two layers at the same speed, but never in the
//! wrong order.

use crate::error::{Error, Result};
use crate::estimate::decay::DecayReport;
use crate::estimate::strike::StrikeFit;
use crate::numeric::{golden_section, isotonic, weighted_least_squares, NelderMead};

/// The felt: the three per-note numbers a preset stores.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FeltParams {
    /// Hammer head mass, kg.
    pub mass: f64,
    /// Felt stiffness `K`, N/m^p.
    pub stiffness: f64,
    /// Felt nonlinearity exponent `p`.
    pub exponent: f64,
}

/// Everything about the collision that is known before the fit starts: the
/// string the hammer meets and the felt constants shared by the whole compass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContactConfig {
    /// Transverse wave impedance of one string, kg/s.
    pub impedance: f64,
    /// Strings the hammer meets at once. They load it in parallel.
    pub strings: f64,
    /// Round trip from the strike point to the agraffe and back, seconds.
    pub reflection_seconds: f64,
    /// Velocity reflection coefficient of the agraffe.
    pub reflection_gain: f64,
    /// Hunt-Crossley hysteresis coefficient, s/m.
    pub hysteresis: f64,
    pub sample_rate: f64,
    /// Integration substeps per audio sample. The felt spring is the fastest
    /// thing in the instrument and needs several steps per cycle.
    pub oversample: usize,
    /// Longest contact the integration will follow, seconds.
    pub max_contact_s: f64,
}

impl Default for ContactConfig {
    fn default() -> Self {
        Self {
            impedance: 2.2,
            strings: 3.0,
            reflection_seconds: 0.115 / 261.6256,
            reflection_gain: 0.85,
            hysteresis: 0.15,
            sample_rate: 48_000.0,
            oversample: 8,
            max_contact_s: 0.02,
        }
    }
}

impl ContactConfig {
    /// The contact a note's own geometry implies. The wave reaches the agraffe
    /// in `x L / c` and `L / c` is `1 / (2 f0)`, so the round trip is
    /// `x / f0` — the same identity the engine builds its hammer with.
    pub fn for_note(&self, f0_hz: f64, strike_position: f64, strings: f64, impedance: f64) -> Self {
        Self {
            impedance,
            strings,
            reflection_seconds: strike_position / f0_hz,
            ..*self
        }
    }
}

/// A hammer/string contact force, sampled at the integrator's substep rate.
#[derive(Clone, Debug)]
pub struct ForcePulse {
    pub dt: f64,
    /// Total force on the string group, newtons.
    pub force: Vec<f64>,
}

impl ForcePulse {
    pub fn duration_s(&self) -> f64 {
        self.force.len() as f64 * self.dt
    }

    /// The impulse the hammer delivered, N s.
    pub fn impulse(&self) -> f64 {
        self.force.iter().sum::<f64>() * self.dt
    }

    pub fn peak(&self) -> f64 {
        self.force.iter().copied().fold(0.0, f64::max)
    }

    /// Magnitude of the pulse's Fourier transform at `hz`, N s.
    ///
    /// Evaluated directly rather than through an FFT: the estimator needs it at
    /// a few dozen partial frequencies, which are not on any convenient grid,
    /// and a direct sum over a pulse this short costs less than the transform
    /// it would replace. The sum advances a unit phasor by one complex multiply
    /// per sample instead of calling a transcendental — the fit evaluates this
    /// tens of thousands of times — and renormalizes it often enough that the
    /// accumulated drift stays far below the measurement it is compared with.
    pub fn magnitude_at(&self, hz: f64) -> f64 {
        const RENORMALIZE: usize = 256;
        let omega = -2.0 * std::f64::consts::PI * hz * self.dt;
        let (step_re, step_im) = (omega.cos(), omega.sin());
        let (mut phase_re, mut phase_im) = (1.0f64, 0.0f64);
        let (mut re, mut im) = (0.0, 0.0);
        for (i, &f) in self.force.iter().enumerate() {
            re += f * phase_re;
            im += f * phase_im;
            let next_re = phase_re * step_re - phase_im * step_im;
            phase_im = phase_re * step_im + phase_im * step_re;
            phase_re = next_re;
            if i % RENORMALIZE == RENORMALIZE - 1 {
                let scale = (phase_re * phase_re + phase_im * phase_im).sqrt();
                phase_re /= scale;
                phase_im /= scale;
            }
        }
        (re * re + im * im).sqrt() * self.dt
    }
}

/// Integrates one strike into its force pulse.
pub fn contact_pulse(felt: &FeltParams, velocity: f64, contact: &ContactConfig) -> ForcePulse {
    let dt = 1.0 / (contact.sample_rate * contact.oversample as f64);
    let steps = (contact.max_contact_s / dt).round() as usize;
    let mut force = Vec::with_capacity(steps.min(1 << 16));
    if velocity <= 0.0 || felt.mass <= 0.0 || felt.stiffness <= 0.0 {
        return ForcePulse { dt, force };
    }
    let (m, k, p) = (felt.mass, felt.stiffness, felt.exponent);
    let two_z = 2.0 * contact.impedance * contact.strings;

    // Deepest compression this strike can reach, from the energy balance
    // `(1/2) m v^2 = K c^(p+1) / (p+1)`, and the felt stiffness there. The
    // reflection is a delayed positive feedback path of gain
    // `k_felt t_ref / 2Z`; past unity the lossless lumped model diverges, so
    // the surplus is handed to the spring the reflection is equivalent to once
    // several round trips have passed.
    let c_max = ((p + 1.0) * m * velocity * velocity / (2.0 * k)).powf(1.0 / (p + 1.0));
    let loop_gain = p * k * c_max.powf(p - 1.0) * contact.reflection_seconds / two_z;
    let carried = (1.0 / loop_gain).min(1.0);
    let delayed = contact.reflection_gain * carried;
    let string_stiffness = (1.0 - carried) * two_z / contact.reflection_seconds;

    let history_len = ((contact.reflection_seconds / dt).round() as usize).clamp(1, steps.max(1));
    let mut history = vec![0.0f64; history_len];
    let mut read = 0usize;

    let (mut x, mut y, mut v) = (0.0f64, 0.0f64, velocity);
    let mut compression_rate = v;
    let mut touched = false;
    for _ in 0..steps {
        let c = x - y;
        let f = if c > 0.0 {
            let hysteresis = (1.0 + contact.hysteresis * compression_rate).clamp(0.0, 2.0);
            k * c.powf(p) * hysteresis
        } else {
            0.0
        };
        touched |= f > 0.0;
        force.push(f);

        let string_reaction = delayed * history[read] + string_stiffness * y;
        history[read] = f;
        read = (read + 1) % history_len;

        // Semi-implicit in the contact point: the felt's local stiffness
        // `dF/dc = p F / c` passes 1e6 N/m at a hard treble strike, and solving
        // the step for the contact velocity is what keeps the integration
        // bounded without paying for ten times the substeps.
        let felt_stiffness = if c > 0.0 { p * f / c } else { 0.0 };
        let contact_velocity = (f - string_reaction) / (two_z + felt_stiffness * dt);
        compression_rate = v - contact_velocity;
        v -= f / m * dt;
        x += v * dt;
        y += contact_velocity * dt;
        if touched && x - y <= 0.0 {
            break;
        }
    }
    ForcePulse { dt, force }
}

/// One point of an excitation spectrum: what the hammer put into one partial.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpectrumPoint {
    pub frequency_hz: f64,
    /// Time-zero amplitude with the strike comb divided out.
    pub amplitude: f64,
    /// Inverse-variance weight of this point's log-amplitude.
    pub weight: f64,
}

/// One velocity layer's excitation spectrum.
#[derive(Clone, Debug)]
pub struct LayerSpectrum {
    /// Index of the layer in the source library, ascending with loudness.
    pub layer: u8,
    pub points: Vec<SpectrumPoint>,
}

/// How much to trust a comb-corrected amplitude.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpectrumWeighting {
    /// Partials whose comb factor is below this are dropped outright: dividing
    /// by a number that small amplifies the measurement's error by more than
    /// the correction is worth.
    pub min_comb: f64,
    /// Relative accuracy of a tracked, decay-corrected, back-extrapolated
    /// amplitude — the noise floor of the measurement itself, before the comb.
    pub amplitude_accuracy: f64,
    /// Relative uncertainty of the fitted strike position. It is what makes the
    /// partials near a comb null untrustworthy even when they clear
    /// `min_comb`: `d ln|sin(k pi x)| / d ln x` is `k pi x cot(k pi x)`, which
    /// runs away at the nulls, so a strike point known to a percent leaves a
    /// partial sitting on the shoulder of a null known to tens of percent.
    pub position_uncertainty: f64,
}

impl Default for SpectrumWeighting {
    fn default() -> Self {
        Self {
            min_comb: 0.1,
            amplitude_accuracy: 0.02,
            position_uncertainty: 0.01,
        }
    }
}

impl LayerSpectrum {
    /// Builds a layer's spectrum from its fitted decays and the note's strike
    /// position: the time-zero amplitude of each partial, divided by the comb,
    /// weighted by how much that division can be trusted.
    pub fn from_decays(
        layer: u8,
        decays: &DecayReport,
        strike: &StrikeFit,
        weighting: &SpectrumWeighting,
    ) -> Self {
        let x = strike.position;
        let points = decays
            .partials
            .iter()
            .filter(|fit| fit.frequency_hz > 0.0 && fit.initial_amplitude() > 0.0)
            .filter_map(|fit| {
                let comb = strike.comb_at(fit.k);
                if comb < weighting.min_comb {
                    return None;
                }
                let angle = f64::from(fit.k) * std::f64::consts::PI * x;
                let sensitivity = (angle / angle.tan()).abs() * weighting.position_uncertainty;
                let variance = weighting.amplitude_accuracy.powi(2) + sensitivity * sensitivity;
                Some(SpectrumPoint {
                    frequency_hz: fit.frequency_hz,
                    amplitude: fit.initial_amplitude() / comb,
                    weight: 1.0 / variance.max(1e-12),
                })
            })
            .collect();
        Self { layer, points }
    }

    /// A spectrum whose points are all equally trusted — what a caller that has
    /// already corrected and vetted its amplitudes hands in.
    pub fn uniform(layer: u8, points: &[(f64, f64)]) -> Self {
        Self {
            layer,
            points: points
                .iter()
                .map(|&(frequency_hz, amplitude)| SpectrumPoint {
                    frequency_hz,
                    amplitude,
                    weight: 1.0,
                })
                .collect(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HammerConfig {
    pub contact: ContactConfig,
    /// Bounds on hammer speed, m/s. A real hammer runs from a whisper at
    /// ~0.2 m/s to ~6 m/s fortissimo.
    pub min_velocity: f64,
    pub max_velocity: f64,
    /// Newtons-to-amplitude calibration of the recording chain. `Some` when it
    /// is known — the felt stiffness is only identifiable then; `None` fits it
    /// alongside, and `K` is then meaningful only together with the fitted
    /// gain.
    pub gain: Option<f64>,
    /// Passes of "velocities, then felt". Two is usually enough; the extra
    /// rounds cost milliseconds.
    pub rounds: usize,
    /// Golden-section iterations per layer velocity.
    pub velocity_iterations: usize,
    /// Simplex budget for the felt parameters, per round.
    pub max_evaluations: usize,
    /// Whether the mass is fitted or held at its starting value. Held is the
    /// right choice when the compass curve for mass is trusted and the
    /// recording is short of high partials to date the pulse with.
    pub fit_mass: bool,
}

impl Default for HammerConfig {
    fn default() -> Self {
        Self {
            contact: ContactConfig::default(),
            min_velocity: 0.1,
            max_velocity: 8.0,
            gain: None,
            rounds: 2,
            velocity_iterations: 16,
            max_evaluations: 300,
            fit_mass: true,
        }
    }
}

/// One measured amplitude as the fit sees it: log domain, with its weight.
#[derive(Clone, Copy, Debug)]
struct Observation {
    frequency_hz: f64,
    log_amplitude: f64,
    weight: f64,
}

#[derive(Clone, Debug)]
pub struct HammerFit {
    pub felt: FeltParams,
    /// Hammer speed fitted for each layer, m/s, nondecreasing and in the order
    /// the layers were given.
    pub velocities: Vec<f64>,
    /// The newtons-to-amplitude constant, fitted or as configured.
    pub gain: f64,
    /// RMS of the fit residual over every partial of every layer, in dB.
    pub residual_db: f64,
}

/// Fits the felt parameters and the layer velocities jointly.
pub fn fit_hammer(
    layers: &[LayerSpectrum],
    start: &FeltParams,
    config: &HammerConfig,
) -> Result<HammerFit> {
    if layers.is_empty() || layers.iter().all(|layer| layer.points.is_empty()) {
        return Err(Error::Estimate("hammer fit got no spectra".into()));
    }
    if config.max_velocity <= config.min_velocity {
        return Err(Error::Config("hammer velocity range is empty".into()));
    }
    let measured: Vec<Vec<Observation>> = layers
        .iter()
        .map(|layer| {
            layer
                .points
                .iter()
                .filter(|point| {
                    point.frequency_hz > 0.0 && point.amplitude > 0.0 && point.weight > 0.0
                })
                .map(|point| Observation {
                    frequency_hz: point.frequency_hz,
                    log_amplitude: point.amplitude.ln(),
                    weight: point.weight,
                })
                .collect()
        })
        .collect();
    if measured.iter().all(|points| points.is_empty()) {
        return Err(Error::Estimate(
            "hammer fit got no usable partial amplitudes".into(),
        ));
    }

    // Start the layers spread geometrically over the velocity range: no
    // information, but in the right order and inside the bounds.
    let count = layers.len();
    let mut velocities: Vec<f64> = (0..count)
        .map(|i| {
            let u = if count > 1 {
                i as f64 / (count - 1) as f64
            } else {
                0.5
            };
            config.min_velocity * (config.max_velocity / config.min_velocity).powf(u)
        })
        .collect();
    let mut felt = *start;
    let mut gain = config
        .gain
        .unwrap_or_else(|| optimal_gain(&measured, &felt, &velocities, config));

    // The velocities are fitted *inside* the felt's objective, not alternately
    // with it. Alternating converges to whatever pairing of felt and speeds it
    // starts near — a stiffer felt struck more gently makes nearly the same
    // level, and only the shapes tell them apart — so the felt has to be judged
    // by the residual it leaves *after* the speeds have been given their best
    // chance, which is what variable projection means.
    for _ in 0..config.rounds.max(1) {
        let mut start_point = vec![felt.stiffness.ln(), felt.exponent];
        if config.fit_mass {
            start_point.push(felt.mass.ln());
        }
        let solver = NelderMead {
            max_evaluations: config.max_evaluations,
            tolerance: 1e-9,
            initial_step: 0.2,
        };
        let minimum = solver.minimize(&start_point, |p| {
            let Some(candidate) = felt_from(p, &felt, config) else {
                return f64::MAX;
            };
            let speeds = fit_velocities(&measured, &candidate, gain, config);
            total_residual(&measured, &candidate, &speeds, gain, config).0
        });
        if let Some(candidate) = felt_from(&minimum.point, &felt, config) {
            felt = candidate;
        }
        velocities = fit_velocities(&measured, &felt, gain, config);
        gain = config
            .gain
            .unwrap_or_else(|| optimal_gain(&measured, &felt, &velocities, config));
    }

    let (mean_square, mass) = total_residual(&measured, &felt, &velocities, gain, config);
    if mass <= 0.0 {
        return Err(Error::Estimate("hammer fit matched no partials".into()));
    }
    Ok(HammerFit {
        felt,
        velocities,
        gain,
        residual_db: 8.685_889_638_065_035 * mean_square.sqrt(),
    })
}

/// The speed each layer was struck at, given the felt: an independent search
/// per layer, then the monotone projection that makes them a velocity map.
fn fit_velocities(
    measured: &[Vec<Observation>],
    felt: &FeltParams,
    gain: f64,
    config: &HammerConfig,
) -> Vec<f64> {
    let logs: Vec<f64> = measured
        .iter()
        .map(|points| {
            if points.is_empty() {
                return f64::NAN;
            }
            golden_section(
                config.min_velocity.ln(),
                config.max_velocity.ln(),
                config.velocity_iterations,
                |log_v| layer_residual(points, felt, log_v.exp(), gain, config).0,
            )
            .0
        })
        .collect();
    // Layers are a monotone function of hammer speed by construction. Imposing
    // it as a projection rather than as a penalty means the fit can put two
    // layers at the same speed but never in the wrong order, and costs nothing
    // when the free fit is already monotone.
    let weights: Vec<f64> = measured
        .iter()
        .map(|points| points.iter().map(|p| p.weight).sum::<f64>().max(f64::MIN_POSITIVE))
        .collect();
    let filled: Vec<f64> = fill_gaps(&logs);
    isotonic(&filled, &weights)
        .iter()
        .map(|log_v| log_v.exp())
        .collect()
}

/// Replaces the entries of layers that had no partials to fit with their
/// neighbours', so the monotone projection sees a full sequence.
fn fill_gaps(logs: &[f64]) -> Vec<f64> {
    let mut filled = logs.to_vec();
    let mut last = logs.iter().copied().find(|v| v.is_finite()).unwrap_or(0.0);
    for value in filled.iter_mut() {
        if value.is_finite() {
            last = *value;
        } else {
            *value = last;
        }
    }
    filled
}

/// Unpacks the simplex's parameters, keeping the felt physical: a stiffness is
/// positive and an exponent between 1 and 6 (felt measurements put it at
/// 2 to 3.5; outside that range the contact model stops describing felt).
fn felt_from(point: &[f64], base: &FeltParams, config: &HammerConfig) -> Option<FeltParams> {
    let stiffness = point[0].exp();
    let exponent = point[1];
    let mass = if config.fit_mass {
        point[2].exp()
    } else {
        base.mass
    };
    if !(stiffness.is_finite() && stiffness > 0.0 && mass.is_finite() && mass > 0.0) {
        return None;
    }
    if !(1.0..=6.0).contains(&exponent) {
        return None;
    }
    Some(FeltParams {
        mass,
        stiffness,
        exponent,
    })
}

/// The gain that minimizes the log-domain residual with everything else fixed:
/// the mean log ratio between measurement and model. Profiling it out this way
/// keeps it off the simplex, where it would only be a slow way of computing an
/// average.
fn optimal_gain(
    measured: &[Vec<Observation>],
    felt: &FeltParams,
    velocities: &[f64],
    config: &HammerConfig,
) -> f64 {
    let (mut sum, mut mass) = (0.0, 0.0);
    for (points, &velocity) in measured.iter().zip(velocities) {
        if points.is_empty() {
            continue;
        }
        let pulse = contact_pulse(felt, velocity, &config.contact);
        for point in points {
            let model = pulse.magnitude_at(point.frequency_hz);
            if model > 0.0 {
                sum += point.weight * (point.log_amplitude - model.ln());
                mass += point.weight;
            }
        }
    }
    if mass <= 0.0 {
        1.0
    } else {
        (sum / mass).exp()
    }
}

/// Weighted mean-square log residual of one layer, and the total weight it was
/// taken over.
fn layer_residual(
    points: &[Observation],
    felt: &FeltParams,
    velocity: f64,
    gain: f64,
    config: &HammerConfig,
) -> (f64, f64) {
    let pulse = contact_pulse(felt, velocity, &config.contact);
    let log_gain = gain.ln();
    let (mut sum, mut mass) = (0.0, 0.0);
    for point in points {
        let model = pulse.magnitude_at(point.frequency_hz);
        if model <= 0.0 || !model.is_finite() {
            return (f64::MAX, 0.0);
        }
        let error = point.log_amplitude - (log_gain + model.ln());
        sum += point.weight * error * error;
        mass += point.weight;
    }
    if mass <= 0.0 {
        (f64::MAX, 0.0)
    } else {
        (sum / mass, mass)
    }
}

fn total_residual(
    measured: &[Vec<Observation>],
    felt: &FeltParams,
    velocities: &[f64],
    gain: f64,
    config: &HammerConfig,
) -> (f64, f64) {
    let (mut sum, mut mass) = (0.0, 0.0);
    for (points, &velocity) in measured.iter().zip(velocities) {
        if points.is_empty() {
            continue;
        }
        let (mean_square, layer_mass) = layer_residual(points, felt, velocity, gain, config);
        if layer_mass <= 0.0 {
            return (f64::MAX, 0.0);
        }
        sum += mean_square * layer_mass;
        mass += layer_mass;
    }
    if mass <= 0.0 {
        (f64::MAX, 0.0)
    } else {
        (sum / mass, mass)
    }
}

/// The engine's MIDI-velocity-to-hammer-speed map: exponential between
/// `velocity_min` at MIDI 1 and `velocity_max` at MIDI 127.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VelocityMap {
    pub velocity_min: f64,
    pub velocity_max: f64,
    /// RMS residual of the fitted layer speeds, in the log domain.
    pub residual: f64,
}

impl VelocityMap {
    pub fn velocity_at(&self, midi: u8) -> f64 {
        let v = f64::from(midi.clamp(1, 127));
        self.velocity_min * (self.velocity_max / self.velocity_min).powf((v - 1.0) / 126.0)
    }
}

/// Fits the engine's two-point exponential velocity map to the speeds measured
/// for a library's layers, given where each layer sits in MIDI velocity.
///
/// The fit is a straight line in `ln v` against MIDI velocity, which is what
/// "exponential map" means; the two endpoints are read off the fitted line
/// rather than off the extreme layers, so a library whose loudest layer is not
/// at 127 still extrapolates the way the engine will.
pub fn fit_velocity_map(layers: &[(u8, f64)]) -> Result<VelocityMap> {
    let usable: Vec<(f64, f64)> = layers
        .iter()
        .filter(|&&(_, v)| v > 0.0 && v.is_finite())
        .map(|&(midi, v)| ((f64::from(midi.clamp(1, 127)) - 1.0) / 126.0, v.ln()))
        .collect();
    if usable.len() < 2 {
        return Err(Error::Estimate(
            "a velocity map needs two layers at different velocities".into(),
        ));
    }
    let basis: Vec<f64> = usable.iter().flat_map(|&(x, _)| [1.0, x]).collect();
    let y: Vec<f64> = usable.iter().map(|&(_, y)| y).collect();
    let weights = vec![1.0; usable.len()];
    let solution = weighted_least_squares(&basis, &y, &weights, 2)
        .ok_or_else(|| Error::Estimate("velocity map fit is singular".into()))?;
    let residual = (usable
        .iter()
        .map(|&(x, y)| (y - (solution[0] + solution[1] * x)).powi(2))
        .sum::<f64>()
        / usable.len() as f64)
        .sqrt();
    Ok(VelocityMap {
        velocity_min: solution[0].exp(),
        velocity_max: (solution[0] + solution[1]).exp(),
        residual,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c4_contact() -> ContactConfig {
        ContactConfig::default()
    }

    fn c4_felt() -> FeltParams {
        // The default preset's C4 hammer, to within the precision it is written
        // with: 6 g, p = 2.65, K from the reference-compression derivation.
        FeltParams {
            mass: 0.0062,
            stiffness: 4.3e9,
            exponent: 2.65,
        }
    }

    /// The spectra a known hammer would produce at a set of speeds, sampled at
    /// the partials of a C4-like string.
    fn layers_from(felt: &FeltParams, velocities: &[f64], contact: &ContactConfig) -> Vec<LayerSpectrum> {
        velocities
            .iter()
            .enumerate()
            .map(|(index, &velocity)| {
                let pulse = contact_pulse(felt, velocity, contact);
                let points: Vec<(f64, f64)> = (1..=24)
                    .map(|k| {
                        let hz = 261.6256 * f64::from(k) * (1.0 + 4e-4 * f64::from(k * k)).sqrt();
                        (hz, pulse.magnitude_at(hz))
                    })
                    .collect();
                LayerSpectrum::uniform(index as u8, &points)
            })
            .collect()
    }

    #[test]
    fn the_contact_pulse_behaves_like_a_hammer() {
        let contact = c4_contact();
        let felt = c4_felt();
        let soft = contact_pulse(&felt, 0.5, &contact);
        let hard = contact_pulse(&felt, 5.0, &contact);
        // Contact lasts a millisecond or two, and a harder strike is shorter
        // and stronger: the felt stiffens as it is compressed.
        assert!(
            (0.4e-3..4.0e-3).contains(&soft.duration_s()),
            "{} s",
            soft.duration_s()
        );
        assert!(hard.duration_s() < soft.duration_s(), "{hard:?}");
        assert!(hard.peak() > 10.0 * soft.peak());
        // Momentum in, momentum out: the impulse cannot exceed what the hammer
        // carried, and a lively strike returns a good part of it.
        let momentum = felt.mass * 5.0;
        assert!(hard.impulse() > momentum && hard.impulse() < 2.0 * momentum, "{}", hard.impulse());
        // The spectrum is low-pass: a 1 ms pulse cannot excite 10 kHz as
        // strongly as it excites 200 Hz.
        assert!(hard.magnitude_at(10_000.0) < 0.5 * hard.magnitude_at(200.0));
    }

    #[test]
    fn a_harder_strike_has_a_brighter_spectrum() {
        // The one qualitative fact the whole velocity fit rests on: loudness
        // and brightness are not independent, so a layer's spectrum tells the
        // fit which speed produced it.
        let contact = c4_contact();
        let felt = c4_felt();
        let ratio = |velocity: f64| {
            let pulse = contact_pulse(&felt, velocity, &contact);
            pulse.magnitude_at(3_000.0) / pulse.magnitude_at(262.0)
        };
        assert!(ratio(4.0) > 2.0 * ratio(0.5), "{} vs {}", ratio(4.0), ratio(0.5));
    }

    #[test]
    fn the_felt_and_the_layer_velocities_come_back_out_of_their_own_spectra() {
        let contact = c4_contact();
        let truth = c4_felt();
        let velocities = [0.4, 0.7, 1.2, 2.0, 3.2, 5.0];
        let layers = layers_from(&truth, &velocities, &contact);
        let config = HammerConfig {
            contact,
            // The engine's own render is the calibrated case: newtons to
            // amplitude is 1, so the stiffness is identifiable.
            gain: Some(1.0),
            rounds: 2,
            ..HammerConfig::default()
        };
        // Start 40 % off in mass and stiffness and 15 % off in exponent.
        let start = FeltParams {
            mass: truth.mass * 1.4,
            stiffness: truth.stiffness * 0.6,
            exponent: truth.exponent * 0.85,
        };
        let fit = fit_hammer(&layers, &start, &config).unwrap();
        assert!(
            (fit.felt.mass / truth.mass - 1.0).abs() < 0.1,
            "mass {} vs {}",
            fit.felt.mass,
            truth.mass
        );
        assert!(
            (fit.felt.exponent / truth.exponent - 1.0).abs() < 0.1,
            "p {} vs {}",
            fit.felt.exponent,
            truth.exponent
        );
        assert!(
            (fit.felt.stiffness / truth.stiffness - 1.0).abs() < 0.1,
            "K {:e} vs {:e}",
            fit.felt.stiffness,
            truth.stiffness
        );
        for (fitted, &truth) in fit.velocities.iter().zip(&velocities) {
            assert!(
                (fitted / truth - 1.0).abs() < 0.1,
                "velocities {:?} vs {velocities:?}",
                fit.velocities
            );
        }
        assert!(fit.residual_db < 1.0, "{fit:?}");
    }

    #[test]
    fn the_fitted_velocity_map_is_monotone_even_when_the_data_is_not() {
        // Two layers recorded in the wrong order, as a mislabelled library
        // would give: the fit may not reproduce the inversion.
        let contact = c4_contact();
        let truth = c4_felt();
        let mut layers = layers_from(&truth, &[0.5, 1.0, 2.0, 4.0], &contact);
        layers.swap(1, 2);
        for (index, layer) in layers.iter_mut().enumerate() {
            layer.layer = index as u8;
        }
        let fit = fit_hammer(
            &layers,
            &truth,
            &HammerConfig {
                contact,
                gain: Some(1.0),
                ..HammerConfig::default()
            },
        )
        .unwrap();
        assert!(
            fit.velocities.windows(2).all(|w| w[0] <= w[1]),
            "{:?}",
            fit.velocities
        );
    }

    #[test]
    fn the_layer_to_midi_map_is_an_exponential_through_the_layers() {
        let truth = VelocityMap {
            velocity_min: 0.2,
            velocity_max: 6.0,
            residual: 0.0,
        };
        let layers: Vec<(u8, f64)> = (0..16)
            .map(|i| {
                let midi = 8 + i * 8;
                (midi as u8, truth.velocity_at(midi as u8))
            })
            .collect();
        let fit = fit_velocity_map(&layers).unwrap();
        assert!((fit.velocity_min / truth.velocity_min - 1.0).abs() < 1e-9, "{fit:?}");
        assert!((fit.velocity_max / truth.velocity_max - 1.0).abs() < 1e-9, "{fit:?}");
        assert!(fit.residual < 1e-12);
    }
}
