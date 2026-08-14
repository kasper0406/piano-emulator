//! The damper: how fast the felt actually stops a string, and the
//! `notes.damper_sigma` that reproduces it.
//!
//! `DECISIONS.md` 183 measured the gap and left the fit undone. Measured
//! identically on the recording and on the render, a release tail falls 20 dB in
//! **0.50 s at C3 and 0.60 s at C5** and the engine's in **0.15 s and 0.10 s**:
//! the damper is roughly three to six times too brisk, on every note in the
//! instrument. That is not a missing mechanism — the engine has a damper and it
//! is a decay rate — it is a table that was never fitted against the recordings.
//!
//! # The material
//!
//! Salamander's `harmL*`/`harmS*`/`harmV3*` regions: a struck note, released,
//! recorded from the release onward. `DECISIONS.md` 183 also established what
//! they are made of — **80 % of `harmLC3`'s energy is C3's own partials**,
//! because a damper takes a few tenths of a second to stop a wound string and
//! the recording contains that decay. Read as a coupling target they are
//! misleading; read as a *damper* target they are exactly right, and that is how
//! they are read here.
//!
//! # The fit: invert a line measured on the engine
//!
//! Not a model inversion. The rate a released partial decays at is
//! `sigma_k + damper_sigma * damper_weight(f_k)` in the string, but what a
//! measurement of the *render* sees is that through the damper's own ramp, the
//! bank's cull threshold, the board's diffuse field and the master chain — and
//! the board's 0.4 s reverberation alone puts a floor under how fast anything
//! can stop. So the engine is asked directly: render the same release at two
//! values of `damper_sigma`, measure the tail with the very code the recording
//! was measured with, and invert the line.
//!
//! The line is in the right variable. The measured quantity is a time to fall
//! 20 dB, `T20 = ln(10) / rate`, and `rate` is affine in `damper_sigma` — so
//! **`1/T20` is affine in `damper_sigma`** and two renders determine it. This is
//! `estimate::directivity`'s pattern (`DECISIONS.md` 137–138) on a different
//! parameter: nothing is searched, and what is inverted is a property of the
//! engine as it stands rather than of a model of it.
//!
//! # What is *not* fitted here, and why
//!
//! `voicing.damper_weight` — the anchors that say how much less the felt grips a
//! partial at 6 kHz than one at 500 Hz. It is a shape *within* one note, and
//! what these recordings measure is one number *per* note. [`band_release`]
//! measures the diagnostic that would move it — the tail's decay in a low band
//! and a high band of the same recording — so the decision is taken from data
//! rather than by default; the per-key table carries the compass either way.

use crate::numeric::weighted_least_squares;

/// Decibels the tail is followed over. Twenty is what `DECISIONS.md` 183
/// measured on both sides, and it is as far as a release recording reliably goes
/// before the room takes over.
pub const RELEASE_DROP_DB: f64 = 20.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DamperConfig {
    /// How far the tail is followed, dB.
    pub drop_db: f64,
    /// Block of the RMS envelope the tail is measured on, seconds.
    pub block_s: f64,
    /// Seconds skipped at the head of a release recording before the peak is
    /// looked for. A `harm*` file starts with the key coming up, and the damper
    /// has not landed yet.
    pub skip_s: f64,
    /// Crossover of the two-band diagnostic, as a multiple of the note's own
    /// fundamental.
    pub band_split: f64,
    /// Smallest and largest `damper_sigma` an inversion may return, 1/s. The
    /// floor is a damper that does not damp; the ceiling is the range the base
    /// preset already spans, times four.
    pub min_sigma: f64,
    pub max_sigma: f64,
}

impl Default for DamperConfig {
    fn default() -> Self {
        Self {
            drop_db: RELEASE_DROP_DB,
            block_s: 0.005,
            skip_s: 0.0,
            band_split: 4.0,
            min_sigma: 0.1,
            max_sigma: 400.0,
        }
    }
}

/// Time for a release tail to fall `drop_db` from its own peak, on an RMS
/// envelope.
///
/// `None` where it never gets there — a recording shorter than its own tail has
/// not measured one, and reading the end of the file as the answer would make
/// every long release look identical.
pub fn tail_decay_s(signal: &[f32], sample_rate: f64, config: &DamperConfig) -> Option<f64> {
    if signal.is_empty() || sample_rate <= 0.0 {
        return None;
    }
    let block = ((config.block_s * sample_rate) as usize).max(1);
    let skip = ((config.skip_s.max(0.0) * sample_rate) as usize).min(signal.len());
    let envelope: Vec<f64> = signal[skip..]
        .chunks(block)
        .map(|chunk| (chunk.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>()
            / chunk.len() as f64)
            .sqrt())
        .collect();
    let (peak_block, &peak) = envelope
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))?;
    if peak <= 0.0 {
        return None;
    }
    let target = peak * 10f64.powf(-config.drop_db / 20.0);
    envelope[peak_block..]
        .iter()
        .position(|&a| a <= target)
        .map(|i| i as f64 * block as f64 / sample_rate)
}

/// The same, in a low band and a high band split at `band_split * f0`.
///
/// The diagnostic for `voicing.damper_weight`: the felt grips a low partial
/// harder than a high one, so a tail whose high band outlasts its low band by
/// more than the anchors say is a tail asking for them to move.
pub fn band_release(
    signal: &[f32],
    sample_rate: f64,
    f0_hz: f64,
    config: &DamperConfig,
) -> Option<(f64, f64)> {
    let split = (f0_hz * config.band_split).clamp(20.0, 0.45 * sample_rate);
    let low = one_pole(signal, sample_rate, split, false);
    let high = one_pole(signal, sample_rate, split, true);
    Some((
        tail_decay_s(&low, sample_rate, config)?,
        tail_decay_s(&high, sample_rate, config)?,
    ))
}

/// Two cascaded one-poles: a 12 dB/octave low- or high-pass at `cutoff`.
///
/// Steeper than one, and it has to be. A single pole leaves the fundamental only
/// 6 dB down an octave into the high band, and a fundamental that outlives the
/// band it leaked into decides the band's measured decay instead of the band's
/// own content. Nothing here reads a level, only a time, so the passband ripple
/// a cascade has does not matter and its skirt does.
fn one_pole(signal: &[f32], sample_rate: f64, cutoff: f64, high: bool) -> Vec<f32> {
    let a = (-std::f64::consts::TAU * cutoff / sample_rate).exp();
    let mut out: Vec<f32> = signal.to_vec();
    for _ in 0..2 {
        let mut state = 0.0f64;
        for x in out.iter_mut() {
            state = (1.0 - a) * f64::from(*x) + a * state;
            *x = (if high { f64::from(*x) - state } else { state }) as f32;
        }
    }
    out
}

/// What the engine's own release does at two or more values of
/// `notes.damper_sigma`, for one key.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DamperLine {
    pub key: u8,
    /// `(damper_sigma, T20 of the rendered release)`.
    pub probes: Vec<(f64, f64)>,
}

impl DamperLine {
    /// The `damper_sigma` whose rendered release falls 20 dB in `t20_s`.
    ///
    /// `1/T20` is affine in `damper_sigma`, so this is a line through the
    /// probes solved at the measured time. `None` when the probes do not draw a
    /// line — two renders that came back at the same rate say the parameter did
    /// nothing and there is nothing to invert.
    pub fn sigma_for(&self, t20_s: f64, config: &DamperConfig) -> Option<f64> {
        if !(t20_s.is_finite() && t20_s > 0.0) || self.probes.len() < 2 {
            return None;
        }
        let points: Vec<(f64, f64)> = self
            .probes
            .iter()
            .filter(|&&(sigma, t20)| sigma.is_finite() && t20.is_finite() && t20 > 0.0)
            .map(|&(sigma, t20)| (sigma, 1.0 / t20))
            .collect();
        if points.len() < 2 {
            return None;
        }
        let basis: Vec<f64> = points.iter().flat_map(|&(x, _)| [1.0, x]).collect();
        let y: Vec<f64> = points.iter().map(|&(_, r)| r).collect();
        let solution = weighted_least_squares(&basis, &y, &vec![1.0; points.len()], 2)?;
        let (intercept, slope) = (solution[0], solution[1]);
        if !(slope.is_finite() && slope > 0.0) {
            return None;
        }
        let sigma = (1.0 / t20_s - intercept) / slope;
        Some(sigma.clamp(config.min_sigma, config.max_sigma))
    }

    /// Whether [`DamperLine::sigma_for`] landed on a bound rather than inside
    /// the range: at the floor the instrument's own free decay is already slower
    /// than the recording's tail, and no damper setting can be that slow.
    pub fn saturated(&self, sigma: f64, config: &DamperConfig) -> bool {
        sigma <= config.min_sigma || sigma >= config.max_sigma
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f64 = 48_000.0;

    /// A tail decaying at `rate` nepers per second, at 200 Hz.
    fn tail(rate: f64, seconds: f64) -> Vec<f32> {
        let frames = (seconds * SR) as usize;
        (0..frames)
            .map(|n| {
                let t = n as f64 / SR;
                ((-rate * t).exp() * (std::f64::consts::TAU * 200.0 * t).sin()) as f32
            })
            .collect()
    }

    #[test]
    fn the_tail_measurement_reads_the_rate_that_produced_it() {
        let config = DamperConfig::default();
        for &t20 in &[0.10, 0.50, 0.60] {
            let rate = std::f64::consts::LN_10 / t20;
            let measured = tail_decay_s(&tail(rate, 3.0), SR, &config).expect("a tail");
            assert!(
                (measured / t20 - 1.0).abs() < 0.06,
                "{t20} s asked, {measured} measured"
            );
        }
        // A recording that never falls 20 dB has not measured a tail.
        assert_eq!(tail_decay_s(&tail(0.2, 0.5), SR, &config), None);
    }

    #[test]
    fn the_engines_own_line_inverts_onto_the_recordings_tail() {
        let config = DamperConfig::default();
        // The engine as the model says it behaves: rate = free + weight * sigma,
        // with a free decay of 2 /s and a weight of 0.9.
        let t20_at = |sigma: f64| std::f64::consts::LN_10 / (2.0 + 0.9 * sigma);
        let line = DamperLine {
            key: 48,
            probes: vec![(23.0, t20_at(23.0)), (6.0, t20_at(6.0))],
        };
        // The engine at the shipped value falls 20 dB in 0.10 s; the recording
        // takes 0.50. `DECISIONS.md` 183's gap, and its answer.
        assert!((t20_at(23.0) - 0.104).abs() < 0.005, "{}", t20_at(23.0));
        let sigma = line.sigma_for(0.50, &config).expect("a line");
        assert!((sigma - 2.9).abs() < 0.1, "{sigma}");
        assert!(!line.saturated(sigma, &config));
        // ... and asking for a tail slower than the string's own free decay
        // saturates rather than returning a negative rate.
        let sigma = line.sigma_for(10.0, &config).expect("a line");
        assert_eq!(sigma, config.min_sigma);
        assert!(line.saturated(sigma, &config));
    }

    #[test]
    fn a_line_with_no_slope_in_it_is_refused_rather_than_divided_by() {
        let config = DamperConfig::default();
        let flat = DamperLine {
            key: 60,
            probes: vec![(5.0, 0.2), (25.0, 0.2)],
        };
        assert_eq!(flat.sigma_for(0.5, &config), None);
        let single = DamperLine {
            key: 60,
            probes: vec![(5.0, 0.2)],
        };
        assert_eq!(single.sigma_for(0.5, &config), None);
    }

    #[test]
    fn the_two_band_diagnostic_separates_a_bright_tail_from_a_dull_one() {
        let config = DamperConfig::default();
        // Two components: 100 Hz falling slowly, 2 kHz falling fast — a damper
        // that grips the high partials harder than the low ones. The crossover
        // is at 4 f0 = 400 Hz, two octaves above the low one and three below the
        // high one, so neither band is deciding the other's answer.
        let frames = (3.0 * SR) as usize;
        let signal: Vec<f32> = (0..frames)
            .map(|n| {
                let t = n as f64 / SR;
                ((-4.0 * t).exp() * (std::f64::consts::TAU * 100.0 * t).sin()
                    + (-20.0 * t).exp() * (std::f64::consts::TAU * 2_000.0 * t).sin())
                    as f32
            })
            .collect();
        let (low, high) = band_release(&signal, SR, 100.0, &config).expect("both bands");
        assert!((low - std::f64::consts::LN_10 / 4.0).abs() < 0.05, "low {low} s");
        assert!((high - std::f64::consts::LN_10 / 20.0).abs() < 0.05, "high {high} s");
    }
}
