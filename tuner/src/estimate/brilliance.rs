//! Brilliance: how much high-frequency energy a render carries against the
//! recording of the same note, and where in time it carries it.
//!
//! `COMPASS.md`'s `centroid` is the power-weighted mean **partial index**. It
//! is register-relative by construction, and the ear's brightness is not: two
//! keys an octave apart with identical `centroid` differ by an octave in the
//! band a listener calls air. Nothing in `renders/` measured absolute
//! frequency until this module, which is why "the recording is slightly more
//! brilliant than the engine" was a listening note with no number under it
//! (`DECISIONS.md` 292).
//!
//! Everything here is a *ratio* against the recording of the same note with
//! the broadband level divided out,
//!
//! ```text
//! HF(band) = 10 log10( E_band(engine) / E_band(reference) )
//!          - 10 log10( E_full(engine) / E_full(reference) )
//! ```
//!
//! because the two signals' levels are 13-22 dB apart across the compass
//! (`COMPASS.md`'s `level e/r`) and any statistic that did not remove that
//! would be measuring `OUTPUT_GAIN`.
//!
//! Two bands, because "brilliant" is two different things. [`HF1`], 2-6 kHz,
//! is where the top of the fitted partial series lives; [`HF2`], 6-12 kHz, is
//! above every fitted mechanism on most of the compass — board, shelf and
//! strike noise alone. Two instants, because a piano's brightness is not a
//! filter: at 0.1 s the strike is still sounding, at 1 s what is left is
//! whatever decayed slowest, and a model can be right at one and wrong at the
//! other for different reasons. [`band_decay_gap`] is the statistic that tells
//! those two apart, and item 294 is what it found.
//!
//! # The tail, and why a crossing time will not do
//!
//! [`fitted_t60`] fits a line through an envelope in dB rather than reading off
//! when it crossed a threshold, for two reasons that both cost a wrong answer.
//! A crossing is what a beat null does — three mistuned strings put 25-47 dB
//! nulls into a bass envelope (`DECISIONS.md` 46) and "the first time it fell
//! 20 dB" then reports the first null. And a recording has a **floor** under
//! it: once the partial is inside the room and the tape, the envelope is the
//! floor's and its slope is zero however long you watch. Refusing to fit under
//! [`FLOOR_MARGIN_DB`] over that floor is what separates a long decay from an
//! unmeasurable one, which is exactly the distinction `DECISIONS.md` 293 turns
//! on.

/// The broadband band a level match is taken over.
pub const FULL: (f64, f64) = (50.0, 20_000.0);
/// The lower brilliance band: the top of the fitted partial series.
pub const HF1: (f64, f64) = (2_000.0, 6_000.0);
/// The upper brilliance band: above every fitted mechanism on most keys.
pub const HF2: (f64, f64) = (6_000.0, 12_000.0);

/// Seconds after the strike from which a signal's own floor is read: what is
/// left when the note is over, which on a recording is the room and the tape.
pub const FLOOR_FROM_S: f64 = 3.0;

/// How far a partial must stand over its own signal's floor before a decay read
/// off it is the partial's decay and not the floor's.
pub const FLOOR_MARGIN_DB: f64 = 10.0;

/// Periods of the band's own frequency in [`narrowband_db`]'s boxcar.
pub const HETERODYNE_CYCLES: f64 = 4.0;

/// Largest correction [`continuation_db`] may put on one partial, dB.
///
/// Twelve because that is the span of the measured rows' own scatter, and
/// because past it a continuation would be asserting a partial the recording
/// never showed it. A band that cannot be reached from inside this cap is a
/// band whose error is not the continuation's to fix.
pub const TRIM_CAP_DB: f64 = 12.0;

/// Smallest share of a band the movable partials must carry before
/// [`trim_gain_db`] will move them at all.
///
/// Under it the solve divides by a number that is mostly rounding: the band is
/// board, strike noise or aliasing rather than string, and no per-partial gain
/// addresses it. Returning zero is the estimator declining a question that is
/// not about partials — which is what it does across the top octave, where the
/// 6-12 kHz excess is not in the partials at all.
pub const MIN_TRIM_SHARE: f64 = 0.02;

fn db(ratio: f64) -> f64 {
    10.0 * ratio.max(1e-30).log10()
}

/// Sums a bin-power spectrum over `[lo, hi)` Hz.
///
/// `power` is the non-negative half of a transform, DC first, so its own length
/// gives the bin spacing and no sample rate has to be passed alongside it.
pub fn band(power: &[f64], sample_rate: f64, (lo, hi): (f64, f64)) -> f64 {
    if power.len() < 2 {
        return 1e-30;
    }
    let spacing = sample_rate / (2.0 * (power.len() - 1) as f64);
    let first = (lo / spacing).ceil() as usize;
    let last = ((hi / spacing).floor() as usize).min(power.len() - 1);
    if first > last {
        return 1e-30;
    }
    power[first..=last].iter().sum::<f64>().max(1e-30)
}

/// The level-matched band ratio: how much more of `bnd` the engine carries than
/// the reference once their broadband levels are equal, in dB.
pub fn hf_ratio(engine: &[f64], reference: &[f64], sample_rate: f64, bnd: (f64, f64)) -> f64 {
    db(band(engine, sample_rate, bnd) / band(reference, sample_rate, bnd))
        - db(band(engine, sample_rate, FULL) / band(reference, sample_rate, FULL))
}

/// How much faster a band dies on one signal than on another, in dB over the
/// interval between the two instants' spectra.
///
/// Nothing is normalised: a band's own drop between two instants is already
/// free of every gain in the chain, so the difference of two drops is free of
/// the level offset, of the master gain, and of the band's own filter. Negative
/// means `engine`'s band dies faster than `reference`'s.
pub fn band_decay_gap(
    engine_early: &[f64],
    engine_late: &[f64],
    reference_early: &[f64],
    reference_late: &[f64],
    sample_rate: f64,
    bnd: (f64, f64),
) -> f64 {
    db(band(engine_late, sample_rate, bnd) / band(engine_early, sample_rate, bnd))
        - db(band(reference_late, sample_rate, bnd) / band(reference_early, sample_rate, bnd))
}

/// The envelope of the band around `hz`, in dB, one point per millisecond.
///
/// A complex heterodyne down to DC followed by a boxcar of
/// [`HETERODYNE_CYCLES`] periods: a band-pass a quarter of `hz` wide, which
/// separates a partial from its own neighbours at every key on the compass and
/// is the same filter on both signals.
pub fn narrowband_db(mono: &[f32], hz: f64, sample_rate: f64) -> Vec<f64> {
    let width = ((sample_rate / hz) * HETERODYNE_CYCLES).round().max(4.0) as usize;
    let (mut re, mut im) = (0.0f64, 0.0f64);
    let mut ring: Vec<(f64, f64)> = vec![(0.0, 0.0); width];
    let step = (sample_rate / 1000.0).max(1.0) as usize;
    let mut out = Vec::with_capacity(mono.len() / step + 1);
    for (n, &x) in mono.iter().enumerate() {
        let phase = -std::f64::consts::TAU * hz * n as f64 / sample_rate;
        let (s, c) = phase.sin_cos();
        let (dr, di) = (f64::from(x) * c, f64::from(x) * s);
        let old = ring[n % width];
        re += dr - old.0;
        im += di - old.1;
        ring[n % width] = (dr, di);
        if n % step == 0 && n >= width {
            out.push(20.0 * ((re * re + im * im).sqrt() / width as f64).max(1e-30).log10());
        }
    }
    out
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// A signal's own floor in the band, in dB under the envelope's peak.
pub fn floor_under_peak(env: &[f64]) -> f64 {
    let peak = env.iter().copied().fold(f64::MIN, f64::max);
    let from = ((FLOOR_FROM_S * 1000.0) as usize).min(env.len());
    peak - median(env[from..].to_vec())
}

/// The band's own T60 in seconds: least squares through the envelope in dB from
/// its peak to the last instant it still stands [`FLOOR_MARGIN_DB`] over the
/// signal's own floor.
///
/// `None` when the note is inside its own floor before there is enough of it to
/// fit — which is not a long decay, it is an unmeasurable one.
pub fn fitted_t60(env: &[f64]) -> Option<f64> {
    let peak = env.iter().copied().fold(f64::MIN, f64::max);
    let from = ((FLOOR_FROM_S * 1000.0) as usize).min(env.len());
    let floor = median(env[from..].to_vec());
    if !floor.is_finite() {
        return None;
    }
    let top = env
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).expect("finite"))
        .map(|(i, _)| i)?;
    let end = env
        .iter()
        .enumerate()
        .skip(top)
        .find(|(_, &v)| v < floor + FLOOR_MARGIN_DB)
        .map_or(env.len(), |(i, _)| i);
    // At least 200 ms of measurable decay and at least 6 dB of it: under either,
    // a slope is an extrapolation.
    if end < top + 200 || peak - (floor + FLOOR_MARGIN_DB) < 6.0 {
        return None;
    }
    let pts = &env[top..end];
    let n = pts.len() as f64;
    let mx = (n - 1.0) / 2_000.0;
    let my = pts.iter().sum::<f64>() / n;
    let (mut num, mut den) = (0.0, 0.0);
    for (i, &y) in pts.iter().enumerate() {
        let dx = i as f64 / 1000.0 - mx;
        num += dx * (y - my);
        den += dx * dx;
    }
    let slope = num / den;
    (slope < -1e-6).then(|| -60.0 / slope)
}

// ---------------------------------------------------------------------------
// The envelope continuation
// ---------------------------------------------------------------------------

/// A band's geometric centre — where [`continuation_db`]'s line is pinned.
pub fn band_centre((lo, hi): (f64, f64)) -> f64 {
    (lo * hi).sqrt()
}

/// Share of a band's measured power carried by the partials **above** `reach`.
///
/// Each partial is counted over `±f0/4`, the width [`narrowband_db`] uses, so
/// no bin is counted for two partials.
pub fn above_reach_share(
    power: &[f64],
    sample_rate: f64,
    partial_hz: &[f64],
    reach: usize,
    bnd: (f64, f64),
) -> f64 {
    let total = band(power, sample_rate, bnd);
    let half = partial_hz.first().copied().unwrap_or(100.0) * 0.25;
    let above: f64 = partial_hz
        .iter()
        .enumerate()
        .filter(|&(i, &hz)| i >= reach && hz >= bnd.0 && hz < bnd.1)
        .map(|(_, &hz)| band(power, sample_rate, (hz - half, hz + half)))
        .sum();
    (above / total).clamp(0.0, 1.0)
}

/// The gain the above-`reach` partials need for the band to land on the
/// recording, in dB.
///
/// Exact rather than "move the band by its error": the band also holds partials
/// the fit already owns, the board, and the strike noise, and moving the whole
/// band by `w` would ask the few partials that *can* move to carry all of it.
/// With `s` the share those partials carry, `P_f + g P_a = W (P_f + P_a)` gives
/// `g = (W - (1 - s)) / s`, which is the correction and its own feasibility
/// test in one: `g <= 0` says the band cannot be brought down by moving those
/// partials at all, however far they are moved.
pub fn trim_gain_db(
    engine: &[f64],
    reference: &[f64],
    sample_rate: f64,
    partial_hz: &[f64],
    reach: usize,
    bnd: (f64, f64),
) -> f64 {
    let s = above_reach_share(engine, sample_rate, partial_hz, reach, bnd);
    if s < MIN_TRIM_SHARE {
        return 0.0;
    }
    let want = 10f64.powf(-hf_ratio(engine, reference, sample_rate, bnd) / 10.0);
    let g = (want - (1.0 - s)) / s;
    if g <= 1e-6 {
        return -TRIM_CAP_DB;
    }
    db(g).clamp(-TRIM_CAP_DB, TRIM_CAP_DB)
}

/// What one partial at `hz` is moved by, given the two bands' corrections.
///
/// A straight line in `ln f` through the two band centres, held flat above the
/// upper one and ramped to zero at the lower band's own bottom edge. Smooth in
/// `ln f` on purpose: `COMPASS.md`'s `irregular` is the mean absolute step
/// between adjacent partials, and a continuation with a step in it would buy
/// brilliance by writing exactly the jaggedness item 284 spent itself removing.
pub fn continuation_db(hz: f64, trim: [f64; 2]) -> f64 {
    let (c1, c2) = (band_centre(HF1), band_centre(HF2));
    if hz <= HF1.0 {
        return 0.0;
    }
    if hz < c1 {
        return trim[0] * (hz / HF1.0).ln() / (c1 / HF1.0).ln();
    }
    if hz >= c2 {
        return trim[1];
    }
    trim[0] + (trim[1] - trim[0]) * (hz / c1).ln() / (c2 / c1).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f64 = 48_000.0;

    /// A flat power spectrum of `bins` bins at `level`.
    fn flat(bins: usize, level: f64) -> Vec<f64> {
        vec![level; bins]
    }

    #[test]
    fn a_pure_level_difference_is_not_a_brilliance_difference() {
        let a = flat(2049, 1.0);
        let b: Vec<f64> = a.iter().map(|x| x * 1_000.0).collect();
        for bnd in [HF1, HF2] {
            assert!(
                hf_ratio(&b, &a, SR, bnd).abs() < 1e-9,
                "a gain moved the ratio in {bnd:?}"
            );
        }
    }

    #[test]
    fn a_tilt_is_read_off_as_the_tilt_it_is() {
        // Six decibels more in 6-12 kHz and nothing else: the ratio has to see
        // the 6 dB in the upper band and (nearly) nothing in the lower, the
        // remainder being what the level match took off for the extra energy.
        let a = flat(2049, 1.0);
        let mut b = a.clone();
        let spacing = SR / (2.0 * 2048.0);
        for (i, v) in b.iter_mut().enumerate() {
            let hz = i as f64 * spacing;
            if (HF2.0..HF2.1).contains(&hz) {
                *v *= 4.0;
            }
        }
        let up = hf_ratio(&b, &a, SR, HF2);
        let flatband = hf_ratio(&b, &a, SR, HF1);
        assert!((up - flatband - 6.02).abs() < 0.05, "tilt read {up} / {flatband}");
    }

    /// One decaying sinusoid, sampled the way a render is.
    fn decaying(hz: f64, t60: f64, seconds: f64) -> Vec<f32> {
        let n = (seconds * SR) as usize;
        (0..n)
            .map(|i| {
                let t = i as f64 / SR;
                let a = 10f64.powf(-3.0 * t / t60);
                (a * (std::f64::consts::TAU * hz * t).sin()) as f32
            })
            .collect()
    }

    #[test]
    fn the_t60_of_a_known_exponential_comes_back() {
        for want in [0.8, 2.0, 5.0] {
            let env = narrowband_db(&decaying(1_000.0, want, 6.0), 1_000.0, SR);
            let got = fitted_t60(&env).expect("a clean exponential is measurable");
            assert!(
                (got / want - 1.0).abs() < 0.05,
                "T60 {got:.3} against {want:.3}"
            );
        }
    }

    #[test]
    fn a_partial_inside_its_own_floor_measures_nothing_rather_than_a_long_decay() {
        // The failure item 293 turns on. A note that is over, sitting on a room
        // 12 dB under its peak: the envelope out there is the room's and flat,
        // and a fit that took it would call a dead note an eternal one. The
        // envelope is written directly rather than synthesised, because what is
        // under test is the refusal and not the band-pass.
        let env: Vec<f64> = (0..6_000)
            .map(|i| {
                let t = i as f64 / 1000.0;
                (-30.0 * t).max(-12.0)
            })
            .collect();
        assert!(
            (floor_under_peak(&env) - 12.0).abs() < 1e-9,
            "floor {}",
            floor_under_peak(&env)
        );
        assert!(
            fitted_t60(&env).is_none(),
            "a decay was fitted under a floor only 12 dB down"
        );
        // The same decay with the room 40 dB down instead is measurable, and
        // comes back at the 2 s it was written with.
        let deep: Vec<f64> = (0..6_000)
            .map(|i| (-30.0 * i as f64 / 1000.0).max(-40.0))
            .collect();
        let got = fitted_t60(&deep).expect("30 dB/s over 40 dB is measurable");
        assert!((got - 2.0).abs() < 0.05, "T60 {got:.3}");
    }

    #[test]
    fn the_band_pass_separates_a_partial_from_its_neighbour() {
        let n = (2.0 * SR) as usize;
        let mixed: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f64 / SR;
                ((std::f64::consts::TAU * 1_000.0 * t).sin()
                    + (std::f64::consts::TAU * 2_000.0 * t).sin()) as f32
            })
            .collect();
        let at_first = narrowband_db(&mixed, 1_000.0, SR);
        let level = at_first[at_first.len() / 2];
        // One unit-amplitude sinusoid heterodyned to DC reads -6 dB (half the
        // analytic amplitude); the neighbour an octave up must not lift it.
        assert!((level + 6.02).abs() < 0.5, "band-pass read {level:.2} dB");
    }

    #[test]
    fn a_band_decay_gap_is_blind_to_every_gain_in_the_chain() {
        let (e0, e1) = (flat(2049, 1.0), flat(2049, 0.01));
        let (r0, r1) = (flat(2049, 5.0), flat(2049, 0.5));
        // The engine drops 20 dB where the reference drops 10.
        let gap = band_decay_gap(&e0, &e1, &r0, &r1, SR, HF1);
        assert!((gap + 10.0).abs() < 1e-9, "gap {gap}");
    }

    /// A spectrum that is zero everywhere except a narrow line at each partial,
    /// with `scale` applied to the lines inside `HF1`.
    fn lines(partial_hz: &[f64], in_band: f64) -> Vec<f64> {
        let spacing = SR / (2.0 * 2048.0);
        (0..2049)
            .map(|i| {
                let hz = i as f64 * spacing;
                if partial_hz.iter().any(|&p| (hz - p).abs() < 100.0) {
                    if (HF1.0..HF1.1).contains(&hz) {
                        in_band
                    } else {
                        1.0
                    }
                } else {
                    0.0
                }
            })
            .collect()
    }

    #[test]
    fn a_band_the_trim_owns_outright_asks_for_the_bands_own_error() {
        // The case where the solve must reduce to the trivial answer: every
        // scrap of the band is a partial above the reach, so `s = 1` and the
        // gain the movable partials need *is* the band's error.
        let partial_hz: Vec<f64> = (1..=8).map(|k| 1_100.0 * k as f64).collect();
        let engine = lines(&partial_hz, 2.0);
        let reference = lines(&partial_hz, 1.0);
        let s = above_reach_share(&engine, SR, &partial_hz, 0, HF1);
        assert!((s - 1.0).abs() < 1e-9, "share {s}");
        let g = trim_gain_db(&engine, &reference, SR, &partial_hz, 0, HF1);
        let error = hf_ratio(&engine, &reference, SR, HF1);
        assert!((g + error).abs() < 0.02, "trim {g} against an error of {error}");
    }

    #[test]
    fn a_band_the_trim_owns_half_of_cannot_be_halved_and_says_so() {
        // Half the band is fitted partials the trim must not touch, and the
        // band is 6 dB too loud against a note whose weight sits elsewhere, so
        // the level match cannot absorb it. Asking the movable half to take all
        // six on its own means taking it past zero, and the solve's own
        // feasibility test is what catches that rather than a silent 60 dB gain.
        let partial_hz: Vec<f64> = (1..=8).map(|k| 1_100.0 * k as f64).collect();
        let spacing = SR / (2.0 * 2048.0);
        let build = |in_band: f64| -> Vec<f64> {
            (0..2049)
                .map(|i| {
                    let hz = i as f64 * spacing;
                    if !partial_hz.iter().any(|&p| (hz - p).abs() < 100.0) {
                        0.0
                    } else if (HF1.0..HF1.1).contains(&hz) {
                        in_band
                    } else {
                        // The note's weight, so the broadband level match has
                        // nothing to absorb the band's error with.
                        100.0
                    }
                })
                .collect()
        };
        let (engine, reference) = (build(4.0), build(1.0));
        let s = above_reach_share(&engine, SR, &partial_hz, 3, HF1);
        assert!((s - 0.5).abs() < 1e-9, "share {s}");
        let error = hf_ratio(&engine, &reference, SR, HF1);
        assert!(error > 5.0, "the band is only {error:.2} dB out");
        assert_eq!(
            trim_gain_db(&engine, &reference, SR, &partial_hz, 3, HF1),
            -TRIM_CAP_DB
        );
    }

    #[test]
    fn the_continuation_leaves_the_fitted_region_alone_and_is_continuous() {
        let trim = [-4.0, 6.0];
        assert_eq!(continuation_db(500.0, trim), 0.0);
        assert_eq!(continuation_db(HF1.0, trim), 0.0);
        assert!((continuation_db(band_centre(HF1), trim) - trim[0]).abs() < 1e-9);
        assert!((continuation_db(band_centre(HF2), trim) - trim[1]).abs() < 1e-9);
        assert!((continuation_db(20_000.0, trim) - trim[1]).abs() < 1e-9);
        // No step anywhere: the whole point is not to write `irregular`.
        let mut previous = 0.0;
        let mut hz = HF1.0;
        while hz < 20_000.0 {
            let next = continuation_db(hz, trim);
            assert!(
                (next - previous).abs() < 0.5,
                "step of {:.2} dB at {hz:.0} Hz",
                next - previous
            );
            previous = next;
            hz *= 1.02;
        }
    }
}
