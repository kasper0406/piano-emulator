//! The unsampled keys' texture, drawn from the sampled keys' distributions:
//! `notes.partial_gains` and `notes.false_beat` where there is no recording to
//! fit — the 58 keys the library never sampled plus the two it sampled and
//! could measure nothing from.
//!
//! # The defect this exists for
//!
//! `DECISIONS.md` 274 measured the seam a listener had already found. A fitted
//! key renders **1.65 dB** from its own recording's `irregular` — the mean
//! absolute step between adjacent partial levels — and an unfitted one renders
//! **5.78 dB under** it, because 58 of the 88 keys carry no per-partial row at
//! all and every other mechanism in the engine that shapes a spectrum is smooth
//! in `ln k`. So the compass alternates: three keys the piano's own roughness,
//! then two keys of a synthesizer. The listener reported it as C4 "sounding
//! different from the rest of the melody", and C4 is the *fitted* key — the
//! seam is the defect and the fitted note is the correct one.
//!
//! # What may and may not be invented
//!
//! Item 274 closed with "the 58 unfitted keys are not extended and cannot be",
//! and that is right about the thing it is about: **no value here may be a
//! measurement of a key nobody recorded**. What this module does instead is
//! draw from the *distributions* the 28 fitted keys measured — an amount of
//! roughness, a correlation between neighbouring partials, a coverage, a
//! number of splits, a rate, a depth — which are statements about the
//! instrument and not about any key. Three rules make that a synthesis rather
//! than an interpolation, and each is enforced here rather than intended:
//!
//! * **No neighbour's row is copied, ever.** Every cell is a draw; nothing is
//!   read from another key's table. A row that came from the key three
//!   semitones down would put the *same* comb on two keys, and the recording
//!   says the roughness is not shared between notes at the same frequency
//!   (`TUNING_REPORT.md` §3) — that measurement is what refused a global bridge
//!   curve, and it refuses interpolation here for the same reason.
//! * **What is drawn is a statistic, conditioned on register and nothing
//!   else.** Where a distribution turned out to have no register dependence
//!   worth the name — the roughness amount, r = −0.11 over 28 keys — the model
//!   says so and draws from one distribution for the whole compass, rather than
//!   fitting a curve to noise.
//! * **The draw is seeded from the key number**, so a preset re-emitted from
//!   the same distributions is the same preset, and a render is reproducible.
//!
//! # What is *not* synthesized, and the measurement that refuses it
//!
//! A fitted row is a **tilt** plus a **roughness**: a degree-2 polynomial in
//! `ln k`, which is the engine's own error in its smooth envelope
//! (`DECISIONS.md` 231's 7.5 dB at C4, the octave-displaced attack), and a
//! per-partial scatter about it. Only the roughness is drawn here. The tilt is
//! a per-key **colour**, `compass_scan`'s `centroid` reads it directly, and
//! over the 28 fitted keys it is white across the compass: its value at `k = 1`
//! has a lag-1 autocorrelation of **+0.45** between keys three semitones apart,
//! at `k = 2` **+0.06** and at `k = 4` **−0.02**, and a degree-5 polynomial in
//! key explains 39 %, 25 % and 26 % of it. So it cannot be interpolated — the
//! same decomposition that refused `notes.bridge_gain` the removed level in
//! item 282 — and drawing it would write 6.6 dB of standard deviation of
//! invented colour into 60 keys, which is the one thing a `centroid` family
//! that is still open (item 280) must not be given. The seam item 274 measured
//! is a **roughness** seam, and that is what this closes.
//!
//! That refusal is also why a drawn row has a minimum length
//! ([`MIN_DRAWN_CELLS`]). A quadratic in `ln k` goes *through* three points, so
//! a three-cell row has no roughness in it at all — it is entirely its own
//! tilt — and drawing one would be drawing a colour by another route. The
//! fitted three-cell rows are kept because they are measurements of something;
//! a drawn one would not be.
//!
//! # The discipline the drawn row is written under is the fitted row's
//!
//! Items 272 and 273, unchanged and in the same order: the row is railed to its
//! **own** spread ([`ShapingConfig::rail_sigmas`], the same rule as
//! `shaping::rail_cells` and the same implementation, [`own_rail`]), then
//! pinned on **power** against the engine's own rendered spectrum
//! ([`shaping::energy_offset`]), and then closed on the **render** by
//! `fit_motion`'s own `close_on_the_render` — the same function, the same
//! roughness ceiling and the same level band the fitted keys pass through. A
//! drawn row that renders rougher than the recordings of its register is
//! trimmed exactly as a measured one is.
//!
//! # And so are the splits, which is what item 300 fixed
//!
//! A `notes.false_beat` row's `db` is a **request** — how far under the partial
//! a second component sits — and nothing in a preset says how deeply the engine
//! will then beat: that depends on the key's own unison, its damping and the
//! rate the companion sits at. A *fitted* key therefore never writes the number
//! its recording implied; [`FalseBeatLoop`] bisects the request until the
//! **rendered** depth is the recording's. Item 284 drew the request from
//! [`TextureModel::depth`] and wrote it, which is the one drawn quantity that
//! was never closed on the render, and both boards found it: E3 rendered `beat`
//! **10.13 dB against its own recording's 3.3** (`DECISIONS.md` 289) and F4 took
//! the melody's `wobble` column to 2.64 dB against the piano's 1.98 (item 298).
//! The ask's own scatter is 10.55 dB at an R² of 0.251, so writing it unclosed
//! writes three quarters of a residual straight into the instrument.
//!
//! So the drawn ask is now closed exactly as the drawn row is: against a
//! **ceiling** ([`BeatCeiling`] — the recordings' own beat depth by register and
//! partial, drawn per key with its own scatter), on the **render**, by
//! `fit_motion`'s `close_splits_on_the_render`. A drawn split that renders
//! deeper than the piano's own partials of that register is bisected down until
//! it does not; one that renders shallower is left alone, because the draw is a
//! draw and not a target and forcing every drawn key up onto its ceiling would
//! put the unsampled keys systematically *beatier* than the sampled ones — the
//! same seam with its sign reversed; and one whose partial already beats over
//! the ceiling with **no** split at all is thrown away, because a companion only
//! adds and a draw that cannot be brought under the ceiling is not evidence of
//! anything.

use crate::estimate::motion::FalseBeatLoop;
use crate::estimate::shaping::{own_rail, ShapingConfig, MAX_ROW_CELLS};
use crate::preset::{
    index_to_key, key_index, FalseBeat, MAX_FALSE_BEATS_PER_KEY, MAX_FALSE_BEAT_DB,
    MAX_FALSE_BEAT_HZ, MIN_FALSE_BEAT_DB,
};

/// Decibels per neper.
const NEPERS_TO_DB: f64 = 8.685_889_638_065_035;

/// The named constant every draw is seeded from, with the key number.
///
/// A synthesized instrument has to be reproducible or it cannot be scored: two
/// runs of the fit must write the same preset, and a render of key 61 must be
/// the same render tomorrow. Seeding from the key alone would tie the draws of
/// two different *presets* together; seeding from a clock would make the
/// instrument unrepeatable. So the seed is this constant mixed with the key,
/// and the constant is written down rather than derived, which is what makes a
/// deliberate re-draw — a different instrument from the same distributions — a
/// one-line change with a name on it.
///
/// The digits are `piano` in ASCII followed by `DECISIONS.md` 284's number.
pub const TEXTURE_SEED: u64 = 0x_7069_616e_6f00_011c;

/// Lowest rate a fitted false beat could have been measured at, Hz — the
/// solver's own [`FalseBeatLoop::MIN_FITTED_HZ`], one cycle in the record a
/// rate is counted over.
///
/// The fitted rows bottom out at exactly this, 0.370 Hz, so it is the lower
/// edge of the distribution being drawn from, and drawing under it would be
/// drawing outside what was measured rather than inside it. (The *schema*
/// allows 0.2, which is a statement about what a file may say.)
pub const MIN_FITTED_HZ: f64 = FalseBeatLoop::MIN_FITTED_HZ;

/// Highest key that carries a false beat at all.
///
/// `DECISIONS.md` 233's falsification refuses the treble: a rate that tracks
/// the partial number is a *tuning* beat and not a wire's, and every fitted key
/// above 93 came back `ScalesWithPartial` or with nothing in range (96, 99 and
/// 102 all write no rows). Above this key the mechanism is not that the
/// measurement failed, it is that the measurement said no, and a draw must say
/// no too.
pub const HIGHEST_FALSE_BEAT_KEY: u8 = 94;

/// A quantity fitted as `exp(a + b·key)`, with the scatter its own keys had
/// about that line in the log domain.
///
/// Log domain because all three of the quantities this is used for — a count of
/// cells, an amount of roughness in dB, a number of splits — are positive and
/// scale rather than shift: the distance from 2 dB to 4 is the distance from 8
/// to 16, not the distance from 8 to 10.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LogLine {
    pub intercept: f64,
    pub slope: f64,
    /// Robust scatter about the line, in the log domain.
    pub sigma: f64,
    /// Pearson `r` of the fit, reported so a slope that is noise can be read as
    /// noise rather than believed.
    pub correlation: f64,
    pub points: usize,
}

impl LogLine {
    /// Least squares on `(key, ln value)`, with the residual's standard
    /// deviation.
    pub fn fit(points: &[(f64, f64)]) -> LogLine {
        let usable: Vec<(f64, f64)> = points
            .iter()
            .filter(|&&(_, v)| v > 0.0 && v.is_finite())
            .map(|&(k, v)| (k, v.ln()))
            .collect();
        let n = usable.len() as f64;
        if usable.len() < 3 {
            let mean = if usable.is_empty() {
                0.0
            } else {
                usable.iter().map(|p| p.1).sum::<f64>() / n
            };
            return LogLine {
                intercept: mean,
                points: usable.len(),
                ..LogLine::default()
            };
        }
        let (mx, my) = (
            usable.iter().map(|p| p.0).sum::<f64>() / n,
            usable.iter().map(|p| p.1).sum::<f64>() / n,
        );
        let sxx: f64 = usable.iter().map(|p| (p.0 - mx).powi(2)).sum();
        let sxy: f64 = usable.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
        let syy: f64 = usable.iter().map(|p| (p.1 - my).powi(2)).sum();
        let slope = if sxx > 0.0 { sxy / sxx } else { 0.0 };
        let intercept = my - slope * mx;
        let residual: f64 = usable
            .iter()
            .map(|p| (p.1 - (intercept + slope * p.0)).powi(2))
            .sum::<f64>()
            / n;
        LogLine {
            intercept,
            slope,
            sigma: residual.sqrt(),
            correlation: if sxx > 0.0 && syy > 0.0 {
                sxy / (sxx * syy).sqrt()
            } else {
                0.0
            },
            points: usable.len(),
        }
    }

    pub fn at(&self, key: u8) -> f64 {
        (self.intercept + self.slope * f64::from(key)).exp()
    }

    /// The line's value at `key` times one draw of its own scatter.
    pub fn draw(&self, key: u8, rng: &mut Draw) -> f64 {
        (self.intercept + self.slope * f64::from(key) + self.sigma * rng.normal()).exp()
    }
}

/// A quantity fitted as `exp(a + b·key + c·key²)`: the same idea as
/// [`LogLine`] where the compass really does bend.
///
/// One quantity needs it — the recording's own `irregular`, which is flat at
/// 6-10 dB from A0 to C5 and then climbs to 45 by F#7 as a note's partials thin
/// out toward Nyquist. Fitted over the 28 sampled keys a straight line in `ln`
/// explains 52 % of it and a quadratic **88 %**, and the residual scatter falls
/// from ×1.51 to **×1.23**.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LogCurve {
    pub coefficients: [f64; 3],
    pub sigma: f64,
    pub r_squared: f64,
    pub points: usize,
}

impl LogCurve {
    pub fn fit(points: &[(f64, f64)]) -> LogCurve {
        let usable: Vec<(f64, f64)> = points
            .iter()
            .filter(|&&(_, v)| v > 0.0 && v.is_finite())
            .map(|&(k, v)| (k, v.ln()))
            .collect();
        if usable.len() < 4 {
            let line = LogLine::fit(points);
            return LogCurve {
                coefficients: [line.intercept, line.slope, 0.0],
                sigma: line.sigma,
                points: usable.len(),
                ..LogCurve::default()
            };
        }
        // Normal equations of a quadratic in the key, centred so the third
        // moment does not lose its digits: the abscissa spans 21..108 and its
        // fourth power spans 1.4e8.
        let n = usable.len() as f64;
        let centre = usable.iter().map(|p| p.0).sum::<f64>() / n;
        let x: Vec<f64> = usable.iter().map(|p| p.0 - centre).collect();
        let y: Vec<f64> = usable.iter().map(|p| p.1).collect();
        let moment = |p: u32| x.iter().map(|v| v.powi(p as i32)).sum::<f64>();
        let cross = |p: u32| {
            x.iter()
                .zip(&y)
                .map(|(v, &yy)| v.powi(p as i32) * yy)
                .sum::<f64>()
        };
        let a = [
            [n, moment(1), moment(2)],
            [moment(1), moment(2), moment(3)],
            [moment(2), moment(3), moment(4)],
        ];
        let b = [cross(0), cross(1), cross(2)];
        let Some(c) = solve3(a, b) else {
            let line = LogLine::fit(points);
            return LogCurve {
                coefficients: [line.intercept, line.slope, 0.0],
                sigma: line.sigma,
                points: usable.len(),
                ..LogCurve::default()
            };
        };
        // Back to uncentred coefficients, so `at` is a plain polynomial in the
        // key and the numbers in a report are readable as such.
        let coefficients = [
            c[0] - c[1] * centre + c[2] * centre * centre,
            c[1] - 2.0 * c[2] * centre,
            c[2],
        ];
        let mean = y.iter().sum::<f64>() / n;
        let (mut rss, mut tss) = (0.0, 0.0);
        for (p, &yy) in usable.iter().zip(&y) {
            let at = coefficients[0] + coefficients[1] * p.0 + coefficients[2] * p.0 * p.0;
            rss += (yy - at).powi(2);
            tss += (yy - mean).powi(2);
        }
        LogCurve {
            coefficients,
            sigma: (rss / n).sqrt(),
            r_squared: if tss > 0.0 { 1.0 - rss / tss } else { 0.0 },
            points: usable.len(),
        }
    }

    pub fn at(&self, key: u8) -> f64 {
        let k = f64::from(key);
        (self.coefficients[0] + self.coefficients[1] * k + self.coefficients[2] * k * k).exp()
    }

    pub fn draw(&self, key: u8, rng: &mut Draw) -> f64 {
        self.at(key) * (self.sigma * rng.normal()).exp()
    }
}

/// The recordings' own **beat depth**, per key and per partial:
/// `ln db = a + b·key + c·k`.
///
/// This is the ceiling a *drawn* split is closed on, and it is the one thing
/// item 284 drew without closing (`DECISIONS.md` 289, 298). A fitted key's split
/// depth is bisected by [`FalseBeatLoop`] until the **rendered** beat depth is
/// the recording's own at that partial; a drawn key has no recording, and taking
/// the depth straight from [`TextureModel::depth`] — the fitted *asks*, whose
/// scatter is 10.55 dB against an R² of 0.251 — put E3's rendered beat at
/// 10.13 dB against its recording's 3.3 and F4's melody wobble at 2.64 dB
/// against the piano's 1.98. So the drawn ask is kept as an ask and the
/// **render** is held under what the piano's own partials of this register do.
///
/// Two terms and not three: a beat depth is a positive quantity that scales
/// (the distance from 2 dB to 4 is the distance from 8 to 16), the register
/// term is the only structure the compass shows in it, and the partial term is
/// carried because [`TextureModel::depth`]'s own fit found one and because the
/// piano's upper partials beat deeper than its fundamentals. Both sub-fits are
/// quoted in [`BeatCeiling::r_squared_key_only`] and
/// [`BeatCeiling::r_squared_partial_only`] so that a term which is noise can be
/// read as noise rather than believed.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BeatCeiling {
    /// `[a, b, c]` of `ln db = a + b·key + c·k`.
    pub coefficients: [f64; 3],
    /// Robust scatter about the surface, in the log domain, and the two halves
    /// it decomposes into: how much of it is one **key** beating more than its
    /// register (`sigma_key`, the scatter of a key's own median residual) and
    /// how much is one **partial** of a key beating more than the rest of that
    /// key (`sigma_cell`, the scatter about it).
    ///
    /// Both are drawn, and the split is the reason: a piano's beat depth is not
    /// a property of the key alone — the same wire beats 3 dB at one partial and
    /// 12 at another — so a ceiling drawn once per key would put every decibel
    /// of the scatter between keys and none inside them, and would then refuse a
    /// whole key's splits on one unlucky number. Measured on the shipped preset
    /// the two are **×1.99 between keys and ×1.70 within one**, which is a real
    /// key-to-key term and a within-key term of nearly its size, and neither is
    /// small enough to drop.
    pub sigma: f64,
    pub sigma_key: f64,
    pub sigma_cell: f64,
    pub r_squared: f64,
    /// The same fit with the partial term dropped, and with the register term
    /// dropped.
    pub r_squared_key_only: f64,
    pub r_squared_partial_only: f64,
    pub points: usize,
    pub keys: usize,
}

impl BeatCeiling {
    /// Least squares on `(key, k, ln depth)`.
    ///
    /// Depths at or under zero are dropped rather than floored: a partial whose
    /// envelope does not beat has no depth to take a logarithm of, and putting
    /// a floor on it would be inventing the quietest reading in the set.
    pub fn fit(points: &[(u8, u16, f64)]) -> BeatCeiling {
        let usable: Vec<(f64, f64, f64)> = points
            .iter()
            .filter(|&&(_, _, db)| db > 0.0 && db.is_finite())
            .map(|&(key, k, db)| (f64::from(key), f64::from(k), db.ln()))
            .collect();
        let mut keys: Vec<u8> = points
            .iter()
            .filter(|&&(_, _, db)| db > 0.0 && db.is_finite())
            .map(|&(key, _, _)| key)
            .collect();
        keys.sort_unstable();
        keys.dedup();
        let n = usable.len() as f64;
        if usable.len() < 4 {
            let mean = if usable.is_empty() {
                0.0
            } else {
                usable.iter().map(|p| p.2).sum::<f64>() / n
            };
            return BeatCeiling {
                coefficients: [mean, 0.0, 0.0],
                points: usable.len(),
                keys: keys.len(),
                ..BeatCeiling::default()
            };
        }
        let dot = |f: &dyn Fn(&(f64, f64, f64)) -> f64, g: &dyn Fn(&(f64, f64, f64)) -> f64| {
            usable.iter().map(|p| f(p) * g(p)).sum::<f64>()
        };
        let one = |_: &(f64, f64, f64)| 1.0;
        let key = |p: &(f64, f64, f64)| p.0;
        let part = |p: &(f64, f64, f64)| p.1;
        let y = |p: &(f64, f64, f64)| p.2;
        let a = [
            [dot(&one, &one), dot(&one, &key), dot(&one, &part)],
            [dot(&key, &one), dot(&key, &key), dot(&key, &part)],
            [dot(&part, &one), dot(&part, &key), dot(&part, &part)],
        ];
        let b = [dot(&one, &y), dot(&key, &y), dot(&part, &y)];
        let Some(coefficients) = solve3(a, b) else {
            let mean = usable.iter().map(|p| p.2).sum::<f64>() / n;
            return BeatCeiling {
                coefficients: [mean, 0.0, 0.0],
                points: usable.len(),
                keys: keys.len(),
                ..BeatCeiling::default()
            };
        };
        let mean = usable.iter().map(|p| p.2).sum::<f64>() / n;
        let tss: f64 = usable.iter().map(|p| (p.2 - mean).powi(2)).sum();
        let residual: Vec<f64> = usable
            .iter()
            .map(|p| p.2 - (coefficients[0] + coefficients[1] * p.0 + coefficients[2] * p.1))
            .collect();
        let rss: f64 = residual.iter().map(|v| v * v).sum();
        let r2_of = |f: &dyn Fn(&(f64, f64, f64)) -> f64| -> f64 {
            let line = LogLine::fit(
                &usable
                    .iter()
                    .map(|p| (f(p), p.2.exp()))
                    .collect::<Vec<_>>(),
            );
            let rss: f64 = usable
                .iter()
                .map(|p| (p.2 - (line.intercept + line.slope * f(p))).powi(2))
                .sum();
            if tss > 0.0 {
                1.0 - rss / tss
            } else {
                0.0
            }
        };
        // The two halves of the scatter: a key's own median residual against
        // the surface, and each cell's residual against its key's median.
        let mut key_medians: Vec<(f64, f64)> = Vec::new();
        let mut within: Vec<f64> = Vec::new();
        for &k in &keys {
            let mine: Vec<f64> = usable
                .iter()
                .zip(&residual)
                .filter(|(p, _)| (p.0 - f64::from(k)).abs() < 0.5)
                .map(|(_, &r)| r)
                .collect();
            if mine.is_empty() {
                continue;
            }
            let centre = median_of(&mine);
            key_medians.push((f64::from(k), centre));
            within.extend(mine.iter().map(|r| r - centre));
        }
        BeatCeiling {
            coefficients,
            // Robust, not the plain standard deviation: the recordings' own
            // depths carry a handful of readings taken at a null, and a ceiling
            // is exactly the statistic those would inflate.
            sigma: robust_sigma(&residual),
            sigma_key: robust_sigma(&key_medians.iter().map(|p| p.1).collect::<Vec<_>>()),
            sigma_cell: robust_sigma(&within),
            r_squared: if tss > 0.0 { 1.0 - rss / tss } else { 0.0 },
            r_squared_key_only: r2_of(&key),
            r_squared_partial_only: r2_of(&part),
            points: usable.len(),
            keys: keys.len(),
        }
    }

    /// The surface's own value at one cell, dB.
    pub fn at(&self, key: u8, k: u16) -> f64 {
        (self.coefficients[0]
            + self.coefficients[1] * f64::from(key)
            + self.coefficients[2] * f64::from(k))
        .exp()
    }

    /// The ceiling for one cell: `at(key, k) · exp(sigma_key · z_key +
    /// sigma_cell · z_cell)`.
    ///
    /// Two draws and not one. `z_key` is the key's own — a wire that beats more
    /// than its register does so at every partial — and `z_cell` is the
    /// partial's, because the recordings' own scatter is mostly *inside* a key
    /// (see [`BeatCeiling::sigma_cell`]). Drawing only the first would refuse a
    /// whole key's splits on one unlucky number; drawing only the second would
    /// give every key the same average wire.
    pub fn draw(&self, key: u8, k: u16, z_key: f64, z_cell: f64) -> f64 {
        self.at(key, k) * (self.sigma_key * z_key + self.sigma_cell * z_cell).exp()
    }
}

fn solve3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> Option<[f64; 3]> {
    for i in 0..3 {
        let pivot = (i..3).max_by(|&p, &q| a[p][i].abs().total_cmp(&a[q][i].abs()))?;
        a.swap(i, pivot);
        b.swap(i, pivot);
        if a[i][i].abs() < 1e-12 {
            return None;
        }
        for j in i + 1..3 {
            let f = a[j][i] / a[i][i];
            for k in i..3 {
                a[j][k] -= f * a[i][k];
            }
            b[j] -= f * b[i];
        }
    }
    let mut out = [0.0; 3];
    for i in (0..3).rev() {
        let mut s = b[i];
        for (k, &solved) in out.iter().enumerate().skip(i + 1) {
            s -= a[i][k] * solved;
        }
        out[i] = s / a[i][i];
    }
    Some(out)
}

/// The 28 fitted keys' distributions, as one object.
///
/// Every field is fitted by [`fit_texture`] from a preset that already carries
/// the measured rows; nothing here is a constant somebody chose, and the two
/// that *are* constants — [`TEXTURE_SEED`] and [`HIGHEST_FALSE_BEAT_KEY`] — are
/// a seed and a verdict rather than a parameter.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TextureModel {
    /// How many cells a row has: `exp(a + b·key)`, capped by the key's own
    /// partial count. This is the *recording's* reach up the series and not the
    /// bank's — at C4 the engine has 80 partials and the tracker measured 16.
    pub cells: LogLine,
    /// The roughness amount: `1.4826 MAD` of the row's cells about their own
    /// degree-2 curve in `ln k`, in dB — the **roughness** half of a fitted row
    /// and not the whole of it, because the tilt half is not drawn (see this
    /// module's header) and an amount that included it would put a colour's
    /// worth of spread into a row that has no colour.
    ///
    /// Fitted only from rows of at least [`MIN_AMOUNT_CELLS`] cells: under that
    /// a degree-2 curve has as many parameters as the row has evidence, its
    /// residual is a fit artefact rather than a roughness, and the three-cell
    /// treble rows have a detrended MAD of exactly zero.
    pub amount: LogLine,
    /// Lag-1 autocorrelation of a row's cells about its own smooth tilt, pooled
    /// over the rows long enough to have one.
    pub rho: f64,
    pub rho_rows: usize,
    /// The recording's own `irregular` by register — the ceiling a drawn row is
    /// trimmed against, exactly as a fitted row is trimmed against its own
    /// recording's.
    pub target: LogCurve,
    /// Fraction of keys at or under [`HIGHEST_FALSE_BEAT_KEY`] whose wire
    /// measured any split at all.
    pub false_beat_probability: f64,
    /// How many splits such a key has: `exp(a + b·key)`, at least one.
    pub false_beat_count: LogLine,
    /// Relative frequency with which partial `k` (1-based) carries one of a
    /// key's splits.
    pub partial_weights: [f64; MAX_FALSE_BEATS_PER_KEY],
    /// The rate, `ln hz`: one distribution for the whole compass, because the
    /// measurement says the rate knows neither the partial number nor the key.
    pub rate_ln_mean: f64,
    pub rate_ln_sigma: f64,
    /// Correlation between the partial number and the rate, within a key, and
    /// between the key and the rate: both reported because both being zero is
    /// the property the draw has to preserve.
    pub rate_vs_partial: f64,
    pub rate_vs_key: f64,
    /// The depth **asked for**: `db = a + b·key + c·k`, with the residual's own
    /// scatter. This is the companion level the fitted rows carry, which is a
    /// request to the engine and not a measurement of the piano — 75 % of it is
    /// scatter, and a fitted key's request was bisected against its own
    /// recording's rendered depth. It seeds the draw; what closes it is
    /// [`TextureModel::beat_ceiling`].
    pub depth: [f64; 3],
    pub depth_sigma: f64,
    pub depth_r_squared: f64,
    /// The depth the recordings' own partials **have**, by register and partial
    /// number: the ceiling the drawn asks are closed on, on the render.
    pub beat_ceiling: BeatCeiling,
    /// Keys the model was fitted from.
    pub fitted_keys: Vec<u8>,
    pub false_beat_rows: usize,
}

/// One key's synthesized texture, before the render has had its say.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SynthesizedTexture {
    pub key: u8,
    /// The row, in dB, median-centred: `cells[i]` is partial `i + 1`.
    pub gains_db: Vec<f64>,
    /// The amount that was drawn, in dB, before the rails.
    pub drawn_amount_db: f64,
    /// The amount the row has after the rails, which is what it was written
    /// with.
    pub railed_amount_db: f64,
    pub railed_cells: usize,
    pub rail_db: f64,
    /// The `irregular` ceiling drawn for this key from the recordings' own
    /// register curve.
    pub target_irregular: f64,
    pub false_beat: Vec<FalseBeat>,
    /// The **rendered** beat depth each of those splits may not exceed, dB,
    /// parallel to `false_beat` and drawn from
    /// [`TextureModel::beat_ceiling`] — the recordings' own depth at this
    /// register and partial, with one draw of this key's own scatter.
    ///
    /// The row's `db` is an ask and this is what the ask is closed against on
    /// the engine, which is the whole of `DECISIONS.md` 300: nothing in a
    /// preset says how deeply a written companion will actually beat, and the
    /// fitted keys learn it by rendering.
    pub beat_ceiling_db: Vec<f64>,
    /// `RSS(rate = s k) / RSS(rate = c)` on the drawn rates: the same
    /// falsification `estimate::motion::fit_false_beat` applies to a measured
    /// key, applied to a drawn one. Always at or over one, because a draw that
    /// failed it was refused.
    pub model_ratio: f64,
}

/// Fits every distribution from the keys of `gains` that carry a row.
///
/// `targets` is the recording's own `irregular` at each fitted key, measured by
/// the caller through the same `series::Series` the compass scores with, and
/// `beat_depths` is the recording's own [`crate::motion::Motion::beat_depth_db`]
/// per key and partial, measured through the same `motion` code the fitted keys'
/// own targets came from. They are the two numbers in this model that come from
/// outside the preset, and the reason they do is that they are the *acceptance*
/// criteria and have to be measured the way the acceptance test measures them.
///
/// `beat_depths` is taken only over the keys at or under
/// [`HIGHEST_FALSE_BEAT_KEY`], because that is the only range the ceiling is
/// ever applied in: above it item 233's falsification refuses the mechanism
/// outright, and a treble key's beat — which is its tuning's and not its wire's
/// — is not evidence about the ceiling a drawn split is held under.
pub fn fit_texture(
    gains: &[Vec<f32>],
    false_beat: &[Vec<FalseBeat>],
    targets: &[(u8, f64)],
    beat_depths: &[(u8, u16, f64)],
    config: &ShapingConfig,
) -> TextureModel {
    let mut model = TextureModel {
        target: LogCurve::fit(
            &targets
                .iter()
                .map(|&(key, v)| (f64::from(key), v))
                .collect::<Vec<_>>(),
        ),
        beat_ceiling: BeatCeiling::fit(
            &beat_depths
                .iter()
                .copied()
                .filter(|&(key, _, _)| key <= HIGHEST_FALSE_BEAT_KEY)
                .collect::<Vec<_>>(),
        ),
        ..TextureModel::default()
    };

    // ---- the gain rows
    let mut cells = Vec::new();
    let mut amounts = Vec::new();
    let mut rho_terms: Vec<(f64, f64)> = Vec::new();
    for (index, row) in gains.iter().enumerate() {
        if row.is_empty() {
            continue;
        }
        let key = index_to_key(index);
        model.fitted_keys.push(key);
        let db: Vec<f64> = row.iter().map(|&g| 20.0 * f64::from(g).log10()).collect();
        cells.push((f64::from(key), row.len() as f64));
        // Both statistics are taken about the row's own tilt, because a
        // degree-2 curve in `ln k` is a *shape* and not a tie between
        // neighbouring partials: leaving it in would read the tilt's curvature
        // as agreement between adjacent cells and its span as roughness.
        if db.len() >= MIN_AMOUNT_CELLS {
            let residual = detrend(&db, config.envelope_degree);
            amounts.push((f64::from(key), robust_sigma(&residual)));
            if let Some(r) = lag_one(&residual) {
                // Fisher's z, weighted by the degrees of freedom, which is how
                // correlations from records of different lengths are pooled.
                rho_terms.push(((residual.len() as f64 - 3.0).max(1.0), atanh(r)));
            }
        }
    }
    model.cells = LogLine::fit(&cells);
    model.amount = LogLine::fit(&amounts);
    let weight: f64 = rho_terms.iter().map(|t| t.0).sum();
    model.rho = if weight > 0.0 {
        (rho_terms.iter().map(|t| t.0 * t.1).sum::<f64>() / weight).tanh()
    } else {
        0.0
    };
    model.rho_rows = rho_terms.len();

    // ---- the splits
    let fitted: Vec<u8> = model.fitted_keys.clone();
    let row_of = |key: u8| -> &[FalseBeat] {
        key_index(key)
            .and_then(|i| false_beat.get(i))
            .map_or(&[][..], |row| row.as_slice())
    };
    let eligible: Vec<u8> = fitted
        .iter()
        .copied()
        .filter(|&key| key <= HIGHEST_FALSE_BEAT_KEY)
        .collect();
    let written: Vec<u8> = eligible
        .iter()
        .copied()
        .filter(|&key| !row_of(key).is_empty())
        .collect();
    model.false_beat_probability = if eligible.is_empty() {
        0.0
    } else {
        written.len() as f64 / eligible.len() as f64
    };
    model.false_beat_count = LogLine::fit(
        &written
            .iter()
            .map(|&key| (f64::from(key), row_of(key).len() as f64))
            .collect::<Vec<_>>(),
    );

    let rows: Vec<(u8, FalseBeat)> = fitted
        .iter()
        .flat_map(|&key| row_of(key).iter().map(move |r| (key, *r)))
        .collect();
    model.false_beat_rows = rows.len();
    for (_, row) in &rows {
        if (1..=MAX_FALSE_BEATS_PER_KEY).contains(&usize::from(row.k)) {
            model.partial_weights[usize::from(row.k) - 1] += 1.0;
        }
    }
    let total: f64 = model.partial_weights.iter().sum();
    if total > 0.0 {
        for w in &mut model.partial_weights {
            *w /= total;
        }
    }
    if !rows.is_empty() {
        let ln: Vec<f64> = rows.iter().map(|(_, r)| f64::from(r.hz).ln()).collect();
        let n = ln.len() as f64;
        model.rate_ln_mean = ln.iter().sum::<f64>() / n;
        model.rate_ln_sigma =
            (ln.iter().map(|v| (v - model.rate_ln_mean).powi(2)).sum::<f64>() / n).sqrt();
        // Within-key centring for the partial correlation: what has to be zero
        // is that a *key's* rates rise with `k`, and pooling the raw pairs
        // would measure the register instead.
        let mut kk = Vec::new();
        let mut hh = Vec::new();
        for &key in &fitted {
            let row = row_of(key);
            if row.len() < 3 {
                continue;
            }
            let n = row.len() as f64;
            let mk = row.iter().map(|r| f64::from(r.k)).sum::<f64>() / n;
            let mh = row.iter().map(|r| f64::from(r.hz).ln()).sum::<f64>() / n;
            for r in row {
                kk.push(f64::from(r.k) - mk);
                hh.push(f64::from(r.hz).ln() - mh);
            }
        }
        model.rate_vs_partial = pearson(&kk, &hh);
        model.rate_vs_key = pearson(
            &rows.iter().map(|(key, _)| f64::from(*key)).collect::<Vec<_>>(),
            &ln,
        );
        model.depth = fit_depth(&rows);
        let residual: Vec<f64> = rows
            .iter()
            .map(|(key, r)| {
                f64::from(r.db)
                    - (model.depth[0]
                        + model.depth[1] * f64::from(*key)
                        + model.depth[2] * f64::from(r.k))
            })
            .collect();
        let n = residual.len() as f64;
        model.depth_sigma = (residual.iter().map(|v| v * v).sum::<f64>() / n).sqrt();
        let mean = rows.iter().map(|(_, r)| f64::from(r.db)).sum::<f64>() / n;
        let tss: f64 = rows
            .iter()
            .map(|(_, r)| (f64::from(r.db) - mean).powi(2))
            .sum();
        model.depth_r_squared = if tss > 0.0 {
            1.0 - residual.iter().map(|v| v * v).sum::<f64>() / tss
        } else {
            0.0
        };
    }
    model
}

impl TextureModel {
    /// Draws one key's texture. Deterministic in `key` and [`TEXTURE_SEED`].
    ///
    /// `partial_count` is the key's own bank: a row may not be longer than it
    /// and a split may not name a partial past it, both of which the schema
    /// refuses.
    pub fn synthesize(&self, key: u8, partial_count: usize, config: &ShapingConfig) -> SynthesizedTexture {
        let mut out = SynthesizedTexture {
            key,
            ..SynthesizedTexture::default()
        };
        // Four independent streams, so that changing what one of them draws
        // does not shift the others: the row's length must not depend on how
        // many splits the same key happened to get.
        let mut shape = Draw::for_key(key, 1);
        let mut amount = Draw::for_key(key, 2);
        let mut ceiling = Draw::for_key(key, 3);
        let mut splits = Draw::for_key(key, 4);

        out.target_irregular = self.target.draw(key, &mut ceiling);

        let cells = (self.cells.draw(key, &mut shape).round() as usize)
            .clamp(1, partial_count.min(MAX_ROW_CELLS));
        // Too short to hold a roughness at all: nothing is written. See
        // [`MIN_DRAWN_CELLS`].
        let cells = if cells < MIN_DRAWN_CELLS { 0 } else { cells };
        out.drawn_amount_db = self.amount.draw(key, &mut amount);
        let mut row = ar_one(cells, self.rho, &mut shape);
        // The tilt is projected out, so the row is a **roughness** and nothing
        // else. A random sequence has a low-order component like any other, and
        // left in it would be an invented per-key colour of two to four
        // decibels — the one thing this module refuses to draw — and it would
        // also defeat the trim, since `flatten_row` scales a row's departures
        // from its own degree-2 curve and would read part of the draw as the
        // curve. What is left is orthogonal to `1`, `ln k` and `ln^2 k`, which
        // is the same basis the amount was fitted in.
        //
        // A row too short for that basis has only its **level** taken out, for
        // the same reason `shaping::write_row` writes such a row's cells as
        // measured: a quadratic through three points in `ln k` goes *through*
        // them, and projecting one out of a three-cell row leaves nothing but
        // the arithmetic's own rounding — which, rescaled to the drawn amount,
        // is a sign pattern of dust standing at the rails and not a draw at
        // all. The two halves of the compass agree about what the level-only
        // rule costs: the long fitted rows' detrended spread has median 4.40 dB
        // and the nine short ones' raw spread has median 4.31, which is why one
        // amount distribution describes both.
        row = detrend(
            &row,
            if cells >= MIN_AMOUNT_CELLS {
                config.envelope_degree
            } else {
                0
            },
        );
        // Unit robust sigma, then the drawn amount: what is written is the
        // statistic that was fitted, cell for cell, rather than a Gaussian
        // whose sample spread happened to land somewhere near it.
        let sigma = robust_sigma(&row);
        if sigma > 0.0 {
            for v in &mut row {
                *v *= out.drawn_amount_db / sigma;
            }
        }
        let median = median_of(&row);
        for v in &mut row {
            *v -= median;
        }
        // The rails, by the fitted rows' own rule and the same implementation.
        let (centre, rail) = own_rail(&row.iter().map(|v| v / NEPERS_TO_DB).collect::<Vec<_>>(), config);
        let (lo, hi) = (
            NEPERS_TO_DB * (centre - rail),
            NEPERS_TO_DB * (centre + rail),
        );
        out.rail_db = NEPERS_TO_DB * rail;
        for v in &mut row {
            if *v > hi {
                *v = hi;
                out.railed_cells += 1;
            } else if *v < lo {
                *v = lo;
                out.railed_cells += 1;
            }
        }
        out.railed_amount_db = robust_sigma(&row);
        out.gains_db = row;

        let (rows, ratio) = self.draw_false_beat(key, partial_count, &mut splits);
        // The ceiling those asks will be closed against on the render: one draw
        // of this key's own scatter, spread over its partials by the surface's
        // own `k` term. A fifth stream, so that adding it moved no cell of the
        // rows the first four draw.
        let mut beat = Draw::for_key(key, 5);
        let z_key = beat.normal();
        out.beat_ceiling_db = rows
            .iter()
            .map(|row| self.beat_ceiling.draw(key, row.k, z_key, beat.normal()))
            .collect();
        out.false_beat = rows;
        out.model_ratio = ratio;
        out
    }

    /// The splits, with `DECISIONS.md` 233's falsification applied to the draw.
    ///
    /// A drawn key can fail it by luck — eight rates drawn independently of `k`
    /// can still happen to rise with `k` — and a key that fails it is a key the
    /// estimator would have refused to write, so the draw is taken again. After
    /// [`FALSIFICATION_ATTEMPTS`] failures the key is written empty, which is
    /// also what the estimator does.
    fn draw_false_beat(
        &self,
        key: u8,
        partial_count: usize,
        rng: &mut Draw,
    ) -> (Vec<FalseBeat>, f64) {
        if key > HIGHEST_FALSE_BEAT_KEY || partial_count == 0 {
            return (Vec::new(), 0.0);
        }
        if rng.uniform() >= self.false_beat_probability {
            return (Vec::new(), 0.0);
        }
        let reachable = partial_count.min(MAX_FALSE_BEATS_PER_KEY);
        let count = (self.false_beat_count.draw(key, rng).round() as usize).clamp(1, reachable);
        // Which partials: sampled without replacement in proportion to how
        // often the fitted rows named each one.
        let mut weights: Vec<f64> = self.partial_weights[..reachable].to_vec();
        let mut chosen: Vec<u16> = Vec::with_capacity(count);
        for _ in 0..count {
            let total: f64 = weights.iter().sum();
            if total <= 0.0 {
                break;
            }
            let mut pick = rng.uniform() * total;
            let mut at = weights.len() - 1;
            for (i, w) in weights.iter().enumerate() {
                if pick < *w {
                    at = i;
                    break;
                }
                pick -= *w;
            }
            weights[at] = 0.0;
            chosen.push(at as u16 + 1);
        }
        chosen.sort_unstable();
        if chosen.is_empty() {
            return (Vec::new(), 0.0);
        }
        for _ in 0..FALSIFICATION_ATTEMPTS {
            let rates: Vec<f64> = chosen
                .iter()
                .map(|_| {
                    truncated(
                        || (self.rate_ln_mean + self.rate_ln_sigma * rng.normal()).exp(),
                        MIN_FITTED_HZ,
                        f64::from(MAX_FALSE_BEAT_HZ),
                    )
                })
                .collect();
            let ratio = flat_over_proportional(&chosen, &rates);
            // One split cannot be tested and is not evidence of a tuning beat
            // either; the estimator writes those too.
            if chosen.len() >= 3 && ratio < 1.0 {
                continue;
            }
            let rows: Vec<FalseBeat> = chosen
                .iter()
                .zip(&rates)
                .map(|(&k, &hz)| {
                    let centre = self.depth[0]
                        + self.depth[1] * f64::from(key)
                        + self.depth[2] * f64::from(k);
                    let db = truncated(
                        || centre + self.depth_sigma * rng.normal(),
                        f64::from(MIN_FALSE_BEAT_DB),
                        f64::from(MAX_FALSE_BEAT_DB),
                    );
                    FalseBeat {
                        k,
                        hz: hz as f32,
                        db: db as f32,
                    }
                })
                .collect();
            return (rows, ratio);
        }
        (Vec::new(), 0.0)
    }
}

/// Fewest cells a **drawn** row must have before it is written at all.
///
/// `envelope_degree + 2`. A row of three cells has no roughness in it: a
/// quadratic in `ln k` goes through three points, so every three-cell row *is*
/// its own tilt, and a tilt is the one thing this module refuses to draw (see
/// the header). The fitted rows of that length are kept because they are
/// measurements of something; a drawn one would be an invented colour with no
/// roughness under it.
pub const MIN_DRAWN_CELLS: usize = 4;

/// Fewest cells a fitted row must have before its detrended spread is read as
/// a roughness.
///
/// Eight against the degree-2 curve's three parameters. Measured on the shipped
/// preset the gate keeps 19 of the 28 fitted rows, and what it drops is exactly
/// the rows that cannot answer: the four treble rows of three or four cells
/// have a detrended MAD of **0.00 dB**, because a quadratic through four points
/// in `ln k` goes through them.
pub const MIN_AMOUNT_CELLS: usize = 8;

/// How many times a drawn set of rates may fail `DECISIONS.md` 233's
/// falsification before the key is written empty.
const FALSIFICATION_ATTEMPTS: usize = 16;

/// `RSS(rate = s k) / RSS(rate = c)`: over one is a rate the partial number
/// does not predict, which is a wire's beat and not a tuning's.
///
/// The same statistic as `estimate::motion`'s `model_ratio`, on the pairs
/// rather than on its `Companion`, so that the draw is tested by the test it
/// has to pass.
pub fn flat_over_proportional(partials: &[u16], rates: &[f64]) -> f64 {
    let n = partials.len().min(rates.len());
    if n < 2 {
        return 0.0;
    }
    let mean = rates[..n].iter().sum::<f64>() / n as f64;
    let flat: f64 = rates[..n].iter().map(|r| (r - mean).powi(2)).sum();
    let (mut num, mut den) = (0.0, 0.0);
    for i in 0..n {
        num += f64::from(partials[i]) * rates[i];
        den += f64::from(partials[i]).powi(2);
    }
    let slope = if den > 0.0 { num / den } else { 0.0 };
    let proportional: f64 = (0..n)
        .map(|i| (rates[i] - slope * f64::from(partials[i])).powi(2))
        .sum();
    if flat <= 0.0 {
        return f64::INFINITY;
    }
    proportional / flat
}

/// A stationary AR(1) sequence of unit variance: `z[i] = rho z[i-1] +
/// sqrt(1 - rho^2) e[i]`.
///
/// The correlation between neighbouring cells is a property of the piano the
/// fitted rows measured, and a row of independent draws would not have it: a
/// spectrum whose partials are tied to their neighbours is smoother at the same
/// spread than one whose partials are not, and `irregular` — the metric the
/// compass scores and the metric the listener heard — is exactly the statistic
/// that difference lands in.
fn ar_one(n: usize, rho: f64, rng: &mut Draw) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    let rho = rho.clamp(-0.95, 0.95);
    let innovation = (1.0 - rho * rho).sqrt();
    let mut out = Vec::with_capacity(n);
    let mut last = rng.normal();
    out.push(last);
    for _ in 1..n {
        last = rho * last + innovation * rng.normal();
        out.push(last);
    }
    out
}

/// Draws until the value is inside `[lo, hi]`, and clamps if it never is.
///
/// Rejection and not clamping, because the fitted distributions have hard edges
/// that are *measurement* edges — the slowest rate a 2.7 s record can show, the
/// quietest companion the schema can name — and clamping a normal against one
/// piles a spike of mass on it that the fitted rows do not have.
fn truncated(mut draw: impl FnMut() -> f64, lo: f64, hi: f64) -> f64 {
    for _ in 0..64 {
        let value = draw();
        if (lo..=hi).contains(&value) {
            return value;
        }
    }
    draw().clamp(lo, hi)
}

/// `1.4826 MAD` about the median: the spread the rails and the amount are both
/// stated in, and the one `shaping::rail_cells` uses.
pub fn robust_sigma(values: &[f64]) -> f64 {
    let centre = median_of(values);
    1.4826 * median_of(&values.iter().map(|v| (v - centre).abs()).collect::<Vec<_>>())
}

fn median_of(values: &[f64]) -> f64 {
    let mut v: Vec<f64> = values.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    let mid = v.len() / 2;
    if v.len() % 2 == 0 {
        0.5 * (v[mid - 1] + v[mid])
    } else {
        v[mid]
    }
}

/// A row's departures from its own degree-`degree` polynomial in `ln k`.
fn detrend(db: &[f64], degree: usize) -> Vec<f64> {
    let x: Vec<f64> = (1..=db.len()).map(|k| (k as f64).ln()).collect();
    let weights = vec![1.0; db.len()];
    match crate::numeric::weighted_polyfit(&x, db, &weights, degree) {
        Some(curve) => x
            .iter()
            .zip(db)
            .map(|(&xx, &yy)| yy - crate::numeric::poly_eval(&curve, xx))
            .collect(),
        None => db.to_vec(),
    }
}

fn lag_one(values: &[f64]) -> Option<f64> {
    if values.len() < 4 {
        return None;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance: f64 = values.iter().map(|v| (v - mean).powi(2)).sum();
    if variance <= 0.0 {
        return None;
    }
    let covariance: f64 = values
        .windows(2)
        .map(|w| (w[0] - mean) * (w[1] - mean))
        .sum();
    Some(covariance / variance)
}

fn atanh(r: f64) -> f64 {
    r.clamp(-0.999, 0.999).atanh()
}

fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }
    let mx = x[..n].iter().sum::<f64>() / n as f64;
    let my = y[..n].iter().sum::<f64>() / n as f64;
    let sxx: f64 = x[..n].iter().map(|v| (v - mx).powi(2)).sum();
    let syy: f64 = y[..n].iter().map(|v| (v - my).powi(2)).sum();
    if sxx <= 0.0 || syy <= 0.0 {
        return 0.0;
    }
    let sxy: f64 = (0..n).map(|i| (x[i] - mx) * (y[i] - my)).sum();
    sxy / (sxx * syy).sqrt()
}

/// `db = a + b·key + c·k`, by least squares.
fn fit_depth(rows: &[(u8, FalseBeat)]) -> [f64; 3] {
    let n = rows.len() as f64;
    if rows.len() < 4 {
        let mean = rows.iter().map(|(_, r)| f64::from(r.db)).sum::<f64>() / n.max(1.0);
        return [mean, 0.0, 0.0];
    }
    let key: Vec<f64> = rows.iter().map(|(key, _)| f64::from(*key)).collect();
    let part: Vec<f64> = rows.iter().map(|(_, r)| f64::from(r.k)).collect();
    let db: Vec<f64> = rows.iter().map(|(_, r)| f64::from(r.db)).collect();
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    let ones = vec![1.0; rows.len()];
    let a = [
        [n, dot(&ones, &key), dot(&ones, &part)],
        [dot(&key, &ones), dot(&key, &key), dot(&key, &part)],
        [dot(&part, &ones), dot(&part, &key), dot(&part, &part)],
    ];
    let b = [dot(&ones, &db), dot(&key, &db), dot(&part, &db)];
    solve3(a, b).unwrap_or([db.iter().sum::<f64>() / n, 0.0, 0.0])
}

/// The draw itself: `splitmix64`, which is what a seeded stream has to be here.
///
/// Not a library generator, and the reason is the same one the caches of item
/// 283 are keyed by content: a preset emitted today has to be the preset
/// emitted next year, and a dependency's generator is free to change its
/// stream between versions without changing its name. This one is fourteen
/// lines and its constants are in its own source, so the instrument cannot
/// change under a `cargo update`.
#[derive(Clone, Debug)]
pub struct Draw {
    state: u64,
    /// Box–Muller produces two normals at a time; the second is kept.
    spare: Option<f64>,
}

impl Draw {
    /// A stream for one key, distinct per `stream` index.
    pub fn for_key(key: u8, stream: u64) -> Draw {
        Draw {
            state: TEXTURE_SEED
                .wrapping_add(u64::from(key).wrapping_mul(0x9e37_79b9_7f4a_7c15))
                .wrapping_add(stream.wrapping_mul(0xbf58_476d_1ce4_e5b9)),
            spare: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform on `[0, 1)`, from the top 53 bits.
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Standard normal, Box–Muller.
    pub fn normal(&mut self) -> f64 {
        if let Some(spare) = self.spare.take() {
            return spare;
        }
        let u = self.uniform().max(f64::MIN_POSITIVE);
        let v = self.uniform();
        let radius = (-2.0 * u.ln()).sqrt();
        let angle = std::f64::consts::TAU * v;
        self.spare = Some(radius * angle.sin());
        radius * angle.cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> TextureModel {
        TextureModel {
            cells: LogLine {
                intercept: 5.058,
                slope: -0.03793,
                sigma: 0.183,
                ..LogLine::default()
            },
            amount: LogLine {
                intercept: 1.461,
                slope: 0.00089,
                sigma: 0.356,
                ..LogLine::default()
            },
            rho: 0.113,
            target: LogCurve {
                coefficients: [3.476, -0.06590, 0.000679],
                sigma: 0.206,
                ..LogCurve::default()
            },
            false_beat_probability: 0.8,
            false_beat_count: LogLine {
                intercept: 1.980,
                slope: -0.01495,
                sigma: 0.524,
                ..LogLine::default()
            },
            partial_weights: [0.183, 0.169, 0.099, 0.113, 0.113, 0.127, 0.099, 0.099],
            rate_ln_mean: 0.017,
            rate_ln_sigma: 0.584,
            depth: [-46.445, 0.3387, 2.076],
            depth_sigma: 10.55,
            beat_ceiling: BeatCeiling {
                coefficients: [-1.825, 0.04662, 0.1183],
                sigma: 0.829,
                sigma_key: 0.686,
                sigma_cell: 0.531,
                ..BeatCeiling::default()
            },
            ..TextureModel::default()
        }
    }

    #[test]
    fn a_key_draws_the_same_texture_every_time() {
        let config = ShapingConfig::default();
        let model = model();
        let first = model.synthesize(61, 80, &config);
        let second = model.synthesize(61, 80, &config);
        assert_eq!(first, second);
        // ... and two keys do not draw the same one.
        let other = model.synthesize(62, 80, &config);
        assert_ne!(first.gains_db, other.gains_db);
    }

    #[test]
    fn a_drawn_row_lands_on_the_amount_it_drew() {
        let config = ShapingConfig::default();
        let model = model();
        for key in [25u8, 40, 61, 74, 88] {
            let drawn = model.synthesize(key, 80, &config);
            let sigma = robust_sigma(&drawn.gains_db);
            assert!(
                (sigma - drawn.drawn_amount_db).abs() <= 0.35 * drawn.drawn_amount_db,
                "key {key}: wrote {sigma:.2} dB of spread against a drawn {:.2}",
                drawn.drawn_amount_db
            );
        }
    }

    /// A short row is either a draw or nothing at all, and never the
    /// arithmetic's own rounding.
    ///
    /// A quadratic through three points in `ln k` goes through them, so
    /// projecting one out of a three-cell row leaves ~1e-16 — and rescaling
    /// *that* to the drawn amount writes a sign pattern of dust standing at the
    /// rails. [`MIN_DRAWN_CELLS`] is the rule that forbids it and this is the
    /// gate on the rule, at the two lengths either side of it.
    #[test]
    fn a_short_row_carries_its_draw_and_not_the_arithmetics_dust() {
        let config = ShapingConfig::default();
        let model = model();
        for (key, partials) in [(96u8, 3usize), (102, 3), (105, 2), (108, 3)] {
            let drawn = model.synthesize(key, partials, &config);
            assert!(
                drawn.gains_db.is_empty(),
                "key {key}: {partials} cells is under MIN_DRAWN_CELLS and wrote {:?}",
                drawn.gains_db
            );
        }
        for (key, partials) in [(93u8, 4usize), (91, 5), (85, 6), (80, 7)] {
            let drawn = model.synthesize(key, partials, &config);
            assert!(
                (MIN_DRAWN_CELLS..=partials).contains(&drawn.gains_db.len()),
                "key {key}: {partials} partials drew {} cells",
                drawn.gains_db.len()
            );
            let sigma = robust_sigma(&drawn.gains_db);
            assert!(
                (sigma - drawn.drawn_amount_db).abs() <= 0.35 * drawn.drawn_amount_db,
                "key {key}: {partials} cells wrote {sigma:.2} dB of spread against a drawn \
                 {:.2} — {:?}",
                drawn.drawn_amount_db,
                drawn.gains_db
            );
            // The dust case rails every cell it does not put at zero; a draw
            // rails at most the one the key's own 6 dB floor reaches.
            assert!(
                drawn.railed_cells <= 1,
                "key {key} railed {} of {partials} cells: {:?}",
                drawn.railed_cells,
                drawn.gains_db
            );
        }
    }

    #[test]
    fn the_drawn_rows_land_inside_the_fitted_distributions() {
        let config = ShapingConfig::default();
        let model = model();
        let mut amounts = Vec::new();
        let mut cells = Vec::new();
        for key in 21..=108u8 {
            let drawn = model.synthesize(key, 80, &config);
            if !drawn.gains_db.is_empty() {
                amounts.push(drawn.railed_amount_db);
            }
            cells.push(drawn.gains_db.len() as f64);
        }
        let median = median_of(&amounts);
        // The fitted rows' own median is 4.40 dB.
        assert!(
            (3.0..6.0).contains(&median),
            "median drawn amount {median:.2} dB"
        );
        // A row is either long enough to hold a roughness or it is not written.
        assert!(cells
            .iter()
            .all(|&c| c == 0.0 || (MIN_DRAWN_CELLS as f64..=48.0).contains(&c)));
        assert!(
            cells.iter().filter(|&&c| c > 0.0).count() >= 60,
            "only {} of 88 keys drew a row",
            cells.iter().filter(|&&c| c > 0.0).count()
        );
    }

    #[test]
    fn the_treble_draws_no_false_beat_and_the_rest_pass_the_falsification() {
        let config = ShapingConfig::default();
        let model = model();
        let mut written = 0;
        for key in 21..=108u8 {
            let drawn = model.synthesize(key, 80.min(usize::from(120 - key)), &config);
            if key > HIGHEST_FALSE_BEAT_KEY {
                assert!(drawn.false_beat.is_empty(), "key {key} drew a split");
                continue;
            }
            if drawn.false_beat.is_empty() {
                continue;
            }
            written += 1;
            assert!(drawn.false_beat.len() <= MAX_FALSE_BEATS_PER_KEY);
            let partials: Vec<u16> = drawn.false_beat.iter().map(|r| r.k).collect();
            let rates: Vec<f64> = drawn.false_beat.iter().map(|r| f64::from(r.hz)).collect();
            if partials.len() >= 3 {
                assert!(
                    flat_over_proportional(&partials, &rates) >= 1.0,
                    "key {key} wrote rates that track the partial number"
                );
            }
            let mut seen = partials.clone();
            seen.sort_unstable();
            seen.dedup();
            assert_eq!(seen.len(), partials.len(), "key {key} split one partial twice");
            for row in &drawn.false_beat {
                assert!((MIN_FITTED_HZ..=f64::from(MAX_FALSE_BEAT_HZ)).contains(&f64::from(row.hz)));
                assert!((MIN_FALSE_BEAT_DB..=MAX_FALSE_BEAT_DB).contains(&row.db));
            }
        }
        assert!(written > 20, "only {written} keys drew a split");
    }

    #[test]
    fn the_rates_a_key_draws_do_not_know_the_partial_number() {
        let config = ShapingConfig::default();
        let model = model();
        let (mut k, mut hz) = (Vec::new(), Vec::new());
        for key in 21..=HIGHEST_FALSE_BEAT_KEY {
            let drawn = model.synthesize(key, 40, &config);
            if drawn.false_beat.len() < 3 {
                continue;
            }
            let n = drawn.false_beat.len() as f64;
            let mk = drawn.false_beat.iter().map(|r| f64::from(r.k)).sum::<f64>() / n;
            let mh = drawn
                .false_beat
                .iter()
                .map(|r| f64::from(r.hz).ln())
                .sum::<f64>()
                / n;
            for row in &drawn.false_beat {
                k.push(f64::from(row.k) - mk);
                hz.push(f64::from(row.hz).ln() - mh);
            }
        }
        let r = pearson(&k, &hz);
        assert!(r.abs() < 0.2, "within-key corr(k, ln hz) = {r:+.3}");
    }

    #[test]
    fn the_ar_one_sequence_has_the_correlation_it_was_asked_for() {
        let mut rng = Draw::for_key(60, 9);
        let row = ar_one(20_000, 0.4, &mut rng);
        let r = lag_one(&row).expect("long enough");
        assert!((r - 0.4).abs() < 0.03, "lag-1 came back {r:+.3}");
        let mean = row.iter().sum::<f64>() / row.len() as f64;
        let sd = (row.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / row.len() as f64).sqrt();
        assert!((sd - 1.0).abs() < 0.05, "sd {sd:.3}");
    }

    /// Every drawn split carries a ceiling, and the ceiling is a positive
    /// number that rises with the partial number the way the recordings' own
    /// depths do.
    ///
    /// The parallel arrays are the contract `fit_motion`'s
    /// `close_splits_on_the_render` zips: a row without a ceiling would be
    /// closed against its neighbour's.
    #[test]
    fn every_drawn_split_carries_a_ceiling_of_its_own() {
        let config = ShapingConfig::default();
        let model = model();
        let mut seen = 0usize;
        for key in 21..=HIGHEST_FALSE_BEAT_KEY {
            let drawn = model.synthesize(key, 40, &config);
            assert_eq!(
                drawn.false_beat.len(),
                drawn.beat_ceiling_db.len(),
                "key {key} drew {} splits and {} ceilings",
                drawn.false_beat.len(),
                drawn.beat_ceiling_db.len()
            );
            for (row, &ceiling) in drawn.false_beat.iter().zip(&drawn.beat_ceiling_db) {
                assert!(
                    ceiling.is_finite() && ceiling > 0.0,
                    "key {key} k{} drew a ceiling of {ceiling}",
                    row.k
                );
                seen += 1;
            }
        }
        assert!(seen > 40, "only {seen} splits drawn over the compass");
    }

    /// The ceilings a key draws are the recordings' own distribution and not a
    /// constant: over the compass their median tracks the fitted surface, and
    /// within one key the partials do not all get the same number.
    #[test]
    fn the_drawn_ceilings_land_inside_the_fitted_distribution() {
        let config = ShapingConfig::default();
        let model = model();
        let mut ratios = Vec::new();
        let mut spread_within = 0usize;
        let mut keys_with_splits = 0usize;
        for key in 21..=HIGHEST_FALSE_BEAT_KEY {
            let drawn = model.synthesize(key, 40, &config);
            if drawn.false_beat.len() < 2 {
                continue;
            }
            keys_with_splits += 1;
            let mut normalised = Vec::new();
            for (row, &ceiling) in drawn.false_beat.iter().zip(&drawn.beat_ceiling_db) {
                let at = model.beat_ceiling.at(key, row.k);
                ratios.push((ceiling / at).ln());
                normalised.push(ceiling / at);
            }
            // Two partials of one key share the key's draw and not the cell's.
            if normalised
                .windows(2)
                .any(|w| (w[0] - w[1]).abs() > 1e-9)
            {
                spread_within += 1;
            }
        }
        let median = median_of(&ratios);
        assert!(
            median.abs() < 0.35,
            "the drawn ceilings sit {median:+.3} in ln off the surface they were drawn from"
        );
        let sigma = robust_sigma(&ratios);
        assert!(
            (0.5 * model.beat_ceiling.sigma..2.0 * model.beat_ceiling.sigma).contains(&sigma),
            "drawn scatter {sigma:.3} against a fitted {:.3}",
            model.beat_ceiling.sigma
        );
        assert!(
            spread_within * 2 > keys_with_splits,
            "only {spread_within} of {keys_with_splits} keys drew a ceiling that varies \
             between its own partials"
        );
    }

    /// The ceiling is a fit and not an assertion: given a surface it recovers
    /// it, and it says how much of what it recovered is each term.
    #[test]
    fn the_beat_ceiling_recovers_a_surface_it_is_given() {
        let points: Vec<(u8, u16, f64)> = (21..=94u8)
            .flat_map(|key| {
                (1..=8u16).map(move |k| {
                    let db = (-1.8 + 0.045 * f64::from(key) + 0.12 * f64::from(k)).exp();
                    (key, k, db)
                })
            })
            .collect();
        let fitted = BeatCeiling::fit(&points);
        assert!((fitted.coefficients[0] + 1.8).abs() < 1e-6);
        assert!((fitted.coefficients[1] - 0.045).abs() < 1e-8);
        assert!((fitted.coefficients[2] - 0.12).abs() < 1e-8);
        assert!(fitted.sigma < 1e-9, "sigma {:.3e}", fitted.sigma);
        assert!(fitted.r_squared > 0.999);
        assert_eq!(fitted.points, points.len());
        assert_eq!(fitted.keys, 74);
        // A surface that really is two terms is not explained by either alone.
        assert!(fitted.r_squared_key_only < fitted.r_squared);
        assert!(fitted.r_squared_partial_only < fitted.r_squared);
        // A depth that cannot be a depth is dropped rather than floored.
        let with_dead = BeatCeiling::fit(
            &points
                .iter()
                .copied()
                .chain([(60u8, 1u16, 0.0), (60, 2, -3.0)])
                .collect::<Vec<_>>(),
        );
        assert_eq!(with_dead.points, points.len());
    }

    #[test]
    fn the_curve_recovers_a_quadratic_it_is_given() {
        let points: Vec<(f64, f64)> = (21..=108)
            .map(|k| {
                let x = f64::from(k);
                (x, (1.0 - 0.05 * x + 0.0008 * x * x).exp())
            })
            .collect();
        let curve = LogCurve::fit(&points);
        assert!((curve.coefficients[0] - 1.0).abs() < 1e-6);
        assert!((curve.coefficients[1] + 0.05).abs() < 1e-8);
        assert!((curve.coefficients[2] - 0.0008).abs() < 1e-10);
        assert!(curve.sigma < 1e-6);
    }
}
