//! The upper partials' decay: how fast a partial the fitted rows never reached
//! actually dies, on the recording and on the engine's own render of the same
//! note.
//!
//! `DECISIONS.md` 295 convicted this. Over 0.1 -> 1 s the engine's 2-6 kHz band
//! decays 11.2 dB less than the recording's in the tenor and its 6-12 kHz
//! 15.3 dB less, against a velocity-layer floor of 0.42 / 0.86 dB — so a
//! brightness correction fitted at the strike is carried through a tail that is
//! already far too long, and a phrase integrates the tail. What shapes a
//! partial's decay in the engine is `notes.sigma0` + `notes.sigma1 (f/1000)^2`
//! multiplied by that partial's `notes.partial_sigma_scale` cell, and above the
//! reach the recording gave a key that cell is exactly 1.0
//! (`string::PartialShaping`): the law alone, extrapolated three octaves past
//! anything that was ever measured.
//!
//! # Why this is not `estimate::decay`
//!
//! [`decay::fit_decays`](crate::estimate::decay::fit_decays) reads
//! [`NoteTrajectories`](crate::trajectory::NoteTrajectories), which is a
//! *tracker's* output: a partial exists there only where peak-picking found and
//! associated it frame after frame, and on a real recording that stops between
//! partial 13 (A4) and partial 15 (C4) — 3.8 kHz, the bottom of the band the
//! conviction is about. The reach is the tracker's and not the decay stage's,
//! and it cannot be argued upward.
//!
//! What can be measured that far up is a **known** frequency's envelope, because
//! the inharmonic model already says where partial `k` of this key is and the
//! only question is how fast what is there dies. That is a band-pass and a
//! straight line in dB, which is what this module is: one long-window transform
//! per key, the bin at each predicted partial read out of every frame, and
//! [`fit_tail_to`] through the result.
//!
//! # The two rules that make it a measurement
//!
//! **The window is the note's own.** [`window_for`] sizes the transform from
//! `f0`, not from the partial: a Hann main lobe is `4 sr / N` wide, so a window
//! of [`WINDOW_CYCLES`] periods of `f0` puts the neighbouring partials outside
//! it at *every* `k`. `estimate::brilliance::narrowband_db`'s boxcar is a
//! quarter of `hz` wide instead, which separates a partial from its neighbours
//! only while `k < 4` — its own `PARTIAL_PROBE` at `k = 8` and `k = 16` was
//! already reading several partials at once, and that is corrected here rather
//! than in it, because its 6 keys x 5 partials are a printed diagnostic and
//! this is a fit.
//!
//! **Nothing is fitted within [`FLOOR_MARGIN_DB`] of the signal's own floor**
//! (`DECISIONS.md` 89, 293). A recording has a room, a tape and the rest of the
//! instrument under it; a render has its board field and its sympathetic halo.
//! Once the partial is inside that, the envelope is the floor's and its slope is
//! zero however long you watch — which is how item 293's broadband reading came
//! to call a dead top octave an eternal one. A partial inside its own floor
//! measures **nothing** here, and nothing is what is written for it.

use crate::estimate::brilliance::{HF1, HF2};
use crate::estimate::texture::LogLine;
use crate::stft::{Stft, StftConfig};

/// Periods of the note's own `f0` in the analysis window.
///
/// A Hann main lobe is `4 sr / N` wide; `N = 8 sr / f0` puts its half-width at
/// `f0 / 2`, so the nearest neighbouring partial sits on the far side of the
/// first null with the window's own -31 dB sidelobes under it. Eight and not
/// four because four puts the neighbour *on* the null, where a partial pulled
/// by inharmonicity or by a beat does not stay.
pub const WINDOW_CYCLES: f64 = 8.0;

/// Longest analysis window, in samples. At 48 kHz this is 341 ms, which is
/// [`WINDOW_CYCLES`] periods of 23.4 Hz — under A0's 27.5, so no key on the
/// compass is clamped by it and it is a bound on the arithmetic rather than on
/// the measurement.
pub const MAX_WINDOW: usize = 1 << 14;

/// Shortest analysis window, in samples: the transform still has to have bins.
pub const MIN_WINDOW: usize = 1 << 7;

/// Advance between envelope points, in seconds.
///
/// Five milliseconds is a fortieth of the shortest decay this module is asked
/// about (the top octave's 200 ms) and a thousandth of the longest, so no fit
/// here is short of points; the cost of the transform is inverse in it.
pub const HOP_S: f64 = 0.005;

/// Seconds after the strike from which a signal's own floor is read.
///
/// The same instant `estimate::brilliance::FLOOR_FROM_S` uses, and for the same
/// reason: what is left when the note is over.
pub const FLOOR_FROM_S: f64 = 3.0;

/// How far a partial must stand over its own signal's floor before a decay read
/// off it is the partial's and not the floor's. `DECISIONS.md` 89 and 293.
pub const FLOOR_MARGIN_DB: f64 = 10.0;

/// Least measurable decay, in dB, before a slope through it is an
/// extrapolation.
pub const MIN_MEASURABLE_DB: f64 = 12.0;

/// Least measurable span, in seconds.
pub const MIN_SPAN_S: f64 = 0.10;

/// Largest residual about the fitted line, in dB rms, that still leaves the
/// envelope a decay rather than a beat pattern or a floor with a shoulder.
///
/// Six because a partial of a three-string unison beats: the deepest measured
/// nulls are 25-47 dB (`DECISIONS.md` 46) but they are narrow, and an rms over
/// the whole fitted span sees a few decibels of them. Past six what is being
/// fitted is not a straight line in dB.
pub const MAX_RESIDUAL_DB: f64 = 4.0;

/// How far apart the two halves of a fitted prefix's slopes may be before what
/// was fitted is not one decay.
pub const MAX_HALF_RATIO: f64 = 2.5;

/// Least share of the floor-limited range a shortened prefix may keep.
///
/// The shortening is there to walk back off a floor that is not stationary — a
/// render's board field and halo decay with the note — which costs the last
/// part of the range and no more. It is *not* a licence to go looking for
/// whatever short stretch of a beating envelope happens to be a straight line:
/// a partial that is only a decay over a fifth of its own measurable range is a
/// partial this module has nothing to say about.
pub const MIN_PREFIX_SHARE: f64 = 0.5;

/// One partial's measured tail.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TailFit {
    /// Seconds to fall 60 dB along the fitted line.
    pub t60_s: f64,
    /// How far the signal's own late level sits under this partial's peak, dB.
    pub floor_db: f64,
    /// Seconds of envelope the line was fitted over.
    pub span_s: f64,
    /// RMS of the envelope about the fitted line, dB.
    pub residual_db: f64,
}

impl TailFit {
    /// The decay rate the engine's `sigma` means: `ln(1000) / T60`.
    pub fn sigma(&self) -> f64 {
        crate::estimate::decay::LN_1000 / self.t60_s
    }
}

/// Analysis window for a key whose fundamental is `f0_hz`: the power of two at
/// or above [`WINDOW_CYCLES`] periods of it, clamped.
pub fn window_for(f0_hz: f64, sample_rate: f64) -> usize {
    if !f0_hz.is_finite() || f0_hz <= 0.0 {
        return MAX_WINDOW;
    }
    let want = WINDOW_CYCLES * sample_rate / f0_hz;
    let mut n = MIN_WINDOW;
    while (n as f64) < want && n < MAX_WINDOW {
        n <<= 1;
    }
    n.clamp(MIN_WINDOW, MAX_WINDOW)
}

/// The dB envelope of every partial in `partial_hz`, one long-window transform
/// over the whole signal, [`HOP_S`] apart.
///
/// The window is sized from `f0_hz` — the note's partial *spacing* — so that one
/// bin is one partial at every `k`. Frames are timestamped at the centre of
/// their window by [`Stft`], and this returns them in order from the first
/// complete window, so index `i` is `i * HOP_S` seconds into the *analysable*
/// part of the signal; [`fit_tail_to`] only ever measures differences and a floor,
/// both of which are blind to that offset.
pub fn partial_envelopes(
    mono: &[f32],
    partial_hz: &[f64],
    f0_hz: f64,
    sample_rate: f64,
) -> Vec<Vec<f64>> {
    let window = window_for(f0_hz, sample_rate);
    let hop = ((HOP_S * sample_rate).round() as usize).max(1);
    let Ok(config) = StftConfig::new(window, hop, window) else {
        return vec![Vec::new(); partial_hz.len()];
    };
    let Ok(stft) = Stft::new(config) else {
        return vec![Vec::new(); partial_hz.len()];
    };
    let spacing = sample_rate / window as f64;
    let bins = stft.bins();
    let index: Vec<Option<usize>> = partial_hz
        .iter()
        .map(|&hz| {
            let bin = (hz / spacing).round() as usize;
            (hz.is_finite() && hz > 0.0 && bin < bins).then_some(bin)
        })
        .collect();
    let mut out: Vec<Vec<f64>> = vec![Vec::new(); partial_hz.len()];
    stft.for_each_frame(mono, sample_rate, |_, magnitude| {
        for (slot, bin) in out.iter_mut().zip(&index) {
            let Some(bin) = *bin else { continue };
            slot.push(20.0 * f64::from(magnitude[bin]).max(1e-30).log10());
        }
    });
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

/// How far a partial can be watched down before it is inside the signal's own
/// floor: `peak - (floor + FLOOR_MARGIN_DB)`, in dB.
///
/// `None` when the signal is too short for a floor to be read at all.
pub fn measurable_db(env: &[f64], dt_s: f64) -> Option<f64> {
    if env.len() < 4 || dt_s <= 0.0 {
        return None;
    }
    let from = ((FLOOR_FROM_S / dt_s) as usize).min(env.len());
    if from >= env.len() {
        return None;
    }
    let floor = median(env[from..].to_vec());
    let peak = env.iter().copied().fold(f64::MIN, f64::max);
    (floor.is_finite() && peak.is_finite()).then_some(peak - (floor + FLOOR_MARGIN_DB))
}

/// Least squares in dB through the **trusted prefix** of one partial's
/// envelope: from its peak down `drop_db`, shortened until what is being fitted
/// is a straight line.
///
/// Three rules, and each of them is a way this measurement goes wrong on real
/// audio rather than a tolerance somebody picked.
///
/// * **`drop_db` is imposed from outside, and the same number is given to both
///   signals.** A piano partial is a double decay — prompt then aftersound — so
///   a line through 60 dB of it and a line through 25 dB of it are not the same
///   statistic, and the recording's floor is much higher than a render's. Fit
///   each side to whatever its own floor allows and the engine is *systematically*
///   the slower one, by construction and at every key. [`measure_key`] therefore
///   takes the smaller of the two sides' [`measurable_db`] and hands it to both.
/// * **The prefix is shortened until the residual is inside
///   [`MAX_RESIDUAL_DB`]**, geometrically, deterministically, and never past
///   [`MIN_PREFIX_SHARE`] of the range the floor allowed. A signal's floor
///   is not stationary — a render's is its board field and its halo, which decay
///   with the note — so an envelope can flatten onto a floor well above the one
///   read at [`FLOOR_FROM_S`], and a line through the flattening reads a decay
///   that is too slow. What is fitted is the longest prefix that is still a
///   decay.
/// * **The two halves of that prefix must agree**, within
///   [`MAX_HALF_RATIO`]. A three-string unison nulls a partial 25-47 dB deep
///   (`DECISIONS.md` 46) and a short prefix caught on the way into a null is a
///   clean steep line that means nothing. Splitting the fit is what tells a
///   decay from a beat, and it costs one more pass over the same points.
///
/// `None` is *not a long decay* — it is an unmeasurable one, which is the
/// distinction `DECISIONS.md` 293 turns on.
pub fn fit_tail_to(env: &[f64], dt_s: f64, drop_db: f64) -> Option<TailFit> {
    if env.len() < 4 || dt_s <= 0.0 || drop_db < MIN_MEASURABLE_DB {
        return None;
    }
    let from = ((FLOOR_FROM_S / dt_s) as usize).min(env.len());
    if from >= env.len() {
        return None;
    }
    let floor = median(env[from..].to_vec());
    if !floor.is_finite() {
        return None;
    }
    let (top, &peak) = env
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).expect("finite"))?;
    let stop = (peak - drop_db).max(floor + FLOOR_MARGIN_DB);
    let mut end = env
        .iter()
        .enumerate()
        .skip(top)
        .find(|(_, &v)| v < stop)
        .map_or(env.len(), |(i, _)| i);
    let full = end.saturating_sub(top);
    let least = ((MIN_SPAN_S / dt_s).ceil() as usize)
        .max(4)
        .max((MIN_PREFIX_SHARE * full as f64).ceil() as usize);
    while end > top + least {
        let pts = &env[top..end];
        if let Some(line) = line_through(pts, dt_s) {
            let fall = -line.slope * (pts.len() - 1) as f64 * dt_s;
            if line.residual_db <= MAX_RESIDUAL_DB && fall >= MIN_MEASURABLE_DB {
                let half = pts.len() / 2;
                let (a, b) = (
                    line_through(&pts[..half], dt_s)?,
                    line_through(&pts[half..], dt_s)?,
                );
                let ratio = (a.slope / b.slope).abs();
                if a.slope < 0.0
                    && b.slope < 0.0
                    && (MAX_HALF_RATIO.recip()..=MAX_HALF_RATIO).contains(&ratio)
                {
                    return Some(TailFit {
                        t60_s: -60.0 / line.slope,
                        floor_db: peak - floor,
                        span_s: (pts.len() - 1) as f64 * dt_s,
                        residual_db: line.residual_db,
                    });
                }
            }
        }
        // Deterministic and geometric: a tenth of what is left each time, so a
        // long prefix is shortened in a bounded number of passes and the answer
        // does not depend on the sampling rate.
        let next = top + ((end - top) * 9) / 10;
        end = if next < end { next } else { end - 1 };
    }
    None
}

/// A least-squares line through an envelope in dB.
struct Line {
    /// dB per second; negative for a decay.
    slope: f64,
    residual_db: f64,
}

fn line_through(pts: &[f64], dt_s: f64) -> Option<Line> {
    if pts.len() < 4 {
        return None;
    }
    let n = pts.len() as f64;
    let mx = (n - 1.0) * dt_s / 2.0;
    let my = pts.iter().sum::<f64>() / n;
    let (mut num, mut den) = (0.0, 0.0);
    for (i, &y) in pts.iter().enumerate() {
        let dx = i as f64 * dt_s - mx;
        num += dx * (y - my);
        den += dx * dx;
    }
    if den <= 0.0 {
        return None;
    }
    let slope = num / den;
    let residual_db = (pts
        .iter()
        .enumerate()
        .map(|(i, &y)| {
            let e = y - (my + slope * (i as f64 * dt_s - mx));
            e * e
        })
        .sum::<f64>()
        / n)
        .sqrt();
    (slope.is_finite() && residual_db.is_finite()).then_some(Line { slope, residual_db })
}

/// The two instants a partial's fall is read between.
///
/// The same pair `estimate::brilliance::band_decay_gap` uses, and deliberately:
/// that band statistic is what `DECISIONS.md` 295 convicted the tail on and what
/// this milestone is gated against, and reading it on the **partial** instead of
/// on the band is the whole of the difference between a diagnosis and a fix. At
/// 0.1 s the strike is still sounding and at 1 s what is left is whatever
/// decayed slowest.
pub const INSTANTS: (f64, f64) = (0.1, 1.0);

/// This partial's level at each of the two [`INSTANTS`], in dB.
///
/// `None` for an instant at which the partial is within [`FLOOR_MARGIN_DB`] of
/// the signal's own floor — the same refusal as [`fit_tail_to`]'s and for the
/// same reason, and it is what stops a partial that is already over from
/// reporting the floor's own flatness as an eternal decay.
///
/// Two levels and not a fitted slope, because a fitted slope has to decide
/// *what* to fit — where the decay stops being one exponential, where a beat
/// null is, how far down the floor lets it look — and every one of those
/// decisions is made differently on the two signals. Two instants are the same
/// two instants on both, so their difference is free of all of it. This is what
/// the row is fitted from; [`fit_tail_to`] is what the audit *reports*, because
/// a T60 is what a decay rate means to a reader.
pub fn levels_at_instants(env: &[f64], dt_s: f64) -> Levels {
    let none = Levels {
        at: [None, None],
        resolvable_db: f64::NAN,
    };
    if env.len() < 4 || dt_s <= 0.0 {
        return none;
    }
    let from = ((FLOOR_FROM_S / dt_s) as usize).min(env.len());
    if from >= env.len() {
        return none;
    }
    let floor = median(env[from..].to_vec());
    if !floor.is_finite() {
        return none;
    }
    let resolvable_db = floor + FLOOR_MARGIN_DB;
    let at = |t: f64| -> Option<f64> {
        let i = (t / dt_s).round() as usize;
        let v = *env.get(i)?;
        (v.is_finite() && v >= resolvable_db).then_some(v)
    };
    Levels {
        at: [at(INSTANTS.0), at(INSTANTS.1)],
        resolvable_db,
    }
}

/// What one signal says about one partial at the two [`INSTANTS`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Levels {
    /// The level at each instant, or `None` where the partial is already inside
    /// this signal's own floor there.
    pub at: [Option<f64>; 2],
    /// The lowest level this signal can resolve: its own floor plus
    /// [`FLOOR_MARGIN_DB`]. A `None` above is not "no information" — it is the
    /// statement that the partial is **under** this number, which is a bound and
    /// is used as one.
    pub resolvable_db: f64,
}

impl Levels {
    /// The late level where it was measured, and the resolvable floor where it
    /// was not — an *upper* bound on where the partial actually is.
    pub fn late_or_bound(&self) -> f64 {
        self.at[1].unwrap_or(self.resolvable_db)
    }

    /// Whether the late reading is a bound rather than a measurement.
    pub fn late_is_bound(&self) -> bool {
        self.at[1].is_none()
    }
}

/// One partial of one key, measured on both signals.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PartialTail {
    pub k: usize,
    pub hz: f64,
    /// The common dB range both fits were held to: the smaller of the two
    /// signals' own [`measurable_db`].
    pub drop_db: f64,
    /// This partial's levels at the two [`INSTANTS`] on the engine's render.
    pub engine_db: Levels,
    /// The same on the recording.
    pub reference_db: Levels,
    pub engine: Option<TailFit>,
    pub reference: Option<TailFit>,
}

impl PartialTail {
    /// The factor this partial's `sigma` must be multiplied by for the engine's
    /// tail to be the recording's: the ratio of the two [`fall_db`]s, since a
    /// `sigma` *is* a fall per second.
    ///
    /// Both sides are measured through the same band-pass on the same partial of
    /// the same note, so every constant between a preset's `sigma` and a
    /// rendered envelope — the polarization split, the three strings, the
    /// radiated damping, the master gain — is in both and cancels. That is what
    /// makes this a correction to the *render* rather than to the schema, and it
    /// is why it can be closed on the render by iterating it.
    pub fn correction(&self) -> Option<f64> {
        let (e, r) = (self.engine_fall_db()?, self.reference_fall_db()?);
        if e <= MIN_FALL_DB || r <= MIN_FALL_DB {
            return None;
        }
        let c = r / e;
        c.is_finite().then_some(c)
    }

    /// Whether this partial is evidence about a *rate*.
    ///
    /// Only the two **early** readings have to be measurements — the partial has
    /// to have been there to decay — and both late readings may be the bound
    /// each signal's own floor puts on them ([`Levels::late_or_bound`]).
    ///
    /// "This partial is under the quietest thing this signal can resolve" is a
    /// *fact* and not a missing measurement, and `DECISIONS.md` 89 and 293's
    /// rule is never to **fit** inside a floor; using the floor as a ceiling on
    /// where a partial got to is the opposite of fitting inside it.
    ///
    /// Requiring the *engine's* late reading to be a measurement was tried first
    /// and is a **ratchet**, which is worth recording because it is invisible
    /// until the loop is iterated: every pass that damped the engine took its
    /// quietest partials under their own floor and out of the set, so what was
    /// left was the partials that had not moved, and the sum went on reporting
    /// "too slow" however much it had already been corrected. It drove the
    /// bass's own partials to 1.5 times the recording's rate
    /// (`DECISIONS.md` 304). With both bounds in, a rendered partial that is
    /// deep under its own floor reads a *large* fall — a render's floor is far
    /// deeper than a recording's — so the correction it asks for goes to one and
    /// the loop stops itself.
    pub fn trusted(&self) -> bool {
        self.engine_db.at[0].is_some() && self.reference_db.at[0].is_some()
    }

    /// How far this partial falls over [`INSTANTS`] on the engine's render, a
    /// lower bound where its late reading is one.
    pub fn engine_fall_db(&self) -> Option<f64> {
        Some(self.engine_db.at[0]? - self.engine_db.late_or_bound())
    }

    /// The same on the recording.
    pub fn reference_fall_db(&self) -> Option<f64> {
        Some(self.reference_db.at[0]? - self.reference_db.late_or_bound())
    }

    /// The same ratio taken off the two fitted T60s. Reported, never written:
    /// see [`fall_db`] for which of the two the row is fitted from.
    pub fn t60_ratio(&self) -> Option<f64> {
        let (e, r) = (self.engine?, self.reference?);
        let c = e.t60_s / r.t60_s;
        c.is_finite().then_some(c)
    }
}

/// Every partial of one key, measured on an engine render and on the recording.
///
/// `partial_hz` is the engine's own series for the key — the frequencies its
/// bank really rings at — and both signals are read at those frequencies,
/// because a band-pass placed at two different frequencies on the two signals
/// would be measuring the tuning and not the decay. The recording's partial is
/// inside the window's main lobe by construction: the compass's worst
/// engine-versus-recording partial error is a few cents.
pub fn measure_key(
    engine: &[f32],
    reference: &[f32],
    partial_hz: &[f64],
    f0_hz: f64,
    sample_rate: f64,
) -> Vec<PartialTail> {
    let e = partial_envelopes(engine, partial_hz, f0_hz, sample_rate);
    let r = partial_envelopes(reference, partial_hz, f0_hz, sample_rate);
    partial_hz
        .iter()
        .enumerate()
        .map(|(i, &hz)| {
            // The one number both fits are given: the smaller of what the two
            // signals' own floors allow. See `fit_tail_to`.
            let drop = match (measurable_db(&e[i], HOP_S), measurable_db(&r[i], HOP_S)) {
                (Some(a), Some(b)) => a.min(b),
                _ => f64::NAN,
            };
            PartialTail {
                k: i + 1,
                hz,
                drop_db: drop,
                engine_db: levels_at_instants(&e[i], HOP_S),
                reference_db: levels_at_instants(&r[i], HOP_S),
                engine: fit_tail_to(&e[i], HOP_S, drop),
                reference: fit_tail_to(&r[i], HOP_S, drop),
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// The correction a key's row carries
// ---------------------------------------------------------------------------

/// Fewest trusted partials in a band before its sum is a decay rather than one
/// partial's luck.
pub const MIN_BAND_CELLS: usize = 3;

/// Least share of a band the key's own partials must carry before a per-partial
/// `sigma` is what the band's decay is about at all.
///
/// `estimate::brilliance::MIN_TRIM_SHARE`'s device and its argument: under it
/// the band is the board's diffuse field, the strike noise or the sympathetic
/// halo rather than this note's string, and no `notes.partial_sigma_scale` cell
/// addresses it. A half and not a fiftieth because this is a *correction* and
/// not a solve — moving the partials until a band that is mostly not partials
/// lands on the recording is how A0's 2-6 kHz, where the bank has no partial at
/// all above 2512 Hz, came back with every one of its top cells over-damped by
/// a factor of 1.7 for no change in the band (`DECISIONS.md` 304).
pub const MIN_BAND_SHARE: f64 = 0.5;

/// Smallest fall, in dB, that is a decay rather than two readings of one level.
pub const MIN_FALL_DB: f64 = 0.5;

/// Largest factor one pass may put on a partial's `sigma`.
///
/// The schema's own rail (`MAX_PARTIAL_SIGMA_SCALE`) is 4, and a correction that
/// wants more than the whole legal range in a single pass is a measurement that
/// has gone wrong rather than a piano that decays sixteen times too slowly. The
/// loop that applies this is iterated and closed on the render, so a genuine
/// factor of four is reached in two passes and a spurious one is not reached at
/// all.
pub const MAX_PASS_FACTOR: f64 = 4.0;

/// What one band did between the two [`INSTANTS`] on both signals.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BandFall {
    /// How far the band's energy fell on the engine's render, dB.
    pub engine_db: f64,
    /// The same on the recording.
    pub reference_db: f64,
    /// Share of the band's power the key's own partials carry — one by
    /// construction where the two falls above are partial sums, and the field is
    /// kept because a caller that wants to drive the whole band has to say so.
    pub partial_share: f64,
    /// The **median** partial's own correction in this band: how much slower the
    /// typical partial is than the recording, where the two falls above are a
    /// sum and therefore belong to the partials that fall least.
    ///
    /// This is the stop, and it is why the two statistics are both here. A sum
    /// is the right objective — a band's energy is what the ear and the gate
    /// hear — and it is the wrong thing to run to convergence, because the only
    /// way to move a sum that a few slow partials own is to over-damp all of
    /// them. Driven to the sum alone the loop took the compass's median partial
    /// to 0.65 of the recording's rate and the realism board from 4.86 to 5.15
    /// dB of mean log-mel (`DECISIONS.md` 304). A band whose median partial has
    /// reached the recording's rate is a band this table has done what it can
    /// for.
    pub partial_median_ratio: f64,
    /// The width of the reference's own velocity-layer floor on this statistic:
    /// `band_decay_gap` between the recording and the layer next door.
    pub floor_db: f64,
    /// The standard error of [`Self::partial_median_ratio`], in `ln`.
    ///
    /// The stop is a comparison against **one**, and a median over a dozen
    /// partials whose ratios scatter by a factor of two either way does not
    /// resolve one to better than a few per cent. Without this the stop is a
    /// **ratchet**: a converged band reads 1.02 or 1.03 from noise alone, every
    /// pass multiplies that in, and eight passes over an already-fitted preset
    /// took the sampled tenor keys' 6-12 kHz band decay 2.5 dB further down for
    /// no measured change in their partials (`DECISIONS.md` 319). A band whose
    /// median is over one by less than its own error is a band this table
    /// cannot show is wrong, which is items 89 and 293's rule about floors
    /// applied to a statistic rather than to a level.
    pub partial_median_ratio_error: f64,
}

/// What one band's partials did over [`INSTANTS`] on **one** signal.
///
/// Two statistics of the same set of partials, because the fit needs both and
/// they are not the same number: a band's own energy is a **sum**, so it belongs
/// to the partials that fall least, while the median is what the typical partial
/// did. [`BandFall`] takes the first as its objective and the second as its
/// stop, and `DECISIONS.md` 306 is the measured case for each.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SideFall {
    /// How far the band's summed partial energy fell, dB.
    pub band_db: f64,
    /// How far the median partial of the band fell, dB, over the partials that
    /// fell at all ([`MIN_FALL_DB`]).
    pub median_db: f64,
    /// That median's own standard error, in `ln`: [`median_ln_error`] over the
    /// same partials. What it is for is [`BandFall::partial_median_ratio_error`].
    pub median_ln_error: f64,
    /// How many partials the band was read over.
    pub cells: usize,
}

/// What one band's **partials** did over [`INSTANTS`] on both signals.
///
/// This and not the whole band's own energy is what a `notes.partial_sigma_scale`
/// row can move, and the difference is not academic: driven on the whole band
/// over twelve passes the loop railed 28 % of its cells, drove the bass's own
/// partials to 1.7 times the recording's rate, and cost the realism board
/// 4.86 -> 5.15 dB of mean log-mel, while the band it was driving stopped
/// moving anyway (`DECISIONS.md` 304). A band holds the board's diffuse field,
/// the strike noise and the sympathetic halo as well as this note's string, and
/// what is not the string is not this table's to fix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PartialBandFall {
    pub engine: SideFall,
    pub reference: SideFall,
    /// The standard error of [`Self::median_ratio`], in `ln`. See
    /// [`BandFall::partial_median_ratio_error`].
    pub median_ratio_ln_error: f64,
    /// The median over the band's partials of one partial's own
    /// [`PartialTail::correction`] — every partial against *itself* on the two
    /// signals. This is the paired statistic and the one a sampled key's stop is
    /// taken from; a key with no recording has to make do with the ratio of the
    /// two medians, and [`DecayModel`] records how far apart the two are.
    pub median_ratio: f64,
}

/// Whether a key's own bank reaches the middle of a band.
///
/// **A rule about the population and not about the correction.** A key whose
/// series ends inside a band still has partials at that band's bottom edge, and
/// correcting *those* is exactly what this table is for — [`partial_band_fall`]
/// reads them and nothing else, so no board field or halo is in the number it
/// returns. What such a key cannot do is stand for the band in
/// [`DecayModel`]'s population: "how far this key's 6-12 kHz falls" read off
/// three partials sitting at 6.0-6.6 kHz is not the same quantity as the same
/// question asked of a key with thirty of them spread over the octave.
///
/// Measured, and it is why the rule exists: C2's bank ends at 6.6 kHz and its
/// three top partials fell **2.5 dB** over the interval where every key from C3
/// up falls 18-45 dB, and that one point was in the population every unsampled
/// key takes its 6-12 kHz target from. It alone took the target at C4 from
/// 15.2 dB to 12.6 and the population's scatter from x1.36 to x1.52
/// (`DECISIONS.md` 320).
pub fn bank_owns_band(tails: &[PartialTail], bnd: (f64, f64)) -> bool {
    tails.last().is_some_and(|t| t.hz >= band_centre(bnd))
}

/// [`PartialBandFall`] for one band of one key, `None` where the key's bank does
/// not reach the band ([`bank_owns_band`]) or where fewer than
/// [`MIN_BAND_CELLS`] of its partials are evidence.
///
/// A partial counts only where it is evidence — [`PartialTail::trusted`] — and
/// the recording's late reading may be the bound its own floor puts on it,
/// which under-states the recording's fall and therefore the correction.
pub fn partial_band_fall(tails: &[PartialTail], bnd: (f64, f64)) -> Option<PartialBandFall> {
    let cells: Vec<&PartialTail> = tails
        .iter()
        .filter(|t| t.hz >= bnd.0 && t.hz < bnd.1 && t.trusted())
        .collect();
    if cells.len() < MIN_BAND_CELLS {
        return None;
    }
    let side = |early: fn(&PartialTail) -> Option<f64>,
                late: fn(&PartialTail) -> f64,
                fall: fn(&PartialTail) -> Option<f64>| {
        let power = |db: f64| 10f64.powf(db / 10.0);
        let sum_early: f64 = cells.iter().filter_map(|t| early(t)).map(power).sum();
        let sum_late: f64 = cells.iter().map(|t| power(late(t))).sum();
        let falls: Vec<f64> = cells
            .iter()
            .filter_map(|t| fall(t))
            .filter(|f| *f > MIN_FALL_DB)
            .collect();
        SideFall {
            band_db: 10.0 * (sum_early.max(1e-30) / sum_late.max(1e-30)).log10(),
            median_db: median(falls.clone()),
            median_ln_error: median_ln_error(&falls),
            cells: cells.len(),
        }
    };
    Some(PartialBandFall {
        engine: side(
            |t| t.engine_db.at[0],
            |t| t.engine_db.late_or_bound(),
            |t| t.engine_fall_db(),
        ),
        reference: side(
            |t| t.reference_db.at[0],
            |t| t.reference_db.late_or_bound(),
            |t| t.reference_fall_db(),
        ),
        median_ratio: median(cells.iter().filter_map(|t| t.correction()).collect()),
        median_ratio_ln_error: median_ln_error(
            &cells.iter().filter_map(|t| t.correction()).collect::<Vec<_>>(),
        ),
    })
}

/// The standard error of the **median** of `values`, in `ln`.
///
/// `1.2533 sigma / sqrt(n)` — the median's own asymptotic error — with `sigma`
/// taken robustly as `1.4826 MAD` of the logs, because these are ratios and
/// falls, whose scatter is multiplicative, and because one partial caught in a
/// beat null must not be allowed to widen it.
///
/// Zero for fewer than [`MIN_BAND_CELLS`] values, which is a set nothing is
/// fitted from anyway.
pub fn median_ln_error(values: &[f64]) -> f64 {
    let logs: Vec<f64> = values
        .iter()
        .filter(|v| v.is_finite() && **v > 0.0)
        .map(|v| v.ln())
        .collect();
    if logs.len() < MIN_BAND_CELLS {
        return 0.0;
    }
    let centre = median(logs.clone());
    let mad = median(logs.iter().map(|v| (v - centre).abs()).collect());
    1.2533 * 1.4826 * mad / (logs.len() as f64).sqrt()
}

/// The same for a band with no recording behind it: the engine's side alone,
/// over the partials the *engine* still resolves.
///
/// A drawn band's target comes from [`DecayModel`] and its engine side is
/// measured here, which is what makes the close on the render the same close a
/// sampled key gets — item 300's rule that what may be drawn is what the
/// instrument does and what must be measured is what the engine does about it.
pub fn engine_band_fall(tails: &[PartialTail], bnd: (f64, f64)) -> Option<SideFall> {
    let cells: Vec<&PartialTail> = tails
        .iter()
        .filter(|t| t.hz >= bnd.0 && t.hz < bnd.1 && t.engine_db.at[0].is_some())
        .collect();
    if cells.len() < MIN_BAND_CELLS {
        return None;
    }
    let power = |db: f64| 10f64.powf(db / 10.0);
    let early: f64 = cells.iter().filter_map(|t| t.engine_db.at[0]).map(power).sum();
    let late: f64 = cells
        .iter()
        .map(|t| power(t.engine_db.late_or_bound()))
        .sum();
    let falls: Vec<f64> = cells
        .iter()
        .filter_map(|t| t.engine_fall_db())
        .filter(|f| *f > MIN_FALL_DB)
        .collect();
    Some(SideFall {
        band_db: 10.0 * (early.max(1e-30) / late.max(1e-30)).log10(),
        median_db: median(falls.clone()),
        median_ln_error: median_ln_error(&falls),
        cells: cells.len(),
    })
}

/// The highest partial index a key's recording still measured a fall on: the
/// reach a row may be written to, and no further.
pub fn reach(tails: &[PartialTail]) -> usize {
    tails
        .iter()
        .filter(|t| t.correction().is_some())
        .map(|t| t.k)
        .max()
        .unwrap_or(0)
}

/// A key's measured decay correction, summarised as one number per band.
///
/// Two medians and a line between them, and not a per-partial table, for the
/// reason `estimate::brilliance::continuation_db` is a line: the per-partial
/// scatter here is a factor of two either way (`DECISIONS.md` 304's own
/// numbers) and writing it cell by cell would put that scatter into the decay
/// of every partial of every note. What is being corrected is a **law** —
/// `sigma0 + sigma1 (f/1000)^2`, one curve for the whole key — so what replaces
/// it has to be smooth in `ln f` too, and the two bands the gate is scored in
/// are where the evidence is.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TailCorrection {
    /// The factor to multiply a partial's `sigma` by at the geometric centre of
    /// [`HF1`] and of [`HF2`]; `None` where the band said nothing.
    pub band: [Option<f64>; 2],
    /// Whether each band was offered at all.
    pub cells: [usize; 2],
    /// The highest partial index with a correction on both signals: the reach
    /// the *recordings* license a row to be written to.
    pub reach: usize,
}

impl TailCorrection {
    /// One key's correction, one band at a time, off the band's own decay.
    ///
    /// A `sigma` is a fall per second, so the factor a band's partials need is
    /// the ratio of the two falls. Three rules are in here and each is a way
    /// this goes wrong:
    ///
    /// * **The band and not the partial.** A median of the per-partial ratios
    ///   was tried first and is wrong, and the control is in `DECISIONS.md` 304:
    ///   driven to a median of 1.00 at every register, the tenor's own 2-6 kHz
    ///   band decay gap moved 10.2 -> 7.6 dB and no further. A band's energy is a
    ///   **sum**, so it belongs to the partials that fall least, and a median
    ///   weights the partial that is already right as heavily as the one that
    ///   never decays at all. What is corrected here is what the gate measures.
    /// * **The floor is a stop and not a tolerance.** `floor_db` is
    ///   `band_decay_gap` taken between the recording and the velocity layer next
    ///   door — two recordings of one piano — so a band already inside it is a
    ///   band this statistic cannot tell from the reference, and correcting past
    ///   it is fitting the recording's own noise. That is `DECISIONS.md` 293's
    ///   lesson taken to the band.
    /// * **A band that is not falling cannot say by how much it is wrong**, only
    ///   that one pass is not enough; it gets [`MAX_PASS_FACTOR`], and the loop
    ///   that calls this is iterated.
    /// * **A band the key's partials do not own is not this table's to move** —
    ///   [`MIN_BAND_SHARE`] — and a band whose *median* partial has already
    ///   reached the recording's rate is one this table has done what it can
    ///   for — [`BandFall::partial_median_ratio`].
    /// * **"Reached" means inside the median's own standard error**, not "not
    ///   greater than one" — [`BandFall::partial_median_ratio_error`]. A stop at
    ///   exactly one is a ratchet, because a converged band reads a per cent or
    ///   two over it from noise alone and every pass multiplies that in.
    pub fn from_band_falls(bands: [Option<BandFall>; 2], reach: usize) -> Self {
        let mut out = Self { reach, ..Self::default() };
        for (i, fall) in bands.into_iter().enumerate() {
            let Some(fall) = fall else { continue };
            out.cells[i] = 1;
            if fall.partial_share < MIN_BAND_SHARE
                || fall.partial_median_ratio.ln() <= fall.partial_median_ratio_error
                || fall.reference_db <= MIN_FALL_DB
                || (fall.reference_db - fall.engine_db).abs() <= fall.floor_db
            {
                continue;
            }
            out.band[i] = Some(
                fall.partial_median_ratio
                    .clamp(MAX_PASS_FACTOR.recip(), MAX_PASS_FACTOR),
            );
        }
        out
    }

    /// The same correction taken `share` of the way, in the log domain.
    ///
    /// A band holds partials the correction does not own and a sum is not
    /// linear in its terms, so one pass's solve is exact only to first order;
    /// damping turns that into convergence instead of ringing. The same device
    /// and the same reason as `brilliance.rs`'s `TRIM_DAMPING`.
    pub fn damped(&self, share: f64) -> Self {
        Self {
            band: self.band.map(|b| b.map(|v| v.powf(share))),
            ..*self
        }
    }

    /// Whether anything was measured well enough to write.
    pub fn is_empty(&self) -> bool {
        self.band[0].is_none() && self.band[1].is_none()
    }

    /// The factor partial at `hz` has its `sigma` multiplied by.
    ///
    /// The same shape as `estimate::brilliance::continuation_db` and for the
    /// same reasons: one at and below [`HF1`]'s bottom edge, ramped to the lower
    /// band's median at that band's geometric centre, a straight line in `ln f`
    /// to the upper band's median at its centre, and held flat above it. Held
    /// flat and not extrapolated, because a power law taken three octaves past
    /// its last evidence is an assertion; and one below 2 kHz, because that is
    /// where the fitted rows already are and where the measurement says the
    /// engine is already right (`DECISIONS.md` 304: the bass reads 1.07 over the
    /// whole compass and the tenor 1.05 under 1 kHz).
    ///
    /// **A band that was offered and refused is one, not the other band's
    /// factor.** [`Self::cells`] is what tells the two apart: a band nobody
    /// could read at all (no cells) takes its neighbour's factor, because that
    /// is the only evidence there is and a key whose top octave resolves three
    /// partials in one band and two in the other should not have a shelf between
    /// them; but a band whose own partials said *stop* — its median has reached
    /// the recording's rate, or its whole gap is inside the recording's own
    /// velocity-layer floor — gets one, and the ramp carries the curve down to
    /// it. Extending the lower band's damping over a refusal is a **ratchet**:
    /// the tenor's 2-6 kHz goes on asking for a little more every pass, and
    /// before this rule every one of those passes damped 6-12 kHz too, so that
    /// eight further passes over an already-fitted preset took the sampled tenor
    /// keys' 6-12 kHz band 2.5 dB further down while their own partials did not
    /// move. [`BandFall::partial_median_ratio_error`] is the other half of the
    /// same defect (`DECISIONS.md` 319).
    pub fn at(&self, hz: f64) -> f64 {
        if self.band[0].is_none() && self.band[1].is_none() {
            return 1.0;
        }
        // A refused band is one; a band with nothing in it borrows its
        // neighbour's factor.
        let factor = |i: usize| -> f64 {
            self.band[i].unwrap_or(if self.cells[i] > 0 {
                1.0
            } else {
                self.band[1 - i].unwrap_or(1.0)
            })
        };
        let (b0, b1) = (factor(0), factor(1));
        let (c1, c2) = (band_centre(HF1), band_centre(HF2));
        let ln = if hz <= HF1.0 {
            0.0
        } else if hz < c1 {
            b0.ln() * (hz / HF1.0).ln() / (c1 / HF1.0).ln()
        } else if hz >= c2 {
            b1.ln()
        } else {
            b0.ln() + (b1.ln() - b0.ln()) * (hz / c1).ln() / (c2 / c1).ln()
        };
        ln.exp().clamp(MAX_PASS_FACTOR.recip(), MAX_PASS_FACTOR)
    }
}

/// One key's `notes.partial_sigma_scale` row with a measured correction on it.
///
/// The existing cells are **multiplied**, not replaced: a cell the decay stage
/// wrote is a measurement of that partial against its own recording
/// (`DECISIONS.md` 200-201) and this is a measurement of the same partial's
/// *rate* against the same recording, so the two compose. Cells the row never
/// had are 1.0 and become the correction alone, which is the whole of the
/// extension.
///
/// The row stops at `reach` — the highest partial the recording still measured a
/// fall on — and at the key's own bank. Past that the recording has nothing to
/// say and the cell stays 1.0, which is `string::PartialShaping`'s own
/// convention for a partial nothing was learned about.
pub fn extend_row(row: &[f32], partial_hz: &[f64], correction: &TailCorrection) -> Vec<f32> {
    let top = correction.reach.min(partial_hz.len());
    let mut out: Vec<f32> = (1..=top.max(row.len().min(partial_hz.len())))
        .map(|k| {
            let was = row.get(k - 1).copied().unwrap_or(1.0);
            let factor = if k <= top {
                correction.at(partial_hz[k - 1])
            } else {
                1.0
            };
            (f64::from(was) * factor).clamp(
                f64::from(crate::preset::MIN_PARTIAL_SIGMA_SCALE),
                f64::from(crate::preset::MAX_PARTIAL_SIGMA_SCALE),
            ) as f32
        })
        .collect();
    while out.last() == Some(&1.0) {
        out.pop();
    }
    out
}

/// The same row with its sub-[`LOW_BAND`]`.1` cells multiplied by `factor`.
///
/// The band [`extend_row`] never touches, written once per run rather than per
/// pass: [`LowDecay`] is a statement about the *piano* read off the keys the
/// library sampled, not a correction measured on this render, so iterating it
/// would compound a number that has no fixed point to converge to. The passes
/// that follow close the two high bands on a render that already has it, which
/// is the same order a sampled key's own row was built in — `shaping` first and
/// the correction on top.
pub fn low_row(row: &[f32], partial_hz: &[f64], factor: f64) -> Vec<f32> {
    let low = partial_hz.iter().take_while(|&&hz| hz < LOW_BAND.1).count();
    if low < MIN_LOW_CELLS || !(factor.is_finite() && factor > 0.0) {
        return row.to_vec();
    }
    let mut out: Vec<f32> = (1..=low.max(row.len()))
        .map(|k| {
            let was = f64::from(row.get(k - 1).copied().unwrap_or(1.0));
            let scaled = if k <= low { was * factor } else { was };
            scaled.clamp(
                f64::from(crate::preset::MIN_PARTIAL_SIGMA_SCALE),
                f64::from(crate::preset::MAX_PARTIAL_SIGMA_SCALE),
            ) as f32
        })
        .collect();
    while out.last() == Some(&1.0) {
        out.pop();
    }
    out
}

/// A band's geometric centre, where [`TailCorrection::at`]'s line is pinned.
pub fn band_centre((lo, hi): (f64, f64)) -> f64 {
    (lo * hi).sqrt()
}

// ---------------------------------------------------------------------------
// What a key nobody could measure draws from
// ---------------------------------------------------------------------------

/// One sampled key's contribution to [`DecayModel`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecayPoint {
    pub key: u8,
    /// How far the *recording's* band fell over [`INSTANTS`], dB, per band —
    /// [`SideFall::band_db`], the sum.
    pub reference_fall_db: [Option<f64>; 2],
    /// How far the *recording's* median partial of that band fell, dB, per band
    /// — [`SideFall::median_db`], the stop's own statistic.
    pub reference_partial_fall_db: [Option<f64>; 2],
    /// The **frequency** of the highest partial that recording still measured a
    /// fall on, Hz. A frequency and not an index: see [`Ceiling`].
    pub ceiling_hz: f64,
}

/// What one key with no recording of its own is closed against.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DrawnDecay {
    /// The band fall this key's render is closed on, dB — a **target** taken
    /// from the compass around it, not a measurement of this key.
    pub target_fall_db: [f64; 2],
    /// The median partial fall for the same band, the same way. This is what the
    /// stop is taken against, because the engine's side of the stop is a median
    /// partial too.
    pub target_partial_fall_db: [f64; 2],
    /// The frequency the row may be written up to. See [`Ceiling`].
    pub ceiling_hz: f64,
}

/// The frequency above which the recordings measured nothing, one number for
/// the whole library.
///
/// **A frequency and not a partial index, and a constant and not a line.** How
/// far up a key's series a recording can be read is `min(bank, ceiling / f0)`,
/// and both of those factors are already known for any key: the bank is the
/// preset's and `f0` is the key's. What is *not* known for an unsampled key is
/// the ceiling, and that is a property of the recording chain's own noise floor
/// against a piano's spectrum — the same microphones and the same room for
/// every key in the library.
///
/// `DECISIONS.md` 307 drew the reach as a log-line in the partial *index*
/// instead, which is the same quantity divided by a `f0` that spans seven
/// octaves, and the result was biased where the two effects part company: the
/// bass's reach is its **bank** ending at 2.5-7 kHz and the tenor's is this
/// ceiling at 9-15 kHz, so one line through both under-predicted the middle by
/// half an octave and left the drawn tenor rows stopping at 7 kHz, in the middle
/// of [`HF2`] and short of the band they were being corrected for
/// (`DECISIONS.md` 320). Leave-one-out over the 30 sampled keys, predicting each
/// key's own measured reach from a model fitted without it: the index line is
/// out by a factor of **1.20** at the median key and this ceiling by **1.07**.
///
/// **The scatter is reported and never drawn.** Where one key's recording ran
/// out is a fact about the recording, not about the piano, and item 300's rule
/// is that what may be drawn is what the *instrument* does. Drawing it would put
/// a random edge — the frequency at which a row stops correcting — at a
/// different place on every neighbouring key, which is the seam this milestone
/// is closing rather than a per-key individuality worth having.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Ceiling {
    /// The median of the sampled keys' own ceilings, Hz.
    pub hz: f64,
    /// Their scatter about it as a factor: the median absolute deviation in
    /// `ln`, exponentiated. Reported, not drawn.
    pub spread: f64,
    pub points: usize,
}

impl Ceiling {
    pub fn fit(ceilings: &[f64]) -> Ceiling {
        let usable: Vec<f64> = ceilings
            .iter()
            .copied()
            .filter(|hz| hz.is_finite() && *hz > 0.0)
            .collect();
        if usable.is_empty() {
            return Ceiling::default();
        }
        let hz = median(usable.clone());
        Ceiling {
            hz,
            spread: median(usable.iter().map(|v| (v / hz).ln().abs()).collect()).exp(),
            points: usable.len(),
        }
    }
}

/// Pearson `r` of a set of pairs, zero where either side does not vary.
fn correlation(pairs: &[(f64, f64)]) -> f64 {
    let n = pairs.len() as f64;
    if pairs.len() < 3 {
        return 0.0;
    }
    let (mx, my) = (
        pairs.iter().map(|p| p.0).sum::<f64>() / n,
        pairs.iter().map(|p| p.1).sum::<f64>() / n,
    );
    let sxx: f64 = pairs.iter().map(|p| (p.0 - mx).powi(2)).sum();
    let syy: f64 = pairs.iter().map(|p| (p.1 - my).powi(2)).sum();
    let sxy: f64 = pairs.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
    if sxx <= 0.0 || syy <= 0.0 {
        return 0.0;
    }
    sxy / (sxx * syy).sqrt()
}

/// How far up a key's own series a ceiling in hertz reaches: the number of
/// leading partials at or under it.
///
/// `partial_hz` is ascending, so this is a `take_while` and it is bounded by the
/// key's own bank without a second rule.
pub fn reach_to(partial_hz: &[f64], ceiling_hz: f64) -> usize {
    partial_hz.iter().take_while(|&&hz| hz <= ceiling_hz).count()
}

/// What a key with no recording of its own is closed against: the compass's own
/// decay, read off the keys the library did sample.
///
/// Two layers, and the second is what `DECISIONS.md` 320 added. A **line** in
/// `ln` through the sampled keys, per band, which is the register trend and
/// carries the whole compass; and the sampled keys' own **residuals about that
/// line, interpolated** in the key, which is the local truth. A key three
/// semitones from a sampled one is given that key's own departure from the
/// trend, faded linearly into the next sampled key's, and the line alone where
/// there is no sampled key on either side.
///
/// **Why not a draw of the line's scatter, which is what item 307 shipped.** The
/// two are the same on average and completely different key by key, and the
/// difference is measurable on the render: a target drawn from a distribution of
/// scatter x1.31-1.45 lands a *random* factor away from what the sampled key
/// next door measured, and the fit then closes each key on its own target, so
/// the row of a drawn key and the row of the sampled key beside it end a factor
/// of 1.5 apart in the tenor and A4's 6-12 kHz cells sit at the schema's rail
/// with A#4's at 1.26. That is the fitted-against-drawn seam item 320 was opened
/// on, and it is not something more passes fix: both keys are at their own fixed
/// points. A decay rate is not an idiosyncrasy — `TUNING.md`'s stage 1 fits
/// *every* smooth quantity as a curve across the compass and lets all 88 keys
/// read from it — so what the unsampled keys want is the compass's own curve
/// through the measurements, and item 284's rule that no number is read off
/// another key's recording is a rule about **individuality** (a false beat, a
/// gain scatter), which this is not. `estimate::texture`'s draws are untouched
/// and keep their scatter for exactly that reason.
///
/// What is taken from the compass is still the **target**, never the correction.
/// A correction is a statement about this engine's render, which changes every
/// time the preset does; a band fall is a statement about the piano, which does
/// not. The key is then closed on the render against that target exactly as a
/// sampled key is closed against its own recording — item 300's `BeatCeiling`
/// pattern, and the same reason: what may be drawn is what the instrument does,
/// and what must be measured is what the engine does about it.
///
/// **Two targets and not one**, because a sampled key's fit uses two statistics
/// of its own recording and a key without one has to answer both: the band's
/// summed fall is what the gate is scored on and the median partial's fall is
/// what the stop is taken on. Taking only the first and dividing it by the
/// engine's *median* partial fall — which is what item 307 shipped — compares a
/// sum against a median, and a sum always falls less than the median of its own
/// terms, so every drawn key stopped early. Measured on the shipped preset: the
/// drawn tenor keys' 6-12 kHz band decay gap sat at +9.03 dB against the sampled
/// keys' -2.10 (`DECISIONS.md` 320).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DecayModel {
    /// `exp(a + b·key)` dB of fall over [`INSTANTS`] in [`HF1`] and [`HF2`]:
    /// the band's own sum.
    pub fall: [LogLine; 2],
    /// The same for the band's median partial.
    pub partial_fall: [LogLine; 2],
    /// How far up the row may be written, as a frequency.
    pub ceiling: Ceiling,
    /// How far the two lines' residuals move together, per band: the Pearson
    /// `r` of `ln fall` against `ln partial_fall` about their own lines, over
    /// the keys that measured both. Reported rather than used, now that the
    /// residuals are interpolated rather than drawn: it says whether the two
    /// statistics are one fact about a key, which is the assumption under
    /// interpolating them separately as well.
    pub residual_correlation: [f64; 2],
    /// The sampled keys themselves, ascending: what the residuals are
    /// interpolated between.
    pub points: Vec<DecayPoint>,
}

impl DecayModel {
    pub fn fit(points: &[DecayPoint]) -> DecayModel {
        let line = |pick: fn(&DecayPoint, usize) -> Option<f64>, i: usize| {
            LogLine::fit(
                &points
                    .iter()
                    .filter_map(|p| Some((f64::from(p.key), pick(p, i)?)))
                    .collect::<Vec<_>>(),
            )
        };
        let fall = [0usize, 1].map(|i| line(|p, i| p.reference_fall_db[i], i));
        let partial_fall = [0usize, 1].map(|i| line(|p, i| p.reference_partial_fall_db[i], i));
        let mut sorted = points.to_vec();
        sorted.sort_by_key(|p| p.key);
        DecayModel {
            residual_correlation: [0usize, 1].map(|i| {
                let pairs: Vec<(f64, f64)> = points
                    .iter()
                    .filter_map(|p| {
                        let (a, b) = (p.reference_fall_db[i]?, p.reference_partial_fall_db[i]?);
                        (a > 0.0 && b > 0.0).then(|| {
                            (
                                a.ln() - fall[i].at(p.key).ln(),
                                b.ln() - partial_fall[i].at(p.key).ln(),
                            )
                        })
                    })
                    .collect();
                correlation(&pairs)
            }),
            fall,
            partial_fall,
            ceiling: Ceiling::fit(&points.iter().map(|p| p.ceiling_hz).collect::<Vec<_>>()),
            points: sorted,
        }
    }

    /// One key's target. A pure function of the sampled keys' measurements and
    /// the key number — nothing is drawn, so a re-emitted preset is the same
    /// preset without a seed having to say so.
    pub fn draw(&self, key: u8) -> DrawnDecay {
        DrawnDecay {
            target_fall_db: [0usize, 1]
                .map(|b| self.at(key, &self.fall[b], |p| p.reference_fall_db[b])),
            target_partial_fall_db: [0usize, 1]
                .map(|b| self.at(key, &self.partial_fall[b], |p| p.reference_partial_fall_db[b])),
            ceiling_hz: self.ceiling.hz,
        }
    }

    /// The line at `key` times the sampled keys' own departure from it,
    /// interpolated: exact at a sampled key, linear in the key between two, and
    /// held flat past the last one on either side.
    fn at(&self, key: u8, line: &LogLine, pick: impl Fn(&DecayPoint) -> Option<f64>) -> f64 {
        let residual: Vec<(f64, f64)> = self
            .points
            .iter()
            .filter_map(|p| {
                let v = pick(p)?;
                (v > 0.0 && line.at(p.key) > 0.0)
                    .then(|| (f64::from(p.key), v.ln() - line.at(p.key).ln()))
            })
            .collect();
        line.at(key) * interpolate(&residual, f64::from(key)).exp()
    }
}

// ---------------------------------------------------------------------------
// The band below the correction curve, which only a sampled key ever had
// ---------------------------------------------------------------------------

/// The band [`TailCorrection::at`] holds at one: everything under [`HF1`].
///
/// Not a band this module ever *measures* — it is where `estimate::shaping`'s
/// per-partial cells live, and the correction curve deliberately stops there
/// (`DECISIONS.md` 304, and [`TailCorrection::at`]'s own doc). It is named
/// because the seam of `DECISIONS.md` 334 is exactly its edge.
pub const LOW_BAND: (f64, f64) = (0.0, HF1.0);

/// How many cells a key must have under [`LOW_BAND`]`.1` before its
/// [`low_mean`] is a **band** statistic rather than one partial's own
/// idiosyncrasy — and, on the other side, before [`LowDecay`] will write one.
///
/// [`MIN_BAND_CELLS`]'s own number and its own reason, and it is not a
/// formality: on the shipped preset the sampled keys carry 67 cells here at A0
/// and **one** from C6 up, and those single-cell keys read 0.42, 1.00 and 0.67
/// — a factor of two either way, which is the per-partial scatter
/// (`DECISIONS.md` 285's 4.5 dB roughness) and not a register trend. Fitting
/// the line through them and then writing the result onto the top octave
/// lengthened the fundamental of six keys above E6 by a factor of 1.6-1.8 and
/// took the compass from **9 flags to 13** — four new ones on `jitter`, which
/// is what a unison whose partners are held up against each other for longer
/// does. Refused as a population point and refused as a target, both for the
/// same reason: three partials is where this stops being one partial.
pub const MIN_LOW_CELLS: usize = MIN_BAND_CELLS;

/// The geometric mean of a `notes.partial_sigma_scale` row's cells under
/// [`LOW_BAND`]`.1`, and how many there were.
///
/// This is the one number a sampled key's row says about its own fundamental
/// region, and it is read off the *preset* rather than off a render because
/// that is where the quantity lives: `estimate::shaping` measured it from the
/// recording's own per-partial T60s and nothing since has touched it.
pub fn low_mean(row: &[f32], partial_hz: &[f64]) -> Option<(f64, usize)> {
    let cells: Vec<f64> = row
        .iter()
        .zip(partial_hz)
        .filter(|(_, &hz)| hz < LOW_BAND.1)
        .map(|(&v, _)| f64::from(v))
        .filter(|v| v.is_finite() && *v > 0.0)
        .collect();
    if cells.len() < MIN_LOW_CELLS {
        return None;
    }
    let n = cells.len();
    Some((
        (cells.iter().map(|v| v.ln()).sum::<f64>() / n as f64).exp(),
        n,
    ))
}

/// What the sub-[`HF1`] half of a `partial_sigma_scale` row is at a key nobody
/// recorded.
///
/// **The band the draw never covered, and the seam it left.** `DECISIONS.md`
/// 320 made the two high bands a compass quantity — the line through the
/// sampled keys times their own interpolated departures from it — and left this
/// one alone, because [`TailCorrection::at`] returns exactly 1.0 at and below
/// [`HF1`]'s bottom edge and there was therefore nothing for a *correction* to
/// draw. But the row under 2 kHz is not empty at a sampled key: `shaping`
/// writes it, and on the shipped preset **every one of the 24 sampled keys that
/// has cells there has a geometric mean below one** (1.00 at A0, 0.90 at A3,
/// 0.75 at C4, 0.59 at A4, 0.39 at C5) while **all 37 drawn keys read exactly
/// 1.000**, because nothing ever wrote them. That is a fitted/unfitted step in
/// one band, and it is the largest thing the melody gate's tail `hf` column can
/// see: the column is a *share*, its denominator is the fundamental, and C4's
/// own cells hold its fundamental 4.2 dB higher at 0.5 s than the law alone
/// would where D4's and E4's do not (`DECISIONS.md` 334).
///
/// **Drawn as a line and not as a scatter**, which is item 320(d)'s argument
/// one band lower and its evidence is the same: over the sampled keys the
/// residual about `exp(a + b·key)` has a lag-1 autocorrelation across the
/// compass of **+0.05**, so the departure of one key is not evidence about the
/// next one and interpolating it is all that may be done with it; drawing its
/// ×1.24 scatter instead would land a *random* factor from what the sampled key
/// next door measured, which is the seam and not the cure.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LowDecay {
    /// `exp(a + b·key)` through the sampled keys' own [`low_mean`].
    pub line: LogLine,
    /// Those keys and their means, ascending: what the residual is interpolated
    /// between.
    pub points: Vec<(u8, f64)>,
}

impl LowDecay {
    pub fn fit(points: &[(u8, f64)]) -> LowDecay {
        let mut sorted = points.to_vec();
        sorted.sort_by_key(|p| p.0);
        LowDecay {
            line: LogLine::fit(
                &sorted
                    .iter()
                    .map(|&(k, v)| (f64::from(k), v))
                    .collect::<Vec<_>>(),
            ),
            points: sorted,
        }
    }

    /// The factor a key's sub-[`HF1`] cells are multiplied by: the line times
    /// the sampled keys' interpolated departure from it, exactly
    /// [`DecayModel::at`]. One where nothing was fitted at all.
    pub fn at(&self, key: u8) -> f64 {
        if self.points.is_empty() {
            return 1.0;
        }
        let residual: Vec<(f64, f64)> = self
            .points
            .iter()
            .filter_map(|&(k, v)| {
                let line = self.line.at(k);
                (v > 0.0 && line > 0.0).then(|| (f64::from(k), v.ln() - line.ln()))
            })
            .collect();
        let at = self.line.at(key) * interpolate(&residual, f64::from(key)).exp();
        if at.is_finite() && at > 0.0 {
            at
        } else {
            1.0
        }
    }
}

/// How far past the last key that measured a band its own departure from the
/// line is still carried, in semitones, before the model is the line alone.
///
/// An octave, and it is load-bearing rather than a taste: the last sampled key
/// that measures 2-6 kHz at all is D#6, and its own band fall is **9.2 dB**
/// where C6 three semitones below reads 34.2. Carried flat, that one reading
/// governs the twenty-one keys above it and took the top octave's band decay
/// gap from +10.35 to +11.38 dB and three keys' rows away with it. A local
/// departure is evidence about its own neighbourhood; an octave away it is not
/// evidence at all, and what is left is the compass's own line
/// (`DECISIONS.md` 320).
pub const RESIDUAL_TAPER_KEYS: f64 = 12.0;

/// Linear interpolation through `(x, y)` ascending, faded to zero over
/// [`RESIDUAL_TAPER_KEYS`] outside them, and zero where there is nothing.
fn interpolate(points: &[(f64, f64)], x: f64) -> f64 {
    let taper = |y: f64, distance: f64| y * (1.0 - distance / RESIDUAL_TAPER_KEYS).max(0.0);
    let Some(&(first_x, first_y)) = points.first() else {
        return 0.0;
    };
    if x <= first_x {
        return taper(first_y, first_x - x);
    }
    let Some(&(last_x, last_y)) = points.last() else {
        return 0.0;
    };
    if x >= last_x {
        return taper(last_y, x - last_x);
    }
    let i = points
        .iter()
        .position(|&(px, _)| px > x)
        .expect("x is under the last point");
    let ((x0, y0), (x1, y1)) = (points[i - 1], points[i]);
    if x1 <= x0 {
        return y0;
    }
    y0 + (y1 - y0) * (x - x0) / (x1 - x0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f64 = 48_000.0;

    /// A sum of decaying sinusoids, sampled the way a render is.
    fn note(f0: f64, partials: &[(usize, f64, f64)], seconds: f64) -> Vec<f32> {
        let n = (seconds * SR) as usize;
        (0..n)
            .map(|i| {
                let t = i as f64 / SR;
                partials
                    .iter()
                    .map(|&(k, amp, t60)| {
                        amp * 10f64.powf(-3.0 * t / t60)
                            * (std::f64::consts::TAU * f0 * k as f64 * t).sin()
                    })
                    .sum::<f64>() as f32
            })
            .collect()
    }

    /// A stationary noise floor, which is what separates a recording from a
    /// synthetic exponential: without one, the quietest partial's envelope runs
    /// down onto the f32 sum's own rounding, which is not stationary and not a
    /// floor anything here is written for.
    fn hiss(signal: &[f32], amplitude: f32) -> Vec<f32> {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        signal
            .iter()
            .map(|&x| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let u = (state >> 40) as f32 / 8_388_608.0 - 1.0;
                x + amplitude * u
            })
            .collect()
    }

    #[test]
    fn the_window_is_the_notes_own_spacing_and_not_the_partials() {
        // The defect this module exists not to have: a filter sized from the
        // partial is wider than the partial spacing as soon as k passes four.
        for (f0, want_at_least) in [(27.5, 1 << 14), (261.63, 1 << 11), (2093.0, 1 << 8)] {
            let n = window_for(f0, SR);
            assert!(
                n >= want_at_least,
                "f0 {f0} got a window of {n}, wanted {want_at_least} or more"
            );
            // The main lobe's half-width is 2 sr / N and must clear f0 / 2.
            assert!(
                2.0 * SR / n as f64 <= f0,
                "f0 {f0}: main lobe {:.1} Hz against a spacing of {f0}",
                4.0 * SR / n as f64
            );
        }
    }

    #[test]
    fn a_known_per_partial_t60_comes_back_at_every_partial() {
        // Four partials, four different decays, all inside one note over a
        // stationary noise floor: the whole point is reading k = 16 without
        // k = 15 and k = 17 leaking into it.
        let f0 = 261.63;
        let truth = [(1usize, 1.0, 4.0), (4, 0.7, 2.0), (8, 0.5, 1.2), (16, 0.4, 0.8)];
        let signal = hiss(&note(f0, &truth, 6.0), 1.0e-5);
        let hz: Vec<f64> = (1..=16).map(|k| f0 * k as f64).collect();
        let env = partial_envelopes(&signal, &hz, f0, SR);
        for &(k, _, t60) in &truth {
            let drop = measurable_db(&env[k - 1], HOP_S).expect("a floor is readable");
            let got = fit_tail_to(&env[k - 1], HOP_S, drop)
                .unwrap_or_else(|| panic!("k={k} measured nothing over {drop:.0} dB"))
                .t60_s;
            assert!(
                (got / t60 - 1.0).abs() < 0.10,
                "k={k}: T60 {got:.3} against {t60:.3}"
            );
        }
    }

    #[test]
    fn a_partial_inside_its_own_floor_measures_nothing_rather_than_a_long_decay() {
        // `DECISIONS.md` 293's failure, in this module's units: a note that is
        // over, sitting on a floor 12 dB under its peak. The envelope out there
        // is the floor's and flat, and a fit that took it would call a dead
        // partial an eternal one.
        let shallow: Vec<f64> = (0..1_200)
            .map(|i| (-30.0 * i as f64 * HOP_S).max(-12.0))
            .collect();
        assert!(measurable_db(&shallow, HOP_S).expect("a floor is readable") < MIN_MEASURABLE_DB);
        assert!(fit_tail_to(&shallow, HOP_S, 60.0).is_none());
        // The same decay over a floor 60 dB down is measurable and comes back
        // at the 2 s it was written with.
        let deep: Vec<f64> = (0..1_200)
            .map(|i| (-30.0 * i as f64 * HOP_S).max(-60.0))
            .collect();
        let drop = measurable_db(&deep, HOP_S).expect("a floor is readable");
        assert!((drop - 50.0).abs() < 1e-9, "measurable {drop}");
        let got = fit_tail_to(&deep, HOP_S, drop).expect("30 dB/s over 50 dB is measurable");
        assert!((got.t60_s - 2.0).abs() < 0.05, "T60 {:.3}", got.t60_s);
        assert!((got.floor_db - 60.0).abs() < 1e-9, "floor {}", got.floor_db);
    }

    #[test]
    fn both_sides_are_fitted_over_one_range_and_not_over_their_own() {
        // The bias this rule exists to remove. One partial, the *same* decay on
        // both signals, but the recording has a floor 30 dB under the peak and
        // the render's is 80 down. Fit each to its own floor and the render's
        // line runs through the slow half of a double decay that the
        // recording's never reaches, and the correction reads a defect that is
        // not there.
        let double_decay = |t: f64| {
            let fast = 10f64.powf(-2.0 * t);
            let slow = 0.05 * 10f64.powf(-0.25 * t);
            20.0 * (fast + slow).log10()
        };
        let engine: Vec<f64> = (0..1_200)
            .map(|i| double_decay(i as f64 * HOP_S).max(-80.0))
            .collect();
        let reference: Vec<f64> = (0..1_200)
            .map(|i| double_decay(i as f64 * HOP_S).max(-30.0))
            .collect();
        let own = |env: &[f64]| {
            let d = measurable_db(env, HOP_S).expect("a floor is readable");
            fit_tail_to(env, HOP_S, d).expect("measurable").t60_s
        };
        let separate = own(&engine) / own(&reference);
        let common = {
            let d = measurable_db(&engine, HOP_S)
                .unwrap()
                .min(measurable_db(&reference, HOP_S).unwrap());
            fit_tail_to(&engine, HOP_S, d).expect("measurable").t60_s
                / fit_tail_to(&reference, HOP_S, d).expect("measurable").t60_s
        };
        assert!(
            (common - 1.0).abs() < 0.05,
            "one range says {common:.3} where the truth is 1.0"
        );
        assert!(
            separate > 1.3,
            "the two-range reading was supposed to be biased and read {separate:.3}"
        );
    }

    #[test]
    fn the_correction_is_the_ratio_of_the_two_falls() {
        let both = PartialTail {
            k: 3,
            hz: 900.0,
            drop_db: 50.0,
            engine_db: Levels { at: [Some(0.0), Some(-10.0)], resolvable_db: -90.0 },
            reference_db: Levels { at: [Some(0.0), Some(-40.0)], resolvable_db: -90.0 },
            engine: tail(2.0),
            reference: tail(0.5),
        };
        assert!((both.correction().expect("both measured") - 4.0).abs() < 1e-12);
        assert!((both.t60_ratio().expect("both measured") - 4.0).abs() < 1e-12);
        let half = PartialTail {
            reference_db: Levels { at: [None, None], resolvable_db: -90.0 },
            ..both
        };
        assert!(half.correction().is_none());
        assert!(!half.trusted());
        // A partial that has not moved between the two instants says nothing
        // about a rate, and a ratio of two numbers near zero is not a
        // correction.
        let still = PartialTail {
            engine_db: Levels { at: [Some(0.0), Some(-0.1)], resolvable_db: -90.0 },
            reference_db: Levels { at: [Some(0.0), Some(-0.2)], resolvable_db: -90.0 },
            ..both
        };
        assert!(still.correction().is_none());
        // The recording's partial gone under its own floor is a bound and not a
        // gap: it fell at least to the floor, and that is what is read.
        let bounded = PartialTail {
            reference_db: Levels { at: [Some(0.0), None], resolvable_db: -30.0 },
            ..both
        };
        assert!(bounded.trusted());
        assert!((bounded.reference_fall_db().expect("bounded") - 30.0).abs() < 1e-12);
        assert!(still.correction().is_none());
    }

    fn tail(t60: f64) -> Option<TailFit> {
        Some(TailFit {
            t60_s: t60,
            floor_db: 60.0,
            span_s: 1.0,
            residual_db: 0.5,
        })
    }

    fn cell(k: usize, hz: f64, engine: f64, reference: f64) -> PartialTail {
        PartialTail {
            k,
            hz,
            drop_db: 40.0,
            engine_db: Levels { at: [Some(0.0), Some(-engine)], resolvable_db: -120.0 },
            reference_db: Levels { at: [Some(0.0), Some(-reference)], resolvable_db: -120.0 },
            engine: tail(1.0),
            reference: tail(1.0),
        }
    }

    #[test]
    fn a_band_inside_the_recordings_own_layer_floor_is_left_alone() {
        let fall = |engine, reference, floor| {
            Some(BandFall {
                engine_db: engine,
                reference_db: reference,
                partial_share: 0.9,
                partial_median_ratio: 2.0,
                floor_db: floor,
                partial_median_ratio_error: 0.0,
            })
        };
        // Ten decibels against twenty, and a floor a tenth of a decibel wide:
        // the band is out of its floor, and what is written is the *median*
        // partial's own ratio.
        let c = TailCorrection::from_band_falls([fall(10.0, 20.0, 0.1), None], 12);
        assert!((c.band[0].expect("out of its floor") - 2.0).abs() < 1e-12);
        assert_eq!(c.band[1], None);
        assert_eq!(c.reach, 12);
        // The same two falls with a floor eleven decibels wide: two recordings
        // of one piano differ by more than this, so nothing is written.
        let inside = TailCorrection::from_band_falls([fall(10.0, 20.0, 11.0), None], 12);
        assert!(inside.is_empty());
        // A partial that is not falling at all gets the pass's whole authority
        // and not an infinity.
        let stuck = Some(BandFall {
            engine_db: 1.0,
            reference_db: 20.0,
            partial_share: 0.9,
            partial_median_ratio: 1e9,
            floor_db: 0.1,
            partial_median_ratio_error: 0.0,
        });
        assert_eq!(
            TailCorrection::from_band_falls([stuck, None], 12).band[0],
            Some(MAX_PASS_FACTOR)
        );
        // And a recording whose band does not fall either says nothing.
        let neither = TailCorrection::from_band_falls([fall(0.0, 0.0, 0.1), None], 12);
        assert!(neither.is_empty());
        // A band the partials do not own is not this table's to move, however
        // far out it is: A0 has no partial at all above 2512 Hz.
        let board = Some(BandFall {
            engine_db: 10.0,
            reference_db: 20.0,
            partial_share: MIN_BAND_SHARE - 0.01,
            partial_median_ratio: 2.0,
            floor_db: 0.1,
            partial_median_ratio_error: 0.0,
        });
        assert!(TailCorrection::from_band_falls([board, None], 12).is_empty());
        // And a band whose typical partial has already caught the recording is
        // done, however far the *sum* still is: what is left of the sum belongs
        // to partials this table cannot reach without over-damping the rest.
        let done = Some(BandFall {
            engine_db: 10.0,
            reference_db: 20.0,
            partial_share: 0.9,
            partial_median_ratio: 1.0,
            floor_db: 0.1,
            partial_median_ratio_error: 0.0,
        });
        assert!(TailCorrection::from_band_falls([done, None], 12).is_empty());
        // And a band whose median is over one by less than the median's own
        // standard error says nothing either: without this the stop is a
        // ratchet, because a converged band reads a per cent or two over one
        // from noise alone and every pass multiplies it in.
        let noisy = BandFall {
            engine_db: 10.0,
            reference_db: 20.0,
            partial_share: 0.9,
            partial_median_ratio: 1.03,
            floor_db: 0.1,
            partial_median_ratio_error: 0.12,
        };
        assert!(TailCorrection::from_band_falls([Some(noisy), None], 12).is_empty());
        let resolved = BandFall {
            partial_median_ratio_error: 0.01,
            ..noisy
        };
        assert_eq!(
            TailCorrection::from_band_falls([Some(resolved), None], 12).band[0],
            Some(1.03)
        );
    }

    #[test]
    fn the_medians_own_error_falls_with_the_root_of_the_count() {
        // A decade of ratios scattered by a factor of two: the error of their
        // median has to shrink like 1/sqrt(n) and has to be read in the log,
        // because a ratio of 2 and a ratio of 0.5 are the same distance out.
        let sample = |n: usize| -> Vec<f64> {
            (0..n)
                .map(|i| if i % 2 == 0 { 2.0 } else { 0.5 })
                .collect()
        };
        let (four, sixteen) = (median_ln_error(&sample(4)), median_ln_error(&sample(16)));
        assert!(four > 0.0 && sixteen > 0.0);
        assert!(
            ((four / sixteen) - 2.0).abs() < 0.2,
            "{four} against {sixteen}"
        );
        // No scatter, no error; and a set too small to be a band says nothing.
        assert_eq!(median_ln_error(&[1.5; 8]), 0.0);
        assert_eq!(median_ln_error(&[1.0, 4.0]), 0.0);
    }

    #[test]
    fn the_reach_is_the_last_partial_the_recording_measured() {
        let mut tails: Vec<PartialTail> = (1..=6)
            .map(|k| cell(k, 500.0 * k as f64, 5.0, 10.0))
            .collect();
        assert_eq!(reach(&tails), 6);
        tails[5].reference_db = Levels { at: [None, None], resolvable_db: -90.0 };
        assert_eq!(reach(&tails), 5);
    }

    #[test]
    fn the_correction_leaves_the_fitted_region_alone_and_is_continuous() {
        let c = TailCorrection {
            band: [Some(0.5), Some(3.0)],
            cells: [8, 8],
            reach: 40,
        };
        assert_eq!(c.at(500.0), 1.0);
        assert_eq!(c.at(HF1.0), 1.0);
        assert!((c.at(band_centre(HF1)) - 0.5).abs() < 1e-9);
        assert!((c.at(band_centre(HF2)) - 3.0).abs() < 1e-9);
        assert!((c.at(20_000.0) - 3.0).abs() < 1e-9);
        // No step anywhere: a jump in the decay law between two neighbouring
        // partials is audible as a shelf in the tail.
        let (mut previous, mut hz) = (1.0, HF1.0);
        while hz < 20_000.0 {
            let next = c.at(hz);
            assert!(
                (next / previous).ln().abs() < 0.02,
                "step of {:.3} at {hz:.0} Hz",
                next / previous
            );
            previous = next;
            hz *= 1.01;
        }
    }

    #[test]
    fn the_row_is_multiplied_where_the_recording_reached_and_left_alone_above_it() {
        let partial_hz: Vec<f64> = (1..=12).map(|k| 1_000.0 * k as f64).collect();
        let c = TailCorrection {
            band: [Some(2.0), Some(2.0)],
            cells: [8, 8],
            reach: 6,
        };
        let row = vec![1.5f32, 1.0, 1.0, 1.0];
        let out = extend_row(&row, &partial_hz, &c);
        // k=1 is under 2 kHz: the fitted cell survives untouched.
        assert!((out[0] - 1.5).abs() < 1e-6, "k=1 became {}", out[0]);
        // k=4 (4 kHz) is inside the reach and inside the band: doubled from 1.0.
        assert!((out[3] - 2.0).abs() < 1e-5, "k=4 became {}", out[3]);
        // k=7 (7 kHz) is past the reach: nothing was learned, so nothing is
        // written, and the row stops.
        assert!(out.len() <= 6, "the row ran to {} cells", out.len());
        // The schema's rail is the ceiling, not the correction.
        let hard = TailCorrection {
            band: [Some(4.0), Some(4.0)],
            ..c
        };
        let railed = extend_row(&[3.0, 1.0, 1.0, 3.0], &partial_hz, &hard);
        assert!(
            (railed[0] - 3.0).abs() < 1e-6,
            "k=1 is under 2 kHz and moved to {}",
            railed[0]
        );
        assert!(
            (railed[3] - f64::from(crate::preset::MAX_PARTIAL_SIGMA_SCALE) as f32).abs() < 1e-6,
            "k=4 railed at {}",
            railed[3]
        );
    }

    #[test]
    fn the_draw_is_deterministic_and_lands_inside_the_line_it_came_from() {
        // A surface with a real register slope: the tenor's band falls further
        // than the bass's, which is what the compass measures. The ceiling is
        // one frequency for every key, which is what a recording chain's own
        // noise floor is.
        let points: Vec<DecayPoint> = (0..30)
            .map(|i| {
                let key = 21 + 3 * i as u8;
                DecayPoint {
                    key,
                    reference_fall_db: [
                        Some((0.5 + 0.02 * f64::from(key)).exp()),
                        Some((0.3 + 0.02 * f64::from(key)).exp()),
                    ],
                    reference_partial_fall_db: [
                        Some((0.9 + 0.02 * f64::from(key)).exp()),
                        Some((0.7 + 0.02 * f64::from(key)).exp()),
                    ],
                    ceiling_hz: 11_000.0,
                }
            })
            .collect();
        let model = DecayModel::fit(&points);
        assert!(
            (model.fall[0].slope - 0.02).abs() < 1e-9,
            "slope {}",
            model.fall[0].slope
        );
        assert!(model.fall[0].sigma < 1e-9, "a clean line has scatter");
        assert!((model.ceiling.hz - 11_000.0).abs() < 1e-9);
        assert!((model.ceiling.spread - 1.0).abs() < 1e-9, "a constant scatters");
        // Same key, same answer, twice, and different keys differ.
        assert_eq!(model.draw(60), model.draw(60));
        assert_ne!(model.draw(60), model.draw(61));
        // And each draw sits on the line it came from, since these lines have
        // no scatter to draw.
        let drawn = model.draw(60);
        assert!(
            (drawn.target_fall_db[0] / model.fall[0].at(60) - 1.0).abs() < 1e-9,
            "drawn {:.3} against the line's {:.3}",
            drawn.target_fall_db[0],
            model.fall[0].at(60)
        );
        assert!(
            (drawn.target_partial_fall_db[1] / model.partial_fall[1].at(60) - 1.0).abs() < 1e-9,
            "drawn {:.3} against the line's {:.3}",
            drawn.target_partial_fall_db[1],
            model.partial_fall[1].at(60)
        );
        // The median partial falls further than the band's sum, at every key
        // and in both bands: that is what a sum over terms of different rates
        // does, and it is why the stop needs its own draw.
        for key in [30u8, 60, 90] {
            let d = model.draw(key);
            for b in 0..2 {
                assert!(
                    d.target_partial_fall_db[b] > d.target_fall_db[b],
                    "key {key} band {b}: {:.2} against {:.2}",
                    d.target_partial_fall_db[b],
                    d.target_fall_db[b]
                );
            }
        }
    }

    #[test]
    fn the_ceiling_is_one_frequency_and_reaches_a_different_partial_at_every_key() {
        // The defect item 320 found: a reach drawn as a partial *index* has to
        // carry the seven octaves of `f0` that separate A0 from C8 inside a
        // straight line, and it cannot. A ceiling in hertz divided by the key's
        // own series is arithmetic and not a fit.
        let ceiling = Ceiling::fit(&[9_000.0, 11_000.0, 13_000.0]);
        assert!((ceiling.hz - 11_000.0).abs() < 1e-9);
        assert_eq!(ceiling.points, 3);
        assert!(
            (ceiling.spread - (13_000f64 / 11_000.0)).abs() < 0.02,
            "spread {}",
            ceiling.spread
        );
        // C4 and C6, each with its own series: the same ceiling, and the reach
        // it licenses differs by the two octaves between them.
        let c4: Vec<f64> = (1..=60).map(|k| 261.63 * k as f64).collect();
        let c6: Vec<f64> = (1..=60).map(|k| 1_046.5 * k as f64).collect();
        assert_eq!(reach_to(&c4, ceiling.hz), 42);
        assert_eq!(reach_to(&c6, ceiling.hz), 10);
        // And the bank binds where the series ends before the ceiling does,
        // with no second rule for it: A0's bank stops at 2.5 kHz.
        let a0: Vec<f64> = (1..=80).map(|k| 27.5 * k as f64).collect();
        assert_eq!(reach_to(&a0, ceiling.hz), 80);
        // An empty population is a ceiling of nothing rather than a panic.
        assert_eq!(Ceiling::fit(&[]).points, 0);
        assert_eq!(reach_to(&c4, Ceiling::fit(&[]).hz), 0);
    }

    #[test]
    fn a_bands_sum_falls_less_than_its_median_partial_and_both_are_reported() {
        // Three partials, one of which barely decays: the sum is that partial's
        // and the median is the typical one's. A stop that compares a drawn
        // *sum* against a rendered *median* is comparing these two columns.
        let tails = vec![
            cell(8, 2_500.0, 30.0, 40.0),
            cell(9, 3_000.0, 28.0, 38.0),
            cell(10, 3_500.0, 1.0, 2.0),
        ];
        let fall = partial_band_fall(&tails, HF1).expect("three trusted cells");
        assert_eq!(fall.engine.cells, 3);
        assert!(
            fall.engine.median_db > fall.engine.band_db + 10.0,
            "median {:.2} against the sum's {:.2}",
            fall.engine.median_db,
            fall.engine.band_db
        );
        assert!(fall.reference.median_db > fall.reference.band_db + 10.0);
        // The paired statistic: every partial is 10 dB further down on the
        // recording, so the median ratio is the ratio of the two medians here.
        assert!((fall.median_ratio - 38.0 / 28.0).abs() < 1e-9, "{}", fall.median_ratio);
        // The engine's own side, over the same partials, is the same reading.
        let engine = engine_band_fall(&tails, HF1).expect("three engine cells");
        assert!((engine.band_db - fall.engine.band_db).abs() < 1e-9);
        assert!((engine.median_db - fall.engine.median_db).abs() < 1e-9);
    }


    #[test]
    fn a_beating_envelope_is_refused_rather_than_averaged() {
        // A partial whose strings null it 26 dB deep twice a second is not a
        // straight line in dB, and what a least-squares slope through it
        // returns depends on where the record happens to stop. Shortening the
        // prefix does not rescue it either: a prefix caught on the way into a
        // null is steep and clean, and the two halves is what refuses it.
        let env: Vec<f64> = (0..1_200)
            .map(|i| {
                let t = i as f64 * HOP_S;
                let beat = 20.0
                    * ((std::f64::consts::TAU * t).cos().abs() + 0.05).log10();
                (-20.0 * t + beat).max(-100.0)
            })
            .collect();
        assert!(
            fit_tail_to(&env, HOP_S, 60.0).is_none(),
            "a 26 dB beat pattern was fitted as a decay: {:?}",
            fit_tail_to(&env, HOP_S, 60.0)
        );
    }

    // -----------------------------------------------------------------------
    // The band under the correction curve (`DECISIONS.md` 334-335)
    // -----------------------------------------------------------------------

    /// A harmonic series at `f0`, which is what the low band is counted over.
    fn series(f0: f64, n: usize) -> Vec<f64> {
        (1..=n).map(|k| f0 * k as f64).collect()
    }

    #[test]
    fn the_low_band_is_the_geometric_mean_of_the_cells_under_two_kilohertz() {
        // C4: seven partials under 2 kHz out of a row of ten.
        let hz = series(261.6, 12);
        let row: Vec<f32> = vec![0.5, 0.5, 0.5, 0.5, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0];
        let (mean, cells) = low_mean(&row, &hz).expect("seven cells is a band");
        assert_eq!(cells, 7, "2 kHz is between partial 7 and partial 8 of C4");
        // Four halves and three doubles: ln mean = (4·ln0.5 + 3·ln2)/7.
        let want = ((4.0 * 0.5f64.ln() + 3.0 * 2.0f64.ln()) / 7.0).exp();
        assert!(
            (mean - want).abs() < 1e-9,
            "the mean is not the geometric one: {mean} against {want}"
        );
        // The cells above 2 kHz are not in it, whatever they are.
        let mut louder = row.clone();
        for cell in louder.iter_mut().skip(7) {
            *cell = 0.25;
        }
        assert_eq!(low_mean(&louder, &hz), low_mean(&row, &hz));
    }

    #[test]
    fn a_key_with_fewer_than_three_partials_under_two_kilohertz_has_no_low_band() {
        // F#5: 741 Hz, so partials 1 and 2 are under 2 kHz and partial 3 is not.
        let hz = series(741.0, 8);
        assert_eq!(hz.iter().filter(|&&f| f < LOW_BAND.1).count(), 2);
        assert_eq!(
            low_mean(&[0.5, 0.5, 0.5, 0.5], &hz),
            None,
            "two cells were taken as a band statistic"
        );
        // And nothing is written there either, which is the other half of the
        // rule: two cells of one key are one partial's idiosyncrasy.
        let row = vec![0.9f32, 0.9, 0.9, 0.9];
        assert_eq!(low_row(&row, &hz, 0.4), row);
    }

    #[test]
    fn the_low_line_is_exact_where_a_key_measured_it_and_interpolates_between() {
        // Three sampled keys a minor third apart, and a key between two of them.
        let low = LowDecay::fit(&[(57, 0.9), (60, 0.75), (63, 0.83)]);
        for &(key, want) in &[(57u8, 0.9), (60, 0.75), (63, 0.83)] {
            assert!(
                (low.at(key) / want - 1.0).abs() < 1e-6,
                "the model is not exact at a key that measured it: {} against {want}",
                low.at(key)
            );
        }
        // Between two sampled keys it is the line times the two departures
        // interpolated, which is bracketed by them in `ln`.
        let between = low.at(61);
        assert!(
            between > 0.75 && between < 0.83,
            "a key between 0.75 and 0.83 drew {between}"
        );
    }

    #[test]
    fn the_low_line_tapers_to_itself_past_the_last_key_that_measured_it() {
        let low = LowDecay::fit(&[(57, 0.9), (60, 0.75), (63, 0.5)]);
        // At the last point the departure is carried whole; a full taper past
        // it, the model is the line alone.
        let far = 63 + RESIDUAL_TAPER_KEYS as u8;
        assert!(
            (low.at(far) / low.line.at(far) - 1.0).abs() < 1e-6,
            "an octave past the last measurement the departure is still being carried"
        );
        // An empty model asks for nothing rather than for zero.
        assert_eq!(LowDecay::fit(&[]).at(60), 1.0);
    }

    #[test]
    fn a_drawn_low_row_scales_under_the_edge_and_leaves_everything_above_it() {
        let hz = series(261.6, 12);
        let row: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.5, 2.5, 2.5];
        let out = low_row(&row, &hz, 0.75);
        assert_eq!(out.len(), row.len());
        for (k, cell) in out.iter().enumerate() {
            let want = if hz[k] < LOW_BAND.1 { 0.75 } else { f64::from(row[k]) };
            assert!(
                (f64::from(*cell) - want).abs() < 1e-6,
                "partial {} at {:.0} Hz reads {cell} and not {want}",
                k + 1,
                hz[k]
            );
        }
        // It is a multiplier and not a replacement: a key that already carries
        // cells there keeps their shape.
        let carried: Vec<f32> = vec![0.5, 2.0, 0.5, 2.0, 0.5, 2.0, 0.5];
        let out = low_row(&carried, &hz, 0.5);
        for (k, cell) in out.iter().enumerate() {
            assert!(
                (f64::from(*cell) - f64::from(carried[k]) * 0.5).abs() < 1e-6,
                "cell {} was replaced rather than scaled",
                k + 1
            );
        }
        // And it obeys the schema's rails rather than the arithmetic.
        let out = low_row(&[0.3; 7], &hz, 0.1);
        assert!(
            out.iter().all(|&c| c >= crate::preset::MIN_PARTIAL_SIGMA_SCALE),
            "a drawn low row went under the schema's floor: {out:?}"
        );
    }
}
