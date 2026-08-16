//! How one partial *moves*: its instantaneous frequency, its beat depth, and
//! the rate its envelope beats at.
//!
//! Everything else in this crate measures a partial's amplitude and its
//! frequency as single numbers per frame — `f_k(t)`, `a_k(t)` on a 43 ms hop.
//! That is the right resolution for a decay rate and the wrong one for the
//! percept `docs/history/FUNDAMENTALS.md` Part II is about, which lives at 0.1–20 Hz *inside*
//! one partial and is a fraction of a cent deep. This module is the measurement
//! that resolves it, promoted out of `forensics/src/bin/jitter_forensics.rs`
//! (the forensics that found the artefact) and
//! `forensics/src/bin/eigenmode_prototype.rs`
//! (which added the beat-rate and placement statistics) so that three callers
//! can share one implementation:
//!
//! * [`crate::realism`]'s Columns A and B — the scoreboard gates of
//!   `docs/history/FUNDAMENTALS.md` §7.7;
//! * [`crate::estimate::motion`] — which inverts the same statistics into
//!   `notes.false_beat` and `[voicing.strike_direction]`;
//! * the two forensic instruments above, which keep their own reporting.
//!
//! # The measurement
//!
//! One forward transform per signal ([`Spectrum::new`]), then per partial:
//!
//! 1. Find the partial's own spectral peak near its nominal frequency, and how
//!    far it stands over the median of its neighbourhood ([`Spectrum::peak_near`]).
//!    A partial that does not clear [`MIN_PEAK_DB`] is not measured: what would
//!    come back is the phase of whatever else radiates there.
//! 2. Multiply by a Gaussian centred on that peak, inverse-transform, and
//!    demodulate down to zero. The Gaussian's time constant is
//!    [`SMOOTH_SIGMA_S`], so its frequency width is 31.8 Hz and the smoothing
//!    the track gets is the filter's and nothing else's.
//! 3. Differentiate the phase on a [`TRACK_HZ`] grid over
//!    [`WINDOW_LO_S`]–[`WINDOW_HI_S`]. That is the instantaneous frequency; the
//!    modulus is the envelope.
//!
//! From the two tracks, six numbers ([`Motion`]). Two of them carry the whole
//! of Part II's argument:
//!
//! * [`Motion::band_cents`] — RMS of the frequency deviation restricted to
//!   [`MOD_LO_HZ`]–[`MOD_HI_HZ`]. The restriction is what separates a beat
//!   (one line under 20 Hz) from everything else radiating near the partial
//!   (broadband, and it fills the whole 500 Hz the grid can express).
//! * [`Motion::placement`] — the same deviation power-weighted by the partial's
//!   own instantaneous power, over the unweighted version. Near 1 is a wobble
//!   that rides the loud part of the note; a small fraction is a spike at the
//!   null of a beat, which is what a sum of free-running sinusoids must produce
//!   and what the recording does not.
//!
//! # The floor
//!
//! Nothing here reads zero. A single decaying sinusoid — the control the
//! forensics ran through this identical code — reads 0.00–0.05 cents of
//! [`Motion::band_cents`], because the demodulation of a decaying signal is not
//! exactly stationary. [`IF_FLOOR_CENTS`] is that floor, and
//! `docs/history/FUNDAMENTALS.md`'s verification errata require it to be clamped in before
//! any ratio of two cells is taken: without it a cell where both signals are at
//! the floor reads a ratio of 30 instead of 1.

use std::f64::consts::TAU;

use rustfft::{num_complex::Complex64, FftPlanner};

use crate::SAMPLE_RATE;

/// Sample rate everything here is quoted on.
const SR: f64 = SAMPLE_RATE as f64;

/// Time-domain standard deviation of the Gaussian band-pass, i.e. the smoothing
/// the frequency track gets. Its frequency-domain width is
/// `1 / (2 pi SMOOTH_SIGMA_S)` = 31.8 Hz, so the nearest neighbouring partial of
/// the lowest key the columns use (A2, 110 Hz apart) is 3.5 sigma out and 54 dB
/// down.
pub const SMOOTH_SIGMA_S: f64 = 0.005;

/// Rate the demodulated track is decimated to before the phase is
/// differentiated. Far above the filter's own 32 Hz bandwidth, so nothing is
/// lost, and far below any carrier, so no phase difference can wrap.
pub const TRACK_HZ: f64 = 1000.0;

/// The modulation band both the frequency track and the envelope are reported
/// over. Matches `renders/timbre-ladder/ANALYSIS.md` metric 3 and
/// `renders/jitter/JITTER.md`, so every number here is comparable with theirs.
pub const MOD_LO_HZ: f64 = 0.1;
pub const MOD_HI_HZ: f64 = 20.0;

/// The analysis window, in seconds since the strike. Past the attack and the
/// hammer's noise, and inside the part of the record every key still sounds in.
pub const WINDOW_LO_S: f64 = 0.3;
pub const WINDOW_HI_S: f64 = 3.0;

/// Transform length the band-pass is applied in: 5.46 s at 48 kHz, longer than
/// anything analysed, so the filter never wraps into its own input.
pub const FFT_N: usize = 1 << 18;

/// How far a partial must stand over the median of its own neighbourhood before
/// it is measured at all, in dB.
pub const MIN_PEAK_DB: f64 = 10.0;

/// The measurement's own floor on [`Motion::band_cents`], in cents: what a
/// single decaying sinusoid reads through this code. Clamp both sides in before
/// taking any ratio (`docs/history/FUNDAMENTALS.md` verification errata).
pub const IF_FLOOR_CENTS: f64 = 0.05;

/// Everything one partial of one signal contributes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Motion {
    /// Power-weighted mean instantaneous frequency, Hz — the pitch the
    /// deviations below are quoted against.
    pub mean_hz: f64,
    /// The partial's own bin over the median bin of the neighbourhood it was
    /// found in, dB.
    pub peak_db: f64,
    /// RMS of the frequency deviation inside [`MOD_LO_HZ`]–[`MOD_HI_HZ`],
    /// cents. Column A's `J`.
    pub band_cents: f64,
    /// RMS of the *unrestricted* deviation, cents.
    pub raw_cents: f64,
    /// The unrestricted deviation weighted by the partial's own power, cents.
    pub weighted_cents: f64,
    /// Peak-to-trough span of the band-limited log envelope, dB (p95 − p5).
    /// Column B's `D`, and what the false-beat level is inverted from.
    pub beat_depth_db: f64,
    /// Dominant rate of the band-limited log envelope's own movement, Hz — see
    /// [`beat_rate`] for why it is an interpolated spectral peak and not the
    /// sign count `docs/history/FUNDAMENTALS.md` §7.4 used. This is the statistic that
    /// decides *which* mechanism a beat comes from: a unison mistuning is a
    /// frequency ratio, so partial `k` beats at `k` times the fundamental's
    /// rate, and a split in the wire's own geometry does not know what `k` is.
    pub beat_rate_hz: f64,
    /// Slope of the log envelope over the first third of the window, dB/s
    /// (negative) — the prompt sound.
    pub prompt_db_s: f64,
    /// The same over the last half — the aftersound.
    pub tail_db_s: f64,
    /// Where the tail's straight line extrapolates back to at the strike,
    /// relative to the prompt's: the aftersound level, in dB under the prompt.
    /// `docs/history/FUNDAMENTALS.md` §7.3's own statistic, and the one the eigenmode
    /// prototype broke at C6 (4.9 -> 21.2 dB).
    pub aftersound_db: f64,
}

impl Motion {
    /// Where in the note the wobble sits: near 1 is a wobble that rides the
    /// loud part of the partial, a small fraction is a spike at the null of a
    /// beat. Column A's `L`.
    pub fn placement(&self) -> f64 {
        if self.raw_cents > 0.0 {
            self.weighted_cents / self.raw_cents
        } else {
            0.0
        }
    }

    /// [`Motion::band_cents`] with the measurement's own floor clamped in.
    pub fn floored_cents(&self) -> f64 {
        self.band_cents.max(IF_FLOOR_CENTS)
    }

    /// The two-component pair this partial's envelope implies: how loud the
    /// companion is, in dB under the partial, from
    /// `D = 20 log10((1 + r) / (1 - r))`.
    ///
    /// This is `docs/history/FUNDAMENTALS.md` §7.4's inversion, and it is the only reading of
    /// a measured beat depth that is in the units the preset is written in.
    /// Returns `None` for a depth so large that `r` would reach 1, which is a
    /// partial that goes through an exact null rather than a pair.
    pub fn companion_db(&self) -> Option<f64> {
        if self.beat_depth_db <= 0.0 || !self.beat_depth_db.is_finite() {
            return None;
        }
        let x = 10f64.powf(self.beat_depth_db / 20.0);
        let r = (x - 1.0) / (x + 1.0);
        (r > 0.0 && r < 1.0).then(|| 20.0 * r.log10())
    }
}

/// The forward transform of one signal, computed once and reused by every
/// partial of it.
pub struct Spectrum {
    bins: Vec<Complex64>,
    planner: FftPlanner<f64>,
}

impl Spectrum {
    /// Transforms `signal`, which must start at the strike.
    pub fn new(signal: &[f64]) -> Spectrum {
        let mut planner = FftPlanner::<f64>::new();
        let mut bins: Vec<Complex64> = (0..FFT_N)
            .map(|n| Complex64::new(signal.get(n).copied().unwrap_or(0.0), 0.0))
            .collect();
        planner.plan_fft_forward(FFT_N).process(&mut bins);
        Spectrum { bins, planner }
    }

    /// Frequency of bin `m`.
    fn hz(m: usize) -> f64 {
        m as f64 * SR / FFT_N as f64
    }

    /// The strongest bin within `±half_width` of `nominal`, refined by a
    /// parabolic fit, and how far it stands over the band's median magnitude.
    pub fn peak_near(&self, nominal: f64, half_width: f64) -> (f64, f64) {
        let bin = |hz: f64| ((hz * FFT_N as f64 / SR).round() as isize).max(1) as usize;
        let lo = bin(nominal - half_width).max(1);
        let hi = bin(nominal + half_width).min(FFT_N / 2 - 2);
        if hi <= lo {
            return (nominal, 0.0);
        }
        let mag = |m: usize| self.bins[m].norm();
        let mut best = lo;
        for m in lo..=hi {
            if mag(m) > mag(best) {
                best = m;
            }
        }
        let mut band: Vec<f64> = (lo..=hi).map(mag).collect();
        band.sort_by(|a, b| a.partial_cmp(b).expect("finite magnitudes"));
        let median = band[band.len() / 2].max(f64::MIN_POSITIVE);
        let peak_db = 20.0 * (mag(best) / median).log10();
        // Parabolic interpolation in log magnitude: the Gaussian band-pass is
        // centred on the result, and a bin is 0.18 Hz wide, so this only has to
        // be good to a fraction of a bin.
        let (a, b, c) = (
            mag(best - 1).max(f64::MIN_POSITIVE).ln(),
            mag(best).max(f64::MIN_POSITIVE).ln(),
            mag(best + 1).max(f64::MIN_POSITIVE).ln(),
        );
        let denom = a - 2.0 * b + c;
        let delta = if denom.abs() > 1e-12 {
            (0.5 * (a - c) / denom).clamp(-0.5, 0.5)
        } else {
            0.0
        };
        (Spectrum::hz(best) + delta * SR / FFT_N as f64, peak_db)
    }

    /// The partial's analytic signal, band-passed by a Gaussian centred on
    /// `carrier` and demodulated down to zero, decimated to [`TRACK_HZ`] and cut
    /// to the analysis window.
    fn demodulate(&mut self, carrier: f64) -> Vec<Complex64> {
        // Never wider than a quarter of the carrier: a Gaussian centred at
        // 110 Hz with a 32 Hz width would be cut off at DC, and a cut-off
        // Gaussian is a sharp edge whose ringing is not the partial's phase.
        let sigma_f = (1.0 / (TAU * SMOOTH_SIGMA_S)).min(carrier / 4.0);
        let mut z = vec![Complex64::new(0.0, 0.0); FFT_N];
        // Positive frequencies only, at twice the amplitude: that is the
        // analytic signal, and the Gaussian is what separates this partial from
        // its neighbours. Six sigma out the weight is 1.5e-8, so the sum is over
        // the partial's own neighbourhood and nothing else.
        let span = (6.0 * sigma_f * FFT_N as f64 / SR).ceil() as usize;
        let centre = (carrier * FFT_N as f64 / SR).round() as usize;
        let lo = centre.saturating_sub(span).max(1);
        let hi = (centre + span).min(FFT_N / 2 - 1);
        for (m, bin) in z.iter_mut().enumerate().take(hi + 1).skip(lo) {
            let u = (Spectrum::hz(m) - carrier) / sigma_f;
            *bin = self.bins[m] * (2.0 * (-0.5 * u * u).exp());
        }
        self.planner.plan_fft_inverse(FFT_N).process(&mut z);
        let scale = 1.0 / FFT_N as f64;
        let step = (SR / TRACK_HZ).round() as usize;
        let from = (WINDOW_LO_S * SR) as usize;
        // One extra sample at each end: the phase is differentiated, and the
        // deviation is quoted over exactly the window.
        let to = (WINDOW_HI_S * SR) as usize + step;
        (from..=to)
            .step_by(step)
            .map(|n| {
                let phase = -TAU * carrier * n as f64 / SR;
                z[n] * scale * Complex64::from_polar(1.0, phase)
            })
            .collect()
    }
}

/// Demodulates the partial nearest `nominal_hz` and measures it, or `None` if
/// nothing stands over the neighbourhood there.
pub fn partial_motion(spectrum: &mut Spectrum, nominal_hz: f64, half_width: f64) -> Option<Motion> {
    let (carrier_hz, peak_db) = spectrum.peak_near(nominal_hz, half_width);
    if peak_db < MIN_PEAK_DB {
        return None;
    }
    let y = spectrum.demodulate(carrier_hz);
    if y.len() < 16 {
        return None;
    }
    // Instantaneous frequency from the phase increment. `arg(y[j+1] conj y[j])`
    // is the increment already wrapped into (−pi, pi], which cannot alias here:
    // the filter is 32 Hz wide and the grid is 1 kHz.
    let mut inst = Vec::with_capacity(y.len() - 1);
    let mut weight = Vec::with_capacity(y.len() - 1);
    for j in 0..y.len() - 1 {
        let d = y[j + 1] * y[j].conj();
        inst.push(carrier_hz + d.arg() * TRACK_HZ / TAU);
        // The weight is the geometric mean of the two endpoints' powers, so it
        // sits at the same instant the increment does.
        weight.push((y[j].norm_sqr() * y[j + 1].norm_sqr()).sqrt());
    }
    let total: f64 = weight.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return None;
    }
    let mean_hz: f64 = inst.iter().zip(&weight).map(|(f, w)| f * w / total).sum();
    if !mean_hz.is_finite() || mean_hz <= 0.0 {
        return None;
    }
    let cents: Vec<f64> = inst
        .iter()
        .map(|f| {
            if *f > 0.0 {
                1200.0 * (f / mean_hz).log2()
            } else {
                // A negative instantaneous frequency is what an exact null looks
                // like on this grid; it is a real excursion, and clamping it to
                // the widest value the grid can express is the honest reading
                // rather than dropping it.
                -1200.0 * (TRACK_HZ / 2.0 / mean_hz).log2().abs()
            }
        })
        .collect();
    let amp_db: Vec<f64> = weight.iter().map(|w| 10.0 * w.max(1e-300).log10()).collect();
    let peak_power = weight.iter().copied().fold(0.0f64, f64::max);
    let weight: Vec<f64> = weight.iter().map(|w| w / peak_power).collect();

    let n = cents.len() as f64;
    let raw_cents = (cents.iter().map(|c| c * c).sum::<f64>() / n).sqrt();
    let wsum: f64 = weight.iter().sum();
    let weighted_cents = (cents
        .iter()
        .zip(&weight)
        .map(|(c, w)| c * c * w)
        .sum::<f64>()
        / wsum.max(f64::MIN_POSITIVE))
    .sqrt();

    // The frequency track is detrended linearly — a partial whose pitch drifts
    // as its unison dies (`docs/history/TUNING_REPORT.md` §6) is drift and not jitter — and
    // the log envelope cubically, which is what a two-exponential decay looks
    // like over three seconds and is the detrend `ANALYSIS.md` metric 3 uses.
    let band = band_limited(&detrended(&cents, 1), TRACK_HZ);
    let band_cents = (band.iter().map(|c| c * c).sum::<f64>() / n).sqrt();

    let envelope = band_limited(&detrended(&amp_db, 3), TRACK_HZ);
    let beat_rate_hz = beat_rate(&envelope, TRACK_HZ);
    let mut sorted = envelope;
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite envelope"));
    let at = |q: f64| sorted[((sorted.len() - 1) as f64 * q) as usize];

    // The double decay, off the same track: a straight line through the prompt
    // and another through the tail, each extrapolated back to the strike. The
    // two windows do not touch, so a handover inside the note belongs to
    // neither line.
    let t = |j: usize| WINDOW_LO_S + j as f64 / TRACK_HZ;
    let n_all = amp_db.len();
    let prompt = line_through(&amp_db[..n_all / 3], t(0), 1.0 / TRACK_HZ);
    let tail = line_through(&amp_db[n_all / 2..], t(n_all / 2), 1.0 / TRACK_HZ);

    Some(Motion {
        mean_hz,
        peak_db,
        band_cents,
        raw_cents,
        weighted_cents,
        beat_depth_db: at(0.95) - at(0.05),
        beat_rate_hz,
        prompt_db_s: prompt.1,
        tail_db_s: tail.1,
        aftersound_db: prompt.0 - tail.0,
    })
}

/// Least squares through `(t0 + j dt, y[j])`, returned as `(value at t = 0,
/// slope per second)`.
fn line_through(y: &[f64], t0: f64, dt: f64) -> (f64, f64) {
    let n = y.len();
    if n < 2 {
        return (y.first().copied().unwrap_or(0.0), 0.0);
    }
    let (mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0);
    for (j, &value) in y.iter().enumerate() {
        let x = t0 + j as f64 * dt;
        sx += x;
        sy += value;
        sxx += x * x;
        sxy += x * value;
    }
    let n = n as f64;
    let denom = n * sxx - sx * sx;
    if denom.abs() <= 1e-12 || !denom.is_finite() {
        return (sy / n, 0.0);
    }
    let slope = (n * sxy - sx * sy) / denom;
    ((sy - slope * sx) / n, slope)
}

/// The dominant modulation rate of an already band-limited, zero-mean log
/// envelope, in Hz.
///
/// # Why this is the transform and not the sign count
///
/// `docs/history/FUNDAMENTALS.md` §7.4 and the eigenmode prototype counted the envelope's own
/// **sign changes** ([`crossing_rate`]), on the argument that a 2.7 s window
/// resolves 0.37 Hz and an FFT would put 0.7 and 1.0 Hz in the same bin. That is
/// true of a bin *index* and false of an interpolated peak, and the count has a
/// bias the interpolated peak does not: the log of a two-component envelope is
///
/// ```text
///     ln |1 + r e^{i theta}| = - sum_{n >= 1} (-r)^n cos(n theta) / n
/// ```
///
/// — a whole harmonic series whose `n`-th term is `r^n / n`. The **fundamental
/// always dominates**, so a peak-picker returns `Delta f` at every ratio; but
/// the harmonics add sign changes, so a *count* over-reads, and it over-reads
/// most where the companion is loudest, which is exactly the cell this
/// measurement exists for. Measured on the engine's own render of a known split
/// (`tuner/tests/calibration.rs`), the count returns 1.85 Hz for a 1.4 Hz split
/// at −6 dB where the peak returns 1.42.
///
/// So: Hann-windowed power spectrum of the band-limited track, strongest bin
/// inside [`MOD_LO_HZ`]–[`MOD_HI_HZ`], refined by a parabola in log power. The
/// count survives as the fallback for a track with no peak inside the band, and
/// as [`crossing_rate`] for callers that want the published statistic.
pub fn beat_rate(x: &[f64], rate: f64) -> f64 {
    let n = x.len();
    if n < 16 {
        return crossing_rate(x, rate);
    }
    let window: Vec<f64> = (0..n)
        .map(|j| 0.5 - 0.5 * (TAU * j as f64 / n as f64).cos())
        .collect();
    let mut buf: Vec<Complex64> = x
        .iter()
        .zip(&window)
        .map(|(&v, &w)| Complex64::new(v * w, 0.0))
        .collect();
    FftPlanner::<f64>::new().plan_fft_forward(n).process(&mut buf);
    let bin_hz = rate / n as f64;
    let lo = (MOD_LO_HZ / bin_hz).ceil().max(1.0) as usize;
    let hi = ((MOD_HI_HZ / bin_hz).floor() as usize).min(n / 2 - 2);
    if hi <= lo + 1 {
        return crossing_rate(x, rate);
    }
    let power = |m: usize| buf[m].norm_sqr();
    let mut best = lo;
    for m in lo..=hi {
        if power(m) > power(best) {
            best = m;
        }
    }
    if best <= lo || best >= hi || power(best) <= 0.0 {
        return crossing_rate(x, rate);
    }
    let (a, b, c) = (
        power(best - 1).max(1e-300).ln(),
        power(best).max(1e-300).ln(),
        power(best + 1).max(1e-300).ln(),
    );
    let denom = a - 2.0 * b + c;
    let delta = if denom.abs() > 1e-12 {
        (0.5 * (a - c) / denom).clamp(-0.5, 0.5)
    } else {
        0.0
    };
    (best as f64 + delta) * bin_hz
}

/// Mean rate of an already band-limited, zero-mean track, from its own sign
/// changes: `crossings / (2 * span)`. [`beat_rate`]'s fallback, and the
/// statistic `docs/history/FUNDAMENTALS.md` §7.4's tables were taken with.
pub fn crossing_rate(x: &[f64], rate: f64) -> f64 {
    if x.len() < 4 {
        return 0.0;
    }
    let crossings = x
        .windows(2)
        .filter(|w| (w[0] < 0.0) != (w[1] < 0.0))
        .count();
    crossings as f64 * rate / (2.0 * (x.len() - 1) as f64)
}

/// `x` with everything outside [`MOD_LO_HZ`]–[`MOD_HI_HZ`] removed, zero phase.
///
/// The track is already detrended, so it has no step at the wrap and a
/// rectangular mask is the right filter.
pub fn band_limited(x: &[f64], rate: f64) -> Vec<f64> {
    let n = x.len();
    if n < 16 {
        return x.to_vec();
    }
    let mut planner = FftPlanner::<f64>::new();
    let mut buf: Vec<Complex64> = x.iter().map(|&v| Complex64::new(v, 0.0)).collect();
    planner.plan_fft_forward(n).process(&mut buf);
    let bin_hz = rate / n as f64;
    for (m, b) in buf.iter_mut().enumerate() {
        let hz = if m <= n / 2 {
            m as f64 * bin_hz
        } else {
            (n - m) as f64 * bin_hz
        };
        if !(MOD_LO_HZ..=MOD_HI_HZ).contains(&hz) {
            *b = Complex64::new(0.0, 0.0);
        }
    }
    planner.plan_fft_inverse(n).process(&mut buf);
    buf.iter().map(|c| c.re / n as f64).collect()
}

/// `x` with a least-squares polynomial of `degree` in time removed.
pub fn detrended(x: &[f64], degree: usize) -> Vec<f64> {
    let n = x.len();
    let cols = degree + 1;
    // Normal equations on the Vandermonde in `u = 2 j / (n - 1) - 1`, which
    // keeps the matrix well conditioned for the degrees used here.
    let u = |j: usize| 2.0 * j as f64 / (n - 1).max(1) as f64 - 1.0;
    let mut a = vec![0.0f64; cols * cols];
    let mut b = vec![0.0f64; cols];
    let mut p = vec![1.0f64; cols];
    for (j, &value) in x.iter().enumerate() {
        p[0] = 1.0;
        for c in 1..cols {
            p[c] = p[c - 1] * u(j);
        }
        for r in 0..cols {
            for c in 0..cols {
                a[r * cols + c] += p[r] * p[c];
            }
            b[r] += p[r] * value;
        }
    }
    // Gaussian elimination with partial pivoting.
    for c in 0..cols {
        let mut pivot = c;
        for r in c + 1..cols {
            if a[r * cols + c].abs() > a[pivot * cols + c].abs() {
                pivot = r;
            }
        }
        if a[pivot * cols + c].abs() < 1e-12 {
            return x.to_vec();
        }
        for k in 0..cols {
            a.swap(c * cols + k, pivot * cols + k);
        }
        b.swap(c, pivot);
        for r in 0..cols {
            if r == c {
                continue;
            }
            let f = a[r * cols + c] / a[c * cols + c];
            for k in c..cols {
                a[r * cols + k] -= f * a[c * cols + k];
            }
            b[r] -= f * b[c];
        }
    }
    let coeff: Vec<f64> = (0..cols).map(|c| b[c] / a[c * cols + c]).collect();
    (0..n)
        .map(|j| {
            let mut p = 1.0;
            let mut fit = 0.0;
            for &c in &coeff {
                fit += c * p;
                p *= u(j);
            }
            x[j] - fit
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `duration` seconds of the sum of `components`, each `(hz, amplitude,
    /// decay s^-1)`, from the strike.
    fn signal(components: &[(f64, f64, f64)], duration: f64) -> Vec<f64> {
        let n = (duration * SR) as usize;
        (0..n)
            .map(|j| {
                let t = j as f64 / SR;
                components
                    .iter()
                    .map(|&(hz, a, sigma)| a * (-sigma * t).exp() * (TAU * hz * t).cos())
                    .sum()
            })
            .collect()
    }

    /// The floor every number this module produces is read against: one
    /// decaying sinusoid is still and its envelope does not beat.
    #[test]
    fn one_decaying_sinusoid_reads_the_measurements_floor() {
        let mut spectrum = Spectrum::new(&signal(&[(440.0, 1.0, 0.8)], 4.0));
        let motion = partial_motion(&mut spectrum, 440.0, 20.0).expect("a partial");
        assert!(
            motion.band_cents < IF_FLOOR_CENTS,
            "band_cents {} should be under the floor",
            motion.band_cents
        );
        assert!(
            motion.beat_depth_db < 0.5,
            "beat depth {} dB on one sinusoid",
            motion.beat_depth_db
        );
        assert!((motion.mean_hz - 440.0).abs() < 0.05, "{}", motion.mean_hz);
    }

    /// A pair a known distance apart returns that distance from its envelope's
    /// sign changes — at the amplitude ratio the recordings actually show
    /// (`docs/history/FUNDAMENTALS.md` §7.4: a companion 4–7 dB down), which is where the
    /// log envelope is still nearly one line.
    #[test]
    fn a_pair_beats_at_the_rate_it_is_split_by() {
        for split in [1.0, 1.5, 2.2] {
            let mut spectrum = Spectrum::new(&signal(
                &[(440.0, 1.0, 0.5), (440.0 + split, 0.5, 0.5)],
                4.0,
            ));
            let motion = partial_motion(&mut spectrum, 440.0, 20.0).expect("a partial");
            // One sign change over the window is 0.185 Hz, which is the
            // resolution this statistic has and all it claims; under 1 Hz the
            // cubic detrend biases it upward, which the module doc measures.
            assert!(
                (motion.beat_rate_hz - split).abs() <= 0.19,
                "split {split} came back as {}",
                motion.beat_rate_hz
            );
        }
    }

    /// Two *equal* components pass through an exact null, and the null is where
    /// all of their frequency movement is — which is the free-running
    /// construction's signature and what `placement` was promoted to catch.
    #[test]
    fn an_equal_pair_nulls_and_puts_all_its_wobble_there() {
        let mut spectrum = Spectrum::new(&signal(&[(440.0, 1.0, 0.5), (440.7, 1.0, 0.5)], 4.0));
        let motion = partial_motion(&mut spectrum, 440.0, 20.0).expect("a partial");
        assert!(
            motion.beat_depth_db > 15.0,
            "an equal pair nulls: {} dB",
            motion.beat_depth_db
        );
        assert!(
            motion.placement() < 0.4,
            "placement {} for a free pair",
            motion.placement()
        );
    }

    /// The companion inversion is the identity on a pair built to a known ratio.
    #[test]
    fn the_companion_a_beat_implies_is_the_pair_that_made_it() {
        for db in [-6.0, -12.0] {
            let a = 10f64.powf(db / 20.0);
            let mut spectrum = Spectrum::new(&signal(
                // Equal decay rates, so the ratio does not sweep across the
                // window and the depth is the one the pair was built with.
                &[(440.0, 1.0, 0.4), (441.0, a, 0.4)],
                4.0,
            ));
            let motion = partial_motion(&mut spectrum, 440.0, 20.0).expect("a partial");
            let implied = motion.companion_db().expect("a companion");
            assert!(
                (implied - db).abs() < 1.5,
                "asked {db} dB, implied {implied:.2} from {:.2} dB of depth",
                motion.beat_depth_db
            );
        }
    }
}

