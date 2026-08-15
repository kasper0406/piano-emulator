//! Per-partial forensics on the rendered instrument.
//!
//! `renders/jitter/JITTER.md` and `renders/jitter/EIGENMODE.md` measured the
//! string's interior with a complex demodulation of each partial, offline, in
//! `tuner/`. Everything the string milestone is gated on is a statement about
//! that measurement, so the measurement itself is ported here, verbatim in
//! structure, and the gates are assertions on it rather than paragraphs in a
//! report:
//!
//! * **equivalence** — the eigenmode construction replaced the free-running one
//!   (`FUNDAMENTALS.md` §5, `DECISIONS.md` 223-229). It cannot be bit-exact, so
//!   what is pinned is what a listener could tell apart, on
//!   `presets/default.toml`, and the contract is **0.5 cents of pitch, 5 % of
//!   T60, 0.5 dB of level**. Each of the three is asserted at that number on the
//!   quantity the construction sets, with whatever the *harness* adds asserted
//!   separately beside it rather than folded into one looser bound
//!   (`DECISIONS.md` 259): the partial's built centre inside 0.5 cents and the
//!   tracker's own window bias inside 0.75 on top; the strike's peak inside
//!   0.5 dB from A0 to C6, with C7 and C8 pinned at the +1.0 / +3.0 the
//!   construction is documented to cost there. The third, the whole-note T60,
//!   is pinned where it is *solved* —
//!   `string::tests::every_partials_t60_lands_on_its_own_anchor`, at 5 % on the
//!   p90 and at 5 % *plus one beat period* on every cell, because the statistic
//!   is the last crossing of a beating envelope and jumps by exactly that.
//! * **the metronome** — the shipped construction put the *same three beat
//!   rates* (0.270 / 0.350 / 0.520 Hz, `voicing.horizontal_offset_hz`) inside
//!   every partial of every key, so a held chord pulsed coherently at a rate no
//!   note chose. Nothing may share a beat rate with anything else any more.
//! * **the double decay** — the prompt sound and the aftersound are what a
//!   piano is recognised by, and a construction that stops beating by also
//!   deleting the tail has not solved anything.

use std::f64::consts::TAU;

use piano_emulator::preset::{FalseBeat, Preset, StrikeDirection};
use piano_emulator::string::PianoString;
use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_emulator::types::{Event, SAMPLE_RATE};
use rustfft::{num_complex::Complex64, FftPlanner};

const SR: f64 = SAMPLE_RATE as f64;

/// The three keys `renders/jitter` measures, so every number here has a
/// published counterpart.
const KEYS: [(u8, &str); 3] = [(60, "C4"), (45, "A2"), (84, "C6")];
const MAX_PARTIAL: usize = 4;
const VELOCITY: u8 = 90;

/// How long a probe render is. Long enough for the tail window below and for
/// the transform not to wrap into its own input.
const RENDER_S: f64 = 4.5;

/// `JITTER.md`'s analysis window, in seconds since the strike.
const T0_S: f64 = 0.3;
const T1_S: f64 = 3.0;
/// The two windows the double decay is read from.
const PROMPT_LO_S: f64 = 0.10;
const PROMPT_HI_S: f64 = 0.60;
const TAIL_LO_S: f64 = 1.50;
const TAIL_HI_S: f64 = 3.50;

/// Time-domain standard deviation of the Gaussian band-pass — 31.8 Hz wide,
/// never wider than a quarter of the carrier. `JITTER.md`'s value.
const SMOOTH_SIGMA_S: f64 = 0.005;
/// Rate the demodulated track is decimated to.
const TRACK_HZ: f64 = 1000.0;
/// The modulation band the envelope is read over.
const MOD_LO_HZ: f64 = 0.1;
const MOD_HI_HZ: f64 = 20.0;
/// 5.46 s at 48 kHz, longer than anything analysed.
const FFT_N: usize = 1 << 18;
/// A partial this far under its own background is not tracked.
const MIN_PEAK_DB: f64 = 10.0;

// ------------------------------------------------------------ demodulation

struct Spectrum {
    bins: Vec<Complex64>,
}

impl Spectrum {
    fn new(signal: &[f64], planner: &mut FftPlanner<f64>) -> Spectrum {
        let mut bins: Vec<Complex64> = (0..FFT_N)
            .map(|n| Complex64::new(signal.get(n).copied().unwrap_or(0.0), 0.0))
            .collect();
        planner.plan_fft_forward(FFT_N).process(&mut bins);
        Spectrum { bins }
    }

    fn hz(m: usize) -> f64 {
        m as f64 * SR / FFT_N as f64
    }

    /// The strongest bin within `+-half_width` of `nominal`, refined by a
    /// parabolic fit, and how far it stands over the band's median magnitude.
    fn peak_near(&self, nominal: f64, half_width: f64) -> (f64, f64) {
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
        band.sort_by(|a, b| a.partial_cmp(b).expect("magnitudes are finite"));
        let median = band[band.len() / 2].max(f64::MIN_POSITIVE);
        let peak_db = 20.0 * (mag(best) / median).log10();
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
    /// `carrier`, demodulated to zero and decimated to [`TRACK_HZ`].
    fn demodulate(
        &self,
        carrier: f64,
        t0: f64,
        t1: f64,
        planner: &mut FftPlanner<f64>,
    ) -> Vec<Complex64> {
        let sigma_f = (1.0 / (TAU * SMOOTH_SIGMA_S)).min(carrier / 4.0);
        let mut z = vec![Complex64::new(0.0, 0.0); FFT_N];
        let span = (6.0 * sigma_f * FFT_N as f64 / SR).ceil() as usize;
        let centre = (carrier * FFT_N as f64 / SR).round() as usize;
        let lo = centre.saturating_sub(span).max(1);
        let hi = (centre + span).min(FFT_N / 2 - 1);
        for (m, bin) in z.iter_mut().enumerate().take(hi + 1).skip(lo) {
            let u = (Spectrum::hz(m) - carrier) / sigma_f;
            *bin = self.bins[m] * (2.0 * (-0.5 * u * u).exp());
        }
        planner.plan_fft_inverse(FFT_N).process(&mut z);
        let scale = 1.0 / FFT_N as f64;
        let step = (SR / TRACK_HZ).round() as usize;
        let from = (t0 * SR) as usize;
        let to = ((t1 * SR) as usize + step).min(FFT_N - 1);
        (from..=to)
            .step_by(step)
            .map(|n| {
                let phase = -TAU * carrier * n as f64 / SR;
                z[n] * scale * Complex64::from_polar(1.0, phase)
            })
            .collect()
    }
}

/// Everything one partial of one render contributes to the gates.
struct PartialStats {
    /// Power-weighted mean instantaneous frequency over the analysis window.
    mean_hz: f64,
    /// Peak-to-trough span of the band-limited log envelope, dB.
    beat_depth_db: f64,
    /// Rate the strongest modulation line sits at, from a zero-padded transform
    /// of the band-limited envelope — located rather than binned.
    beat_line_hz: f64,
    /// Slope of the log envelope over the prompt window, dB/s.
    prompt_db_s: f64,
    /// Slope of the log envelope over the tail window, dB/s.
    tail_db_s: f64,
    /// Where the tail's straight line extrapolates back to at the strike,
    /// relative to the prompt's — the aftersound level, dB below the prompt.
    aftersound_db: f64,
}

fn track_partial(
    spectrum: &Spectrum,
    nominal_hz: f64,
    half_width: f64,
    planner: &mut FftPlanner<f64>,
) -> Option<PartialStats> {
    let (carrier_hz, peak_db) = spectrum.peak_near(nominal_hz, half_width);
    if peak_db < MIN_PEAK_DB {
        return None;
    }
    let y = spectrum.demodulate(carrier_hz, T0_S, T1_S, planner);
    if y.len() < 64 {
        return None;
    }
    let mut inst = Vec::with_capacity(y.len() - 1);
    let mut weight = Vec::with_capacity(y.len() - 1);
    for j in 0..y.len() - 1 {
        let d = y[j + 1] * y[j].conj();
        inst.push(carrier_hz + d.arg() * TRACK_HZ / TAU);
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

    let amp_db: Vec<f64> = weight.iter().map(|w| 10.0 * w.max(1e-300).log10()).collect();
    let residual = detrended(&amp_db, 3);
    let band = band_limited(&residual, TRACK_HZ);
    let mut sorted = band.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("envelope values are finite"));
    let at = |q: f64| sorted[((sorted.len() - 1) as f64 * q) as usize];
    let beat_line_hz = strongest_line(&band, TRACK_HZ);

    let long = spectrum.demodulate(carrier_hz, PROMPT_LO_S, TAIL_HI_S, planner);
    let decay_db: Vec<f64> = long
        .iter()
        .map(|z| 20.0 * z.norm().max(1e-300).log10())
        .collect();
    let (prompt_db_s, prompt_at_0) =
        line_fit(&decay_db, PROMPT_LO_S, TRACK_HZ, PROMPT_LO_S, PROMPT_HI_S)
            .unwrap_or((f64::NAN, f64::NAN));
    let (tail_db_s, tail_at_0) = line_fit(&decay_db, PROMPT_LO_S, TRACK_HZ, TAIL_LO_S, TAIL_HI_S)
        .unwrap_or((f64::NAN, f64::NAN));

    Some(PartialStats {
        mean_hz,
        beat_depth_db: at(0.95) - at(0.05),
        beat_line_hz,
        prompt_db_s,
        tail_db_s,
        aftersound_db: tail_at_0 - prompt_at_0,
    })
}

/// Where the strongest modulation line of an already band-limited envelope
/// sits, and how far it stands over the median of the same band.
///
/// Zero-padded eight times, because the point of the statistic is *which rate*
/// and the analysis window is only 2.7 s: a bare transform puts 0.27, 0.35 and
/// 0.52 Hz in the same bin, which is precisely the three numbers that have to
/// be told apart.
fn strongest_line(x: &[f64], rate: f64) -> f64 {
    let n = x.len();
    if n < 64 {
        return 0.0;
    }
    let padded = (8 * n).next_power_of_two();
    let window: Vec<f64> = (0..n)
        .map(|j| 0.5 - 0.5 * (TAU * j as f64 / n as f64).cos())
        .collect();
    let mut buf: Vec<Complex64> = (0..padded)
        .map(|j| {
            if j < n {
                Complex64::new(x[j] * window[j], 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            }
        })
        .collect();
    FftPlanner::<f64>::new()
        .plan_fft_forward(padded)
        .process(&mut buf);
    let bin_hz = rate / padded as f64;
    let lo = (MOD_LO_HZ / bin_hz).ceil().max(1.0) as usize;
    let hi = ((MOD_HI_HZ / bin_hz).floor() as usize).min(padded / 2 - 1);
    if hi <= lo {
        return 0.0;
    }
    let power: Vec<f64> = (lo..=hi).map(|m| buf[m].norm_sqr()).collect();
    let mut best = 0usize;
    for (i, &p) in power.iter().enumerate() {
        if p > power[best] {
            best = i;
        }
    }
    (lo + best) as f64 * bin_hz
}

/// `x` with everything outside [`MOD_LO_HZ`]–[`MOD_HI_HZ`] removed, zero phase.
fn band_limited(x: &[f64], rate: f64) -> Vec<f64> {
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
fn detrended(x: &[f64], degree: usize) -> Vec<f64> {
    let n = x.len();
    let cols = degree + 1;
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

/// Least-squares slope and intercept of `y` against `t`, over the samples whose
/// time falls in `[lo, hi]`. Returns `(slope per second, value at t = 0)`.
fn line_fit(y: &[f64], t0: f64, rate: f64, lo: f64, hi: f64) -> Option<(f64, f64)> {
    let mut n = 0.0f64;
    let (mut st, mut sy, mut stt, mut sty) = (0.0, 0.0, 0.0, 0.0);
    for (i, &v) in y.iter().enumerate() {
        let t = t0 + i as f64 / rate;
        if t < lo || t > hi || !v.is_finite() {
            continue;
        }
        n += 1.0;
        st += t;
        sy += v;
        stt += t * t;
        sty += t * v;
    }
    if n < 8.0 {
        return None;
    }
    let denom = n * stt - st * st;
    if denom.abs() < 1e-12 {
        return None;
    }
    let slope = (n * sty - st * sy) / denom;
    Some((slope, (sy - slope * st) / n))
}

// ----------------------------------------------------------------- the rig

/// Renders one note of `preset` and returns its mono sum, frame 0 at the strike.
fn render(preset: &Preset, key: u8, vel: u8) -> Vec<f64> {
    let events = [RenderEvent::new(0.0, Event::NoteOn { key, vel })];
    let (l, r) = render_to_buffer(preset, &events, RENDER_S as f32);
    l.iter()
        .zip(&r)
        .map(|(a, b)| 0.5 * (f64::from(*a) + f64::from(*b)))
        .collect()
}

/// Every partial of one key, measured.
fn measure(preset: &Preset, key: u8, vel: u8) -> Vec<Option<PartialStats>> {
    let mono = render(preset, key, vel);
    let mut planner = FftPlanner::<f64>::new();
    let spectrum = Spectrum::new(&mono, &mut planner);
    let params = preset.string_params(key);
    (1..=MAX_PARTIAL)
        .map(|k| {
            let nominal = f64::from(params.partial_freq(k));
            let half = (nominal * 0.01).max(3.0);
            track_partial(&spectrum, nominal, half, &mut planner)
        })
        .collect()
}

/// The cells the gates below are read over: the three keys `renders/jitter`
/// measures, four partials each.
fn cells(preset: &Preset) -> Vec<(&'static str, usize, f64, PartialStats)> {
    let mut out = Vec::new();
    for (key, name) in KEYS {
        let params = preset.string_params(key);
        for (i, cell) in measure(preset, key, VELOCITY).into_iter().enumerate() {
            let Some(stats) = cell else { continue };
            // A partial whose log envelope swings by more than 25 dB inside the
            // analysis window has gone under the room before the window closes,
            // and what is being measured after that is the room: C6's third and
            // fourth read 46.9 and 69.7 dB of "beat depth" for exactly that
            // reason, and every statistic taken on them is noise. Named here
            // rather than silently included. The deepest cell that *is* a
            // partial reads 14.5 dB.
            if stats.beat_depth_db > 25.0 || !stats.aftersound_db.is_finite() {
                continue;
            }
            out.push((name, i + 1, f64::from(params.partial_freq(i + 1)), stats));
        }
    }
    out
}

/// Every partial sits where the preset's tuning asks for it.
///
/// This is the equivalence contract's pitch half, and it is stated against
/// nominal pitch rather than against the free-running construction because the
/// free-running one did not meet it: `voicing.horizontal_offset_hz` added a
/// fixed *number of hertz* to every horizontal mode, which is 0.35 Hz at C4's
/// fundamental and the same 0.35 Hz at A0's — 22 cents there — so the composite
/// came out sharp by an amount that grew as the note got lower. Measured on the
/// same cells with the same code, the free-running construction read
/// **+0.87 / +0.75 / +0.40 / +0.89** cents at C4, **+1.72 / +0.67 / +0.25 /
/// −0.09** at A2 and **+0.92 / +1.09 / +0.36 / +0.09** at C6; this one reads
/// −0.65 to +0.23 over the same twelve.
///
/// **The contract is 0.5 cents, and this splits the one-cent gate into the two
/// halves that add up to it** rather than leaving the looser number to stand for
/// both. What the construction sets is where the partial's radiated centre is,
/// and that is inside 0.5 cents over every partial of every key
/// (`string::tests::partials_sit_where_the_formula_asks_for_them`, worst 0.166).
/// What a render adds is the tracker: a 2.7 s complex demodulation through the
/// soundboard's diffuse field, with the partial's own beat inside the window and
/// its neighbours' skirts on either side. Asserting the two separately says
/// which is which, and the second is a measurement of the harness rather than of
/// the instrument. Over these eleven cells the construction's own offset is
/// **0.073 cents** at worst — a seventh of the contract, and against the 0.166
/// the construction-side test measures over the whole compass — while the
/// tracker's is **0.694**, negative on nine of the eleven, which is a window
/// bias and not a tuning (`DECISIONS.md` 259).
#[test]
fn every_partial_sits_at_the_pitch_the_preset_asks_for() {
    let preset = Preset::default();
    let measured = cells(&preset);
    assert!(measured.len() >= 10, "only {} cells tracked", measured.len());
    let mut worst = (0.0f64, 0.0f64);
    for (name, k, nominal, stats) in &measured {
        let key = KEYS
            .iter()
            .find(|(_, n)| n == name)
            .map(|(key, _)| *key)
            .expect("a cell names one of the keys it was measured on");
        let string = PianoString::new(
            preset.string_params(key),
            &preset.voicing,
            preset.partial_shaping(key),
        );
        // Where the construction put the partial, against where the preset's
        // own formula asks for it: the contract's half-cent.
        let built = 1200.0 * (f64::from(string.partial_freq(*k)) / nominal).log2();
        // ... and what the render adds on top of that, which is the tracker.
        let rendered = 1200.0 * (stats.mean_hz / nominal).log2();
        worst = (worst.0.max(built.abs()), worst.1.max((rendered - built).abs()));
        assert!(
            built.abs() < 0.5,
            "{name} k={k} is *built* {built:+.3} cents off its nominal {nominal:.3} Hz"
        );
        // 0.75 and not the contract's 0.5, because this half is not the
        // contract: it is what the harness costs, and it is measured rather
        // than allowed for. C6's third partial renders 0.708 cents flat, of
        // which 0.051 is where the construction put it and 0.656 is the window.
        assert!(
            (rendered - built).abs() < 0.75,
            "{name} k={k} renders {rendered:+.3} cents off nominal where it was built \
             {built:+.3}: the tracker moved it by {:+.3}",
            rendered - built
        );
        assert!(
            rendered.abs() < 1.0,
            "{name} k={k} sits {rendered:+.3} cents off its nominal {nominal:.3} Hz"
        );
    }
    println!(
        "pitch: worst construction offset {:.3} cents, worst tracker offset {:.3} cents",
        worst.0, worst.1
    );
}

/// The metronome is gone from the rendered note.
///
/// `voicing.horizontal_offset_hz` put the *same three beat rates* — 0.270,
/// 0.350 and 0.520 Hz — inside every partial of every key, so a held chord
/// modulated coherently at a rate no note in it had chosen
/// (`renders/jitter/JITTER.md`, every row of every component table;
/// `FUNDAMENTALS.md` §2.3).
///
/// What a render can say about that, and what it cannot, are worth separating.
/// The three shipped numbers are **not** tested for literally here: on
/// `presets/default.toml` the unison's own detune beats live at 0.4-1.1 Hz and
/// land within a few hundredths of 0.520 by coincidence, so a literal test is a
/// coincidence trap rather than a measurement. What a render *can* say is that
/// nothing is note-independent, and that is what is asserted — no rate is the
/// dominant modulation line of more than a quarter of the cells, over three keys
/// two and a half octaves apart.
///
/// The strong form of the claim — that no *pair* of modes anywhere on the
/// compass shares a beat rate, and that the three shipped numbers appear in
/// fewer than one cell in a hundred — is exhaustive and lives where it can be
/// checked exhaustively, in
/// `string::tests::no_beat_rate_is_shared_across_the_compass`.
#[test]
fn no_shared_beat_rate_survives_into_the_render() {
    let preset = Preset::default();
    let measured = cells(&preset);
    let lines: Vec<f64> = measured.iter().map(|(_, _, _, s)| s.beat_line_hz).collect();
    assert!(lines.len() >= 10);

    // No rate is note-independent: nothing is the dominant line of more than a
    // quarter of the cells. Three keys two and a half octaves apart, four
    // partials each — under `horizontal_offset_hz` a single number was a beat
    // rate of every one of them.
    // One bin of the zero-padded transform is 0.031 Hz, so two rates inside
    // 0.02 Hz of each other are the same measured rate.
    for &a in &lines {
        let shared = lines.iter().filter(|&&b| (a - b).abs() < 0.02).count();
        assert!(
            shared * 3 <= lines.len(),
            "{shared} of {} cells modulate at {a:.3} Hz",
            lines.len()
        );
    }
    // ... and they are spread rather than clustered: the fastest cell moves at
    // more than twice the rate of the slowest, where a metronome would put every
    // one of them on the same number.
    let (lo, hi) = lines.iter().fold((f64::MAX, 0.0f64), |(l, h), &x| (l.min(x), h.max(x)));
    assert!(
        hi > 2.0 * lo,
        "every cell modulates between {lo:.3} and {hi:.3} Hz — that is one rate, not twelve"
    );
}

/// The double decay survives the change of construction.
///
/// It is the property a piano is recognised by and the one a unison that has
/// stopped beating is most at risk of losing: `renders/jitter/EIGENMODE.md`
/// measured the prototype's and found it kept at C4 and A2 and broken at C6.
/// Read as two straight lines through the partial's log envelope — 0.1–0.6 s and
/// 1.5–3.5 s — the claim is that the second is shallower than the first, on
/// every partial of the two keys the prototype kept it on.
#[test]
fn the_prompt_sound_and_the_aftersound_are_still_two_decays() {
    let preset = Preset::default();
    for (name, k, _, stats) in cells(&preset) {
        if name == "C6" {
            continue;
        }
        assert!(
            stats.tail_db_s > stats.prompt_db_s + 1.0,
            "{name} k={k}: prompt {:.2} dB/s, tail {:.2} dB/s — the tail is not the \
             slower of the two, so there is no aftersound",
            stats.prompt_db_s,
            stats.tail_db_s
        );
    }
}

/// C6, where the prototype's aftersound broke, and where it does not here.
///
/// `FUNDAMENTALS.md` §7.3 measured the prototype's C6 aftersound level going
/// 4.9 → 21.2 dB of error against the recording and blamed `detune_cents`
/// fitted through the free-running forward model — `mu = 1.74` on the shipped
/// preset, which the verification errata corrects to ~1.05 under the resolved
/// `radiated_share`. Softened, not overturned: the group is close enough to
/// locking that its three vertical modes come out within a few decibels of each
/// other and there is little left to be a quiet slow survivor. What this asserts
/// is the part that does not depend on which preset is loaded — C6's fundamental
/// still has a tail slower than its prompt — and it is the cell to watch if the
/// treble ever needs `detune_cents` re-fitted under the new forward model
/// (`FUNDAMENTALS.md` §7.5 step 4).
#[test]
fn c6_still_has_an_aftersound_under_its_fundamental() {
    let preset = Preset::default();
    let c6: Vec<_> = cells(&preset)
        .into_iter()
        .filter(|(name, _, _, _)| *name == "C6")
        .collect();
    assert!(!c6.is_empty(), "C6 tracked nothing at all");
    let (_, k, _, stats) = &c6[0];
    assert_eq!(*k, 1);
    assert!(
        stats.tail_db_s > stats.prompt_db_s + 1.0,
        "C6 k=1: prompt {:.2} dB/s, tail {:.2} dB/s",
        stats.prompt_db_s,
        stats.tail_db_s
    );
}

/// The note's loudness does not move.
///
/// The equivalence contract's level half, on the quantity the gain staging
/// actually sets — the peak the strike produces, which is what
/// `calibrate::MechanismCalibration` quotes every `[noise]` level against. The
/// free-running construction's numbers, measured with this code on this preset,
/// are the pins.
///
/// **The contract is 0.5 dB and it is pinned over the whole compass**, six keys
/// at three velocities each, rather than over the three bass and midrange keys
/// this test used to cover at 0.7 dB. The two keys where 0.5 dB does not hold
/// are the two `DECISIONS.md` 229 names, and they are pinned at the size it
/// measured rather than left out: **C7 up to +1.01 dB and C8 up to +2.99**,
/// where A0 to C6 come back inside +0.49. The treble is where the coupled
/// construction differs most, because a short high partial's whole decay is a
/// couple of beat periods long and the T60 normalisation has the least room to
/// place it (`DECISIONS.md` 259).
///
/// What is *not* pinned here, and cannot be, is the level in the middle of the
/// note: the prompt decay is now derived from the bridge rather than asserted by
/// `Voicing::vertical_decay_factor`, and the derived one is shallower, so the
/// 0.2–2.0 s RMS rises by up to 2.3 dB in the bass. That is the documented break
/// (`DECISIONS.md` 228).
#[test]
fn the_strike_lands_at_the_level_it_always_did() {
    // The master gain has moved once since `b929658`, deliberately and by a
    // known factor: `types::OUTPUT_GAIN` 9.0 -> 4.95 to put the ten-note
    // fortissimo chord back on the safety limiter's threshold
    // (`DECISIONS.md` 42, 277). That is a pure scale on the sounding path — the
    // two end-of-note floors were frozen in internal units in the same change
    // so that it would be — so it is carried here as one term rather than
    // folded into eighteen numbers, and every departure below is the same
    // departure it was before the recalibration, to a hundredth of a decibel.
    const MASTER_GAIN_DB: f64 = -5.1927; // 20 log10(4.95 / 9.0)

    // Peak of the render, dB, from the free-running construction at `b929658`,
    // and the budget this key is held to against it.
    const PINNED: [(u8, u8, f64, f64); 18] = [
        (21, 40, -32.59, 0.5),
        (21, 90, -19.73, 0.5),
        (21, 120, -12.40, 0.5),
        (45, 40, -34.46, 0.5),
        (45, 90, -21.59, 0.5),
        (45, 120, -14.27, 0.5),
        (60, 40, -35.43, 0.5),
        (60, 90, -21.41, 0.5),
        (60, 120, -13.71, 0.5),
        (84, 40, -34.12, 0.5),
        (84, 90, -21.28, 0.5),
        (84, 120, -14.00, 0.5),
        // The two documented exceptions, at the size they were measured.
        (96, 40, -32.08, 1.1),
        (96, 90, -18.53, 1.1),
        (96, 120, -10.79, 1.1),
        (108, 40, -32.78, 3.1),
        (108, 90, -19.52, 3.1),
        (108, 120, -11.55, 3.1),
    ];
    let preset = Preset::default();
    let mut worst = (0.0f64, 0u8, 0u8);
    for (key, vel, pinned, budget) in PINNED {
        let want = pinned + MASTER_GAIN_DB;
        let events = [RenderEvent::new(0.0, Event::NoteOn { key, vel })];
        let (l, r) = render_to_buffer(&preset, &events, 3.0);
        let peak = l.iter().chain(r.iter()).fold(0.0f32, |m, &x| m.max(x.abs()));
        let got = 20.0 * f64::from(peak).max(1e-30).log10();
        if (got - want).abs() > worst.0 {
            worst = ((got - want).abs(), key, vel);
        }
        assert!(
            (got - want).abs() < budget,
            "key {key} at velocity {vel} peaks at {got:.2} dBFS against the \
             free-running construction's {want:.2}, past its {budget} dB budget"
        );
    }
    println!(
        "strike level against the free-running construction: worst {:.2} dB, at key {} velocity {}",
        worst.0, worst.1, worst.2
    );
}


// ------------------------------------- the two motion mechanisms, rendered

/// `presets/default.toml` with one key's unison narrowed to 0.3 cents.
///
/// The false beat has to be *told apart* from the unison, which is the whole of
/// `FUNDAMENTALS.md` §7.4's argument, and on the default preset at C4 the two
/// live in the same band: 2.9 cents of detune is 0.88 Hz of spread at the second
/// partial, which is the size of the split under test. Narrowing the unison of
/// the one key the test measures leaves the mechanism alone and takes the other
/// mechanism out of the measurement — the same discipline `renders/jitter`'s
/// `02_no_detune` control uses, and the reason that control exists.
fn quiet_unison(key: u8) -> Preset {
    let mut preset = Preset::default();
    preset.notes.detune_cents[piano_emulator::types::key_index(key).expect("a real key")] = 0.3;
    preset
}

/// ... and the same key with one within-string split on partial `k`.
fn with_split(key: u8, k: usize, hz: f32, db: f32) -> Preset {
    let mut preset = quiet_unison(key);
    let i = piano_emulator::types::key_index(key).expect("a real key");
    preset.notes.false_beat = vec![Vec::new(); 88];
    preset.notes.false_beat[i] = vec![FalseBeat {
        k: k as u16,
        hz,
        db,
    }];
    preset.validate().expect("a split preset validates");
    preset
}

/// A within-string split beats on the partial it names, at the rate it names,
/// and leaves every other partial of the key exactly where it was.
///
/// This is the render's half of the mechanism; the construction's half — that
/// the companion stands at the *level* the table asks for, to a tenth of a
/// decibel — is `string::tests::a_false_beat_splits_the_partial_it_names_and_
/// nothing_beside_it`, because the level is a statement about the strike instant
/// and a render measures a window.
///
/// What the window says instead is the rate, and the rate is the finding: the
/// recording's companions sit **0.7–1.5 Hz** away at a spacing that does not
/// scale with the partial number (`FUNDAMENTALS.md` §7.4), where a unison
/// mistuning must scale with it and the bridge's polarization split is a hundred
/// times narrower. Three splits are asked for and three rates come back.
///
/// The *depth* over this window is deliberately not asserted against the level.
/// The split plane is the horizontal one, so it decays more slowly than the mode
/// it beats against, and the amplitude ratio therefore sweeps through 1 somewhere
/// inside any long window whatever level it started at — §2.4's guaranteed null,
/// arriving here as the mechanism working rather than as a defect. What is
/// asserted is that the beat is *measurable* where the control is flat.
#[test]
fn a_false_beat_beats_on_the_partial_it_names_and_nowhere_else() {
    const KEY: u8 = 60;
    const SPLIT: usize = 2;
    let control = measure(&quiet_unison(KEY), KEY, VELOCITY);
    let split = measure(&with_split(KEY, SPLIT, 1.0, -6.0), KEY, VELOCITY);

    let want = control[SPLIT - 1].as_ref().expect("the control tracks k=2");
    let got = split[SPLIT - 1].as_ref().expect("the split tracks k=2");
    assert!(
        want.beat_depth_db < 0.5,
        "the control already beats by {:.2} dB, so the split is not what is being \
         measured",
        want.beat_depth_db
    );
    assert!(
        got.beat_depth_db > 5.0,
        "the split partial beats by only {:.2} dB",
        got.beat_depth_db
    );
    assert!(
        (got.beat_line_hz - 1.0).abs() < 0.15,
        "the split beats at {:.3} Hz where the table asked for 1.0",
        got.beat_line_hz
    );

    // Nowhere else: every other partial of the same key is what it was, to the
    // bit, because a false beat is a defect of one wire at one partial.
    for k in 1..=MAX_PARTIAL {
        if k == SPLIT {
            continue;
        }
        let (a, b) = (&control[k - 1], &split[k - 1]);
        match (a, b) {
            (Some(a), Some(b)) => {
                // Not bit-for-bit, and it cannot be: the two presets are
                // different files, so the render differs by the last bits of
                // every f32 in it. The scale that matters is the one the
                // forensics measure at — 0.05 dB is the shipped engine's whole
                // velocity spread — and against a 13 dB beat it is nothing.
                assert!(
                    (a.beat_depth_db - b.beat_depth_db).abs() < 0.05,
                    "partial {k} started beating: {:.3} -> {:.3} dB",
                    a.beat_depth_db,
                    b.beat_depth_db
                );
                assert_eq!(a.beat_line_hz, b.beat_line_hz, "partial {k} changed rate");
                assert!(
                    (a.mean_hz - b.mean_hz).abs() < 1.0e-3,
                    "partial {k} moved {:.4} Hz",
                    b.mean_hz - a.mean_hz
                );
            }
            (None, None) => {}
            _ => panic!("partial {k} appeared or vanished"),
        }
    }

    // The rate is the table's, over the band the mechanism was measured in: the
    // recording's C4 and A2 companions sit at 0.74-1.48 Hz and its C6 ones at
    // 2.22-5.19 (`renders/jitter/EIGENMODE.md`).
    for (hz, tolerance) in [(0.5f32, 0.2f64), (1.0, 0.15), (2.5, 0.2)] {
        let cells = measure(&with_split(KEY, SPLIT, hz, -6.0), KEY, VELOCITY);
        let line = cells[SPLIT - 1]
            .as_ref()
            .expect("the split tracks k=2")
            .beat_line_hz;
        assert!(
            (line - f64::from(hz)).abs() < tolerance,
            "a {hz} Hz split beats at {line:.3} Hz"
        );
    }
}

/// Velocity moves the rendered beat structure, and without a strike direction it
/// cannot.
///
/// `FUNDAMENTALS.md`'s single cleanest discriminator, and the one column
/// (`B2`, velocity coherence) that no construction has ever passed: the
/// recording's beat depth moves **1.90 dB** over velocities 40 / 90 / 120 and
/// the shipped engine's moves **0.054** — "nothing stochastic or
/// amplitude-coupled can hold 0.008 cents across an 80-point velocity span"
/// (Part II §II.3). The reason is structural rather than a fitting failure:
/// `u = s_j g_k` scales uniformly with velocity, so `c = V^-1 u` scales
/// uniformly and every ratio in the mixture is a constant (§7.3).
///
/// Both halves are asserted here, on the same key, the same partials and the
/// same statistic: the neutral preset is **exactly** invariant — bit for bit,
/// which is what a deterministic renderer of a velocity-independent mixture has
/// to be — and one `[voicing.strike_direction]` moves it by decibels.
#[test]
fn only_a_strike_direction_makes_the_render_depend_on_velocity() {
    const KEY: u8 = 60;
    const VELOCITIES: [u8; 3] = [40, 90, 120];

    // The engine as it is: the mixture is a constant, so the beat structure is
    // the *same number* at every velocity and not merely a close one.
    let flat = Preset::default();
    let mut reference: Vec<Vec<f64>> = Vec::new();
    for vel in VELOCITIES {
        reference.push(
            measure(&flat, KEY, vel)
                .iter()
                .map(|c| c.as_ref().map_or(f64::NAN, |s| s.beat_depth_db))
                .collect(),
        );
    }
    for k in 0..MAX_PARTIAL {
        for row in &reference[1..] {
            // 0.05 dB, which is the 0.054 the forensics measured on the
            // shipped engine over these same three layers and a fortieth of the
            // recording's 1.90 — the number Column B2 fails on. The residual is
            // the *hammer*, whose felt is nonlinear, and not the mixture, which
            // is a constant: measured here at 0.002 to 0.016 dB.
            assert!(
                (row[k] - reference[0][k]).abs() < 0.05,
                "partial {} moved with velocity without a strike direction: \
                 {:.4} against {:.4} dB",
                k + 1,
                row[k],
                reference[0][k]
            );
        }
    }

    let mut voiced = Preset::default();
    voiced.voicing.strike_direction = Some(StrikeDirection {
        vh_db_at_pp: -8.0,
        vh_db_at_ff: 8.0,
        share_tilt: 0.2,
    });
    voiced.validate().expect("a voiced strike direction validates");
    let mut depths: Vec<Vec<f64>> = Vec::new();
    for vel in VELOCITIES {
        depths.push(
            measure(&voiced, KEY, vel)
                .iter()
                .map(|c| c.as_ref().map_or(f64::NAN, |s| s.beat_depth_db))
                .collect(),
        );
    }
    // Column B2's own statistic: the spread of the beat depth across the three
    // velocity layers the forensics render, averaged over the partials. The
    // reference reads 1.90 dB and the shipped engine 0.054.
    let mut spread = 0.0f64;
    let mut counted = 0.0f64;
    for k in 0..MAX_PARTIAL {
        let column: Vec<f64> = depths.iter().map(|row| row[k]).collect();
        if column.iter().any(|x| !x.is_finite()) {
            continue;
        }
        let (lo, hi) = column
            .iter()
            .fold((f64::MAX, 0.0f64), |(l, h), &x| (l.min(x), h.max(x)));
        spread += hi - lo;
        counted += 1.0;
    }
    assert!(counted >= 3.0, "only {counted} partials tracked at all three velocities");
    let mean = spread / counted;
    assert!(
        mean > 1.0,
        "the beat depth moves {mean:.3} dB over the velocity span, against the \
         recording's 1.90 and the velocity-independent construction's 0.000"
    );
}

/// The metronome, asked of the instrument that is actually shipped.
///
/// `string::tests::no_beat_rate_is_shared_across_the_compass` runs the same
/// census on `presets/default.toml` with **no** per-partial shaping, so it reads
/// the construction alone. This one reads the fitted preset with its own
/// `notes.false_beat` on it, which is the only place a fixed rate could come
/// back: a false beat is written in *hertz*, so several keys whose fit ran into
/// the schema's 3.0 Hz ceiling carry the same asked offset — six of the 71
/// shipped rows do, and three more sit on the fit's own 0.37 Hz floor
/// (`DECISIONS.md` 250).
///
/// What that is worth is the point of the test. The offset is one term on the
/// diagonal of one block, so what a listener hears is the eigenvalue split it
/// produces and not the number itself, and it reaches one partial of one key.
/// Against `horizontal_offset_hz`, which was a beat rate of **every** partial of
/// **every** key, the bar here is the same one the construction is held to: no
/// rate shared by more than one cell in fifty.
#[test]
fn no_beat_rate_is_shared_across_the_measured_presets_compass() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../presets/salamander-c5.toml");
    let preset = Preset::load(&path).expect("the measured preset loads");
    let mut cells: Vec<(u8, usize, Vec<f64>)> = Vec::new();
    for key in 21..=108u8 {
        let string = PianoString::new(
            preset.string_params(key),
            &preset.voicing,
            preset.partial_shaping(key),
        );
        for k in 1..=string.partial_count() {
            let modes = string.partial_modes(k);
            let mut rates = Vec::new();
            for (i, a) in modes.iter().enumerate() {
                for b in &modes[i + 1..] {
                    rates.push(f64::from((a.hz - b.hz).abs()));
                }
            }
            cells.push((key, k, rates));
        }
    }
    assert!(cells.len() > 3000, "only {} cells", cells.len());

    // One millihertz is a beat period of a quarter of an hour: two rates that
    // close are one rate by any measure a listener has.
    const SAME_HZ: f64 = 1.0e-3;
    let mut bins: std::collections::HashMap<i64, std::collections::HashSet<(u8, usize)>> =
        std::collections::HashMap::new();
    for (key, k, rates) in &cells {
        for r in rates {
            if !(0.05..5.0).contains(r) {
                continue;
            }
            for bin in [(r / SAME_HZ).floor() as i64, (r / SAME_HZ).ceil() as i64] {
                bins.entry(bin).or_default().insert((*key, *k));
            }
        }
    }
    let (worst_bin, worst) = bins
        .iter()
        .max_by_key(|(_, c)| c.len())
        .map(|(b, c)| (*b, c.len()))
        .unwrap_or((0, 0));
    // The three the deleted field asserted, and the two rails the fit can run
    // into, counted by name.
    let share = |hz: f64| {
        cells
            .iter()
            .filter(|(_, _, r)| r.iter().any(|x| (x - hz).abs() < 5.0e-3))
            .count()
    };
    println!(
        "measured preset: the most-shared beat rate is {:.3} Hz, in {worst} of {} cells \
         ({:.2} %); 0.270/0.350/0.520 Hz in {}/{}/{}; the schema's 3.0 Hz ceiling in {}, its \
         0.37 Hz floor in {}",
        worst_bin as f64 * SAME_HZ,
        cells.len(),
        100.0 * worst as f64 / cells.len() as f64,
        share(0.270),
        share(0.350),
        share(0.520),
        share(3.0),
        share(0.37),
    );
    assert!(
        worst * 50 < cells.len(),
        "one beat rate is shared by {worst} of {} cells",
        cells.len()
    );
}

/// The band truncation is inert on every preset in the repository.
///
/// `PianoString::new` stops the series at the first partial whose solved modes
/// leave the Nyquist band, which is the same rule `StringParams::partial_count`
/// applies to the undetuned series and is what makes the construction total on
/// any preset `validate` accepts (`DECISIONS.md` 257). Total is not the same as
/// silent, so what has to be pinned is that it never fires on an instrument: the
/// backstop exists for a tuning or a fitted decay law that spends more band than
/// the gap between `MAX_PARTIAL_RATIO` and one half, and neither shipped file is
/// within 1.7 kHz of doing that.
#[test]
fn the_shipped_presets_build_every_partial_their_series_asks_for() {
    for name in ["default.toml", "salamander-c5.toml"] {
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../presets/{name}"));
        let preset = Preset::load(&path).expect("a shipped preset loads");
        let mut highest = 0.0f32;
        for key in 21..=108u8 {
            let params = preset.string_params(key);
            let asked = params.partial_count();
            let string =
                PianoString::new(params, &preset.voicing, preset.partial_shaping(key));
            assert_eq!(
                string.partial_count(),
                asked,
                "{name} key {key}: the series was truncated at {} of {asked} partials",
                string.partial_count()
            );
            for k in 1..=asked {
                for mode in string.partial_modes(k) {
                    highest = highest.max(mode.hz);
                }
            }
        }
        println!("{name}: highest mode {highest:.1} Hz of {} Hz", 0.5 * SAMPLE_RATE);
        assert!(
            highest < 0.5 * SAMPLE_RATE,
            "{name} put a mode at {highest} Hz"
        );
    }
}
