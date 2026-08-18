//! **Does the reference's own mono sum pay for its nodal band?** — the three
//! measurements the energy-conserving mechanism milestone is owed before any
//! engine change (`DECISIONS.md` 392-397).
//!
//! ```text
//! cargo run -p forensics --bin mono_mechanism -- <section>
//! sections: keys  mono  theta  lag  all
//! ```
//!
//! * `keys` — the four sections `DECISIONS.md` 407-408 are built on, which is
//!   the attribution `mono` deliberately pools away: **which** keys carry each
//!   band's mono difference and how much of the band they own; the same
//!   difference at **1/24 octave** with both sides printed separately and the
//!   recording's own pair-over-mono beside them; the same statistic through
//!   seven instruments that differ in **one stage each**, so the comb can be
//!   charged to a stage rather than guessed at; and the **headroom** an
//!   energy-conserving mechanism needs against what the fitted knobs reach.
//!
//! * `mono` — the recording's mono behaviour at sixth-octave resolution over
//!   100-800 Hz on the thirty recorded keys, against the engine's own mono in
//!   the same bands. The question is whether the reference's mono sum carries
//!   the nodal band's cancellation, how deep, and **how much of that our fitted
//!   mono already inherited** through the pan-pot path.
//! * `theta` — the Givens angle `theta(f)` that would reproduce the recording's
//!   `r0` from the engine's own geometric mid/side, and what it costs the mono
//!   sum per band. Compared against the headroom `mono` measured.
//! * `lag` — the spacing readback under the current lobe, under no lobe, under
//!   a Givens rotation, and under allpasses of known group delay; and the
//!   compensation that removes the bias.
//!
//! # Why every stereo stage can be applied offline
//!
//! `soundboard` builds both paths as `L = mid + side`, `R = mid - side`, and
//! the lobe is `side += lift * B(mid)` on each of them with the *same* `B`.
//! Linearity then gives, for the summed output, `side_total += lift*B(mid_total)`
//! — so the lobe, a Givens rotation, an allpass, or the exact inverse of any of
//! them can be applied to a *rendered* pair and is bit-equivalent to rendering
//! with it, as long as the safety limiter stays disengaged (it is transparent
//! below -1 dBFS and a single key at v90 is nowhere near). The DC blocker and
//! the master shelf are the same filter on both channels, so they commute with
//! the mid/side split. `lag`'s first table checks that claim against real
//! renders rather than assuming it.

use std::f64::consts::TAU;
use std::path::Path;

use piano_emulator::preset::{MicVoicing, Preset};

/// The band **this instrument** measures with: two edges and a `lift`.
///
/// It is deliberately not `preset::ModalBand` any more. Every stage in this
/// file is applied *offline* to rendered pairs — that is the finding the file
/// exists to record — so it never puts a band into a preset, and the engine's
/// own band now carries a rotation angle rather than a lift
/// (`soundboard::ModalRotation`). Keeping the three numbers here keeps the
/// tables this file printed re-derivable exactly as they were printed.
#[derive(Clone, Copy, Debug, PartialEq)]
struct ModalBand {
    lo_hz: f32,
    hi_hz: f32,
    lift: f32,
}

/// The shipped preset's band read as this file's own.
///
/// The engine's band carries a `lift` again (`DECISIONS.md` 406 reverted the
/// rotation), and this file's own `ModalBand` is a `lift` too, so the reading
/// is the field itself. It is kept as a separate type deliberately: every stage
/// in this file is applied *offline* to rendered pairs — that is the finding the
/// file exists to record — so it never puts a band into a preset, and it must go
/// on measuring what it measured whichever mechanism the engine carries.
fn shipped_band(preset: &Preset) -> Option<ModalBand> {
    preset.voicing.mics.and_then(|m| m.modal).map(|b| ModalBand {
        lo_hz: b.lo_hz,
        hi_hz: b.hi_hz,
        lift: b.lift,
    })
}
use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_emulator::types::Event;
use piano_tuner::audio::Audio;
use piano_tuner::estimate::mics::{interchannel_lag, LagConfig, ENGINE_LAG_PER_ITD, SPEED_OF_SOUND};
use piano_tuner::realism::{self, RecordedKeys};
use piano_tuner::sampler::{SamplerEvent, SAMPLER_VERSION};
use piano_tuner::{cache, SampleLibrary, Sampler, TimedEvent, SAMPLE_RATE};

use rayon::prelude::*;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

type C64 = Complex<f64>;

/// One key's rendered pair as spectra, with the transform length beside them:
/// `(left, right, n)`. Every stage of `lag` reads the same rendered set, so the
/// triple is carried rather than recomputed.
type KeySpectra = (Vec<C64>, Vec<C64>, usize);

const SFZ: &str = "data/salamander/SalamanderGrandPiano-V3+20200602.sfz";
const DATA: &str = "data/salamander";
const VELOCITY: u8 = 90;
const RENDER_S: f64 = 3.0;
const PREROLL: usize = realism::STEREO_PREROLL_SAMPLES;
const PREROLL_S: f64 = PREROLL as f64 / 48_000.0;
const SR: f64 = 48_000.0;

/// The span the mono question is asked over, and the span every share is
/// normalised inside — so a band's number is a *local* shape and not the
/// engine's standing broadband tilt (`DECISIONS.md` 343).
const SPAN_HZ: (f64, f64) = (100.0, 810.0);
/// The nodal band under investigation.
/// Sixth-octave centres land at 179.6, 201.6, 226.3, 254.0 and 285.1 Hz, so the
/// edges are set a hair outside the band `DECISIONS.md` 392-396 names in order
/// to include the 179.6 Hz point, which is the deepest one.
const NODAL_HZ: (f64, f64) = (175.0, 300.0);

/// The engine's own Butterworth section Qs, copied from `soundboard` (private
/// there). Eighth-order highpass at `lo`, fourth-order lowpass at `hi`.
const HIGH_Q: [f64; 4] = [0.509_796_2, 0.601_344_9, 0.899_976_2, 2.562_915_4];
const LOW_Q: [f64; 2] = [0.541_196_1, 1.306_562_9];

// ---------------------------------------------------------------------------
// Grid
// ---------------------------------------------------------------------------

/// Sixth-octave centres on `realism::stereo_profile`'s own grid (from 40 Hz),
/// restricted to [`SPAN_HZ`].
fn grid() -> Vec<f64> {
    let ratio = 2.0f64.powf(1.0 / realism::STEREO_PROFILE_PER_OCTAVE as f64);
    let mut hz = realism::STEREO_PROFILE_RANGE_HZ.0;
    let mut out = Vec::new();
    while hz <= SPAN_HZ.1 {
        if hz >= SPAN_HZ.0 {
            out.push(hz);
        }
        hz *= ratio;
    }
    out
}

fn band_edges(hz: f64) -> (f64, f64) {
    let half = 2.0f64.powf(0.5 / realism::STEREO_PROFILE_PER_OCTAVE as f64);
    (hz / half, hz * half)
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

fn shipped() -> Preset {
    Preset::load(Path::new("presets/salamander-c5.toml")).expect("the measured preset loads")
}

fn without_lobe(preset: &Preset) -> Preset {
    let mut p = preset.clone();
    if let Some(mics) = preset.voicing.mics {
        p.voicing.mics = Some(MicVoicing { modal: None, ..mics });
    }
    p
}

fn render_key(preset: &Preset, key: u8) -> (Vec<f32>, Vec<f32>) {
    let events = [RenderEvent::new(
        PREROLL_S as f32,
        Event::NoteOn {
            key,
            vel: u16::from(VELOCITY),
        },
    )];
    let (l, r) = render_to_buffer(preset, &events, (PREROLL_S + RENDER_S) as f32);
    (l[PREROLL..].to_vec(), r[PREROLL..].to_vec())
}

fn reference_key(key: u8, velocity: u8) -> Result<Audio, piano_tuner::Error> {
    let sfz = Path::new(SFZ);
    let mut print = cache::Fingerprint::new();
    print
        .str("tests/stereo/reference")
        .u64(u64::from(SAMPLER_VERSION))
        .file(sfz)?
        .u64(u64::from(SAMPLE_RATE))
        .u64(u64::from(key))
        .u64(u64::from(velocity))
        .f64(RENDER_S);
    let path = cache::reference_dir(Path::new(DATA)).join(format!(
        "stereo-key{key:03}-v{velocity:03}-{}.wav",
        print.hex()
    ));
    cache::audio(&path, || {
        let mut sampler = Sampler::new(sfz)?;
        let events = [TimedEvent::new(
            0.0,
            SamplerEvent::NoteOn { key, vel: velocity },
        )];
        let rendered = sampler.render(&events, RENDER_S + 0.2)?;
        let mono = rendered.mono();
        let onset = piano_tuner::detect_onset(&mono, f64::from(SAMPLE_RATE));
        let skip = (onset * f64::from(SAMPLE_RATE)).round() as usize;
        let frames = (RENDER_S * f64::from(SAMPLE_RATE)) as usize;
        let channels: Vec<Vec<f32>> = rendered
            .channels
            .iter()
            .map(|c| {
                (0..frames)
                    .map(|n| c.get(skip + n).copied().unwrap_or(0.0))
                    .collect()
            })
            .collect();
        Audio::new(SAMPLE_RATE, channels)
    })
}

// ---------------------------------------------------------------------------
// Spectra and band energies
// ---------------------------------------------------------------------------

fn forward(x: &[f32], n: usize, planner: &mut FftPlanner<f64>) -> Vec<C64> {
    let fft = planner.plan_fft_forward(n);
    let mut buf: Vec<C64> = (0..n)
        .map(|i| C64::new(f64::from(x.get(i).copied().unwrap_or(0.0)), 0.0))
        .collect();
    fft.process(&mut buf);
    buf
}

fn inverse(mut buf: Vec<C64>, planner: &mut FftPlanner<f64>) -> Vec<f32> {
    let n = buf.len();
    let fft = planner.plan_fft_inverse(n);
    fft.process(&mut buf);
    let s = 1.0 / n as f64;
    buf.iter().map(|c| (c.re * s) as f32).collect()
}

/// Energies of one band of one stereo pair: `[E_L, E_R, E_M, E_S, Re<L,R>]`.
fn band_energies(a: &[C64], b: &[C64], lo: f64, hi: f64) -> [f64; 5] {
    let n = a.len();
    let bin = |hz: f64| (hz * n as f64 / SR).round() as usize;
    let (blo, bhi) = (bin(lo).max(1), bin(hi).min(n / 2));
    let mut acc = [0.0f64; 5];
    if bhi < blo {
        return acc;
    }
    for j in blo..=bhi {
        for &(x, y) in &[(a[j], b[j]), (a[n - j], b[n - j])] {
            acc[0] += x.norm_sqr();
            acc[1] += y.norm_sqr();
            acc[2] += ((x + y) * 0.5).norm_sqr();
            acc[3] += ((x - y) * 0.5).norm_sqr();
            acc[4] += (x * y.conj()).re;
        }
    }
    let s = 1.0 / n as f64;
    for v in &mut acc {
        *v *= s;
    }
    acc
}

// ---------------------------------------------------------------------------
// The engine's lobe, analytically
// ---------------------------------------------------------------------------

fn biquad_response(hz: f64, q: f64, high: bool, f: f64) -> C64 {
    let w = (TAU * hz / SR).clamp(1.0e-6, 3.0);
    let (sin, cos) = w.sin_cos();
    let alpha = sin / (2.0 * q);
    let a0 = 1.0 + alpha;
    let (b0, b1, b2) = if high {
        let g = (1.0 + cos) / 2.0;
        (g, -2.0 * g, g)
    } else {
        let g = (1.0 - cos) / 2.0;
        (g, 2.0 * g, g)
    };
    let (b0, b1, b2, a1, a2) = (
        b0 / a0,
        b1 / a0,
        b2 / a0,
        -2.0 * cos / a0,
        (1.0 - alpha) / a0,
    );
    let z1 = C64::from_polar(1.0, -TAU * f / SR);
    let z2 = z1 * z1;
    (b0 + b1 * z1 + b2 * z2) / (1.0 + a1 * z1 + a2 * z2)
}

/// `lift * B(f)` — the mode-controlled band's transfer function exactly as the
/// engine's cascade realises it (the same sections, the same order).
fn lobe_response(band: &ModalBand, f: f64) -> C64 {
    let mut h = C64::new(f64::from(band.lift), 0.0);
    for q in HIGH_Q {
        h *= biquad_response(f64::from(band.lo_hz), q, true, f);
    }
    for q in LOW_Q {
        h *= biquad_response(f64::from(band.hi_hz), q, false, f);
    }
    h
}

/// Frequency of FFT bin `j` of a length-`n` transform, as a *signed* frequency
/// folded to `[0, SR/2]` — the response of a real filter at `n-j` is the
/// conjugate of its response at `j`, which is what keeps the output real.
fn bin_hz(j: usize, n: usize) -> (f64, bool) {
    if j <= n / 2 {
        (j as f64 * SR / n as f64, false)
    } else {
        ((n - j) as f64 * SR / n as f64, true)
    }
}

/// Applies `side += H(f) * mid` to a spectrum pair, in place.
fn apply_lobe(a: &mut [C64], b: &mut [C64], band: &ModalBand) {
    let n = a.len();
    for j in 0..n {
        let (f, conj) = bin_hz(j, n);
        let mut h = lobe_response(band, f);
        if conj {
            h = h.conj();
        }
        let (m, s) = ((a[j] + b[j]) * 0.5, (a[j] - b[j]) * 0.5);
        let s2 = s + h * m;
        a[j] = m + s2;
        b[j] = m - s2;
    }
}

/// Item 393's repair, exactly as it was built and reverted: the band-limited
/// mid copy is passed through **four first-order allpass sections** at 1.2x,
/// 1.99x, 3.31x and 5.5x the band's geometric centre before it is added to the
/// side. `MIC_MODAL_DIFFUSION` in the reverted `soundboard`.
///
/// This is the stage `DECISIONS.md` 395 was measuring when it recorded
/// "-21 to -24 %" spacing readbacks below `lo_hz = 225`. It is here so the
/// attribution can be settled rather than assumed.
fn apply_lobe_diffused(a: &mut [C64], b: &mut [C64], band: &ModalBand) {
    let centre = (f64::from(band.lo_hz) * f64::from(band.hi_hz)).sqrt();
    let corners = [1.2, 1.99, 3.31, 5.5].map(|k: f64| k * centre);
    let n = a.len();
    for j in 0..n {
        let (f, conj) = bin_hz(j, n);
        let mut h = lobe_response(band, f);
        let mut d = C64::new(1.0, 0.0);
        for c in corners {
            d *= allpass_response(c, f);
        }
        h *= d;
        if conj {
            h = h.conj();
        }
        let (m, s) = ((a[j] + b[j]) * 0.5, (a[j] - b[j]) * 0.5);
        let s2 = s + h * m;
        a[j] = m + s2;
        b[j] = m - s2;
    }
}

/// The exact inverse of [`apply_lobe`]: recovers `(mid, side_geo)` and rebuilds
/// the pair the geometry alone would have produced. This is the compensation
/// `lag` proposes for the estimator.
fn undo_lobe(a: &mut [C64], b: &mut [C64], band: &ModalBand) {
    let n = a.len();
    for j in 0..n {
        let (f, conj) = bin_hz(j, n);
        let mut h = lobe_response(band, f);
        if conj {
            h = h.conj();
        }
        let (m, s) = ((a[j] + b[j]) * 0.5, (a[j] - b[j]) * 0.5);
        let s2 = s - h * m;
        a[j] = m + s2;
        b[j] = m - s2;
    }
}

/// The energy-conserving form: `mid *= cos(theta)`, `side += mid*sin(theta)`,
/// with `tan(theta(f)) = |lift*B(f)|` — the same band shape as the lobe, turned
/// into a rotation. Zero-phase, so this isolates the mechanism from any
/// realisation's own group delay.
fn apply_givens(a: &mut [C64], b: &mut [C64], band: &ModalBand) {
    let n = a.len();
    for j in 0..n {
        let (f, _) = bin_hz(j, n);
        let g = lobe_response(band, f).norm();
        let (c, s_) = (1.0 / (1.0 + g * g).sqrt(), g / (1.0 + g * g).sqrt());
        let (m, s) = ((a[j] + b[j]) * 0.5, (a[j] - b[j]) * 0.5);
        let m2 = m * c;
        let s2 = s + m * s_;
        a[j] = m2 + s2;
        b[j] = m2 - s2;
    }
}

/// Minimum-phase filter with the given magnitude, by the cepstral fold.
///
/// The zero-phase Givens above is a *measurement* of the mechanism and not a
/// shippable filter: a real one is causal and therefore carries group delay.
/// This builds the causal filter with exactly that magnitude, so the readback
/// can be measured on something the engine could actually run.
fn min_phase(mag: &[f64], planner: &mut FftPlanner<f64>) -> Vec<C64> {
    let n = mag.len();
    let mut buf: Vec<C64> = mag
        .iter()
        .map(|&m| C64::new(m.max(1e-9).ln(), 0.0))
        .collect();
    planner.plan_fft_forward(n).process(&mut buf);
    let s = 1.0 / n as f64;
    for (j, v) in buf.iter_mut().enumerate() {
        *v *= s;
        if j == 0 || (n % 2 == 0 && j == n / 2) {
            // keep
        } else if j < n / 2 {
            *v *= 2.0;
        } else {
            *v = C64::new(0.0, 0.0);
        }
    }
    planner.plan_fft_inverse(n).process(&mut buf);
    buf.iter().map(|c| c.exp()).collect()
}

/// The Givens rotation realised **causally**: both `cos(theta)` on the mid and
/// `sin(theta)` on the injection as minimum-phase filters of the same
/// magnitudes.
fn apply_givens_min_phase(a: &mut [C64], b: &mut [C64], band: &ModalBand) {
    let n = a.len();
    let mut planner = FftPlanner::<f64>::new();
    let (mut cm, mut sm) = (vec![0.0f64; n], vec![0.0f64; n]);
    for j in 0..n {
        let (f, _) = bin_hz(j, n);
        let g = lobe_response(band, f).norm();
        cm[j] = 1.0 / (1.0 + g * g).sqrt();
        sm[j] = g / (1.0 + g * g).sqrt();
    }
    let c = min_phase(&cm, &mut planner);
    let s_ = min_phase(&sm, &mut planner);
    for j in 0..n {
        let (m, s) = ((a[j] + b[j]) * 0.5, (a[j] - b[j]) * 0.5);
        let m2 = m * c[j];
        let s2 = s + m * s_[j];
        a[j] = m2 + s2;
        b[j] = m2 - s2;
    }
}

fn undo_givens(a: &mut [C64], b: &mut [C64], band: &ModalBand) {
    let n = a.len();
    for j in 0..n {
        let (f, _) = bin_hz(j, n);
        let g = lobe_response(band, f).norm();
        let (c, s_) = (1.0 / (1.0 + g * g).sqrt(), g / (1.0 + g * g).sqrt());
        let (m2, s2) = ((a[j] + b[j]) * 0.5, (a[j] - b[j]) * 0.5);
        let m = m2 / c;
        let s = s2 - m * s_;
        a[j] = m + s;
        b[j] = m - s;
    }
}

/// A first-order allpass `(-c + z^-1)/(1 - c z^-1)` with corner `hz`, applied to
/// the **side** path. Its group delay at DC is `2/(w0)`-ish; the exact value at
/// the frequency of interest is reported rather than assumed.
fn allpass_response(hz: f64, f: f64) -> C64 {
    let w = TAU * hz / SR;
    let c = (1.0 - w.sin()) / w.cos();
    let z1 = C64::from_polar(1.0, -TAU * f / SR);
    (C64::new(-c, 0.0) + z1) / (C64::new(1.0, 0.0) - z1 * c)
}

/// Group delay of that allpass at `f`, in seconds, by finite difference of its
/// phase.
fn allpass_group_delay_s(hz: f64, f: f64) -> f64 {
    let df = 0.5;
    let p1 = allpass_response(hz, f - df).arg();
    let p2 = allpass_response(hz, f + df).arg();
    let mut d = p2 - p1;
    while d > std::f64::consts::PI {
        d -= TAU;
    }
    while d < -std::f64::consts::PI {
        d += TAU;
    }
    -d / (TAU * 2.0 * df)
}

fn apply_side_allpass(a: &mut [C64], b: &mut [C64], hz: f64) {
    let n = a.len();
    for j in 0..n {
        let (f, conj) = bin_hz(j, n);
        let mut h = allpass_response(hz, f);
        if conj {
            h = h.conj();
        }
        let (m, s) = ((a[j] + b[j]) * 0.5, (a[j] - b[j]) * 0.5);
        let s2 = s * h;
        a[j] = m + s2;
        b[j] = m - s2;
    }
}

/// The same allpass on **both** channels — a common path, which no interchannel
/// estimator may see at all. The control for the one above.
fn apply_common_allpass(a: &mut [C64], b: &mut [C64], hz: f64) {
    let n = a.len();
    for j in 0..n {
        let (f, conj) = bin_hz(j, n);
        let mut h = allpass_response(hz, f);
        if conj {
            h = h.conj();
        }
        a[j] *= h;
        b[j] *= h;
    }
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

fn median(v: &mut Vec<f64>) -> f64 {
    v.retain(|x| x.is_finite());
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

#[allow(dead_code)]
fn mad(v: &[f64]) -> f64 {
    let c: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if c.is_empty() {
        return f64::NAN;
    }
    let m = median(&mut c.clone());
    let mut d: Vec<f64> = c.iter().map(|x| (x - m).abs()).collect();
    1.4826 * median(&mut d)
}

fn db(x: f64) -> f64 {
    10.0 * x.max(1e-300).log10()
}

// ---------------------------------------------------------------------------
// Section 1: the mono question
// ---------------------------------------------------------------------------

/// One key's band table, for one take.
struct Take {
    /// `[E_L, E_R, E_M, E_S, Re<L,R>]` per band of [`grid`].
    bands: Vec<[f64; 5]>,
}

#[allow(dead_code)]
impl Take {
    fn of(left: &[f32], right: &[f32], grid: &[f64]) -> Take {
        let n = left.len().max(right.len()).next_power_of_two();
        let mut planner = FftPlanner::<f64>::new();
        let a = forward(left, n, &mut planner);
        let b = forward(right, n, &mut planner);
        Take::from_spectra(&a, &b, grid)
    }

    fn from_spectra(a: &[C64], b: &[C64], grid: &[f64]) -> Take {
        Take {
            bands: grid
                .iter()
                .map(|&hz| {
                    let (lo, hi) = band_edges(hz);
                    band_energies(a, b, lo, hi)
                })
                .collect(),
        }
    }

    fn mono_total(&self) -> f64 {
        self.bands.iter().map(|e| e[2]).sum()
    }
    fn pair_total(&self) -> f64 {
        self.bands.iter().map(|e| e[0] + e[1]).sum()
    }
    /// This band's share of the take's own 100-800 Hz mono energy, dB.
    fn mono_share_db(&self, i: usize) -> f64 {
        db(self.bands[i][2] / self.mono_total())
    }
    /// The same for the pair average `(E_L + E_R)/2`.
    fn pair_share_db(&self, i: usize) -> f64 {
        db(0.5 * (self.bands[i][0] + self.bands[i][1]) / (0.5 * self.pair_total()))
    }
    /// `10 log10((E_L + E_R) / 2 E_M)` — what the mono fold-down does not carry.
    fn pair_db(&self, i: usize) -> f64 {
        db((self.bands[i][0] + self.bands[i][1]) / (2.0 * self.bands[i][2]))
    }
    fn r0(&self, i: usize) -> f64 {
        let e = self.bands[i];
        e[4] / (e[0] * e[1]).sqrt()
    }
    /// Is this band worth reading on this key? Within 40 dB of the key's own
    /// loudest band inside the span — below that a band is the board's floor.
    fn readable(&self, i: usize) -> bool {
        let peak = self
            .bands
            .iter()
            .map(|e| e[2])
            .fold(0.0f64, f64::max);
        self.bands[i][2] > peak * 1e-4
    }
}

#[allow(dead_code)]
struct KeyRow {
    key: u8,
    label: String,
    reference: Take,
    engine_bare: Take,
    engine_shipped: Take,
    engine_synth: Take,
}

fn mono_section(grid: &[f64]) -> Result<Vec<KeyRow>, Box<dyn std::error::Error>> {
    let preset = shipped();
    let bare = without_lobe(&preset);
    let band = shipped_band(&preset).expect("the shipped preset declares a mode-controlled band");
    let library = SampleLibrary::from_sfz(Path::new(SFZ))?;
    let recorded = RecordedKeys::from_library(&library)?;

    let rows: Vec<KeyRow> = recorded
        .keys()
        .par_iter()
        .map(|&key| -> Result<KeyRow, piano_tuner::Error> {
            let r = reference_key(key, VELOCITY)?;
            let reference = Take::of(&r.channels[0], &r.channels[1], grid);
            let (bl, br) = render_key(&bare, key);
            let (sl, sr) = render_key(&preset, key);
            let n = bl.len().next_power_of_two();
            let mut planner = FftPlanner::<f64>::new();
            let mut a = forward(&bl, n, &mut planner);
            let mut b = forward(&br, n, &mut planner);
            let engine_bare = Take::from_spectra(&a, &b, grid);
            apply_lobe(&mut a, &mut b, &band);
            let engine_synth = Take::from_spectra(&a, &b, grid);
            Ok(KeyRow {
                key,
                label: realism::note_name(key),
                reference,
                engine_bare,
                engine_shipped: Take::of(&sl, &sr, grid),
                engine_synth,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

/// Pooled, level-matched band ratio: every key's band energy divided by that
/// key's own 100-810 Hz total, summed over the keys, and the two sums divided.
///
/// A sixth-octave band of *one* key is dominated by whether a partial happens to
/// land in it, which is why the per-key median of a share carries a 3-7 dB MAD
/// and says nothing. Pooling the level-matched energies weights each key by how
/// much it actually has in the band, which is the estimator the question wants:
/// *of the energy the recording puts in this band, how much does the engine
/// put there?*
fn pooled_db(rows: &[&KeyRow], num: &dyn Fn(&KeyRow) -> (f64, f64), den: &dyn Fn(&KeyRow) -> (f64, f64)) -> f64 {
    let (mut a, mut b) = (0.0f64, 0.0f64);
    for r in rows {
        let (na, ta) = num(r);
        let (nb, tb) = den(r);
        a += na / ta;
        b += nb / tb;
    }
    db(a / b)
}

/// Jackknife standard error of a pooled ratio over the keys, dB.
fn jackknife_db(
    rows: &[&KeyRow],
    num: &dyn Fn(&KeyRow) -> (f64, f64),
    den: &dyn Fn(&KeyRow) -> (f64, f64),
) -> f64 {
    let n = rows.len();
    if n < 3 {
        return f64::NAN;
    }
    let full: Vec<f64> = (0..n)
        .map(|drop| {
            let (mut a, mut b) = (0.0f64, 0.0f64);
            for (j, r) in rows.iter().enumerate() {
                if j == drop {
                    continue;
                }
                let (na, ta) = num(r);
                let (nb, tb) = den(r);
                a += na / ta;
                b += nb / tb;
            }
            db(a / b)
        })
        .collect();
    let mean = full.iter().sum::<f64>() / n as f64;
    let var = full.iter().map(|x| (x - mean).powi(2)).sum::<f64>() * (n - 1) as f64 / n as f64;
    var.sqrt()
}

fn report_mono(grid: &[f64], rows: &[KeyRow]) {
    println!(
        "\n## 1. The recording's mono sum through the nodal band — {} recorded keys at v{VELOCITY}\n",
        rows.len()
    );
    println!(
        "Every take is level-matched on its own {:.0}-{:.0} Hz total and the keys are then \
**pooled**, so a column is a local shape and the standing broadband tilt is out of it. \
`+-` is a jackknife standard error over the keys.\n",
        SPAN_HZ.0, SPAN_HZ.1
    );
    println!(
        "| Hz | n | REF pair-over-mono dB | REF r0 | REF mid/side dB | ENG mono - REF mono dB | \
+- | ENG mono - REF pair-avg dB | ENG(bare) pair dB | ENG(shipped) pair dB | offline-vs-rendered dB |"
    );
    println!("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");

    let mono_of = |t: &Take, i: usize| (t.bands[i][2], t.mono_total());
    let pairavg_of = |t: &Take, i: usize| (0.5 * (t.bands[i][0] + t.bands[i][1]), 0.5 * t.pair_total());

    let mut nodal_pair = Vec::new();
    let mut base_pair = Vec::new();
    let mut nodal_dmono = Vec::new();
    let mut base_dmono = Vec::new();
    let mut curve = Vec::new();

    for (i, &hz) in grid.iter().enumerate() {
        let live: Vec<&KeyRow> = rows
            .iter()
            .filter(|r| r.reference.readable(i) && r.engine_bare.readable(i))
            .collect();
        if live.len() < 5 {
            continue;
        }
        // `pair_db` is a within-take ratio, so pooling it needs no level match
        // at all: sum the two energies over the keys and divide.
        let ref_pair = {
            let (mut p, mut m) = (0.0, 0.0);
            for r in &live {
                let t = r.reference.mono_total();
                p += (r.reference.bands[i][0] + r.reference.bands[i][1]) / t;
                m += 2.0 * r.reference.bands[i][2] / t;
            }
            db(p / m)
        };
        let mut r0v: Vec<f64> = live.iter().map(|r| r.reference.r0(i)).collect();
        let ref_r0 = median(&mut r0v);
        let ref_ms = {
            let (mut m, mut s) = (0.0, 0.0);
            for r in &live {
                let t = r.reference.mono_total();
                m += r.reference.bands[i][2] / t;
                s += r.reference.bands[i][3] / t;
            }
            db(m / s)
        };
        let dmono = pooled_db(
            &live,
            &|r| mono_of(&r.engine_shipped, i),
            &|r| mono_of(&r.reference, i),
        );
        let dse = jackknife_db(
            &live,
            &|r| mono_of(&r.engine_shipped, i),
            &|r| mono_of(&r.reference, i),
        );
        let dpair = pooled_db(
            &live,
            &|r| mono_of(&r.engine_shipped, i),
            &|r| pairavg_of(&r.reference, i),
        );
        let bare_pair = {
            let (mut p, mut m) = (0.0, 0.0);
            for r in &live {
                let t = r.engine_bare.mono_total();
                p += (r.engine_bare.bands[i][0] + r.engine_bare.bands[i][1]) / t;
                m += 2.0 * r.engine_bare.bands[i][2] / t;
            }
            db(p / m)
        };
        let ship_pair = {
            let (mut p, mut m) = (0.0, 0.0);
            for r in &live {
                let t = r.engine_shipped.mono_total();
                p += (r.engine_shipped.bands[i][0] + r.engine_shipped.bands[i][1]) / t;
                m += 2.0 * r.engine_shipped.bands[i][2] / t;
            }
            db(p / m)
        };
        let mut synth: Vec<f64> = live
            .iter()
            .map(|r| r.engine_synth.pair_db(i) - r.engine_shipped.pair_db(i))
            .collect();
        let synth_err = median(&mut synth);
        println!(
            "| {hz:.0} | {} | {ref_pair:+.2} | {ref_r0:+.3} | {ref_ms:+.2} | **{dmono:+.2}** | \
{dse:.2} | {dpair:+.2} | {bare_pair:+.2} | {ship_pair:+.2} | {synth_err:+.3} |",
            live.len()
        );
        curve.push((hz, ref_pair, dmono, dse));
        if (NODAL_HZ.0..=NODAL_HZ.1).contains(&hz) {
            nodal_pair.push(ref_pair);
            nodal_dmono.push(dmono);
        } else {
            base_pair.push(ref_pair);
            base_dmono.push(dmono);
        }
    }

    let mi = median(&mut nodal_pair.clone());
    let mo = median(&mut base_pair.clone());
    println!(
        "\n**(a) The reference's own mono sum does carry the nodal band's cost.** Pooled \
pair-over-mono is **{mi:+.2} dB** inside {:.0}-{:.0} Hz against **{mo:+.2} dB** in the rest of \
{:.0}-{:.0} Hz — a **{:+.2} dB** localised mono cancellation, peaking at **{:+.2} dB**. Its `r0` \
goes negative over exactly those bands, which is a real pair straddling a nodal line and not a \
level difference (`DECISIONS.md` 393a).",
        NODAL_HZ.0,
        NODAL_HZ.1,
        SPAN_HZ.0,
        SPAN_HZ.1,
        mi - mo,
        nodal_pair.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );
    let di = median(&mut nodal_dmono.clone());
    let do_ = median(&mut base_dmono.clone());
    println!(
        "\n**(b) How much of it our mono already inherited.** Engine mono less reference mono, \
pooled and level-matched, is **{di:+.2} dB** inside the nodal band and **{do_:+.2} dB** outside \
it. **Excess headroom inside the band: {:+.2} dB** — that is what a mono-costing mechanism can \
spend before the engine's mono is *darker* there than the recording's.",
        di - do_
    );
    println!("\nThe curve, for the record (Hz, ref pair-over-mono dB, eng-ref mono dB +- se):\n");
    for (hz, p, d, se) in &curve {
        println!("  {hz:6.0}  {p:+6.2}   {d:+6.2} +- {se:.2}");
    }
}


// ---------------------------------------------------------------------------
// Section 2: theta(f)
// ---------------------------------------------------------------------------

/// **How far the existing knobs reach against the headroom a nodal mechanism
/// needs** (`DECISIONS.md` 407).
///
/// The mechanism's mono cost at a band is exactly the pair energy it adds
/// there, so the source must stand **`REF pair - ENG(bare) pair`** decibels
/// *above* the recording's own mono before the mechanism is applied, or the
/// fold-down lands under the recording once it is. That is the "required"
/// column. Against it: what the engine actually stands at now, and what it
/// stands at when the only frequency-shaping knob the direct-plus-board path
/// has — `soundboard.body_modes` under 400 Hz — is pushed to two, four and
/// eight times its fitted gains, with `board_mix` opened to 0.6 as well.
fn report_headroom(grid: &[f64], rows: &[KeyRow]) -> Result<(), Box<dyn std::error::Error>> {
    let base = without_lobe(&shipped());
    let scaled = |k: f32, mix: Option<f32>| {
        let mut p = base.clone();
        for m in &mut p.soundboard.body_modes {
            if m.hz >= 140.0 && m.hz <= 400.0 {
                m.gain *= k;
            }
        }
        if let Some(mix) = mix {
            p.soundboard.board_mix = mix;
        }
        p
    };
    let v2 = scaled(2.0, None);
    let v4 = scaled(4.0, None);
    let v8 = scaled(8.0, None);
    let v8m = scaled(8.0, Some(0.6));
    // The one fitted parameter that *is* the direct path's radiated spectrum:
    // `notes.partial_gains`, per key and per partial. Every partial whose own
    // frequency lands between 160 and 300 Hz is lifted by 9 dB — the deficit
    // the table below asks for at 180 Hz — so that what the per-partial fit
    // could supply if it were re-fitted against an uncancelled target can be
    // told apart from what it cannot reach at all.
    let mut lifted = base.clone();
    {
        let f0 = lifted.notes.f0_hz.clone();
        let bb = lifted.notes.inharmonicity_b.clone();
        for (k, row) in lifted.notes.partial_gains.iter_mut().enumerate() {
            let (f, b) = (f0[k], bb[k]);
            for (i, g) in row.iter_mut().enumerate() {
                let n = (i + 1) as f32;
                let hz = f * n * (1.0 + b * n * n).sqrt();
                if (160.0..=300.0).contains(&hz) {
                    *g *= 2.818_383; // +9 dB
                }
            }
        }
    }
    let variants: [(&str, &Preset); 6] = [
        ("x2", &v2),
        ("x4", &v4),
        ("x8", &v8),
        ("x8, board_mix 0.6", &v8m),
        ("partials +9 dB in 160-300", &lifted),
        ("shipped bare", &base),
    ];
    let mut cols: Vec<Vec<f64>> = Vec::new();
    let mono_of = |t: &Take, i: usize| (t.bands[i][2], t.mono_total());
    for (_, preset) in variants {
        let takes: Vec<Take> = rows
            .par_iter()
            .map(|r| {
                let (l, rr) = render_key(preset, r.key);
                Take::of(&l, &rr, grid)
            })
            .collect();
        cols.push(
            (0..grid.len())
                .map(|i| {
                    let (mut e, mut rf) = (0.0, 0.0);
                    for (t, r) in takes.iter().zip(rows) {
                        let (a, b) = mono_of(t, i);
                        let (c, d) = mono_of(&r.reference, i);
                        e += a / b;
                        rf += c / d;
                    }
                    db(e / rf)
                })
                .collect(),
        );
    }
    println!("\n## 0d. The headroom a nodal mechanism needs, against what the knobs reach\n");
    print!("| Hz | REF pair dB | ENG bare pair dB | **required** | now |");
    for (name, _) in &variants[..5] {
        print!(" {name} |");
    }
    println!();
    print!("|---:|---:|---:|---:|---:|");
    for _ in &variants[..5] {
        print!("---:|");
    }
    println!();
    for (i, &hz) in grid.iter().enumerate() {
        let pooled_pair = |f: &dyn Fn(&KeyRow) -> &Take| {
            let (mut p, mut m) = (0.0, 0.0);
            for r in rows {
                let t = f(r);
                let tot = t.mono_total();
                p += (t.bands[i][0] + t.bands[i][1]) / tot;
                m += 2.0 * t.bands[i][2] / tot;
            }
            db(p / m)
        };
        let refp = pooled_pair(&|r| &r.reference);
        let engp = pooled_pair(&|r| &r.engine_bare);
        print!(
            "| {hz:.0} | {refp:+.2} | {engp:+.2} | **{:+.2}** | {:+.2} |",
            refp - engp,
            cols[5][i]
        );
        for c in &cols[..5] {
            print!(" {:+.2} |", c[i]);
        }
        println!();
    }
    Ok(())
}

/// **Which stage of the engine carries the comb** (`DECISIONS.md` 407).
///
/// The same pooled 1/24-octave mono shape as [`report_halo`], on the same keys,
/// through four instruments that differ in one stage each: the shipped preset
/// with no lobe; the same with `soundboard.body_modes` flattened (every gain
/// zero); the same with `voicing.bridge.peaks` flattened (every gain 0 dB); and
/// the same with `board_mix` at zero, which removes the whole board path. A
/// bump that is the body modes' disappears in the second, a bump that is the
/// bridge's disappears in the third, and one that survives all three is the
/// backbone's or the strings' own.
fn report_parts(rows: &[KeyRow]) -> Result<(), Box<dyn std::error::Error>> {
    let ratio = 2.0f64.powf(1.0 / 24.0);
    let mut fine = Vec::new();
    let mut hz = SPAN_HZ.0;
    while hz <= SPAN_HZ.1 {
        fine.push(hz);
        hz *= ratio;
    }
    let base = without_lobe(&shipped());
    let mut flat_body = base.clone();
    for m in &mut flat_body.soundboard.body_modes {
        m.gain = 0.0;
    }
    let mut flat_peaks = base.clone();
    if let Some(bridge) = flat_peaks.voicing.bridge.as_mut() {
        for p in &mut bridge.peaks {
            p.gain_db = 0.0;
        }
    }
    let mut no_board = base.clone();
    no_board.soundboard.board_mix = 0.0;
    let mut no_symp = base.clone();
    no_symp.voicing.resonance_coupling = 0.0;
    let mut no_strike = base.clone();
    for a in &mut no_strike.noise.strike.level_db {
        a.db = -200.0;
    }
    let mut only_direct = no_symp.clone();
    only_direct.soundboard.board_mix = 0.0;
    for a in &mut only_direct.noise.strike.level_db {
        a.db = -200.0;
    }
    let variants: [(&str, &Preset); 7] = [
        ("bare", &base),
        ("no body modes", &flat_body),
        ("no bridge peaks", &flat_peaks),
        ("no board path", &no_board),
        ("no sympathetic", &no_symp),
        ("no strike noise", &no_strike),
        ("strings alone", &only_direct),
    ];
    let half = 2.0f64.powf(0.5 / 24.0);
    let take_of = |l: &[f32], r: &[f32]| {
        let n = l.len().max(r.len()).next_power_of_two();
        let mut planner = FftPlanner::<f64>::new();
        let a = forward(l, n, &mut planner);
        let b = forward(r, n, &mut planner);
        Take {
            bands: fine
                .iter()
                .map(|&hz| band_energies(&a, &b, hz / half, hz * half))
                .collect(),
        }
    };
    println!("\n## 0c. The comb, stage by stage — pooled engine less recording, 1/24 octave, dB\n");
    let refs: Vec<Take> = rows
        .par_iter()
        .map(|r| -> Result<Take, piano_tuner::Error> {
            let a = reference_key(r.key, VELOCITY)?;
            Ok(take_of(&a.channels[0], &a.channels[1]))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut cols: Vec<Vec<f64>> = Vec::new();
    for (_, preset) in variants {
        let takes: Vec<Take> = rows
            .par_iter()
            .map(|r| {
                let (l, rr) = render_key(preset, r.key);
                take_of(&l, &rr)
            })
            .collect();
        cols.push(
            (0..fine.len())
                .map(|i| {
                    let (mut e, mut rf) = (0.0, 0.0);
                    for (t, reference) in takes.iter().zip(&refs) {
                        e += t.bands[i][2] / t.mono_total();
                        rf += reference.bands[i][2] / reference.mono_total();
                    }
                    db(e / rf)
                })
                .collect(),
        );
    }
    print!("| Hz |");
    for (name, _) in variants {
        print!(" {name} |");
    }
    println!();
    print!("|---:|");
    for _ in variants {
        print!("---:|");
    }
    println!();
    for (i, &hz) in fine.iter().enumerate() {
        print!("| {hz:.0} |");
        for c in &cols {
            print!(" {:+.2} |", c[i]);
        }
        println!();
    }
    Ok(())
}

/// **What the mono excess *is*, at 1/24 octave and in the time domain**
/// (`DECISIONS.md` 407).
///
/// `keys` says *which* keys carry it; this says what it looks like. Two
/// readings on the same renders:
///
/// * the pooled engine-less-recording mono share at **1/24 octave** over
///   100-810 Hz, which separates "a resonance" (a few tenths of an octave
///   wide) from "a tilt" (the whole span);
/// * for the treble keys, whose 100-810 Hz content is *all* halo and no
///   partial, the **decay** of a 254 Hz bandpass of the engine's own render
///   against the recording's — a body mode rings, a noise burst does not.
fn report_halo(rows: &[KeyRow]) -> Result<(), Box<dyn std::error::Error>> {
    let ratio = 2.0f64.powf(1.0 / 24.0);
    let mut fine = Vec::new();
    let mut hz = SPAN_HZ.0;
    while hz <= SPAN_HZ.1 {
        fine.push(hz);
        hz *= ratio;
    }
    println!("\n## 0b. The excess at 1/24 octave, pooled over the treble keys and over all keys\n");
    println!(
        "| Hz | ENG share dB | REF share dB | ENG-REF dB | REF pair-over-mono dB | treble keys dB |"
    );
    println!("|---:|---:|---:|---:|---:|---:|");
    let preset = shipped();
    let bare = without_lobe(&preset);
    // Re-render at the finer grid: the `Take`s carried by `rows` are on the
    // sixth-octave grid and cannot be re-banded.
    let takes: Vec<(u8, Take, Take)> = rows
        .par_iter()
        .map(|r| -> Result<(u8, Take, Take), piano_tuner::Error> {
            let refa = reference_key(r.key, VELOCITY)?;
            let half = 2.0f64.powf(0.5 / 24.0);
            let grid: Vec<f64> = fine.clone();
            let take = |l: &[f32], rr: &[f32]| {
                let n = l.len().max(rr.len()).next_power_of_two();
                let mut planner = FftPlanner::<f64>::new();
                let a = forward(l, n, &mut planner);
                let b = forward(rr, n, &mut planner);
                Take {
                    bands: grid
                        .iter()
                        .map(|&hz| band_energies(&a, &b, hz / half, hz * half))
                        .collect(),
                }
            };
            let (bl, br) = render_key(&bare, r.key);
            Ok((
                r.key,
                take(&refa.channels[0], &refa.channels[1]),
                take(&bl, &br),
            ))
        })
        .collect::<Result<Vec<_>, _>>()?;
    for (i, &hz) in fine.iter().enumerate() {
        let pooled = |sel: &dyn Fn(u8) -> bool| {
            let (mut e, mut rf) = (0.0, 0.0);
            for (key, reference, engine) in &takes {
                if !sel(*key) {
                    continue;
                }
                e += engine.bands[i][2] / engine.mono_total();
                rf += reference.bands[i][2] / reference.mono_total();
            }
            db(e / rf)
        };
        let side = |eng: bool| {
            let mut acc = 0.0;
            for (_, reference, engine) in &takes {
                let t = if eng { engine } else { reference };
                acc += t.bands[i][2] / t.mono_total();
            }
            db(acc / takes.len() as f64)
        };
        let ref_pair = {
            let (mut p, mut m) = (0.0, 0.0);
            for (_, reference, _) in &takes {
                let t = reference.mono_total();
                p += (reference.bands[i][0] + reference.bands[i][1]) / t;
                m += 2.0 * reference.bands[i][2] / t;
            }
            db(p / m)
        };
        println!(
            "| {hz:.0} | {:+.2} | {:+.2} | {:+.2} | {:+.2} | {:+.2} |",
            side(true),
            side(false),
            pooled(&|_| true),
            ref_pair,
            pooled(&|k| k >= 81)
        );
    }
    Ok(())
}

/// **Which keys carry the mono excess, band by band** (`DECISIONS.md` 407).
///
/// `mono` pools the keys, which is the right estimator for "of the energy the
/// recording puts here, how much does the engine put here" and is deliberately
/// blind to *which* key put it there. The repair has to be written into a fit
/// that acts per key and per partial, so the attribution is owed: this prints,
/// per key, that key's own 100-810 Hz-normalised mono share in each band,
/// engine (bare) less recording, and beside it the key's share of the pooled
/// band energy so a large per-key number on a key that has nothing in the band
/// can be told from one that carries it.
fn report_keys(grid: &[f64], rows: &[KeyRow]) {
    println!("\n## 0. Which keys carry each band's mono excess (engine bare - recording, dB)\n");
    let want: Vec<usize> = (0..grid.len()).collect();
    print!("| key |");
    for &i in &want {
        print!(" {:.0} |", grid[i]);
    }
    println!();
    print!("|---|");
    for _ in &want {
        print!("---:|");
    }
    println!();
    // Each band's pooled reference energy, for the weight column.
    let ref_tot: Vec<f64> = want
        .iter()
        .map(|&i| {
            rows.iter()
                .map(|r| r.reference.bands[i][2] / r.reference.mono_total())
                .sum::<f64>()
        })
        .collect();
    for r in rows {
        print!("| {} |", r.label);
        for (c, &i) in want.iter().enumerate() {
            if !r.reference.readable(i) || !r.engine_bare.readable(i) {
                print!("  . |");
                continue;
            }
            let d = r.engine_bare.mono_share_db(i) - r.reference.mono_share_db(i);
            let w = (r.reference.bands[i][2] / r.reference.mono_total()) / ref_tot[c].max(1e-300);
            print!(" {d:+.1}<{:.0}%> |", 100.0 * w);
        }
        println!();
    }
    println!(
        "\n`<n%>` is that key's share of the pooled reference energy in the band — the weight the \
pooled row of section 1 gives it."
    );
}

fn report_theta(grid: &[f64], rows: &[KeyRow]) {
    println!("\n## 2. The Givens angle the recording asks for, and what it costs the mono sum\n");
    println!(
        "`rho = E_M/E_S`. The recording's `r0` fixes the target `rho* = (1+r0)/(1-r0)` (exact \
when the two channels are level-matched, which `DECISIONS.md` 393 measured them to be to \
2-4 dB). A rotation by `theta` on the engine's own geometric pair gives \
`rho' = rho0 cos^2 / (1 + rho0 sin^2)`, so `sin^2 theta = (1 - rho*/rho0)/(rho* + 1)` and the \
mono sum pays `-10 log10(cos^2 theta)`.\n"
    );
    println!(
        "| Hz | REF rho* dB | REF r0 | ENG rho0 dB (bare) | sin^2 th | **theta deg** | \
**mono cost dB** | headroom dB | +- | **cost - headroom** | resulting r0 | current lift g | \
lobe's manufactured pair dB |"
    );
    println!("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
    let band = shipped_band(&shipped()).expect("a band");
    let mono_of = |t: &Take, i: usize| (t.bands[i][2], t.mono_total());
    let mut inside: Vec<(f64, f64, f64, f64)> = Vec::new();
    let mut infeasible = Vec::new();
    let mut theta_curve: Vec<(f64, f64)> = Vec::new();
    for (i, &hz) in grid.iter().enumerate() {
        let live: Vec<&KeyRow> = rows
            .iter()
            .filter(|r| r.reference.readable(i) && r.engine_bare.readable(i))
            .collect();
        if live.len() < 5 {
            continue;
        }
        // Pooled, level-matched mid/side ratios: the quantity a rotation
        // actually controls, on both sides.
        let pooled_ratio = |f: &dyn Fn(&KeyRow) -> &Take| -> f64 {
            let (mut m, mut s) = (0.0, 0.0);
            for r in &live {
                let t = f(r);
                let tot = t.mono_total();
                m += t.bands[i][2] / tot;
                s += t.bands[i][3] / tot;
            }
            m / s
        };
        let rho_star = pooled_ratio(&|r| &r.reference);
        let rho0 = pooled_ratio(&|r| &r.engine_bare);
        let mut r0v: Vec<f64> = live.iter().map(|r| r.reference.r0(i)).collect();
        let r0 = median(&mut r0v);
        let t = (1.0 - rho_star / rho0) / (rho_star + 1.0);
        let feasible = t > 0.0 && t < 1.0;
        let theta = if feasible { t.sqrt().asin().to_degrees() } else { f64::NAN };
        let cost = if feasible { -10.0 * (1.0 - t).log10() } else { f64::NAN };
        let headroom = pooled_db(
            &live,
            &|r| mono_of(&r.engine_shipped, i),
            &|r| mono_of(&r.reference, i),
        );
        let se = jackknife_db(
            &live,
            &|r| mono_of(&r.engine_shipped, i),
            &|r| mono_of(&r.reference, i),
        );
        // What r0 the rotated engine would then read, if its channels stay
        // level-matched: r0 = (rho - 1)/(rho + 1) at rho = rho*.
        let out_r0 = (rho_star - 1.0) / (rho_star + 1.0);
        let g = lobe_response(&band, hz).norm();
        let lobe_pair = 10.0 * (1.0 + g * g).log10();
        theta_curve.push((hz, if feasible { t.sqrt() } else { 0.0 }));
        if (NODAL_HZ.0..=NODAL_HZ.1).contains(&hz) {
            if feasible {
                inside.push((hz, cost, headroom, se));
            } else {
                infeasible.push(hz);
            }
        }
        println!(
            "| {hz:.0} | {:+.2} | {r0:+.3} | {:+.2} | {t:+.3} | **{theta:.1}** | **{cost:+.2}** | \
{headroom:+.2} | {se:.2} | {:+.2} | {out_r0:+.3} | {g:.3} | {lobe_pair:+.2} |",
            db(rho_star),
            db(rho0),
            cost - headroom
        );
    }
    // ---- Closing the loop: apply the derived theta(f) and re-measure. ----
    verify_theta(grid, rows, &theta_curve);

    if !inside.is_empty() {
        let costs: Vec<f64> = inside.iter().map(|x| x.1).collect();
        let heads: Vec<f64> = inside.iter().map(|x| x.2).collect();
        let nets: Vec<f64> = inside.iter().map(|x| x.2 - x.1).collect();
        let mc = median(&mut costs.clone());
        let mh = median(&mut heads.clone());
        let mn = median(&mut nets.clone());
        println!(
            "\n**Verdict input.** Inside {:.0}-{:.0} Hz the rotation that reproduces the \
recording's mid/side ratio costs the mono sum **{mc:+.2} dB** median (peak **{:+.2} dB** at \
{:.0} Hz). The engine's mono currently sits **{mh:+.2} dB** above the recording's there. \
Residual after the rotation: **{mn:+.2} dB** — negative means the engine's mono ends up *darker* \
than the recording's in the nodal band.",
            NODAL_HZ.0,
            NODAL_HZ.1,
            costs.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            inside
                .iter()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .map(|x| x.0)
                .unwrap_or(0.0),
        );
        if !infeasible.is_empty() {
            println!(
                "\nBands inside the nodal range where a rotation is **not** the mechanism \
(the engine already has more side than the recording asks for, so `sin^2 theta < 0`): {:?} Hz.",
                infeasible
            );
        }
    }
}

/// `sin(theta)` at an arbitrary frequency, piecewise-linear in log frequency
/// between the sixth-octave points the table derived, and zero outside them.
fn sin_theta_at(curve: &[(f64, f64)], f: f64) -> f64 {
    if curve.is_empty() || f <= curve[0].0 || f >= curve[curve.len() - 1].0 {
        return 0.0;
    }
    for w in curve.windows(2) {
        if f >= w[0].0 && f <= w[1].0 {
            let u = (f.ln() - w[0].0.ln()) / (w[1].0.ln() - w[0].0.ln());
            return w[0].1 + u * (w[1].1 - w[0].1);
        }
    }
    0.0
}

/// **The prediction, applied and re-measured.** The table above is arithmetic on
/// pooled band energies and assumes the injected mid is orthogonal to the
/// geometric side. That is only *nearly* true — the geometric side is a
/// difference of a signal and a delayed copy of itself, which is in quadrature
/// with it at low frequency but not exactly — so the honest thing is to run the
/// rotation and read the boards off the result.
fn verify_theta(grid: &[f64], rows: &[KeyRow], curve: &[(f64, f64)]) {
    let preset = shipped();
    let bare = without_lobe(&preset);
    let rotated: Vec<Take> = rows
        .par_iter()
        .map(|row| {
            let (l, r) = render_key(&bare, row.key);
            let n = l.len().next_power_of_two();
            let mut planner = FftPlanner::<f64>::new();
            let mut a = forward(&l, n, &mut planner);
            let mut b = forward(&r, n, &mut planner);
            for j in 0..n {
                let (f, _) = bin_hz(j, n);
                let st = sin_theta_at(curve, f).clamp(0.0, 0.999);
                let ct = (1.0 - st * st).sqrt();
                let (m, s) = ((a[j] + b[j]) * 0.5, (a[j] - b[j]) * 0.5);
                let m2 = m * ct;
                let s2 = s + m * st;
                a[j] = m2 + s2;
                b[j] = m2 - s2;
            }
            Take::from_spectra(&a, &b, grid)
        })
        .collect();

    let references: Vec<Take> = rows
        .iter()
        .map(|r| Take {
            bands: r.reference.bands.clone(),
        })
        .collect();
    println!(
        "\n### The same rotation, applied to the engine's own renders and re-measured\n"
    );
    println!(
        "| Hz | theta deg | ENG r0 after | REF r0 | ENG pair-over-mono after | REF pair-over-mono | mono change dB | predicted cost dB | ENG mono - REF mono after dB |"
    );
    println!("|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
    for (i, &hz) in grid.iter().enumerate() {
        let live: Vec<usize> = (0..rows.len())
            .filter(|&k| rows[k].reference.readable(i) && rows[k].engine_bare.readable(i))
            .collect();
        if live.len() < 5 {
            continue;
        }
        let st = sin_theta_at(curve, hz);
        let mut r0v: Vec<f64> = live.iter().map(|&k| rotated[k].r0(i)).collect();
        let mut refr0: Vec<f64> = live.iter().map(|&k| rows[k].reference.r0(i)).collect();
        let pooled_pair = |takes: &[Take]| -> f64 {
            let (mut p, mut m) = (0.0, 0.0);
            for &k in &live {
                let x = &takes[k];
                let tot = x.mono_total();
                p += (x.bands[i][0] + x.bands[i][1]) / tot;
                m += 2.0 * x.bands[i][2] / tot;
            }
            db(p / m)
        };
        // Mono change is measured against the *bare* engine, unnormalised:
        // the rotation is the only thing that moved.
        let (mut a, mut b) = (0.0, 0.0);
        for &k in &live {
            a += rotated[k].bands[i][2];
            b += rows[k].engine_bare.bands[i][2];
        }
        let dmono = db(a / b);
        // And where that leaves the engine against the recording, level-matched
        // on the *bare* engine's own 100-810 total so the rotation's loss is
        // not normalised away.
        let (mut ea, mut eb) = (0.0, 0.0);
        for &k in &live {
            ea += rotated[k].bands[i][2] / rows[k].engine_bare.mono_total();
            eb += rows[k].reference.bands[i][2] / rows[k].reference.mono_total();
        }
        let after = db(ea / eb);
        let cost = if st > 0.0 { -20.0 * (1.0 - st * st).sqrt().log10() } else { 0.0 };
        println!(
            "| {hz:.0} | {:.1} | {:+.3} | {:+.3} | {:+.2} | {:+.2} | {dmono:+.2} | {:+.2} | {after:+.2} |",
            st.asin().to_degrees(),
            median(&mut r0v),
            median(&mut refr0),
            pooled_pair(&rotated),
            pooled_pair(&references),
            cost
        );
    }
}

// ---------------------------------------------------------------------------
// Section 3: the estimator's group-delay bias
// ---------------------------------------------------------------------------

/// The bass keys `tuner/tests/mics.rs` reads the spacing off.
const LAG_KEYS: [u8; 11] = [21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51];

#[derive(Clone, Copy)]
enum Stage {
    None,
    Lobe(ModalBand),
    LobeDiffused(ModalBand),
    LobeThenUndo(ModalBand),
    Givens(ModalBand),
    GivensMinPhase(ModalBand),
    GivensThenUndo(ModalBand),
    SideAllpass(f64),
    CommonAllpass(f64),
}

fn stage_name(s: &Stage) -> String {
    match s {
        Stage::None => "no lobe (the geometry alone)".into(),
        Stage::Lobe(b) => format!("lobe {:.0}-{:.0} x{:.2}", b.lo_hz, b.hi_hz, b.lift),
        Stage::LobeDiffused(b) => format!(
            "lobe {:.0}-{:.0} x{:.2} + item 393's FOUR-ALLPASS DIFFUSER",
            b.lo_hz, b.hi_hz, b.lift
        ),
        Stage::LobeThenUndo(b) => format!(
            "lobe {:.0}-{:.0} x{:.2} + COMPENSATION",
            b.lo_hz, b.hi_hz, b.lift
        ),
        Stage::Givens(b) => format!(
            "Givens (zero-phase) {:.0}-{:.0} tan th={:.2}",
            b.lo_hz, b.hi_hz, b.lift
        ),
        Stage::GivensMinPhase(b) => format!(
            "Givens (MINIMUM-PHASE, causal) {:.0}-{:.0} tan th={:.2}",
            b.lo_hz, b.hi_hz, b.lift
        ),
        Stage::GivensThenUndo(b) => format!(
            "Givens {:.0}-{:.0} tan th={:.2} + COMPENSATION",
            b.lo_hz, b.hi_hz, b.lift
        ),
        Stage::SideAllpass(hz) => format!(
            "allpass {hz:.0} Hz on the SIDE (tau = {:.3} ms at 100 Hz)",
            1e3 * allpass_group_delay_s(*hz, 100.0)
        ),
        Stage::CommonAllpass(hz) => format!(
            "allpass {hz:.0} Hz on BOTH (tau = {:.3} ms at 100 Hz) — the control",
            1e3 * allpass_group_delay_s(*hz, 100.0)
        ),
    }
}

fn apply_stage(a: &mut [C64], b: &mut [C64], stage: &Stage) {
    match stage {
        Stage::None => {}
        Stage::Lobe(band) => apply_lobe(a, b, band),
        Stage::LobeDiffused(band) => apply_lobe_diffused(a, b, band),
        Stage::LobeThenUndo(band) => {
            apply_lobe(a, b, band);
            undo_lobe(a, b, band);
        }
        Stage::Givens(band) => apply_givens(a, b, band),
        Stage::GivensMinPhase(band) => apply_givens_min_phase(a, b, band),
        Stage::GivensThenUndo(band) => {
            apply_givens(a, b, band);
            undo_givens(a, b, band);
        }
        Stage::SideAllpass(hz) => apply_side_allpass(a, b, *hz),
        Stage::CommonAllpass(hz) => apply_common_allpass(a, b, *hz),
    }
}

fn band_at(lo: f32, hi: f32, lift: f32) -> ModalBand {
    ModalBand { lo_hz: lo, hi_hz: hi, lift }
}

fn lag_section() -> Result<(), Box<dyn std::error::Error>> {
    let preset = shipped();
    let bare = without_lobe(&preset);
    let mics = preset.voicing.mics.expect("a pair");
    let shipped_band = shipped_band(&preset).expect("a band");
    let config = LagConfig::default();

    println!("\n## 3. The spacing readback, and the estimator's group-delay bias\n");
    println!(
        "Eleven bass keys, `LagConfig` default ({:.0}-{:.0} Hz, {:.0} s), \
`spacing = |median lag| * c / ENGINE_LAG_PER_ITD` with `ENGINE_LAG_PER_ITD = {ENGINE_LAG_PER_ITD:.3}`. \
Every stage below the first is applied **offline** to the same renders, which is exact for this \
construction; the validation row proves it.\n",
        config.band_hz.0, config.band_hz.1, config.window_s
    );

    // Validation: the offline lobe against a real render with the lobe.
    {
        let spacing = 0.12f32;
        let with = MicVoicing { spacing_m: spacing, ..mics };
        let mut p_bare = bare.clone();
        p_bare.voicing.mics = Some(MicVoicing { modal: None, ..with });
        let mut p_full = preset.clone();
        p_full.voicing.mics = Some(with);
        let pairs: Vec<(f64, f64)> = LAG_KEYS
            .par_iter()
            .map(|&key| {
                let (bl, br) = render_key(&p_bare, key);
                let (fl, fr) = render_key(&p_full, key);
                let n = (2 * bl.len()).next_power_of_two();
                let mut planner = FftPlanner::<f64>::new();
                let mut a = forward(&bl, n, &mut planner);
                let mut b = forward(&br, n, &mut planner);
                apply_lobe(&mut a, &mut b, &shipped_band);
                let sl = inverse(a, &mut planner);
                let sr = inverse(b, &mut planner);
                let real = interchannel_lag(&fl, &fr, SR, &config).expect("two channels");
                let synth = interchannel_lag(&sl[..bl.len()], &sr[..bl.len()], SR, &config)
                    .expect("two channels");
                (real.lag_s, synth.lag_s)
            })
            .collect();
        let worst = pairs
            .iter()
            .map(|(a, b)| 1e6 * (a - b).abs())
            .fold(0.0f64, f64::max);
        println!(
            "**Validation** — the offline lobe against the engine's own rendered lobe, eleven \
keys at 0.12 m: worst per-key lag disagreement **{worst:.1} us** ({:.4} ms sample period is \
{:.1} us).\n",
            1e3 / SR,
            1e6 / SR
        );
    }

    let stages: Vec<Stage> = vec![
        Stage::None,
        Stage::Lobe(shipped_band),
        Stage::Lobe(band_at(218.0, 300.0, 2.25)),
        Stage::Lobe(band_at(200.0, 280.0, 2.25)),
        Stage::Lobe(band_at(180.0, 260.0, 2.25)),
        Stage::Lobe(band_at(160.0, 240.0, 2.25)),
        Stage::LobeDiffused(shipped_band),
        Stage::LobeDiffused(band_at(218.0, 300.0, 2.25)),
        Stage::LobeThenUndo(shipped_band),
        Stage::LobeThenUndo(band_at(180.0, 260.0, 2.25)),
        Stage::Givens(shipped_band),
        Stage::Givens(band_at(218.0, 300.0, 2.25)),
        Stage::Givens(band_at(180.0, 260.0, 2.25)),
        Stage::GivensMinPhase(shipped_band),
        Stage::GivensMinPhase(band_at(218.0, 300.0, 2.25)),
        Stage::GivensMinPhase(band_at(180.0, 260.0, 2.25)),
        Stage::GivensThenUndo(band_at(180.0, 260.0, 2.25)),
        Stage::SideAllpass(60.0),
        Stage::SideAllpass(120.0),
        Stage::SideAllpass(300.0),
        Stage::CommonAllpass(120.0),
    ];

    println!("| stage | 0.12 m | 0.24 m | 0.48 m | worst error |");
    println!("|---|---:|---:|---:|---:|");

    // Render once per (spacing, key) with no lobe; every stage reuses them.
    let spacings = [0.12f32, 0.24, 0.48];
    let spectra: Vec<Vec<KeySpectra>> = spacings
        .iter()
        .map(|&spacing| {
            let mut p = bare.clone();
            p.voicing.mics = Some(MicVoicing {
                spacing_m: spacing,
                modal: None,
                ..mics
            });
            LAG_KEYS
                .par_iter()
                .map(|&key| {
                    let (l, r) = render_key(&p, key);
                    let len = l.len();
                    let n = (2 * len).next_power_of_two();
                    let mut planner = FftPlanner::<f64>::new();
                    (
                        forward(&l, n, &mut planner),
                        forward(&r, n, &mut planner),
                        len,
                    )
                })
                .collect()
        })
        .collect();

    let mut rows: Vec<(String, [f64; 3], [f64; 3])> = Vec::new();
    for stage in &stages {
        let mut recovered = [0.0f64; 3];
        let mut lags = [0.0f64; 3];
        for (si, &spacing) in spacings.iter().enumerate() {
            let mut per_key: Vec<f64> = spectra[si]
                .par_iter()
                .map(|(a0, b0, len)| {
                    let mut a = a0.clone();
                    let mut b = b0.clone();
                    apply_stage(&mut a, &mut b, stage);
                    let mut planner = FftPlanner::<f64>::new();
                    let l = inverse(a, &mut planner);
                    let r = inverse(b, &mut planner);
                    interchannel_lag(&l[..*len], &r[..*len], SR, &config)
                        .expect("two channels")
                        .lag_s
                })
                .collect();
            let m = median(&mut per_key);
            lags[si] = m;
            recovered[si] = m.abs() * SPEED_OF_SOUND / ENGINE_LAG_PER_ITD / f64::from(spacing);
        }
        rows.push((stage_name(stage), recovered, lags));
    }

    for (name, rec, lags) in &rows {
        let worst = rec
            .iter()
            .map(|r| (r - 1.0).abs())
            .fold(0.0f64, f64::max);
        println!(
            "| {name} | {:+.0} % ({:+.3} ms) | {:+.0} % ({:+.3} ms) | {:+.0} % ({:+.3} ms) | \
{}{:.0} % |",
            100.0 * (rec[0] - 1.0),
            1e3 * lags[0],
            100.0 * (rec[1] - 1.0),
            1e3 * lags[1],
            100.0 * (rec[2] - 1.0),
            1e3 * lags[2],
            if worst > 0.20 { "**RED** " } else { "" },
            100.0 * worst
        );
    }

    // The bias as a delay, at the spacing the gate cares about most.
    let base = rows[0].2[0];
    println!("\n**The bias, as a delay at 0.12 m** (median lag less the no-lobe median):\n");
    println!("| stage | median lag ms | bias vs no-lobe us | as % of readback |");
    println!("|---|---:|---:|---:|");
    for (name, rec, lags) in &rows {
        println!(
            "| {name} | {:+.4} | {:+.1} | {:+.1} % |",
            1e3 * lags[0],
            1e6 * (lags[0] - base),
            100.0 * (rec[0] - rows[0].1[0])
        );
    }
    println!(
        "\n**What `ENGINE_LAG_PER_ITD` would be if the estimator inverted the declared stage \
first.** `K = |median lag| * c / d` per spacing; with the compensation the three agree, which is \
what makes it a property of the geometry rather than of the mic voicing.\n"
    );
    println!("| stage | K at 0.12 | K at 0.24 | K at 0.48 | geometric mean | spread |");
    println!("|---|---:|---:|---:|---:|---:|");
    for (name, _rec, lags) in &rows {
        let k: Vec<f64> = spacings
            .iter()
            .enumerate()
            .map(|(i, &d)| lags[i].abs() * SPEED_OF_SOUND / f64::from(d))
            .collect();
        let gm = (k[0] * k[1] * k[2]).cbrt();
        let spread = k.iter().map(|x| (x / gm - 1.0).abs()).fold(0.0f64, f64::max);
        println!(
            "| {name} | {:.3} | {:.3} | {:.3} | **{gm:.3}** | +-{:.1} % |",
            k[0],
            k[1],
            k[2],
            100.0 * spread
        );
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let section = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "all".to_string());
    let grid = grid();
    if section == "mono" || section == "theta" || section == "keys" || section == "all" {
        let rows = mono_section(&grid)?;
        if section == "keys" || section == "all" {
            report_keys(&grid, &rows);
            report_halo(&rows)?;
            report_parts(&rows)?;
            report_headroom(&grid, &rows)?;
        }
        if section == "mono" || section == "all" {
            report_mono(&grid, &rows);
        }
        if section == "theta" || section == "all" {
            report_theta(&grid, &rows);
        }
    }
    if section == "lag" || section == "all" {
        lag_section()?;
    }
    Ok(())
}
