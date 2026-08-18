//! **Can an antisymmetric (side-only) radiator buy the recording's nodal band
//! without touching the mono fold-down?** — the fifth attempt's enabling
//! hypothesis, measured (`DECISIONS.md` 417+, after 405-416).
//!
//! ```text
//! cargo run --release -p forensics --bin side_injection -- [variant]
//! variants: fdn | board-ap | mid-ap | all   (default: all)
//! ```
//!
//! # The hypothesis this instrument tests
//!
//! Items 399-416 refused four mechanisms. Every one of them *transformed* the
//! signal already there — a Givens rotation, an allpass pair, a colouration of
//! the drive — and so either paid a mono bill (rotation) or moved the mono sum
//! and the pair together (radiation). A soundboard **nodal line** is different
//! in kind: it is an antisymmetric radiator, so what it puts into the two
//! capsules is `+s` and `−s`, and `(L + R)/2` does not see it at all. The
//! recording is the existence proof — its pair stands +5..+9.4 dB above its own
//! mono over 180-300 Hz.
//!
//! # What is measured, and what turns out to matter
//!
//! The instrument fits a per-sixth-octave level for a decorrelated side-only
//! source over 160-320 Hz so that the treated engine's **pair-over-mono**
//! lands on the recording's, then re-measures everything the gates read. The
//! mono fold-down is asserted unmoved (§3) — that is the hypothesis's whole
//! claim and it holds exactly.
//!
//! The load-bearing column is **§4's `pair share`**, which no previous item
//! printed per band: what each take puts in the band as a fraction of its own
//! *pair* energy. `pair-over-mono` is a ratio of two of the take's own numbers
//! and says nothing about how loud the band is; `pair share` does. See the
//! verdict the run prints.

use std::f64::consts::TAU;
use std::path::Path;

use piano_emulator::preset::{MicVoicing, Preset};
use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_emulator::types::Event;
use piano_tuner::audio::Audio;
use piano_tuner::realism::{self, ChannelItem, RecordedKeys, StereoItem, VelocityLayers};
use piano_tuner::sampler::{SamplerEvent, SAMPLER_VERSION};
use piano_tuner::{cache, SampleLibrary, Sampler, TimedEvent, SAMPLE_RATE};

use rayon::prelude::*;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

type C64 = Complex<f64>;

const SFZ: &str = "data/salamander/SalamanderGrandPiano-V3+20200602.sfz";
const DATA: &str = "data/salamander";
const VELOCITY: u8 = 90;
const RENDER_S: f64 = 3.0;
const PREROLL: usize = realism::STEREO_PREROLL_SAMPLES;
const PREROLL_S: f64 = PREROLL as f64 / 48_000.0;
const SR: f64 = 48_000.0;

/// The span every share is normalised inside, `DECISIONS.md` 343's rule.
const SPAN_HZ: (f64, f64) = (100.0, 810.0);

/// **The injected band's centres.** The seven sixth-octave centres 160 · 2^(k/6),
/// k = 0..6. [`INJECT_HZ`] is these bands' own outer edges, so the injection
/// lives in exactly seven bands of [`grid`] and leaks into no other one at all.
/// The edges are the recording's own profile's: its pair-over-mono crosses zero
/// between 127 and 160 Hz on the way up and is back inside a decibel of the
/// engine's by 359 Hz (both printed in §4).
const INJECT_CENTRES: (f64, f64) = (160.0, 320.0);

/// The injected range, in Hz: the 160 Hz band's own lower edge to the 320 Hz
/// band's own upper edge. Computed from [`band_edges`] rather than written down,
/// because a hand-written edge that misses a band's own by one FFT bin silently
/// takes that bin out of the fold-down — which is the one thing this instrument
/// may not do.
fn inject_hz() -> (f64, f64) {
    (
        band_edges(INJECT_CENTRES.0).0,
        band_edges(INJECT_CENTRES.1).1,
    )
}

// ---------------------------------------------------------------------------
// Grid
// ---------------------------------------------------------------------------

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

/// The indices of [`grid`] the injection has a control point on.
fn control_points(grid: &[f64]) -> Vec<usize> {
    (0..grid.len())
        .filter(|&i| {
            grid[i] > INJECT_CENTRES.0 * 0.99 && grid[i] < INJECT_CENTRES.1 * 1.01
        })
        .collect()
}

/// The FFT bin a frequency falls in, exactly as [`band_energies`] rounds it.
fn bin_of(hz: f64, n: usize) -> usize {
    (hz * n as f64 / SR).round() as usize
}

// ---------------------------------------------------------------------------
// Presets and rendering
// ---------------------------------------------------------------------------

fn shipped() -> Preset {
    Preset::load(Path::new("presets/salamander-c5.toml")).expect("the measured preset loads")
}

/// The shipped preset with `[voicing.mics.modal]` deleted: the capsule pair and
/// nothing else, which is items 405-416's own control and the instrument the
/// injection is meant to *replace* the lobe on.
fn bare(preset: &Preset) -> Preset {
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
// Spectra
// ---------------------------------------------------------------------------

fn forward(x: &[f32], n: usize, planner: &mut FftPlanner<f64>) -> Vec<C64> {
    let fft = planner.plan_fft_forward(n);
    let mut buf: Vec<C64> = (0..n)
        .map(|i| C64::new(f64::from(x.get(i).copied().unwrap_or(0.0)), 0.0))
        .collect();
    fft.process(&mut buf);
    buf
}

fn inverse(mut buf: Vec<C64>, planner: &mut FftPlanner<f64>) -> Vec<f64> {
    let n = buf.len();
    let fft = planner.plan_fft_inverse(n);
    fft.process(&mut buf);
    let s = 1.0 / n as f64;
    buf.iter().map(|c| c.re * s).collect()
}

fn bin_hz(j: usize, n: usize) -> f64 {
    let half = n / 2;
    if j <= half {
        j as f64 * SR / n as f64
    } else {
        (n - j) as f64 * SR / n as f64
    }
}

/// `[E_L, E_R, E_M, E_S, Re<L,R>]` of one band, both halves of the transform.
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

fn f64_db(x: f64) -> f64 {
    20.0 * x.max(1e-300).log10()
}

fn db(x: f64) -> f64 {
    10.0 * x.max(1e-300).log10()
}

fn median(v: &mut Vec<f64>) -> f64 {
    v.retain(|x| x.is_finite());
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

// ---------------------------------------------------------------------------
// The three candidate side sources
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Source {
    /// **The real implementation's own signal.** The board path's difference
    /// tap `l − r`, recovered from a `board_mix = 1.0` render by dividing out
    /// the one-pole coherence highpass the engine puts on it, then band-limited
    /// to [`INJECT_HZ`] instead of highpassed. This is literally
    /// `soundboard::Fdn`'s `s = 0.5*(l - r)` before `side_b` acts on it, so a
    /// shipped stage would need no new signal at all — only a different filter
    /// on one that already exists.
    Fdn,
    /// **The surrogate the mandate names.** The board-path-only render's own
    /// *mid*, band-limited and pushed through a three-stage Schroeder allpass
    /// chain so that it is decorrelated from everything already in the pair.
    /// Same spectral colour as the board's field, an independent realisation.
    BoardAllpass,
    /// The same decorrelation applied to the **whole bare render's mid** —
    /// direct path included, so it is alive from the first sample where the
    /// two board-derived sources are not. The optimistic bound.
    MidAllpass,
}

impl Source {
    fn name(self) -> &'static str {
        match self {
            Source::Fdn => "fdn",
            Source::BoardAllpass => "board-ap",
            Source::MidAllpass => "mid-ap",
        }
    }
}

/// The engine's board-side coherence highpass, exactly as `soundboard::Fdn`
/// runs it: `lp += b (s - lp); hp = s - lp`, i.e.
/// `H(z) = (1-b)(1 - z^-1) / (1 - (1-b) z^-1)`.
fn coherence_highpass(b: f64, f: f64) -> C64 {
    let z1 = C64::from_polar(1.0, -TAU * f / SR);
    C64::new(1.0 - b, 0.0) * (C64::new(1.0, 0.0) - z1) / (C64::new(1.0, 0.0) - z1 * (1.0 - b))
}

/// The one-pole coefficient the shipped preset's mic voicing produces
/// (`Mics::new`).
fn diffuse_b(v: &MicVoicing) -> f64 {
    const MIC_DIFFUSE_POLE_K: f64 = 0.4266;
    const SPEED_OF_SOUND: f64 = 343.0;
    let hz = MIC_DIFFUSE_POLE_K * SPEED_OF_SOUND / f64::from(v.spacing_m) * f64::from(v.diffuse_coherence);
    let w = (TAU * hz / SR).min(std::f64::consts::PI);
    1.0 - (-w).exp()
}

/// A Schroeder allpass `(-c + z^-m)/(1 - c z^-m)` — unit modulus at every
/// frequency, so it moves no energy anywhere, and `m` samples of delay per
/// bounce, which is what decorrelates.
fn schroeder(m: usize, c: f64, f: f64) -> C64 {
    let zm = C64::from_polar(1.0, -TAU * f * m as f64 / SR);
    (C64::new(-c, 0.0) + zm) / (C64::new(1.0, 0.0) - zm * c)
}

/// Three of them, coprime lengths, ~35 ms of total spread: enough that the
/// phase sweeps many turns across a sixth-octave band, which is what makes the
/// band-integrated cross term with the original vanish.
const AP: [(usize, f64); 3] = [(521, 0.7), (853, 0.7), (1_223, 0.7)];

fn decorrelate(f: f64) -> C64 {
    AP.iter().fold(C64::new(1.0, 0.0), |acc, &(m, c)| acc * schroeder(m, c, f))
}

// ---------------------------------------------------------------------------
// Per-key material
// ---------------------------------------------------------------------------

/// Everything one key contributes, with the injected band's raw bins kept so
/// the fit can re-evaluate it without another transform.
struct Key {
    #[allow(dead_code)]
    key: u8,
    label: String,
    /// Base engine (bare) band energies over the whole grid.
    base: Vec<[f64; 5]>,
    /// Recording's band energies.
    reference: Vec<[f64; 5]>,
    /// Shipped engine (lobe present) band energies.
    shipped: Vec<[f64; 5]>,
    /// `mono_total` of each take over the span — unchanged by any injection.
    base_mono_total: f64,
    ref_mono_total: f64,
    shipped_mono_total: f64,
    base_pair_total: f64,
    ref_pair_total: f64,
    /// Positive-frequency bins of the injected range: `(bin, A, B, X)`.
    bins: Vec<(usize, C64, C64, C64)>,
    /// The first of them.
    blo: usize,
    /// Transform length the bins came from.
    n: usize,
    /// Base render, for the boards.
    base_l: Vec<f32>,
    base_r: Vec<f32>,
    /// Full injection source spectrum, unit gain, already masked to the
    /// injected bins — for the boards.
    xspec: Vec<C64>,
    /// Reference and alternate audio, for the boards.
    reference_audio: Audio,
    alternate_audio: Audio,
    shipped_audio: Audio,
}

fn totals(bands: &[[f64; 5]]) -> (f64, f64) {
    (
        bands.iter().map(|e| e[2]).sum(),
        bands.iter().map(|e| e[0] + e[1]).sum(),
    )
}

fn build_key(
    key: u8,
    grid: &[f64],
    preset: &Preset,
    bare_p: &Preset,
    board_p: &Preset,
    alt_vel: u8,
    source: Source,
) -> Result<Key, piano_tuner::Error> {
    let reference_audio = reference_key(key, VELOCITY)?;
    let alternate_audio = reference_key(key, alt_vel)?;
    let (bl, br) = render_key(bare_p, key);
    let (sl, sr) = render_key(preset, key);
    let (cl, cr) = render_key(board_p, key);
    let len = bl.len();
    let n = len.next_power_of_two();
    let mut planner = FftPlanner::<f64>::new();
    let a = forward(&bl, n, &mut planner);
    let b = forward(&br, n, &mut planner);

    // --- the injection source, as a full spectrum, at unit gain ---
    let mics = preset.voicing.mics.expect("the measured preset has a pair");
    let bcoef = diffuse_b(&mics);
    let mut x: Vec<C64> = match source {
        Source::Fdn => {
            // side of the board-only render = HP(0.5*(l - r)); divide the
            // highpass out to recover the raw difference tap.
            let ca = forward(&cl, n, &mut planner);
            let cb = forward(&cr, n, &mut planner);
            (0..n)
                .map(|j| {
                    let f = bin_hz(j, n);
                    let s = (ca[j] - cb[j]) * 0.5;
                    let mut h = coherence_highpass(bcoef, f);
                    if j > n / 2 {
                        h = h.conj();
                    }
                    if h.norm() < 1e-12 {
                        C64::new(0.0, 0.0)
                    } else {
                        s / h
                    }
                })
                .collect()
        }
        Source::BoardAllpass => {
            let ca = forward(&cl, n, &mut planner);
            let cb = forward(&cr, n, &mut planner);
            (0..n)
                .map(|j| {
                    let f = bin_hz(j, n);
                    let m = (ca[j] + cb[j]) * 0.5;
                    let mut h = decorrelate(f);
                    if j > n / 2 {
                        h = h.conj();
                    }
                    m * h
                })
                .collect()
        }
        Source::MidAllpass => (0..n)
            .map(|j| {
                let f = bin_hz(j, n);
                let m = (a[j] + b[j]) * 0.5;
                let mut h = decorrelate(f);
                if j > n / 2 {
                    h = h.conj();
                }
                m * h
            })
            .collect(),
    };
    // Hard band limit, **by bin and not by hertz**: the mask is exactly the
    // union of the seven bands' own bin ranges as `band_energies` rounds them,
    // so not one bin the fold-down is measured over can fall outside the mask
    // and no statistic outside the injected bands can move at all.
    let (ilo, ihi) = inject_hz();
    let (blo, bhi) = (bin_of(ilo, n).max(1), bin_of(ihi, n).min(n / 2));
    for (j, v) in x.iter_mut().enumerate() {
        let jj = if j <= n / 2 { j } else { n - j };
        if jj < blo || jj > bhi {
            *v = C64::new(0.0, 0.0);
        }
    }
    let bins: Vec<(usize, C64, C64, C64)> =
        (blo..=bhi).map(|j| (j, a[j], b[j], x[j])).collect();


    let ref_bands: Vec<[f64; 5]> = {
        let ra = forward(&reference_audio.channels[0], n, &mut planner);
        let rb = forward(&reference_audio.channels[1], n, &mut planner);
        grid.iter()
            .map(|&hz| {
                let (lo, hi) = band_edges(hz);
                band_energies(&ra, &rb, lo, hi)
            })
            .collect()
    };
    let base_bands: Vec<[f64; 5]> = grid
        .iter()
        .map(|&hz| {
            let (lo, hi) = band_edges(hz);
            band_energies(&a, &b, lo, hi)
        })
        .collect();
    let shipped_bands: Vec<[f64; 5]> = {
        let sa = forward(&sl, n, &mut planner);
        let sb = forward(&sr, n, &mut planner);
        grid.iter()
            .map(|&hz| {
                let (lo, hi) = band_edges(hz);
                band_energies(&sa, &sb, lo, hi)
            })
            .collect()
    };

    let (bm, bp) = totals(&base_bands);
    let (rm, rp) = totals(&ref_bands);
    let (sm, _) = totals(&shipped_bands);
    let shipped_audio = Audio::new(SAMPLE_RATE, vec![sl, sr])?;

    Ok(Key {
        key,
        label: realism::note_name(key),
        base: base_bands,
        reference: ref_bands,
        shipped: shipped_bands,
        base_mono_total: bm,
        ref_mono_total: rm,
        shipped_mono_total: sm,
        base_pair_total: bp,
        ref_pair_total: rp,
        bins,
        blo,
        n,
        base_l: bl,
        base_r: br,
        xspec: x,
        reference_audio,
        alternate_audio,
        shipped_audio,
    })
}

// ---------------------------------------------------------------------------
// The injection curve
// ---------------------------------------------------------------------------

/// **The injection curve, parameterised in energy.**
///
/// The fitted numbers `u[c]` are *energy* weights at the seven control points,
/// and the applied amplitude is `sqrt(Σ_c hat_c(f) u_c)` with `hat_c` the
/// piecewise-linear-in-log-frequency hat that is one at its own centre, zero at
/// its neighbours' and zero at both edges of the injected range.
///
/// The parameterisation is the reason the fit is a linear solve rather than a
/// search. A source decorrelated from the pair adds `|G|²|X|²` to each channel,
/// so the pair energy a band gains is `Σ_bins 2|X|² Σ_c hat_c u_c` — **linear in
/// `u`** — and the whole fit is one 7x7 system with a measured Jacobian.
/// Whatever is left over is the *coherent* part of the injection, which §1
/// measures separately and which the Newton rounds absorb.
#[derive(Clone, Debug)]
struct Curve {
    /// `(centre Hz, energy weight)`, ascending.
    nodes: Vec<(f64, f64)>,
    lo: f64,
    hi: f64,
}

impl Curve {
    fn new(centres: &[f64]) -> Curve {
        let (lo, hi) = inject_hz();
        Curve {
            nodes: centres.iter().map(|&f| (f, 0.0)).collect(),
            lo,
            hi,
        }
    }
    /// Energy weight at `f`: **constant across each control point's own
    /// sixth-octave band**, zero outside the injected range.
    ///
    /// It is piecewise constant and not interpolated on purpose, and the choice
    /// is *generous to the hypothesis*. A band's score integrates across the
    /// band, so an interpolated curve peaked at one centre spills into its
    /// neighbours and the seven bands stop being seven free parameters — with a
    /// triangular basis the fit stalls 4.4 dB out at 285 Hz because 254 Hz's
    /// own weight already overshoots it. A brick per band makes the system
    /// diagonal and the fit exact, so nothing below is an artifact of the fit
    /// failing. What a *realisable* stage would put here is
    /// `soundboard::Radiation`'s own design — third-octave peaking sections
    /// overlapping by half, Newton-inverted to pass through the declared points
    /// (`DECISIONS.md` 412(b)) — which ripples between the points and is
    /// therefore strictly less sharp than this. Every shape error measured
    /// below is a floor on what that design could do, not a ceiling.
    fn energy_at(&self, f: f64) -> f64 {
        if f <= self.lo || f >= self.hi {
            return 0.0;
        }
        let brick = |g: f64| -> f64 {
            if g <= self.lo || g >= self.hi {
                return 0.0;
            }
            for &(c, u) in &self.nodes {
                let (lo, hi) = band_edges(c);
                if g >= lo && g < hi {
                    return u;
                }
            }
            0.0
        };
        // The step is ramped over a 24th of an octave centred on each band
        // edge. Without it the magnitude is a 60 dB brick wall and its
        // minimum-phase impulse response is longer than the transform, which
        // time-aliases; with it the realised stage and the fit's own arithmetic
        // agree to a thousandth of a decibel (§3's offline-vs-rendered row).
        let w = 2.0f64.powf(1.0 / 48.0).ln();
        let lf = f.ln();
        let here = brick(f);
        let (below, above) = (brick((lf - w).exp()), brick((lf + w).exp()));
        if below != here {
            // distance to the edge, found by bisection on the brick
            let (mut a, mut b) = (lf - w, lf);
            for _ in 0..24 {
                let m = 0.5 * (a + b);
                if brick(m.exp()) == here {
                    b = m;
                } else {
                    a = m;
                }
            }
            let t = (lf - b) / w;
            return here + (below - here) * 0.5 * (1.0 - t);
        }
        if above != here {
            let (mut a, mut b) = (lf, lf + w);
            for _ in 0..24 {
                let m = 0.5 * (a + b);
                if brick(m.exp()) == here {
                    a = m;
                } else {
                    b = m;
                }
            }
            let t = (a - lf) / w;
            return here + (above - here) * 0.5 * (1.0 - t);
        }
        here
    }
    /// The amplitude the stage applies.
    fn amp_at(&self, f: f64) -> f64 {
        self.energy_at(f).max(0.0).sqrt()
    }
    fn with(&self, u: &[f64]) -> Curve {
        let mut c = self.clone();
        for (n, &v) in c.nodes.iter_mut().zip(u) {
            n.1 = v;
        }
        c
    }
}

/// The treated band energies of one key over the whole grid, given the
/// **realised** complex gain of the injection filter.
///
/// It is the realised gain and not the fitted magnitude because the source is
/// not fully incoherent with the pair it is added to (§1's `coh` column reads
/// 0.13-0.40), so the cross term `2 Re<A, S>` is a real part of the band's
/// energy and it turns with the filter's phase. Using a real gain here and a
/// minimum-phase one in the render puts the two **12 dB** apart on a bass key;
/// using the same filter for both puts them at the level §3 prints.
///
/// Outside the injected bands the result is the base one unchanged, because the
/// source is exactly zero there — by bin, not by hertz.
fn treated_bands(k: &Key, grid: &[f64], gain: &[C64]) -> Vec<[f64; 5]> {
    let mut out = k.base.clone();
    let n = k.n;
    let scale = 2.0 / n as f64;
    for (i, &hz) in grid.iter().enumerate() {
        // The same criterion `control_points` uses. Adjacent sixth-octave bands'
        // edges *meet*, so a `hi <= inject_lo` test is a floating-point tie and
        // decides by one ulp; the centre is unambiguous.
        if hz < INJECT_CENTRES.0 * 0.99 || hz > INJECT_CENTRES.1 * 1.01 {
            continue;
        }
        let (lo, hi) = band_edges(hz);
        let (jlo, jhi) = (bin_of(lo, n).max(1), bin_of(hi, n).min(n / 2));
        assert!(
            jlo >= k.blo && jhi <= k.blo + k.bins.len() - 1,
            "band {hz:.0} Hz runs outside the stored bins — the mask and the bands disagree"
        );
        let mut acc = [0.0f64; 5];
        for j in jlo..=jhi {
            let (jj, a, b, x) = k.bins[j - k.blo];
            debug_assert_eq!(jj, j);
            let s = x * gain[j];
            let (p, q) = (a + s, b - s);
            acc[0] += p.norm_sqr();
            acc[1] += q.norm_sqr();
            acc[2] += ((p + q) * 0.5).norm_sqr();
            acc[3] += ((p - q) * 0.5).norm_sqr();
            acc[4] += (p * q.conj()).re;
        }
        for v in &mut acc {
            *v *= scale;
        }
        out[i] = acc;
    }
    out
}

/// Pooled, level-matched pair-over-mono of one set of takes, per band.
fn pooled_pair_over_mono(bands: &[Vec<[f64; 5]>], totals: &[f64], i: usize) -> f64 {
    let (mut p, mut m) = (0.0, 0.0);
    for (b, &t) in bands.iter().zip(totals) {
        p += (b[i][0] + b[i][1]) / t;
        m += 2.0 * b[i][2] / t;
    }
    db(p / m)
}

/// Pooled, level-matched **pair share**: this band's pair energy as a fraction
/// of the take's own 100-810 Hz pair energy. The absolute-loudness ledger.
fn pooled_pair_share(bands: &[Vec<[f64; 5]>], totals: &[f64], i: usize) -> f64 {
    let (mut p, mut t) = (0.0, 0.0);
    for (b, &tot) in bands.iter().zip(totals) {
        p += (b[i][0] + b[i][1]) / tot;
        t += 1.0;
    }
    db(p / t)
}

fn pooled_mono_share(bands: &[Vec<[f64; 5]>], totals: &[f64], i: usize) -> f64 {
    let (mut p, mut t) = (0.0, 0.0);
    for (b, &tot) in bands.iter().zip(totals) {
        p += b[i][2] / tot;
        t += 1.0;
    }
    db(p / t)
}

fn pooled_r0(bands: &[Vec<[f64; 5]>], totals: &[f64], i: usize) -> f64 {
    let (mut c, mut l, mut r) = (0.0, 0.0, 0.0);
    for (b, &tot) in bands.iter().zip(totals) {
        c += b[i][4] / tot;
        l += b[i][0] / tot;
        r += b[i][1] / tot;
    }
    c / (l * r).sqrt()
}

// ---------------------------------------------------------------------------
// The fit
// ---------------------------------------------------------------------------

struct Fit {
    curve: Curve,
    /// Per-round peak |treated − target| over the fitted bands, dB.
    history: Vec<f64>,
    /// The recording's own pair-over-mono in the fitted bands: the target.
    target: Vec<f64>,
    /// What the fitted curve actually delivers there.
    achieved: Vec<f64>,
    /// The pair energy the band needs, and what a unit-energy weight buys.
    needed: Vec<f64>,
}

/// Gaussian elimination with partial pivoting on a small dense system.
fn solve(mut m: Vec<Vec<f64>>, mut r: Vec<f64>) -> Option<Vec<f64>> {
    let n = r.len();
    for col in 0..n {
        let piv = (col..n).max_by(|&i, &j| m[i][col].abs().total_cmp(&m[j][col].abs()))?;
        if m[piv][col].abs() < 1e-30 {
            return None;
        }
        m.swap(col, piv);
        r.swap(col, piv);
        for row in (col + 1)..n {
            let f = m[row][col] / m[col][col];
            for c in col..n {
                m[row][c] -= f * m[col][c];
            }
            r[row] -= f * r[col];
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let mut acc = r[row];
        for c in (row + 1)..n {
            acc -= m[row][c] * x[c];
        }
        x[row] = acc / m[row][row];
    }
    Some(x)
}

/// **The fit.** Newton with a measured Jacobian on the pooled pair energy, the
/// arrangement `DECISIONS.md` 412's response design uses, in the energy
/// parameterisation [`Curve`] explains.
fn fit(keys: &[Key], grid: &[f64], cp: &[usize]) -> Fit {
    let mono_tot: Vec<f64> = keys.iter().map(|k| k.base_mono_total).collect();
    let ref_tot: Vec<f64> = keys.iter().map(|k| k.ref_mono_total).collect();
    let ref_bands: Vec<Vec<[f64; 5]>> = keys.iter().map(|k| k.reference.clone()).collect();
    let target: Vec<f64> = cp
        .iter()
        .map(|&i| pooled_pair_over_mono(&ref_bands, &ref_tot, i))
        .collect();

    // Pooled base pair sum and pooled mono sum per fitted band; both are level
    // matched on each key's own 100-810 Hz mono total, which the injection
    // cannot move.
    let pooled_pair = |bands: &[Vec<[f64; 5]>], i: usize| -> f64 {
        bands
            .iter()
            .zip(&mono_tot)
            .map(|(b, &t)| (b[i][0] + b[i][1]) / t)
            .sum()
    };
    let base_bands: Vec<Vec<[f64; 5]>> = keys.iter().map(|k| k.base.clone()).collect();
    let mut needed = Vec::new();
    let mut p0 = Vec::new();
    for (c, &i) in cp.iter().enumerate() {
        let m: f64 = keys
            .iter()
            .map(|k| 2.0 * k.base[i][2] / k.base_mono_total)
            .sum();
        let base = pooled_pair(&base_bands, i);
        p0.push(base);
        needed.push(m * 10f64.powf(target[c] / 10.0) - base);
    }

    let curve0 = Curve::new(&cp.iter().map(|&i| grid[i]).collect::<Vec<_>>());
    let nbins = keys[0].n;
    let measure = |u: &[f64]| -> Vec<f64> {
        let c = curve0.with(u);
        let g = realised_gain(&c, nbins);
        let t: Vec<Vec<[f64; 5]>> = keys.par_iter().map(|k| treated_bands(k, grid, &g)).collect();
        cp.iter().map(|&i| pooled_pair(&t, i) - p0[cp.iter().position(|&x| x == i).unwrap()])
            .collect()
    };

    // Round zero: one control point at a time, to size each one.
    let mut u = vec![0.0; cp.len()];
    for c in 0..cp.len() {
        let mut probe = vec![0.0; cp.len()];
        probe[c] = 1.0;
        let s = measure(&probe);
        u[c] = if s[c] > 0.0 {
            (needed[c] / s[c]).max(0.0)
        } else {
            0.0
        };
    }

    // The Jacobian, by finite difference around that point.
    let s0 = measure(&u);
    let mut jac = vec![vec![0.0; cp.len()]; cp.len()];
    for c in 0..cp.len() {
        let d = (u[c] * 0.25).max(1e-6);
        let mut probe = u.clone();
        probe[c] += d;
        let s = measure(&probe);
        for i in 0..cp.len() {
            jac[i][c] = (s[i] - s0[i]) / d;
        }
    }

    // Newton with a fixed Jacobian and adaptive damping. The system is not
    // quite linear — the source is partly coherent with the pair, so the cross
    // term goes as sqrt(u) — and an undamped step limit-cycles at half a
    // decibel; halving the step whenever a round does not improve takes it to
    // hundredths.
    let mut history = Vec::new();
    let mut best = u.clone();
    let mut best_err = f64::INFINITY;
    let mut damping = 1.0f64;
    for _ in 0..24 {
        let s = measure(&u);
        let mut worst = 0.0f64;
        for c in 0..cp.len() {
            let got =
                db((p0[c] + s[c]).max(1e-300)) - db((p0[c] + needed[c]).max(1e-300));
            worst = worst.max(got.abs());
        }
        history.push(worst);
        let from = if worst < best_err {
            best_err = worst;
            best = u.clone();
            s
        } else {
            damping *= 0.5;
            u = best.clone();
            measure(&u)
        };
        let r: Vec<f64> = (0..cp.len()).map(|i| needed[i] - from[i]).collect();
        let Some(step) = solve(jac.clone(), r) else {
            break;
        };
        for c in 0..cp.len() {
            u[c] = (u[c] + damping * step[c]).max(0.0);
        }
    }

    let curve = curve0.with(&best);
    let g = realised_gain(&curve, nbins);
    let treated: Vec<Vec<[f64; 5]>> =
        keys.par_iter().map(|k| treated_bands(k, grid, &g)).collect();
    let achieved = cp
        .iter()
        .map(|&i| pooled_pair_over_mono(&treated, &mono_tot, i))
        .collect();
    Fit {
        curve,
        history,
        target,
        achieved,
        needed,
    }
}

// ---------------------------------------------------------------------------
// Treated audio, for the gates' own statistics
// ---------------------------------------------------------------------------

/// The **minimum-phase** realisation of a magnitude response, by the real
/// cepstrum — the same construction `soundboard::Radiation`'s design assumes
/// and the same one `mono_mechanism` uses.
///
/// It matters here for two of the measurements and for nothing else: a
/// zero-phase band-pass rings symmetrically about `t = 0`, which would put
/// injected energy *before* the strike and make both §1's first-10-ms reading
/// and §5d's lag structure fiction. The magnitude is untouched, so every band
/// energy the fit was closed on is unchanged.
fn min_phase(mag: &[f64], planner: &mut FftPlanner<f64>) -> Vec<C64> {
    let n = mag.len();
    let peak = mag.iter().cloned().fold(0.0f64, f64::max).max(1e-300);
    let mut c: Vec<C64> = mag
        .iter()
        .map(|&m| C64::new((m / peak).max(1e-3).ln(), 0.0))
        .collect();
    planner.plan_fft_inverse(n).process(&mut c);
    let s = 1.0 / n as f64;
    let mut q = vec![C64::new(0.0, 0.0); n];
    q[0] = c[0] * s;
    for j in 1..n / 2 {
        q[j] = c[j] * (2.0 * s);
    }
    q[n / 2] = c[n / 2] * s;
    planner.plan_fft_forward(n).process(&mut q);
    q.iter().map(|v| (*v).exp() * peak).collect()
}

/// The realised complex gain of the injection filter over `n` bins: the
/// **minimum-phase** completion of the curve's magnitude.
fn realised_gain(curve: &Curve, n: usize) -> Vec<C64> {
    let mut planner = FftPlanner::<f64>::new();
    let mag: Vec<f64> = (0..n).map(|j| curve.amp_at(bin_hz(j, n))).collect();
    min_phase(&mag, &mut planner)
}

/// The treated pair in the time domain, plus the mono-discipline residual.
///
/// `L' = L + s`, `R' = R − s` with `s` the fitted source. The claim under test
/// is that `(L' + R')/2 = (L + R)/2` **identically**, so the residual is
/// returned rather than argued: once in the arithmetic the offline stage runs
/// in (`f64`), and once after both channels are rounded to `f32`, which is what
/// the engine would actually write.
fn treated_audio(k: &Key, gain: &[C64]) -> (Audio, Vec<f64>, f64, f64, f64) {
    let n = k.n;
    let len = k.base_l.len();
    let mut planner = FftPlanner::<f64>::new();
    let xs: Vec<C64> = k.xspec.iter().zip(gain).map(|(x, h)| x * h).collect();
    let s = inverse(xs, &mut planner);
    let mut peak = 0.0f64;
    let mut worst64 = 0.0f64;
    let mut worst32 = 0.0f64;
    let (mut lo, mut ro) = (Vec::with_capacity(len), Vec::with_capacity(len));
    for i in 0..len {
        let (l, r) = (f64::from(k.base_l[i]), f64::from(k.base_r[i]));
        let m0 = 0.5 * (l + r);
        let (lt, rt) = (l + s[i], r - s[i]);
        worst64 = worst64.max((0.5 * (lt + rt) - m0).abs());
        let (l32, r32) = (lt as f32, rt as f32);
        let m32 = 0.5 * (f64::from(l32) + f64::from(r32));
        worst32 = worst32.max((m32 - m0).abs());
        peak = peak.max(m0.abs()).max(lt.abs()).max(rt.abs());
        lo.push(l32);
        ro.push(r32);
    }
    (
        Audio::new(SAMPLE_RATE, vec![lo, ro]).expect("two channels"),
        s[..len].to_vec(),
        worst64,
        worst32,
        peak,
    )
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

fn stereo_row(c: &realism::StereoColumn) -> String {
    format!(
        "{:+.3} (bar {:.3}){}",
        c.engine_r0,
        c.bar,
        if c.pass { "" } else { " **RED**" }
    )
}

fn run(source: Source, keys: &[Key], grid: &[f64], cp: &[usize]) {
    println!("\n\n# Source `{}`\n", source.name());

    // ----- §1: what the source is, before any level is fitted -----
    println!("## 1. The injected signal, at unit weight\n");
    println!(
        "`headroom` is the pair energy the band would gain at `u = 1` against the base engine's \
own pair energy there — how hard the fitted weight has to work. `coh` is |cos| of the angle \
between the source and the pair's **existing** side signal, pooled over the keys: zero means \
the injection adds energy and nothing else, one means it is the same signal again and the \
addition is coherent. The fit does not assume either.\n"
    );
    println!("| Hz | source pair headroom dB | coh with existing side |");
    println!("|---:|---:|---:|");
    let unit = Curve::new(&cp.iter().map(|&i| grid[i]).collect::<Vec<_>>())
        .with(&vec![1.0; cp.len()]);
    for &i in cp {
        let (lo, hi) = band_edges(grid[i]);
        let (mut xe, mut se, mut cross, mut base) = (0.0, 0.0, 0.0, 0.0);
        for k in keys {
            let n = k.n;
            let (jlo, jhi) = (bin_of(lo, n).max(1), bin_of(hi, n).min(n / 2));
            let t = k.base_mono_total * n as f64 / 2.0;
            for j in jlo..=jhi {
                let (_, a, b, x) = k.bins[j - k.blo];
                let g = unit.amp_at(j as f64 * SR / n as f64);
                let side = (a - b) * 0.5;
                xe += (x * g).norm_sqr() / t;
                se += side.norm_sqr() / t;
                cross += ((x * g) * side.conj()).re / t;
                base += (a.norm_sqr() + b.norm_sqr()) / t;
            }
        }
        println!(
            "| {:.0} | {:+.2} | {:.3} |",
            grid[i],
            db(2.0 * xe / base),
            (cross / (xe * se).sqrt()).abs()
        );
    }

    let f = fit(keys, grid, cp);
    println!("\n## 2. The fitted per-sixth-octave injection level\n");
    println!(
        "| Hz | energy weight u | amplitude sqrt(u) | dB | target pair/mono | achieved | miss |"
    );
    println!("|---:|---:|---:|---:|---:|---:|---:|");
    for (c, &i) in cp.iter().enumerate() {
        let u = f.curve.nodes[c].1;
        println!(
            "| {:.0} | {u:.5} | {:.4} | {:+.2} | {:+.2} | {:+.2} | {:+.3} |",
            grid[i],
            u.max(0.0).sqrt(),
            10.0 * u.max(1e-12).log10(),
            f.target[c],
            f.achieved[c],
            f.achieved[c] - f.target[c],
        );
    }
    let _ = &f.needed;
    println!(
        "\nFit convergence, peak |treated − target| over the {} fitted bands per Newton round: {}",
        cp.len(),
        f.history
            .iter()
            .map(|x| format!("{x:.3}"))
            .collect::<Vec<_>>()
            .join(" → ")
    );

    // --- treated bands, all keys ---
    //
    // Two ways, deliberately: the fit's own bin-domain arithmetic, and the
    // **rendered** treated pair put back through `band_energies`. Every table
    // below is the rendered one; the two are printed against each other in §3
    // because an offline stage that is not what it claims to render is the one
    // failure mode this whole methodology has (`mono_mechanism`'s own
    // validation row, 0.1 us / 0.001 dB).
    let gain = realised_gain(&f.curve, keys[0].n);
    let treated_fit: Vec<Vec<[f64; 5]>> = keys
        .par_iter()
        .map(|k| treated_bands(k, grid, &gain))
        .collect();
    let rendered: Vec<(Audio, f64, f64, f64)> = keys
        .par_iter()
        .map(|k| {
            let (a, _, w64, w32, peak) = treated_audio(k, &gain);
            (a, w64, w32, peak)
        })
        .collect();
    let treated: Vec<Vec<[f64; 5]>> = rendered
        .par_iter()
        .zip(keys)
        .map(|((a, ..), k)| {
            let n = k.n;
            let mut planner = FftPlanner::<f64>::new();
            let la = forward(&a.channels[0], n, &mut planner);
            let rb = forward(&a.channels[1], n, &mut planner);
            grid.iter()
                .map(|&hz| {
                    let (lo, hi) = band_edges(hz);
                    band_energies(&la, &rb, lo, hi)
                })
                .collect()
        })
        .collect();
    let base_bands: Vec<Vec<[f64; 5]>> = keys.iter().map(|k| k.base.clone()).collect();
    let ship_bands: Vec<Vec<[f64; 5]>> = keys.iter().map(|k| k.shipped.clone()).collect();
    let ref_bands: Vec<Vec<[f64; 5]>> = keys.iter().map(|k| k.reference.clone()).collect();
    let mono_tot: Vec<f64> = keys.iter().map(|k| k.base_mono_total).collect();
    let ship_tot: Vec<f64> = keys.iter().map(|k| k.shipped_mono_total).collect();
    let ref_tot: Vec<f64> = keys.iter().map(|k| k.ref_mono_total).collect();
    let base_ptot: Vec<f64> = keys.iter().map(|k| k.base_pair_total).collect();
    let ref_ptot: Vec<f64> = keys.iter().map(|k| k.ref_pair_total).collect();
    let treat_ptot: Vec<f64> = treated
        .iter()
        .map(|b| b.iter().map(|e| e[0] + e[1]).sum())
        .collect();

    // --- §3: the mono fold-down ---
    println!("\n## 3. The mono fold-down, asserted rather than argued\n");
    let mut worst_band = 0.0f64;
    for i in 0..grid.len() {
        for (t, b) in treated.iter().zip(keys.iter().map(|k| &k.base)) {
            let d = db(t[i][2]) - db(b[i][2]);
            if d.is_finite() {
                worst_band = worst_band.max(d.abs());
            }
        }
    }
    if std::env::var("SI_DEBUG").is_ok() {
        let zero = Curve::new(&cp.iter().map(|&i| grid[i]).collect::<Vec<_>>());
        for k in keys.iter().take(3) {
            let (a, _, _, _, _) = treated_audio(k, &realised_gain(&zero, k.n));
            let n = k.n;
            let mut pl = FftPlanner::<f64>::new();
            let la = forward(&a.channels[0], n, &mut pl);
            let rb = forward(&a.channels[1], n, &mut pl);
            let ba = forward(&k.base_l, n, &mut pl);
            let bb = forward(&k.base_r, n, &mut pl);
            let mut w = 0.0f64;
            let mut wi = 0;
            for (i, &hz) in grid.iter().enumerate() {
                let (lo, hi) = band_edges(hz);
                let e1 = band_energies(&la, &rb, lo, hi);
                let e0 = band_energies(&ba, &bb, lo, hi);
                let d = (db(e1[0]) - db(e0[0])).abs();
                if d > w { w = d; wi = i; }
            }
            eprintln!("DEBUG zero-curve {} worst {:.4} dB at {:.0} Hz", k.label, w, grid[wi]);
        }
    }
    let w64 = rendered.iter().map(|r| r.1).fold(0.0f64, f64::max);
    let w32 = rendered.iter().map(|r| r.2).fold(0.0f64, f64::max);
    let pk = rendered.iter().map(|r| r.3).fold(0.0f64, f64::max);
    // Readable bands only: a band 60 dB under the key's own loudest is the
    // `f32` write's own quantisation floor, and the injection raises that floor
    // for every band because it raises the peak.
    let mut offline_vs_rendered = 0.0f64;
    let mut ovr_where = (0usize, 0usize);
    let mut ovr_all: Vec<f64> = Vec::new();
    for (kx, (t, o)) in treated.iter().zip(&treated_fit).enumerate() {
        let peak = t.iter().map(|e| e[0] + e[1]).fold(0.0f64, f64::max);
        for i in 0..grid.len() {
            if t[i][0] + t[i][1] < peak * 1e-4 {
                continue;
            }
            for c in [0usize, 1] {
                let d = db(t[i][c]) - db(o[i][c]);
                if d.is_finite() {
                    ovr_all.push(d.abs());
                    if d.abs() > offline_vs_rendered {
                        offline_vs_rendered = d.abs();
                        ovr_where = (kx, i);
                    }
                }
            }
        }
    }
    let ovr_median = median(&mut ovr_all);
    println!(
        "* Worst per-band mono energy change over 30 keys x 19 bands: **{worst_band:.2e} dB**.\n\
* Worst per-sample `|(L'+R')/2 − (L+R)/2|` in the stage's own `f64` arithmetic: \
**{:.1} dBFS re the render's peak** ({w64:.3e} absolute).\n\
* The same after both channels are rounded to `f32`, which is what the engine writes: \
**{:.1} dBFS re peak** ({w32:.3e} absolute). The repository's own mono-discipline bound is \
−116 dB (`DECISIONS.md` 412).\n\
* **Offline vs rendered**: worst per-key per-band disagreement between the fit's own bin-domain \
arithmetic and the same stage measured off the rendered treated pair: **{offline_vs_rendered:.4} dB** \
(at {ovr_key} {ovr_hz:.0} Hz; median over keys and readable bands {ovr_med:.4} dB, readable being within 40 dB of the key's own loudest).",
        f64_db(w64 / pk),
        f64_db(w32 / pk),
        ovr_key = keys[ovr_where.0].label,
        ovr_hz = grid[ovr_where.1],
        ovr_med = ovr_median,
    );
    // The bound is the `f32` write and nothing else: the stage's own arithmetic
    // leaves the fold-down at −335 dBFS, and what is left is the two channels
    // being rounded to `f32` before the sum is taken, which is the same
    // rounding every render in this repository already carries (item 412's
    // −116 dB). A tighter bound here would be asserting that `f32` is exact.
    assert!(
        worst_band < 1e-4,
        "a side-only injection moved the mono fold-down beyond the f32 write: {worst_band:.3e} dB"
    );

    // --- §4: the sixth-octave table ---
    println!("\n## 4. Per sixth-octave: recording, untreated engine, treated engine\n");
    println!(
        "`pair/mono` is `10log10((E_L+E_R)/2E_M)` pooled and level-matched. `pair share` is the \
band's pair energy as a fraction of the take's own 100-810 Hz pair energy — the loudness \
column, which no item from 405 to 416 printed. `mono share` likewise on the fold-down. \
`inj` marks a fitted band.\n"
    );
    println!(
        "| Hz | inj | REF pair/mono | ENG ship | ENG bare | **TREATED** | REF r0 | ship r0 | \
bare r0 | **treated r0** | REF pair share | bare pair share | **treated pair share** | \
**treated − REF** | REF mono share | bare mono share |"
    );
    println!("|---:|:--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
    let mut loudness: Vec<(f64, f64)> = Vec::new();
    for (i, &hz) in grid.iter().enumerate() {
        let inj = cp.contains(&i);
        let rp = pooled_pair_over_mono(&ref_bands, &ref_tot, i);
        let sp = pooled_pair_over_mono(&ship_bands, &ship_tot, i);
        let bp = pooled_pair_over_mono(&base_bands, &mono_tot, i);
        let tp = pooled_pair_over_mono(&treated, &mono_tot, i);
        let rs = pooled_pair_share(&ref_bands, &ref_ptot, i);
        let bs = pooled_pair_share(&base_bands, &base_ptot, i);
        let ts = pooled_pair_share(&treated, &treat_ptot, i);
        let rm = pooled_mono_share(&ref_bands, &ref_tot, i);
        let bm = pooled_mono_share(&base_bands, &mono_tot, i);
        loudness.push((hz, ts - rs));
        println!(
            "| {hz:.0} | {} | {rp:+.2} | {sp:+.2} | {bp:+.2} | **{tp:+.2}** | {:+.3} | {:+.3} | \
{:+.3} | **{:+.3}** | {rs:+.2} | {bs:+.2} | **{ts:+.2}** | **{:+.2}** | {rm:+.2} | {bm:+.2} |",
            if inj { "x" } else { "" },
            pooled_r0(&ref_bands, &ref_tot, i),
            pooled_r0(&ship_bands, &ship_tot, i),
            pooled_r0(&base_bands, &mono_tot, i),
            pooled_r0(&treated, &mono_tot, i),
            ts - rs
        );
    }
    let worst_loud = loudness
        .iter()
        .filter(|(hz, _)| (inject_hz().0..=inject_hz().1).contains(hz))
        .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
        .copied()
        .unwrap_or((0.0, 0.0));
    println!(
        "\n**Loudness ledger.** Inside the injected range the treated pair share stands \
**{:+.2} dB** from the recording's at its worst ({:.0} Hz).",
        worst_loud.1, worst_loud.0
    );

    // --- §5: the gates' own statistics ---
    println!("\n## 5. The gates' own statistics, on the treated renders\n");
    let audios: Vec<Audio> = rendered.iter().map(|(a, ..)| a.clone()).collect();
    // **The strike window** (`DECISIONS.md` 379's own measurement, on the
    // injected band). A nodal line radiates from the first sample; a
    // board-derived tap cannot, because the FDN's shortest delay line is 149
    // samples and its difference is exactly zero for 3.1 ms.
    {
        let (ilo, ihi) = inject_hz();
        let win = (0.010 * SR) as usize;
        let ms_db = |l: &[f32], r: &[f32]| -> f64 {
            let n = win.next_power_of_two() * 2;
            let mut planner = FftPlanner::<f64>::new();
            let a = forward(&l[..win], n, &mut planner);
            let b = forward(&r[..win], n, &mut planner);
            let e = band_energies(&a, &b, ilo, ihi);
            db(e[2] / e[3].max(1e-300))
        };
        let mut refv: Vec<f64> = keys
            .iter()
            .map(|k| ms_db(&k.reference_audio.channels[0], &k.reference_audio.channels[1]))
            .collect();
        let mut barev: Vec<f64> = keys.iter().map(|k| ms_db(&k.base_l, &k.base_r)).collect();
        let mut shipv: Vec<f64> = keys
            .iter()
            .map(|k| ms_db(&k.shipped_audio.channels[0], &k.shipped_audio.channels[1]))
            .collect();
        let mut treatv: Vec<f64> = audios
            .iter()
            .map(|a| ms_db(&a.channels[0], &a.channels[1]))
            .collect();
        println!(
            "**The strike window.** Mid-over-side in {:.0}-{:.0} Hz over the first 10 ms, median \
over the keys: recording **{:+.2} dB**, engine bare **{:+.2}**, engine shipped **{:+.2}**, \
treated **{:+.2}**. `DECISIONS.md` 379 refused the first lobe design for reading +9.9 dB here \
against the recording's −1.6.\n",
            ilo,
            ihi,
            median(&mut refv),
            median(&mut barev),
            median(&mut shipv),
            median(&mut treatv),
        );
    }

    let bare_audios: Vec<Audio> = keys
        .par_iter()
        .map(|k| {
            Audio::new(SAMPLE_RATE, vec![k.base_l.clone(), k.base_r.clone()]).expect("two channels")
        })
        .collect();

    let mk_items = |engine: &[Audio]| -> (Vec<StereoItem>, Vec<ChannelItem>) {
        let s: Vec<StereoItem> = keys
            .iter()
            .zip(engine)
            .map(|(k, a)| StereoItem {
                label: k.label.clone(),
                engine: realism::stereo_image_of(a).expect("two channels"),
                reference: realism::stereo_image_of(&k.reference_audio).expect("two channels"),
                alternate: realism::stereo_image_of(&k.alternate_audio).expect("two channels"),
            })
            .collect();
        let c: Vec<ChannelItem> = keys
            .iter()
            .zip(engine)
            .map(|(k, a)| ChannelItem {
                label: k.label.clone(),
                engine: realism::channel_shape_of(a).expect("two channels"),
                reference: realism::channel_shape_of(&k.reference_audio).expect("two channels"),
                alternate: realism::channel_shape_of(&k.alternate_audio).expect("two channels"),
            })
            .collect();
        (s, c)
    };
    let ship_audios: Vec<Audio> = keys.iter().map(|k| k.shipped_audio.clone()).collect();
    let (ship_s, ship_c) = mk_items(&ship_audios);
    let (bare_s, bare_c) = mk_items(&bare_audios);
    let (treat_s, treat_c) = mk_items(&audios);

    println!("### 5a. `realism::stereo_columns` — the coherence board's `r0`\n");
    println!("| band | REF r0 | ENG shipped | ENG bare | **TREATED** | bar |");
    println!("|---|---:|---:|---:|---:|---:|");
    let sc_s = realism::stereo_columns(&ship_s);
    let sc_b = realism::stereo_columns(&bare_s);
    let sc_t = realism::stereo_columns(&treat_s);
    for i in 0..sc_t.len() {
        println!(
            "| {} | {:+.3} | {} | {} | {} | {:.3} |",
            sc_t[i].name,
            sc_t[i].reference_r0,
            stereo_row(&sc_s[i]),
            stereo_row(&sc_b[i]),
            stereo_row(&sc_t[i]),
            sc_t[i].bar
        );
    }

    println!("\n### 5b. `realism::channel_columns` — the third red's own board\n");
    println!(
        "| band | take | dev_L | dev_R | error | bar | pass | pair_bal | pair_bar | mono_bal | \
mono_pooled | per_key | per_key_bar |"
    );
    println!("|---|---|---:|---:|---:|---:|:--:|---:|---:|---:|---:|---:|---:|");
    let cc_s = realism::channel_columns(&ship_c);
    let cc_b = realism::channel_columns(&bare_c);
    let cc_t = realism::channel_columns(&treat_c);
    for i in 0..cc_t.len() {
        if !(cc_t[i].lo_hz >= 63.0 && cc_t[i].hi_hz <= 2_000.0) {
            continue;
        }
        for (tag, c) in [("shipped", &cc_s[i]), ("bare", &cc_b[i]), ("TREATED", &cc_t[i])] {
            println!(
                "| {} | {tag} | {:+.2} | {:+.2} | {:.2} | {:.2} | {} | {:+.2} | {:.2} | {:+.2} | \
{:+.2} | {:.2} | {:.2} |",
                c.name,
                c.engine_left_db,
                c.engine_right_db,
                c.error,
                c.bar,
                if c.pass { "yes" } else { "**RED**" },
                c.pair_balance,
                c.pair_bar,
                c.mono_balance,
                c.mono_pooled,
                c.per_key_error,
                c.per_key_bar,
            );
        }
        println!(
            "| {} | *recording* | {:+.2} | {:+.2} | . | . | . | . | . | . | . | . | . |",
            cc_t[i].name, cc_t[i].reference_left_db, cc_t[i].reference_right_db
        );
    }

    println!("\n### 5c. The same board at a sixth of an octave, over the injected range\n");
    let fc_b = realism::channel_fine_columns(&bare_c);
    let fc_t = realism::channel_fine_columns(&treat_c);
    println!(
        "| Hz | REF L | REF R | bare L | bare R | **treated L** | **treated R** | error bare | \
error treated | bar | pair_bal treated | mono_bal treated |"
    );
    println!("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
    for i in 0..fc_t.len() {
        let c = &fc_t[i];
        if c.hi_hz < 120.0 || c.lo_hz > 520.0 {
            continue;
        }
        println!(
            "| {} | {:+.2} | {:+.2} | {:+.2} | {:+.2} | **{:+.2}** | **{:+.2}** | {:.2} | \
**{:.2}** | {:.2} | {:+.2} | {:+.2} |",
            c.name,
            c.reference_left_db,
            c.reference_right_db,
            fc_b[i].engine_left_db,
            fc_b[i].engine_right_db,
            c.engine_left_db,
            c.engine_right_db,
            fc_b[i].error,
            c.error,
            c.bar,
            c.pair_balance,
            c.mono_balance,
        );
    }

    println!("\n### 5d. Peak-|r| lag structure (`StereoBand::lag_ms`), median over the keys\n");
    println!("| band | REF lag ms | REF peak_r | ship lag | bare lag | **treated lag** | **treated peak_r** |");
    println!("|---|---:|---:|---:|---:|---:|---:|");
    for b in 0..realism::STEREO_BANDS.len() {
        let pick = |items: &[StereoItem], f: fn(&realism::StereoBand) -> f64, eng: bool| {
            let mut v: Vec<f64> = items
                .iter()
                .map(|it| {
                    let img = if eng { &it.engine } else { &it.reference };
                    f(&img.bands[b])
                })
                .collect();
            median(&mut v)
        };
        println!(
            "| {} | {:+.3} | {:+.3} | {:+.3} | {:+.3} | **{:+.3}** | **{:+.3}** |",
            realism::STEREO_BANDS[b].0,
            pick(&treat_s, |x| x.lag_ms, false),
            pick(&treat_s, |x| x.peak_r, false),
            pick(&ship_s, |x| x.lag_ms, true),
            pick(&bare_s, |x| x.lag_ms, true),
            pick(&treat_s, |x| x.lag_ms, true),
            pick(&treat_s, |x| x.peak_r, true),
        );
    }

    // --- §6: the asymmetry the injection cannot reach ---
    println!(
        "\n### 5f. **The L/R asymmetry a symmetric side source cannot touch**\n\n\
`spread` is `dev_L − dev_R` — how differently the two loudspeakers depart from the take's own \
mono shape. A side-only injection adds `+s` to one channel and `−s` to the other, so it adds \
the *same* energy `|s|²` to both and can only move the spread through the cross term with what \
is already there. This column is therefore the part of the third red the hypothesis cannot \
address even in principle, measured rather than argued.\n"
    );
    println!(
        "| Hz | REF spread | bare spread | **treated spread** | REF − treated | \\|REF−treated\\| / \\|REF−bare\\| |"
    );
    println!("|---:|---:|---:|---:|---:|---:|");
    let mut left: Vec<f64> = Vec::new();
    for i in 0..fc_t.len() {
        let c = &fc_t[i];
        if c.hi_hz < 150.0 || c.lo_hz > 340.0 {
            continue;
        }
        let rs = c.reference_left_db - c.reference_right_db;
        let bs = fc_b[i].engine_left_db - fc_b[i].engine_right_db;
        let ts = c.engine_left_db - c.engine_right_db;
        left.push((rs - ts).abs());
        println!(
            "| {} | {rs:+.2} | {bs:+.2} | **{ts:+.2}** | {:+.2} | {:.0} % |",
            c.name,
            rs - ts,
            100.0 * (rs - ts).abs() / (rs - bs).abs().max(1e-9)
        );
    }
    println!(
        "\nMedian |REF − treated| spread over the injected bands: **{:.2} dB**.",
        median(&mut left)
    );

    println!("\n### 5e. Nothing outside the injected range moved\n");
    let mut worst_outside = 0.0f64;
    let mut where_outside = 0.0f64;
    for (i, &hz) in grid.iter().enumerate() {
        if hz >= INJECT_CENTRES.0 * 0.99 && hz <= INJECT_CENTRES.1 * 1.01 {
            continue;
        }
        let d = pooled_pair_over_mono(&treated, &mono_tot, i)
            - pooled_pair_over_mono(&base_bands, &mono_tot, i);
        if d.abs() > worst_outside {
            worst_outside = d.abs();
            where_outside = hz;
        }
    }
    println!(
        "Worst pair-over-mono change in an un-injected sixth-octave band: **{worst_outside:.2e} dB** \
at {where_outside:.0} Hz. The source is exactly zero outside {:.0}-{:.0} Hz, so this is the \
arithmetic saying so.",
        inject_hz().0, inject_hz().1
    );
    for i in 0..sc_t.len() {
        if sc_t[i].hi_hz <= 125.0 || sc_t[i].lo_hz >= 500.0 {
            println!(
                "* {}: bare r0 {:+.3} → treated {:+.3} (bar {:.3})",
                sc_t[i].name, sc_b[i].engine_r0, sc_t[i].engine_r0, sc_t[i].bar
            );
        }
    }

    // --- §6: the acceptance items 408/411 wrote, on the treated instrument ---
    println!(
        "\n## 6. `DECISIONS.md` 408's own table, reproduced and then re-read on the treated \
instrument\n\n\
`required = REF pair/mono − ENG(bare) pair/mono` and `standing = ENG(bare) mono share − REF \
mono share` are item 408's two columns, and item 411's acceptance for this whole track is \
**standing rising to meet required**. `treated standing` is the same column after the \
injection.\n"
    );
    println!("| Hz | required | standing | **treated standing** | moved by |");
    println!("|---:|---:|---:|---:|---:|");
    for (i, &hz) in grid.iter().enumerate() {
        let req = pooled_pair_over_mono(&ref_bands, &ref_tot, i)
            - pooled_pair_over_mono(&base_bands, &mono_tot, i);
        let standing = pooled_mono_share(&base_bands, &mono_tot, i)
            - pooled_mono_share(&ref_bands, &ref_tot, i);
        let treated_standing = pooled_mono_share(&treated, &mono_tot, i)
            - pooled_mono_share(&ref_bands, &ref_tot, i);
        println!(
            "| {hz:.0} | {req:+.2} | {standing:+.2} | **{treated_standing:+.2}** | {:+.2e} |",
            treated_standing - standing
        );
    }
}

// ---------------------------------------------------------------------------

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arg = std::env::args().nth(1).unwrap_or_else(|| "all".to_string());
    let sources: Vec<Source> = match arg.as_str() {
        "fdn" => vec![Source::Fdn],
        "board-ap" => vec![Source::BoardAllpass],
        "mid-ap" => vec![Source::MidAllpass],
        _ => vec![Source::Fdn, Source::BoardAllpass, Source::MidAllpass],
    };
    let grid = grid();
    let cp = control_points(&grid);
    let preset = shipped();
    let bare_p = bare(&preset);
    let mut board_p = bare_p.clone();
    board_p.soundboard.board_mix = 1.0;

    let library = SampleLibrary::from_sfz(Path::new(SFZ))?;
    let recorded = RecordedKeys::from_library(&library)?;
    let layers = VelocityLayers::from_library(&library)?;
    let alt_vel = layers.alternate(VELOCITY);

    println!("# side_injection — an antisymmetric radiator, fitted and measured\n");
    println!(
        "{} recorded keys at v{VELOCITY}, {RENDER_S} s from the strike; the recording's second \
layer is v{alt_vel}, which is what every bar below is made of. Injected range \
**{:.1}-{:.1} Hz** ({} sixth-octave control points at {}). Base instrument: the shipped preset \
with `[voicing.mics.modal]` deleted — the injection **replaces** the lobe rather than stacking \
on it.\n",
        recorded.keys().len(),
        inject_hz().0,
        inject_hz().1,
        cp.len(),
        cp.iter()
            .map(|&i| format!("{:.0}", grid[i]))
            .collect::<Vec<_>>()
            .join(", ")
    );

    for source in sources {
        let keys: Vec<Key> = recorded
            .keys()
            .par_iter()
            .map(|&k| build_key(k, &grid, &preset, &bare_p, &board_p, alt_vel, source))
            .collect::<Result<Vec<_>, _>>()?;
        run(source, &keys, &grid, &cp);
    }
    Ok(())
}
