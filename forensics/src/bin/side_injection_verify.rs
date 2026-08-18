//! **Adversarial verification of `side_injection`** — independent re-measurement
//! of its load-bearing numbers, plus the two checks the probe did not run:
//! the injected source's coherence with the **mid** (the mono-smuggling
//! question), and a **held-out velocity layer** evaluation of the v90-fitted
//! levels (the circularity question).
//!
//! ```text
//! cargo run --release -p forensics --bin side_injection_verify
//! ```
//!
//! The fitted energy weights are hardcoded from the probe's own report
//! (`renders/side-injection/SIDE_INJECTION.md` §2, source `fdn`), so this
//! binary re-measures the *treatment the report describes* rather than
//! re-running the fit that produced it.

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
const SPAN_HZ: (f64, f64) = (100.0, 810.0);
const INJECT_CENTRES: (f64, f64) = (160.0, 320.0);

/// The probe's fitted energy weights for source `fdn`, §2 of its report.
/// Centres are the exact grid values 160·2^(k/6), not the report's rounded
/// labels — a rounded 180.0 shifts the brick edges by 0.23 % and moves
/// 0.14 dB at 202 Hz.
const FITTED_W: [f64; 7] = [
    0.06105, 9.15743, 4.21302, 0.67987, 2.11107, 0.18804, 0.13863,
];

fn fitted_nodes() -> Vec<(f64, f64)> {
    (0..7)
        .map(|k| (160.0 * 2.0f64.powf(k as f64 / 6.0), FITTED_W[k]))
        .collect()
}

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

fn inject_hz() -> (f64, f64) {
    (
        band_edges(INJECT_CENTRES.0).0,
        band_edges(INJECT_CENTRES.1).1,
    )
}

fn bin_of(hz: f64, n: usize) -> usize {
    (hz * n as f64 / SR).round() as usize
}

fn bin_hz(j: usize, n: usize) -> f64 {
    let half = n / 2;
    if j <= half {
        j as f64 * SR / n as f64
    } else {
        (n - j) as f64 * SR / n as f64
    }
}

fn shipped() -> Preset {
    Preset::load(Path::new("presets/salamander-c5.toml")).expect("the measured preset loads")
}

fn bare(preset: &Preset) -> Preset {
    let mut p = preset.clone();
    if let Some(mics) = preset.voicing.mics {
        p.voicing.mics = Some(MicVoicing { modal: None, ..mics });
    }
    p
}

fn render_key(preset: &Preset, key: u8, velocity: u8) -> (Vec<f32>, Vec<f32>) {
    let events = [RenderEvent::new(
        PREROLL_S as f32,
        Event::NoteOn {
            key,
            vel: u16::from(velocity),
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

/// `[E_L, E_R, E_M, E_S, Re<L,R>]` — written independently, positive
/// frequencies only, scaled x2 (conjugate symmetry), same bin rounding.
fn band5(a: &[C64], b: &[C64], lo: f64, hi: f64) -> [f64; 5] {
    let n = a.len();
    let (blo, bhi) = (bin_of(lo, n).max(1), bin_of(hi, n).min(n / 2));
    let mut acc = [0.0f64; 5];
    for j in blo..=bhi {
        let (x, y) = (a[j], b[j]);
        acc[0] += x.norm_sqr();
        acc[1] += y.norm_sqr();
        acc[2] += ((x + y) * 0.5).norm_sqr();
        acc[3] += ((x - y) * 0.5).norm_sqr();
        acc[4] += (x * y.conj()).re;
    }
    let s = 2.0 / n as f64;
    for v in &mut acc {
        *v *= s;
    }
    acc
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

fn coherence_highpass(b: f64, f: f64) -> C64 {
    let z1 = C64::from_polar(1.0, -TAU * f / SR);
    C64::new(1.0 - b, 0.0) * (C64::new(1.0, 0.0) - z1) / (C64::new(1.0, 0.0) - z1 * (1.0 - b))
}

fn diffuse_b(v: &MicVoicing) -> f64 {
    const MIC_DIFFUSE_POLE_K: f64 = 0.426_63;
    const SPEED_OF_SOUND: f64 = 343.0;
    let hz =
        MIC_DIFFUSE_POLE_K * SPEED_OF_SOUND / f64::from(v.spacing_m) * f64::from(v.diffuse_coherence);
    let w = (TAU * hz / SR).min(std::f64::consts::PI);
    1.0 - (-w).exp()
}

/// The probe's `Curve::energy_at`, reproduced: brick per band with a
/// 24th-octave ramp at each edge.
fn energy_at(f: f64) -> f64 {
    let (lo, hi) = inject_hz();
    if f <= lo || f >= hi {
        return 0.0;
    }
    let nodes = fitted_nodes();
    let brick = |g: f64| -> f64 {
        if g <= lo || g >= hi {
            return 0.0;
        }
        for &(c, u) in &nodes {
            let (l, h) = band_edges(c);
            if g >= l && g < h {
                return u;
            }
        }
        0.0
    };
    let w = 2.0f64.powf(1.0 / 48.0).ln();
    let lf = f.ln();
    let here = brick(f);
    let (below, above) = (brick((lf - w).exp()), brick((lf + w).exp()));
    if below != here {
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

fn realised_gain(n: usize) -> Vec<C64> {
    let mut planner = FftPlanner::<f64>::new();
    let mag: Vec<f64> = (0..n)
        .map(|j| energy_at(bin_hz(j, n)).max(0.0).sqrt())
        .collect();
    min_phase(&mag, &mut planner)
}

/// One key's material at one velocity: bare render, board render (mix 1.0),
/// the recovered fdn source spectrum (masked), and the reference.
struct KeyV {
    label: String,
    bare_l: Vec<f32>,
    bare_r: Vec<f32>,
    xspec: Vec<C64>,
    n: usize,
    reference: Audio,
    alternate: Audio,
}

fn build_key(
    key: u8,
    preset: &Preset,
    bare_p: &Preset,
    board_p: &Preset,
    engine_vel: u8,
    ref_vel: u8,
    alt_vel: u8,
) -> Result<KeyV, piano_tuner::Error> {
    let reference = reference_key(key, ref_vel)?;
    let alternate = reference_key(key, alt_vel)?;
    let (bl, br) = render_key(bare_p, key, engine_vel);
    let (cl, cr) = render_key(board_p, key, engine_vel);
    let n = bl.len().next_power_of_two();
    let mut planner = FftPlanner::<f64>::new();
    let mics = preset.voicing.mics.expect("the measured preset has a pair");
    let bcoef = diffuse_b(&mics);
    let ca = forward(&cl, n, &mut planner);
    let cb = forward(&cr, n, &mut planner);
    let mut x: Vec<C64> = (0..n)
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
        .collect();
    let (ilo, ihi) = inject_hz();
    let (blo, bhi) = (bin_of(ilo, n).max(1), bin_of(ihi, n).min(n / 2));
    for (j, v) in x.iter_mut().enumerate() {
        let jj = if j <= n / 2 { j } else { n - j };
        if jj < blo || jj > bhi {
            *v = C64::new(0.0, 0.0);
        }
    }
    Ok(KeyV {
        label: realism::note_name(key),
        bare_l: bl,
        bare_r: br,
        xspec: x,
        n,
        reference,
        alternate,
    })
}

/// Treated pair in the time domain (f32, as the engine writes), plus the two
/// mono residuals and the peak.
fn treat(k: &KeyV, gain: &[C64]) -> (Audio, f64, f64, f64) {
    let len = k.bare_l.len();
    let mut planner = FftPlanner::<f64>::new();
    let xs: Vec<C64> = k.xspec.iter().zip(gain).map(|(x, h)| x * h).collect();
    let s = inverse(xs, &mut planner);
    let mut peak = 0.0f64;
    let (mut w64, mut w32) = (0.0f64, 0.0f64);
    let (mut lo, mut ro) = (Vec::with_capacity(len), Vec::with_capacity(len));
    for i in 0..len {
        let (l, r) = (f64::from(k.bare_l[i]), f64::from(k.bare_r[i]));
        let m0 = 0.5 * (l + r);
        let (lt, rt) = (l + s[i], r - s[i]);
        w64 = w64.max((0.5 * (lt + rt) - m0).abs());
        let (l32, r32) = (lt as f32, rt as f32);
        w32 = w32.max((0.5 * (f64::from(l32) + f64::from(r32)) - m0).abs());
        peak = peak.max(m0.abs()).max(lt.abs()).max(rt.abs());
        lo.push(l32);
        ro.push(r32);
    }
    (
        Audio::new(SAMPLE_RATE, vec![lo, ro]).expect("two channels"),
        w64,
        w32,
        peak,
    )
}

fn pooled_pair_over_mono(bands: &[Vec<[f64; 5]>], totals: &[f64], i: usize) -> f64 {
    let (mut p, mut m) = (0.0, 0.0);
    for (b, &t) in bands.iter().zip(totals) {
        p += (b[i][0] + b[i][1]) / t;
        m += 2.0 * b[i][2] / t;
    }
    db(p / m)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let g = grid();
    let cp: Vec<usize> = (0..g.len())
        .filter(|&i| g[i] > INJECT_CENTRES.0 * 0.99 && g[i] < INJECT_CENTRES.1 * 1.01)
        .collect();
    let preset = shipped();
    let bare_p = bare(&preset);
    let mut board_p = bare_p.clone();
    board_p.soundboard.board_mix = 1.0;

    let library = SampleLibrary::from_sfz(Path::new(SFZ))?;
    let recorded = RecordedKeys::from_library(&library)?;
    let layers = VelocityLayers::from_library(&library)?;
    let alt_vel = layers.alternate(VELOCITY);

    println!("# side_injection_verify — adversarial re-measurement\n");
    println!(
        "fdn source, fitted u hardcoded from the probe's §2. Alternate layer v{alt_vel}.\n"
    );

    // ---- primary: engine at v90, reference at v90, alternate layer as bar ----
    let keys: Vec<KeyV> = recorded
        .keys()
        .par_iter()
        .map(|&k| build_key(k, &preset, &bare_p, &board_p, VELOCITY, VELOCITY, alt_vel))
        .collect::<Result<Vec<_>, _>>()?;

    let gain = realised_gain(keys[0].n);
    let treated: Vec<(Audio, f64, f64, f64)> =
        keys.par_iter().map(|k| treat(k, &gain)).collect();

    // §V1: coherence of the injected source (at fitted level) with the MID and
    // with the SIDE of the bare pair, per fitted band, pooled over keys.
    println!("## V1. Source coherence with mid and side (pooled, per fitted band)\n");
    println!("| Hz | |coh| with mid | |coh| with side | injected/mid energy dB |");
    println!("|---:|---:|---:|---:|");
    for &i in &cp {
        let (lo, hi) = band_edges(g[i]);
        let (mut xe, mut me, mut se, mut xm, mut xs) = (0.0, 0.0, 0.0, C64::new(0.0, 0.0), C64::new(0.0, 0.0));
        for k in &keys {
            let n = k.n;
            let (jlo, jhi) = (bin_of(lo, n).max(1), bin_of(hi, n).min(n / 2));
            let mut planner = FftPlanner::<f64>::new();
            let a = forward(&k.bare_l, n, &mut planner);
            let b = forward(&k.bare_r, n, &mut planner);
            for j in jlo..=jhi {
                let src = k.xspec[j] * gain[j];
                let mid = (a[j] + b[j]) * 0.5;
                let side = (a[j] - b[j]) * 0.5;
                xe += src.norm_sqr();
                me += mid.norm_sqr();
                se += side.norm_sqr();
                xm += src * mid.conj();
                xs += src * side.conj();
            }
        }
        println!(
            "| {:.0} | {:.3} | {:.3} | {:+.2} |",
            g[i],
            (xm.norm()) / (xe * me).sqrt(),
            (xs.norm()) / (xe * se).sqrt(),
            db(xe / me),
        );
    }

    // §V2: the headline table, re-measured off the treated f32 audio.
    println!("\n## V2. Treated pair/mono and r0 per fitted band (independent code)\n");
    let mut planner = FftPlanner::<f64>::new();
    let per_key: Vec<(Vec<[f64; 5]>, Vec<[f64; 5]>, Vec<[f64; 5]>, f64, f64)> = keys
        .iter()
        .zip(&treated)
        .map(|(k, (a, ..))| {
            let n = k.n;
            let ba = forward(&k.bare_l, n, &mut planner);
            let bb = forward(&k.bare_r, n, &mut planner);
            let ta = forward(&a.channels[0], n, &mut planner);
            let tb = forward(&a.channels[1], n, &mut planner);
            let ra = forward(&k.reference.channels[0], n, &mut planner);
            let rb = forward(&k.reference.channels[1], n, &mut planner);
            let bband: Vec<[f64; 5]> = g
                .iter()
                .map(|&hz| {
                    let (lo, hi) = band_edges(hz);
                    band5(&ba, &bb, lo, hi)
                })
                .collect();
            let tband: Vec<[f64; 5]> = g
                .iter()
                .map(|&hz| {
                    let (lo, hi) = band_edges(hz);
                    band5(&ta, &tb, lo, hi)
                })
                .collect();
            let rband: Vec<[f64; 5]> = g
                .iter()
                .map(|&hz| {
                    let (lo, hi) = band_edges(hz);
                    band5(&ra, &rb, lo, hi)
                })
                .collect();
            let bt: f64 = bband.iter().map(|e| e[2]).sum();
            let rt: f64 = rband.iter().map(|e| e[2]).sum();
            (bband, tband, rband, bt, rt)
        })
        .collect();
    let bare_bands: Vec<Vec<[f64; 5]>> = per_key.iter().map(|x| x.0.clone()).collect();
    let treat_bands: Vec<Vec<[f64; 5]>> = per_key.iter().map(|x| x.1.clone()).collect();
    let ref_bands: Vec<Vec<[f64; 5]>> = per_key.iter().map(|x| x.2.clone()).collect();
    let mono_tot: Vec<f64> = per_key.iter().map(|x| x.3).collect();
    let ref_tot: Vec<f64> = per_key.iter().map(|x| x.4).collect();

    println!("| Hz | REF pair/mono | bare | treated | probe said treated |");
    println!("|---:|---:|---:|---:|---:|");
    let probe_said = [2.27, 9.37, 6.01, 4.61, 3.61, 2.99, 1.72];
    for (c, &i) in cp.iter().enumerate() {
        println!(
            "| {:.0} | {:+.2} | {:+.2} | {:+.2} | {:+.2} |",
            g[i],
            pooled_pair_over_mono(&ref_bands, &ref_tot, i),
            pooled_pair_over_mono(&bare_bands, &mono_tot, i),
            pooled_pair_over_mono(&treat_bands, &mono_tot, i),
            probe_said[c],
        );
    }

    // Mono invariance, re-derived.
    let mut worst_band = 0.0f64;
    for i in 0..g.len() {
        for (t, b) in treat_bands.iter().zip(&bare_bands) {
            let d = (db(t[i][2]) - db(b[i][2])).abs();
            if d.is_finite() {
                worst_band = worst_band.max(d);
            }
        }
    }
    let w64 = treated.iter().map(|t| t.1).fold(0.0f64, f64::max);
    let w32 = treated.iter().map(|t| t.2).fold(0.0f64, f64::max);
    let pk = treated.iter().map(|t| t.3).fold(0.0f64, f64::max);
    println!(
        "\nMono: worst band change {:.2e} dB; per-sample f64 {:.1} dBFS re peak; f32 {:.1} dBFS re peak.",
        worst_band,
        20.0 * (w64 / pk).max(1e-300).log10(),
        20.0 * (w32 / pk).max(1e-300).log10(),
    );

    // Gate statistics through the repository's own instruments.
    let audios: Vec<Audio> = treated.iter().map(|t| t.0.clone()).collect();
    let bare_audios: Vec<Audio> = keys
        .iter()
        .map(|k| Audio::new(SAMPLE_RATE, vec![k.bare_l.clone(), k.bare_r.clone()]).expect("two"))
        .collect();
    let mk = |engine: &[Audio]| -> (Vec<StereoItem>, Vec<ChannelItem>) {
        let s: Vec<StereoItem> = keys
            .iter()
            .zip(engine)
            .map(|(k, a)| StereoItem {
                label: k.label.clone(),
                engine: realism::stereo_image_of(a).expect("two channels"),
                reference: realism::stereo_image_of(&k.reference).expect("two channels"),
                alternate: realism::stereo_image_of(&k.alternate).expect("two channels"),
            })
            .collect();
        let c: Vec<ChannelItem> = keys
            .iter()
            .zip(engine)
            .map(|(k, a)| ChannelItem {
                label: k.label.clone(),
                engine: realism::channel_shape_of(a).expect("two channels"),
                reference: realism::channel_shape_of(&k.reference).expect("two channels"),
                alternate: realism::channel_shape_of(&k.alternate).expect("two channels"),
            })
            .collect();
        (s, c)
    };
    let (treat_s, treat_c) = mk(&audios);
    let (_bare_s, bare_c) = mk(&bare_audios);
    println!("\n`realism::stereo_columns` on the treated pair (gate bands):\n");
    println!("| band | REF r0 | treated r0 | bar | pass |");
    println!("|---|---:|---:|---:|:--:|");
    for c in realism::stereo_columns(&treat_s) {
        println!(
            "| {} | {:+.3} | {:+.3} | {:.3} | {} |",
            c.name,
            c.reference_r0,
            c.engine_r0,
            c.bar,
            if c.pass { "yes" } else { "RED" }
        );
    }
    println!("\n`realism::channel_fine_columns` spread (dev_L − dev_R), 150-340 Hz:\n");
    println!("| Hz | REF spread | bare | treated |");
    println!("|---:|---:|---:|---:|");
    let fc_t = realism::channel_fine_columns(&treat_c);
    let fc_b = realism::channel_fine_columns(&bare_c);
    for i in 0..fc_t.len() {
        let c = &fc_t[i];
        if c.hi_hz < 150.0 || c.lo_hz > 340.0 {
            continue;
        }
        println!(
            "| {} | {:+.2} | {:+.2} | {:+.2} |",
            c.name,
            c.reference_left_db - c.reference_right_db,
            fc_b[i].engine_left_db - fc_b[i].engine_right_db,
            c.engine_left_db - c.engine_right_db,
        );
    }

    // Strike window.
    {
        let (ilo, ihi) = inject_hz();
        let win = (0.010 * SR) as usize;
        let ms_db = |l: &[f32], r: &[f32]| -> f64 {
            let n = win.next_power_of_two() * 2;
            let mut planner = FftPlanner::<f64>::new();
            let a = forward(&l[..win], n, &mut planner);
            let b = forward(&r[..win], n, &mut planner);
            let e = band5(&a, &b, ilo, ihi);
            db(e[2] / e[3].max(1e-300))
        };
        let mut refv: Vec<f64> = keys
            .iter()
            .map(|k| ms_db(&k.reference.channels[0], &k.reference.channels[1]))
            .collect();
        let mut treatv: Vec<f64> = audios
            .iter()
            .map(|a| ms_db(&a.channels[0], &a.channels[1]))
            .collect();
        println!(
            "\nStrike window (first 10 ms, mid-over-side in the injected range): recording {:+.2} dB, treated {:+.2} dB.",
            median(&mut refv),
            median(&mut treatv)
        );
    }

    // §V3: held-out velocity. Same fitted u; engine and reference both at the
    // alternate layer's velocity. The v90 recording layer serves as the bar.
    println!("\n## V3. Held-out layer: v90-fitted levels applied at v{alt_vel}\n");
    let keys_h: Vec<KeyV> = recorded
        .keys()
        .par_iter()
        .map(|&k| build_key(k, &preset, &bare_p, &board_p, alt_vel, alt_vel, VELOCITY))
        .collect::<Result<Vec<_>, _>>()?;
    let gain_h = realised_gain(keys_h[0].n);
    let treated_h: Vec<(Audio, f64, f64, f64)> =
        keys_h.par_iter().map(|k| treat(k, &gain_h)).collect();
    let mut planner2 = FftPlanner::<f64>::new();
    let per_key_h: Vec<(Vec<[f64; 5]>, Vec<[f64; 5]>, Vec<[f64; 5]>, f64, f64)> = keys_h
        .iter()
        .zip(&treated_h)
        .map(|(k, (a, ..))| {
            let n: usize = k.n;
            let ba = forward(&k.bare_l, n, &mut planner2);
            let bb = forward(&k.bare_r, n, &mut planner2);
            let ta = forward(&a.channels[0], n, &mut planner2);
            let tb = forward(&a.channels[1], n, &mut planner2);
            let ra = forward(&k.reference.channels[0], n, &mut planner2);
            let rb = forward(&k.reference.channels[1], n, &mut planner2);
            let f = |aa: &Vec<C64>, bb2: &Vec<C64>| -> Vec<[f64; 5]> {
                g.iter()
                    .map(|&hz| {
                        let (lo, hi) = band_edges(hz);
                        band5(aa, bb2, lo, hi)
                    })
                    .collect()
            };
            let bband = f(&ba, &bb);
            let tband = f(&ta, &tb);
            let rband = f(&ra, &rb);
            let bt: f64 = bband.iter().map(|e| e[2]).sum();
            let rt: f64 = rband.iter().map(|e| e[2]).sum();
            (bband, tband, rband, bt, rt)
        })
        .collect();
    let bare_h: Vec<Vec<[f64; 5]>> = per_key_h.iter().map(|x| x.0.clone()).collect();
    let treat_hb: Vec<Vec<[f64; 5]>> = per_key_h.iter().map(|x| x.1.clone()).collect();
    let ref_h: Vec<Vec<[f64; 5]>> = per_key_h.iter().map(|x| x.2.clone()).collect();
    let mono_th: Vec<f64> = per_key_h.iter().map(|x| x.3).collect();
    let ref_th: Vec<f64> = per_key_h.iter().map(|x| x.4).collect();
    println!("| Hz | REF(v{alt_vel}) pair/mono | bare(v{alt_vel}) | treated(v{alt_vel}) | miss | v90 target |");
    println!("|---:|---:|---:|---:|---:|---:|");
    let v90_target = [2.28, 9.38, 6.02, 4.61, 3.61, 2.99, 1.72];
    for (c, &i) in cp.iter().enumerate() {
        let r = pooled_pair_over_mono(&ref_h, &ref_th, i);
        let t = pooled_pair_over_mono(&treat_hb, &mono_th, i);
        println!(
            "| {:.0} | {:+.2} | {:+.2} | {:+.2} | {:+.2} | {:+.2} |",
            g[i],
            r,
            pooled_pair_over_mono(&bare_h, &mono_th, i),
            t,
            t - r,
            v90_target[c],
        );
    }

    Ok(())
}
