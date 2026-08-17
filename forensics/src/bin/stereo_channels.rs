//! Per-channel forensics on the virtual microphone pair.
//!
//! Every fidelity gate in this repository scores the **mono sum**, and
//! `soundboard::Mics` is written so the mono sum cannot move. This instrument
//! measures the two things that construction leaves unscored: what each
//! channel's own spectrum does, and what the side path does to broadband
//! content against tonal content.
//!
//! ```text
//! cargo run --release -p forensics --bin stereo_channels -- <section> [preset]
//! sections: analytic  keys  phrase  noise  melody  all
//! ```

use std::path::{Path, PathBuf};

use piano_emulator::preset::{MicVoicing, NoiseAnchor, Preset, SILENT_LEVEL_DB};
use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_emulator::types::Event;
use piano_tuner::audio::Audio;
use piano_tuner::estimate::melody::{self, LineNote, NoteTexture, Window};
use piano_tuner::realism::{self, Phrase, RecordedKeys};
use piano_tuner::sampler::{engine_events, SamplerEvent, SAMPLER_VERSION};
use piano_tuner::{cache, SampleLibrary, Sampler, TimedEvent, SAMPLE_RATE};

use rustfft::num_complex::Complex32;
use rustfft::FftPlanner;

const SFZ: &str = "data/salamander/SalamanderGrandPiano-V3+20200602.sfz";
const DATA: &str = "data/salamander";
const SPEED_OF_SOUND: f64 = 343.0;
const MIC_DIFFUSE_POLE_K: f64 = 0.426_63;

/// The keys the per-key work is done on: recorded takes only, because only a
/// recorded key has per-channel *truth* — the real AB pair's own two capsules.
/// Six of them, spanning the melody register 57-67 and one either side.
const KEYS: [u8; 6] = [54, 57, 60, 63, 66, 69];
const VELOCITY: u8 = 90;
const RENDER_S: f64 = 3.0;
const PREROLL: usize = realism::STEREO_PREROLL_SAMPLES;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let section = args
        .first()
        .cloned()
        .unwrap_or_else(|| "all".to_string());
    let preset_path = PathBuf::from(
        args.get(1)
            .cloned()
            .unwrap_or_else(|| "presets/salamander-c5.toml".into()),
    );
    let shipped = Preset::load(&preset_path)?;

    if section == "analytic" || section == "all" {
        analytic(&shipped);
    }
    if section == "keys" || section == "all" {
        keys(&shipped)?;
    }
    if section == "phrase" || section == "all" {
        phrase(&shipped)?;
    }
    if section == "noise" || section == "all" {
        noise(&shipped)?;
    }
    if section == "melody" || section == "all" {
        melody_channels(&shipped)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Preset variants
// ---------------------------------------------------------------------------

fn panpot(preset: &Preset) -> Preset {
    let mut p = preset.clone();
    p.voicing.mics = None;
    p
}

fn pair_only(preset: &Preset) -> Preset {
    let mut p = preset.clone();
    if let Some(mics) = preset.voicing.mics {
        p.voicing.mics = Some(MicVoicing {
            modal: None,
            ..mics
        });
    }
    p
}

fn no_strike(preset: &Preset) -> Preset {
    let mut p = preset.clone();
    p.noise.strike.level_db = vec![NoiseAnchor {
        key: 21,
        db: SILENT_LEVEL_DB,
    }];
    p
}

// ---------------------------------------------------------------------------
// 1. The analytic per-channel transfer function of the pair
// ---------------------------------------------------------------------------

/// `Mics::taps`, reimplemented from `engine/src/soundboard.rs` so the pair's
/// per-channel transfer function can be written down rather than measured.
fn taps(mics: &MicVoicing, pan: f64) -> (f64, f64, f64) {
    let half = 0.5 * f64::from(mics.spacing_m);
    let h = f64::from(mics.height_m);
    let x = pan.clamp(-1.0, 1.0) * f64::from(mics.span_m);
    let dl = ((x + half).powi(2) + h * h).sqrt();
    let dr = ((x - half).powi(2) + h * h).sqrt();
    let (al, ar) = (1.0 / dl, 1.0 / dr);
    let n = 1.0 / (al * al + ar * ar).sqrt();
    (al * n, ar * n, (dl - dr) / SPEED_OF_SOUND)
}

/// `soundboard::pan_for_key` — the pan the engine gives a key. Read out of the
/// engine by rendering is unnecessary: item 351 quotes it as
/// `(2 * key_position - 1) * 0.6`.
fn pan_for_key(key: u8) -> f64 {
    let position = (f64::from(key) - 21.0) / 87.0;
    (2.0 * position - 1.0) * 0.6
}

fn analytic(preset: &Preset) {
    let Some(mics) = preset.voicing.mics else {
        println!("no [voicing.mics]");
        return;
    };
    println!("== 1. the pair's per-channel transfer function, from the code ==\n");
    println!(
        "spacing {:.4} m  height {:.3} m  span {:.2} m  width {:.4}  diffuse {:.4}",
        mics.spacing_m, mics.height_m, mics.span_m, mics.width, mics.diffuse_coherence
    );
    let pole = MIC_DIFFUSE_POLE_K * SPEED_OF_SOUND / f64::from(mics.spacing_m)
        * f64::from(mics.diffuse_coherence);
    println!("board-field diffuse corner {pole:.0} Hz");
    if let Some(m) = mics.modal {
        println!(
            "modal lobe {:.1}-{:.1} Hz  lift {:.4}   -> in-band L = {:+.2} dB, R = {:+.2} dB \
             (mono sum unchanged)",
            m.lo_hz,
            m.hi_hz,
            m.lift,
            20.0 * (1.0 + f64::from(m.lift)).log10(),
            20.0 * (1.0 - f64::from(m.lift)).abs().log10(),
        );
    }
    println!();
    println!(
        "Per source the direct path is L = c*x + (w/2)(uL*x(t-dL) - uR*x(t-dR)), R = c*x - (same),"
    );
    println!(
        "with c = (gl+gr)/2 the pan-pot's own sum. A real spaced pair has L = uL*x(t-dL) alone:"
    );
    println!("one delayed copy, no comb. Here each channel holds an UNDELAYED copy and a delayed");
    println!("one, so each channel is a two-tap comb of the source with itself.\n");
    println!(
        "{:<5} {:>7} {:>7} {:>7} {:>7} {:>8} {:>8} {:>8} {:>8} {:>9} {:>9}",
        "key", "pan", "c", "uL", "uR", "delta_us", "f1_Hz", "combL_dB", "combR_dB", "L_rms_dB",
        "R_rms_dB"
    );
    for key in [21u8, 33, 45, 54, 57, 60, 63, 66, 69, 81, 96, 108] {
        let pan = pan_for_key(key);
        let angle = (pan + 1.0) * std::f64::consts::FRAC_PI_4;
        let (gl, gr) = (angle.cos(), angle.sin());
        let c = 0.5 * (gl + gr);
        let (ul, ur, delta) = taps(&mics, pan);
        let w = 0.5 * f64::from(mics.width);
        // Only the farther capsule is delayed. Whichever it is, the channel
        // that carries the *undelayed* term at full weight is the comb.
        // L(f) = c + w*uL*e^{-jwdL} - w*uR*e^{-jwdR}; one of dL, dR is zero.
        let (l_dc, l_rot, r_dc, r_rot) = if delta >= 0.0 {
            // source right of centre: dR = 0, dL = delta
            (c - w * ur, w * ul, c + w * ur, -w * ul)
        } else {
            (c + w * ul, -w * ur, c - w * ul, w * ur)
        };
        let comb = |dc: f64, rot: f64| {
            let hi = dc.abs() + rot.abs();
            let lo = (dc.abs() - rot.abs()).abs();
            20.0 * (hi / lo.max(1e-12)).log10()
        };
        let rms = |dc: f64, rot: f64| 10.0 * (dc * dc + rot * rot).log10();
        let f1 = if delta.abs() > 0.0 {
            1.0 / delta.abs()
        } else {
            f64::INFINITY
        };
        println!(
            "{:<5} {:>7.3} {:>7.4} {:>7.4} {:>7.4} {:>8.1} {:>8.0} {:>8.1} {:>8.1} {:>9.2} {:>9.2}",
            melody::note_name(key),
            pan,
            c,
            ul,
            ur,
            delta * 1e6,
            f1,
            comb(l_dc, l_rot),
            comb(r_dc, r_rot),
            rms(l_dc, l_rot),
            rms(r_dc, r_rot),
        );
    }
    println!();
    println!("f1 is the first comb NOTCH of the combed channel (notches at k/delta, k>=1);");
    println!("the other channel's extrema sit at the same places with the sign swapped.");
    println!();
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

fn render_key(preset: &Preset, key: u8) -> (Vec<f32>, Vec<f32>) {
    let preroll_s = PREROLL as f32 / SAMPLE_RATE as f32;
    let events = [RenderEvent::new(
        preroll_s,
        Event::NoteOn {
            key,
            vel: u16::from(VELOCITY),
        },
    )];
    let (l, r) = render_to_buffer(preset, &events, preroll_s + RENDER_S as f32);
    (l[PREROLL..].to_vec(), r[PREROLL..].to_vec())
}

fn reference_key(key: u8) -> Result<Audio, Box<dyn std::error::Error>> {
    let sfz = Path::new(SFZ);
    let mut print = cache::Fingerprint::new();
    print
        .str("tests/stereo/reference")
        .u64(u64::from(SAMPLER_VERSION))
        .file(sfz)?
        .u64(u64::from(SAMPLE_RATE))
        .u64(u64::from(key))
        .u64(u64::from(VELOCITY))
        .f64(RENDER_S);
    let dir = cache::reference_dir(Path::new(DATA));
    let path = dir.join(format!(
        "stereo-key{key:03}-v{:03}-{}.wav",
        VELOCITY,
        print.hex()
    ));
    let audio = cache::audio(&path, || {
        let mut sampler = Sampler::new(sfz)?;
        let events = [TimedEvent::new(
            0.0,
            SamplerEvent::NoteOn {
                key,
                vel: VELOCITY,
            },
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
    })?;
    Ok(audio)
}

fn render_phrase(preset: &Preset, phrase: &Phrase) -> Audio {
    let events: Vec<RenderEvent> = engine_events::to_render_events(&phrase.events);
    let (l, r) = render_to_buffer(preset, &events, phrase.duration_s as f32);
    Audio::new(SAMPLE_RATE, vec![l, r]).expect("stereo")
}

fn reference_phrase(phrase: &Phrase) -> Result<Audio, Box<dyn std::error::Error>> {
    let sfz = Path::new(SFZ);
    let mut key = cache::Fingerprint::new();
    key.str("tests/melody/reference")
        .u64(u64::from(SAMPLER_VERSION))
        .file(sfz)?
        .u64(u64::from(SAMPLE_RATE))
        .str(phrase.name)
        .str("reference")
        .f64(phrase.duration_s);
    let path = cache::reference_dir(Path::new(DATA))
        .join(format!("melody-{}-reference-{}.wav", phrase.name, key.hex()));
    let rendered = cache::audio(&path, || {
        let mut sampler = Sampler::new(sfz)?;
        sampler.render(&phrase.events, phrase.duration_s)
    })?;
    Ok(melody::align_reference(&rendered, phrase.events[0].time_s))
}

// ---------------------------------------------------------------------------
// Spectra
// ---------------------------------------------------------------------------

/// Welch power spectrum, 8192-point Hann, half-overlap. Long enough to resolve
/// a comb whose teeth are 2.7 kHz apart many times over, and averaged so the
/// note's own partial structure does not read as ripple.
fn power_spectrum(signal: &[f32]) -> Vec<f64> {
    const N: usize = 8192;
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N);
    let window: Vec<f32> = (0..N)
        .map(|i| 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / N as f64).cos() as f32)
        .collect();
    let mut acc = vec![0.0f64; N / 2 + 1];
    let mut frames = 0usize;
    let mut start = 0usize;
    while start + N <= signal.len() {
        let mut buffer: Vec<Complex32> = (0..N)
            .map(|i| Complex32::new(signal[start + i] * window[i], 0.0))
            .collect();
        fft.process(&mut buffer);
        for (slot, c) in acc.iter_mut().zip(buffer.iter().take(N / 2 + 1)) {
            *slot += f64::from(c.norm_sqr());
        }
        frames += 1;
        start += N / 2;
    }
    if frames == 0 {
        return acc;
    }
    acc.iter_mut().for_each(|s| *s /= frames as f64);
    acc
}

fn bin_hz(i: usize) -> f64 {
    i as f64 * f64::from(SAMPLE_RATE) / 8192.0
}

fn band_db(power: &[f64], lo: f64, hi: f64) -> f64 {
    let mut sum = 0.0;
    for (i, &p) in power.iter().enumerate() {
        let f = bin_hz(i);
        if f >= lo && f < hi {
            sum += p;
        }
    }
    10.0 * sum.max(1e-30).log10()
}

/// One-sixth-octave smoothing of a power spectrum, on a log-frequency grid.
///
/// A note's partials are 200-400 Hz apart in this register and a comb whose
/// teeth are 2.7-4.7 kHz apart is what is being looked for, so the partial
/// structure has to go before the ripple can be read. Sixth-octave is 480 Hz
/// wide at 4 kHz — several partials, a fraction of a comb tooth.
fn smoothed(power: &[f64], lo: f64, hi: f64, points: usize) -> Vec<(f64, f64)> {
    let ratio = (hi / lo).powf(1.0 / (points - 1) as f64);
    let half = 2f64.powf(1.0 / 12.0); // sixth-octave: +/- a twelfth
    (0..points)
        .map(|i| {
            let f = lo * ratio.powi(i as i32);
            let (a, b) = (f / half, f * half);
            let mut sum = 0.0;
            let mut n = 0usize;
            for (j, &p) in power.iter().enumerate() {
                let g = bin_hz(j);
                if g >= a && g < b {
                    sum += p;
                    n += 1;
                }
            }
            (f, if n == 0 { f64::NAN } else { 10.0 * (sum / n as f64).max(1e-30).log10() })
        })
        .collect()
}

/// The analytic per-channel magnitude of the direct path at `hz`, from the
/// construction in `soundboard::add_voice`: `L = c + w*uL*e^-jwdL - w*uR*e^-jwdR`.
fn analytic_channel_db(mics: &MicVoicing, pan: f64, hz: f64) -> (f64, f64) {
    let angle = (pan + 1.0) * std::f64::consts::FRAC_PI_4;
    let c = 0.5 * (angle.cos() + angle.sin());
    let (ul, ur, delta) = taps(mics, pan);
    let w = 0.5 * f64::from(mics.width);
    let (dl, dr) = if delta >= 0.0 { (delta, 0.0) } else { (0.0, -delta) };
    let phase = |d: f64| {
        let t = -std::f64::consts::TAU * hz * d;
        (t.cos(), t.sin())
    };
    let (cl, sl) = phase(dl);
    let (cr, sr) = phase(dr);
    let sre = w * ul * cl - w * ur * cr;
    let sim = w * ul * sl - w * ur * sr;
    let l = ((c + sre).powi(2) + sim * sim).sqrt();
    let r = ((c - sre).powi(2) + sim * sim).sqrt();
    // Referenced to the pan-pot's own two gains, which is what the ratio a
    // render measures is against.
    (
        20.0 * (l / angle.cos()).log10(),
        20.0 * (r / angle.sin()).log10(),
    )
}

fn keys(preset: &Preset) -> Result<(), Box<dyn std::error::Error>> {
    let mics = preset.voicing.mics.expect("[voicing.mics]");
    println!("== 2. per-channel spectra, recorded keys at v{VELOCITY} ==\n");
    let flat = panpot(preset);
    println!("(a) THE COMB. Sixth-octave 10*log10(P_mics / P_panpot) per channel, dB —");
    println!("    the mic stage's own per-channel response, the source cancelled. `pred` is");
    println!("    the analytic direct-path magnitude from soundboard::add_voice.\n");
    let grid: Vec<f64> = (0..17)
        .map(|i| 500.0 * 2f64.powf(i as f64 / 4.0))
        .collect();
    print!("{:<5} {:<7}", "key", "chan");
    for f in &grid {
        print!(" {:>6.0}", f);
    }
    println!();
    for &key in &KEYS {
        let pan = pan_for_key(key);
        let (ml, mr) = render_key(preset, key);
        let (pl, pr) = render_key(&flat, key);
        let a = smoothed(&power_spectrum(&ml), 400.0, 14000.0, 200);
        let b = smoothed(&power_spectrum(&pl), 400.0, 14000.0, 200);
        let c = smoothed(&power_spectrum(&mr), 400.0, 14000.0, 200);
        let d = smoothed(&power_spectrum(&pr), 400.0, 14000.0, 200);
        let pick = |curve: &[(f64, f64)], f: f64| {
            curve
                .iter()
                .min_by(|x, y| (x.0 - f).abs().partial_cmp(&(y.0 - f).abs()).unwrap())
                .map(|p| p.1)
                .unwrap_or(f64::NAN)
        };
        for (name, num, den) in [("L", &a, &b), ("R", &c, &d)] {
            print!("{:<5} {:<7}", melody::note_name(key), name);
            for &f in &grid {
                print!(" {:>6.1}", pick(num, f) - pick(den, f));
            }
            println!();
            print!("{:<5} {:<7}", "", "pred");
            for &f in &grid {
                let (l, r) = analytic_channel_db(&mics, pan, f);
                print!(" {:>6.1}", if name == "L" { l } else { r });
            }
            println!();
        }
    }
    println!();
    println!("    and the same thing as one number per key/channel: peak-to-trough of the");
    println!("    measured curve over 1-10 kHz, and of the analytic direct path.\n");
    println!(
        "{:<5} {:<6} {:>12} {:>12} {:>12} {:>12}",
        "key", "chan", "meas_p2p", "pred_p2p", "meas_notch", "pred_notch"
    );
    for &key in &KEYS {
        let pan = pan_for_key(key);
        let (ml, mr) = render_key(preset, key);
        let (pl, pr) = render_key(&flat, key);
        for (name, num, den) in [("L", &ml, &pl), ("R", &mr, &pr)] {
            let a = smoothed(&power_spectrum(num), 1000.0, 10000.0, 120);
            let b = smoothed(&power_spectrum(den), 1000.0, 10000.0, 120);
            let curve: Vec<(f64, f64)> = a
                .iter()
                .zip(&b)
                .filter(|(x, y)| x.1.is_finite() && y.1.is_finite())
                .map(|(x, y)| (x.0, x.1 - y.1))
                .collect();
            let hi = curve.iter().map(|p| p.1).fold(f64::MIN, f64::max);
            let lo = curve.iter().map(|p| p.1).fold(f64::MAX, f64::min);
            let at_lo = curve
                .iter()
                .min_by(|x, y| x.1.partial_cmp(&y.1).unwrap())
                .map(|p| p.0)
                .unwrap_or(0.0);
            let pred: Vec<(f64, f64)> = curve
                .iter()
                .map(|p| {
                    let (l, r) = analytic_channel_db(&mics, pan, p.0);
                    (p.0, if name == "L" { l } else { r })
                })
                .collect();
            let phi = pred.iter().map(|p| p.1).fold(f64::MIN, f64::max);
            let plo = pred.iter().map(|p| p.1).fold(f64::MAX, f64::min);
            let pat = pred
                .iter()
                .min_by(|x, y| x.1.partial_cmp(&y.1).unwrap())
                .map(|p| p.0)
                .unwrap_or(0.0);
            println!(
                "{:<5} {:<6} {:>12.1} {:>12.1} {:>11.0}H {:>11.0}H",
                melody::note_name(key),
                name,
                hi - lo,
                phi - plo,
                at_lo,
                pat
            );
        }
    }
    println!();

    println!("(b) per-channel band levels, each take referenced to its OWN MONO broadband");
    println!("    (which is how renders/stereo levels the takes), dB. `dev` is the channel's");
    println!("    departure from the same take's mono in that band — the number a mono board");
    println!("    cannot see. The reference's dev is a real AB pair's own.\n");
    for &key in &KEYS {
        let reference = reference_key(key)?;
        let (ml, mr) = render_key(preset, key);
        let (pl, pr) = render_key(&flat, key);
        println!("  {} :", melody::note_name(key));
        println!(
            "  {:<12} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
            "take/chan", "63-125", "125-250", "250-500", "500-2k", "2k-6k", "6k-12k"
        );
        for (name, l, r) in [
            ("reference", &reference.channels[0], &reference.channels[1]),
            ("engine", &ml, &mr),
            ("panpot", &pl, &pr),
        ] {
            let mono: Vec<f32> = l.iter().zip(r).map(|(a, b)| 0.5 * (a + b)).collect();
            let pm = power_spectrum(&mono);
            let anchor = band_db(&pm, 20.0, 20000.0);
            for (chan, sig) in [("L", l), ("R", r)] {
                let p = power_spectrum(sig);
                print!("  {:<12}", format!("{name} {chan}"));
                for &(_, lo, hi) in &realism::STEREO_BANDS {
                    print!(" {:>9.2}", band_db(&p, lo, hi) - anchor);
                }
                println!();
            }
            print!("  {:<12}", format!("{name} dev"));
            for &(_, lo, hi) in &realism::STEREO_BANDS {
                let a = band_db(&power_spectrum(l), lo, hi);
                let b = band_db(&power_spectrum(r), lo, hi);
                let m = band_db(&pm, lo, hi);
                print!(" {:>9.2}", 10.0 * (10f64.powf(a / 10.0) + 10f64.powf(b / 10.0)).log10() - 3.0 - m);
            }
            println!();
            print!("  {:<12}", format!("{name} L-R"));
            for &(_, lo, hi) in &realism::STEREO_BANDS {
                print!(
                    " {:>9.2}",
                    band_db(&power_spectrum(l), lo, hi) - band_db(&power_spectrum(r), lo, hi)
                );
            }
            println!();
        }
        println!();
    }

    println!("(c) does the REFERENCE comb per channel? |L/R| in sixth octaves. A real spaced");
    println!("    pair puts its whole interchannel difference in PHASE: |L(f)/R(f)| is the");
    println!("    two inverse-distance gains and is flat. A mid+side construction puts it in");
    println!("    MAGNITUDE: |c+s|/|c-s|, which combs. Peak-to-trough over 1-10 kHz, dB.\n");
    println!(
        "{:<5} {:>12} {:>12} {:>12}",
        "key", "reference", "engine", "panpot"
    );
    for &key in &KEYS {
        let reference = reference_key(key)?;
        let (ml, mr) = render_key(preset, key);
        let (pl, pr) = render_key(&flat, key);
        print!("{:<5}", melody::note_name(key));
        for (l, r) in [
            (&reference.channels[0], &reference.channels[1]),
            (&ml, &mr),
            (&pl, &pr),
        ] {
            let a = smoothed(&power_spectrum(l), 1000.0, 10000.0, 120);
            let b = smoothed(&power_spectrum(r), 1000.0, 10000.0, 120);
            let curve: Vec<f64> = a
                .iter()
                .zip(&b)
                .filter(|(x, y)| x.1.is_finite() && y.1.is_finite())
                .map(|(x, y)| x.1 - y.1)
                .collect();
            let hi = curve.iter().cloned().fold(f64::MIN, f64::max);
            let lo = curve.iter().cloned().fold(f64::MAX, f64::min);
            print!(" {:>12.1}", hi - lo);
        }
        println!();
    }
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// 3. The phrase
// ---------------------------------------------------------------------------

fn phrase(preset: &Preset) -> Result<(), Box<dyn std::error::Error>> {
    println!("== 3. the chords phrase, per channel ==\n");
    let flat = panpot(preset);
    let pair = pair_only(preset);
    let chords = realism::chords_pedal();
    let reference = reference_phrase(&chords)?;
    let takes: Vec<(&str, Audio)> = vec![
        ("reference", reference),
        ("engine", render_phrase(preset, &chords)),
        ("pair", render_phrase(&pair, &chords)),
        ("panpot", render_phrase(&flat, &chords)),
    ];
    println!("Each take referenced to its OWN MONO broadband level, which is exactly how");
    println!("renders/stereo sets the four gains. So these numbers are what a listener");
    println!("hears when the four files are played at the board's own matched level.\n");
    println!(
        "{:<14} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "take/chan", "63-125", "125-250", "250-500", "500-2k", "2k-6k", "6k-12k", "broad"
    );
    let mut brightness: Vec<(&str, f64, f64, f64, f64)> = Vec::new();
    for (name, audio) in &takes {
        let pm = power_spectrum(&audio.mono());
        let anchor = band_db(&pm, 20.0, 20000.0);
        let mut chan_powers: Vec<Vec<f64>> = Vec::new();
        for (chan, i) in [("L", 0usize), ("R", 1usize)] {
            let p = power_spectrum(&audio.channels[i]);
            print!("{:<14}", format!("{name} {chan}"));
            for &(_, lo, hi) in &realism::STEREO_BANDS {
                print!(" {:>9.2}", band_db(&p, lo, hi) - anchor);
            }
            println!(" {:>9.2}", band_db(&p, 20.0, 20000.0) - anchor);
            chan_powers.push(p);
        }
        print!("{:<14}", format!("{name} mono"));
        for &(_, lo, hi) in &realism::STEREO_BANDS {
            print!(" {:>9.2}", band_db(&pm, lo, hi) - anchor);
        }
        println!(" {:>9.2}", 0.0);
        // The channel pair's total power against the mono sum's, per band: the
        // side lift, which a mono board cannot see.
        print!("{:<14}", format!("{name} dev"));
        for &(_, lo, hi) in &realism::STEREO_BANDS {
            let a = band_db(&chan_powers[0], lo, hi);
            let b = band_db(&chan_powers[1], lo, hi);
            let m = band_db(&pm, lo, hi);
            print!(
                " {:>9.2}",
                10.0 * (10f64.powf(a / 10.0) + 10f64.powf(b / 10.0)).log10() - 3.0 - m
            );
        }
        println!();
        print!("{:<14}", format!("{name} L-R"));
        for &(_, lo, hi) in &realism::STEREO_BANDS {
            print!(
                " {:>9.2}",
                band_db(&chan_powers[0], lo, hi) - band_db(&chan_powers[1], lo, hi)
            );
        }
        println!();
        // Tilt: 2-6 kHz against 250-500 Hz, per channel and on the mono sum.
        let tilt = |p: &[f64]| band_db(p, 2000.0, 6000.0) - band_db(p, 250.0, 500.0);
        let tilt6 = |p: &[f64]| band_db(p, 6000.0, 12000.0) - band_db(p, 250.0, 500.0);
        brightness.push((
            name,
            tilt(&chan_powers[0]),
            tilt(&chan_powers[1]),
            tilt(&pm),
            tilt6(&pm),
        ));
        let _ = tilt6;
        println!();
    }
    println!("brilliance as a TILT — 2-6 kHz over 250-500 Hz, dB — per channel and on the");
    println!("mono sum, and engine minus reference on each. A mono board scores only the");
    println!("last column.\n");
    println!(
        "{:<12} {:>9} {:>9} {:>9}   {:>9} {:>9} {:>9}",
        "take", "L", "R", "mono", "dL", "dR", "dmono"
    );
    let r = brightness[0];
    for t in &brightness {
        println!(
            "{:<12} {:>9.2} {:>9.2} {:>9.2}   {:>9.2} {:>9.2} {:>9.2}",
            t.0,
            t.1,
            t.2,
            t.3,
            t.1 - r.1,
            t.2 - r.2,
            t.3 - r.3
        );
    }
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// 4. Noise to tone, per channel
// ---------------------------------------------------------------------------

fn noise(preset: &Preset) -> Result<(), Box<dyn std::error::Error>> {
    println!("== 4. the strike against the note, per channel ==\n");
    let quiet = no_strike(preset);
    let flat = panpot(preset);
    let flat_quiet = no_strike(&flat);
    println!("the event is additive in Voice::process, so the sample-wise difference of the");
    println!("two renders IS the burst through the whole chain. Per channel, its level");
    println!("against the note's own content in the same band, first 30 ms (dB):\n");
    println!(
        "{:<5} {:<8} {:<5} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "key", "take", "chan", "63-125", "125-250", "250-500", "500-2k", "2k-6k", "6k-12k"
    );
    let win = (0.03 * f64::from(SAMPLE_RATE)) as usize;
    for &key in &KEYS {
        for (take, on, off) in [("mics", preset, &quiet), ("panpot", &flat, &flat_quiet)] {
            let (al, ar) = render_key(on, key);
            let (bl, br) = render_key(off, key);
            for (chan, a, b) in [("L", &al, &bl), ("R", &ar, &br)] {
                let burst: Vec<f32> = a[..win].iter().zip(&b[..win]).map(|(x, y)| x - y).collect();
                let tone = &b[..win];
                let pb = power_spectrum_short(&burst);
                let pt = power_spectrum_short(tone);
                print!("{:<5} {:<8} {:<5}", melody::note_name(key), take, chan);
                for &(_, lo, hi) in &realism::STEREO_BANDS {
                    print!(
                        " {:>9.1}",
                        band_db_short(&pb, lo, hi) - band_db_short(&pt, lo, hi)
                    );
                }
                println!();
            }
        }
    }
    println!();
    println!("and the M10 statistic itself — attack tonality, arith/geo mean of the power");
    println!("spectrum of the first 30 ms from the strike, dB. Large is a line spectrum,");
    println!("zero a continuum, so a hammer too loud against its note reads LOW.\n");
    println!(
        "{:<5} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "key", "ref L", "ref R", "ref mono", "eng L", "eng R", "eng mono", "pan L", "pan mono"
    );
    for &key in &KEYS {
        let reference = reference_key(key)?;
        let (ml, mr) = render_key(preset, key);
        let (pl, pr) = render_key(&flat, key);
        let mono = |a: &[f32], b: &[f32]| -> Vec<f32> {
            a.iter().zip(b).map(|(x, y)| 0.5 * (x + y)).collect()
        };
        let ntt = |s: &[f32]| {
            piano_tuner::estimate::attack::noise_to_tone_db(s, 0.0, f64::from(SAMPLE_RATE))
        };
        let rm = reference.mono();
        println!(
            "{:<5} {:>9.2} {:>9.2} {:>9.2} {:>9.2} {:>9.2} {:>9.2} {:>9.2} {:>9.2}",
            melody::note_name(key),
            ntt(&reference.channels[0]),
            ntt(&reference.channels[1]),
            ntt(&rm),
            ntt(&ml),
            ntt(&mr),
            ntt(&mono(&ml, &mr)),
            ntt(&pl),
            ntt(&mono(&pl, &pr)),
        );
    }
    println!();
    strike_on_the_line(preset)?;
    Ok(())
}

/// The plainest statement of "how loud is the hammer": the burst's RMS against
/// the note's own, per channel, over every strike of the melody line.
fn strike_on_the_line(preset: &Preset) -> Result<(), Box<dyn std::error::Error>> {
    println!("the same question on the melody line, where the complaint was made: the");
    println!("burst's RMS against the note's own over the first 30 ms of every strike,");
    println!("per channel and on the mono sum, dB. `dev` is channel minus mono — the");
    println!("part the M10 fit could not see because it closed on the mono sum.\n");
    let line = melody::line_for(Window::Head);
    let notes = melody::line_notes_for(Window::Head);
    let variants: [(&str, Preset); 3] = [
        ("mics", preset.clone()),
        ("pair", pair_only(preset)),
        ("panpot", panpot(preset)),
    ];
    println!(
        "{:<10} {:>9} {:>9} {:>9} {:>10} {:>10}",
        "take", "L", "R", "mono", "devL", "devR"
    );
    for (name, on) in &variants {
        let off = no_strike(on);
        let a = render_phrase(on, &line);
        let b = render_phrase(&off, &line);
        let sr = f64::from(SAMPLE_RATE);
        let win = (0.03 * sr) as usize;
        let mut out = [0.0f64; 3];
        for (i, chan) in [0usize, 1, 2].iter().enumerate() {
            let (sa, sb): (Vec<f32>, Vec<f32>) = if *chan == 2 {
                (a.mono(), b.mono())
            } else {
                (a.channels[*chan].clone(), b.channels[*chan].clone())
            };
            let mut burst = 0.0f64;
            let mut tone = 0.0f64;
            for note in notes.iter().filter(|n| n.measurable()) {
                let strike = melody::note_onset(&sb, sr, note.onset_s);
                let lo = (strike * sr) as usize;
                let hi = (lo + win).min(sa.len());
                for j in lo..hi {
                    let d = f64::from(sa[j] - sb[j]);
                    burst += d * d;
                    tone += f64::from(sb[j]).powi(2);
                }
            }
            out[i] = 10.0 * (burst.max(1e-30) / tone.max(1e-30)).log10();
        }
        println!(
            "{:<10} {:>9.2} {:>9.2} {:>9.2} {:>10.2} {:>10.2}",
            name,
            out[0],
            out[1],
            out[2],
            out[0] - out[2],
            out[1] - out[2]
        );
        // The same thing per band, pooled over the line's 28 strikes: the
        // burst's spectrum against the note's, channel-pair energy and mono.
        let mut burst_ch = vec![vec![0.0f64; 0]; 3];
        let mut tone_ch = vec![vec![0.0f64; 0]; 3];
        for chan in 0..3usize {
            let (sa, sb): (Vec<f32>, Vec<f32>) = if chan == 2 {
                (a.mono(), b.mono())
            } else {
                (a.channels[chan].clone(), b.channels[chan].clone())
            };
            let mut bacc: Vec<f32> = Vec::new();
            let mut tacc: Vec<f32> = Vec::new();
            for note in notes.iter().filter(|n| n.measurable()) {
                let strike = melody::note_onset(&sb, sr, note.onset_s);
                let lo = (strike * sr) as usize;
                let hi = (lo + win).min(sa.len());
                for j in lo..hi {
                    bacc.push(sa[j] - sb[j]);
                    tacc.push(sb[j]);
                }
            }
            burst_ch[chan] = power_spectrum_short_long(&bacc);
            tone_ch[chan] = power_spectrum_short_long(&tacc);
        }
        for (label, chans) in [("L", vec![0usize]), ("R", vec![1]), ("pair", vec![0, 1]), ("mono", vec![2])] {
            print!("  {:<8} {:<6}", name, label);
            for &(_, lo, hi) in &realism::STEREO_BANDS {
                let bsum: f64 = chans
                    .iter()
                    .map(|&c| 10f64.powf(band_db(&burst_ch[c], lo, hi) / 10.0))
                    .sum();
                let tsum: f64 = chans
                    .iter()
                    .map(|&c| 10f64.powf(band_db(&tone_ch[c], lo, hi) / 10.0))
                    .sum();
                print!(" {:>9.1}", 10.0 * (bsum / tsum.max(1e-30)).log10());
            }
            println!();
        }
    }
    println!();
    Ok(())
}

/// Welch spectrum on the pooled attack samples, at the resolution `band_db`
/// expects (8192-point, the same grid as [`power_spectrum`]).
fn power_spectrum_short_long(signal: &[f32]) -> Vec<f64> {
    power_spectrum(signal)
}

fn power_spectrum_short(signal: &[f32]) -> Vec<f64> {
    const N: usize = 2048;
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N);
    let mut buffer = vec![Complex32::new(0.0, 0.0); N];
    let len = signal.len().min(N);
    for i in 0..len {
        let w = 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / len as f64).cos();
        buffer[i] = Complex32::new(signal[i] * w as f32, 0.0);
    }
    fft.process(&mut buffer);
    buffer[..=N / 2].iter().map(|c| f64::from(c.norm_sqr())).collect()
}

fn band_db_short(power: &[f64], lo: f64, hi: f64) -> f64 {
    let df = f64::from(SAMPLE_RATE) / 2048.0;
    let mut sum = 0.0;
    for (i, &p) in power.iter().enumerate() {
        let f = i as f64 * df;
        if f >= lo && f < hi {
            sum += p;
        }
    }
    10.0 * sum.max(1e-30).log10()
}

// ---------------------------------------------------------------------------
// 5. The melody line, per note, per channel
// ---------------------------------------------------------------------------

/// The mode-controlled lobe's exact gain at one frequency: an eighth-order
/// Butterworth highpass at `lo`, a fourth-order lowpass at `hi`, times `lift`.
/// Written out rather than rendered because it *is* the code
/// (`soundboard::ModalLobe`) and the whole per-note effect follows from it.
fn lobe_gain(lo: f64, hi: f64, lift: f64, f: f64) -> f64 {
    let hp = 1.0 / (1.0 + (lo / f).powi(16)).sqrt();
    let lp = 1.0 / (1.0 + (f / hi).powi(8)).sqrt();
    lift * hp * lp
}

/// What the lobe does to one note, per channel, at its own fundamental.
fn lobe_table(preset: &Preset) {
    let Some(mics) = preset.voicing.mics else { return };
    let Some(band) = mics.modal else { return };
    let (lo, hi, lift) = (
        f64::from(band.lo_hz),
        f64::from(band.hi_hz),
        f64::from(band.lift),
    );
    println!("the mode-controlled lobe, note by note: side += lift*bandpass(mid), so");
    println!("L = mid*(1+g), R = mid*(1-g) at the note's own fundamental. The mono sum");
    println!("is mid at every g, which is why no board in this repository sees any of it.\n");
    println!(
        "{:<5} {:>9} {:>8} {:>10} {:>10} {:>12}",
        "key", "f0_Hz", "g", "L_dB", "R_dB", "pair_dB"
    );
    for key in 55u8..=68 {
        let f0 = 440.0 * 2f64.powf((f64::from(key) - 69.0) / 12.0);
        let g = lobe_gain(lo, hi, lift, f0);
        let l = 20.0 * (1.0 + g).log10();
        let r = 20.0 * (1.0 - g).abs().log10();
        let pair = 10.0 * (((1.0 + g).powi(2) + (1.0 - g).powi(2)) / 2.0).log10();
        let mark = if melody::line_keys().contains(&key) {
            " <- melody"
        } else {
            ""
        };
        println!(
            "{:<5} {:>9.1} {:>8.3} {:>10.2} {:>10.2} {:>12.2}{mark}",
            melody::note_name(key),
            f0,
            g,
            l,
            r,
            pair
        );
    }
    println!();
}

/// Every note of the line, as a LEVEL: what a listener means by "that note
/// stands out". The four gate metrics are all shape statistics and none of
/// them is a loudness, which is why the gate can be green on a line whose C4
/// is six decibels up in one channel.
fn line_levels(preset: &Preset) -> Result<(), Box<dyn std::error::Error>> {
    let line = melody::line_for(Window::Head);
    let notes = melody::line_notes_for(Window::Head);
    let engine = render_phrase(preset, &line);
    let pair = render_phrase(&pair_only(preset), &line);
    let pan = render_phrase(&panpot(preset), &line);
    let reference = reference_phrase(&line)?;
    let sr = f64::from(SAMPLE_RATE);
    println!("per-note LEVEL over the note window, dB, as each key's departure from the");
    println!("line's own Theil-Sen trend in pitch — the melody gate's own device, applied");
    println!("to loudness. `pair` is the two channels' summed energy: what the ear gets.\n");
    let takes: Vec<(&str, &Audio)> = vec![
        ("reference", &reference),
        ("engine", &engine),
        ("pair-only", &pair),
        ("panpot", &pan),
    ];
    for (band_name, blo, bhi) in [
        ("broadband", 20.0, 20000.0),
        ("200-350Hz", 200.0, 350.0),
    ] {
        println!("  {band_name}:");
        print!("  {:<16}", "take/chan");
        for &k in &melody::line_keys() {
            print!(" {:>14}", melody::note_name(k));
        }
        println!();
        for (name, audio) in &takes {
            for chan in ["L", "R", "pair", "mono"] {
                let signal: Vec<f32> = match chan {
                    "L" => audio.channels[0].clone(),
                    "R" => audio.channels[1].clone(),
                    "mono" => audio.mono(),
                    _ => audio.channels[0]
                        .iter()
                        .zip(&audio.channels[1])
                        .map(|(a, b)| (a * a + b * b).sqrt())
                        .collect(),
                };
                let detect = audio.mono();
                let mut per: Vec<(u8, Vec<f64>)> = Vec::new();
                for note in notes.iter().filter(|n| n.measurable()) {
                    let strike = melody::note_onset(&detect, sr, note.onset_s);
                    let lo = ((strike + 0.03) * sr) as usize;
                    let hi = (((strike + 0.30) * sr) as usize).min(signal.len());
                    if hi <= lo {
                        continue;
                    }
                    let slice = &signal[lo..hi];
                    let level = if band_name == "broadband" {
                        10.0 * (slice.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>()
                            / (hi - lo) as f64)
                            .max(1e-30)
                            .log10()
                    } else {
                        band_db(&power_spectrum(slice), blo, bhi)
                    };
                    match per.iter_mut().find(|(k, _)| *k == note.key) {
                        Some((_, v)) => v.push(level),
                        None => per.push((note.key, vec![level])),
                    }
                }
                per.sort_by_key(|(k, _)| *k);
                let medians: Vec<(u8, f64)> = per
                    .iter()
                    .map(|(k, v)| {
                        let mut v = v.clone();
                        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        (*k, v[v.len() / 2])
                    })
                    .collect();
                let points: Vec<(f64, f64)> =
                    medians.iter().map(|(k, x)| (f64::from(*k), *x)).collect();
                let (slope, intercept) = melody::theil_sen(&points);
                print!("  {:<16}", format!("{name} {chan}"));
                for (k, x) in &medians {
                    print!(" {:>7.1}/{:<6.2}", x, x - (intercept + slope * f64::from(*k)));
                }
                println!();
            }
        }
        println!();
    }
    Ok(())
}

/// The lobe's predicted per-note lift, measured through the whole chain: each
/// key of the melody register struck alone, its own fundamental read with a
/// heterodyne, engine minus pan-pot, per channel.
fn measured_lift(preset: &Preset) -> Result<(), Box<dyn std::error::Error>> {
    let flat = panpot(preset);
    let pair = pair_only(preset);
    let sr = f64::from(SAMPLE_RATE);
    let level = |signal: &[f32], hz: f64| -> f64 {
        let env = piano_tuner::estimate::brilliance::narrowband_db(
            &signal[..(0.5 * sr) as usize],
            hz,
            sr,
        );
        let mut v = env.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    println!("measured through the whole chain — each key struck alone at v{VELOCITY}, its own");
    println!("fundamental read with a heterodyne, engine minus pan-pot per channel, dB.");
    println!("`pred` is the lobe's analytic gain from the table above.
");
    println!(
        "{:<5} {:>8} {:>8} {:>8} {:>8} {:>8} {:>9} {:>9}",
        "key", "f0", "L", "R", "predL", "predR", "pairE", "predPair"
    );
    let band = preset.voicing.mics.and_then(|m| m.modal);
    for key in 55u8..=68 {
        let f0 = 440.0 * 2f64.powf((f64::from(key) - 69.0) / 12.0);
        let (ml, mr) = render_key(preset, key);
        let (pl, pr) = render_key(&flat, key);
        let (rl, rr) = render_key(&pair, key);
        let dl = level(&ml, f0) - level(&rl, f0);
        let dr = level(&mr, f0) - level(&rr, f0);
        let pe = 10.0
            * ((10f64.powf(level(&ml, f0) / 10.0) + 10f64.powf(level(&mr, f0) / 10.0))
                / (10f64.powf(level(&pl, f0) / 10.0) + 10f64.powf(level(&pr, f0) / 10.0)))
            .log10();
        let (predl, predr, predp) = match band {
            Some(b) => {
                let g = lobe_gain(
                    f64::from(b.lo_hz),
                    f64::from(b.hi_hz),
                    f64::from(b.lift),
                    f0,
                );
                (
                    20.0 * (1.0 + g).log10(),
                    20.0 * (1.0 - g).abs().log10(),
                    10.0 * (((1.0 + g).powi(2) + (1.0 - g).powi(2)) / 2.0).log10(),
                )
            }
            None => (0.0, 0.0, 0.0),
        };
        println!(
            "{:<5} {:>8.1} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>9.2} {:>9.2}",
            melody::note_name(key),
            f0,
            dl,
            dr,
            predl,
            predr,
            pe,
            predp
        );
    }
    println!();
    Ok(())
}

fn melody_channels(preset: &Preset) -> Result<(), Box<dyn std::error::Error>> {
    lobe_table(preset);
    measured_lift(preset)?;
    line_levels(preset)?;
    println!("== 5. the melody line, per note, per channel ==\n");
    let flat = panpot(preset);
    let library = SampleLibrary::from_sfz(Path::new(SFZ))?;
    let recorded = RecordedKeys::from_library(&library)?;
    let sr = f64::from(SAMPLE_RATE);
    let partial_hz = |key: u8| -> Vec<f64> {
        let params = preset.string_params(key);
        (1..=piano_tuner::series::PARTIALS)
            .map(|k| f64::from(params.partial_freq(k)))
            .collect()
    };
    for window in [Window::Head, Window::Tail] {
        let line = melody::line_for(window);
        let notes: Vec<LineNote> = melody::line_notes_for(window);
        let engine = render_phrase(preset, &line);
        let pan = render_phrase(&flat, &line);
        let reference = reference_phrase(&line)?;
        // One channel at a time, presented as a mono Audio: this instrument
        // asks what each channel does on its own, and `measure_line`'s own
        // `channel` column is 0 by definition on a one-channel signal.
        let per = |signal: &[f32]| {
            let audio = Audio::new(SAMPLE_RATE, vec![signal.to_vec()]).expect("mono");
            melody::measure_line(&audio, sr, &notes, &partial_hz, window)
        };
        let takes: Vec<(&str, Vec<NoteTexture>)> = vec![
            ("ref L", per(&reference.channels[0])),
            ("ref R", per(&reference.channels[1])),
            ("ref mono", per(&reference.mono())),
            ("eng L", per(&engine.channels[0])),
            ("eng R", per(&engine.channels[1])),
            ("eng mono", per(&engine.mono())),
            ("pan L", per(&pan.channels[0])),
            ("pan mono", per(&pan.mono())),
        ];
        println!("--- {} window ---", window.name());
        println!("per-key medians, and each key's DEPARTURE from the line's own trend");
        println!("(the melody gate's own statistic), per channel:\n");
        for (metric, index) in [("roughness", 0usize), ("wobble", 1), ("hf", 2), ("strike", 3)] {
            if metric == "strike" && window == Window::Tail {
                continue;
            }
            println!("  {metric}:");
            print!("  {:<10}", "take");
            let key_list: Vec<u8> = melody::per_key(&takes[0].1).iter().map(|(k, _)| *k).collect();
            for &k in &key_list {
                print!(" {:>14}", melody::note_name(k));
            }
            println!();
            for (name, textures) in &takes {
                let per_key = melody::per_key(textures);
                let values: Vec<f64> = per_key.iter().map(|(_, v)| v[index]).collect();
                // Departure from the line's own trend in pitch: Theil-Sen
                // through (key, value), the melody gate's own device.
                let points: Vec<(f64, f64)> = per_key
                    .iter()
                    .map(|(k, v)| (f64::from(*k), v[index]))
                    .collect();
                let (slope, intercept) = melody::theil_sen(&points);
                print!("  {name:<10}");
                for (i, (k, _)) in per_key.iter().enumerate() {
                    let trend = intercept + slope * f64::from(*k);
                    print!(" {:>7.2}/{:<6.2}", values[i], values[i] - trend);
                }
                println!();
            }
            println!();
        }
        let _ = &recorded;
    }
    Ok(())
}
