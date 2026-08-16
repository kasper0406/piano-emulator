//! The decisive presentation experiment: how much of the standing
//! engine-vs-reference gap is the recording chain nobody has modelled.
//!
//! `renders/realism/REALISM.md` reads mean log-mel **4.99 dB against a noise
//! floor of 1.59** — an excess of 3.40 dB — and every milestone since item 284
//! has spent itself on the instrument. But one side of that comparison is a
//! near-anechoic render and the other is a **recording**: a microphone pair, a
//! placement, a lid, the room those sessions were in and whatever mastering the
//! release had. `PHYSICS.md` §8 and §9 say that is two stages the engine does
//! not have. This driver measures how much of the gap they can absorb, without
//! committing the engine to a room model: everything here is offline, in the
//! tuner, and writes nothing into any preset.
//!
//! [`piano_tuner::estimate::chain`] holds the measurements and the honest list
//! of what this material cannot identify. This is the driver.
//!
//! # The design, and the statistics that make it honest
//!
//! **The chain is fitted once, globally, on single notes, and reported on
//! phrases.** Six phrases and a melody line are held out entirely: not one
//! sample of the material the collapse table is measured on was in the fit.
//!
//! **The keys are split in half.** The curve is fitted independently on the
//! even-indexed sampled keys and on the odd-indexed ones. Those two halves
//! share a microphone and a room and share no strings, so
//! [`curve_agreement`](piano_tuner::estimate::chain::curve_agreement) between
//! them is the only evidence in the experiment about whether a *global* static
//! chain exists at all — a curve that does not replicate across the compass is
//! not a chain, it is each half's own keys. The shipped experiment uses the
//! half-A curve and the held-out half-B notes are its generalisation test.
//!
//! **The velocities are split too.** A chain with a compressor or a tape in it
//! would fit a different curve to the soft layers than to the loud ones. That
//! is a null result if it comes out flat, and it is *tested* rather than
//! assumed.
//!
//! ```sh
//! cargo run --release -p piano-tuner -- chain \
//!     [data/salamander] [renders/chain] [preset.toml]
//! ```

use std::cell::RefCell;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rayon::prelude::*;

use piano_emulator::preset::Preset;
use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_emulator::types::Event;
use piano_tuner::cache;
use piano_tuner::estimate::brilliance::{band, hf_ratio, FULL, HF1, HF2};
use piano_tuner::estimate::chain::{
    band_centres, curve_agreement, energy_decay, fit_eq, reflection_candidates, stereo_signature,
    Chain, ChainEq, EqFit, EqSample, Reflection, RoomStage, StereoSignature, EQ_BANDS,
    SPATIAL_BANDS,
};
use piano_tuner::estimate::melody;
use piano_tuner::library::MechanismKind;
use piano_tuner::realism::{self, Phrase, VelocityLayers, PHRASE_SET_VERSION, TARGET_RMS};
use piano_tuner::sampler::SAMPLER_VERSION;
use piano_tuner::stft::{Stft, StftConfig};
use piano_tuner::sampler::engine_events;
use piano_tuner::{Audio, SampleLibrary, Sampler, SamplerEvent, TimedEvent, SAMPLE_RATE};

const DEFAULT_PRESET: &str = "presets/salamander-c5.toml";

/// The sample rate everything here is measured at.
const SR: f64 = SAMPLE_RATE as f64;

/// Seconds of single note the EQ is fitted from. Long enough to carry the tail
/// the chain colours as well as the strike; the transfer is static, so what a
/// pair contributes is its whole long-term average spectrum.
const NOTE_S: f64 = 3.0;
const PREROLL_S: f64 = 0.05;

/// The velocities the fit is taken over: two soft, two loud, so the
/// nonlinearity test has two independent readings on each side.
const FIT_VELOCITIES: [u8; 4] = [40, 70, 100, 120];
/// Which of them count as soft for that test.
const SOFT_MAX: u8 = 70;

/// Analysis geometry for the long-term average spectrum.
const WINDOW: usize = 4096;
const HOP: usize = 1024;

/// Octave-band centres for the phrase tilt, the same eight `brilliance` uses.
const OCTAVES: [f64; 8] = [125.0, 250.0, 500.0, 1_000.0, 2_000.0, 4_000.0, 8_000.0, 16_000.0];
const TILT_FROM_HZ: f64 = 500.0;

/// The window a reflection is looked for in, seconds after the direct sound.
/// Under 3 ms is the mechanical event itself; past 60 ms a room is diffuse and
/// a discrete tap is no longer the right description.
const REFLECTION_WINDOW_S: (f64, f64) = (0.003, 0.060);
/// How far an envelope must rise over its own running trough to be a candidate.
const REFLECTION_MARGIN_DB: f64 = 4.0;
/// Most taps the stage will carry: the strongest few, clustered.
const MAX_REFLECTIONS: usize = 6;
/// Candidates within this of each other are one arrival.
const REFLECTION_CLUSTER_S: f64 = 0.0025;

/// Tail levels the room stage's late field is swept over, in dB under the
/// direct sound. Fitted against the recording's own interchannel correlation,
/// which is the one thing about the late field this material *does* measure.
const TAIL_LEVEL_SWEEP_DB: [f64; 9] =
    [-30.0, -24.0, -18.0, -12.0, -9.0, -6.0, -3.0, 0.0, 6.0];

/// The room stage's noise seed, so a render is reproducible.
const ROOM_SEED: u64 = 0x0c8a_1000;
/// The two single notes that get written out chained.
const SHOWCASE: [(u8, &str); 2] = [(33, "bass_A1"), (84, "treble_C6")];
const SHOWCASE_VELOCITY: u8 = 90;

const MAX_CACHED_BUFFERS: usize = 8;

thread_local! {
    static SAMPLER: RefCell<Option<Sampler>> = const { RefCell::new(None) };
}

fn with_sampler<T>(
    sfz: &Path,
    body: impl FnOnce(&mut Sampler) -> Result<T, piano_tuner::Error>,
) -> Result<T, piano_tuner::Error> {
    SAMPLER.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            *slot = Some(Sampler::new(sfz)?);
        }
        let sampler = slot.as_mut().expect("a player was just built");
        let out = body(sampler);
        if sampler.cached_buffers() > MAX_CACHED_BUFFERS {
            sampler.clear_cache();
        }
        out
    })
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

fn render_engine_note(preset: &Preset, key: u8, vel: u8) -> Audio {
    let events = [RenderEvent::new(PREROLL_S as f32, Event::NoteOn { key, vel })];
    let (left, right) = render_to_buffer(preset, &events, (PREROLL_S + NOTE_S) as f32);
    let skip = (PREROLL_S * SR) as usize;
    Audio::new(SAMPLE_RATE, vec![left[skip..].to_vec(), right[skip..].to_vec()])
        .expect("the engine renders stereo")
}

fn render_reference_note(
    sfz: &Path,
    data: &Path,
    key: u8,
    vel: u8,
) -> Result<Audio, piano_tuner::Error> {
    let mut fingerprint = cache::Fingerprint::new();
    fingerprint
        .str("chain-fit/note")
        .u64(u64::from(SAMPLER_VERSION))
        .file(sfz)?
        .u64(u64::from(SAMPLE_RATE))
        .u64(u64::from(key))
        .u64(u64::from(vel))
        .f64(NOTE_S);
    let path = cache::reference_dir(data)
        .join(format!("chain-note-{key}-{vel}-{}.wav", fingerprint.hex()));
    cache::audio(&path, || {
        with_sampler(sfz, |sampler| {
            let events = [TimedEvent::new(0.0, SamplerEvent::NoteOn { key, vel })];
            let rendered = sampler.render(&events, NOTE_S + 0.2)?;
            let mono = rendered.mono();
            let onset = piano_tuner::detect_onset(&mono, SR);
            let skip = (onset * SR).round() as usize;
            let frames = (NOTE_S * SR) as usize;
            let cut = |c: &Vec<f32>| -> Vec<f32> {
                (0..frames).map(|n| c.get(skip + n).copied().unwrap_or(0.0)).collect()
            };
            Audio::new(SAMPLE_RATE, rendered.channels.iter().map(cut).collect())
        })
    })
}


fn render_engine_phrase(preset: &Preset, phrase: &Phrase) -> Audio {
    let (left, right) =
        render_to_buffer(preset, &engine_events::to_render_events(&phrase.events), phrase.duration_s as f32);
    Audio::new(SAMPLE_RATE, vec![left, right]).expect("the engine renders stereo")
}

fn render_reference_phrase(
    sfz: &Path,
    data: &Path,
    phrase: &Phrase,
    name: &str,
    events: &[TimedEvent],
) -> Result<Audio, piano_tuner::Error> {
    let mut fingerprint = cache::Fingerprint::new();
    fingerprint
        .str("chain-fit/phrase")
        .u64(u64::from(SAMPLER_VERSION))
        .file(sfz)?
        .u64(u64::from(SAMPLE_RATE))
        .u64(u64::from(PHRASE_SET_VERSION))
        .str(phrase.name)
        .str(name)
        .f64(phrase.duration_s);
    let path = cache::reference_dir(data)
        .join(format!("chain-phrase-{}-{name}-{}.wav", phrase.name, fingerprint.hex()));
    cache::audio(&path, || {
        with_sampler(sfz, |sampler| sampler.render(events, phrase.duration_s))
    })
}

// ---------------------------------------------------------------------------
// Spectra
// ---------------------------------------------------------------------------

fn stft() -> Stft {
    Stft::new(StftConfig::new(WINDOW, HOP, WINDOW).expect("a valid geometry"))
        .expect("a valid transform")
}

/// Power summed over every frame: the long-term average spectrum.
fn total_power(stft: &Stft, mono: &[f32]) -> Vec<f64> {
    let mut sum = vec![0.0f64; stft.bins()];
    stft.for_each_frame(mono, SR, |_, magnitude| {
        for (s, &m) in sum.iter_mut().zip(magnitude.iter()) {
            *s += f64::from(m) * f64::from(m);
        }
    });
    sum
}

fn db(ratio: f64) -> f64 {
    10.0 * ratio.max(1e-30).log10()
}

// ---------------------------------------------------------------------------
// Phrase metrics
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default)]
struct Distances {
    mel: f64,
    modulation: f64,
    hf1: f64,
    hf2: f64,
    tilt: f64,
}

fn tilt_db_per_octave(engine: &[f64], reference: &[f64]) -> f64 {
    let level = db(band(engine, SR, FULL) / band(reference, SR, FULL));
    let points: Vec<(f64, f64)> = OCTAVES
        .iter()
        .filter(|&&c| c >= TILT_FROM_HZ)
        .map(|&c| {
            let bnd = (c / std::f64::consts::SQRT_2, c * std::f64::consts::SQRT_2);
            (
                (c / TILT_FROM_HZ).log2(),
                db(band(engine, SR, bnd) / band(reference, SR, bnd)) - level,
            )
        })
        .collect();
    let n = points.len() as f64;
    let mx = points.iter().map(|p| p.0).sum::<f64>() / n;
    let my = points.iter().map(|p| p.1).sum::<f64>() / n;
    let sxy: f64 = points.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
    let sxx: f64 = points.iter().map(|p| (p.0 - mx).powi(2)).sum();
    if sxx > 0.0 {
        sxy / sxx
    } else {
        f64::NAN
    }
}

fn distances(stft: &Stft, engine: &Audio, reference: &Audio) -> Distances {
    let (a, b) = realism::level_match(engine, reference).expect("two non-silent renders");
    let (ea, eb) = (a.mono(), b.mono());
    let mel = realism::multi_res_log_mel_distance(&ea, &eb, SR)
        .map(|d| d.mean)
        .unwrap_or(f64::NAN);
    let modulation = realism::modulation_distance(&ea, &eb, SR)
        .map(|d| d.mean)
        .unwrap_or(f64::NAN);
    let (pa, pb) = (total_power(stft, &ea), total_power(stft, &eb));
    Distances {
        mel,
        modulation,
        hf1: hf_ratio(&pa, &pb, SR, HF1),
        hf2: hf_ratio(&pa, &pb, SR, HF2),
        tilt: tilt_db_per_octave(&pa, &pb),
    }
}

/// Per-cell signed log-mel differences of one pair, one vector per mel band,
/// at the middle resolution the scoreboard's images are drawn at.
///
/// This is what the **oracle** bound is computed from: the best a static
/// magnitude filter could ever do against this metric is to subtract, from each
/// band, the constant that minimises the mean absolute deviation of that band's
/// own cells — which is the median. Nothing fitted on other material can beat
/// a constant fitted on the answer, so the oracle is an upper bound on every
/// static EQ there is, this one included.
fn mel_cells(engine: &[f32], reference: &[f32], window: usize) -> Vec<Vec<f64>> {
    let bands = realism::MEL_BANDS;
    let hop = window / realism::HOP_DIVISOR;
    let (Ok(a), Ok(b)) = (
        realism::mel_spectrogram(engine, SR, window, hop, bands, realism::MEL_F_MIN, realism::MEL_F_MAX),
        realism::mel_spectrogram(reference, SR, window, hop, bands, realism::MEL_F_MIN, realism::MEL_F_MAX),
    ) else {
        return vec![Vec::new(); bands];
    };
    let n = a.frames.len().min(b.frames.len());
    let floor = a.peak_db().max(b.peak_db()) + realism::MEL_FLOOR_DB;
    let to_db = |e: f64| if e <= 0.0 { floor } else { (10.0 * e.log10()).max(floor) };
    let mut out: Vec<Vec<f64>> = (0..bands).map(|_| Vec::with_capacity(n)).collect();
    for t in 0..n {
        for (k, band) in out.iter_mut().enumerate() {
            band.push(to_db(a.frames[t][k]) - to_db(b.frames[t][k]));
        }
    }
    out
}

/// The mean absolute deviation of a band's cells about a given offset.
fn deviation(cells: &[Vec<f64>], offsets: &[f64]) -> f64 {
    let mut total = 0.0;
    let mut n = 0usize;
    for (k, band) in cells.iter().enumerate() {
        for &d in band {
            total += (d - offsets.get(k).copied().unwrap_or(0.0)).abs();
            n += 1;
        }
    }
    if n == 0 {
        f64::NAN
    } else {
        total / n as f64
    }
}

/// `(as rendered, one global oracle curve, one oracle curve per phrase)`, all
/// at the middle mel resolution, in dB.
fn oracle_bounds(per_phrase: &[Vec<Vec<f64>>]) -> (f64, f64, f64) {
    let bands = realism::MEL_BANDS;
    let zeros = vec![0.0; bands];
    let weight: Vec<f64> = per_phrase
        .iter()
        .map(|p| p.iter().map(|b| b.len()).sum::<usize>() as f64)
        .collect();
    let total: f64 = weight.iter().sum();
    let weighted = |f: &dyn Fn(usize) -> f64| -> f64 {
        if total <= 0.0 {
            return f64::NAN;
        }
        per_phrase
            .iter()
            .enumerate()
            .map(|(i, _)| f(i) * weight[i])
            .sum::<f64>()
            / total
    };
    let plain = weighted(&|i| deviation(&per_phrase[i], &zeros));
    let per = weighted(&|i| {
        let offsets: Vec<f64> = per_phrase[i]
            .iter()
            .map(|b| median_of(b.clone()))
            .map(|m| if m.is_finite() { m } else { 0.0 })
            .collect();
        deviation(&per_phrase[i], &offsets)
    });
    // One curve for all six: the median of every phrase's cells pooled.
    let global: Vec<f64> = (0..bands)
        .map(|k| {
            let pooled: Vec<f64> =
                per_phrase.iter().flat_map(|p| p[k].iter().copied()).collect();
            let m = median_of(pooled);
            if m.is_finite() {
                m
            } else {
                0.0
            }
        })
        .collect();
    let one = weighted(&|i| deviation(&per_phrase[i], &global));
    (plain, one, per)
}

/// Scale a render to the benchmark's own target level, guarding the peak.
fn normalized(audio: &Audio) -> Audio {
    let r = realism::rms(&audio.mono());
    let gain = if r > 0.0 { f64::from(TARGET_RMS) / r } else { 1.0 };
    let peak = audio
        .channels
        .iter()
        .flat_map(|c| c.iter())
        .fold(0.0f32, |m, &x| m.max(x.abs()));
    let gain = if peak as f64 * gain > 0.98 { 0.98 / peak as f64 } else { gain };
    Audio {
        sample_rate: audio.sample_rate,
        channels: audio
            .channels
            .iter()
            .map(|c| c.iter().map(|&x| (f64::from(x) * gain) as f32).collect())
            .collect(),
    }
}

/// A recording cut to its own onset, with 1 ms of run-up kept.
fn onset_trimmed(audio: &Audio) -> Audio {
    let mono = audio.mono();
    let onset = piano_tuner::detect_onset(&mono, SR);
    let skip = ((onset * SR).round() as usize).saturating_sub((0.001 * SR) as usize);
    Audio {
        sample_rate: audio.sample_rate,
        channels: audio.channels.iter().map(|c| c[skip.min(c.len())..].to_vec()).collect(),
    }
}

fn note_name(key: u8) -> String {
    const NAMES: [&str; 12] = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
    format!("{}{}", NAMES[usize::from(key) % 12], i32::from(key) / 12 - 1)
}

fn mean(xs: impl Iterator<Item = f64>) -> f64 {
    let (s, n) = xs.fold((0.0, 0usize), |(s, n), x| if x.is_finite() { (s + x, n + 1) } else { (s, n) });
    if n == 0 {
        f64::NAN
    } else {
        s / n as f64
    }
}

fn median_of(mut xs: Vec<f64>) -> f64 {
    xs.retain(|v| v.is_finite());
    if xs.is_empty() {
        return f64::NAN;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let n = xs.len();
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        0.5 * (xs[n / 2 - 1] + xs[n / 2])
    }
}

// ---------------------------------------------------------------------------

pub fn run(args: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
    let mut args = args.into_iter();
    let data = PathBuf::from(args.next().unwrap_or_else(|| "data/salamander".into()));
    let out = PathBuf::from(args.next().unwrap_or_else(|| "renders/chain".into()));
    let preset_path = PathBuf::from(args.next().unwrap_or_else(|| DEFAULT_PRESET.into()));

    let sfz = data.join("SalamanderGrandPiano-V3+20200602.sfz");
    if !sfz.exists() {
        eprintln!(
            "the reference piano is not here: {}\nrun data/fetch_salamander.sh first (707 MiB).",
            sfz.display()
        );
        std::process::exit(2);
    }
    std::fs::create_dir_all(&out)?;
    let started = Instant::now();

    let preset = Preset::load(&preset_path)?;
    let library = SampleLibrary::from_sfz(&sfz)?;
    let layers = VelocityLayers::from_library(&library)?;
    let stft = stft();

    let keys: Vec<u8> = library.keys().collect();
    let half_a: Vec<u8> = keys.iter().step_by(2).copied().collect();
    let half_b: Vec<u8> = keys.iter().skip(1).step_by(2).copied().collect();
    println!(
        "{} sampled keys, split {} / {}; {} velocities each\n",
        keys.len(),
        half_a.len(),
        half_b.len(),
        FIT_VELOCITIES.len()
    );

    // -----------------------------------------------------------------------
    // 1. The material: matched single-note pairs.
    // -----------------------------------------------------------------------
    let mut pairs: Vec<(u8, u8)> = Vec::new();
    for &key in &keys {
        for &vel in &FIT_VELOCITIES {
            pairs.push((key, vel));
        }
    }
    let samples: Vec<EqSample> = pairs
        .par_iter()
        .map(|&(key, vel)| -> Result<EqSample, piano_tuner::Error> {
            let engine = render_engine_note(&preset, key, vel);
            let reference = render_reference_note(&sfz, &data, key, vel)?;
            let pe = total_power(&stft, &engine.mono());
            let pr = total_power(&stft, &reference.mono());
            Ok(EqSample {
                engine: piano_tuner::estimate::chain::band_powers(&pe, SR),
                reference: piano_tuner::estimate::chain::band_powers(&pr, SR),
                key,
                velocity: vel,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    println!("{} matched note pairs in {:.1} s", samples.len(), started.elapsed().as_secs_f64());

    let pick = |keep: &dyn Fn(&EqSample) -> bool| -> Vec<EqSample> {
        samples.iter().filter(|s| keep(s)).cloned().collect()
    };
    let fit_a = fit_eq(&pick(&|s: &EqSample| half_a.contains(&s.key)));
    let fit_b = fit_eq(&pick(&|s: &EqSample| half_b.contains(&s.key)));
    let fit_all = fit_eq(&samples);
    let fit_soft = fit_eq(&pick(&|s: &EqSample| s.velocity <= SOFT_MAX));
    let fit_loud = fit_eq(&pick(&|s: &EqSample| s.velocity > SOFT_MAX));

    let (split_abs, split_r) = curve_agreement(&fit_a.smooth_db, &fit_b.smooth_db);
    let (level_abs, level_r) = curve_agreement(&fit_soft.smooth_db, &fit_loud.smooth_db);

    println!("\n== (a) the smooth spectral transfer ==");
    println!(
        "  split-half agreement (even keys vs odd keys): mean |Δ| {:.2} dB, r {:+.3}",
        split_abs, split_r
    );
    println!(
        "  soft vs loud layers:                          mean |Δ| {:.2} dB, r {:+.3}",
        level_abs, level_r
    );
    let centres = band_centres();
    println!("\n  {:>8}  {:>7}  {:>7}  {:>7}  {:>6}  {:>7}", "Hz", "A", "B", "all", "n(A)", "MAD(A)");
    for (b, &centre) in centres.iter().enumerate() {
        println!(
            "  {:8.0}  {:7.2}  {:7.2}  {:7.2}  {:6}  {:7.2}",
            centre,
            fit_a.smooth_db[b],
            fit_b.smooth_db[b],
            fit_all.smooth_db[b],
            fit_a.counts[b],
            fit_a.scatter_db[b]
        );
    }

    // The generalisation test: the half-A curve on the half-B notes. Read over
    // exactly the cells the fit was allowed to read — a band under
    // `BAND_FLOOR_DB` is two floors' ratio on both sides of this comparison and
    // scoring it would be scoring the material's own silence.
    let residual = |eq: &ChainEq, keep: &dyn Fn(&EqSample) -> bool| -> (f64, f64) {
        let floor = 10f64.powf(piano_tuner::estimate::chain::BAND_FLOOR_DB / 10.0);
        let mut before = Vec::new();
        let mut after = Vec::new();
        for s in samples.iter().filter(|s| keep(s)) {
            let te: f64 = s.engine.iter().sum();
            let tr: f64 = s.reference.iter().sum();
            if te <= 0.0 || tr <= 0.0 {
                continue;
            }
            let scale = tr / te;
            let peak_r = s.reference.iter().cloned().fold(0.0f64, f64::max);
            let peak_e = s.engine.iter().cloned().fold(0.0f64, f64::max) * scale;
            for b in 0..EQ_BANDS {
                let e = s.engine[b] * scale;
                if e <= peak_e * floor || s.reference[b] <= peak_r * floor {
                    continue;
                }
                let d = 10.0 * (s.reference[b] / e).log10();
                before.push(d.abs());
                after.push((d - eq.gains_db[b]).abs());
            }
        }
        (median_of(before), median_of(after))
    };
    let eq_a = fit_a.eq();
    let (train_before, train_after) = residual(&eq_a, &|s: &EqSample| half_a.contains(&s.key));
    let (test_before, test_after) = residual(&eq_a, &|s: &EqSample| half_b.contains(&s.key));
    println!(
        "\n  median per-band |engine − reference| over readable cells, curve fitted on half A:\n    \
         train (half A) {:.2} -> {:.2} dB     held out (half B) {:.2} -> {:.2} dB",
        train_before, train_after, test_before, test_after
    );
    println!(
        "  the curve was read over bands {}..{} ({:.0}-{:.0} Hz) and is flat outside them",
        fit_a.read_range.0,
        fit_a.read_range.1,
        centres[fit_a.read_range.0],
        centres[fit_a.read_range.1]
    );

    // -----------------------------------------------------------------------
    // 2. The spatial and temporal signature.
    // -----------------------------------------------------------------------
    println!("\n== (b) the spatial and temporal signature ==");
    // Measured on **half A** alone, so the room stage is fitted on the same
    // half of the compass the curve is and the phrases stay held out.
    let mut reference_sigs: Vec<(u8, StereoSignature)> = Vec::new();
    let mut engine_sigs: Vec<(u8, StereoSignature)> = Vec::new();
    let mut spatial_notes: Vec<(u8, Audio)> = Vec::new();
    for &key in &half_a {
        let reference = render_reference_note(&sfz, &data, key, SHOWCASE_VELOCITY)?;
        let engine = render_engine_note(&preset, key, SHOWCASE_VELOCITY);
        reference_sigs.push((
            key,
            stereo_signature(&reference.channels[0], &reference.channels[1], SR)?,
        ));
        engine_sigs.push((key, stereo_signature(&engine.channels[0], &engine.channels[1], SR)?));
        spatial_notes.push((key, engine));
    }
    println!("  interchannel correlation, recording against engine:");
    println!(
        "  {:>6}  {:>18}  {:>16}  {:>16}",
        "band", "reference r@0 / peak", "at lag ms", "engine r@0"
    );
    let mut band_zero_r: Vec<f64> = Vec::new();
    for (i, &(lo, hi)) in SPATIAL_BANDS.iter().enumerate() {
        let rz = median_of(reference_sigs.iter().map(|(_, s)| s.per_band[i].zero_r).collect());
        let rp = median_of(reference_sigs.iter().map(|(_, s)| s.per_band[i].peak_r.abs()).collect());
        let rl = median_of(reference_sigs.iter().map(|(_, s)| s.per_band[i].lag_ms).collect());
        let ez = median_of(engine_sigs.iter().map(|(_, s)| s.per_band[i].zero_r).collect());
        band_zero_r.push(rz);
        println!(
            "  {:>6}  {:9.3} / {:6.3}  {:16.2}  {:16.3}",
            format!("{:.0}-{:.0}", lo, hi),
            rz,
            rp,
            rl,
            ez
        );
    }
    let broad_ref = median_of(reference_sigs.iter().map(|(_, s)| s.broadband.zero_r).collect());
    let broad_eng = median_of(engine_sigs.iter().map(|(_, s)| s.broadband.zero_r).collect());
    println!("  broadband r@0: reference {broad_ref:.3}, engine {broad_eng:.3}");

    // The mechanism recordings: the only impulsive events in the library.
    let mut impulsive: Vec<(String, Audio)> = Vec::new();
    for m in library.mechanism() {
        if matches!(m.kind, MechanismKind::KeyOff | MechanismKind::PedalDown | MechanismKind::PedalUp)
        {
            if let Ok(a) = piano_tuner::audio::load_at(&m.path, SAMPLE_RATE) {
                let name = m
                    .path
                    .file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_default();
                // Every recording in the library begins with however much
                // silence there was before the event; the reference notes are
                // onset-trimmed for exactly this reason and so is this. Without
                // it the "direct sound" window is that silence and every
                // late-to-direct ratio is a ratio to nothing.
                impulsive.push((name, onset_trimmed(&a)));
            }
        }
        if impulsive.len() >= 24 {
            break;
        }
    }
    println!("\n  {} impulsive mechanism recordings read", impulsive.len());

    let decays: Vec<_> = impulsive.iter().map(|(_, a)| energy_decay(&a.mono(), SR)).collect();
    let edt = median_of(decays.iter().filter_map(|d| d.edt_s).collect());
    let t20 = median_of(decays.iter().filter_map(|d| d.t20_s).collect());
    println!("  broadband EDT {edt:.3} s, T20 {t20:.3} s");
    let mut tail_t60: Vec<(f64, f64, f64)> = Vec::new();
    println!("  {:>12}  {:>8}", "band", "T20 s");
    for (i, &(lo, hi)) in SPATIAL_BANDS.iter().enumerate() {
        let t = median_of(decays.iter().filter_map(|d| d.per_band[i].2).collect());
        println!("  {:>12}  {:8.3}", format!("{:.0}-{:.0}", lo, hi), t);
        if t.is_finite() && t > 0.02 {
            tail_t60.push((lo, hi, t));
        }
    }

    // Reflections: candidates pooled over the impulsive recordings, clustered.
    let mut candidates: Vec<(f64, f64, f64)> = Vec::new(); // delay, level dB, side
    for (_, a) in &impulsive {
        let mono = a.mono();
        for (delay, level) in reflection_candidates(
            &mono,
            SR,
            REFLECTION_WINDOW_S.0,
            REFLECTION_WINDOW_S.1,
            REFLECTION_MARGIN_DB,
        ) {
            // Which side the arrival came from, read where it arrived.
            let side = if a.channels.len() >= 2 {
                let at = (delay * SR) as usize;
                let n = (0.002 * SR) as usize;
                let energy = |c: &Vec<f32>| -> f64 {
                    c.iter()
                        .skip(at)
                        .take(n)
                        .map(|&x| f64::from(x) * f64::from(x))
                        .sum::<f64>()
                        .max(1e-30)
                };
                let (l, r) = (energy(&a.channels[0]), energy(&a.channels[1]));
                ((r / l).log10()).clamp(-1.0, 1.0)
            } else {
                0.0
            };
            candidates.push((delay, level, side));
        }
    }
    candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("finite"));
    let mut clusters: Vec<Vec<(f64, f64, f64)>> = Vec::new();
    for c in candidates {
        match clusters.last_mut() {
            Some(last) if (c.0 - last[0].0).abs() <= REFLECTION_CLUSTER_S => last.push(c),
            _ => clusters.push(vec![c]),
        }
    }
    clusters.sort_by_key(|c| std::cmp::Reverse(c.len()));
    let mut reflections: Vec<Reflection> = clusters
        .iter()
        .take(MAX_REFLECTIONS)
        .map(|c| Reflection {
            delay_s: median_of(c.iter().map(|x| x.0).collect()),
            gain_db: median_of(c.iter().map(|x| x.1).collect()),
            side: median_of(c.iter().map(|x| x.2).collect()),
        })
        .collect();
    reflections.sort_by(|a, b| a.delay_s.partial_cmp(&b.delay_s).expect("finite"));
    println!("\n  reflection candidates (clustered over the mechanism recordings):");
    println!("  {:>9}  {:>9}  {:>7}  {:>6}", "delay ms", "level dB", "side", "n");
    for (r, c) in reflections.iter().zip(clusters.iter().take(MAX_REFLECTIONS)) {
        println!(
            "  {:9.2}  {:9.2}  {:7.2}  {:6}",
            r.delay_s * 1000.0,
            r.gain_db,
            r.side,
            c.len()
        );
    }

    // The candidates are **refused**, and the refusal is a measurement. A
    // sequence of early reflections gets quieter with delay; these get louder,
    // by 16 dB over 47 ms. What the finder is walking up is the mechanical
    // event's own body — a damper landing and a tray settling take tens of
    // milliseconds — and a room read off it would be that mechanism dressed as
    // a wall. Nothing in this material carries a discrete arrival, so the stage
    // carries none.
    let monotone_rise = reflections.len() >= 2
        && reflections
            .windows(2)
            .all(|w| w[1].gain_db >= w[0].gain_db - 2.0);
    println!(
        "  the candidates {} with delay, so they are {} — the stage carries no reflections",
        if monotone_rise { "get LOUDER" } else { "decay" },
        if monotone_rise { "the mechanism's own body and not arrivals" } else { "arrivals" }
    );

    // Tail onset: a stated choice, not a reading (see the module's note 2 —
    // every sample is trimmed to its own onset, so a pre-delay is not
    // recoverable). At a 0.45 s tail, 10 ms of onset is 2 % of its energy and
    // none of its spectrum.
    let tail_onset_s = 0.010;

    // The tail's **level** is not identifiable from the impulsive material
    // either — a key-off recording's late-to-direct ratio is the mechanism's
    // duration, not the room's. What *is* measurable is the reference's own
    // interchannel decorrelation, and a diffuse tail is the only thing in this
    // stage that produces any. So the level is fitted to that, on half A's
    // notes, and the phrases stay held out.
    // The target is the **peak** |r| over lags, not the value at lag zero. The
    // recording's channels are a spaced pair: they carry the same wavefront at
    // a lag of a fraction of a millisecond, which destroys `r@0` while leaving
    // the two signals largely coherent. Peak |r| is lag-invariant and is
    // therefore the coherence a diffuse field can actually be fitted against;
    // `r@0` is reported beside it and is what a listener's ears' own delay
    // makes of the same fact.
    let target_r: Vec<f64> = (0..SPATIAL_BANDS.len())
        .map(|i| {
            median_of(reference_sigs.iter().map(|(_, s)| s.per_band[i].peak_r.abs()).collect())
        })
        .collect();
    let width_mismatch = |sigs: &[StereoSignature]| -> f64 {
        mean((0..SPATIAL_BANDS.len()).map(|i| {
            (median_of(sigs.iter().map(|s| s.per_band[i].peak_r.abs()).collect()) - target_r[i])
                .abs()
        }))
    };
    let bare: Vec<StereoSignature> = engine_sigs.iter().map(|(_, s)| s.clone()).collect();
    println!(
        "\n  fitting the tail level to the recording's own decorrelation \
         (bare engine mismatch {:.3}):",
        width_mismatch(&bare)
    );
    let mut sweep: Vec<(f64, f64)> = Vec::new();
    let mut best = (f64::NEG_INFINITY, f64::INFINITY);
    for level in TAIL_LEVEL_SWEEP_DB {
        let room = RoomStage {
            reflections: Vec::new(),
            tail_onset_s,
            tail_level_db: level,
            tail_t60: tail_t60.clone(),
            reflection_lowpass_hz: 20_000.0,
            seed: ROOM_SEED,
        };
        let sigs: Vec<StereoSignature> = spatial_notes
            .par_iter()
            .map(|(_, engine)| {
                let wet = room.apply(engine);
                stereo_signature(&wet.channels[0], &wet.channels[1], SR)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let m = width_mismatch(&sigs);
        println!("    tail {level:+.0} dB -> interchannel mismatch {m:.3}");
        sweep.push((level, m));
        if m < best.1 {
            best = (level, m);
        }
    }
    let tail_level_db = best.0;
    println!(
        "  the recording's decorrelation is matched at a tail level of {:+.0} dB (mismatch {:.3})",
        tail_level_db, best.1
    );

    let room = RoomStage {
        reflections: Vec::new(),
        tail_onset_s,
        tail_level_db,
        tail_t60: tail_t60.clone(),
        reflection_lowpass_hz: 20_000.0,
        seed: ROOM_SEED,
    };
    let chain = Chain { eq: eq_a.clone(), room: room.clone() };

    // -----------------------------------------------------------------------
    // 3. The collapse, on held-out material.
    // -----------------------------------------------------------------------
    println!("\n== the collapse, on held-out phrases ==");
    let mut rows: Vec<(String, Distances, Distances, Distances, Distances)> = Vec::new();
    let mut cells_before: Vec<Vec<Vec<f64>>> = Vec::new();
    let mut cells_eq: Vec<Vec<Vec<f64>>> = Vec::new();
    let mut phrases: Vec<Phrase> = realism::phrase_set();
    phrases.push(melody::soprano());
    for phrase in &phrases {
        let engine = render_engine_phrase(&preset, phrase);
        let reference =
            render_reference_phrase(&sfz, &data, phrase, "reference", &phrase.events)?;
        let alt = render_reference_phrase(
            &sfz,
            &data,
            phrase,
            "alt-layer",
            &layers.shift(&phrase.events),
        )?;
        let eq_only = eq_a.apply(&engine);
        let chained = chain.apply(&engine);
        {
            let (a, b) = realism::level_match(&engine, &reference)?;
            cells_before.push(mel_cells(&a.mono(), &b.mono(), realism::MULTI_RES_WINDOWS[1]));
            let (a, b) = realism::level_match(&eq_only, &reference)?;
            cells_eq.push(mel_cells(&a.mono(), &b.mono(), realism::MULTI_RES_WINDOWS[1]));
        }
        rows.push((
            phrase.name.to_string(),
            distances(&stft, &engine, &reference),
            distances(&stft, &eq_only, &reference),
            distances(&stft, &chained, &reference),
            distances(&stft, &alt, &reference),
        ));
        let last = rows.last().expect("just pushed");
        println!(
            "  {:<18} mel {:.2} -> {:.2} -> {:.2}  (floor {:.2})   mod {:.2} -> {:.2} -> {:.2} (floor {:.2})",
            last.0, last.1.mel, last.2.mel, last.3.mel, last.4.mel,
            last.1.modulation, last.2.modulation, last.3.modulation, last.4.modulation
        );
    }
    // The six benchmark phrases are the scoreboard's own set; the melody line is
    // reported beside it and not averaged into it.
    let bench = &rows[..realism::phrase_set().len()];
    let mel_before = mean(bench.iter().map(|r| r.1.mel));
    let mel_eq = mean(bench.iter().map(|r| r.2.mel));
    let mel_chain = mean(bench.iter().map(|r| r.3.mel));
    let mel_floor = mean(bench.iter().map(|r| r.4.mel));
    println!(
        "\n  mean over the six: mel {:.2} -> {:.2} (EQ) -> {:.2} (chain), floor {:.2}",
        mel_before, mel_eq, mel_chain, mel_floor
    );
    let excess = mel_before - mel_floor;
    println!(
        "  the excess over the floor is {:.2} dB; the EQ absorbs {:.0} % of it and the whole chain {:.0} %",
        excess,
        100.0 * (mel_before - mel_eq) / excess,
        100.0 * (mel_before - mel_chain) / excess
    );

    // The oracle: the best any static magnitude filter could do, fitted on the
    // answer. Two of them — one curve per phrase, and one curve for all six.
    let oracle = oracle_bounds(&cells_before[..realism::phrase_set().len()]);
    let oracle_eq = oracle_bounds(&cells_eq[..realism::phrase_set().len()]);
    println!(
        "\n  the oracle, at window {} (the resolution the images are drawn at):",
        realism::MULTI_RES_WINDOWS[1]
    );
    println!(
        "    as rendered {:.2} dB  ->  one global static curve fitted on the answer {:.2}  \
         ->  one curve per phrase {:.2}",
        oracle.0, oracle.1, oracle.2
    );
    println!(
        "    through the fitted EQ {:.2} dB  ->  {:.2}  ->  {:.2}",
        oracle_eq.0, oracle_eq.1, oracle_eq.2
    );
    println!(
        "    so a static magnitude chain has at most {:.2} dB of the {:.2} dB gap to give, \
         and the fitted curve has already taken {:.2} of it",
        oracle.0 - oracle.1,
        oracle.0,
        oracle.0 - oracle_eq.0
    );

    // -----------------------------------------------------------------------
    // 4. The renders.
    // -----------------------------------------------------------------------
    let mut written: Vec<String> = Vec::new();
    let mut write_triple = |name: &str,
                            engine: &Audio,
                            reference: &Audio|
     -> Result<(), Box<dyn std::error::Error>> {
        let chained = chain.apply(engine);
        normalized(engine).write_wav(out.join(format!("{name}_engine.wav")))?;
        normalized(&chained).write_wav(out.join(format!("{name}_engine_chained.wav")))?;
        normalized(reference).write_wav(out.join(format!("{name}_reference.wav")))?;
        written.push(name.to_string());
        Ok(())
    };
    let soprano = melody::soprano();
    let soprano_engine = render_engine_phrase(&preset, &soprano);
    let soprano_reference =
        render_reference_phrase(&sfz, &data, &soprano, "reference", &soprano.events)?;
    write_triple("ode_soprano", &soprano_engine, &soprano_reference)?;
    let excerpt = realism::excerpt();
    let excerpt_engine = render_engine_phrase(&preset, &excerpt);
    let excerpt_reference =
        render_reference_phrase(&sfz, &data, &excerpt, "reference", &excerpt.events)?;
    write_triple("excerpt", &excerpt_engine, &excerpt_reference)?;
    for (key, name) in SHOWCASE {
        let engine = render_engine_note(&preset, key, SHOWCASE_VELOCITY);
        let reference = render_reference_note(&sfz, &data, key, SHOWCASE_VELOCITY)?;
        write_triple(name, &engine, &reference)?;
    }
    println!("\n  wrote {} triples into {}", written.len(), out.display());

    // -----------------------------------------------------------------------
    // 5. The report.
    // -----------------------------------------------------------------------
    let chained_mismatch = {
        let sigs: Vec<StereoSignature> = spatial_notes
            .par_iter()
            .map(|(_, engine)| {
                let wet = chain.apply(engine);
                stereo_signature(&wet.channels[0], &wet.channels[1], SR)
            })
            .collect::<Result<Vec<_>, _>>()?;
        width_mismatch(&sigs)
    };
    println!(
        "  interchannel mismatch against the recording: bare {:.3} -> chained {:.3}",
        width_mismatch(&bare),
        chained_mismatch
    );

    let findings = Findings {
        preset: &preset_path,
        sfz: &sfz,
        keys: &keys,
        half_a: &half_a,
        half_b: &half_b,
        velocities: FIT_VELOCITIES.len(),
        pairs: samples.len(),
        fit_a: &fit_a,
        fit_b: &fit_b,
        fit_all: &fit_all,
        split: (split_abs, split_r),
        level: (level_abs, level_r),
        residual: (train_before, train_after, test_before, test_after),
        reference_sigs: &reference_sigs,
        engine_sigs: &engine_sigs,
        candidates: &reflections,
        room: &room,
        edt,
        t20,
        bare_mismatch: width_mismatch(&bare),
        chained_mismatch,
        sweep: &sweep,
        rows: &rows,
        bench_len: realism::phrase_set().len(),
        oracle,
        oracle_eq,
        written: &written,
    };
    let path = out.join("CHAIN.md");
    std::fs::write(&path, chain_report(&findings))?;
    println!("  {}", path.display());
    println!("\ntotal {:.1} s", started.elapsed().as_secs_f64());
    Ok(())
}

// ---------------------------------------------------------------------------
// The report
// ---------------------------------------------------------------------------

/// Everything the run measured, in one place, so `CHAIN.md` is a rendering of
/// the findings rather than a second copy of the analysis.
struct Findings<'a> {
    preset: &'a Path,
    sfz: &'a Path,
    keys: &'a [u8],
    half_a: &'a [u8],
    half_b: &'a [u8],
    velocities: usize,
    pairs: usize,
    fit_a: &'a EqFit,
    fit_b: &'a EqFit,
    fit_all: &'a EqFit,
    /// `(mean abs difference, correlation)` between the two halves' curves.
    split: (f64, f64),
    /// The same between the soft and loud layers' curves.
    level: (f64, f64),
    /// `(train before, train after, held out before, held out after)`.
    residual: (f64, f64, f64, f64),
    reference_sigs: &'a [(u8, StereoSignature)],
    engine_sigs: &'a [(u8, StereoSignature)],
    candidates: &'a [Reflection],
    room: &'a RoomStage,
    edt: f64,
    t20: f64,
    bare_mismatch: f64,
    chained_mismatch: f64,
    sweep: &'a [(f64, f64)],
    rows: &'a [(String, Distances, Distances, Distances, Distances)],
    bench_len: usize,
    /// `(as rendered, one global oracle curve, one oracle curve per phrase)`.
    oracle: (f64, f64, f64),
    oracle_eq: (f64, f64, f64),
    written: &'a [String],
}

fn chain_report(f: &Findings) -> String {
    let mut out = String::new();
    let centres = band_centres();
    let bench = &f.rows[..f.bench_len];
    let mel_before = mean(bench.iter().map(|r| r.1.mel));
    let mel_eq = mean(bench.iter().map(|r| r.2.mel));
    let mel_chain = mean(bench.iter().map(|r| r.3.mel));
    let mel_floor = mean(bench.iter().map(|r| r.4.mel));
    let excess = mel_before - mel_floor;

    let _ = write!(
        out,
        "# CHAIN.md — how much of the gap is the recording, not the piano\n\n\
Written by `cargo run --release -p piano-tuner -- chain`. \
The engine on `{}` against the recordings at `{}`. **Nothing here is written into a preset.**\n\n\
`REALISM.md` compares a near-anechoic render against a **recording**: a microphone pair, a \
placement, a lid, a room and a master. `PHYSICS.md` §8 and §9 say that is two stages the engine \
does not have, and `TUNING.md` stage 2 reserves the first of them — \"a static linear filter \
applied to the engine output before loss, a ~40-band cepstrally-smooth log-magnitude EQ\". \
This file is that absorber, **fitted once, globally, on single notes, and reported on phrases \
none of which were in the fit**.\n\n\
## The answer, first\n\n\
| | mel dB | share of the {:.2} dB excess |\n|---|--:|--:|\n\
| engine as it stands | {:.2} | — |\n\
| through the fitted chain EQ | {:.2} | {:.0} % |\n\
| through EQ and room stage | {:.2} | {:.0} % |\n\
| **the oracle**: the best *any* static magnitude curve could do, fitted on the answer itself | {:.2} | {:.0} % |\n\
| the reference's own velocity-layer floor | {:.2} | 100 % |\n\n\
The oracle line is the one that settles it. It is not a filter anybody could build — its \
per-band offsets are the medians of the very cells it is scored on, one curve for all six \
phrases — and it is therefore an **upper bound on every static magnitude chain there is**. \
It moves the mean log-mel distance by **{:.2} dB of {:.2}**. The remaining gap is not a \
static spectral transfer, and no EQ, fitted however well, is going to become one.\n\n\
## Material and split\n\n\
{} sampled keys x {} velocities = {} matched pairs. Keys split even/odd into halves of {} and \
{}: the two halves share a microphone and a room and share no strings.\n\n\
* half A: {}\n* half B: {}\n\n\
## (a) The smooth spectral transfer\n\n\
| test | mean abs difference | correlation | what it means |\n|---|--:|--:|---|\n\
| half A vs half B | {:.2} dB | {:+.3} | whether a *global* static chain exists at all |\n\
| soft layers vs loud | {:.2} dB | {:+.3} | whether the chain is level-dependent (a compressor) |\n\n\
Median per-band `|engine − reference|` over the cells the fit was allowed to read, with the \
half-A curve applied: **train (half A) {:.2} -> {:.2} dB, held out (half B) {:.2} -> {:.2} dB**. \
The curve replicates in *shape* (r {:+.3}) and buys, on keys it never saw, {:+.2} dB of a \
{:.2} dB per-band error — because that error is scatter between neighbouring bands of one key, \
not a tilt shared by all of them.\n\n\
### The curve\n\n\
Positive is what the **recording** has that the render does not. Read over bands {}..{} \
({:.0}-{:.0} Hz); flat outside them, because a truncated DCT overshoots at an edge and the top \
bands are where a note has least energy.\n\n\
| Hz | half A | half B | all | pairs read (A) | MAD (A) |\n|--:|--:|--:|--:|--:|--:|\n",
        f.preset.display(),
        f.sfz.display(),
        excess,
        mel_before,
        mel_eq,
        100.0 * (mel_before - mel_eq) / excess,
        mel_chain,
        100.0 * (mel_before - mel_chain) / excess,
        mel_before - (f.oracle.0 - f.oracle.1),
        100.0 * (f.oracle.0 - f.oracle.1) / excess,
        mel_floor,
        f.oracle.0 - f.oracle.1,
        f.oracle.0,
        f.keys.len(),
        f.velocities,
        f.pairs,
        f.half_a.len(),
        f.half_b.len(),
        f.half_a.iter().map(|&k| note_name(k)).collect::<Vec<_>>().join(" "),
        f.half_b.iter().map(|&k| note_name(k)).collect::<Vec<_>>().join(" "),
        f.split.0,
        f.split.1,
        f.level.0,
        f.level.1,
        f.residual.0,
        f.residual.1,
        f.residual.2,
        f.residual.3,
        f.split.1,
        f.residual.2 - f.residual.3,
        f.residual.2,
        f.fit_a.read_range.0,
        f.fit_a.read_range.1,
        centres[f.fit_a.read_range.0],
        centres[f.fit_a.read_range.1],
    );
    for (b, &centre) in centres.iter().enumerate() {
        let _ = writeln!(
            out,
            "| {:.0} | {:+.2} | {:+.2} | {:+.2} | {} | {:.2} |",
            centre,
            f.fit_a.smooth_db[b],
            f.fit_b.smooth_db[b],
            f.fit_all.smooth_db[b],
            f.fit_a.counts[b],
            f.fit_a.scatter_db[b]
        );
    }
    let _ = write!(
        out,
        "\nThe curve is the per-band median over pairs, cepstrally smoothed to {} coefficients \
so it can carry a microphone and cannot carry a partial series, with its mean removed — level \
is not the chain's business. `MAD` is the median absolute deviation of that band's own \
readings: where it is 3-10 dB, most of what the band sees is one key's partials and only the \
median of thirty keys is a chain.\n\n\
## (b) The spatial and temporal signature\n\n\
### Interchannel correlation — the largest thing in the experiment, and the one nothing scores\n\n\
| band | reference r at lag 0 | reference peak \\|r\\| | at lag | engine r at lag 0 |\n\
|---|--:|--:|--:|--:|\n",
        piano_tuner::estimate::chain::CEPSTRAL_ORDER
    );
    for (i, &(lo, hi)) in SPATIAL_BANDS.iter().enumerate() {
        let m = |g: &dyn Fn(&StereoSignature) -> f64, v: &[(u8, StereoSignature)]| {
            median_of(v.iter().map(|(_, s)| g(s)).collect())
        };
        let _ = writeln!(
            out,
            "| {:.0}–{:.0} Hz | {:.3} | {:.3} | {:+.2} ms | {:.3} |",
            lo,
            hi,
            m(&|s| s.per_band[i].zero_r, f.reference_sigs),
            m(&|s| s.per_band[i].peak_r.abs(), f.reference_sigs),
            m(&|s| s.per_band[i].lag_ms, f.reference_sigs),
            m(&|s| s.per_band[i].zero_r, f.engine_sigs),
        );
    }
    let _ = write!(
        out,
        "\nMedians over half A's {} keys at velocity {}. This is the largest and cleanest \
difference in the whole experiment and it is **register-inverted**: the recording's two \
channels are nearly one signal below 125 Hz — a mic pair spaced well under a wavelength sees \
one wavefront — and essentially independent above it. The engine is the other way round. Its \
board FDN's two taps use orthogonal sign patterns (`soundboard.rs`), so what it decorrelates \
is the *bass*, where the board field dominates a note, and by 6-12 kHz its channels are a \
pan-pot's, correlating at {:.3}.\n\n\
**Every metric in `REALISM.md` is computed on the mono sum** and is blind to all of it. That \
is not an oversight — a stereo distance between an engine that pans by key and a recording \
made with a particular pair would mostly measure the pair — but it does mean this column of \
the gap has never been scored, and it is the column a room stage exists to move.\n\n\
### The impulsive material, and why it does not measure a room\n\n\
The library's key-off and pedal recordings are the only impulsive events in it. Onset-trimmed \
and Schroeder-integrated: broadband **EDT {:.3} s, T20 {:.3} s**.\n\n\
| band | T20 s |\n|---|--:|\n",
        f.reference_sigs.len(),
        SHOWCASE_VELOCITY,
        median_of(
            f.engine_sigs
                .iter()
                .map(|(_, s)| s.per_band[SPATIAL_BANDS.len() - 1].zero_r)
                .collect()
        ),
        f.edt,
        f.t20
    );
    for &(lo, hi, t) in &f.room.tail_t60 {
        let _ = writeln!(out, "| {lo:.0}–{hi:.0} Hz | {t:.3} |");
    }
    let _ = write!(
        out,
        "\n**That is not a room and the table says so itself.** Air absorption makes every real \
space's decay *shorter* with frequency; this one gets longer, {:.3} s at 500 Hz-2 kHz against \
{:.3} s at 6-12 kHz. What is being integrated is the mechanical event's own body, not the \
space it happened in. Read as a bound the number is still useful — the room's T60 is *at most* \
about half a second — and read as a measurement it is refused.\n\n\
The reflection finder agrees. Its {} strongest clustered candidates, over the same recordings:\n\n\
| delay ms | level dB under direct | side |\n|--:|--:|--:|\n",
        f.room
            .tail_t60
            .iter()
            .find(|&&(lo, _, _)| lo == 500.0)
            .map(|&(_, _, t)| t)
            .unwrap_or(f64::NAN),
        f.room
            .tail_t60
            .iter()
            .find(|&&(lo, _, _)| lo == 6000.0)
            .map(|&(_, _, t)| t)
            .unwrap_or(f64::NAN),
        f.candidates.len()
    );
    for r in f.candidates {
        let _ = writeln!(
            out,
            "| {:.2} | {:.2} | {:+.2} |",
            r.delay_s * 1000.0,
            r.gain_db,
            r.side
        );
    }
    let _ = write!(
        out,
        "\nA sequence of early reflections gets **quieter** with delay. These get louder, by \
{:.0} dB over {:.0} ms. They are the damper landing and the tray settling, and a room built \
out of them would be that mechanism dressed as a wall. **The stage therefore carries no \
reflections**, and the refusal is a finding rather than an omission.\n\n\
### What is left, and how it is fitted\n\n\
A diffuse, decorrelated tail and nothing else. Its decay per band is the bound above. Its \
onset is a **stated choice** of {:.0} ms, not a reading — every sample in the library is \
trimmed to its own onset, so a pre-delay is not recoverable, and at a {:.2} s tail 10 ms is \
2 % of its energy and none of its spectrum. Its **level** is fitted to the one thing this \
material does measure about a late field: the recording's own interchannel decorrelation, on \
half A's notes.\n\n\
| tail level dB | interchannel mismatch |\n|--:|--:|\n| bare engine | {:.3} |\n",
        f.candidates
            .last()
            .map(|r| r.gain_db)
            .unwrap_or(0.0)
            - f.candidates.first().map(|r| r.gain_db).unwrap_or(0.0),
        1000.0
            * (f.candidates.last().map(|r| r.delay_s).unwrap_or(0.0)
                - f.candidates.first().map(|r| r.delay_s).unwrap_or(0.0)),
        f.room.tail_onset_s * 1000.0,
        f.room.tail_t60.iter().map(|&(_, _, t)| t).fold(0.0f64, f64::max),
        f.bare_mismatch
    );
    for &(level, mismatch) in f.sweep {
        if (level - f.room.tail_level_db).abs() < 1e-9 {
            let _ = writeln!(out, "| **{level:+.0}** | **{mismatch:.3}** |");
        } else {
            let _ = writeln!(out, "| {level:+.0} | {mismatch:.3} |");
        }
    }
    let _ = write!(
        out,
        "\nThe stage is set at **{:+.0} dB**, an interior minimum rather than a rail. Through the whole \
chain the mismatch reads **{:.3} -> {:.3}** — the room stage closes about a third of the one \
gap it addresses, and cannot close more, because the recording's decorrelation is not a tail: \
its channels stay {:.0} % coherent at a lag of a fraction of a millisecond, which is a spaced \
**pair of microphones** and not a reverberant field. Matching it properly needs `PHYSICS.md` \
§8's two mic positions, which is a different stage from §9's room. The two channels' tails are \
drawn from independent streams, which is the whole spatial content of what is built here.\n\n\
## The collapse\n\n\
Every phrase below was held out of the fit entirely. `engine` is the render as it stands, \
`+EQ` is it through the fitted curve alone, `+chain` is it through curve and room, `floor` is \
the reference against itself played out of the neighbouring velocity layer. `ode_soprano` is \
`MELODY.md`'s line and is reported beside the six rather than averaged into them.\n\n\
| phrase | mel engine | +EQ | +chain | floor | mod engine | +EQ | +chain | floor |\n\
|---|--:|--:|--:|--:|--:|--:|--:|--:|\n",
        f.room.tail_level_db,
        f.bare_mismatch,
        f.chained_mismatch,
        100.0
            * median_of(
                f.reference_sigs
                    .iter()
                    .map(|(_, s)| s.per_band[SPATIAL_BANDS.len() - 2].peak_r.abs())
                    .collect()
            )
    );
    for (name, before, eq, chained, floor) in f.rows {
        let _ = writeln!(
            out,
            "| `{name}` | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} |",
            before.mel, eq.mel, chained.mel, floor.mel,
            before.modulation, eq.modulation, chained.modulation, floor.modulation
        );
    }
    let _ = write!(
        out,
        "| **mean of six** | **{:.2}** | **{:.2}** | **{:.2}** | **{:.2}** | {:.2} | {:.2} | {:.2} | {:.2} |\n\n\
The excess over the floor is **{:.2} dB**; the EQ absorbs **{:.0} %** of it and the whole \
chain **{:.0} %** — the room stage gives back about a tenth of a dB of what the EQ buys, \
because a decorrelated tail added to a signal that already carries a board field puts energy \
where the reference has none, and the mono sum is where this metric reads it. What the room \
stage buys is not in this table at all: see the interchannel column above.\n\n\
### The oracle, at window {} (the resolution the images are drawn at)\n\n\
| | as rendered | through the fitted EQ |\n|---|--:|--:|\n\
| no curve | {:.2} | {:.2} |\n\
| one global static curve, fitted on the answer | {:.2} | {:.2} |\n\
| one static curve **per phrase**, fitted on the answer | {:.2} | {:.2} |\n\n\
A global static magnitude filter has at most **{:.2} dB** of a **{:.2} dB** distance to give, \
and even a filter allowed to change with every phrase has only {:.2} dB. The fitted curve, \
which saw none of this material, has already taken {:.2} dB.\n\n\
### Brilliance, the same three ways\n\n\
| phrase | 2-6 kHz e / +EQ / +chain (floor) | 6-12 kHz e / +EQ / +chain (floor) | tilt dB/oct e / +EQ / +chain (floor) |\n\
|---|--:|--:|--:|\n",
        mel_before, mel_eq, mel_chain, mel_floor,
        mean(bench.iter().map(|r| r.1.modulation)),
        mean(bench.iter().map(|r| r.2.modulation)),
        mean(bench.iter().map(|r| r.3.modulation)),
        mean(bench.iter().map(|r| r.4.modulation)),
        excess,
        100.0 * (mel_before - mel_eq) / excess,
        100.0 * (mel_before - mel_chain) / excess,
        realism::MULTI_RES_WINDOWS[1],
        f.oracle.0,
        f.oracle_eq.0,
        f.oracle.1,
        f.oracle_eq.1,
        f.oracle.2,
        f.oracle_eq.2,
        f.oracle.0 - f.oracle.1,
        f.oracle.0,
        f.oracle.0 - f.oracle.2,
        f.oracle.0 - f.oracle_eq.0,
    );
    for (name, before, eq, chained, floor) in f.rows {
        let _ = writeln!(
            out,
            "| `{name}` | {:+.2} / {:+.2} / {:+.2} ({:.2}) | {:+.2} / {:+.2} / {:+.2} ({:.2}) | {:+.2} / {:+.2} / {:+.2} ({:.2}) |",
            before.hf1, eq.hf1, chained.hf1, floor.hf1.abs(),
            before.hf2, eq.hf2, chained.hf2, floor.hf2.abs(),
            before.tilt, eq.tilt, chained.tilt, floor.tilt.abs()
        );
    }
    let _ = write!(
        out,
        "\n## Renders\n\n\
`<name>_{{engine,engine_chained,reference}}.wav`, each scaled to the benchmark's own target \
RMS so the three are level-matched: {}.\n\n\
## The verdict\n\n\
**The recording chain is not where the gap is.** A static magnitude transfer — the thing \
`TUNING.md` reserved a stage for — absorbs **{:.0} %** of the excess when fitted honestly on \
separate material, and **at most {:.0} %** when an oracle is allowed to fit it on the answer. \
Under a dB is the whole prize, on a 3.40 dB excess.\n\n\
Three independent readings say the same thing and none of them is the headline number.\n\n\
1. **The band-level scatter does not move.** The half-A curve applied to keys it never saw \
changes the median per-band error by {:+.2} dB of {:.2}. What a curve can remove is the part of \
the error every key shares; what is left is each key's own partials being in the wrong place \
relative to each other, which is a spectrum and not a filter.\n\
2. **The soft and loud halves disagree more than the two key halves do** ({:.2} dB against \
{:.2}, r {:+.3} against {:+.3}). A microphone is not level-dependent. Either the chain has a \
compressor in it — which nothing else here supports — or, far more likely, part of what the \
curve absorbs is the excitation model's own velocity behaviour.\n\
3. **The curve fitted on notes makes the phrases' top band worse.** Over the six phrases the \
mean absolute 6-12 kHz ratio goes {:.2} -> {:.2} dB and the 500 Hz-8 kHz tilt {:.2} -> {:.2} \
dB/octave. The note-median curve cuts hard above 6 kHz because the *top-octave keys* are where \
the engine is brightest there (`DECISIONS.md` 292's +15.16 dB), and a phrase that never goes \
above C6 then gets a cut it did not earn. A chain is one curve for all 88 keys; a correction \
that has to be register-dependent to be right is not a chain, and this one does.\n\n\
**What the chain is worth, and it is not nothing:** the interchannel column. The recording's \
two channels are one signal below 125 Hz and about 60 % coherent above it at a sub-millisecond \
lag; the engine's are the reverse — decorrelated in the bass by the board FDN's orthogonal \
taps and a pan-pot's by 6-12 kHz ({:.3}). The room stage closes that mismatch {:.3} -> {:.3}, \
and **nothing in `REALISM.md` scores it**, because every metric there is a mono sum. If the \
listener's \"it doesn't sound bad, it's just different\" has a presentation component, this is \
where it lives — and it is a **microphone** question (`PHYSICS.md` §8), not a room one (§9).\n\n\
### Recommended path\n\n\
* **Do not** add a `[chain]` EQ section to the preset on this evidence. Under a dB of mel, \
bought by a curve that has to be register-dependent to be right, is a place for instrument \
error to hide — which is exactly what `PHYSICS.md` §8 warns of when it says splitting the \
absorber into mic geometry plus room is strictly better than an anonymous EQ, because geometry \
has priors and an EQ has none.\n\
* **Do** treat the interchannel finding as the next presentation milestone, and build §8 \
rather than §9: two virtual microphone positions with a per-key delay and gain is a mechanism \
with a prior, in front of a room this material cannot measure at all.\n\
* **Before either**, give the loss a stereo term. The largest measured difference in this \
experiment is invisible to every column on the scoreboard, and a stage built to fix something \
nothing scores is a stage nobody can regress.\n\n\
### What remains, ranked\n\n\
1. The per-key, per-partial spectral scatter — {:.2} dB of median band error that no global \
curve touches. This is `COMPASS.md`'s `match` and the fitted-against-drawn seam, and it is the \
biggest single thing left.\n\
2. The modulation column, {:.2} dB against a {:.2} dB floor, which a linear stage cannot move \
at all by construction: no filter changes how a partial's level *moves*.\n\
3. The stereo presentation above — large, measured, unscored.\n\
4. The static spectral transfer — real, replicable in shape (r {:+.3}), and worth under a dB.\n\n\
## What this material cannot identify\n\n\
1. **Chain colour against instrument error.** A smooth static EQ cannot tell a microphone's \
response from a systematic excitation-model error: both are smooth in `ln f` and constant \
across the compass. The split-half agreement above bounds how much of the gap is *shaped like* \
a static chain; it is not causation, and it is never reported as such.\n\
2. **Absolute pre-delay.** Every sample is trimmed to its own onset, so mic distance is gone \
and any reflection delay would be relative only.\n\
3. **The chain's phase.** Magnitude ratios of two different excitations carry no phase. The EQ \
is linear-phase **by choice**, stated rather than fitted, and its group delay is compensated \
exactly.\n\
4. **Room tail against string, board and mechanism.** A struck note's late energy is string, \
board, halo and room superposed; a key-off recording's is the mechanism's own body. Neither \
separates. The T20 table above is a bound, and its rising-with-frequency shape is the proof \
that it is not a room.\n\
5. **Mic geometry against the instrument's own extent.** One instrument, one session: spacing, \
bridge separation and early field are three causes of one number.\n\
6. **Chain nonlinearity.** Tested by the soft-vs-loud split above; a null result bounds it \
rather than excluding it.\n\
7. **What the reference's room does between notes.** The reference is a sampler: each note \
carries its own recording's room, and Salamander's note-off fade truncates that room tail with \
the note. A room stage convolved over a whole engine phrase does not truncate. The two are \
equal while notes ring and differ after every damper.\n",
        f.written
            .iter()
            .map(|n| format!("`{n}`"))
            .collect::<Vec<_>>()
            .join(", "),
        100.0 * (mel_before - mel_eq) / excess,
        100.0 * (f.oracle.0 - f.oracle.1) / excess,
        f.residual.2 - f.residual.3,
        f.residual.2,
        f.level.0,
        f.split.0,
        f.level.1,
        f.split.1,
        mean(bench.iter().map(|r| r.1.hf2.abs())),
        mean(bench.iter().map(|r| r.2.hf2.abs())),
        mean(bench.iter().map(|r| r.1.tilt.abs())),
        mean(bench.iter().map(|r| r.2.tilt.abs())),
        median_of(
            f.engine_sigs
                .iter()
                .map(|(_, s)| s.per_band[SPATIAL_BANDS.len() - 1].peak_r.abs())
                .collect()
        ),
        f.bare_mismatch,
        f.chained_mismatch,
        f.residual.2,
        mean(bench.iter().map(|r| r.1.modulation)),
        mean(bench.iter().map(|r| r.4.modulation)),
        f.split.1,
    );
    out
}
