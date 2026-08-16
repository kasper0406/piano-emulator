//! How bright the engine is against the piano, per key, per phrase and over
//! time — the audit behind `DECISIONS.md` 292-295.
//!
//! The listening note this was written for is "the recording is slightly more
//! brilliant than the engine". Brilliance had no number under it before this:
//! `COMPASS.md`'s `centroid` is the power-weighted mean *partial index*, which
//! is register-relative by construction and says nothing about where a key's
//! energy sits in **absolute** frequency, and the ear's brightness is absolute.
//! Two keys an octave apart with identical `centroid` differ by an octave in
//! the band a listener calls air.
//!
//! [`piano_tuner::estimate::brilliance`] holds the measurements and their
//! reasons; this is the driver that runs them over the compass, over the six
//! benchmark phrases, and over the three named suspects.
//!
//! # What it prints, in the order it prints it
//!
//! 1. **Per key**: the level-matched 2-6 kHz and 6-12 kHz ratios at 0.1 s and
//!    at 1 s, each against the reference's own **velocity-layer spread** — the
//!    same key out of the recording layer next door, which is what this metric
//!    cannot resolve and therefore what a finding has to beat. Then each band's
//!    own decay between the two instants, engine minus recording, which is the
//!    column that separates "too dark" from "dies at the wrong rate".
//! 2. **Per register**, the same, averaged; and the same summed with every key
//!    level-matched to its own recording first. The per-key numbers scatter by
//!    5-17 dB and are meant to — one key's 2 kHz band holds a dozen partials
//!    whose levels the engine already misses by `COMPASS.md`'s `match`. What a
//!    listener hears is all of them at once, so the sum is the statistic that
//!    answers the listening note; the per-partial scatter cancels there and a
//!    systematic tilt does not.
//! 3. **The fundamental's tail**: a fitted T60 of the band around `f0` on both
//!    signals, with the recording's own floor beside it. This is suspect (a),
//!    and the floor column is why it was refused.
//! 4. **The phrases**: the two bands and a **tilt** — the least-squares slope,
//!    in dB per octave, of the engine-minus-reference octave-band difference
//!    over 500 Hz - 8 kHz — on the same material `REALISM.md` scores.
//!
//! # The two experiments
//!
//! `<shelf_gain_db> [<shelf_hz>]` overrides `[soundboard]`'s master shelf in
//! memory, so suspect (b) can be swept without editing a preset. Item 293's
//! acquittal is that sweep.
//!
//! `--trim <passes>` runs the **envelope continuation**: each key's partials
//! above the reach its own recording gave it are moved by
//! [`continuation_db`](piano_tuner::estimate::brilliance::continuation_db),
//! re-rendered and re-measured, until the two bands stop moving. It is kept
//! because it is the falsification in item 295 — it brings the compass into
//! the floor and takes the phrase set out of it — and because the next
//! milestone should start from the working instrument rather than from the
//! argument.
//!
//! ```sh
//! cargo run --release -p piano-tuner -- brilliance \
//!     data/salamander presets/salamander-c5.toml
//! cargo run --release -p piano-tuner -- brilliance \
//!     data/salamander presets/salamander-c5.toml 0
//! cargo run --release -p piano-tuner -- brilliance \
//!     data/salamander presets/salamander-c5.toml --trim 6
//! ```

use std::cell::RefCell;
use std::path::{Path, PathBuf};

use rayon::prelude::*;

use piano_emulator::preset::Preset;
use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_emulator::types::Event;
use piano_tuner::cache;
use piano_tuner::realism::{self, Phrase, VelocityLayers, PHRASE_SET_VERSION};
use piano_tuner::sampler::SAMPLER_VERSION;
use piano_tuner::estimate::brilliance::{
    band, band_decay, continuation_db, fitted_t60, floor_under_peak, hf_ratio, narrowband_db,
    BandDecay, FLOOR_FROM_S,
    trim_gain_db, FULL, HF1, HF2, TRIM_CAP_DB,
};
use piano_tuner::estimate::shaping::MAX_ROW_CELLS;
use piano_tuner::stft::{Stft, StftConfig};
use piano_tuner::sampler::engine_events;
use piano_tuner::{Audio, SampleLibrary, Sampler, SamplerEvent, TimedEvent, SAMPLE_RATE};

/// The velocity the compass, the fits and the motion columns all use.
const VELOCITY: u8 = 90;
/// Seconds of note, matching `compass` so the reference cache is shared.
const RENDER_S: f64 = 3.6;
const PREROLL_S: f64 = 0.05;

const FIRST_KEY: u8 = 21;
const LAST_KEY: u8 = 108;

/// Analysis window: 4096 samples, 85 ms at 48 kHz.
///
/// Short enough that the 0.1 s reading is still inside the strike and does not
/// average the note's first half-second into it, long enough that the 2 kHz
/// band edge is 170 bins wide and the window's own skirt is nowhere near it.
const WINDOW: usize = 4096;

/// Where the two readings start, in seconds after the onset.
const INSTANTS: [f64; 2] = [0.1, 1.0];

/// Octave-band centres for the phrase tilt.
const OCTAVES: [f64; 8] = [125.0, 250.0, 500.0, 1_000.0, 2_000.0, 4_000.0, 8_000.0, 16_000.0];
/// The octave bands the tilt slope is fitted over: everything from 500 Hz up.
const TILT_FROM_HZ: f64 = 500.0;

/// The keys the per-partial decay probe is run on: one per register, all of
/// them keys the library sampled, so nothing here is an interpolation.
const PROBE_KEYS: [u8; 6] = [33, 45, 57, 60, 72, 84];
/// The partials it is run on. Powers of two so that one row spans four octaves
/// of the same note.
const PARTIAL_PROBE: [usize; 5] = [1, 2, 4, 8, 16];

/// The sample rate every measurement here is taken at.
const SR: f64 = SAMPLE_RATE as f64;

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
// Band energies
// ---------------------------------------------------------------------------

/// Power per FFT bin of one Hann-windowed frame starting at `start`.
fn frame_power(stft: &Stft, mono: &[f32], start: usize) -> Option<Vec<f64>> {
    if start + WINDOW > mono.len() {
        return None;
    }
    let spectrum = stft.analyze(&mono[start..start + WINDOW], f64::from(SAMPLE_RATE));
    let frame = spectrum.frames.first()?;
    Some(
        frame
            .magnitude
            .iter()
            .map(|&m| f64::from(m) * f64::from(m))
            .collect(),
    )
}

/// Power summed over every frame of a whole signal.
fn total_power(stft: &Stft, mono: &[f32]) -> Vec<f64> {
    let mut sum = vec![0.0f64; stft.bins()];
    stft.for_each_frame(mono, f64::from(SAMPLE_RATE), |_, magnitude| {
        for (s, &m) in sum.iter_mut().zip(magnitude.iter()) {
            *s += f64::from(m) * f64::from(m);
        }
    });
    sum
}

fn db(ratio: f64) -> f64 {
    10.0 * ratio.max(1e-30).log10()
}

/// [`band_decay`] over the two instants of this tool, off two renders, with the
/// headroom each signal's own late-time floor leaves it.
///
/// A gap whose reference side has fallen into the recording's own floor is a
/// **bound** and not a measurement, and above 6 kHz that is most of this
/// compass (`DECISIONS.md` 319) — so this returns the whole reading and the
/// tables print the mark rather than the number alone.
fn decay_gap(stft: &Stft, engine: &[f32], reference: &[f32], bnd: (f64, f64)) -> Option<BandDecay> {
    let at = |signal: &[f32], t: f64| {
        frame_power(stft, signal, (t * f64::from(SAMPLE_RATE)) as usize)
    };
    let floor = |signal: &[f32]| at(signal, FLOOR_FROM_S);
    let (Some(e0), Some(e1), Some(ef), Some(r0), Some(r1), Some(rf)) = (
        at(engine, INSTANTS[0]),
        at(engine, INSTANTS[1]),
        floor(engine),
        at(reference, INSTANTS[0]),
        at(reference, INSTANTS[1]),
        floor(reference),
    ) else {
        return None;
    };
    Some(band_decay(&e0, &e1, &ef, &r0, &r1, &rf, SR, bnd))
}

/// The gap alone, for a column that has no room for its mark.
fn gap_db(cell: Option<BandDecay>) -> f64 {
    cell.map_or(f64::NAN, |d| d.gap_db)
}

/// The bound the *median* cell of a column carries, read off the same cells the
/// median is.
fn bound_mark(cells: &[BandDecay]) -> &'static str {
    BandDecay {
        gap_db: f64::NAN,
        engine_headroom_db: median(cells.iter().map(|d| d.engine_headroom_db)),
        reference_headroom_db: median(cells.iter().map(|d| d.reference_headroom_db)),
    }
    .mark()
}

/// Both bands at both instants: `[hf1@0.1, hf2@0.1, hf1@1.0, hf2@1.0]`.
fn key_ratios(stft: &Stft, engine: &[f32], reference: &[f32]) -> [f64; 4] {
    let mut out = [f64::NAN; 4];
    for (i, &t) in INSTANTS.iter().enumerate() {
        let start = (t * f64::from(SAMPLE_RATE)) as usize;
        let (Some(e), Some(r)) = (
            frame_power(stft, engine, start),
            frame_power(stft, reference, start),
        ) else {
            continue;
        };
        out[2 * i] = hf_ratio(&e, &r, SR, HF1);
        out[2 * i + 1] = hf_ratio(&e, &r, SR, HF2);
    }
    out
}

/// Share of one pass's correction actually applied. The band a partial is
/// trimmed for contains partials the trim does not own, so the solve is exact
/// only to first order; damping turns that into convergence instead of ringing.
const TRIM_DAMPING: f64 = 0.8;

/// The bands the compass-wide curve is accumulated in: the eight octaves, then
/// the two brilliance bands, then the broadband the level match is taken over.
fn curve_bands() -> Vec<(f64, f64)> {
    let mut v: Vec<(f64, f64)> = OCTAVES
        .iter()
        .map(|&c| (c / std::f64::consts::SQRT_2, c * std::f64::consts::SQRT_2))
        .collect();
    v.push(HF1);
    v.push(HF2);
    v.push(FULL);
    v
}

/// One key's band powers at one instant, already level-matched: the engine's
/// are scaled so that its broadband power equals the reference's, which is what
/// makes a *sum over the compass* a fair comparison rather than a comparison of
/// the master gain.
///
/// A per-key match rather than one global gain because the compass's own level
/// error is 13-22 dB and register-dependent (`COMPASS.md`'s `level e/r`): a
/// single gain would let the bass, where the engine is 22 dB quiet, decide what
/// the treble's brilliance looks like.
fn matched_powers(spectrum: &[f64], reference: &[f64], bands: &[(f64, f64)]) -> Vec<f64> {
    let scale = band(reference, SR, FULL) / band(spectrum, SR, FULL);
    bands.iter().map(|&b| band(spectrum, SR, b) * scale).collect()
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

fn render_engine(preset: &Preset, key: u8) -> Audio {
    let events = [RenderEvent::new(
        PREROLL_S as f32,
        Event::NoteOn { key, vel: VELOCITY },
    )];
    let (left, right) = render_to_buffer(preset, &events, (PREROLL_S + RENDER_S) as f32);
    let skip = (PREROLL_S * f64::from(SAMPLE_RATE)) as usize;
    Audio::new(
        SAMPLE_RATE,
        vec![left[skip..].to_vec(), right[skip..].to_vec()],
    )
    .expect("the engine renders stereo")
}

/// The recording of one note, onset-aligned exactly the way `compass`
/// aligns it — same recipe, same cache entry.
fn render_reference(
    sampler: &mut Sampler,
    key: u8,
    vel: u8,
) -> Result<Audio, piano_tuner::Error> {
    let events = [TimedEvent::new(0.0, SamplerEvent::NoteOn { key, vel })];
    let rendered = sampler.render(&events, RENDER_S + 0.2)?;
    let mono = rendered.mono();
    let onset = piano_tuner::detect_onset(&mono, f64::from(SAMPLE_RATE));
    let skip = (onset * f64::from(SAMPLE_RATE)).round() as usize;
    let frames = (RENDER_S * f64::from(SAMPLE_RATE)) as usize;
    let cut = |c: &Vec<f32>| -> Vec<f32> {
        (0..frames)
            .map(|n| c.get(skip + n).copied().unwrap_or(0.0))
            .collect()
    };
    Audio::new(
        SAMPLE_RATE,
        rendered.channels.iter().map(cut).collect(),
    )
}


fn render_phrase(preset: &Preset, phrase: &Phrase) -> Audio {
    let (left, right) = render_to_buffer(
        preset,
        &engine_events::to_render_events(&phrase.events),
        phrase.duration_s as f32,
    );
    Audio::new(SAMPLE_RATE, vec![left, right]).expect("the engine renders stereo")
}

// ---------------------------------------------------------------------------
// Reporting helpers
// ---------------------------------------------------------------------------

fn note_name(key: u8) -> String {
    const NAMES: [&str; 12] = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    format!(
        "{}{}",
        NAMES[usize::from(key) % 12],
        i32::from(key) / 12 - 1
    )
}

fn mean(xs: impl Iterator<Item = f64>) -> f64 {
    let (sum, n) = xs.fold((0.0, 0usize), |(s, n), x| {
        if x.is_finite() {
            (s + x, n + 1)
        } else {
            (s, n)
        }
    });
    if n == 0 {
        f64::NAN
    } else {
        sum / n as f64
    }
}

fn median(xs: impl Iterator<Item = f64>) -> f64 {
    let mut v: Vec<f64> = xs.filter(|x| x.is_finite()).collect();
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

struct Row {
    key: u8,
    engine: [f64; 4],
    layer: [f64; 4],
    /// `[instant][band]` band powers, level-matched to the reference key by key.
    engine_bands: [Vec<f64>; INSTANTS.len()],
    reference_bands: [Vec<f64>; INSTANTS.len()],
    layer_bands: [Vec<f64>; INSTANTS.len()],
    /// Engine-minus-reference decay of `[full, 2-6k, 6-12k]` over the interval.
    decay_gap: [Option<BandDecay>; 3],
    layer_decay_gap: [Option<BandDecay>; 3],
    /// The fundamental's own fitted T60 on the engine and on the recording.
    tail: [Option<f64>; 2],
    /// The same, for each of [`PARTIAL_PROBE`]; empty off [`PROBE_KEYS`].
    partial_t60: Vec<(Option<f64>, Option<f64>)>,
    reference_floor_db: f64,
    engine_floor_db: f64,
    /// The gain, in dB, that this key's partials **above its fitted row** would
    /// need for each of the two bands to land on the recording.
    trim_db: [f64; 2],
}

pub fn run(args: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
    let mut args = args.into_iter();
    let data = PathBuf::from(args.next().unwrap_or_else(|| "data/salamander".into()));
    let preset_path = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "presets/salamander-c5.toml".into()),
    );

    let sfz = data.join("SalamanderGrandPiano-V3+20200602.sfz");
    if !sfz.exists() {
        eprintln!("the reference piano is not here: {}", sfz.display());
        std::process::exit(2);
    }
    let mut preset = Preset::load(&preset_path)?;
    // `-- <data> <preset> [<shelf_gain_db>] [<shelf_hz>] [--trim <passes>]`:
    // the shelf override tries suspect (b) without editing the preset, and
    // `--trim` runs the envelope continuation of item 293 in memory.
    let rest: Vec<String> = args.collect();
    let flag = |name: &str| -> Option<f64> {
        rest.iter()
            .position(|a| a == name)
            .and_then(|i| rest.get(i + 1))
            .and_then(|v| v.parse().ok())
    };
    let trim_passes = flag("--trim").unwrap_or(0.0) as usize;
    let positional: Vec<f32> = rest
        .iter()
        .take_while(|a| !a.starts_with("--"))
        .filter_map(|a| a.parse().ok())
        .collect();
    if let Some(&g) = positional.first() {
        preset.soundboard.shelf_gain_db = g;
    }
    if let Some(&hz) = positional.get(1) {
        preset.soundboard.shelf_hz = hz;
    }
    preset.validate()?;
    let library = SampleLibrary::from_sfz(&sfz)?;
    let layers = VelocityLayers::from_library(&library)?;
    let alt_velocity = layers.alternate(VELOCITY);

    let stft = Stft::new(StftConfig::new(WINDOW, WINDOW, WINDOW)?)?;
    let bands = curve_bands();

    println!(
        "brilliance audit: engine on {}, reference {}",
        preset_path.display(),
        sfz.display()
    );
    println!(
        "  bands {:.0}-{:.0} Hz and {:.0}-{:.0} Hz, at {} s and {} s, level-matched over {:.0}-{:.0} Hz",
        HF1.0, HF1.1, HF2.0, HF2.1, INSTANTS[0], INSTANTS[1], FULL.0, FULL.1
    );
    println!("  noise floor: the same key at velocity {alt_velocity} (the layer next door)");
    println!(
        "  master shelf: {:+.2} dB above {:.0} Hz\n",
        preset.soundboard.shelf_gain_db, preset.soundboard.shelf_hz
    );

    // ---- the compass ------------------------------------------------------
    let reference_cache = cache::reference_dir(&data);
    // Byte-for-byte the key `compass` uses, so its warm cache is this
    // tool's warm cache: same sampler, same SFZ, same rate, same velocity, same
    // length, same onset-aligned cut.
    let mut compass_key = cache::Fingerprint::new();
    compass_key
        .str("compass-scan/reference")
        .u64(u64::from(SAMPLER_VERSION))
        .file(&sfz)?
        .u64(u64::from(SAMPLE_RATE))
        .u64(u64::from(VELOCITY))
        .f64(RENDER_S);
    let mut alt_key = cache::Fingerprint::new();
    alt_key
        .str("brilliance/alt-layer")
        .u64(u64::from(SAMPLER_VERSION))
        .file(&sfz)?
        .u64(u64::from(SAMPLE_RATE))
        .u64(u64::from(alt_velocity))
        .f64(RENDER_S);

    let keys: Vec<u8> = (FIRST_KEY..=LAST_KEY).collect();
    // The row lengths before any trim: the reach the *recordings* gave each key,
    // which is what the continuation continues from and must never move — a
    // reach re-read from the trimmed preset would say the trim has nothing left
    // to correct and the loop would stop after one pass.
    let reach: Vec<usize> = keys
        .iter()
        .map(|&k| preset.notes.partial_gains[usize::from(k - FIRST_KEY)].len())
        .collect();
    let base_rows: Vec<Vec<f32>> = preset.notes.partial_gains.clone();

    // The whole compass, measured against the recordings, on whatever preset it
    // is handed. A closure and not a straight loop because the trim below has to
    // run it again after every pass: a continuation fitted to one render and
    // never checked on the next is a fit that was never closed.
    let measure = |preset: &Preset, reach: &[usize]| -> Result<Vec<Row>, piano_tuner::Error> {
        keys
        .par_iter()
        .map(|&key| -> Result<Row, piano_tuner::Error> {
            let engine_audio = render_engine(preset, key);
            let mut k = compass_key;
            k.u64(u64::from(key));
            let path = reference_cache.join(format!("compass-key{key:03}-{}.wav", k.hex()));
            let reference_audio =
                cache::audio(&path, || with_sampler(&sfz, |s| render_reference(s, key, VELOCITY)))?;
            let mut a = alt_key;
            a.u64(u64::from(key));
            let alt_path =
                reference_cache.join(format!("brilliance-alt-key{key:03}-{}.wav", a.hex()));
            let alt_audio = cache::audio(&alt_path, || {
                with_sampler(&sfz, |s| render_reference(s, key, alt_velocity))
            })?;
            let engine = engine_audio.mono();
            let reference = reference_audio.mono();
            let alt = alt_audio.mono();
            let f0 = f64::from(preset.string_params(key).partial_freq(1));
            let engine_env = narrowband_db(&engine, f0, SR);
            let reference_env = narrowband_db(&reference, f0, SR);
            let mut engine_bands = [const { Vec::new() }; INSTANTS.len()];
            let mut reference_bands = [const { Vec::new() }; INSTANTS.len()];
            let mut layer_bands = [const { Vec::new() }; INSTANTS.len()];
            for (i, &t) in INSTANTS.iter().enumerate() {
                let start = (t * f64::from(SAMPLE_RATE)) as usize;
                let (Some(e), Some(r), Some(a)) = (
                    frame_power(&stft, &engine, start),
                    frame_power(&stft, &reference, start),
                    frame_power(&stft, &alt, start),
                ) else {
                    continue;
                };
                engine_bands[i] = matched_powers(&e, &r, &bands);
                layer_bands[i] = matched_powers(&a, &r, &bands);
                reference_bands[i] = bands.iter().map(|&b| band(&r, SR, b)).collect();
            }
            Ok(Row {
                key,
                engine: key_ratios(&stft, &engine, &reference),
                layer: key_ratios(&stft, &alt, &reference),
                engine_bands,
                reference_bands,
                layer_bands,
                decay_gap: [FULL, HF1, HF2]
                    .map(|b| decay_gap(&stft, &engine, &reference, b)),
                layer_decay_gap: [FULL, HF1, HF2]
                    .map(|b| decay_gap(&stft, &alt, &reference, b)),
                tail: [fitted_t60(&engine_env), fitted_t60(&reference_env)],
                partial_t60: if PROBE_KEYS.contains(&key) {
                    let params = preset.string_params(key);
                    PARTIAL_PROBE
                        .iter()
                        .map(|&k| {
                            let hz = f64::from(params.partial_freq(k));
                            if hz > 0.45 * f64::from(SAMPLE_RATE) {
                                return (None, None);
                            }
                            (
                                fitted_t60(&narrowband_db(&engine, hz, SR)),
                                fitted_t60(&narrowband_db(&reference, hz, SR)),
                            )
                        })
                        .collect()
                } else {
                    Vec::new()
                },
                reference_floor_db: floor_under_peak(&reference_env),
                engine_floor_db: floor_under_peak(&engine_env),
                trim_db: {
                    let params = preset.string_params(key);
                    let n = params.partial_count().min(MAX_ROW_CELLS);
                    let partial_hz: Vec<f64> =
                        (1..=n).map(|k| f64::from(params.partial_freq(k))).collect();
                    let reach = reach[usize::from(key - FIRST_KEY)];
                    let start = (INSTANTS[0] * f64::from(SAMPLE_RATE)) as usize;
                    match (
                        frame_power(&stft, &engine, start),
                        frame_power(&stft, &reference, start),
                    ) {
                        (Some(e), Some(r)) => [HF1, HF2]
                            .map(|b| trim_gain_db(&e, &r, SR, &partial_hz, reach, b)),
                        _ => [0.0; 2],
                    }
                },
            })
        })
        .collect::<Result<Vec<_>, _>>()
    };

    let mut trim: Vec<Vec<f64>> = keys.iter().map(|_| Vec::new()).collect();

    let mut rows = measure(&preset, &reach)?;
    for pass in 1..=trim_passes {
        for (i, &key) in keys.iter().enumerate() {
            let params = preset.string_params(key);
            let n = params.partial_count().min(MAX_ROW_CELLS);
            if n <= reach[i] {
                continue;
            }
            let t = rows[i].trim_db;
            if trim[i].is_empty() {
                trim[i] = vec![0.0; n - reach[i]];
            }
            let mut row: Vec<f32> = base_rows[i].clone();
            row.resize(reach[i], 1.0);
            for k in reach[i] + 1..=n {
                let hz = f64::from(params.partial_freq(k));
                let cell = &mut trim[i][k - reach[i] - 1];
                *cell = (*cell + TRIM_DAMPING * continuation_db(hz, t))
                    .clamp(-TRIM_CAP_DB, TRIM_CAP_DB);
                row.push(10f64.powf(*cell / 20.0) as f32);
            }
            while row.last() == Some(&1.0) {
                row.pop();
            }
            preset.notes.partial_gains[i] = row;
        }
        preset.validate()?;
        rows = measure(&preset, &reach)?;
        let reg = |lo: u8, hi: u8, b: usize| {
            mean(rows.iter().filter(|r| (lo..=hi).contains(&r.key)).map(|r| r.engine[b]))
        };
        println!(
            "  trim pass {pass}: tenor 2-6k {:+.2} 6-12k {:+.2}   top 2-6k {:+.2} 6-12k {:+.2}",
            reg(48, 71, 0),
            reg(48, 71, 1),
            reg(84, 108, 0),
            reg(84, 108, 1),
        );
    }
    let rows = rows;
    let preset = preset;

    println!("| key | note | 2-6k @0.1 | 6-12k @0.1 | 2-6k @1.0 | 6-12k @1.0 | floor @0.1 | floor @1.0 | d full | d 2-6k | d 6-12k | d floor |");
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|");
    for r in &rows {
        println!(
            "| {} | {} | {:+.2} | {:+.2} | {:+.2} | {:+.2} | {:.2}/{:.2} | {:.2}/{:.2} | {:+.1} | {:+.1} | {:+.1} | {:.1}/{:.1} |",
            r.key,
            note_name(r.key),
            r.engine[0],
            r.engine[1],
            r.engine[2],
            r.engine[3],
            r.layer[0].abs(),
            r.layer[1].abs(),
            r.layer[2].abs(),
            r.layer[3].abs(),
            gap_db(r.decay_gap[0]),
            gap_db(r.decay_gap[1]),
            gap_db(r.decay_gap[2]),
            gap_db(r.layer_decay_gap[1]).abs(),
            gap_db(r.layer_decay_gap[2]).abs(),
        );
    }

    let registers: [(&str, u8, u8); 4] = [
        ("A0-B2 bass", 21, 47),
        ("C3-B4 tenor", 48, 71),
        ("C5-B5 treble", 72, 83),
        ("C6-C8 top", 84, 108),
    ];
    println!("\n{:<14} {:>10} {:>11} {:>10} {:>11}", "register", "2-6k @0.1", "6-12k @0.1", "2-6k @1.0", "6-12k @1.0");
    for &(name, lo, hi) in &registers {
        let sel = || rows.iter().filter(move |r| (lo..=hi).contains(&r.key));
        println!(
            "{name:<14} {:>+10.2} {:>+11.2} {:>+10.2} {:>+11.2}",
            mean(sel().map(|r| r.engine[0])),
            mean(sel().map(|r| r.engine[1])),
            mean(sel().map(|r| r.engine[2])),
            mean(sel().map(|r| r.engine[3])),
        );
        println!(
            "{:<14} {:>10.2} {:>11.2} {:>10.2} {:>11.2}",
            "  layer |.|",
            mean(sel().map(|r| r.layer[0].abs())),
            mean(sel().map(|r| r.layer[1].abs())),
            mean(sel().map(|r| r.layer[2].abs())),
            mean(sel().map(|r| r.layer[3].abs())),
        );
    }
    println!(
        "{:<14} {:>+10.2} {:>+11.2} {:>+10.2} {:>+11.2}",
        "ALL mean",
        mean(rows.iter().map(|r| r.engine[0])),
        mean(rows.iter().map(|r| r.engine[1])),
        mean(rows.iter().map(|r| r.engine[2])),
        mean(rows.iter().map(|r| r.engine[3])),
    );
    println!(
        "{:<14} {:>10.2} {:>11.2} {:>10.2} {:>11.2}",
        "ALL |.| ",
        mean(rows.iter().map(|r| r.engine[0].abs())),
        mean(rows.iter().map(|r| r.engine[1].abs())),
        mean(rows.iter().map(|r| r.engine[2].abs())),
        mean(rows.iter().map(|r| r.engine[3].abs())),
    );
    println!(
        "{:<14} {:>10.2} {:>11.2} {:>10.2} {:>11.2}",
        "ALL layer |.|",
        mean(rows.iter().map(|r| r.layer[0].abs())),
        mean(rows.iter().map(|r| r.layer[1].abs())),
        mean(rows.iter().map(|r| r.layer[2].abs())),
        mean(rows.iter().map(|r| r.layer[3].abs())),
    );

    println!(
        "\n{:<14} {:>10} {:>10} {:>10} {:>18}",
        "band decay 0.1 -> 1 s, engine - reference (dB); negative = the engine's band dies faster", "", "", "", ""
    );
    println!(
        "{:<14} {:>10} {:>10} {:>10} {:>18}",
        "register", "full", "2-6k", "6-12k", "floor 2-6k/6-12k"
    );
    println!(
        "  a cell marked `≥` has its reference side inside the recording's own floor and is a lower bound; \
         `≤` is the engine's side, `?` is both (`DECISIONS.md` 319)"
    );
    for &(name, lo, hi) in &registers {
        let sel = || rows.iter().filter(move |r| (lo..=hi).contains(&r.key));
        let cell = |b: usize| -> String {
            let cells: Vec<BandDecay> = sel().filter_map(|r| r.decay_gap[b]).collect();
            format!(
                "{:>+9.2}{}",
                median(cells.iter().map(|d| d.gap_db)),
                bound_mark(&cells)
            )
        };
        println!(
            "{name:<14} {} {} {} {:>9.2}/{:<8.2}",
            cell(0),
            cell(1),
            cell(2),
            median(sel().map(|r| gap_db(r.layer_decay_gap[1]).abs())),
            median(sel().map(|r| gap_db(r.layer_decay_gap[2]).abs())),
        );
    }
    println!(
        "{:<14} {:>+10.2} {:>+10.2} {:>+10.2} {:>9.2}/{:<8.2}",
        "ALL median",
        median(rows.iter().map(|r| gap_db(r.decay_gap[0]))),
        median(rows.iter().map(|r| gap_db(r.decay_gap[1]))),
        median(rows.iter().map(|r| gap_db(r.decay_gap[2]))),
        median(rows.iter().map(|r| gap_db(r.layer_decay_gap[1]).abs())),
        median(rows.iter().map(|r| gap_db(r.layer_decay_gap[2]).abs())),
    );

    // ---- the fundamental's own tail ---------------------------------------
    //
    // `top_octave.rs` reads the engine 60 dB down at 1.3-1.9 s where the
    // recordings take 3.4-3.7, and reads it broadband. This is the same
    // question asked of the **partial** and asked only where the answer exists:
    // a recording has a room and a noise floor under it, and once the note has
    // fallen into that floor the "envelope" is the floor's and its decay is
    // zero however long you watch. `ref floor` is how far the recording's own
    // late level sits under the note's peak — the deepest drop that can be
    // measured on it at all.
    println!("\nthe fundamental's tail: fitted T60 of the band around f0, seconds");
    println!(
        "{:>4} {:>5} {:>9} {:>9} {:>9} {:>10} {:>10}",
        "key", "note", "eng T60", "ref T60", "eng/ref", "ref floor", "eng floor"
    );
    let show = |v: Option<f64>| v.map_or("     -".to_string(), |t| format!("{t:6.2}"));
    for r in rows.iter().filter(|r| r.key >= 84) {
        println!(
            "{:>4} {:>5} {:>9} {:>9} {:>9} {:>10.1} {:>10.1}",
            r.key,
            note_name(r.key),
            show(r.tail[0]),
            show(r.tail[1]),
            match (r.tail[0], r.tail[1]) {
                (Some(e), Some(f)) => format!("{:6.2}", e / f),
                _ => "     -".to_string(),
            },
            r.reference_floor_db,
            r.engine_floor_db,
        );
    }
    for &(name, lo, hi) in &registers {
        let sel = || rows.iter().filter(move |r| (lo..=hi).contains(&r.key));
        println!(
            "{name:<14} eng T60 {:>6.2}  ref T60 {:>6.2}   n {:>2}/{:<2}   ref floor {:>5.1} dB",
            median(sel().filter_map(|r| r.tail[0])),
            median(sel().filter_map(|r| r.tail[1])),
            sel().filter(|r| r.tail[0].is_some()).count(),
            sel().filter(|r| r.tail[1].is_some()).count(),
            median(sel().map(|r| r.reference_floor_db)),
        );
    }

    // ---- how a partial's decay depends on its frequency -------------------
    //
    // The two halves of the register table above — dark at 0.1 s, bright at
    // 1 s — are one statement about *time*, and this is where it is read off
    // the mechanism. The engine damps a partial at
    // `sigma0 + sigma1 (f/1000)^2`, so what the ear calls brilliance dying away
    // is `sigma1`; if `sigma1` is small the note's high partials outlive its
    // fundamental and the note gets brighter as it decays, which no piano does.
    println!("\nper-partial T60, seconds: the engine's band around partial k against the recording's");
    print!("{:>4} {:>5}", "key", "note");
    for k in PARTIAL_PROBE {
        print!("{:>16}", format!("k={k}"));
    }
    println!("   (eng / ref)");
    for &key in &PROBE_KEYS {
        let Some(r) = rows.iter().find(|r| r.key == key) else {
            continue;
        };
        print!("{:>4} {:>5}", key, note_name(key));
        for (e, f) in &r.partial_t60 {
            print!(
                "{:>16}",
                match (e, f) {
                    (Some(e), Some(f)) => format!("{e:6.2} /{f:6.2}"),
                    (Some(e), None) => format!("{e:6.2} /     -"),
                    (None, Some(f)) => format!("     - /{f:6.2}"),
                    _ => "     - /     -".to_string(),
                }
            );
        }
        println!();
    }

    // ---- the compass played at once ---------------------------------------
    //
    // The per-key numbers above scatter by 5-17 dB, and they are meant to: one
    // key's 2 kHz band holds a dozen partials whose individual levels the
    // engine already misses by `COMPASS.md`'s `match`, 7-24 dB. What a listener
    // hears is not one of those keys, it is all of them, so the statistic that
    // answers the listening note is the **sum**: every key level-matched to its
    // own recording first, then the whole compass added up band by band. The
    // per-partial scatter cancels there and a systematic tilt does not.
    println!("\ncompass summed, engine - reference, level-matched per key (dB):");
    let labels: Vec<String> = OCTAVES
        .iter()
        .map(|c| format!("{c:.0}"))
        .chain(["2-6k".into(), "6-12k".into()])
        .collect();
    print!("{:<20}", "band Hz");
    for l in &labels {
        print!("{l:>8}");
    }
    println!();
    let summed = |pick: fn(&Row) -> &[Vec<f64>; INSTANTS.len()],
                  i: usize,
                  b: usize,
                  keys: (u8, u8)|
     -> f64 {
        let rows: Vec<&Row> = rows
            .iter()
            .filter(|r| (keys.0..=keys.1).contains(&r.key))
            .collect();
        let (mut num, mut den) = (0.0, 0.0);
        for r in &rows {
            let (v, rf) = (pick(r), &r.reference_bands[i]);
            if v[i].is_empty() || rf.is_empty() {
                continue;
            }
            num += v[i][b];
            den += rf[b];
        }
        db(num / den)
            - db(rows
                .iter()
                .filter(|r| !pick(r)[i].is_empty())
                .map(|r| pick(r)[i][bands.len() - 1])
                .sum::<f64>()
                / rows
                    .iter()
                    .filter(|r| !r.reference_bands[i].is_empty())
                    .map(|r| r.reference_bands[i][bands.len() - 1])
                    .sum::<f64>())
    };
    for (i, t) in INSTANTS.iter().enumerate() {
        for &(name, lo, hi) in registers.iter().chain(&[("ALL", FIRST_KEY, LAST_KEY)]) {
            print!("{:<20}", format!("{name} @{t} s"));
            for b in 0..labels.len() {
                print!("{:>+8.2}", summed(|r| &r.engine_bands, i, b, (lo, hi)));
            }
            println!();
        }
        print!("{:<20}", format!("layer floor @{t} s"));
        for b in 0..labels.len() {
            print!("{:>+8.2}", summed(|r| &r.layer_bands, i, b, (FIRST_KEY, LAST_KEY)));
        }
        println!();
    }

    // ---- the phrases ------------------------------------------------------
    let mut phrase_key = cache::Fingerprint::new();
    phrase_key
        .str("realism-bench/reference")
        .u64(u64::from(SAMPLER_VERSION))
        .file(&sfz)?
        .u64(u64::from(SAMPLE_RATE))
        .u64(u64::from(PHRASE_SET_VERSION));

    println!("\nphrases (engine - reference, level-matched; floor is the layer next door)");
    println!(
        "{:<18} {:>9} {:>10} {:>12} {:>9} {:>10} {:>12}",
        "phrase", "2-6k", "6-12k", "tilt dB/oct", "f 2-6k", "f 6-12k", "f tilt"
    );
    let phrases: Vec<Phrase> = realism::phrase_set();
    let phrase_rows: Vec<PhraseRow> = phrases
        .into_par_iter()
        .map(|phrase| -> Result<PhraseRow, piano_tuner::Error> {
            let engine = render_phrase(&preset, &phrase);
            let cached = |name: &str, events: &[TimedEvent]| -> Result<Audio, piano_tuner::Error> {
                let mut key = phrase_key;
                key.str(name).str(phrase.name).f64(phrase.duration_s);
                let path = reference_cache
                    .join(format!("realism-{}-{name}-{}.wav", phrase.name, key.hex()));
                cache::audio(&path, || {
                    with_sampler(&sfz, |s| s.render(events, phrase.duration_s))
                })
            };
            let reference = cached("reference", &phrase.events)?;
            let alt = cached("alt-layer", &layers.shift(&phrase.events))?;
            let e = total_power(&stft, &engine.mono());
            let r = total_power(&stft, &reference.mono());
            let a = total_power(&stft, &alt.mono());
            Ok(PhraseRow {
                name: phrase.name.to_string(),
                v: [
                    hf_ratio(&e, &r, SR, HF1),
                    hf_ratio(&e, &r, SR, HF2),
                    tilt(&e, &r),
                    hf_ratio(&a, &r, SR, HF1),
                    hf_ratio(&a, &r, SR, HF2),
                    tilt(&a, &r),
                ],
                engine_oct: octave_diff(&e, &r),
                floor_oct: octave_diff(&a, &r),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    for row in &phrase_rows {
        println!(
            "{:<18} {:>+9.2} {:>+10.2} {:>+12.2} {:>+9.2} {:>+10.2} {:>+12.2}",
            row.name, row.v[0], row.v[1], row.v[2], row.v[3], row.v[4], row.v[5]
        );
    }
    println!(
        "{:<18} {:>+9.2} {:>+10.2} {:>+12.2} {:>+9.2} {:>+10.2} {:>+12.2}",
        "mean",
        mean(phrase_rows.iter().map(|r| r.v[0])),
        mean(phrase_rows.iter().map(|r| r.v[1])),
        mean(phrase_rows.iter().map(|r| r.v[2])),
        mean(phrase_rows.iter().map(|r| r.v[3])),
        mean(phrase_rows.iter().map(|r| r.v[4])),
        mean(phrase_rows.iter().map(|r| r.v[5])),
    );

    println!("\noctave bands, engine - reference, level-matched (dB):");
    print!("{:<18}", "band Hz");
    for c in OCTAVES {
        print!("{c:>8.0}");
    }
    println!();
    for row in &phrase_rows {
        print!("{:<18}", row.name);
        for v in &row.engine_oct {
            print!("{v:>+8.2}");
        }
        println!();
    }
    print!("{:<18}", "mean");
    for i in 0..OCTAVES.len() {
        print!("{:>+8.2}", mean(phrase_rows.iter().map(|r| r.engine_oct[i])));
    }
    println!();
    print!("{:<18}", "floor |.|");
    for i in 0..OCTAVES.len() {
        print!("{:>8.2}", mean(phrase_rows.iter().map(|r| r.floor_oct[i].abs())));
    }
    println!();
    Ok(())
}

struct PhraseRow {
    name: String,
    /// `[2-6k, 6-12k, tilt, floor 2-6k, floor 6-12k, floor tilt]`.
    v: [f64; 6],
    engine_oct: Vec<f64>,
    floor_oct: Vec<f64>,
}

/// The engine-minus-reference difference in each octave band, level-matched.
fn octave_diff(engine: &[f64], reference: &[f64]) -> Vec<f64> {
    let offset = db(band(engine, SR, FULL) / band(reference, SR, FULL));
    OCTAVES
        .iter()
        .map(|&c| {
            let bnd = (c / std::f64::consts::SQRT_2, c * std::f64::consts::SQRT_2);
            db(band(engine, SR, bnd) / band(reference, SR, bnd)) - offset
        })
        .collect()
}

/// Least-squares slope in dB per octave of the engine-minus-reference
/// octave-band difference, over the bands from [`TILT_FROM_HZ`] up.
fn tilt(engine: &[f64], reference: &[f64]) -> f64 {
    let offset = db(band(engine, SR, FULL) / band(reference, SR, FULL));
    let points: Vec<(f64, f64)> = OCTAVES
        .iter()
        .filter(|&&c| c >= TILT_FROM_HZ)
        .map(|&c| {
            let bnd = (c / std::f64::consts::SQRT_2, c * std::f64::consts::SQRT_2);
            (
                c.log2(),
                db(band(engine, SR, bnd) / band(reference, SR, bnd)) - offset,
            )
        })
        .collect();
    let n = points.len() as f64;
    let mx = points.iter().map(|p| p.0).sum::<f64>() / n;
    let my = points.iter().map(|p| p.1).sum::<f64>() / n;
    let num: f64 = points.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
    let den: f64 = points.iter().map(|p| (p.0 - mx).powi(2)).sum();
    if den == 0.0 {
        f64::NAN
    } else {
        num / den
    }
}
