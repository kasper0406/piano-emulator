//! Every key of the compass, struck alone, scored against its own neighbours
//! and against the recording of the same note.
//!
//! The scoreboard in `renders/realism/REALISM.md` is six phrases and one number
//! per column: it says how far the instrument is from the recordings *on
//! average over the compass*. That average is exactly the wrong statistic for
//! the failure this tool exists to catch — one key that does not fit the rest.
//! A single sour note is 1/88th of a phrase and it moves a mean log-mel
//! distance by hundredths of a decibel, while a listener finds it in one pass.
//!
//! So: 88 solo notes at one velocity, seven numbers each, and a robust outlier
//! score. The score is **relative to the key's own neighbours**, because
//! everything about a piano changes down the compass — level, brightness,
//! decay, how much a unison beats — and none of that is a defect. What is a
//! defect is a discontinuity: a key that differs from the keys around it by far
//! more than any other key differs from its own.
//!
//! # The seven numbers
//!
//! Measured identically on the engine's render and on the sampler's, over the
//! same window, so every one of them can also be quoted as a difference.
//!
//! | metric | what it catches |
//! |---|---|
//! | `level` | RMS of the first second, dBFS — a key that is loud or weak |
//! | `centroid` | spectral centroid in semitones over `f0` — a key of the wrong colour, register-free |
//! | `irregular` | mean absolute step between adjacent partial levels, dB — a *jagged* harmonic series, which is what a bad `notes.partial_gains` row makes and what no smooth model can make |
//! | `beat` | median [`Motion::beat_depth_db`] over the measured partials — a key that wobbles |
//! | `jitter` | median [`Motion::band_cents`] — a key whose pitch moves |
//! | `match` | mean absolute per-partial distance from the recording of the same note, dB, common offset removed — a key of the wrong colour, measured against the piano rather than against its neighbours |
//! | `decay` | median [`Motion::tail_db_s`] — a key that dies at the wrong rate |
//!
//! `irregular` is the one that is not in any existing report and it is the one
//! that found the note this tool was written for. Every mechanism in the engine
//! that shapes a spectrum is smooth in `ln k` — the hammer, the bridge, the
//! comb, the microphone — with exactly one exception, the
//! `notes.partial_gains` row, which is a free number per partial. So a jagged
//! harmonic series is a *table* defect by construction, and the metric points
//! straight at the table that carries it.
//!
//! Since `DECISIONS.md` 284 that table covers most of the compass in two ways:
//! 28 keys carry rows measured against their own recordings, and 49 carry rows
//! **drawn** from those keys' distributions. The `sampled` column says which a
//! key is, because the two are meant to be indistinguishable here and that is
//! the claim this report exists to check.
//!
//! # The score
//!
//! Per metric, per key: the residual against the median of the [`NEIGHBOURS`]
//! nearest keys **strung the same way**, divided by a robust sigma
//! (`1.4826 * MAD`) taken over every residual the metric has. A key is flagged
//! on any metric past [`FLAG_Z`]. The reference's own metrics go through the
//! same mill so that the report can say whether the recording does the same
//! thing — a genuine oddity of the *piano* shows up in both columns and is not
//! a bug.
//!
//! ```sh
//! cargo run --release -p piano-tuner -- compass \
//!     data/salamander renders/compass presets/salamander-c5.toml
//! ```

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;

use piano_emulator::preset::Preset;
use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_emulator::types::Event;
use piano_tuner::cache;
use piano_tuner::motion::Motion;
use piano_tuner::realism::{StereoBand, StereoImage, STEREO_BANDS};
use piano_tuner::sampler::SAMPLER_VERSION;
use piano_tuner::series::{amp_db, Series, PARTIALS, WINDOW_S as SPECTRUM_WINDOW};
use piano_tuner::{
    audio, realism, Audio, SampleLibrary, Sampler, SamplerEvent, TimedEvent, SAMPLE_RATE,
};

/// Velocity every key is struck at: the middle layer, the one the fits and the
/// motion columns both use.
const VELOCITY: u8 = 90;

/// Seconds of note. The partial motion window ends at 3.0 s.
const RENDER_S: f64 = 3.6;

/// Silence before the strike, in samples.
///
/// `realism::STEREO_PREROLL_SAMPLES`, which is `tuner/tests/stereo.rs`'s and
/// `tools::mics`'s, because this board reads a **stereo image** of the same
/// material and a window that does not open at the strike measures its own edge
/// (`DECISIONS.md` 378). A whole number of engine blocks, asserted here as it is
/// there: the number is only right by construction if something checks.
const PREROLL: usize = realism::STEREO_PREROLL_SAMPLES;

const _: () = assert!(
    PREROLL % piano_emulator::types::BLOCK == 0,
    "the preroll must be a whole number of engine blocks or the window starts inside the note"
);

const PREROLL_S: f64 = PREROLL as f64 / 48_000.0;

/// How many neighbours a key is scored against.
///
/// The neighbours are the nearest keys **strung the same way** — the same number
/// of unison strings — and not simply the nearest keys. A two-string key beats
/// where a one-string key cannot, so the 1→2 and 2→3 boundaries are step changes
/// in `beat` and `jitter` that belong to the instrument. Scoring across them
/// convicts four keys of being the boundary. Eight is wide enough that one bad
/// key cannot set its own baseline and narrow enough that the register's real
/// trend is flat across it: on this compass the eight nearest same-`N` keys of a
/// boundary key span at most nine semitones.
const NEIGHBOURS: usize = 8;

/// How many band-pass widths apart two partials must be before the motion
/// metrics are taken at all.
///
/// `motion.rs`'s Gaussian band-pass is 31.8 Hz wide, and its own header says why
/// that number matters: "the nearest neighbouring partial of the lowest key the
/// columns use (A2, 110 Hz apart) is 3.5 sigma out and 54 dB down". Below A2 it
/// is not. At C2 the partials are 65 Hz apart, two sigma, and the fundamental's
/// track is a mixture of the fundamental and 18 dB of whatever the second
/// partial is doing — which at C2 is 25 dB louder. `beat`, `jitter` and `decay`
/// are therefore not measured under this separation on either signal, rather
/// than measured and compared: two contaminated readings do not cancel. The
/// keys it silences (A0-F#2 on this compass) keep `level`, `centroid` and
/// `irregular`, and `irregular` is the metric that found the note this tool was
/// written for.
const MIN_PARTIAL_SIGMAS: f64 = 3.0;

/// Robust `z` at which a key is called an outlier.
///
/// Chosen the only way a threshold like this can be: on the measured
/// distribution. Over the compass the residuals are Gaussian-cored with a
/// handful of tails, and 4.0 is where the flagged set stops growing one key at
/// a time and starts admitting the ordinary scatter of the fitted keys.
const FLAG_Z: f64 = 4.0;

/// The lowest and highest MIDI key of an 88-key piano.
const FIRST_KEY: u8 = 21;
const LAST_KEY: u8 = 108;

/// How many decoded recordings one worker's sampler holds before it lets them
/// go.
///
/// This replaces a `clear_cache()` every twelfth key with the bound that was
/// actually meant: the scan touches every key once, so nothing is ever reused
/// and the cache is pure high-water mark. A few dozen recordings is a few
/// hundred megabytes; eight is the working set of one key with room for its
/// release group.
const MAX_CACHED_BUFFERS: usize = 8;

thread_local! {
    /// One reference player per worker thread.
    ///
    /// The scan is data-parallel because a render depends on nothing but its own
    /// events: [`Sampler::render`] seeds its round-robin draw from
    /// [`SamplerConfig::seed`](piano_tuner::SamplerConfig) at the top of every
    /// call, and the decoded-buffer cache is a speed device that no output
    /// depends on. So a key rendered on its own player is the same bytes as the
    /// same key rendered eighty-eighth on a shared one, which is the property
    /// that lets this loop run on as many threads as there are cores and still
    /// write the same `COMPASS.md`.
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

pub fn run(args: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
    let mut args = args.into_iter();
    let data = PathBuf::from(args.next().unwrap_or_else(|| "data/salamander".into()));
    let out = PathBuf::from(args.next().unwrap_or_else(|| "renders/compass".into()));
    let preset_path = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "presets/salamander-c5.toml".into()),
    );
    // Remaining arguments are keys to print a per-partial table for.
    let detail: Vec<u8> = args.filter_map(|a| a.parse::<u8>().ok()).collect();

    // Whichever library this tree is, rather than Salamander's own filename:
    // `adapter::instrument_path` resolves a described library through its
    // LibrarySpec and an undescribed one by its single map (DECISIONS.md 521).
    let sfz = piano_tuner::adapter::instrument_path(&data)?;
    if !sfz.exists() {
        eprintln!(
            "the reference piano is not here: {}\nrun data/fetch_salamander.sh first (707 MiB).",
            sfz.display()
        );
        std::process::exit(2);
    }
    std::fs::create_dir_all(&out)?;

    let preset = Preset::load(&preset_path)?;
    let library = SampleLibrary::from_sfz(&sfz)?;
    let recorded_keys = realism::RecordedKeys::from_library(&library)?;
    let sampled: Vec<u8> = recorded_keys.keys().to_vec();

    // The reference side of the scan, keyed by everything it is a function of.
    // The engine side is deliberately *not* cached: it is the thing under test
    // and it is cheap, so it is re-rendered on every run.
    let reference_cache = cache::reference_dir(&data);
    let mut reference_key = cache::Fingerprint::new();
    reference_key
        .str("compass-scan/reference")
        .u64(u64::from(SAMPLER_VERSION))
        .file(&sfz)?
        .u64(u64::from(SAMPLE_RATE))
        .u64(u64::from(VELOCITY))
        .f64(RENDER_S);

    println!(
        "compass scan: vel {VELOCITY}, {} keys, engine on {}",
        LAST_KEY - FIRST_KEY + 1,
        preset_path.display()
    );
    println!("  reference cache: {}", reference_cache.display());

    // One key is one independent measurement: two renders that share no state,
    // and a fixed set of numbers taken off them. Rayon runs them across the
    // cores and `collect` puts them back in key order, so `COMPASS.md` is the
    // same file at any thread count.
    let done = AtomicUsize::new(0);
    let keys: Vec<u8> = (FIRST_KEY..=LAST_KEY).collect();
    let measured: Vec<(Scan, Option<String>)> = keys
        .par_iter()
        .map(
            |&key| -> Result<(Scan, Option<String>), piano_tuner::Error> {
                let params = preset.string_params(key);
                let partial_hz: Vec<f64> = (1..=PARTIALS)
                    .map(|k| f64::from(params.partial_freq(k)))
                    .collect();
                let engine_audio = render_engine(&preset, key);
                let mut key_print = reference_key;
                key_print.u64(u64::from(key));
                let path =
                    reference_cache.join(format!("compass-key{key:03}-{}.wav", key_print.hex()));
                let reference_audio =
                    cache::audio(&path, || with_sampler(&sfz, |s| render_reference(s, key)))?;
                let (reference, reference_spectrum) =
                    Metrics::measure(&reference_audio.mono(), &partial_hz, None);
                // `match` is the one metric of the seven that scores the engine
                // against *the recording of this key*, and two keys in three have
                // no recording of their own: what the player answers with there is
                // a neighbour's take, resampled. Rendering it is right — it is what
                // a listener hears — but scoring against it measures the resampler,
                // so the column is left empty and the key is marked
                // `transposed — unscored` (`DECISIONS.md` 328).
                let recorded = sampled.contains(&key);
                let (mut engine, _) = Metrics::measure(
                    &engine_audio.mono(),
                    &partial_hz,
                    recorded.then_some(&reference_spectrum),
                );
                if !recorded {
                    engine.match_db = f64::NAN;
                }
                // The STEREO column, on the two channels rather than the mono sum
                // every metric above is measured from (`DECISIONS.md` 314, 317 (a)).
                // One band per key — the one this key's fundamental falls in — so
                // that the report is a line down the compass rather than a surface,
                // and so that the band quoted is always one the note actually
                // fills.
                let stereo_band = StereoImage::band_for(partial_hz[0]);
                let stereo_engine = realism::stereo_image_of(&engine_audio)?.bands[stereo_band];
                let stereo_reference =
                    realism::stereo_image_of(&reference_audio)?.bands[stereo_band];
                let detail_text = detail
                    .contains(&key)
                    .then(|| detail_report(key, &partial_hz, &engine_audio, &reference_audio, &out))
                    .transpose()?;
                let count = done.fetch_add(1, Ordering::Relaxed) + 1;
                print!("\r  {count} keys   ");
                use std::io::Write;
                std::io::stdout().flush().ok();
                Ok((
                    Scan {
                        key,
                        f0: partial_hz[0],
                        unison: usize::from(preset.notes.unison[usize::from(key - FIRST_KEY)]),
                        sampled: recorded,
                        engine,
                        reference,
                        stereo_band,
                        stereo_engine,
                        stereo_reference,
                    },
                    detail_text,
                ))
            },
        )
        .collect::<Result<Vec<_>, _>>()?;
    println!("\r  {} keys rendered      ", measured.len());

    // The detail tables are printed here rather than where they were measured,
    // so that they come out in key order however the work was scheduled.
    let mut scans: Vec<Scan> = Vec::with_capacity(measured.len());
    for (scan, detail_text) in measured {
        if let Some(text) = detail_text {
            print!("{text}");
        }
        scans.push(scan);
    }

    let readable: Vec<&Scan> = scans
        .iter()
        .filter(|sc| sc.stereo_engine.readable() && sc.stereo_reference.readable())
        .collect();
    let inverted = readable
        .iter()
        .filter(|sc| sc.stereo_engine.r0 * sc.stereo_reference.r0 < 0.0)
        .count();
    println!(
        "\nSTEREO r@0 in each key's fundamental band (not a mono sum): engine {:+.3}, \
recording {:+.3} (medians over {} readable keys); {inverted} of them on opposite sides of zero",
        median(
            &readable
                .iter()
                .map(|sc| sc.stereo_engine.r0)
                .collect::<Vec<_>>()
        ),
        median(
            &readable
                .iter()
                .map(|sc| sc.stereo_reference.r0)
                .collect::<Vec<_>>()
        ),
        readable.len(),
    );

    let engine_z = score(&scans, |s| s.engine);
    let reference_z = score(&scans, |s| s.reference);
    let report = out.join("COMPASS.md");
    std::fs::write(
        &report,
        write_report(&scans, &engine_z, &reference_z, &preset_path, &sfz),
    )?;

    let flags = flagged(&scans, &engine_z);
    println!("\noutliers (|z| >= {FLAG_Z:.1}):");
    if flags.is_empty() {
        println!("  none");
    }
    for f in &flags {
        println!(
            "  {:>4} {:<4} n={}  {:<10} z {:6.1}   (recording z {:5.1})",
            scans[f.key_index].key,
            note_name(scans[f.key_index].key),
            scans[f.key_index].unison,
            f.metric,
            f.z,
            reference_z[f.key_index].get(f.metric),
        );
    }
    println!(
        "  {} flags over {} keys",
        flags.len(),
        flags
            .iter()
            .map(|f| f.key_index)
            .collect::<std::collections::BTreeSet<_>>()
            .len()
    );
    println!("\n{}", report.display());
    Ok(())
}

// ---------------------------------------------------------------------------
// Rendering one key
// ---------------------------------------------------------------------------

fn render_engine(preset: &Preset, key: u8) -> Audio {
    let events = [RenderEvent::new(
        PREROLL_S as f32,
        Event::NoteOn {
            key,
            vel: u16::from(VELOCITY),
        },
    )];
    let (left, right) = render_to_buffer(preset, &events, (PREROLL_S + RENDER_S) as f32);
    debug_assert_eq!(events[0].frame(), PREROLL);
    Audio::new(
        SAMPLE_RATE,
        vec![left[PREROLL..].to_vec(), right[PREROLL..].to_vec()],
    )
    .expect("the engine renders stereo")
}

/// The recording of the same note, aligned on its own onset so that the two
/// windows measure the same part of the note. The sampler's own attack offset
/// is a property of where the microphone was, not of the piano.
fn render_reference(sampler: &mut Sampler, key: u8) -> Result<Audio, piano_tuner::Error> {
    let events = [TimedEvent::new(
        0.0,
        SamplerEvent::NoteOn { key, vel: VELOCITY },
    )];
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
    let channels: Vec<Vec<f32>> = rendered.channels.iter().map(cut).collect();
    Audio::new(SAMPLE_RATE, channels)
}

// ---------------------------------------------------------------------------
// The six numbers
// ---------------------------------------------------------------------------

/// Names in the order [`Metrics::values`] returns them.
const METRIC_NAMES: [&str; 7] = [
    "level",
    "centroid",
    "irregular",
    "match",
    "beat",
    "jitter",
    "decay",
];

#[derive(Clone, Copy, Debug, Default)]
struct Metrics {
    /// RMS of the first second, dBFS.
    level_db: f64,
    /// Spectral centroid of the partial series, in semitones over `f0`.
    centroid_st: f64,
    /// Mean absolute step between adjacent partial levels, dB.
    irregular_db: f64,
    /// Mean absolute per-partial distance from the recording of the same note,
    /// dB, after the common offset between the two has been removed. Zero on
    /// the recording's own column by construction.
    match_db: f64,
    /// Median beat depth over the partials that were measurable, dB. `NaN`
    /// where the partials are too close together to be measured apart — see
    /// [`MIN_PARTIAL_SIGMAS`].
    beat_db: f64,
    /// Median band-limited frequency deviation, cents. `NaN` under
    /// [`MIN_PARTIAL_SIGMAS`].
    jitter_cents: f64,
    /// Median tail slope, dB/s. `NaN` under [`MIN_PARTIAL_SIGMAS`].
    decay_db_s: f64,
}

impl Metrics {
    fn values(&self) -> [f64; METRIC_NAMES.len()] {
        [
            self.level_db,
            self.centroid_st,
            self.irregular_db,
            self.match_db,
            self.beat_db,
            self.jitter_cents,
            self.decay_db_s,
        ]
    }

    /// `against` is the spectrum of the recording of the same note, or `None`
    /// when this *is* the recording.
    fn measure(mono: &[f32], partial_hz: &[f64], against: Option<&Series>) -> (Metrics, Series) {
        let sr = f64::from(SAMPLE_RATE);
        let lo = (SPECTRUM_WINDOW.0 * sr) as usize;
        let hi = ((SPECTRUM_WINDOW.1 * sr) as usize).min(mono.len());
        let window = &mono[lo.min(hi)..hi];
        let level_db = amp_db(realism::rms(window));

        let spectrum = Series::measure(window, partial_hz, sr);
        let centroid_st = spectrum.centroid_semitones();
        let irregular_db = spectrum.irregularity();
        let match_db = against.map_or(0.0, |r| spectrum.distance_from(r));

        // The partials of this key have to be resolvable apart before anything
        // that reads one partial's own movement means anything.
        let band_hz = 1.0 / (std::f64::consts::TAU * piano_tuner::motion::SMOOTH_SIGMA_S);
        if partial_hz[0] < MIN_PARTIAL_SIGMAS * band_hz {
            return (
                Metrics {
                    level_db,
                    centroid_st,
                    irregular_db,
                    match_db,
                    beat_db: f64::NAN,
                    jitter_cents: f64::NAN,
                    decay_db_s: f64::NAN,
                },
                spectrum,
            );
        }
        let signal: Vec<f64> = mono.iter().map(|&v| f64::from(v)).collect();
        let motions = realism::measure_partials(&signal, partial_hz);
        let measured: Vec<Motion> = motions.iter().flatten().copied().collect();
        (
            Metrics {
                level_db,
                centroid_st,
                irregular_db,
                match_db,
                beat_db: median(&measured.iter().map(|m| m.beat_depth_db).collect::<Vec<_>>()),
                jitter_cents: median(
                    &measured
                        .iter()
                        .map(|m| m.floored_cents())
                        .collect::<Vec<_>>(),
                ),
                decay_db_s: median(&measured.iter().map(|m| m.tail_db_s).collect::<Vec<_>>()),
            },
            spectrum,
        )
    }
}

// ---------------------------------------------------------------------------
// Scoring against the neighbours
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Scan {
    key: u8,
    f0: f64,
    unison: usize,
    sampled: bool,
    engine: Metrics,
    reference: Metrics,
    /// Which of [`STEREO_BANDS`] this key's fundamental falls in, clamped at
    /// both ends — A0's 27.5 Hz is under the lowest band and reads there.
    stereo_band: usize,
    /// The engine's and the recording's interchannel behaviour in that band.
    /// **Not a mono sum**, and not scored against the neighbours the seven
    /// metrics above are: the recording's own r@0 is a fact about where the
    /// microphones were, which changes smoothly down the compass and is not a
    /// thing a key can be an outlier of.
    stereo_engine: StereoBand,
    stereo_reference: StereoBand,
}

/// One key's six robust `z` scores.
#[derive(Clone, Copy, Debug, Default)]
struct Zs([f64; METRIC_NAMES.len()]);

impl Zs {
    fn worst(&self) -> (&'static str, f64) {
        let mut best = (METRIC_NAMES[0], 0.0f64);
        for (name, &z) in METRIC_NAMES.iter().zip(&self.0) {
            if z.abs() > best.1 {
                best = (name, z.abs());
            }
        }
        best
    }

    fn get(&self, name: &str) -> f64 {
        METRIC_NAMES
            .iter()
            .position(|&n| n == name)
            .map(|i| self.0[i])
            .unwrap_or(0.0)
    }
}

/// One key, one metric, over the line.
struct Flag {
    key_index: usize,
    metric: &'static str,
    z: f64,
}

/// Every `(key, metric)` pair at or over [`FLAG_Z`], worst first.
///
/// Per pair rather than per key: a key can be wrong in two ways at once, and
/// which ways it is wrong is the attribution. Reporting only its worst metric
/// throws away exactly the corroboration a reader needs.
fn flagged(scans: &[Scan], zs: &[Zs]) -> Vec<Flag> {
    let mut out: Vec<Flag> = Vec::new();
    for (key_index, z) in zs.iter().enumerate() {
        for (metric, &value) in METRIC_NAMES.iter().zip(&z.0) {
            if value.abs() >= FLAG_Z {
                out.push(Flag {
                    key_index,
                    metric,
                    z: value,
                });
            }
        }
    }
    let _ = scans;
    out.sort_by(|a, b| b.z.abs().total_cmp(&a.z.abs()));
    out
}

/// Metrics whose population is the keys the library actually **recorded**.
///
/// `match` is the only one of the seven that compares the engine against the
/// recording of the same note, so it is the only one the transposition policy
/// of `DECISIONS.md` 328 touches. Its neighbourhood is drawn from recorded keys
/// too: a residual against eight neighbours of which six carry no score is not
/// a residual, and a `match` column scored against a transposed take was
/// measuring the resampler on both sides of the subtraction.
const RECORDED_ONLY: [&str; 1] = ["match"];

/// The indices of the [`NEIGHBOURS`] nearest keys strung the same way as `i`,
/// nearest first, excluding `i` itself.
///
/// This is the comparison the whole report rests on and it is a judgement about
/// what a piano is: a key is odd when it does not fit the keys *of its own
/// kind*. Every metric here steps at the 1→2 and 2→3 string boundaries because
/// the number of strings is what a unison beat is made of, and a scan that
/// ignored that would spend its flags convicting four keys of being a boundary.
///
/// `recorded_only` narrows the candidates to keys the library recorded, which
/// is what a [`RECORDED_ONLY`] metric's population is.
fn same_kind_neighbours_of(scans: &[Scan], i: usize, recorded_only: bool) -> Vec<usize> {
    let mut candidates: Vec<usize> = (0..scans.len())
        .filter(|&j| j != i && scans[j].unison == scans[i].unison)
        .filter(|&j| !recorded_only || scans[j].sampled)
        .collect();
    candidates.sort_by_key(|&j| (scans[j].key as i32 - scans[i].key as i32).abs());
    candidates.truncate(NEIGHBOURS);
    candidates
}

/// The residual of one metric at one key: its value less the median of its
/// same-kind neighbours. `None` where the metric was not measured, at the key
/// or at its neighbours.
fn residual_at(scans: &[Scan], column: &[f64], i: usize, recorded_only: bool) -> Option<f64> {
    if !column[i].is_finite() {
        return None;
    }
    let neighbours: Vec<f64> = same_kind_neighbours_of(scans, i, recorded_only)
        .into_iter()
        .map(|j| column[j])
        .filter(|v| v.is_finite())
        .collect();
    // The recorded population is a third as dense as the compass, and at the
    // ends of it a key can have only three same-`N` takes in the whole library.
    // Requiring four of eight there would silence the metric exactly where the
    // library is thinnest, which is a different defect from the one the rule is
    // for; three is still a median.
    let minimum = if recorded_only { 3 } else { NEIGHBOURS / 2 };
    (neighbours.len() >= minimum).then(|| column[i] - median(&neighbours))
}

fn score(scans: &[Scan], pick: impl Fn(&Scan) -> Metrics) -> Vec<Zs> {
    let series: Vec<[f64; METRIC_NAMES.len()]> = scans.iter().map(|s| pick(s).values()).collect();
    let mut out = vec![Zs::default(); scans.len()];
    for m in 0..METRIC_NAMES.len() {
        let recorded_only = RECORDED_ONLY.contains(&METRIC_NAMES[m]);
        let column: Vec<f64> = series.iter().map(|v| v[m]).collect();
        let residual: Vec<Option<f64>> = (0..column.len())
            .map(|i| residual_at(scans, &column, i, recorded_only))
            .collect();
        // The scale is taken over the residuals that exist. A key the metric
        // could not be measured at contributes nothing rather than a zero: a
        // quarter of the compass reading exactly zero would halve the MAD and
        // double every other key's score.
        let present: Vec<f64> = residual.iter().flatten().copied().collect();
        let sigma = mad_sigma(&present);
        for (i, r) in residual.iter().enumerate() {
            out[i].0[m] = match (r, sigma > 0.0) {
                (Some(r), true) => r / sigma,
                _ => 0.0,
            };
        }
    }
    out
}

fn median(values: &[f64]) -> f64 {
    let mut v: Vec<f64> = values.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// `1.4826 * MAD` — the consistent estimator of a Gaussian's sigma that a
/// handful of outliers cannot move, which is the whole reason the threshold is
/// stated on it rather than on a standard deviation the outliers would inflate.
fn mad_sigma(residual: &[f64]) -> f64 {
    let centre = median(residual);
    let deviations: Vec<f64> = residual.iter().map(|r| (r - centre).abs()).collect();
    1.4826 * median(&deviations)
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

/// One key's per-partial table, as text rather than as a print.
///
/// It is built rather than printed because the scan is parallel: the tables are
/// emitted in key order by the caller, which is what keeps a run's console
/// output — and not only its files — the same whatever order the work finished
/// in. The two WAVs it writes are named by key, so writing them here is
/// order-free.
fn detail_report(
    key: u8,
    partial_hz: &[f64],
    engine: &Audio,
    reference: &Audio,
    out: &Path,
) -> Result<String, piano_tuner::Error> {
    use std::fmt::Write as _;
    let mut s = String::new();
    let sr = f64::from(SAMPLE_RATE);
    let lo = (SPECTRUM_WINDOW.0 * sr) as usize;
    let hi = (SPECTRUM_WINDOW.1 * sr) as usize;
    let e_mono = engine.mono();
    let r_mono = reference.mono();
    let e_spec = Series::measure(&e_mono[lo..hi.min(e_mono.len())], partial_hz, sr);
    let r_spec = Series::measure(&r_mono[lo..hi.min(r_mono.len())], partial_hz, sr);
    let (e_lev, r_lev) = (&e_spec.levels_db, &r_spec.levels_db);
    let e_sig: Vec<f64> = e_mono.iter().map(|&v| f64::from(v)).collect();
    let r_sig: Vec<f64> = r_mono.iter().map(|&v| f64::from(v)).collect();
    let e_mot = realism::measure_partials(&e_sig, partial_hz);
    let r_mot = realism::measure_partials(&r_sig, partial_hz);
    let _ = writeln!(s, "\n  key {key} ({}):", note_name(key));
    let _ = writeln!(
        s,
        "    {:>2} {:>8}  {:>7} {:>7} {:>7} {:>4} | {:>6} {:>6} | {:>6} {:>6} | {:>7} {:>7}",
        "k",
        "hz",
        "eng dB",
        "ref dB",
        "diff",
        "seen",
        "eBeat",
        "rBeat",
        "eRate",
        "rRate",
        "eTail",
        "rTail"
    );
    for k in 0..partial_hz.len() {
        let f = |m: &Option<Motion>, g: fn(&Motion) -> f64| {
            m.as_ref()
                .map(g)
                .map_or("  --  ".to_string(), |v| format!("{v:6.2}"))
        };
        let _ = writeln!(
            s,
            "    {:>2} {:>8.1}  {:>7.1} {:>7.1} {:>+7.1} {:>4} | {} {} | {} {} | {:>7} {:>7}",
            k + 1,
            partial_hz[k],
            e_lev[k],
            r_lev[k],
            e_lev[k] - r_lev[k],
            match (e_spec.present[k], r_spec.present[k]) {
                (true, true) => "both",
                (true, false) => "eng",
                (false, true) => "ref",
                (false, false) => "-",
            },
            f(&e_mot[k], |m| m.beat_depth_db),
            f(&r_mot[k], |m| m.beat_depth_db),
            f(&e_mot[k], |m| m.beat_rate_hz),
            f(&r_mot[k], |m| m.beat_rate_hz),
            f(&e_mot[k], |m| m.tail_db_s),
            f(&r_mot[k], |m| m.tail_db_s),
        );
    }
    let dir = out.join("detail");
    std::fs::create_dir_all(&dir)?;
    let name = note_name(key);
    write_normalised(&dir.join(format!("key{key}_{name}_engine.wav")), engine)?;
    write_normalised(
        &dir.join(format!("key{key}_{name}_reference.wav")),
        reference,
    )?;
    Ok(s)
}

/// Writes a solo note at a level a listener can compare: peak to -3 dBFS.
fn write_normalised(path: &Path, audio: &Audio) -> Result<(), piano_tuner::Error> {
    let peak = audio
        .channels
        .iter()
        .flat_map(|c| c.iter())
        .fold(0.0f32, |m, &v| m.max(v.abs()));
    let gain = if peak > 0.0 { 0.708 / peak } else { 1.0 };
    let scaled: Vec<Vec<f32>> = audio
        .channels
        .iter()
        .map(|c| c.iter().map(|&v| v * gain).collect())
        .collect();
    Audio::new(audio.sample_rate, scaled)?.write_wav(path)?;
    Ok(())
}

fn write_report(
    scans: &[Scan],
    engine_z: &[Zs],
    reference_z: &[Zs],
    preset: &Path,
    sfz: &Path,
) -> String {
    let mut s = String::new();
    s.push_str("# The compass, key by key\n\n");
    s.push_str(&format!(
        "Every key of an 88-note piano struck alone at velocity {VELOCITY}, held for {RENDER_S:.1} s, \
through the engine on `{}` and through the recordings of the Yamaha C5 at `{}`. Seven numbers each, \
and a robust outlier score against the nearest keys strung the same way.\n\n",
        preset.display(),
        sfz.display()
    ));
    s.push_str(
        "The scoreboard in `renders/realism/REALISM.md` averages over the compass, which is the \
wrong statistic for one sour key: a single bad note is 1/88th of a phrase and moves a mean by \
hundredths of a decibel, while a listener finds it in one pass. This report is that listener's \
statistic.\n\n",
    );
    s.push_str("## What is measured\n\n");
    s.push_str(
        "| metric | definition | what it catches |\n|---|---|---|\n\
| `level` | RMS of 0.10-1.10 s, dBFS | a key that is loud or weak |\n\
| `centroid` | power-weighted mean partial index, semitones over `f0` | a key of the wrong colour, register-free |\n\
| `irregular` | mean absolute step between adjacent partial levels, dB | a **jagged** harmonic series |\n\
| `match` | mean absolute per-partial distance from the recording of the same note, dB, common offset removed | a key of the wrong colour, measured against the piano itself |\n\
| `beat` | median `Motion::beat_depth_db` over the measurable partials | a key that wobbles |\n\
| `jitter` | median `Motion::band_cents` | a key whose pitch moves |\n\
| `decay` | median `Motion::tail_db_s` | a key that dies at the wrong rate |\n\n",
    );
    s.push_str(&format!(
        "`z` is the residual against the median of the {NEIGHBOURS} nearest keys **strung the same \
way**, over `1.4826 * MAD` taken across every residual the metric has. Same-`N` neighbours \
because the 1->2 and 2->3 string boundaries are step changes in `beat` and `jitter` that belong \
to the instrument: a scan that ignored them would spend its flags convicting four keys of being \
a boundary. A key is an **outlier** at `|z| >= {FLAG_Z:.1}`, on any one of the seven.\n\n\
`beat`, `jitter` and `decay` are not measured at all where a key's partials sit closer together \
than {MIN_PARTIAL_SIGMAS:.0} widths of the band-pass that has to separate them (31.8 Hz, \
`motion::SMOOTH_SIGMA_S`) - A0 to F#2 here. Under that separation the fundamental's track is a \
mixture of the fundamental and 18 dB of whatever the second partial is doing, on the recording \
exactly as much as on the engine, and two contaminated readings do not cancel. Those keys keep \
`level`, `centroid`, `irregular` and `match`, and `irregular` is the one that found the note \
this tool was written for.\n\n\
`irregular` is in no other report. Every mechanism in the engine that shapes a spectrum is \
smooth in `ln k` - the hammer, the bridge admittance, the strike comb, the microphone - with \
exactly one exception, the `notes.partial_gains` row, which is a free number per partial. \
A jagged harmonic series is therefore a *table* defect by construction, and the metric points \
at the table that carries it. Since `DECISIONS.md` 284 that table covers most of the compass in \
two ways: 28 keys carry rows measured against their own recordings, and 49 carry rows **drawn** \
from those keys' distributions. The `sampled` column says which a key is, because the two are \
meant to be indistinguishable here and that is the claim this report exists to check.\n\n\
## Which reference notes are scored\n\n\
`DECISIONS.md` 328. The library records one key every minor third and plays the other two in \
three by **transposing** the nearest take. A transposed reference note is listening material - \
it is exactly what a player of this library hears - but it is not a measurement of the piano at \
that key: its inharmonicity, its unison beat, its decay and its brightness are the neighbour's, \
resampled. Six of the seven metrics here are scored against a key's own *neighbours* and are \
unaffected. The seventh, `match`, is the distance from **the recording of the same note**, and \
that is only a fact where the recording exists. So `match` is measured, and its neighbourhood \
taken, on recorded keys alone; every other key is rendered, listened to, and marked \
`transposed - unscored`.\n\n"
    ));

    // ---- the ranked outlier list ----
    let flags = flagged(scans, engine_z);
    s.push_str("## Outliers, ranked\n\n");
    s.push_str(&format!(
        "One row per `(key, metric)` pair at or over `|z| = {FLAG_Z:.1}`, worst first: a key can \
be wrong in two ways at once, and which ways it is wrong is the attribution. `recording z` is \
the same metric's score on the *recording* of the same key - a genuine oddity of the piano \
shows up in both columns and is not a defect of the model.\n\n"
    ));
    s.push_str(
        "| key | note | N | sampled | metric | engine z | engine | neighbours | recording z |\n",
    );
    s.push_str("|---|---|---|---|---|---|---|---|---|\n");
    for f in &flags {
        let i = f.key_index;
        let column: Vec<f64> = scans
            .iter()
            .map(|sc| sc.engine.values()[metric_index(f.metric)])
            .collect();
        let neighbours: Vec<f64> =
            same_kind_neighbours_of(scans, i, RECORDED_ONLY.contains(&f.metric))
                .into_iter()
                .map(|j| column[j])
                .filter(|v| v.is_finite())
                .collect();
        s.push_str(&format!(
            "| {} | {} | {} | {} | `{}` | **{:+.1}** | {:.2} | {:.2} | {:+.1} |\n",
            scans[i].key,
            note_name(scans[i].key),
            scans[i].unison,
            if scans[i].sampled { "yes" } else { "-" },
            f.metric,
            f.z,
            column[i],
            median(&neighbours),
            reference_z[i].get(f.metric),
        ));
    }
    if flags.is_empty() {
        s.push_str("| - | - | - | - | - | - | - | - | - |\n");
        s.push_str("\n**No key of the compass is an outlier.**\n");
    }
    let keys_flagged = flags
        .iter()
        .map(|f| f.key_index)
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    s.push_str(&format!(
        "\n{} flags over {keys_flagged} of {} keys.\n\n",
        flags.len(),
        scans.len()
    ));

    // ---- the stereo line ----
    s.push_str("## The stereo image, key by key  *(STEREO, not a mono sum)*\n\n");
    s.push_str(
        "Every one of the seven metrics above is measured on the mono sum, and this section is \
the one that is not. `DECISIONS.md` 314 measured the interchannel correlation of the recording \
against the engine's and found them **inverted**: the recording's two channels are +0.945 \
correlated below 125 Hz, falling to about zero through the mid and treble, which is a spaced \
pair of microphones over the strings; the engine read negative in the bass, where the \
soundboard FDN's two opposite-sign output taps decorrelated it, and near +1 in the treble, \
where `soundboard::pan_for_key` scaled one mono voice into two channels — 46 of 88 keys on the \
opposite side of zero from the recording, median |Δ| 0.784. `PHYSICS.md` §8 is now built \
(`voicing.mics`, `DECISIONS.md` 351-358): two virtual capsules over the string band, a \
per-source delay and gain from where along the bass-treble axis that source sits, and a \
frequency-dependent coherence on the board's diffuse field — with the pair's own geometry \
since inverted out of the interchannel delays the recording carries (`DECISIONS.md` 359-367), \
and with the board's **mode-controlled band** under it (`[voicing.mics.modal]`, `DECISIONS.md` \
368-377), which is the part of the image no spacing can produce: where the plate's own modes \
put a nodal line between the two capsules, both hear the same field with opposite signs and \
the pair sees more difference than sum — built as an anti-phase copy of the pair's own sum \
since `DECISIONS.md` 379, and read here through a window that opens at the strike rather than \
96 samples after it (`DECISIONS.md` 378). \
The line below is what all of that is worth key by key. Read the two summary numbers together: the \
count of keys on the wrong side of zero counts keys sitting *near* zero, and the median |Δ| is \
how far the image is off. It is **not a room** — §9 is refused by measurement in item 315.\n\n",
    );
    s.push_str(&format!(
        "`r@0` is the normalised interchannel correlation at lag zero, **signed**, read in the \
band this key's fundamental falls in (clamped at both ends: A0-B1 are under the lowest band and \
read there, which is where their second partial is). `Δ` is engine minus recording, and its \
sign is the whole finding. `peak |r| @ lag` is the largest |r| over ±{:.0} ms and where it sits, \
positive meaning the *right* channel leads. `M/S` is 10·log10 of mid over side energy in the \
band. A row is `-` where the band holds less than {:.0} dB under the whole note's energy on \
either side, which is where a correlation is a ratio of two noise floors.\n\n",
        realism::STEREO_MAX_LAG_S * 1000.0,
        realism::STEREO_BAND_FLOOR_DB
    ));
    s.push_str(
        "This column is **not** scored against the neighbours the seven above are. Where the \
microphones were is a fact that changes smoothly down the compass, not something one key can be \
an outlier of; the statistic that matters here is the whole line, and `renders/realism/REALISM.md`'s \
Columns S and `tuner/tests/stereo.rs` are where it carries a bar.\n\n",
    );
    s.push_str(
        "| key | note | f0 | band | engine r@0 | recording r@0 | Δ | engine peak \\|r\\| @ lag | recording peak \\|r\\| @ lag | engine M/S | recording M/S |\n",
    );
    s.push_str("|---|---|---|---|---|---|---|---|---|---|---|\n");
    for sc in scans.iter() {
        let readable = sc.stereo_engine.readable() && sc.stereo_reference.readable();
        if !readable {
            s.push_str(&format!(
                "| {} | {} | {:.1} | `{}` | - | - | - | - | - | - | - |\n",
                sc.key,
                note_name(sc.key),
                sc.f0,
                STEREO_BANDS[sc.stereo_band].0
            ));
            continue;
        }
        s.push_str(&format!(
            "| {} | {} | {:.1} | `{}` | {:+.3} | {:+.3} | **{:+.3}** | {:.3} @ {:+.2} ms | {:.3} @ {:+.2} ms | {:+.1} dB | {:+.1} dB |\n",
            sc.key,
            note_name(sc.key),
            sc.f0,
            STEREO_BANDS[sc.stereo_band].0,
            sc.stereo_engine.r0,
            sc.stereo_reference.r0,
            sc.stereo_engine.r0 - sc.stereo_reference.r0,
            sc.stereo_engine.peak_r.abs(),
            sc.stereo_engine.lag_ms,
            sc.stereo_reference.peak_r.abs(),
            sc.stereo_reference.lag_ms,
            sc.stereo_engine.mid_side_db,
            sc.stereo_reference.mid_side_db,
        ));
    }
    let readable: Vec<&Scan> = scans
        .iter()
        .filter(|sc| sc.stereo_engine.readable() && sc.stereo_reference.readable())
        .collect();
    let inverted = readable
        .iter()
        .filter(|sc| sc.stereo_engine.r0 * sc.stereo_reference.r0 < 0.0)
        .count();
    s.push_str(&format!(
        "\n**{} of {} readable keys have the engine on the opposite side of zero from the \
recording.** Median r@0 over them: engine {:+.3}, recording {:+.3}; median |Δ| {:.3}.\n\n",
        inverted,
        readable.len(),
        median(
            &readable
                .iter()
                .map(|sc| sc.stereo_engine.r0)
                .collect::<Vec<_>>()
        ),
        median(
            &readable
                .iter()
                .map(|sc| sc.stereo_reference.r0)
                .collect::<Vec<_>>()
        ),
        median(
            &readable
                .iter()
                .map(|sc| (sc.stereo_engine.r0 - sc.stereo_reference.r0).abs())
                .collect::<Vec<_>>()
        ),
    ));

    // ---- the full table ----
    s.push_str("## Every key\n\n");
    s.push_str(
        "`e` is the engine, `r` the recording. The `z` column is the worst of the six. \
`source` is where the reference at that key comes from: `recorded` when the library has a take \
of that note, `transposed` when it does not and the player is resampling a neighbour's take. \
A transposed key's `match` cell reads **`transposed — unscored`**: the note is still rendered \
and it is still what a listener hears, but there is no recording of *that* key to be a \
distance from.\n\n",
    );
    s.push_str(
        "| key | note | f0 | N | source | level e/r | centroid e/r | irregular e/r | match | beat e/r | jitter e/r | decay e/r | worst z |\n",
    );
    s.push_str("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");
    for (i, sc) in scans.iter().enumerate() {
        let (name, z) = engine_z[i].worst();
        let pair = |a: f64, b: f64, places: usize| -> String {
            let one = |v: f64| {
                if v.is_finite() {
                    format!("{v:.places$}")
                } else {
                    "-".to_string()
                }
            };
            format!("{}/{}", one(a), one(b))
        };
        s.push_str(&format!(
            "| {} | {} | {:.1} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} {:.1} |\n",
            sc.key,
            note_name(sc.key),
            sc.f0,
            sc.unison,
            if sc.sampled { "recorded" } else { "transposed" },
            pair(sc.engine.level_db, sc.reference.level_db, 1),
            pair(sc.engine.centroid_st, sc.reference.centroid_st, 1),
            pair(sc.engine.irregular_db, sc.reference.irregular_db, 1),
            match (sc.sampled, sc.engine.match_db.is_finite()) {
                (_, true) => format!("{:.1}", sc.engine.match_db),
                // A recorded key whose two spectra share no partial the series
                // could read: the recording exists, the distance does not.
                (true, false) => "-".to_string(),
                (false, false) => "*transposed — unscored*".to_string(),
            },
            pair(sc.engine.beat_db, sc.reference.beat_db, 1),
            pair(sc.engine.jitter_cents, sc.reference.jitter_cents, 2),
            pair(sc.engine.decay_db_s, sc.reference.decay_db_s, 1),
            name,
            z,
        ));
    }
    s.push_str(&format!(
        "\n```sh\ncargo run --release -p piano-tuner -- compass \\\n    data/salamander renders/compass {}\n```\n",
        preset.display()
    ));
    s
}

fn metric_index(name: &str) -> usize {
    METRIC_NAMES.iter().position(|&n| n == name).unwrap_or(0)
}

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

/// Unused import guard: `audio` is re-exported for callers that want to load a
/// render back off disk while iterating on this tool.
#[allow(dead_code)]
fn _keep_audio_in_scope(p: &Path) -> Option<Audio> {
    audio::load(p).ok()
}
