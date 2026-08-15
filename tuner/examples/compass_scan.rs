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
//! comb, the microphone — with exactly one exception, the fitted
//! `notes.partial_gains` row, which is a free number per partial. So a jagged
//! harmonic series is a *fitted* defect by construction, and the metric points
//! straight at the table that carries it.
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
//! cargo run --release -p piano-tuner --example compass_scan \
//!     -- data/salamander renders/compass presets/salamander-c5.toml
//! ```

use std::f64::consts::TAU;
use std::path::{Path, PathBuf};

use piano_emulator::preset::Preset;
use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_emulator::types::Event;
use piano_tuner::motion::Motion;
use piano_tuner::{audio, realism, Audio, Sampler, SampleLibrary, SamplerEvent, TimedEvent, SAMPLE_RATE};

/// Velocity every key is struck at: the middle layer, the one the fits and the
/// motion columns both use.
const VELOCITY: u8 = 90;

/// Seconds of note. The partial motion window ends at 3.0 s.
const RENDER_S: f64 = 3.6;

/// Seconds of silence before the strike, so an onset is never at sample zero.
const PREROLL_S: f64 = 0.05;

/// Partials the spectrum metrics are taken over. Twelve reaches 785 Hz at A0
/// and 12.5 kHz at C8; above that a bass note's partials are closer together
/// than the band-pass is wide.
const PARTIALS: usize = 12;

/// Window the level and the spectrum are measured over, seconds from the
/// strike: past the hammer's noise and inside the part every key still sounds
/// in.
const SPECTRUM_WINDOW: (f64, f64) = (0.10, 1.10);

/// How far a partial must stand over the signal *between* partials before it is
/// counted as present, in dB.
///
/// Without this the spectrum metrics measure the analysis floor at the top of
/// the compass, where a key's twelfth partial is past the rendered band on one
/// side and past Nyquist on the other: two floor readings differ by whatever
/// the floor happens to be doing, and the treble reads a 25 dB `irregular`
/// that has nothing to do with the note. The test is a *local* one — the level
/// midway to each neighbouring partial — so a partial that is genuinely 26 dB
/// under its neighbours still counts, which is the whole point.
const PRESENCE_DB: f64 = 8.0;

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let data = PathBuf::from(args.next().unwrap_or_else(|| "data/salamander".into()));
    let out = PathBuf::from(args.next().unwrap_or_else(|| "renders/compass".into()));
    let preset_path = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "presets/salamander-c5.toml".into()),
    );
    // Remaining arguments are keys to print a per-partial table for.
    let detail: Vec<u8> = args.filter_map(|a| a.parse::<u8>().ok()).collect();

    let sfz = data.join("SalamanderGrandPiano-V3+20200602.sfz");
    if !sfz.exists() {
        eprintln!(
            "the reference piano is not here: {}\nrun data/fetch_salamander.sh first (707 MiB).",
            sfz.display()
        );
        std::process::exit(2);
    }
    std::fs::create_dir_all(&out)?;

    let preset = Preset::load(&preset_path)?;
    let mut sampler = Sampler::new(&sfz)?;
    let library = SampleLibrary::from_sfz(&sfz)?;
    let sampled: Vec<u8> = library.keys().collect();

    println!(
        "compass scan: vel {VELOCITY}, {} keys, engine on {}",
        LAST_KEY - FIRST_KEY + 1,
        preset_path.display()
    );

    let mut scans: Vec<Scan> = Vec::new();
    for key in FIRST_KEY..=LAST_KEY {
        let params = preset.string_params(key);
        let partial_hz: Vec<f64> = (1..=PARTIALS)
            .map(|k| f64::from(params.partial_freq(k)))
            .collect();
        let engine_audio = render_engine(&preset, key);
        let reference_audio = render_reference(&mut sampler, key)?;
        if key % 12 == 0 {
            sampler.clear_cache();
        }
        let (reference, reference_spectrum) =
            Metrics::measure(&reference_audio.mono(), &partial_hz, None);
        let (engine, _) = Metrics::measure(
            &engine_audio.mono(),
            &partial_hz,
            Some(&reference_spectrum),
        );
        if detail.contains(&key) {
            print_detail(key, &partial_hz, &engine_audio, &reference_audio, &out)?;
        }
        scans.push(Scan {
            key,
            f0: partial_hz[0],
            unison: usize::from(preset.notes.unison[usize::from(key - FIRST_KEY)]),
            sampled: sampled.contains(&key),
            engine,
            reference,
        });
        print!("\r  key {key}   ");
        use std::io::Write;
        std::io::stdout().flush().ok();
    }
    println!("\r  {} keys rendered      ", scans.len());

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
            vel: VELOCITY,
        },
    )];
    let (left, right) = render_to_buffer(preset, &events, (PREROLL_S + RENDER_S) as f32);
    let skip = (PREROLL_S * f64::from(SAMPLE_RATE)) as usize;
    Audio::new(
        SAMPLE_RATE,
        vec![left[skip..].to_vec(), right[skip..].to_vec()],
    )
    .expect("the engine renders stereo")
}

/// The recording of the same note, aligned on its own onset so that the two
/// windows measure the same part of the note. The sampler's own attack offset
/// is a property of where the microphone was, not of the piano.
fn render_reference(sampler: &mut Sampler, key: u8) -> Result<Audio, piano_tuner::Error> {
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
    fn measure(mono: &[f32], partial_hz: &[f64], against: Option<&Spectrum>) -> (Metrics, Spectrum) {
        let sr = f64::from(SAMPLE_RATE);
        let lo = (SPECTRUM_WINDOW.0 * sr) as usize;
        let hi = ((SPECTRUM_WINDOW.1 * sr) as usize).min(mono.len());
        let window = &mono[lo.min(hi)..hi];
        let level_db = amp_db(realism::rms(window));

        let spectrum = Spectrum::measure(window, partial_hz, sr);
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
                    &measured.iter().map(|m| m.floored_cents()).collect::<Vec<_>>(),
                ),
                decay_db_s: median(&measured.iter().map(|m| m.tail_db_s).collect::<Vec<_>>()),
            },
            spectrum,
        )
    }
}

/// One note's harmonic series as measured: a level per partial, and whether the
/// partial is really there.
///
/// The projection is Goertzel-style onto each partial's own frequency through a
/// Hann window rather than a DFT bin, because the window is a second long, its
/// bins are 1 Hz apart, and the two modes of a coupled unison are a tenth of
/// that: what is wanted is the partial's energy *however it is split inside its
/// own neighbourhood*, so a skirt of +-3 bins is summed in power.
struct Spectrum {
    /// Level of partial `k`, dB.
    levels_db: Vec<f64>,
    /// Whether partial `k` stands [`PRESENCE_DB`] over the signal between it
    /// and its neighbours.
    present: Vec<bool>,
}

impl Spectrum {
    fn measure(window: &[f32], partial_hz: &[f64], sr: f64) -> Spectrum {
        if window.is_empty() || partial_hz.is_empty() {
            return Spectrum {
                levels_db: vec![f64::NEG_INFINITY; partial_hz.len()],
                present: vec![false; partial_hz.len()],
            };
        }
        let n = window.len();
        // Hann, so a neighbouring partial 1/T away is 31 dB down.
        let taper: Vec<f64> = (0..n)
            .map(|i| 0.5 - 0.5 * (TAU * i as f64 / n as f64).cos())
            .collect();
        let bin = sr / n as f64;
        let at = |hz: f64| -> f64 {
            if hz <= 0.0 || hz >= 0.45 * sr {
                return f64::NEG_INFINITY;
            }
            let mut power = 0.0;
            for d in -3..=3i32 {
                let f = hz + f64::from(d) * bin;
                if f <= 0.0 {
                    continue;
                }
                let (mut re, mut im) = (0.0f64, 0.0f64);
                let w = TAU * f / sr;
                for (i, (&s, &t)) in window.iter().zip(&taper).enumerate() {
                    let phase = w * i as f64;
                    let v = f64::from(s) * t;
                    re += v * phase.cos();
                    im -= v * phase.sin();
                }
                power += re * re + im * im;
            }
            amp_db(2.0 * power.sqrt() / n as f64)
        };
        let levels_db: Vec<f64> = partial_hz.iter().map(|&hz| at(hz)).collect();
        // The floor is measured where the note is not: 45 % of the way to each
        // neighbouring partial, whichever side reads lower.
        let spacing = partial_hz[0];
        let present: Vec<bool> = partial_hz
            .iter()
            .zip(&levels_db)
            .map(|(&hz, &db)| {
                let floor = at(hz - 0.45 * spacing).min(at(hz + 0.45 * spacing));
                db.is_finite() && (!floor.is_finite() || db - floor >= PRESENCE_DB)
            })
            .collect();
        Spectrum {
            levels_db,
            present,
        }
    }

    /// The partials that are really there, as `(k, level)`.
    fn sounding(&self) -> Vec<(usize, f64)> {
        self.levels_db
            .iter()
            .enumerate()
            .zip(&self.present)
            .filter(|((_, db), &p)| p && db.is_finite())
            .map(|((i, &db), _)| (i + 1, db))
            .collect()
    }

    /// Power-weighted mean partial index, expressed in semitones over `f0`.
    ///
    /// Register-free by construction: a note whose energy sits on its second
    /// partial reads 12 semitones whether it is A0 or C8, so the number can be
    /// compared across the compass without a trend to remove first.
    fn centroid_semitones(&self) -> f64 {
        let mut num = 0.0;
        let mut den = 0.0;
        for (k, db) in self.sounding() {
            let p = 10f64.powf(db / 10.0);
            num += p * k as f64;
            den += p;
        }
        if den <= 0.0 {
            return 0.0;
        }
        12.0 * (num / den).log2()
    }

    /// Mean absolute per-partial distance from another spectrum of the same
    /// note, dB, after the common offset between the two is removed.
    ///
    /// The offset is the **median** of the per-partial differences, so one
    /// railed partial cannot move it. Removing it is what makes this a measure
    /// of *colour* rather than of loudness: a note that is uniformly 6 dB quiet
    /// is a level error, which `level` already reports, and it is not what a
    /// listener calls a note that does not fit.
    ///
    /// Taken only where both spectra have the partial: a partial the recording
    /// does not have is not a distance, it is a missing measurement.
    fn distance_from(&self, other: &Spectrum) -> f64 {
        let pairs: Vec<(f64, f64)> = self
            .levels_db
            .iter()
            .zip(&self.present)
            .zip(other.levels_db.iter().zip(&other.present))
            .filter(|((a, &pa), (b, &pb))| pa && pb && a.is_finite() && b.is_finite())
            .map(|((&a, _), (&b, _))| (a, b))
            .collect();
        if pairs.len() < 3 {
            return f64::NAN;
        }
        let offset = median(&pairs.iter().map(|(a, b)| a - b).collect::<Vec<_>>());
        pairs.iter().map(|(a, b)| (a - offset - b).abs()).sum::<f64>() / pairs.len() as f64
    }

    /// Mean absolute step between the levels of adjacent sounding partials, dB.
    ///
    /// Every mechanism in the engine that shapes a spectrum is smooth in `ln k`
    /// — the hammer, the bridge admittance, the strike comb, the microphone —
    /// with exactly one exception, the fitted `notes.partial_gains` row, which
    /// is a free number per partial. So this metric cannot be made large by the
    /// physics, and a key that reads large has a table problem.
    ///
    /// Steps are taken only between partials that are adjacent or one apart: a
    /// step across a long silent stretch of the series is a step across
    /// whatever the note stopped doing, not a step in the note.
    fn irregularity(&self) -> f64 {
        let sounding = self.sounding();
        let steps: Vec<f64> = sounding
            .windows(2)
            .filter(|w| w[1].0 - w[0].0 <= 2)
            .map(|w| (w[1].1 - w[0].1).abs())
            .collect();
        if steps.is_empty() {
            return 0.0;
        }
        steps.iter().sum::<f64>() / steps.len() as f64
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

/// The indices of the [`NEIGHBOURS`] nearest keys strung the same way as `i`,
/// nearest first, excluding `i` itself.
///
/// This is the comparison the whole report rests on and it is a judgement about
/// what a piano is: a key is odd when it does not fit the keys *of its own
/// kind*. Every metric here steps at the 1→2 and 2→3 string boundaries because
/// the number of strings is what a unison beat is made of, and a scan that
/// ignored that would spend its flags convicting four keys of being a boundary.
fn same_kind_neighbours(scans: &[Scan], i: usize) -> Vec<usize> {
    let mut candidates: Vec<usize> = (0..scans.len())
        .filter(|&j| j != i && scans[j].unison == scans[i].unison)
        .collect();
    candidates.sort_by_key(|&j| (scans[j].key as i32 - scans[i].key as i32).abs());
    candidates.truncate(NEIGHBOURS);
    candidates
}

/// The residual of one metric at one key: its value less the median of its
/// same-kind neighbours. `None` where the metric was not measured, at the key
/// or at its neighbours.
fn residual_at(scans: &[Scan], column: &[f64], i: usize) -> Option<f64> {
    if !column[i].is_finite() {
        return None;
    }
    let neighbours: Vec<f64> = same_kind_neighbours(scans, i)
        .into_iter()
        .map(|j| column[j])
        .filter(|v| v.is_finite())
        .collect();
    (neighbours.len() >= NEIGHBOURS / 2).then(|| column[i] - median(&neighbours))
}

fn score(scans: &[Scan], pick: impl Fn(&Scan) -> Metrics) -> Vec<Zs> {
    let series: Vec<[f64; METRIC_NAMES.len()]> = scans.iter().map(|s| pick(s).values()).collect();
    let mut out = vec![Zs::default(); scans.len()];
    for m in 0..METRIC_NAMES.len() {
        let column: Vec<f64> = series.iter().map(|v| v[m]).collect();
        let residual: Vec<Option<f64>> = (0..column.len())
            .map(|i| residual_at(scans, &column, i))
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

fn print_detail(
    key: u8,
    partial_hz: &[f64],
    engine: &Audio,
    reference: &Audio,
    out: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let sr = f64::from(SAMPLE_RATE);
    let lo = (SPECTRUM_WINDOW.0 * sr) as usize;
    let hi = (SPECTRUM_WINDOW.1 * sr) as usize;
    let e_mono = engine.mono();
    let r_mono = reference.mono();
    let e_spec = Spectrum::measure(&e_mono[lo..hi.min(e_mono.len())], partial_hz, sr);
    let r_spec = Spectrum::measure(&r_mono[lo..hi.min(r_mono.len())], partial_hz, sr);
    let (e_lev, r_lev) = (&e_spec.levels_db, &r_spec.levels_db);
    let e_sig: Vec<f64> = e_mono.iter().map(|&v| f64::from(v)).collect();
    let r_sig: Vec<f64> = r_mono.iter().map(|&v| f64::from(v)).collect();
    let e_mot = realism::measure_partials(&e_sig, partial_hz);
    let r_mot = realism::measure_partials(&r_sig, partial_hz);
    println!("\n  key {key} ({}):", note_name(key));
    println!(
        "    {:>2} {:>8}  {:>7} {:>7} {:>7} {:>4} | {:>6} {:>6} | {:>6} {:>6} | {:>7} {:>7}",
        "k", "hz", "eng dB", "ref dB", "diff", "seen", "eBeat", "rBeat", "eRate", "rRate", "eTail",
        "rTail"
    );
    for k in 0..partial_hz.len() {
        let f = |m: &Option<Motion>, g: fn(&Motion) -> f64| {
            m.as_ref().map(g).map_or("  --  ".to_string(), |v| format!("{v:6.2}"))
        };
        println!(
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
    write_normalised(&dir.join(format!("key{key}_{name}_reference.wav")), reference)?;
    Ok(())
}

/// Writes a solo note at a level a listener can compare: peak to -3 dBFS.
fn write_normalised(path: &Path, audio: &Audio) -> Result<(), Box<dyn std::error::Error>> {
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
exactly one exception, the fitted `notes.partial_gains` row, which is a free number per partial. \
A jagged harmonic series is therefore a *fitted* defect by construction, and the metric points \
at the table that carries it.\n\n"
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
        "| key | note | N | fitted | metric | engine z | engine | neighbours | recording z |\n",
    );
    s.push_str("|---|---|---|---|---|---|---|---|---|\n");
    for f in &flags {
        let i = f.key_index;
        let column: Vec<f64> = scans
            .iter()
            .map(|sc| sc.engine.values()[metric_index(f.metric)])
            .collect();
        let neighbours: Vec<f64> = same_kind_neighbours(scans, i)
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

    // ---- the full table ----
    s.push_str("## Every key\n\n");
    s.push_str(
        "`e` is the engine, `r` the recording. The `z` column is the worst of the six.\n\n",
    );
    s.push_str(
        "| key | note | f0 | N | level e/r | centroid e/r | irregular e/r | match | beat e/r | jitter e/r | decay e/r | worst z |\n",
    );
    s.push_str("|---|---|---|---|---|---|---|---|---|---|---|---|\n");
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
            "| {} | {} | {:.1} | {} | {} | {} | {} | {:.1} | {} | {} | {} | {} {:.1} |\n",
            sc.key,
            note_name(sc.key),
            sc.f0,
            sc.unison,
            pair(sc.engine.level_db, sc.reference.level_db, 1),
            pair(sc.engine.centroid_st, sc.reference.centroid_st, 1),
            pair(sc.engine.irregular_db, sc.reference.irregular_db, 1),
            sc.engine.match_db,
            pair(sc.engine.beat_db, sc.reference.beat_db, 1),
            pair(sc.engine.jitter_cents, sc.reference.jitter_cents, 2),
            pair(sc.engine.decay_db_s, sc.reference.decay_db_s, 1),
            name,
            z,
        ));
    }
    s.push_str(&format!(
        "\n```sh\ncargo run --release -p piano-tuner --example compass_scan \\\n    -- data/salamander renders/compass {}\n```\n",
        preset.display()
    ));
    s
}

fn metric_index(name: &str) -> usize {
    METRIC_NAMES.iter().position(|&n| n == name).unwrap_or(0)
}

fn amp_db(amp: f64) -> f64 {
    if amp > 0.0 {
        20.0 * amp.log10()
    } else {
        f64::NEG_INFINITY
    }
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
