//! Stage 2, the per-partial half: the five fields of the excitation, the decay
//! and the attack, fitted against a real instrument.
//!
//! `survey` (stage 1) fits what one *smooth law per note* can carry. This is
//! what is left over when those laws are as right as they can be, and every one
//! of them is a per-partial or per-attack measurement the laws cannot hold:
//!
//! 1. **`notes.comb_floor`** and **`notes.partial_gains`** — the excitation's two
//!    faults of opposite sign, from the layer-median of every sampled note's own
//!    time-zero spectrum against the comb the engine will play
//!    (`estimate::shaping`; `TUNING_REPORT.md` §3, `ANALYSIS.md` §4a).
//! 2. **`notes.partial_sigma_scale`** — the per-partial correction to the fitted
//!    decay law, from the same layers' prompt rates, written only where the
//!    decay stage's own gates pass (`estimate::shaping`; `TUNING_REPORT.md` §2).
//! 3. **`[noise.strike]`** — the hammer's own noise, from the onset residual of
//!    the struck notes themselves (`estimate::attack`; `REALISM.md`'s +5.2 dB of
//!    attack tonality, `ANALYSIS.md` §8.3).
//! 4. **`notes.damper_sigma`** — the damper's grip, from the release recordings'
//!    20-dB tails inverted on a line measured on the engine itself
//!    (`estimate::damper`; `DECISIONS.md` 183).
//!
//! **This tool is not re-entrant, and `--preset` and `--out` may not be the
//! same file.** Every one of the four measurements above is taken *against a
//! render of the base preset* — the comb floor from `CombLine`, the gains from
//! the comb that line describes, the strike from the engine's own onset
//! residual, the damper from `DamperLine` — so a base that already carries the
//! answers measures an instrument that has already been corrected. Run over the
//! shipped preset rather than the survey base it was fitted from, `comb_line`
//! renders a comb whose nulls its own `partial_gains` have already filled: the
//! engine's line goes flat (at key 45, −31.0 / −11.5 / −11.5 / −11.9 dB across
//! the four probe floors, against a line that separates them by 19 dB from the
//! base) and `floor_for` saturates to 0.450 at every key, where the same run
//! from the base reproduces the shipped 0.200 and 0.0 exactly. The second pass
//! moves `comb_floor` at 41 keys. `DECISIONS.md` 214 and 218.
//!
//! So: fit from the **survey base**, write to a **scratch file**, and splice the
//! sections you meant to re-fit into the preset by hand — which is how
//! `DECISIONS.md` 210–213's `[noise.strike]` was taken.
//!
//! **`notes.partial_gains` has moved.** `tuner/examples/fit_motion.rs` fits it as
//! the full measured ratio of the recording's time-zero spectrum to the engine's
//! own render of the same note (`DECISIONS.md` 237, 243), against a probe whose
//! own row is cleared first — which is what makes that fit re-entrant where this
//! one is not. What is left here is the roughness half plus
//! [`envelope_tilt`](piano_tuner::estimate::shaping::envelope_tilt), kept because
//! `notes.comb_floor` is fitted against the gains this file writes and the two
//! were derived together; run `fit_motion` afterwards and its table replaces
//! this one outright.
//!
//! ```sh
//! cargo run --release -p piano-tuner --example fit_partials -- \
//!     data/salamander/SalamanderGrandPiano-V3+20200602.sfz \
//!     --preset presets/salamander-c5.toml --cache data/cache/salamander \
//!     --out /tmp/salamander-fit.toml
//! ```
//!
//! Without `--out` it measures and prints and writes nothing, which is how the
//! tables in `DECISIONS.md` were taken.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_emulator::types::Event;
use piano_tuner::estimate::attack::{
    density_bands, fit_strike, residual_metrics, AttackConfig, AttackResidual,
};
use piano_tuner::estimate::damper::{band_release, tail_decay_s, DamperConfig, DamperLine};
use piano_tuner::estimate::decay::DecayCurve;
use piano_tuner::estimate::noise::NoiseConfig;
use piano_tuner::estimate::decay::DecayReport;
use piano_tuner::estimate::shaping::{
    envelope_tilt, fit_note, measured_deepest, CombLine, DecaySplit, EngineComb, NoteShaping,
    ShapingConfig,
};
use piano_tuner::library::MechanismKind;
use piano_tuner::pipeline::{analyze_note, analyze_trajectories};
use piano_tuner::residual::onset_residual;
use piano_tuner::preset::{
    equal_temperament, key_index, NoteEstimate, Preset, PresetBuilder, MAX_PARTIAL_GAIN,
    MIN_PARTIAL_GAIN, NUM_KEYS,
};
use piano_tuner::survey::{load_signal, trajectories_for, SurveyConfig};
use piano_tuner::trajectory::InharmonicModel;
use piano_tuner::{audio, Error, Result, SampleLibrary, SAMPLE_RATE};

/// Velocity every engine reference gesture is played at — `TUNING_REPORT.md`
/// §5's own convention, and `[noise.strike]`'s nominal drive.
const REFERENCE_VELOCITY: u8 = 90;
/// The damper probe: hold, release, and listen. Long enough that the tail is a
/// tail and not the note.
const HOLD_S: f32 = 1.0;
const RELEASE_RENDER_S: f32 = 5.0;
/// The two `damper_sigma` multipliers the line is measured at. Wide enough that
/// the two rendered tails differ by far more than the measurement's own
/// resolution, close enough that neither is a preset nobody would ship.
const PROBE_SCALES: [f64; 2] = [1.0, 0.25];
/// The comb floors the engine is probed at, and how long each probe is
/// rendered for. Four points over the schema's whole range, which piecewise
/// linear interpolation in the amplitude domain reads to about 0.02; one
/// velocity layer, because a comb is a property of where the hammer lands and
/// not of how hard; and four seconds, because what is read off the probe is the
/// excitation at `t = 0`.
const COMB_PROBES: [f32; 4] = [0.0, 0.1, 0.25, 0.45];
const COMB_RENDER_S: f32 = 4.0;

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("fit_partials: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}

struct Options {
    sfz: PathBuf,
    preset: PathBuf,
    cache: Option<PathBuf>,
    out: Option<PathBuf>,
    keys: Option<Vec<u8>>,
}

fn parse() -> Result<Options> {
    let mut sfz = None;
    let mut preset = None;
    let mut options = Options {
        sfz: PathBuf::new(),
        preset: PathBuf::new(),
        cache: None,
        out: None,
        keys: None,
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let value = |i: &mut usize| -> Result<String> {
            *i += 1;
            args.get(*i)
                .cloned()
                .ok_or_else(|| Error::Config(format!("{} needs a value", args[*i - 1])))
        };
        match args[i].as_str() {
            "--preset" => preset = Some(PathBuf::from(value(&mut i)?)),
            "--cache" => options.cache = Some(PathBuf::from(value(&mut i)?)),
            "--out" => options.out = Some(PathBuf::from(value(&mut i)?)),
            "--keys" => {
                let list = value(&mut i)?;
                options.keys = Some(
                    list.split(',')
                        .map(|k| {
                            k.trim()
                                .parse()
                                .map_err(|_| Error::Config(format!("bad key {k:?}")))
                        })
                        .collect::<Result<Vec<u8>>>()?,
                );
            }
            other if other.starts_with('-') => {
                return Err(Error::Config(format!("unknown option {other:?}")))
            }
            other => sfz = Some(PathBuf::from(other)),
        }
        i += 1;
    }
    options.sfz = sfz.ok_or_else(|| Error::Config("no instrument file".into()))?;
    options.preset = preset.ok_or_else(|| Error::Config("needs --preset".into()))?;
    Ok(options)
}

/// Refuses the one invocation that destroys its own input, and says so about
/// the one that merely mismeasures.
///
/// The header explains why both are wrong; this is here because a header is not
/// a guard and the destructive form is a single flag away from the useful one.
/// The refusal is on `--out` naming the file `--preset` was read from — there
/// is no run for which that is the right thing to do, because the *next* run
/// would then have no base to measure against. The warning is on a base that
/// already carries any of this tool's own answers, which is the same mistake
/// reached by copying the shipped preset to a new name first.
fn refuse_self_overwrite(options: &Options) -> Result<()> {
    let Some(out) = &options.out else {
        return Ok(());
    };
    // A path that does not exist yet cannot be the file the preset was read
    // from, and `canonicalize` is what resolves the two ways of spelling one
    // that does.
    let (Ok(from), Ok(to)) = (
        std::fs::canonicalize(&options.preset),
        std::fs::canonicalize(out),
    ) else {
        return Ok(());
    };
    if from != to {
        return Ok(());
    }
    Err(Error::Config(format!(
        "--out {} is the file --preset was read from. Every measurement here is \
         taken against a render of the base preset, so fitting from an already \
         fitted file measures an instrument that has already been corrected — \
         the engine's comb line goes flat and every key's `comb_floor` \
         saturates. Write to a scratch file and splice the sections you meant \
         to re-fit (`DECISIONS.md` 214, 218).",
        out.display()
    )))
}

/// Warns when the base preset already carries what this tool writes.
fn warn_if_already_fitted(base: &Preset) {
    let mut carried: Vec<String> = Vec::new();
    let rows = |table: &[Vec<f32>]| table.iter().filter(|row| !row.is_empty()).count();
    for (name, count) in [
        ("notes.partial_gains", rows(&base.notes.partial_gains)),
        (
            "notes.partial_sigma_scale",
            rows(&base.notes.partial_sigma_scale),
        ),
        (
            "notes.comb_floor",
            base.notes.comb_floor.iter().filter(|&&f| f > 0.0).count(),
        ),
        (
            "[noise.strike]",
            base.noise
                .strike
                .level_db
                .iter()
                .filter(|a| a.db > -199.0)
                .count(),
        ),
    ] {
        if count > 0 {
            carried.push(format!("{name} ({count})"));
        }
    }
    if carried.is_empty() {
        return;
    }
    eprintln!(
        "fit_partials: warning — the base preset already carries {}. This tool is \
         not re-entrant: it measures against a render of the base, so a base that \
         already holds these answers reads an instrument that has already been \
         corrected and the numbers below are not the ones a fit from the survey \
         base would give (`DECISIONS.md` 214, 218).",
        carried.join(", ")
    );
}

fn run() -> Result<()> {
    let options = parse()?;
    refuse_self_overwrite(&options)?;
    let mut library = SampleLibrary::from_sfz(&options.sfz)?;
    if let Some(keys) = &options.keys {
        library = library.restricted_to(keys);
    }
    let base = Preset::load(&options.preset)?;
    warn_if_already_fitted(&base);
    let survey = SurveyConfig {
        cache_dir: options.cache.clone(),
        ..SurveyConfig::default()
    };
    println!(
        "{}: {} keys, {} recordings, over {}",
        options.sfz.display(),
        library.key_count(),
        library.sample_count(),
        options.preset.display()
    );

    let band = strike_band(&library, &base, &survey);
    let attack = band.config;
    let mut notes = analyse(&library, &base, &survey, &attack)?;
    fit_comb_floors(&mut notes, &base, &survey);
    refit_with_floors(&mut notes, &base);
    fit_spectral_envelope(&mut notes, &base, &survey);
    report_shaping(&notes);
    let mut strike = report_strike(&notes, &base, &attack);
    strike_offset(&base, &mut strike.strike, &band);
    let damper = report_damper(&library, &base)?;

    let Some(out) = &options.out else {
        println!("\nnothing written (no --out)");
        return Ok(());
    };
    // One estimate per key, so that a key the excitation fit and the damper fit
    // both had something to say about carries both — `PresetBuilder::note`
    // replaces by key, and two calls would keep only the second.
    let mut estimates: BTreeMap<u8, NoteEstimate> = BTreeMap::new();
    for note in &notes {
        estimates
            .entry(note.key)
            .or_insert_with(|| NoteEstimate::new(note.key))
            .comb_floor = note.comb_floor;
    }
    if let Some((key, sigma)) = damper {
        estimates
            .entry(key)
            .or_insert_with(|| NoteEstimate::new(key))
            .damper_sigma = Some(sigma);
    }
    let mut builder = PresetBuilder::new(base.clone());
    for estimate in estimates.into_values() {
        builder = builder.note(estimate);
    }
    builder = builder
        .partial_gains(rows(&notes, |n| n.shaping.gains.clone()))
        .partial_sigma_scale(rows(&notes, |n| n.shaping.sigma_scale.clone()));
    let mut noise = base.noise.clone();
    noise.strike = strike.strike.clone();
    builder = builder.noise(noise);
    let preset = builder.build()?;

    // The attribution comment at the head of the file is not part of the schema
    // and would be lost by a round trip; it is carried over verbatim.
    let previous = std::fs::read_to_string(&options.preset)?;
    let mut head = String::new();
    for line in previous.lines().take_while(|line| line.starts_with('#')) {
        head.push_str(line);
        head.push('\n');
    }
    let mut text = head;
    if !text.is_empty() {
        text.push('\n');
    }
    text.push_str(&preset.to_toml());
    std::fs::write(out, text)?;
    println!("\nwrote {}", out.display());
    Ok(())
}

/// One key's ragged row, or an empty one for a key nobody measured.
fn rows(notes: &[Note], get: impl Fn(&Note) -> Vec<f32>) -> Vec<Vec<f32>> {
    let mut table = vec![Vec::new(); NUM_KEYS];
    for note in notes {
        if let Some(index) = key_index(note.key) {
            table[index] = get(note);
        }
    }
    table
}

// ----------------------------------------------------------------- analysis

struct Note {
    key: u8,
    shaping: NoteShaping,
    /// The time-zero spectra of every layer, and their envelope fits: kept so
    /// that the gains can be re-measured once the comb floor is known without
    /// tracking the library a second time.
    spectra: Vec<Vec<(u32, f64)>>,
    reports: Vec<DecayReport>,
    /// The floor the engine's own line asks for, and the depth it was inverted
    /// at.
    comb_floor: Option<f64>,
    residuals: Vec<AttackResidual>,
    /// Third-octave density of the onset residual at the nominal velocity, in dB
    /// under its own loudest band: the spectrum the four numbers of
    /// `[noise.strike]` are a summary of.
    spectrum: Vec<(f64, f64)>,
    /// Why the residual was not measured, where it was not.
    refused: Option<String>,
    /// The time-zero spectrum of the layer a velocity-90 blow triggers, kept so
    /// that the engine's own render can be measured against the same one.
    reference_spectrum: Vec<(u32, f64)>,
}

/// Every sampled key: the per-partial fits from the cached trajectories, and the
/// onset residuals from the recordings themselves.
/// The keys the strike's band is measured at: the five the compass anchors
/// itself on, which are the five inside `estimate::attack`'s own gates.
const STRIKE_BAND_KEYS: [u8; 5] = [45, 57, 69, 81, 93];
/// How long an engine probe is rendered for when its own onset residual is
/// measured. The residual is 150 ms and the tracker needs a window either side
/// of it.
const ATTACK_RENDER_S: f32 = 2.0;

/// The band `[noise.strike]` is measured in, from what the engine's own attack
/// is missing.
///
/// One render per probe key with the strike silenced, tracked and subtracted by
/// exactly the code the recording goes through, gives the engine's own onset
/// residual; `estimate::attack::deficit_band` compares the two third-octave
/// densities and returns the run of bands the engine is short in. The median of
/// the probes' edges is the band, because the schema holds one centroid and one
/// limit for the whole compass.
///
/// Falls back to the schema's whole range when nothing can be measured, which is
/// the band the first fit used.
/// What [`strike_band`] measured: the band, and the recordings' own densities
/// inside it, kept so that [`strike_offset`] can compare the engine with the
/// very spectra the band was chosen from rather than re-track them.
struct StrikeBand {
    config: AttackConfig,
    /// `(key, third-octave density in dB against that residual's own peak)`.
    recorded: Vec<(u8, Vec<(f64, f64)>)>,
}

fn strike_band(library: &SampleLibrary, base: &Preset, survey: &SurveyConfig) -> StrikeBand {
    let default = AttackConfig::default();
    let mut probe = base.clone();
    probe.noise.strike = piano_tuner::preset::StrikeNoise::default();
    let Some(engine) = engine_preset(&probe) else {
        return StrikeBand { config: default, recorded: Vec::new() };
    };
    let mut edges: Vec<(f64, f64)> = Vec::new();
    let mut kept: Vec<(u8, Vec<(f64, f64)>)> = Vec::new();
    for key in STRIKE_BAND_KEYS {
        let Some(index) = key_index(key) else { continue };
        let Ok(note_config) = survey.note_config(equal_temperament(key)) else {
            continue;
        };
        // The recording, at the level the instrument plays it.
        let Some(sample) = library
            .layers(key)
            .iter()
            .find(|s| (s.lovel..=s.hivel).contains(&REFERENCE_VELOCITY))
        else {
            continue;
        };
        let Ok(trajectories) = trajectories_for(sample, &note_config, survey) else {
            continue;
        };
        let onset_s = trajectories.onset_s;
        let Ok(analysis) = analyze_trajectories(trajectories, &note_config) else {
            continue;
        };
        let gain = 10f64.powf(sample.volume_db / 20.0) as f32;
        let Ok(raw) = load_signal(&sample.path, survey) else {
            continue;
        };
        let signal: Vec<f32> = raw.iter().map(|&x| x * gain).collect();
        let partial_hz: Vec<f64> = analysis
            .decays
            .partials
            .iter()
            .map(|fit| fit.frequency_hz)
            .filter(|f| f.is_finite() && *f > 0.0)
            .collect();
        let Some(recorded) = onset_residual(
            &signal,
            f64::from(SAMPLE_RATE),
            &partial_hz,
            onset_s,
            default.residual_s,
        )
        .and_then(|r| density_bands(&r, f64::from(SAMPLE_RATE), &default))
        else {
            continue;
        };

        // The same measurement on the engine's own attack.
        let events = [RenderEvent::new(
            0.05,
            Event::NoteOn {
                key,
                vel: REFERENCE_VELOCITY,
            },
        )];
        let (left, right) = render_to_buffer(&engine, &events, ATTACK_RENDER_S);
        let mono: Vec<f32> = left.iter().zip(&right).map(|(&l, &r)| 0.5 * (l + r)).collect();
        let seed = InharmonicModel::harmonic(f64::from(base.notes.f0_hz[index]));
        let Ok(rendered) = analyze_note(&mono, f64::from(SAMPLE_RATE), seed, &note_config) else {
            continue;
        };
        let engine_hz: Vec<f64> = rendered
            .decays
            .partials
            .iter()
            .map(|fit| fit.frequency_hz)
            .filter(|f| f.is_finite() && *f > 0.0)
            .collect();
        let Some(made) = onset_residual(
            &mono,
            f64::from(SAMPLE_RATE),
            &engine_hz,
            rendered.trajectories.onset_s,
            default.residual_s,
        )
        .and_then(|r| density_bands(&r, f64::from(SAMPLE_RATE), &default))
        else {
            continue;
        };

        let recorded = piano_tuner::estimate::attack::density_db(&recorded);
        let made = piano_tuner::estimate::attack::density_db(&made);
        kept.push((key, recorded.clone()));
        if let Some(band) = piano_tuner::estimate::attack::deficit_band(&recorded, &made, &default) {
            println!(
                "  key {key:>3}: the engine's own attack is short from {:.0} Hz to {:.0} Hz",
                band.0, band.1
            );
            edges.push(band);
        } else {
            println!("  key {key:>3}: nothing missing by {:.0} dB", default.deficit_db);
        }
    }
    println!("\nthe band `[noise.strike]` is fitted in, from what the engine's attack is missing:");
    let Some(band) = median_band(&edges) else {
        println!("  nothing measured; the schema's whole range is used");
        return StrikeBand { config: default, recorded: kept };
    };
    let fitted = default.in_band(band);
    println!(
        "  {:.0} Hz .. {:.0} Hz, the median of {} probe keys",
        fitted.band_hz.0,
        fitted.band_hz.1,
        edges.len()
    );
    StrikeBand { config: fitted, recorded: kept }
}

/// The levels the strike is probed at when its own line is measured on the
/// engine, in dB against the fitted anchors. Far enough apart that the render
/// separates the burst from what the engine already has in the band.
const STRIKE_PROBES: [f64; 3] = [0.0, -8.0, -16.0];

/// Corrects `[noise.strike]`'s anchors by what the engine's own render says,
/// and returns the correction in dB.
///
/// The same argument as `CombLine` (`DECISIONS.md` 199) and the damper line
/// (203): what a residual *measures* is not what the engine has to *play*, and
/// the two are separated by things that are real and not small — the tracker's
/// own subtraction error inside an attack, the string partials above its reach
/// that count as residual, and a burst normalised on its peak rather than on its
/// density. All three ride along identically in a render, so the level is
/// inverted on the engine instead of being trusted from the recording: the
/// fitted strike is rendered at [`STRIKE_PROBES`], the engine's own onset
/// residual density is measured inside the fitted band by exactly the code the
/// recording went through, and the offset that puts the two densities on top of
/// each other is read off the line. The median over the probe keys moves the
/// whole table, which is `PresetBuilder`'s single-measurement rule.
fn strike_offset(base: &Preset, strike: &mut piano_tuner::preset::StrikeNoise, band: &StrikeBand) {
    if band.recorded.is_empty() {
        return;
    }
    let (lo, hi) = band.config.band_hz;
    let mut offsets: Vec<f64> = Vec::new();
    println!("\nthe strike's level, inverted on the engine's own render in {lo:.0}–{hi:.0} Hz:");
    for (key, recorded) in &band.recorded {
        let target = mean_in_band(recorded, lo, hi);
        let mut line: Vec<(f64, f64)> = Vec::new();
        for probe in STRIKE_PROBES {
            let mut candidate = base.clone();
            candidate.noise.strike = strike.clone();
            for anchor in candidate.noise.strike.level_db.iter_mut() {
                anchor.db += probe as f32;
            }
            let Some(made) = rendered_residual_density(&candidate, *key, &band.config) else {
                continue;
            };
            line.push((probe, mean_in_band(&made, lo, hi)));
        }
        let Some(offset) = crossing(&line, target) else {
            continue;
        };
        println!(
            "  key {key:>3}: recording {target:+6.1} dB, engine {:+6.1} dB as fitted — {offset:+6.1} dB",
            line.first().map_or(f64::NAN, |&(_, db)| db)
        );
        offsets.push(offset);
    }
    offsets.sort_by(f64::total_cmp);
    let Some(&offset) = offsets.get(offsets.len() / 2) else {
        return;
    };
    println!("  the median of {} probe keys: {offset:+.1} dB", offsets.len());
    for anchor in strike.level_db.iter_mut() {
        anchor.db += offset as f32;
    }
}

/// Mean third-octave density inside a band, in dB.
fn mean_in_band(density: &[(f64, f64)], lo: f64, hi: f64) -> f64 {
    let inside: Vec<f64> = density
        .iter()
        .filter(|&&(hz, _)| hz >= lo && hz <= hi)
        .map(|&(_, db)| db)
        .collect();
    if inside.is_empty() {
        return f64::NAN;
    }
    inside.iter().sum::<f64>() / inside.len() as f64
}

/// The engine's own onset residual density for one key, against its own peak.
fn rendered_residual_density(
    preset: &Preset,
    key: u8,
    config: &AttackConfig,
) -> Option<Vec<(f64, f64)>> {
    let index = key_index(key)?;
    let engine = engine_preset(preset)?;
    let note_config = SurveyConfig::default()
        .note_config(equal_temperament(key))
        .ok()?;
    let events = [RenderEvent::new(
        0.05,
        Event::NoteOn {
            key,
            vel: REFERENCE_VELOCITY,
        },
    )];
    let (left, right) = render_to_buffer(&engine, &events, ATTACK_RENDER_S);
    let mono: Vec<f32> = left.iter().zip(&right).map(|(&l, &r)| 0.5 * (l + r)).collect();
    let seed = InharmonicModel::harmonic(f64::from(preset.notes.f0_hz[index]));
    let analysis = analyze_note(&mono, f64::from(SAMPLE_RATE), seed, &note_config).ok()?;
    let partial_hz: Vec<f64> = analysis
        .decays
        .partials
        .iter()
        .map(|fit| fit.frequency_hz)
        .filter(|f| f.is_finite() && *f > 0.0)
        .collect();
    let bands = onset_residual(
        &mono,
        f64::from(SAMPLE_RATE),
        &partial_hz,
        analysis.trajectories.onset_s,
        config.residual_s,
    )
    .and_then(|r| density_bands(&r, f64::from(SAMPLE_RATE), &AttackConfig::default()))?;
    Some(piano_tuner::estimate::attack::density_db(&bands))
}

/// Where a monotone-enough line of `(offset dB, measured dB)` crosses `target`,
/// piecewise linearly, held at its ends.
fn crossing(line: &[(f64, f64)], target: f64) -> Option<f64> {
    if line.len() < 2 || !target.is_finite() {
        return None;
    }
    let mut points: Vec<(f64, f64)> = line.iter().copied().filter(|&(_, y)| y.is_finite()).collect();
    points.sort_by(|a, b| a.1.total_cmp(&b.1));
    if points.len() < 2 {
        return None;
    }
    if target <= points[0].1 {
        return Some(points[0].0);
    }
    for pair in points.windows(2) {
        let ((x0, y0), (x1, y1)) = (pair[0], pair[1]);
        if target <= y1 {
            if (y1 - y0).abs() < 1e-9 {
                return Some(x0);
            }
            return Some(x0 + (x1 - x0) * (target - y0) / (y1 - y0));
        }
    }
    points.last().map(|&(x, _)| x)
}

/// The median of the low edges and the median of the high edges.
fn median_band(edges: &[(f64, f64)]) -> Option<(f64, f64)> {
    if edges.is_empty() {
        return None;
    }
    let mut lo: Vec<f64> = edges.iter().map(|&(l, _)| l).collect();
    let mut hi: Vec<f64> = edges.iter().map(|&(_, h)| h).collect();
    lo.sort_by(f64::total_cmp);
    hi.sort_by(f64::total_cmp);
    Some((lo[lo.len() / 2], hi[hi.len() / 2]))
}

fn analyse(
    library: &SampleLibrary,
    base: &Preset,
    survey: &SurveyConfig,
    attack: &AttackConfig,
) -> Result<Vec<Note>> {
    let keys: Vec<u8> = library.keys().collect();
    let next = AtomicUsize::new(0);
    let mut done: Vec<Option<Note>> = keys.iter().map(|_| None).collect();
    let total = keys.len();
    let workers = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(total.max(1));
    std::thread::scope(|scope| {
        let (tx, rx) = std::sync::mpsc::channel();
        for _ in 0..workers {
            let (next, keys, tx) = (&next, &keys, tx.clone());
            scope.spawn(move || loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                let Some(&key) = keys.get(index) else {
                    return;
                };
                let note = analyse_key(library, base, survey, attack, key);
                if tx.send((index, note)).is_err() {
                    return;
                }
            });
        }
        drop(tx);
        let mut count = 0usize;
        for (index, note) in rx {
            count += 1;
            eprint!("\r  {count}/{total} keys   ");
            done[index] = note;
        }
    });
    eprintln!();
    Ok(done.into_iter().flatten().collect())
}

type LayerSignal = (u8, u8, u8, f64, Option<Vec<f32>>);

fn analyse_key(
    library: &SampleLibrary,
    base: &Preset,
    survey: &SurveyConfig,
    attack: &AttackConfig,
    key: u8,
) -> Option<Note> {
    let index = key_index(key)?;
    let note_config = survey.note_config(equal_temperament(key)).ok()?;

    let mut analyses = Vec::new();
    let mut signals: Vec<LayerSignal> = Vec::new();
    for sample in library.layers(key) {
        let Ok(trajectories) = trajectories_for(sample, &note_config, survey) else {
            continue;
        };
        let onset_s = trajectories.onset_s;
        let Ok(analysis) = analyze_trajectories(trajectories, &note_config) else {
            continue;
        };
        // The recording at the level the instrument plays it, which is what a
        // level relative to a strike of the same key has to be measured on.
        let gain = 10f64.powf(sample.volume_db / 20.0) as f32;
        let signal: Option<Vec<f32>> = load_signal(&sample.path, survey)
            .ok()
            .map(|s| s.iter().map(|&x| x * gain).collect());
        signals.push((sample.midi_velocity(), sample.lovel, sample.hivel, onset_s, signal));
        analyses.push(analysis);
    }
    if analyses.is_empty() {
        return None;
    }

    let reports: Vec<&piano_tuner::estimate::DecayReport> =
        analyses.iter().map(|a| &a.decays).collect();
    let comb = EngineComb::new(
        f64::from(base.notes.strike_position[index]),
        f64::from(base.notes.contact_width[index]),
        f64::from(base.notes.comb_floor[index]),
    );
    let curve = DecayCurve {
        sigma0: f64::from(base.notes.sigma0[index]),
        sigma1: f64::from(base.notes.sigma1[index]),
        residual: 0.0,
    };
    let split = DecaySplit {
        horizontal_gain_db: f64::from(base.voicing.horizontal_gain_db),
        horizontal_decay_ratio: f64::from(base.voicing.horizontal_decay_ratio),
    };
    let shaping = fit_note(
        key,
        &reports,
        comb,
        curve,
        split,
        &note_config.decay,
        &ShapingConfig::default(),
    );
    let spectra: Vec<Vec<(u32, f64)>> = analyses
        .iter()
        .map(|a| {
            a.decays
                .partials
                .iter()
                .filter(|fit| fit.k >= 1 && fit.initial_amplitude() > 0.0)
                .map(|fit| (fit.k, fit.initial_amplitude()))
                .collect()
        })
        .collect();
    let owned_reports: Vec<DecayReport> = analyses.iter().map(|a| a.decays.clone()).collect();
    let reference_spectrum = signals
        .iter()
        .zip(&spectra)
        .find(|((_, lovel, hivel, _, _), _)| (*lovel..=*hivel).contains(&REFERENCE_VELOCITY))
        .map(|(_, spectrum)| spectrum.clone())
        .unwrap_or_default();

    // The reference every level here is quoted against: the peak of the layer a
    // velocity-90 blow would trigger, at the level the instrument plays it.
    let reference_peak = signals
        .iter()
        .find(|(_, lovel, hivel, _, _)| (*lovel..=*hivel).contains(&REFERENCE_VELOCITY))
        .and_then(|(_, _, _, _, signal)| signal.as_ref())
        .map(|s| s.iter().fold(0.0f64, |m, &x| m.max(f64::from(x).abs())))
        .unwrap_or(0.0);

    let mut residuals = Vec::new();
    let mut spectrum = Vec::new();
    let mut refused = None;
    if reference_peak > 0.0 {
        for ((velocity, lovel, hivel, onset_s, signal), analysis) in signals.iter().zip(&analyses) {
            let Some(signal) = signal else { continue };
            let partial_hz: Vec<f64> = analysis
                .decays
                .partials
                .iter()
                .map(|fit| fit.frequency_hz)
                .filter(|f| f.is_finite() && *f > 0.0)
                .collect();
            let f0 = partial_hz.iter().copied().fold(f64::INFINITY, f64::min);
            if f0 < attack.min_f0_hz || !f0.is_finite() {
                refused = Some(format!(
                    "lowest tracked partial {f0:.1} Hz is under the {:.0} Hz this subtraction \
                     can separate inside an attack",
                    attack.min_f0_hz
                ));
                continue;
            }
            if let Some(metrics) = residual_metrics(
                key,
                *velocity,
                signal,
                f64::from(SAMPLE_RATE),
                &partial_hz,
                *onset_s,
                reference_peak,
                attack,
            ) {
                residuals.push(metrics);
            }
            // The spectrum is reported at the nominal velocity, which is the
            // layer every level in `[noise]` is quoted against.
            if (*lovel..=*hivel).contains(&REFERENCE_VELOCITY) {
                if let Some(bands) = onset_residual(
                    signal,
                    f64::from(SAMPLE_RATE),
                    &partial_hz,
                    *onset_s,
                    attack.residual_s,
                )
                // Reported over the schema's whole range whatever band the fit
                // narrowed to, because this table is what the band was chosen
                // *from* and a spectrum printed inside the answer is not one.
                .and_then(|r| density_bands(&r, f64::from(SAMPLE_RATE), &AttackConfig::default()))
                {
                    let peak = bands.iter().fold(0.0f64, |m, &(_, p)| m.max(p));
                    if peak > 0.0 {
                        spectrum = bands
                            .into_iter()
                            .map(|(hz, p)| (hz, 10.0 * (p.max(1e-30) / peak).log10()))
                            .collect();
                    }
                }
            }
        }
    }
    Some(Note {
        key,
        shaping,
        spectra,
        reports: owned_reports,
        comb_floor: None,
        residuals,
        spectrum,
        refused,
        reference_spectrum,
    })
}

/// The comb floor of every key, inverted on a line measured on the engine.
///
/// Four renders per key, at one velocity, analysed with the same settings the
/// recordings were: what is compared is how deep the *deepest measured partial*
/// stands, on both sides, so the leakage and the smooth-reference bias that
/// separate a comb's own depth from a measured one are on both sides of the
/// comparison. See `estimate::shaping::CombLine`.
fn fit_comb_floors(notes: &mut [Note], base: &Preset, survey: &SurveyConfig) {
    let config = ShapingConfig::default();
    println!(
        "\n key   deepest dB   at k   engine's line, dB at floor {COMB_PROBES:?}        floor"
    );
    let keys: Vec<u8> = notes.iter().map(|n| n.key).collect();
    let lines: Vec<Option<CombLine>> = {
        let next = AtomicUsize::new(0);
        let mut done: Vec<Option<CombLine>> = keys.iter().map(|_| None).collect();
        let workers = std::thread::available_parallelism()
            .map_or(1, |n| n.get())
            .min(keys.len().max(1));
        std::thread::scope(|scope| {
            let (tx, rx) = std::sync::mpsc::channel();
            for _ in 0..workers {
                let (next, keys, tx) = (&next, &keys, tx.clone());
                scope.spawn(move || loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    let Some(&key) = keys.get(index) else { return };
                    let line = comb_line(base, survey, key, &config);
                    if tx.send((index, line)).is_err() {
                        return;
                    }
                });
            }
            drop(tx);
            for (index, line) in rx {
                done[index] = line;
            }
        });
        done
    };
    for (note, line) in notes.iter_mut().zip(lines) {
        let deepest = measured_deepest(&note.spectra, &config);
        let Some((db, k)) = deepest else { continue };
        let Some(line) = line else { continue };
        let floor = line.floor_for(db).filter(|_| note.shaping.floor_measurable);
        println!(
            "{:>4} {:>12.1} {:>6} {:>40} {:>12}",
            note.key,
            db,
            k,
            line.probes
                .iter()
                .map(|&(_, d)| format!("{d:.1}"))
                .collect::<Vec<_>>()
                .join(" "),
            floor.map_or("-".into(), |f| format!(
                "{f:.3}{}",
                if line.saturated(f) { " (sat)" } else { "" }
            )),
        );
        note.comb_floor = floor;
    }
}

/// The half of the excitation the roughness fit throws away, measured on the
/// engine and multiplied back in.
///
/// [`piano_tuner::estimate::shaping::envelope_tilt`] carries the argument; this
/// is the render that feeds it. Each measured key is played through the engine
/// **with everything fitted so far already in it** — the comb floor, the
/// roughness gains — so that what the two smooth envelopes differ by is the
/// engine's own error in the hammer's tilt and nothing that has already been
/// corrected. The diffuse field is switched off for the same reason `comb_line`
/// switches it off: several decibels of frequency-dependent gain on every
/// partial would be measured as envelope.
fn fit_spectral_envelope(notes: &mut [Note], base: &Preset, survey: &SurveyConfig) {
    let config = ShapingConfig::default();
    println!(
        "\n key   tilt dB at k=1..4                       span dB   strongest partial  \
         recording / engine"
    );
    for note in notes.iter_mut() {
        let Some(index) = key_index(note.key) else { continue };
        if note.reference_spectrum.is_empty() {
            continue;
        }
        let mut probe = base.clone();
        probe.notes.comb_floor[index] = note.comb_floor.unwrap_or(0.0) as f32;
        probe.notes.partial_gains[index] = note.shaping.gains.clone();
        probe.soundboard.board_mix = 0.0;
        let Some(rendered) = rendered_spectrum(&probe, survey, note.key) else {
            continue;
        };
        let partials = note.shaping.gains.len().max(rendered.len());
        let Some(tilt) = envelope_tilt(&note.reference_spectrum, &rendered, partials, &config)
        else {
            continue;
        };
        let loudest = |spectrum: &[(u32, f64)]| {
            spectrum
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).expect("finite"))
                .map(|&(k, _)| k)
                .unwrap_or(0)
        };
        let db = |g: f32| 20.0 * f64::from(g).log10();
        let mut gains = note.shaping.gains.clone();
        gains.resize(partials, 1.0);
        for (g, t) in gains.iter_mut().zip(&tilt) {
            *g = (*g * t).clamp(MIN_PARTIAL_GAIN, MAX_PARTIAL_GAIN);
        }
        while gains.last() == Some(&1.0) {
            gains.pop();
        }
        let span = tilt.iter().fold(f64::MIN, |m, &g| m.max(db(g)))
            - tilt.iter().fold(f64::MAX, |m, &g| m.min(db(g)));
        println!(
            "{:>4}   {:>7.2} {:>7.2} {:>7.2} {:>7.2}   {:>13.1}   {:>8} / {:<8}",
            note.key,
            db(tilt[0]),
            tilt.get(1).copied().map_or(f64::NAN, db),
            tilt.get(2).copied().map_or(f64::NAN, db),
            tilt.get(3).copied().map_or(f64::NAN, db),
            span,
            loudest(&note.reference_spectrum),
            loudest(&rendered),
        );
        note.shaping.gains = gains;
    }
}

/// One key's time-zero spectrum as the engine renders it, measured by the code
/// that measured the recording.
fn rendered_spectrum(probe: &Preset, survey: &SurveyConfig, key: u8) -> Option<Vec<(u32, f64)>> {
    let index = key_index(key)?;
    let note_config = survey.note_config(equal_temperament(key)).ok()?;
    let engine = piano_emulator::preset::Preset::from_toml(&probe.to_toml()).ok()?;
    let events = [RenderEvent::new(
        0.05,
        Event::NoteOn {
            key,
            vel: REFERENCE_VELOCITY,
        },
    )];
    let (left, right) = render_to_buffer(&engine, &events, COMB_RENDER_S);
    let mono: Vec<f32> = left
        .iter()
        .zip(&right)
        .map(|(&l, &r)| 0.5 * (l + r))
        .collect();
    let seed = InharmonicModel::harmonic(f64::from(probe.notes.f0_hz[index]));
    let analysis = analyze_note(&mono, f64::from(SAMPLE_RATE), seed, &note_config).ok()?;
    Some(
        analysis
            .decays
            .partials
            .iter()
            .filter(|fit| fit.k >= 1 && fit.initial_amplitude() > 0.0)
            .map(|fit| (fit.k, fit.initial_amplitude()))
            .collect(),
    )
}

/// One key's line: the engine rendered at each probe floor and measured the way
/// the recording was.
fn comb_line(
    base: &Preset,
    survey: &SurveyConfig,
    key: u8,
    config: &ShapingConfig,
) -> Option<CombLine> {
    let index = key_index(key)?;
    let note_config = survey.note_config(equal_temperament(key)).ok()?;
    let mut line = CombLine {
        key,
        probes: Vec::new(),
    };
    for floor in COMB_PROBES {
        let mut probe = base.clone();
        probe.notes.comb_floor[index] = floor;
        // The diffuse field is several dB of frequency-dependent gain on every
        // partial and would be measured as roughness; the gate switches it off
        // for the same reason (`tests/calibration.rs`).
        probe.soundboard.board_mix = 0.0;
        let Ok(engine) = piano_emulator::preset::Preset::from_toml(&probe.to_toml()) else {
            continue;
        };
        let events = [RenderEvent::new(
            0.05,
            Event::NoteOn {
                key,
                vel: REFERENCE_VELOCITY,
            },
        )];
        let (left, right) = render_to_buffer(&engine, &events, COMB_RENDER_S);
        let mono: Vec<f32> = left
            .iter()
            .zip(&right)
            .map(|(&l, &r)| 0.5 * (l + r))
            .collect();
        let seed = InharmonicModel::harmonic(f64::from(base.notes.f0_hz[index]));
        let Ok(analysis) = analyze_note(&mono, f64::from(SAMPLE_RATE), seed, &note_config) else {
            continue;
        };
        let spectrum: Vec<(u32, f64)> = analysis
            .decays
            .partials
            .iter()
            .filter(|fit| fit.k >= 1 && fit.initial_amplitude() > 0.0)
            .map(|fit| (fit.k, fit.initial_amplitude()))
            .collect();
        if let Some((db, _)) = measured_deepest(std::slice::from_ref(&spectrum), config) {
            line.probes.push((f64::from(floor), db));
        }
    }
    (line.probes.len() >= 2).then_some(line)
}

/// The gains and the decay corrections, re-measured with each key's fitted comb
/// floor in the reference — which is what stops the two mechanisms from
/// double-counting the same null.
fn refit_with_floors(notes: &mut [Note], base: &Preset) {
    let config = ShapingConfig::default();
    let split = DecaySplit {
        horizontal_gain_db: f64::from(base.voicing.horizontal_gain_db),
        horizontal_decay_ratio: f64::from(base.voicing.horizontal_decay_ratio),
    };
    for note in notes.iter_mut() {
        let Some(index) = key_index(note.key) else { continue };
        let comb = EngineComb::new(
            f64::from(base.notes.strike_position[index]),
            f64::from(base.notes.contact_width[index]),
            note.comb_floor.unwrap_or(0.0),
        );
        let curve = DecayCurve {
            sigma0: f64::from(base.notes.sigma0[index]),
            sigma1: f64::from(base.notes.sigma1[index]),
            residual: 0.0,
        };
        let reports: Vec<&DecayReport> = note.reports.iter().collect();
        note.shaping = fit_note(
            note.key,
            &reports,
            comb,
            curve,
            split,
            &SurveyConfig::default().note.decay,
            &config,
        );
    }
}

// ------------------------------------------------------------------ reports

fn report_shaping(notes: &[Note]) {
    println!(
        "\n key   deepest  at k   bare comb    floor   rough   gains  span dB        clamped   \
         sigma  trusted"
    );
    for note in notes {
        let s = &note.shaping;
        let (low, high) = s.gain_span_db();
        println!(
            "{:>4} {:>9} {:>5} {:>11.1} {:>8} {:>7.2} {:>7} {:>7.1}..{:<7.1} {:>4}   {:>5} {:>8}",
            note.key,
            s.deepest_db.map_or("-".into(), |d| format!("{d:.1}")),
            s.deepest_k.map_or("-".into(), |k| k.to_string()),
            s.bare_comb_db,
            note.comb_floor.map_or("-".into(), |f| format!("{f:.3}")),
            s.roughness_db,
            s.gains.len(),
            low,
            high,
            s.clamped_gains,
            s.sigma_scale.len(),
            s.trusted_rates,
        );
    }
    let rows = notes.iter().filter(|n| !n.shaping.gains.is_empty()).count();
    let sigma_rows = notes
        .iter()
        .filter(|n| !n.shaping.sigma_scale.is_empty())
        .count();
    println!(
        "\nper-partial: {rows} keys carry gains, {sigma_rows} carry decay corrections, \
         {} carry a comb floor",
        notes
            .iter()
            .filter(|n| n.comb_floor.is_some_and(|f| f > 0.0))
            .count()
    );
}

fn report_strike(
    notes: &[Note],
    base: &Preset,
    attack: &AttackConfig,
) -> piano_tuner::estimate::attack::StrikeFitReport {
    let measurements: Vec<AttackResidual> = notes
        .iter()
        .flat_map(|note| note.residuals.iter().copied())
        .collect();
    println!("\n key   layers   level@90 dB   velocity dB   centroid Hz   rolloff Hz   decay s   flat dB");
    for note in notes {
        if note.residuals.is_empty() {
            continue;
        }
        let median = |get: fn(&AttackResidual) -> f64| -> f64 {
            let mut v: Vec<f64> = note.residuals.iter().map(get).filter(|x| x.is_finite()).collect();
            v.sort_by(f64::total_cmp);
            v.get(v.len() / 2).copied().unwrap_or(f64::NAN)
        };
        let report = fit_strike(
            &note.residuals,
            &base.noise.strike,
            attack,
            &NoiseConfig::default(),
        );
        println!(
            "{:>4} {:>8} {:>13.1} {:>13.1} {:>13.0} {:>12.0} {:>9.3} {:>9.1}",
            note.key,
            note.residuals.len(),
            report
                .per_key_db
                .first()
                .map_or(f64::NAN, |&(_, db)| db),
            report
                .per_key_velocity_db
                .first()
                .map_or(f64::NAN, |&(_, s)| s),
            median(|m| m.centroid_hz),
            median(|m| m.bandwidth_hz),
            median(|m| m.decay_s),
            median(|m| m.flatness_db),
        );
    }
    for note in notes {
        if let Some(reason) = &note.refused {
            println!("  key {:>3}: no residual — {reason}", note.key);
        }
    }
    println!("\nthe residual's own spectrum at the nominal velocity, third-octave density in dB \
              under each key's loudest band:");
    for note in notes {
        if note.spectrum.is_empty() {
            continue;
        }
        if ![21u8, 45, 60, 84, 96].contains(&note.key) {
            continue;
        }
        print!("  key {:>3}:", note.key);
        for &(hz, db) in note.spectrum.iter().step_by(3) {
            print!(" {hz:.0}:{db:.0}");
        }
        println!();
    }
    let report = fit_strike(
        &measurements,
        &base.noise.strike,
        attack,
        &NoiseConfig::default(),
    );
    println!(
        "\n[noise.strike]: {:.0} Hz centroid, {:.0} Hz band limit, {:.3} s, {:.1} dB of velocity, \
         {} anchors over {} keys / {} recordings (median flatness {:.1} dB)",
        report.strike.centroid_hz,
        report.strike.bandwidth_hz,
        report.strike.decay_s,
        report.strike.velocity_db,
        report.strike.level_db.len(),
        report.keys,
        report.recordings,
        report.flatness_db,
    );
    for (key, db) in report.anchors() {
        println!("    key {key:>3}: {db:>6.1} dB");
    }
    report
}

/// The damper, from the release recordings and a line measured on the engine.
///
/// The answer is **one number**, not eighty-eight. Each measured key gives a
/// ratio between the `damper_sigma` its own recorded tail asks for and the one
/// the preset has, and what is written is the median of those ratios applied to
/// the whole table — `PresetBuilder`'s documented single-measurement rule, which
/// scales the base curve rather than flattening it. Three things say that is the
/// right shape of answer:
///
/// * The per-key ratios scatter over a factor of ten while their median is
///   stable, because a `harm*` recording is the damped string *plus* whatever
///   else the instrument was doing, and how much else differs by key.
/// * Fewer than half the keys invert at all: at the rest even a damper turned
///   off entirely leaves the engine's tail shorter than the recording's, so the
///   line saturates and the key has measured a bound rather than a value.
/// * Nothing in these recordings distinguishes one key's damper from its
///   neighbour's by more than that scatter, and the base table's compass shape
///   was designed rather than measured — a curve through ten noisy points would
///   replace a designed shape with a fitted wiggle.
fn report_damper(library: &SampleLibrary, base: &Preset) -> Result<Option<(u8, f64)>> {
    /// One release recording: the key it belongs to, its 20 dB tail, and the
    /// same tail measured in a low band and a high one.
    type Release = (u8, f64, Option<(f64, f64)>);
    let config = DamperConfig::default();
    let mut measured: Vec<Release> = Vec::new();
    for sample in library.mechanism_of(MechanismKind::StringResonance) {
        let Some(key) = sample.key else { continue };
        let Some(index) = key_index(key) else { continue };
        let Ok(recording) = audio::load_at(&sample.path, SAMPLE_RATE) else {
            continue;
        };
        let mono = recording.mono();
        let Some(t20) = tail_decay_s(&mono, f64::from(SAMPLE_RATE), &config) else {
            continue;
        };
        let bands = band_release(
            &mono,
            f64::from(SAMPLE_RATE),
            f64::from(base.notes.f0_hz[index]),
            &config,
        );
        measured.push((key, t20, bands));
    }
    let mut keys: Vec<u8> = measured.iter().map(|&(key, _, _)| key).collect();
    keys.sort_unstable();
    keys.dedup();

    println!(
        "\n key   recorded T20 s   low band   high band   hi/lo   engine T20 1.00x / 0.25x   \
         asks for   base   ratio"
    );
    let mut ratios: Vec<f64> = Vec::new();
    let mut fitted_keys: Vec<(u8, f64, f64)> = Vec::new();
    let mut band_ratios: Vec<f64> = Vec::new();
    for key in keys {
        let mut times: Vec<f64> = measured
            .iter()
            .filter(|&&(k, _, _)| k == key)
            .map(|&(_, t, _)| t)
            .collect();
        times.sort_by(f64::total_cmp);
        let recorded = times[times.len() / 2];
        let bands = measured
            .iter()
            .find(|&&(k, _, b)| k == key && b.is_some())
            .and_then(|&(_, _, b)| b);
        if let Some((low, high)) = bands.filter(|&(l, h)| l > 0.0 && h > 0.0) {
            band_ratios.push(high / low);
        }
        let Some(index) = key_index(key) else { continue };
        let base_sigma = f64::from(base.notes.damper_sigma[index]);
        let mut line = DamperLine {
            key,
            probes: Vec::new(),
        };
        for scale in PROBE_SCALES {
            let mut probe = base.clone();
            probe.notes.damper_sigma[index] = (base_sigma * scale) as f32;
            if probe.validate().is_err() {
                continue;
            }
            if let Some(t20) = rendered_release_t20(&probe, key, &config) {
                line.probes.push((base_sigma * scale, t20));
            }
        }
        let fitted = line
            .sigma_for(recorded, &config)
            .filter(|&s| !line.saturated(s, &config));
        println!(
            "{:>4} {:>16.3} {:>10} {:>11} {:>7} {:>12} / {:<8} {:>10} {:>6.2} {:>7}",
            key,
            recorded,
            bands.map_or("-".into(), |(l, _)| format!("{l:.3}")),
            bands.map_or("-".into(), |(_, h)| format!("{h:.3}")),
            bands.map_or("-".into(), |(l, h)| format!("{:.2}", h / l)),
            line.probes
                .first()
                .map_or("-".into(), |&(_, t)| format!("{t:.3}")),
            line.probes
                .get(1)
                .map_or("-".into(), |&(_, t)| format!("{t:.3}")),
            fitted.map_or("(saturated)".into(), |s| format!("{s:.2}")),
            base_sigma,
            fitted.map_or("-".into(), |s| format!("{:.3}", s / base_sigma)),
        );
        if let Some(sigma) = fitted {
            ratios.push(sigma / base_sigma);
            fitted_keys.push((key, sigma, recorded));
        }
    }
    let mut recorded_all: Vec<f64> = measured.iter().map(|&(_, t, _)| t).collect();
    recorded_all.sort_by(f64::total_cmp);
    band_ratios.sort_by(f64::total_cmp);
    println!(
        "\ndamper: {} of {} keys invert; the recordings' median T20 is {:.3} s, and their high \
         band outlasts their low band by a median factor of {:.2} — which is what \
         `voicing.damper_weight` already delivers (0.85 of the grip an octave up is a 1.18 \
         ratio), so the anchors are left where they are.",
        ratios.len(),
        measured.len(),
        recorded_all
            .get(recorded_all.len() / 2)
            .copied()
            .unwrap_or(f64::NAN),
        band_ratios
            .get(band_ratios.len() / 2)
            .copied()
            .unwrap_or(f64::NAN),
    );
    if ratios.is_empty() {
        return Ok(None);
    }
    ratios.sort_by(f64::total_cmp);
    let scale = ratios[ratios.len() / 2];
    // The anchor is the key nearest the middle of the compass that inverted, so
    // that the one measurement `PresetBuilder` scales the table through is the
    // best-conditioned one.
    let anchor = fitted_keys
        .iter()
        .min_by_key(|&&(key, _, _)| u8::abs_diff(key, 60))
        .map(|&(key, _, _)| key)
        .expect("non-empty");
    let index = key_index(anchor).expect("a surveyed key");
    let wanted = f64::from(base.notes.damper_sigma[index]) * scale;
    println!(
        "damper_sigma x {scale:.3} across the compass (anchored at key {anchor}: {:.2} -> \
         {wanted:.2} /s), from ratios spanning {:.3}..{:.3}",
        base.notes.damper_sigma[index],
        ratios[0],
        ratios[ratios.len() - 1],
    );
    // Verification, on the engine itself and with the recordings' own code: what
    // the fitted table actually delivers at every key that was measured.
    println!("\n key   recorded T20 s   fitted render T20 s");
    let mut fitted_preset = base.clone();
    for sigma in fitted_preset.notes.damper_sigma.iter_mut() {
        *sigma = (f64::from(*sigma) * scale) as f32;
    }
    for &(key, _, recorded) in &fitted_keys {
        match rendered_release_t20(&fitted_preset, key, &config) {
            Some(t20) => println!("{key:>4} {recorded:>16.3} {t20:>21.3}"),
            None => println!("{key:>4} {recorded:>16.3} {:>21}", "-"),
        }
    }
    Ok(Some((anchor, wanted)))
}

/// The engine's own release tail for one key, measured the way the recording is.
fn rendered_release_t20(preset: &Preset, key: u8, config: &DamperConfig) -> Option<f64> {
    let engine = engine_preset(preset)?;
    let events = [
        RenderEvent::new(0.05, Event::NoteOn { key, vel: REFERENCE_VELOCITY }),
        RenderEvent::new(0.05 + HOLD_S, Event::NoteOff { key, vel: 64 }),
    ];
    let (left, right) = render_to_buffer(&engine, &events, RELEASE_RENDER_S);
    let start = (((0.05 + HOLD_S) * SAMPLE_RATE as f32) as usize).min(left.len());
    let tail: Vec<f32> = left[start..]
        .iter()
        .zip(&right[start..])
        .map(|(&l, &r)| 0.5 * (l + r))
        .collect();
    tail_decay_s(&tail, f64::from(SAMPLE_RATE), config)
}

/// The tuner's preset as the engine's, through the file both crates agree on.
fn engine_preset(preset: &Preset) -> Option<piano_emulator::preset::Preset> {
    piano_emulator::preset::Preset::from_toml(&preset.to_toml()).ok()
}
