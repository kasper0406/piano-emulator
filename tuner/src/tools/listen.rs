//! `piano-tuner listen` — the listening material for a preset, against **its
//! own** library's recordings.
//!
//! Every other render-producing tool in this crate is pointed at Salamander,
//! because until now there was one measured preset. A range of pianos needs
//! one more thing per preset than a board: two pieces of music the ear can
//! judge, rendered by the engine on that preset and by the sampler on **the
//! library that preset was estimated from** — never against another piano's
//! recordings, which would be comparing two instruments and calling it an
//! error.
//!
//! Two pieces, chosen because they are the two the boards already argue over:
//!
//! - **the melody line** (`estimate::melody::soprano`, the Ode line the melody
//!   board scores), which is where pitch, level and the stereo image are
//!   audible one note at a time;
//! - **a pedalled chord phrase** (`realism::chords_pedal`), which is where the
//!   sympathetic halo, the released decay and the pedal mechanism are.
//!
//! ### Levels, and why the two sides do not share a gain
//!
//! The engine take and the reference take are normalised **separately**, each
//! to the same RMS over its own whole phrase, and the gains applied are
//! printed into the folder's own `README.md`. A microphone twelve centimetres
//! above somebody else's strings has no level in common with a modelled
//! radiation, so a shared gain would be a coincidence rather than a
//! measurement — `tools::ab` reached the same conclusion for the same reason.
//! What a shared gain *would* preserve — a difference in loudness between two
//! takes of the same engine — is not what is being listened to here.
//!
//! ### What is genuinely recorded, and what is a resampled neighbour
//!
//! The evaluation policy is that only genuinely recorded keys are fitted and
//! scored. Listening material is not scored, so it may use the whole tune —
//! but which of its notes are the library's own takes and which are its
//! resampler is a property of the material, so the folder's `README.md` says,
//! note by note.
//!
//! ```text
//! piano-tuner listen <data-dir> <preset.toml> [renders/<name>]
//! ```

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use piano_emulator::preset::Preset;
use piano_emulator::render::render_to_buffer;
use piano_tuner::adapter::{instrument_path, LibrarySpec};
use piano_tuner::audio::Audio;
use piano_tuner::sampler::engine_events;
use piano_tuner::{estimate, realism, Phrase, SampleLibrary, Sampler, SamplerConfig, SAMPLE_RATE};

type Exit = std::result::Result<(), Box<dyn std::error::Error>>;

/// RMS every take is brought to, over its own whole phrase.
const TARGET_RMS: f32 = 0.05;

/// Peak a take is pulled back to if the RMS match would clip it. A clipped
/// comparison is not a comparison.
const PEAK_CEILING: f32 = 0.97;

pub fn run(args: Vec<String>) -> Exit {
    let mut args = args.into_iter();
    let data = PathBuf::from(args.next().unwrap_or_else(|| "data/salamander".into()));
    let preset_path = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "presets/salamander-c5.toml".into()),
    );
    let out = PathBuf::from(args.next().unwrap_or_else(|| {
        let stem = preset_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("preset");
        format!("renders/{stem}")
    }));

    let sfz = instrument_path(&data)?;
    let library = SampleLibrary::from_sfz(&sfz)?;
    let spec = data
        .file_name()
        .and_then(|n| n.to_str())
        .and_then(LibrarySpec::find);
    let preset = Preset::load(&preset_path)?;
    std::fs::create_dir_all(&out)?;

    println!("listening material for {}", preset_path.display());
    println!("  against {}", sfz.display());
    println!("  into    {}", out.display());

    let pieces = [
        (
            "melody",
            estimate::melody::soprano(),
            "the Ode line the melody board scores: pitch, level and image, one note at a time",
        ),
        (
            "chords",
            realism::chords_pedal(),
            "a pedalled chord phrase: the sympathetic halo, the released decay and the pedal",
        ),
    ];

    let mut report = String::new();
    let _ = writeln!(report, "# Listening material — `{}`\n", preset_path.display());
    if let Some(spec) = spec {
        let _ = writeln!(report, "**{}**\n", spec.instrument);
        let _ = writeln!(report, "- library: `{}`", spec.id);
        let _ = writeln!(report, "- credit: {}", spec.credit);
        let _ = writeln!(report, "- licence: {}", spec.licence);
        let _ = writeln!(report, "- source: <{}>", spec.source_url);
        let _ = writeln!(
            report,
            "- shape: {} recorded keys x {} velocity layers, delivered at {} Hz{}\n",
            spec.layout.keys().len(),
            spec.bands.count(),
            spec.delivered_rate_hz,
            if spec.is_native_rate() {
                " (native)"
            } else {
                " (resampled once, offline, at fetch)"
            }
        );
    }
    let _ = writeln!(
        report,
        "The engine take and the reference take are normalised **separately**, each to \
         an RMS of {TARGET_RMS} over its own whole phrase. Nothing here is scored — the \
         boards are — and nothing here is a level comparison; see `tools/listen.rs`.\n"
    );

    for (name, phrase, description) in pieces {
        println!("\n{name}: {description}");
        let engine = render_engine(&preset, &phrase)?;
        let mut sampler = Sampler::with_config(&sfz, SamplerConfig::default())?;
        let reference = sampler.render(&phrase.events, phrase.duration_s)?;

        let (engine, engine_gain) = normalise(engine);
        let (reference, reference_gain) = normalise(reference);
        let engine_path = out.join(format!("{name}_engine.wav"));
        let reference_path = out.join(format!("{name}_reference.wav"));
        engine.write_wav(&engine_path)?;
        reference.write_wav(&reference_path)?;
        println!(
            "  {} ({:+.2} dB)  /  {} ({:+.2} dB)",
            engine_path.display(),
            db(engine_gain),
            reference_path.display(),
            db(reference_gain)
        );

        let _ = writeln!(report, "## `{name}` — {description}\n");
        let _ = writeln!(
            report,
            "| take | file | gain applied | length |\n|---|---|---|---|"
        );
        let _ = writeln!(
            report,
            "| engine | `{name}_engine.wav` | {:+.2} dB | {:.1} s |",
            db(engine_gain),
            engine.duration_s()
        );
        let _ = writeln!(
            report,
            "| reference | `{name}_reference.wav` | {:+.2} dB | {:.1} s |\n",
            db(reference_gain),
            reference.duration_s()
        );

        let keys = phrase_keys(&phrase);
        let recorded: Vec<u8> = keys
            .iter()
            .copied()
            .filter(|k| !library.layers(*k).is_empty())
            .collect();
        let _ = writeln!(
            report,
            "{} of this phrase's {} distinct keys are the library's **own takes** \
             ({}); the rest are its resampler playing a neighbour, which is \
             listening material and is never scored.\n",
            recorded.len(),
            keys.len(),
            if recorded.is_empty() {
                "none".to_string()
            } else {
                recorded
                    .iter()
                    .map(|k| k.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            }
        );
        println!(
            "  {} of {} distinct keys are genuinely recorded",
            recorded.len(),
            keys.len()
        );
    }

    let readme = out.join("README.md");
    std::fs::write(&readme, report)?;
    println!("\nwrote {}", readme.display());
    Ok(())
}

fn render_engine(preset: &Preset, phrase: &Phrase) -> Result<Audio, Box<dyn std::error::Error>> {
    let events = engine_events::to_render_events(&phrase.events);
    let (left, right) = render_to_buffer(preset, &events, phrase.duration_s as f32);
    Ok(Audio::new(SAMPLE_RATE, vec![left, right])?)
}

/// Brings a take to [`TARGET_RMS`] over its whole length, backing off if that
/// would clip. Returns the gain that was applied, so the folder's README can
/// print it: a normalisation nobody can see the size of is a normalisation
/// that can hide a defect.
fn normalise(audio: Audio) -> (Audio, f32) {
    let mut sum = 0.0f64;
    let mut count = 0usize;
    let mut peak = 0.0f32;
    for channel in &audio.channels {
        for &sample in channel {
            sum += f64::from(sample) * f64::from(sample);
            count += 1;
            peak = peak.max(sample.abs());
        }
    }
    if count == 0 || sum <= 0.0 {
        return (audio, 1.0);
    }
    let rms = (sum / count as f64).sqrt() as f32;
    let mut gain = TARGET_RMS / rms;
    if peak * gain > PEAK_CEILING {
        gain = PEAK_CEILING / peak.max(f32::MIN_POSITIVE);
    }
    let channels = audio
        .channels
        .iter()
        .map(|c| c.iter().map(|&s| s * gain).collect())
        .collect();
    (
        Audio::new(audio.sample_rate, channels).expect("same shape"),
        gain,
    )
}

fn db(gain: f32) -> f64 {
    20.0 * f64::from(gain).max(1e-12).log10()
}

fn phrase_keys(phrase: &Phrase) -> Vec<u8> {
    let mut keys: Vec<u8> = phrase
        .events
        .iter()
        .filter_map(|e| match e.event {
            piano_tuner::SamplerEvent::NoteOn { key, vel } if vel > 0 => Some(key),
            _ => None,
        })
        .collect();
    keys.sort_unstable();
    keys.dedup();
    keys
}

#[allow(dead_code)]
fn exists(path: &Path) -> bool {
    path.exists()
}
