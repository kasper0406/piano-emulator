//! `piano-tuner stereo` — the A/B for `PHYSICS.md` §8, and the one board in
//! this repository that is **not** a mono sum.
//!
//! ```sh
//! cargo run --release -p piano-tuner -- stereo [data/salamander] [renders/stereo]
//! ```
//!
//! Two pieces of music — the Ode melody line alone and unpedalled, and the
//! pedalled chord phrase where the board rings longest — through the three
//! things the engine has been, plus the recording:
//!
//! | file | what it is |
//! |---|---|
//! | `*_panpot.wav` | `[voicing.mics]` deleted: one mono voice scaled into two channels, and the board's two orthogonal taps. What the instrument was before item 351. |
//! | `*_pair.wav` | the virtual capsule pair alone (`[voicing.mics.modal]` deleted): item 359's fitted geometry, four of six bands. |
//! | `*_engine.wav` | the shipped preset: the pair **and** the board's mode-controlled band. |
//! | `*_reference.wav` | the Salamander recording of the same music. |
//!
//! # What is being listened for, and why the levels are shared
//!
//! All three engine renders have **the same mono sum, sample for sample**
//! (`soundboard::Mics`), so they are matched to one gain and not to three: any
//! level difference between them is a difference in *side* energy, which is the
//! whole subject, and normalising it away would be normalising away the
//! finding. The recording gets its own gain, because somebody else's
//! microphones twelve centimetres above somebody else's strings are not
//! comparable to a rendered level — `tools::ab` makes the same split for the
//! same reason.
//!
//! So: fold any two of the first three to mono and they are identical. Listen
//! to them in stereo and the pan-pot is a point between the speakers, the pair
//! is an instrument with a width that grows with pitch, and the shipped one
//! has a low-mid that steps *outside* the speakers where the board's modes are.
//! The recording is the thing all three are being measured against.

use std::path::{Path, PathBuf};

use piano_emulator::preset::Preset;
use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_tuner::audio::Audio;
use piano_tuner::cache;
use piano_tuner::estimate::melody;
use piano_tuner::realism::{self, Phrase, StereoImage, StereoProfilePoint};
use piano_tuner::sampler::{engine_events, SAMPLER_VERSION};
use piano_tuner::{Sampler, SAMPLE_RATE};

/// Level the group is matched to, as RMS of the mono fold-down. `tools::ab`'s
/// own target, so the two boards' files sit at the same loudness.
const TARGET_RMS: f32 = 0.05;

/// Ceiling the shared peak guard holds the group under.
const PEAK_CEILING: f32 = 0.7;

/// The four renders of one piece of music, in the order the report prints them.
struct Take {
    name: &'static str,
    title: &'static str,
    audio: Audio,
}

pub fn run(args: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
    let mut args = args.into_iter();
    let data = PathBuf::from(args.next().unwrap_or_else(|| "data/salamander".into()));
    let out = PathBuf::from(args.next().unwrap_or_else(|| "renders/stereo".into()));
    let preset_path = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "presets/salamander-c5.toml".into()),
    );
    let sfz = data.join("SalamanderGrandPiano-V3+20200602.sfz");
    if !sfz.exists() {
        eprintln!(
            "the reference piano is not here: {}\nrun data/fetch_salamander.sh first (707 MiB).",
            sfz.display()
        );
        std::process::exit(2);
    }
    std::fs::create_dir_all(&out)?;

    let shipped = Preset::load(&preset_path)?;
    let mics = shipped
        .voicing
        .mics
        .ok_or("this board needs a preset with [voicing.mics]")?;
    // The three engine variants, each one field poorer than the last.
    let mut pair_only = shipped.clone();
    pair_only.voicing.mics = Some(piano_emulator::preset::MicVoicing {
        modal: None,
        ..mics
    });
    let mut panpot = shipped.clone();
    panpot.voicing.mics = None;

    let pieces = [melody::soprano(), realism::chords_pedal()];
    let mut sections = String::new();
    for phrase in &pieces {
        let mut takes = vec![
            Take {
                name: "panpot",
                title: "the pan-pot, `[voicing.mics]` absent",
                audio: render_engine(&panpot, phrase),
            },
            Take {
                name: "pair",
                title: "the capsule pair alone, `[voicing.mics.modal]` absent",
                audio: render_engine(&pair_only, phrase),
            },
            Take {
                name: "engine",
                title: "the shipped preset: the pair and the mode-controlled band",
                audio: render_engine(&shipped, phrase),
            },
        ];
        // One gain for all three engine takes — they share a mono sum, so a
        // per-take normalisation would be measuring the side signal and
        // removing it.
        let engine_rms = rms(&takes[2].audio.mono());
        let reference = render_reference(&data, &sfz, phrase)?;
        let reference_rms = rms(&reference.mono());
        if engine_rms <= 0.0 || reference_rms <= 0.0 {
            return Err(format!("{} rendered silence", phrase.name).into());
        }
        let mut gains = vec![f64::from(TARGET_RMS) / engine_rms; 3];
        gains.push(f64::from(TARGET_RMS) / reference_rms);
        takes.push(Take {
            name: "reference",
            title: "the Salamander recording of the same music",
            audio: reference,
        });
        // A shared peak guard, so the group stays comparable if any one of
        // them would clip.
        let loudest = takes
            .iter()
            .zip(&gains)
            .map(|(t, &g)| peak(&t.audio) * g as f32)
            .fold(0.0f32, f32::max);
        if loudest > PEAK_CEILING {
            let guard = f64::from(PEAK_CEILING / loudest);
            gains.iter_mut().for_each(|g| *g *= guard);
        }
        for (take, &gain) in takes.iter_mut().zip(&gains) {
            take.audio = scale(&take.audio, gain as f32);
        }

        for take in &takes {
            take.audio
                .write_wav(out.join(format!("{}_{}.wav", phrase.name, take.name)))?;
        }
        sections.push_str(&section(phrase, &takes));
    }

    let report = format!(
        "# The stereo image, heard\n\n\
         *`piano-tuner stereo` — `PHYSICS.md` §8, `DECISIONS.md` 313-317 and 346-379.*\n\n\
         Every other board in `renders/` scores a **mono sum**, on purpose: a stereo\n\
         distance would mostly measure somebody else's microphones. This one is the\n\
         exception, and it exists because the largest single difference ever measured\n\
         between this engine and the recording was invisible to all of them — the\n\
         recording's two channels correlate at **+0.95 below 125 Hz and about zero\n\
         above it**, and the engine's did exactly the reverse (item 314).\n\n\
         Two pieces of music, four takes each. The three engine takes share **one\n\
         gain**, because they share a mono sum sample for sample: fold any two of\n\
         them down and the difference is `f32` rounding. Everything you can hear\n\
         between them is side energy, which is the subject. The recording is matched\n\
         separately.\n\n\
         The tables are `realism::stereo_image`: the lag-zero interchannel\n\
         correlation per band, the largest |r| over ±5 ms and where it sits, and the\n\
         mid-over-side energy ratio. The row to read is `125-250` and `250-500`,\n\
         where the recording is **negative** — its difference is larger than its sum,\n\
         which is a soundboard's modes seen from two capsules that straddle a nodal\n\
         line, and which a mono fold-down throws away.\n\n\
         {sections}"
    );
    std::fs::write(out.join("STEREO.md"), &report)?;
    println!("{report}");
    println!("wrote {}", out.join("STEREO.md").display());
    Ok(())
}

fn section(phrase: &Phrase, takes: &[Take]) -> String {
    use std::fmt::Write as _;
    let mut s = String::new();
    let _ = write!(
        s,
        "## `{}` — {}\n\n\
         | take | file | 63-125 | 125-250 | 250-500 | 500-2k | 2k-6k | 6k-12k | broadband |\n\
         |---|---|---:|---:|---:|---:|---:|---:|---:|\n",
        phrase.name, phrase.description
    );
    let images: Vec<StereoImage> = takes
        .iter()
        .map(|t| realism::stereo_image_of(&t.audio).expect("two channels"))
        .collect();
    for (take, image) in takes.iter().zip(&images) {
        let _ = write!(s, "| {} | `{}_{}.wav` |", take.title, phrase.name, take.name);
        for band in &image.bands {
            let _ = write!(s, " {:+.3} |", band.r0);
        }
        let _ = writeln!(s, " {:+.3} |", image.broadband.r0);
    }
    let _ = write!(
        s,
        "\nMid over side, dB — the same rows, the statistic a mono fold-down loses:\n\n\
         | take | 63-125 | 125-250 | 250-500 | 500-2k | 2k-6k | 6k-12k |\n\
         |---|---:|---:|---:|---:|---:|---:|\n"
    );
    for (take, image) in takes.iter().zip(&images) {
        let _ = write!(s, "| {} |", take.title);
        for band in &image.bands {
            let _ = write!(s, " {:+.1} |", band.mid_side_db);
        }
        let _ = writeln!(s);
    }
    let _ = write!(
        s,
        "\nPeak |r| and the lag it sits at, milliseconds (positive: the right \
         channel leads):\n\n\
         | take | 63-125 | 125-250 | 250-500 | 500-2k | 2k-6k | 6k-12k |\n\
         |---|---:|---:|---:|---:|---:|---:|\n"
    );
    for (take, image) in takes.iter().zip(&images) {
        let _ = write!(s, "| {} |", take.title);
        for band in &image.bands {
            let _ = write!(s, " {:.2} @ {:+.2} |", band.peak_r.abs(), band.lag_ms);
        }
        let _ = writeln!(s);
    }

    // The curve the coarse bands are a summary of: this is where the
    // recording's step at 140 Hz and its negative lobe are visible as a shape.
    let profiles: Vec<Vec<StereoProfilePoint>> = takes
        .iter()
        .map(|t| realism::stereo_profile_of(&t.audio).expect("two channels"))
        .collect();
    let _ = write!(
        s,
        "\nThe same thing as a curve — sixth-octave `r0`, 63 Hz to 4 kHz, which is \
         where the shape the six bands summarise is actually visible:\n\n\
         | Hz | {} |\n|---:|{}\n",
        takes
            .iter()
            .map(|t| t.name)
            .collect::<Vec<_>>()
            .join(" | "),
        "---:|".repeat(takes.len())
    );
    for (i, point) in profiles[0].iter().enumerate() {
        if point.hz < 63.0 || point.hz > 4_000.0 {
            continue;
        }
        let _ = write!(s, "| {:.0} |", point.hz);
        for profile in &profiles {
            match profile.get(i) {
                Some(p) if p.level_db > -60.0 => {
                    let _ = write!(s, " {:+.3} |", p.r0);
                }
                _ => {
                    let _ = write!(s, " — |");
                }
            }
        }
        let _ = writeln!(s);
    }
    let _ = writeln!(s);
    s
}

fn render_engine(preset: &Preset, phrase: &Phrase) -> Audio {
    let events: Vec<RenderEvent> = engine_events::to_render_events(&phrase.events);
    let (left, right) = render_to_buffer(preset, &events, phrase.duration_s as f32);
    Audio::new(SAMPLE_RATE, vec![left, right]).expect("the engine renders stereo")
}

fn render_reference(
    data: &Path,
    sfz: &Path,
    phrase: &Phrase,
) -> Result<Audio, piano_tuner::Error> {
    let mut key = cache::Fingerprint::new();
    key.str("renders/stereo/reference")
        .u64(u64::from(SAMPLER_VERSION))
        .file(sfz)?
        .u64(u64::from(SAMPLE_RATE))
        .str(phrase.name)
        .f64(phrase.duration_s);
    let path = cache::reference_dir(data).join(format!("stereo-ab-{}-{}.wav", phrase.name, key.hex()));
    cache::audio(&path, || {
        let mut sampler = Sampler::new(sfz)?;
        sampler.render(&phrase.events, phrase.duration_s)
    })
}

fn rms(signal: &[f32]) -> f64 {
    if signal.is_empty() {
        return 0.0;
    }
    (signal.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>() / signal.len() as f64).sqrt()
}

fn peak(audio: &Audio) -> f32 {
    audio
        .channels
        .iter()
        .flat_map(|c| c.iter())
        .fold(0.0f32, |m, &x| m.max(x.abs()))
}

fn scale(audio: &Audio, gain: f32) -> Audio {
    Audio::new(
        audio.sample_rate,
        audio
            .channels
            .iter()
            .map(|c| c.iter().map(|&x| x * gain).collect())
            .collect(),
    )
    .expect("a scaled copy has the same shape")
}
