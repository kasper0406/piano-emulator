//! A/B material for `TUNING.md`'s Phase D: the same music through the
//! hand-tuned preset and through the one estimated from the Salamander
//! recordings, plus — for the single notes — the Salamander recording itself
//! next to both.
//!
//! Three groups land in `renders/salamander-ab/`:
//!
//! * **Single notes** across the compass, at the sampled keys, one file per
//!   preset and one straight out of the library. This is the comparison that
//!   can be judged: same note, same velocity, same level.
//! * **The demo**, which is the whole instrument under pedals and dynamics.
//! * **A pedal phrase**, where what is being listened to is the sympathetic
//!   halo and the decay of a released chord — the two things the estimated
//!   decay curve changes most.
//!
//! Level matching is done on the RMS of the first second, which is the note's
//! prompt sound: a decay that differs between the two presets must be *heard*
//! rather than removed by the normalisation, so the files are lined up where
//! they start rather than over their whole length. A peak guard follows,
//! because a matched level is no use if it clips.
//!
//! The two presets share one gain and the source recording gets its own. Any
//! difference in radiated level between two presets of the same engine is a
//! real difference and has to survive the normalisation; the level of somebody
//! else's microphone twelve centimetres above the strings is not comparable to
//! either of them and is simply brought alongside.
//!
//! ```text
//! cargo run --release --example salamander_ab -- [data/salamander] [renders/salamander-ab]
//! ```

use std::path::{Path, PathBuf};

use piano_emulator::preset::Preset;
use piano_emulator::render::{demo_sequence, render_to_buffer, RenderEvent, DEMO_DURATION_S};
use piano_emulator::types::{Event, PedalEvent};
use piano_tuner::{audio, SampleLibrary, SAMPLE_RATE};

/// Keys the A/B renders cover: sampled by Salamander, and spread over the
/// compass so that the bass, the tenor, the middle and the treble each get a
/// comparison.
const KEYS: [u8; 8] = [21, 33, 45, 57, 60, 72, 84, 96];

/// Velocity the single notes are struck at, and the layer the source sample is
/// taken from — the layer whose band contains it.
const VELOCITY: u8 = 90;

/// How long each single note is rendered and how much of the source is kept.
const NOTE_SECONDS: f32 = 8.0;

/// Level every A/B group is matched to, as RMS over its first second.
const TARGET_RMS: f32 = 0.05;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let root = PathBuf::from(args.next().unwrap_or_else(|| "data/salamander".into()));
    let out = PathBuf::from(args.next().unwrap_or_else(|| "renders/salamander-ab".into()));
    std::fs::create_dir_all(&out)?;

    let presets = [
        ("default", Preset::load(Path::new("presets/default.toml"))?),
        (
            "salamander",
            Preset::load(Path::new("presets/salamander-c5.toml"))?,
        ),
    ];
    let library = SampleLibrary::from_sfz(root.join("SalamanderGrandPiano-V3+20200602.sfz")).ok();

    for key in KEYS {
        let name = note_name(key);
        let group: Vec<(String, Stereo)> = presets
            .iter()
            .map(|(label, preset)| {
                let events = [RenderEvent::new(0.0, Event::NoteOn { key, vel: VELOCITY })];
                (
                    format!("note_{name}_{label}"),
                    render_to_buffer(preset, &events, NOTE_SECONDS),
                )
            })
            .collect();
        write_matched(&out, &group)?;
        if let Some(source) = library.as_ref().and_then(|l| source_note(l, key)) {
            write_matched(&out, &[(format!("note_{name}_source"), source)])?;
        }
    }

    let phrases: [(&str, Vec<RenderEvent>, f32); 2] = [
        ("demo", demo_sequence(), DEMO_DURATION_S),
        ("pedal", pedal_phrase(), 14.0),
    ];
    for (phrase, events, seconds) in &phrases {
        let group: Vec<(String, Stereo)> = presets
            .iter()
            .map(|(label, preset)| {
                (
                    format!("{phrase}_{label}"),
                    render_to_buffer(preset, events, *seconds),
                )
            })
            .collect();
        write_matched(&out, &group)?;
    }
    println!("wrote {}", out.display());
    Ok(())
}

type Stereo = (Vec<f32>, Vec<f32>);

/// A phrase for listening to the pedal: a bass octave and a chord taken under
/// the sustain pedal, the keys released while the pedal holds, a second chord
/// struck into the ringing instrument, and finally the pedal lifted so that the
/// dampers land on everything at once.
fn pedal_phrase() -> Vec<RenderEvent> {
    let mut events = vec![RenderEvent::new(0.0, Event::Pedal(PedalEvent::Sustain(1.0)))];
    let mut strike = |at: f32, keys: &[u8], vel: u8, hold: f32| {
        for &key in keys {
            events.push(RenderEvent::new(at, Event::NoteOn { key, vel }));
            events.push(RenderEvent::new(at + hold, Event::NoteOff { key }));
        }
    };
    strike(0.05, &[33, 45], 96, 0.6);
    strike(1.20, &[52, 57, 60, 64], 80, 0.5);
    strike(3.50, &[59, 62, 67, 71], 88, 0.5);
    strike(6.00, &[36, 48, 55, 60, 64, 67], 104, 0.8);
    events.push(RenderEvent::new(10.0, Event::Pedal(PedalEvent::Sustain(0.0))));
    events.sort_by(|a, b| a.time_s.total_cmp(&b.time_s));
    events
}

/// The library's own recording of `key` at the layer [`VELOCITY`] would
/// trigger, trimmed to the same length as the renders.
fn source_note(library: &SampleLibrary, key: u8) -> Option<Stereo> {
    let sample = library
        .layers(key)
        .iter()
        .find(|s| (s.lovel..=s.hivel).contains(&VELOCITY))?;
    let recording = audio::load_at(&sample.path, SAMPLE_RATE).ok()?;
    let frames = ((NOTE_SECONDS * SAMPLE_RATE as f32) as usize).min(recording.frames());
    let channel = |i: usize| recording.channels[i.min(recording.channel_count() - 1)][..frames].to_vec();
    Some((channel(0), channel(1)))
}

/// Writes a group of renders at one common gain, chosen so the loudest of them
/// starts at [`TARGET_RMS`].
///
/// One gain for the whole group, not one each: what is being compared is two
/// presets of the same instrument, and a per-file gain would flatten exactly
/// the difference in radiated level that is part of the comparison.
fn write_matched(dir: &Path, group: &[(String, Stereo)]) -> Result<(), Box<dyn std::error::Error>> {
    let reference = group
        .iter()
        .map(|(_, audio)| onset_rms(audio))
        .fold(0.0f32, f32::max);
    if reference <= 0.0 {
        return Ok(());
    }
    let mut gain = TARGET_RMS / reference;
    let peak = group
        .iter()
        .map(|(_, (l, r))| {
            l.iter()
                .chain(r.iter())
                .fold(0.0f32, |m, &v| m.max(v.abs()))
        })
        .fold(0.0f32, f32::max);
    if peak * gain > 0.98 {
        gain = 0.98 / peak;
    }
    for (name, audio) in group {
        write_wav(&dir.join(format!("{name}.wav")), audio, gain)?;
    }
    Ok(())
}

/// RMS of the first second, summed to mono: the prompt sound, which is what a
/// listener matches levels by.
fn onset_rms((left, right): &Stereo) -> f32 {
    let n = (SAMPLE_RATE as usize).min(left.len());
    if n == 0 {
        return 0.0;
    }
    let sum: f32 = (0..n).map(|i| (left[i] + right[i]).powi(2)).sum();
    (sum / n as f32).sqrt()
}

fn write_wav(path: &Path, (left, right): &Stereo, gain: f32) -> Result<(), hound::Error> {
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for (&l, &r) in left.iter().zip(right) {
        writer.write_sample(l * gain)?;
        writer.write_sample(r * gain)?;
    }
    writer.finalize()
}

fn note_name(key: u8) -> String {
    const NAMES: [&str; 12] = [
        "C", "Cs", "D", "Ds", "E", "F", "Fs", "G", "Gs", "A", "As", "B",
    ];
    format!(
        "{}{}",
        NAMES[usize::from(key) % 12],
        i32::from(key) / 12 - 1
    )
}
