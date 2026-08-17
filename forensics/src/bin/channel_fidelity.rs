//! The per-channel spectral board, on the surfaces the stereo gate already
//! uses, and the recording evidence behind the repair.
//!
//! ```text
//! cargo run --release -p forensics --bin channel_fidelity -- <section> [preset]
//! sections: board  evidence  all
//! ```
//!
//! * `board` — `realism::channel_columns` over the thirty recorded keys at v90
//!   and over the six phrases of the scoreboard's set, for the shipped preset,
//!   the capsule pair with no mode-controlled band, and the pan-pot. The
//!   pan-pot's row is the control: a pan-potted pair's two channels *are* its
//!   mono sum scaled, so every one of its numbers must be 0.00.
//! * `evidence` — what the recording's own two channels do inside the
//!   mode-controlled band, at sixth-octave resolution: `r0`, the mid-over-side
//!   ratio, and the **level difference between the two channels**. The last one
//!   is the number that decides what a "nodal line" may be built out of.

use std::path::{Path, PathBuf};

use piano_emulator::preset::{MicVoicing, Preset};
use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_emulator::types::Event;
use piano_tuner::audio::Audio;
use piano_tuner::realism::{
    self, ChannelItem, ChannelShape, Phrase, RecordedKeys, VelocityLayers, PHRASE_SET_VERSION,
};
use piano_tuner::sampler::{engine_events, SamplerEvent, SAMPLER_VERSION};
use piano_tuner::{cache, SampleLibrary, Sampler, TimedEvent, SAMPLE_RATE};

use rayon::prelude::*;

const SFZ: &str = "data/salamander/SalamanderGrandPiano-V3+20200602.sfz";
const DATA: &str = "data/salamander";
const VELOCITY: u8 = 90;
const RENDER_S: f64 = 3.0;
const PREROLL: usize = realism::STEREO_PREROLL_SAMPLES;
const PREROLL_S: f64 = PREROLL as f64 / 48_000.0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let section = args.first().cloned().unwrap_or_else(|| "all".to_string());
    let preset_path = PathBuf::from(
        args.get(1)
            .cloned()
            .unwrap_or_else(|| "presets/salamander-c5.toml".into()),
    );
    let shipped = Preset::load(&preset_path)?;
    if section == "board" || section == "all" {
        board(&shipped)?;
    }
    if section == "evidence" || section == "all" {
        evidence()?;
    }
    Ok(())
}

fn panpot(preset: &Preset) -> Preset {
    let mut p = preset.clone();
    p.voicing.mics = None;
    p
}

fn pair_only(preset: &Preset) -> Preset {
    let mut p = preset.clone();
    if let Some(mics) = preset.voicing.mics {
        p.voicing.mics = Some(MicVoicing { modal: None, ..mics });
    }
    p
}

fn render_key(preset: &Preset, key: u8) -> Audio {
    let events = [RenderEvent::new(
        PREROLL_S as f32,
        Event::NoteOn {
            key,
            vel: u16::from(VELOCITY),
        },
    )];
    let (l, r) = render_to_buffer(preset, &events, (PREROLL_S + RENDER_S) as f32);
    Audio::new(SAMPLE_RATE, vec![l[PREROLL..].to_vec(), r[PREROLL..].to_vec()])
        .expect("the engine renders stereo")
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
            .map(|c| (0..frames).map(|n| c.get(skip + n).copied().unwrap_or(0.0)).collect())
            .collect();
        Audio::new(SAMPLE_RATE, channels)
    })
}

fn render_phrase(preset: &Preset, phrase: &Phrase) -> Audio {
    let (l, r) = render_to_buffer(
        preset,
        &engine_events::to_render_events(&phrase.events),
        phrase.duration_s as f32,
    );
    Audio::new(SAMPLE_RATE, vec![l, r]).expect("stereo")
}

fn reference_phrase(
    phrase: &Phrase,
    events: &[TimedEvent],
    name: &str,
) -> Result<Audio, piano_tuner::Error> {
    let sfz = Path::new(SFZ);
    let mut key = cache::Fingerprint::new();
    key.str("realism-bench/reference")
        .u64(u64::from(SAMPLER_VERSION))
        .file(sfz)?
        .u64(u64::from(SAMPLE_RATE))
        .u64(u64::from(PHRASE_SET_VERSION))
        .str(name)
        .str(phrase.name)
        .f64(phrase.duration_s);
    let path = cache::reference_dir(Path::new(DATA))
        .join(format!("realism-{}-{name}-{}.wav", phrase.name, key.hex()));
    cache::audio(&path, || {
        let mut sampler = Sampler::new(sfz)?;
        sampler.render(events, phrase.duration_s)
    })
}

struct KeyRow {
    label: String,
    reference: ChannelShape,
    alternate: ChannelShape,
    key: u8,
}

struct PhraseRow {
    phrase: Phrase,
    reference: ChannelShape,
    alternate: ChannelShape,
}

fn board(shipped: &Preset) -> Result<(), Box<dyn std::error::Error>> {
    let sfz = Path::new(SFZ);
    let library = SampleLibrary::from_sfz(sfz)?;
    let recorded = RecordedKeys::from_library(&library)?;
    let layers = VelocityLayers::from_library(&library)?;
    let other = layers.alternate(VELOCITY);

    let keys: Vec<KeyRow> = recorded
        .keys()
        .par_iter()
        .map(|&key| -> Result<KeyRow, piano_tuner::Error> {
            Ok(KeyRow {
                label: realism::note_name(key),
                reference: realism::channel_shape_of(&reference_key(key, VELOCITY)?)?,
                alternate: realism::channel_shape_of(&reference_key(key, other)?)?,
                key,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let phrases: Vec<PhraseRow> = realism::phrase_set()
        .into_par_iter()
        .map(|phrase| -> Result<PhraseRow, piano_tuner::Error> {
            let reference = reference_phrase(&phrase, &phrase.events, "reference")?;
            let shifted = layers.shift(&phrase.events);
            let alternate = reference_phrase(&phrase, &shifted, "alt-layer")?;
            Ok(PhraseRow {
                reference: realism::channel_shape_of(&reference)?,
                alternate: realism::channel_shape_of(&alternate)?,
                phrase,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    println!(
        "per-channel spectral fidelity: {} recorded keys at v{VELOCITY} (floor layer v{other}) \
and {} phrases\n",
        keys.len(),
        phrases.len()
    );

    for (name, preset) in [
        ("the shipped preset (pair + mode-controlled band)", shipped.clone()),
        ("the capsule pair alone", pair_only(shipped)),
        ("the pan-pot — the control, every number must be 0.00", panpot(shipped)),
    ] {
        let items: Vec<ChannelItem> = keys
            .par_iter()
            .map(|row| ChannelItem {
                label: row.label.clone(),
                engine: realism::channel_shape_of(&render_key(&preset, row.key))
                    .expect("stereo"),
                reference: row.reference.clone(),
                alternate: row.alternate.clone(),
            })
            .collect();
        let columns = realism::channel_columns(&items);
        println!(
            "### {name} — {} recorded keys ({} red, worst band {:.2} dB over its bar)\n{}",
            keys.len(),
            columns.iter().filter(|c| !c.pass).count(),
            columns
                .iter()
                .filter(|c| c.bar.is_finite() && c.bar > 0.0)
                .map(|c| c.error - c.bar)
                .fold(f64::NEG_INFINITY, f64::max),
            realism::channel_report(&columns)
        );

        let items: Vec<ChannelItem> = phrases
            .par_iter()
            .map(|row| ChannelItem {
                label: row.phrase.name.to_string(),
                engine: realism::channel_shape_of(&render_phrase(&preset, &row.phrase))
                    .expect("stereo"),
                reference: row.reference.clone(),
                alternate: row.alternate.clone(),
            })
            .collect();
        let columns = realism::channel_columns(&items);
        println!(
            "### {name} — {} phrases ({} red)\n{}",
            phrases.len(),
            columns.iter().filter(|c| !c.pass).count(),
            realism::channel_report(&columns)
        );
    }
    Ok(())
}

/// What the recording's own two channels do where the engine puts its
/// mode-controlled band.
///
/// The question this answers: a nodal line straddled by two capsules is an
/// *in-phase inversion*, `L = +a·x` and `R = −a·x`. That is an anti-correlation
/// **and** — the part nothing in this repository had measured — a `|L| = |R|`.
/// The engine's lobe realises the inversion as `L = m(1+g)`, `R = m(1−g)`, which
/// is an anti-correlation *and* a level difference of `20 log10 |(1+g)/(1−g)|`,
/// which for the shipped `g` reaches thirty decibels. So: does the recording,
/// in the band where its `r0` is negative, show a level difference between its
/// two channels, or does it not?
fn evidence() -> Result<(), Box<dyn std::error::Error>> {
    let sfz = Path::new(SFZ);
    let library = SampleLibrary::from_sfz(sfz)?;
    let recorded = RecordedKeys::from_library(&library)?;
    let takes: Vec<Vec<realism::StereoProfilePoint>> = recorded
        .keys()
        .par_iter()
        .map(|&key| -> Result<Vec<realism::StereoProfilePoint>, piano_tuner::Error> {
            let audio = reference_key(key, VELOCITY)?;
            realism::stereo_profile_of(&audio)
        })
        .collect::<Result<Vec<_>, _>>()?;
    // The same keys' per-channel level difference, band by band, on the same
    // sixth-octave grid — read straight off the profile's own energies.
    let balance: Vec<Vec<f64>> = recorded
        .keys()
        .par_iter()
        .map(|&key| -> Result<Vec<f64>, piano_tuner::Error> {
            let audio = reference_key(key, VELOCITY)?;
            Ok(sixth_octave_balance(&audio))
        })
        .collect::<Result<Vec<_>, _>>()?;

    println!("\nthe recording's own two channels, sixth-octave, median over {} recorded keys at v{VELOCITY}", takes.len());
    println!("| Hz | r0 | mid/side dB | \\|L\\|-\\|R\\| dB | \\|L\\|-\\|R\\| p90 |");
    println!("|---:|---:|---:|---:|---:|");
    let points = takes[0].len();
    for i in 0..points {
        let hz = takes[0][i].hz;
        if !(100.0..900.0).contains(&hz) {
            continue;
        }
        let mut r: Vec<f64> = takes.iter().map(|t| t[i].r0).filter(|v| v.is_finite()).collect();
        let mut ms: Vec<f64> = takes
            .iter()
            .map(|t| t[i].mid_side_db)
            .filter(|v| v.is_finite())
            .collect();
        let mut b: Vec<f64> = balance
            .iter()
            .filter_map(|t| t.get(i).copied())
            .filter(|v| v.is_finite())
            .map(f64::abs)
            .collect();
        r.sort_by(f64::total_cmp);
        ms.sort_by(f64::total_cmp);
        b.sort_by(f64::total_cmp);
        if r.is_empty() || b.is_empty() {
            continue;
        }
        println!(
            "| {hz:.0} | {:+.3} | {:+.2} | {:.2} | {:.2} |",
            r[r.len() / 2],
            ms[ms.len() / 2],
            b[b.len() / 2],
            b[(b.len() * 9 / 10).min(b.len() - 1)]
        );
    }
    Ok(())
}

/// `|L| − |R|` per sixth-octave band, dB, on the same grid
/// `realism::stereo_profile` uses.
fn sixth_octave_balance(audio: &Audio) -> Vec<f64> {
    use rustfft::num_complex::Complex32;
    use rustfft::FftPlanner;
    let (left, right) = (&audio.channels[0], &audio.channels[1]);
    let n = left.len().max(right.len()).next_power_of_two();
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n);
    let spectrum = |s: &[f32]| {
        let mut buf: Vec<Complex32> = (0..n)
            .map(|i| Complex32::new(s.get(i).copied().unwrap_or(0.0), 0.0))
            .collect();
        fft.process(&mut buf);
        buf
    };
    let (a, b) = (spectrum(left), spectrum(right));
    let sr = f64::from(audio.sample_rate);
    let ratio = 2.0f64.powf(1.0 / realism::STEREO_PROFILE_PER_OCTAVE as f64);
    let half = ratio.sqrt();
    let bin = |hz: f64| (hz * n as f64 / sr).round() as usize;
    let mut out = Vec::new();
    let mut hz = realism::STEREO_PROFILE_RANGE_HZ.0;
    while hz <= realism::STEREO_PROFILE_RANGE_HZ.1 {
        let (blo, bhi) = (bin(hz / half).max(1), bin(hz * half).min(n / 2));
        if bhi >= blo {
            let mut ea = 0.0f64;
            let mut eb = 0.0f64;
            for j in blo..=bhi {
                ea += f64::from(a[j].norm_sqr());
                eb += f64::from(b[j].norm_sqr());
            }
            out.push(10.0 * (ea.max(1e-300) / eb.max(1e-300)).log10());
        }
        hz *= ratio;
    }
    out
}
