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
//! | `*_engine.wav` | the shipped preset. |
//! | `*_m17.wav` | **written only when the shipped preset has no `[voicing.mics.modal]`**: today's pair with the band item 418 fitted and item 449 shipped put back, and nothing else moved. The instrument item 451's corner stage exists to retire. |
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
    // **The instrument the corner stage exists to retire, kept on the page as
    // a take** (`DECISIONS.md` 451). Once the shipped preset carries no
    // `[voicing.mics.modal]`, `pair_only` *is* `shipped` and the A/B that
    // matters is no longer "the pair with and without a band" but "the band
    // that shipped through item 449 against the one that does not have it".
    // The band is `melody::M17_MODAL_BAND` — the same constant the melody
    // board's two falsifications install — so this take is built from whatever
    // preset ships rather than from a file somebody has to keep in step.
    let m17 = (shipped.voicing.mics.and_then(|m| m.modal).is_none()).then(|| {
        let mut p = shipped.clone();
        let (lo_hz, hi_hz, lift) = melody::M17_MODAL_BAND;
        p.voicing.mics = Some(piano_emulator::preset::MicVoicing {
            modal: Some(piano_emulator::preset::ModalBand { lo_hz, hi_hz, lift }),
            ..mics
        });
        p
    });

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
                title: if m17.is_some() {
                    "the shipped preset: the pair, and no mode-controlled band"
                } else {
                    "the shipped preset: the pair and the mode-controlled band"
                },
                audio: render_engine(&shipped, phrase),
            },
        ];
        if let Some(m17) = &m17 {
            takes.push(Take {
                name: "m17",
                title: "the band item 418 fitted and item 449 shipped, put back on \
today's pair — the instrument this milestone replaced",
                audio: render_engine(m17, phrase),
            });
        }
        // One gain for every engine take — they share a mono sum, so a
        // per-take normalisation would be measuring the side signal and
        // removing it.
        let engine_rms = rms(&takes[2].audio.mono());
        let reference = render_reference(&data, &sfz, phrase)?;
        let reference_rms = rms(&reference.mono());
        if engine_rms <= 0.0 || reference_rms <= 0.0 {
            return Err(format!("{} rendered silence", phrase.name).into());
        }
        let mut gains = vec![f64::from(TARGET_RMS) / engine_rms; takes.len()];
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

    // **What this board is an A/B of changes with the band** (`DECISIONS.md`
    // 451). With no `[voicing.mics.modal]` in the shipped preset the `_pair`
    // take and the `_engine` take are the same instrument, and the comparison
    // a listener wants is against the band that used to be there — which is
    // what `_m17` is.
    let milestone = if m17.is_some() {
        "\n## What removing the band changed, and what these files are\n\n\
         **The shipped preset carries no `[voicing.mics.modal]`.** So `_engine` and\n\
         `_pair` are the same instrument, sample for sample, and the fourth engine take\n\
         `_m17` is the band that shipped from item 418 to item 449 — `174.3-456.5 Hz`\n\
         at a lift of `0.99` — put back on today's pair and on nothing else. That is\n\
         the A/B this board now exists for, and it is the one the listener made: play\n\
         `ode_soprano_m17.wav` and `ode_soprano_engine.wav` one after the other.\n\n\
         What to listen for is **not** a width difference. The band's two edges bracket\n\
         every fundamental of this line — 261.6 to 392.0 Hz inside 174.3-456.5 — and\n\
         **none** of those notes' second partials, which start at 523 Hz. `L = m(1+B)`\n\
         and `R = m(1−B)`, so wherever `Re B > 0` the band is not a widener but a\n\
         **pan**: on `_m17` each note's *pitch* is pulled towards one loudspeaker while\n\
         that same note's *colour* stays where the pair put it, and the ear is handed a\n\
         note arriving from two places at once. Measured on this very render, F4's\n\
         fundamental sits **21.8 dB** away from its own second, third and fourth\n\
         partials in the image, and G4's **16.8**, where the recording's F4 sits 1.4 dB\n\
         from its own — `renders/melody/MELODY.md`'s `splitting` column, item 451. A\n\
         listener hears that as the note coming apart, part of what was reported —\n\
         the C4 percept itself traced further, to items 452-453's mono-domain level\n\
         and decay defects — and the preset's own `f0` table is right to 0.3 cents.\n\n\
         The mono fold-down of all four engine takes is identical, so none of this is a\n\
         level and none of it is audible in mono.\n\n"
    } else {
        ""
    };
    let report = format!(
        "# The stereo image, heard\n\n\
         *`piano-tuner stereo` — `PHYSICS.md` §8, `DECISIONS.md` 313-317, 346-379,\n\
         392-425 and 446-453.*\n{milestone}\n\
         Every other board in `renders/` scores a **mono sum**, on purpose: a stereo\n\
         distance would mostly measure somebody else's microphones. This one is the\n\
         exception, and it exists because the largest single difference ever measured\n\
         between this engine and the recording was invisible to all of them — the\n\
         recording's two channels correlate at **+0.95 below 125 Hz and about zero\n\
         above it**, and the engine's did exactly the reverse (item 314).\n\n\
         Two pieces of music. The engine takes share **one\n\
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
         The **fourth** table under each piece is `realism::channel_shape`, and it is\n\
         there because the three above it are not enough: a correlation is normalised\n\
         per channel and a mid-over-side ratio is a sum, so all three are blind to\n\
         what one loudspeaker's own spectrum does. A stage can leave the mono sum\n\
         bit-identical, match the recording's coherence band for band, and still put\n\
         one speaker 9 dB up and the other 21 dB *down* at a single note's\n\
         fundamental — which is what the mode-controlled band did, and what a listener\n\
         reported three separate ways while every gate stayed green (item 392).\n\n\
         ## What item 418's rail changed, complaint by complaint\n\n\
         `[voicing.mics.modal].lift` is railed at **one** since item 418 and the pair is\n\
         refitted under it, so these takes are not the ones items 392-394 were written\n\
         on. The lobe is `L = m(1 + B)`, `R = m(1 − B)` where `B` is the band's\n\
         **complex** response — items 392-418 wrote it as a real `g` and item 423 is the\n\
         correction, which is worth up to 13 dB and changes two of the three findings.\n\
         What the rail makes **unbuildable** is `|B| > 1`: no channel inversion in\n\
         either loudspeaker (on the pre-418 preset the left was inverted over\n\
         232.0-272.3 Hz and the right over 316.0-357.4), no pitch-dependent flip of\n\
         which one carries it, and a pair-energy ceiling of `10 log10(1 + |B|²)` =\n\
         **+3.01 dB** against the old lobe's realised **+6.18**. What it does **not**\n\
         remove is per-channel level loss, and it **deepens** it: `1 ± B` is smallest\n\
         where `|B|` is nearest *one*, so a lift of 0.99 across a wide band is a worse\n\
         null than a lift of 2.12 across a narrow one. The old lobe's deepest\n\
         one-channel loss was **−20.5 dB at 349.8 Hz**, in the right channel; the\n\
         shipped one reaches **−33.1 dB at 221.4 Hz — in the *left* channel, at A3's\n\
         own fundamental** — with either channel more than 10 dB down over 0.286\n\
         octaves against the old lobe's 0.105. There are **no exact nulls and never\n\
         were**: at `|B| = 1` the loss is `2|sin(arg B / 2)|`, so item 392's \"unity\n\
         crossings at 213.0 and 359.6 Hz\" null nothing.\n\n\
         Re-measured on these renders' own instrument with\n\
         `forensics/src/bin/stereo_channels.rs`, against item 392's numbers for the same\n\
         statistics (item 425):\n\n\
         * **(a) \"the C4 of the melody line stands out\" — resolved.** The old band's\n\
           gain peaked at 261.7 Hz, half a cent from C4, and pair-over-mono energy read\n\
           **C4 +6.42 dB**, the loudest note of the line by 0.9 dB, with adjacent\n\
           semitones 6.7 dB apart. The shipped band's own prediction is flat across the\n\
           line at **+2.45 to +2.94 dB**, and measured through the whole chain C4 reads\n\
           **+4.07** with E4 above it at **+4.36**; the recording's own C4 sits +1.72\n\
           above its line's trend. No note is singled out any more.\n\n\
         * **(b) \"the hammer noise is too loud\" — the inversion is gone, a loss of\n\
           the same order is not.** The complaint was a denominator: the burst was\n\
           untouched and the note's own *fundamental* was leaving one channel, so what\n\
           was left there read as noise. Measured against a pan-pot at each note's\n\
           fundamental, **F#4 goes R −18.33 → −6.54 dB** and **F4 R −9.61 → −10.09**:\n\
           the worst per-channel loss on the line improves by 8 dB and moves note. The\n\
           lobe's own notch moves the *other* way — deepest right-channel loss −20.5 dB\n\
           at 349.8 Hz before and −20.2 at 373.9 after, the same depth one semitone up —\n\
           because in a channel the lobe has nulled what is left is the pair's\n\
           **geometric** side, and that is what sets the depth and the frequency a\n\
           listener actually gets. **What is structurally gone is the sign**: `L +7.17 /\n\
           R −9.61` at F4 against `L +0.70 / R +9.43` at C4 was one loudspeaker carrying\n\
           the note inverted against the other, with the flip landing in the middle of\n\
           the tune, and no preset this schema accepts can do that now. **What is not\n\
           gone is the loss**, whose deepest point under the rail has moved to the left\n\
           channel at 221.4 Hz (item 423).\n\n\
         * **(c) \"the reference chords have more brilliance\" — the right channel's\n\
           share is retired, the left channel's is where it was.** Brilliance as a\n\
           2-6 kHz over 250-500 Hz tilt, engine minus reference: `dmono` is **−8.15 dB\n\
           and identical on all three engine takes**, which is the control that says the\n\
           mono deficit is pre-existing and the microphone section is innocent of it. On\n\
           top of that, the lobe's own share — engine minus the pair-only take, per\n\
           channel — moves **L −4.95 → −4.23 dB** and **R −2.64 → +0.18**. The larger\n\
           one, and the one the complaint was about, is 0.7 dB better and still there.\n\n\
         So one of the three complaints is gone, one has lost its mechanism but not its\n\
         level, and one is half gone. The residual is what the two documented reds\n\
         measure (`CONTEXT.md`), and item 421 is the arithmetic that says a bar may not\n\
         move to close them.\n\n\
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

    // **What each loudspeaker plays**, which none of the three tables above
    // can say: they are a correlation, a lag and a sum, and all three are
    // blind to one channel's spectrum. See `realism::ChannelBand`.
    let _ = write!(
        s,
        "\nPer-channel spectrum, dB — each channel's share of its own broadband \
         energy minus the same take's mono share, `L / R`. **A pan-pot of one note \
         reads 0.0 / 0.0**; anything else is what the stereo stage did to each \
         loudspeaker, and it is the column three listening complaints turned out \
         to live in (`DECISIONS.md` 392-394):\n\n\
         | take | 63-125 | 125-250 | 250-500 | 500-2k | 2k-6k | 6k-12k |\n\
         |---|---:|---:|---:|---:|---:|---:|\n"
    );
    let shapes: Vec<realism::ChannelShape> = takes
        .iter()
        .map(|t| realism::channel_shape_of(&t.audio).expect("two channels"))
        .collect();
    for (take, shape) in takes.iter().zip(&shapes) {
        let _ = write!(s, "| {} |", take.title);
        for band in &shape.bands {
            let _ = write!(s, " {:+.2} / {:+.2} |", band.dev_left_db, band.dev_right_db);
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
