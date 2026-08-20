//! The stereo gate: the engine's interchannel image against the recording's.
//!
//! `DECISIONS.md` 313-317 is the chain experiment, and its largest measured
//! result is the one nothing on any scoreboard could see. `chain::stereo_signature`
//! read the recording's two channels and found **+0.945 correlation at lag zero
//! below 125 Hz**, falling to about nothing through the mid and treble, with a
//! peak |r| of 0.57-0.65 everywhere at lags of −0.23 to +1.98 ms. That is a
//! spaced pair of microphones — an AKG pair about 12 cm above the strings — two
//! capsules well inside a wavelength of each other in the bass seeing one
//! wavefront, and seeing the same sound about 60 % coherent a fraction of a
//! millisecond apart above it. The engine is **inverted in every band**:
//! −0.577 in the bass, where `soundboard`'s FDN sends two orthogonal-sign taps
//! to the two channels, rising to +0.964 at 6-12 kHz, which is what
//! `soundboard::pan_for_key` is — one mono voice scaled into two channels, a
//! pan-pot. Every metric in `REALISM.md` is computed on the mono sum and is
//! blind to all of it.
//!
//! Item 317 (a) is the instruction: **give the loss a stereo term first**,
//! because a stage built to fix something nothing scores is a stage nobody can
//! regress. This file is that term with a bar under it.
//!
//! # This gate is green, and the shape of how it got there is the record
//!
//! It is the pattern `tests/melody.rs` established and `DECISIONS.md` 298 and
//! 330 both used: a gate written from a measurement, failing on the instrument
//! as it stands, and then closed by a mechanism rather than by a tolerance.
//! Four milestones, and each one is visible in a different set of bands.
//!
//! **All six red** when the gate was written (`DECISIONS.md` 346-350): the
//! numbers in the paragraph above, with the engine on the wrong side of zero in
//! every band and a bass peak |r| railing at −4.03 ms, which is what "the FDN
//! decorrelates at no particular delay" looks like.
//!
//! **Two red** once `PHYSICS.md` §8 was built and then fitted — `[voicing.mics]`,
//! two virtual capsules over the string band with a per-source delay and gain
//! and a frequency-dependent coherence on the board's diffuse field
//! (`DECISIONS.md` 351-358), with its five numbers inverted out of the
//! recording's own interchannel delays rather than swept (359-367). That
//! **inverted the inversion**: 63-125 Hz went −0.577 → +0.961 against the
//! recording's +0.953, and 6-12 kHz +0.912 → +0.089 against +0.050.
//!
//! **None red at the window this file was reading, and three red at the window
//! it should have been reading**, with the board's **mode-controlled band**
//! (`DECISIONS.md` 368-377). What was left was 125-500 Hz, and item 357 was
//! right that no two-point geometry could close it: the recording reads +0.953 below 125 Hz
//! and −0.115 one octave above, and `sin(kd)/kd`, a pure interchannel delay, or
//! any mixture of them cannot fall from +0.95 through zero across one octave.
//! What closed it was measuring the recording at a resolution that shows a
//! *shape* instead of six numbers (`piano-tuner mics --stage profile`): its
//! sixth-octave interchannel correlation is `+0.94` at 127 Hz, `+0.07` at 160
//! and **`−0.53` at 180**, holds negative through 254, and is inside ±0.2 of
//! zero everywhere above 500 — three regimes of a *plate*, not one curve of a
//! microphone pair, and repeated to within 0.1 by the same keys' other velocity
//! layer. `soundboard::ModalLobe` is those three regimes, and
//! `the_capsule_pair_without_the_mode_controlled_band_fails_in_the_middle`
//! below is the control that says which two bands it is carrying.
//!
//! **None red, at a window that opens where the note does** (`DECISIONS.md`
//! 378-379). The milestone above was measured through a window that began
//! **96 samples after the strike**: this file asked for 0.05 s of preroll,
//! which is 2400 samples, and the engine's block is 128, so the note began at
//! 2304 and the window at 2400. Two milliseconds of every note were outside it
//! and it opened in the middle of a signal — and the verdict turned on that.
//! Struck at the head of a block the same instrument read `+0.936 / +0.204 /
//! +0.218` in the first three bands against the recording's `+0.953 / −0.115 /
//! −0.226`, which is **three red**, not none. Item 378 is the window and the
//! control that licenses it; item 379 is what the honest window then showed,
//! which is that the mode-controlled band was built out of the wrong signal.
//! It band-limited the *difference* of the board's two decorrelated taps and
//! scaled it up, and a difference cannot be a nodal line — two capsules
//! straddling one hear the same field with opposite signs — nor can it act
//! during the strike, because the FDN's shortest line is 149 samples and its
//! difference is exactly zero for the first 3.1 ms of every note. Measured in
//! 10 ms frames, C5's first frame read `+9.9 dB` mid over side in 125-250 Hz
//! where the recording's reads `−1.6 dB`. `soundboard::ModalLobe` now adds an
//! anti-phase copy of the **sum**, on the direct path as well as the board's,
//! and `[voicing.mics]` was refitted at the aligned window by
//! `piano-tuner mics --stage band` — a stage that moves the band and the two
//! trims together, because since the change they build one side signal and are
//! no longer separable.
//!
//! **And then a fifth board, because none of the four columns above is a
//! spectrum** (`DECISIONS.md` 392-395). `r0`, the peak |r|, its lag and the
//! mid-over-side ratio are the whole of what this file scored, and all four are
//! blind to what **one channel** does on its own: correlation is normalised per
//! channel by construction, and mid-over-side is a sum. So the instrument the
//! milestone above shipped could leave the mono fold-down bit-identical, pass
//! every band of the coherence table, and still put one loudspeaker **9 dB up
//! and the other 21 dB down at a single note's fundamental** — which is what
//! `soundboard::ModalLobe`'s in-phase inversion did at its unity-gain crossings
//! (213.0 and 359.6 Hz), and which a listener reported three separate ways
//! while 696 tests stayed green. `realism::channel_columns` is that fifth
//! board, `each_loudspeaker_has_the_recordings_spectrum_where_the_mic_pair_acts`
//! is its gate, and `soundboard::MIC_MODAL_DIFFUSION` is what closed it.
//!
//! It is still **not** a room: §9's reverberant field is refused by measurement
//! in item 315 and stays out of scope. What was added is a property of the
//! board — where its modes begin to put a nodal line between two capsules 12 cm
//! apart, and where modal overlap stops there being a sign to see.
//!
//! # Where the window starts, and why that is not a free parameter
//!
//! Every render this file scores puts the strike on the **first sample** of the
//! window: `PREROLL` is `realism::STEREO_PREROLL_SAMPLES`, a whole number of
//! engine blocks, asserted at compile time here and in `tools::mics`, with a
//! run-time assertion that nothing sounds before it. Two things go wrong
//! otherwise and only one of them is obvious. A window that opens *inside* a
//! note opens with a step, and a step is broadband — that alone took the
//! engine's 6-12 kHz column from readable on 15 of these keys to readable on
//! 29. And a window that opens *after* the strike is missing the strike, which
//! in 125-500 Hz is most of what a treble key has here.
//! `the_recordings_image_does_not_move_when_the_window_does` is the control:
//! the *recording* is unmoved by the same three placements, which is what
//! licenses reading it from an onset detector while the engine is read from the
//! strike itself — and it prints the engine's own figure beside it, which is
//! not the same number and is item 379's open half.
//!
//! Writing the gate before the mechanism is the order `DECISIONS.md` 317 (a)
//! asks for: a stage built to fix something nothing scores is a stage nobody
//! can regress.
//!
//! # The material, and where the bar comes from
//!
//! The 30 keys the Salamander library actually **recorded** (`DECISIONS.md`
//! 328), struck alone at velocity 90. Solo notes rather than the scoreboard's
//! phrases, for two reasons: a phrase is a mixture of keys and the geometry
//! this is about is per-key, and a recorded key has a **second recording of
//! itself** — its neighbouring velocity layer — which is what the floor is made
//! of. Transposed keys are not used at all: a resampled take keeps its mic
//! image, so its correlation is a real measurement, but its *velocity layers*
//! are two transpositions of one take rather than two takes, and a floor built
//! from those would be a measurement of the resampler (the same argument item
//! 328 makes about `match` and item 331 makes about the melody's bar).
//!
//! The score per band is `|engine r@0 − reference r@0|` on the medians over the
//! keys, and the bar is `max(floor, scatter/sqrt(n)) · realism::STEREO_ALLOWANCE`
//! — the same median taken on the *second take*, against the precision with
//! which 30 keys pin a median that moves by `scatter` across them. Both are the
//! recording disagreeing with itself; neither is anything the engine did. The
//! pooled `scatter` is deliberately **not** the bar: a recording's r@0 moves
//! across the compass because the keys sit in different places relative to the
//! microphones, and that motion is a thing the engine is meant to reproduce
//! rather than to be excused from. The per-key distance is reported beside the
//! pooled one, so a model that fixes a band's median without fixing its image
//! is visible as such.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use piano_emulator::preset::Preset;
use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_emulator::types::{Event, BLOCK};
use piano_tuner::audio::Audio;
use piano_tuner::cache;
use piano_tuner::realism::{
    self, RecordedKeys, StereoColumn, StereoImage, StereoItem, VelocityLayers,
};
use piano_tuner::sampler::{Sampler, SamplerEvent, TimedEvent, SAMPLER_VERSION};
use piano_tuner::{SampleLibrary, SAMPLE_RATE};

/// The velocity every key is struck at: the middle layer, the one the fits, the
/// compass and the motion columns all use.
const VELOCITY: u8 = 90;

/// Seconds of note the image is read over. Long enough that the bass bands hold
/// several cycles of the lowest key's fundamental and short enough that the top
/// octave has not decayed into the recording's noise.
const RENDER_S: f64 = 3.0;

/// Silence before the strike, in samples: [`realism::STEREO_PREROLL_SAMPLES`],
/// which carries the whole argument for why this is a whole number of blocks
/// and what it cost when it was not.
const PREROLL: usize = realism::STEREO_PREROLL_SAMPLES;

/// The window has to begin **at** the strike, and an event takes effect at the
/// head of the block that contains it, so a preroll that is not a whole number
/// of blocks starts the window inside the note. Checked here rather than
/// trusted: it is one `const` away from being wrong again.
const _: () = assert!(
    PREROLL % BLOCK == 0,
    "the preroll must be a whole number of engine blocks or the window starts inside the note"
);

/// The same number in seconds, for the event list.
const PREROLL_S: f64 = PREROLL as f64 / SAMPLE_RATE as f64;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn sfz() -> Option<PathBuf> {
    let path = repo()
        .join("data/salamander")
        .join("SalamanderGrandPiano-V3+20200602.sfz");
    path.exists().then_some(path)
}

fn shipped_preset() -> Preset {
    Preset::load(&repo().join("presets/salamander-c5.toml")).expect("the measured preset loads")
}

// ---------------------------------------------------------------------------
// Rendering one key, three ways
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
    assert_eq!(
        events[0].frame(),
        PREROLL,
        "the strike must land on the first sample of the window"
    );
    assert!(
        left[..PREROLL].iter().all(|&x| x == 0.0),
        "there is sound before the strike, so the window does not start at it"
    );
    Audio::new(
        SAMPLE_RATE,
        vec![left[PREROLL..].to_vec(), right[PREROLL..].to_vec()],
    )
    .expect("the engine renders stereo")
}

/// The recording of the same note at some velocity, trimmed to its own onset so
/// that both sides read the same part of the note, and cached to disk the way
/// every other reference render in this repository is: it is a function of the
/// sampler, the library and the key, none of which move when the engine does.
fn render_reference(sfz: &Path, key: u8, velocity: u8) -> Audio {
    let mut print = cache::Fingerprint::new();
    print
        .str("tests/stereo/reference")
        .u64(u64::from(SAMPLER_VERSION))
        .file(sfz)
        .expect("the sfz is readable")
        .u64(u64::from(SAMPLE_RATE))
        .u64(u64::from(key))
        .u64(u64::from(velocity))
        .f64(RENDER_S);
    let dir = cache::reference_dir(&repo().join("data/salamander"));
    let path = dir.join(format!(
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
            .map(|c| {
                (0..frames)
                    .map(|n| c.get(skip + n).copied().unwrap_or(0.0))
                    .collect()
            })
            .collect();
        Audio::new(SAMPLE_RATE, channels)
    })
    .expect("the recording of a key the library recorded")
}

/// The same signal made mono and put back into two channels: a **pan-pot of the
/// recording**, which is the engine's own construction applied to the piano's
/// own sound. Its correlation is +1 in every band by construction.
fn pan_potted(audio: &Audio) -> Audio {
    let mono = audio.mono();
    Audio::new(audio.sample_rate, vec![mono.clone(), mono]).expect("two channels")
}

// ---------------------------------------------------------------------------
// The measurement, once for the whole file
// ---------------------------------------------------------------------------

struct Measured {
    /// One item per recorded key: engine, recording, and the recording's other
    /// velocity layer.
    items: Vec<StereoItem>,
    /// The same renders on the per-channel board (`DECISIONS.md` 393).
    channels: Vec<realism::ChannelItem>,
    /// The same items with the *recording* on the engine's side: the control
    /// that says the columns do not red out on a signal that is right.
    itself: Vec<StereoItem>,
    /// The same items with a **pan-potted copy of the recording** on the
    /// engine's side: the control that says the columns catch the defect they
    /// name, on the real material, with the engine out of the picture.
    panned: Vec<StereoItem>,
    panned_channels: Vec<realism::ChannelItem>,
    /// The shipped preset with `[voicing.mics.modal]` deleted — the instrument
    /// as `DECISIONS.md` 359-367 left it, a capsule pair and nothing else.
    /// The control under the whole of 369-372.
    without_modal: Vec<StereoItem>,
    without_modal_channels: Vec<realism::ChannelItem>,
    /// **The same pair with a mode-controlled band on it**, whichever way the
    /// shipped preset happens to fall (`DECISIONS.md` 452): the shipped band
    /// when there is one, and `melody::M17_MODAL_BAND` — the band item 418
    /// fitted and item 449 shipped — when there is not.
    ///
    /// The control `the_per_channel_column_sees_the_mode_controlled_band_and_
    /// only_it` is a statement about the *column*, not about the preset: it
    /// says that this board moves when a band is added and moves it in the
    /// band's own bands. Once nothing ships a band, "the shipped preset with
    /// the band deleted" is the shipped preset, the difference is zero, and the
    /// control would assert that a column measuring the band correctly is
    /// broken. So the control is taken between *these two* instead, and it
    /// asks the same question either way round.
    with_modal_channels: Vec<realism::ChannelItem>,
}

/// The recording's side of the comparison: one row per recorded key, the take
/// at [`VELOCITY`] and the take at the neighbouring layer, both as images.
///
/// It is a function of the library alone, so it is measured once for the whole
/// file and every engine the file renders is scored against the same numbers.
struct Reference {
    key: u8,
    label: String,
    reference: StereoImage,
    alternate: StereoImage,
    /// The recording's own mono sum put back into two channels: the pan-pot
    /// control's engine side.
    panned: StereoImage,
    /// The same three takes as **per-channel spectral shapes**
    /// (`realism::channel_shape`), which is the board `DECISIONS.md` 393 added.
    reference_channels: realism::ChannelShape,
    alternate_channels: realism::ChannelShape,
    panned_channels: realism::ChannelShape,
}

fn references() -> Option<&'static Vec<Reference>> {
    static ONCE: OnceLock<Option<Vec<Reference>>> = OnceLock::new();
    ONCE.get_or_init(|| {
        let sfz = sfz()?;
        let library = SampleLibrary::from_sfz(&sfz).expect("the library reads");
        let recorded = RecordedKeys::from_library(&library).expect("the library records keys");
        let layers = VelocityLayers::from_library(&library).expect("it has velocity layers");
        let other = layers.alternate(VELOCITY);
        assert_ne!(
            other, VELOCITY,
            "the floor needs a second layer to be a second recording"
        );
        let image = |a: &Audio| realism::stereo_image_of(a).expect("two channels");
        let shape = |a: &Audio| realism::channel_shape_of(a).expect("two channels");
        Some(
            recorded
                .keys()
                .iter()
                .map(|&key| {
                    let reference = render_reference(&sfz, key, VELOCITY);
                    let alternate = render_reference(&sfz, key, other);
                    let panned = pan_potted(&reference);
                    Reference {
                        key,
                        label: realism::note_name(key),
                        panned: image(&panned),
                        panned_channels: shape(&panned),
                        reference_channels: shape(&reference),
                        alternate_channels: shape(&alternate),
                        reference: image(&reference),
                        alternate: image(&alternate),
                    }
                })
                .collect(),
        )
    })
    .as_ref()
}

/// Scores one preset's renders against [`references`], on both boards, off one
/// set of renders.
fn score(preset: &Preset) -> (Vec<StereoItem>, Vec<realism::ChannelItem>) {
    let rows = references().expect("a library");
    let rendered: Vec<Audio> = rows.iter().map(|r| render_engine(preset, r.key)).collect();
    (
        rows.iter()
            .zip(&rendered)
            .map(|(r, audio)| StereoItem {
                label: r.label.clone(),
                engine: realism::stereo_image_of(audio).expect("two channels"),
                reference: r.reference.clone(),
                alternate: r.alternate.clone(),
            })
            .collect(),
        rows.iter()
            .zip(&rendered)
            .map(|(r, audio)| realism::ChannelItem {
                label: r.label.clone(),
                engine: realism::channel_shape_of(audio).expect("two channels"),
                reference: r.reference_channels.clone(),
                alternate: r.alternate_channels.clone(),
            })
            .collect(),
    )
}

fn measured() -> Option<&'static Measured> {
    static ONCE: OnceLock<Option<Measured>> = OnceLock::new();
    ONCE.get_or_init(|| {
        let rows = references()?;
        let preset = shipped_preset();
        let (items, channels) = score(&preset);
        let side = |pick: fn(&Reference) -> &StereoImage| -> Vec<StereoItem> {
            rows.iter()
                .map(|r| StereoItem {
                    label: r.label.clone(),
                    engine: pick(r).clone(),
                    reference: r.reference.clone(),
                    alternate: r.alternate.clone(),
                })
                .collect()
        };
        let mut bare = preset.clone();
        bare.voicing.mics = bare.voicing.mics.map(|m| piano_emulator::preset::MicVoicing {
            modal: None,
            ..m
        });
        let channel_side = |pick: fn(&Reference) -> &realism::ChannelShape| {
            rows.iter()
                .map(|r| realism::ChannelItem {
                    label: r.label.clone(),
                    engine: pick(r).clone(),
                    reference: r.reference_channels.clone(),
                    alternate: r.alternate_channels.clone(),
                })
                .collect::<Vec<realism::ChannelItem>>()
        };
        let (without_modal, without_modal_channels) = score(&bare);
        // One more render set only when the shipped preset has no band of its
        // own; when it has one, the banded instrument is the shipped one and is
        // already measured.
        let shipped_band = preset.voicing.mics.and_then(|m| m.modal);
        let with_modal_channels = match shipped_band {
            Some(_) => channels.clone(),
            None => {
                let mut banded = preset.clone();
                let (lo_hz, hi_hz, lift) = piano_tuner::estimate::melody::M17_MODAL_BAND;
                banded.voicing.mics =
                    banded.voicing.mics.map(|m| piano_emulator::preset::MicVoicing {
                        modal: Some(piano_emulator::preset::ModalBand { lo_hz, hi_hz, lift }),
                        ..m
                    });
                score(&banded).1
            }
        };
        Some(Measured {
            items,
            channels,
            itself: side(|r| &r.reference),
            panned: side(|r| &r.panned),
            panned_channels: channel_side(|r| &r.panned_channels),
            without_modal,
            without_modal_channels,
            with_modal_channels,
        })
    })
    .as_ref()
}

/// **A sweep instrument, not a gate.** Renders the gate's material through one
/// or more `[voicing.mics]` settings and prints the columns for each.
///
/// ```text
/// MIC_SWEEP='0.12,0.30,0.70,1.0,1.0; 0.20,0.20,0.70,1.2,1.5' \
///   cargo test --release -p piano-tuner --test stereo -- --ignored --nocapture mic_geometry
/// ```
///
/// The fields are `spacing_m, height_m, span_m, width, diffuse_coherence` and,
/// optionally, the mode-controlled band's `lo_hz, hi_hz, lift`;
/// the empty string scores the shipped preset as it stands. It is `#[ignore]`d
/// because it asserts nothing — the gate above is what asserts.
#[test]
#[ignore]
fn mic_geometry_sweep() {
    if references().is_none() {
        eprintln!("no data/salamander in this tree; skipping the sweep");
        return;
    }
    let spec = std::env::var("MIC_SWEEP").unwrap_or_default();
    for setting in spec.split(';').map(str::trim).filter(|s| !s.is_empty()) {
        let n: Vec<f32> = setting
            .split(',')
            .map(|f| f.trim().parse().expect("five numbers"))
            .collect();
        assert!(
            n.len() == 5 || n.len() == 8,
            "spacing,height,span,width,coherence[,modal lo_hz,hi_hz,lift]"
        );
        let mut preset = shipped_preset();
        preset.voicing.mics = Some(piano_emulator::preset::MicVoicing {
            spacing_m: n[0],
            height_m: n[1],
            span_m: n[2],
            width: n[3],
            diffuse_coherence: n[4],
            source_extent_m: 0.0,
            modal: (n.len() == 8).then(|| piano_emulator::preset::ModalBand {
                lo_hz: n[5],
                hi_hz: n[6],
                lift: n[7],
            }),
        });
        preset.validate().expect("a legal geometry");
        let columns = realism::stereo_columns(&score(&preset).0);
        let reds = columns.iter().filter(|c| !c.pass).count();
        println!(
            "\n=== mics {setting} — {reds} of {} bands red{}",
            columns.len(),
            report("engine against the recording", &columns)
        );
    }
    if spec.is_empty() {
        let columns = realism::stereo_columns(&score(&shipped_preset()).0);
        let reds = columns.iter().filter(|c| !c.pass).count();
        println!(
            "\n=== shipped preset — {reds} of {} bands red{}",
            columns.len(),
            report("engine against the recording", &columns)
        );
    }
}

fn report(what: &str, columns: &[StereoColumn]) -> String {
    format!(
        "\nSTEREO columns on {} recorded keys at velocity {VELOCITY} ({what}):\n{}",
        columns.first().map_or(0, |c| c.items),
        realism::stereo_report(columns)
    )
}

// ---------------------------------------------------------------------------
// The gate
// ---------------------------------------------------------------------------

/// **The gate of `DECISIONS.md` 346-379, and the standing record of item 314.**
///
/// Per band, the engine's lag-zero interchannel correlation against the
/// recording's, over the keys the library recorded, against a bar made of the
/// recording's own disagreement with itself.
///
/// It failed in every band when it was written — in the bass because the
/// board's two output taps were anti-phase where the recording's two capsules
/// see one wavefront, in the treble because a pan-pot correlates at +1 where a
/// spaced pair does not correlate at all, and in the middle because the board
/// was one radiator where the recording shows a mode-controlled plate. All
/// three are now mechanisms in `soundboard`, and the module header says which
/// milestone closed which bands.
///
/// # Red again in the two nodal bands since `DECISIONS.md` 418, on purpose
///
/// This gate was green in all six bands from item 379 to item 417, and it was
/// green **on the mechanism item 392 convicted**. `r0 = (E_mid − E_side) /
/// (E_mid + E_side)`, so the recording's `−0.115` at 125-250 Hz and `−0.226` at
/// 250-500 are *more difference than sum* — `E_side/E_mid` of 1.26 and 1.58 —
/// and the only thing in this engine that could produce them was
/// `[voicing.mics.modal]`'s lift, whose whole contribution to that ratio is
/// `lift²·|B|²`. **A ratio above one is a lift above one**, and a lift above one
/// is `1 − g < 0`: one loudspeaker carrying the note inverted against the
/// other, at a frequency that moves with pitch, with two outright nulls inside
/// the audible band. Item 417 accepted that price as too high — a listener
/// found the artifact three separate ways while this gate was green — and item
/// 418 railed the lift at the null.
///
/// So these two bands are red because the instrument stopped manufacturing what
/// made them green, and the refit's reach under the rail is `E_side/E_mid` of
/// **0.80 and 1.03** against the 1.26 and 1.58 asked for. It is the same
/// shortfall the per-channel board's `pair_db` columns carry, in the other of
/// the two units it can be read in, and item 418 records the frontier and the
/// three-way conflict that bounds it. **Do not close this by moving the bar**:
/// the bar is the recording against its own second take and the shortfall is
/// the mechanism, not the measurement.
#[test]
#[ignore = "D418/D486/D463 known gap, re-barred to the neutral policy's ceiling on the side energy (r0 = 0, which is D418's own lift rail in this board's units): all six bands are red on the point D487 installs, worst 0.977 of 0.120 at 125-250 — the pair the owner's verdict of D485 leaves carries 0.06-0.53 dB of side where the recording carries 2.8-3.9; run with --ignored to read the current distance"]
fn the_engines_stereo_image_is_the_recordings_in_every_band() {
    let Some(m) = measured() else {
        eprintln!("no data/salamander in this tree; skipping the stereo gate");
        return;
    };
    let columns = realism::stereo_columns(&m.items);
    // **The neutral-policy re-bar, printed before anything is asserted**
    // (`DECISIONS.md` 486, `realism::NEUTRAL_SIDE_CEILING_R0`) — item 417's own
    // rule for an acceptance: one nobody can read is indistinguishable from a
    // widened bar. The statistic has not moved and neither has the bar; what
    // has moved is the **target**, in the bands where the recording's own `r0`
    // is negative and therefore asks for more difference than sum. `r0 < 0` is
    // `E_side > E_mid` is `|T| > 1`, which is where `|1 − T|` can vanish and one
    // loudspeaker inverts against the other — the thing item 418 railed
    // `MIC_MODAL_LIFT` at one to forbid after a listener found it three ways.
    println!(
        "the neutral policy's ceiling on the side energy, excluded from this gate's target \
(`DECISIONS.md` 486, on the owner's verdict of item 485; r0 = 0 is E_side = E_mid is the \
lift rail of item 418 in this board's own units). The bar is unchanged in every band — it \
is still the recording against its own second take, or the material's own uncertainty:"
    );
    for c in &columns {
        if c.items == 0 {
            continue;
        }
        println!(
            "  {:>7}: recording asks r0 {:+.3} → target {:+.3} (excluded {:+.3}); \
engine {:+.3}, |err| {:.3} against a bar of {:.3} — {}",
            c.name,
            c.reference_r0,
            c.target_r0,
            c.excluded_r0,
            c.engine_r0,
            c.error,
            c.bar,
            if c.excluded_r0.abs() < 1e-12 {
                "untouched: the recording sees more sum than difference here"
            } else {
                "re-barred: the recording sees more difference than sum here, which no mono-exact pair can hold without a channel inversion"
            },
        );
    }
    let red: Vec<&StereoColumn> = columns.iter().filter(|c| !c.pass).collect();
    let mut lines = String::new();
    for c in &red {
        let _ = write!(
            lines,
            "\n  {:>8}: engine {:+.3} where the recording reads {:+.3} — |err| {:.3} \
against a bar of {:.3} (floor {:.3}, scatter {:.3}, x{:.2}), worst key {} at {:.3}",
            c.name,
            c.engine_r0,
            c.reference_r0,
            c.error,
            c.bar,
            c.floor,
            c.scatter,
            realism::STEREO_ALLOWANCE,
            c.worst.as_ref().map(|w| w.0.as_str()).unwrap_or("?"),
            c.worst.as_ref().map(|w| w.1).unwrap_or(f64::NAN),
        );
    }
    assert!(
        red.is_empty(),
        "{} of {} bands are outside what the recording says about itself.{lines}\n{}",
        red.len(),
        columns.len(),
        report("engine against the recording", &columns)
    );
}

/// **The per-channel gate of `DECISIONS.md` 392-394, and the standing record of
/// what three listening complaints had in common.**
///
/// Per band, `realism::channel_columns`: what each loudspeaker's spectrum does
/// against the take's own mono spectrum, engine against recording, over the
/// keys the library recorded, against a bar made of the recording's own
/// disagreement with itself. `realism::ChannelBand`'s own header says what the
/// statistic is and why the mono boards and the coherence board are both blind
/// to it.
///
/// **Three statistics, not one** (`DECISIONS.md` 395). `dev_L`/`dev_R` are
/// *shapes* — each channel's band share against the take's own mono share —
/// and a shape cannot see loudness: the lobe manufactures `2(1 + g²)` of pair
/// energy against the fold-down's `1`, up to +6.18 dB of it, and moves no
/// shape at all. So the column carries `pair_balance` beside them, the pair's
/// own energy against its mono sum, against the recording's own value. And
/// both of those are medians over thirty keys, so `per_key_error` — the same
/// distance taken key by key — is asserted too, against the recording's own
/// key-to-key spread. A board right on the median and wrong at every key is
/// the C4-shaped failure at key granularity.
///
/// **What is asserted, and why it is not all six bands.** The two bands the
/// **mode-controlled lobe acts in — 125-250 and 250-500 Hz — must pass, and no
/// band may be further out than a pan-potted engine's own value.** The first
/// half is this milestone's own work: the lobe is the only thing in the engine
/// that deliberately puts a per-channel spectral difference there, it read
/// **2.63 dB against a bar of 1.09** when this column was written, and the
/// repair is judged on it. The second half is the rest of the compass, where
/// what the column measures is the pan-pot's board taps and the pair geometry
/// — jobs of other milestones, in bands the engine is 1-2 dB out in with no
/// microphone section at all (a pan-potted engine reads 5.68 dB out at
/// 63-125 Hz). Asserting those absolutely would be asserting a milestone that
/// has not happened; asserting that the microphone section does not make them
/// *worse* is exactly the "must not have bought its two bands by spending the
/// other four" rule the mode-controlled band already carries below.
///
/// # The target excludes the capsule-placement asymmetry, and says so out loud
///
/// `DECISIONS.md` 417's side-injection probe refuted the whole class of
/// symmetric repairs at once: the recording's nodal band is asymmetric **in
/// level** between its two capsules — `dev_L − dev_R` up to **+5.85 dB at
/// 178 Hz** — and no side source, incoherent or coherent, can move that
/// statistic under control, while item 406(a)'s successor form is excluded by
/// the very property it was chosen for (`|L| = |R|`). Reproducing it needs
/// per-channel per-band gains fitted to where two microphones stood on one
/// afternoon, which the standing no-room and no-mic-idiosyncrasy policy
/// refuses. So it is **accepted as unscored**, in the same class as item 328's
/// transposed notes, and subtracted from this gate's target rather than
/// chased: `ChannelColumn::asymmetry`, half the reference's own spread, which
/// is the floor the statistic itself puts under any model whose two channels
/// depart symmetrically from their own mono. The gate prints it, per octave and
/// at the sixth-octave resolution the probe measured it at, **before** it
/// asserts anything — an acceptance nobody can read is indistinguishable from a
/// widened bar.
///
/// **This engine is not a symmetric model and the exclusion says so** (item
/// 424). A nodal-line lobe is `L = m(1 + g)`, `R = m(1 − g)`: per-channel by
/// construction, and the engine's own spread reads −1.27 dB at 125-250 Hz and
/// +3.64 at 250-500 against the reference's +1.52 and −1.30 — opposite in sign
/// in both. So the floor **sizes** the exclusion and item 417's policy is what
/// **justifies** it, and what makes the difference safe here is measured rather
/// than argued: the exclusion changes no verdict this gate asserts. The two
/// bands asserted absolutely are red against the unexcluded bar too (2.38
/// against 1.15, 2.47 against 1.09), the other four are asserted against a
/// pan-potted engine's own error with the unexcluded `bar` as the slack, and
/// the engine is *worse* than the symmetric floor in both nodal bands rather
/// than spending it. Where the exclusion does move a target is the fit, which
/// reads `ChannelColumn::reachable` so that fit and gate close on one
/// definition.
///
/// `the_acceptance_still_fails_on_the_lobe_it_was_re_barred_against` is the
/// falsification: the pre-418 unclamped lobe is refused by both crates' schemas
/// *and* is red on this board against the reachable bar, so the exclusion is
/// narrower than the defect it could be accused of hiding.
///
/// # And it is still red, which is item 418's frontier and not a surprise
///
/// The rail costs what it was always going to cost. `pair_db` is
/// `10 log10(1 + E_side/E_mid)` and the lobe's whole contribution to
/// `E_side/E_mid` is `lift²·|B|²`, so **one is a ceiling of +3.01 dB** where the
/// recording's own two nodal bands read **+3.54 and +3.88**; the same ratio read
/// as a correlation is `r0 = (1 − ρ)/(1 + ρ)`, and the recording's `−0.115` and
/// `−0.226` are `ρ = 1.26` and `1.58` — *more difference than sum*, which for
/// this construction is a lift above the null. The refit under the rail reaches
/// `ρ = 0.80` and `1.03`. Every red left on this board and the two the
/// coherence board now carries are that one shortfall read four ways, and item
/// 418 records the map.
#[test]
#[ignore = "D418/D486/D463 known gap, re-barred with the coherence gate: the same shortfall read per channel — pair_db balance -0.16 to -2.95 across the six bands on the point D487 installs, worst 2.95 of 0.56 at 125-250; run with --ignored to read the current distance"]
fn each_loudspeaker_has_the_recordings_spectrum_where_the_mic_pair_acts() {
    let Some(m) = measured() else {
        eprintln!("no data/salamander in this tree; skipping the per-channel gate");
        return;
    };
    let columns = realism::channel_columns(&m.channels);
    let text = format!(
        "\n{}\nthe same board with no microphone section at all, for comparison:\n{}",
        realism::channel_report(&columns),
        realism::channel_report(&realism::channel_columns(&m.panned_channels))
    );
    // **The same board at a sixth of an octave, printed first** — the resolution
    // the mode-controlled band's own shape lives at (`realism::STEREO_FINE_BANDS`,
    // `DECISIONS.md` 404). `pair_db` and `mono_db` are shapes over frequency,
    // the band is 0.96 octaves wide, and the two scoreboard bands that contain
    // it are an octave each — so an angle that is right at the band's bottom
    // edge and eight decibels too large at its top averages into a column
    // reading inside its bar. This board cannot average that away, and the
    // control beside it is the same preset with `[voicing.mics.modal]` deleted,
    // whose fold-down is the pan-pot's own to `f32` rounding: its distance from
    // the recording **is** the headroom a nodal line has to spend.
    //
    // It is printed **before** the assertions and not asserted itself: this
    // gate is the project's third documented red, so anything asserted under it
    // never runs, and the point of this board is to be readable on exactly the
    // run that fails. `DECISIONS.md` 404-406 is what it measured and what that
    // refuted.
    let fine = realism::channel_fine_columns(&m.channels);
    let fine_bare = realism::channel_fine_columns(&m.without_modal_channels);
    println!(
        "the per-channel board at a sixth of an octave:\n{}\nand the same with the \
mode-controlled band deleted, which is the headroom the fold-down has:\n{}",
        realism::channel_report(&fine),
        realism::channel_report(&fine_bare)
    );
    let middle: Vec<&realism::ChannelColumn> = columns
        .iter()
        .filter(|c| c.items > 0 && c.lo_hz >= 125.0 && c.hi_hz <= 500.0)
        .collect();
    assert_eq!(
        middle.len(),
        2,
        "the two bands the mode-controlled band lives in must both be readable{text}"
    );
    // **The acceptance is printed before it is taken, so it is never silent**
    // (`DECISIONS.md` 417-418). What the recording asks in these bands has a
    // component no symmetric pair of capsules can produce — the reference
    // session's own placement across the board's nodal lines, `spread = dev_L −
    // dev_R` running to +5.85 dB at 178 Hz, which item 417's side-injection
    // probe refuted every mechanism against and which the standing
    // no-mic-idiosyncrasy policy accepts as unscored, exactly as item 328
    // excludes the library's transposed notes from every fitted quantity.
    // `ChannelColumn::asymmetry` is that component, half the reference spread,
    // and the verdict below is taken against `bar + asymmetry`.
    println!(
        "the capsule-placement asymmetry excluded from this gate's target \
(`DECISIONS.md` 417; half the recording's own dev_L − dev_R, out of the reference alone).\n\
  The excluded figure is the floor this statistic puts under a model whose two channels depart \
*symmetrically* from their own mono; a nodal-line lobe is `L = m(1+g)`, `R = m(1-g)` and is not \
one, so the engine's own spread is printed beside it and the two bands' verdicts are stated \
against both bars (`DECISIONS.md` 424):"
    );
    for c in &columns {
        if c.items == 0 {
            continue;
        }
        println!(
            "  {:>7}: reference spread {:+.2} dB → excluded {:.2} dB; bar {:.2} → reachable \
{:.2}; the engine's own spread is {:+.2} dB and it reads {:.2}, where a symmetric model's \
best would be {:.2} — {}",
            c.name,
            c.reference_left_db - c.reference_right_db,
            c.asymmetry,
            c.bar,
            c.reachable,
            c.engine_left_db - c.engine_right_db,
            c.error,
            c.asymmetry,
            if c.error <= c.asymmetry {
                "inside the symmetric floor, so this band is spending the exclusion"
            } else {
                "worse than that floor, so the exclusion is not what carries this band"
            },
        );
    }
    // **And the second exclusion, which is item 486's and not item 417's.**
    // `pair_db` is `10 log10(1 + E_side/E_mid)`, so `E_side/E_mid = 1` is
    // +3.0103 dB — the same ceiling the coherence board reads as `r0 = 0`, and
    // the same one item 418 railed `MIC_MODAL_LIFT` at. Where the recording's
    // own pair energy is above it, this board has stopped asking for the
    // difference.
    println!(
        "the neutral policy's ceiling on the side energy, excluded from this gate's \
`pair_db` target (`DECISIONS.md` 486; +{:.4} dB is E_side = E_mid, which is item 418's \
lift rail in this column's own units):",
        realism::NEUTRAL_PAIR_CEILING_DB
    );
    for c in &columns {
        if c.items == 0 {
            continue;
        }
        println!(
            "  {:>7}: recording asks pair_db {:+.2} → target {:+.2} (excluded {:.2}); \
engine {:+.2}, balance {:+.2} against a bar of {:.2}",
            c.name,
            c.reference_pair_db,
            c.target_pair_db,
            c.excluded_pair_db,
            c.engine_pair_db,
            c.pair_balance,
            c.pair_bar,
        );
    }
    println!(
        "  and the same at a sixth of an octave, where the probe measured it \
(`renders/side-injection/SIDE_INJECTION.md` §5f):"
    );
    for c in &fine {
        if c.items == 0 || c.lo_hz < 125.0 || c.hi_hz > 500.0 {
            continue;
        }
        println!(
            "  {:>7}: reference spread {:+.2} dB → excluded {:.2} dB",
            c.name,
            c.reference_left_db - c.reference_right_db,
            c.asymmetry
        );
    }
    for c in &middle {
        assert!(
            c.pass,
            "{}: the two channels' spectra are {:.2} dB from the recording's against a reachable \
bar of {:.2} — the recording's own bar {:.2} (floor {:.2}, scatter {:.2}, x{:.2}) plus the \
{:.2} dB of capsule-placement asymmetry item 417 excludes (half the reference's own spread of \
{:+.2} dB) — engine L {:+.2} / R {:+.2} where the recording reads L {:+.2} / R {:+.2}{text}",
            c.name,
            c.error,
            c.reachable,
            c.bar,
            c.floor,
            c.scatter,
            realism::STEREO_ALLOWANCE,
            c.asymmetry,
            c.reference_left_db - c.reference_right_db,
            c.engine_left_db,
            c.engine_right_db,
            c.reference_left_db,
            c.reference_right_db,
            text = text
        );
    }
    // **The loudness half.** `dev_L`/`dev_R` are shapes and cannot see it: the
    // lobe adds `2(1 + g^2)` of pair energy where the mono fold-down keeps
    // `1`, and item 392 measured up to +6.18 dB of it with nothing pushing
    // back. `ChannelColumn::pair_balance` is that energy against the
    // recording's own, per band, signed, on the same thirty keys — and it is
    // asserted in the same two bands and against the same pan-pot rule as the
    // shape half, for the same reasons.
    for c in &middle {
        assert!(
            c.pair_pass,
            "{}: the two loudspeakers carry {:+.2} dB of pair energy against their own mono \
fold-down where the recording carries {:+.2} — a balance of {:+.2} dB against a bar of {:.2} \
(take-to-take {:.2}, key-to-key {:.2}/sqrt(n), x{:.2}){text}",
            c.name,
            c.engine_pair_db,
            c.reference_pair_db,
            c.pair_balance,
            c.pair_bar,
            c.pair_floor,
            c.pair_scatter,
            realism::STEREO_ALLOWANCE,
            text = text
        );
    }
    // **And the per-key half.** Both of the above are medians over thirty keys,
    // and a board right on the median and wrong at every key is exactly the
    // C4-shaped failure at key granularity. The bar is the recording's own
    // key-to-key sigma; `ChannelColumn::per_key_error` says why it is that and
    // not the take-to-take floor.
    for c in &middle {
        assert!(
            c.per_key_pass,
            "{}: the two channels' spectra are {:.2} dB from the recording's at the median \
*key* against a bar of {:.2} (the recording's own key-to-key sigma {:.2}, x{:.2}; its \
take-to-take floor is {:.2}) — worst key {}{text}",
            c.name,
            c.per_key_error,
            c.per_key_bar,
            c.scatter,
            realism::STEREO_ALLOWANCE,
            c.per_key_floor,
            c.worst
                .as_ref()
                .map_or_else(|| "—".to_string(), |(k, d)| format!("{k} at {d:.2}")),
            text = text
        );
    }
    let pan = realism::channel_columns(&m.panned_channels);
    for (c, p) in columns.iter().zip(&pan) {
        if c.items == 0 || p.items == 0 || !c.error.is_finite() || !p.error.is_finite() {
            continue;
        }
        assert!(
            c.pair_balance.abs() <= p.pair_balance.abs() + c.pair_bar,
            "{}: the microphone section takes this band's pair energy from {:+.2} dB out to \
{:+.2}, which is more than a bar ({:.2}) worse than having no section at all{text}",
            c.name,
            p.pair_balance,
            c.pair_balance,
            c.pair_bar,
            text = text
        );
        assert!(
            c.error <= p.error + c.bar,
            "{}: the microphone section takes this band from {:.2} dB out to {:.2}, which is \
more than a bar ({:.2}) worse than having no section at all{text}",
            c.name,
            p.error,
            c.error,
            c.bar,
            text = text
        );
    }
}

/// **The falsification the re-barring is kept honest by**: put the unclamped
/// lobe back and the acceptance above must go red again.
///
/// `DECISIONS.md` 418 does two things to this gate at once — it clamps the
/// instrument (`soundboard::MIC_MODAL_LIFT` at one) and it re-bars the target
/// (item 417's capsule-placement asymmetry excluded, `ChannelColumn::asymmetry`)
/// — and either of those, done alone and badly, is a way to make a red test
/// green without fixing anything. This is the test that says the second did not
/// swallow the first: the exact `[voicing.mics.modal]` the tree shipped before
/// item 418, restored on top of everything else this milestone changed, read
/// against the *reachable* bar rather than the old one.
///
/// Two assertions, and the pair of them is the point.
///
/// * **The schema refuses it.** A lift of 2.124 is not a preset any more, in
///   either crate's copy of the rails, so the defect cannot come back through a
///   file.
/// * **And the board refuses it too**, on the same renders and the same
///   statistic the acceptance passes on — so the exclusion is narrower than the
///   defect it is accused of hiding. If a future edit widens `asymmetry` until
///   the old lobe fits under it, this goes red.
#[test]
fn the_acceptance_still_fails_on_the_lobe_it_was_re_barred_against() {
    let Some(m) = measured() else {
        eprintln!("no data/salamander in this tree; skipping the falsification");
        return;
    };
    // `DECISIONS.md` 401's own numbers: the band `presets/salamander-c5.toml`
    // carried from item 379 to item 417, lift and all.
    const PRE_418: piano_emulator::preset::ModalBand = piano_emulator::preset::ModalBand {
        lo_hz: 229.425_17,
        hi_hz: 307.391_63,
        lift: 2.124_296_2,
    };
    let mut unclamped = shipped_preset();
    unclamped.voicing.mics = unclamped
        .voicing
        .mics
        .map(|mics| piano_emulator::preset::MicVoicing {
            modal: Some(PRE_418),
            ..mics
        });
    let refusal = unclamped.validate().expect_err(
        "the pre-418 lobe must be refused by the schema — that is the rail item 418 added",
    );
    println!("the schema's own refusal of the unclamped lobe: {refusal}");
    assert!(
        format!("{refusal}").contains("voicing.mics.modal.lift"),
        "the refusal must name the lift and not something else: {refusal}"
    );

    // Rendered anyway — `render_to_buffer` does not validate — so the board can
    // be asked the same question the acceptance asks.
    let (_, channels) = score(&unclamped);
    let columns = realism::channel_columns(&channels);
    let text = format!(
        "\nthe pre-418 unclamped lobe, on this milestone's own board:\n{}",
        realism::channel_report(&columns)
    );
    let middle: Vec<&realism::ChannelColumn> = columns
        .iter()
        .filter(|c| c.items > 0 && c.lo_hz >= 125.0 && c.hi_hz <= 500.0)
        .collect();
    assert_eq!(middle.len(), 2, "both middle bands must be readable{text}");
    let red = middle.iter().filter(|c| !c.pass).count();
    assert!(
        red > 0,
        "the unclamped lobe passes the re-barred acceptance, which means the exclusion is \
wider than the defect: {}{text}",
        middle
            .iter()
            .map(|c| format!(
                "{} {:.2} against a reachable {:.2} (bar {:.2} + excluded {:.2})",
                c.name, c.error, c.reachable, c.bar, c.asymmetry
            ))
            .collect::<Vec<_>>()
            .join(", "),
        text = text
    );
    // And the two instruments are told apart on the halves the rail *did*
    // close, so this is a difference between two instruments and not a board
    // that fails everything. The shipped one is inside the recording's own
    // pair-energy on neither band (item 418's frontier) and the unclamped one
    // is inside on both — because the energy it is inside with is manufactured,
    // which is the whole of item 392. Printed, both ways, and asserted only on
    // the direction: the clamped instrument must carry **less** pair energy than
    // the unclamped one in the bands the lobe acts in.
    let shipped: Vec<realism::ChannelColumn> = realism::channel_columns(&m.channels)
        .into_iter()
        .filter(|c| c.items > 0 && c.lo_hz >= 125.0 && c.hi_hz <= 500.0)
        .collect();
    for (c, s) in middle.iter().zip(&shipped) {
        println!(
            "  {}: unclamped pair {:+.2} dB (balance {:+.2}) against clamped {:+.2} \
(balance {:+.2}), the recording's own {:+.2}",
            c.name,
            c.engine_pair_db,
            c.pair_balance,
            s.engine_pair_db,
            s.pair_balance,
            c.reference_pair_db
        );
        assert!(
            s.engine_pair_db < c.engine_pair_db,
            "{}: the clamped instrument carries {:+.2} dB of pair energy against the unclamped \
one's {:+.2} — the rail is supposed to take energy out of the two loudspeakers, not put it \
in{text}",
            s.name,
            s.engine_pair_db,
            c.engine_pair_db,
            text = text
        );
    }
}

/// **The control the per-channel column needs to be a measurement**: the same
/// pair with and without a mode-controlled lobe, everything else the same.
///
/// The lobe is what items 392-418 repaired, so the column has to be able to
/// see it — and it has to see it *in the lobe's own two bands* and nowhere
/// else, or it is measuring something the lobe did not do. What it must not do
/// is red out on the capsule pair alone in bands the pair never touches.
///
/// **Both instruments are named rather than assumed** (`DECISIONS.md` 452).
/// This used to be "the shipped preset against the shipped preset with the
/// band deleted", which stopped being two instruments the moment nothing
/// shipped a band: the difference would be zero and the control would report
/// that a column measuring the lobe perfectly is broken. `Measured::
/// with_modal_channels` is the banded side whichever way the preset falls —
/// the shipped band when there is one, `melody::M17_MODAL_BAND` when there is
/// not — so the control asks its own question in both worlds. It is a
/// statement about the **column**, not about what ships.
#[test]
fn the_per_channel_column_sees_the_mode_controlled_band_and_only_it() {
    let Some(m) = measured() else {
        eprintln!("no data/salamander in this tree; skipping the per-channel control");
        return;
    };
    let with = realism::channel_columns(&m.with_modal_channels);
    let without = realism::channel_columns(&m.without_modal_channels);
    let text = format!(
        "\nwith a mode-controlled band:\n{}\nwithout one:\n{}",
        realism::channel_report(&with),
        realism::channel_report(&without)
    );
    let moved: Vec<f64> = with
        .iter()
        .zip(&without)
        .filter(|(a, b)| a.items > 0 && b.items > 0)
        .map(|(a, b)| (a.engine_left_db - b.engine_left_db).abs())
        .collect();
    assert!(
        !moved.is_empty(),
        "no band was readable on both sides of the control{text}"
    );
    // The lobe's own bands move; the bands it cannot reach do not. `500-2k`
    // is the first band above it and the loosest statement of the two, because
    // the fourth-order upper edge is still 12 dB down an octave over `hi_hz`.
    let inside: f64 = with
        .iter()
        .zip(&without)
        .filter(|(a, _)| a.lo_hz >= 125.0 && a.hi_hz <= 500.0)
        .map(|(a, b)| (a.engine_left_db - b.engine_left_db).abs())
        .fold(0.0, f64::max);
    let outside: f64 = with
        .iter()
        .zip(&without)
        .filter(|(a, _)| a.hi_hz <= 125.0 || a.lo_hz >= 2_000.0)
        .map(|(a, b)| (a.engine_left_db - b.engine_left_db).abs())
        .fold(0.0, f64::max);
    assert!(
        inside > 0.5,
        "adding the mode-controlled band moved its own bands by only {inside:.2} dB — this \
column cannot be measuring it{text}"
    );
    assert!(
        outside < inside,
        "adding the mode-controlled band moved a band it does not live in ({outside:.2} dB) by \
more than one it does ({inside:.2}){text}"
    );
}

/// **The mono-sum gate, and the neutrality gate, in one render.**
///
/// The microphone pair is written as *mid plus side* and replaces only the
/// side, so the mono fold-down of the new image is the mono fold-down of the
/// pan-pot — for every source, every pan and every geometry
/// (`soundboard::Mics`, and `soundboard::tests::
/// the_microphone_pair_leaves_the_mono_sum_exactly_where_the_pan_pot_put_it`
/// asserts it on the board directly). This is the same claim end to end, on the
/// shipped preset's own demo, in the bands the scoreboard reads: **every mono
/// board in this repository is a function of `(L + R)/2`, and if that signal
/// does not move, none of them can.**
///
/// The bar is 0.5 dB band-wise, which is the milestone's stated discipline;
/// what it measures is 0.000 dB, because the only thing between the two sums is
/// `f32` rounding.
///
/// The second half is `DECISIONS.md` 103's contract: the *same* preset with
/// `[voicing.mics]` deleted renders the pan-pot bit for bit. Not "within a
/// tolerance" — the identical samples, which is what "absent means the old
/// model" has to mean.
#[test]
fn the_mono_fold_down_and_the_preset_without_the_section_are_both_unmoved() {
    use piano_emulator::render::{demo_sequence, DEMO_DURATION_S};

    let with = shipped_preset();
    let mics = with
        .voicing
        .mics
        .expect("the shipped preset is the one with a microphone pair");
    let mut without = with.clone();
    without.voicing.mics = None;

    let demo = demo_sequence();
    let (wl, wr) = render_to_buffer(&with, &demo, DEMO_DURATION_S);
    let (bl, br) = render_to_buffer(&without, &demo, DEMO_DURATION_S);
    assert!(wl.iter().any(|v| v.abs() > 0.1), "the demo made no sound");

    // (a) The image did move — otherwise the rest of this proves nothing.
    let moved = wl
        .iter()
        .zip(&bl)
        .map(|(a, b)| f64::from(a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(
        moved > 1.0e-3,
        "the microphone pair changed the left channel by {moved:e}: it is not doing anything"
    );

    // (b) ... and the mono sum did not, band by band. `stereo_image` reports
    // each band's share of the whole signal's energy, so the band's own level
    // is that share times the signal's energy, and the two are compared in dB.
    let mono = |l: &[f32], r: &[f32]| -> (StereoImage, f64) {
        let m: Vec<f32> = l.iter().zip(r).map(|(&a, &b)| 0.5 * (a + b)).collect();
        let energy: f64 = m.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
        let image = realism::stereo_image(&m, &m, f64::from(SAMPLE_RATE)).expect("an image");
        (image, 10.0 * energy.log10())
    };
    let (wi, we) = mono(&wl, &wr);
    let (bi, be) = mono(&bl, &br);
    let mut worst: f64 = 0.0;
    let mut lines = String::new();
    for (b, &(name, _, _)) in realism::STEREO_BANDS.iter().enumerate() {
        let (a, c) = (wi.bands[b], bi.bands[b]);
        let delta = (a.level_db + we) - (c.level_db + be);
        worst = worst.max(delta.abs());
        let _ = write!(lines, "\n  {name:>8}: {:+.4} dB", delta);
    }
    let broadband = we - be;
    // Printed, not only asserted: the milestone's discipline is 0.5 dB and what
    // this reads is `f32` rounding, and the difference between those two is
    // only visible if the number is on the page.
    println!(
        "the mono fold-down of the shipped demo, with the microphone pair and the \
board's mode-controlled band against the pan-pot: {worst:.4} dB worst band, \
{broadband:+.4} dB broadband{lines}"
    );
    assert!(
        worst < 0.5 && broadband.abs() < 0.5,
        "the mono fold-down moved by {worst:.4} dB band-wise ({broadband:+.4} dB broadband) \
         with mics {mics:?}{lines}"
    );
    // It is not merely inside the bar, it is rounding: state the number the
    // gate actually reads so a regression that spends the whole 0.5 dB is
    // visible as a change rather than as a pass.
    assert!(
        worst < 0.01,
        "the mono fold-down moved by {worst:.4} dB band-wise, which is more than \
         `f32` rounding{lines}"
    );

    // (c) The neutrality contract: the same preset without the section is the
    // pan-pot, sample for sample.
    let (nl, nr) = render_to_buffer(&without, &demo, DEMO_DURATION_S);
    assert_eq!(nl, bl, "left channel is not reproducible");
    assert_eq!(nr, br, "right channel is not reproducible");
}

/// Where the window is allowed to start, in samples past the strike, for the
/// two controls below.
///
/// Ninety-six samples is exactly the misalignment `DECISIONS.md` 378 found in
/// this file — a 0.05 s preroll against a 128-sample block — and thirty-two is
/// what a 0.03 s or 0.11 s one gives, so these are the three placements the
/// gate has actually been read at rather than three round numbers.
const WINDOW_SHIFTS: [usize; 3] = [0, 32, 96];

/// The image of the same signal read from `shift` samples in, with the step at
/// the window's edge faded out over the shift.
///
/// The fade is not cosmetic and it is the difference between measuring a signal
/// and measuring a window: a window that opens in the middle of a note opens
/// with a step, and a step is broadband. Without it the engine's 6-12 kHz band
/// goes from readable on 15 of these keys to readable on 29 purely by moving
/// the window 2 ms, and A0's band sits 16 dB higher than it does when the
/// window opens in silence. With it, that column comes back to within a
/// decibel or two of where the aligned window has it, which is what says the
/// difference was the edge. What is left after the fade is *content* — the
/// first two milliseconds of the note, which is the half of this that is real.
///
/// Every shift reads a window of the same *length* — the signal less the widest
/// shift — so what moves between them is where the window is and nothing else.
fn image_from(audio: &Audio, shift: usize) -> StereoImage {
    let widest = WINDOW_SHIFTS.iter().copied().max().unwrap_or(0);
    let faded = |c: &Vec<f32>| -> Vec<f32> {
        let mut v: Vec<f32> = c[shift..c.len() - widest + shift].to_vec();
        for (i, x) in v.iter_mut().take(shift).enumerate() {
            *x *= 0.5 - 0.5 * (std::f32::consts::PI * i as f32 / shift as f32).cos();
        }
        v
    };
    realism::stereo_image(
        &faded(&audio.channels[0]),
        &faded(&audio.channels[1]),
        f64::from(SAMPLE_RATE),
    )
    .expect("two channels")
}

/// Median over the keys of one band's `r0`, for the items that are readable in
/// that band at every one of [`WINDOW_SHIFTS`].
fn shift_medians(images: &[Vec<StereoImage>], band: usize) -> Vec<f64> {
    (0..WINDOW_SHIFTS.len())
        .map(|s| {
            let mut v: Vec<f64> = images
                .iter()
                .filter(|row| row.iter().all(|im| im.bands[band].readable()))
                .map(|row| row[s].bands[band].r0)
                .collect();
            v.sort_by(f64::total_cmp);
            if v.is_empty() {
                f64::NAN
            } else if v.len() % 2 == 1 {
                v[v.len() / 2]
            } else {
                0.5 * (v[v.len() / 2 - 1] + v[v.len() / 2])
            }
        })
        .collect()
}

/// **The control under the pinned window** (`DECISIONS.md` 378).
///
/// The engine's window starts at the strike because the strike is a sample the
/// renderer chose; the recording's starts where `detect_onset` says the note
/// begins, which is an *estimate* with a few samples of slop in it. Comparing
/// one against the other is only legitimate if the recording's image does not
/// depend on that slop — so this asserts it does not, over three times the
/// misalignment that made the finding.
///
/// It also prints the engine's own figure beside it, because they are not the
/// same number and the difference is the open half of item 379: the recording's
/// 125-500 Hz is a field that is still there two milliseconds later, and the
/// engine's is largely a strike.
#[test]
fn the_recordings_image_does_not_move_when_the_window_does() {
    let Some(rows) = references() else {
        eprintln!("no data/salamander in this tree; skipping the window control");
        return;
    };
    let sfz = sfz().expect("a library");
    let reference: Vec<Vec<StereoImage>> = rows
        .iter()
        .map(|r| {
            let audio = render_reference(&sfz, r.key, VELOCITY);
            WINDOW_SHIFTS.iter().map(|&s| image_from(&audio, s)).collect()
        })
        .collect();
    let preset = shipped_preset();
    let engine: Vec<Vec<StereoImage>> = rows
        .iter()
        .map(|r| {
            let audio = render_engine(&preset, r.key);
            WINDOW_SHIFTS.iter().map(|&s| image_from(&audio, s)).collect()
        })
        .collect();

    let columns = realism::stereo_columns(&measured().expect("a library").items);
    let mut lines = String::new();
    let mut worst: Vec<String> = Vec::new();
    for (b, c) in columns.iter().enumerate() {
        let (rm, em) = (shift_medians(&reference, b), shift_medians(&engine, b));
        let swing = |v: &[f64]| -> f64 {
            v.iter().copied().fold(f64::MIN, f64::max) - v.iter().copied().fold(f64::MAX, f64::min)
        };
        let _ = write!(
            lines,
            "\n  {:>8}: recording {} (swing {:.3}, bar {:.3}) — engine {} (swing {:.3})",
            c.name,
            rm.iter().map(|v| format!("{v:+.3}")).collect::<Vec<_>>().join(" "),
            swing(&rm),
            c.bar,
            em.iter().map(|v| format!("{v:+.3}")).collect::<Vec<_>>().join(" "),
            swing(&em),
        );
        if rm.iter().all(|v| v.is_finite()) && swing(&rm) > c.bar {
            worst.push(format!("{} by {:.3} against {:.3}", c.name, swing(&rm), c.bar));
        }
    }
    println!(
        "the image against where the window starts, at {WINDOW_SHIFTS:?} samples past the \
strike, over {} recorded keys at velocity {VELOCITY}:{lines}",
        rows.len()
    );
    assert!(
        worst.is_empty(),
        "the recording's own image moves with the window in {}: {}{lines}",
        worst.len(),
        worst.join("; ")
    );
}

/// The control that says the bar is passable and the columns do not simply red
/// out: the *recording itself* on the engine's side of the comparison. Every
/// band must pass, and it is not a tautology — the floor and the scatter are
/// measured on other signals, so a bar of zero anywhere would fail this.
#[test]
fn the_recording_passes_its_own_bar_in_every_band() {
    let Some(m) = measured() else {
        eprintln!("no data/salamander in this tree; skipping the stereo control");
        return;
    };
    let columns = realism::stereo_columns(&m.itself);
    for c in &columns {
        assert!(
            c.items > 0,
            "{} was readable on no key at all{}",
            c.name,
            report("the recording against itself", &columns)
        );
        assert!(
            c.bar > 0.0 && c.bar.is_finite(),
            "{}: a bar of {:.4} is not a bar{}",
            c.name,
            c.bar,
            report("the recording against itself", &columns)
        );
        // **The control splits in two, which is the honest form of an
        // exclusion** (`DECISIONS.md` 486, taking item 466's own shape). Against
        // the recording's *own* value the machinery has no bias of its own and
        // reads exactly zero in every band. Against the neutral policy's ceiling
        // on the side energy it stands off by exactly the excluded amount in
        // each of the **four** bands where it sees more difference than sum —
        // 125-250 and 250-500 by 0.115 and 0.226, and 500 Hz-2 kHz and 2-6 kHz
        // by 0.002 and 0.012, a hair under zero — and it **fails** only where
        // that exclusion is outside the band's own bar, which on this material
        // is 250-500 alone. That is the size of the exclusion, written down as
        // a test rather than as a claim.
        assert!(
            (c.engine_r0 - c.reference_r0).abs() < 1e-9,
            "{}: the recording is not its own image, {:+.4} against {:+.4}{}",
            c.name,
            c.engine_r0,
            c.reference_r0,
            report("the recording against itself", &columns)
        );
        assert!(
            (c.error - c.excluded_r0.abs()).abs() < 1e-9,
            "{}: the recording's distance from the neutral target is {:.4} where the \
             exclusion is {:.4} — the two have to be the same number or the target is \
             not what item 486 says it is{}",
            c.name,
            c.error,
            c.excluded_r0.abs(),
            report("the recording against itself", &columns)
        );
        assert_eq!(
            c.pass,
            c.excluded_r0.abs() <= c.bar,
            "{}: the recording passes iff the exclusion in this band is inside its own \
             bar, and here they disagree — {:.4} excluded against a bar of {:.4}{}",
            c.name,
            c.excluded_r0.abs(),
            c.bar,
            report("the recording against itself", &columns)
        );
    }
    let excluded: Vec<String> = columns
        .iter()
        .filter(|c| c.excluded_r0.abs() > 1e-9)
        .map(|c| format!("{} {:+.3}", c.name, c.excluded_r0))
        .collect();
    println!(
        "the neutral policy excludes this much of the recording's own decorrelation: {}",
        if excluded.is_empty() {
            "nothing".to_string()
        } else {
            excluded.join(", ")
        }
    );
    assert!(
        !excluded.is_empty(),
        "no band of this recording asks for more difference than sum, so item 486's \
         exclusion is of nothing and the policy has nothing to say here{}",
        report("the recording against itself", &columns)
    );
}

/// The control that says the gate catches the thing it names, with the engine
/// out of the picture entirely: the **recording's own mono sum, put back into
/// two channels**. That is precisely `soundboard::pan_for_key`'s construction
/// applied to the piano's own sound — same spectrum, same envelope, same
/// everything the mono metrics measure, and a stereo image that is +1 in every
/// band.
///
/// What it asserts is the shape of the finding rather than a blanket failure:
/// the **bass is where a pan-pot is nearly right**, because the recording reads
/// +0.945 there and one wavefront is one wavefront, and every band *above* it is
/// where the microphone spacing lives and where a pan-pot cannot be right. So
/// the bands from 125 Hz up must go red on a signal that is the recording in
/// every other respect. A gate that only fails is not a gate either.
#[test]
fn a_pan_potted_copy_of_the_recording_fails_above_the_bass() {
    let Some(m) = measured() else {
        eprintln!("no data/salamander in this tree; skipping the pan-pot control");
        return;
    };
    let columns = realism::stereo_columns(&m.panned);
    let text = report("a pan-potted copy of the recording", &columns);
    for c in &columns {
        if c.items == 0 {
            continue;
        }
        assert!(
            c.engine_r0 > 0.99,
            "{}: a pan-pot must read +1 at lag zero, read {:+.4}{}",
            c.name,
            c.engine_r0,
            text
        );
    }
    // The bass is the band where a pan-pot is nearly right — the recording is
    // +0.945 there and one wavefront is one wavefront. Everything above it is
    // where the mic spacing lives, and that is what must fail.
    let above_bass: Vec<&StereoColumn> = columns
        .iter()
        .filter(|c| c.items > 0 && c.lo_hz >= 125.0)
        .collect();
    assert!(
        !above_bass.is_empty(),
        "no band above the bass was readable{text}"
    );
    for c in &above_bass {
        assert!(
            !c.pass,
            "{}: a pan-pot of the recording must not pass, {:.4} against {:.4}{}",
            c.name, c.error, c.bar, text
        );
    }
}

/// **The control under `PHYSICS.md` §8's third regime**: the same preset with
/// `[voicing.mics.modal]` deleted, which is the instrument `DECISIONS.md` 367
/// shipped — a fitted pair of virtual capsules and nothing else.
///
/// It must fail, and it must fail *in 125-500 Hz specifically*, because that
/// is what item 357 predicted from the model and item 362 measured: a
/// two-point geometry cannot fall from `+0.95` below 125 Hz to `-0.115` one
/// octave up, and the pair that fits the recording's delays (0.11 m) is five
/// times narrower than the one its mid-band coherence would need (0.6-0.7 m).
/// Everything *outside* those two bands must still pass, because the pair was
/// already right there and the mode-controlled band is not allowed to have
/// bought its two bands by spending the other four.
#[test]
fn the_capsule_pair_without_the_mode_controlled_band_fails_in_the_middle() {
    let Some(m) = measured() else {
        eprintln!("no data/salamander in this tree; skipping the modal-band control");
        return;
    };
    let columns = realism::stereo_columns(&m.without_modal);
    let text = report("the capsule pair with no mode-controlled band", &columns);
    let middle: Vec<&StereoColumn> = columns
        .iter()
        .filter(|c| c.items > 0 && c.lo_hz >= 125.0 && c.hi_hz <= 500.0)
        .collect();
    assert_eq!(
        middle.len(),
        2,
        "the two bands the control is about must both be readable{text}"
    );
    for c in &middle {
        assert!(
            !c.pass,
            "{}: without the mode-controlled band this must be red, {:.4} against {:.4}{}",
            c.name, c.error, c.bar, text
        );
        // And red on the *positive* side: the defect is a band that stays
        // correlated where the recording goes anti-phase, not a band that
        // misses by an arbitrary amount.
        assert!(
            c.engine_r0 > c.reference_r0,
            "{}: the pair alone reads {:+.3} under the recording's {:+.3} — that is not \
             the finding this control exists for{}",
            c.name,
            c.engine_r0,
            c.reference_r0,
            text
        );
    }
    // **The outside-the-middle clause is gone, and it is the band that took it
    // with it** (`DECISIONS.md` 487). What that clause asserted was that
    // `[voicing.mics.modal]` had not bought its two bands by spending the other
    // four — a statement whose subject is the band, and the preset that ships no
    // longer has one, so `m.without_modal` and the shipped instrument are now
    // the same render and the clause has nothing to compare. What is left is the
    // half that still has a subject and is still item 357's finding: a
    // two-point geometry cannot hold 125-500 Hz, asserted above. The four bands
    // outside it are printed rather than gated, and where they are red they are
    // red on the instrument that ships — which is item 418's own gate's
    // business (`the_engines_stereo_image_is_the_recordings_in_every_band`) and
    // is carried there with its distance.
    for c in columns
        .iter()
        .filter(|c| c.items > 0 && (c.hi_hz <= 125.0 || c.lo_hz >= 500.0))
    {
        println!(
            "  outside the middle, printed and not gated since item 487: {} reads \
{:+.3} against a target of {:+.3}, |err| {:.4} of {:.4} — {}",
            c.name,
            c.engine_r0,
            c.target_r0,
            c.error,
            c.bar,
            if c.pass { "inside" } else { "outside" },
        );
    }
}

/// The band a key's fundamental is read in, asserted on the shipped library's
/// own recorded keys rather than on a table of constants: this is what
/// `COMPASS.md`'s stereo line quotes per key, and the clamp at the bottom
/// (A0-B1 are under the lowest band) is a decision worth pinning.
#[test]
fn every_recorded_key_lands_in_a_band_the_note_actually_fills() {
    let Some(sfz) = sfz() else {
        eprintln!("no data/salamander in this tree; skipping");
        return;
    };
    let library = SampleLibrary::from_sfz(&sfz).expect("the library reads");
    let recorded = RecordedKeys::from_library(&library).expect("the library records keys");
    let preset = shipped_preset();
    for &key in recorded.keys() {
        let f0 = f64::from(preset.string_params(key).partial_freq(1));
        let band = StereoImage::band_for(f0);
        let (_, lo, hi) = realism::STEREO_BANDS[band];
        assert!(
            f0 < hi,
            "{} at {f0:.1} Hz is not under the top of its band",
            realism::note_name(key)
        );
        assert!(
            f0 >= lo || band == 0,
            "{} at {f0:.1} Hz fell out of the bottom of a band that is not the lowest",
            realism::note_name(key)
        );
    }
}
