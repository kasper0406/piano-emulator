//! `piano-tuner mics` — fits `[voicing.mics]` off the recording's own stereo
//! image and writes it into a preset.
//!
//! `PHYSICS.md` §8 gave the engine a virtual pair of capsules
//! (`DECISIONS.md` 351-358), and its five numbers came out of a sweep run from
//! an `#[ignore]`d test over one surface, with two of the five — the height and
//! the span — "swept and left round" (item 355's own words). This subcommand is
//! what replaces the sweep, and it is stage 2's usual shape: measure the
//! recording, invert the part of it the model has a term for, and close the
//! rest on the engine's own render.
//!
//! ```sh
//! cargo run --release -p piano-tuner -- mics \
//!     data/salamander presets/salamander-c5.toml --out presets/salamander-c5.toml
//! ```
//!
//! # The stages, and why they are separate
//!
//! | stage | fields | material | what it is |
//! |---|---|---|---|
//! | `geometry` | `spacing_m`, `span_m` (`height_m` held) | the recording's own two channels | a TDOA inversion — [`estimate::mics`] |
//! | `profile` | none — it prints | the recording's own two channels | the sixth-octave interchannel curve, engine beside recording beside the recording's second take |
//! | `coherence` | `width`, `diffuse_coherence` | engine renders against the same recordings | a two-parameter search on `realism::stereo_columns` |
//! | `modal` | `[voicing.mics.modal]`'s two edges and its lift | the same | a coarse grid then the same search |
//! | `band` | the two edges, the lift, `width` and `diffuse_coherence` | the same | `modal`'s grid and search over all five at once |
//!
//! `band` is what to run: since `DECISIONS.md` 379 the mode-controlled term is
//! an anti-phase copy of the pair's own sum rather than a gain on a signal
//! orthogonal to it, so inside the band `width` and `lift` build **one** side
//! signal and cannot be fitted one after the other. `coherence` and `modal`
//! are kept because they are what the earlier milestones ran and because
//! either alone is still the right instrument for asking what one half can do
//! on its own.
//!
//! `profile` asserts nothing and writes nothing: it is the measurement item 357
//! said was missing when it named "the board's mode-controlled nodal lines"
//! and refused to model them without one. Six scoreboard bands are enough to
//! *score* a coherence and not enough to *see* one, and what the sixth-octave
//! curve shows is a three-regime plate rather than a two-point geometry —
//! `DECISIONS.md` 369, and `soundboard::ModalLobe` is what it is acted on with.
//!
//! The split is the model's, not a convenience. The geometry is **when** the
//! two capsules hear a source and nothing else: move the pair and every
//! interchannel delay moves with it, whatever the board is doing. `width` and
//! `diffuse_coherence` are **how much** of the resulting difference reaches the
//! output — a gain on the direct difference and a corner frequency on the
//! board's — and neither moves a delay. So the first stage can be inverted from
//! the recording alone, in closed physical terms, with no engine in the loop at
//! all; and the second cannot, because how coherent the engine's two channels
//! end up is a property of the whole chain (how much of the sound arrives
//! through the diffuse path, what the polarization spread is doing, where the
//! duplex sits) and this crate would have to mirror all of it to predict.
//! `estimate::directivity`'s header makes the same argument about the same
//! boundary, and this tool is where it is acted on twice.
//!
//! # The height is not fitted, and that is a measurement too
//!
//! The recordings' own documentation — `data/salamander/readme.txt`, Alexander
//! Holm's original note — says **"Two AKG c414 disposed in an AB position ~12cm
//! above the strings"**. That is the one number about this microphone pair that
//! is not an inference, and it is the height. It is therefore *held* at 0.12 m
//! and the span is fitted against it, which is also what makes the fit
//! well-posed: the delay curve constrains the ratio `span / height` far better
//! than either alone (see `estimate::mics`'s header), so one of the two has to
//! come from somewhere else, and here one of them is written down.
//!
//! The spacing is **not** written down — "AB position" is a family of setups,
//! not a distance — and it is what the inversion is for.
//!
//! # Two surfaces, not one
//!
//! Both the thirty recorded keys of `tuner/tests/stereo.rs` and the six phrases
//! of `renders/realism/REALISM.md`'s Columns S, because they disagree: a
//! geometry fitted on the notes alone improved their 63-125 Hz band and made
//! the phrases' worse than it started (`DECISIONS.md` 364). A note's coherence
//! is two capsules' view of one source; a phrase's is their view of several at
//! once.
//!
//! # Held-out material
//!
//! The fit runs at one velocity ([`FIT_VELOCITY`]). The check runs at the
//! others: the same keys struck in velocity layers the fit never saw, which for
//! this library are genuinely different *recordings* rather than the same take
//! scaled. A microphone pair does not move between takes, so a geometry fitted
//! at one dynamic that stops describing the image at another is a geometry that
//! fitted something else — the strike, the register, the layer's own noise
//! floor. That is the only honest test available for a stage with five
//! parameters and thirty keys, and it is reported whether it passes or not.

use std::path::PathBuf;

use rayon::prelude::*;

use piano_emulator::preset::{MicVoicing, ModalBand, Preset};
use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_emulator::types::Event;
use piano_tuner::cache;
use piano_tuner::estimate::mics::{
    fit_geometry, interchannel_lag, GeometryConfig, GeometryFit, KeyLag, LagConfig, MicGeometry,
};
use piano_tuner::estimate::melody;
use piano_tuner::numeric::NelderMead;
use piano_tuner::realism::{
    self, ChannelColumn, ChannelItem, ChannelShape, StereoColumn, StereoImage, StereoItem,
};
use piano_tuner::sampler::SAMPLER_VERSION;
use piano_tuner::realism::{Phrase, PHRASE_SET_VERSION};
use piano_tuner::sampler::engine_events;
use piano_tuner::{Audio, SampleLibrary, Sampler, SamplerEvent, TimedEvent, SAMPLE_RATE};

/// Velocity the fit is made at: the middle layer, the one every other stage-2
/// fit and both boards use.
const FIT_VELOCITY: u8 = 90;

/// Seconds of note the image is read over — `tuner/tests/stereo.rs`'s own, so
/// that what this tool minimises and what the gate asserts are the same window.
const RENDER_S: f64 = 3.0;

/// Silence before the strike, in samples: `realism::STEREO_PREROLL_SAMPLES`,
/// which is `tuner/tests/stereo.rs`'s own, so that what this tool minimises and
/// what the gate asserts are read over the same window. It carries the argument
/// for why the number is a whole number of engine blocks and what it cost when
/// it was not (`DECISIONS.md` 378).
const PREROLL: usize = realism::STEREO_PREROLL_SAMPLES;

/// The strike has to land on the first sample of the window, and an event takes
/// effect at the head of the block that contains it. Checked rather than
/// trusted, in both places that render this material.
const _: () = assert!(
    PREROLL % piano_emulator::types::BLOCK == 0,
    "the preroll must be a whole number of engine blocks or the window starts inside the note"
);

const PREROLL_S: f64 = PREROLL as f64 / 48_000.0;

/// The height the pair is held at, metres: `data/salamander/readme.txt`.
const DOCUMENTED_HEIGHT_M: f64 = 0.12;

/// First step of the pattern search, as a log-ratio: `exp(0.5)` is a factor of
/// 1.65, which crosses the whole legal range of every knob in five steps.
const SEARCH_STEP: f64 = 0.5;

/// Smallest improvement, in bars, the search will move a parameter for.
///
/// A hundredth of one band's take-to-take repeatability. Below that the
/// objective is reading the difference between two renders of the same
/// instrument rather than anything about the recording, and a search with no
/// such floor walks a flat direction to whichever bound it is pointed at —
/// which is exactly what the first run of this fit did with
/// `diffuse_coherence`, riding it from 5.0 to the ceiling for a total gain of
/// 0.08 bars spread over twenty accepted steps. A fitted preset should not
/// carry a number at its rail unless the rail is what the material asked for.
const SEARCH_EPSILON: f64 = 0.02;

/// Fraction of a band's bar the objective stops pushing at.
///
/// Not 1.0, which is where the gate's own threshold is, and the difference is
/// the whole reason the number exists: an objective that goes flat exactly at
/// the bar leaves a band **parked on it**, passing by a thousandth, and the
/// first version of this fit did precisely that — it walked 2-6 kHz to 0.072
/// against a bar of 0.072 and 6-12 kHz to 0.098 against 0.098. A band that
/// passes by nothing passes only this take. Half the bar is the margin the
/// material itself asks for: the bar is already `max(floor, scatter/sqrt(n))`
/// times an allowance, so half of it is about one floor — one take-to-take
/// disagreement of the recording with itself.
const PASS_MARGIN: f64 = 0.5;

/// Evaluations the simplex polish is allowed after the compass search.
const POLISH_EVALUATIONS: usize = 120;

/// Bars charged for every band of the **recorded-keys** surface left red.
///
/// The two surfaces disagree about 250-500 Hz by more than either's own floor
/// — the recording reads `−0.226` on thirty solo recorded keys and `+0.348` on
/// six phrases, against floors of 0.039 and 0.018 — so one coherence curve
/// cannot satisfy both band aggregates unless the engine's *within-band* energy
/// distribution matches the recording's, which it does not exactly. A summed
/// objective therefore splits the difference, and measured it splits it about
/// evenly: at its own optimum the notes are 1.13 bars out in that band and the
/// phrases 0.91, which is a good balance and still a **red gate**, because the
/// two bars differ (0.059 against 0.089) and one threshold falls on each side.
///
/// This says which surface is authoritative when they cannot both be had, and
/// the reasons are not "because it is the gate":
///
/// * it is the only surface made of keys the library **recorded**, with no
///   resampling anywhere in it — item 328's rule for every other fitted
///   quantity in this repository;
/// * it is **thirty** items against six, and it spans the whole compass rather
///   than whatever six phrases happen to play;
/// * it is **one source at a time**, which is what a per-source geometry is
///   about — item 364's own sentence, "a note's coherence is two capsules' view
///   of *one* source and a phrase's is their view of several at once".
///
/// Item 364's objection to fitting on the notes alone was empirical and
/// specific — it made the phrases *worse than they started* — and it does not
/// apply here: the phrase surface improves from 23.88 bars to 18.77 at the
/// constrained optimum, against 18.04 at the unconstrained one. **The price is
/// 0.73 bars on the phrase board and it is stated rather than hidden**; what it
/// buys is that both surfaces' 250-500 Hz columns pass at once, which neither
/// the unconstrained optimum nor the pair alone manages.
const RED_BAND_PENALTY: f64 = 10.0;

/// Step at which the search stops, as a log-ratio: half a per cent, which is
/// finer than the fourth decimal a preset writes and far finer than anything
/// the material can distinguish.
const SEARCH_FLOOR: f64 = 0.005;

/// Where the mode-controlled band's search starts, read off the recording's
/// own sixth-octave profile (`--stage profile`) rather than chosen.
///
/// The measured median `r0` is `+0.940` at 127 Hz, `+0.301` at 143, `+0.065`
/// at 160 and `-0.529` at 180: it crosses zero at about **165 Hz**, which is
/// the lower edge. It comes back through zero between 254 Hz (`-0.470`) and
/// 285 (`+0.448`), so **270 Hz** is the upper one. Inside, the mid/side ratio
/// is `-2.4` to `-4.0` dB, and since `soundboard::ModalLobe`'s lift *is* the
/// side-over-mid amplitude the band carries, that is a lift of `10^(3.5/20)`,
/// about **1.5** — a number read straight off the recording rather than
/// converted through a model of the diffuse taps, which is what it had to be
/// while the lobe was a gain on the difference (`DECISIONS.md` 379). All three
/// are starting points and all three move.
///
/// The lift's reading is **1.5 and the rail is 1.0** (`DECISIONS.md` 418), so
/// the start is the rail's own top: the recording asks for more side-over-mid
/// inside the band than a pair of capsules straddling a nodal line can carry
/// without one of them going through zero, and item 417's disposition is that
/// the part above the null is the reference session's placement rather than a
/// piano. Reading it and then not being able to have it is the finding, and the
/// start of the search is where it is clipped.
///
/// The lower edge's reading is **165 Hz and the search's own rail is 170**, for
/// the reason [`Knob::bounds`] gives: the estimator's spacing readback is
/// biased by the lobe's group delay, and 170 is where that bias comes inside
/// the gate's 20 % under the clamped lift. The start is the rail, not the
/// reading, and the five hertz between them is stated rather than absorbed.
const PROFILE_LOBE_START_HZ: f32 = 172.0;
const PROFILE_LOBE_END_HZ: f32 = 270.0;
const PROFILE_LOBE_LIFT: f32 = 0.99;

// ---------------------------------------------------------------------------
// Material
// ---------------------------------------------------------------------------

/// The engine-render cache for this invocation, opened once by [`run`] before
/// anything renders (`DECISIONS.md` 398). A `None` handle renders every time,
/// which is what every other caller of `renders::` gets and is the behaviour
/// this replaces.
static ENGINE_CACHE: std::sync::OnceLock<piano_tuner::renders::EngineRenders> =
    std::sync::OnceLock::new();

fn render_engine(preset: &Preset, key: u8, velocity: u8) -> Audio {
    // **Content-keyed on the whole input**, so a hit is the answer to exactly
    // this question or it is not read: the preset's own TOML bytes, the engine
    // fingerprinted by what it sounds like, and the material (`renders::`,
    // `DECISIONS.md` 398). Measured on this file's own throughput yardstick —
    // `--stage grid`, a fixed 65 candidates x (30 keys + 6 phrases) — **132.2 s
    // cold against 92.7 s warm**, 2.03 s per candidate against 1.43. That is
    // **1.43x** and not item 398's 8.3x, because only the *key* renders go
    // through here: the six phrases per candidate are a different shape from
    // `renders::NoteSpec` and are still rendered every time, and they are the
    // long ones. Caching them is the obvious next thing and it is a change to
    // `renders::`, not to this file.
    if let Some(cache) = ENGINE_CACHE.get() {
        return cache.note(
            preset,
            piano_tuner::renders::NoteSpec::new(key, velocity, PREROLL_S + RENDER_S, PREROLL),
        );
    }
    let events = [RenderEvent::new(
        PREROLL_S as f32,
        Event::NoteOn {
            key,
            vel: u16::from(velocity),
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

/// The recording of one key at one velocity, trimmed to its own onset and
/// cached exactly as `tuner/tests/stereo.rs` caches it — same fingerprint, so
/// the tool and the gate share one set of files on disk and can never be
/// measuring two different trims of the same take.
fn render_reference(
    data: &std::path::Path,
    sfz: &std::path::Path,
    key: u8,
    velocity: u8,
) -> Result<Audio, piano_tuner::Error> {
    let mut print = cache::Fingerprint::new();
    print
        .str("tests/stereo/reference")
        .u64(u64::from(SAMPLER_VERSION))
        .file(sfz)?
        .u64(u64::from(SAMPLE_RATE))
        .u64(u64::from(key))
        .u64(u64::from(velocity))
        .f64(RENDER_S);
    let path = cache::reference_dir(data).join(format!(
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
}

/// One phrase of `realism::phrase_set`, with the two reference images that do
/// not move when the engine does.
///
/// **The second surface the stereo table is printed on**, and it is in the fit
/// because it disagrees with the first. `renders/realism/REALISM.md`'s Columns S
/// are six phrases — mixtures of keys, most of them transposed, three of them
/// pedalled — and `tuner/tests/stereo.rs` is thirty keys struck alone. A
/// geometry moves them differently and can move them *oppositely*: a fit made
/// on the notes alone took the notes' 63-125 Hz band from 0.037 to 0.009 while
/// taking the phrases' from 0.149 to 0.254, because a note's coherence is the
/// two capsules' view of **one** source and a phrase's is their view of several
/// at once, and steepening the pan-to-position map improves the first while
/// spreading the second. Fitting both is the only way the answer is about the
/// microphones rather than about which page it was read from.
struct PhraseRow {
    phrase: Phrase,
    reference: StereoImage,
    alternate: StereoImage,
}

fn render_phrase(preset: &Preset, phrase: &Phrase) -> Audio {
    let (left, right) = render_to_buffer(
        preset,
        &engine_events::to_render_events(&phrase.events),
        phrase.duration_s as f32,
    );
    Audio::new(SAMPLE_RATE, vec![left, right]).expect("the engine renders stereo")
}

/// The engine's own pan axis, mirrored: `soundboard::pan_for_key`.
fn pan_for_key(key: u8) -> f64 {
    let position = f64::from(key.clamp(21, 108) - 21) / 87.0;
    (2.0 * position - 1.0) * 0.6
}

/// One recorded key's reference side, measured once and reused by every
/// candidate geometry.
struct Row {
    key: u8,
    label: String,
    pan: f64,
    /// The delay the recording carries at the fit velocity.
    lag: KeyLag,
    /// The same delay measured on the *other* velocity layer: a second
    /// recording of the same key, and so the floor under the delay term the
    /// same way the alternate image is the floor under the coherence term.
    alternate_lag: KeyLag,
    reference: StereoImage,
    alternate: StereoImage,
    /// The same two takes' **per-channel** spectral shape
    /// (`realism::channel_shape`), which is the board item 393 added and the
    /// third term of this stage's objective.
    reference_channels: ChannelShape,
    alternate_channels: ChannelShape,
}

// ---------------------------------------------------------------------------
// The driver
// ---------------------------------------------------------------------------

pub fn run(args: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
    let mut positional: Vec<String> = Vec::new();
    let mut out: Option<PathBuf> = None;
    let mut stages: Vec<String> = Vec::new();
    let mut holdout = true;
    let mut set: Option<MicVoicing> = None;
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" => out = Some(PathBuf::from(args.next().ok_or("--out needs a path")?)),
            "--stage" => stages.push(args.next().ok_or("--stage needs a name")?),
            "--no-holdout" => holdout = false,
            "--set" => {
                let text = args.next().ok_or("--set needs five comma-separated numbers")?;
                let n: Vec<f32> = text
                    .split(',')
                    .map(|f| f.trim().parse::<f32>())
                    .collect::<Result<Vec<f32>, _>>()?;
                if n.len() != 5 && n.len() != 8 {
                    return Err("--set takes spacing,height,span,width,coherence \
                                and optionally modal lo_hz,hi_hz,lift"
                        .into());
                }
                set = Some(MicVoicing {
                    spacing_m: n[0],
                    height_m: n[1],
                    span_m: n[2],
                    width: n[3],
                    diffuse_coherence: n[4],
                    modal: (n.len() == 8).then(|| ModalBand {
                        lo_hz: n[5],
                        hi_hz: n[6],
                        lift: n[7],
                    }),
                });
            }
            _ => positional.push(arg),
        }
    }
    let data = PathBuf::from(
        positional
            .first()
            .cloned()
            .unwrap_or_else(|| "data/salamander".into()),
    );
    let preset_path = PathBuf::from(
        positional
            .get(1)
            .cloned()
            .unwrap_or_else(|| "presets/salamander-c5.toml".into()),
    );
    const STAGES: [&str; 7] = [
        "geometry", "profile", "coherence", "image", "modal", "band", "grid",
    ];
    if let Some(unknown) = stages.iter().find(|s| !STAGES.contains(&s.as_str())) {
        return Err(format!(
            "unknown stage {unknown:?}; stages are {}",
            STAGES.join(", ")
        )
        .into());
    }
    let wants = |name: &str| set.is_none() && (stages.is_empty() || stages.iter().any(|s| s == name));

    let sfz = data.join("SalamanderGrandPiano-V3+20200602.sfz");
    if !sfz.exists() {
        eprintln!(
            "the reference piano is not here: {}\nrun data/fetch_salamander.sh first (707 MiB).",
            sfz.display()
        );
        std::process::exit(2);
    }
    // The engine-render cache, rooted beside the reference one. Opened before
    // anything renders, so every render in this invocation goes through it.
    let _ = ENGINE_CACHE.set(piano_tuner::renders::EngineRenders::at_data_root(&data));
    let base = Preset::load(&preset_path)?;
    let library = SampleLibrary::from_sfz(&sfz)?;
    let recorded = realism::RecordedKeys::from_library(&library)?;
    let layers = realism::VelocityLayers::from_library(&library)?;
    let alternate_velocity = layers.alternate(FIT_VELOCITY);

    println!(
        "mics: {} recorded keys at v{FIT_VELOCITY}, floor layer v{alternate_velocity}, base {}",
        recorded.keys().len(),
        preset_path.display()
    );

    // ---- the recording's side, measured once -----------------------------
    let lag_config = LagConfig::default();
    let rows: Vec<Row> = recorded
        .keys()
        .par_iter()
        .map(|&key| -> Result<Row, piano_tuner::Error> {
            let reference = render_reference(&data, &sfz, key, FIT_VELOCITY)?;
            let alternate = render_reference(&data, &sfz, key, alternate_velocity)?;
            let measured = interchannel_lag(
                &reference.channels[0],
                &reference.channels[1],
                f64::from(SAMPLE_RATE),
                &lag_config,
            )?;
            let other = interchannel_lag(
                &alternate.channels[0],
                &alternate.channels[1],
                f64::from(SAMPLE_RATE),
                &lag_config,
            )?;
            Ok(Row {
                key,
                label: realism::note_name(key),
                pan: pan_for_key(key),
                lag: KeyLag {
                    pan: pan_for_key(key),
                    lag_s: measured.lag_s,
                    confidence: measured.confidence,
                    ild_db: measured.ild_db,
                },
                alternate_lag: KeyLag {
                    pan: pan_for_key(key),
                    lag_s: other.lag_s,
                    confidence: other.confidence,
                    ild_db: other.ild_db,
                },
                reference: realism::stereo_image_of(&reference)?,
                alternate: realism::stereo_image_of(&alternate)?,
                reference_channels: realism::channel_shape_of(&reference)?,
                alternate_channels: realism::channel_shape_of(&alternate)?,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    // The census behind `LagConfig`'s defaults, kept so the band and window it
    // picks are re-checkable rather than asserted: every recorded key's delay
    // and PHAT peak in seven bands over four window lengths.
    //
    // ```sh
    // MIC_CENSUS=1 cargo run --release -p piano-tuner -- mics data/salamander
    // ```
    //
    // What it shows on this library (`DECISIONS.md` 361): peaks of 0.85-1.00
    // over 40-160 Hz and 0.11-0.44 over 200-4000 Hz, which is why the delay is
    // read in the bass and over the whole note rather than in the mid over an
    // onset window, as a direct path normally would be.
    if std::env::var("MIC_CENSUS").is_ok() {
        let bands = [
            (40.0, 160.0),
            (63.0, 250.0),
            (63.0, 500.0),
            (125.0, 500.0),
            (200.0, 1000.0),
            (200.0, 4000.0),
            (1000.0, 6000.0),
        ];
        let windows = [0.05, 0.2, 1.0, 3.0];
        for &window_s in &windows {
            println!("\n=== window {window_s} s");
            print!("| key | pan |");
            for b in &bands {
                print!(" {:.0}-{:.0} |", b.0, b.1);
            }
            println!();
            for r in &rows {
                let audio = render_reference(&data, &sfz, r.key, FIT_VELOCITY)?;
                print!("| {} | {:+.3} |", r.label, r.pan);
                for &band_hz in &bands {
                    let l = interchannel_lag(
                        &audio.channels[0],
                        &audio.channels[1],
                        f64::from(SAMPLE_RATE),
                        &LagConfig {
                            band_hz,
                            window_s,
                            ..LagConfig::default()
                        },
                    );
                    match l {
                        Ok(l) => print!(" {:+.2}/{:.2} |", 1e3 * l.lag_s, l.confidence),
                        Err(_) => print!(" — |"),
                    }
                }
                println!();
            }
        }
    }

    // ---- the phrase surface, measured once -------------------------------
    //
    // Cached under `realism-bench`'s own fingerprint, so this shares the files
    // `piano-tuner bench` already wrote and the two tools can never be scoring
    // two different renders of one phrase.
    let mut phrase_key = cache::Fingerprint::new();
    phrase_key
        .str("realism-bench/reference")
        .u64(u64::from(SAMPLER_VERSION))
        .file(&sfz)?
        .u64(u64::from(SAMPLE_RATE))
        .u64(u64::from(PHRASE_SET_VERSION));
    let phrase_dir = cache::reference_dir(&data);
    let phrases: Vec<PhraseRow> = realism::phrase_set()
        .into_par_iter()
        .map(|phrase| -> Result<PhraseRow, piano_tuner::Error> {
            let cached = |name: &str, events: &[TimedEvent]| -> Result<Audio, piano_tuner::Error> {
                let mut key = phrase_key;
                key.str(name).str(phrase.name).f64(phrase.duration_s);
                let path = phrase_dir.join(format!(
                    "realism-{}-{name}-{}.wav",
                    phrase.name,
                    key.hex()
                ));
                cache::audio(&path, || {
                    let mut sampler = Sampler::new(&sfz)?;
                    sampler.render(events, phrase.duration_s)
                })
            };
            let reference = cached("reference", &phrase.events)?;
            let alternate = cached("alt-layer", &layers.shift(&phrase.events))?;
            Ok(PhraseRow {
                reference: realism::stereo_image_of(&reference)?,
                alternate: realism::stereo_image_of(&alternate)?,
                phrase,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    println!("  and {} phrases of the scoreboard's own set", phrases.len());

    // ---- the melody surface, measured once -------------------------------
    //
    // `DECISIONS.md` 447. Two phrases through the recordings and their
    // neighbouring velocity layer, cached under the melody board's own
    // fingerprint so this tool and `piano-tuner melody` and `tests/melody.rs`
    // all read one set of files.
    let melody_board = MelodyBoard::measure(&data, &sfz, &layers, &recorded, &base)?;
    println!(
        "  and the melody board's {} columns on the Ode line and the recorded ladder \
(scored here: {})",
        melody::METRICS.len(),
        MELODY_METRICS.join(", ")
    );

    // ---- stage 1: the geometry -------------------------------------------
    let start = base.voicing.mics.map(|m| MicGeometry {
        spacing_m: f64::from(m.spacing_m),
        height_m: f64::from(m.height_m),
        span_m: f64::from(m.span_m),
    });
    let geometry = if wants("geometry") || (set.is_none() && stages.is_empty()) {
        let lags: Vec<KeyLag> = rows.iter().map(|r| r.lag).collect();
        let fit = fit_geometry(
            &lags,
            &GeometryConfig {
                height_m: DOCUMENTED_HEIGHT_M,
                ..GeometryConfig::default()
            },
        )?;
        print_geometry(&rows, &fit, start);
        fit.geometry
    } else {
        start.ok_or("this stage needs a preset that already has [voicing.mics]")?
    };

    // ---- stage 1b: the coherence *curve*, off the recording alone ---------
    //
    // `DECISIONS.md` 357 named the 125-500 Hz shortfall and said what was
    // missing was measured directivity. Six scoreboard bands are enough to
    // score a curve and not enough to see one, so this stage prints the
    // recording's own sixth-octave interchannel profile — and the engine's
    // beside it — over the same thirty keys, median and quartiles.
    if wants("profile") {
        let engine_preset = base.clone();
        let profile = |a: &Audio| realism::stereo_profile_of(a);
        let mut reference_rows: Vec<Vec<realism::StereoProfilePoint>> = Vec::new();
        let mut alternate_rows: Vec<Vec<realism::StereoProfilePoint>> = Vec::new();
        let mut engine_rows: Vec<Vec<realism::StereoProfilePoint>> = Vec::new();
        for r in &rows {
            reference_rows.push(profile(&render_reference(&data, &sfz, r.key, FIT_VELOCITY)?)?);
            alternate_rows.push(profile(&render_reference(
                &data,
                &sfz,
                r.key,
                alternate_velocity,
            )?)?);
            engine_rows.push(profile(&render_engine(&engine_preset, r.key, FIT_VELOCITY))?);
        }
        let stat = |rows: &[Vec<realism::StereoProfilePoint>],
                    i: usize,
                    pick: fn(&realism::StereoProfilePoint) -> f64|
         -> (f64, f64, f64) {
            let mut v: Vec<f64> = rows
                .iter()
                .filter_map(|p| p.get(i))
                .filter(|p| p.level_db > -60.0)
                .map(pick)
                .filter(|x| x.is_finite())
                .collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if v.is_empty() {
                return (f64::NAN, f64::NAN, f64::NAN);
            }
            let q = |f: f64| v[((v.len() - 1) as f64 * f).round() as usize];
            (q(0.25), q(0.5), q(0.75))
        };
        println!("\n=== the recording's interchannel profile, sixth-octave, {} recorded keys at v{FIT_VELOCITY}", rows.len());
        println!("| Hz | ref r0 (q1/med/q3) | ref M/S dB | alt r0 | alt M/S | engine r0 | engine M/S | n |");
        println!("|---:|---:|---:|---:|---:|---:|---:|---:|");
        for i in 0..reference_rows[0].len() {
            let hz = reference_rows[0][i].hz;
            let (rl, rm, rh) = stat(&reference_rows, i, |p| p.r0);
            let (_, sm, _) = stat(&reference_rows, i, |p| p.mid_side_db);
            let (_, am, _) = stat(&alternate_rows, i, |p| p.r0);
            let (_, asm, _) = stat(&alternate_rows, i, |p| p.mid_side_db);
            let (_, em, _) = stat(&engine_rows, i, |p| p.r0);
            let (_, esm, _) = stat(&engine_rows, i, |p| p.mid_side_db);
            let n = reference_rows
                .iter()
                .filter(|p| p.get(i).is_some_and(|p| p.level_db > -60.0))
                .count();
            println!(
                "| {hz:.0} | {rl:+.3}/{rm:+.3}/{rh:+.3} | {sm:+.2} | {am:+.3} | {asm:+.2} | {em:+.3} | {esm:+.2} | {n} |"
            );
        }
    }

    // ---- stage 2: how much of the difference reaches the output -----------
    //
    // Two searches, and the pair of them is the report. `coherence` holds the
    // geometry at the delays the recording measured and moves only the two
    // trims; `image` lets the geometry move as well, on the render score
    // alone. What separates them is `DECISIONS.md` 357 in numbers: how much of
    // the recording's band-by-band coherence a pair placed where the delays
    // say it is can reproduce, against how much a pair placed anywhere can.
    let measured = MicVoicing {
        spacing_m: geometry.spacing_m as f32,
        height_m: geometry.height_m as f32,
        span_m: geometry.span_m as f32,
        width: base.voicing.mics.map_or(1.0, |m| m.width),
        diffuse_coherence: base.voicing.mics.map_or(1.0, |m| m.diffuse_coherence),
        modal: base.voicing.mics.and_then(|m| m.modal),
    };
    let mut voicing = set.unwrap_or(measured);
    let mut trimmed = None;
    if wants("coherence") {
        println!("\ncoherence: width and diffuse_coherence, geometry held at the measured delays");
        let fit = search(
            &base,
            &rows,
            &phrases,
            &melody_board,
            measured,
            TRIM_KNOBS,
            false,
            &Feasible::default(),
        );
        trimmed = Some(fit);
        voicing = fit;
    }
    if wants("image") {
        println!("\nimage: all four, on the render score alone");
        let from_measured = search(
            &base,
            &rows,
            &phrases,
            &melody_board,
            voicing,
            IMAGE_KNOBS,
            false,
            &Feasible::default(),
        );
        let from_shipped = base.voicing.mics.map(|shipped| {
            search(
                &base,
                &rows,
                &phrases,
                &melody_board,
                MicVoicing {
                    height_m: DOCUMENTED_HEIGHT_M as f32,
                    ..shipped
                },
                IMAGE_KNOBS,
                false,
                &Feasible::default(),
            )
        });
        // The surface is smooth but not convex, so the search is started twice:
        // at the geometry the delays measured, and at the values it would
        // replace. A fit that cannot beat the shipped numbers from the shipped
        // numbers has not been given the chance to.
        let floor_ms = delay_floor_ms(&rows);
        let total = |v: MicVoicing| {
            gate_excess(&columns_for_voicing(&base, &rows, v, FIT_VELOCITY))
                + melody_excess_for(&base, &melody_board, v)
                + gate_excess(&phrase_columns_for_voicing(&base, &phrases, v))
                + delay_excess(&rows, geometry_of(&v), floor_ms)
        };
        let mut best = from_measured;
        let score = total(best);
        if let Some(other) = from_shipped {
            let value = total(other);
            println!(
                "  from the measured geometry: {score:.3} bars out; \
from the shipped values: {value:.3}"
            );
            if value < score {
                best = other;
            }
        }
        voicing = best;
    }
    // ---- the throughput yardstick ----------------------------------------
    //
    // `modal_grid` alone: a **fixed** batch of [`MODAL_GRID_LO_HZ`] x
    // [`MODAL_GRID_HI_HZ`] x [`MODAL_GRID_LIFT`] + 1 candidates, each one thirty
    // key renders and six phrase renders. It is the one part of the fit whose
    // cost is the same on every invocation and on every preset, which is what
    // makes it the thing to time a change to the search's parallelism against —
    // the compass and the simplex both change how many points they visit when
    // the surface moves, so a wall time taken on them measures the surface as
    // much as the machine. `DECISIONS.md` 392.
    if wants("grid") && !(wants("band") || wants("modal")) {
        let start = MicVoicing {
            modal: Some(voicing.modal.unwrap_or(ModalBand {
                lo_hz: PROFILE_LOBE_START_HZ,
                hi_hz: PROFILE_LOBE_END_HZ,
                lift: PROFILE_LOBE_LIFT,
            })),
            ..voicing
        };
        let feasible = Feasible::default();
        let clock = std::time::Instant::now();
        let best = modal_grid(&base, &rows, &phrases, &melody_board, start, &feasible);
        let elapsed = clock.elapsed().as_secs_f64();
        let cells = 1 + MODAL_GRID_LO_HZ.len() * MODAL_GRID_HI_HZ.len() * MODAL_GRID_LIFT.len();
        println!(
            "  grid: {cells} candidates x ({} keys + {} phrases) in {elapsed:.1} s \
({:.2} s per candidate, {:.0} renders/s)",
            rows.len(),
            phrases.len(),
            elapsed / cells as f64,
            (cells * (rows.len() + phrases.len())) as f64 / elapsed,
        );
        if let Some(b) = best.modal {
            println!("  grid best: {:.1}-{:.1} Hz x{:.3}", b.lo_hz, b.hi_hz, b.lift);
        }
        return Ok(());
    }

    // ---- stage 3: the board's mode-controlled band ------------------------
    //
    // Started from what the recording's own profile says rather than from the
    // middle of the range: the lower edge where the measured `r0` crosses
    // zero, the upper edge where it comes back, and a lift of two because the
    // measured mid/side ratio in between is about -3 dB. A search started at a
    // reading of the data is a search that can be said to have refined a
    // measurement; one started at a round number is a sweep with a nicer name.
    if wants("band") || wants("modal") {
        let joint = wants("band");
        let knobs = if joint { BAND_KNOBS } else { MODAL_KNOBS };
        if joint {
            println!(
                "\nband: the board's mode-controlled band and the two trims together, \
geometry held"
            );
        } else {
            println!("\nmodal: the board's mode-controlled band, geometry and trims held");
        }
        let read_off = ModalBand {
            lo_hz: PROFILE_LOBE_START_HZ,
            hi_hz: PROFILE_LOBE_END_HZ,
            lift: PROFILE_LOBE_LIFT,
        };
        let mut start = MicVoicing {
            modal: Some(voicing.modal.unwrap_or(read_off)),
            ..voicing
        };
        // A base preset written before item 418's rail carries a lift above it
        // (the shipped one is 2.124), and a search started outside its own
        // bounds spends its first steps walking back inside them. Every axis is
        // brought into range before the grid, which for the three axes already
        // inside is the identity.
        for knob in knobs {
            let value = knob.get(&start);
            let (lo, hi) = knob.bounds();
            knob.set(&mut start, value.clamp(lo, hi));
        }
        // **Relaxed first, then constrained** — the standard order for a
        // penalty method, and here it is not a formality. `RED_BAND_PENALTY`
        // is ten bars, which is larger than the whole continuous variation of
        // the objective, so under it the surface is dominated by an integer
        // count of red bands and a coarse grid can no longer tell one basin
        // from another: run this way round from the start, the grid picks
        // 281-570 Hz x1.7 and the compass sits down at 40.97 bars. Item 363's
        // own objective finds the basin; the constraint then decides where in
        // it to stop.
        let feasible = Feasible::default();
        let coarse = modal_grid(&base, &rows, &phrases, &melody_board, start, &feasible);
        println!("  (relaxed: item 363's objective, to find the basin)");
        let relaxed = search(
            &base,
            &rows,
            &phrases,
            &melody_board,
            coarse,
            knobs,
            false,
            &feasible,
        );
        // A penalty method has to start somewhere it can move. See [`Feasible`].
        let from = feasible.take().unwrap_or(relaxed);
        match (from.modal, relaxed.modal) {
            (Some(f), Some(r)) => println!(
                "  (constrained: every red band of the gate charged {RED_BAND_PENALTY:.0} bars; \
starting at the best band the relaxed pass saw pass it, {:.1}-{:.1} Hz x{:.3}, \
rather than at its own optimum {:.1}-{:.1} Hz x{:.3})",
                f.lo_hz, f.hi_hz, f.lift, r.lo_hz, r.hi_hz, r.lift
            ),
            _ => println!("  (constrained: no band passed the gate in the relaxed pass)"),
        }
        let constrained = search(
            &base,
            &rows,
            &phrases,
            &melody_board,
            from,
            knobs,
            true,
            &feasible,
        );
        let banded = modal_refine(&base, &rows, &phrases, &melody_board, constrained);
        voicing = choose_band_or_none(&base, &rows, &phrases, &melody_board, banded, &feasible);
    }

    if let (Some(trim), true) = (trimmed, wants("image")) {
        let held = gate_excess(&columns_for_voicing(&base, &rows, trim, FIT_VELOCITY));
        let free = gate_excess(&columns_for_voicing(&base, &rows, voicing, FIT_VELOCITY));
        println!(
            "\nthe cost of respecting the measured delays: {held:.3} bars out with the \
geometry held, {free:.3} with it free"
        );
    }

    // ---- what it bought ---------------------------------------------------
    let mut fitted = base.clone();
    fitted.voicing.mics = Some(voicing);
    fitted.validate()?;

    println!("\n[voicing.mics] fitted");
    println!("  spacing_m         {:.4}", voicing.spacing_m);
    println!("  height_m          {:.4}   (held: readme.txt)", voicing.height_m);
    println!("  span_m            {:.4}", voicing.span_m);
    println!("  width             {:.4}", voicing.width);
    println!("  diffuse_coherence {:.4}", voicing.diffuse_coherence);
    match voicing.modal {
        None => println!("  [modal]           absent — the diffuse coherence alone"),
        Some(b) => println!(
            "  [modal] lo_hz {:.1}  hi_hz {:.1}  lift {:.4}",
            b.lo_hz, b.hi_hz, b.lift
        ),
    }
    let floor_ms = delay_floor_ms(&rows);
    println!(
        "  against the recording's own delays: {:.3} ms weighted RMS, {:.2} bars out \
(take-to-take floor {:.3} ms; the delay inversion's own best {:.3} ms; a pair that is not \
there {:.3} ms; the preset this replaces {:.3} ms)",
        delay_residual(&rows, geometry_of(&voicing)),
        delay_excess(&rows, geometry_of(&voicing), floor_ms),
        floor_ms,
        delay_residual(&rows, geometry),
        delay_residual(&rows, MicGeometry::new(0.001, 1.0, 1.0)),
        base.voicing
            .mics
            .map_or(f64::NAN, |m| delay_residual(&rows, geometry_of(&m))),
    );

    let (before, before_channels) = boards_for(&base, &rows, FIT_VELOCITY);
    let (after, after_channels) = boards_for(&fitted, &rows, FIT_VELOCITY);
    println!(
        "\nSTEREO columns, {} recorded keys at v{FIT_VELOCITY} — before ({} red, {:.2} bars out, summed |err| {:.3}):\n{}",
        rows.len(),
        before.iter().filter(|c| !c.pass).count(),
        gate_excess(&before),
        band_error(&before),
        realism::stereo_report(&before)
    );
    println!(
        "STEREO columns, {} recorded keys at v{FIT_VELOCITY} — after ({} red, {:.2} bars out, summed |err| {:.3}):\n{}",
        rows.len(),
        after.iter().filter(|c| !c.pass).count(),
        gate_excess(&after),
        band_error(&after),
        realism::stereo_report(&after)
    );
    println!(
        "PER-CHANNEL columns, the same renders — before ({} red, {:.2} bars out) and after ({} red, {:.2} bars out):\n{}\n{}",
        before_channels.iter().filter(|c| !c.pass).count(),
        channel_excess(&before_channels),
        after_channels.iter().filter(|c| !c.pass).count(),
        channel_excess(&after_channels),
        realism::channel_report(&before_channels),
        realism::channel_report(&after_channels)
    );

    let melody_before = melody_board.columns(&base);
    let melody_after = melody_board.columns(&fitted);
    println!(
        "MELODY columns, the Ode line and the recorded ladder — before ({} of {} red, {:.2} bars out) \
and after ({} red, {:.2} bars out).  Scored here: {}.\n{}\n{}",
        melody_before
            .iter()
            .filter(|c| MELODY_METRICS.contains(&c.metric) && !c.pass)
            .count(),
        MELODY_METRICS.len(),
        melody_excess(&melody_before),
        melody_after
            .iter()
            .filter(|c| MELODY_METRICS.contains(&c.metric) && !c.pass)
            .count(),
        melody_excess(&melody_after),
        MELODY_METRICS.join(", "),
        melody::report(&melody_before),
        melody::report(&melody_after),
    );

    let phrases_before = phrase_columns(&base, &phrases);
    let phrases_after = phrase_columns(&fitted, &phrases);
    println!(
        "STEREO columns, the scoreboard's six phrases — before ({} red, {:.2} bars out) \
and after ({} red, {:.2} bars out):\n{}\n{}",
        phrases_before.iter().filter(|c| !c.pass).count(),
        gate_excess(&phrases_before),
        phrases_after.iter().filter(|c| !c.pass).count(),
        gate_excess(&phrases_after),
        realism::stereo_report(&phrases_before),
        realism::stereo_report(&phrases_after)
    );

    if holdout {
        for velocity in held_out_velocities(&layers) {
            let held = held_out_rows(&data, &sfz, &layers, &rows, velocity)?;
            let columns = columns_for(&fitted, &held, velocity);
            let base_columns = columns_for(&base, &held, velocity);
            println!(
                "HELD OUT — v{velocity}, never fitted ({} red after, {} before; {:.2} bars out after, {:.2} before; summed |err| {:.3} after, {:.3} before):\n{}",
                columns.iter().filter(|c| !c.pass).count(),
                base_columns.iter().filter(|c| !c.pass).count(),
                gate_excess(&columns),
                gate_excess(&base_columns),
                band_error(&columns),
                band_error(&base_columns),
                realism::stereo_report(&columns)
            );
        }
    }

    if let Some(path) = out {
        fitted.save(&path)?;
        println!("wrote {}", path.display());
    } else {
        println!("(no --out: nothing written)");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------

/// One field of `[voicing.mics]`, with the engine's own bounds on it.
///
/// The bounds are `soundboard::MAX_MIC_SPACING_M`, `MIC_HEIGHT_M`, `MIC_SPAN_M`,
/// `MIC_WIDTH` and `MIC_DIFFUSE_COHERENCE`, pulled in a hair so that a fitted
/// preset is strictly inside what `Preset::validate` accepts rather than on its
/// edge — a value written at the rail reads as a fit that wanted to keep going,
/// and this way it is one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Knob {
    Spacing,
    Span,
    Width,
    Coherence,
    /// The mode-controlled band's bottom edge, `[voicing.mics.modal].lo_hz`.
    ModalLo,
    /// Its top edge.
    ModalHi,
    /// How much more difference than sum the pair sees inside it.
    ModalLift,
}

/// The trims: what a pair does with its difference, not where it is.
const TRIM_KNOBS: &[Knob] = &[Knob::Width, Knob::Coherence];

/// Everything but the height, which `data/salamander/readme.txt` states.
const IMAGE_KNOBS: &[Knob] = &[Knob::Spacing, Knob::Span, Knob::Width, Knob::Coherence];

/// The mode-controlled band alone, with the pair held where the delays put it.
///
/// A separate stage rather than three more axes on `IMAGE_KNOBS`, and for the
/// reason `DECISIONS.md` 362 gives: the geometry is pinned by a *measurement
/// the engine is not in* — the recording's own interchannel delays — and the
/// band is not. Letting the search trade one against the other would spend the
/// pinned half to buy the free one, which is exactly what the free
/// four-parameter coherence fit did when it walked the spacing to 0.66 m.
const MODAL_KNOBS: &[Knob] = &[Knob::ModalLo, Knob::ModalHi, Knob::ModalLift];

/// The band **and** the two trims, moved together.
///
/// The `coherence` and `modal` stages were separable while the lobe was a gain
/// on the board's *difference*: `width` scaled the direct difference, the
/// coherence pole scaled the diffuse one, and the lobe scaled a third signal
/// that was orthogonal to both. Since `DECISIONS.md` 379 the lobe is an
/// anti-phase copy of the **sum**, and inside its band the side is
/// `width/2 · geometry + lift · sum` — one signal built out of two knobs, so
/// the two are no longer separable and a stage that pretends they are searches
/// a ridge one axis at a time. Measured, that is not a nicety: run as two
/// stages, the trims are fitted with no band present at all, which is a fit to
/// an instrument that is not going to ship.
///
/// The geometry stays out for the reason `MODAL_KNOBS` gives — it is pinned by
/// a measurement the engine is not in — so this is the trims and the band and
/// nothing else.
const BAND_KNOBS: &[Knob] = &[
    Knob::Width,
    Knob::Coherence,
    Knob::ModalLo,
    Knob::ModalHi,
    Knob::ModalLift,
];

impl Knob {
    fn bounds(self) -> (f64, f64) {
        match self {
            Knob::Spacing => (0.04, 0.99),
            Knob::Span => (0.5, 1.5),
            Knob::Width => (0.05, 1.99),
            Knob::Coherence => (0.26, 7.9),
            // The engine bounds the edges at 40 Hz and 2 kHz; the search is
            // held inside the range the profile actually resolves — above the
            // lowest recorded fundamental and below the band where every
            // measured value is already within 0.2 of zero.
            //
            // **The lower bound is 170 Hz and it is a measurement of the
            // spacing-readback gate, re-taken under item 418's rail.** The lobe
            // is not common to the two channels, so its group delay enters the
            // phase-transform delay reading `tuner/tests/mics.rs`'s
            // `the_estimator_reads_back_a_spacing_the_engine_was_given` is built
            // on. Item 395 put that boundary at about **225 Hz** and it is what
            // emptied the M12 feasible set; swept again under the clamped lobe
            // (`where_the_bands_lower_edge_starts_biasing_the_reading`, the same
            // three spacings and the same 20 % tolerance) the worst reading is
            // **+8 % at 170 Hz, +13 % at 165, +24 % at 160 and +40 % at 155** at
            // a lift of 0.99, and +14 / +18 / +23 / +32 % at 0.5. So the rail
            // moves fifty-five hertz *down*: a weaker lobe biases the reading
            // less, which is the first thing item 418's clamp bought back and
            // the reason its refit has room item 395's did not.
            Knob::ModalLo => (170.0, 400.0),
            Knob::ModalHi => (200.0, 1_500.0),
            // **Under the rail, and a hair inside it** (`DECISIONS.md` 418).
            // `soundboard::MIC_MODAL_LIFT` is clamped at 1.0 — the lift is the
            // side-over-mid amplitude the band carries, so one is where
            // `1 − g` reaches zero and one loudspeaker is nulled outright, and
            // above it `1 − g` changes sign and which speaker carries the
            // fundamental flips with pitch (item 392's two unity crossings at
            // 213.0 and 359.6 Hz). The search stops at 0.99 for the reason
            // every other knob here stops short of its rail: a fitted number
            // written at a bound is a fit that wanted to keep going.
            Knob::ModalLift => (0.05, 0.99),
        }
    }

    fn get(self, v: &MicVoicing) -> f64 {
        f64::from(match self {
            Knob::Spacing => v.spacing_m,
            Knob::Span => v.span_m,
            Knob::Width => v.width,
            Knob::Coherence => v.diffuse_coherence,
            Knob::ModalLo => v.modal.map_or(f32::NAN, |b| b.lo_hz),
            Knob::ModalHi => v.modal.map_or(f32::NAN, |b| b.hi_hz),
            Knob::ModalLift => v.modal.map_or(f32::NAN, |b| b.lift),
        })
    }

    fn set(self, v: &mut MicVoicing, value: f64) {
        let (lo, hi) = self.bounds();
        let value = value.clamp(lo, hi) as f32;
        match self {
            Knob::Spacing => v.spacing_m = value,
            Knob::Span => v.span_m = value,
            Knob::Width => v.width = value,
            Knob::Coherence => v.diffuse_coherence = value,
            Knob::ModalLo => {
                if let Some(b) = &mut v.modal {
                    // The engine rejects a crossed band, so the search may not
                    // propose one: an edge that would pass its partner is held
                    // a semitone short of it.
                    b.lo_hz = value.min(b.hi_hz / 1.06);
                }
            }
            Knob::ModalHi => {
                if let Some(b) = &mut v.modal {
                    b.hi_hz = value.max(b.lo_hz * 1.06);
                }
            }
            Knob::ModalLift => {
                if let Some(b) = &mut v.modal {
                    b.lift = value;
                }
            }
        }
    }
}

/// A coarse grid over the mode-controlled band before the compass refines it.
///
/// **Because the surface has more than one basin and the compass finds the
/// nearest one.** Started at the reading of the recording's own profile
/// ([`PROFILE_LOBE_START_HZ`] and its two companions) the compass walks the
/// band *wider and weaker* — 208 Hz to 875 Hz at a lift of 1.67, 29.66 bars out
/// — and stops there, while a narrower, stronger band at 190-330 Hz and 2.4
/// scores **24.39** on the same objective. The two are not on one slope: the
/// lower edge and the lift trade almost exactly against each other, so a search
/// that moves one axis at a time sees a ridge between the basins and reads it
/// as a wall. Sixty-four cells, log-spaced over the range the profile supports,
/// cost about a tenth of the compass they precede and remove the question.
///
/// The cells are the search's own bounds pulled in to where the recording's
/// sixth-octave profile has structure: it crosses zero going down between 127
/// and 180 Hz and comes back between 254 and 320, so the lower edge is gridded
/// over 150-281 Hz and the upper over 280-815; the lift over 1.2-3.4, which
/// spans "the diffuse taps as they are" to "three times what they are".
///
/// **The lift row is the rail's** (`DECISIONS.md` 418). It used to span 1.2-3.4
/// — "the diffuse taps as they are" to "three times what they are" — and every
/// cell of it is now illegal: `soundboard::MIC_MODAL_LIFT` stops at 1.0 because
/// that is where `1 − g` reaches zero. The row spans the legal range instead,
/// from a quarter of the null to a hundredth under it, and the top of it is
/// where the whole of the pair energy a lobe can manufacture now lives:
/// `10 log10(1 + g²)` is **+3.01 dB** at `g = 1` against the +6.18 item 392
/// convicted.
const MODAL_GRID_LO_HZ: [f32; 4] = [172.0, 200.0, 235.0, 281.0];
const MODAL_GRID_HI_HZ: [f32; 4] = [280.0, 400.0, 570.0, 815.0];
const MODAL_GRID_LIFT: [f32; 4] = [0.25, 0.5, 0.75, 0.99];

fn modal_grid(
    preset: &Preset,
    rows: &[Row],
    phrases: &[PhraseRow],
    melody_board: &MelodyBoard,
    start: MicVoicing,
    feasible: &Feasible,
) -> MicVoicing {
    // Item 363's objective, not the penalised one: the grid's job is to choose
    // a basin, and a ten-bar penalty turns the surface into an integer count of
    // red bands that no coarse grid can read. See the modal stage's own comment.
    let objective = relaxed_objective(preset, rows, phrases, melody_board, feasible);
    // **Every cell at once.** The grid is a fixed, independent batch and
    // nothing in it reads anything a sibling wrote, so it is one `par_iter`
    // rather than sixty-five serial evaluations of a thirty-way one. See
    // [`batch`].
    let mut cells: Vec<MicVoicing> = vec![start];
    for &lo_hz in &MODAL_GRID_LO_HZ {
        for &hi_hz in &MODAL_GRID_HI_HZ {
            for &lift in &MODAL_GRID_LIFT {
                cells.push(MicVoicing {
                    modal: Some(ModalBand { lo_hz, hi_hz, lift }),
                    ..start
                });
            }
        }
    }
    let scored = batch(&cells, &objective);
    println!("  grid: the profile's own reading scores {:.3} bars out", scored[0]);
    let mut best = start;
    let mut score = scored[0];
    for (candidate, &value) in cells.iter().zip(&scored).skip(1) {
        if value < score {
            score = value;
            best = *candidate;
            let b = candidate.modal.expect("a grid cell always has a band");
            println!(
                "  grid: {:.0}-{:.0} Hz x{:.1} -> {score:.3} bars out",
                b.lo_hz, b.hi_hz, b.lift
            );
        }
    }
    best
}

/// Score a whole batch of candidates at once.
///
/// **The one structural thing that made this phase's fits affordable**
/// (`DECISIONS.md` 392). Every objective evaluation here is already a thirty-way
/// `par_iter` over the recorded keys and a six-way one over the phrases, and
/// measured that reaches **3.6 cores of a 14-core machine**: thirty renders is
/// a short queue with a long tail, and the six phrases at the end of it are a
/// six-way queue. The searches around it, on the other hand, all evaluate
/// *independent batches* — sixty-five grid cells, six compass probes, twenty-six
/// diagonal neighbours — and running the batch in parallel fills the machine
/// from the outside instead. Rayon's work stealing composes the two: the outer
/// iterator hands out candidates, the inner ones subdivide whatever is left.
fn batch(candidates: &[MicVoicing], objective: &(impl Fn(MicVoicing) -> f64 + Sync)) -> Vec<f64> {
    candidates.par_iter().map(|&v| objective(v)).collect()
}

/// Item 363's objective — the two surfaces and the delay term, no penalty —
/// noting every candidate that passes the gate outright. See [`Feasible`].
fn relaxed_objective<'a>(
    preset: &'a Preset,
    rows: &'a [Row],
    phrases: &'a [PhraseRow],
    melody_board: &'a MelodyBoard,
    feasible: &'a Feasible,
) -> impl Fn(MicVoicing) -> f64 + Sync + 'a {
    let floor_ms = delay_floor_ms(rows);
    move |v: MicVoicing| -> f64 {
        let mut candidate = preset.clone();
        candidate.voicing.mics = Some(v);
        let (notes, channels) = boards_for(&candidate, rows, FIT_VELOCITY);
        let melody = melody_board.columns(&candidate);
        let value = gate_excess(&notes)
            + channel_excess(&channels)
            + pair_excess(&channels)
            + melody_excess(&melody)
            + gate_excess(&phrase_columns_for_voicing(preset, phrases, v))
            + delay_excess(rows, geometry_of(&v), floor_ms);
        if melody
            .iter()
            .all(|c| !MELODY_METRICS.contains(&c.metric) || c.pass)
            && notes.iter().all(|c| c.items == 0 || c.pass)
            && channels
                .iter()
                .all(|c| c.items == 0 || (c.pass && (c.pair_pass || !modal_channel_band(c))))
        {
            feasible.offer(v, value);
        }
        value
    }
}

/// The best point with **no red band on the gate** that any search has seen.
///
/// Every candidate the relaxed pass evaluates already has the gate's columns
/// computed, so noticing which of them are feasible costs nothing — and it is
/// what makes the constrained pass work at all. The relaxed optimum is a narrow
/// band at 212-225 Hz with a lift of 4.81, and *every* point within ten per
/// cent of it on all three axes is red: a penalty method started there cannot
/// move, which is the ordinary failure of a penalty method started at an
/// infeasible point. Started instead at the best feasible point the relaxed
/// pass walked past, it has somewhere to go.
///
/// A `Mutex` rather than a `Cell` because [`batch`] evaluates candidates on
/// every core at once; the lock is taken once per *render set*, which is
/// several seconds of work, so it costs nothing measurable. Ties are broken by
/// the candidate's own numbers rather than by arrival order, so which thread
/// gets there first cannot change the answer.
#[derive(Default)]
struct Feasible {
    best: std::sync::Mutex<Option<(MicVoicing, f64)>>,
}

impl Feasible {
    fn offer(&self, voicing: MicVoicing, score: f64) {
        let mut slot = self.best.lock().expect("the feasible set is never poisoned");
        let better = match *slot {
            None => true,
            Some((held, s)) => (score, key_of(voicing)) < (s, key_of(held)),
        };
        if better {
            *slot = Some((voicing, score));
        }
    }

    fn take(&self) -> Option<MicVoicing> {
        self.best
            .lock()
            .expect("the feasible set is never poisoned")
            .map(|(v, _)| v)
    }
}

/// A total order on a candidate, used only to break ties deterministically.
fn key_of(v: MicVoicing) -> [f32; 8] {
    let b = v.modal.unwrap_or(ModalBand {
        lo_hz: 0.0,
        hi_hz: 0.0,
        lift: 0.0,
    });
    [
        v.spacing_m,
        v.height_m,
        v.span_m,
        v.width,
        v.diffuse_coherence,
        b.lo_hz,
        b.hi_hz,
        b.lift,
    ]
}

/// The mode-controlled band's objective: item 363's two surfaces and its delay
/// term, with every red band of the recorded-keys surface charged
/// [`RED_BAND_PENALTY`] on top.
fn modal_objective<'a>(
    preset: &'a Preset,
    rows: &'a [Row],
    phrases: &'a [PhraseRow],
    melody_board: &'a MelodyBoard,
) -> impl Fn(MicVoicing) -> f64 + Sync + 'a {
    let floor_ms = delay_floor_ms(rows);
    move |v: MicVoicing| -> f64 {
        let mut candidate = preset.clone();
        candidate.voicing.mics = Some(v);
        let (notes, channels) = boards_for(&candidate, rows, FIT_VELOCITY);
        // Both boards of the authoritative surface are charged, and for the
        // same reason: a red band on either is a statement the recording makes
        // that the engine does not. The per-channel board is charged only in
        // the two bands the mode-controlled lobe is allowed to act in — every
        // other band of it is the pan-pot's and the pair geometry's, which this
        // stage cannot move and must not be scored on. See
        // [`MODAL_CHANNEL_BANDS`].
        //
        // The melody board's two stereo columns are charged the same way and
        // for the same reason (`DECISIONS.md` 447): they are gates, they are
        // this stage's own to move, and a red one is a statement the recording
        // makes about a *tune* that the engine does not.
        let melody = melody_board.columns(&candidate);
        let reds = notes.iter().filter(|c| c.items > 0 && !c.pass).count()
            + channels
                .iter()
                .filter(|c| c.items > 0 && modal_channel_band(c) && !(c.pass && c.pair_pass))
                .count()
            + melody
                .iter()
                .filter(|c| MELODY_METRICS.contains(&c.metric) && !c.pass)
                .count();
        gate_excess(&notes)
            + channel_excess(&channels)
            + pair_excess(&channels)
            + melody_excess(&melody)
            + RED_BAND_PENALTY * reds as f64
            + gate_excess(&phrase_columns_for_voicing(preset, phrases, v))
            + delay_excess(rows, geometry_of(&v), floor_ms)
    }
}

/// The bands of the per-channel board the mode-controlled lobe is charged for
/// being red in: the two it lives in.
///
/// `MIC_MODAL_HZ` lets a band be declared anywhere from 40 Hz to 2 kHz, but
/// the fit's own bounds and the recording's profile put it inside 125-500, and
/// outside that the per-channel column is measuring the pan-pot and the pair
/// geometry — neither of which this stage's three knobs can move. Charging ten
/// bars for a band nothing here can reach would make the objective a constant
/// plus noise.
fn modal_channel_band(c: &ChannelColumn) -> bool {
    c.lo_hz >= 125.0 && c.hi_hz <= 500.0
}

/// **The one candidate no search over `[voicing.mics.modal]`'s three knobs can
/// reach: the section with no band in it at all** (`DECISIONS.md` 451).
///
/// Every stage above this one moves `lo_hz`, `hi_hz` and `lift` inside their
/// bounds, and `Knob::set` on a `None` band is a no-op by construction, so the
/// grid, the compass, the simplex and the diagonal refinement between them can
/// make a band narrow, wide, weak or strong and can never make it *absent*.
/// That is not a small gap in the search space. `soundboard::ModalLobe` is a
/// twelfth-order cascade whose response `B` is complex, `L = m(1 + B)` and
/// `R = m(1 − B)`, so **a lift of zero is not the same instrument as no band**:
/// the shipped grid's weakest cell is 0.25 and even at 0.05 — the knob's own
/// floor — the cascade's phase is still turning inside the band. Absence is a
/// corner of the feasible set and a corner has to be evaluated, not
/// approached.
///
/// So it is evaluated, on **the same constrained objective** the banded point
/// was chosen with, and with its own two trims refitted first: `width` and
/// `diffuse_coherence` are fitted *together* with the band since item 379
/// (`BAND_KNOBS` says why), so holding the banded point's trims on a bandless
/// pair would be scoring an instrument nobody would ship. Whichever of the two
/// scores lower is what this stage returns, and both scores are printed, which
/// is the frontier this comparison is worth quoting from.
fn choose_band_or_none(
    preset: &Preset,
    rows: &[Row],
    phrases: &[PhraseRow],
    melody_board: &MelodyBoard,
    banded: MicVoicing,
    feasible: &Feasible,
) -> MicVoicing {
    println!("\nband or no band: the corner of the feasible set the three knobs cannot reach");
    let bare = search(
        preset,
        rows,
        phrases,
        melody_board,
        MicVoicing {
            modal: None,
            ..banded
        },
        TRIM_KNOBS,
        true,
        feasible,
    );
    let objective = modal_objective(preset, rows, phrases, melody_board);
    let scored = batch(&[banded, bare], &objective);
    let (with, without) = (scored[0], scored[1]);
    let describe = |v: MicVoicing, score: f64| {
        format!(
            "width {:.4} x coherence {:.4}, {} -> {score:.3} bars out",
            v.width,
            v.diffuse_coherence,
            v.modal.map_or_else(
                || "no [modal] band".to_string(),
                |b| format!("{:.1}-{:.1} Hz x{:.4}", b.lo_hz, b.hi_hz, b.lift)
            )
        )
    };
    println!("  with a band:    {}", describe(banded, with));
    println!("  with none:      {}", describe(bare, without));
    if without < with {
        println!(
            "  -> the band is removed: {:.3} bars better on the same objective",
            with - without
        );
        bare
    } else {
        println!(
            "  -> the band is kept: absence costs {:.3} bars on the same objective",
            without - with
        );
        banded
    }
}

/// Multiplicative steps the diagonal refinement tries, largest first.
const MODAL_REFINE_STEPS: [f64; 5] = [1.25, 1.12, 1.06, 1.03, 1.015];

/// The last word on the mode-controlled band: a **diagonal** pattern search —
/// every combination of up, down and stay on all three knobs at once.
///
/// [`search`] moves one axis at a time and then hands a simplex the diagonals.
/// On this surface neither is enough, and the reason is that the surface has
/// *steps* in it: a band enters or leaves the readable set as the engine's own
/// levels move, so the objective is piecewise-flat at the scale the searches
/// finish at. Measured — the compass and the simplex together settle at
/// 202-252 Hz x3.48 and 20.467 bars, and 205-245 Hz x3.80, a third of a
/// semitone away on two axes and 9 % away on the third, scores **20.37** and
/// takes the notes' gate from one red to none. Twenty-six neighbours at four
/// step sizes is a maximal positive basis (Torczon 1997): it cannot be caught
/// by a ridge that is diagonal in any of the three planes, which is the one
/// failure a compass has by construction, and it costs about a hundred renders.
fn modal_refine(
    preset: &Preset,
    rows: &[Row],
    phrases: &[PhraseRow],
    melody_board: &MelodyBoard,
    start: MicVoicing,
) -> MicVoicing {
    let objective = modal_objective(preset, rows, phrases, melody_board);
    let mut best = start;
    let mut score = objective(best);
    for step in MODAL_REFINE_STEPS {
        loop {
            // The twenty-six neighbours are a batch, and a batch is one
            // `par_iter` (see [`batch`]). Taking the *best* of the twenty-six
            // rather than the first improvement is the textbook form of a
            // pattern search anyway — it is what makes the positive basis the
            // thing that decides the step, not the order the loops happen to
            // be nested in.
            let mut neighbours = Vec::with_capacity(26);
            for lo in [1.0 / step, 1.0, step] {
                for hi in [1.0 / step, 1.0, step] {
                    for lift in [1.0 / step, 1.0, step] {
                        if lo == 1.0 && hi == 1.0 && lift == 1.0 {
                            continue;
                        }
                        let mut candidate = best;
                        Knob::ModalLo.set(&mut candidate, Knob::ModalLo.get(&best) * lo);
                        Knob::ModalHi.set(&mut candidate, Knob::ModalHi.get(&best) * hi);
                        Knob::ModalLift.set(&mut candidate, Knob::ModalLift.get(&best) * lift);
                        neighbours.push(candidate);
                    }
                }
            }
            let scored = batch(&neighbours, &objective);
            let Some((i, &value)) = scored
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.total_cmp(b.1))
            else {
                break;
            };
            if value >= score - SEARCH_EPSILON {
                break;
            }
            score = value;
            best = neighbours[i];
        }
        println!(
            "  refine x{step:.4}: {} -> {score:.3} bars out",
            best.modal.map_or_else(
                || "absent".to_string(),
                |b| format!("{:.1}-{:.1} Hz x{:.4}", b.lo_hz, b.hi_hz, b.lift)
            )
        );
    }
    best
}

/// Bounded pattern search over the free knobs, in log coordinates.
///
/// A pattern search rather than a simplex, for one reason: the objective is a
/// median over thirty renders of a *readable* set of bands, and a band can
/// enter or leave that set as the engine's own levels move, which puts small
/// steps in the surface. Nelder-Mead reads a step as curvature and collapses
/// onto it; a compass search either finds a better point along an axis or
/// halves its step, and cannot be fooled into converging on a discontinuity.
/// Log coordinates because every one of these is a positive scale.
#[allow(clippy::too_many_arguments)]
fn search(
    preset: &Preset,
    rows: &[Row],
    phrases: &[PhraseRow],
    melody_board: &MelodyBoard,
    start: MicVoicing,
    free: &[Knob],
    hard: bool,
    feasible: &Feasible,
) -> MicVoicing {
    let plain = relaxed_objective(preset, rows, phrases, melody_board, feasible);
    // The geometry stages keep item 363's objective exactly; only the
    // mode-controlled band is fitted under the hard gate, and only after a
    // relaxed pass has chosen the basin. See [`RED_BAND_PENALTY`].
    let penalised = modal_objective(preset, rows, phrases, melody_board);
    let objective = move |v: MicVoicing| -> f64 {
        if hard {
            penalised(v)
        } else {
            plain(v)
        }
    };
    let mut best = start;
    let mut score = objective(best);
    // Compass, simplex, compass again, until neither moves it. **Because one
    // round is not convergence on this surface**: a band enters or leaves the
    // readable set as the engine's own levels move, which puts small steps in
    // the objective, and the two searches fail on steps in opposite ways — the
    // compass walks past a diagonal valley, the simplex reads a step as
    // curvature and contracts onto it. Measured: one round of the
    // mode-controlled band stopped at 202-252 Hz x3.48 and 20.467 bars, and a
    // point a whole third of a semitone away — 205-245 Hz x3.80 — scores
    // **20.37** and takes the notes' gate from one red to none. A fit that
    // leaves a better point that close to the one it reports has not finished.
    for round in 0..SEARCH_ROUNDS {
        let before = score;
        compass(&objective, &mut best, &mut score, free);
        polish(&objective, &mut best, &mut score, free);
        println!("  round {}: {score:.3} bars out", round + 1);
        if before - score <= SEARCH_EPSILON {
            break;
        }
    }
    best
}

/// Rounds of compass-then-simplex [`search`] will run before it gives up on
/// finding anything more. Three is what the mode-controlled band needed to stop
/// moving; the loop exits early whenever a round buys less than
/// [`SEARCH_EPSILON`], which is what usually happens on the second.
const SEARCH_ROUNDS: usize = 4;

/// One compass pass: axis-aligned, in log coordinates, halving the step.
fn compass(
    objective: &(impl Fn(MicVoicing) -> f64 + Sync),
    best: &mut MicVoicing,
    score: &mut f64,
    free: &[Knob],
) {
    let mut step = SEARCH_STEP;
    let mut evaluations = 1usize;
    while step > SEARCH_FLOOR {
        // One step of the compass is `2 * free.len()` probes around one point,
        // and they do not depend on one another: [`batch`] runs them at once
        // and the step takes the best of them.
        let mut probes: Vec<MicVoicing> = Vec::with_capacity(2 * free.len());
        for &knob in free {
            for direction in [1.0, -1.0] {
                let mut candidate = *best;
                knob.set(&mut candidate, knob.get(best) * (direction * step).exp());
                if knob.get(&candidate) == knob.get(best) {
                    continue;
                }
                probes.push(candidate);
            }
        }
        let scored = batch(&probes, objective);
        evaluations += probes.len();
        let improved = match scored.iter().enumerate().min_by(|a, b| a.1.total_cmp(b.1)) {
            Some((i, &value)) if value < *score - SEARCH_EPSILON => {
                *score = value;
                *best = probes[i];
                true
            }
            _ => false,
        };
        if !improved {
            step *= 0.5;
        }
        println!(
            "  step {step:.4}: spacing {:.3} span {:.3} width {:.3} coherence {:.3} \
modal {} -> {score:.3} bars out ({evaluations} sets rendered)",
            best.spacing_m,
            best.span_m,
            best.width,
            best.diffuse_coherence,
            best.modal.map_or_else(
                || "absent".to_string(),
                |b| format!("{:.0}-{:.0} Hz x{:.3}", b.lo_hz, b.hi_hz, b.lift)
            )
        );
    }
}

/// A simplex on top of the compass, because the compass cannot walk a diagonal
/// valley and this surface has one: spacing and width trade almost exactly
/// against each other in the mid bands, where a wider pair and a quieter
/// difference make nearly the same side signal — and so, at the other end of
/// the spectrum, do the mode-controlled band's lower edge and its lift.
fn polish(
    objective: &(impl Fn(MicVoicing) -> f64 + Sync),
    best: &mut MicVoicing,
    score: &mut f64,
    free: &[Knob],
) {
    let start: Vec<f64> = free.iter().map(|k| k.get(best).ln()).collect();
    let simplex = NelderMead {
        max_evaluations: POLISH_EVALUATIONS,
        tolerance: 1e-4,
        initial_step: 0.05,
    };
    let voicing_at = |p: &[f64]| -> MicVoicing {
        let mut candidate = *best;
        for (knob, value) in free.iter().zip(p) {
            knob.set(&mut candidate, value.exp());
        }
        candidate
    };
    // The simplex's own points, a batch at a time: see
    // [`NelderMead::minimize_batched`]. It visits the same points in the same
    // order and stops in the same place; what changes is that a set of thirty
    // renders no longer waits for the set before it.
    let minimum = simplex.minimize_batched(&start, |points| {
        let candidates: Vec<MicVoicing> = points.iter().map(|p| voicing_at(p)).collect();
        batch(&candidates, objective)
    });
    let polished = voicing_at(&minimum.point);
    let value = objective(polished);
    println!(
        "  polish: spacing {:.3} span {:.3} width {:.3} coherence {:.3} modal {} \
-> {value:.3} bars out ({} more sets rendered)",
        polished.spacing_m,
        polished.span_m,
        polished.width,
        polished.diffuse_coherence,
        polished.modal.map_or_else(
            || "absent".to_string(),
            |b| format!("{:.0}-{:.0} Hz x{:.3}", b.lo_hz, b.hi_hz, b.lift)
        ),
        minimum.evaluations
    );
    if value < *score - SEARCH_EPSILON {
        *best = polished;
        *score = value;
    }
}

/// The stereo columns of one candidate `[voicing.mics]`.
fn columns_for_voicing(
    preset: &Preset,
    rows: &[Row],
    voicing: MicVoicing,
    velocity: u8,
) -> Vec<StereoColumn> {
    let mut candidate = preset.clone();
    candidate.voicing.mics = Some(voicing);
    columns_for(&candidate, rows, velocity)
}

/// The stereo columns of one preset against the phrase set.
///
/// No level match, unlike `bench`: a correlation and a mid-over-side ratio are
/// both invariant to a gain applied to a whole signal, so the two agree band
/// for band and this saves a pass over six phrases per evaluation.
fn phrase_columns(preset: &Preset, phrases: &[PhraseRow]) -> Vec<StereoColumn> {
    let items: Vec<StereoItem> = phrases
        .par_iter()
        .map(|p| StereoItem {
            label: p.phrase.name.to_string(),
            engine: realism::stereo_image_of(&render_phrase(preset, &p.phrase))
                .expect("two channels"),
            reference: p.reference.clone(),
            alternate: p.alternate.clone(),
        })
        .collect();
    realism::stereo_columns(&items)
}

fn phrase_columns_for_voicing(
    preset: &Preset,
    phrases: &[PhraseRow],
    voicing: MicVoicing,
) -> Vec<StereoColumn> {
    let mut candidate = preset.clone();
    candidate.voicing.mics = Some(voicing);
    phrase_columns(&candidate, phrases)
}

/// The stereo columns of one preset against the reference rows.
fn columns_for(preset: &Preset, rows: &[Row], velocity: u8) -> Vec<StereoColumn> {
    boards_for(preset, rows, velocity).0
}

/// **Both boards off one set of renders**: the coherence columns
/// ([`realism::stereo_columns`]) and the per-channel spectral columns
/// ([`realism::channel_columns`]).
///
/// One pass rather than two, because the second board costs a forward FFT pair
/// on a signal that has just been rendered and rendering is the whole cost —
/// and because two passes could not be guaranteed to be scoring the same
/// samples.
fn boards_for(
    preset: &Preset,
    rows: &[Row],
    velocity: u8,
) -> (Vec<StereoColumn>, Vec<ChannelColumn>) {
    let rendered: Vec<(StereoImage, ChannelShape)> = rows
        .par_iter()
        .map(|r| {
            let audio = render_engine(preset, r.key, velocity);
            (
                realism::stereo_image_of(&audio).expect("two channels"),
                realism::channel_shape_of(&audio).expect("two channels"),
            )
        })
        .collect();
    let images: Vec<StereoItem> = rows
        .iter()
        .zip(&rendered)
        .map(|(r, (image, _))| StereoItem {
            label: r.label.clone(),
            engine: image.clone(),
            reference: r.reference.clone(),
            alternate: r.alternate.clone(),
        })
        .collect();
    let shapes: Vec<ChannelItem> = rows
        .iter()
        .zip(&rendered)
        .map(|(r, (_, shape))| ChannelItem {
            label: r.label.clone(),
            engine: shape.clone(),
            reference: r.reference_channels.clone(),
            alternate: r.alternate_channels.clone(),
        })
        .collect();
    (
        realism::stereo_columns(&images),
        realism::channel_columns(&shapes),
    )
}

// ---------------------------------------------------------------------------
// The melody board, as a term of this objective
// ---------------------------------------------------------------------------

/// The columns of `piano-tuner melody` that this fit's knobs can move.
///
/// `strike`, `roughness`, `wobble` and `hf` are all functions of the mono
/// fold-down and of tables no microphone parameter touches, so charging them
/// here would add a constant and its render noise. `channel`, `balance` and
/// `splitting` are the three the pair writes, and they are three rather than
/// two since `DECISIONS.md` 451: a pair can put a note's fundamental in the
/// right loudspeaker and its own overtones in the other one, and the first two
/// columns are both blind to that — `channel` because it is a sum over the
/// pair, `balance` because it reads one frequency per note.
const MELODY_METRICS: [&str; 3] = ["channel", "balance", "splitting"];

/// **The melody board's three stereo columns, in this objective** — the term
/// `DECISIONS.md` 447 adds and item 451 widens, with item 416's own lesson
/// applied: *close on what the gates read*.
///
/// The reason it is here and not checked afterwards is the whole of item 446.
/// The band this stage fits acts over 174-456 Hz; every fundamental of the Ode
/// line falls inside it; and the boards this fit already closes on are a
/// per-key **spectral** surface, a per-key **coherence** surface and a
/// per-phrase one, none of which is a per-note reading of *which loudspeaker a
/// tune's pitches come out of*. Item 421 chose the shipped point partly on the
/// melody `channel` column — but by checking it after the fit had finished, so
/// the fit never traded anything for it. This makes it a term.
///
/// The reference side is measured **once**: it is the recordings playing two
/// fixed phrases and it does not move when a candidate does. Per candidate the
/// engine renders the soprano line and the recorded ladder, which is about
/// twenty-eight seconds of audio against the thirty keys' ninety — and the
/// keys go through [`ENGINE_CACHE`] while these do not, so on a warm cache it
/// is the larger half. That cost is stated in item 447's budget rather than
/// hidden.
struct MelodyBoard {
    recorded: realism::RecordedKeys,
    line: Phrase,
    line_notes: Vec<melody::LineNote>,
    line_reference: Vec<(u8, [f64; 8])>,
    line_layer: Vec<(u8, [f64; 8])>,
    ladder: Phrase,
    ladder_notes: Vec<melody::LineNote>,
    ladder_reference: Vec<(u8, [f64; 8])>,
    ladder_layer: Vec<(u8, [f64; 8])>,
}

impl MelodyBoard {
    /// The recordings' half, measured once. Only [`melody::Window::Head`] is
    /// built: `channel` and `balance` are both head-window columns
    /// (`melody::METRIC_IS_BALANCE`), and the tail line is another twenty-five
    /// seconds of render per candidate for three columns this stage cannot
    /// move.
    fn measure(
        data: &std::path::Path,
        sfz: &std::path::Path,
        layers: &realism::VelocityLayers,
        recorded: &realism::RecordedKeys,
        preset: &Preset,
    ) -> Result<Self, piano_tuner::Error> {
        let window = melody::Window::Head;
        let ladder_keys = melody::ladder_keys(recorded, &melody::line_keys());
        let line = melody::line_for(window);
        let line_notes = melody::line_notes_for(window);
        let ladder = melody::ladder(&ladder_keys, window);
        let ladder_notes = melody::ladder_notes(&ladder_keys, window);
        let hz = partial_hz_of(preset);
        let side = |phrase: &Phrase,
                    notes: &[melody::LineNote],
                    name: &str,
                    events: &[TimedEvent]|
         -> Result<Vec<(u8, [f64; 8])>, piano_tuner::Error> {
            let audio = melody::reference_line(sfz, data, phrase, name, events)?;
            Ok(melody::per_key(&melody::measure_line(
                &audio,
                f64::from(SAMPLE_RATE),
                notes,
                &hz,
                window,
            )))
        };
        Ok(MelodyBoard {
            recorded: recorded.clone(),
            line_reference: side(&line, &line_notes, "reference", &line.events)?,
            line_layer: side(&line, &line_notes, "alt-layer", &layers.shift(&line.events))?,
            ladder_reference: side(&ladder, &ladder_notes, "reference", &ladder.events)?,
            ladder_layer: side(
                &ladder,
                &ladder_notes,
                "alt-layer",
                &layers.shift(&ladder.events),
            )?,
            line,
            line_notes,
            ladder,
            ladder_notes,
        })
    }

    /// One candidate's melody columns, off two engine renders.
    fn columns(&self, preset: &Preset) -> Vec<melody::Column> {
        let hz = partial_hz_of(preset);
        let engine = |phrase: &Phrase, notes: &[melody::LineNote]| -> Vec<(u8, [f64; 8])> {
            let events = engine_events::to_render_events(&phrase.events);
            let (left, right) = render_to_buffer(preset, &events, phrase.duration_s as f32);
            let audio =
                Audio::new(SAMPLE_RATE, vec![left, right]).expect("the engine renders stereo");
            melody::per_key(&melody::measure_line(
                &audio,
                f64::from(SAMPLE_RATE),
                notes,
                &hz,
                melody::Window::Head,
            ))
        };
        melody::compare(
            melody::Window::Head,
            &melody::Lines::new(
                engine(&self.line, &self.line_notes),
                self.line_reference.clone(),
                self.line_layer.clone(),
            ),
            &melody::Lines::new(
                engine(&self.ladder, &self.ladder_notes),
                self.ladder_reference.clone(),
                self.ladder_layer.clone(),
            ),
            &self.recorded,
        )
    }
}

/// The frequencies a key's partials are read at: the preset's own table, which
/// is what both sides of every melody column are measured with.
fn partial_hz_of(preset: &Preset) -> impl Fn(u8) -> Vec<f64> + '_ {
    move |key: u8| -> Vec<f64> {
        let params = preset.string_params(key);
        (1..=piano_tuner::series::PARTIALS)
            .map(|k| f64::from(params.partial_freq(k)))
            .collect()
    }
}

/// The melody board's exceedance, in the same currency [`gate_excess`] uses.
///
/// Both halves of a column are charged and each only where it is a verdict:
/// `balance` and `splitting` are gated on their median **and** on the line's
/// own spread (`melody::METRIC_IS_SPREAD`), `channel` on the median alone. The
/// medians are not taken over the same keys — `splitting`'s is the line's and
/// the other two are the recorded ladder's (`melody::METRIC_ON_LINE`) — which
/// is a property of the columns and needs nothing here. A fit that drives this to zero is a fit that turns those columns of
/// `tuner/tests/melody.rs` green, which is the only thing it can earn here.
fn melody_excess(columns: &[melody::Column]) -> f64 {
    columns
        .iter()
        .filter(|c| MELODY_METRICS.contains(&c.metric))
        .map(|c| {
            let mut excess = 0.0;
            if c.gated_on_balance && c.balance.is_finite() && c.balance_bar > 0.0 {
                excess += (c.balance.abs() / c.balance_bar - PASS_MARGIN).max(0.0);
            }
            if c.gated_on_spread && c.standout.is_finite() && c.bar > 0.0 {
                excess += (c.standout / c.bar - PASS_MARGIN).max(0.0);
            }
            excess
        })
        .sum()
}

/// The melody term for one voicing.
fn melody_excess_for(preset: &Preset, board: &MelodyBoard, voicing: MicVoicing) -> f64 {
    let mut candidate = preset.clone();
    candidate.voicing.mics = Some(voicing);
    melody_excess(&board.columns(&candidate))
}

/// The per-channel board's exceedance, in the same currency [`gate_excess`]
/// uses: `sum over bands of max(0, |err| / bar - margin)`.
///
/// **In the objective from item 393 on, and it is the term that decides where
/// in the basin the band stops.** The coherence columns cannot see it — `r0`
/// is normalised per channel and the mid-over-side ratio is a sum — so a fit
/// made on them alone is free to buy its correlation with a band that puts one
/// loudspeaker 9 dB up and the other 20 dB down, and the fit that shipped
/// did exactly that.
/// **The loudness half of the per-channel board**, in the two bands the lobe
/// acts in: how far the pair's own energy against the take's mono fold-down is
/// from the recording's, in units of the recording's repeatability.
///
/// It is the one term of this objective that is not a *shape*. `dev_L` and
/// `dev_R` are both referenced to the take's own mono spectrum, and `r0` and
/// the mid-over-side ratio are both ratios inside the pair, so a lobe that
/// doubles the acoustic energy the two loudspeakers put in the room while
/// leaving the fold-down alone moves none of them — which is exactly what
/// `DECISIONS.md` 392 found and what three listening complaints heard. Only
/// the modal bands are charged, for [`modal_channel_band`]'s reason: outside
/// them this is the pan-pot's and the geometry's number.
fn pair_excess(columns: &[ChannelColumn]) -> f64 {
    columns
        .iter()
        .filter(|c| {
            modal_channel_band(c)
                && c.pair_balance.is_finite()
                && c.pair_bar.is_finite()
                && c.pair_bar > 0.0
        })
        .map(|c| (c.pair_balance.abs() / c.pair_bar - PASS_MARGIN).max(0.0))
        .sum()
}

/// The per-channel shape half of the objective, in the **gate's own currency**.
///
/// `ChannelColumn::reachable` and not `::bar`, because that is what the gate's
/// verdict is taken against since item 418: the bar is what the recording asks
/// of a model that could place its capsules where the session did, and half the
/// reference spread of that ask is the capsule-placement asymmetry item 417
/// accepted as unscored. A fit closed on `bar` would spend the whole search
/// chasing a component no symmetric pair can produce — which is what every
/// mechanism milestone from 393 to 414 spent itself on — and would trade real
/// bands away to buy a hundredth of it.
fn channel_excess(columns: &[ChannelColumn]) -> f64 {
    columns
        .iter()
        .filter(|c| c.error.is_finite() && c.reachable.is_finite() && c.reachable > 0.0)
        .map(|c| (c.error / c.reachable - PASS_MARGIN).max(0.0))
        .sum()
}

/// **The objective: how far outside the recording's own repeatability the
/// engine is, summed over the bands, in units of that repeatability.**
///
/// `sum over bands of max(0, |err| / bar - 1)`. Zero for a band the gate
/// passes, and growing linearly in the gate's own currency for one it does
/// not — so a fit that drives this to zero is a fit that turns
/// `tuner/tests/stereo.rs` green, and nothing else it can do earns anything.
///
/// **Not the plain summed `|err|`**, which was the first version and is the
/// wrong thing by a wide margin. The bars differ by more than an order of
/// magnitude between bands — 0.009 in 63-125 Hz against 0.120 one octave up —
/// because the recording repeats itself far more exactly down there. Summing
/// raw correlations therefore *sells the bass*: measured on this material, a
/// geometry that takes 63-125 Hz from +0.965 to +0.494 (against the
/// recording's +0.953) and buys 0.6 of correlation back across 125-500 Hz
/// scores 1.30 where the bass-true one scores 2.21 — and it is 51 bars out in
/// the band where the other is 1.3, on the one measurement of the whole
/// finding that is repeatable to a hundredth. In the gate's units the same two
/// score 57 and 24, in the right order.
///
/// Nothing here is a function of the engine: `bar` is built out of the
/// reference and its own second take alone, which
/// `realism::tests::the_stereo_bar_is_built_out_of_the_reference_alone` pins to
/// twelve decimals. Minimising a distance in units of a threshold that cannot
/// move is minimising the distance.
fn gate_excess(columns: &[StereoColumn]) -> f64 {
    columns
        .iter()
        .filter(|c| c.error.is_finite() && c.bar.is_finite() && c.bar > 0.0)
        .map(|c| (c.error / c.bar - PASS_MARGIN).max(0.0))
        .sum()
}

/// The same distance in plain correlation units, summed over the readable
/// bands. Reported beside [`gate_excess`] — it is the number
/// `DECISIONS.md` 358 quoted — and never minimised.
fn band_error(columns: &[StereoColumn]) -> f64 {
    columns
        .iter()
        .filter(|c| c.error.is_finite())
        .map(|c| c.error)
        .sum()
}

/// **The delay half of the objective**: how far the geometry's own predicted
/// interchannel delays are from the ones the recording carries, in units of
/// what the recording's two takes of each key disagree by.
///
/// The same shape as [`gate_excess`] and for the same reason. A pair is
/// *where* it is as well as *how coherent* it makes two channels, and a fit
/// that reads only the second is free to put the capsules anywhere that
/// happens to decorrelate a band. The floor is measured, not chosen: the same
/// delay taken on the key's other velocity layer — a second recording of the
/// same note through the same microphones, which cannot have moved between
/// takes — so the term is zero exactly when the modelled delays are as close
/// to the measured ones as the measurement is to itself.
fn delay_excess(rows: &[Row], geometry: MicGeometry, floor_ms: f64) -> f64 {
    let bar = (floor_ms * realism::STEREO_ALLOWANCE).max(1e-3);
    (delay_residual(rows, geometry) / bar - PASS_MARGIN).max(0.0)
}

/// The delay floor: weighted RMS of `v90 delay − other-layer delay` over the
/// keys, milliseconds.
fn delay_floor_ms(rows: &[Row]) -> f64 {
    let (mut num, mut den) = (0.0, 0.0);
    for r in rows {
        let w = r.lag.confidence.min(r.alternate_lag.confidence);
        num += w * (r.lag.lag_s - r.alternate_lag.lag_s).powi(2);
        den += w;
    }
    1e3 * (num / den).sqrt()
}

/// Weighted RMS distance, milliseconds, between the delays the recording
/// carries and the ones a geometry predicts.
fn delay_residual(rows: &[Row], geometry: MicGeometry) -> f64 {
    let (mut num, mut den) = (0.0, 0.0);
    for r in rows {
        num += r.lag.confidence * (r.lag.lag_s - geometry.itd_s(r.pan)).powi(2);
        den += r.lag.confidence;
    }
    1e3 * (num / den).sqrt()
}

fn geometry_of(v: &MicVoicing) -> MicGeometry {
    MicGeometry {
        spacing_m: f64::from(v.spacing_m),
        height_m: f64::from(v.height_m),
        span_m: f64::from(v.span_m),
    }
}

fn print_geometry(rows: &[Row], fit: &GeometryFit, start: Option<MicGeometry>) {
    println!(
        "\ngeometry from {} measured delays (height held at {:.3} m):",
        fit.items, fit.geometry.height_m
    );
    println!("  spacing_m {:.4}   span_m {:.4}", fit.geometry.spacing_m, fit.geometry.span_m);
    println!(
        "  residual {:.3} ms against a no-pair null of {:.3} ms; aspect is {:.1}x worse \
determined than the spacing; converged {}",
        fit.residual_ms, fit.null_ms, fit.conditioning, fit.converged
    );
    if let Some(start) = start {
        println!(
            "  the preset it replaces: spacing_m {:.4}, height_m {:.4}, span_m {:.4} \
(residual {:.3} ms)",
            start.spacing_m,
            start.height_m,
            start.span_m,
            delay_residual(rows, start)
        );
    }
    println!("\n| key | pan | measured lag | fitted lag | residual | peak | ILD meas / model |");
    println!("|---|---:|---:|---:|---:|---:|---:|");
    for r in rows {
        println!(
            "| {} | {:+.3} | {:+.3} ms | {:+.3} ms | {:+.3} ms | {:.2} | {:+.1} / {:+.1} dB |",
            r.label,
            r.pan,
            1e3 * r.lag.lag_s,
            1e3 * fit.geometry.itd_s(r.pan),
            1e3 * (r.lag.lag_s - fit.geometry.itd_s(r.pan)),
            r.lag.confidence,
            r.lag.ild_db,
            fit.geometry.ild_db(r.pan),
        );
    }
}

/// The velocity layers the fit never saw: one clearly softer and one clearly
/// louder than [`FIT_VELOCITY`], picked out of the library's own bands rather
/// than named, so a differently layered library still gets two.
fn held_out_velocities(layers: &realism::VelocityLayers) -> Vec<u8> {
    let bands = layers.bands();
    let middle = |i: usize| -> u8 {
        let (lo, hi) = bands[i];
        (((u16::from(lo) + u16::from(hi)) / 2) as u8).max(1)
    };
    let fit_band = layers.band_of(FIT_VELOCITY).unwrap_or(bands.len() / 2);
    let mut out = Vec::new();
    if fit_band >= 3 {
        out.push(middle(fit_band - 3));
    }
    if fit_band + 3 < bands.len() {
        out.push(middle(fit_band + 3));
    }
    out
}

/// The reference side re-measured at a held-out velocity: the same keys, the
/// recording of that layer, and its own neighbouring layer as the floor.
fn held_out_rows(
    data: &std::path::Path,
    sfz: &std::path::Path,
    layers: &realism::VelocityLayers,
    rows: &[Row],
    velocity: u8,
) -> Result<Vec<Row>, piano_tuner::Error> {
    let other = layers.alternate(velocity);
    rows.par_iter()
        .map(|r| -> Result<Row, piano_tuner::Error> {
            let reference = render_reference(data, sfz, r.key, velocity)?;
            let alternate = render_reference(data, sfz, r.key, other)?;
            let measured = interchannel_lag(
                &reference.channels[0],
                &reference.channels[1],
                f64::from(SAMPLE_RATE),
                &LagConfig::default(),
            )?;
            Ok(Row {
                key: r.key,
                label: r.label.clone(),
                pan: r.pan,
                lag: KeyLag {
                    pan: r.pan,
                    lag_s: measured.lag_s,
                    confidence: measured.confidence,
                    ild_db: measured.ild_db,
                },
                alternate_lag: r.alternate_lag,
                reference: realism::stereo_image_of(&reference)?,
                alternate: realism::stereo_image_of(&alternate)?,
                reference_channels: realism::channel_shape_of(&reference)?,
                alternate_channels: realism::channel_shape_of(&alternate)?,
            })
        })
        .collect()
}
