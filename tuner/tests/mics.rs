//! Self-calibration for the microphone-pair estimator: can it read back a
//! geometry the engine was *told* to have?
//!
//! `TUNING.md`'s standing rule, and the one `tuner/tests/calibration.rs` applies
//! to every other estimator in the crate: a fit is worth its output only if,
//! given a render of the engine with a parameter set to a known value, it
//! returns that value. Unit tests on synthetic delays
//! (`estimate::mics::tests`) prove the *arithmetic*; this file is the part that
//! can fail for a real reason — the estimator is pointed at a piano, with a
//! soundboard's diffuse field, three unison strings, a duplex, mechanism noise
//! and a limiter between the geometry and the two channels it has to be read
//! out of.
//!
//! # Why this is not obviously possible
//!
//! The engine's microphone stage is **not** a delay pair. It is mid plus side,
//! and the mid is delay-free by construction so that no mono scoreboard moves
//! when the geometry does (`DECISIONS.md` 352). For one source at pan `p`, with
//! `δ` the geometric delay of the farther capsule, what comes out is
//!
//! ```text
//! L = (m - w·u_R/2)·x(t) + (w·u_L/2)·x(t-δ)
//! R = (m + w·u_R/2)·x(t) - (w·u_L/2)·x(t-δ)
//! ```
//!
//! — a two-tap pair, not a delayed one. Its cross-correlation carries a lobe at
//! `+δ` weighted `(m + w·u_R/2)(w·u_L/2)`, one at `-δ` weighted
//! `-(m - w·u_R/2)(w·u_L/2)`, and a zero-lag lobe weighted `m² - (w/2)²·…`. At
//! the width the pair is actually run at those come to about `+0.66`, `-0.09`
//! and `-0.06`: the geometric lobe is seven times the next largest and the
//! zero-lag one has nearly cancelled itself, so a delay estimator sees the
//! geometry and not the construction. That is a claim about a particular corner
//! of the parameter space, which is why it is asserted here against the engine
//! rather than argued in a comment.
//!
//! # The material
//!
//! The keys whose own strings radiate in the band the delay is read in
//! ([`estimate::mics::LagConfig`]'s 40-160 Hz, which is where the *recording*
//! is coherent enough to be read at all). Above that band a key's low
//! frequencies are the board's response rather than the key's own source, and
//! the board is not panned — so a treble key measures the diffuse field's own
//! delay, which is zero by construction and would drag any fit towards a pair
//! that is not there. The recording has the same property and the tool that
//! fits it says so; here it is the reason the compass is cut at D#3.

use piano_emulator::preset::{MicVoicing, Preset};
use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_emulator::types::Event;
use piano_tuner::estimate::mics::{
    fit_geometry, interchannel_lag, GeometryConfig, KeyLag, LagConfig, MicGeometry,
    ENGINE_LAG_PER_ITD, SPEED_OF_SOUND,
};
use piano_tuner::{Audio, SAMPLE_RATE};

use rayon::prelude::*;

/// Velocity every key is struck at.
const VELOCITY: u8 = 90;

/// Seconds of note, and the silence before it.
///
/// The preroll is `realism::STEREO_PREROLL_SAMPLES` — a whole number of engine
/// blocks, so the strike lands on the first sample of the window. This file
/// reads an *interchannel* statistic off a single key, which is exactly the
/// measurement `DECISIONS.md` 378 found reading its own window edge: at the
/// 0.05 s preroll this file used to ask for, the note began 96 samples before
/// the window did, and the recovered spacing at 12 cm read `+28 %` where the
/// aligned window reads `+13 %`.
const RENDER_S: f64 = 3.0;
const PREROLL: usize = piano_tuner::realism::STEREO_PREROLL_SAMPLES;

const _: () = assert!(
    PREROLL % piano_emulator::types::BLOCK == 0,
    "the preroll must be a whole number of engine blocks or the window starts inside the note"
);

const PREROLL_S: f64 = PREROLL as f64 / 48_000.0;

/// The keys the delay is measured on: A0 to D#3, whose fundamentals and low
/// partials are inside `LagConfig`'s band. Every third semitone, which is the
/// spacing the library itself sampled at.
const KEYS: [u8; 11] = [21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51];

fn repo() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn shipped_preset() -> Preset {
    Preset::load(&repo().join("presets/salamander-c5.toml")).expect("the measured preset loads")
}

fn pan_for_key(key: u8) -> f64 {
    (2.0 * f64::from(key.clamp(21, 108) - 21) / 87.0 - 1.0) * 0.6
}

fn render(preset: &Preset, key: u8) -> Audio {
    let events = [RenderEvent::new(
        PREROLL_S as f32,
        Event::NoteOn {
            key,
            vel: u16::from(VELOCITY),
        },
    )];
    let (left, right) = render_to_buffer(preset, &events, (PREROLL_S + RENDER_S) as f32);
    assert_eq!(events[0].frame(), PREROLL, "the strike must open the window");
    Audio::new(
        SAMPLE_RATE,
        vec![left[PREROLL..].to_vec(), right[PREROLL..].to_vec()],
    )
    .expect("the engine renders stereo")
}

/// Every key's interchannel delay, measured off the engine's own render the
/// same way the tool measures it off the recording.
fn delays(preset: &Preset) -> Vec<KeyLag> {
    let config = LagConfig::default();
    KEYS.par_iter()
        .map(|&key| {
            let audio = render(preset, key);
            let lag = interchannel_lag(
                &audio.channels[0],
                &audio.channels[1],
                f64::from(SAMPLE_RATE),
                &config,
            )
            .expect("the engine renders two channels of a note");
            KeyLag {
                pan: pan_for_key(key),
                lag_s: lag.lag_s,
                confidence: lag.confidence,
                ild_db: lag.ild_db,
            }
        })
        .collect()
}

fn with_mics(base: &Preset, mics: MicVoicing) -> Preset {
    let mut preset = base.clone();
    preset.voicing.mics = Some(mics);
    preset.validate().expect("a legal geometry");
    preset
}

/// **The self-calibration gate: a spacing the engine was given, read back out
/// of its own renders.**
///
/// Three spacings an octave apart — 12, 24 and 48 cm — rendered through the
/// whole instrument at eleven bass keys, and recovered as
///
/// ```text
/// spacing = |median lag| · c / ENGINE_LAG_PER_ITD
/// ```
///
/// A **median**, not a fit. The 88-key inversion [`fit_geometry`] is what runs
/// on the recording, where the delays are spread across the compass and the
/// curve's *shape* carries information; here every key is far enough off-axis
/// that the geometry has saturated at `±d/c` and the curve has no shape left —
/// so the statistic that recovers the spacing is the level of the plateau, and
/// the robust estimator of a plateau is its median. Running the shaped
/// inversion on a saturated plateau is what it cannot do: `spacing` and
/// `span/height` trade against each other exactly along the plateau's own
/// direction, and the fit walks the valley to whichever bound it is nearest.
///
/// [`ENGINE_LAG_PER_ITD`] is the calibration constant and its own doc comment
/// is where it comes from. What this test asserts is that it is **still true**:
/// three truths, five times apart end to end, each recovered to 20 %.
#[test]
fn the_estimator_reads_back_a_spacing_the_engine_was_given() {
    let base = shipped_preset();
    let shipped = base
        .voicing
        .mics
        .expect("the shipped preset carries a microphone pair");
    let mut lines = String::new();
    let mut failures = 0;
    for &spacing in &[0.12f32, 0.24, 0.48] {
        let measured = delays(&with_mics(
            &base,
            MicVoicing {
                spacing_m: spacing,
                ..shipped
            },
        ));
        let mut lags: Vec<f64> = measured.iter().map(|k| k.lag_s).collect();
        lags.sort_by(f64::total_cmp);
        let median = lags[lags.len() / 2];
        let recovered = median.abs() * SPEED_OF_SOUND / ENGINE_LAG_PER_ITD;
        let error = recovered / f64::from(spacing) - 1.0;
        lines.push_str(&format!(
            "\n  spacing {spacing:.2} m -> {recovered:.3} m ({:+.0} %) from a median lag of \
{:+.3} ms",
            100.0 * error,
            1e3 * median
        ));
        if error.abs() > 0.20 {
            failures += 1;
        }
    }
    // Printed whether it passes or not, so the calibration is re-derivable from
    // a green run rather than only visible in a red one — `DECISIONS.md` 400's
    // rule, and item 418's frontier is quoted from exactly this line.
    println!("the spacings read back out of the engine's own renders:{lines}");
    assert_eq!(
        failures, 0,
        "the estimator did not read back the geometry the engine was given:{lines}"
    );
}

/// **Where the mode-controlled band starts biasing the reading**, swept, so the
/// constraint the fit is held under is a measurement of this tree rather than a
/// number carried forward from `DECISIONS.md` 395.
///
/// The lobe is not common to the two channels — it is added to the side and
/// subtracted from it — so its own group delay enters a phase-transform delay
/// reading. Item 395 measured the boundary at about `lo_hz = 225` and item 400
/// undid it exactly, for a *rotation* whose arithmetic has an inverse; the
/// rotation is gone (item 406) and the lobe's inverse is not the same
/// arithmetic, so the boundary is a live constraint again. Item 418 re-measures
/// it under the rail, where the lobe is weaker and the bias is therefore
/// smaller, and holds `piano-tuner mics`' own `Knob::ModalLo` at what comes
/// out.
///
/// `#[ignore]`d because it is an instrument and not a gate — twelve bands times
/// three spacings is 396 renders — and run by name:
///
/// ```sh
/// cargo test --release -p piano-tuner --test mics -- --ignored --nocapture \
///     where_the_bands_lower_edge_starts_biasing_the_reading
/// ```
#[test]
#[ignore]
fn where_the_bands_lower_edge_starts_biasing_the_reading() {
    // `MICS_SWEEP_PRESET` points the sweep at a candidate instead of the
    // shipped preset — the case this was written for is a refit in flight,
    // where the boundary has to be known *before* the preset that respects it
    // exists. It is an override on an `#[ignore]`d instrument and nothing a
    // gate reads.
    let base = match std::env::var("MICS_SWEEP_PRESET") {
        Ok(path) => Preset::load(std::path::Path::new(&path)).expect("the candidate loads"),
        Err(_) => shipped_preset(),
    };
    let shipped = base
        .voicing
        .mics
        .expect("the shipped preset carries a microphone pair");
    let band = shipped.modal.expect("the shipped preset carries a band");
    println!("| lo_hz | hi_hz | lift | 0.12 m | 0.24 m | 0.48 m | worst |");
    println!("|---:|---:|---:|---:|---:|---:|---:|");
    for lift in [0.5f32, 0.99] {
        for lo_hz in [155.0f32, 160.0, 165.0, 170.0, 185.0, 230.0] {
            let mut errors = Vec::new();
            for &spacing in &[0.12f32, 0.24, 0.48] {
                let preset = with_mics(
                    &base,
                    MicVoicing {
                        spacing_m: spacing,
                        modal: Some(piano_emulator::preset::ModalBand {
                            lo_hz,
                            hi_hz: band.hi_hz.max(lo_hz * 1.06),
                            lift,
                        }),
                        ..shipped
                    },
                );
                let mut lags: Vec<f64> = delays(&preset).iter().map(|k| k.lag_s).collect();
                lags.sort_by(f64::total_cmp);
                let median = lags[lags.len() / 2];
                let recovered = median.abs() * SPEED_OF_SOUND / ENGINE_LAG_PER_ITD;
                errors.push(recovered / f64::from(spacing) - 1.0);
            }
            let worst = errors.iter().cloned().fold(0.0f64, |a, b| a.max(b.abs()));
            println!(
                "| {lo_hz:.0} | {:.0} | {lift:.2} | {:+.0} % | {:+.0} % | {:+.0} % | {:+.0} % |",
                band.hi_hz.max(lo_hz * 1.06),
                100.0 * errors[0],
                100.0 * errors[1],
                100.0 * errors[2],
                100.0 * worst
            );
        }
    }
}

/// The same thing said the other way: the readout is **monotone** in the
/// spacing, and by the right factor.
///
/// A calibration constant can absorb a scale error; it cannot absorb a
/// nonlinearity. Doubling the spacing must double the delay, and this asserts
/// that with no constant in it at all — so it still fails if
/// [`ENGINE_LAG_PER_ITD`] is ever re-measured wrongly.
#[test]
fn doubling_the_spacing_doubles_the_delay_the_renders_carry() {
    let base = shipped_preset();
    let shipped = base
        .voicing
        .mics
        .expect("the shipped preset carries a microphone pair");
    let median_lag = |spacing: f32| -> f64 {
        let mut lags: Vec<f64> = delays(&with_mics(
            &base,
            MicVoicing {
                spacing_m: spacing,
                ..shipped
            },
        ))
        .iter()
        .map(|k| k.lag_s.abs())
        .collect();
        lags.sort_by(f64::total_cmp);
        lags[lags.len() / 2]
    };
    let (small, middle, large) = (median_lag(0.12), median_lag(0.24), median_lag(0.48));
    for (a, b, name) in [
        (small, middle, "0.12 -> 0.24"),
        (middle, large, "0.24 -> 0.48"),
    ] {
        let ratio = b / a;
        assert!(
            (0.7..1.5).contains(&(ratio / 2.0)),
            "{name} m of spacing changed the median delay by {ratio:.2}x, not about 2x \
({:.3} ms -> {:.3} ms)",
            1e3 * a,
            1e3 * b
        );
    }
}

/// The control that makes the gate above mean something: **an engine with no
/// microphone pair must not read as one.**
///
/// With `[voicing.mics]` absent the engine is the pan-pot and the two
/// orthogonal board taps — no delay anywhere in the chain — so there is nothing
/// for the inversion to find, and what it must not do is find something anyway.
/// The test is the inversion's own null: a fit whose residual is no better than
/// "every delay is zero" has measured nothing, and that is what is asserted.
#[test]
fn an_engine_with_no_pair_gives_the_inversion_nothing_to_find() {
    let mut preset = shipped_preset();
    preset.voicing.mics = None;
    let measured = delays(&preset);
    let largest = measured
        .iter()
        .map(|k| 1e3 * k.lag_s.abs())
        .fold(0.0, f64::max);
    let fit = fit_geometry(&measured, &GeometryConfig::default()).expect("eleven delays");
    assert!(
        fit.residual_ms >= 0.5 * fit.null_ms,
        "a pan-potted engine was inverted into a pair: residual {:.3} ms against a null of \
{:.3} ms, largest delay {largest:.3} ms, recovered {:?}",
        fit.residual_ms,
        fit.null_ms,
        fit.geometry
    );
}

/// The shipped instrument reads back the pair it is documented to have.
///
/// Not a tolerance on a fitted number — that is the test above — but the
/// standing statement that `presets/salamander-c5.toml`'s `[voicing.mics]` is a
/// geometry the engine's own output still carries: the delays the preset's
/// capsule positions predict are closer to the ones its renders show than a
/// pair that is not there would be.
#[test]
fn the_shipped_pair_is_visible_in_the_shipped_instruments_own_renders() {
    let preset = shipped_preset();
    let mics = preset
        .voicing
        .mics
        .expect("the shipped preset carries a microphone pair");
    let stated = MicGeometry::new(
        f64::from(mics.spacing_m),
        f64::from(mics.height_m),
        f64::from(mics.span_m),
    );
    let measured = delays(&preset);
    let (mut modelled, mut null, mut weight) = (0.0, 0.0, 0.0);
    for k in &measured {
        modelled += k.confidence * (k.lag_s - stated.itd_s(k.pan)).powi(2);
        null += k.confidence * k.lag_s.powi(2);
        weight += k.confidence;
    }
    let (modelled, null) = (
        1e3 * (modelled / weight).sqrt(),
        1e3 * (null / weight).sqrt(),
    );
    assert!(
        modelled < null,
        "the shipped geometry explains none of its own renders' delays: {modelled:.3} ms \
against a no-pair null of {null:.3} ms"
    );
}
