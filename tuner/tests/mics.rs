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
    fit_geometry, interchannel_lag, GeometryConfig, KeyLag, KnownBand, LagConfig, MicGeometry,
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

/// Every key's interchannel delay, measured off the engine's own render.
///
/// **Not quite the way the tool measures it off the recording, and the
/// difference is `DECISIONS.md` 465.** The engine's own mode-controlled band
/// puts an interchannel phase between the two channels that no path difference
/// put there, and it is computable from the preset — so a reading of an
/// *engine* render is handed the band it is known to carry and subtracts it,
/// where a reading of the recording is handed `None` and never anything else.
fn delays(preset: &Preset) -> Vec<KeyLag> {
    let config = LagConfig {
        band: preset.voicing.mics.and_then(|m| m.modal).map(|b| KnownBand {
            lo_hz: f64::from(b.lo_hz),
            hi_hz: f64::from(b.hi_hz),
            lift: f64::from(b.lift),
        }),
        ..LagConfig::default()
    };
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

/// **The phase the estimator subtracts is the phase the engine adds.**
///
/// [`KnownBand`] mirrors `soundboard::MIC_MODAL_HIGH_Q`, `MIC_MODAL_LOW_Q` and
/// the cookbook forms `soundboard::Biquad::butterworth` builds — the same
/// device `SPEED_OF_SOUND` is, and the same risk: a mirrored constant is a
/// second copy of a decision, and two copies drift. So it is checked against
/// the engine rather than argued.
///
/// The instrument is a preset with **no geometric side at all** (`width = 0`)
/// and **no board field** (`board_mix = 0`), where the mic stage reduces to
/// `L = m(1 + B)`, `R = m(1 − B)` exactly, so the pair's own interchannel phase
/// *is* the band's. Read off the two rendered channels bin by bin over the
/// band's own span, it must be what `KnownBand::interchannel_phase` returns.
#[test]
fn the_known_bands_response_is_the_engines_own() {
    use rustfft::{num_complex::Complex32, FftPlanner};

    let mut preset = shipped_preset();
    let shipped = preset
        .voicing
        .mics
        .expect("the shipped preset carries a microphone pair");
    let band = shipped.modal.expect("and a band");
    preset.soundboard.board_mix = 0.0;
    preset.voicing.mics = Some(MicVoicing {
        width: 0.0,
        ..shipped
    });
    preset.validate().expect("a legal instrument");
    // Softly struck, and that is not a detail: the master chain ends in a
    // `soft_clip`, which is a *nonlinearity*, and two channels that differ by
    // 8 dB inside the band come out of it with two different curvatures. A
    // pianissimo note keeps the whole comparison linear.
    let audio = {
        let events = [RenderEvent::new(
            PREROLL_S as f32,
            Event::NoteOn { key: 40, vel: 20 },
        )];
        let (left, right) = render_to_buffer(&preset, &events, (PREROLL_S + RENDER_S) as f32);
        Audio::new(
            SAMPLE_RATE,
            vec![left[PREROLL..].to_vec(), right[PREROLL..].to_vec()],
        )
        .expect("the engine renders stereo")
    };
    let known = KnownBand {
        lo_hz: f64::from(band.lo_hz),
        hi_hz: f64::from(band.hi_hz),
        lift: f64::from(band.lift),
    };

    let n = 1 << 16;
    let mut planner = FftPlanner::<f64>::new();
    let forward = planner.plan_fft_forward(n);
    let spectrum = |channel: &[f32]| -> Vec<Complex32> {
        let mut buffer: Vec<rustfft::num_complex::Complex<f64>> = (0..n)
            .map(|i| {
                rustfft::num_complex::Complex::new(
                    channel.get(i).copied().unwrap_or(0.0) as f64,
                    0.0,
                )
            })
            .collect();
        forward.process(&mut buffer);
        buffer
            .into_iter()
            .map(|c| Complex32::new(c.re as f32, c.im as f32))
            .collect()
    };
    let (l, r) = (
        spectrum(&audio.channels[0]),
        spectrum(&audio.channels[1]),
    );
    // **At the note's own partials, and only there.** A struck string radiates
    // at a comb of decaying sinusoids; on the skirts between them what the
    // transform holds is one partial's *leakage* in both channels at once, so
    // the ratio there is the filter's response at the **partial's** frequency
    // and not at the bin's, and comparing it against the model at the bin is
    // comparing two different frequencies. At a local maximum of the mid the
    // partial is the signal, and the ratio is the response.
    let mid: Vec<f64> = (0..n / 2)
        .map(|j| f64::from(((l[j] + r[j]) * 0.5).norm()))
        .collect();
    let ceiling = mid.iter().cloned().fold(0.0f64, f64::max);
    let mut ranked: Vec<(f64, f64, f64)> = (1..n / 2 - 1)
        .filter_map(|j| {
            let hz = j as f64 * f64::from(SAMPLE_RATE) / n as f64;
            if !(60.0..=1_200.0).contains(&hz) {
                return None;
            }
            // A partial: a local maximum of the mid, well over the floor.
            if !(mid[j] > mid[j - 1] && mid[j] >= mid[j + 1] && mid[j] > 0.02 * ceiling) {
                return None;
            }
            // **Not inside a hair of the notch.** `R = m(1 − B)` and the lift
            // is 0.99, so where `|1 − B|` is a hundredth the channel's phase is
            // the argument of a difference of two nearly equal numbers: the
            // engine computes it in `f32` through a running biquad and the
            // model in `f64` in closed form, and a coefficient difference of
            // one part in a million turns into a radian. That is a property of
            // the construction (item 423 measured the same thing as a −33 dB
            // one-channel loss) and not of the mirror, so the comparison is
            // made where the model is not singular.
            let b = known.response(hz, f64::from(SAMPLE_RATE));
            let depth = |s: f64| ((1.0 + s * b.0).powi(2) + b.1 * b.1).sqrt();
            if depth(1.0).min(depth(-1.0)) < 0.15 {
                return None;
            }
            let weight = f64::from(l[j].norm()) * f64::from(r[j].norm());
            let measured = f64::from((l[j] * r[j].conj()).arg());
            let modelled = known.interchannel_phase(hz, f64::from(SAMPLE_RATE));
            let mut error = measured - modelled;
            while error > std::f64::consts::PI {
                error -= std::f64::consts::TAU;
            }
            while error < -std::f64::consts::PI {
                error += std::f64::consts::TAU;
            }
            Some((weight, hz, error))
        })
        .collect();
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
    ranked.truncate(24);
    let (worst, at_hz) = ranked
        .iter()
        .fold((0.0f64, 0.0f64), |acc, &(_, hz, e)| {
            if e.abs() > acc.0 {
                (e.abs(), hz)
            } else {
                acc
            }
        });
    println!(
        "the band's interchannel phase, engine against model: worst {worst:.4} rad at {at_hz:.1} Hz \
over the {} loudest partials",
        ranked.len()
    );
    assert!(ranked.len() >= 8, "the probe sounded at {} partials", ranked.len());
    assert!(
        worst < 0.05,
        "the estimator's copy of the band is not the engine's: {worst:.4} rad out at {at_hz:.1} Hz"
    );
}

/// **What the band's own group delay was doing to the readback, band by band
/// and with and without the subtraction of `DECISIONS.md` 465.**
///
/// The instrument behind item 465, and the table its doc comment quotes. Five
/// bands — the one that ships, the one item 462 refused, two of the sweep's own
/// neighbours and no band at all — times three spacings an octave apart, read
/// twice each: once on the raw cross-spectrum, once with `arg[(1+B)/(1−B)]`
/// removed from every bin. What it shows is the whole claim: the raw readings
/// scatter with the band's *width* and the corrected ones do not move at all.
///
/// `#[ignore]`d because it is an instrument and not a gate — thirty conditions
/// of eleven renders each — and run by name:
///
/// ```sh
/// cargo test --release -p piano-tuner --test mics -- --ignored --nocapture \
///     what_the_bands_group_delay_was_doing_to_the_readback
/// ```
#[test]
#[ignore]
fn what_the_bands_group_delay_was_doing_to_the_readback() {
    let base = shipped_preset();
    let shipped = base
        .voicing
        .mics
        .expect("the shipped preset carries a microphone pair");
    let bands: [(&str, Option<piano_emulator::preset::ModalBand>); 6] = [
        ("170-456 shipped", shipped.modal),
        (
            "170-234 refused",
            Some(piano_emulator::preset::ModalBand {
                lo_hz: 170.0,
                hi_hz: 234.0,
                lift: 0.99,
            }),
        ),
        (
            "170-280",
            Some(piano_emulator::preset::ModalBand {
                lo_hz: 170.0,
                hi_hz: 280.0,
                lift: 0.99,
            }),
        ),
        (
            "170-200 narrowest",
            Some(piano_emulator::preset::ModalBand {
                lo_hz: 170.0,
                hi_hz: 200.0,
                lift: 0.99,
            }),
        ),
        (
            "400-800",
            Some(piano_emulator::preset::ModalBand {
                lo_hz: 400.0,
                hi_hz: 800.0,
                lift: 0.99,
            }),
        ),
        ("none", None),
    ];
    println!("| band | corrected | 0.12 m | 0.24 m | 0.48 m |");
    println!("|---|---|--:|--:|--:|");
    for (name, modal) in bands {
        for corrected in [false, true] {
            let mut cells = Vec::new();
            for &spacing in &[0.12f32, 0.24, 0.48] {
                let preset = with_mics(
                    &base,
                    MicVoicing {
                        spacing_m: spacing,
                        modal,
                        ..shipped
                    },
                );
                let config = LagConfig {
                    band: corrected.then(|| modal).flatten().map(|b| KnownBand {
                        lo_hz: f64::from(b.lo_hz),
                        hi_hz: f64::from(b.hi_hz),
                        lift: f64::from(b.lift),
                    }),
                    ..LagConfig::default()
                };
                let mut lags: Vec<f64> = KEYS
                    .par_iter()
                    .map(|&key| {
                        let audio = render(&preset, key);
                        interchannel_lag(
                            &audio.channels[0],
                            &audio.channels[1],
                            f64::from(SAMPLE_RATE),
                            &config,
                        )
                        .expect("two channels")
                        .lag_s
                    })
                    .collect();
                lags.sort_by(f64::total_cmp);
                let median = lags[lags.len() / 2];
                cells.push(format!(
                    "{:+.3} ms ({:+.0} %)",
                    1e3 * median,
                    100.0 * (median.abs() * SPEED_OF_SOUND / ENGINE_LAG_PER_ITD
                        / f64::from(spacing)
                        - 1.0)
                ));
            }
            println!(
                "| {name} | {} | {} |",
                if corrected { "yes" } else { "raw" },
                cells.join(" | ")
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
