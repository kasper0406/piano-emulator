//! Stage 2, the motion half: the two mechanisms that make a partial *move*, and
//! the two tables the new forward model invalidated.
//!
//! ```text
//! cargo run --release -p piano-tuner --example fit_motion -- \
//!     data/salamander/SalamanderGrandPiano-V3+20200602.sfz \
//!     --preset presets/salamander-c5.toml --out presets/salamander-c5.toml
//! ```
//!
//! Four fits, in the order they must run in, because each later one is measured
//! against a render that already contains the earlier ones:
//!
//! 1. **`notes.false_beat`** — per key and per partial, the companion the
//!    recording's own beat depth and rate imply, with `DECISIONS.md` 233's
//!    falsification (*uncorrelated across `k` or it is not a false beat*) run per
//!    key. `estimate::motion::fit_false_beat`.
//! 2. **`[voicing.strike_direction]`** — one global velocity law. Its **sign**
//!    is regressed from the same companion measured at every velocity layer the
//!    library has (`estimate::motion::fit_strike_direction`, a within-cell
//!    regression); its **size** is inverted on the engine
//!    (`estimate::motion::SwingLine`), because a beat depth saturates and the
//!    column the field exists to move is a spread. Pinned to zero at the
//!    reference velocity so that nothing fitted at velocity 90 moves.
//! 3. **`notes.detune_cents`** — re-fitted where, and only where, it is still
//!    identifiable. The coupled construction locks the bass and midrange unison
//!    (`FUNDAMENTALS.md` §7.3: *with anti-veering there will be no beats*), so
//!    the map from tuning to beat rate is no longer an identity there and there
//!    is nothing to invert. The partition is not a new judgement: it is the
//!    false-beat fit's own verdict — a key whose measured rates track `k` is
//!    beating because of its tuning, and that is exactly the key whose tuning a
//!    beat rate can be inverted from. The aftersound `FUNDAMENTALS.md` §7.5
//!    step 4 names as the check is *reported* against the recording's, not
//!    inverted; [`DETUNE_LO`] carries the measurement that says why.
//! 4. **`notes.partial_gains`** — the full measured ratio `a_k(0) recorded /
//!    a_k(0) as the engine itself renders it` (`DECISIONS.md` 231, 237), taken
//!    against a probe whose own row is **cleared** first, which is what makes
//!    this fit re-entrant where `fit_partials` is not.
//!
//! Without `--out` it measures and prints and writes nothing.

use std::collections::BTreeMap;
use std::path::PathBuf;

use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_emulator::types::Event;
use piano_tuner::estimate::motion::{
    fit_false_beat, fit_strike_direction, strike_direction_for, FalseBeatLoop, FalseBeatVerdict,
    MotionConfig, SwingLine, VelocityCell,
};
use piano_tuner::estimate::shaping::{
    flatten_row, measured_over_rendered_report, ShapingConfig,
};
use piano_tuner::motion::{partial_motion, Motion, Spectrum, WINDOW_HI_S};
use piano_tuner::pipeline::analyze_note;
use piano_tuner::preset::{
    equal_temperament, key_index, FalseBeat, Preset, MAX_PARTIAL_GAIN, MIN_PARTIAL_GAIN,
};
use piano_tuner::series::{amp_db, Series, PARTIALS, WINDOW_S};
use piano_tuner::survey::SurveyConfig;
use piano_tuner::trajectory::InharmonicModel;
use piano_tuner::{audio, detect_onset, Sample, SampleLibrary, SAMPLE_RATE};

const SR: f64 = SAMPLE_RATE as f64;

/// Velocity every fit is anchored at — `TUNING_REPORT.md` §5's own convention,
/// and the velocity every per-note table in the preset was measured at.
const REFERENCE_VELOCITY: u8 = 90;

/// Seconds of silence before the strike in every render, and how long each note
/// is rendered for. The analysis window ends at
/// [`piano_tuner::motion::WINDOW_HI_S`]; the extra half-second is the Gaussian
/// band-pass's own tail.
const PREROLL_S: f64 = 0.05;
const RENDER_S: f64 = WINDOW_HI_S + 1.5;

/// How long a render for a *spectrum* has to be: `analyze_note` fits a decay and
/// extrapolates it back to the strike, and a short render is a short fit.
const SPECTRUM_RENDER_S: f32 = 4.0;

/// How long a render for the compass's own spectrum statistic has to be: the
/// window ends at 1.10 s and the level is an RMS over it.
const SERIES_RENDER_S: f32 = 1.6;

/// Velocity layers the gains are taken over. The reference layer plus its two
/// neighbours: the engine is rendered at each one's own velocity, so what is
/// left in the ratio is the mismatch and not the blow, and three of them is what
/// makes the median mean anything.
const GAIN_LAYER_SPAN: i32 = 1;

/// How many partials the gains are written for at most.
const MAX_GAIN_PARTIALS: usize = 48;

/// The keys the swing line is measured on, and the partials: exactly Column B's
/// own cells (`realism::MOTION_KEYS`, `MOTION_PARTIALS`), because what the field
/// is fitted against is the column it exists to move. Fitting it on some other
/// cell set and hoping would be the mistake `FUNDAMENTALS.md` §II.2 convicts the
/// scoreboard of.
const SWING_KEYS: [u8; 4] = [45, 60, 69, 84];
const SWING_PARTIALS: u32 = 4;

/// The swings the engine is probed at, in dB of pianissimo-to-fortissimo range.
/// Zero first, which is the velocity-independent construction and reads the
/// line's own floor.
const SWING_PROBES: &[f64] = &[0.0, 4.0, 8.0, 16.0];

/// The three velocities Column B is defined at.
const COLUMN_B_VELOCITIES: [u8; 3] = [40, 90, 120];

/// The detune search, in cents: a **grid**, not a bisection, because what the
/// tuning now sets is not monotone in it.
///
/// # What the detune is fitted against, and why it is not the beat rate
///
/// It used to be the beat rate, and under the free-running construction that
/// was an identity: two strings a ratio apart beat at their difference. Under
/// the coupled one it is not. `FUNDAMENTALS.md` §3.2's anti-veering pulls the
/// group's frequencies *together*, so a narrow unison locks and beats at
/// nothing, and — measured here — the recording's own companions come back flat
/// in `k` at almost every key, which is the false-beat fit's verdict and means
/// the beat that is there is the **wire's** and not the tuning's. There is no
/// beat rate left to invert at 28 of 30 keys.
///
/// What the tuning does still set is the **aftersound** — at zero detuning the
/// antisymmetric eigenvectors radiate nothing and `|w . v_m|` grows with the
/// mistuning (§5.1) — and that was tried as the objective first, because §7.5
/// step 4 names C6's aftersound as the check. It does not carry a fit: swept
/// over this grid the rendered aftersound of the fundamental runs over ranges of
/// **13 to 96 dB** with no monotone shape (key 69: −56.1 to +39.8 dB), because
/// the statistic is two straight lines through 2.7 s of a demodulated log
/// envelope that is also beating. So the aftersound is *reported* here, against
/// the recording's own, as §7.5's check — and it is not inverted.
const DETUNE_LO: f64 = 0.05;
const DETUNE_HI: f64 = 6.0;
const DETUNE_STEPS: usize = 12;
/// How close the best grid point's beat *slope* must come to the recording's,
/// as a fraction, before the answer is written. The statistic matched is the
/// slope `s` of `rate = s k` through the origin over every partial the recording
/// measured — which is what a unison mistuning **is**, a frequency ratio, and is
/// the only thing these keys are here for.
const DETUNE_TOLERANCE: f64 = 0.25;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let sfz = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "data/salamander/SalamanderGrandPiano-V3+20200602.sfz".into()),
    );
    let mut preset_path = PathBuf::from("presets/salamander-c5.toml");
    let mut out: Option<PathBuf> = None;
    let mut only: Vec<u8> = Vec::new();
    let mut stages: Vec<String> = Vec::new();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--preset" => preset_path = PathBuf::from(args.next().expect("--preset <file>")),
            "--out" => out = Some(PathBuf::from(args.next().expect("--out <file>"))),
            "--key" => only.push(args.next().expect("--key <n>").parse()?),
            "--stage" => stages.push(args.next().expect("--stage <name>")),
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let wants = |name: &str| stages.is_empty() || stages.iter().any(|s| s == name);

    let library = SampleLibrary::from_sfz(&sfz)?;
    let mut preset = Preset::load(&preset_path)?;
    let survey = SurveyConfig::default();
    let config = MotionConfig::default();

    let mut keys: Vec<u8> = {
        let mut set: Vec<u8> = library.samples().map(|s| s.key).collect();
        set.sort_unstable();
        set.dedup();
        set
    };
    if !only.is_empty() {
        keys.retain(|k| only.contains(k));
    }
    println!(
        "{} keys from {}, preset {}",
        keys.len(),
        sfz.display(),
        preset_path.display()
    );

    // ---- 1. the false beat, and the coverage of what it refused -------------
    let mut fits = BTreeMap::new();
    if wants("false_beat") {
        println!(
            "\n== notes.false_beat ==\n key  partials   flat/prop  r(k,rate)  verdict          rows (k: hz / dB)"
        );
        for &key in &keys {
            let Some(sample) = layer_for(&library, key, REFERENCE_VELOCITY) else {
                continue;
            };
            let Ok(signal) = recording(sample) else { continue };
            let measured = measure(&preset, key, &signal, config.max_partial);
            let mut fit = fit_false_beat(key, &measured, &config);
            // The recording says *which* partials and at what rate; the engine
            // says at what level, because the asked level is quoted against one
            // block and the depth is measured on the whole partial. See
            // `FalseBeatLoop`.
            let unwritten = if fit.rows.is_empty() {
                0
            } else {
                let (rows, unwritten) = solve_on_the_render(&preset, key, &fit.rows, &measured, &config);
                fit.rows = rows;
                if fit.rows.is_empty() {
                    fit.verdict = FalseBeatVerdict::NoneInRange;
                }
                unwritten
            };
            println!(
                "{:>4}  {:>8}  {:>9.2}  {:>9.2}  {:<15}  {:>2} unwritten  {}",
                key,
                fit.measured.len(),
                fit.model_ratio,
                fit.rate_correlation,
                verdict_name(fit.verdict),
                unwritten,
                fit.rows
                    .iter()
                    .map(|r| format!("{}: {:.2} / {:.1}", r.k, r.hz, r.db))
                    .collect::<Vec<_>>()
                    .join("  ")
            );
            fits.insert(key, fit);
        }
        let written: Vec<u8> = fits
            .iter()
            .filter(|(_, f)| f.verdict == FalseBeatVerdict::Written)
            .map(|(k, _)| *k)
            .collect();
        println!(
            "coverage: {} of {} keys written, {} refused as the unison's own beat, \
             {} measured nothing in range",
            written.len(),
            fits.len(),
            fits.values()
                .filter(|f| f.verdict == FalseBeatVerdict::ScalesWithPartial)
                .count(),
            fits.values()
                .filter(|f| {
                    matches!(
                        f.verdict,
                        FalseBeatVerdict::NoneInRange | FalseBeatVerdict::TooFewPartials
                    )
                })
                .count()
        );
        write_false_beats(&mut preset, &fits);
    }

    // ---- 2. the strike direction -------------------------------------------
    if wants("strike_direction") {
        println!("\n== [voicing.strike_direction] ==");
        let mut cells: Vec<VelocityCell> = Vec::new();
        let mut group = 0u32;
        for (&key, fit) in &fits {
            if fit.verdict != FalseBeatVerdict::Written {
                continue;
            }
            let partials: Vec<u32> = fit.rows.iter().map(|r| u32::from(r.k)).collect();
            let groups: Vec<u32> = partials
                .iter()
                .map(|_| {
                    group += 1;
                    group
                })
                .collect();
            for sample in library.layers(key) {
                let Ok(signal) = recording(sample) else { continue };
                let measured = measure(&preset, key, &signal, config.max_partial);
                for (k, motion) in measured {
                    let Some(at) = partials.iter().position(|p| *p == k) else {
                        continue;
                    };
                    if let Some(db) = motion.companion_db() {
                        cells.push(VelocityCell {
                            group: groups[at],
                            velocity: sample.midi_velocity(),
                            db,
                        });
                    }
                }
            }
        }
        let Some(regression) = fit_strike_direction(&cells, REFERENCE_VELOCITY, &config) else {
            println!("{} readings: too few for a regression", cells.len());
            return Ok(());
        };
        println!(
            "{} readings over {} cells: within-cell slope {:+.2} dB (r {:+.3}, residual \
             {:.2} dB), median per-cell slope {:+.2} dB (IQR {:.2})",
            regression.cells,
            regression.groups,
            regression.swing_db,
            regression.correlation,
            regression.residual_db,
            regression.median_cell_slope,
            regression.cell_slope_iqr
        );

        // The target: what the recording's own beat depth does across the three
        // velocities Column B is defined at, on the cells this fit has a
        // companion for.
        let probe_keys: Vec<u8> = SWING_KEYS.to_vec();
        let reference = recorded_velocity_spread(&library, &preset, &probe_keys);
        println!(
            "target: the recording's mean per-cell spread over velocities {:?} is \
             {:.2} dB of beat depth and {:.3} cents of frequency, over {} keys x {} \
             partials — Column B2's own denominator, and the fit is for B2 = 1",
            COLUMN_B_VELOCITIES,
            reference.0,
            reference.1,
            probe_keys.len(),
            SWING_PARTIALS
        );

        // The line: the same statistic on the engine, at a handful of swings.
        // The sign comes from the *median per-cell* slope, not from the pooled
        // one: the pooled figure is snapped to zero under `min_swing_db`, and a
        // snapped number has no sign to read.
        let sign = if regression.median_cell_slope < 0.0 {
            -1.0
        } else {
            1.0
        };
        let mut line = SwingLine::default();
        for &swing in SWING_PROBES {
            let mut probe = preset.clone();
            probe.voicing.strike_direction = (swing != 0.0)
                .then(|| strike_direction_for(sign * swing, REFERENCE_VELOCITY));
            let (depth, cents) = rendered_velocity_spread(&probe, &probe_keys);
            // The line's ordinate is Column B2 itself: the geometric mean of the
            // two ratios, pooled exactly as `realism::motion_columns` pools
            // them, so what is inverted is the gate and not a proxy for it.
            let coherence = ((depth / reference.0).max(0.0) * (cents / reference.1).max(0.0))
                .sqrt();
            println!(
                "  swing {:>5.1} dB -> spread {:.3} dB / {:.3} c, B2 {:.3}",
                sign * swing,
                depth,
                cents,
                coherence
            );
            line.probes.push((swing, coherence));
        }
        match line.swing_for(1.0) {
            Some(swing) => {
                let direction = strike_direction_for(sign * swing, REFERENCE_VELOCITY);
                println!(
                    "  -> swing {:+.2} dB: vh_db_at_pp {:+.2}, vh_db_at_ff {:+.2}, \
                     share_tilt {:.2}",
                    sign * swing,
                    direction.vh_db_at_pp,
                    direction.vh_db_at_ff,
                    direction.share_tilt
                );
                preset.voicing.strike_direction = (swing != 0.0).then_some(direction);
            }
            None => println!("  -> the engine's spread does not move with the swing; nothing written"),
        }
    }

    // ---- 3. detune, where the beat still identifies it ----------------------
    if wants("detune") {
        println!(
            "\n== notes.detune_cents ==\n key  strings  recorded s  rendered s {:.2}..{:.2} c  \
             identifiable  cents            aftersound dB: rec / engine",
            DETUNE_LO, DETUNE_HI
        );
        let grid: Vec<f64> = (0..DETUNE_STEPS)
            .map(|i| {
                DETUNE_LO
                    * (DETUNE_HI / DETUNE_LO).powf(i as f64 / (DETUNE_STEPS - 1) as f64)
            })
            .collect();
        for &key in &keys {
            let Some(index) = key_index(key) else { continue };
            if preset.notes.unison[index] < 2 {
                continue;
            }
            let Some(fit) = fits.get(&key) else { continue };
            let recorded_aftersound = layer_for(&library, key, REFERENCE_VELOCITY)
                .and_then(|sample| recording(sample).ok())
                .and_then(|signal| aftersound_of(&preset, key, &signal))
                .unwrap_or(f64::NAN);
            let before = aftersound_db(&preset, key);
            // The partition: a key whose measured rates track `k` is beating
            // because of its tuning, and only there is a beat rate a
            // measurement of the tuning. Everywhere else the beat is the wire's
            // (the false-beat fit's own verdict) and there is nothing to
            // invert — see [`DETUNE_LO`].
            let identifiable = fit.verdict == FalseBeatVerdict::ScalesWithPartial;
            let mut error = f64::NAN;
            let (mut lo, mut hi) = (f64::NAN, f64::NAN);
            let mut target = f64::NAN;
            let was = f64::from(preset.notes.detune_cents[index]);
            if identifiable {
                let recorded: Vec<(u32, f64)> =
                    fit.measured.iter().map(|c| (c.k, c.hz)).collect();
                let top = recorded.iter().map(|&(k, _)| k).max().unwrap_or(1);
                target = beat_slope(&recorded);
                let slope_at = |cents: f64| {
                    let mut probe = preset.clone();
                    probe.notes.detune_cents[index] = cents as f32;
                    let signal = render(&probe, key, REFERENCE_VELOCITY, RENDER_S);
                    let rendered = measure(&probe, key, &signal, top);
                    let paired: Vec<(u32, f64)> = recorded
                        .iter()
                        .filter_map(|&(k, _)| {
                            rendered
                                .iter()
                                .find(|(index, _)| *index == k)
                                .map(|(_, m)| (k, m.beat_rate_hz))
                        })
                        .collect();
                    beat_slope(&paired)
                };
                let scored: Vec<(f64, f64)> =
                    grid.iter().map(|&cents| (cents, slope_at(cents))).collect();
                lo = scored.iter().map(|s| s.1).fold(f64::INFINITY, f64::min);
                hi = scored.iter().map(|s| s.1).fold(f64::NEG_INFINITY, f64::max);
                let best = scored
                    .iter()
                    .min_by(|a, b| (a.1 - target).abs().total_cmp(&(b.1 - target).abs()))
                    .copied()
                    .expect("a non-empty grid");
                error = (best.1 - target).abs() / target;
                if error <= DETUNE_TOLERANCE {
                    preset.notes.detune_cents[index] = best.0 as f32;
                }
            }
            let after = aftersound_db(&preset, key);
            println!(
                "{:>4}  {:>7}  {:>10.2}  {:>10.2}..{:<9.2}  {:<12}  {:>5.3} -> {:<5.3}  \
                 {:>7.1}: {:.1} -> {:.1}",
                key,
                preset.notes.unison[index],
                target,
                lo,
                hi,
                if !identifiable {
                    "no (the wire)".to_string()
                } else if error <= DETUNE_TOLERANCE {
                    format!("yes ({:+.0} %)", 100.0 * error)
                } else {
                    format!("no ({:+.0} %)", 100.0 * error)
                },
                was,
                f64::from(preset.notes.detune_cents[index]),
                recorded_aftersound,
                before,
                after
            );
        }
    }

    // ---- 4. the full-envelope gains ----------------------------------------
    let loudest = |spectrum: &[(u32, f64)]| {
        spectrum
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).expect("finite"))
            .map_or(0, |&(k, _)| k)
    };
    if wants("partial_gains") {
        println!(
            "\n== notes.partial_gains ==\n key  layers  partials   dB at k=1..4                \
             span    rail  railed   step rec/row   level   keep  irreg rec/eng  moved  \
             loudest rec/eng"
        );
        let shaping = ShapingConfig::default();
        let mut agreed = 0usize;
        let mut checked = 0usize;
        for &key in &keys {
            let Some(index) = key_index(key) else { continue };
            let layers = gain_layers(&library, key);
            if layers.is_empty() {
                continue;
            }
            // Cleared, not carried: the ratio is then the absolute correction
            // and running this tool twice is running it once.
            let mut probe = preset.clone();
            probe.notes.partial_gains[index] = Vec::new();
            let mut recorded = Vec::new();
            let mut rendered = Vec::new();
            for sample in &layers {
                let Some(spectrum) = recorded_spectrum(&survey, &preset, key, sample) else {
                    continue;
                };
                let Some(engine) =
                    rendered_spectrum(&survey, &probe, key, sample.midi_velocity())
                else {
                    continue;
                };
                recorded.push(spectrum);
                rendered.push(engine);
            }
            let Some(report) =
                measured_over_rendered_report(&recorded, &rendered, MAX_GAIN_PARTIALS, &shaping)
            else {
                println!("{key:>4}  {:>6}  nothing fitted", recorded.len());
                continue;
            };
            // Closed on the render, which is the only place the acceptance
            // criterion lives. See `close_on_the_render`.
            let mut fitted = report.gains.clone();
            fitted.truncate(partial_count(&preset, key));
            let target = reference_series(&library, key, &preset);
            let closed = close_on_the_render(&preset, &probe, key, &fitted, target, &shaping);
            let gains = closed.row.clone();
            let db = |g: f32| 20.0 * f64::from(g).log10();
            let span = gains.iter().fold(f64::MIN, |m, &g| m.max(db(g)))
                - gains.iter().fold(f64::MAX, |m, &g| m.min(db(g)));
            preset.notes.partial_gains[index] = gains
                .iter()
                .map(|g| g.clamp(MIN_PARTIAL_GAIN, MAX_PARTIAL_GAIN))
                .collect();
            let mut row = preset.notes.partial_gains[index].clone();
            while row.last() == Some(&1.0) {
                row.pop();
            }
            preset.notes.partial_gains[index] = row;

            let reference = recorded.first().cloned().unwrap_or_default();
            let after = rendered_spectrum(&survey, &preset, key, REFERENCE_VELOCITY)
                .unwrap_or_default();
            if loudest(&reference) == loudest(&after) {
                agreed += 1;
            }
            checked += 1;
            println!(
                "{:>4}  {:>6}  {:>8}   {:>6.2} {:>6.2} {:>6.2} {:>6.2}   {:>6.2}  {:>6.2} {:>4}   \
                 {:>5.2}/{:<5.2}  {:>+6.2}   {:>4.2}  {:>5.2}/{:<5.2}  {:>+5.2}  {} / {}",
                key,
                recorded.len(),
                preset.notes.partial_gains[index].len(),
                gains.first().copied().map_or(f64::NAN, db),
                gains.get(1).copied().map_or(f64::NAN, db),
                gains.get(2).copied().map_or(f64::NAN, db),
                gains.get(3).copied().map_or(f64::NAN, db),
                span,
                report.rail_db,
                report.railed,
                report.target_step_db,
                report.written_step_db,
                report.level_db,
                closed.keep,
                closed.target_irregular,
                closed.rendered_irregular,
                closed.level_moved_db,
                loudest(&reference),
                loudest(&after),
            );
        }
        if checked > 0 {
            println!(
                "acceptance: the engine's loudest partial matches the recording's at \
                 {agreed} of {checked} keys"
            );
        }
    }

    if let Some(path) = out {
        preset.validate()?;
        preset.save(&path)?;
        println!("\nwrote {}", path.display());
    } else {
        println!("\nnothing written (pass --out <file>)");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The gains, closed on the render
// ---------------------------------------------------------------------------

/// How far **under** the recording's own `irregular` the render has to land
/// before the loop stops shrinking the row's roughness, in dB.
///
/// The target is the recording's roughness and not zero — the measured 5-10 dB
/// of per-partial scatter is the whole reason the field exists, and a loop that
/// aimed at a smooth series would give back the defect
/// `estimate::shaping`'s header convicts the excitation of. What this margin
/// buys is two things the target alone does not:
///
/// * The recording's `irregular` is measured through the sampler's own noise
///   floor and the engine's is not, so landing exactly on it transfers the
///   recording's measurement noise into the instrument.
/// * The compass scores a key against its **neighbours**, and 58 of the 88 are
///   unsampled and carry no row at all. Measured with no margin, C2 renders at
///   9.80 dB of `irregular` against its own recording's 9.81 — a 0.01 dB match —
///   and still reads `z` +4.2 because the keys either side of it render at 4.5.
///   Half a decibel is what that costs to clear, and it leaves C2 inside the
///   band its own recording sits in.
const IRREGULAR_MARGIN_DB: f64 = 0.5;

/// How far the rendered level of a fitted key may sit from where the bare key
/// put it, in dB: the **band the recording's own compass has**, and not zero.
///
/// Measured over the 88 keys of `renders/compass/COMPASS.md`, the *recording's*
/// `level` residual against its own eight nearest same-`N` neighbours has a
/// robust sigma of **1.48 dB** (p90 3.25, worst 6.40). Two sigmas of that is the
/// width of the piano's own key-to-key level scatter, and a fitted key that
/// stays inside it cannot be the key a listener picks out of a melody: on the
/// same compass the *engine's* residual sigma was 3.04 dB with a worst of 23.25,
/// which is the defect.
///
/// # What this constant is *not* worth, measured (`DECISIONS.md` 282)
///
/// An earlier version of this comment claimed that pinning to zero instead
/// would throw away **0.32 dB of mean log-mel**, and that the width of the band
/// was therefore buying the scoreboard something. It is not. The stage was
/// re-emitted at three settings of this constant and both boards scored on each:
///
/// | `LEVEL_BAND_DB` | mean log-mel | compass |
/// |---|---|---|
/// | `0.0` — the row may not move the level at all | 4.70 | 16 flags / 13 keys |
/// | `2.96` — shipped | 4.70 | 16 flags / 13 keys |
/// | `1e9` — no bound at all | 4.73 | 16 flags / 13 keys |
///
/// So the band is worth **0.00 dB** of mean log-mel against pinning to zero and
/// **+0.03** against lifting it, and no `level` flag appears on the compass at
/// any of the three. The 0.32 dB belongs to `estimate::shaping::energy_offset`,
/// which is a different mechanism in a different file: putting *its*
/// removed scalar back into every
/// row reads **4.45** — and costs `F#5` a rendered level **17.9 dB over its own
/// neighbours**, which is the leak item 272 exists to close.
///
/// What the band still earns is one key: lifted, `D#4` gains a `centroid` flag
/// at z +6.0. That, and not the scoreboard, is the reason it is kept.
const LEVEL_BAND_DB: f64 = 2.96;

/// How close to the band edge the loop has to land before it stops.
const LEVEL_TOLERANCE_DB: f64 = 0.15;

/// Bisection steps on the roughness, and level corrections after it.
const KEEP_STEPS: usize = 7;
const LEVEL_ROUNDS: usize = 2;

struct ClosedRow {
    row: Vec<f32>,
    /// The fraction of the fitted roughness that survived.
    keep: f64,
    target_irregular: f64,
    rendered_irregular: f64,
    /// What the row did to the rendered level, dB — the number
    /// [`LEVEL_BAND_DB`] bounds.
    level_moved_db: f64,
}

/// Puts the fitted row on the engine and adjusts it until the **rendered** note
/// passes the two tests the compass will apply to it.
///
/// # Why this cannot be done in the estimator
///
/// `estimate::shaping` normalises the row for energy and smooths it until the
/// series it *predicts* — the engine's measured spectrum times the row — is no
/// jaggier than the recording's. That prediction is not the render. The row
/// multiplies a modal input gain, the note it produces is a coupled unison whose
/// partials are pairs of eigenmodes, and the compass measures a one-second RMS
/// through the whole master chain rather than an amplitude extrapolated back to
/// the strike. Measured, the gap is real and one-sided: after the estimator's
/// own discipline C4's predicted series reads 11.8 dB of step against its
/// recording's 12.8 and the *rendered* note still reads 13.9 against 7.0.
///
/// So the last word is the render's, which is `estimate::directivity`'s pattern
/// and `CombLine`'s and `FalseBeatLoop`'s (`DECISIONS.md` 137, 199, 211, 264): a
/// quantity that is only meaningful as "what does the engine do with this" is
/// inverted on the engine.
///
/// # The two tests, in the order they are solved
///
/// 1. **Roughness.** [`piano_tuner::estimate::shaping::flatten_row`] scales the
///    row's departures from its own smooth tilt by `keep` and leaves the tilt
///    alone, and `keep` is bisected until the rendered `irregular` is
///    [`IRREGULAR_MARGIN_DB`] under the recording's own. `keep = 1` — the whole
///    fitted roughness — is tried first and kept where it passes, because the
///    target is the recording's roughness and not zero.
/// 2. **Level.** The row is then shifted bodily until the rendered
///    0.10-1.10 s RMS is inside [`LEVEL_BAND_DB`] of where the *bare* key put
///    it — the band the recording's own compass has, so what is removed is only
///    the part of the level no key of the piano does (`DECISIONS.md` 272).
fn close_on_the_render(
    preset: &Preset,
    probe: &Preset,
    key: u8,
    fitted: &[f32],
    target: Option<Series>,
    shaping: &ShapingConfig,
) -> ClosedRow {
    let Some(index) = key_index(key) else {
        return ClosedRow {
            row: fitted.to_vec(),
            keep: 1.0,
            target_irregular: f64::NAN,
            rendered_irregular: f64::NAN,
            level_moved_db: 0.0,
        };
    };
    let with = |row: &[f32]| -> (f64, f64) {
        let mut candidate = preset.clone();
        candidate.notes.partial_gains[index] = row.to_vec();
        rendered_series(&candidate, key)
    };
    let (bare_level, _) = rendered_series(probe, key);
    let Some(target) = target else {
        return ClosedRow {
            row: fitted.to_vec(),
            keep: 1.0,
            target_irregular: f64::NAN,
            rendered_irregular: f64::NAN,
            level_moved_db: 0.0,
        };
    };
    let target_irregular = target.irregularity();

    // 1. the roughness.
    let mut keep = 1.0f64;
    let (_, irregular) = with(fitted);
    let mut row = fitted.to_vec();
    if irregular > target_irregular - IRREGULAR_MARGIN_DB {
        let (mut lo, mut hi) = (0.0f64, 1.0f64);
        for _ in 0..KEEP_STEPS {
            let mid = 0.5 * (lo + hi);
            let candidate = flatten_row(fitted, mid, 0.0, shaping);
            let (_, got) = with(&candidate);
            if got > target_irregular - IRREGULAR_MARGIN_DB {
                hi = mid;
            } else {
                lo = mid;
                keep = mid;
                row = candidate;
            }
        }
        if keep <= 0.0 {
            row = flatten_row(fitted, 0.0, 0.0, shaping);
        }
    }

    // 2. the level.
    let mut lift = 0.0f64;
    for _ in 0..LEVEL_ROUNDS {
        let (level, _) = with(&row);
        let moved = level - bare_level;
        let excess = moved - moved.clamp(-LEVEL_BAND_DB, LEVEL_BAND_DB);
        if excess.abs() <= LEVEL_TOLERANCE_DB {
            break;
        }
        lift -= excess;
        row = flatten_row(fitted, keep, lift, shaping);
    }
    let (level, got) = with(&row);
    ClosedRow {
        row,
        keep,
        target_irregular,
        rendered_irregular: got,
        level_moved_db: level - bare_level,
    }
}

/// `(level dBFS, series)` of one key struck alone through `preset`, measured
/// exactly as `compass_scan` measures it.
fn rendered_series(preset: &Preset, key: u8) -> (f64, f64) {
    let engine = match piano_emulator::preset::Preset::from_toml(&preset.to_toml()) {
        Ok(engine) => engine,
        Err(_) => return (f64::NAN, f64::NAN),
    };
    let events = [RenderEvent::new(
        PREROLL_S as f32,
        Event::NoteOn {
            key,
            vel: REFERENCE_VELOCITY,
        },
    )];
    let (left, right) = render_to_buffer(&engine, &events, SERIES_RENDER_S);
    let onset = (PREROLL_S * SR) as usize;
    let mono: Vec<f32> = left
        .iter()
        .zip(&right)
        .skip(onset)
        .map(|(&l, &r)| 0.5 * (l + r))
        .collect();
    let params = engine.string_params(key);
    let partial_hz: Vec<f64> = (1..=PARTIALS)
        .map(|k| f64::from(params.partial_freq(k)))
        .collect();
    let lo = (WINDOW_S.0 * SR) as usize;
    let hi = ((WINDOW_S.1 * SR) as usize).min(mono.len());
    let window = &mono[lo.min(hi)..hi];
    (
        amp_db(piano_tuner::realism::rms(window)),
        Series::measure(window, &partial_hz, SR).irregularity(),
    )
}

/// How many partials this key's bank has: the longest a row for it may be.
fn partial_count(preset: &Preset, key: u8) -> usize {
    piano_emulator::preset::Preset::from_toml(&preset.to_toml())
        .map(|engine| engine.string_params(key).partial_count())
        .unwrap_or(0)
}

/// The recording of `key` at the reference velocity, measured the same way.
fn reference_series(library: &SampleLibrary, key: u8, preset: &Preset) -> Option<Series> {
    let sample = layer_for(library, key, REFERENCE_VELOCITY)?;
    let audio = audio::load_at(&sample.path, SAMPLE_RATE).ok()?;
    let mono = audio.mono();
    let onset = (detect_onset(&mono, SR) * SR).round() as usize;
    let index = key_index(key)?;
    let engine_params = piano_emulator::preset::Preset::from_toml(&preset.to_toml())
        .ok()
        .map(|p| p.string_params(key))?;
    let _ = index;
    let partial_hz: Vec<f64> = (1..=PARTIALS)
        .map(|k| f64::from(engine_params.partial_freq(k)))
        .collect();
    let lo = onset + (WINDOW_S.0 * SR) as usize;
    let hi = (onset + (WINDOW_S.1 * SR) as usize).min(mono.len());
    if lo >= hi {
        return None;
    }
    Some(Series::measure(&mono[lo..hi], &partial_hz, SR))
}

fn verdict_name(verdict: FalseBeatVerdict) -> &'static str {
    match verdict {
        FalseBeatVerdict::Written => "written",
        FalseBeatVerdict::TooFewPartials => "too few",
        FalseBeatVerdict::ScalesWithPartial => "scales with k",
        FalseBeatVerdict::NoneInRange => "none in range",
    }
}

/// Runs [`FalseBeatLoop`] on one key: the level of every row is bisected until
/// the *rendered* beat depth is the recording's, and the rate is stepped until
/// the rendered frequency deviation is too.
///
/// The probe has this key's own row cleared before the first render — round
/// zero is the baseline, which is what decides whether the engine's unison
/// already out-beats the piano — so the stage is re-entrant: running it twice
/// on its own output gives the same answer.
///
/// Returns the solved rows and how many partials came back with no row at all,
/// because the engine's own unison was already closer to the recording than
/// anything the mechanism could add.
fn solve_on_the_render(
    preset: &Preset,
    key: u8,
    seed: &[FalseBeat],
    recorded: &[(u32, Motion)],
    config: &MotionConfig,
) -> (Vec<FalseBeat>, usize) {
    let mut probe = preset.clone();
    let mut loops = FalseBeatLoop::new(seed, recorded);
    while loops.running() {
        set_false_beat(&mut probe, key, &loops.rows());
        let signal = render(&probe, key, REFERENCE_VELOCITY, RENDER_S);
        let rendered = measure(&probe, key, &signal, config.max_partial);
        loops.observe(&rendered);
    }
    (loops.solved(), loops.unwritten())
}

/// Replaces one key's `notes.false_beat` row, growing the table to full length
/// first so that a preset that had none can still be probed.
fn set_false_beat(preset: &mut Preset, key: u8, rows: &[FalseBeat]) {
    let Some(index) = key_index(key) else { return };
    if preset.notes.false_beat.len() != piano_tuner::preset::NUM_KEYS {
        preset.notes.false_beat = vec![Vec::new(); piano_tuner::preset::NUM_KEYS];
    }
    preset.notes.false_beat[index] = rows.to_vec();
}

fn write_false_beats(
    preset: &mut Preset,
    fits: &BTreeMap<u8, piano_tuner::estimate::motion::FalseBeatFit>,
) {
    let mut table: Vec<Vec<FalseBeat>> = vec![Vec::new(); piano_tuner::preset::NUM_KEYS];
    let mut any = false;
    for (&key, fit) in fits {
        if let Some(index) = key_index(key) {
            if !fit.rows.is_empty() {
                any = true;
            }
            table[index] = fit.rows.clone();
        }
    }
    preset.notes.false_beat = if any { table } else { Vec::new() };
}

/// The library layer of `key` that a strike at `velocity` would trigger.
fn layer_for(library: &SampleLibrary, key: u8, velocity: u8) -> Option<&Sample> {
    library
        .layers(key)
        .iter()
        .find(|s| (s.lovel..=s.hivel).contains(&velocity))
}

/// The reference layer and its immediate neighbours.
fn gain_layers(library: &SampleLibrary, key: u8) -> Vec<Sample> {
    let layers = library.layers(key);
    let Some(centre) = layers
        .iter()
        .position(|s| (s.lovel..=s.hivel).contains(&REFERENCE_VELOCITY))
    else {
        return Vec::new();
    };
    let lo = centre.saturating_sub(GAIN_LAYER_SPAN as usize);
    let hi = (centre + GAIN_LAYER_SPAN as usize).min(layers.len() - 1);
    layers[lo..=hi].to_vec()
}

/// The recording, on the engine's clock, cut so that frame 0 is the strike.
fn recording(sample: &Sample) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let audio = audio::load_at(&sample.path, SAMPLE_RATE)?;
    let mono = audio.mono();
    let onset = detect_onset(&mono, SR);
    let start = (onset * SR).round() as usize;
    let frames = (RENDER_S * SR) as usize;
    Ok((0..frames)
        .map(|n| f64::from(mono.get(start + n).copied().unwrap_or(0.0)))
        .collect())
}

/// Every partial of `key` this signal resolves, measured the one way.
fn measure(preset: &Preset, key: u8, signal: &[f64], max_partial: u32) -> Vec<(u32, Motion)> {
    let Some(index) = key_index(key) else {
        return Vec::new();
    };
    let f0 = f64::from(preset.notes.f0_hz[index]);
    let b = f64::from(preset.notes.inharmonicity_b[index]);
    let b4 = f64::from(preset.notes.inharmonicity_b4[index]);
    let mut spectrum = Spectrum::new(signal);
    (1..=max_partial)
        .filter_map(|k| {
            let k2 = f64::from(k) * f64::from(k);
            let nominal = f64::from(k) * f0 * (1.0 + b * k2 + b4 * k2 * k2).sqrt();
            partial_motion(&mut spectrum, nominal, 0.35 * f0).map(|m| (k, m))
        })
        .collect()
}

/// The engine's own render of one note, mono, from the strike.
fn render(preset: &Preset, key: u8, vel: u8, seconds: f64) -> Vec<f64> {
    let engine = piano_emulator::preset::Preset::from_toml(&preset.to_toml())
        .expect("the tuner's preset is the engine's");
    let events = [RenderEvent::new(
        PREROLL_S as f32,
        Event::NoteOn { key, vel },
    )];
    let (left, right) = render_to_buffer(&engine, &events, (PREROLL_S + seconds) as f32);
    let skip = (PREROLL_S * SR) as usize;
    left.iter()
        .zip(&right)
        .skip(skip)
        .map(|(&l, &r)| 0.5 * (f64::from(l) + f64::from(r)))
        .collect()
}

/// Where the tail's straight line extrapolates back to at the strike, relative
/// to the prompt's — the aftersound level, in dB under the prompt. This is
/// `FUNDAMENTALS.md` §7.3's own statistic, and C6's fundamental is where the
/// prototype broke it (4.9 -> 21.2 dB).
fn aftersound_db(preset: &Preset, key: u8) -> f64 {
    let signal = render(preset, key, REFERENCE_VELOCITY, RENDER_S);
    aftersound_of(preset, key, &signal).unwrap_or(f64::NAN)
}

/// Least-squares slope of `rate = s k` through the origin: what a unison
/// mistuning has to look like, in hertz per partial index.
fn beat_slope(rates: &[(u32, f64)]) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for &(k, hz) in rates {
        num += f64::from(k) * hz;
        den += f64::from(k) * f64::from(k);
    }
    if den > 0.0 {
        num / den
    } else {
        0.0
    }
}

/// The same statistic on a signal the caller already has.
fn aftersound_of(preset: &Preset, key: u8, signal: &[f64]) -> Option<f64> {
    measure(preset, key, signal, 1)
        .first()
        // A fundamental that does not stand over its own neighbourhood has no
        // envelope to read two lines off, and the number a demodulation of the
        // background returns is not a small one.
        .filter(|(_, m)| m.peak_db >= 15.0 && m.aftersound_db.abs() < 60.0)
        .map(|(_, m)| m.aftersound_db)
}

/// The mean, over Column B's own cells, of the beat depth's spread across
/// [`COLUMN_B_VELOCITIES`] — measured on the recordings, which is the target the
/// swing line is inverted against, and its frequency twin beside it.
fn recorded_velocity_spread(
    library: &SampleLibrary,
    preset: &Preset,
    keys: &[u8],
) -> (f64, f64) {
    let mut depth: Vec<Vec<f64>> = Vec::new();
    let mut cents: Vec<Vec<f64>> = Vec::new();
    for &key in keys {
        let mut by_partial: BTreeMap<u32, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
        for &velocity in &COLUMN_B_VELOCITIES {
            let Some(sample) = layer_for(library, key, velocity) else {
                continue;
            };
            let Ok(signal) = recording(sample) else { continue };
            for (k, motion) in measure(preset, key, &signal, SWING_PARTIALS) {
                let entry = by_partial.entry(k).or_default();
                entry.0.push(motion.beat_depth_db);
                entry.1.push(motion.band_cents);
            }
        }
        for (_, (d, c)) in by_partial {
            if d.len() == COLUMN_B_VELOCITIES.len() {
                depth.push(d);
                cents.push(c);
            }
        }
    }
    (
        piano_tuner::estimate::motion::velocity_spread(&depth),
        piano_tuner::estimate::motion::velocity_spread(&cents),
    )
}

/// The same statistic on the engine.
fn rendered_velocity_spread(probe: &Preset, keys: &[u8]) -> (f64, f64) {
    let mut depth: Vec<Vec<f64>> = Vec::new();
    let mut cents: Vec<Vec<f64>> = Vec::new();
    for &key in keys {
        let mut by_partial: BTreeMap<u32, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
        for &velocity in &COLUMN_B_VELOCITIES {
            let signal = render(probe, key, velocity, RENDER_S);
            for (k, motion) in measure(probe, key, &signal, SWING_PARTIALS) {
                let entry = by_partial.entry(k).or_default();
                entry.0.push(motion.beat_depth_db);
                entry.1.push(motion.band_cents);
            }
        }
        for (_, (d, c)) in by_partial {
            if d.len() == COLUMN_B_VELOCITIES.len() {
                depth.push(d);
                cents.push(c);
            }
        }
    }
    (
        piano_tuner::estimate::motion::velocity_spread(&depth),
        piano_tuner::estimate::motion::velocity_spread(&cents),
    )
}

/// One layer's time-zero spectrum as the recording has it./// One layer's time-zero spectrum as the recording has it.
fn recorded_spectrum(
    survey: &SurveyConfig,
    preset: &Preset,
    key: u8,
    sample: &Sample,
) -> Option<Vec<(u32, f64)>> {
    let index = key_index(key)?;
    let audio = audio::load_at(&sample.path, SAMPLE_RATE).ok()?;
    let mono = audio.mono();
    let note_config = survey.note_config(equal_temperament(key)).ok()?;
    let seed = InharmonicModel::harmonic(f64::from(preset.notes.f0_hz[index]));
    let analysis = analyze_note(&mono, SR, seed, &note_config).ok()?;
    Some(spectrum_of(&analysis))
}

/// The same, as the engine renders it.
fn rendered_spectrum(
    survey: &SurveyConfig,
    probe: &Preset,
    key: u8,
    vel: u8,
) -> Option<Vec<(u32, f64)>> {
    let index = key_index(key)?;
    let engine = piano_emulator::preset::Preset::from_toml(&probe.to_toml()).ok()?;
    let events = [RenderEvent::new(0.05, Event::NoteOn { key, vel })];
    let (left, right) = render_to_buffer(&engine, &events, SPECTRUM_RENDER_S);
    let mono: Vec<f32> = left
        .iter()
        .zip(&right)
        .map(|(&l, &r)| 0.5 * (l + r))
        .collect();
    let note_config = survey.note_config(equal_temperament(key)).ok()?;
    let seed = InharmonicModel::harmonic(f64::from(probe.notes.f0_hz[index]));
    let analysis = analyze_note(&mono, SR, seed, &note_config).ok()?;
    Some(spectrum_of(&analysis))
}

fn spectrum_of(analysis: &piano_tuner::NoteAnalysis) -> Vec<(u32, f64)> {
    analysis
        .decays
        .partials
        .iter()
        .filter(|fit| fit.k >= 1 && fit.initial_amplitude() > 0.0)
        .map(|fit| (fit.k, fit.initial_amplitude()))
        .collect()
}
