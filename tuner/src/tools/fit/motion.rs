//! Stage 2, the motion half: the two mechanisms that make a partial *move*, and
//! the two tables the new forward model invalidated.
//!
//! ```text
//! cargo run --release -p piano-tuner -- fit \
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
//!    (`docs/history/FUNDAMENTALS.md` §7.3: *with anti-veering there will be no beats*), so
//!    the map from tuning to beat rate is no longer an identity there and there
//!    is nothing to invert. The partition is not a new judgement: it is the
//!    false-beat fit's own verdict — a key whose measured rates track `k` is
//!    beating because of its tuning, and that is exactly the key whose tuning a
//!    beat rate can be inverted from. The aftersound `docs/history/FUNDAMENTALS.md` §7.5
//!    step 4 names as the check is *reported* against the recording's, not
//!    inverted; [`DETUNE_LO`] carries the measurement that says why.
//! 4. **`notes.partial_gains`** — the full measured ratio `a_k(0) recorded /
//!    a_k(0) as the engine itself renders it` (`DECISIONS.md` 231, 237), taken
//!    against a probe whose own row is **cleared** first, which is what makes
//!    this fit re-entrant where `--stage partials` is not.
//! 5. **The unsampled keys' texture** — `notes.partial_gains` and
//!    `notes.false_beat` **drawn** for the 58 keys the library never sampled
//!    (plus A7 and C8, which it sampled and which measured nothing) from the
//!    distributions the other 28 measured: `estimate::texture`, seeded from the
//!    key number, disciplined by the same rails, the same power pin and the
//!    same `close_on_the_render` as the fitted rows, and recorded in
//!    `notes.synthesized_texture`. `DECISIONS.md` 284-291. Its splits are then
//!    closed on the render too, by `close_splits_on_the_render`, against the
//!    recordings' own beat depth by register and partial — the half item 284
//!    left open and both boards found (`DECISIONS.md` 289, 298, 300).
//!
//! Without `--out` it measures and prints and writes nothing.
//! `--draw-over-measured` is item 290's control and not a way to build an
//! instrument: it draws over the 28 *measured* keys and leaves the rest bare.

use std::collections::BTreeMap;
use std::path::PathBuf;

use rayon::prelude::*;

use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_emulator::types::Event;
use piano_tuner::estimate::motion::{
    fit_false_beat, fit_strike_direction, strike_direction_for, FalseBeatLoop, FalseBeatVerdict,
    MotionConfig, SwingLine, VelocityCell,
};
use piano_tuner::estimate::shaping::{
    energy_offset, flatten_row, measured_over_rendered_report, ShapingConfig, MAX_ROW_CELLS,
};
use piano_tuner::estimate::texture::{fit_texture, SynthesizedTexture, TextureModel};
use piano_tuner::motion::{partial_motion, Motion, Spectrum, WINDOW_HI_S};
use piano_tuner::pipeline::analyze_note;
use piano_tuner::preset::{
    equal_temperament, key_index, FalseBeat, Preset, MAX_PARTIAL_GAIN, MIN_FALSE_BEAT_DB,
    MIN_PARTIAL_GAIN,
};
use piano_tuner::series::{amp_db, Series, PARTIALS, WINDOW_S};
use piano_tuner::survey::SurveyConfig;
use piano_tuner::trajectory::InharmonicModel;
use piano_tuner::{audio, detect_onset, Sample, SampleLibrary, SAMPLE_RATE};

const SR: f64 = SAMPLE_RATE as f64;

/// Decibels per neper.
const NEPERS_TO_DB: f64 = 8.685_889_638_065_035;

/// Velocity every fit is anchored at — `docs/history/TUNING_REPORT.md` §5's own convention,
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

/// The keys the swing line is measured on, and the partials: exactly Column B's
/// own cells (`realism::MOTION_KEYS`, `MOTION_PARTIALS`), because what the field
/// is fitted against is the column it exists to move. Fitting it on some other
/// cell set and hoping would be the mistake `docs/history/FUNDAMENTALS.md` §II.2 convicts the
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
/// the coupled one it is not. `docs/history/FUNDAMENTALS.md` §3.2's anti-veering pulls the
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

pub fn run(args: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
    let mut args = args.into_iter();
    let sfz = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "data/salamander/SalamanderGrandPiano-V3+20200602.sfz".into()),
    );
    let mut preset_path = PathBuf::from("presets/salamander-c5.toml");
    let mut out: Option<PathBuf> = None;
    let mut only: Vec<u8> = Vec::new();
    let mut stages: Vec<String> = Vec::new();
    let mut draw_over_measured = false;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--preset" => preset_path = PathBuf::from(args.next().expect("--preset <file>")),
            "--out" => out = Some(PathBuf::from(args.next().expect("--out <file>"))),
            "--key" => only.push(args.next().expect("--key <n>").parse()?),
            "--stage" => stages.push(args.next().expect("--stage <name>")),
            "--draw-over-measured" => draw_over_measured = true,
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let wants = |name: &str| stages.is_empty() || stages.iter().any(|s| s == name);

    let library = SampleLibrary::from_sfz(&sfz)?;
    let mut preset = Preset::load(&preset_path)?;
    grow_per_key_tables(&mut preset);
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
            // **A row may only name a partial this key's bank actually has**
            // (`DECISIONS.md` 522). The recording is the authority on *which*
            // partials beat and at what rate, and it will happily resolve a
            // fifth partial at the top of the compass; the engine's string
            // bank for that key may hold four, and the schema rejects a
            // `notes.false_beat` row for a partial that does not exist — as it
            // should, since there is nothing for such a row to modulate. This
            // could not happen on a minor-third library because the top key's
            // bank was always the larger of the two; the first whole-tone
            // library found it in `solve_on_the_render`'s first render, as a
            // panic rather than as a bad number, which is the good failure.
            let bank = partial_count(&preset, key) as u32;
            let measured: Vec<(u32, Motion)> = measure(&preset, key, &signal, config.max_partial)
                .into_iter()
                .filter(|(k, _)| *k <= bank)
                .collect();
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
                measured_over_rendered_report(&recorded, &rendered, MAX_ROW_CELLS, &shaping)
            else {
                println!("{key:>4}  {:>6}  nothing fitted", recorded.len());
                continue;
            };
            // Closed on the render, which is the only place the acceptance
            // criterion lives. See `close_on_the_render`.
            let mut fitted = report.gains.clone();
            fitted.truncate(partial_count(&preset, key));
            let target = reference_series(&library, key, &preset).map(|s| s.irregularity());
            let closed =
                close_on_the_render(&preset, &probe, key, &fitted, target, LEVEL_BAND_DB, &shaping);
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

    // ---- 5. the unsampled keys' texture, drawn from the sampled ones' -------
    if wants("texture") {
        let shaping = ShapingConfig::default();
        synthesize_texture(
            &mut preset,
            &library,
            &only,
            draw_over_measured,
            &survey,
            &shaping,
            &config,
        );
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
/// * The compass scores a key against its **neighbours**. When this constant
///   was set, 58 of the 88 were unsampled and carried no row at all: measured
///   with no margin, C2 rendered at 9.80 dB of `irregular` against its own
///   recording's 9.81 — a 0.01 dB match — and still read `z` +4.2, because the
///   keys either side of it rendered at 4.5. Those keys now carry drawn rows
///   (`DECISIONS.md` 284) and are trimmed against the same margin, so what it
///   buys now is the first reason on both halves of the compass rather than the
///   second on one.
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

/// The same band for a **drawn** row: zero.
///
/// [`LEVEL_BAND_DB`] is the width of the piano's own key-to-key level scatter,
/// and what it licenses is a *measured* row carrying the part of a key's level
/// that the piano itself has. A drawn row has measured nothing about this key's
/// level and must therefore carry none of it: what `estimate::texture` draws is
/// a roughness, and a roughness that moved a note's loudness would be a level
/// control with a random number in it.
///
/// Item 282 measured that pinning the *fitted* rows this way costs the
/// scoreboard 0.00 dB and moves no compass flag, so the two settings are not in
/// tension — the band is kept where a row has evidence and closed where it does
/// not.
const DRAWN_LEVEL_BAND_DB: f64 = 0.0;

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
    /// Whether the render ended up under the ceiling at all.
    ///
    /// False is the case `temper`'s header names from the other side: even a
    /// row flattened to its own tilt renders rougher than the recordings of
    /// this register, so the roughness is the engine's and the row is not the
    /// place to fix it. A *measured* row is kept anyway — it is evidence, and
    /// F#7 is the compass's own example (`DECISIONS.md` 280) — and a **drawn**
    /// one is refused by `synthesize_texture`, because a draw that cannot be
    /// brought under the ceiling is not evidence of anything.
    reached: bool,
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
    target_irregular: Option<f64>,
    level_band_db: f64,
    shaping: &ShapingConfig,
) -> ClosedRow {
    let Some(index) = key_index(key) else {
        return ClosedRow {
            row: fitted.to_vec(),
            keep: 1.0,
            target_irregular: f64::NAN,
            rendered_irregular: f64::NAN,
            reached: false,
            level_moved_db: 0.0,
        };
    };
    let with = |row: &[f32]| -> (f64, f64) {
        let mut candidate = preset.clone();
        candidate.notes.partial_gains[index] = row.to_vec();
        rendered_series(&candidate, key)
    };
    let (bare_level, _) = rendered_series(probe, key);
    let Some(target_irregular) = target_irregular else {
        return ClosedRow {
            row: fitted.to_vec(),
            keep: 1.0,
            target_irregular: f64::NAN,
            rendered_irregular: f64::NAN,
            reached: false,
            level_moved_db: 0.0,
        };
    };

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
        let excess = moved - moved.clamp(-level_band_db, level_band_db);
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
        reached: got <= target_irregular - IRREGULAR_MARGIN_DB + LEVEL_TOLERANCE_DB,
        level_moved_db: level - bare_level,
    }
}

// ---------------------------------------------------------------------------
// The drawn splits, closed on the render
// ---------------------------------------------------------------------------

/// Bisection steps on a drawn split's level, after the baseline and after the
/// draw itself have been rendered.
///
/// Six halvings of a bracket that is at most the schema's 40 dB wide leaves
/// 0.6 dB of level, which is under a decibel of the quantity being bounded: the
/// rendered depth moves roughly decibel for decibel with the companion's level
/// while the companion is well under the partial. The cost is
/// `2 + CEILING_STEPS` renders of one key, once, for the 35 keys that draw a
/// split at all.
const CEILING_STEPS: usize = 6;

/// How far under its ceiling a drawn split has to land before the bisection
/// stops asking, dB.
///
/// The same shape of margin as [`IRREGULAR_MARGIN_DB`] and for the first of its
/// two reasons: the ceiling is drawn from depths measured through the sampler's
/// own noise floor, and landing exactly on it would transfer that measurement's
/// noise into the instrument.
const BEAT_MARGIN_DB: f64 = 0.25;

/// One key's drawn splits after the render has had its say.
struct ClosedSplits {
    rows: Vec<FalseBeat>,
    /// Per row of `rows`: the ceiling it was held under and what the render
    /// finally read at that partial, dB.
    ceiling_db: Vec<f64>,
    rendered_db: Vec<f64>,
    /// Rows whose level the ceiling moved, and by how much in total.
    trimmed: usize,
    trimmed_db: f64,
    /// Rows thrown away because the partial beats over its ceiling with the
    /// least this mechanism can add: `(k, the depth the quietest render read,
    /// the ceiling)`. Printed rather than counted, because a refusal is a
    /// statement about the engine's own unison at that partial and the number
    /// is the evidence for it.
    refused: Vec<(u16, f64, f64)>,
    /// Rows the render never resolved, which are kept as drawn: nothing can be
    /// concluded from a partial that did not measure, which is
    /// [`FalseBeatLoop::observe`]'s own rule.
    unobserved: usize,
}

/// Puts the drawn splits on the engine and bisects each one's level until the
/// **rendered** beat depth of its partial is under the ceiling drawn for it.
///
/// # Why this cannot be done in the estimator
///
/// The same reason `close_on_the_render` gives for the gain rows, in a place
/// where the fitted keys already act on it. `notes.false_beat` names a
/// *companion* — a second component `db` under the partial, `hz` away — and the
/// depth that companion produces is a property of the render: the partial is a
/// pair of coupled eigenmodes with its own beating, the master chain is between
/// it and the measurement, and a rate slow against the 2.7 s window shows only
/// part of a cycle. A fitted key therefore never writes what its recording
/// implied — [`FalseBeatLoop`] bisects the ask until the rendered depth **is**
/// the recording's — and before `DECISIONS.md` 300 a drawn key wrote its ask
/// straight out of a distribution whose residual is 10.55 dB.
///
/// # The three verdicts, and the order they are reached in
///
/// 1. **The baseline**, this key with its splits cleared, exactly as
///    [`FalseBeatLoop`]'s round zero. A partial that already beats over its
///    ceiling with no split at all is *refused*: a companion only adds, so
///    there is no level that reaches the target, and the roughness the key has
///    is its unison's and not a wire's to write.
/// 2. **The draw as it stands.** Under the ceiling, it is kept whole — the
///    amount is a draw and not a target, and lifting a quiet key onto its
///    ceiling would be the seam with its sign reversed.
/// 3. **The bisection**, between the schema's quietest companion and the drawn
///    ask, keeping the loudest level whose render came in under the ceiling.
///    The depth is monotone in the level, so that level is simply the bracket's
///    own lower end and there is no separate best-point bookkeeping — which is
///    the one place this differs from [`FalseBeatLoop`], whose objective also
///    contains a frequency deviation, is not monotone in anything, and has to
///    keep the best point it visited rather than the one it stopped on.
///
/// The rate is never touched. It is drawn i.i.d. of the partial number and of
/// the register because that is what the fitted rows measured (`DECISIONS.md`
/// 285), it passes item 233's falsification as drawn, and a loop that moved it
/// against a rendered deviation would be fitting a quantity this key has no
/// recording of.
fn close_splits_on_the_render(
    base: &Preset,
    key: u8,
    drawn: &[FalseBeat],
    ceilings: &[f64],
    config: &MotionConfig,
) -> ClosedSplits {
    let mut out = ClosedSplits {
        rows: Vec::new(),
        ceiling_db: Vec::new(),
        rendered_db: Vec::new(),
        trimmed: 0,
        trimmed_db: 0.0,
        refused: Vec::new(),
        unobserved: 0,
    };
    assert_eq!(
        drawn.len(),
        ceilings.len(),
        "key {key} drew {} splits and {} ceilings",
        drawn.len(),
        ceilings.len()
    );
    if drawn.is_empty() {
        return out;
    }
    let depths = |probe: &Preset| -> Vec<(u32, Motion)> {
        let signal = render(probe, key, REFERENCE_VELOCITY, RENDER_S);
        measure(probe, key, &signal, config.max_partial)
    };
    let depth_of = |seen: &[(u32, Motion)], k: u16| -> Option<f64> {
        seen.iter()
            .find(|(index, _)| *index == u32::from(k))
            .filter(|(_, m)| m.peak_db >= config.min_peak_db)
            .map(|(_, m)| m.beat_depth_db)
    };

    // 1. the baseline: this key with no splits at all.
    let mut probe = base.clone();
    set_false_beat(&mut probe, key, &[]);
    let bare = depths(&probe);

    struct State {
        row: FalseBeat,
        ceiling: f64,
        /// The bracket on the ask, in dB: `lo` is quiet enough to be under the
        /// ceiling and `hi` is the drawn ask.
        lo: f64,
        hi: f64,
        /// The loudest ask whose render came in under the ceiling, and what
        /// that render read.
        best: Option<(f64, f64)>,
        /// The last depth any round read at this partial, for the report that
        /// has to say why a draw was thrown away.
        seen: f64,
        settled: bool,
    }
    impl State {
        /// What to ask the engine for in this round: the draw itself first,
        /// then the midpoint of the bracket, and nothing more once the draw has
        /// been accepted whole.
        fn ask(&self, round: usize) -> f64 {
            match self.best {
                Some((db, _)) if self.settled => db,
                _ if round == 0 => self.hi,
                _ => 0.5 * (self.lo + self.hi),
            }
        }
    }
    let mut states: Vec<State> = Vec::new();
    for (row, &ceiling) in drawn.iter().zip(ceilings) {
        match depth_of(&bare, row.k) {
            // The partial already out-beats the piano's own register with
            // nothing written. A companion only adds.
            Some(depth) if depth > ceiling - BEAT_MARGIN_DB => {
                out.refused.push((row.k, depth, ceiling));
                continue;
            }
            Some(depth) => states.push(State {
                row: *row,
                ceiling,
                lo: f64::from(MIN_FALSE_BEAT_DB),
                hi: f64::from(row.db),
                best: None,
                seen: depth,
                settled: false,
            }),
            // Not resolved on the render: nothing can be concluded, so the draw
            // stands. `FalseBeatLoop::observe` does the same.
            None => {
                out.unobserved += 1;
                out.rows.push(*row);
                out.ceiling_db.push(ceiling);
                out.rendered_db.push(f64::NAN);
            }
        }
    }
    if states.is_empty() {
        return out.sorted();
    }

    // 2. the draw as it stands, and 3. the bisection.
    for round in 0..=CEILING_STEPS {
        let rows: Vec<FalseBeat> = states
            .iter()
            .map(|s| FalseBeat {
                db: s.ask(round) as f32,
                ..s.row
            })
            .collect();
        set_false_beat(&mut probe, key, &rows);
        let seen = depths(&probe);
        for state in &mut states {
            let asked = state.ask(round);
            let Some(depth) = depth_of(&seen, state.row.k) else {
                // The render stopped resolving this partial. The state stands,
                // which for round zero means the drawn ask stands.
                continue;
            };
            state.seen = depth;
            if depth <= state.ceiling - BEAT_MARGIN_DB {
                state.best = Some((asked, depth));
                state.lo = asked;
                if round == 0 {
                    // The draw is already under its ceiling: it is a draw and
                    // not a target, so nothing is trimmed.
                    state.settled = true;
                }
            } else {
                state.hi = asked;
            }
        }
        if states.iter().all(|s| s.settled) {
            break;
        }
    }

    for state in states {
        let Some((db, depth)) = state.best else {
            // Every level the bracket held rendered over the ceiling, including
            // the schema's quietest companion. Same verdict as the baseline's.
            out.refused.push((state.row.k, state.seen, state.ceiling));
            continue;
        };
        if (db - f64::from(state.row.db)).abs() > 1e-6 {
            out.trimmed += 1;
            out.trimmed_db += f64::from(state.row.db) - db;
        }
        out.rows.push(FalseBeat {
            db: db as f32,
            ..state.row
        });
        out.ceiling_db.push(state.ceiling);
        out.rendered_db.push(depth);
    }
    out.sorted()
}

impl ClosedSplits {
    /// Rows in partial order, which is the order the schema and every report
    /// read them in.
    fn sorted(mut self) -> ClosedSplits {
        let mut order: Vec<usize> = (0..self.rows.len()).collect();
        order.sort_by_key(|&i| self.rows[i].k);
        self.rows = order.iter().map(|&i| self.rows[i]).collect();
        self.ceiling_db = order.iter().map(|&i| self.ceiling_db[i]).collect();
        self.rendered_db = order.iter().map(|&i| self.rendered_db[i]).collect();
        self
    }
}

// ---------------------------------------------------------------------------
// The unsampled keys' texture, drawn and then closed on the same render
// ---------------------------------------------------------------------------

/// Draws `notes.partial_gains` and `notes.false_beat` for every key that has no
/// measured row, and writes the provenance list beside them.
///
/// # The order, and why each step is where it is
///
/// 1. **Clear first.** Every key `notes.synthesized_texture` names is emptied
///    before anything is fitted, so the distributions are measured from the
///    *measured* keys only and a second run of this stage is the first run.
///    That is item 214's re-entrancy rule applied to a draw: the fit must be a
///    function of the recordings, not of its own last output.
/// 2. **Fit the distributions** from what is left, with two numbers per fitted
///    key coming from outside the preset: the recording's own `irregular`,
///    measured through `series::Series`, which is the ceiling a row is trimmed
///    against; and the recording's own beat depth per partial, measured through
///    `motion`, which is the ceiling a *split* is trimmed against. Both are
///    measured the way the acceptance test measures them, because that is what
///    an acceptance criterion is.
/// 3. **Draw, then discipline, splits first.** The splits are closed on the
///    render by [`close_splits_on_the_render`] before the row is, because that
///    is the order the fitted keys are fitted in — stage 1 solves
///    `notes.false_beat` on a preset that does not yet carry stage 4's row — and
///    because the row is then pinned and trimmed on a key that beats the way it
///    will ship. Then the row: rails from its own spread, the power pin on the
///    engine's own rendered spectrum, and [`close_on_the_render`] — the fitted
///    keys' own function, with the drawn ceiling in place of the recording's.
///    A drawn row that renders rougher than the recordings of its register is
///    trimmed exactly as a measured one is, and a drawn row that renders smoother
///    is left alone exactly as a measured one is: the amount is a *draw*, not a
///    target, and forcing every drawn key up onto its ceiling would put the
///    unsampled keys systematically rougher than the sampled ones — the same
///    seam with its sign reversed. The splits are closed under the same two
///    rules.
///
/// The loop is parallel and collects in key order (item 283): each key renders
/// its own engine from its own clone of the preset and shares nothing.
fn synthesize_texture(
    preset: &mut Preset,
    library: &SampleLibrary,
    only: &[u8],
    draw_over_measured: bool,
    survey: &SurveyConfig,
    shaping: &ShapingConfig,
    motion: &MotionConfig,
) {
    println!("\n== notes.partial_gains / notes.false_beat, synthesized ==");
    // 1. clear
    for key in std::mem::take(&mut preset.notes.synthesized_texture) {
        if let Some(index) = key_index(key) {
            if index < preset.notes.partial_gains.len() {
                preset.notes.partial_gains[index] = Vec::new();
            }
            set_false_beat(preset, key, &[]);
        }
    }
    if preset.notes.partial_gains.len() != piano_tuner::preset::NUM_KEYS {
        preset.notes.partial_gains = vec![Vec::new(); piano_tuner::preset::NUM_KEYS];
    }
    if preset.notes.false_beat.len() != piano_tuner::preset::NUM_KEYS {
        preset.notes.false_beat = vec![Vec::new(); piano_tuner::preset::NUM_KEYS];
    }

    // 2. fit
    let measured: Vec<u8> = (piano_tuner::preset::LOWEST_KEY..=piano_tuner::preset::HIGHEST_KEY)
        .filter(|&key| key_index(key).is_some_and(|i| !preset.notes.partial_gains[i].is_empty()))
        .collect();
    let targets: Vec<(u8, f64)> = measured
        .par_iter()
        .filter_map(|&key| {
            reference_series(library, key, preset).map(|s| (key, s.irregularity()))
        })
        .collect();
    // The second measurement from outside the preset: how deeply the piano's
    // own partials beat, key by key and partial by partial, read with the same
    // `measure` the fitted keys' own `FalseBeatLoop` targets are read with. It
    // is the ceiling the drawn splits are closed on — `DECISIONS.md` 300 — and
    // it cannot be taken from the preset, because what a preset carries is the
    // companion level a fit *asked* for and not the depth it produced.
    let beat_depths: Vec<(u8, u16, f64)> = measured
        .par_iter()
        .flat_map(|&key| {
            let Some(sample) = layer_for(library, key, REFERENCE_VELOCITY) else {
                return Vec::new();
            };
            let Ok(signal) = recording(sample) else {
                return Vec::new();
            };
            measure(preset, key, &signal, motion.max_partial)
                .into_iter()
                .filter(|(_, m)| m.peak_db >= motion.min_peak_db)
                .map(|(k, m)| (key, k as u16, m.beat_depth_db))
                .collect::<Vec<_>>()
        })
        .collect();
    let model = fit_texture(
        &preset.notes.partial_gains,
        &preset.notes.false_beat,
        &targets,
        &beat_depths,
        shaping,
    );
    report_model(&model);

    // The control the refusal rate has to be read against, measured on the
    // *sampled* keys and not assumed: how often the engine's own unison already
    // beats deeper than the recording of the same key at the same partial, with
    // `notes.false_beat` cleared. Those are the cells `FalseBeatLoop` drops for
    // a fitted key — a companion only adds, so no level reaches the target — and
    // a drawn key's ceiling refuses exactly the same cells for exactly the same
    // reason. If the two rates agree, the splits a drawn key loses are the
    // instrument's own statistic rather than the ceiling's severity.
    let bare_over: Vec<(bool, f64)> = measured
        .par_iter()
        .filter(|&&key| key <= piano_tuner::estimate::texture::HIGHEST_FALSE_BEAT_KEY)
        .flat_map(|&key| {
            let Some(index) = key_index(key) else {
                return Vec::new();
            };
            let mut probe = preset.clone();
            probe.notes.false_beat[index] = Vec::new();
            let signal = render(&probe, key, REFERENCE_VELOCITY, RENDER_S);
            let rendered = measure(&probe, key, &signal, motion.max_partial);
            beat_depths
                .iter()
                .filter(|&&(k, _, _)| k == key)
                .filter_map(|&(_, k, recorded)| {
                    rendered
                        .iter()
                        .find(|(index, _)| *index == u32::from(k))
                        .filter(|(_, m)| m.peak_db >= motion.min_peak_db)
                        .map(|(_, m)| (m.beat_depth_db >= recorded, m.beat_depth_db - recorded))
                })
                .collect::<Vec<_>>()
        })
        .collect();
    if !bare_over.is_empty() {
        let over = bare_over.iter().filter(|c| c.0).count();
        let mut signed: Vec<f64> = bare_over.iter().map(|c| c.1).collect();
        signed.sort_by(f64::total_cmp);
        println!(
            "  control    with no split written at all, the engine's own unison already beats \
             over the recording's own depth at {over} of {} sampled cells ({:.0} %), median \
             {:+.2} dB — the cells a fitted key's loop drops, and the same cells a drawn key's \
             ceiling refuses",
            bare_over.len(),
            100.0 * over as f64 / bare_over.len() as f64,
            signed[signed.len() / 2],
        );
    }

    // `--draw-over-measured` is `DECISIONS.md` 284's own control and not a way
    // to build an instrument: it draws over the **measured** keys and leaves
    // the unsampled ones bare, which is the shipped preset with the same 28
    // rows *drawn* instead of fitted. The scoreboard read on it is what
    // separates "texture costs the mel board" from "these particular 60 keys
    // cost it", because a mean absolute distance is minimised by the reference's
    // own mean and a row that is right in distribution and wrong in detail is
    // further from it than no row at all.
    let drawn: Vec<u8> = (piano_tuner::preset::LOWEST_KEY..=piano_tuner::preset::HIGHEST_KEY)
        .filter(|key| measured.contains(key) == draw_over_measured)
        .filter(|key| only.is_empty() || only.contains(key))
        .collect();
    if draw_over_measured {
        println!("CONTROL: drawing over the {} measured keys", measured.len());
        for &key in &measured {
            if let Some(index) = key_index(key) {
                preset.notes.partial_gains[index] = Vec::new();
                preset.notes.false_beat[index] = Vec::new();
            }
        }
    }
    println!(
        "\n key  cells  amount drawn/written  rail railed   level   keep  irreg target/rendered  \
         moved   splits (k: hz / dB)"
    );

    // 3. draw and close
    let rows: Vec<(u8, SynthesizedTexture, ClosedSplits, ClosedRow)> = drawn
        .par_iter()
        .filter_map(|&key| {
            let index = key_index(key)?;
            let partials = partial_count(preset, key);
            let synthesized = model.synthesize(key, partials, shaping);
            // The splits first, in the order the fitted keys are fitted in:
            // stage 1 solves `notes.false_beat` on a preset that does not yet
            // carry stage 4's gain row, so a drawn key closes its splits on the
            // same instrument a measured one did.
            let mut bare = preset.clone();
            bare.notes.partial_gains[index] = Vec::new();
            bare.notes.false_beat[index] = Vec::new();
            let splits = close_splits_on_the_render(
                &bare,
                key,
                &synthesized.false_beat,
                &synthesized.beat_ceiling_db,
                motion,
            );
            // The instrument the row is measured on and against: this key with
            // its own closed splits and no gain row at all.
            let mut base = preset.clone();
            base.notes.partial_gains[index] = Vec::new();
            base.notes.false_beat[index] = splits.rows.clone();
            // The power pin, on the engine's own spectrum — the same quantity
            // `measured_over_rendered_report` pins against, and the same
            // function.
            let engine = rendered_spectrum(survey, &base, key, REFERENCE_VELOCITY)?;
            let cells = synthesized.gains_db.len();
            let row: Vec<Option<f64>> = synthesized
                .gains_db
                .iter()
                .map(|db| Some(db / NEPERS_TO_DB))
                .collect();
            let mut engine_line = vec![None; cells];
            for (k, amplitude) in engine {
                if k >= 1 && (k as usize) <= cells && amplitude > 0.0 {
                    engine_line[k as usize - 1] = Some(amplitude.ln());
                }
            }
            let (levelled, _) = energy_offset(&row, &engine_line);
            let fitted: Vec<f32> = levelled
                .iter()
                .map(|cell| match cell {
                    Some(ln) => (ln.exp() as f32).clamp(MIN_PARTIAL_GAIN, MAX_PARTIAL_GAIN),
                    None => 1.0,
                })
                .collect();
            let closed = close_on_the_render(
                &base,
                &base,
                key,
                &fitted,
                Some(synthesized.target_irregular),
                DRAWN_LEVEL_BAND_DB,
                shaping,
            );
            Some((key, synthesized, splits, closed))
        })
        .collect();

    let mut refused = 0usize;
    for (key, synthesized, splits, closed) in &rows {
        let Some(index) = key_index(*key) else { continue };
        // A draw the ceiling refused writes nothing. See `ClosedRow::reached`:
        // the key already renders rougher than the recordings of its register
        // without any row at all, and the answer to that is not a row.
        let mut row: Vec<f32> = if closed.reached {
            closed
                .row
                .iter()
                .map(|g| g.clamp(MIN_PARTIAL_GAIN, MAX_PARTIAL_GAIN))
                .collect()
        } else {
            refused += 1;
            Vec::new()
        };
        while row.last() == Some(&1.0) {
            row.pop();
        }
        preset.notes.partial_gains[index] = row;
        preset.notes.false_beat[index] = splits.rows.clone();
        // Named only if something was written. A key whose gain row the ceiling
        // refused and whose wire drew no split carries no drawn number, so it
        // has no provenance to record: the field says which *rows* are drawn,
        // and an empty row is not one.
        if !preset.notes.partial_gains[index].is_empty()
            || !preset.notes.false_beat[index].is_empty()
        {
            preset.notes.synthesized_texture.push(*key);
        }
        let db: Vec<f64> = preset.notes.partial_gains[index]
            .iter()
            .map(|&g| 20.0 * f64::from(g).log10())
            .collect();
        println!(
            "{:>4}  {:>5}  {:>6.2}/{:<6.2}  {:>5.2} {:>5}  {:>+6.2}  {:>4.2}  {:>5.2}/{:<5.2}  \
             {:>+5.2}  {}{}",
            key,
            preset.notes.partial_gains[index].len(),
            synthesized.drawn_amount_db,
            piano_tuner::estimate::texture::robust_sigma(&db),
            synthesized.rail_db,
            synthesized.railed_cells,
            closed.level_moved_db,
            closed.keep,
            synthesized.target_irregular,
            closed.rendered_irregular,
            closed.level_moved_db,
            if closed.reached { "" } else { "REFUSED  " },
            // Per written split: the partial, its rate, the ask that survived
            // the ceiling, the ceiling it was held under and what the render
            // read at it. A row whose ask the draw wrote whole prints the same
            // number twice, which is how a key that needed no trim reads.
            splits
                .rows
                .iter()
                .zip(splits.ceiling_db.iter().zip(&splits.rendered_db))
                .map(|(r, (ceiling, rendered))| {
                    format!(
                        "{}: {:.2}/{:.1} <= {:.1} ({:.1})",
                        r.k, r.hz, r.db, rendered, ceiling
                    )
                })
                .chain(splits.refused.iter().map(|&(k, bare, ceiling)| {
                    format!("{k}: refused ({bare:.1} > {ceiling:.1})")
                }))
                .collect::<Vec<_>>()
                .join("  ")
        );
    }
    let written: usize = rows.iter().map(|r| r.2.rows.len()).sum();
    let with_splits = rows.iter().filter(|r| !r.2.rows.is_empty()).count();
    let drew: usize = rows.iter().map(|r| r.1.false_beat.len()).sum();
    let trimmed: usize = rows.iter().map(|r| r.2.trimmed).sum();
    let trimmed_db: f64 = rows.iter().map(|r| r.2.trimmed_db).sum();
    let refused_splits: usize = rows.iter().map(|r| r.2.refused.len()).sum();
    let unobserved: usize = rows.iter().map(|r| r.2.unobserved).sum();
    println!(
        "drawn: {} keys, {} gain cells, {} split rows over {} keys; \
         {refused} rows refused by their own ceiling; {} keys keep their measured rows",
        rows.len(),
        rows.iter()
            .filter_map(|r| key_index(r.0))
            .map(|i| preset.notes.partial_gains[i].len())
            .sum::<usize>(),
        written,
        with_splits,
        measured.len()
    );
    println!(
        "splits closed on the render: {drew} drawn, {trimmed} trimmed by \
         {:.1} dB on average, {refused_splits} refused (the partial already beats \
         over its ceiling with nothing written), {unobserved} kept unclosed (the \
         render did not resolve the partial)",
        if trimmed > 0 {
            trimmed_db / trimmed as f64
        } else {
            0.0
        }
    );
}

/// Prints the fitted distributions, which are the milestone's own report.
fn report_model(model: &TextureModel) {
    println!(
        "fitted from {} keys ({} split rows)\n  \
         cells      exp({:+.5} key {:+.3}), scatter x{:.2}, r {:+.3} -> {:.0} at A0, {:.0} at C4, \
         {:.0} at F#7\n  \
         amount     exp({:+.5} key {:+.3}) dB, scatter x{:.2}, r {:+.3} -> {:.2} at A0, {:.2} at \
         C4, {:.2} at F#7\n  \
         lag-1 rho  {:+.3} over {} rows\n  \
         ceiling    exp({:+.3} {:+.5} key {:+.7} key^2), scatter x{:.2}, R2 {:.3} -> {:.1} at A0, \
         {:.1} at C4, {:.1} at F#7\n  \
         splits     P(any) {:.2} at or under key {}, none above; count exp({:+.5} key {:+.3}), \
         scatter x{:.2}, r {:+.3}\n  \
         rate       lognormal ln-mean {:+.3} ln-sd {:.3} ({:.2} Hz median, {:.2}..{:.2} band), \
         r(k, rate) {:+.3} within a key, r(key, rate) {:+.3}\n  \
         depth      {:.2} {:+.4} key {:+.3} k dB, scatter {:.2} dB, R2 {:.3}\n  \
         beat       exp({:+.3} {:+.5} key {:+.4} k) dB over {} cells at {} keys, scatter x{:.2} \
         (x{:.2} between keys, x{:.2} within one), R2 {:.3} (key alone {:.3}, k alone {:.3}) -> \
         {:.2} at A0 k1, {:.2} at C4 k1, {:.2} at C4 k8\n  \
         partials   {}",
        model.fitted_keys.len(),
        model.false_beat_rows,
        model.cells.slope,
        model.cells.intercept,
        model.cells.sigma.exp(),
        model.cells.correlation,
        model.cells.at(21),
        model.cells.at(60),
        model.cells.at(102),
        model.amount.slope,
        model.amount.intercept,
        model.amount.sigma.exp(),
        model.amount.correlation,
        model.amount.at(21),
        model.amount.at(60),
        model.amount.at(102),
        model.rho,
        model.rho_rows,
        model.target.coefficients[0],
        model.target.coefficients[1],
        model.target.coefficients[2],
        model.target.sigma.exp(),
        model.target.r_squared,
        model.target.at(21),
        model.target.at(60),
        model.target.at(102),
        model.false_beat_probability,
        piano_tuner::estimate::texture::HIGHEST_FALSE_BEAT_KEY,
        model.false_beat_count.slope,
        model.false_beat_count.intercept,
        model.false_beat_count.sigma.exp(),
        model.false_beat_count.correlation,
        model.rate_ln_mean,
        model.rate_ln_sigma,
        model.rate_ln_mean.exp(),
        piano_tuner::estimate::texture::MIN_FITTED_HZ,
        f64::from(piano_tuner::preset::MAX_FALSE_BEAT_HZ),
        model.rate_vs_partial,
        model.rate_vs_key,
        model.depth[0],
        model.depth[1],
        model.depth[2],
        model.depth_sigma,
        model.depth_r_squared,
        model.beat_ceiling.coefficients[0],
        model.beat_ceiling.coefficients[1],
        model.beat_ceiling.coefficients[2],
        model.beat_ceiling.points,
        model.beat_ceiling.keys,
        model.beat_ceiling.sigma.exp(),
        model.beat_ceiling.sigma_key.exp(),
        model.beat_ceiling.sigma_cell.exp(),
        model.beat_ceiling.r_squared,
        model.beat_ceiling.r_squared_key_only,
        model.beat_ceiling.r_squared_partial_only,
        model.beat_ceiling.at(21, 1),
        model.beat_ceiling.at(60, 1),
        model.beat_ceiling.at(60, 8),
        model
            .partial_weights
            .iter()
            .enumerate()
            .map(|(i, w)| format!("k{}: {:.2}", i + 1, w))
            .collect::<Vec<_>>()
            .join("  "),
    );
}

/// `(level dBFS, series)` of one key struck alone through `preset`, measured
/// exactly as `compass` measures it.
fn rendered_series(preset: &Preset, key: u8) -> (f64, f64) {
    let engine = match piano_emulator::preset::Preset::from_toml(&preset.to_toml()) {
        Ok(engine) => engine,
        Err(_) => return (f64::NAN, f64::NAN),
    };
    let events = [RenderEvent::new(
        PREROLL_S as f32,
        Event::NoteOn {
            key,
            vel: u16::from(REFERENCE_VELOCITY),
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


/// Grows the two per-key texture tables to the full compass, so that a stage
/// may index into them by key.
///
/// **Both are absent-means-old and may arrive empty**, and `presets/default.toml`
/// — the base every new preset is built from — carries neither.
/// `synthesize_texture` has grown them since it was written; the
/// `partial_gains` stage never did, and never had to, because
/// `presets/salamander-c5.toml` grew its table long ago and every run since has
/// been over a preset that already carried one. **The first factory run that
/// started from the default on a new library found it**, as `index out of
/// bounds: the len is 0 but the index is 0` at the first key of
/// `notes.partial_gains` — and a stage that cannot run on a fresh preset is a
/// stage that cannot make a second piano (`DECISIONS.md` 522).
///
/// A table that is already the right length is left exactly as it is: this must
/// not clear a preset that has been fitted before.
///
/// `notes.partial_sigma_scale` is grown here too, and it is **not** a table this
/// stage writes. It is `tail`'s, and `tools/tail.rs` indexes it by key without a
/// guard of its own — the identical bug, a third time, at `tail.rs:490`. That
/// file belongs to the halo workstream, so the one-line guard it owes is queued
/// rather than made here (the same reason `instrument_path` has three
/// non-adopters, item 521); growing it at the end of `fit`, which every preset
/// passes through before `tail` ever sees it, is what makes the factory run
/// end-to-end on a new library today. **Delete this third clause when
/// `tail.rs` grows its own.**
fn grow_per_key_tables(preset: &mut Preset) {
    if preset.notes.partial_gains.len() != piano_tuner::preset::NUM_KEYS {
        preset.notes.partial_gains = vec![Vec::new(); piano_tuner::preset::NUM_KEYS];
    }
    if preset.notes.false_beat.len() != piano_tuner::preset::NUM_KEYS {
        preset.notes.false_beat = vec![Vec::new(); piano_tuner::preset::NUM_KEYS];
    }
    if preset.notes.partial_sigma_scale.len() != piano_tuner::preset::NUM_KEYS {
        preset.notes.partial_sigma_scale = vec![Vec::new(); piano_tuner::preset::NUM_KEYS];
    }
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

/// Writes the fitted rows into the table, leaving every key this fit did not
/// measure as it found it.
///
/// A merge and not a replacement, because since `DECISIONS.md` 284 the table
/// also carries the 60 unsampled keys' **drawn** rows and this stage runs over
/// the 30 sampled ones. Replacing it wholesale would silently empty the other
/// sixty and leave `notes.synthesized_texture` naming keys with nothing in
/// them.
fn write_false_beats(
    preset: &mut Preset,
    fits: &BTreeMap<u8, piano_tuner::estimate::motion::FalseBeatFit>,
) {
    if fits.is_empty() {
        return;
    }
    if preset.notes.false_beat.len() != piano_tuner::preset::NUM_KEYS {
        preset.notes.false_beat = vec![Vec::new(); piano_tuner::preset::NUM_KEYS];
    }
    for (&key, fit) in fits {
        if let Some(index) = key_index(key) {
            preset.notes.false_beat[index] = fit.rows.clone();
        }
    }
    if preset.notes.false_beat.iter().all(Vec::is_empty) {
        preset.notes.false_beat = Vec::new();
    }
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
        Event::NoteOn { key, vel: u16::from(vel) },
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
/// `docs/history/FUNDAMENTALS.md` §7.3's own statistic, and C6's fundamental is where the
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

/// One layer's time-zero spectrum as the recording has it.
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
    let events = [RenderEvent::new(0.05, Event::NoteOn { key, vel: u16::from(vel) })];
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


#[cfg(test)]
mod growth_tests {
    use super::*;

    /// The falsification for `DECISIONS.md` 522's second half: a factory run
    /// that starts from `presets/default.toml` — which carries neither table —
    /// used to panic at the first key of the `partial_gains` stage.
    #[test]
    fn a_fresh_preset_grows_its_per_key_tables_before_a_stage_indexes_them() {
        let repo = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("the tuner sits in the workspace")
            .to_path_buf();
        let mut preset = Preset::load(repo.join("presets/default.toml")).expect("the base loads");
        // The precondition: this is what the factory actually starts from.
        assert!(
            preset.notes.partial_gains.is_empty(),
            "presets/default.toml has grown a partial_gains table; this test's \
             precondition is gone and the bug it pins can no longer be reached \
             from the default"
        );
        grow_per_key_tables(&mut preset);
        assert_eq!(preset.notes.partial_gains.len(), piano_tuner::preset::NUM_KEYS);
        assert_eq!(preset.notes.false_beat.len(), piano_tuner::preset::NUM_KEYS);
        assert_eq!(
            preset.notes.partial_sigma_scale.len(),
            piano_tuner::preset::NUM_KEYS,
            "tail indexes this one by key and has no guard of its own"
        );
        // Every key of the compass is now indexable, which is the whole claim.
        for key in piano_tuner::preset::LOWEST_KEY..=piano_tuner::preset::HIGHEST_KEY {
            let index = key_index(key).expect("on the keyboard");
            assert!(preset.notes.partial_gains[index].is_empty());
            assert!(preset.notes.false_beat[index].is_empty());
        }
    }

    /// And it must not clear a preset that has already been fitted: growth is
    /// growth, not a reset.
    #[test]
    fn growing_a_table_that_is_already_full_length_changes_nothing() {
        let repo = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("the tuner sits in the workspace")
            .to_path_buf();
        let mut preset = Preset::load(repo.join("presets/default.toml")).expect("the base loads");
        preset.notes.partial_gains = vec![Vec::new(); piano_tuner::preset::NUM_KEYS];
        preset.notes.partial_gains[0] = vec![1.0, 2.0, 3.0];
        preset.notes.false_beat = vec![Vec::new(); piano_tuner::preset::NUM_KEYS];
        let before = preset.notes.partial_gains.clone();
        grow_per_key_tables(&mut preset);
        assert_eq!(preset.notes.partial_gains, before);
    }
}
