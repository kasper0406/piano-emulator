//! Stage 2, the mechanism's balance: how loud the hammer is against the note it
//! belongs to, engine against the recording of the same key, and the level of
//! `[noise.strike]` that puts the two on top of each other.
//!
//! `DECISIONS.md` 338-341. `estimate::attack` fits this event's *colour* and a
//! level referenced to the note's **peak**; `fit --stage partials` then corrects
//! that level on the engine's own render. Neither closes on a ratio, and a ratio
//! is what a listener hears — so anything that moves the attack's tonal content
//! without moving the note's peak moves the balance and nothing in the factory
//! notices. That is what happened between the milestone this event was fitted in
//! and the one that found it: the event never moved and the instrument around it
//! did.
//!
//! # What is measured
//!
//! Per recorded key and per velocity, on isolated notes:
//!
//! * the **recording's** attack tonality
//!   ([`piano_tuner::estimate::attack::noise_to_tone_db`]) — the arithmetic over
//!   the geometric mean of the power spectrum of the first 30 ms from its own
//!   onset, which is a noise-to-tone ratio needing no level match;
//! * the **engine's**, as the preset ships;
//! * the **engine's with the event silenced** — the tonal attack alone;
//! * the offset on the event's level that puts the second on the first.
//!
//! **Recorded keys only** (`DECISIONS.md` 328): every row is the engine at a key
//! against a recording *of that key*. The transposed keys are still played by
//! everything else in the repository and are not scored here.
//!
//! # Why the inversion is exact
//!
//! Two renders per note — with the event and without it — and the sample-wise
//! difference **is** the event, through the board, the master gain and its own
//! filters. Every other level of it is then
//! [`mix`](piano_tuner::estimate::attack::mix) and no render is repeated. So the
//! answer is not a search over presets and not a prediction: it is the same
//! output-referenced inversion `CombLine`, the damper line and `strike_offset`
//! are (`DECISIONS.md` 199, 203, 211), with the estimator between the render and
//! the number removed.
//!
//! # What it writes
//!
//! `--out <file>` applies the fitted correction to `[noise.strike]`: the level
//! at the nominal drive to every anchor, and the slope in drive to
//! `velocity_db`. Those are the event's only two level fields, and they are
//! exactly what a line through the per-note offsets has.
//!
//! **It is re-entrant.** The correction is measured on whatever preset it is
//! given, so running it over its own output measures a corrected instrument and
//! asks for nothing more; `estimate::attack`'s `the_balance_is_a_fixed_point`
//! gates the arithmetic and the second pass over the shipped preset returns
//! −0.00 dB on both fields.
//!
//! # The other four events, and why they are here
//!
//! `--stage mechanism` writes the *other* four events — `key_off`,
//! `damper_lift`, `pedal_down`, `pedal_up` — through
//! [`estimate::noise::fit_noise_screened`], which is the same code `survey`
//! runs. It has no engine in it and no render: those four are read off the
//! library's own mechanism recordings against strikes of the same key, so
//! nothing about them depends on the six stages that run between `survey` and
//! here. It exists because they are written by stage **1** and are therefore
//! unreachable on a finished preset without re-running the whole factory —
//! which is exactly the position `DECISIONS.md` 531 found the range's two new
//! presets in, shipping mechanism tables that stage 1 should have refused.
//!
//! The tables a library's recordings do not earn are taken from `--base`
//! (`presets/default.toml` by default) rather than from the preset being
//! written, so running this over a contaminated preset **repairs** it instead
//! of preserving it. `[noise.strike]` is never touched by this stage: it is the
//! balance fit's, above, and it is kept from the preset being written.
//!
//! ```sh
//! cargo run --release -p piano-tuner -- noise \
//!     [data/salamander] [presets/salamander-c5.toml] [--out <f>] [--key <n>]
//! cargo run --release -p piano-tuner -- noise \
//!     [data] [preset.toml] --stage mechanism [--base presets/default.toml] [--out <f>]
//! ```

use std::cell::RefCell;
use std::path::{Path, PathBuf};

use rayon::prelude::*;

use piano_emulator::preset::{NoiseAnchor, Preset, SILENT_LEVEL_DB};
use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_emulator::types::Event;
use piano_tuner::cache;
use piano_tuner::estimate::attack::{
    balance_reading, fit_balance, BalanceReading, BalanceVerdict,
};
use piano_tuner::estimate::melody::note_onset;
use piano_tuner::estimate::noise::{fit_noise_screened, NoiseConfig};
use piano_tuner::realism::RecordedKeys;
use piano_tuner::survey::measure_mechanism;
use piano_tuner::sampler::SAMPLER_VERSION;
use piano_tuner::{Audio, SampleLibrary, Sampler, TimedEvent, SAMPLE_RATE};

/// The velocities the balance is read at.
///
/// Five, spanning the library's own range, because the correction has a **slope
/// in drive** and a slope needs more than the nominal point: measured on the
/// preset this tool was written for, the offset the balance asks for runs from
/// −17 dB at velocity 24 to −3 dB at velocity 110, which is a velocity law and
/// not a level. They are MIDI velocities rather than layer indices so that the
/// same list means the same thing on a library with a different number of
/// layers.
pub const VELOCITIES: [u8; 5] = [24, 48, 72, 88, 110];

/// Seconds of note rendered, and how long the key is held.
const HOLD_S: f64 = 0.5;
const RENDER_S: f64 = 0.8;
const PREROLL_S: f64 = 0.05;
const SR: f64 = SAMPLE_RATE as f64;

/// Fewest readings a correction may be fitted from.
///
/// Thirty is one velocity at every recorded key, or five velocities at six of
/// them; under it the line is being drawn through a register rather than
/// through the compass.
const MIN_READINGS: usize = 30;

const MAX_CACHED_BUFFERS: usize = 8;

thread_local! {
    static SAMPLER: RefCell<Option<Sampler>> = const { RefCell::new(None) };
}

fn with_sampler<T>(
    sfz: &Path,
    body: impl FnOnce(&mut Sampler) -> Result<T, piano_tuner::Error>,
) -> Result<T, piano_tuner::Error> {
    SAMPLER.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            *slot = Some(Sampler::new(sfz)?);
        }
        let sampler = slot.as_mut().expect("a player was just built");
        let out = body(sampler);
        if sampler.cached_buffers() > MAX_CACHED_BUFFERS {
            sampler.clear_cache();
        }
        out
    })
}

/// The same preset with `[noise.strike]` silenced outright.
pub fn without_strike(preset: &Preset) -> Preset {
    let mut out = preset.clone();
    out.noise.strike.level_db = vec![NoiseAnchor {
        key: 21,
        db: SILENT_LEVEL_DB,
    }];
    out
}

/// The three signals one render gives: the left channel, the right channel and
/// the mono sum.
///
/// **All three, because the balance is not a mono quantity** (`DECISIONS.md`
/// 392-394). The mechanism's burst is added to the *mid* like every other
/// source and reaches both capsules; the note's own fundamental goes through
/// the mode-controlled lobe and does not. So the ratio of the one to the other
/// — which is the whole of what this tool measures — is a **different number in
/// each loudspeaker**, and the mono sum is the one place it cannot be seen.
/// Measured on the instrument that shipped: over the soprano line's five
/// pitches, engine minus recording read **−4.30 dB on the mono sum, +5.05 dB
/// on the left and −2.60 on the right** — the mono reading asks for a *louder*
/// event where the left channel asks for a much quieter one, and the listener
/// heard the left channel.
struct Take {
    left: Vec<f32>,
    right: Vec<f32>,
    mono: Vec<f32>,
}

impl Take {
    fn channel(&self, c: usize) -> &[f32] {
        match c {
            0 => &self.left,
            1 => &self.right,
            _ => &self.mono,
        }
    }

    /// The sample-wise difference of two takes, channel by channel — which is
    /// the event itself, through the board, the master gain and the microphone
    /// pair. See the module header.
    fn minus(&self, other: &Take) -> Take {
        let sub = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(&x, &y)| x - y).collect();
        Take {
            left: sub(&self.left, &other.left),
            right: sub(&self.right, &other.right),
            mono: sub(&self.mono, &other.mono),
        }
    }
}

/// The three channels the balance is read on, in the order [`Take::channel`]
/// indexes them.
const CHANNELS: [&str; 3] = ["L", "R", "mono"];

fn render_engine(preset: &Preset, key: u8, vel: u8) -> Take {
    let events = [
        RenderEvent::new(PREROLL_S as f32, Event::NoteOn { key, vel: u16::from(vel) }),
        RenderEvent::new(
            (PREROLL_S + HOLD_S) as f32,
            Event::NoteOff { key, vel: 64 },
        ),
    ];
    let (left, right) = render_to_buffer(preset, &events, RENDER_S as f32);
    let mono = left.iter().zip(&right).map(|(&l, &r)| 0.5 * (l + r)).collect();
    Take { left, right, mono }
}

fn render_reference(
    sampler: &mut Sampler,
    key: u8,
    vel: u8,
) -> Result<Audio, piano_tuner::Error> {
    let events = TimedEvent::note(PREROLL_S, key, vel, HOLD_S);
    sampler.render(&events, RENDER_S)
}

/// `--stage mechanism`: the four events the *library* measures, screened.
///
/// No render and no engine — see the module header. The base preset supplies
/// every table the recordings do not earn, and `[noise.strike]` is carried over
/// from the preset being written, because it belongs to the balance stage above
/// and this one has no opinion about it.
fn run_mechanism(
    sfz: &Path,
    preset_path: &Path,
    base_path: &Path,
    out: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    // The tuner's own mirror of the schema, not the engine's: this stage
    // writes a preset file and `estimate::noise` speaks that side of it.
    use piano_tuner::preset::Preset;
    let preset = Preset::load(preset_path)?;
    let config = NoiseConfig::default();
    let library = SampleLibrary::from_sfz(sfz)?;
    let measurements = measure_mechanism(&library, &config);
    println!(
        "the mechanism, from {} against strikes of the same key; base {}, preset {}",
        sfz.display(),
        base_path.display(),
        preset_path.display()
    );
    if measurements.is_empty() {
        println!("\nno mechanism recordings in this library — nothing to write");
        return Ok(());
    }
    // The base's mechanism, but this preset's own hammer noise: `fit_noise`
    // carries `strike` through from the base it is handed, and the base here is
    // `presets/default.toml`, whose strike is silence.
    let mut base = Preset::load(base_path)?.noise;
    base.strike = preset.noise.strike.clone();
    let (fitted, screening) = fit_noise_screened(&measurements, &base, &config);
    crate::print_mechanism(&measurements, &fitted, &screening);

    let mut written = preset.clone();
    written.noise = fitted;
    written.description = screening.describe(&written.description);
    written.validate()?;
    for (name, event) in written.noise.events() {
        println!(
            "\n[noise.{name}] {:.2} Hz, {:.4} s, {:.2} dB of velocity",
            event.centroid_hz, event.decay_s, event.velocity_db
        );
        for anchor in &event.level_db {
            println!("  key {:>3}  {:+.3} dB", anchor.key, anchor.db);
        }
    }
    if written.noise == preset.noise && written.description == preset.description {
        println!("\nnothing moved: this preset already carries what the gate allows");
    }
    match out {
        Some(path) => {
            written.save(path)?;
            println!("\nwrote {}", path.display());
        }
        None => println!("\n(no --out: nothing written)"),
    }
    Ok(())
}

pub fn run(args: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
    let positional: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    let data = PathBuf::from(
        positional
            .first()
            .map(|s| s.as_str())
            .unwrap_or("data/salamander"),
    );
    let preset_path = PathBuf::from(
        positional
            .get(1)
            .map(|s| s.as_str())
            .unwrap_or("presets/salamander-c5.toml"),
    );
    let mut out: Option<PathBuf> = None;
    let mut only: Option<Vec<u8>> = None;
    let mut mechanism = false;
    let mut base = PathBuf::from("presets/default.toml");
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => {
                out = Some(PathBuf::from(&args[i + 1]));
                i += 1;
            }
            "--key" => {
                only = Some(vec![args[i + 1].parse()?]);
                i += 1;
            }
            "--stage" => {
                match args[i + 1].as_str() {
                    "mechanism" => mechanism = true,
                    "balance" => mechanism = false,
                    other => return Err(format!("no such noise stage: {other}").into()),
                }
                i += 1;
            }
            "--base" => {
                base = PathBuf::from(&args[i + 1]);
                i += 1;
            }
            _ => {}
        }
        i += 1;
    }

    // Whichever library this tree is, rather than Salamander's own filename:
    // `adapter::instrument_path` resolves a described library through its
    // LibrarySpec and an undescribed one by its single map (DECISIONS.md 521).
    let sfz = piano_tuner::adapter::instrument_path(&data)?;
    if !sfz.exists() {
        eprintln!(
            "the reference piano is not here: {}\nrun data/fetch_salamander.sh first (707 MiB).",
            sfz.display()
        );
        std::process::exit(2);
    }
    // The balance stage measures the engine's own render of the preset it is
    // given, so writing over that preset mid-run would corrupt the thing being
    // measured. The mechanism stage has no render in it — its input is the
    // library and `--base`, and the preset it is handed contributes only
    // `[noise.strike]`, which it carries through untouched — so writing in
    // place is how a preset gets regenerated through it, and it is idempotent.
    if !mechanism && out.as_deref() == Some(preset_path.as_path()) {
        return Err("--out may not be the preset being measured".into());
    }

    if mechanism {
        return run_mechanism(&sfz, &preset_path, &base, out.as_deref());
    }
    let preset = Preset::load(&preset_path)?;
    let quiet = without_strike(&preset);
    let library = SampleLibrary::from_sfz(&sfz)?;
    let recorded = RecordedKeys::from_library(&library)?;
    let keys: Vec<u8> = match &only {
        Some(k) => k.clone(),
        None => recorded.keys().to_vec(),
    };

    let reference_cache = cache::reference_dir(&data);
    let mut reference_key = cache::Fingerprint::new();
    reference_key
        .str("noise-balance/reference")
        .u64(u64::from(SAMPLER_VERSION))
        .file(&sfz)?
        .u64(u64::from(SAMPLE_RATE))
        .f64(RENDER_S)
        .f64(HOLD_S);

    println!(
        "the mechanism's balance: {} recorded keys x {} velocities, engine on {}",
        keys.len(),
        VELOCITIES.len(),
        preset_path.display()
    );
    println!(
        "  [noise.strike] as it stands: level {:+.2} .. {:+.2} dB over {} anchors, \
         velocity_db {:.2}, centroid {:.0} Hz, band {:.0} Hz, decay {:.3} s",
        preset
            .noise
            .strike
            .level_db
            .iter()
            .map(|a| a.db)
            .fold(f32::INFINITY, f32::min),
        preset
            .noise
            .strike
            .level_db
            .iter()
            .map(|a| a.db)
            .fold(f32::NEG_INFINITY, f32::max),
        preset.noise.strike.level_db.len(),
        preset.noise.strike.velocity_db,
        preset.noise.strike.centroid_hz,
        preset.noise.strike.bandwidth_hz,
        preset.noise.strike.decay_s,
    );

    let cells: Vec<(u8, u8)> = keys
        .iter()
        .flat_map(|&key| VELOCITIES.iter().map(move |&vel| (key, vel)))
        .collect();
    // Three readings per cell — left, right, mono — off **one** pair of
    // renders. The worse of the two channels is the one that is fitted and the
    // one that is printed; `mono` is carried beside it so the change of
    // statistic is visible rather than asserted. See [`Take`].
    let cell_readings: Vec<[BalanceReading; 3]> = cells
        .par_iter()
        .map(|&(key, vel)| -> Result<[BalanceReading; 3], piano_tuner::Error> {
            let engine = render_engine(&preset, key, vel);
            let tone = render_engine(&quiet, key, vel);
            let burst = engine.minus(&tone);
            let mut cell_print = reference_key;
            cell_print.u64(u64::from(key)).u64(u64::from(vel));
            let path = reference_cache.join(format!(
                "balance-key{key:03}-v{vel:03}-{}.wav",
                cell_print.hex()
            ));
            let reference = cache::audio(&path, || {
                with_sampler(&sfz, |s| render_reference(s, key, vel))
            })?;
            let reference_mono = reference.mono();
            // The onsets are read on the mono sums of both sides, once, so
            // that the three channels are compared over **the same window**:
            // a per-channel onset search would move the window as well as the
            // signal and the columns would no longer be a difference.
            let reference_onset = note_onset(&reference_mono, SR, PREROLL_S);
            let engine_onset = note_onset(&engine.mono, SR, PREROLL_S);
            let reference_channel = |c: usize| -> Vec<f32> {
                match (c, reference.channel_count()) {
                    (_, 0..=1) => reference_mono.clone(),
                    (2, _) => reference_mono.clone(),
                    (c, _) => reference.channels[c].clone(),
                }
            };
            Ok(std::array::from_fn(|c| {
                balance_reading(
                    key,
                    vel,
                    &reference_channel(c),
                    reference_onset,
                    tone.channel(c),
                    burst.channel(c),
                    engine_onset,
                    SR,
                )
            }))
        })
        .collect::<Result<Vec<_>, _>>()?;

    /// The reading of the channel the engine is furthest from the recording in.
    ///
    /// **The worse, not the average.** A mechanism that is right in one
    /// loudspeaker and six decibels out in the other is six decibels out; an
    /// average would call it three, and the ear does not.
    fn worse(cell: &[BalanceReading; 3]) -> (usize, BalanceReading) {
        let away = |r: &BalanceReading| (r.engine_db - r.reference_db).abs();
        if away(&cell[0]) >= away(&cell[1]) {
            (0, cell[0])
        } else {
            (1, cell[1])
        }
    }

    // **The correction is fitted on the mono sum, and the channels are
    // reported beside it.** The event is added to the *mid* like every other
    // source, so how loud it is, is a mono quantity and a level fitted to one
    // loudspeaker would put the fold-down wrong by the same amount. What the
    // channels are for is the *spread*: if the two disagree, the number the
    // level is fitted to is not the number a listener hears, and the repair is
    // upstream in `soundboard::MIC_MODAL_DIFFUSION` rather than here
    // (`DECISIONS.md` 393). This tool's job is to make that visible on every
    // run, which is what it could not do while it folded down first.
    let readings: Vec<BalanceReading> = cell_readings.iter().map(|c| c[2]).collect();

    println!(
        "\nattack tonality of the first 30 ms, dB — a line spectrum is large, a continuum is zero\n"
    );
    println!(
        "{:>4} {:>4} {:>4} {:>9} {:>9} {:>9} {:>9} {:>9} {:>10}",
        "key", "vel", "worse", "reference", "engine", "tone only", "eng-ref", "in that ch", "offset"
    );
    for (cell, r) in cell_readings.iter().zip(&readings) {
        let (which, worst) = worse(cell);
        println!(
            "{:>4} {:>4} {:>4} {:>9.2} {:>9.2} {:>9.2} {:>+9.2} {:>+9.2} {:>10}",
            r.key,
            r.midi_velocity,
            CHANNELS[which],
            r.reference_db,
            r.engine_db,
            r.tone_db,
            r.engine_db - r.reference_db,
            worst.engine_db - worst.reference_db,
            match (r.offset_db, r.verdict) {
                (Some(db), _) => format!("{db:+.2}"),
                (None, BalanceVerdict::Floor) => "floor".to_string(),
                (None, _) => "ceiling".to_string(),
            }
        );
    }

    let column = |pick: &dyn Fn(&BalanceReading) -> f64| -> (f64, f64) {
        let mut v: Vec<f64> = readings.iter().map(pick).filter(|x| x.is_finite()).collect();
        if v.is_empty() {
            return (f64::NAN, f64::NAN);
        }
        v.sort_by(f64::total_cmp);
        (
            v[v.len() / 2],
            v.iter().map(|x| x.abs()).sum::<f64>() / v.len() as f64,
        )
    };
    let (imbalance, imbalance_abs) = column(&|r| r.engine_db - r.reference_db);
    let (tone_only, tone_only_abs) = column(&|r| r.tone_db - r.reference_db);
    println!(
        "\nengine minus reference on the mono sum: median {imbalance:+.2} dB, mean |·| \
         {imbalance_abs:.2} dB over {} notes",
        readings.len()
    );
    {
        // The spread the fold-down hides: per cell, how far the worse
        // loudspeaker is from the recording, and how far the two are from each
        // other. On the instrument `DECISIONS.md` 392 opened on, the mono sum
        // read -4.30 dB over the soprano line's pitches while the left channel
        // read +5.05 and the right -2.60 — the fold-down asked for a *louder*
        // event where the left channel asked for a much quieter one.
        let column = |pick: &dyn Fn(&[BalanceReading; 3]) -> f64| -> (f64, f64) {
            let mut v: Vec<f64> = cell_readings.iter().map(pick).filter(|x| x.is_finite()).collect();
            v.sort_by(f64::total_cmp);
            if v.is_empty() {
                return (f64::NAN, f64::NAN);
            }
            (v[v.len() / 2], *v.last().expect("non-empty"))
        };
        let (worse_median, worse_max) =
            column(&|c| (worse(c).1.engine_db - worse(c).1.reference_db).abs());
        let (split_median, split_max) = column(&|c| (c[0].engine_db - c[1].engine_db).abs());
        let (ref_split_median, ref_split_max) =
            column(&|c| (c[0].reference_db - c[1].reference_db).abs());
        let (l, r) = (
            cell_readings.iter().filter(|c| worse(c).0 == 0).count(),
            cell_readings.iter().filter(|c| worse(c).0 == 1).count(),
        );
        println!(
            "  the worse loudspeaker of each note: |engine-ref| median {worse_median:.2} dB, \
             worst {worse_max:.2} ({l} notes worse in L, {r} in R)"
        );
        println!(
            "  and how far the two loudspeakers are from *each other*: engine median \
             {split_median:.2} dB, worst {split_max:.2}; the recording's own {ref_split_median:.2} \
             and {ref_split_max:.2} — this is the number the mono sum cannot carry, and the \
             engine's must sit inside the recording's"
        );
    }
    println!(
        "the same with the event silenced: median {tone_only:+.2} dB, mean |·| {tone_only_abs:.2} dB \
         — the sign says which side of the piano the tonal attack is on"
    );
    for &vel in &VELOCITIES {
        let mut a: Vec<f64> = readings
            .iter()
            .filter(|r| r.midi_velocity == vel)
            .map(|r| r.engine_db - r.reference_db)
            .filter(|x| x.is_finite())
            .collect();
        let mut b: Vec<f64> = readings
            .iter()
            .filter(|r| r.midi_velocity == vel)
            .filter_map(|r| r.offset_db)
            .collect();
        a.sort_by(f64::total_cmp);
        b.sort_by(f64::total_cmp);
        if a.is_empty() {
            continue;
        }
        println!(
            "  vel {vel:>3}: engine-ref {:+6.2} dB   offset {} (n {})",
            a[a.len() / 2],
            if b.is_empty() {
                "     —".to_string()
            } else {
                format!("{:+6.2}", b[b.len() / 2])
            },
            b.len()
        );
    }

    let Some(fit) = fit_balance(&readings, MIN_READINGS) else {
        println!(
            "\nnot enough readings inverted to fit a correction ({} of {})",
            readings.iter().filter(|r| r.offset_db.is_some()).count(),
            readings.len()
        );
        return Ok(());
    };
    println!(
        "\nthe correction, Theil-Sen through {} inverted readings ({} floor, {} ceiling):",
        fit.closed, fit.floor, fit.ceiling
    );
    println!("  level at the nominal drive  {:+.2} dB", fit.level_db);
    println!(
        "  velocity_db                 {:.2} -> {:.2}  ({:+.2})",
        preset.noise.strike.velocity_db,
        f64::from(preset.noise.strike.velocity_db) + fit.velocity_db,
        fit.velocity_db
    );
    println!("  scatter about the line      {:.2} dB", fit.scatter_db);

    if let Some(path) = out {
        let mut written = preset.clone();
        for anchor in written.noise.strike.level_db.iter_mut() {
            anchor.db += fit.level_db as f32;
        }
        written.noise.strike.velocity_db += fit.velocity_db as f32;
        written.validate()?;
        written.save(&path)?;
        println!("\n{}", path.display());
        for anchor in &written.noise.strike.level_db {
            println!("  key {:>3}  {:+.3} dB", anchor.key, anchor.db);
        }
        println!("  velocity_db {:.3}", written.noise.strike.velocity_db);
    }
    Ok(())
}
