//! Stage 2, the render-and-measure half: the duplex segments, the sympathetic
//! halo and the per-key stereo spread, fitted against a real instrument.
//!
//! `survey` (stage 1) fits everything an *isolated note* can identify. Three
//! things it cannot, `docs/history/TUNING_REPORT.md` says so explicitly, and all three are
//! here:
//!
//! 1. **`notes.duplex`** — from Salamander's `harmL*`/`harmS*`/`harmV3*`
//!    release-resonance recordings, free-tracked with the inharmonic seed
//!    removed (`estimate::duplex`). Measured frequencies, never ratios.
//! 2. **`voicing.resonance_coupling` and `[voicing.bridge]`** — §4's
//!    between-partial census and §5's `harm*` levels are targets, and the fit
//!    is a loop: render the engine, measure it with the very code the
//!    recordings were measured with, step, render again (`estimate::halo`).
//! 3. **`notes.pan_spread`** — the drift of each register's stereo image,
//!    inverted on a line measured per key on the engine rather than on the
//!    compass median that overshot C5 and C6 (`estimate::directivity`).
//!
//! ```sh
//! cargo run --release -p piano-tuner -- sympathetic \
//!     data/salamander/SalamanderGrandPiano-V3+20200602.sfz \
//!     --preset presets/salamander-c5.toml --out presets/salamander-c5.toml
//! ```
//!
//! Without `--out` it measures and prints and writes nothing, which is how the
//! before/after tables in `DECISIONS.md` were taken.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_emulator::types::Event;
use piano_tuner::estimate::directivity::{
    balance_drift, DirectivityConfig, KeyDriftLine, MAX_PAN_SPREAD,
};
use piano_tuner::estimate::duplex::{
    assert_not_harmonic, duplex_row, partial_frequencies, DuplexConfig,
};
use piano_tuner::estimate::halo::{
    between_partials, peaks_from_body_modes, refine, salamander_targets,
    HaloConfig, HaloError, HaloVoicing,
};
use piano_tuner::library::MechanismKind;
use piano_tuner::preset::{key_index, DuplexMode, Preset, NUM_KEYS};
use piano_tuner::survey::SurveyConfig;
use piano_tuner::{audio, Error, Result, SampleLibrary, SAMPLE_RATE};

/// Velocity every engine reference strike is taken at — `docs/history/TUNING_REPORT.md`
/// §5's own convention.
const REFERENCE_VELOCITY: u8 = 90;
/// How long a note is held before the key comes up, and how long the render
/// runs. The halo has to be given time to be heard on its own.
const HOLD_S: f32 = 1.0;
const RENDER_S: f32 = 5.0;
/// Passes of the halo fit. It converges in five or six; the cap is there
/// because each pass is a dozen renders.
const HALO_PASSES: usize = 8;
/// Damping on the halo step (`estimate::halo::refine`).
const HALO_RATE: f64 = 0.6;
/// How much room the coupling must still have above where it stands, once the
/// segments have taken their share of the undamped loop bound.
///
/// They are never damped, so what they take they keep, and what they take comes
/// out of the coupling that moves the halo — but "what they take" is a *loop
/// gain*, not a bare response factor. Until `DECISIONS.md` 481 this was written
/// as a ceiling of 0.05 on `duplex_response_factor`, which corresponded to a
/// coupling ceiling of about 3.3 against a fitted coupling of 0.0104: three
/// hundred times more headroom than the halo fit could ever use, and it cost
/// nothing to reserve because a segment was inaudible anyway. It is not free
/// now — under 481's normalisation the measured instrument lands 20.7 dB under
/// that ceiling — so the reserve is stated as what it is for: the coupling may
/// still move by this factor before the segments become the binding
/// constraint, which is more room than any halo fit has ever asked for.
const DUPLEX_COUPLING_HEADROOM: f32 = 4.0;

struct Options {
    sfz: PathBuf,
    base: PathBuf,
    out: Option<PathBuf>,
    /// `--only duplex`: run stage 1 and leave the halo and the pan spread
    /// exactly as the file has them.
    ///
    /// It exists because `DECISIONS.md` 481 re-decided what `notes.duplex`
    /// means and nothing else, so the rows have to be re-estimated while the
    /// coupling, the bridge and the spread stay where the fits that own them
    /// put them. Re-running the whole stage to move one table would put three
    /// closed-on-the-render fits back in play for no reason.
    only_duplex: bool,
    /// `--only pan-spread`: run stage 3 alone and leave the duplex rows and the
    /// halo exactly as the file has them.
    ///
    /// It exists for the same reason `--only duplex` does, one item later:
    /// `DECISIONS.md` 485 re-decides what stage 3 *writes* and nothing else, so
    /// the rows it retires have to be retired without putting the duplex
    /// estimate of items 481-484 and the halo's closed-on-the-render loop back
    /// in play. Under item 485 the stage renders, measures and prints exactly
    /// what it always did and then writes the **null**: see
    /// [`retire_pan_spread`].
    only_pan_spread: bool,
}

fn parse(args: Vec<String>) -> Result<Options> {
    let mut sfz = None;
    let (mut base, mut out) = (None, None);
    let mut only_duplex = false;
    let mut only_pan_spread = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--only" => {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| Error::Config("--only needs a stage".into()))?;
                match value.as_str() {
                    "duplex" => only_duplex = true,
                    "pan-spread" => only_pan_spread = true,
                    other => {
                        return Err(Error::Config(format!(
                            "--only {other}: the stages that can be run alone are `duplex` \
                             and `pan-spread`"
                        )))
                    }
                }
                i += 1;
            }
            "--preset" | "--out" => {
                let value = args
                    .get(i + 1)
                    .ok_or_else(|| Error::Config(format!("{} needs a path", args[i])))?;
                if args[i] == "--preset" {
                    base = Some(PathBuf::from(value));
                } else {
                    out = Some(PathBuf::from(value));
                }
                i += 1;
            }
            other => sfz = Some(PathBuf::from(other)),
        }
        i += 1;
    }
    Ok(Options {
        sfz: sfz.ok_or_else(|| Error::Config("no instrument file".into()))?,
        base: base.ok_or_else(|| Error::Config("--preset is required".into()))?,
        out,
        only_duplex,
        only_pan_spread,
    })
}

pub fn run(args: Vec<String>) -> Result<()> {
    let options = parse(args)?;
    let library = SampleLibrary::from_sfz(&options.sfz)?;
    let text = std::fs::read_to_string(&options.base)?;
    let mut preset = Preset::from_toml(&text)?;
    let survey = SurveyConfig::default();
    println!(
        "{}: {} keys, {} recordings, {} release resonances",
        options.sfz.display(),
        library.key_count(),
        library.sample_count(),
        library.mechanism_of(MechanismKind::StringResonance).len()
    );

    // **Stage 3 alone, before stage 1 has run** (`DECISIONS.md` 485): the
    // retirement writes a null and needs neither the duplex rows nor the halo,
    // and re-estimating them to reach it would put two closed fits back in play.
    if options.only_pan_spread {
        retire_pan_spread(&library, &mut preset, &survey)?;
        preset.validate()?;
        println!(
            "\nfinal (--only pan-spread): voicing.polarization_pan_spread {:.4} and \
             notes.pan_spread {} rows; the duplex, the coupling and the bridge untouched",
            preset.voicing.polarization_pan_spread,
            preset.notes.pan_spread.len(),
        );
        return write_out(&options, &preset, &text);
    }

    // ---------------------------------------------------------- 1. duplex
    let duplex = fit_duplex(&library, &preset)?;
    preset.notes.duplex = duplex;

    if options.only_duplex {
        preset.validate()?;
        println!(
            "\nfinal (--only duplex): resonance_coupling {:.5} and the bridge, the spread and \
             everything else untouched; duplex loop gain {:.5}",
            preset.voicing.resonance_coupling,
            preset.duplex_loop_gain(),
        );
    } else {
        // ------------------------------------------------------------ 2. halo
        let peaks = peaks_from_body_modes(&preset);
        let voicing = fit_halo(&preset, &peaks, &survey)?;
        voicing.apply(&mut preset, peaks.clone())?;

        // ------------------------------------------------------ 3. pan spread
        // **A retirement since `DECISIONS.md` 485, not a fit**: the stage still
        // renders and still prints the drift, and what it writes is the null.
        // [`fit_pan_spread`] is kept beside it because it is the measurement a
        // successor building item 467's gain trim inverts, and because deleting
        // the instrument that made the diagnosis possible is how a repository
        // forgets why a field is zero.
        retire_pan_spread(&library, &mut preset, &survey)?;

        preset.validate()?;
        println!(
            "\nfinal: resonance_coupling {:.5}, backbone {:+.2} dB, tilt {:+.2} dB, \
             bridge loop gain {:.4}, duplex loop gain {:.5}",
            preset.voicing.resonance_coupling,
            voicing.backbone_gain_db,
            voicing.treble_tilt_db,
            voicing.loop_gain(&peaks),
            preset.duplex_loop_gain(),
        );
    }

    write_out(&options, &preset, &text)
}

/// Write the fitted preset, carrying the credit comment at the head of the file
/// over by hand: it is not part of the schema and a round trip would lose it.
fn write_out(options: &Options, preset: &Preset, text: &str) -> Result<()> {
    let Some(out) = &options.out else {
        println!("\n(no --out: nothing written)");
        return Ok(());
    };
    let mut header = String::new();
    for line in text.lines().take_while(|line| line.starts_with('#')) {
        header.push_str(line);
        header.push('\n');
    }
    std::fs::write(out, format!("{header}{}", preset.to_toml()))?;
    println!("\nwrote {}", out.display());
    Ok(())
}

// ------------------------------------------------------------------ duplex

/// One duplex row per key, from the release-resonance recordings.
fn fit_duplex(library: &SampleLibrary, preset: &Preset) -> Result<Vec<Vec<DuplexMode>>> {
    let config = DuplexConfig::default();
    let mut table = vec![Vec::new(); NUM_KEYS];
    // The loudest tier of each key: Salamander ships three (`harmL*` from
    // velocity 45 up, `harmS*` below it, `harmV3*` untracked), and the loud one
    // is what §5 measured and what has the most above the recording's floor.
    let mut best: BTreeMap<u8, (&Path, f64, u8)> = BTreeMap::new();
    for sample in library.mechanism_of(MechanismKind::StringResonance) {
        let Some(key) = sample.key else { continue };
        let entry = (sample.path.as_path(), sample.volume_db, sample.lovel);
        match best.get(&key) {
            Some(&(_, _, lovel)) if lovel >= sample.lovel => {}
            _ => {
                best.insert(key, entry);
            }
        }
    }

    // Two passes, because the level convention needs the whole instrument
    // before it can write any of it: a shift that keeps the loudest segment
    // under the schema's ceiling, and one that keeps the whole undamped bank
    // out of the coupling's way, and neither can be known from one key.
    // Clamping each row on its own would flatten the compass to one level; one
    // shift for all of them keeps the relative structure, which is the part the
    // gate proves is recoverable.
    struct Measured<'a> {
        key: u8,
        index: usize,
        modes: Vec<piano_tuner::estimate::duplex::ResidualMode>,
        reference: f64,
        partials: Vec<f64>,
        path: &'a Path,
    }
    let mut measured: Vec<Measured> = Vec::new();
    for (&key, &(path, volume_db, _)) in &best {
        let Some(index) = key_index(key) else { continue };
        // The strike this key's segments are quoted against, at the level the
        // instrument plays both — the same convention as every other level in
        // `docs/history/TUNING_REPORT.md` §5.
        let Some(strike) = library.nearest_layer(key, REFERENCE_VELOCITY) else {
            continue;
        };
        let strike_audio = audio::load_at(&strike.path, SAMPLE_RATE)?;
        let Some(loudest) = piano_tuner::estimate::duplex::strongest_peak(
            &strike_audio.mono(),
            f64::from(SAMPLE_RATE),
            &config,
        ) else {
            continue;
        };
        let reference = loudest * 10f64.powf(strike.volume_db / 20.0);

        let recording = audio::load_at(path, SAMPLE_RATE)?;
        let gain = 10f32.powf((volume_db / 20.0) as f32);
        let signal: Vec<f32> = recording.mono().iter().map(|&x| x * gain).collect();
        let partials = partial_frequencies(
            f64::from(preset.notes.f0_hz[index]),
            f64::from(preset.notes.inharmonicity_b[index]),
            f64::from(preset.notes.inharmonicity_b4[index]),
            80,
        );
        let modes = piano_tuner::estimate::duplex::residual_modes_above(
            &signal,
            f64::from(SAMPLE_RATE),
            &partials,
            f64::from(preset.notes.f0_hz[index]),
            &config,
        )?;
        measured.push(Measured { key, index, modes, reference, partials, path });
    }

    let loudest_asked = measured
        .iter()
        .flat_map(|m| m.modes.iter().map(|mode| mode.amplitude / m.reference))
        .map(|ratio| piano_tuner::estimate::duplex::gain_for_level(20.0 * ratio.log10()))
        .fold(f64::NEG_INFINITY, f64::max);
    // Two things bound the shift, and the second is the binding one.
    //
    // The schema's own +6 dB ceiling is the obvious one, and since
    // `DECISIONS.md` 481 re-normalised the field it is no longer the binding
    // one: the loudest segment this library carries asks for +4.0 dB, so the
    // measurement fits inside the schema with no shift at all. The other is the
    // *loop*: 88 permanently undamped banks stand in the sympathetic path, and
    // whatever share of the bound they take is taken away from the coupling —
    // which is the parameter that actually moves the halo. That one is stated
    // as `DUPLEX_COUPLING_HEADROOM`, and the measured instrument is under it,
    // so what ships is the measurement and not a policy.
    let ceiling_shift =
        (f64::from(piano_tuner::preset::MAX_DUPLEX_GAIN_DB) - loudest_asked).min(0.0);
    let mut probe = preset.clone();
    probe.notes.duplex = {
        let config = DuplexConfig { shift_db: ceiling_shift, ..config };
        let mut rows = vec![Vec::new(); NUM_KEYS];
        for m in &measured {
            rows[m.index] = duplex_row(&m.modes, m.reference, &config);
        }
        rows
    };
    let factor = probe.duplex_response_factor();
    let loop_gain = probe.duplex_loop_gain();
    let budget = f64::from(piano_tuner::preset::MAX_DUPLEX_LOOP_GAIN / DUPLEX_COUPLING_HEADROOM);
    let shift_db = if f64::from(loop_gain) > budget {
        ceiling_shift + 20.0 * (budget / f64::from(loop_gain)).log10()
    } else {
        ceiling_shift
    };
    let config = DuplexConfig { shift_db, ..config };
    println!(
        "\nduplex segments, from the release resonances\n  the loudest segment on the instrument \
         asks for {loudest_asked:+.1} dB against the schema's {:+.1} ceiling ({ceiling_shift:+.1} \
         dB of shift); at that level the segments realise {factor:.3} per unit of drive and close \
         an undamped loop of {loop_gain:.5} against the {budget:.5} that leaves the coupling \
         {DUPLEX_COUPLING_HEADROOM}x of room, so the shift is {shift_db:+.1} dB and the levels \
         are relative",
        piano_tuner::preset::MAX_DUPLEX_GAIN_DB
    );
    println!("  key   n   strongest Hz   re strike   T60 s   cents off the nearest partial");
    for m in &measured {
        let row = duplex_row(&m.modes, m.reference, &config);
        if row.is_empty() {
            println!(
                "  {:>3}   0   (nothing over the floor that outlives {} s)",
                m.key, config.min_t60_s
            );
            continue;
        }
        // A row that is only the note again would be a measurement of the
        // tracker, not of the piano.
        if let Err(error) = assert_not_harmonic(&row, &m.partials, 10.0) {
            println!("  {:>3}   -   refused ({}): {error}", m.key, m.path.display());
            continue;
        }
        let detunings = piano_tuner::estimate::duplex::detuning_cents(&row, &m.partials);
        println!(
            "  {:>3} {:>3}   {:>12.1}   {:>+9.1}   {:>5.2}   {}",
            m.key,
            row.len(),
            row[0].hz,
            20.0 * (m.modes[0].amplitude / m.reference).log10(),
            row[0].t60_s,
            detunings
                .iter()
                .map(|d| format!("{d:+.0}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
        table[m.index] = row;
    }
    // The finding `PHYSICS.md` §3 asks a measurement to make or refute, as one
    // number: how far a segment sits from the nearest partial of its own note.
    let mut offsets: Vec<f64> = table
        .iter()
        .enumerate()
        .flat_map(|(index, row)| {
            let partials = partial_frequencies(
                f64::from(preset.notes.f0_hz[index]),
                f64::from(preset.notes.inharmonicity_b[index]),
                f64::from(preset.notes.inharmonicity_b4[index]),
                80,
            );
            piano_tuner::estimate::duplex::detuning_cents(row, &partials)
        })
        .collect();
    offsets.sort_by(f64::total_cmp);
    if !offsets.is_empty() {
        let median = offsets[offsets.len() / 2];
        let sharp = offsets.iter().filter(|&&d| d > 0.0).count();
        println!(
            "  {} segments, median {median:+.0} cents off the nearest partial, {sharp} of them \
             sharp of it (the guard excludes +-{:.0} cents, so the distribution is what \
             survives that)",
            offsets.len(),
            config.guard_cents
        );
    }
    let keyed = table.iter().filter(|row| !row.is_empty()).count();
    println!(
        "  {keyed} of {NUM_KEYS} keys got a measured table; the rest get an empty one, \
         not an invented one"
    );
    if keyed == 0 {
        return Ok(Vec::new());
    }
    Ok(table)
}

// -------------------------------------------------------------------- halo

/// The engine, rendered and measured, until §4's and §5's numbers come back.
fn fit_halo(
    preset: &Preset,
    peaks: &[piano_tuner::preset::BridgePeak],
    survey: &SurveyConfig,
) -> Result<HaloVoicing> {
    let targets = salamander_targets();
    let duplex_factor = preset.duplex_response_factor();
    let mut voicing = HaloVoicing {
        coupling: preset.voicing.resonance_coupling,
        ..HaloVoicing::default()
    };
    println!(
        "\nhalo fit: {} targets, {HALO_PASSES} passes at rate {HALO_RATE}; the segments \
         already occupy {duplex_factor:.3} of the loop, leaving a coupling ceiling of {:.5}",
        targets.len(),
        voicing.coupling_ceiling(peaks, duplex_factor)
    );
    println!("  pass  coupling  backbone  tilt |  {}",
        targets.iter().map(|t| format!("{:>12}", t.name)).collect::<Vec<_>>().join(""));

    let mut best: Option<(f64, HaloVoicing, Vec<HaloError>)> = None;
    for pass in 0..HALO_PASSES {
        let mut candidate = preset.clone();
        voicing.apply(&mut candidate, peaks.to_vec())?;
        let errors = measure_targets(&candidate, &targets, survey)?;
        let cost: f64 = errors
            .iter()
            .map(|e| (e.error_db().abs() / e.target.tolerance_db.max(0.25)).powi(2))
            .sum();
        println!(
            "  {pass:>4}  {:>8.5}  {:>+8.2}  {:>+4.1} |  {}",
            voicing.coupling,
            voicing.backbone_gain_db,
            voicing.treble_tilt_db,
            errors
                .iter()
                .map(|e| format!("{:>12.1}", e.measured_db))
                .collect::<Vec<_>>()
                .join("")
        );
        if best.as_ref().map_or(true, |&(c, _, _)| cost < c) {
            best = Some((cost, voicing, errors.clone()));
        }
        if errors.iter().all(HaloError::inside_tolerance) {
            println!("  every target inside its band after {} passes", pass + 1);
            break;
        }
        voicing = refine(voicing, &errors, peaks, duplex_factor, HALO_RATE);
    }
    let (_, voicing, errors) = best.expect("at least one pass ran");
    println!("\n  target        want      got     error   inside?");
    for error in &errors {
        println!(
            "  {:<12} {:>+7.1}  {:>+7.1}  {:>+7.1}   {}",
            error.target.name,
            error.target.target_db,
            error.measured_db,
            error.error_db(),
            if error.inside_tolerance() { "yes" } else { "NO" }
        );
    }
    Ok(voicing)
}

/// Every target, measured on renders of one candidate preset.
fn measure_targets(
    candidate: &Preset,
    targets: &[piano_tuner::estimate::halo::HaloTarget],
    survey: &SurveyConfig,
) -> Result<Vec<HaloError>> {
    let engine = piano_emulator::preset::Preset::from_toml(&candidate.to_toml())
        .map_err(|e| Error::Preset(e.to_string()))?;
    let mut errors = Vec::new();
    for &target in targets {
        let measured_db = if target.name.starts_with("harm") {
            halo_level(&engine, target.key)
        } else {
            let f0 = f64::from(candidate.notes.f0_hz[key_index(target.key).unwrap()]);
            let note_config = survey.note_config(f0)?;
            let (left, right) = render_to_buffer(
                &engine,
                &[RenderEvent::new(
                    0.0,
                    Event::NoteOn {
                        key: target.key,
                        vel: u16::from(REFERENCE_VELOCITY),
                    },
                )],
                RENDER_S,
            );
            let mono = mono(&left, &right);
            between_partials(
                &mono,
                f64::from(SAMPLE_RATE),
                f0,
                &note_config,
                &HaloConfig::default(),
            )
            .map(|b| b.at_late_db)
            .unwrap_or(f64::NAN)
        };
        errors.push(HaloError { target, measured_db });
    }
    Ok(errors)
}

/// `docs/history/TUNING_REPORT.md` §5's `harm*` measurement, on the engine —
/// and it is [`piano_tuner::estimate::halo::engine_halo_level`] and not a
/// second copy of it (`DECISIONS.md` 501).
///
/// The recording it is held against is a sample of the halo *alone* —
/// Salamander records the string resonance separately from the key-off thump —
/// so the engine's halo has to be isolated the same way, and it is, by
/// subtraction: the same note, struck and released, rendered once as the
/// instrument is and once with nothing to couple through (no bus, no
/// segments). The engine is deterministic, so the difference is exactly the
/// sympathetic contribution, and no arbitrary "wait for the damper" window has
/// to be chosen on one side and not the other.
///
/// It lives in `estimate::halo` because the `halo` column
/// (`tuner/tests/halo.rs`) reads the same quantity, and a fit whose objective
/// and whose gate are two copies of one measurement is a fit that can converge
/// on a number the gate does not score.
fn halo_level(engine: &piano_emulator::preset::Preset, key: u8) -> f64 {
    piano_tuner::estimate::halo::engine_halo_level(
        engine,
        key,
        REFERENCE_VELOCITY,
        f64::from(HOLD_S),
    )
    .map_or(f64::NAN, |level| level.peak_db)
}

fn mono(left: &[f32], right: &[f32]) -> Vec<f32> {
    left.iter().zip(right).map(|(&l, &r)| 0.5 * (l + r)).collect()
}

// -------------------------------------------------------------- pan spread

/// **Stage 3 under `DECISIONS.md` 485: it measures exactly what it always
/// measured and writes the null.**
///
/// [`fit_pan_spread`] inverts a *drift* — how far a note's interchannel balance
/// moves between 0.3 s and 2 s, which is a real measured property of the
/// recordings and 1.2-6.2 dB of it — into `voicing.polarization_pan_spread`,
/// which is a **position**: `voice.rs` renders the horizontal polarization at
/// `pan + spread·sign` and the vertical at `pan − spread·sign`, with `sign`
/// flipping on `key % 2`. Item 467 measured what that costs through a spaced
/// pair rather than through a pan-pot — the two planes of C4 at pan −0.42 and
/// +0.30, over a metre of string band apart, and the sign alternating note by
/// note — and priced the exchange at **4:1 against**: about 2.9 dB of the drift
/// bought with about ±11 dB of static image. It named the replacement (a
/// per-polarization interchannel *gain* trim at the mic stage: same drift, one
/// position, about 3 dB of image) and did not build it, because the image
/// budget of item 470 had no owner.
///
/// Item 485 is that owner, and its verdict is about this mechanism by name: the
/// per-key lean is not to dominate. A term that alternates its sign with the
/// key parity is the largest per-key lean in the tree, so the stage stops
/// writing one. What it still does is **render, measure and print** — the
/// recordings' own drift at each ladder key, the engine's drift as the file
/// stands, and the engine's drift with the spread out — so that the size of
/// what is given up is on the page and a successor building item 467's trim
/// knows what it has to buy back.
fn retire_pan_spread(
    library: &SampleLibrary,
    preset: &mut Preset,
    survey: &SurveyConfig,
) -> Result<()> {
    let config = DirectivityConfig::default();
    let before = preset.clone();
    let engine_zero = engine_with_spread(&before, 0.0)?;
    let engine_as_is = piano_emulator::preset::Preset::from_toml(&before.to_toml())
        .map_err(|e| Error::Preset(e.to_string()))?;

    println!(
        "\nstage 3 is a RETIREMENT (DECISIONS.md 485, on item 467's diagnosis): the drift is \
         real and the position it was bought with is the dominant per-key lean in the tree, \
         so the table is written as the null and the drift is printed rather than fitted."
    );
    println!("  key   recorded   engine now   engine with the spread out   spread now");
    let keys: Vec<u8> = library.keys().collect();
    let (mut kept, mut lost, mut n) = (0.0f64, 0.0f64, 0usize);
    for key in keys {
        let Some(sample) = library.nearest_layer(key, REFERENCE_VELOCITY) else {
            continue;
        };
        let Some(index) = key_index(key) else { continue };
        let f0 = f64::from(before.notes.f0_hz[index]);
        let note_config = survey.note_config(f0)?;
        let recording = audio::load_at(&sample.path, SAMPLE_RATE)?;
        if recording.channel_count() < 2 {
            continue;
        }
        let Ok(recorded) = balance_drift(
            &recording.channels[0],
            &recording.channels[1],
            f0,
            f64::from(SAMPLE_RATE),
            &note_config,
            &config,
        ) else {
            continue;
        };
        let drift_of = |engine: &piano_emulator::preset::Preset| -> Option<f64> {
            let (left, right) = render_to_buffer(
                engine,
                &[RenderEvent::new(
                    0.0,
                    Event::NoteOn {
                        key,
                        vel: u16::from(REFERENCE_VELOCITY),
                    },
                )],
                8.0,
            );
            balance_drift(&left, &right, f0, f64::from(SAMPLE_RATE), &note_config, &config)
                .ok()
                .map(|d| d.drift_db)
        };
        let (Some(now), Some(without)) = (drift_of(&engine_as_is), drift_of(&engine_zero)) else {
            continue;
        };
        println!(
            "  {key:>3}   {:>8.2}   {:>10.2}   {:>27.2}   {:>10.3}",
            recorded.drift_db,
            now,
            without,
            before
                .notes
                .pan_spread
                .get(index)
                .copied()
                .unwrap_or(before.voicing.polarization_pan_spread),
        );
        kept += without.abs();
        lost += (now - without).abs();
        n += 1;
    }
    if n > 0 {
        let band = piano_tuner::estimate::directivity::MEASURED_DRIFT_BAND;
        println!(
            "  over {n} keys: {:.2} dB of drift survives the retirement and {:.2} dB is given \
             up, against the {:.1}-{:.1} dB the recordings carry — item 467's replacement \
             (a per-polarization interchannel gain trim at the mic stage) is what buys it back",
            kept / n as f64,
            lost / n as f64,
            band.0,
            band.1,
        );
    }
    preset.voicing.polarization_pan_spread = 0.0;
    preset.notes.pan_spread = Vec::new();
    Ok(())
}

/// `notes.pan_spread`, from the recordings' drift and two engine renders per
/// key — **the fit `DECISIONS.md` 485 retired**, kept because it is the
/// instrument item 467's replacement mechanism has to be fitted with and
/// because a field that is zero without the measurement beside it is a field
/// nobody can ever put back.
#[allow(dead_code)]
fn fit_pan_spread(
    library: &SampleLibrary,
    preset: &Preset,
    survey: &SurveyConfig,
) -> Result<Vec<f32>> {
    let config = DirectivityConfig::default();
    let engine_zero = engine_with_spread(preset, 0.0)?;
    let engine_top = engine_with_spread(preset, MAX_PAN_SPREAD as f32)?;

    let mut measured: Vec<(u8, f64)> = Vec::new();
    let mut lines: Vec<KeyDriftLine> = Vec::new();
    println!("\nstereo drift, recording against the engine's own line");
    println!("  key   recorded   engine@0   engine@0.4   spread");
    for key in library.keys().collect::<Vec<u8>>() {
        let Some(sample) = library.nearest_layer(key, REFERENCE_VELOCITY) else {
            continue;
        };
        let Some(index) = key_index(key) else { continue };
        let f0 = f64::from(preset.notes.f0_hz[index]);
        let note_config = survey.note_config(f0)?;
        let recording = audio::load_at(&sample.path, SAMPLE_RATE)?;
        if recording.channel_count() < 2 {
            continue;
        }
        let Ok(recorded) = balance_drift(
            &recording.channels[0],
            &recording.channels[1],
            f0,
            f64::from(SAMPLE_RATE),
            &note_config,
            &config,
        ) else {
            continue;
        };
        let line_at = |engine: &piano_emulator::preset::Preset| -> Option<f64> {
            let (left, right) = render_to_buffer(
                engine,
                &[RenderEvent::new(0.0, Event::NoteOn { key, vel: u16::from(REFERENCE_VELOCITY) })],
                8.0,
            );
            balance_drift(&left, &right, f0, f64::from(SAMPLE_RATE), &note_config, &config)
                .ok()
                .map(|d| d.drift_db)
        };
        let (Some(at_zero_db), Some(at_ceiling_db)) = (line_at(&engine_zero), line_at(&engine_top))
        else {
            continue;
        };
        let line = KeyDriftLine {
            key,
            at_zero_db,
            at_ceiling_db,
        };
        println!(
            "  {key:>3}   {:>8.2}   {:>8.2}   {:>10.2}   {:>6.3}",
            recorded.drift_db,
            at_zero_db,
            at_ceiling_db,
            line.spread_for(recorded.drift_db)
        );
        measured.push((key, recorded.drift_db));
        lines.push(line);
    }
    let table = piano_tuner::estimate::directivity::pan_spread_table(&measured, &lines)?;
    println!(
        "  table spans {:.3}..{:.3} against the one global {:.3} it replaces",
        table.iter().copied().fold(f32::INFINITY, f32::min),
        table.iter().copied().fold(0.0f32, f32::max),
        preset.voicing.polarization_pan_spread
    );

    // What it actually does, which is the only claim worth making: the drift
    // the engine renders with the fitted table, against the band
    // `docs/history/TUNING_REPORT.md` §5 measured (1.2-6.2 dB) and against the one global
    // spread the file used to carry.
    let mut fitted = preset.clone();
    fitted.notes.pan_spread = table.clone();
    let fitted = piano_emulator::preset::Preset::from_toml(&fitted.to_toml())
        .map_err(|e| Error::Preset(e.to_string()))?;
    let band = piano_tuner::estimate::directivity::MEASURED_DRIFT_BAND;
    let (mut inside, mut total) = (0usize, 0usize);
    println!("  key   global 0.4   fitted   in the 1.2-6.2 dB band?");
    for line in &lines {
        let Some(index) = key_index(line.key) else { continue };
        let f0 = f64::from(preset.notes.f0_hz[index]);
        let note_config = survey.note_config(f0)?;
        let (left, right) = render_to_buffer(
            &fitted,
            &[RenderEvent::new(0.0, Event::NoteOn { key: line.key, vel: u16::from(REFERENCE_VELOCITY) })],
            8.0,
        );
        let Ok(drift) =
            balance_drift(&left, &right, f0, f64::from(SAMPLE_RATE), &note_config, &config)
        else {
            continue;
        };
        total += 1;
        let ok = (band.0..=band.1).contains(&drift.drift_db);
        inside += usize::from(ok);
        println!(
            "  {:>3}   {:>10.2}   {:>6.2}   {}",
            line.key,
            line.at_ceiling_db,
            drift.drift_db,
            if ok { "yes" } else { "no" }
        );
    }
    let was = lines
        .iter()
        .filter(|l| (band.0..=band.1).contains(&l.at_ceiling_db))
        .count();
    println!("  {inside} of {total} keys inside the band, against {was} at the global 0.4");
    Ok(table)
}

fn engine_with_spread(preset: &Preset, spread: f32) -> Result<piano_emulator::preset::Preset> {
    let mut candidate = preset.clone();
    candidate.voicing.polarization_pan_spread = spread;
    candidate.notes.pan_spread = Vec::new();
    piano_emulator::preset::Preset::from_toml(&candidate.to_toml())
        .map_err(|e| Error::Preset(e.to_string()))
}
