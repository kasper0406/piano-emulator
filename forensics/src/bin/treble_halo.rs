//! **The treble sympathetic halo, measured before anything is built.**
//!
//! `docs/history/TUNING_REPORT.md` §4 says the recordings of the top two
//! octaves are mostly *not* the struck string one second on, and that the
//! engine's render of the same note is tens of decibels under them. §4's own
//! update then says the number it printed cannot be used as a fitting target,
//! because the between-partial census on an 85 ms window sits on a **leakage
//! floor** at about −48 dB — the note's own decaying partials smeared outside
//! the guard band — and `estimate::halo`'s `between C6` and `between C7` rows
//! have been aimed at that floor ever since.
//!
//! This instrument is the answer to "then what *is* the statistic?", and it
//! prints three candidates side by side on the same renders:
//!
//! * `--census` — §4's own between-partial census on a ladder of window
//!   lengths, against the **floor**: the engine with `resonance_coupling` at
//!   zero and `notes.duplex` emptied, which has no halo in it by construction,
//!   so whatever the census reads there is leakage and nothing else.
//! * `--sub` — the **sub-fundamental band**. A struck string's lowest mode is
//!   its own fundamental and inharmonicity only pushes the rest *up*, so a
//!   struck treble key puts **nothing at all** below `f0`. Everything a
//!   recording holds there is the rest of the instrument answering, which is
//!   the halo by definition and needs no tracker, no guard band and no census.
//! * `--harm` — §5's own reading: Salamander's `harmL*` release-resonance
//!   recordings, which are the halo *recorded alone*, against a strike of the
//!   same key. It reaches D#6 and stops, because above the damper break there
//!   is no release to record.
//!
//! `--ablate` is the diagnosis: the same statistic on the engine with one
//! mechanism moved at a time — the bus, the segments, the board's diffuse field
//! — so that the shortfall is charged to a path rather than guessed at.
//! `--frontier` is what the fit can still reach with the two knobs it has.
//!
//! `--write` is the listening material, and its one rule is item 506's: **a
//! halo is a ratio against its own key's strike, so both sides are divided by
//! their own strike before anything is written**, and the whole set is then
//! played at one gain. Written any other way the pair reproduces the two
//! recordings' gain staging as though it were the defect — it played +31.07 dB
//! at C6 where the column convicts +19.61 — and the instrument now prints the
//! played gap against the column's own digit rather than claiming it in prose.
//!
//! ```sh
//! cargo run --release -p forensics --bin treble_halo -- --sub --harm --ablate
//! ```

use std::path::{Path, PathBuf};

use piano_emulator::preset::Preset as EnginePreset;
use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_emulator::types::{Event, SAMPLE_RATE};
use piano_tuner::estimate::halo::{between_partials, resonance_level, HaloConfig};
use piano_tuner::library::{MechanismKind, SampleLibrary};
use piano_tuner::residual::frame_spectrum;
use piano_tuner::survey::SurveyConfig;

/// Velocity the engine is struck at, and the layer the recording is read from.
const VELOCITY: u16 = 90;

/// Windows the census ladder is taken on, samples at 48 kHz: 85 ms (the shipped
/// `HaloConfig`), 171, 341 (§4's update) and 683 ms.
const WINDOWS: [usize; 4] = [4_096, 8_192, 16_384, 32_768];

/// The window the sub-fundamental band is read on, samples: 341 ms.
const SUB_WINDOW: usize = 16_384;

/// Where the sub-fundamental band stops, as a fraction of the key's own `f0`.
const SUB_HI: f64 = 0.85;
/// Where it starts, as a fraction of `f0`.
const SUB_LO: f64 = 0.20;

/// How long the key is held before it is released, and how long the render
/// runs, seconds — the same numbers `tools::sympathetic` uses.
const HOLD_S: f32 = 1.0;
const RENDER_S: f32 = 5.0;

/// Salamander's attack groups' `amp_veltrack`, read off the SFZ.
const ATTACK_VELTRACK: f64 = 73.0;

fn mono(l: &[f32], r: &[f32]) -> Vec<f32> {
    l.iter().zip(r).map(|(&a, &b)| 0.5 * (a + b)).collect()
}

fn db(x: f64) -> f64 {
    10.0 * x.max(1.0e-300).log10()
}

/// The preset with nothing sympathetic in it: no bus, no segments. Whatever a
/// halo statistic reads here is its own floor.
fn without_halo(preset: &EnginePreset) -> EnginePreset {
    let mut bare = preset.clone();
    bare.voicing.resonance_coupling = 0.0;
    bare.notes.duplex = Vec::new();
    bare
}

/// The mechanism silenced: `harm*` is a recording of the strings alone, so the
/// key-off thump must not be counted as halo on either side.
fn without_mechanism(preset: &EnginePreset) -> EnginePreset {
    let mut quiet = preset.clone();
    for event in [
        &mut quiet.noise.key_off,
        &mut quiet.noise.damper_lift,
        &mut quiet.noise.pedal_down,
        &mut quiet.noise.pedal_up,
    ] {
        for anchor in &mut event.level_db {
            anchor.db = -200.0;
        }
    }
    quiet
}

fn engine_note(preset: &EnginePreset, key: u8, seconds: f32) -> Vec<f32> {
    let (l, r) = render_to_buffer(
        preset,
        &[RenderEvent::new(0.0, Event::NoteOn { key, vel: VELOCITY })],
        seconds,
    );
    mono(&l, &r)
}

/// One recording, and the gain in dB the instrument plays it at.
fn recorded_note(library: &SampleLibrary, key: u8, velocity: u8) -> Option<(Vec<f32>, f64)> {
    let sample = library.nearest_layer(key, velocity)?;
    let audio = piano_tuner::audio::load_at(&sample.path, SAMPLE_RATE as u32).ok()?;
    let gain_db =
        sample.volume_db + ATTACK_VELTRACK / 100.0 * 40.0 * (f64::from(velocity) / 127.0).log10();
    Some((audio.mono(), gain_db))
}

/// The *other* velocity layer of the same key: the take floor's partner, a
/// second independent recording of the same piano playing very nearly the same
/// note (`realism::VelocityLayers`).
fn other_layer(library: &SampleLibrary, key: u8, velocity: u8) -> Option<(Vec<f32>, f64, u8)> {
    let layers = library.layers(key);
    let here = layers
        .iter()
        .position(|s| s.lovel <= velocity && velocity <= s.hivel)?;
    let other = if here + 1 < layers.len() {
        here + 1
    } else {
        here.checked_sub(1)?
    };
    let sample = &layers[other];
    let vel = sample.midi_velocity();
    let audio = piano_tuner::audio::load_at(&sample.path, SAMPLE_RATE as u32).ok()?;
    let gain_db =
        sample.volume_db + ATTACK_VELTRACK / 100.0 * 40.0 * (f64::from(vel) / 127.0).log10();
    Some((audio.mono(), gain_db, vel))
}

// ------------------------------------------------------------------ the census

fn census(signal: &[f32], f0: f64, survey: &SurveyConfig, window: usize, at_s: f64) -> f64 {
    let Ok(note_config) = survey.note_config(f0) else {
        return f64::NAN;
    };
    let config = HaloConfig {
        window,
        at_s,
        ..HaloConfig::default()
    };
    between_partials(signal, f64::from(SAMPLE_RATE), f0, &note_config, &config)
        .map(|b| b.at_late_db)
        .unwrap_or(f64::NAN)
}

// --------------------------------------------------- the sub-fundamental band

/// First sample at half the signal's own peak: a struck note's onset, found the
/// same way on a render and on a recording so that any bias is common.
fn onset(signal: &[f32]) -> usize {
    let peak = signal.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
    signal
        .iter()
        .position(|&x| x.abs() >= 0.5 * peak)
        .unwrap_or(0)
}

/// The sub-fundamental statistic: energy strictly below the key's own
/// fundamental, against the energy of everything at and above it, in one late
/// window. `None` when the window does not fit.
fn sub_fundamental_db(signal: &[f32], f0: f64, at_s: f64, window: usize) -> Option<f64> {
    let start = onset(signal) + (at_s * f64::from(SAMPLE_RATE)) as usize;
    if start + window > signal.len() {
        return None;
    }
    let magnitude = frame_spectrum(signal, start, window, 1).ok()?;
    let bin_hz = f64::from(SAMPLE_RATE) / window as f64;
    let (mut below, mut above) = (0.0f64, 0.0f64);
    for (bin, &value) in magnitude.iter().enumerate() {
        let hz = bin as f64 * bin_hz;
        let power = f64::from(value) * f64::from(value);
        if hz >= SUB_LO * f0 && hz <= SUB_HI * f0 {
            below += power;
        } else if hz > SUB_HI * f0 && hz <= 12_000.0 {
            above += power;
        }
    }
    Some(db(below / above.max(1.0e-300)))
}

// ------------------------------------------------------------------- the harm

/// The recorded halo of one key, at the level the instrument plays it after a
/// hold of `HOLD_S`.
fn recorded_halo(
    library: &SampleLibrary,
    key: u8,
    velocity: u8,
    rt: &std::collections::BTreeMap<String, f64>,
    tier: &str,
) -> Option<(Vec<f32>, f64, f64)> {
    let sample = library
        .mechanism_of(MechanismKind::StringResonance)
        .into_iter()
        .find(|s| {
            s.key == Some(key)
                && s.lovel <= velocity
                && velocity <= s.hivel
                && s.path
                    .file_name()
                    .is_some_and(|n| n.to_string_lossy().starts_with(tier))
        })?;
    let name = sample.path.file_name()?.to_string_lossy().to_string();
    let rt_decay = rt.get(&name).copied().unwrap_or(0.0);
    let audio = piano_tuner::audio::load_at(&sample.path, SAMPLE_RATE as u32).ok()?;
    let veltrack = sample.amp_veltrack.unwrap_or(0.0);
    // SFZ's `rt_decay` is dB per second of *hold* taken off a release region,
    // which is the whole reason a release sample is comparable to a strike at
    // all: the halo the engine renders has been decaying for the hold too.
    let gain_db = sample.volume_db
        + veltrack / 100.0 * 40.0 * (f64::from(velocity) / 127.0).log10()
        - rt_decay * f64::from(HOLD_S);
    Some((audio.mono(), gain_db, rt_decay))
}

/// The engine's halo, isolated by subtraction — `tools::sympathetic::halo_level`
/// as an instrument, with the signal handed back rather than only its level.
fn engine_halo(preset: &EnginePreset, key: u8) -> (Vec<f32>, Vec<f32>) {
    let quiet = without_mechanism(preset);
    let bare = without_halo(&quiet);
    let events = [
        RenderEvent::new(0.0, Event::NoteOn { key, vel: VELOCITY }),
        RenderEvent::new(HOLD_S, Event::NoteOff { key, vel: 64 }),
    ];
    let (wl, wr) = render_to_buffer(&quiet, &events, RENDER_S);
    let (bl, br) = render_to_buffer(&bare, &events, RENDER_S);
    let with = mono(&wl, &wr);
    let without = mono(&bl, &br);
    let halo: Vec<f32> = with
        .iter()
        .zip(&without)
        .skip((HOLD_S * SAMPLE_RATE) as usize)
        .map(|(&a, &b)| a - b)
        .collect();
    let (sl, sr) = render_to_buffer(
        &quiet,
        &[RenderEvent::new(0.0, Event::NoteOn { key, vel: VELOCITY })],
        2.0,
    );
    (halo, mono(&sl, &sr))
}

/// The same, with the sustain pedal down from before the strike: every damper
/// off every string, which is the only way this engine lets the bus reach a
/// key below the damper break.
fn engine_halo_pedalled(preset: &EnginePreset, key: u8, amount: f32) -> (Vec<f32>, Vec<f32>) {
    use piano_emulator::types::PedalEvent;
    let quiet = without_mechanism(preset);
    let bare = without_halo(&quiet);
    let events = [
        RenderEvent::new(0.0, Event::Pedal(PedalEvent::Sustain(amount))),
        RenderEvent::new(0.01, Event::NoteOn { key, vel: VELOCITY }),
        RenderEvent::new(HOLD_S, Event::NoteOff { key, vel: 64 }),
        RenderEvent::new(HOLD_S, Event::Pedal(PedalEvent::Sustain(0.0))),
    ];
    let (wl, wr) = render_to_buffer(&quiet, &events, RENDER_S);
    let (bl, br) = render_to_buffer(&bare, &events, RENDER_S);
    let with = mono(&wl, &wr);
    let without = mono(&bl, &br);
    let halo: Vec<f32> = with
        .iter()
        .zip(&without)
        .skip((HOLD_S * SAMPLE_RATE) as usize)
        .map(|(&a, &b)| a - b)
        .collect();
    let (sl, sr) = render_to_buffer(
        &quiet,
        &[RenderEvent::new(0.0, Event::NoteOn { key, vel: VELOCITY })],
        2.0,
    );
    (halo, mono(&sl, &sr))
}

fn note_name(key: u8) -> String {
    piano_tuner::realism::note_name(key)
}

// ------------------------------------------------- the listening material's
// ------------------------------------------------- common reference

/// The raw sample peak — the quantity `resonance_level` divides by
/// (`residual::transient_metrics::peak`), and therefore the only denominator a
/// written file may use if its own peak is to *be* its `peak_db`.
fn peak_of(signal: &[f32]) -> f64 {
    signal.iter().fold(0.0f64, |m, &x| m.max(f64::from(x).abs()))
}

/// One side's halo put on the reference the **column** reads it against: its
/// own key's strike, with each side's own gain staging applied to both.
///
/// The column's statistic is a *ratio*, and the two sides' absolute levels have
/// no reason to agree — Salamander's C6 played at the SFZ's own gain peaks 11.5
/// dB above the engine's C6 — so a listening pair written at each side's raw
/// level plays a gap that is **not** the one the column convicts (item 506).
/// After this the invariant holds by construction:
///
/// `20 log10(peak(strike_referenced(h, gh, s, gs))) == resonance_level(h, gh,
/// s, gs).peak_db`
///
/// so two files written this way stand apart by exactly the column's own
/// per-key error, and one common make-up gain over the whole set moves neither
/// that gap nor the gaps between keys.
fn strike_referenced(
    halo: &[f32],
    halo_gain_db: f64,
    strike: &[f32],
    strike_gain_db: f64,
) -> Option<Vec<f32>> {
    let strike_peak = peak_of(strike) * 10f64.powf(strike_gain_db / 20.0);
    if !strike_peak.is_finite() || strike_peak <= 0.0 {
        return None;
    }
    let gain = 10f64.powf(halo_gain_db / 20.0) / strike_peak;
    Some(halo.iter().map(|&x| (f64::from(x) * gain) as f32).collect())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| args.iter().any(|a| a == name);
    let preset_path = PathBuf::from(
        args.iter()
            .find(|a| a.ends_with(".toml"))
            .cloned()
            .unwrap_or_else(|| "presets/salamander-c5.toml".into()),
    );
    let sfz = Path::new("data/salamander").join("SalamanderGrandPiano-V3+20200602.sfz");

    // The pair is taken out: this is a mono statistic and mono discipline says
    // the fold-down is the pan-pot's to about −120 dBFS, so `[voicing.mics]`
    // cannot move it — and taking it out keeps the reading independent of a
    // stereo refit that a different workstream owns.
    let preset = {
        let text = std::fs::read_to_string(&preset_path)?;
        let mut p = piano_tuner::preset::Preset::from_toml(&text).map_err(|e| e.to_string())?;
        p.voicing.mics = None;
        EnginePreset::from_toml(&p.to_toml()).map_err(|e| e.to_string())?
    };
    let bare = without_halo(&preset);
    let tuner_preset =
        piano_tuner::preset::Preset::from_toml(&std::fs::read_to_string(&preset_path)?)
            .map_err(|e| e.to_string())?;
    let library = SampleLibrary::from_sfz(&sfz)?;
    let survey = SurveyConfig::default();
    let f0_of = |key: u8| f64::from(tuner_preset.notes.f0_hz[(key - 21) as usize]);

    let mut recorded: Vec<u8> = library.keys().collect();
    recorded.sort_unstable();

    // ------------------------------------------------------------- the census
    if flag("--census") {
        println!("§4's census at 1.0 s, per window — does the recording stand clear of the floor?");
        println!("  key  window     ms   recording      engine       floor   rec-floor   eng-floor");
        for key in [72u8, 84, 96] {
            let f0 = f0_of(key);
            let Some((rec, _)) = recorded_note(&library, key, VELOCITY as u8) else {
                continue;
            };
            let eng = engine_note(&preset, key, 4.0);
            let flr = engine_note(&bare, key, 4.0);
            for window in WINDOWS {
                let r = census(&rec, f0, &survey, window, 1.0);
                let e = census(&eng, f0, &survey, window, 1.0);
                let f = census(&flr, f0, &survey, window, 1.0);
                println!(
                    "  {key:>3}  {window:>6}  {:>5.0}   {r:>+9.2}   {e:>+9.2}   {f:>+9.2}   \
                     {:>+9.2}   {:>+9.2}",
                    1000.0 * window as f64 / f64::from(SAMPLE_RATE),
                    r - f,
                    e - f
                );
            }
        }
        println!();
    }

    // ----------------------------------------------- the sub-fundamental band
    if flag("--sub") {
        println!(
            "the sub-fundamental band [{SUB_LO} f0, {SUB_HI} f0] at 1.0 s on a {:.0} ms window,\n\
             where the struck string has no mode at all:",
            1000.0 * SUB_WINDOW as f64 / f64::from(SAMPLE_RATE)
        );
        println!(
            "  key  name    f0 Hz     band Hz     recording  other take    engine     floor  \
             short by"
        );
        for &key in recorded.iter().filter(|&&k| k >= 48) {
            let f0 = f0_of(key);
            let Some((rec, _)) = recorded_note(&library, key, VELOCITY as u8) else {
                continue;
            };
            let r = sub_fundamental_db(&rec, f0, 1.0, SUB_WINDOW);
            let o = other_layer(&library, key, VELOCITY as u8)
                .and_then(|(s, _, _)| sub_fundamental_db(&s, f0, 1.0, SUB_WINDOW));
            let e = sub_fundamental_db(&engine_note(&preset, key, 4.0), f0, 1.0, SUB_WINDOW);
            let f = sub_fundamental_db(&engine_note(&bare, key, 4.0), f0, 1.0, SUB_WINDOW);
            let cell = |v: Option<f64>| v.map_or("     —".to_string(), |x| format!("{x:>+6.2}"));
            println!(
                "  {key:>3}  {:<5} {f0:>7.1}   {:>5.0}-{:>5.0}    {:>8}    {:>8}  {:>8}  {:>8}  \
                 {:>8}",
                note_name(key),
                SUB_LO * f0,
                SUB_HI * f0,
                cell(r),
                cell(o),
                cell(e),
                cell(f),
                match (r, e) {
                    (Some(r), Some(e)) => format!("{:+.2}", r - e),
                    _ => "—".to_string(),
                }
            );
        }
        println!();
    }

    // --------------------------------------------------------------- the harm
    if flag("--harm") {
        let rt = rt_decay_table(&sfz)?;
        println!("§5's `harm*` release resonances: the halo recorded alone, against a strike of");
        println!("the same key, after a {HOLD_S} s hold.");
        println!(
            "  key  name    the recording (harmL, v90)     the neighbouring layer    |          \
             the engine            | short by"
        );
        println!(
            "                level  decay s  centroid       level  decay s  centroid  |     \
             level  decay s  centroid |    level   colour"
        );
        // The recording read twice, from the library's two release tiers: the
        // take floor of a statistic whose bar has to have no engine in it.
        let read = |key: u8,
                    velocity: u8,
                    tier: &str|
         -> Option<piano_tuner::estimate::halo::ResonanceLevel> {
            let (strike, strike_gain) = recorded_note(&library, key, velocity)?;
            let (halo, halo_gain, _) = recorded_halo(&library, key, velocity, &rt, tier)?;
            resonance_level(
                &halo,
                halo_gain,
                &strike,
                strike_gain,
                f64::from(SAMPLE_RATE),
            )
        };
        let mut level_gap: Vec<f64> = Vec::new();
        let mut colour_gap: Vec<f64> = Vec::new();
        let mut level_floor: Vec<f64> = Vec::new();
        let mut colour_floor: Vec<f64> = Vec::new();
        for &key in recorded.iter().filter(|&&k| k >= 48) {
            let neighbour = other_layer(&library, key, VELOCITY as u8).map_or(VELOCITY as u8, |(_, _, v)| v);
            let (Some(rec), Some(alt)) = (
                read(key, VELOCITY as u8, "harmL"),
                read(key, neighbour, "harmL"),
            ) else {
                continue;
            };
            let (ehalo, estrike) = engine_halo(&preset, key);
            let Some(eng) = resonance_level(&ehalo, 0.0, &estrike, 0.0, f64::from(SAMPLE_RATE))
            else {
                continue;
            };
            let semitones = |a: f64, b: f64| 12.0 * (a / b).log2();
            level_gap.push(rec.peak_db - eng.peak_db);
            colour_gap.push(semitones(eng.centroid_hz, rec.centroid_hz));
            level_floor.push(rec.peak_db - alt.peak_db);
            colour_floor.push(semitones(alt.centroid_hz, rec.centroid_hz));
            println!(
                "  {key:>3}  {:<5} {:>+8.2} {:>7.2} {:>9.0}    {:>+8.2} {:>7.2} {:>9.0}  | \
                 {:>+8.2} {:>7.2} {:>9.0} | {:>+7.2}  {:>+6.1}",
                note_name(key),
                rec.peak_db,
                rec.decay_s,
                rec.centroid_hz,
                alt.peak_db,
                alt.decay_s,
                alt.centroid_hz,
                eng.peak_db,
                eng.decay_s,
                eng.centroid_hz,
                rec.peak_db - eng.peak_db,
                semitones(eng.centroid_hz, rec.centroid_hz),
            );
        }
        let bar = |v: &[f64]| -> f64 {
            let mut a: Vec<f64> = v.iter().map(|x| x.abs()).collect();
            a.sort_by(f64::total_cmp);
            let median = a[a.len() / 2];
            let mut d: Vec<f64> = a.iter().map(|x| (x - median).abs()).collect();
            d.sort_by(f64::total_cmp);
            1.4826 * d[d.len() / 2] / (a.len() as f64).sqrt()
        };
        let worst = |v: &[f64]| v.iter().fold(0.0f64, |a, &x| a.max(x.abs()));
        let median = |v: &[f64]| {
            let mut a: Vec<f64> = v.to_vec();
            a.sort_by(f64::total_cmp);
            a[a.len() / 2]
        };
        println!(
            "\n  level : worst {:+.2} dB, median {:+.2} dB over {} keys;  take floor worst \
             {:+.2}, bar {:.2}",
            worst(&level_gap),
            median(&level_gap),
            level_gap.len(),
            worst(&level_floor),
            bar(&level_floor)
        );
        println!(
            "  colour: worst {:+.2} st, median {:+.2} st;  take floor worst {:+.2}, bar {:.2}",
            worst(&colour_gap),
            median(&colour_gap),
            worst(&colour_floor),
            bar(&colour_floor)
        );
        println!();
    }

    // ------------------------------------------------------------ the ablation
    if flag("--ablate") {
        let rt = rt_decay_table(&sfz)?;
        println!("the diagnosis: the column's own statistic, one path moved at a time.");
        let mut variants: Vec<(String, EnginePreset, bool)> = Vec::new();
        variants.push(("shipped".into(), preset.clone(), false));
        let mut no_bus = preset.clone();
        no_bus.voicing.resonance_coupling = 0.0;
        variants.push(("bus off".into(), no_bus, false));
        let mut no_duplex = preset.clone();
        no_duplex.notes.duplex = Vec::new();
        variants.push(("segments off".into(), no_duplex, false));
        let mut max_bus = preset.clone();
        max_bus.voicing.resonance_coupling = piano_emulator::resonance::MAX_COUPLING;
        variants.push(("bus at MAX_COUPLING".into(), max_bus, false));
        for scale in [4.0f32, 10.0] {
            let mut long = preset.clone();
            long.soundboard.fdn_t60_lf *= scale;
            long.soundboard.fdn_t60_hf *= scale;
            variants.push((format!("fdn t60 x{scale:.0}"), long, false));
        }
        let mut wet = preset.clone();
        wet.soundboard.board_mix = 1.0;
        variants.push(("board_mix 1.0".into(), wet, false));
        let mut hf = preset.clone();
        hf.soundboard.fdn_t60_hf = preset.soundboard.fdn_t60_lf;
        variants.push(("fdn t60_hf = t60_lf".into(), hf, false));
        // **The decisive one.** With no pedal, the only strings the bus may
        // drive are the ones above the damper break — 91 and up — so the halo
        // of a struck C6 can only be built out of pitches *above* it. With the
        // pedal down every string in the instrument is free. If that is where
        // the missing 20 dB is, the engine's damper is a gate where the piano's
        // is a loss.
        for amount in [0.1f32, 0.2, 0.3, 0.5, 1.0] {
            variants.push((format!("sustain pedal {amount}"), preset.clone(), true));
        }
        {
            let c = 0.005f32;
            let mut v = preset.clone();
            v.voicing.resonance_coupling = c;
            variants.push((format!("coupling {c}"), v, false));
        }

        let keys = [60u8, 72, 84, 87];
        println!(
            "  halo level against the key's own strike, dB{}",
            keys.iter()
                .map(|&k| format!("{:>14}", note_name(k)))
                .collect::<Vec<_>>()
                .join("")
        );
        for (name, variant, pedal) in &variants {
            let cells: Vec<String> = keys
                .iter()
                .map(|&key| {
                    let (halo, strike) = if *pedal {
                        let amount = name
                            .rsplit(' ')
                            .next()
                            .and_then(|v| v.parse::<f32>().ok())
                            .unwrap_or(1.0);
                        engine_halo_pedalled(variant, key, amount)
                    } else {
                        engine_halo(variant, key)
                    };
                    resonance_level(&halo, 0.0, &strike, 0.0, f64::from(SAMPLE_RATE)).map_or(
                        "             -".into(),
                        |l| format!("{:>8.1} @{:>4.0}", l.peak_db, l.centroid_hz),
                    )
                })
                .collect();
            println!("  {name:<33} {}", cells.join(""));
        }
        let recs: Vec<String> = keys
            .iter()
            .map(|&key| {
                let Some((strike, sg)) = recorded_note(&library, key, VELOCITY as u8) else {
                    return "             -".into();
                };
                let Some((halo, hg, _)) = recorded_halo(&library, key, VELOCITY as u8, &rt, "harmL")
                else {
                    return "             -".into();
                };
                resonance_level(&halo, hg, &strike, sg, f64::from(SAMPLE_RATE)).map_or(
                    "             -".into(),
                    |l| format!("{:>8.1} @{:>4.0}", l.peak_db, l.centroid_hz),
                )
            })
            .collect();
        println!("  {:<33} {}", "THE RECORDING", recs.join(""));
    }
    // ------------------------------------------------------------ the frontier
    // What the halo fit can still reach with the two knobs it has. The backbone
    // gain and the coupling are degenerate in the sound and are held apart by
    // one thing only — the loop bound — so the honest frontier is a sweep of
    // the backbone with the coupling at its own ceiling at every point.
    if flag("--frontier") {
        use piano_tuner::estimate::halo::{peaks_from_body_modes, HaloVoicing};
        let text = std::fs::read_to_string(&preset_path)?;
        let base = piano_tuner::preset::Preset::from_toml(&text).map_err(|e| e.to_string())?;
        let peaks = peaks_from_body_modes(&base);
        let duplex_factor = base.duplex_response_factor();
        println!(
            "the halo fit's own frontier: the backbone lifted, the coupling at its ceiling,\n\
             the duplex occupying {duplex_factor:.3} of the loop."
        );
        println!(
            "  backbone dB   coupling   loop gain {}",
            [60u8, 72, 84, 87]
                .iter()
                .map(|&k| format!("{:>11}", note_name(k)))
                .collect::<Vec<_>>()
                .join("")
        );
        for backbone_gain_db in [0.0f64, 3.0, 6.0, 12.0, 20.0] {
            for treble_tilt_db in [0.0f64, 12.0] {
                let mut voicing = HaloVoicing {
                    coupling: base.voicing.resonance_coupling,
                    backbone_gain_db,
                    treble_tilt_db,
                };
                voicing.coupling = voicing.coupling_ceiling(&peaks, duplex_factor);
                let mut candidate = base.clone();
                candidate.voicing.mics = None;
                if voicing.apply(&mut candidate, peaks.clone()).is_err() {
                    println!("  {backbone_gain_db:>+11.1}   refused by the schema");
                    continue;
                }
                let Ok(engine) = EnginePreset::from_toml(&candidate.to_toml()) else {
                    println!("  {backbone_gain_db:>+11.1}   refused by the engine");
                    continue;
                };
                let cells: Vec<String> = [60u8, 72, 84, 87]
                    .iter()
                    .map(|&key| {
                        piano_tuner::estimate::halo::engine_halo_level(&engine, key, 90, 1.0)
                            .map_or("          -".into(), |l| format!("{:>11.1}", l.peak_db))
                    })
                    .collect();
                println!(
                    "  {backbone_gain_db:>+7.1}/{treble_tilt_db:>+4.0}   {:>8.5}   {:>9.4} {}",
                    voicing.coupling,
                    voicing.loop_gain(&peaks),
                    cells.join("")
                );
            }
        }
        let recs: Vec<String> = [60u8, 72, 84, 87]
            .iter()
            .map(|&key| {
                let rt = rt_decay_table(&sfz).unwrap_or_default();
                let _ = &rt;
                piano_tuner::estimate::halo::recorded_halo_level(&library, key, 90, 1.0, 73.0)
                    .map_or("          -".into(), |l| format!("{:>11.1}", l.peak_db))
            })
            .collect();
        println!("  {:<31}{}", "THE RECORDING", recs.join(""));
    }

    // -------------------------------------------------------------- the A/B
    // What a listener can judge: a treble phrase with the sympathetic path in
    // and out, the difference alone at the level it really is, and — the one
    // that matters — the halo of one key on its own, the engine's beside the
    // recording's, **each against its own side's strike** so that the gap a
    // listener hears is the gap the column convicts.
    if flag("--write") {
        let dir = Path::new("renders/halo");
        std::fs::create_dir_all(dir)?;
        let rt = rt_decay_table(&sfz)?;
        let phrase: Vec<RenderEvent> = [
            (0.00f32, 72u8),
            (0.18, 76),
            (0.36, 79),
            (0.54, 84),
            (0.72, 88),
            (0.90, 91),
            (1.20, 96),
            (1.50, 91),
            (1.68, 88),
            (1.86, 84),
            (2.04, 79),
            (2.22, 72),
        ]
        .iter()
        .flat_map(|&(t, key)| {
            [
                RenderEvent::new(t, Event::NoteOn { key, vel: VELOCITY }),
                RenderEvent::new(t + 0.12, Event::NoteOff { key, vel: 64 }),
            ]
        })
        .collect();
        let write = |name: &str, l: &[f32], r: &[f32]| -> Result<(), Box<dyn std::error::Error>> {
            piano_tuner::audio::Audio::new(SAMPLE_RATE as u32, vec![l.to_vec(), r.to_vec()])?
                .write_wav(dir.join(name))?;
            Ok(())
        };
        let (l, r) = render_to_buffer(&preset, &phrase, 8.0);
        let (bl, br) = render_to_buffer(&bare, &phrase, 8.0);
        write("treble-phrase-halo-on.wav", &l, &r)?;
        write("treble-phrase-halo-off.wav", &bl, &br)?;
        let dl: Vec<f32> = l.iter().zip(&bl).map(|(&a, &b)| a - b).collect();
        let dr: Vec<f32> = r.iter().zip(&br).map(|(&a, &b)| a - b).collect();
        write("treble-phrase-halo-only.wav", &dl, &dr)?;
        println!(
            "renders/halo/: the twelve-note treble phrase, the sympathetic path in and out, and \
             the difference alone at {:+.2} dB of the phrase.",
            db(
                dl.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>()
                    / bl.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>()
            )
        );
        // One key's halo, twice, **on a common strike reference** (item 506).
        //
        // The column's statistic is a *ratio*: `resonance_level::peak_db` is the
        // halo's peak over the peak of a strike of the same key, each side on
        // its own scale. The two sides' strikes do not stand at the same
        // absolute level and there is no reason they should — the engine's C6
        // strike peaks at −24.18 dBFS where the library's, played at the SFZ's
        // own gain, peaks at −12.72 — so writing the two halos *raw* plays a gap
        // that is not the column's: C6 played +31.07 dB where the column
        // convicts +19.61, and C5 was honest only because its two strikes
        // happen to agree to 0.02 dB.
        //
        // So each halo is divided by **its own side's strike peak**, which is
        // exactly the denominator `resonance_level` divides by (`transient
        // _metrics::peak` is the raw sample peak), and a written file's own peak
        // is therefore its `peak_db`. One make-up gain shared by every file in
        // the set then makes them audible without touching any ratio — within a
        // key, which is the A/B, or between keys, which is the slope.
        // Each entry: file name, the halo already divided by its own strike, and
        // the `peak_db` that division makes its peak.
        let mut pending: Vec<(String, Vec<f32>, f64)> = Vec::new();
        // The two sides' strike peaks, absolute — the disagreement that makes
        // the division necessary, printed rather than asserted.
        let mut strikes: Vec<(u8, f64, f64)> = Vec::new();
        for key in [72u8, 84] {
            let (halo, strike) = engine_halo(&preset, key);
            let Some(scaled) = strike_referenced(&halo, 0.0, &strike, 0.0) else {
                continue;
            };
            let peak_db = db(peak_of(&scaled).powi(2));
            pending.push((format!("{}-halo-engine.wav", note_name(key)), scaled, peak_db));

            let (Some((rec, halo_gain, _)), Some((rec_strike, strike_gain))) = (
                recorded_halo(&library, key, VELOCITY as u8, &rt, "harmL"),
                recorded_note(&library, key, VELOCITY as u8),
            ) else {
                continue;
            };
            let Some(scaled) = strike_referenced(&rec, halo_gain, &rec_strike, strike_gain) else {
                continue;
            };
            strikes.push((
                key,
                db(peak_of(&strike).powi(2)),
                db((peak_of(&rec_strike) * 10f64.powf(strike_gain / 20.0)).powi(2)),
            ));
            let peak_db = db(peak_of(&scaled).powi(2));
            pending.push((
                format!("{}-halo-recording.wav", note_name(key)),
                scaled,
                peak_db,
            ));
        }
        // The one gain the whole set is played at: the loudest file lands at
        // LISTEN_PEAK_DBFS and every other file keeps its distance from it.
        const LISTEN_PEAK_DBFS: f64 = -3.0;
        let loudest = pending.iter().map(|p| p.2).fold(f64::NEG_INFINITY, f64::max);
        let makeup_db = LISTEN_PEAK_DBFS - loudest;
        let makeup = 10f64.powf(makeup_db / 20.0) as f32;
        for (name, samples, _) in &pending {
            let played: Vec<f32> = samples.iter().map(|&x| x * makeup).collect();
            write(name, &played, &played)?;
        }
        println!(
            "renders/halo/: C5 and C6's halo alone, the engine's beside the recording's, each \
             divided by its own side's strike and the set played {makeup_db:+.2} dB up, so a \
             file's peak is its own `peak_db` and the gap between a pair is the column's."
        );
        // The check the raw files did not have: what a listener hears, against
        // the column's own digit for the same key, read through the column's own
        // functions rather than through this file's copies of them.
        println!(
            "  the strikes the two sides are divided by, absolute (this is why the division is \
             not optional):"
        );
        for &(key, eng, rec) in &strikes {
            println!(
                "    {key:>3}  {:<5} engine {eng:>7.2} dBFS   recording {rec:>7.2} dBFS   \
                 they differ by {:>+6.2}",
                note_name(key),
                rec - eng
            );
        }
        println!("  key  name   engine dBFS  recording dBFS   played gap   the column   agree");
        for key in [72u8, 84] {
            let name_of = |side: &str| format!("{}-halo-{side}.wav", note_name(key));
            let peak_db = |name: String| {
                pending
                    .iter()
                    .find(|p| p.0 == name)
                    .map(|p| p.2 + makeup_db)
            };
            let (Some(eng), Some(rec)) = (peak_db(name_of("engine")), peak_db(name_of("recording")))
            else {
                continue;
            };
            let column = match (
                piano_tuner::estimate::halo::engine_halo_level(
                    &preset,
                    key,
                    VELOCITY as u8,
                    f64::from(HOLD_S),
                ),
                piano_tuner::estimate::halo::recorded_halo_level(
                    &library,
                    key,
                    VELOCITY as u8,
                    f64::from(HOLD_S),
                    ATTACK_VELTRACK,
                ),
            ) {
                (Some(e), Some(r)) => r.peak_db - e.peak_db,
                _ => f64::NAN,
            };
            let played = rec - eng;
            println!(
                "  {key:>3}  {:<5} {eng:>11.2}  {rec:>14.2}   {played:>+10.2}   {column:>+10.2}   \
                 {}",
                note_name(key),
                if (played - column).abs() <= 0.05 {
                    "yes"
                } else {
                    "NO"
                }
            );
        }
    }

    Ok(())
}

/// `rt_decay`, in dB per second of hold, for each key's release-resonance
/// region — parsed off the SFZ because `library::MechanismSample` does not
/// carry it and it is the whole difference between a release sample's level and
/// a strike's.
fn rt_decay_table(
    sfz: &Path,
) -> Result<std::collections::BTreeMap<String, f64>, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(sfz)?;
    let mut table = std::collections::BTreeMap::new();
    let mut rt = 0.0f64;
    let mut is_resonance = false;
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with("<group>") {
            rt = field(line, "rt_decay").unwrap_or(0.0);
            is_resonance = line.contains("trigger=release") && !line.contains("pitch_keytrack=0");
        } else if line.starts_with("<region>") && is_resonance {
            if let Some(at) = line.find("sample=") {
                let path = line[at + 7..].split_whitespace().next().unwrap_or("");
                if let Some(name) = path.rsplit(['/', '\\']).next() {
                    table.insert(name.to_string(), rt);
                }
            }
        }
    }
    Ok(table)
}

fn field(line: &str, name: &str) -> Option<f64> {
    let at = line.find(&format!("{name}="))? + name.len() + 1;
    line[at..].split_whitespace().next()?.parse::<f64>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The gain staging the recording side carries and the engine side does
    /// not: the SFZ's `volume` + velocity law on the strike, and the same plus
    /// `rt_decay` on the release resonance.
    const HALO_GAIN_DB: f64 = -7.5;
    const STRIKE_GAIN_DB: f64 = -3.25;

    /// Two sides of one key, with their strikes at deliberately different
    /// absolute levels: an "engine" whose strike peaks at 0.06 and a
    /// "recording" whose strike, once its gain is applied, peaks 23.5 dB above
    /// it — where the real pair is 11.46 dB apart at C6. **Both sides' halos
    /// are 40 dB under their own played strike**, so the honest gap between
    /// them is zero and anything a listener hears is arithmetic.
    fn sides() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let tone = |amplitude: f64, hz: f64, n: usize| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let t = i as f64 / f64::from(SAMPLE_RATE);
                    (amplitude * (-2.0 * t).exp() * (std::f64::consts::TAU * hz * t).sin()) as f32
                })
                .collect()
        };
        let n = 24_000;
        let under = 10f64.powf(-40.0 / 20.0);
        (
            tone(0.06, 523.0, n),
            tone(0.06 * under, 311.0, n),
            tone(0.9, 523.0, n),
            // Written on the file's own scale, so it has to give back the gain
            // difference the two groups are played at.
            tone(
                0.9 * under * 10f64.powf((STRIKE_GAIN_DB - HALO_GAIN_DB) / 20.0),
                311.0,
                n,
            ),
        )
    }

    /// The invariant the listening material rests on: a strike-referenced
    /// file's own peak **is** its `peak_db`, so the gap between two of them is
    /// the column's own error and nothing else.
    #[test]
    fn a_strike_referenced_halos_peak_is_the_columns_own_digit() {
        let (e_strike, e_halo, r_strike, r_halo) = sides();
        // The recording side carries gain staging on both of its files; the
        // engine side carries none, exactly as `--write` reads them.
        let (halo_gain, strike_gain) = (HALO_GAIN_DB, STRIKE_GAIN_DB);
        for (halo, hg, strike, sg) in [
            (&e_halo, 0.0, &e_strike, 0.0),
            (&r_halo, halo_gain, &r_strike, strike_gain),
        ] {
            let written = strike_referenced(halo, hg, strike, sg).expect("a strike with energy");
            let level = resonance_level(halo, hg, strike, sg, f64::from(SAMPLE_RATE))
                .expect("both signals have peaks");
            let played_db = 20.0 * peak_of(&written).log10();
            assert!(
                (played_db - level.peak_db).abs() < 1.0e-6,
                "written {played_db} against the column's {}",
                level.peak_db
            );
        }
    }

    /// The defect this replaces, reproduced on the arithmetic it was written
    /// with (item 506): each halo at its own raw level, which plays the two
    /// strikes' disagreement on top of the column's verdict. Here the true gap
    /// is zero and the raw pair plays 23.5 dB of it; at C6 it played +31.07
    /// where the column convicts +19.61.
    #[test]
    fn writing_each_halo_raw_plays_the_two_strikes_disagreement_as_if_it_were_the_defect() {
        let (e_strike, e_halo, r_strike, r_halo) = sides();
        let (halo_gain, strike_gain) = (HALO_GAIN_DB, STRIKE_GAIN_DB);
        let column = resonance_level(&r_halo, halo_gain, &r_strike, strike_gain, f64::from(SAMPLE_RATE))
            .expect("recording")
            .peak_db
            - resonance_level(&e_halo, 0.0, &e_strike, 0.0, f64::from(SAMPLE_RATE))
                .expect("engine")
                .peak_db;
        assert!(column.abs() < 1.0e-6, "the two sides are equally haloed: {column}");

        // The old way: the recording at its own SFZ gain, the engine raw.
        let raw_engine = 20.0 * peak_of(&e_halo).log10();
        let raw_recording = 20.0 * peak_of(&r_halo).log10() + halo_gain;
        let raw_gap = raw_recording - raw_engine;
        let strikes_disagree = (20.0 * peak_of(&r_strike).log10() + strike_gain)
            - 20.0 * peak_of(&e_strike).log10();
        assert!(strikes_disagree > 20.0, "the fixture's strikes: {strikes_disagree}");
        assert!(
            (raw_gap - (column + strikes_disagree)).abs() < 1.0e-6,
            "raw {raw_gap}, column {column}, strikes {strikes_disagree}"
        );

        // And the fix removes exactly that term.
        let engine = strike_referenced(&e_halo, 0.0, &e_strike, 0.0).expect("engine");
        let recording =
            strike_referenced(&r_halo, halo_gain, &r_strike, strike_gain).expect("recording");
        let played = 20.0 * peak_of(&recording).log10() - 20.0 * peak_of(&engine).log10();
        assert!((played - column).abs() < 1.0e-6, "{played} against {column}");
    }

    /// One make-up gain over the whole set moves no ratio: not the pair's, and
    /// not the distance between two keys, which is where the column's slope is.
    #[test]
    fn the_sets_common_make_up_gain_moves_neither_a_pair_nor_the_slope() {
        let (e_strike, e_halo, r_strike, r_halo) = sides();
        let quiet: Vec<f32> = r_halo.iter().map(|&x| x * 0.1).collect();
        let files: Vec<Vec<f32>> = [
            strike_referenced(&e_halo, 0.0, &e_strike, 0.0),
            strike_referenced(&r_halo, HALO_GAIN_DB, &r_strike, STRIKE_GAIN_DB),
            strike_referenced(&quiet, HALO_GAIN_DB, &r_strike, STRIKE_GAIN_DB),
        ]
        .into_iter()
        .map(|f| f.expect("a strike with energy"))
        .collect();
        let before: Vec<f64> = files.iter().map(|f| 20.0 * peak_of(f).log10()).collect();
        let makeup = 10f64.powf((-3.0 - before.iter().cloned().fold(f64::MIN, f64::max)) / 20.0)
            as f32;
        let after: Vec<f64> = files
            .iter()
            .map(|f| {
                let played: Vec<f32> = f.iter().map(|&x| x * makeup).collect();
                20.0 * peak_of(&played).log10()
            })
            .collect();
        assert!((after.iter().cloned().fold(f64::MIN, f64::max) + 3.0).abs() < 1.0e-4);
        for i in 0..files.len() {
            for j in 0..files.len() {
                let moved = (after[i] - after[j]) - (before[i] - before[j]);
                assert!(moved.abs() < 1.0e-4, "pair {i},{j} moved {moved}");
            }
        }
    }
}
