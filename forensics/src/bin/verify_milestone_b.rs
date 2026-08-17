//! Independent audio verification of the sympathetic-resonance milestone: the
//! bridge admittance, the duplex segments and the per-key stereo spread.
//!
//! Nothing here trusts the fit's own report. Everything is re-measured from
//! rendered audio with the same code `docs/history/TUNING_REPORT.md` §4 measured the
//! recordings with, and the recordings are re-measured beside it:
//!
//! 1. **The spectrum census**, before and after, at C4 / C6 / C7 across
//!    velocities, against the Salamander recording of the same key and layer.
//!    The headline number is `between@1s`: the broadband energy between the
//!    partials one second after the strike, which the report has at −3.5 dB on
//!    a fortissimo C7 against the engine's −48.
//! 2. **The engine's own halo**, isolated: the same gesture rendered with the
//!    coupling on and off, subtracted, so what is left is only what the bus
//!    and the segments put there.
//! 3. **Render health on the measured preset** — non-finite samples, clipping,
//!    DC and derivative outliers over the demo, the pedal phrase, the halo
//!    phrase and 45 s of dense random playing.
//! 4. **Neutrality**: the default preset's demo rendered from the file and
//!    from the built-in, sample for sample, plus the same for the measured
//!    preset with its new sections stripped against the preset as it stood
//!    before them.
//! 5. **What the new sections cost**, before against after, on `SPEC.md`'s
//!    worst case.
//! 6. **The five targets the fit was aimed at**, re-measured on the file that
//!    was written, so that the fit's own report is checked against the preset
//!    it emitted rather than against the candidate it last held in memory.
//! 7. **Where the between-partial floor comes from**, by taking the engine
//!    apart: the census number the milestone was supposed to move cannot be
//!    moved by anything the milestone touched, and this says what does move
//!    it.
//!
//! ```text
//! cargo run --release -p forensics --bin verify_milestone_b -- [before.toml] [1 2 3 4 5]
//! ```
//!
//! `before.toml` is the stage-1 preset as it stood before this milestone
//! (`git show HEAD:presets/salamander-c5.toml`); without it the passes that
//! need a baseline are skipped.

use std::path::{Path, PathBuf};

use piano_emulator::render::{
    demo_sequence, halo_sequence, render_to_buffer, RenderEvent, DEMO_DURATION_S, HALO_DURATION_S,
};
use piano_emulator::types::{Event, PedalEvent};
use piano_tuner::estimate::halo::{
    between_partials, resonance_level, salamander_targets, HaloConfig,
};
use piano_tuner::pipeline::track_refined;
use piano_tuner::preset::{equal_temperament, Preset};
use piano_tuner::residual::{
    band_split, classify_peaks, frame_spectrum, partial_levels, PeakClass, ResidualConfig,
};
use piano_tuner::stft::find_peaks;
use piano_tuner::survey::SurveyConfig;
use piano_tuner::trajectory::InharmonicModel;
use piano_tuner::{audio, SampleLibrary, SAMPLE_RATE};

const SR: f64 = SAMPLE_RATE as f64;

/// The keys the milestone is about: the one the report says the engine already
/// gets right, and the two it says it does not.
const KEYS: [u8; 3] = [60, 84, 96];

/// Velocities the census runs at, covering the report's own layers.
const VELOCITIES: [u8; 4] = [40, 68, 90, 108];

const NOTE_SECONDS: f32 = 8.0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let root = repo.join("data/salamander");
    let after_path = repo.join("presets/salamander-c5.toml");

    let mut args = std::env::args().skip(1).peekable();
    let before_path = match args.peek() {
        Some(a) if a.ends_with(".toml") => Some(PathBuf::from(args.next().expect("peeked"))),
        _ => None,
    };
    let sections: Vec<String> = args.collect();
    let wanted = |s: &str| sections.is_empty() || sections.iter().any(|a| a == s);

    let after = piano_emulator::preset::Preset::load(&after_path)?;
    let before = before_path
        .as_ref()
        .map(|p| piano_emulator::preset::Preset::load(p))
        .transpose()?;
    let tuner_preset = Preset::load(&after_path)?;
    let library = SampleLibrary::from_sfz(root.join("SalamanderGrandPiano-V3+20200602.sfz")).ok();
    let config = SurveyConfig::default();
    let residual = ResidualConfig::default();

    if wanted("1") {
        census_pass(
            library.as_ref(),
            before.as_ref(),
            &after,
            &tuner_preset,
            &config,
            &residual,
        );
    }
    if wanted("2") {
        halo_pass(before.as_ref(), &after);
    }
    if wanted("3") {
        if let Some(before) = before.as_ref() {
            println!("\n--- the same material on the preset as it stood before the milestone");
            health_pass(before);
        }
        health_pass(&after);
    }
    if wanted("4") {
        neutrality_pass(&repo, before.as_ref(), &after)?;
    }
    if wanted("5") {
        cost_pass(before.as_ref(), &after);
    }
    if wanted("6") {
        target_pass(before.as_ref(), &after, &tuner_preset, &config)?;
    }
    if wanted("7") {
        floor_pass(&after, &tuner_preset, &config)?;
    }
    Ok(())
}

// ------------------------------------------------------------- 1. the census

/// What one census frame came to, in the columns `docs/history/TUNING_REPORT.md` §4 uses.
#[derive(Clone, Copy, Debug)]
struct Census {
    peaks: usize,
    transverse: usize,
    unexplained: usize,
    loudest_unexplained_db: f64,
    between_at_strike_db: f64,
    between_at_one_second_db: f64,
}

fn census_pass(
    library: Option<&SampleLibrary>,
    before: Option<&piano_emulator::preset::Preset>,
    after: &piano_emulator::preset::Preset,
    tuner_preset: &Preset,
    config: &SurveyConfig,
    residual: &ResidualConfig,
) {
    println!("\n=== 1. spectrum census: what radiates that is not the struck string\n");
    println!(
        "        (TUNING_REPORT section 4: between@1s is -44.3 dB on a recorded C4 ff against"
    );
    println!("         the engine's -47.0; -22.1/-26.4 at C6 against -47.7; -15.9/-13.0/-3.5 at");
    println!("         C7 against -48.2)\n");
    println!(
        "{:>12} {:>4} {:>4} {:>6} {:>7} {:>7} {:>9} {:>10} {:>11}",
        "source", "key", "vel", "peaks", "transv", "unexpl", "loudest", "between@0", "between@1s"
    );
    for key in KEYS {
        for vel in VELOCITIES {
            if let Some(library) = library {
                if let Some(sample) = library
                    .layers(key)
                    .iter()
                    .find(|s| (s.lovel..=s.hivel).contains(&vel))
                {
                    if let Ok(recording) = audio::load_at(&sample.path, SAMPLE_RATE) {
                        report(
                            "salamander",
                            key,
                            sample.midi_velocity(),
                            census_one(&recording.mono(), key, tuner_preset, config, residual),
                        );
                    }
                }
            }
            if let Some(before) = before {
                let signal = render_mono(before, key, vel);
                report(
                    "before",
                    key,
                    vel,
                    census_one(&signal, key, tuner_preset, config, residual),
                );
            }
            let signal = render_mono(after, key, vel);
            report(
                "after",
                key,
                vel,
                census_one(&signal, key, tuner_preset, config, residual),
            );
        }
        println!();
    }
}

fn report(source: &str, key: u8, vel: u8, census: Option<Census>) {
    let Some(c) = census else {
        println!("{source:>12} {key:>4} {vel:>4}   (not tracked)");
        return;
    };
    println!(
        "{source:>12} {key:>4} {vel:>4} {:>6} {:>7} {:>7} {:>9.1} {:>10.1} {:>11.1}",
        c.peaks,
        c.transverse,
        c.unexplained,
        c.loudest_unexplained_db,
        c.between_at_strike_db,
        c.between_at_one_second_db,
    );
}

fn render_mono(preset: &piano_emulator::preset::Preset, key: u8, vel: u8) -> Vec<f32> {
    let events = [RenderEvent::new(0.0, Event::NoteOn { key, vel: u16::from(vel) })];
    let (l, r) = render_to_buffer(preset, &events, NOTE_SECONDS);
    l.iter().zip(&r).map(|(&a, &b)| 0.5 * (a + b)).collect()
}

/// `residuals.rs`'s `census_one`, returning its numbers instead of printing
/// them. Term for term the same measurement, so that a row here and a row of
/// `docs/history/TUNING_REPORT.md` §4 mean the same thing.
fn census_one(
    signal: &[f32],
    key: u8,
    preset: &Preset,
    config: &SurveyConfig,
    residual: &ResidualConfig,
) -> Option<Census> {
    let note_config = config.note_config(equal_temperament(key)).ok()?;
    let (trajectories, fit) = track_refined(
        signal,
        SR,
        InharmonicModel::harmonic(equal_temperament(key)),
        &note_config,
    )
    .ok()?;
    let loudest = trajectories
        .tracks
        .iter()
        .filter_map(|t| t.peak())
        .map(|p| p.amplitude)
        .fold(0.0f64, f64::max);
    let partials: Vec<(u32, f64)> = trajectories
        .tracks
        .iter()
        .filter(|t| {
            t.peak()
                .is_some_and(|p| p.amplitude >= loudest * 10f64.powf(-residual.level_db / 20.0))
        })
        .filter_map(|t| t.weighted_frequency().map(|f| (t.k, f)))
        .collect();
    if partials.is_empty() {
        return None;
    }
    let neighbours: Vec<(u8, f64)> = (21..=108)
        .filter(|&k| k != key)
        .filter_map(|k| preset.f0(k).map(|f| (k, f64::from(f))))
        .collect();

    let window = note_config.tracker.stft.window;
    let onset = (trajectories.onset_s * SR) as usize;
    let sustain = onset + (SR * 0.5) as usize;
    let spectrum = frame_spectrum(signal, sustain, window, 2).ok()?;
    let fft_size = window * 2;
    let mut peaks = Vec::new();
    find_peaks(&spectrum, SR, fft_size, -70.0, &mut peaks);
    let band = (0.75 * fit.model.partial(1), 12_000.0);
    peaks.retain(|p| (band.0..=band.1).contains(&p.frequency_hz));
    let lobe = 4.0 * SR / window as f64;
    let reference = partial_levels(
        &spectrum,
        SR,
        fft_size,
        &partials.iter().map(|&(_, f)| f).collect::<Vec<f64>>(),
        lobe,
    )
    .into_iter()
    .flatten()
    .fold(0.0f64, f64::max)
    .max(f64::MIN_POSITIVE);
    let census = classify_peaks(&peaks, &partials, &neighbours, reference, lobe, residual);

    let short = 4096;
    let guard = (4.0 * SR / short as f64).max(3.0);
    let frequencies: Vec<f64> = partials.iter().map(|&(_, f)| f).collect();
    let between = |start: usize| -> f64 {
        frame_spectrum(signal, start, short, 1).map_or(f64::NAN, |frame| {
            band_split(&frame, SR, short, &frequencies, guard, band).between_db()
        })
    };

    Some(Census {
        peaks: census.len(),
        transverse: census
            .iter()
            .filter(|p| matches!(p.class, PeakClass::Transverse { .. }))
            .count(),
        unexplained: census
            .iter()
            .filter(|p| p.class == PeakClass::Unexplained)
            .count(),
        loudest_unexplained_db: census
            .iter()
            .filter(|p| p.class == PeakClass::Unexplained)
            .map(|p| p.level_db)
            .fold(f64::NEG_INFINITY, f64::max),
        between_at_strike_db: between(onset),
        between_at_one_second_db: between(onset + SR as usize),
    })
}

// -------------------------------------------------------------- 2. the halo

/// The halo on its own: the same gesture rendered with the instrument coupled
/// and with it uncoupled, subtracted sample by sample.
///
/// Two gestures, because the milestone has two mechanisms. A held bass octave
/// under a treble strike is the bus (the treble note drives strings whose
/// dampers are up); a staccato treble note with everything else damped is the
/// duplex (nothing else can ring at all).
fn halo_pass(before: Option<&piano_emulator::preset::Preset>, after: &piano_emulator::preset::Preset) {
    println!("\n=== 2. the halo isolated: what the note left behind, minus the note\n");
    println!("        (levels are dB relative to the peak of the strike that caused them.");
    println!("         `harm*` in TUNING_REPORT section 5 measures -31 dB at C3 and -39 at C5,");
    println!("         ringing 1-2 s, as the peak of the release resonance against the strike)\n");

    // Three gestures, one per path the milestone opened.
    //
    // The first two are subtractions against a *silent* control: the same
    // events rendered with the keys not held, so what is left is only what the
    // held strings radiated. The third has nothing to subtract - a staccato
    // treble note with every damper down is the duplex on its own.
    /// A gesture: what it is called, the events, the control events to
    /// subtract (empty means "subtract the same preset without its segments"),
    /// how long to render, and when the strike is over.
    type Gesture = (&'static str, Vec<RenderEvent>, Vec<RenderEvent>, f32, f32);

    let gestures: [Gesture; 3] = [
        (
            "held C2/C3/E3/G3, C5-E5-G5 struck into them",
            held_chord(true),
            held_chord(false),
            6.0,
            1.2,
        ),
        (
            "held C3, G4 struck into it (G4 = C3's third partial)",
            vec![
                RenderEvent::new(0.1, Event::KeyDown { key: 48 }),
                RenderEvent::new(0.5, Event::NoteOn { key: 67, vel: 108 }),
                RenderEvent::new(0.7, Event::NoteOff { key: 67, vel: 96 }),
            ],
            vec![
                RenderEvent::new(0.5, Event::NoteOn { key: 67, vel: 108 }),
                RenderEvent::new(0.7, Event::NoteOff { key: 67, vel: 96 }),
            ],
            6.0,
            1.0,
        ),
        (
            "staccato C6, damped, minus the same without segments",
            vec![
                RenderEvent::new(0.5, Event::NoteOn { key: 84, vel: 108 }),
                RenderEvent::new(0.65, Event::NoteOff { key: 84, vel: 64 }),
            ],
            Vec::new(),
            4.0,
            1.0,
        ),
    ];
    println!(
        "{:>10} {:>46} {:>13} {:>10} {:>10}",
        "preset", "gesture", "halo peak", "at +0.5 s", "at +1.5 s"
    );
    for (name, events, control, seconds, from) in &gestures {
        for (label, preset) in [("before", before), ("after", Some(after))] {
            let Some(preset) = preset else { continue };
            let (l, r) = render_to_buffer(preset, events, *seconds);
            let mono: Vec<f32> = l.iter().zip(&r).map(|(&a, &b)| 0.5 * (a + b)).collect();
            let halo: Vec<f32> = if control.is_empty() {
                // The duplex on its own: the same events on the same preset
                // with `notes.duplex` taken out, so the difference is only
                // what the segments radiated. C6 is one of the 23 keys the
                // estimator gave a table to, and its damper works, so nothing
                // else can still be sounding.
                let mut bare = preset.clone();
                bare.notes.duplex = Vec::new();
                let (cl, cr) = render_to_buffer(&bare, events, *seconds);
                mono.iter()
                    .zip(cl.iter().zip(&cr))
                    .map(|(&m, (&a, &b))| m - 0.5 * (a + b))
                    .collect()
            } else {
                let (cl, cr) = render_to_buffer(preset, control, *seconds);
                mono.iter()
                    .zip(cl.iter().zip(&cr))
                    .map(|(&m, (&a, &b))| m - 0.5 * (a + b))
                    .collect()
            };
            let strike = mono.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            let re_strike = |v: f64| 20.0 * (v / f64::from(strike).max(1e-30)).log10();
            let window_at = |t: f32| -> f64 {
                let start = (t * SAMPLE_RATE as f32) as usize;
                let end = (start + SAMPLE_RATE as usize / 2).min(halo.len());
                if start >= end {
                    return f64::NEG_INFINITY;
                }
                re_strike(f64::from(rms(&halo[start..end])))
            };
            // The halo's own peak, taken from the point the strike is over so
            // that the strike itself is not what is measured.
            let peak = halo[(from * SAMPLE_RATE as f32) as usize..]
                .iter()
                .fold(0.0f32, |m, &v| m.max(v.abs()));
            println!(
                "{label:>10} {name:>46} {:>13.1} {:>10.1} {:>10.1}",
                re_strike(f64::from(peak)),
                window_at(*from),
                window_at(from + 1.0)
            );
        }
    }
}

/// The halo phrase's second movement on its own: four keys pressed silently
/// and a C major triad two octaves above struck into them, or the same triad
/// with nothing held.
fn held_chord(hold: bool) -> Vec<RenderEvent> {
    let mut events = Vec::new();
    if hold {
        for key in [36u8, 48, 52, 55] {
            events.push(RenderEvent::new(0.1, Event::KeyDown { key }));
        }
    }
    for key in [72u8, 76, 79] {
        events.push(RenderEvent::new(0.5, Event::NoteOn { key, vel: 118 }));
        events.push(RenderEvent::new(0.64, Event::NoteOff { key, vel: 96 }));
    }
    events
}

// ------------------------------------------------------- 3. render health

fn health_pass(preset: &piano_emulator::preset::Preset) {
    println!("\n=== 3. render health on presets/salamander-c5.toml\n");
    println!(
        "{:>28} {:>10} {:>12} {:>8} {:>10} {:>9} {:>22}",
        "material", "non-finite", "peak dBFS", "clipped", "DC dB", "clicks", "largest derivative"
    );
    let material: [(&str, Vec<RenderEvent>, f32); 4] = [
        ("demo", demo_sequence(), DEMO_DURATION_S),
        ("pedal phrase", pedal_phrase(), 14.0),
        ("halo phrase", halo_sequence(), HALO_DURATION_S),
        ("45 s of dense playing", dense_phrase(), 45.0),
    ];
    for (name, events, seconds) in &material {
        let (left, right) = render_to_buffer(preset, events, *seconds);
        for (channel_name, channel) in [("L", &left), ("R", &right)] {
            let nan = channel.iter().filter(|x| !x.is_finite()).count();
            let peak = channel.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
            let clipped = channel.iter().filter(|x| x.abs() >= 1.0).count();
            let dc = channel.iter().map(|&x| f64::from(x)).sum::<f64>() / channel.len() as f64;
            let outliers = derivative_outliers(channel, 1e-4);
            let clicks = outliers.iter().filter(|o| o.1 > 12.0).count();
            println!(
                "{:>26} {channel_name} {nan:>10} {:>12.2} {clipped:>8} {:>10.1} {clicks:>9} \
                 {:>15.1} at {:>4.2} s",
                name,
                20.0 * f64::from(peak).max(1e-30).log10(),
                20.0 * dc.abs().max(1e-12).log10(),
                outliers.first().map_or(0.0, |o| o.1),
                outliers.first().map_or(0.0, |o| o.0),
            );
        }
    }
}

/// Samples whose first difference stands above the RMS first difference of the
/// surrounding 43 ms, with |difference| at least `floor`. Worst first, as
/// (seconds, ratio, step) — `verify_milestone_a.rs`'s scan.
fn derivative_outliers(signal: &[f32], floor: f64) -> Vec<(f64, f64, f64)> {
    let window = 2048usize;
    let d: Vec<f64> = signal
        .windows(2)
        .map(|w| f64::from(w[1]) - f64::from(w[0]))
        .collect();
    let mut out = Vec::new();
    for (start, chunk) in d.chunks(window).enumerate() {
        let rms = (chunk.iter().map(|&x| x * x).sum::<f64>() / chunk.len() as f64).sqrt();
        for (i, &x) in chunk.iter().enumerate() {
            if x.abs() > floor {
                out.push((
                    (start * window + i) as f64 / SR,
                    x.abs() / rms.max(1e-12),
                    x.abs(),
                ));
            }
        }
    }
    out.sort_by(|a, b| b.1.total_cmp(&a.1));
    out
}

// ------------------------------------------------------------ 4. neutrality

fn neutrality_pass(
    repo: &Path,
    before: Option<&piano_emulator::preset::Preset>,
    after: &piano_emulator::preset::Preset,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== 4. neutrality: a preset without the new sections is the old instrument\n");
    let file = piano_emulator::preset::Preset::load(&repo.join("presets/default.toml"))?;
    let built_in = piano_emulator::preset::Preset::default();
    let demo = demo_sequence();
    let (file_l, file_r) = render_to_buffer(&file, &demo, DEMO_DURATION_S);
    let (ref_l, ref_r) = render_to_buffer(&built_in, &demo, DEMO_DURATION_S);
    println!(
        "   presets/default.toml demo vs the built-in default: L {}, R {} ({} samples)",
        verdict(&file_l, &ref_l),
        verdict(&file_r, &ref_r),
        file_l.len()
    );

    // And the measured preset with its new sections taken back out has to be
    // the preset as it stood before them — the "absent means the old
    // behaviour" contract, on the file that exercises it.
    if let Some(before) = before {
        let mut stripped = after.clone();
        stripped.voicing.bridge = None;
        stripped.notes.duplex = Vec::new();
        stripped.notes.pan_spread = Vec::new();
        stripped.voicing.resonance_coupling = before.voicing.resonance_coupling;
        let (a_l, a_r) = render_to_buffer(&stripped, &demo, DEMO_DURATION_S);
        let (b_l, b_r) = render_to_buffer(before, &demo, DEMO_DURATION_S);
        println!(
            "   presets/salamander-c5.toml with the three sections stripped vs the same \
             preset before this milestone: L {}, R {}",
            verdict(&a_l, &b_l),
            verdict(&a_r, &b_r)
        );
    }
    Ok(())
}

fn verdict(a: &[f32], b: &[f32]) -> String {
    if a == b {
        return "bit-identical".to_string();
    }
    let difference: f32 = (a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        / a.len().max(1) as f32)
        .sqrt();
    format!("differs, {difference:e} RMS")
}

// ------------------------------------------------------------------ 5. cost

fn cost_pass(before: Option<&piano_emulator::preset::Preset>, after: &piano_emulator::preset::Preset) {
    println!("\n=== 5. what the milestone costs: SPEC's worst case, before against after\n");
    const AUDIO_S: f32 = 10.0;
    let mut events = vec![RenderEvent::new(
        0.0,
        Event::Pedal(PedalEvent::Sustain(1.0)),
    )];
    for (i, key) in (21u8..=108).enumerate() {
        let t = i as f32 * 0.02;
        events.push(RenderEvent::new(t, Event::NoteOn { key, vel: 90 }));
        events.push(RenderEvent::new(t + 0.1, Event::NoteOff { key, vel: 64 }));
    }
    let measure = |preset: &piano_emulator::preset::Preset| {
        let mut best = f32::INFINITY;
        for _ in 0..3 {
            let start = std::time::Instant::now();
            let (l, _) = render_to_buffer(preset, &events, AUDIO_S);
            assert!(l.iter().all(|v| v.is_finite()));
            best = best.min(start.elapsed().as_secs_f32() / AUDIO_S);
        }
        best
    };
    if let Some(before) = before {
        println!(
            "   before this milestone: {:.1}% of one core",
            100.0 * measure(before)
        );
    }
    println!(
        "   presets/salamander-c5.toml: {:.1}% of one core (design goal 50%)",
        100.0 * measure(after)
    );
}

// ------------------------------------------------------------------ material

fn pedal_phrase() -> Vec<RenderEvent> {
    let mut events = vec![RenderEvent::new(0.0, Event::Pedal(PedalEvent::Sustain(1.0)))];
    let mut strike = |at: f32, keys: &[u8], vel: u8, hold: f32| {
        for &key in keys {
            events.push(RenderEvent::new(at, Event::NoteOn { key, vel: u16::from(vel) }));
            events.push(RenderEvent::new(at + hold, Event::NoteOff { key, vel: 64 }));
        }
    };
    strike(0.05, &[33, 45], 96, 0.6);
    strike(1.20, &[52, 57, 60, 64], 80, 0.5);
    strike(3.50, &[59, 62, 67, 71], 88, 0.5);
    strike(6.00, &[36, 48, 55, 60, 64, 67], 104, 0.8);
    events.push(RenderEvent::new(10.0, Event::Pedal(PedalEvent::Sustain(0.0))));
    events.sort_by(|a, b| a.time_s.total_cmp(&b.time_s));
    events
}

fn dense_phrase() -> Vec<RenderEvent> {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut events = Vec::new();
    let mut t = 0.1f32;
    while t < 42.0 {
        let key = 21 + (next() % 88) as u8;
        let vel = 20 + (next() % 108) as u8;
        let hold = 0.2 + (next() % 2300) as f32 / 1000.0;
        events.push(RenderEvent::new(t, Event::NoteOn { key, vel: u16::from(vel) }));
        events.push(RenderEvent::new(t + hold, Event::NoteOff { key, vel: 64 }));
        t += 0.03 + (next() % 120) as f32 / 1000.0;
    }
    for i in 0..14 {
        let at = 1.5 + 3.0 * i as f32;
        let value = if i % 2 == 0 { 1.0 } else { 0.0 };
        events.push(RenderEvent::new(at, Event::Pedal(PedalEvent::Sustain(value))));
    }
    events.sort_by(|a, b| a.time_s.total_cmp(&b.time_s));
    events
}

fn rms(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    (values.iter().map(|&v| v * v).sum::<f32>() / values.len() as f32).sqrt()
}

// ------------------------------------------------------ 6. the fit's targets

/// `docs/history/TUNING_REPORT.md` §4 and §5's five numbers, measured on the presets as
/// they sit on disk.
///
/// The gestures are `sympathetic.rs`'s, because a verification that used a
/// different gesture would be measuring a different quantity: the `harm*` rows
/// are a strike-and-release with the mechanism silenced and the uncoupled
/// render subtracted, and the `between` rows are one struck note a second in.
fn target_pass(
    before: Option<&piano_emulator::preset::Preset>,
    after: &piano_emulator::preset::Preset,
    tuner_preset: &Preset,
    config: &SurveyConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== 6. the five targets the sympathetic fit was aimed at\n");
    println!(
        "{:>14} {:>10} {:>12} {:>12} {:>12} {:>10}",
        "target", "want", "tolerance", "before", "after", "moved"
    );
    for target in salamander_targets() {
        let measure = |preset: &piano_emulator::preset::Preset| -> f64 {
            if target.name.starts_with("harm") {
                halo_level(preset, target.key)
            } else {
                let index = usize::from(target.key - 21);
                let f0 = f64::from(tuner_preset.notes.f0_hz[index]);
                let Ok(note_config) = config.note_config(f0) else {
                    return f64::NAN;
                };
                let signal = render_mono_seconds(preset, target.key, 90, 5.0);
                between_partials(&signal, SR, f0, &note_config, &HaloConfig::default())
                    .map(|b| b.at_late_db)
                    .unwrap_or(f64::NAN)
            }
        };
        let after_db = measure(after);
        let before_db = before.map_or(f64::NAN, measure);
        println!(
            "{:>14} {:>10.1} {:>12.2} {:>12.1} {:>12.1} {:>10.1}",
            target.name,
            target.target_db,
            target.tolerance_db,
            before_db,
            after_db,
            after_db - before_db,
        );
    }
    Ok(())
}

fn render_mono_seconds(
    preset: &piano_emulator::preset::Preset,
    key: u8,
    vel: u8,
    seconds: f32,
) -> Vec<f32> {
    let events = [RenderEvent::new(0.0, Event::NoteOn { key, vel: u16::from(vel) })];
    let (l, r) = render_to_buffer(preset, &events, seconds);
    l.iter().zip(&r).map(|(&a, &b)| 0.5 * (a + b)).collect()
}

/// `sympathetic.rs`'s `halo_level`, term for term: the sympathetic
/// contribution isolated by subtracting the uncoupled render, with the
/// mechanism silenced so a key-off thump is not counted as halo, quoted
/// against a strike of the same key at the same velocity.
fn halo_level(engine: &piano_emulator::preset::Preset, key: u8) -> f64 {
    const HOLD_S: f32 = 1.0;
    const RENDER_S: f32 = 5.0;
    let mut quiet = engine.clone();
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
    let mut bare = quiet.clone();
    bare.voicing.resonance_coupling = 0.0;
    bare.notes.duplex = Vec::new();

    let events = [
        RenderEvent::new(0.0, Event::NoteOn { key, vel: 90 }),
        RenderEvent::new(HOLD_S, Event::NoteOff { key, vel: 64 }),
    ];
    let (wl, wr) = render_to_buffer(&quiet, &events, RENDER_S);
    let (bl, br) = render_to_buffer(&bare, &events, RENDER_S);
    let halo: Vec<f32> = wl
        .iter()
        .zip(&wr)
        .zip(bl.iter().zip(&br))
        .skip((HOLD_S * SAMPLE_RATE as f32) as usize)
        .map(|((&a, &b), (&c, &d))| 0.5 * (a + b) - 0.5 * (c + d))
        .collect();
    let (sl, sr) = render_to_buffer(
        &quiet,
        &[RenderEvent::new(0.0, Event::NoteOn { key, vel: 90 })],
        2.0,
    );
    let strike: Vec<f32> = sl.iter().zip(&sr).map(|(&a, &b)| 0.5 * (a + b)).collect();
    resonance_level(&halo, 0.0, &strike, 0.0, SR).map_or(f64::NAN, |level| level.peak_db)
}

// -------------------------------------------------- 7. the between-partial floor

/// What actually sets `between@1s` in the engine's renders.
///
/// Section 1 finds the number unmoved by the whole milestone, to a tenth of a
/// decibel. That is only interesting if one knows what is holding it there, so
/// this takes the instrument apart one path at a time and re-measures. Each
/// row is the shipped preset with one thing changed.
fn floor_pass(
    preset: &piano_emulator::preset::Preset,
    tuner_preset: &Preset,
    config: &SurveyConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== 7. what holds the engine's between-partial energy where it is\n");
    let mut coupled = preset.clone();
    coupled.voicing.resonance_coupling *= 20.0;
    let mut no_board = preset.clone();
    no_board.soundboard.board_mix = 0.0;
    let mut no_body = preset.clone();
    no_body.soundboard.body_mix = 0.0;
    let mut dry = preset.clone();
    dry.soundboard.fdn_t60_lf = 0.05;
    dry.soundboard.fdn_t60_hf = 0.05;
    let mut silent = preset.clone();
    silent.voicing.resonance_coupling = 0.0;
    silent.notes.duplex = Vec::new();

    let variants: [(&str, &piano_emulator::preset::Preset); 6] = [
        ("as shipped", preset),
        ("coupling x20", &coupled),
        ("no sympathetic coupling, no duplex", &silent),
        ("no board path (board_mix = 0)", &no_board),
        ("no body modes (body_mix = 0)", &no_body),
        ("diffuse field 50 ms instead of seconds", &dry),
    ];
    println!(
        "{:>40} {:>14} {:>14} {:>14}",
        "variant", "C4 between@1s", "C6 between@1s", "C7 between@1s"
    );
    for (name, variant) in variants {
        let mut row = Vec::new();
        for key in KEYS {
            let index = usize::from(key - 21);
            let f0 = f64::from(tuner_preset.notes.f0_hz[index]);
            let note_config = config.note_config(f0)?;
            let signal = render_mono_seconds(variant, key, 108, 5.0);
            row.push(
                between_partials(&signal, SR, f0, &note_config, &HaloConfig::default())
                    .map(|b| b.at_late_db)
                    .unwrap_or(f64::NAN),
            );
        }
        println!(
            "{name:>40} {:>14.1} {:>14.1} {:>14.1}",
            row[0], row[1], row[2]
        );
    }

    // Whether that is a floor of the *instrument* or of the *measurement*: a
    // Hann window's own leakage from the partials lands between them, and if
    // it is what is being read then it moves with the window and the recorded
    // numbers, which are 25 dB above it, are unaffected either way.
    println!("\n   the same number on the shipped preset at four window lengths:");
    for window in [2_048usize, 4_096, 16_384, 65_536] {
        let mut row = Vec::new();
        for key in KEYS {
            let index = usize::from(key - 21);
            let f0 = f64::from(tuner_preset.notes.f0_hz[index]);
            let note_config = config.note_config(f0)?;
            let signal = render_mono_seconds(preset, key, 108, 5.0);
            let halo_config = HaloConfig {
                window,
                ..HaloConfig::default()
            };
            row.push(
                between_partials(&signal, SR, f0, &note_config, &halo_config)
                    .map(|b| b.at_late_db)
                    .unwrap_or(f64::NAN),
            );
        }
        println!(
            "{:>40} {:>14.1} {:>14.1} {:>14.1}",
            format!("window {window} ({:.0} ms)", 1000.0 * window as f64 / SR),
            row[0],
            row[1],
            row[2]
        );
    }
    Ok(())
}
