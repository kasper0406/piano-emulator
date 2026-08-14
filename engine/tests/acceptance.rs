//! The acceptance tests from SPEC.md, in the spec's order.
//!
//! Everything here runs offline through `Engine::process` — the same code path
//! the audio callback uses — so what is measured is what comes out of the
//! device. `tests/smoke.rs` holds the coarser end-to-end invariants.

use piano_emulator::preset::Preset;
use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_emulator::string::StringParams;
use piano_emulator::types::{key_index, Event, PedalEvent, BLOCK, SAMPLE_RATE};
use rustfft::{num_complex::Complex32, FftPlanner};

/// Every test runs the instrument as shipped — or, when `PIANO_PRESET` names a
/// file, the instrument in that file.
///
/// The override exists for estimated presets: most of what is asserted below is
/// a property of the *model* (nothing is NaN, nothing clips, a damped note
/// stops, the halo needs the pedal, the worst case fits the budget) and has to
/// hold for any preset the engine will play, while a few assertions are windows
/// around numbers a particular piano was tuned to and may legitimately differ
/// for another one. Running the suite against a candidate preset is how the two
/// kinds get told apart; `TUNING.md`'s Phase D reports the second kind as
/// measured-versus-window rather than moving the window.
fn preset() -> Preset {
    match std::env::var("PIANO_PRESET") {
        Ok(path) if !path.is_empty() => Preset::load(std::path::Path::new(&path))
            .unwrap_or_else(|e| panic!("PIANO_PRESET={path}: {e}")),
        _ => Preset::default(),
    }
}

fn string_params(key: u8) -> StringParams {
    preset().string_params(key)
}

/// The same instrument with its action silenced.
///
/// The mechanism noise is a preset field like any other, so switching it off is
/// something a preset can say rather than something the tests have to reach
/// into the engine for. Used where a measurement is about the strings and a
/// thump in the same window would answer a different question.
fn silent_mechanism(mut preset: Preset) -> Preset {
    for event in [
        &mut preset.noise.key_off,
        &mut preset.noise.damper_lift,
        &mut preset.noise.pedal_down,
        &mut preset.noise.pedal_up,
    ] {
        for anchor in &mut event.level_db {
            anchor.db = -200.0;
        }
    }
    // The hammer's own noise is silent in every shipped preset, but a preset
    // named by `PIANO_PRESET` may voice it, and "the action, silenced" has to
    // mean all five events.
    for anchor in &mut preset.noise.strike.level_db {
        anchor.db = -200.0;
    }
    preset.validate().expect("a silenced action is a legal preset");
    preset
}

/// The strongest sympathetic coupling this preset may legally ask for.
///
/// `resonance::MAX_COUPLING` stopped being the answer the day `voicing.bridge`
/// arrived: the bound is on the *loop*, `resonance_coupling * max|B|`
/// (`DECISIONS.md` 149), and the duplex adds a second one on top of it (156).
/// On `presets/salamander-c5.toml`, whose fitted bridge peaks at 26.7 dB, the
/// scalar ceiling makes a loop gain of 1.08 and the validator refuses it — so
/// a test that wants "as coupled as the schema allows" has to ask the preset
/// rather than the constant, or it fails on every voiced instrument for a
/// reason that has nothing to do with what it is measuring.
///
/// Found by asking the validator instead of re-deriving its arithmetic, which
/// is the point: this is a test helper, and a second copy of the bound here
/// would be a second thing to keep in step. A quarter is close enough to the
/// ceiling for the measurements below, which want a strongly coupled
/// instrument and not a precisely coupled one.
fn strongest_legal_coupling(preset: &Preset) -> f32 {
    let mut probe = preset.clone();
    let mut candidate = piano_emulator::resonance::MAX_COUPLING;
    for _ in 0..40 {
        probe.voicing.resonance_coupling = candidate;
        if probe.validate().is_ok() {
            return candidate;
        }
        candidate *= 0.75;
    }
    panic!("no sympathetic coupling at all is legal for this preset")
}

// ---------------------------------------------------------------- helpers

fn peak(signal: &[f32]) -> f32 {
    signal.iter().fold(0.0f32, |m, &v| m.max(v.abs()))
}

fn rms(signal: &[f32]) -> f32 {
    (signal.iter().map(|v| v * v).sum::<f32>() / signal.len().max(1) as f32).sqrt()
}

fn db(amplitude: f32) -> f32 {
    20.0 * amplitude.max(1.0e-30).log10()
}

fn window(signal: &[f32], from_s: f32, to_s: f32) -> &[f32] {
    let a = (from_s * SAMPLE_RATE) as usize;
    let b = ((to_s * SAMPLE_RATE) as usize).min(signal.len());
    &signal[a.min(b)..b]
}

/// Renders a sequence and returns the mono sum of the two channels. The tests
/// below measure energy and partial frequencies, neither of which the stereo
/// image should change.
fn render_mono(events: &[RenderEvent], duration_s: f32) -> Vec<f32> {
    let (l, r) = render_to_buffer(&preset(), events, duration_s);
    l.iter().zip(&r).map(|(a, b)| a + b).collect()
}

/// A single strike, released at `release_s` (never, if that is past the end).
fn strike(key: u8, vel: u8, release_s: f32, duration_s: f32) -> Vec<f32> {
    let mut events = vec![RenderEvent::new(0.0, Event::NoteOn { key, vel })];
    if release_s < duration_s {
        events.push(RenderEvent::new(release_s, Event::NoteOff { key, vel: 64 }));
    }
    render_mono(&events, duration_s)
}

/// Hann-windowed, zero-padded magnitude spectrum of `signal`.
fn spectrum(signal: &[f32], fft_size: usize) -> Vec<f32> {
    assert!(signal.len() <= fft_size);
    let mut buffer: Vec<Complex32> = (0..fft_size)
        .map(|i| {
            let x = signal.get(i).copied().unwrap_or(0.0);
            let w = 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / signal.len() as f32).cos();
            Complex32::new(x * if i < signal.len() { w } else { 0.0 }, 0.0)
        })
        .collect();
    FftPlanner::new()
        .plan_fft_forward(fft_size)
        .process(&mut buffer);
    buffer[..fft_size / 2].iter().map(|c| c.norm()).collect()
}

/// Frequency of the strongest peak of `mag` within `tolerance` (as a ratio) of
/// `around`, refined by parabolic interpolation of the log magnitudes. Log
/// magnitudes because a Gaussian-ish main lobe is parabolic in dB, which is
/// what makes the interpolation accurate to a small fraction of a bin.
fn peak_near(mag: &[f32], fft_size: usize, around: f32, tolerance: f32) -> f32 {
    let bin = |f: f32| (f * fft_size as f32 / SAMPLE_RATE) as usize;
    let lo = bin(around * (1.0 - tolerance)).max(1);
    let hi = bin(around * (1.0 + tolerance)).min(mag.len() - 2);
    assert!(lo <= hi, "search band for {around} Hz is empty");
    let k = (lo..=hi)
        .max_by(|&a, &b| mag[a].total_cmp(&mag[b]))
        .expect("non-empty band");
    let (a, b, c) = (
        mag[k - 1].max(1e-30).ln(),
        mag[k].max(1e-30).ln(),
        mag[k + 1].max(1e-30).ln(),
    );
    let delta = 0.5 * (a - c) / (a - 2.0 * b + c);
    (k as f32 + delta) * SAMPLE_RATE / fft_size as f32
}

fn cents(measured: f32, expected: f32) -> f32 {
    1200.0 * (measured / expected).log2()
}

/// Amplitude envelope of the band around `f`, sampled every `hop` samples.
///
/// Complex demodulation with a 4 Hz low pass: narrow enough to isolate one
/// partial of one note, wide enough to pass the unison beat itself, which a
/// short-window DFT would alias into the measurement.
fn band_envelope(signal: &[f32], f: f32, hop: usize) -> Vec<f32> {
    let w = std::f64::consts::TAU * f as f64 / SAMPLE_RATE as f64;
    let a = (-std::f32::consts::TAU * 4.0 / SAMPLE_RATE).exp() as f64;
    let (mut re, mut im) = (0.0f64, 0.0f64);
    let mut out = Vec::with_capacity(signal.len() / hop + 1);
    for (n, &x) in signal.iter().enumerate() {
        let phase = w * n as f64;
        re = a * re + (1.0 - a) * x as f64 * phase.cos();
        im = a * im - (1.0 - a) * x as f64 * phase.sin();
        if n % hop == 0 {
            out.push(2.0 * (re * re + im * im).sqrt() as f32);
        }
    }
    out
}

/// Time at which a partial's envelope has fallen 60 dB below its peak.
///
/// The envelope of a unison group is not monotonic — it beats — so the
/// crossing is taken from a running maximum over `smooth_s`, i.e. the answer is
/// the last moment the partial is still audible rather than the first moment it
/// happens to be in a beat null.
fn t60(signal: &[f32], f: f32, smooth_s: f32) -> f32 {
    const HOP_S: f32 = 0.01;
    let hop = (HOP_S * SAMPLE_RATE) as usize;
    let env = band_envelope(signal, f, hop);
    let span = (smooth_s / HOP_S) as usize;
    let target = env.iter().cloned().fold(0.0f32, f32::max) * 1.0e-3;
    let mut last = 0.0;
    for (i, _) in env.iter().enumerate() {
        let hi = env[i..(i + span).min(env.len())]
            .iter()
            .cloned()
            .fold(0.0f32, f32::max);
        if hi >= target {
            last = i as f32 * HOP_S;
        }
    }
    last
}

// ------------------------------------------------- 1. tuning/inharmonicity

/// Partials 1..8 must sit on `f_k = k f0 sqrt(1 + B k^2)`, and far enough off
/// the harmonic series that the stretch is real rather than rounding.
#[test]
fn partials_follow_the_inharmonic_series() {
    const FFT: usize = 1 << 18;
    for key in [45u8, 60, 69] {
        let y = strike(key, 90, 10.0, 3.0);
        // Past the hammer noise, and short compared with the unison beat
        // period: three strings a couple of cents apart cannot be resolved by
        // any window, so they merge into one peak, and only while they are
        // still near in phase does that peak sit at the group's real pitch.
        // Later in the note the merged lobe is lopsided and the peak wanders by
        // a cent or two — which is also why a tuner listens to the attack.
        let mag = spectrum(window(&y, 0.05, 0.65), FFT);
        let params = string_params(key);

        for k in 1..=8 {
            let expected = params.partial_freq(k);
            // Half a semitone: wide enough to catch a mistuned partial, narrow
            // enough that neighbouring partials cannot be picked up instead.
            let measured = peak_near(&mag, FFT, expected, 0.028);
            let error = cents(measured, expected);
            assert!(
                error.abs() < 3.0,
                "key {key} partial {k}: {measured:.2} Hz vs {expected:.2} Hz ({error:.2} cents)"
            );
        }

        // ... and the eighth partial is not where a harmonic series would put it.
        let harmonic = 8.0 * params.f0;
        let stretch = cents(params.partial_freq(8), harmonic);
        assert!(
            stretch > 5.0,
            "key {key}: partial 8 stretched only {stretch:.1} cents"
        );
        let measured = peak_near(&mag, FFT, params.partial_freq(8), 0.028);
        assert!(
            cents(measured, harmonic) > 5.0,
            "key {key}: measured partial 8 at {measured:.1} Hz is harmonic ({harmonic:.1} Hz)"
        );
    }
}

// -------------------------------------------------------- 2. decay sanity

#[test]
fn the_fundamental_decays_over_a_pianistic_time() {
    let c4 = strike(60, 90, 30.0, 25.0);
    let measured = t60(&c4, string_params(60).partial_freq(1), 1.5);
    assert!(
        (8.0..20.0).contains(&measured),
        "C4 fundamental T60 {measured:.1} s, expected 8..20 s"
    );

    let c7 = strike(96, 90, 30.0, 4.0);
    let measured = t60(&c7, string_params(96).partial_freq(1), 0.3);
    assert!(
        (0.3..2.0).contains(&measured),
        "C7 fundamental T60 {measured:.2} s, expected 0.3..2 s"
    );
}

#[test]
fn releasing_a_key_with_the_pedal_up_stops_the_note() {
    for key in [36u8, 48, 60, 79] {
        let y = strike(key, 90, 1.0, 2.5);
        let before = rms(window(&y, 0.9, 1.0));
        let after = rms(window(&y, 1.5, 1.6));
        assert!(
            db(after / before) < -40.0,
            "key {key}: only {:.1} dB down 0.5 s after release",
            db(after / before)
        );
    }
}

// ------------------------------------------------------------- 3. beating

/// Three detuned unison strings must make the fundamental's envelope wobble,
/// with a beat period in the seconds — not decay monotonically.
#[test]
fn a_unison_group_beats() {
    const HOP_S: f32 = 0.02;
    let y = strike(60, 90, 30.0, 12.0);
    let hop = (HOP_S * SAMPLE_RATE) as usize;
    let env = band_envelope(&y, string_params(60).partial_freq(1), hop);

    // Beat troughs: a sample lower than both of its neighbours a quarter of a
    // second away, so the exponential decay itself cannot produce one.
    let step = (0.25 / HOP_S) as usize;
    let troughs: Vec<f32> = (step..env.len() - step)
        .filter(|&i| env[i] < env[i - step] && env[i] < env[i + step])
        .map(|i| i as f32 * HOP_S)
        .collect();
    assert!(!troughs.is_empty(), "the fundamental decays monotonically");

    // Consecutive troughs are one beat period apart; group the raw indices into
    // troughs by taking the first of each run.
    let mut periods = Vec::new();
    let mut previous = troughs[0];
    for &t in &troughs[1..] {
        if t - previous > 0.5 {
            periods.push(t - previous);
            previous = t;
        }
    }
    assert!(
        !periods.is_empty(),
        "only one beat trough in 12 s: {troughs:?}"
    );
    for p in &periods {
        assert!(
            (0.5..10.0).contains(p),
            "beat period {p:.2} s outside 0.5..10 s ({periods:?})"
        );
    }
}

// --------------------------------------------------------------- 4. pedal

#[test]
fn the_sustain_pedal_holds_a_released_note() {
    const KEY: u8 = 48; // C3
    const RELEASE_S: f32 = 1.0;
    const PROBE_S: f32 = RELEASE_S + 2.0;

    let energy = |events: Vec<RenderEvent>| {
        let y = render_mono(&events, PROBE_S + 0.2);
        rms(window(&y, PROBE_S, PROBE_S + 0.2))
    };
    let with_pedal = |pedal: f32| {
        vec![
            RenderEvent::new(0.0, Event::Pedal(PedalEvent::Sustain(pedal))),
            RenderEvent::new(0.01, Event::NoteOn { key: KEY, vel: 90 }),
            RenderEvent::new(RELEASE_S, Event::NoteOff { key: KEY, vel: 64 }),
        ]
    };

    // The reference is the note left sustaining because the key is still down.
    let held = energy(vec![RenderEvent::new(
        0.01,
        Event::NoteOn { key: KEY, vel: 90 },
    )]);
    let pedalled = energy(with_pedal(1.0));
    let damped = energy(with_pedal(0.0));

    assert!(
        db(pedalled / held) > -12.0,
        "pedal down is {:.1} dB below the held note, expected within 12 dB",
        db(pedalled / held)
    );
    assert!(
        db(damped / held) < -40.0,
        "pedal up is only {:.1} dB below the held note, expected 40 dB",
        db(damped / held)
    );
}

#[test]
fn half_pedal_damps_partially() {
    let energy = |pedal: f32| {
        let events = vec![
            RenderEvent::new(0.0, Event::Pedal(PedalEvent::Sustain(pedal))),
            RenderEvent::new(0.01, Event::NoteOn { key: 48, vel: 90 }),
            RenderEvent::new(1.0, Event::NoteOff { key: 48, vel: 64 }),
        ];
        let y = render_mono(&events, 1.8);
        db(rms(window(&y, 1.6, 1.8)))
    };
    let (up, half, down) = (energy(0.0), energy(0.5), energy(1.0));
    assert!(
        up < half - 6.0 && half < down - 6.0,
        "half pedal is not between the two: {up:.1} / {half:.1} / {down:.1} dB"
    );
}

// ------------------------------------------------- 5. sympathetic resonance

/// A string nobody struck must pick up energy from one that was struck, and
/// only while its damper is off the strings.
#[test]
fn undamped_strings_resonate_with_the_ones_being_played() {
    use piano_emulator::engine::Engine;
    use piano_emulator::types::BLOCK;

    // C3 is never struck; G4 is struck hard and released. With the pedal down
    // C3's dampers are off its strings, so the bank must gain energy.
    let energy_of_c3 = |sustain: f32| {
        let (mut engine, _tx) = Engine::new(&preset());
        engine.handle_event(Event::Pedal(PedalEvent::Sustain(sustain)));
        engine.handle_event(Event::NoteOn { key: 67, vel: 120 });
        let (mut l, mut r) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
        let blocks = (1.5 * SAMPLE_RATE / BLOCK as f32) as usize;
        for b in 0..blocks {
            if b == (0.2 * SAMPLE_RATE / BLOCK as f32) as usize {
                engine.handle_event(Event::NoteOff { key: 67, vel: 64 });
            }
            engine.process(&mut l, &mut r);
        }
        engine.voice(key_index(48).unwrap()).string().energy()
    };

    let coupled = energy_of_c3(1.0);
    let isolated = energy_of_c3(0.0);
    assert!(
        coupled > 0.0,
        "C3 picked up nothing at all with the pedal down"
    );
    assert!(
        coupled > isolated * 100.0,
        "halo {coupled:e} is not clear of the damped case {isolated:e}"
    );
}

/// The audible form of the same thing: with the pedal down a struck-and-
/// released note leaves a halo ringing a second later; with the pedal up the
/// instrument is silent.
#[test]
fn the_pedal_down_halo_outlives_the_note() {
    let render = |sustain: f32| {
        let events = vec![
            RenderEvent::new(0.0, Event::Pedal(PedalEvent::Sustain(sustain))),
            RenderEvent::new(0.01, Event::NoteOn { key: 55, vel: 120 }),
            RenderEvent::new(0.3, Event::NoteOff { key: 55, vel: 64 }),
        ];
        let y = render_mono(&events, 1.6);
        db(rms(window(&y, 1.3, 1.6)))
    };
    let halo = render(1.0);
    let dry = render(0.0);
    assert!(halo > dry + 30.0, "halo {halo:.1} dB vs dry {dry:.1} dB");
}

/// ... and a prepared string that picks up enough to be *heard* goes on
/// ringing after the note that excited it has stopped. The idle and culling
/// thresholds are a floor under the audible, not a gate on the gesture.
///
/// Measured on C3 alone, by subtracting a render in which C3 was never pressed
/// from one in which it was pressed silently — the only way to see one string's
/// contribution, since nothing else about the two renders differs.
///
/// The coupling is raised to the maximum a preset may ask for, and that is the
/// point of the test rather than a convenience: at the shipped coupling C3
/// picks up about -95 dBFS from a fortissimo G4, five decibels over the
/// engine's own -100 dBFS audibility floor (`types::IDLE_ENERGY`), so there is
/// nothing left to ring on with once the exciter stops — which is
/// `TUNING_REPORT.md`'s backlog item 5, a stage-2 *coupling level*, and not a
/// property of the thresholds. Raise the level and the persistence comes with
/// it, which is what this asserts.
#[test]
fn a_prepared_string_rings_on_after_the_note_that_excited_it() {
    const PREPARED: u8 = 48; // C3
    const EXCITER: u8 = 67; // G4, which lands on C3's third partial
    let mut preset = silent_mechanism(preset());
    preset.voicing.resonance_coupling = strongest_legal_coupling(&preset);
    preset.validate().expect("a strongly coupled instrument is a legal preset");

    let render = |prepare: bool| {
        let mut events = vec![
            RenderEvent::new(0.5, Event::NoteOn { key: EXCITER, vel: 120 }),
            RenderEvent::new(2.5, Event::NoteOff { key: EXCITER, vel: 64 }),
        ];
        if prepare {
            events.push(RenderEvent::new(0.0, Event::KeyDown { key: PREPARED }));
        }
        render_to_buffer(&preset, &events, 5.5)
    };
    let ((hl, hr), (cl, cr)) = (render(true), render(false));
    let alone: Vec<f32> = hl
        .iter()
        .zip(&cl)
        .zip(hr.iter().zip(&cr))
        .map(|((a, b), (c, d))| (a - b) + (c - d))
        .collect();

    let ringing = db(rms(window(&alone, 1.5, 2.5)));
    // A second and a half after the exciter's damper landed.
    let after = db(rms(window(&alone, 4.0, 5.0)));
    assert!(
        ringing > -120.0,
        "the prepared string picked up nothing at all ({ringing:.0} dBFS)"
    );
    assert!(
        after > ringing - 30.0,
        "the prepared string was at {ringing:.0} dBFS under the note and {after:.0} dBFS \
         a second and a half after it stopped: it was cut off, not left to decay"
    );
}

/// The other half of what the duplex segments are for: a key that is never
/// touched answers another key's note through the bridge, with its own strings
/// under their dampers the whole time.
///
/// C5's segments are tuned to C6's second partial — an aliquot's placement —
/// and C6 is struck *staccato* with the pedal up. C5's speaking length is
/// damped throughout and cannot ring at all, so whatever C5 contributes came
/// through the resonance bus, through the bridge admittance, into a bank that
/// has no damper; and because it has no damper it is still contributing a
/// second after C6's own damper landed.
///
/// The coupling is raised to the maximum a preset may ask for, for the same
/// reason `a_prepared_string_rings_on_after_the_note_that_excited_it` raises it:
/// the shipped level puts one key's sympathetic contribution near the engine's
/// own audibility floor, which is `TUNING_REPORT.md`'s backlog item 5 — a
/// stage-2 coupling level — and not a property of this path.
#[test]
fn a_duplex_segment_answers_another_keys_note_through_the_bridge() {
    use piano_emulator::engine::Engine;
    use piano_emulator::preset::{DuplexMode, MAX_DUPLEX_GAIN_DB};
    use piano_emulator::types::NUM_KEYS;

    const ANSWERS: u8 = 72; // C5, damped from the first sample to the last
    const STRUCK: u8 = 84; // C6

    let mut voiced = silent_mechanism(preset());
    // The base preset may already carry a duplex table — the measured one
    // carries a hundred segments — and the control has to differ from the
    // instrument under test by exactly the one segment this test adds, so
    // both sides start with none.
    voiced.notes.duplex = vec![Vec::new(); NUM_KEYS];
    let mut plain = voiced.clone();
    voiced.notes.duplex[key_index(ANSWERS).unwrap()] = vec![DuplexMode {
        hz: string_params(STRUCK).partial_freq(2),
        gain_db: MAX_DUPLEX_GAIN_DB,
        t60_s: 1.5,
    }];
    // The segment is part of the loop, so the ceiling is asked of the preset
    // that has it, and the control is given the same number.
    let coupling = strongest_legal_coupling(&voiced);
    voiced.voicing.resonance_coupling = coupling;
    plain.voicing.resonance_coupling = coupling;
    voiced.validate().expect("one segment on one key is a legal preset");
    plain.validate().expect("the control is a legal preset");

    let run = |preset: &Preset| {
        let (mut engine, _tx) = Engine::new(preset);
        engine.handle_event(Event::NoteOn {
            key: STRUCK,
            vel: 120,
        });
        let (mut l, mut r) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
        let mut out = Vec::new();
        let mut answered = 0.0f32;
        for b in 0..(1.6 * SAMPLE_RATE / BLOCK as f32) as usize {
            if b == (0.3 * SAMPLE_RATE / BLOCK as f32) as usize {
                engine.handle_event(Event::NoteOff {
                    key: STRUCK,
                    vel: 64,
                });
            }
            engine.process(&mut l, &mut r);
            out.extend(l.iter().zip(&r).map(|(a, b)| a + b));
            let voice = engine.voice(key_index(ANSWERS).unwrap());
            answered = answered.max(voice.duplex().energy());
            assert_eq!(voice.string().damper(), 1.0, "C5's damper came off");
            assert!(voice.string().energy() < 1.0e-12, "C5's string rang");
        }
        (answered, out)
    };

    let (answered, with) = run(&voiced);
    let (nothing, without) = run(&plain);
    assert_eq!(nothing, 0.0, "the control key has no segments to ring");
    assert!(
        answered > 0.0,
        "C5's segments picked up nothing from C6 at all"
    );

    // ... and it reaches the output. The difference between the two renders is
    // this one bank, since nothing else about the two presets differs.
    let alone: Vec<f32> = with.iter().zip(&without).map(|(a, b)| a - b).collect();
    let note = db(rms(window(&with, 0.05, 0.25)));
    let during = db(rms(window(&alone, 0.05, 0.25)));
    let after = db(rms(window(&alone, 0.9, 1.2)));
    assert!(
        during > note - 55.0,
        "C6 sounded at {note:.0} dBFS and C5's segments answered at {during:.0}"
    );
    // A second after C6's damper landed, C5's undamped segments are not merely
    // audible in what is left of the instrument — they *are* what is left. The
    // two measurements come out equal to a tenth of a decibel, having stood
    // 42 dB apart while C6 was sounding.
    let remaining = db(rms(window(&with, 0.9, 1.2)));
    assert!(
        after > remaining - 3.0,
        "a second after the note stopped the segments were at {after:.0} dBFS \
         against the whole instrument's {remaining:.0}"
    );
}

// ---------------------------------------------------------- 6. stability

#[test]
fn thirty_seconds_of_dense_playing_stays_safe() {
    let mut events = vec![RenderEvent::new(
        0.0,
        Event::Pedal(PedalEvent::Sustain(1.0)),
    )];
    // Deterministic pseudo-random playing across the whole compass, with the
    // pedal pumped so dampers keep re-engaging under a full instrument.
    let mut state = 0x9e37_79b9u32;
    for i in 0..1200 {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let key = 21 + (state >> 16) as u8 % 88;
        let vel = 20 + (state >> 8) as u8 % 107;
        let t = i as f32 * 0.024;
        events.push(RenderEvent::new(t, Event::NoteOn { key, vel }));
        events.push(RenderEvent::new(t + 0.35, Event::NoteOff { key, vel: 64 }));
        if i % 100 == 0 {
            let pedal = if (i / 100) % 2 == 0 { 1.0 } else { 0.0 };
            events.push(RenderEvent::new(t, Event::Pedal(PedalEvent::Sustain(pedal))));
        }
    }
    let (l, r) = render_to_buffer(&preset(), &events, 30.0);

    assert!(l.iter().chain(r.iter()).all(|v| v.is_finite()));
    assert!(peak(&l).max(peak(&r)) <= 1.0, "clipped past 0 dBFS");
    for (name, channel) in [("left", &l), ("right", &r)] {
        let dc = channel.iter().sum::<f32>() / channel.len() as f32;
        assert!(db(dc) < -60.0, "{name} DC {:.1} dBFS", db(dc));
    }

    // ... and an engine that was told nothing renders digital silence.
    let (sl, sr) = render_to_buffer(&preset(), &[], 2.0);
    assert!(sl.iter().chain(sr.iter()).all(|&v| v == 0.0));
}

/// The bridge admittance's stability contract, exercised rather than argued.
///
/// `Preset::validate` refuses any preset whose `resonance_coupling * max|B(f)|`
/// exceeds `MAX_BRIDGE_LOOP_GAIN`, and the derivation behind that bound is in
/// `resonance.rs`. This is the gate the derivation is worth nothing without:
/// the *most extreme preset the bound still admits* — forty stacked resonances
/// and a backbone that swings 34 dB, with the coupling then set to put the
/// effective loop gain exactly on the ceiling — playing a minute of dense
/// pedal-down music with every damper up and every string driven.
///
/// A minute and not thirty seconds because a marginally unstable loop grows
/// exponentially but slowly: at the T60s in the bass an instability with a
/// doubling time of several seconds is invisible in a short render and
/// deafening in a long one. Release only, because a minute of 88-voice
/// polyphony through a hundred-section filter is 20-30x slower unoptimised.
#[test]
#[cfg_attr(debug_assertions, ignore = "a minute of 88 voices needs --release")]
fn the_most_extreme_bridge_a_preset_may_ask_for_stays_bounded_for_a_minute() {
    use piano_emulator::preset::{BridgeAnchor, BridgePeak, BridgeVoicing};
    use piano_emulator::resonance::{BridgeFilter, MAX_BRIDGE_LOOP_GAIN, MAX_COUPLING};

    // A backbone that swings the full width of the schema, and forty
    // resonances spread over the bridge's modal region, alternating sign so
    // they are separate modes rather than one stack (a stack is refused, which
    // `resonance::tests::stacked_peaks_are_refused_by_the_loop_gain_check`
    // covers). Sharp: half of them at the schema's `Q` ceiling.
    let bridge = BridgeVoicing {
        backbone: [
            (20.0, -18.0),
            (60.0, 4.0),
            (160.0, 12.0),
            (400.0, -6.0),
            (1_100.0, 8.0),
            (2_600.0, -12.0),
            (6_000.0, 6.0),
            (16_000.0, -22.0),
        ]
        .into_iter()
        .map(|(hz, gain_db)| BridgeAnchor { hz, gain_db })
        .collect(),
        peaks: (0..40)
            .map(|i| BridgePeak {
                // 23 Hz to 15 kHz, geometrically — so a peak lands on or very
                // near a partial of nearly every key in the compass.
                hz: 23.0 * (15_000.0f32 / 23.0).powf(i as f32 / 39.0),
                q: if i % 2 == 0 { 50.0 } else { 3.0 },
                gain_db: if i % 3 == 0 { -16.0 } else { 14.0 },
            })
            .collect(),
        // ... and the largest share of the strings' own damping the schema
        // will hand to those resonances, so the render also exercises the
        // partials whose decay the admittance has slowed by the clamp's full
        // factor of four (`string::RADIATED_FACTOR_RANGE`) — the longest-ringing
        // strings this schema can describe, under the densest playing.
        radiated_share: piano_emulator::preset::MAX_RADIATED_SHARE,
    };

    // Put the loop exactly on the ceiling: whatever this filter's realised
    // maximum turns out to be, the coupling is the largest one the validator
    // will accept with it. That is the worst case the contract permits, and it
    // is derived from the filter rather than guessed at.
    let max_b = BridgeFilter::new(&bridge).max_magnitude();
    let mut preset = preset();
    preset.voicing.bridge = Some(bridge);
    preset.voicing.resonance_coupling = (MAX_BRIDGE_LOOP_GAIN / max_b).min(MAX_COUPLING);
    preset
        .validate()
        .expect("the extreme bridge must still be a legal preset");
    let effective = preset.voicing.resonance_coupling * max_b;
    println!(
        "bridge peaks at {:.1} dB; coupling {:.4} puts the loop at {effective:.3}",
        db(max_b),
        preset.voicing.resonance_coupling
    );
    assert!(effective > 0.9 * MAX_BRIDGE_LOOP_GAIN.min(MAX_COUPLING * max_b));

    // Dense pedal-down playing: the pedal never comes up, so every one of the
    // 88 strings is undamped and driven for the whole minute, and the notes
    // are never released, so nothing is ever culled.
    let mut events = vec![RenderEvent::new(
        0.0,
        Event::Pedal(PedalEvent::Sustain(1.0)),
    )];
    let mut state = 0x2545_f491u32;
    // 2400 notes over the first 48 seconds — 50 a second — which leaves the
    // last twelve as decay. The first version of this test spaced them 25 ms
    // apart, which put the last note at exactly 60.0 s and left the "after the
    // playing stopped" window still being played into: it passed on the
    // default preset by six hundredths of a decibel and failed on the measured
    // one by eight, neither of which was about stability.
    for i in 0..2400 {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let key = 21 + (state >> 16) as u8 % 88;
        let vel = 40 + (state >> 8) as u8 % 87;
        events.push(RenderEvent::new(i as f32 * 0.02, Event::NoteOn { key, vel }));
    }
    let (l, r) = render_to_buffer(&preset, &events, 60.0);

    assert!(
        l.iter().chain(r.iter()).all(|v| v.is_finite()),
        "the bus diverged to a non-finite sample"
    );
    assert!(peak(&l).max(peak(&r)) <= 1.0, "clipped past 0 dBFS");
    assert!(rms(&l) > 1.0e-3, "the extreme bridge made no sound at all");
    // Bounded is not enough on its own — a loop sitting just under unity would
    // ride the limiter and come out as a minute of full-scale mush. The last
    // five seconds are seven seconds of decay past the last note, so they have
    // to be far *quieter* than the playing that fed them: a stable instrument
    // is 15 dB down by then and this one is, on both presets.
    let late = db(rms(window(&l, 55.0, 60.0)));
    let during = db(rms(window(&l, 20.0, 25.0)));
    assert!(
        late < during - 10.0,
        "seven seconds after the playing stopped the instrument is at {late:.1} dB \
         against {during:.1} dB during it: the loop is feeding itself"
    );
}

// ------------------------------------------------------- 7. the mechanism

/// The peak of a whole render, as a level in dB.
fn peak_db(events: &[RenderEvent], duration_s: f32) -> f32 {
    db(peak(&render_mono(events, duration_s)))
}

/// Stereo magnitude `sqrt(l^2 + r^2)`, which an equal-power pan preserves — so
/// two sounds at different pan positions can be compared by level.
fn magnitude(left: &[f32], right: &[f32]) -> Vec<f32> {
    left.iter()
        .zip(right)
        .map(|(&l, &r)| (l * l + r * r).sqrt())
        .collect()
}

/// Peak of the stereo magnitude of a whole render, as an amplitude.
fn magnitude_peak(events: &[RenderEvent], duration_s: f32) -> f32 {
    let (l, r) = render_to_buffer(&preset(), events, duration_s);
    peak(&magnitude(&l, &r))
}

fn range(from_s: f32, to_s: f32) -> std::ops::Range<usize> {
    (from_s * SAMPLE_RATE) as usize..(to_s * SAMPLE_RATE) as usize
}

/// `TUNING_REPORT.md` §5, the second table: a key-off plays at -25 to -39 dB
/// relative to a velocity-90 strike of the same key, and the engine made no
/// sound at all. This is that column, measured back out of the finished chain.
///
/// What it is asserted against is the level the *preset* asks for, which for
/// the shipped instrument is §5's own measured column and for an estimated
/// preset is that piano's. That makes this a claim about the engine — a burst
/// triggered at a level of `x` dB has to arrive at the ear `x` dB under the
/// strike — rather than about which piano is loaded, and it is the claim that
/// `calibrate.rs` exists to make true: the levels are quoted against a strike
/// at the *output*, so the reference is measured through the board rather than
/// assumed at its input.
///
/// Sixteen releases per key, averaged, because the peak of one realization of a
/// short noise band is itself a random number and scatters by 2-3 dB
/// (`DECISIONS.md` 114) — the *design* level is what is under test, and it is
/// the mean of many events. The statistic is the peak of the stereo magnitude,
/// which an equal-power pan preserves, so a key at the edge of the stage is
/// compared with its own strike on equal terms.
#[test]
fn a_note_off_thumps_at_the_level_the_recordings_measured() {
    use piano_emulator::types::{interp_anchors, key_position};

    // key, and what `rel1`/`rel37`/`rel40`/`rel52`/`rel76` measured for it —
    // the five anchors `presets/default.toml` is written from.
    const MEASURED: [(u8, f32); 5] = [
        (21, -37.3),
        (57, -30.2),
        (60, -35.4),
        (72, -25.4),
        (96, -33.5),
    ];
    let preset = preset();
    let anchors: Vec<(f32, f32)> = preset
        .noise
        .key_off
        .level_db
        .iter()
        .map(|a| (key_position(a.key), a.db))
        .collect();

    let mut total = 0.0f32;
    for (key, table) in MEASURED {
        let strike = magnitude_peak(&[RenderEvent::new(0.0, Event::NoteOn { key, vel: 90 })], 1.0);
        // Releases of a key that is not down: the thump on its own, with no
        // string sound anywhere near it. Spaced a second apart, which is four
        // envelope lifetimes of the longest key-off in the table.
        const RELEASES: usize = 16;
        let releases: Vec<RenderEvent> = (0..RELEASES)
            .map(|i| RenderEvent::new(i as f32, Event::NoteOff { key, vel: 64 }))
            .collect();
        let (l, r) = render_to_buffer(&preset, &releases, RELEASES as f32);
        let magnitude = magnitude(&l, &r);
        let thump: f32 = (0..RELEASES)
            .map(|i| peak(&magnitude[range(i as f32, i as f32 + 1.0)]) / RELEASES as f32)
            .sum();
        assert!(
            db(thump) > -80.0,
            "key {key}: the note-off made no sound ({:.1} dBFS)",
            db(thump)
        );
        let measured = db(thump) - db(strike);
        let asked = interp_anchors(key_position(key), &anchors);
        // Printed as well as asserted: how far the render sits from the level
        // it was asked for is the number to look at when the board or the gain
        // staging moves.
        println!("key {key:>3}: rendered {measured:>6.1} dB, preset asks {asked:>6.1}, table {table:>6.1}");
        // Two decibels per key and 1.2 across the compass, against the six and
        // 2.5 this test allowed while the reference was a constant. What is
        // left is the sampling error of the eight realizations `calibrate.rs`
        // measures the board's peak gain from — the mean over the compass sits
        // about 0.8 dB under the ask because those eight sit that far above the
        // population — against the 2-3 dB by which each individual event is
        // *meant* to scatter (`DECISIONS.md` 114).
        assert!(
            (measured - asked).abs() < 2.0,
            "key {key}: the note-off renders at {measured:.1} dB re a strike where the \
             preset asks for {asked:.1} (the recordings say {table:.1})"
        );
        total += measured - asked;
    }
    let mean = total / MEASURED.len() as f32;
    assert!(
        mean.abs() < 1.2,
        "the key-off level is {mean:+.1} dB off what the preset asks across the compass"
    );
}

/// `pedalD1` and `pedalU1`: -35.8 dB with a six-second 70 Hz rumble going down,
/// -42.4 dB over 0.3 s coming up. The engine used to move the dampers in
/// silence.
#[test]
fn the_pedal_makes_a_sound_going_down_and_coming_up() {
    let strike = peak_db(
        &[RenderEvent::new(0.0, Event::NoteOn { key: 60, vel: 90 })],
        1.0,
    );

    let down = peak_db(
        &[RenderEvent::new(
            0.0,
            Event::Pedal(PedalEvent::Sustain(1.0)),
        )],
        1.0,
    );
    assert!(
        (down - strike + 35.8).abs() < 4.0,
        "pedal down is {:.1} dB re a strike, recordings say -35.8",
        down - strike
    );

    // Released long after the down rumble has gone, so what is measured is the
    // release and not the tail of the press.
    let events = [
        RenderEvent::new(0.0, Event::Pedal(PedalEvent::Sustain(1.0))),
        RenderEvent::new(8.0, Event::Pedal(PedalEvent::Sustain(0.0))),
    ];
    let y = render_mono(&events, 10.0);
    let up = db(peak(window(&y, 8.0, 10.0)));
    assert!(
        (up - strike + 42.4).abs() < 4.0,
        "pedal up is {:.1} dB re a strike, recordings say -42.4",
        up - strike
    );

    // ... and the rumble really is a rumble: seconds long and low.
    let rumble = render_mono(
        &[RenderEvent::new(
            0.0,
            Event::Pedal(PedalEvent::Sustain(1.0)),
        )],
        6.0,
    );
    let late = db(rms(window(&rumble, 4.0, 5.0)));
    let early = db(rms(window(&rumble, 0.1, 1.1)));
    assert!(
        late > early - 40.0,
        "the pedal-down rumble is {:.1} dB down after four seconds, expected a six-second decay",
        late - early
    );
}

/// A pedal press that lifts nothing is nearly silent: the sound is the damper
/// rail moving, so a chord held by the keys leaves it almost nothing to move.
#[test]
fn the_pedal_is_quiet_when_the_keys_already_hold_the_dampers() {
    let mut everything = vec![];
    for key in 21u8..=90 {
        everything.push(RenderEvent::new(0.0, Event::KeyDown { key }));
    }
    everything.push(RenderEvent::new(0.5, Event::Pedal(PedalEvent::Sustain(1.0))));
    let held = render_mono(&everything, 2.0);
    let free = render_mono(
        &[RenderEvent::new(
            0.5,
            Event::Pedal(PedalEvent::Sustain(1.0)),
        )],
        2.0,
    );
    let (a, b) = (
        db(peak(window(&held, 0.5, 2.0))),
        db(peak(window(&free, 0.5, 2.0))),
    );
    assert!(
        a < b - 4.0,
        "the pedal was as loud over a fully held keyboard ({a:.1} dB) as over an empty one ({b:.1})"
    );
}

/// `PHYSICS.md` §6's first acceptance test: a key pressed too gently to reach
/// escapement lifts its damper and nothing else, so that string answers a note
/// struck elsewhere **with the pedal up** — the whole point of the gesture, and
/// the thing the engine could not do at all.
#[test]
fn a_silently_held_key_answers_a_struck_note_with_the_pedal_up() {
    use piano_emulator::engine::Engine;

    let energy_of_c3 = |prepare: Option<Event>| {
        let (mut engine, _tx) = Engine::new(&preset());
        if let Some(event) = prepare {
            engine.handle_event(event);
        }
        engine.handle_event(Event::NoteOn { key: 67, vel: 120 });
        let (mut l, mut r) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
        let blocks = (1.5 * SAMPLE_RATE / BLOCK as f32) as usize;
        for b in 0..blocks {
            if b == (0.2 * SAMPLE_RATE / BLOCK as f32) as usize {
                engine.handle_event(Event::NoteOff { key: 67, vel: 64 });
            }
            engine.process(&mut l, &mut r);
        }
        engine.voice(key_index(48).unwrap()).string().energy()
    };

    let silent_press = energy_of_c3(Some(Event::KeyDown { key: 48 }));
    // The same gesture written as a velocity-zero note-on.
    let too_soft_to_sound = energy_of_c3(Some(Event::NoteOn { key: 48, vel: 0 }));
    let nothing = energy_of_c3(None);

    assert!(
        silent_press > nothing * 100.0,
        "the silently held C3 picked up {silent_press:e} against {nothing:e} for a key at rest"
    );
    assert!(
        too_soft_to_sound > nothing * 100.0,
        "a velocity-zero note-on did not prepare the string"
    );
    // ... and it really was silent: preparing C3 must not have struck it. Its
    // own answer is orders of magnitude below what the hammer would put there.
    let struck = {
        let (mut engine, _tx) = Engine::new(&preset());
        engine.handle_event(Event::NoteOn { key: 48, vel: 0 });
        let (mut l, mut r) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
        engine.process(&mut l, &mut r);
        engine.voice(key_index(48).unwrap()).string().energy()
    };
    assert_eq!(struck, 0.0, "a silent press struck the string");
    // ... while velocity 1 — a genuine pianissimo note in a recorded
    // performance — must strike: no sounding velocity is a silent press.
    let quietest = {
        let (mut engine, _tx) = Engine::new(&preset());
        engine.handle_event(Event::NoteOn { key: 48, vel: 1 });
        let (mut l, mut r) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
        engine.process(&mut l, &mut r);
        engine.voice(key_index(48).unwrap()).string().energy()
    };
    assert!(quietest > 0.0, "a velocity-1 note-on was swallowed as a silent press");
}

/// `PHYSICS.md` §6's second: how fast the key comes back sets how fast the
/// damper falls, so a note let go slowly rings on where one dropped stops.
///
/// Measured on an instrument whose mechanism is silenced, because release
/// velocity reaches *two* things — the damper's ramp and the key-off thump —
/// and this is the claim about the damper. The thump's own velocity law is
/// `noise::tests::release_velocity_moves_the_level_and_the_nominal_one_hits_
/// the_table`.
#[test]
fn a_slow_release_rings_on_where_a_fast_one_stops() {
    let preset = silent_mechanism(preset());
    let after_release = |release_vel: u8| {
        let events = [
            RenderEvent::new(0.0, Event::NoteOn { key: 48, vel: 100 }),
            RenderEvent::new(
                1.0,
                Event::NoteOff {
                    key: 48,
                    vel: release_vel,
                },
            ),
        ];
        let (l, r) = render_to_buffer(&preset, &events, 1.6);
        let y: Vec<f32> = l.iter().zip(&r).map(|(a, b)| a + b).collect();
        db(rms(window(&y, 1.15, 1.30)))
    };
    let slow = after_release(1);
    let nominal = after_release(64);
    let fast = after_release(127);
    assert!(
        slow > nominal + 2.0 && nominal >= fast,
        "release velocity did not reach the damper: {slow:.1} / {nominal:.1} / {fast:.1} dB"
    );
}

/// `PHYSICS.md` §6's third: a damper that is touching but not seated limits the
/// string's deflection nonlinearly, so a half-pedalled note is not the same
/// note with a bigger `σ`.
///
/// The control is exact rather than approximate. Half-pedalling multiplies the
/// damper's extra decay rate by the pedal position, so an instrument whose
/// `damper_sigma` is halved and whose damper is *fully* down has, partial by
/// partial, exactly the damping of the first instrument at half pedal — and no
/// partly-engaged damper anywhere, so no felt in contact. Same note, same
/// linear damping: the only thing left between them is the nonlinearity.
///
/// Both are rendered with the mechanism silent and the sympathetic bus
/// uncoupled, because the pedal move that produces the half-pedal also makes a
/// tray noise and lifts the whole instrument's dampers, and either of those
/// would answer the question instead of the felt.
#[test]
fn a_half_pedalled_note_is_not_merely_a_faster_decay() {
    const KEY: u8 = 48;
    let render = |half: bool| {
        let mut preset = silent_mechanism(preset());
        preset.voicing.resonance_coupling = 0.0;
        let sustain = if half {
            0.5
        } else {
            for sigma in &mut preset.notes.damper_sigma {
                *sigma *= 0.5;
            }
            0.0
        };
        let events = [
            RenderEvent::new(0.0, Event::NoteOn { key: KEY, vel: 120 }),
            RenderEvent::new(1.0, Event::Pedal(PedalEvent::Sustain(sustain))),
            RenderEvent::new(1.0, Event::NoteOff { key: KEY, vel: 64 }),
        ];
        let (l, r) = render_to_buffer(&preset, &events, 2.0);
        l.iter().zip(&r).map(|(a, b)| a + b).collect::<Vec<f32>>()
    };
    let (half, linear) = (render(true), render(false));

    // The felt in contact takes energy out as well as colouring what is left,
    // but the two notes have to stay comparable or "the spectrum differs" would
    // only be saying that one of them is quieter.
    let level = |y: &[f32]| db(rms(window(y, 1.0, 1.05)));
    assert!(
        (level(&half) - level(&linear)).abs() < 6.0,
        "the control is not level-matched: {:.1} against {:.1} dB",
        level(&half),
        level(&linear)
    );

    // What a soft limit does that a decay rate cannot: fold the waveform, and
    // put energy where the note's own partials are not. Read as the balance
    // between the top of the spectrum and the fundamental region — which a
    // linear damper can only move the *other* way, because the felt's
    // frequency response grips low partials hardest.
    const FFT: usize = 1 << 15;
    let colour = |y: &[f32]| {
        let mag = spectrum(window(y, 1.0, 1.05), FFT);
        let bin = |f: f32| (f * FFT as f32 / SAMPLE_RATE) as usize;
        let band = |lo: f32, hi: f32| {
            mag[bin(lo)..bin(hi).min(mag.len())]
                .iter()
                .map(|m| m * m)
                .sum::<f32>()
        };
        db(band(1_500.0, 6_000.0).sqrt() / band(100.0, 400.0).sqrt())
    };
    let (bright, plain) = (colour(&half), colour(&linear));
    assert!(
        bright > plain + 3.0,
        "half pedal is only {:.1} dB brighter than the same damping applied linearly \
         ({bright:.1} against {plain:.1}); the felt is doing nothing a bigger sigma \
         could not",
        bright - plain
    );
}

/// ... and the felt limits the string only while it is *arriving*: a key struck
/// again after it was released must have the attack it has from silence, and it
/// must not depend on how loud the note before it was.
///
/// The engine starts the damper's lift and the hammer's blow at the same
/// instant, so for the first ~10 ms of every re-strike the damper is between
/// the string and its rest position; the real action lifts the damper early in
/// the key's travel and has it clear before the hammer arrives. When the
/// limiter did not test the damper's *direction* it spent those two blocks
/// clamping the new attack against the level the *previous* note had when its
/// damper landed — 27 to 68 dB of choke on the first two blocks here, and 23 dB
/// of it purely historical. Measured on the attack transient itself, in 3 ms
/// windows: nothing else in the engine has that time constant, and a whole-note
/// level would average the effect away.
///
/// The mechanism is silenced because a key-off thump and a damper-lift burst
/// both land inside the windows this reads.
#[test]
fn a_restrike_attacks_like_a_strike_from_silence() {
    const KEY: u8 = 60;
    const AT: f32 = 3.0;
    let preset = silent_mechanism(preset());
    let attack = |previous: Option<u8>| -> Vec<f32> {
        let mut events = Vec::new();
        if let Some(vel) = previous {
            events.push(RenderEvent::new(0.0, Event::NoteOn { key: KEY, vel }));
            events.push(RenderEvent::new(1.0, Event::NoteOff { key: KEY, vel: 64 }));
        }
        events.push(RenderEvent::new(AT, Event::NoteOn { key: KEY, vel: 120 }));
        let (l, r) = render_to_buffer(&preset, &events, AT + 0.05);
        let mono: Vec<f32> = l.iter().zip(&r).map(|(a, b)| a + b).collect();
        [(0.0, 0.003), (0.003, 0.006), (0.006, 0.010), (0.010, 0.020)]
            .iter()
            .map(|&(a, b)| db(rms(window(&mono, AT + a, AT + b))))
            .collect()
    };
    let clean = attack(None);
    let after_soft = attack(Some(30));
    let after_loud = attack(Some(120));
    // Nothing of the earlier note is left to add to the new one: whatever the
    // windows below show is the new strike, not a sum.
    let (l, r) = render_to_buffer(
        &preset,
        &[
            RenderEvent::new(0.0, Event::NoteOn { key: KEY, vel: 120 }),
            RenderEvent::new(1.0, Event::NoteOff { key: KEY, vel: 64 }),
        ],
        AT,
    );
    let residual: Vec<f32> = l.iter().zip(&r).map(|(a, b)| a + b).collect();
    assert!(
        db(rms(window(&residual, AT - 0.1, AT))) < db(rms(window(&residual, 0.0, 0.02))) - 100.0,
        "the released note is still ringing where the re-strike is measured"
    );

    for (i, (&c, (&soft, &loud))) in clean
        .iter()
        .zip(after_soft.iter().zip(&after_loud))
        .enumerate()
    {
        assert!(
            (soft - c).abs() < 1.0 && (loud - c).abs() < 1.0,
            "window {i}: the re-strike attacks at {soft:.1} / {loud:.1} dB against \
             {c:.1} dB from silence"
        );
        assert!(
            (soft - loud).abs() < 0.5,
            "window {i}: the re-strike's attack depends on the previous note's level \
             ({soft:.1} dB after a pp note, {loud:.1} after a ff one)"
        );
    }
}

/// ... and the felt arrives, lands and lets go without a click.
///
/// The damper's position advances once per block, and the felt's threshold is
/// exponential in it, so anything that reads that position and holds it for the
/// block is a staircase in the *output*: 13.3 dB of threshold at every boundary
/// of a nominal release, plus a larger step wherever the limiter switched off.
/// Measured on the shipped presets before this was fixed, every note-off of
/// every key carried four single-sample jumps at four consecutive block
/// boundaries, the worst of them 12 to 14 times the local slope and 17 to 23 dB
/// under the note's own peak — inside the shipped benchmark renders, and
/// nowhere in the recordings they are compared against (`DECISIONS.md` 218).
///
/// The assertion is scale-free on purpose. A step is only a click if it is
/// *isolated*: a jump of a given size in the middle of a loud attack is the
/// waveform, and the same jump in a decayed tail is a transient nothing in the
/// physics produced. So each block boundary is measured against the RMS
/// single-sample step of the block either side of it, which is what the signal
/// is doing anyway, and a boundary is allowed to be no more than five times
/// that. Nothing in the model is synchronised to the block grid, so a boundary
/// that stands out from its own neighbourhood is an artefact of the block loop
/// by construction.
///
/// The action is silenced because a key-off thump is a broadband burst that
/// starts at the note-off — a step by design, and this test is about the string.
#[test]
fn a_release_does_not_click_at_the_block_boundary() {
    const AFTER_S: f32 = 0.04;
    const ALLOWED: f32 = 5.0;
    let preset = silent_mechanism(preset());
    let off_s = 0.5;
    for key in [33u8, 45, 60, 72] {
        for vel in [12u8, 90] {
            let events = [
                RenderEvent::new(0.1, Event::NoteOn { key, vel }),
                RenderEvent::new(off_s, Event::NoteOff { key, vel: 64 }),
            ];
            let (l, r) = render_to_buffer(&preset, &events, 1.5);
            let mono: Vec<f32> = l.iter().zip(&r).map(|(a, b)| 0.5 * (a + b)).collect();
            // There has to be a note there to click: a silent window would pass
            // this test on an engine that had stopped making sound at all.
            let before = rms(window(&mono, off_s - 0.05, off_s));
            let during = rms(window(&mono, off_s, off_s + AFTER_S));
            assert!(
                during > 0.1 * before && before > 1.0e-5,
                "key {key} vel {vel}: nothing is ringing across the release \
                 ({before:.2e} before, {during:.2e} after)"
            );

            let first = (off_s * SAMPLE_RATE) as usize / BLOCK;
            let last = ((off_s + AFTER_S) * SAMPLE_RATE) as usize / BLOCK;
            for block in first..=last {
                let i = block * BLOCK - 1;
                let step = (mono[i + 1] - mono[i]).abs();
                let mut sum = 0.0f64;
                for j in (i - BLOCK)..(i + BLOCK) {
                    sum += f64::from(mono[j + 1] - mono[j]).powi(2);
                }
                let local = (sum / (2 * BLOCK) as f64).sqrt() as f32;
                assert!(
                    step <= ALLOWED * local,
                    "key {key} vel {vel}: the block boundary {:.1} ms after the \
                     note-off jumps {:.1}x the local slope ({step:.3e} against \
                     {local:.3e} rms), which is a click and not a note",
                    (i as f32 - off_s * SAMPLE_RATE) * 1000.0 / SAMPLE_RATE,
                    step / local,
                );
            }
        }
    }
}

// -------------------------------------------------------- 8. performance

/// The spec's worst case: sustain pedal down and a glissando that leaves all 88
/// keys ringing, so every string is live and every undamped string is also
/// being driven by the resonance bus.
///
/// Debug builds are 20-30x slower than release, so this only means anything
/// with optimizations on; `cargo test --release` is where it runs.
#[test]
#[cfg_attr(debug_assertions, ignore = "timing is only meaningful in --release")]
fn the_worst_case_fits_the_performance_budget() {
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

    let start = std::time::Instant::now();
    let (l, r) = render_to_buffer(&preset(), &events, AUDIO_S);
    let ratio = start.elapsed().as_secs_f32() / AUDIO_S;

    assert!(l.iter().chain(r.iter()).all(|v| v.is_finite()));
    assert!(rms(&l) > 1.0e-3, "the glissando made no sound");
    // Printed rather than only asserted: how much of the budget is left is the
    // number that matters when a preset changes how long the notes ring.
    println!("worst case: {:.1}% of one core", 100.0 * ratio);
    assert!(
        ratio < 0.8,
        "worst case took {:.1}% of one core (design goal 50%)",
        100.0 * ratio
    );
}

/// The same worst case with the most duplex the schema allows — six segments on
/// every one of the 88 keys, none of which any damper can ever stop — measured
/// against the same run without them so the difference is the segments and not
/// the machine.
///
/// Two costs are being measured at once and both are the point. The segments
/// themselves are 528 resonators, which is small beside the ~14 000 string
/// partials of a full keyboard. What could have been large is the *waking*: a
/// bank that is never damped keeps its voice out of the branch that renders
/// nothing, so an instrument whose segments all ring is an instrument whose 88
/// voices all run. `Voice::process` splits the two decisions — the strings and
/// the segments live or sleep separately — and this is the measurement that
/// says the split works.
#[test]
#[cfg_attr(debug_assertions, ignore = "timing is only meaningful in --release")]
fn the_worst_case_with_a_duplex_on_every_key_fits_the_performance_budget() {
    use piano_emulator::preset::{DuplexMode, MAX_DUPLEX_MODES};
    use piano_emulator::types::{index_to_note, NUM_KEYS};

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

    // Six segments per key, spread over the band a duplex occupies and rising
    // across the compass, at a level and a decay in the middle of the schema.
    let mut voiced = preset();
    voiced.notes.duplex = (0..NUM_KEYS)
        .map(|i| {
            let position = i as f32 / (NUM_KEYS - 1) as f32;
            (0..MAX_DUPLEX_MODES)
                .map(|k| DuplexMode {
                    hz: 1_500.0
                        * 2.0f32.powf(1.5 * position + k as f32 * 0.31 + i as f32 * 0.0037),
                    gain_db: -20.0,
                    t60_s: 1.5,
                })
                .collect()
        })
        .collect();
    voiced
        .validate()
        .expect("six scattered segments on every key is a legal preset");
    assert_eq!(voiced.duplex_modes(index_to_note(0)).len(), MAX_DUPLEX_MODES);

    let measure = |preset: &Preset| {
        // Best of three: the number is a floor on the cost, not an average of
        // whatever else the machine was doing.
        let mut best = f32::INFINITY;
        for _ in 0..3 {
            let start = std::time::Instant::now();
            let (l, r) = render_to_buffer(preset, &events, AUDIO_S);
            let ratio = start.elapsed().as_secs_f32() / AUDIO_S;
            assert!(l.iter().chain(r.iter()).all(|v| v.is_finite()));
            assert!(peak(&l).max(peak(&r)) <= 1.0, "clipped past 0 dBFS");
            assert!(rms(&l) > 1.0e-3, "the glissando made no sound");
            best = best.min(ratio);
        }
        best
    };
    let bare = measure(&preset());
    let with_duplex = measure(&voiced);
    println!(
        "worst case: {:.1}% of one core without segments, {:.1}% with six on every key \
         (+{:.2} points)",
        100.0 * bare,
        100.0 * with_duplex,
        100.0 * (with_duplex - bare)
    );
    assert!(
        with_duplex < 0.5,
        "the worst case with a full duplex took {:.1}% of one core (design goal 50%)",
        100.0 * with_duplex
    );

    // And the case the waking would show up in most sharply: one note, held,
    // with every other key's dampers down. Nothing but the bus can reach the
    // other 87 voices, and it reaches their segments whether or not their
    // strings are damped — so this is 87 banks running for one note. It has to
    // stay a rounding error beside the worst case above, or a preset with
    // segments would make sparse playing cost what a glissando does.
    let sparse = vec![RenderEvent::new(0.0, Event::NoteOn { key: 60, vel: 110 })];
    let sparse_measure = |preset: &Preset| {
        let mut best = f32::INFINITY;
        for _ in 0..3 {
            let start = std::time::Instant::now();
            let (l, _) = render_to_buffer(preset, &sparse, AUDIO_S);
            best = best.min(start.elapsed().as_secs_f32() / AUDIO_S);
            assert!(l.iter().all(|v| v.is_finite()));
        }
        best
    };
    let (one_bare, one_voiced) = (sparse_measure(&preset()), sparse_measure(&voiced));
    println!(
        "one held note: {:.2}% of one core without segments, {:.2}% with them",
        100.0 * one_bare,
        100.0 * one_voiced
    );
    assert!(
        one_voiced < bare / 4.0,
        "one note with a full duplex cost {:.1}% of one core against a whole \
         keyboard's {:.1}%",
        100.0 * one_voiced,
        100.0 * bare
    );
}

/// The same worst case on the instrument that actually ships with a duplex:
/// `presets/salamander-c5.toml`, whose 100 measured segments over 23 keys are
/// never damped, and whose bridge filter runs on every block.
///
/// The test above bounds the schema's ceiling with a synthetic table; this one
/// bounds the file a user will really load, which is a different question and
/// the one the design goal is written about. It is also the only performance
/// measurement that has the fitted admittance in it — a voiced bridge wakes
/// voices a flat bus lets sleep (`DECISIONS.md` 152), so the measured preset
/// costs more than the default even before the segments are counted.
#[test]
#[cfg_attr(debug_assertions, ignore = "timing is only meaningful in --release")]
fn the_worst_case_with_the_measured_presets_duplex_fits_the_performance_budget() {
    const AUDIO_S: f32 = 10.0;
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../presets/salamander-c5.toml");
    let measured = Preset::load(&path).expect("presets/salamander-c5.toml loads");
    let segments: usize = (0..88)
        .map(|i| measured.duplex_modes(piano_emulator::types::index_to_note(i)).len())
        .sum();
    assert!(
        segments > 0,
        "the measured preset has no duplex table, so this measures nothing"
    );

    let mut events = vec![RenderEvent::new(
        0.0,
        Event::Pedal(PedalEvent::Sustain(1.0)),
    )];
    for (i, key) in (21u8..=108).enumerate() {
        let t = i as f32 * 0.02;
        events.push(RenderEvent::new(t, Event::NoteOn { key, vel: 90 }));
        events.push(RenderEvent::new(t + 0.1, Event::NoteOff { key, vel: 64 }));
    }

    // Best of three: a floor on the cost rather than an average of whatever
    // else the machine was doing.
    let mut best = f32::INFINITY;
    for _ in 0..3 {
        let start = std::time::Instant::now();
        let (l, r) = render_to_buffer(&measured, &events, AUDIO_S);
        let ratio = start.elapsed().as_secs_f32() / AUDIO_S;
        assert!(l.iter().chain(r.iter()).all(|v| v.is_finite()));
        assert!(peak(&l).max(peak(&r)) <= 1.0, "clipped past 0 dBFS");
        assert!(rms(&l) > 1.0e-3, "the glissando made no sound");
        best = best.min(ratio);
    }
    println!(
        "worst case on presets/salamander-c5.toml ({segments} duplex segments, voiced bridge): \
         {:.1}% of one core",
        100.0 * best
    );
    assert!(
        best < 0.5,
        "the worst case on the measured preset took {:.1}% of one core (design goal 50%)",
        100.0 * best
    );
}

/// The worst case with **both** noise paths running: every string ringing under
/// the pedal, and a note starting or stopping every ten milliseconds so that
/// the hammer's burst and the action's burst are alive on many voices at once.
///
/// The strike noise put a second `Burst` on every voice — deliberately, since a
/// staccato is a release 80 ms into an attack and one burst cannot be both — so
/// what this measures is the cost of the pair. The glissando alone would not
/// show it: 88 note-ons over 1.8 s leave the bursts idle for most of the render,
/// while the material below keeps a thousand events inside ten seconds on top of
/// a full keyboard of ringing strings.
#[test]
#[cfg_attr(debug_assertions, ignore = "timing is only meaningful in --release")]
fn the_worst_case_with_the_strike_noise_fits_the_performance_budget() {
    const AUDIO_S: f32 = 10.0;

    // A preset whose hammer noise is voiced at the level the measurements ask
    // for, at the top of the band limit the schema allows — the most expensive
    // legal strike there is.
    let mut voiced = preset();
    voiced.noise.strike.centroid_hz = 2_000.0;
    voiced.noise.strike.bandwidth_hz = 8_000.0;
    voiced.noise.strike.decay_s = 0.3;
    voiced.noise.strike.level_db = vec![piano_emulator::preset::NoiseAnchor {
        key: 21,
        db: -12.0,
    }];
    voiced
        .validate()
        .expect("a voiced hammer noise is a legal preset");

    let mut events = vec![RenderEvent::new(
        0.0,
        Event::Pedal(PedalEvent::Sustain(1.0)),
    )];
    for (i, key) in (21u8..=108).enumerate() {
        let t = i as f32 * 0.02;
        events.push(RenderEvent::new(t, Event::NoteOn { key, vel: 90 }));
        events.push(RenderEvent::new(t + 0.1, Event::NoteOff { key, vel: 64 }));
    }
    // ... and then a note every 10 ms for the rest of the render, so that both
    // bursts are running on a large fraction of the 88 voices at every instant.
    let mut key = 21u8;
    let mut t = 2.0f32;
    while t < AUDIO_S {
        events.push(RenderEvent::new(t, Event::NoteOn { key, vel: 100 }));
        events.push(RenderEvent::new(t + 0.02, Event::NoteOff { key, vel: 100 }));
        key = if key >= 108 { 21 } else { key + 7 };
        t += 0.01;
    }

    let measure = |preset: &Preset| {
        // Best of three: a floor on the cost rather than an average of whatever
        // else the machine was doing.
        let mut best = f32::INFINITY;
        for _ in 0..3 {
            let start = std::time::Instant::now();
            let (l, r) = render_to_buffer(preset, &events, AUDIO_S);
            let ratio = start.elapsed().as_secs_f32() / AUDIO_S;
            assert!(l.iter().chain(r.iter()).all(|v| v.is_finite()));
            assert!(peak(&l).max(peak(&r)) <= 1.0, "clipped past 0 dBFS");
            assert!(rms(&l) > 1.0e-3, "the material made no sound");
            best = best.min(ratio);
        }
        best
    };
    let silent = measure(&silent_mechanism(preset()));
    let both = measure(&voiced);
    println!(
        "worst case with a thousand events: {:.1}% of one core with the action \
         silenced, {:.1}% with the mechanism and the hammer's noise both voiced \
         (+{:.2} points)",
        100.0 * silent,
        100.0 * both,
        100.0 * (both - silent)
    );
    assert!(
        both < 0.5,
        "the worst case with the strike noise took {:.1}% of one core \
         (design goal 50%)",
        100.0 * both
    );
}
