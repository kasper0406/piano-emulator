//! The acceptance tests from SPEC.md, in the spec's order.
//!
//! Everything here runs offline through `Engine::process` — the same code path
//! the audio callback uses — so what is measured is what comes out of the
//! device. `tests/smoke.rs` holds the coarser end-to-end invariants.

use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_emulator::string::StringParams;
use piano_emulator::types::{key_index, note_to_freq, Event, PedalEvent, SAMPLE_RATE};
use rustfft::{num_complex::Complex32, FftPlanner};

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
    let (l, r) = render_to_buffer(events, duration_s);
    l.iter().zip(&r).map(|(a, b)| a + b).collect()
}

/// A single strike, released at `release_s` (never, if that is past the end).
fn strike(key: u8, vel: u8, release_s: f32, duration_s: f32) -> Vec<f32> {
    let mut events = vec![RenderEvent::new(0.0, Event::NoteOn { key, vel })];
    if release_s < duration_s {
        events.push(RenderEvent::new(release_s, Event::NoteOff { key }));
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
        let params = StringParams::for_key(key);

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
        let harmonic = 8.0 * note_to_freq(key);
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
    let measured = t60(&c4, StringParams::for_key(60).partial_freq(1), 1.5);
    assert!(
        (8.0..20.0).contains(&measured),
        "C4 fundamental T60 {measured:.1} s, expected 8..20 s"
    );

    let c7 = strike(96, 90, 30.0, 4.0);
    let measured = t60(&c7, StringParams::for_key(96).partial_freq(1), 0.3);
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
    let env = band_envelope(&y, StringParams::for_key(60).partial_freq(1), hop);

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
            RenderEvent::new(RELEASE_S, Event::NoteOff { key: KEY }),
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
            RenderEvent::new(1.0, Event::NoteOff { key: 48 }),
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
        let (mut engine, _tx) = Engine::new();
        engine.handle_event(Event::Pedal(PedalEvent::Sustain(sustain)));
        engine.handle_event(Event::NoteOn { key: 67, vel: 120 });
        let (mut l, mut r) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
        let blocks = (1.5 * SAMPLE_RATE / BLOCK as f32) as usize;
        for b in 0..blocks {
            if b == (0.2 * SAMPLE_RATE / BLOCK as f32) as usize {
                engine.handle_event(Event::NoteOff { key: 67 });
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
            RenderEvent::new(0.3, Event::NoteOff { key: 55 }),
        ];
        let y = render_mono(&events, 1.6);
        db(rms(window(&y, 1.3, 1.6)))
    };
    let halo = render(1.0);
    let dry = render(0.0);
    assert!(halo > dry + 30.0, "halo {halo:.1} dB vs dry {dry:.1} dB");
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
        events.push(RenderEvent::new(t + 0.35, Event::NoteOff { key }));
        if i % 100 == 0 {
            let pedal = if (i / 100) % 2 == 0 { 1.0 } else { 0.0 };
            events.push(RenderEvent::new(t, Event::Pedal(PedalEvent::Sustain(pedal))));
        }
    }
    let (l, r) = render_to_buffer(&events, 30.0);

    assert!(l.iter().chain(r.iter()).all(|v| v.is_finite()));
    assert!(peak(&l).max(peak(&r)) <= 1.0, "clipped past 0 dBFS");
    for (name, channel) in [("left", &l), ("right", &r)] {
        let dc = channel.iter().sum::<f32>() / channel.len() as f32;
        assert!(db(dc) < -60.0, "{name} DC {:.1} dBFS", db(dc));
    }

    // ... and an engine that was told nothing renders digital silence.
    let (sl, sr) = render_to_buffer(&[], 2.0);
    assert!(sl.iter().chain(sr.iter()).all(|&v| v == 0.0));
}

// -------------------------------------------------------- 7. performance

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
        events.push(RenderEvent::new(t + 0.1, Event::NoteOff { key }));
    }

    let start = std::time::Instant::now();
    let (l, r) = render_to_buffer(&events, AUDIO_S);
    let ratio = start.elapsed().as_secs_f32() / AUDIO_S;

    assert!(l.iter().chain(r.iter()).all(|v| v.is_finite()));
    assert!(rms(&l) > 1.0e-3, "the glissando made no sound");
    assert!(
        ratio < 0.8,
        "worst case took {:.1}% of one core (design goal 50%)",
        100.0 * ratio
    );
}
