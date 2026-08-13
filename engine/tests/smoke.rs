//! End-to-end checks through the offline render path — the same
//! `Engine::process` the audio callback uses.
//!
//! The spectral acceptance tests from the spec (tuning/inharmonicity, decay,
//! beating, pedal, sympathetic resonance, performance) land here as the DSP
//! modules are completed; these are the invariants that must hold at every
//! stage of that work.

use piano_emulator::preset::Preset;
use piano_emulator::render::{demo_sequence, render_to_buffer, RenderEvent, DEMO_DURATION_S};
use piano_emulator::types::{Event, PedalEvent, SAMPLE_RATE};
use rustfft::{num_complex::Complex32, FftPlanner};

/// The instrument as shipped, or the one `PIANO_PRESET` names — see
/// `tests/acceptance.rs`, which uses the same override for the same reason.
fn preset() -> Preset {
    match std::env::var("PIANO_PRESET") {
        Ok(path) if !path.is_empty() => Preset::load(std::path::Path::new(&path))
            .unwrap_or_else(|e| panic!("PIANO_PRESET={path}: {e}")),
        _ => Preset::default(),
    }
}

fn peak(signal: &[f32]) -> f32 {
    signal.iter().fold(0.0f32, |m, &v| m.max(v.abs()))
}

fn rms(signal: &[f32]) -> f32 {
    (signal.iter().map(|v| v * v).sum::<f32>() / signal.len() as f32).sqrt()
}

fn window(signal: &[f32], from_s: f32, to_s: f32) -> &[f32] {
    let a = (from_s * SAMPLE_RATE) as usize;
    let b = ((to_s * SAMPLE_RATE) as usize).min(signal.len());
    &signal[a.min(b)..b]
}

#[test]
fn an_engine_with_no_events_is_bit_exact_silent() {
    let (l, r) = render_to_buffer(&preset(), &[], 2.0);
    assert!(l.iter().chain(r.iter()).all(|&v| v == 0.0));
}

#[test]
fn a_single_note_sounds_at_a_sane_level_and_decays() {
    let events = [
        RenderEvent::new(0.0, Event::NoteOn { key: 60, vel: 80 }),
        RenderEvent::new(1.0, Event::NoteOff { key: 60 }),
    ];
    let (l, _r) = render_to_buffer(&preset(), &events, 3.0);

    // One channel of a centre-panned note, so ~3 dB below the mono peak the
    // gain staging in `types::OUTPUT_GAIN` is calibrated against.
    let attack = peak(window(&l, 0.0, 0.5));
    assert!(
        (0.025..=1.0).contains(&attack),
        "mezzo-forte C4 peaked at {attack}, expected roughly -32..0 dBFS"
    );
    // Damper down at 1 s: 0.5 s later the note must be more than 40 dB gone.
    let damped = rms(window(&l, 1.5, 1.6));
    assert!(
        damped < attack * 0.01,
        "after damping {damped} vs attack {attack}"
    );
}

#[test]
fn the_sustain_pedal_keeps_a_released_note_ringing() {
    let strike = |sustain: f32| {
        let events = [
            RenderEvent::new(0.0, Event::Pedal(PedalEvent::Sustain(sustain))),
            RenderEvent::new(0.05, Event::NoteOn { key: 48, vel: 90 }),
            RenderEvent::new(1.0, Event::NoteOff { key: 48 }),
        ];
        let (l, _r) = render_to_buffer(&preset(), &events, 3.5);
        rms(window(&l, 3.0, 3.2))
    };
    let held = strike(1.0);
    let released = strike(0.0);
    assert!(
        held > released * 100.0,
        "pedal down {held}, pedal up {released}"
    );
}

#[test]
fn dense_playing_stays_finite_and_bounded() {
    let mut events = vec![RenderEvent::new(
        0.0,
        Event::Pedal(PedalEvent::Sustain(1.0)),
    )];
    // Deterministic pseudo-random glissando-ish spray across the whole compass.
    let mut state = 0x2545_f491u32;
    for i in 0..600 {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let key = 21 + (state >> 16) as u8 % 88;
        let vel = 30 + (state >> 8) as u8 % 90;
        let t = i as f32 * 0.02;
        events.push(RenderEvent::new(t, Event::NoteOn { key, vel }));
        events.push(RenderEvent::new(t + 0.4, Event::NoteOff { key }));
    }
    let (l, r) = render_to_buffer(&preset(), &events, 14.0);

    assert!(l.iter().chain(r.iter()).all(|v| v.is_finite()));
    assert!(peak(&l).max(peak(&r)) <= 1.0);
    let dc = l.iter().sum::<f32>() / l.len() as f32;
    assert!(dc.abs() < 1e-3, "DC offset {dc}");
}

/// Frequency of the strongest spectral peak within a semitone of `around`,
/// refined by parabolic interpolation of the log magnitudes around the peak
/// bin. The search is banded because a real piano note is a whole series of
/// partials and the fundamental is not always the loudest of them.
fn peak_frequency_near(signal: &[f32], fft_size: usize, around: f32) -> f32 {
    let mut buffer: Vec<Complex32> = (0..fft_size)
        .map(|i| {
            let x = signal.get(i).copied().unwrap_or(0.0);
            // Hann window: the sidelobes of a rectangular window would swamp
            // neighbouring partials.
            let w = 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / signal.len() as f32).cos();
            Complex32::new(x * if i < signal.len() { w } else { 0.0 }, 0.0)
        })
        .collect();
    FftPlanner::new().plan_fft_forward(fft_size).process(&mut buffer);

    let mag: Vec<f32> = buffer[..fft_size / 2].iter().map(|c| c.norm()).collect();
    let bin = |f: f32| (f * fft_size as f32 / SAMPLE_RATE) as usize;
    let (lo, hi) = (
        bin(around * 0.944).max(1),
        bin(around * 1.059).min(mag.len() - 2),
    );
    let peak = (lo..=hi)
        .max_by(|&a, &b| mag[a].total_cmp(&mag[b]))
        .expect("search band is not empty");
    let (a, b, c) = (
        mag[peak - 1].max(1e-30).ln(),
        mag[peak].max(1e-30).ln(),
        mag[peak + 1].max(1e-30).ln(),
    );
    let delta = 0.5 * (a - c) / (a - 2.0 * b + c);
    (peak as f32 + delta) * SAMPLE_RATE / fft_size as f32
}

#[test]
fn struck_notes_sound_at_the_right_pitch() {
    // The first partial sits at f0 shifted by inharmonicity and by the unison
    // detuning — a few cents at most.
    for key in [45u8, 60, 69] {
        let events = [RenderEvent::new(0.0, Event::NoteOn { key, vel: 90 })];
        let (l, _r) = render_to_buffer(&preset(), &events, 3.0);
        let expected = preset().f0(key);
        let measured = peak_frequency_near(window(&l, 0.2, 1.5), 1 << 18, expected);
        let cents = 1200.0 * (measured / expected).log2();
        assert!(
            cents.abs() < 5.0,
            "key {key}: {measured} Hz vs {expected} Hz ({cents} cents)"
        );
    }
}

#[test]
fn the_demo_renders_without_clipping() {
    let (l, r) = render_to_buffer(&preset(), &demo_sequence(), DEMO_DURATION_S);
    assert!(l.iter().chain(r.iter()).all(|v| v.is_finite()));
    assert!(peak(&l).max(peak(&r)) <= 1.0);
    assert!(rms(&l) > 1e-4, "the demo produced no sound");
}
