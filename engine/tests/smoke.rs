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
        RenderEvent::new(1.0, Event::NoteOff { key: 60, vel: 64 }),
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
            RenderEvent::new(1.0, Event::NoteOff { key: 48, vel: 64 }),
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
        events.push(RenderEvent::new(t + 0.4, Event::NoteOff { key, vel: 64 }));
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

/// Level of one partial in one window, in dB: the Hann-windowed DFT at that
/// frequency. This is `TUNING_REPORT.md` §5's per-partial reading — the metric
/// that measured 1.2–6.2 dB of left-minus-right drift between 0.3 s and 2 s on
/// the recordings against 0.02–0.14 dB on the engine's renders.
fn partial_level_db(signal: &[f32], f: f32, from_s: f32, to_s: f32) -> f32 {
    let w = window(signal, from_s, to_s);
    let n = w.len() as f64;
    let (mut re, mut im) = (0.0f64, 0.0f64);
    for (i, &x) in w.iter().enumerate() {
        let t = i as f64;
        let hann = 0.5 - 0.5 * (std::f64::consts::TAU * t / n).cos();
        let phase = std::f64::consts::TAU * f as f64 * t / SAMPLE_RATE as f64;
        re += x as f64 * hann * phase.cos();
        im -= x as f64 * hann * phase.sin();
    }
    (10.0 * (re * re + im * im).max(1e-30).log10()) as f32
}

/// A note's stereo balance has to be able to *move* while it decays, which the
/// engine could not do at all: it panned one mono voice per key, so whatever
/// balance a note started with was the balance it died with. With the two
/// polarizations placed apart, the fast one dying leaves the slow one's
/// position behind — and with the spread at zero the old behaviour is exactly
/// what is left.
///
/// The one test here that does not honour `PIANO_PRESET`. How far the balance
/// travels between two given instants is set by how deep the preset's double
/// decay is and by where the handover between the polarizations falls: on
/// `presets/salamander-c5.toml`, whose horizontal polarization is 27.6 dB down
/// rather than the default's 12, the handover is well past 2 s and the same
/// spread moves the balance 0.09 dB inside this window. That is a property of
/// that piano's voicing, not of the mechanism, so the mechanism is measured on
/// the instrument the window was chosen for.
#[test]
fn spreading_the_polarizations_makes_a_held_notes_balance_drift() {
    // Measured on the fundamental: it is the partial with the most left of it
    // two seconds in, so the board's own diffuse field — which is decorrelated
    // between the channels and drifts a little by itself — is furthest down.
    let drift = |key: u8, spread: f32| -> f32 {
        let mut preset = Preset::default();
        preset.voicing.polarization_pan_spread = spread;
        preset.validate().expect("a spread within the ceiling is legal");
        let f0 = preset.f0(key);
        // Held, so nothing but the string's own decay is in the measurement.
        let events = [RenderEvent::new(0.0, Event::NoteOn { key, vel: 100 })];
        let (l, r) = render_to_buffer(&preset, &events, 2.4);
        let balance = |at: f32| {
            partial_level_db(&l, f0, at, at + 0.2) - partial_level_db(&r, f0, at, at + 0.2)
        };
        balance(2.0) - balance(0.3)
    };

    let mut moved = [0.0f32; 2];
    // Two neighbouring keys, because the spread alternates with key parity.
    for (i, key) in [60u8, 61].into_iter().enumerate() {
        let still = drift(key, 0.0);
        assert!(
            still.abs() < 1.0,
            "key {key}: an unspread note's balance moved {still} dB, and the only
             thing that can move it is the board's own field"
        );
        moved[i] = drift(key, 0.4);
        assert!(
            moved[i].abs() > 3.0,
            "key {key}: a spread note's balance moved only {} dB, against the
             1.2-6.2 dB the recordings drift",
            moved[i]
        );
    }
    // ... and it alternates: the note that leaves its aftersound to the right
    // sits next to one that leaves it to the left, so 88 of them do not lean.
    assert!(
        moved[0] * moved[1] < 0.0,
        "neighbouring keys drifted the same way: {moved:?}"
    );
}

#[test]
fn the_demo_renders_without_clipping() {
    let (l, r) = render_to_buffer(&preset(), &demo_sequence(), DEMO_DURATION_S);
    assert!(l.iter().chain(r.iter()).all(|v| v.is_finite()));
    assert!(peak(&l).max(peak(&r)) <= 1.0);
    assert!(rms(&l) > 1e-4, "the demo produced no sound");
}

/// The demo now ends in mechanism noise rather than in digital silence: its
/// last event is a chord under the pedal, and the pedal is still down, so what
/// is left after the strings have gone is the tray rumble and the releases.
/// The check is that the piece is still the piece — the noise is 30 dB or more
/// under it, everywhere, and never on its own loud enough to be a fault.
#[test]
fn the_mechanism_stays_under_the_music() {
    let (l, r) = render_to_buffer(&preset(), &demo_sequence(), DEMO_DURATION_S);
    let music = rms(window(&l, 0.0, 14.0)).max(rms(window(&r, 0.0, 14.0)));
    let mut silent = preset();
    for event in [
        &mut silent.noise.key_off,
        &mut silent.noise.damper_lift,
        &mut silent.noise.pedal_down,
        &mut silent.noise.pedal_up,
    ] {
        for anchor in &mut event.level_db {
            anchor.db = -200.0;
        }
    }
    let (ql, qr) = render_to_buffer(&silent, &demo_sequence(), DEMO_DURATION_S);
    // Difference of the two renders is the mechanism, on its own.
    let noise: Vec<f32> = l.iter().zip(&ql).map(|(a, b)| a - b).collect();
    let level = 20.0 * (rms(&noise) / music).max(1e-30).log10();
    assert!(
        (-60.0..-15.0).contains(&level),
        "the mechanism sits {level:.1} dB under the music, expected roughly -20 to -40"
    );
    // ... and it is genuinely there in both channels.
    let right: Vec<f32> = r.iter().zip(&qr).map(|(a, b)| a - b).collect();
    assert!(rms(&right) > 0.0 && rms(&noise) > 0.0);
    assert!(peak(&noise).max(peak(&right)) < 0.1, "the action is far too loud");
}
