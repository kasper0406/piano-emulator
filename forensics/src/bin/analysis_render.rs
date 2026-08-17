//! Renders the audio-quality analysis corpus into `renders/analysis/`.
//!
//! Throwaway measurement material, not part of the instrument: single notes
//! across the compass at several velocities, long held notes for T60, pedal
//! phrases for halo/decay comparisons, and dense chords for headroom checks.
//! The corpus behind `DECISIONS.md` 46's independent audio-quality audit.
//!
//! ```text
//! cargo run --release -p forensics --bin analysis_render
//! ```

use piano_emulator::preset::Preset;
use piano_emulator::render::{render_to_wav, RenderEvent};
use piano_emulator::types::{Event, PedalEvent};
use std::path::PathBuf;

fn note_on(t: f32, key: u8, vel: u8) -> RenderEvent {
    RenderEvent::new(t, Event::NoteOn { key, vel: u16::from(vel) })
}

fn note_off(t: f32, key: u8) -> RenderEvent {
    RenderEvent::new(t, Event::NoteOff { key, vel: 64 })
}

fn pedal(t: f32, p: PedalEvent) -> RenderEvent {
    RenderEvent::new(t, Event::Pedal(p))
}

fn main() {
    let dir = PathBuf::from("renders/analysis");
    std::fs::create_dir_all(&dir).expect("create output dir");
    let preset = Preset::default();
    let save = |name: &str, events: &[RenderEvent], dur: f32| {
        render_to_wav(&dir.join(name), &preset, events, dur).expect("render");
        println!("wrote {name}");
    };

    // Long held single notes for T60 measurement (key held for the full file).
    for &(key, name, dur) in &[
        (21u8, "t60_a0.wav", 32.0f32),
        (60, "t60_c4.wav", 18.0),
        (84, "t60_c6.wav", 6.0),
        (96, "t60_c7.wav", 3.0),
        (108, "t60_c8.wav", 2.0),
    ] {
        save(name, &[note_on(0.02, key, 80)], dur);
    }

    // 3 s strikes for partial-frequency analysis (spec acceptance test notes).
    for &(key, name) in &[(45u8, "part_a2.wav"), (60, "part_c4.wav"), (69, "part_a4.wav")] {
        save(name, &[note_on(0.02, key, 80)], 3.0);
    }

    // Unison beating: C4 and A2 held long enough to see several beat periods.
    save("beat_c4.wav", &[note_on(0.02, 60, 80)], 15.0);
    save("beat_a2.wav", &[note_on(0.02, 45, 80)], 15.0);

    // Velocity ladder at four places on the compass, one note per file.
    for &(key, tag) in &[(36u8, "c2"), (60, "c4"), (69, "a4"), (84, "c6")] {
        for &vel in &[20u8, 40, 60, 80, 100, 120] {
            let name = format!("vel_{tag}_{vel:03}.wav");
            save(&name, &[note_on(0.02, key, vel)], 2.0);
        }
    }

    // Loudness balance: every third key, mezzo-forte, isolated files.
    for key in (21u8..=108).step_by(3) {
        let name = format!("bal_{key:03}.wav");
        save(&name, &[note_on(0.02, key, 80)], 1.5);
    }

    // Pedal halo trio: C3 struck 1 s then released at t = 1.0,
    //   a) sustain pedal down  b) pedal up  c) key simply held (reference).
    let strike = [note_on(0.02, 48, 90), note_off(1.0, 48)];
    let mut down = vec![pedal(0.0, PedalEvent::Sustain(1.0))];
    down.extend_from_slice(&strike);
    save("pedal_down.wav", &down, 6.0);
    save("pedal_up.wav", &strike, 6.0);
    save("pedal_held.wav", &[note_on(0.02, 48, 90)], 6.0);

    // Pedal halo, spec variant: pedal down, strike-and-release one note, listen
    // for broadband halo 1 s later; same with pedal up as control.
    let mut halo = vec![pedal(0.0, PedalEvent::Sustain(1.0)), note_on(0.02, 48, 100), note_off(0.25, 48)];
    save("halo_down.wav", &halo, 4.0);
    halo.remove(0);
    save("halo_up.wav", &halo, 4.0);

    // Click hunting: pedal transitions, half pedal, restrike on ringing string,
    // staccato damper cutoffs.
    let clicky = vec![
        pedal(0.0, PedalEvent::Sustain(1.0)),
        note_on(0.05, 48, 100),
        note_on(0.05, 55, 95),
        note_off(0.60, 48),
        note_off(0.60, 55),
        pedal(1.20, PedalEvent::Sustain(0.45)),
        pedal(1.80, PedalEvent::Sustain(0.0)),
        note_on(2.20, 60, 110),
        note_off(2.35, 60),
        note_on(2.60, 60, 110), // restrike while ringing
        pedal(2.90, PedalEvent::Sustain(1.0)),
        note_off(3.10, 60),
        pedal(3.60, PedalEvent::Sustain(0.0)),
        RenderEvent::new(4.20, Event::AllOff),
    ];
    save("clicks.wav", &clicky, 5.0);

    // Dense chords for headroom: the loudest plausible gesture, ten notes ff,
    // plus repeated big chords under pedal.
    let mut dense = vec![pedal(0.0, PedalEvent::Sustain(1.0))];
    let big: [u8; 10] = [33, 40, 45, 52, 57, 61, 64, 69, 76, 81];
    for (i, t) in [0.05f32, 0.9, 1.75, 2.6].iter().enumerate() {
        for &k in &big {
            dense.push(note_on(*t, k + (i as u8 % 2) * 2, 127));
            dense.push(note_off(*t + 0.7, k + (i as u8 % 2) * 2));
        }
    }
    save("dense_ff.wav", &dense, 7.0);

    // Release decay: strike, hold 1 s, release with pedal up; the tail after
    // note-off must fall > 40 dB within 0.5 s.
    save("release_c4.wav", &[note_on(0.02, 60, 90), note_off(1.0, 60)], 3.0);
    save("release_c2.wav", &[note_on(0.02, 36, 90), note_off(1.0, 36)], 3.0);
}
