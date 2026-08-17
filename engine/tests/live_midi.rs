//! Live MIDI input, end to end, against Core MIDI itself.
//!
//! No hardware and no mocking: the test creates a **virtual source** in its own
//! process, connects the standalone's real input port to it, and plays it. What
//! runs is the same `MIDIInputPortCreateWithProtocol` path
//! `piano-emulator --midi-in` opens, and — crucially — the same **Core MIDI
//! translation**, so what these assertions pin is not our parser reading our
//! own bytes back but the OS's actual 1.0 → 2.0 up-scaling arriving at the
//! velocities the engine plays.
//!
//! Two things it does not cover, and there is no way to cover them without the
//! keyboard: that an SL88 MK2 negotiates MIDI 2.0 with the OS at all, and that
//! its 16-bit velocities are as fine as its data sheet says (Yamaha, the most
//! candid vendor, admits its own controllers sense 10 bits of the 16 —
//! `DISTRIBUTION.md`'s MIDI 2.0 verdict). `README.md`'s live-input section
//! carries those as a manual smoke test.
//!
//! The whole file is skipped, loudly, if Core MIDI will not give this process a
//! client — a sandbox with no `MIDIServer` to talk to. It is not skipped
//! quietly, because a test that silently passes when it did not run is worse
//! than no test.

#![cfg(target_os = "macos")]

use coremidi::{Client, EventBuffer, PacketBuffer, Protocol, VirtualSource};
use piano_emulator::midi::live;
use piano_emulator::midi::ump::upscale_7_to_16;
use piano_emulator::types::{velocity_from_midi, VELOCITY_STEPS};
use piano_emulator::{Event, PedalEvent};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::Arc;
use std::time::Duration;

/// Long enough for a round trip through the MIDI server on a loaded machine,
/// short enough that a genuinely broken path fails the run rather than hanging
/// it.
const DELIVERY: Duration = Duration::from_secs(2);

/// One end-to-end rig: our own virtual source on one side, the standalone's
/// input port on the other, and a channel where the engine's queue would be.
struct Loopback {
    source: VirtualSource,
    events: Receiver<Event>,
    _client: Client,
    _input: live::LiveInput,
}

impl Loopback {
    /// `None` when Core MIDI is unavailable to this process.
    fn open(label: &str) -> Option<Loopback> {
        let name = format!("piano-emulator loopback {label} {}", std::process::id());
        let client = Client::new("piano-emulator loopback").ok()?;
        let source = client.virtual_source(&name).ok()?;
        let (tx, events) = mpsc::channel();
        // Connect by name to our own source only, and publish no virtual
        // destination: two of these must be able to run in one test binary.
        let input = live::open_with_sink(
            Arc::new(move |event| {
                let _ = tx.send(event);
            }),
            Some(&name),
            false,
        )
        .expect("the loopback source is there to connect to");
        Some(Loopback {
            source,
            events,
            _client: client,
            _input: input,
        })
    }

    fn send_midi1(&self, bytes: &[u8]) {
        self.source
            .received(&PacketBuffer::new(0, bytes))
            .expect("sending MIDI 1.0 bytes");
    }

    fn send_ump(&self, words: &[u32]) {
        self.source
            .received(EventBuffer::new(Protocol::Midi20).with_packet(0, words))
            .expect("sending UMP");
    }

    fn next(&self) -> Event {
        match self.events.recv_timeout(DELIVERY) {
            Ok(event) => event,
            Err(RecvTimeoutError::Timeout) => panic!("nothing arrived within {DELIVERY:?}"),
            Err(e) => panic!("the loopback closed: {e:?}"),
        }
    }

    /// Everything that arrives until the channel goes quiet for `quiet`.
    fn drain(&self, quiet: Duration) -> Vec<Event> {
        let mut out = Vec::new();
        while let Ok(event) = self.events.recv_timeout(quiet) {
            out.push(event);
        }
        out
    }
}

fn velocity_of(event: Event) -> f32 {
    match event {
        Event::NoteOn { vel, .. } | Event::NoteOff { vel, .. } => velocity_from_midi(vel),
        other => panic!("{other:?} is not a key event"),
    }
}

fn skipped(what: &str) -> bool {
    println!("SKIPPED: no Core MIDI client in this environment ({what})");
    true
}

/// A MIDI 1.0 keyboard — the hardware nearly everyone owns — reaches the engine
/// as exactly the events a `.mid` file of the same performance would produce.
/// The velocities are the file reader's numbers, not neighbours of them.
#[test]
fn a_midi_1_keyboard_plays_the_notes_the_file_reader_would_have_played() {
    let Some(rig) = Loopback::open("midi1") else {
        assert!(skipped("midi1"));
        return;
    };
    rig.send_midi1(&[0x90, 60, 90]); // C4, mezzo-forte
    rig.send_midi1(&[0x80, 60, 100]); // let go briskly
    rig.send_midi1(&[0x90, 62, 0]); // the running-status note off

    assert_eq!(velocity_of(rig.next()), 90.0);
    assert_eq!(velocity_of(rig.next()), 100.0);
    // The OS turns a note-on-at-zero into a note off at centre; the engine reads
    // it as the nominal release, which is what it means.
    let third = rig.next();
    assert!(matches!(third, Event::NoteOff { key: 62, .. }));
    assert_eq!(velocity_of(third), 64.0);
}

/// **Core MIDI's own up-scaling**, pinned from the outside. This is the
/// measurement `ump::upscale_7_to_16` exists to invert, and if the OS ever
/// changed it this is the test that would say so.
#[test]
fn core_midi_upscales_a_7_bit_velocity_exactly_as_the_specification_says() {
    let Some(rig) = Loopback::open("upscale") else {
        assert!(skipped("upscale"));
        return;
    };
    for v in [1u8, 20, 40, 64, 90, 100, 126, 127] {
        rig.send_midi1(&[0x90, 60, v]);
        let event = rig.next();
        assert_eq!(
            event,
            Event::NoteOn {
                key: 60,
                vel: u16::from(v) * VELOCITY_STEPS
            },
            "velocity {v} came back as {event:?}"
        );
        assert_eq!(velocity_of(event), f32::from(v));
    }
}

/// A MIDI 2.0 source's sixteen bits survive the crossing: velocities *between*
/// two 7-bit points arrive between them, in order, and none of them collapses
/// onto a MIDI 1.0 value. This is `SHIPPING.md` §4's acceptance clause with the
/// SL88 MK2 replaced by a virtual source — the protocol half of it is real, the
/// hardware half is the manual step in `README.md`.
#[test]
fn a_midi_2_source_keeps_all_sixteen_bits_of_its_velocity() {
    let Some(rig) = Loopback::open("midi2") else {
        assert!(skipped("midi2"));
        return;
    };
    let (low, high) = (upscale_7_to_16(90), upscale_7_to_16(91));
    let steps: Vec<u16> = (0..5).map(|i| low + (high - low) * i / 5).collect();
    let mut played = Vec::new();
    for &v16 in &steps {
        // MIDI 2.0 channel voice, note on, group 0 channel 0, key 60.
        rig.send_ump(&[0x4090_3C00, u32::from(v16) << 16]);
        played.push(velocity_of(rig.next()));
    }
    assert_eq!(played[0], 90.0, "the bottom of the step is velocity 90");
    for pair in played.windows(2) {
        assert!(
            pair[1] > pair[0],
            "sixteen-bit velocities collapsed: {played:?}"
        );
    }
    assert!(
        played[4] < 91.0,
        "a velocity inside one MIDI 1.0 step left it: {played:?}"
    );
    // Four distinct hammer speeds inside a single MIDI 1.0 velocity step, which
    // is the whole of what the widening buys.
    assert!(played[4] - played[0] > 0.5);
}

/// Both protocols, same note: a 7-bit velocity 90 and its MIDI 2.0 spelling
/// play *the same* hammer speed, so a controller switching protocols does not
/// switch instruments.
#[test]
fn the_two_protocols_agree_on_the_same_note() {
    let Some(rig) = Loopback::open("agree") else {
        assert!(skipped("agree"));
        return;
    };
    rig.send_midi1(&[0x90, 60, 90]);
    let bytes = rig.next();
    rig.send_ump(&[0x4090_3C00, u32::from(upscale_7_to_16(90)) << 16]);
    let words = rig.next();
    assert_eq!(bytes, words);
    assert_eq!(velocity_of(bytes), 90.0);
}

/// The sustain pedal arrives slewed: one CC 64 becomes a short ramp of
/// positions rather than a step, so the continuous damper model is never asked
/// to jump (`DISTRIBUTION.md`, MIDI 2.0 verdict, "mitigation available today").
/// The other two pedals are switches and pass straight through.
#[test]
fn the_sustain_pedal_arrives_as_a_ramp_and_the_other_two_as_switches() {
    let Some(rig) = Loopback::open("pedals") else {
        assert!(skipped("pedals"));
        return;
    };
    rig.send_midi1(&[0xB0, 64, 127]);
    let ramp: Vec<f32> = rig
        .drain(Duration::from_millis(200))
        .into_iter()
        .map(|e| match e {
            Event::Pedal(PedalEvent::Sustain(v)) => v,
            other => panic!("{other:?} is not a sustain event"),
        })
        .collect();
    assert!(
        ramp.len() >= 4,
        "the pedal stepped rather than travelled: {ramp:?}"
    );
    assert!(ramp.windows(2).all(|w| w[1] > w[0]), "{ramp:?}");
    assert_eq!(*ramp.last().expect("a ramp"), 1.0, "{ramp:?}");

    rig.send_midi1(&[0xB0, 66, 127]);
    assert_eq!(rig.next(), Event::Pedal(PedalEvent::Sostenuto(true)));
    rig.send_midi1(&[0xB0, 67, 10]);
    assert_eq!(rig.next(), Event::Pedal(PedalEvent::UnaCorda(false)));
}

/// The whole path, including the queue the audio thread pops: a note played on
/// a (virtual) keyboard makes the engine sound. Nothing here is inspected by
/// hand — `Engine::process` is the same call the cpal callback makes.
#[test]
fn a_note_played_live_comes_out_of_the_engine() {
    use piano_emulator::midi::EventInput;
    use piano_emulator::{Engine, Preset, BLOCK};

    let name = format!("piano-emulator loopback engine {}", std::process::id());
    let Ok(client) = Client::new("piano-emulator loopback engine") else {
        assert!(skipped("engine"));
        return;
    };
    let source = client.virtual_source(&name).expect("a virtual source");

    let (mut engine, sender) = Engine::new(&Preset::default());
    let input = EventInput::new(sender);
    let _live = live::open(input, Some(&name), false).expect("connecting to the loopback source");

    let (mut l, mut r) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
    engine.process(&mut l, &mut r);
    assert_eq!(
        engine.active_voices(),
        0,
        "the piano was not silent to start"
    );

    source
        .received(&PacketBuffer::new(0, &[0x90, 60, 100]))
        .expect("sending a note");

    // The event crosses two threads, so give it a bounded number of blocks
    // rather than one.
    let mut energy = 0.0f32;
    for _ in 0..200 {
        engine.process(&mut l, &mut r);
        energy += l.iter().chain(r.iter()).map(|x| x * x).sum::<f32>();
        if engine.active_voices() > 0 && energy > 0.0 {
            break;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(engine.active_voices(), 1, "the note never reached a voice");
    assert!(energy > 0.0, "the note reached a voice but made no sound");
}
