//! Live MIDI input through Core MIDI (macOS).
//!
//! `piano-emulator --midi-in` opens one input port, connects it to every
//! hardware source (or to the ones a name matches), and publishes a virtual
//! destination so other applications can play the instrument too. Everything
//! that arrives becomes an [`Event`] through [`ump`] and goes into the same
//! queue the REPL uses — [`EventInput`] — so the two are simply merged and the
//! audio thread cannot tell them apart.
//!
//! # The protocol
//!
//! The port is created with `MIDIInputPortCreateWithProtocol(kMIDIProtocol_2_0)`
//! (`coremidi` 0.9's `Client::input_port_with_protocol`). That is the whole of
//! this instrument's MIDI 2.0 story, and it is enough, because **Core MIDI
//! translates**: a MIDI 1.0 keyboard's bytes arrive at a 2.0 port as MIDI 2.0
//! channel-voice UMP, up-scaled by the specification's own min-center-max rule,
//! and a genuine UMP source's 16-bit velocity arrives untouched. One parser
//! reads both, and `DISTRIBUTION.md`'s "we do not implement MIDI-CI" holds: the
//! OS negotiates, we set a protocol and read what comes.
//!
//! Measured on a Core MIDI loopback in this repository's own test
//! (`engine/tests/live_midi.rs`): `0x90 0x3C 0x5A` from a MIDI 1.0 virtual
//! source arrives as `40903C00 B4D30000` — velocity 46 291, which is exactly
//! `upscale_7_to_16(90)` — and a UMP packet sent with a 16-bit velocity arrives
//! with all sixteen bits intact.
//!
//! If the protocol port cannot be created (it needs macOS 11), the client falls
//! back to `MIDIInputPortCreateWithBlock` and the MIDI 1.0 byte parser. The
//! events are the same; only the resolution is lost.
//!
//! # The sustain pedal is slewed
//!
//! A 7-bit CC 64 has 128 positions and no LSB partner — 14-bit CC pairs only
//! exist for controllers 0–31 — so a slow pedal move arrives as a staircase.
//! Against a *continuous* damper model that staircase is audible in a way it
//! never is against a gated sampler, which is why `DISTRIBUTION.md` asks for a
//! slew limiter in the adapter rather than for anyone's 32-bit controller.
//! [`Slew`] is it: a full-travel move takes [`Slew::TRAVEL_S`], stepped at the
//! engine's own block quantum, and a 32-bit CC 64 from a MIDI 2.0 source goes
//! through the same limiter and is simply never seen to step.
//!
//! # Threads
//!
//! Core MIDI calls the receive block on a thread it owns, at high priority.
//! That thread parses, slews, and pushes into the SPSC queue behind
//! [`EventInput`]'s mutex; it never touches the engine. The audio thread is
//! untouched by this module in every sense — it does not lock, and nothing here
//! runs on it.

use super::ump::{self, Midi1Stream};
use super::EventInput;
use crate::types::{Event, PedalEvent, BLOCK, SAMPLE_RATE};
use coremidi::{
    Client, EventList, InputPort, InputPortWithContext, PacketList, Protocol, Source, Sources,
    VirtualDestination,
};
use std::fmt;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Name this client and its virtual destination appear under in other
/// applications' MIDI setup.
pub const CLIENT_NAME: &str = "Piano Emulator";

/// Where a parsed event goes.
///
/// In the standalone this is [`EventInput::send`] and nothing else. It is a
/// parameter rather than a hard-wired queue for one reason: it lets
/// `engine/tests/live_midi.rs` put a channel here and read back the exact
/// events a real Core MIDI endpoint delivered, which is the only way to test
/// the OS's own translation without a keyboard on the desk.
pub type EventSink = Arc<dyn Fn(Event) + Send + Sync + 'static>;

/// A rate limiter on the sustain pedal, in position units per second.
///
/// It is a *rate* limiter and not a smoother: a small step still lands quickly,
/// a full stamp takes [`Slew::TRAVEL_S`], and the value always arrives exactly
/// on the target rather than approaching it. That matters because the pedal's
/// end points are physical — fully up is dampers on strings — and an
/// exponential smoother would never quite get there.
#[derive(Clone, Copy, Debug)]
pub struct Slew {
    current: f32,
    target: f32,
}

impl Slew {
    /// Seconds for the pedal to travel its whole range.
    ///
    /// `DISTRIBUTION.md`'s "~15 ms". It is long enough that the 128 positions of
    /// a 7-bit CC 64 stop being a staircase — one step is 118 µs of travel,
    /// well inside the damper's own 10 ms ramp — and short enough that a real
    /// stamp is not softened: a pianist's fastest pedal is about 60 ms.
    pub const TRAVEL_S: f32 = 0.015;

    /// The interval the pedal is re-sent at while it is moving.
    ///
    /// One engine block. Sending faster cannot help: `DECISIONS.md` 55 applies
    /// an event at the start of the next block, so two positions inside one
    /// block are one position with the first thrown away.
    pub const TICK: Duration =
        Duration::from_nanos((BLOCK as f64 / SAMPLE_RATE as f64 * 1.0e9) as u64);

    pub fn new(initial: f32) -> Slew {
        Slew {
            current: initial,
            target: initial,
        }
    }

    pub fn position(&self) -> f32 {
        self.current
    }

    pub fn set_target(&mut self, target: f32) {
        self.target = target.clamp(0.0, 1.0);
    }

    /// True when the pedal has arrived and there is nothing left to send.
    pub fn settled(&self) -> bool {
        self.current == self.target
    }

    /// Advances by `dt` seconds. Returns the new position if it moved.
    pub fn advance(&mut self, dt: f32) -> Option<f32> {
        if self.settled() {
            return None;
        }
        let step = (dt / Self::TRAVEL_S).max(0.0);
        let delta = self.target - self.current;
        self.current = if delta.abs() <= step {
            self.target
        } else {
            self.current + step * delta.signum()
        };
        Some(self.current)
    }
}

/// The slew's ticker: one thread, asleep until the pedal moves.
struct PedalSlew {
    shared: Arc<(Mutex<SlewState>, Condvar)>,
    thread: Option<JoinHandle<()>>,
}

struct SlewState {
    slew: Slew,
    stopping: bool,
}

impl PedalSlew {
    fn start(sink: EventSink) -> PedalSlew {
        let shared = Arc::new((
            Mutex::new(SlewState {
                slew: Slew::new(0.0),
                stopping: false,
            }),
            Condvar::new(),
        ));
        let worker = Arc::clone(&shared);
        let thread = std::thread::Builder::new()
            .name("pedal-slew".to_string())
            .spawn(move || {
                let (lock, condvar) = &*worker;
                let mut last = Instant::now();
                let mut state = lock.lock().expect("slew mutex");
                loop {
                    state = if state.slew.settled() {
                        // Nothing to do until a pedal message arrives.
                        last = Instant::now();
                        condvar.wait(state).expect("slew mutex")
                    } else {
                        condvar
                            .wait_timeout(state, Slew::TICK)
                            .expect("slew mutex")
                            .0
                    };
                    if state.stopping {
                        return;
                    }
                    let now = Instant::now();
                    let dt = now.duration_since(last).as_secs_f32();
                    last = now;
                    if let Some(position) = state.slew.advance(dt) {
                        sink(Event::Pedal(PedalEvent::Sustain(position)));
                    }
                }
            })
            .expect("spawning the pedal slew thread");
        PedalSlew {
            shared,
            thread: Some(thread),
        }
    }

    /// Called from the Core MIDI callback thread. Takes a lock and notifies;
    /// allocates nothing.
    fn set_target(&self, position: f32) {
        let (lock, condvar) = &*self.shared;
        if let Ok(mut state) = lock.lock() {
            state.slew.set_target(position);
        }
        condvar.notify_one();
    }
}

impl Drop for PedalSlew {
    fn drop(&mut self) {
        let (lock, condvar) = &*self.shared;
        if let Ok(mut state) = lock.lock() {
            state.stopping = true;
        }
        condvar.notify_all();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// One MIDI source, as `--midi-list` reports it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceInfo {
    pub index: usize,
    pub name: String,
    /// What the endpoint says its own protocol is, when it says anything. Core
    /// MIDI translates either way, so this is information rather than a gate.
    pub protocol: Option<&'static str>,
}

/// Every MIDI source currently on the system, in Core MIDI's own order.
pub fn sources() -> Vec<SourceInfo> {
    Sources
        .into_iter()
        .enumerate()
        .map(|(index, source)| SourceInfo {
            index,
            name: source_name(&source),
            protocol: match source.get_property(&coremidi::Properties::protocol_id()) {
                Ok(1) => Some("MIDI 1.0"),
                Ok(2) => Some("MIDI 2.0"),
                _ => None,
            },
        })
        .collect()
}

fn source_name(source: &Source) -> String {
    source
        .display_name()
        .or_else(|| source.name())
        .unwrap_or_else(|| "(unnamed)".to_string())
}

/// A running live input. Dropping it disconnects and disposes everything.
pub struct LiveInput {
    /// Dropped last-ish; every port below belongs to it.
    _client: Client,
    _port: Port,
    _virtual_destination: Option<VirtualDestination>,
    /// Ordered before the client in the struct so the ticker stops before the
    /// queue it writes to can go away.
    _slew: Arc<PedalSlew>,
    connected: Vec<String>,
    protocol: &'static str,
    virtual_destination_name: Option<String>,
}

/// Which kind of port we ended up with. Both are kept alive; only the
/// constructor cares which.
enum Port {
    Ump(InputPortWithContext<()>),
    Bytes(InputPort),
}

impl LiveInput {
    /// Names of the sources this input is connected to.
    pub fn connected(&self) -> &[String] {
        &self.connected
    }

    /// `"MIDI 2.0"` when the port was created with a protocol, `"MIDI 1.0"`
    /// when it fell back to the byte-oriented one.
    pub fn protocol(&self) -> &'static str {
        self.protocol
    }

    /// The virtual destination other applications can send to, if one was
    /// created.
    pub fn virtual_destination(&self) -> Option<&str> {
        self.virtual_destination_name.as_deref()
    }
}

#[derive(Debug)]
pub enum LiveError {
    /// Core MIDI would not give us a client at all — no MIDI server, or a
    /// sandbox that forbids it.
    Client(i32),
    /// Neither `MIDIInputPortCreateWithProtocol` nor the byte-oriented port
    /// could be created.
    Port(i32),
    /// `--midi-in <name>` matched nothing.
    NoSuchSource {
        wanted: String,
        available: Vec<String>,
    },
    /// There are no MIDI sources at all.
    NoSources,
    /// A source was found but would not connect.
    Connect { name: String, status: i32 },
}

impl fmt::Display for LiveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LiveError::Client(status) => {
                write!(f, "could not open a Core MIDI client (OSStatus {status})")
            }
            LiveError::Port(status) => {
                write!(f, "could not open a MIDI input port (OSStatus {status})")
            }
            LiveError::NoSuchSource { wanted, available } => {
                write!(f, "no MIDI source matches '{wanted}'")?;
                if available.is_empty() {
                    write!(f, " (there are none)")
                } else {
                    write!(f, "; available: {}", available.join(", "))
                }
            }
            LiveError::NoSources => write!(
                f,
                "no MIDI sources: plug a keyboard in, or send to the virtual destination"
            ),
            LiveError::Connect { name, status } => {
                write!(f, "could not connect to '{name}' (OSStatus {status})")
            }
        }
    }
}

impl std::error::Error for LiveError {}

/// Opens live input and starts delivering events into `input`.
///
/// `selector` is `None` for "every source" — the instrument is one piano and a
/// second keyboard is another pair of hands — or a case-insensitive substring
/// of the source name. `virtual_destination` publishes an endpoint other
/// applications can send to; it is what makes the standalone playable from a
/// sequencer, an iPad or the test suite without any hardware.
pub fn open(
    input: EventInput,
    selector: Option<&str>,
    virtual_destination: bool,
) -> Result<LiveInput, LiveError> {
    open_with_sink(
        Arc::new(move |event| {
            input.send(event);
        }),
        selector,
        virtual_destination,
    )
}

/// [`open`], with somewhere other than the engine's queue to put the events.
/// See [`EventSink`] for why this exists.
pub fn open_with_sink(
    out: EventSink,
    selector: Option<&str>,
    virtual_destination: bool,
) -> Result<LiveInput, LiveError> {
    let client = Client::new(CLIENT_NAME).map_err(LiveError::Client)?;
    let slew = Arc::new(PedalSlew::start(Arc::clone(&out)));

    // One sink, used by the port and by the virtual destination, so a note
    // played on the keyboard and a note sent by another application are the
    // same event on the same queue.
    let sink = {
        let slew = Arc::clone(&slew);
        move |event: Event| match event {
            // The pedal is the one message that does not go straight through:
            // it sets the slew's target and the ticker sends the positions.
            Event::Pedal(PedalEvent::Sustain(position)) => slew.set_target(position),
            other => out(other),
        }
    };

    let (port, protocol) = match client.input_port_with_protocol("in", Protocol::Midi20, {
        let sink = sink.clone();
        move |list: &EventList, _: &mut ()| {
            for packet in list.iter() {
                ump::parse_ump(packet.data(), &sink);
            }
        }
    }) {
        Ok(port) => (Port::Ump(port), "MIDI 2.0"),
        Err(_) => {
            // macOS 10.x, or a Core MIDI that will not give us a protocol port.
            // The byte parser keeps running status across packets, so it is
            // built once and moved into the callback.
            let mut stream = Midi1Stream::new();
            let sink = sink.clone();
            let port = client
                .input_port("in", move |packets: &PacketList| {
                    for packet in packets.iter() {
                        stream.push_bytes(packet.data(), &sink);
                    }
                })
                .map_err(LiveError::Port)?;
            (Port::Bytes(port), "MIDI 1.0")
        }
    };

    let (virtual_destination, virtual_destination_name) = if virtual_destination {
        let sink = sink.clone();
        match client.virtual_destination_with_protocol(CLIENT_NAME, Protocol::Midi20, {
            move |list: &EventList| {
                for packet in list.iter() {
                    ump::parse_ump(packet.data(), &sink);
                }
            }
        }) {
            Ok(destination) => (Some(destination), Some(CLIENT_NAME.to_string())),
            // Not fatal: the hardware input is the point, and a name clash with
            // another copy of the app is the usual reason this fails.
            Err(_) => (None, None),
        }
    } else {
        (None, None)
    };

    // One snapshot of the source list, kept as endpoints rather than as
    // indices: a keyboard plugged in between the listing and the connecting
    // would otherwise shift every index after it.
    let available: Vec<(String, Source)> = Sources
        .into_iter()
        .map(|source| (source_name(&source), source))
        .collect();
    let wanted: Vec<&(String, Source)> = match selector {
        None => available.iter().collect(),
        Some(name) => {
            let lower = name.to_ascii_lowercase();
            available
                .iter()
                .filter(|(name, _)| name.to_ascii_lowercase().contains(&lower))
                .collect()
        }
    };
    if wanted.is_empty() {
        // With a virtual destination there is still something to play through,
        // so "no keyboard plugged in" is only fatal when it was asked for by
        // name.
        return Err(match selector {
            Some(name) => LiveError::NoSuchSource {
                wanted: name.to_string(),
                available: available.into_iter().map(|(name, _)| name).collect(),
            },
            None if virtual_destination.is_none() => LiveError::NoSources,
            None => {
                return Ok(LiveInput {
                    _client: client,
                    _port: port,
                    _virtual_destination: virtual_destination,
                    _slew: slew,
                    connected: Vec::new(),
                    protocol,
                    virtual_destination_name,
                })
            }
        });
    }

    let mut connected = Vec::new();
    let mut port = port;
    for (name, source) in wanted {
        let status = match &mut port {
            Port::Ump(p) => p.connect_source(source, ()),
            Port::Bytes(p) => p.connect_source(source),
        };
        match status {
            Ok(()) => connected.push(name.clone()),
            Err(status) => {
                return Err(LiveError::Connect {
                    name: name.clone(),
                    status,
                })
            }
        }
    }

    Ok(LiveInput {
        _client: client,
        _port: port,
        _virtual_destination: virtual_destination,
        _slew: slew,
        connected,
        protocol,
        virtual_destination_name,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A full stamp takes the stated travel time, and lands exactly on the
    /// target rather than approaching it.
    #[test]
    fn the_pedal_takes_the_stated_time_to_travel_its_whole_range() {
        let mut slew = Slew::new(0.0);
        slew.set_target(1.0);
        let dt = Slew::TICK.as_secs_f32();
        let mut elapsed = 0.0;
        let mut positions = Vec::new();
        while let Some(p) = slew.advance(dt) {
            elapsed += dt;
            positions.push(p);
        }
        assert_eq!(slew.position(), 1.0, "the pedal never quite arrived");
        assert!(
            (elapsed - Slew::TRAVEL_S).abs() <= dt,
            "full travel took {elapsed} s, not {} s",
            Slew::TRAVEL_S
        );
        // Enough intermediate positions that the 7-bit staircase is gone: one
        // CC step is 1/127 of the range and the slew emits several inside it.
        assert!(
            positions.len() >= 5,
            "only {} positions in a full stamp",
            positions.len()
        );
        assert!(positions.windows(2).all(|w| w[1] > w[0]));
    }

    /// A rate limiter, not a smoother: a small move costs proportionally less
    /// time, so a light pedal touch is not delayed by the same 15 ms.
    #[test]
    fn a_small_move_is_not_slowed_to_a_full_travel() {
        let mut slew = Slew::new(0.5);
        slew.set_target(0.6);
        let dt = 0.001;
        let mut ticks = 0;
        while slew.advance(dt).is_some() {
            ticks += 1;
        }
        assert_eq!(slew.position(), 0.6);
        assert!(
            (ticks as f32 * dt - 0.1 * Slew::TRAVEL_S).abs() <= dt,
            "a tenth of the travel took {ticks} ms"
        );
    }

    /// A target that moves mid-travel is followed, in both directions, without
    /// overshoot — a pianist reversing the pedal is normal playing.
    #[test]
    fn a_reversal_mid_travel_is_followed_without_overshoot() {
        let mut slew = Slew::new(0.0);
        slew.set_target(1.0);
        let dt = Slew::TICK.as_secs_f32();
        slew.advance(dt);
        slew.advance(dt);
        let midway = slew.position();
        assert!(midway > 0.0 && midway < 1.0);
        slew.set_target(0.0);
        let mut positions = Vec::new();
        while let Some(p) = slew.advance(dt) {
            positions.push(p);
        }
        assert_eq!(slew.position(), 0.0);
        assert!(positions.windows(2).all(|w| w[1] < w[0]));
        assert!(positions.iter().all(|&p| (0.0..=midway).contains(&p)));
    }

    /// A settled pedal produces nothing at all, so an idle instrument puts no
    /// events on the queue.
    #[test]
    fn a_settled_pedal_is_silent_on_the_queue() {
        let mut slew = Slew::new(0.25);
        assert!(slew.settled());
        assert_eq!(slew.advance(1.0), None);
        slew.set_target(0.25);
        assert_eq!(slew.advance(1.0), None);
    }
}
