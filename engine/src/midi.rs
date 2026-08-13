//! Standard MIDI file replay.
//!
//! A `.mid` file is turned into the same timed [`RenderEvent`] list the built-in
//! demo uses, so replay goes through the engine's ordinary event path and comes
//! out of `Engine::process` like everything else. This is both a useful feature
//! on its own and the forward model the interaction-parameter fitting in
//! TUNING.md needs: a ground-truth performance in, our render out.
//!
//! What is read:
//!
//! - note on / note off on every channel, merged (the instrument is one piano);
//!   a note on with velocity 0 is a note off, as the MIDI spec allows;
//! - CC 64 as a *continuous* sustain pedal, so half-pedalling in a Disklavier
//!   recording reaches the dampers as the fractional value it was played at
//!   rather than as a switch;
//! - CC 66 sostenuto and CC 67 una corda, which are switches on the pedal
//!   itself (a lever either catches the damper levers or it does not);
//! - the tempo map: every SetTempo meta event, from any track.
//!
//! Everything else — program changes, aftertouch, pitch bend, lyrics — is
//! ignored: the engine has no use for it.

use crate::render::RenderEvent;
use crate::types::{Event, PedalEvent, HIGHEST_KEY, LOWEST_KEY};
use midly::{MetaMessage, MidiMessage, Smf, Timing, TrackEventKind};
use std::fmt;
use std::path::Path;

/// MIDI controller numbers for the three pedals.
const CC_SUSTAIN: u8 = 64;
const CC_SOSTENUTO: u8 = 66;
const CC_UNA_CORDA: u8 = 67;

/// A switched controller is down at 64 and up below it — the convention every
/// sustain-capable controller follows.
const SWITCH_THRESHOLD: u8 = 64;

/// Default tempo when a file never states one: 120 bpm, i.e. half a second per
/// quarter note.
const DEFAULT_US_PER_BEAT: f64 = 500_000.0;

/// Silence appended after the last event so releases and the pedal-down halo
/// are not cut off mid-decay.
pub const RELEASE_TAIL_S: f32 = 4.0;

/// A performance read from a MIDI file.
#[derive(Clone, Debug, PartialEq)]
pub struct MidiPerformance {
    /// Events in time order, ready for [`crate::render::render_to_buffer`].
    pub events: Vec<RenderEvent>,
    /// Time of the last event, seconds.
    pub last_event_s: f32,
}

impl MidiPerformance {
    /// Render length that lets the last note decay.
    pub fn duration_s(&self) -> f32 {
        self.last_event_s + RELEASE_TAIL_S
    }
}

pub fn load(path: &Path) -> Result<MidiPerformance, MidiError> {
    let bytes = std::fs::read(path).map_err(MidiError::Io)?;
    parse(&bytes)
}

/// Parses a standard MIDI file held in memory.
pub fn parse(bytes: &[u8]) -> Result<MidiPerformance, MidiError> {
    let smf = Smf::parse(bytes).map_err(MidiError::Parse)?;
    if smf.header.format == midly::Format::Sequential {
        // Format 2 tracks are independent sequences rather than parts of one
        // performance; playing them at once would be meaningless.
        return Err(MidiError::Unsupported("format 2 (sequential tracks)"));
    }

    // Absolute ticks per track first: a track's delta times are relative to
    // that track alone, and the tempo map is shared by all of them.
    let tracks: Vec<Vec<(u64, TrackEventKind)>> = smf
        .tracks
        .iter()
        .map(|track| {
            let mut tick = 0u64;
            track
                .iter()
                .map(|e| {
                    tick += e.delta.as_int() as u64;
                    (tick, e.kind)
                })
                .collect()
        })
        .collect();

    let clock = Clock::new(smf.header.timing, &tracks)?;

    let mut events: Vec<(u64, RenderEvent)> = Vec::new();
    for track in &tracks {
        for &(tick, kind) in track {
            let TrackEventKind::Midi { message, .. } = kind else {
                continue;
            };
            let Some(event) = translate(message) else {
                continue;
            };
            events.push((tick, RenderEvent::new(clock.seconds(tick) as f32, event)));
        }
    }
    // Stable by tick, so simultaneous events keep the order the file gives
    // them — a note off written just before a note on of the same key at the
    // same tick must stay in that order or the note is left hanging.
    events.sort_by_key(|(tick, _)| *tick);

    let events: Vec<RenderEvent> = events.into_iter().map(|(_, e)| e).collect();
    let last_event_s = events.last().map_or(0.0, |e| e.time_s);
    Ok(MidiPerformance {
        events,
        last_event_s,
    })
}

fn translate(message: MidiMessage) -> Option<Event> {
    match message {
        // Velocity 0 is the note off half of a running-status note on.
        MidiMessage::NoteOn { key, vel } if vel.as_int() > 0 => {
            playable(key.as_int()).map(|key| Event::NoteOn {
                key,
                vel: vel.as_int(),
            })
        }
        MidiMessage::NoteOn { key, .. } | MidiMessage::NoteOff { key, .. } => {
            playable(key.as_int()).map(|key| Event::NoteOff { key })
        }
        MidiMessage::Controller { controller, value } => {
            let value = value.as_int();
            let switch = value >= SWITCH_THRESHOLD;
            match controller.as_int() {
                CC_SUSTAIN => Some(Event::Pedal(PedalEvent::Sustain(value as f32 / 127.0))),
                CC_SOSTENUTO => Some(Event::Pedal(PedalEvent::Sostenuto(switch))),
                CC_UNA_CORDA => Some(Event::Pedal(PedalEvent::UnaCorda(switch))),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Keys outside the 88 are dropped here rather than silently ignored deeper in,
/// so the event list is exactly what the instrument will play.
fn playable(key: u8) -> Option<u8> {
    (LOWEST_KEY..=HIGHEST_KEY).contains(&key).then_some(key)
}

/// Tick-to-seconds conversion.
///
/// A metrical file measures time in beats, and every SetTempo event changes how
/// long a beat lasts, so the map is piecewise linear with a breakpoint at each
/// tempo change. Timecode files are already in real time and have no tempo map.
struct Clock {
    /// `(tick, seconds at that tick, seconds per tick from there on)`, by tick.
    segments: Vec<(u64, f64, f64)>,
}

impl Clock {
    fn new(timing: Timing, tracks: &[Vec<(u64, TrackEventKind)>]) -> Result<Clock, MidiError> {
        let ticks_per_beat = match timing {
            Timing::Metrical(t) => t.as_int() as f64,
            Timing::Timecode(fps, subframes) => {
                // Already real time: one fixed rate, no tempo map.
                let per_second = fps.as_f32() as f64 * subframes as f64;
                if per_second <= 0.0 {
                    return Err(MidiError::Unsupported("zero-rate timecode division"));
                }
                return Ok(Clock {
                    segments: vec![(0, 0.0, 1.0 / per_second)],
                });
            }
        };
        if ticks_per_beat <= 0.0 {
            return Err(MidiError::Unsupported("zero ticks per beat"));
        }

        let mut changes: Vec<(u64, f64)> = tracks
            .iter()
            .flat_map(|track| track.iter())
            .filter_map(|&(tick, kind)| match kind {
                TrackEventKind::Meta(MetaMessage::Tempo(us)) => Some((tick, us.as_int() as f64)),
                _ => None,
            })
            .collect();
        changes.sort_by_key(|&(tick, _)| tick);

        let mut segments = vec![(0u64, 0.0, DEFAULT_US_PER_BEAT / 1.0e6 / ticks_per_beat)];
        for (tick, us_per_beat) in changes {
            let seconds_per_tick = us_per_beat / 1.0e6 / ticks_per_beat;
            let &(last_tick, last_seconds, last_rate) = segments.last().expect("never empty");
            let seconds = last_seconds + (tick - last_tick) as f64 * last_rate;
            if tick == last_tick {
                // Two tempo events on the same tick: the last one wins.
                *segments.last_mut().expect("never empty") = (tick, seconds, seconds_per_tick);
            } else {
                segments.push((tick, seconds, seconds_per_tick));
            }
        }
        Ok(Clock { segments })
    }

    fn seconds(&self, tick: u64) -> f64 {
        let i = self.segments.partition_point(|&(t, _, _)| t <= tick) - 1;
        let (start, seconds, rate) = self.segments[i];
        seconds + (tick - start) as f64 * rate
    }
}

#[derive(Debug)]
pub enum MidiError {
    Io(std::io::Error),
    Parse(midly::Error),
    Unsupported(&'static str),
}

impl fmt::Display for MidiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MidiError::Io(e) => write!(f, "{e}"),
            MidiError::Parse(e) => write!(f, "not a readable MIDI file: {e}"),
            MidiError::Unsupported(what) => write!(f, "unsupported MIDI file: {what}"),
        }
    }
}

impl std::error::Error for MidiError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preset::Preset;
    use crate::render::render_to_buffer;
    use crate::types::SAMPLE_RATE;

    /// Variable-length quantity, MIDI's delta-time encoding.
    fn varlen(mut value: u32, out: &mut Vec<u8>) {
        let mut stack = vec![(value & 0x7f) as u8];
        value >>= 7;
        while value > 0 {
            stack.push((value & 0x7f) as u8 | 0x80);
            value >>= 7;
        }
        out.extend(stack.iter().rev());
    }

    /// One track event: delta ticks and raw status/data bytes.
    fn event(delta: u32, bytes: &[u8], out: &mut Vec<u8>) {
        varlen(delta, out);
        out.extend_from_slice(bytes);
    }

    /// Builds a single-track format-0 file with `division` ticks per quarter.
    fn smf(division: u16, track: &[u8]) -> Vec<u8> {
        let mut file = b"MThd".to_vec();
        file.extend_from_slice(&6u32.to_be_bytes());
        file.extend_from_slice(&0u16.to_be_bytes()); // format 0
        file.extend_from_slice(&1u16.to_be_bytes()); // one track
        file.extend_from_slice(&division.to_be_bytes());
        let mut body = track.to_vec();
        event(0, &[0xff, 0x2f, 0x00], &mut body); // end of track
        file.extend_from_slice(b"MTrk");
        file.extend_from_slice(&(body.len() as u32).to_be_bytes());
        file.extend_from_slice(&body);
        file
    }

    /// A quarter note per 480 ticks; at the default 120 bpm that is 0.5 s.
    const DIVISION: u16 = 480;

    fn at(p: &MidiPerformance, i: usize) -> (f32, Event) {
        (p.events[i].time_s, p.events[i].event)
    }

    #[test]
    fn notes_and_tempo_changes_land_where_the_file_puts_them() {
        let mut track = Vec::new();
        // 60 bpm from the start, so a 480-tick quarter lasts a second.
        event(0, &[0xff, 0x51, 0x03, 0x0f, 0x42, 0x40], &mut track);
        event(0, &[0x90, 60, 100], &mut track); // C4 on at t = 0
        event(480, &[0x80, 60, 64], &mut track); // C4 off at t = 1 s
                                                 // Back to 120 bpm: the next quarter is half as long.
        event(0, &[0xff, 0x51, 0x03, 0x07, 0xa1, 0x20], &mut track);
        event(480, &[0x90, 64, 90], &mut track); // E4 on at t = 1.5 s
        event(240, &[0x90, 64, 0], &mut track); // velocity 0 = note off, 1.75 s

        let p = parse(&smf(DIVISION, &track)).expect("parses");
        assert_eq!(p.events.len(), 4);
        assert_eq!(at(&p, 0), (0.0, Event::NoteOn { key: 60, vel: 100 }));
        assert_eq!(at(&p, 1), (1.0, Event::NoteOff { key: 60 }));
        assert_eq!(at(&p, 2), (1.5, Event::NoteOn { key: 64, vel: 90 }));
        assert_eq!(at(&p, 3), (1.75, Event::NoteOff { key: 64 }));
        assert_eq!(p.last_event_s, 1.75);
        assert!(p.duration_s() > p.last_event_s);
    }

    #[test]
    fn pedals_arrive_as_continuous_sustain_and_switched_levers() {
        let mut track = Vec::new();
        event(0, &[0xb0, CC_SUSTAIN, 127], &mut track);
        event(240, &[0xb0, CC_SUSTAIN, 40], &mut track); // half pedal
        event(240, &[0xb0, CC_SUSTAIN, 0], &mut track);
        event(0, &[0xb0, CC_SOSTENUTO, 127], &mut track);
        event(0, &[0xb0, CC_UNA_CORDA, 10], &mut track);
        event(0, &[0xb0, 7, 100], &mut track); // volume: ignored

        let p = parse(&smf(DIVISION, &track)).expect("parses");
        let pedals: Vec<Event> = p.events.iter().map(|e| e.event).collect();
        assert_eq!(
            pedals,
            vec![
                Event::Pedal(PedalEvent::Sustain(1.0)),
                Event::Pedal(PedalEvent::Sustain(40.0 / 127.0)),
                Event::Pedal(PedalEvent::Sustain(0.0)),
                Event::Pedal(PedalEvent::Sostenuto(true)),
                Event::Pedal(PedalEvent::UnaCorda(false)),
            ]
        );
        assert!((p.events[1].time_s - 0.25).abs() < 1e-6);
    }

    /// Two tracks played at once, with the tempo map on the first one only —
    /// the shape of every format-1 file a sequencer writes.
    #[test]
    fn tracks_are_merged_and_share_one_tempo_map() {
        let mut tempo = Vec::new();
        event(0, &[0xff, 0x51, 0x03, 0x0f, 0x42, 0x40], &mut tempo); // 60 bpm
        event(0, &[0xff, 0x2f, 0x00], &mut tempo);
        let mut notes = Vec::new();
        event(480, &[0x91, 48, 80], &mut notes); // channel 2, t = 1 s
        event(0, &[0xff, 0x2f, 0x00], &mut notes);

        let mut file = b"MThd".to_vec();
        file.extend_from_slice(&6u32.to_be_bytes());
        file.extend_from_slice(&1u16.to_be_bytes()); // format 1
        file.extend_from_slice(&2u16.to_be_bytes());
        file.extend_from_slice(&DIVISION.to_be_bytes());
        for track in [&tempo, &notes] {
            file.extend_from_slice(b"MTrk");
            file.extend_from_slice(&(track.len() as u32).to_be_bytes());
            file.extend_from_slice(track);
        }

        let p = parse(&file).expect("parses");
        assert_eq!(at(&p, 0), (1.0, Event::NoteOn { key: 48, vel: 80 }));
    }

    #[test]
    fn keys_outside_the_keyboard_are_dropped() {
        let mut track = Vec::new();
        event(0, &[0x90, 12, 100], &mut track); // C0, below A0
        event(0, &[0x90, 120, 100], &mut track); // above C8
        event(0, &[0x90, 21, 100], &mut track); // A0
        let p = parse(&smf(DIVISION, &track)).expect("parses");
        assert_eq!(p.events.len(), 1);
        assert_eq!(p.events[0].event, Event::NoteOn { key: 21, vel: 100 });
    }

    #[test]
    fn garbage_is_rejected_rather_than_played() {
        assert!(parse(b"not a midi file at all").is_err());
        assert!(parse(&[]).is_err());
    }

    /// The point of the module: a file goes in and audio comes out, with the
    /// note sounding at the time the file asked for.
    #[test]
    fn a_parsed_file_renders_through_the_engine() {
        let mut track = Vec::new();
        event(480, &[0x90, 60, 100], &mut track); // C4 at t = 0.5 s
        event(480, &[0x80, 60, 0], &mut track);
        let p = parse(&smf(DIVISION, &track)).expect("parses");
        let (l, r) = render_to_buffer(&Preset::default(), &p.events, 1.5);

        let energy = |from: f32, to: f32| {
            let (a, b) = ((from * SAMPLE_RATE) as usize, (to * SAMPLE_RATE) as usize);
            l[a..b].iter().chain(&r[a..b]).map(|x| x * x).sum::<f32>()
        };
        assert_eq!(energy(0.0, 0.49), 0.0, "sound before the first note on");
        assert!(energy(0.5, 1.0) > 1e-6, "the note never sounded");
    }
}
