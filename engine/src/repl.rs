//! Terminal REPL.
//!
//! Runs on the UI thread and never touches the engine directly — every command
//! becomes an [`Event`] on the SPSC queue, except `render`, which builds its
//! own offline engine.

use crate::engine::EventSender;
use crate::midi;
use crate::preset::Preset;
use crate::render::{
    default_sequence, demo_sequence, halo_sequence, render_to_wav, RenderEvent,
    DEFAULT_DURATION_S, DEMO_DURATION_S, HALO_DURATION_S,
};
use crate::types::{Event, PedalEvent, DEFAULT_RELEASE_VELOCITY, HIGHEST_KEY, LOWEST_KEY};
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Velocity used when a command omits one.
pub const DEFAULT_VELOCITY: u8 = 80;

const NOTE_NAMES: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

/// Parses a scientific-pitch note name (`C4`, `F#3`, `Bb2`, `Ebb-1`) into a
/// MIDI note number, rejecting anything outside the keyboard's A0..C8.
pub fn parse_note(text: &str) -> Option<u8> {
    let lower = text.trim().to_ascii_lowercase();
    let mut chars = lower.chars().peekable();

    let semitone = match chars.next()? {
        'c' => 0,
        'd' => 2,
        'e' => 4,
        'f' => 5,
        'g' => 7,
        'a' => 9,
        'b' => 11,
        _ => return None,
    };

    let mut accidental = 0i32;
    while let Some(&c) = chars.peek() {
        match c {
            '#' | 's' => accidental += 1,
            'b' => accidental -= 1,
            _ => break,
        }
        chars.next();
    }

    let octave: i32 = chars.collect::<String>().parse().ok()?;
    // Scientific pitch notation: C4 = MIDI 60, so A4 = 69 = 440 Hz.
    let midi = (octave + 1) * 12 + semitone + accidental;
    let midi = u8::try_from(midi).ok()?;
    if (LOWEST_KEY..=HIGHEST_KEY).contains(&midi) {
        Some(midi)
    } else {
        None
    }
}

/// Renders a MIDI note number back as a note name (sharps only).
pub fn note_name(key: u8) -> String {
    let octave = key as i32 / 12 - 1;
    format!("{}{}", NOTE_NAMES[(key % 12) as usize], octave)
}

#[derive(Clone, Debug, PartialEq)]
pub enum Command {
    Note { key: u8, vel: u8 },
    /// A key held down without a strike: the damper lifts and nothing sounds.
    Hold { key: u8 },
    Off { key: u8, vel: u8 },
    Chord { keys: Vec<u8>, vel: u8 },
    Pedal(PedalEvent),
    Demo,
    Render { path: PathBuf, source: RenderSource },
    Panic,
    Quit,
    Help,
    /// Blank line.
    Nothing,
}

/// What `render` should play.
#[derive(Clone, Debug, PartialEq)]
pub enum RenderSource {
    /// The compass sweep: what you want to hear when checking evenness.
    Default,
    /// The built-in musical demo.
    Demo,
    /// The sympathetic-resonance phrase: staccato treble, a silently held
    /// bass struck into from above, and a pedal-down wash.
    Halo,
    /// A standard MIDI file.
    Midi(PathBuf),
}

impl RenderSource {
    /// Reads the source into a timed event list and the length to render.
    pub fn resolve(&self) -> Result<(Vec<RenderEvent>, f32), String> {
        match self {
            RenderSource::Default => Ok((default_sequence(), DEFAULT_DURATION_S)),
            RenderSource::Demo => Ok((demo_sequence(), DEMO_DURATION_S)),
            RenderSource::Halo => Ok((halo_sequence(), HALO_DURATION_S)),
            RenderSource::Midi(path) => match midi::load(path) {
                Ok(performance) => {
                    let duration = performance.duration_s();
                    Ok((performance.events, duration))
                }
                Err(e) => Err(format!("could not read {}: {e}", path.display())),
            },
        }
    }
}

/// Extensions `render` treats as a MIDI file rather than a keyword.
fn is_midi_path(arg: &str) -> bool {
    let lower = arg.to_ascii_lowercase();
    lower.ends_with(".mid") || lower.ends_with(".midi")
}

const HELP: &str = "commands (notes are names like C4, F#3, Bb2; A4 = 440 Hz)
  n <note> [vel]           strike a note (vel 1-127, default 80)
  hold <note>              press a key silently: the damper lifts, nothing
                           is struck (sostenuto will still catch it)
  off <note> [rel]         release a key; rel 1-127 is the release velocity,
                           which sets how fast the damper falls (default 64)
  chord <note>... [vel]    strike notes together
  ped sus <0..1>           sustain pedal, continuous (half-pedal works)
  ped sos <0|1>            sostenuto: captures the keys held right now
  ped uc <0|1>             una corda
  demo                     play the built-in demo
  render <out.wav> [what]  render offline: a compass sweep by default,
                           'demo', 'halo' (the sympathetic phrase), or a
                           standard MIDI file (*.mid)
  panic                    all notes and pedals off
  help                     this list
  quit                     exit";

pub fn help_text() -> &'static str {
    HELP
}

/// Parses one input line. The error string is meant to be shown to the user.
pub fn parse_command(line: &str) -> Result<Command, String> {
    let mut parts = line.split_whitespace();
    let Some(verb) = parts.next() else {
        return Ok(Command::Nothing);
    };
    let args: Vec<&str> = parts.collect();

    match verb.to_ascii_lowercase().as_str() {
        "n" | "note" => {
            let key = note_arg(args.first())?;
            let vel = match args.get(1) {
                Some(v) => velocity_arg(v)?,
                None => DEFAULT_VELOCITY,
            };
            if args.len() > 2 {
                return Err("usage: n <note> [vel]".into());
            }
            Ok(Command::Note { key, vel })
        }
        "hold" => {
            if args.len() != 1 {
                return Err("usage: hold <note>".into());
            }
            Ok(Command::Hold {
                key: note_arg(args.first())?,
            })
        }
        "off" => {
            if args.is_empty() || args.len() > 2 {
                return Err("usage: off <note> [rel]".into());
            }
            let vel = match args.get(1) {
                Some(v) => velocity_arg(v)?,
                None => DEFAULT_RELEASE_VELOCITY,
            };
            Ok(Command::Off {
                key: note_arg(args.first())?,
                vel,
            })
        }
        "chord" => {
            if args.is_empty() {
                return Err("usage: chord <note>... [vel]".into());
            }
            // A trailing numeric argument is the velocity, not a note.
            let (notes, vel) = match args.last().and_then(|s| s.parse::<u16>().ok()) {
                Some(_) if args.len() > 1 => {
                    (&args[..args.len() - 1], velocity_arg(args[args.len() - 1])?)
                }
                _ => (&args[..], DEFAULT_VELOCITY),
            };
            let keys = notes
                .iter()
                .map(|n| note_arg(Some(n)))
                .collect::<Result<Vec<u8>, String>>()?;
            Ok(Command::Chord { keys, vel })
        }
        "ped" | "pedal" => parse_pedal(&args),
        "demo" => Ok(Command::Demo),
        "render" => {
            const USAGE: &str = "usage: render <out.wav> [demo | halo | <file.mid>]";
            let Some(path) = args.first() else {
                return Err(USAGE.into());
            };
            if args.len() > 2 {
                return Err(USAGE.into());
            }
            let source = match args.get(1) {
                None => RenderSource::Default,
                Some(a) if a.eq_ignore_ascii_case("demo") => RenderSource::Demo,
                Some(a) if a.eq_ignore_ascii_case("halo") => RenderSource::Halo,
                Some(a) if is_midi_path(a) => RenderSource::Midi(PathBuf::from(a)),
                Some(_) => return Err(USAGE.into()),
            };
            Ok(Command::Render {
                path: PathBuf::from(path),
                source,
            })
        }
        "panic" => Ok(Command::Panic),
        "quit" | "exit" | "q" => Ok(Command::Quit),
        "help" | "?" => Ok(Command::Help),
        other => Err(format!("unknown command '{other}'")),
    }
}

fn parse_pedal(args: &[&str]) -> Result<Command, String> {
    let (Some(which), Some(value)) = (args.first(), args.get(1)) else {
        return Err("usage: ped sus <0..1> | sos <0|1> | uc <0|1>".into());
    };
    match which.to_ascii_lowercase().as_str() {
        "sus" | "sustain" => {
            let v: f32 = value
                .parse()
                .map_err(|_| format!("'{value}' is not a number in 0..1"))?;
            if !(0.0..=1.0).contains(&v) {
                return Err("sustain must be between 0 and 1".into());
            }
            Ok(Command::Pedal(PedalEvent::Sustain(v)))
        }
        "sos" | "sostenuto" => Ok(Command::Pedal(PedalEvent::Sostenuto(bool_arg(value)?))),
        "uc" | "una" => Ok(Command::Pedal(PedalEvent::UnaCorda(bool_arg(value)?))),
        other => Err(format!("unknown pedal '{other}' (sus, sos or uc)")),
    }
}

fn note_arg(arg: Option<&&str>) -> Result<u8, String> {
    let text = arg.ok_or_else(|| "expected a note name".to_string())?;
    parse_note(text).ok_or_else(|| format!("'{text}' is not a note between A0 and C8"))
}

fn velocity_arg(arg: &str) -> Result<u8, String> {
    match arg.parse::<u16>() {
        Ok(v) if (1..=127).contains(&v) => Ok(v as u8),
        _ => Err(format!("'{arg}' is not a velocity between 1 and 127")),
    }
}

fn bool_arg(arg: &str) -> Result<bool, String> {
    match arg.to_ascii_lowercase().as_str() {
        "1" | "on" | "true" => Ok(true),
        "0" | "off" | "false" => Ok(false),
        other => Err(format!("'{other}' is not 0 or 1")),
    }
}

/// Reads commands from stdin until `quit` or EOF. Returns when the user is
/// done. `preset` is the one the live engine was built from, so an offline
/// render started here sounds like what is coming out of the speakers.
pub fn run(mut sender: EventSender, preset: &Preset) -> io::Result<()> {
    println!("piano-emulator — physical model piano");
    println!("{}", help_text());

    let mut stdin = io::stdin().lock();
    let mut line = String::new();
    loop {
        print!("> ");
        io::stdout().flush()?;
        line.clear();
        if stdin.read_line(&mut line)? == 0 {
            println!();
            return Ok(());
        }
        match parse_command(&line) {
            Ok(Command::Quit) => return Ok(()),
            Ok(command) => execute(command, &mut sender, preset),
            Err(message) => {
                println!("{message}");
                println!("type 'help' for the command list");
            }
        }
    }
}

fn execute(command: Command, sender: &mut EventSender, preset: &Preset) {
    match command {
        Command::Nothing | Command::Quit => {}
        Command::Help => println!("{}", help_text()),
        Command::Note { key, vel } => {
            send(sender, Event::NoteOn { key, vel });
            println!("{} vel {vel}", note_name(key));
        }
        Command::Hold { key } => {
            send(sender, Event::KeyDown { key });
            println!("{} held silently", note_name(key));
        }
        Command::Off { key, vel } => send(sender, Event::NoteOff { key, vel }),
        Command::Chord { keys, vel } => {
            for key in &keys {
                send(sender, Event::NoteOn { key: *key, vel });
            }
            let names: Vec<String> = keys.iter().map(|&k| note_name(k)).collect();
            println!("{} vel {vel}", names.join(" "));
        }
        Command::Pedal(pedal) => send(sender, Event::Pedal(pedal)),
        Command::Panic => {
            send(sender, Event::AllOff);
            println!("all notes and pedals off");
        }
        Command::Demo => {
            println!("playing demo ({DEMO_DURATION_S:.0} s)");
            play_live(&demo_sequence(), sender);
        }
        Command::Render { path, source } => {
            let (events, duration) = match source.resolve() {
                Ok(resolved) => resolved,
                Err(message) => {
                    println!("{message}");
                    return;
                }
            };
            match render_to_wav(&path, preset, &events, duration) {
                Ok(()) => println!("wrote {} ({duration:.1} s)", path.display()),
                Err(e) => println!("could not write {}: {e}", path.display()),
            }
        }
    }
}

/// Plays a timed sequence in real time by sleeping between events. The engine
/// keeps rendering on the audio thread throughout.
fn play_live(events: &[RenderEvent], sender: &mut EventSender) {
    let start = Instant::now();
    for scheduled in events {
        let due = Duration::from_secs_f32(scheduled.time_s.max(0.0));
        if let Some(wait) = due.checked_sub(start.elapsed()) {
            std::thread::sleep(wait);
        }
        send(sender, scheduled.event);
    }
}

fn send(sender: &mut EventSender, event: Event) {
    if !sender.send(event) {
        println!("event queue full, dropped {event:?}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn note_names_round_trip() {
        for key in LOWEST_KEY..=HIGHEST_KEY {
            assert_eq!(parse_note(&note_name(key)), Some(key), "key {key}");
        }
    }

    #[test]
    fn accidentals_and_case_are_accepted() {
        assert_eq!(parse_note("C4"), Some(60));
        assert_eq!(parse_note("c4"), Some(60));
        assert_eq!(parse_note("A4"), Some(69));
        assert_eq!(parse_note("F#3"), Some(54));
        assert_eq!(parse_note("Gb3"), Some(54));
        assert_eq!(parse_note("Bb2"), Some(46));
        assert_eq!(parse_note("A#2"), Some(46));
        assert_eq!(parse_note("Fs3"), Some(54));
        assert_eq!(parse_note("A0"), Some(21));
        assert_eq!(parse_note("C8"), Some(108));
    }

    #[test]
    fn out_of_range_and_garbage_are_rejected() {
        assert_eq!(parse_note("G0"), None); // below A0
        assert_eq!(parse_note("C9"), None); // above C8
        assert_eq!(parse_note("H4"), None);
        assert_eq!(parse_note(""), None);
        assert_eq!(parse_note("C"), None);
        assert_eq!(parse_note("4"), None);
    }

    #[test]
    fn note_and_off_commands() {
        assert_eq!(
            parse_command("n C4"),
            Ok(Command::Note {
                key: 60,
                vel: DEFAULT_VELOCITY
            })
        );
        assert_eq!(
            parse_command("N f#3 120"),
            Ok(Command::Note { key: 54, vel: 120 })
        );
        assert_eq!(
            parse_command("off Bb2"),
            Ok(Command::Off {
                key: 46,
                vel: DEFAULT_RELEASE_VELOCITY
            })
        );
        assert!(parse_command("n C4 200").is_err());
        assert!(parse_command("n").is_err());
    }

    /// The two commands `PHYSICS.md` §6 asks for: a key pressed without a
    /// strike, and a release with a velocity of its own.
    #[test]
    fn a_key_can_be_held_silently_and_released_at_a_speed() {
        assert_eq!(parse_command("hold C3"), Ok(Command::Hold { key: 48 }));
        assert_eq!(parse_command("HOLD f#3"), Ok(Command::Hold { key: 54 }));
        assert!(parse_command("hold").is_err());
        assert!(parse_command("hold C3 90").is_err());
        assert!(parse_command("hold X9").is_err());

        assert_eq!(parse_command("off C4 20"), Ok(Command::Off { key: 60, vel: 20 }));
        assert_eq!(
            parse_command("off C4 127"),
            Ok(Command::Off { key: 60, vel: 127 })
        );
        assert!(parse_command("off C4 0").is_err());
        assert!(parse_command("off C4 20 extra").is_err());
        assert!(parse_command("off").is_err());
    }

    #[test]
    fn chords_take_an_optional_trailing_velocity() {
        assert_eq!(
            parse_command("chord C4 E4 G4"),
            Ok(Command::Chord {
                keys: vec![60, 64, 67],
                vel: DEFAULT_VELOCITY
            })
        );
        assert_eq!(
            parse_command("chord C4 E4 G4 100"),
            Ok(Command::Chord {
                keys: vec![60, 64, 67],
                vel: 100
            })
        );
        assert!(parse_command("chord").is_err());
        assert!(parse_command("chord C4 X9").is_err());
    }

    #[test]
    fn pedal_commands() {
        assert_eq!(
            parse_command("ped sus 0.5"),
            Ok(Command::Pedal(PedalEvent::Sustain(0.5)))
        );
        assert_eq!(
            parse_command("ped sos 1"),
            Ok(Command::Pedal(PedalEvent::Sostenuto(true)))
        );
        assert_eq!(
            parse_command("ped uc 0"),
            Ok(Command::Pedal(PedalEvent::UnaCorda(false)))
        );
        assert!(parse_command("ped sus 2").is_err());
        assert!(parse_command("ped foo 1").is_err());
        assert!(parse_command("ped").is_err());
    }

    #[test]
    fn remaining_commands_and_errors() {
        assert_eq!(parse_command("  "), Ok(Command::Nothing));
        assert_eq!(parse_command("demo"), Ok(Command::Demo));
        assert_eq!(parse_command("panic"), Ok(Command::Panic));
        assert_eq!(parse_command("QUIT"), Ok(Command::Quit));
        assert_eq!(parse_command("help"), Ok(Command::Help));
        assert_eq!(
            parse_command("render out.wav demo"),
            Ok(Command::Render {
                path: PathBuf::from("out.wav"),
                source: RenderSource::Demo
            })
        );
        assert_eq!(
            parse_command("render out.wav HALO"),
            Ok(Command::Render {
                path: PathBuf::from("out.wav"),
                source: RenderSource::Halo
            })
        );
        assert_eq!(
            parse_command("render out.wav"),
            Ok(Command::Render {
                path: PathBuf::from("out.wav"),
                source: RenderSource::Default
            })
        );
        assert_eq!(
            parse_command("render out.wav Song.MID"),
            Ok(Command::Render {
                path: PathBuf::from("out.wav"),
                source: RenderSource::Midi(PathBuf::from("Song.MID"))
            })
        );
        assert!(parse_command("render").is_err());
        assert!(parse_command("render out.wav song.wav").is_err());
        assert!(parse_command("render out.wav demo extra").is_err());
        assert!(parse_command("wat").is_err());
    }
}
