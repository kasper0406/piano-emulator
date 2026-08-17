//! Binary entry point.
//!
//! With no subcommand: build the engine, hand it to the audio thread, and run
//! the REPL on this thread. The `render` and `preset` subcommands do their work
//! offline and never open the audio device, which is what makes them usable on
//! a machine with no output — or in a pipeline.

use piano_emulator::audio::AudioOutput;
use piano_emulator::engine::Engine;
use piano_emulator::midi::EventInput;
use piano_emulator::preset::Preset;
use piano_emulator::render::render_to_wav;
use piano_emulator::repl::{self, RenderSource};
use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = "usage:
  piano-emulator [--preset <file.toml>] [--midi-in [name]]
      play interactively (audio out + REPL, and a MIDI keyboard if asked for)
  piano-emulator --midi-list
      list the MIDI sources --midi-in can connect to, and exit
  piano-emulator render <out.wav> [demo | halo | <file.mid>] [--preset <file.toml>] [--duration <s>]
      render offline: a compass sweep by default, the built-in demo, the
      sympathetic-resonance phrase, or a standard MIDI file
  piano-emulator preset <out.toml> [--preset <file.toml>]
      write out a preset — the built-in default unless one is given";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

/// The words that are commands rather than values, so `--midi-in` knows not to
/// swallow one.
const SUBCOMMANDS: [&str; 2] = ["render", "preset"];

fn run(args: &[String]) -> Result<(), String> {
    let mut options = Options::parse(args)?;
    if options.midi_list {
        return list_midi_sources();
    }
    if options.midi_in.is_some() && !options.positional.is_empty() {
        return Err(format!(
            "--midi-in plays live; it does nothing for '{}'\n{USAGE}",
            options.positional[0]
        ));
    }
    let preset = match &options.preset {
        Some(path) => Preset::load(path).map_err(|e| format!("{}: {e}", path.display()))?,
        None => Preset::default(),
    };

    match options.positional.first().map(String::as_str) {
        Some("render") => {
            let out = options.take_positional(1, "render needs an output .wav path")?;
            let source = match options.positional.get(2).map(String::as_str) {
                None => RenderSource::Default,
                Some(a) if a.eq_ignore_ascii_case("demo") => RenderSource::Demo,
                Some(a) if a.eq_ignore_ascii_case("halo") => RenderSource::Halo,
                Some(a) => RenderSource::Midi(PathBuf::from(a)),
            };
            if options.positional.len() > 3 {
                return Err(USAGE.to_string());
            }
            let (events, duration) = source.resolve()?;
            let duration = options.duration.unwrap_or(duration);
            render_to_wav(&PathBuf::from(&out), &preset, &events, duration)
                .map_err(|e| format!("could not write {out}: {e}"))?;
            println!("wrote {out} ({duration:.1} s)");
            Ok(())
        }
        Some("preset") => {
            let out = options.take_positional(1, "preset needs an output .toml path")?;
            preset
                .save(&PathBuf::from(&out))
                .map_err(|e| format!("could not write {out}: {e}"))?;
            println!("wrote {out}");
            Ok(())
        }
        Some(other) => Err(format!("unknown command '{other}'\n{USAGE}")),
        None => play(preset, options.midi_in.take()),
    }
}

fn play(preset: Preset, midi_in: Option<MidiIn>) -> Result<(), String> {
    let (engine, sender) = Engine::new(&preset);
    let input = EventInput::new(sender);
    let output = AudioOutput::start(engine).map_err(|e| format!("could not start audio: {e}"))?;
    println!(
        "audio: {} — {} Hz, {} channels",
        output.device_name(),
        output.sample_rate() as u32,
        output.channels()
    );
    println!("preset: {}", preset.name);

    // Held for as long as the REPL runs: dropping it disconnects the keyboard
    // and stops the pedal slew's thread.
    let _live = match midi_in {
        Some(wanted) => Some(open_midi_in(&input, wanted.source.as_deref())?),
        None => None,
    };

    repl::run(&input, &preset).map_err(|e| format!("input error: {e}"))
}

/// `--midi-in`, on the one platform that has it.
#[derive(Clone, Debug, PartialEq)]
struct MidiIn {
    /// A case-insensitive substring of the source name, or `None` for all of
    /// them.
    source: Option<String>,
}

#[cfg(target_os = "macos")]
fn open_midi_in(
    input: &EventInput,
    wanted: Option<&str>,
) -> Result<piano_emulator::midi::live::LiveInput, String> {
    use piano_emulator::midi::live;
    let live = live::open(input.clone(), wanted, true).map_err(|e| e.to_string())?;
    match live.connected() {
        [] => println!("midi in: no sources connected"),
        names => println!("midi in: {} ({})", names.join(", "), live.protocol()),
    }
    if let Some(name) = live.virtual_destination() {
        println!("midi in: also listening as '{name}'");
    }
    Ok(live)
}

#[cfg(not(target_os = "macos"))]
fn open_midi_in(_input: &EventInput, _wanted: Option<&str>) -> Result<(), String> {
    Err("--midi-in needs Core MIDI, which is macOS only".to_string())
}

#[cfg(target_os = "macos")]
fn list_midi_sources() -> Result<(), String> {
    let sources = piano_emulator::midi::live::sources();
    if sources.is_empty() {
        println!("no MIDI sources");
        return Ok(());
    }
    for source in sources {
        match source.protocol {
            Some(protocol) => println!("  [{}] {} — {protocol}", source.index, source.name),
            None => println!("  [{}] {}", source.index, source.name),
        }
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn list_midi_sources() -> Result<(), String> {
    Err("--midi-list needs Core MIDI, which is macOS only".to_string())
}

/// The command line: flags anywhere, everything else positional.
struct Options {
    positional: Vec<String>,
    preset: Option<PathBuf>,
    duration: Option<f32>,
    midi_in: Option<MidiIn>,
    midi_list: bool,
}

impl Options {
    fn parse(args: &[String]) -> Result<Options, String> {
        let mut options = Options {
            positional: Vec::new(),
            preset: None,
            duration: None,
            midi_in: None,
            midi_list: false,
        };
        let mut rest = args.iter();
        while let Some(arg) = rest.next() {
            match arg.as_str() {
                "--preset" => {
                    let path = rest.next().ok_or("--preset needs a file path")?;
                    options.preset = Some(PathBuf::from(path));
                }
                "--duration" => {
                    let value = rest.next().ok_or("--duration needs a number of seconds")?;
                    let seconds: f32 = value
                        .parse()
                        .map_err(|_| format!("'{value}' is not a number of seconds"))?;
                    if !seconds.is_finite() || seconds <= 0.0 {
                        return Err("--duration must be a positive number of seconds".to_string());
                    }
                    options.duration = Some(seconds);
                }
                "--midi-in" => {
                    // The source name is optional, so it is taken only when the
                    // next argument cannot be anything else: not a flag, and not
                    // a subcommand. `--midi-in` with nothing after it connects
                    // to every source.
                    let source = match rest.clone().next() {
                        Some(next)
                            if !next.starts_with('-') && !SUBCOMMANDS.contains(&next.as_str()) =>
                        {
                            rest.next().cloned()
                        }
                        _ => None,
                    };
                    options.midi_in = Some(MidiIn { source });
                }
                "--midi-list" => options.midi_list = true,
                "-h" | "--help" => return Err(USAGE.to_string()),
                other if other.starts_with('-') => {
                    return Err(format!("unknown option '{other}'\n{USAGE}"))
                }
                other => options.positional.push(other.to_string()),
            }
        }
        Ok(options)
    }

    fn take_positional(&mut self, index: usize, missing: &str) -> Result<String, String> {
        self.positional
            .get(index)
            .cloned()
            .ok_or_else(|| format!("{missing}\n{USAGE}"))
    }
}
