//! Binary entry point.
//!
//! With no subcommand: build the engine, hand it to the audio thread, and run
//! the REPL on this thread. The `render` and `preset` subcommands do their work
//! offline and never open the audio device, which is what makes them usable on
//! a machine with no output — or in a pipeline.

use piano_emulator::audio::AudioOutput;
use piano_emulator::engine::Engine;
use piano_emulator::preset::Preset;
use piano_emulator::render::render_to_wav;
use piano_emulator::repl::{self, RenderSource};
use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = "usage:
  piano-emulator [--preset <file.toml>]
      play interactively (audio out + REPL)
  piano-emulator render <out.wav> [demo | <file.mid>] [--preset <file.toml>] [--duration <s>]
      render offline: a compass sweep by default, the built-in demo, or a
      standard MIDI file
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

fn run(args: &[String]) -> Result<(), String> {
    let mut options = Options::parse(args)?;
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
        None => play(preset),
    }
}

fn play(preset: Preset) -> Result<(), String> {
    let (engine, sender) = Engine::new(&preset);
    let output = AudioOutput::start(engine).map_err(|e| format!("could not start audio: {e}"))?;
    println!(
        "audio: {} — {} Hz, {} channels",
        output.device_name(),
        output.sample_rate() as u32,
        output.channels()
    );
    println!("preset: {}", preset.name);
    repl::run(sender, &preset).map_err(|e| format!("input error: {e}"))
}

/// The command line: flags anywhere, everything else positional.
struct Options {
    positional: Vec<String>,
    preset: Option<PathBuf>,
    duration: Option<f32>,
}

impl Options {
    fn parse(args: &[String]) -> Result<Options, String> {
        let mut options = Options {
            positional: Vec::new(),
            preset: None,
            duration: None,
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
