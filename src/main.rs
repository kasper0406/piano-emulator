//! Binary entry point: build the engine, hand it to the audio thread, and run
//! the REPL on this thread.

use piano_emulator::audio::AudioOutput;
use piano_emulator::engine::Engine;
use piano_emulator::repl;
use std::process::ExitCode;

fn main() -> ExitCode {
    let (engine, sender) = Engine::new();

    let output = match AudioOutput::start(engine) {
        Ok(output) => output,
        Err(e) => {
            eprintln!("could not start audio: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!(
        "audio: {} — {} Hz, {} channels",
        output.device_name(),
        output.sample_rate() as u32,
        output.channels()
    );

    if let Err(e) = repl::run(sender) {
        eprintln!("input error: {e}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
