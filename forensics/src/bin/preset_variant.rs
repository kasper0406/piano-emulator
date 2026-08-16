//! Writes a copy of a preset with one thing changed, so that a standing board
//! can be run on the variant without a hand-edited file.
//!
//! ```text
//! cargo run --release -p forensics --bin preset_variant -- \
//!     <in.toml> <out.toml> [--no-sigma] [--no-strike] [--no-key-off] \
//!     [--strike-db X] [--strike-velocity-db Y]
//! ```

use piano_emulator::preset::{NoiseAnchor, Preset, SILENT_LEVEL_DB};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let positional: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    let input = positional[0];
    let output = positional[1];
    let mut preset = Preset::load(std::path::Path::new(input)).expect("preset");
    let mut what: Vec<String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--no-sigma" => {
                for row in preset.notes.partial_sigma_scale.iter_mut() {
                    row.clear();
                }
                preset.notes.synthesized_decay.clear();
                what.push("every partial_sigma_scale row cleared".into());
            }
            "--no-strike" => {
                preset.noise.strike.level_db = vec![NoiseAnchor {
                    key: 21,
                    db: SILENT_LEVEL_DB,
                }];
                what.push("[noise.strike] silenced".into());
            }
            "--no-key-off" => {
                preset.noise.key_off.level_db = vec![NoiseAnchor {
                    key: 21,
                    db: SILENT_LEVEL_DB,
                }];
                what.push("[noise.key_off] silenced".into());
            }
            "--strike-db" => {
                let db: f32 = args[i + 1].parse().expect("dB");
                for anchor in preset.noise.strike.level_db.iter_mut() {
                    anchor.db += db;
                }
                what.push(format!("[noise.strike] level {db:+.2} dB"));
                i += 1;
            }
            "--strike-velocity-db" => {
                let db: f32 = args[i + 1].parse().expect("dB");
                preset.noise.strike.velocity_db = db;
                what.push(format!("[noise.strike] velocity_db = {db:.2}"));
                i += 1;
            }
            _ => {}
        }
        i += 1;
    }
    preset.validate().expect("a legal preset");
    preset
        .save(std::path::Path::new(output))
        .expect("write the variant");
    println!("{output}: {}", what.join(", "));
}
