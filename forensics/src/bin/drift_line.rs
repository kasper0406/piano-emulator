//! The line `estimate::directivity` inverts `voicing.polarization_pan_spread`
//! through, measured on the engine as it stands.
//!
//! `DRIFT_PER_SPREAD_DB` and `DRIFT_AT_ZERO_DB` are not a model — the module's
//! own header says so — they are the slope and intercept of a straight line
//! fitted to the engine's rendered stereo drift. A forward model that moves
//! moves them, which is why this is a tool and not a comment.
//!
//! ```sh
//! cargo run --release -p forensics --bin drift_line -- presets/salamander-c5.toml
//! ```

use piano_emulator::preset::Preset as EnginePreset;
use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_emulator::types::Event;
use piano_tuner::estimate::directivity::{balance_drift, DirectivityConfig};

/// The eight keys `estimate::directivity`'s constants are documented as having
/// been measured over.
const KEYS: [u8; 8] = [21, 33, 45, 57, 60, 72, 84, 96];
/// The six `tuner/tests/calibration.rs` gates the constants on: A0 and A1 are
/// panned so far left that the spread has almost nowhere to move them.
const GATE_KEYS: [u8; 6] = [45, 57, 60, 72, 84, 96];

fn equal_temperament(key: u8) -> f64 {
    440.0 * 2f64.powf((f64::from(key) - 69.0) / 12.0)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "presets/default.toml".into());
    let board_mix: Option<f32> = std::env::args().nth(2).and_then(|a| a.parse().ok());
    let base = EnginePreset::load(std::path::Path::new(&path))?;
    let config = DirectivityConfig::default();
    let survey = piano_tuner::survey::SurveyConfig::default();
    let sr = f64::from(piano_tuner::SAMPLE_RATE);

    println!("preset {path}, board_mix {board_mix:?}");
    println!("{:>7}  {:>10}  {:>10}", "spread", "8 keys", "6 gate keys");

    let mut points: Vec<(f64, f64, f64)> = Vec::new();
    for step in 0..=4 {
        let spread = 0.1 * step as f32;
        let mut preset = base.clone();
        preset.voicing.polarization_pan_spread = spread;
        if let Some(mix) = board_mix {
            preset.soundboard.board_mix = mix;
        }
        let preset = EnginePreset::from_toml(&preset.to_toml())?;

        let drift_of = |keys: &[u8]| -> f64 {
            let mut d: Vec<f64> = keys
                .iter()
                .filter_map(|&key| {
                    let events = [RenderEvent::new(0.0, Event::NoteOn { key, vel: 90 })];
                    let (left, right) = render_to_buffer(&preset, &events, 8.0);
                    let note_config = survey.note_config(equal_temperament(key)).ok()?;
                    balance_drift(
                        &left,
                        &right,
                        equal_temperament(key),
                        sr,
                        &note_config,
                        &config,
                    )
                    .ok()
                    .map(|x| x.drift_db)
                })
                .collect();
            d.sort_by(f64::total_cmp);
            d[d.len() / 2]
        };
        let all = drift_of(&KEYS);
        let gate = drift_of(&GATE_KEYS);
        println!("{spread:>7.2}  {all:>10.3}  {gate:>10.3}");
        points.push((f64::from(spread), all, gate));
    }

    for (name, pick) in [
        ("8 keys", 1usize),
        ("6 gate keys", 2usize),
    ] {
        let n = points.len() as f64;
        let mx = points.iter().map(|p| p.0).sum::<f64>() / n;
        let my = points
            .iter()
            .map(|p| if pick == 1 { p.1 } else { p.2 })
            .sum::<f64>()
            / n;
        let (mut sxy, mut sxx) = (0.0, 0.0);
        for p in &points {
            let y = if pick == 1 { p.1 } else { p.2 };
            sxy += (p.0 - mx) * (y - my);
            sxx += (p.0 - mx) * (p.0 - mx);
        }
        let slope = sxy / sxx;
        let intercept = my - slope * mx;
        let worst = points
            .iter()
            .map(|p| {
                let y = if pick == 1 { p.1 } else { p.2 };
                (y - (intercept + slope * p.0)).abs()
            })
            .fold(0.0f64, f64::max);
        println!("{name}: slope {slope:.3} dB per unit spread, intercept {intercept:.3} dB, worst residual {worst:.3} dB");
    }
    Ok(())
}
