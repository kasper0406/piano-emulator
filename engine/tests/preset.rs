//! The preset file and the built-in default must be the same instrument.
//!
//! This is the gate on the preset extraction: every parameter the engine used
//! to hold as a constant now travels through `presets/default.toml`, and a
//! render made with that file has to be the render the hand-tuned v1 engine
//! made. Anything less and the tuning pipeline would be fitting a different
//! instrument from the one being played.

use piano_emulator::preset::Preset;
use piano_emulator::render::{demo_sequence, render_to_buffer, DEMO_DURATION_S};
use std::path::PathBuf;

fn default_preset_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../presets/default.toml")
}

fn rms_difference(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    (a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>() / a.len().max(1) as f32).sqrt()
}

#[test]
fn rendering_the_demo_from_the_preset_file_is_bit_identical() {
    let from_file = Preset::load(&default_preset_path()).expect("presets/default.toml loads");
    let built_in = Preset::default();

    let demo = demo_sequence();
    let (file_l, file_r) = render_to_buffer(&from_file, &demo, DEMO_DURATION_S);
    let (ref_l, ref_r) = render_to_buffer(&built_in, &demo, DEMO_DURATION_S);

    // Loud enough that a difference would have somewhere to hide.
    assert!(ref_l.iter().any(|v| v.abs() > 0.1));
    for (channel, from_file, built_in) in [("left", &file_l, &ref_l), ("right", &file_r, &ref_r)] {
        let difference = rms_difference(from_file, built_in);
        assert_eq!(
            from_file, built_in,
            "{channel} channel differs: {difference:e} RMS"
        );
    }
}

/// ... and the preset is genuinely in the signal path: a changed one must
/// change the sound, or the test above would pass on an engine that ignored it.
#[test]
fn a_retuned_preset_produces_a_different_render() {
    let mut stretched = Preset::default();
    for f0 in &mut stretched.notes.f0_hz {
        *f0 *= 1.002; // ~3.5 cents sharp, the top of a Railsback curve
    }
    let events = demo_sequence();
    let (l, _) = render_to_buffer(&stretched, &events, 2.0);
    let (reference, _) = render_to_buffer(&Preset::default(), &events, 2.0);
    assert!(
        rms_difference(&l, &reference) > 1.0e-4,
        "retuning the preset did not reach the strings"
    );
}
