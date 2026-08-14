//! The preset file and the built-in default must be the same instrument.
//!
//! This is the gate on the preset extraction: every parameter the engine used
//! to hold as a constant now travels through `presets/default.toml`, and a
//! render made with that file has to be the render the hand-tuned v1 engine
//! made. Anything less and the tuning pipeline would be fitting a different
//! instrument from the one being played.

use piano_emulator::preset::Preset;
use piano_emulator::render::{demo_sequence, render_to_buffer, RenderEvent, DEMO_DURATION_S};
use piano_emulator::types::Event;
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

/// The checked-in file is not merely *equivalent* to what the engine writes,
/// it is character for character what the engine writes.
///
/// Every field added to the schema since has to leave it alone: the file is the
/// interface to the tuner, which reads and rewrites it through its own copy of
/// the schema, and a field the engine started emitting would break that
/// contract even though every number in the instrument stayed the same. Fields
/// with a neutral value are therefore skipped on the way out — see
/// `preset.rs`'s "Fields a preset may leave out".
#[test]
fn the_checked_in_default_is_character_for_character_what_the_engine_writes() {
    let text = std::fs::read_to_string(default_preset_path()).expect("presets/default.toml reads");
    assert_eq!(
        Preset::default().to_toml(),
        text,
        "presets/default.toml is out of date; regenerate it with \
         `piano-emulator preset presets/default.toml`"
    );
}

/// The instrument the strings make is bit for bit the one v1 made.
///
/// `presets/default.toml` no longer renders the demo byte-identically to the
/// engine that had no action: the mechanism is on by default (its neutral value
/// is a measurement — `DECISIONS.md`), a note-off now thumps, and a damper on
/// its way down soft-limits the string it is landing on. Both are *wanted*, and
/// both are audible: the demo's samples differ from the pre-mechanism render by
/// −12.7 dB RMS.
///
/// What must not have changed is the sounding path underneath them, and that is
/// what this measures. The material is ten strikes across the compass with no
/// release and no pedal move, so no damper is ever between its two seats and no
/// event ever fires: everything it touches — hammer, string, unison, the
/// resonance bus, the soundboard, the master chain — is the code that was there
/// before `notes.inharmonicity_b4`, `notes.contact_width`,
/// `voicing.unison_sigma_scale`, `voicing.polarization_pan_spread` and
/// `noise.rs` were written. The fingerprint below was taken from the engine
/// built at commit `16307c4`, the last one before any of them.
///
/// A failure here is not "update the constant". It means a change reached the
/// strings, and the question to answer is which one and whether it was meant.
#[test]
fn the_sounding_path_is_what_it_was_before_the_mechanism() {
    let mut events = Vec::new();
    for (i, key) in [21u8, 33, 45, 57, 60, 64, 67, 72, 84, 96]
        .into_iter()
        .enumerate()
    {
        events.push(RenderEvent::new(
            0.35 * i as f32,
            Event::NoteOn {
                key,
                vel: 40 + 8 * i as u8,
            },
        ));
    }
    let (l, r) = render_to_buffer(&Preset::default(), &events, 8.0);
    assert!(l.iter().any(|v| v.abs() > 0.02), "the probe made no sound");
    assert_eq!(
        fingerprint(&l, &r),
        "63686423443ec4d3",
        "the sounding path has moved since the engine that had no action"
    );
}

/// FNV-1a over the samples' bits, left channel then right. Any difference at
/// all, in any sample, changes it.
fn fingerprint(left: &[f32], right: &[f32]) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for value in left.iter().chain(right.iter()) {
        for byte in value.to_bits().to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x1000_0000_01b3);
        }
    }
    format!("{hash:016x}")
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
