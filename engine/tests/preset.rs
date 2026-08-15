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

/// The sounding path, pinned.
///
/// Ten strikes across the compass with no release and no pedal move, so no
/// damper is ever between its two seats and no event ever fires: everything the
/// probe touches is the sounding path — hammer, string, unison, the resonance
/// bus, the soundboard, the master chain — and nothing else.
///
/// A failure here is not "update the constant". It means a change reached the
/// strings, and the question to answer is which one and whether it was meant.
///
/// # The four times it has been meant
///
/// The constant stood at `63686423443ec4d3` from the engine built at commit
/// `16307c4` — the last one before `notes.inharmonicity_b4`,
/// `notes.contact_width`, `voicing.unison_sigma_scale`,
/// `voicing.polarization_pan_spread` and `noise.rs` — through every milestone
/// that added a schema field, because item 103's rule is that a preset without a
/// new field renders what it always rendered, and each of them held it.
///
/// It moved to `61e29d4abaa316f9` when the unison stopped being `2N`
/// free-running oscillators and became the `2N` coupled eigenmodes of one
/// bridge (`FUNDAMENTALS.md` §5, `DECISIONS.md` 223-230). That is not a field
/// with a neutral value; it is a different construction of the same instrument
/// from the same numbers, and it *cannot* be bit-exact — the poles are the
/// eigenvalues of a matrix where they used to be the diagonal of it. What was
/// held instead is measured equivalence, and it is pinned in
/// `tests/partials.rs` and in `string`'s own tests: **0.5 cents of pitch, 5 % of
/// T60, 0.5 dB of level**, each asserted at that number on the quantity the
/// construction sets and with the harness's own share stated separately
/// (`DECISIONS.md` 259). The demo's samples differ from the free-running render
/// by **-7.7 dB RMS** (the sweep by -8.8, the measured preset's demo by -12.1),
/// which is most of what a change of construction is worth and is stated rather
/// than avoided.
///
/// It has not moved since, and the one change measured against it that would
/// have moved it was not taken: `decay_scale` returning the best point of its
/// bisection rather than the last is worth 4.5 points of the worst T60 error,
/// costs -61.8 dB RMS here, and takes two of `tuner/tests/calibration.rs`'s
/// round trips red (`DECISIONS.md` 259).
///
/// It moved to `11a08741631adb99` when `types::IDLE_ENERGY` — the threshold that
/// decides *when a note ends*, because a bank that reports idle lets
/// `Voice::process` stop writing samples altogether — was re-derived from the
/// reference recordings' own noise floor instead of a nominal -100 dBFS. At the
/// old value the top octave was switched off at 1.8-2.5 s where the recordings
/// of the same keys ring for 3.5, and the compass read the step as -176.0 dB/s
/// of decay against a neighbourhood of -19.9 (`DECISIONS.md` 275, family 1).
///
/// This is a change to the sounding path and it was meant, so what is held is
/// the size of it, measured on this probe: **-115.0 dB RMS** against a render
/// whose own RMS is -40.4, i.e. 74.6 dB under the signal it is added to, with a
/// largest single-sample difference of 2.3e-5 (-92.9 dBFS). Nothing that can be
/// heard moved; what moved is how far into silence a dead voice is followed.
///
/// It moved to `b77c69714fdb6f21` when `types::OUTPUT_GAIN` was recalibrated
/// 9.0 -> 4.95, which is the one clause `DECISIONS.md` 42 sets that number by:
/// the ten-note fortissimo chord had been driving the safety limiter 5.19 dB
/// past its threshold and now arrives 0.02 dB under it, with no sample of it
/// shaped (`DECISIONS.md` 277).
///
/// **This one is a pure scale and the fingerprint is the only thing that can
/// see it.** The two end-of-note floors were frozen in internal units in the
/// same change precisely so that it would be (`types::FLOOR_REFERENCE_GAIN`),
/// so every sample of this probe is the old sample times 0.55 to within f32
/// rounding: undo the scale and the residual is **-167.9 dBFS RMS, 122.3 dB
/// under the signal**, largest single sample 4.5e-8 (-147.0 dBFS, 129.4 dB
/// under the peak). The probe's own RMS moves -40.41 -> -45.61 dBFS, which is
/// -5.20 against the -5.1927 the constants say. Nothing about the instrument
/// moved; where full scale is did.
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
        "b77c69714fdb6f21",
        "the sounding path has moved since the master-gain calibration"
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
