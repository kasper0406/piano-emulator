//! What a velocity-90 strike measures at the output, so that the mechanism's
//! tabulated levels mean what they say.
//!
//! `TUNING_REPORT.md` §5 quotes every mechanism level as a **peak relative to a
//! velocity-90 strike of the same key** — measured, on both sides, at the
//! microphone. The engine's key-off thump reaches the ear the same way the
//! strike does, through the board (`DECISIONS.md` 110), so the ratio the preset
//! asks for is a ratio of two *outputs*; anchoring the burst at the board's
//! *input* instead leaves the board's own response to the two signals in the
//! answer. It is not small, and it is not the same at either end of the
//! compass: against the constant the engine used, the board passes a 190 Hz
//! mechanism burst 1.3 dB more generously than it passes a C4 strike's peak and
//! 3.4 dB more than an A1's, and a preset whose strikes peak lower than the
//! default's — salamander-c5's do, by 1.4 to 1.9 dB — moved every one of its
//! events by that as well.
//!
//! So both halves are measured here, once, when the engine is built:
//!
//! * the **output peak of a velocity-90 strike** at eight keys across the
//!   compass, interpolated between them like every other per-note table;
//! * the **board's peak gain** for each of the four event shapes, as the mean
//!   output peak of eight noise realizations of a burst of that shape — the
//!   statistic a level has to predict, since what a preset asks for is where
//!   many events land rather than where one does.
//!
//! The amplitude a burst is triggered at is then `strike_peak(key) /
//! board_gain(shape)` times the level the preset asks for, and the render lands
//! on the table for any preset rather than for the one the constant was
//! calibrated on. This is `DECISIONS.md` 114's argument — a level that is a
//! peak has to be measured through the chain that produces it rather than
//! derived — carried the one stage further that includes the board.
//!
//! Setup time only, and deterministic: no allocation, no randomness and no
//! measurement of any kind happens after [`MechanismCalibration::new`] returns.

use crate::noise::{Burst, EventModel, EventShape, NoiseShapes};
use crate::pedal::PedalState;
use crate::preset::Preset;
use crate::resonance::ResonanceBus;
use crate::soundboard::Soundboard;
use crate::types::{interp_anchors, key_position, BLOCK, SAMPLE_RATE};
use crate::voice::Voice;

/// Keys whose strike is rendered, one per octave from A0. Eight renders of
/// 50 ms cost about 2.5 ms; measuring all 88 would cost 30 and buy a few tenths
/// of a decibel, since the compass is deliberately flattened by
/// `notes.bridge_gain` and what is left of it is smooth.
const ANCHOR_KEYS: [u8; 8] = [21, 33, 45, 57, 69, 81, 93, 105];

/// How long a strike is rendered for its peak. The peak of a velocity-90 strike
/// arrives with the attack: over the whole compass, 40 ms of render is within
/// 0.11 dB of 3 s, and 50 ms is within 0.00.
const STRIKE_WINDOW_S: f32 = 0.05;

/// How long a burst is rendered for the board's gain, and over how many noise
/// realizations. The ratio of the two peaks is what is averaged, so the window
/// only has to be long enough to contain both — a quarter of a second holds the
/// peak of every shape in `TUNING_REPORT.md` §5's table.
const BURST_WINDOW_S: f32 = 0.25;
const BURST_DRAWS: u64 = 8;

/// Velocity of the strike every mechanism level is quoted against.
const REFERENCE_VELOCITY: u8 = 90;

/// The key a *pedal* event's level is quoted against. The pedal recordings
/// belong to no key, and the tuner measures them against the sampled key
/// nearest C4 (`estimate::noise`), so the engine reads its own C4.
pub const PEDAL_REFERENCE_KEY: u8 = 60;

/// The two measurements that turn `TUNING_REPORT.md` §5's dB into amplitudes.
#[derive(Clone, Copy, Debug)]
pub struct MechanismCalibration {
    /// Output peak of a velocity-90 strike at each of [`ANCHOR_KEYS`].
    strike_peak: [f32; ANCHOR_KEYS.len()],
    /// The board's peak gain for each event shape, in the order
    /// key-off, damper-lift, pedal-down, pedal-up.
    board_gain: [f32; 4],
}

impl MechanismCalibration {
    /// Measures `preset` through the same voice and board the audio path uses.
    pub fn new(preset: &Preset, shapes: &NoiseShapes) -> MechanismCalibration {
        let mut strike_peak = [0.0f32; ANCHOR_KEYS.len()];
        for (slot, &key) in strike_peak.iter_mut().zip(&ANCHOR_KEYS) {
            *slot = strike_output_peak(preset, shapes, key);
        }
        let board_gain = [
            board_peak_gain(preset, shapes.key_off),
            board_peak_gain(preset, shapes.damper_lift),
            board_peak_gain(preset, shapes.pedal_down),
            board_peak_gain(preset, shapes.pedal_up),
        ];
        MechanismCalibration {
            strike_peak,
            board_gain,
        }
    }

    /// A calibration that silences the mechanism outright.
    ///
    /// This is what the scratch voices [`MechanismCalibration::new`] strikes are
    /// built with, which is both what the measurement wants — the reference is a
    /// strike, not a strike with its own action underneath it — and what keeps
    /// the construction from needing itself.
    pub fn silent() -> MechanismCalibration {
        MechanismCalibration {
            strike_peak: [0.0; ANCHOR_KEYS.len()],
            board_gain: [1.0; 4],
        }
    }

    /// Board-input peak amplitude at which an event of `shape_index` on `key`
    /// renders at the same output peak as a velocity-90 strike of that key —
    /// the amplitude a level of 0 dB in `TUNING_REPORT.md` §5's table means.
    fn reference(&self, key: u8, shape_index: usize) -> f32 {
        let mut anchors = [(0.0f32, 0.0f32); ANCHOR_KEYS.len()];
        for (anchor, (&k, &peak)) in anchors
            .iter_mut()
            .zip(ANCHOR_KEYS.iter().zip(&self.strike_peak))
        {
            *anchor = (key_position(k), peak);
        }
        interp_anchors(key_position(key), &anchors) / self.board_gain[shape_index]
    }

    pub fn key_off(&self, key: u8) -> f32 {
        self.reference(key, 0)
    }

    pub fn damper_lift(&self, key: u8) -> f32 {
        self.reference(key, 1)
    }

    pub fn pedal_down(&self) -> f32 {
        self.reference(PEDAL_REFERENCE_KEY, 2)
    }

    pub fn pedal_up(&self) -> f32 {
        self.reference(PEDAL_REFERENCE_KEY, 3)
    }
}

/// Peak of the stereo magnitude `sqrt(l^2 + r^2)` of a velocity-90 strike of
/// `key`, rendered through one voice and one board.
///
/// The magnitude rather than either channel or their mean: an equal-power pan
/// preserves it, so the reference does not depend on where in the stereo image
/// the key sits, and neither the strike nor the thump has to be re-panned to be
/// compared with the other.
fn strike_output_peak(preset: &Preset, shapes: &NoiseShapes, key: u8) -> f32 {
    let mut voice = Voice::new(key, preset, shapes, &MechanismCalibration::silent());
    let mut board = Soundboard::new(&preset.soundboard);
    // No sympathetic bus: one voice cannot answer itself (`resonance.rs`
    // subtracts a string's own contribution), so an uncoupled bus renders
    // exactly what the instrument does to a single note.
    let bus = ResonanceBus::new(0.0);
    voice.note_on(REFERENCE_VELOCITY, &PedalState::new());
    let mut out = [0.0f32; BLOCK];
    let (mut left, mut right) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
    let mut peak = 0.0f32;
    for _ in 0..blocks(STRIKE_WINDOW_S) {
        board.begin_block();
        voice.process(&mut out, &bus, &mut board);
        board.process(&mut left, &mut right);
        peak = peak.max(magnitude_peak(&left, &right));
    }
    peak
}

/// Mean output peak, over [`BURST_DRAWS`] realizations, of a burst of this
/// shape triggered at unit amplitude.
///
/// The mean of the output *peaks*, which is the statistic the level has to
/// predict: what a preset asks for is where many events land on average, and
/// the mean output peak of events triggered at amplitude `a` is `a` times this.
/// [`EventShape::new`]'s own normalisation cancels out of that product exactly —
/// it scales both this measurement and every rendered burst — so what bounds
/// the answer is estimating one mean from [`BURST_DRAWS`] realizations of a
/// quantity that scatters by 5 to 7 dB, the peak of one draw of a narrow noise
/// band being itself a random number (`DECISIONS.md` 114). Eight draws place it
/// within about half a decibel: on the shipped instrument the rendered key-off
/// comes out 0.8 dB under the level asked for, which is where these eight
/// happen to sit above the population. More draws would cost board renders, and
/// the residual is already a quarter of the 2-3 dB by which each individual
/// event is meant to scatter.
///
/// The burst is driven far below the master limiter's threshold and the answer
/// divided back out; at unit amplitude the limiter would return the ratio of
/// its own ceiling instead of the board's gain.
fn board_peak_gain(preset: &Preset, shape: EventShape) -> f32 {
    const DRIVE: f32 = 1.0e-4;
    let model = EventModel::unit(shape);
    let mut after_total = 0.0f32;
    for draw in 0..BURST_DRAWS {
        let mut burst = Burst::new();
        burst.trigger(&model, 0.0, crate::noise::seed_of(0, draw));
        let mut board = Soundboard::new(&preset.soundboard);
        let mut out = [0.0f32; BLOCK];
        let (mut left, mut right) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
        let mut after = 0.0f32;
        for _ in 0..blocks(BURST_WINDOW_S) {
            board.begin_block();
            out.fill(0.0);
            burst.add(&mut out);
            for x in out.iter_mut() {
                *x *= DRIVE;
            }
            // Centred: the board's pan is equal-power and its wet path is fed
            // from the unpanned sum, so the stereo magnitude it returns is the
            // same wherever the event sits (measured: 0.2 dB across the stage).
            board.add_voice(&out, 0.0);
            board.process(&mut left, &mut right);
            after = after.max(magnitude_peak(&left, &right));
        }
        after_total += after / DRIVE;
    }
    let mean = after_total / BURST_DRAWS as f32;
    if mean > 0.0 {
        mean
    } else {
        1.0
    }
}

fn magnitude_peak(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .fold(0.0f32, |m, (&l, &r)| m.max((l * l + r * r).sqrt()))
}

fn blocks(seconds: f32) -> usize {
    (seconds * SAMPLE_RATE / BLOCK as f32).ceil() as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::amp_to_db;

    /// The two halves of the calibration, on the shipped instrument: a
    /// velocity-90 strike peaks around -19 dBFS and the board passes a
    /// mechanism burst at about +16 dB. Windows rather than values — what the
    /// test pins is that the measurement ran and returned something physical,
    /// while `acceptance::a_note_off_thumps_at_the_level_the_recordings_
    /// measured` checks the number that matters on the finished render.
    #[test]
    fn the_calibration_measures_a_strike_and_the_board() {
        let preset = Preset::default();
        let shapes = NoiseShapes::new(&preset.noise);
        let calibration = MechanismCalibration::new(&preset, &shapes);
        for &key in &ANCHOR_KEYS {
            let db = amp_to_db(strike_output_peak(&preset, &shapes, key));
            assert!(
                (-26.0..-12.0).contains(&db),
                "a velocity-90 strike of key {key} peaks at {db:.1} dBFS"
            );
        }
        for (i, &gain) in calibration.board_gain.iter().enumerate() {
            let db = amp_to_db(gain);
            assert!(
                (6.0..26.0).contains(&db),
                "the board's gain for shape {i} is {db:.1} dB"
            );
        }
        // The reference is an amplitude in the board's input units, and it is
        // the one a 0 dB entry in the table would ask for.
        let c4 = calibration.key_off(60);
        assert!(c4 > 0.0 && c4 < 1.0, "key-off reference {c4}");
    }

    /// Deterministic: two engines built from one preset must calibrate to the
    /// same bits, or the same performance would not render the same samples.
    #[test]
    fn the_calibration_is_deterministic() {
        let preset = Preset::default();
        let shapes = NoiseShapes::new(&preset.noise);
        let a = MechanismCalibration::new(&preset, &shapes);
        let b = MechanismCalibration::new(&preset, &shapes);
        assert_eq!(a.strike_peak, b.strike_peak);
        assert_eq!(a.board_gain, b.board_gain);
    }

    /// A silenced calibration is silence, not a very quiet thump: it is what
    /// the scratch voices are built with, and their strikes must not carry a
    /// mechanism of their own.
    #[test]
    fn a_silent_calibration_asks_for_no_amplitude_at_all() {
        let silent = MechanismCalibration::silent();
        for key in [21u8, 60, 108] {
            assert_eq!(silent.key_off(key), 0.0);
            assert_eq!(silent.damper_lift(key), 0.0);
        }
        assert_eq!(silent.pedal_down(), 0.0);
        assert_eq!(silent.pedal_up(), 0.0);
    }

    /// A preset that is quieter overall must move its mechanism with it, which
    /// is the whole reason this is measured rather than tabulated: the level in
    /// the file is a *ratio* to a strike, so halving the instrument's output
    /// has to halve the thump.
    #[test]
    fn the_reference_follows_the_presets_own_gain_staging() {
        let preset = Preset::default();
        let shapes = NoiseShapes::new(&preset.noise);
        let loud = MechanismCalibration::new(&preset, &shapes);

        let mut quiet_preset = preset.clone();
        for gain in &mut quiet_preset.notes.bridge_gain {
            *gain *= 0.5;
        }
        let quiet = MechanismCalibration::new(&quiet_preset, &shapes);
        let ratio = quiet.key_off(60) / loud.key_off(60);
        assert!(
            (ratio - 0.5).abs() < 0.05,
            "halving the instrument moved the mechanism by a factor of {ratio:.3}"
        );
    }
}
