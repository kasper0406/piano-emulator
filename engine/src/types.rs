//! Shared constants and small value types used across the whole synthesizer.

/// Internal sample rate. The cpal stream is opened at this rate; the offline
/// renderer uses it unconditionally.
pub const SAMPLE_RATE: f32 = 48_000.0;

/// Internal processing block size in frames. Every DSP stage renders in
/// multiples of this; `Engine::process` chunks arbitrary callback sizes down to it.
pub const BLOCK: usize = 128;

/// Lowest modelled key, A0.
pub const LOWEST_KEY: u8 = 21;
/// Highest modelled key, C8.
pub const HIGHEST_KEY: u8 = 108;
/// Number of keys / voices (A0..C8 inclusive).
pub const NUM_KEYS: usize = (HIGHEST_KEY - LOWEST_KEY) as usize + 1;

/// Maximum partials per modal bank (spec cap; also caps per-string allocation).
pub const MAX_PARTIALS: usize = 80;
/// Maximum unison strings per key (3 in the treble).
pub const MAX_UNISON: usize = 3;

/// Highest partial frequency admitted into a bank, as a fraction of the sample rate.
pub const MAX_PARTIAL_RATIO: f32 = 0.45;

/// Lowest key that still has a damper is F#6; from G6 (MIDI 91) up the strings
/// ring freely, as on a concert grand.
pub const FIRST_UNDAMPED_KEY: u8 = 91;

/// Linear gain applied to the summed voice signal on the way out of the
/// soundboard. The model's internal unit is the string's force on the bridge in
/// newtons scaled by `string::EXCITATION_SCALE`, which is numerically small, so
/// this is where the instrument gets its level.
///
/// **Calibrated on one clause and one only** (`DECISIONS.md` 42, 277): the
/// loudest thing a pianist can do — ten notes at fortissimo struck exactly
/// together — arrives at the safety limiter's -1 dBFS threshold rather than past
/// it. Everything else about the instrument's level is a *consequence* of that
/// number and is reported rather than solved for. On `presets/default.toml`,
/// through the finished chain (`forensics/src/bin/output_gain.rs`):
///
/// | clause | mono sum | channel | limiter |
/// |---|---|---|---|
/// | mezzo-forte C4 (vel 80) | -24.36 dBFS | -28.98 | 0 samples |
/// | single fortissimo C4 (vel 127) | -11.98 | -16.66 | 0 |
/// | loudest single strike (C8, vel 127) | -6.97 | -9.42 | 0 |
/// | **ten-note ff chord** | +4.69 | **-1.02** | **0** |
///
/// **The threshold is a per-channel number**, because `soundboard::soft_clip` is
/// applied to each channel on its own. Item 42 read the chord's *mono sum*
/// against it, and item 266 read "6 dB past the threshold" off a mono sum of two
/// channels both saturated near 1.0 — which is +6.02 dBFS whatever the drive
/// was. The overshoot was real and it was 5.2 dB, not 6, and it is the channel
/// peak that measures it.
///
/// Recalibrate whenever the excitation chain or the board's gain changes: lower
/// the constant until the chord clears the limiter, at which point the render
/// *is* the pre-limiter signal (everything between the voices and `soft_clip` is
/// linear) and the answer is exact in one step.
pub const OUTPUT_GAIN: f32 = 4.95;

/// The master gain the two end-of-note floors below are referred to, frozen at
/// item 42's value.
///
/// Item 42 made [`CULL_AMPLITUDE`] and [`IDLE_ENERGY`] proportional to
/// [`OUTPUT_GAIN`] so that a level change would carry them along, and the rule
/// was harmless until the level actually changed. What it means is that the
/// floors sit at a fixed **dBFS**, so lowering the master gain moves the whole
/// instrument down toward them and spends its quietest mechanisms: at
/// `OUTPUT_GAIN` 4.95 the rule retires the sympathetic pickup that `PHYSICS.md`
/// §6's first acceptance test is about —
/// `a_silently_held_key_answers_a_struck_note_with_the_pedal_up` reads the
/// prepared C3's energy as **exactly zero** — because that mechanism was living
/// less than 5 dB over the culling floor and the recalibration was 5.19 dB
/// (`DECISIONS.md` 278).
///
/// So the floors stop following. Items 275-276 say what they actually are —
/// thresholds on the *string's own state*, deciding when a mode is not worth
/// integrating and when a note has ended — and those are properties of the
/// sounding path, not of where full scale happens to be. Freezing them here
/// makes a master-gain recalibration provably a level change and nothing else:
/// every render is the old render times `OUTPUT_GAIN / 9.0` sample for sample.
/// Measured on the ten-strike probe of
/// `preset::the_sounding_path_is_what_it_was_before_the_mechanism`, undoing the
/// 0.55 leaves **-167.9 dBFS RMS, 122.3 dB under the signal** — f32 rounding in
/// the DC blocker's recursion and nothing else. Unfreeze these and that
/// recalibration is a change to the instrument instead, which is what the
/// suite read the first time it was tried: nine tests red, `PHYSICS.md` §6's
/// sympathetic pickup among them.
const FLOOR_REFERENCE_GAIN: f32 = 9.0;

/// Per-mode amplitude below which a resonator contributes less than -90 dBFS to
/// the master output and may be skipped. Expressed in internal (pre-`OUTPUT_GAIN`) units.
///
/// Divided by [`MAX_UNISON`] x 2 since the string became a coupled group, and
/// that factor is the whole of the correction: one partial is now `2N`
/// eigenmodes that **add coherently at the output**, so `2N` modes each a
/// hair under a per-mode threshold are together `2N` times over it. The
/// free-running construction had the same arithmetic on paper and got away with
/// it because its energy sat in one or two modes per partial rather than six.
/// Measured on `a_prepared_string_rings_on_after_the_note_that_excited_it`: with
/// the undivided threshold a sympathetically prepared C3 holding 3.2e-11 of
/// energy was zeroed outright, where the free-running string holding *less*
/// (1.8e-11) survived because all of it was in one mode.
///
/// **The -90 dBFS in the first line is nominal and the real number is measured**
/// (`forensics/src/bin/top_octave.rs`): dividing by a master gain treats the chain
/// past the voice as unity, and the board and the microphone are not, so through
/// the finished chain the floor lands at **-113.0 to -124.9 dBFS** over the 88
/// keys — 23 dB under what is written above, and remarkably flat, which is the
/// measurement that refuses a per-key or per-bank version of this constant.
/// (It read -107.8 to -119.7 before item 277 recalibrated the master gain; the
/// floor is fixed in internal units now, so it moved down with the instrument
/// rather than staying put and eating into it.)
///
/// It is also **not** what ends a note. `cull` runs only on a block with no
/// input, so an undamped key with an active resonance bus is never culled at
/// all; and the amplitude [`IDLE_ENERGY`] corresponds to used to be *above* this
/// one, so a bank reported itself idle — and `Voice::process` stopped writing
/// samples — before the culling floor was ever reached. That is why family 1's
/// truncation was `IDLE_ENERGY` and not this (`DECISIONS.md` 275-276).
pub const CULL_AMPLITUDE: f32 = 3.162e-5 / (FLOOR_REFERENCE_GAIN * 2.0 * MAX_UNISON as f32);

/// Bank energy (sum of `|s_k|^2`) below which a bank reports itself idle.
///
/// This is not a bookkeeping threshold: it is where a note *ends*. A bank that
/// reports idle lets `Voice::process` take the branch that writes nothing at
/// all, so the instrument's output goes to exact zero the block this crosses —
/// and if it crosses while the recording of the same note is still ringing, the
/// engine has rendered digital silence into the middle of a note.
///
/// Which is what it did. At -100 dBFS nominal, measured through the finished
/// chain at -102 to -114 dBFS at the master, the top octave crossed it at
/// 1.8-2.5 s: A7 read -176.0 dB/s of tail decay against a neighbourhood of
/// -19.9 and flagged `decay`, `beat` and `jitter` at once, and so did A#7, B7
/// and C8 (`DECISIONS.md` 270, family 1; `renders/compass/COMPASS.md`). The
/// three flags are one event, because a step to zero has an infinite decay
/// slope, an envelope span equal to the whole dynamic range and a phase that is
/// noise. Nothing about the *level* was wrong — the mechanism is that the
/// crossing time of a fixed floor is `ln(a0/floor)/sigma`, so a 3x step in the
/// top octave's fitted decay is a 3x step in where the floor lands inside the
/// note, and at A7 it landed inside the window the compass fits a tail over.
///
/// So the floor is set where no comparison against the reference can reach it,
/// which is a measurement and not a choice. Over the 88 keys of
/// `data/salamander/SalamanderGrandPiano-V3+20200602.sfz` at velocity 90 the
/// recordings' own noise floor — the quietest 20 ms anywhere in the note —
/// bottoms out at **-109.8 dBFS** (A7; median -56.4). Three decades down from
/// the old value is -140 dBFS nominal, **-142 to -154 at the master** through
/// the measured chain, so a retired bank is at least 32 dB under the quietest
/// thing the reference piano can represent at any key. The top octave then
/// decays through the whole window: A7 goes from 178.2 dB/s between 2.0 and
/// 3.0 s to **15.4**, against a neighbourhood of 19.9
/// (`the_top_octave_does_not_stop_dead_inside_the_note`).
///
/// Three decades and not four. At `1.0e-8` every family-1 number is identical —
/// the note is already decaying freely through the compass's window at `1.0e-6`
/// — and `tuner/tests/calibration.rs`'s
/// `the_pan_spread_comes_back_from_the_drift_it_puts_in_the_image` goes red:
/// its drift is read over an **8 s** render, so a C7 that now rings for all of
/// it moves the median drift 3.46 -> 3.66 dB and the inverted spread 0.379 ->
/// 0.400 against a 0.08 gate. That gate was passing by 0.001 because
/// `estimate::directivity`'s line is itself stale — the engine's measured slope
/// is 10.97 dB per unit spread against the constant's 8.3
/// (`forensics/src/bin/drift_line.rs`) — and re-deriving it is that estimator's
/// milestone, not this one's. Nothing is bought past `1.0e-7` and something is
/// spent, so the floor stops there.
///
/// The cost is that a voice stays awake longer after its note has ended, which
/// is why `engine/tests/acceptance.rs`'s four performance worst cases are the
/// gate on this number and not an afterthought: measured, they move by 0.1
/// point or less (29.7 -> 29.8 %, 30.3 -> 30.3, 29.5 -> 29.6, 29.4 -> 29.5),
/// because the worst case is a keyboard whose voices are all live anyway.
pub const IDLE_ENERGY: f32 =
    (1.0e-7 / FLOOR_REFERENCE_GAIN) * (1.0e-7 / FLOOR_REFERENCE_GAIN);

/// Concert pitch reference: A4 = MIDI 69 = 440 Hz.
pub fn note_to_freq(note: u8) -> f32 {
    440.0 * 2.0f32.powf((note as f32 - 69.0) / 12.0)
}

/// Voice index for a MIDI note number, or `None` if outside A0..C8.
pub fn key_index(note: u8) -> Option<usize> {
    if (LOWEST_KEY..=HIGHEST_KEY).contains(&note) {
        Some((note - LOWEST_KEY) as usize)
    } else {
        None
    }
}

/// Inverse of [`key_index`].
pub fn index_to_note(index: usize) -> u8 {
    LOWEST_KEY + index as u8
}

/// Position of a key in the compass, 0.0 at A0 and 1.0 at C8. The per-note
/// parameter tables are all written as smooth functions of this.
pub fn key_position(note: u8) -> f32 {
    (note.clamp(LOWEST_KEY, HIGHEST_KEY) - LOWEST_KEY) as f32 / (NUM_KEYS - 1) as f32
}

/// Linear interpolation of `y` over the sorted `x` anchor points, clamped at the ends.
/// Used by the per-note parameter tables so they stay smooth and editable.
///
/// The anchors must be strictly ascending in `x`: a repeated `x` divides by a
/// zero span. The tables written in this file are literals; the one that comes
/// from a file is checked by `Preset::validate`.
pub fn interp_anchors(x: f32, anchors: &[(f32, f32)]) -> f32 {
    debug_assert!(!anchors.is_empty());
    debug_assert!(anchors.windows(2).all(|w| w[0].0 < w[1].0));
    if x <= anchors[0].0 {
        return anchors[0].1;
    }
    for w in anchors.windows(2) {
        let (x0, y0) = w[0];
        let (x1, y1) = w[1];
        if x <= x1 {
            let t = (x - x0) / (x1 - x0);
            return y0 + t * (y1 - y0);
        }
    }
    anchors[anchors.len() - 1].1
}

pub fn db_to_amp(db: f32) -> f32 {
    10.0f32.powf(db / 20.0)
}

pub fn amp_to_db(amp: f32) -> f32 {
    20.0 * amp.max(1.0e-30).log10()
}

/// Pedal state changes carried on the event queue.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PedalEvent {
    /// Continuous sustain pedal, 0.0 = up, 1.0 = fully down (half-pedal in between).
    Sustain(f32),
    /// Sostenuto: on capture, the keys currently held keep their dampers lifted.
    Sostenuto(bool),
    /// Una corda: softens the hammer and drops one struck unison string.
    UnaCorda(bool),
}

/// Nominal velocity attributed to the damper lift of a silent press.
///
/// A key pressed slowly enough that the jack escapes without the hammer
/// reaching the string lifts that note's damper and makes no note — the
/// standard way of preparing sympathetic resonance, and written into repertoire
/// (`PHYSICS.md` §6). The gesture is spelled [`Event::KeyDown`] (or a
/// [`Event::NoteOn`] at velocity 0); this constant only sets how briskly the
/// damper-lift noise of that gesture plays. A nonzero MIDI velocity is *never*
/// reinterpreted as a silent press: recorded performances carry genuine
/// pianissimo notes at velocities 1–3, and a threshold would silence them —
/// exactly the kind of silent, data-dependent loss stage-2 replay fitting
/// cannot survive.
pub const ESCAPEMENT_VELOCITY: u8 = 3;

/// Release velocity used when the source has none to give: MIDI's own idle
/// value, and what a note-off written as a note-on with velocity 0 means.
pub const DEFAULT_RELEASE_VELOCITY: u8 = 64;

/// Everything the UI thread can tell the audio thread to do.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Event {
    /// A key press. Any nonzero velocity throws the hammer — velocity 1 is the
    /// quietest playable note, not a silent press. Velocity 0 is the silent
    /// press, same as [`Event::KeyDown`].
    NoteOn { key: u8, vel: u8 },
    /// A key release. `vel` is the *release* velocity: how fast the key comes
    /// back sets how fast the damper falls onto the string, and how loud the
    /// key-off thump is.
    NoteOff { key: u8, vel: u8 },
    /// A key held down without striking — the silent press, made explicit for
    /// callers that do not want to express it as a velocity.
    KeyDown { key: u8 },
    Pedal(PedalEvent),
    AllOff,
}

/// Enable ARM flush-to-zero so denormal string states cannot stall the audio
/// thread. Must be called from the audio thread itself: FPCR is per-thread.
pub fn enable_flush_to_zero() {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: reads and writes the calling thread's own FPCR; setting the FZ
    // bit only changes denormal handling for this thread.
    unsafe {
        let mut fpcr: u64;
        core::arch::asm!("mrs {0}, fpcr", out(reg) fpcr, options(nomem, nostack));
        core::arch::asm!("msr fpcr, {0}", in(reg) fpcr | (1 << 24), options(nomem, nostack));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concert_pitch() {
        assert!((note_to_freq(69) - 440.0).abs() < 1e-3);
        assert!((note_to_freq(60) - 261.6256).abs() < 1e-3);
        assert!((note_to_freq(21) - 27.5).abs() < 1e-4);
    }

    #[test]
    fn key_indexing() {
        assert_eq!(key_index(21), Some(0));
        assert_eq!(key_index(108), Some(NUM_KEYS - 1));
        assert_eq!(key_index(20), None);
        assert_eq!(key_index(109), None);
        assert_eq!(index_to_note(0), 21);
    }

    #[test]
    fn anchor_interpolation_clamps_and_interpolates() {
        let a = [(0.0, 1.0), (1.0, 3.0)];
        assert_eq!(interp_anchors(-1.0, &a), 1.0);
        assert_eq!(interp_anchors(2.0, &a), 3.0);
        assert!((interp_anchors(0.5, &a) - 2.0).abs() < 1e-6);
    }
}
