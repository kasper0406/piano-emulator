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
/// Calibrated by measurement through the finished chain: a mezzo-forte (vel 80)
/// C4 peaks near -19.5 dBFS, a single fortissimo note near -10, and a ten-note
/// fortissimo chord — the loudest thing a pianist can do — arrives at the
/// safety limiter's -1 dBFS threshold rather than 6 dB past it. Recalibrate
/// whenever the excitation chain or the board's gain changes.
pub const OUTPUT_GAIN: f32 = 9.0;

/// Per-mode amplitude below which a resonator contributes less than -90 dBFS to
/// the master output and may be skipped. Expressed in internal (pre-`OUTPUT_GAIN`) units.
pub const CULL_AMPLITUDE: f32 = 3.162e-5 / OUTPUT_GAIN;

/// Bank energy (sum of |s_k|^2) below which a bank reports itself idle, i.e.
/// contributes less than -100 dBFS to the master output.
pub const IDLE_ENERGY: f32 = (1.0e-5 / OUTPUT_GAIN) * (1.0e-5 / OUTPUT_GAIN);

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
pub fn interp_anchors(x: f32, anchors: &[(f32, f32)]) -> f32 {
    debug_assert!(!anchors.is_empty());
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

/// Everything the UI thread can tell the audio thread to do.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Event {
    NoteOn { key: u8, vel: u8 },
    NoteOff { key: u8 },
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
