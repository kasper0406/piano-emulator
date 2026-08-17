//! Duplex and aliquot segments: the parts of the string that have no damper.
//!
//! A piano string does not end at the bridge or at the agraffe. The *front*
//! segment runs from the capo bar or agraffe to the tuning pin and the *rear*
//! segment from the bridge to the hitch pin; both are short, high-pitched, and
//! — the whole point — have no damper on them. They are driven only through the
//! bridge, and they go on ringing after the speaking length has been stopped.
//! Öberg & Askenfelt measured every main and duplex string over D4–C8 on a
//! concert-condition grand, saw both segments in the bridge motion and in the
//! radiated sound, and found in an ABX test that damping the front duplex was
//! *clearly* perceptible to musicians and to naive listeners alike
//! (`PHYSICS.md` §3).
//!
//! # What drives them
//!
//! Two things, and the second is most of what the feature is for:
//!
//! * the key's **own** bridge force — the mono sum the voice has just rendered,
//!   after the damper felt has had its say, because that is what the bridge
//!   actually sees; and
//! * the **resonance bus**, through the bridge admittance, which is how a
//!   segment answers a note played on another key. It is the very same drive
//!   buffer the voice's strings are given (`resonance::ResonanceBus::drive`,
//!   `coupling · (B(bus) − own_gain · own_previous)`): there is one bus path in
//!   the engine and the segments read it rather than building a second one.
//!
//! Their output joins the voice's mono sum, so it reaches the board at the
//! key's own pan and — being part of the sum — feeds the bus in its turn.
//!
//! # What `gain_db` means, and why it is normalised
//!
//! [`DuplexMode::gain_db`] is the segment's response *at its own frequency*, per
//! unit of the bridge force driving it. The raw input gain of a modal resonator
//! is not: a mode with a 3 s decay answers a steady drive sixty times harder
//! than one with a 0.05 s decay at the same input gain, so an un-normalised
//! `gain_db` would mean "louder" and "longer" at once and a preset could not
//! move one without the other. Here the two are independent, which is what
//! makes the field estimable — the tracker's peak levels and its T60s are
//! separate measurements — and what makes the loop bound below computable.
//!
//! # Being undamped is a cost, and the cost is culling
//!
//! Nothing stops these banks: not the key, not the sustain pedal, not sostenuto
//! (a sostenuto rail catches damper levers, and there is no lever to catch), not
//! una corda (which moves the hammer, not the dampers). Only `AllOff` resets
//! them, because that is a panic and not a gesture. So the *only* thing that
//! ever lets a voice go back to sleep is the segments' own decay, and 88 banks
//! that never sleep would keep 88 voices — and their several hundred string
//! partials each — running for the length of the piece. Three things keep that
//! from happening:
//!
//! * `MAX_DUPLEX_T60_S` is 3 s and `PHYSICS.md` §3 asks for 0.5–2 s, which is
//!   shorter than intuition suggests for exactly this reason;
//! * a segment bank being alive does **not** make its voice's strings run —
//!   `Voice::process` decides those separately — so a ringing duplex costs its
//!   own six resonators and not the note's eighty;
//! * `ModalBank`'s culling applies to the segments like everything else: below
//!   `CULL_AMPLITUDE` the modes are zeroed, the bank reports itself idle, and
//!   the voice returns to the branch that writes no samples at all.

use crate::modal::ModalBank;
use crate::preset::DuplexMode;
use crate::types::{db_to_amp, BLOCK, SAMPLE_RATE};

/// Largest undamped loop gain a preset may ask the segments for.
///
/// The derivation is in `Preset::validate_duplex`, which is also where it is
/// enforced: a quarter of unity, the same margin `resonance.rs` holds the
/// bridge admittance to, and for a stronger reason — a string's contribution to
/// the loop dies with the note and a segment's does not.
pub const MAX_DUPLEX_LOOP_GAIN: f32 = 0.25;

/// `sigma · T60`, i.e. `ln(1000)`.
const T60_DECADES: f32 = 3.0 * std::f32::consts::LN_10;

/// The three coefficients one segment becomes: pole radius, pole angle in
/// radians per sample, and the input gain that puts the mode's response at its
/// own frequency on `gain_db`.
///
/// Driven by `x[n] = sin(w_k n)`, the mode `s[n] = a s[n-1] + g x[n]` settles at
/// `|s| = g / (2 (1 - r))` — the factor of two is the negative-frequency image,
/// which contributes nothing at resonance — so the gain that realises a peak
/// response of `G` is `2 G (1 - r)`. Written with `1 - r` rather than
/// `sigma / SAMPLE_RATE` so that it is exact at the long decays, where the two
/// differ by a part in `10^4`.
fn resonator(mode: &DuplexMode) -> (f32, f32, f32) {
    let sigma = T60_DECADES / mode.t60_s;
    let r = (-sigma / SAMPLE_RATE).exp();
    let w = std::f32::consts::TAU * mode.hz / SAMPLE_RATE;
    (r, w, 2.0 * db_to_amp(mode.gain_db) * (1.0 - r))
}

/// Decay rate of a segment, 1/s, for the callers that build the bank.
pub fn sigma(mode: &DuplexMode) -> f32 {
    T60_DECADES / mode.t60_s
}

/// `|D(f)|` of a whole row of segments as *realised*: how much signal the row
/// puts out per unit of drive at `hz`.
///
/// The modes' magnitudes are summed rather than their complex responses. That
/// is the conservative reading and it is the right one here, because this is
/// what the stability bound is computed from and the phases of two segments
/// that happen to land on one frequency are not something a preset controls.
pub fn magnitude(modes: &[DuplexMode], hz: f32) -> f32 {
    let w = std::f32::consts::TAU * hz / SAMPLE_RATE;
    modes
        .iter()
        .map(|mode| {
            let (r, wk, g) = resonator(mode);
            // |g / (1 - a e^{-jw})| with a = r e^{j w_k}, halved for the image.
            let delta = wk - w;
            let (re, im) = (1.0 - r * delta.cos(), -r * delta.sin());
            0.5 * g / (re * re + im * im).sqrt()
        })
        .sum()
}

/// One key's duplex and aliquot segments.
///
/// Empty for a key the preset gives no segments — and for every key of a preset
/// that has no `notes.duplex` table at all, which is the neutral case: the bank
/// is then never touched and the voice behaves exactly as it did before this
/// module existed.
pub struct DuplexBank {
    bank: ModalBank,
    /// The summed drive, so the two input paths reach `ModalBank` as the one
    /// signal it takes. Held here rather than on the stack because it is
    /// written once per block per voice.
    input: [f32; BLOCK],
}

impl DuplexBank {
    pub fn new(modes: &[DuplexMode]) -> DuplexBank {
        let mut bank = ModalBank::with_capacity(modes.len());
        for mode in modes {
            let (_, _, g) = resonator(mode);
            bank.push_mode(mode.hz, sigma(mode), g);
        }
        DuplexBank {
            bank,
            input: [0.0; BLOCK],
        }
    }

    /// True when this key has no segments, which is the case the whole path is
    /// skipped for.
    pub fn is_empty(&self) -> bool {
        self.bank.is_empty()
    }

    /// True when the segments hold too little to be heard. Nothing damps them,
    /// so this is the only way a voice with segments ever goes quiet.
    pub fn is_idle(&self) -> bool {
        self.bank.is_idle()
    }

    /// How many segments this key has.
    pub fn len(&self) -> usize {
        self.bank.len()
    }

    /// Stored energy, the same proxy [`crate::string::PianoString::energy`]
    /// reports: what the segments are holding, whether or not it is loud enough
    /// to matter yet.
    pub fn energy(&self) -> f32 {
        self.bank.energy()
    }

    /// Frequency of segment `k`, for tests and reporting.
    pub fn mode_freq(&self, k: usize) -> f32 {
        self.bank.mode_freq(k)
    }

    /// Drives the segments with the key's own bridge force `own` and, when the
    /// bus has something to give, the resonance drive it has already computed;
    /// **adds** what they radiate into `out`.
    ///
    /// `own` is read before anything is added to `out`, so a segment is never
    /// driven by its own output within a block. Across blocks the only path
    /// back is through the bus, where the voice's own contribution is
    /// subtracted — see the loop bound in `Preset::validate_duplex`.
    pub fn add(&mut self, own: &[f32], drive: Option<&[f32]>, out: &mut [f32]) {
        debug_assert_eq!(own.len(), BLOCK);
        debug_assert_eq!(out.len(), BLOCK);
        if self.bank.is_empty() {
            return;
        }
        match drive {
            Some(drive) => {
                debug_assert_eq!(drive.len(), BLOCK);
                for ((i, &o), &d) in self.input.iter_mut().zip(own).zip(drive) {
                    *i = o + d;
                }
            }
            None => self.input.copy_from_slice(own),
        }
        self.bank.process_add(&self.input, out);
    }

    /// Immediate silence. Reached only from `AllOff`: no key and no pedal may
    /// stop a segment.
    pub fn reset(&mut self) {
        self.bank.reset_state();
        self.input.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preset::MIN_DUPLEX_T60_S;
    use crate::types::amp_to_db;

    fn mode(hz: f32, gain_db: f32, t60_s: f32) -> DuplexMode {
        DuplexMode { hz, gain_db, t60_s }
    }

    fn peak(v: &[f32]) -> f32 {
        v.iter().fold(0.0f32, |m, &x| m.max(x.abs()))
    }

    /// Drives a bank with a unit sinusoid at `hz` for `seconds` and returns the
    /// peak of the last block — the steady-state response, once the resonators
    /// have filled.
    fn steady_state(modes: &[DuplexMode], hz: f32, seconds: f32) -> f32 {
        let mut bank = DuplexBank::new(modes);
        let mut n = 0usize;
        let mut out = [0.0f32; BLOCK];
        let blocks = (seconds * SAMPLE_RATE / BLOCK as f32) as usize;
        for _ in 0..blocks {
            let mut own = [0.0f32; BLOCK];
            for x in own.iter_mut() {
                *x = (std::f32::consts::TAU * hz * n as f32 / SAMPLE_RATE).sin();
                n += 1;
            }
            out.fill(0.0);
            bank.add(&own, None, &mut out);
        }
        peak(&out)
    }

    /// The contract `gain_db` states: the response at the segment's own
    /// frequency, per unit of bridge force, and nothing else.
    #[test]
    fn a_segment_answers_its_own_frequency_at_the_gain_the_preset_asked_for() {
        for gain_db in [-40.0f32, -20.0, -6.0, 0.0, 6.0] {
            for hz in [220.0f32, 1_500.0, 4_400.0, 12_000.0] {
                let got = amp_to_db(steady_state(&[mode(hz, gain_db, 1.0)], hz, 8.0));
                assert!(
                    (got - gain_db).abs() < 0.5,
                    "a {gain_db} dB segment at {hz} Hz answered at {got:.2} dB"
                );
            }
        }
    }

    /// ... and it is that gain whatever the decay, which is the whole reason
    /// the input gain is normalised: level and length are separate parameters
    /// because they are separate measurements.
    #[test]
    fn the_level_of_a_segment_does_not_depend_on_how_long_it_rings() {
        let hz = 3_000.0;
        for t60 in [0.05f32, 0.2, 1.0, 3.0] {
            // Long enough to fill even the slowest resonator.
            let got = amp_to_db(steady_state(&[mode(hz, -12.0, t60)], hz, 8.0 + 4.0 * t60));
            assert!(
                (got + 12.0).abs() < 0.5,
                "a T60 of {t60} s moved the level to {got:.2} dB"
            );
        }
    }

    /// The segments ring for as long as the preset says, measured on the bank
    /// the engine builds rather than on the coefficient it was built from.
    #[test]
    fn a_segment_rings_for_the_t60_it_was_given() {
        for t60 in [0.2f32, 1.0] {
            let mut bank = DuplexBank::new(&[mode(2_000.0, 0.0, t60)]);
            let mut own = [0.0f32; BLOCK];
            own[0] = 1.0;
            let mut out = [0.0f32; BLOCK];
            bank.add(&own, None, &mut out);
            let first = peak(&out);
            // Half the T60 is 30 dB of decay, comfortably clear of the floor.
            let blocks = (0.5 * t60 * SAMPLE_RATE / BLOCK as f32) as usize;
            let silence = [0.0f32; BLOCK];
            for _ in 0..blocks {
                out.fill(0.0);
                bank.add(&silence, None, &mut out);
            }
            let after = peak(&out);
            let measured = amp_to_db(after / first);
            assert!(
                (measured + 30.0).abs() < 2.0,
                "a T60 of {t60} s lost {measured:.1} dB in half of it, expected -30"
            );
        }
    }

    /// The realised-response measurement the stability bound is computed from
    /// has to agree with what the bank actually does — including when a preset
    /// stacks segments on one frequency, which is the case the bound exists to
    /// refuse.
    #[test]
    fn the_measured_response_agrees_with_the_bank_and_sees_stacked_segments() {
        let hz = 2_500.0;
        let one = [mode(hz, 0.0, 1.0)];
        assert!((magnitude(&one, hz) - 1.0).abs() < 1.0e-3);
        assert!((steady_state(&one, hz, 8.0) - magnitude(&one, hz)).abs() < 0.05);

        // Six segments on one frequency answer six times as hard, which is the
        // shape of the loop the validator refuses.
        let stacked: Vec<DuplexMode> = (0..6).map(|_| mode(hz, 0.0, 1.0)).collect();
        assert!((magnitude(&stacked, hz) - 6.0).abs() < 0.01);
        assert!((steady_state(&stacked, hz, 8.0) - 6.0).abs() < 0.3);

        // Segments a real trichord's worth of scatter apart — 25 cents at
        // 2.5 kHz is 36 Hz, against a resonator whose bandwidth is under a
        // hertz — do not answer for each other at all.
        let scattered: Vec<DuplexMode> = (0..6)
            .map(|i| mode(hz * 2.0f32.powf(i as f32 * 25.0 / 1200.0), 0.0, 1.0))
            .collect();
        assert!(
            magnitude(&scattered, hz) < 1.1,
            "scattered segments crowded: {}",
            magnitude(&scattered, hz)
        );
    }

    /// A key with no segments is the case every preset shipped today is in: the
    /// bank must be empty, idle, and add exactly nothing.
    #[test]
    fn a_key_without_segments_costs_nothing_and_stays_idle() {
        let mut bank = DuplexBank::new(&[]);
        assert!(bank.is_empty());
        assert!(bank.is_idle());
        let mut out = [7.0f32; BLOCK];
        bank.add(&[1.0; BLOCK], Some(&[1.0; BLOCK]), &mut out);
        assert!(out.iter().all(|&x| x == 7.0), "an empty bank wrote samples");
        assert_eq!(magnitude(&[], 1_000.0), 0.0);
    }

    /// The bus path is a second input, not a second bank: driving with the bus
    /// alone is what makes a segment answer another key's note, and it reaches
    /// the same resonators the key's own force does.
    #[test]
    fn the_bus_drive_reaches_the_segments_on_its_own() {
        let hz = 3_300.0;
        let mut bank = DuplexBank::new(&[mode(hz, 0.0, 1.0)]);
        let mut out = [0.0f32; BLOCK];
        let mut drive = [0.0f32; BLOCK];
        drive[0] = 1.0;
        // Nothing from this key at all: the string is silent and damped.
        bank.add(&[0.0; BLOCK], Some(&drive), &mut out);
        assert!(peak(&out) > 0.0, "the bus drive did not reach the segments");
        assert!(!bank.is_idle());
        bank.reset();
        assert!(bank.is_idle(), "AllOff left the segments ringing");
    }

    /// A segment that is never driven again goes quiet and is culled, which is
    /// what lets its voice go back to sleep. Nothing else can stop it.
    #[test]
    fn a_segment_left_alone_decays_to_exact_silence() {
        let mut bank = DuplexBank::new(&[mode(4_000.0, 0.0, MIN_DUPLEX_T60_S)]);
        let mut own = [0.0f32; BLOCK];
        own[0] = 1.0;
        let mut out = [0.0f32; BLOCK];
        bank.add(&own, None, &mut out);
        assert!(!bank.is_idle());
        let silence = [0.0f32; BLOCK];
        // Ten T60s: past the culling floor for any level this bank can reach.
        for _ in 0..(10.0 * MIN_DUPLEX_T60_S * SAMPLE_RATE / BLOCK as f32) as usize {
            out.fill(0.0);
            bank.add(&silence, None, &mut out);
        }
        assert!(bank.is_idle(), "an undamped segment never went quiet");
        out.fill(0.0);
        bank.add(&silence, None, &mut out);
        assert!(
            out.iter().all(|&x| x == 0.0),
            "a culled segment still wrote"
        );
    }
}
