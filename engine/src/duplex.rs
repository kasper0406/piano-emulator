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
//! # What drives them (`DECISIONS.md` 481)
//!
//! Three things, and the **first** is what makes the feature audible at all:
//!
//! * the **strike burst** — the hammer's own force pulse, the broadband knock
//!   that crosses the bridge into the segment. A rear duplex is the *same wire*
//!   as the speaking length, continuous over the bridge, so what launches it is
//!   the travelling pulse the hammer put on the string, not the line spectrum
//!   the speaking length settles into afterwards. This is the drive the segments
//!   never had, and its absence is the whole of `DECISIONS.md` 260: a segment
//!   tuned tens of cents off a partial (which is what a real one *is*) sits
//!   between the only frequencies the old drive carried and was handed nothing
//!   to answer;
//! * the key's **own** bridge force — the mono sum the voice has just rendered,
//!   after the damper felt has had its say. This is the *aliquot* path: a
//!   segment nominally tuned to a partial of its own note sings along with it,
//!   which is what aliquot stringing is for;
//! * the **resonance bus**, through the bridge admittance, which is how a
//!   segment answers a note played on another key. It is the very same drive
//!   buffer the voice's strings are given (`resonance::ResonanceBus::drive`,
//!   `coupling · (B(bus) − own_gain · own_previous)`): there is one bus path in
//!   the engine and the segments read it rather than building a second one.
//!
//! The three are summed at the segment's *force* input, exactly as a string's
//! excitation buffer sums the hammer's pulse (newtons) and the bus drive (the
//! engine's signal unit); `resonance.rs` states that conflation and this module
//! inherits it rather than inventing a second convention.
//!
//! Their output joins the voice's mono sum, so it reaches the board at the
//! key's own pan and — being part of the sum — feeds the bus in its turn.
//!
//! # What `gain_db` means (`DECISIONS.md` 481)
//!
//! **A segment is built exactly as a partial of the speaking length is built.**
//! `string.rs`'s per-partial input gain is
//! `bridge_excitation_scale · comb · taper · shaping / SAMPLE_RATE` — an
//! *impulse* normalisation: the `1 / SAMPLE_RATE` turns the per-sample
//! accumulation of the excitation buffer into an integral over the hammer's
//! force pulse, so what the gain states is the mode's answer per newton-second
//! of impulse. [`DuplexMode::gain_db`] is the same thing for a segment, against
//! the same per-key scale and with the `k`-dependent factors gone, because the
//! strike comb and the contact taper are properties of where the hammer meets
//! the *speaking length* and the segment is beyond the bridge (the contact
//! taper reaches it anyway — it is in the pulse itself, which is what drives
//! the bank).
//!
//! So `gain_db` reads: **how hard this segment answers the hammer's knock,
//! relative to the key's own speaking length.** 0 dB is a segment excited
//! exactly as strongly as one of the note's partials would be with a flat comb;
//! −20 dB is one a tenth as strongly excited. Any constant transmission loss
//! across the bridge is inside that number, which is why there is no second
//! coupling constant to fit.
//!
//! Level and length stay independent, which is what makes the field estimable —
//! the tracker's peak levels and its T60s are separate measurements. Under an
//! impulse normalisation that is immediate: the peak of `g r^n cos(w n)` is `g`
//! whatever `r` is.
//!
//! **What this replaces, and why.** Until `DECISIONS.md` 481 `gain_db` was
//! normalised to the segment's *steady* response at its own frequency,
//! `g = 2 G (1 − r)` — a part in ten thousand at a 1.4 s decay. That is the
//! right convention for a resonator that is driven at resonance and the wrong
//! one for a resonator that is *struck*: it makes a pulse-driven segment's level
//! fall as `1/t60`, so a segment asked to ring twice as long came out 6 dB
//! quieter, and it put the whole bank 94 dB under where a measurement had put
//! it (`DUPLEX_LEVEL_OFFSET_DB`). The realised *resonant* response is now
//! `G · scale / (2 sigma)` instead — a derived quantity rather than the field —
//! and it is what the loop bound below is still computed from.
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
/// radians per sample, and the input gain.
///
/// `scale` is the key's [`bridge_excitation_scale_per_hz`], the same factor
/// `string.rs` builds its own partials' gains from; the segment's own frequency
/// supplies the modal mass, because a segment is a *short* piece of the same
/// wire and a shorter string answers one newton-second harder in exact
/// proportion to its fundamental. See the module header for what that makes
/// `gain_db` mean.
///
/// [`bridge_excitation_scale_per_hz`]: crate::string::bridge_excitation_scale_per_hz
fn resonator(mode: &DuplexMode, scale: f32) -> (f32, f32, f32) {
    let sigma = T60_DECADES / mode.t60_s;
    let r = (-sigma / SAMPLE_RATE).exp();
    let w = std::f32::consts::TAU * mode.hz / SAMPLE_RATE;
    (r, w, db_to_amp(mode.gain_db) * scale * mode.hz / SAMPLE_RATE)
}

/// Peak amplitude a segment reaches for one newton-second of impulse crossing
/// the bridge into it — i.e. `gain_db` in linear units, against the key's own
/// bridge scale. This is the quantity the round-trip gate reads back.
pub fn impulse_response(mode: &DuplexMode, scale: f32) -> f32 {
    let (_, _, g) = resonator(mode, scale);
    g * SAMPLE_RATE
}

/// Decay rate of a segment, 1/s, for the callers that build the bank.
pub fn sigma(mode: &DuplexMode) -> f32 {
    T60_DECADES / mode.t60_s
}

/// `|D(f)|` of a whole row of segments as *realised*: how much signal the row
/// puts out per unit of **steady** drive at `hz`.
///
/// This is the quantity the loop bound is computed from, and since
/// `DECISIONS.md` 481 it is a *derived* number rather than the schema field: at
/// a segment's own centre it is `G · scale / (2 sigma)`, so it rises with
/// `t60_s` exactly as a resonator's Q does. That is what makes the bound below
/// mean something — a preset that asks for a very long segment is asking for a
/// sharper resonance, and the validator sees it.
///
/// The modes' magnitudes are summed rather than their complex responses. That
/// is the conservative reading and it is the right one here, because this is
/// what the stability bound is computed from and the phases of two segments
/// that happen to land on one frequency are not something a preset controls.
pub fn magnitude(modes: &[DuplexMode], scale: f32, hz: f32) -> f32 {
    let w = std::f32::consts::TAU * hz / SAMPLE_RATE;
    modes
        .iter()
        .map(|mode| {
            let (r, wk, g) = resonator(mode, scale);
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
    /// `scale` is the key's [`bridge_excitation_scale_per_hz`]: the segments
    /// share the bridge and the wire with the speaking length, and `gain_db` is
    /// stated against exactly that.
    ///
    /// [`bridge_excitation_scale_per_hz`]: crate::string::bridge_excitation_scale_per_hz
    pub fn new(modes: &[DuplexMode], scale: f32) -> DuplexBank {
        let mut bank = ModalBank::with_capacity(modes.len());
        for mode in modes {
            let (_, _, g) = resonator(mode, scale);
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

    /// Drives the segments with the hammer's broadband strike `burst`, the
    /// key's own bridge force `own`, and — when the bus has something to give —
    /// the resonance drive the voice has already computed; **adds** what they
    /// radiate into `out`.
    ///
    /// `own` is read before anything is added to `out`, so a segment is never
    /// driven by its own output within a block. Across blocks the only path
    /// back is through the bus, where the voice's own contribution is
    /// subtracted — see the loop bound in `Preset::validate_duplex`. The burst
    /// is feed-forward from the hammer and closes no loop at all.
    pub fn add(
        &mut self,
        own: &[f32],
        drive: Option<&[f32]>,
        burst: Option<&[f32]>,
        out: &mut [f32],
    ) {
        debug_assert_eq!(own.len(), BLOCK);
        debug_assert_eq!(out.len(), BLOCK);
        if self.bank.is_empty() {
            return;
        }
        self.input.copy_from_slice(own);
        if let Some(drive) = drive {
            debug_assert_eq!(drive.len(), BLOCK);
            for (i, &d) in self.input.iter_mut().zip(drive) {
                *i += d;
            }
        }
        if let Some(burst) = burst {
            debug_assert_eq!(burst.len(), BLOCK);
            for (i, &b) in self.input.iter_mut().zip(burst) {
                *i += b;
            }
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

    /// Strikes a bank with one newton-second of impulse — a single sample of
    /// `SAMPLE_RATE`, which integrates to one — and returns the peak of the
    /// response over `seconds`. This is the drive `gain_db` is stated against.
    fn struck(modes: &[DuplexMode], scale: f32, seconds: f32) -> f32 {
        let mut bank = DuplexBank::new(modes, scale);
        let mut out = [0.0f32; BLOCK];
        let mut burst = [0.0f32; BLOCK];
        burst[0] = SAMPLE_RATE;
        bank.add(&[0.0; BLOCK], None, Some(&burst), &mut out);
        let mut best = peak(&out);
        let silence = [0.0f32; BLOCK];
        for _ in 0..(seconds * SAMPLE_RATE / BLOCK as f32) as usize {
            out.fill(0.0);
            bank.add(&silence, None, None, &mut out);
            best = best.max(peak(&out));
        }
        best
    }

    /// Drives a bank with a unit sinusoid at `hz` for `seconds` and returns the
    /// peak of the last block — the steady-state response, once the resonators
    /// have filled. Since `DECISIONS.md` 481 this is a *derived* quantity (the
    /// segment's Q), not the field, and it is what the loop bound reads.
    fn steady_state(modes: &[DuplexMode], scale: f32, hz: f32, seconds: f32) -> f32 {
        let mut bank = DuplexBank::new(modes, scale);
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
            bank.add(&own, None, None, &mut out);
        }
        peak(&out)
    }

    /// The contract `gain_db` states since `DECISIONS.md` 481: the segment's
    /// answer to the hammer's knock, per newton-second across the bridge,
    /// against the key's own bridge scale and the segment's own modal mass —
    /// and nothing else.
    #[test]
    fn a_segment_answers_the_hammers_knock_at_the_gain_the_preset_asked_for() {
        for gain_db in [-40.0f32, -20.0, -6.0, 0.0, 6.0] {
            for hz in [220.0f32, 1_500.0, 4_400.0, 12_000.0] {
                for scale in [0.5f32, 1.0, 2.0] {
                    let got =
                        amp_to_db(struck(&[mode(hz, gain_db, 1.0)], scale, 0.05) / (scale * hz));
                    assert!(
                        (got - gain_db).abs() < 0.5,
                        "a {gain_db} dB segment at {hz} Hz on a {scale} bridge answered at \
                         {got:.2} dB"
                    );
                }
            }
        }
    }

    /// ... and it is that gain whatever the decay. This is the property the old
    /// normalisation did not have and the reason the field was re-decided: under
    /// `g = 2 G (1 - r)` a struck segment's level fell as `1 / t60`, so a
    /// segment asked to ring twice as long came out 6 dB quieter.
    #[test]
    fn the_level_of_a_struck_segment_does_not_depend_on_how_long_it_rings() {
        let hz = 3_000.0;
        for t60 in [0.05f32, 0.2, 1.0, 3.0] {
            let got = amp_to_db(struck(&[mode(hz, -12.0, t60)], 1.0, 0.05) / hz);
            assert!(
                (got + 12.0).abs() < 0.5,
                "a T60 of {t60} s moved the level to {got:.2} dB"
            );
        }
    }

    /// The realised *resonant* response — the derived quantity, which is what
    /// the loop bound is computed from — is the gain over the mode's own
    /// bandwidth, so it rises with `t60_s` as a Q does.
    #[test]
    fn the_resonant_response_is_the_gain_over_the_segments_own_bandwidth() {
        let hz = 2_500.0;
        for t60 in [0.2f32, 1.0, 3.0] {
            let one = [mode(hz, 0.0, t60)];
            let predicted = hz / (2.0 * sigma(&one[0]));
            assert!(
                (magnitude(&one, 1.0, hz) / predicted - 1.0).abs() < 0.01,
                "t60 {t60}: magnitude {} against the predicted {predicted}",
                magnitude(&one, 1.0, hz)
            );
            let rendered = steady_state(&one, 1.0, hz, 6.0 + 4.0 * t60);
            assert!(
                (rendered / predicted - 1.0).abs() < 0.05,
                "t60 {t60}: the bank realises {rendered} where the bound reads {predicted}"
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
        let single = magnitude(&one, 1.0, hz);
        assert!((steady_state(&one, 1.0, hz, 8.0) / single - 1.0).abs() < 0.05);

        // Six segments on one frequency answer six times as hard, which is the
        // shape of the loop the validator refuses.
        let stacked: Vec<DuplexMode> = (0..6).map(|_| mode(hz, 0.0, 1.0)).collect();
        assert!((magnitude(&stacked, 1.0, hz) / (6.0 * single) - 1.0).abs() < 0.01);
        assert!((steady_state(&stacked, 1.0, hz, 8.0) / (6.0 * single) - 1.0).abs() < 0.05);

        // Segments a real trichord's worth of scatter apart — 25 cents at
        // 2.5 kHz is 36 Hz, against a resonator whose bandwidth is under a
        // hertz — do not answer for each other at all.
        let scattered: Vec<DuplexMode> = (0..6)
            .map(|i| mode(hz * 2.0f32.powf(i as f32 * 25.0 / 1200.0), 0.0, 1.0))
            .collect();
        assert!(
            magnitude(&scattered, 1.0, hz) < 1.1 * single,
            "scattered segments crowded: {}",
            magnitude(&scattered, 1.0, hz)
        );
    }

    /// The finding `DECISIONS.md` 260 named, as a unit test on the mechanism.
    ///
    /// What separates the two drives is **bandwidth**, not level, so that is
    /// what this reads: the same segment is tuned onto a partial and then walked
    /// off it, and each drive is scored by how much it loses on the way. A knock
    /// loses almost nothing — it has energy everywhere — while the note's own
    /// bridge force falls off a cliff, which is why a real duplex, deliberately
    /// tuned tens of cents sharp, was handed nothing by the old path.
    ///
    /// The absolute gap between the two at the levels the engine actually runs
    /// them at is +48 dB at 52 cents on C5, and it is measured outside the unit
    /// tests, by `forensics/duplex_drive`, because it needs a hammer and a
    /// string.
    #[test]
    fn only_the_strike_burst_reaches_a_segment_tuned_off_the_notes_partials() {
        const PARTIAL_HZ: f32 = 2_616.0;
        let at = |cents: f32| PARTIAL_HZ * (cents / 1200.0).exp2();

        // The knock: a millisecond of contact, the hammer's own.
        let contact = (0.001 * SAMPLE_RATE) as usize;
        let mut burst = [0.0f32; BLOCK];
        for (i, b) in burst.iter_mut().take(contact).enumerate() {
            *b = (std::f32::consts::PI * i as f32 / contact as f32).sin();
        }
        let knock = |hz: f32| -> f32 {
            let mut bank = DuplexBank::new(&[mode(hz, 0.0, 1.4)], 1.0);
            let mut out = [0.0f32; BLOCK];
            bank.add(&[0.0; BLOCK], None, Some(&burst), &mut out);
            let mut best = peak(&out);
            let silence = [0.0f32; BLOCK];
            for _ in 0..(0.05 * SAMPLE_RATE / BLOCK as f32) as usize {
                out.fill(0.0);
                bank.add(&silence, None, None, &mut out);
                best = best.max(peak(&out));
            }
            best
        };

        // The old drive: the note's own bridge force, one decaying partial.
        let line = |hz: f32| -> f32 {
            let mut bank = DuplexBank::new(&[mode(hz, 0.0, 1.4)], 1.0);
            let mut out = [0.0f32; BLOCK];
            let mut best = 0.0f32;
            let mut n = 0usize;
            for _ in 0..(2.0 * SAMPLE_RATE / BLOCK as f32) as usize {
                let mut own = [0.0f32; BLOCK];
                for x in own.iter_mut() {
                    let t = n as f32 / SAMPLE_RATE;
                    *x = (-3.0 * t).exp()
                        * (std::f32::consts::TAU * PARTIAL_HZ * t).sin();
                    n += 1;
                }
                out.fill(0.0);
                bank.add(&own, None, None, &mut out);
                best = best.max(peak(&out));
            }
            best
        };

        let knock_loss = amp_to_db(knock(at(52.0)) / knock(at(0.0)));
        let line_loss = amp_to_db(line(at(52.0)) / line(at(0.0)));
        assert!(
            knock_loss > -1.0,
            "the knock lost {knock_loss:.1} dB over 52 cents — it is not broadband"
        );
        assert!(
            line_loss < -20.0,
            "the note\'s own partial reached 52 cents off itself down only \
             {line_loss:.1} dB, which is not the drive DECISIONS.md 157 measured"
        );
    }

    /// The segments ring for as long as the preset says, measured on the bank
    /// the engine builds rather than on the coefficient it was built from.
    #[test]
    fn a_segment_rings_for_the_t60_it_was_given() {
        for t60 in [0.2f32, 1.0] {
            let mut bank = DuplexBank::new(&[mode(2_000.0, 0.0, t60)], 1.0);
            let mut burst = [0.0f32; BLOCK];
            burst[0] = SAMPLE_RATE;
            let mut out = [0.0f32; BLOCK];
            bank.add(&[0.0; BLOCK], None, Some(&burst), &mut out);
            let first = peak(&out);
            // Half the T60 is 30 dB of decay, comfortably clear of the floor.
            let blocks = (0.5 * t60 * SAMPLE_RATE / BLOCK as f32) as usize;
            let silence = [0.0f32; BLOCK];
            for _ in 0..blocks {
                out.fill(0.0);
                bank.add(&silence, None, None, &mut out);
            }
            let after = peak(&out);
            let measured = amp_to_db(after / first);
            assert!(
                (measured + 30.0).abs() < 2.0,
                "a T60 of {t60} s lost {measured:.1} dB in half of it, expected -30"
            );
        }
    }

    /// A key with no segments is the case every preset shipped today is in: the
    /// bank must be empty, idle, and add exactly nothing.
    #[test]
    fn a_key_without_segments_costs_nothing_and_stays_idle() {
        let mut bank = DuplexBank::new(&[], 1.0);
        assert!(bank.is_empty());
        assert!(bank.is_idle());
        let mut out = [7.0f32; BLOCK];
        bank.add(&[1.0; BLOCK], Some(&[1.0; BLOCK]), Some(&[1.0; BLOCK]), &mut out);
        assert!(out.iter().all(|&x| x == 7.0), "an empty bank wrote samples");
        assert_eq!(magnitude(&[], 1.0, 1_000.0), 0.0);
    }

    /// The bus path is a second input, not a second bank: driving with the bus
    /// alone is what makes a segment answer another key's note, and it reaches
    /// the same resonators the key's own force does.
    #[test]
    fn the_bus_drive_reaches_the_segments_on_its_own() {
        let hz = 3_300.0;
        let mut bank = DuplexBank::new(&[mode(hz, 0.0, 1.0)], 1.0);
        let mut out = [0.0f32; BLOCK];
        let mut drive = [0.0f32; BLOCK];
        drive[0] = 1.0;
        // Nothing from this key at all: the string is silent and damped.
        bank.add(&[0.0; BLOCK], Some(&drive), None, &mut out);
        assert!(peak(&out) > 0.0, "the bus drive did not reach the segments");
        assert!(!bank.is_idle());
        bank.reset();
        assert!(bank.is_idle(), "AllOff left the segments ringing");
    }

    /// A segment that is never driven again goes quiet and is culled, which is
    /// what lets its voice go back to sleep. Nothing else can stop it.
    #[test]
    fn a_segment_left_alone_decays_to_exact_silence() {
        let mut bank = DuplexBank::new(&[mode(4_000.0, 0.0, MIN_DUPLEX_T60_S)], 1.0);
        let mut burst = [0.0f32; BLOCK];
        burst[0] = SAMPLE_RATE;
        let mut out = [0.0f32; BLOCK];
        bank.add(&[0.0; BLOCK], None, Some(&burst), &mut out);
        assert!(!bank.is_idle());
        let silence = [0.0f32; BLOCK];
        // Ten T60s: past the culling floor for any level this bank can reach.
        for _ in 0..(10.0 * MIN_DUPLEX_T60_S * SAMPLE_RATE / BLOCK as f32) as usize {
            out.fill(0.0);
            bank.add(&silence, None, None, &mut out);
        }
        assert!(bank.is_idle(), "an undamped segment never went quiet");
        out.fill(0.0);
        bank.add(&silence, None, None, &mut out);
        assert!(
            out.iter().all(|&x| x == 0.0),
            "a culled segment still wrote"
        );
    }
}
