//! Sympathetic resonance bus.
//!
//! Every string's output is summed into a mono bus, and each *undamped* string
//! is driven by `coupling * (bus - own_contribution)`. Subtracting a string's
//! own contribution is not an optimization — without it each string feeds back
//! into itself, which is indistinguishable from reducing its damping, and the
//! decay time drifts with how many other strings happen to be ringing.
//!
//! The bus a string reads is the previous block's sum. One block (2.7 ms) of
//! latency in a diffuse coupling path is inaudible, and it removes the
//! circular dependency between "sum all strings" and "drive all strings".
//!
//! # Stability
//!
//! The bus carries the engine's signal unit — bridge force in newtons times
//! the per-note output scale in `string.rs` — and the strings are driven at
//! their force input. Driven steadily at one of its partials, a string answers
//! with at most `output_scale * sin(k pi x) / sigma_k` of signal: roughly 1 for
//! the slowest bass partials and far less everywhere else. A loop that runs
//! through `m` mutually coincident partials therefore has gain
//! `~ m * coupling`, so the default coupling is roughly two orders of
//! magnitude below the point where a realistic cluster of coincidences could
//! sustain itself; [`MAX_COUPLING`] keeps a caller from tuning past it.
//!
//! That argument depends on the string parameters, so it is backed by a hard
//! guarantee that does not: the drive is clamped to [`DRIVE_CEILING`]. Every
//! modal pole is strictly inside the unit circle, so a bounded excitation can
//! only produce a bounded output — with the clamp in place the coupling loop
//! cannot diverge no matter how the strings are retuned.

use crate::types::{BLOCK, CULL_AMPLITUDE};

/// Largest coupling [`ResonanceBus::set_coupling`] will accept. Well above the
/// spec's 0.005-0.03 range and well below the loop gain analysed above.
pub const MAX_COUPLING: f32 = 0.05;

/// Hard ceiling on the coupling drive, in newtons. Three orders of magnitude
/// above anything the instrument produces in normal playing, so it never
/// colours the sound; it exists so that boundedness is a property of the code
/// rather than of the current parameter tables.
const DRIVE_CEILING: f32 = 1.0;

/// Largest signal a string answers a steady drive at one of its partials with
/// (see the stability discussion above): about 1 for the slowest bass partials
/// and less everywhere else, so 0.5 is a serviceable middle figure. Used only
/// to decide when the bus is too quiet to wake a silent string — see
/// [`ResonanceBus::is_active`].
const MAX_STRING_ADMITTANCE: f32 = 0.5;

pub struct ResonanceBus {
    bus: [f32; BLOCK],
    accum: [f32; BLOCK],
    coupling: f32,
    /// Peak absolute value of `bus`, kept so the engine can decide cheaply
    /// whether a silent voice is worth waking.
    peak: f32,
}

impl ResonanceBus {
    /// `coupling` is the preset's `resonance_coupling`, clamped to the stable
    /// range like every later change to it.
    pub fn new(coupling: f32) -> Self {
        let mut bus = ResonanceBus {
            bus: [0.0; BLOCK],
            accum: [0.0; BLOCK],
            coupling: 0.0,
            peak: 0.0,
        };
        bus.set_coupling(coupling);
        bus
    }

    pub fn coupling(&self) -> f32 {
        self.coupling
    }

    /// Sets the coupling, clamped to `0..=MAX_COUPLING`. A value that is not a
    /// number silences the bus rather than passing through — `f32::clamp`
    /// returns NaN for NaN, and a NaN here would reach every undamped string.
    pub fn set_coupling(&mut self, coupling: f32) {
        self.coupling = if coupling.is_finite() {
            coupling.clamp(0.0, MAX_COUPLING)
        } else {
            0.0
        };
    }

    /// Publishes the block that just finished as the bus the next block reads.
    pub fn begin_block(&mut self) {
        self.bus.copy_from_slice(&self.accum);
        self.accum.fill(0.0);
        self.peak = self.bus.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    }

    /// True when the bus carries enough to make an otherwise silent undamped
    /// string audible, so the engine has to render it. Below this the coupled
    /// drive `coupling * bus` cannot lift any mode past [`CULL_AMPLITUDE`] even
    /// at a partial's exact centre frequency, and every silent voice can be
    /// skipped — which is what keeps a pedal-down chord from costing the same
    /// as the full 88-key worst case.
    pub fn is_active(&self) -> bool {
        self.peak * self.coupling * MAX_STRING_ADMITTANCE > CULL_AMPLITUDE
    }

    /// Bus contents visible during the current block.
    pub fn bus(&self) -> &[f32] {
        &self.bus
    }

    /// Adds one string's output for the current block.
    pub fn contribute(&mut self, block: &[f32]) {
        debug_assert_eq!(block.len(), BLOCK);
        for (a, &x) in self.accum.iter_mut().zip(block) {
            *a += x;
        }
    }

    /// Adds `coupling * (bus - own_previous)` into `out`, where `own_previous`
    /// is the same string's output during the block the bus was summed from.
    pub fn drive(&self, own_previous: &[f32], out: &mut [f32]) {
        debug_assert_eq!(own_previous.len(), BLOCK);
        debug_assert_eq!(out.len(), BLOCK);
        for i in 0..BLOCK {
            let rest = self.bus[i] - own_previous[i];
            out[i] += (self.coupling * rest).clamp(-DRIVE_CEILING, DRIVE_CEILING);
        }
    }

    pub fn reset(&mut self) {
        self.bus.fill(0.0);
        self.accum.fill(0.0);
        self.peak = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bus coupled as the instrument really runs it.
    fn bus() -> ResonanceBus {
        ResonanceBus::new(Preset::default().voicing.resonance_coupling)
    }
    use crate::modal::ModalBank;
    use crate::preset::Preset;
    use crate::types::SAMPLE_RATE;

    /// A one-partial stand-in for a string: same pole and same input gain
    /// convention (`1/(2 Z f_s)`) as `string.rs` builds, so the loop gains in
    /// these tests are the ones the instrument really runs at.
    fn partial(key: u8, detune_hz: f32) -> ModalBank {
        let p = Preset::default().string_params(key);
        let mut bank = ModalBank::with_capacity(1);
        bank.push_mode(
            p.f0 + detune_hz,
            p.partial_sigma(1),
            1.0 / (2.0 * p.impedance * SAMPLE_RATE),
        );
        bank
    }

    fn peak(v: &[f32]) -> f32 {
        v.iter().fold(0.0f32, |m, x| m.max(x.abs()))
    }

    #[test]
    fn bus_lags_by_one_block() {
        let mut r = bus();
        let mut a = [0.0f32; BLOCK];
        a[0] = 1.0;
        r.contribute(&a);
        assert_eq!(r.bus()[0], 0.0);
        r.begin_block();
        assert_eq!(r.bus()[0], 1.0);
        r.begin_block();
        assert_eq!(r.bus()[0], 0.0);
    }

    #[test]
    fn a_lone_string_drives_itself_with_nothing() {
        let mut r = bus();
        let mut own = [0.0f32; BLOCK];
        own[3] = 0.5;
        r.contribute(&own);
        r.begin_block();
        let mut out = [0.0f32; BLOCK];
        r.drive(&own, &mut out);
        assert!(out.iter().all(|&v| v.abs() < 1e-9));
    }

    #[test]
    fn coupling_is_clamped_to_the_stable_range() {
        let mut r = bus();
        r.set_coupling(10.0);
        assert_eq!(r.coupling(), MAX_COUPLING);
        r.set_coupling(-1.0);
        assert_eq!(r.coupling(), 0.0);
    }

    #[test]
    fn drive_is_bounded_however_loud_the_bus_gets() {
        let mut r = bus();
        let huge = [1.0e12f32; BLOCK];
        r.contribute(&huge);
        r.begin_block();
        let mut out = [0.0f32; BLOCK];
        r.drive(&[0.0; BLOCK], &mut out);
        assert!(out.iter().all(|&v| v.abs() <= DRIVE_CEILING));
    }

    /// Two co-tuned strings coupled through the bus at the maximum coupling:
    /// the worst realistic case for the loop gain, run for ten seconds.
    #[test]
    fn coincident_strings_stay_bounded_for_ten_seconds() {
        let mut r = bus();
        r.set_coupling(MAX_COUPLING);
        let mut banks = [partial(21, 0.0), partial(21, 0.0)];
        let mut previous = [[0.0f32; BLOCK]; 2];

        // One hammer-sized impulse into the first string, then nothing.
        let mut excite = [0.0f32; BLOCK];
        excite[0] = 100.0;

        let (mut early, mut late) = (0.0f32, 0.0f32);
        let blocks = (10.0 * SAMPLE_RATE / BLOCK as f32) as usize;
        let second = (SAMPLE_RATE / BLOCK as f32) as usize;
        for b in 0..blocks {
            r.begin_block();
            for (i, bank) in banks.iter_mut().enumerate() {
                let mut input = [0.0f32; BLOCK];
                if b == 0 && i == 0 {
                    input.copy_from_slice(&excite);
                }
                r.drive(&previous[i], &mut input);
                let mut out = [0.0f32; BLOCK];
                bank.process_add(&input, &mut out);
                r.contribute(&out);
                previous[i].copy_from_slice(&out);
            }
            let block_peak = peak(&previous[0]).max(peak(&previous[1]));
            assert!(block_peak.is_finite(), "diverged at block {b}");
            if b < second {
                early = early.max(block_peak);
            } else if b >= blocks - second {
                late = late.max(block_peak);
            }
        }
        assert!(early > 0.0);
        // The pair exchanges energy but the coupling adds none: after ten
        // seconds of a 25 s T60 the pair must be quieter, not louder.
        assert!(late < early, "grew from {early} to {late}");
    }

    /// The point of the whole module: a string that was never struck must pick
    /// up energy from one that was.
    #[test]
    fn an_unstruck_string_picks_up_the_bus() {
        let mut r = bus();
        // C4 and a second string a fifth of a Hz away — the unison-style near
        // coincidence that produces the strongest halo.
        let mut banks = [partial(60, 0.0), partial(60, 0.2)];
        let mut previous = [[0.0f32; BLOCK]; 2];

        let blocks = (1.0 * SAMPLE_RATE / BLOCK as f32) as usize;
        for b in 0..blocks {
            r.begin_block();
            for (i, bank) in banks.iter_mut().enumerate() {
                let mut input = [0.0f32; BLOCK];
                if b == 0 && i == 0 {
                    input[0] = 100.0;
                }
                r.drive(&previous[i], &mut input);
                let mut out = [0.0f32; BLOCK];
                bank.process_add(&input, &mut out);
                r.contribute(&out);
                previous[i].copy_from_slice(&out);
            }
        }
        let (struck, halo) = (peak(&previous[0]), peak(&previous[1]));
        assert!(halo > 0.0, "no sympathetic response at all");
        assert!(halo < struck, "halo {halo} is not quieter than {struck}");
    }
}
