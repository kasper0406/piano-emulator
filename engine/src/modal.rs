//! Modal resonator bank — the primitive every sounding object is built from.
//!
//! Each mode `k` is a complex one-pole driven by a common real input `x[n]`:
//!
//! ```text
//! s_k[n] = a_k * s_k[n-1] + g_k * x[n],   a_k = r_k * e^(i*w_k)
//! y[n]   = sum_k Im(s_k[n])
//! r_k    = exp(-sigma_k / SAMPLE_RATE),   w_k = 2*pi*f_k / SAMPLE_RATE
//! ```
//!
//! The input gain `g_k` is **complex**. A bank whose modes are the free modes of
//! one object — the duplex segments, the board, one uncoupled string — only ever
//! needs a real one, and [`ModalBank::push_mode`] is that case written as
//! `g_im = 0`. What needs the imaginary half is a bank whose modes are the
//! *eigenmodes of a coupled group*: the strike projects onto a non-orthogonal
//! eigenbasis, so each mode starts at its own phase, and that phase is the
//! difference between a unison that beats once and settles and one that beats
//! forever (`docs/history/FUNDAMENTALS.md` §5.2, `engine::string`). It costs one FMA per mode
//! per sample in [`Chunk::step`].
//!
//! State is stored SoA. The recurrence is serial along the sample axis, so the
//! block loop runs [`LANES`] modes at a time with the state held in registers:
//! the mode axis is the one that vectorizes.

use crate::types::{CULL_AMPLITUDE, IDLE_ENERGY, SAMPLE_RATE};
use std::ops::Range;

/// Modes processed simultaneously by the inner loop — one NEON f32 vector. The
/// mode arrays are padded to a multiple of this with silent entries
/// (`a = 0`, `g = 0`) so the block loop never needs a tail case.
const LANES: usize = 8;

/// One vector of resonators, held in registers for the length of a block.
struct Chunk {
    re: [f32; LANES],
    im: [f32; LANES],
    a_re: [f32; LANES],
    a_im: [f32; LANES],
    g_re: [f32; LANES],
    g_im: [f32; LANES],
}

impl Chunk {
    /// One sample of the complex recurrence across the lanes; returns the
    /// summed imaginary part, i.e. this chunk's contribution to `y[n]`.
    #[inline(always)]
    fn step(&mut self, x: f32) -> f32 {
        let mut acc = 0.0;
        for l in 0..LANES {
            let (re, im) = (self.re[l], self.im[l]);
            let next_re = self.a_re[l] * re - self.a_im[l] * im + self.g_re[l] * x;
            let next_im = self.a_re[l] * im + self.a_im[l] * re + self.g_im[l] * x;
            self.re[l] = next_re;
            self.im[l] = next_im;
            acc += next_im;
        }
        acc
    }

    /// The same step with no input at all: `s <- a s`.
    ///
    /// Most of what a piano is doing at any instant is *ringing*, not being
    /// driven — a hammer pulse is two milliseconds and a note is seconds — so
    /// this is the case that decides the budget, and it is four multiplies and
    /// three adds per lane against the driven six and five. It is also exactly
    /// the recurrence, not an approximation of it: `x` is zero, so the two gain
    /// terms are zero.
    #[inline(always)]
    fn step_free(&mut self) -> f32 {
        let mut acc = 0.0;
        for l in 0..LANES {
            let (re, im) = (self.re[l], self.im[l]);
            let next_re = self.a_re[l] * re - self.a_im[l] * im;
            let next_im = self.a_re[l] * im + self.a_im[l] * re;
            self.re[l] = next_re;
            self.im[l] = next_im;
            acc += next_im;
        }
        acc
    }
}

pub struct ModalBank {
    // Resonator state, SoA.
    re: Vec<f32>,
    im: Vec<f32>,
    // Current per-sample pole a_k = r_cur * e^(i*w_k).
    a_re: Vec<f32>,
    a_im: Vec<f32>,
    // Input gains, complex: `g_im` is zero for every bank whose modes are the
    // free modes of one object, and non-zero only where the modes are the
    // eigenmodes of a coupled group.
    g_re: Vec<f32>,
    g_im: Vec<f32>,
    // Unit phasor e^(i*w_k), kept so the pole can be rebuilt from a new radius.
    cos_w: Vec<f32>,
    sin_w: Vec<f32>,
    // Mode descriptions.
    freq: Vec<f32>,
    sigma_base: Vec<f32>,
    sigma_extra: Vec<f32>,
    // Pole radius: undamped, current, and the value `sigma_base + sigma_extra` implies.
    r_base: Vec<f32>,
    r_cur: Vec<f32>,
    r_tgt: Vec<f32>,
    /// Modes in use; the vectors above are padded up to a multiple of `LANES`.
    len: usize,
    /// Some mode's radius has not reached its target yet.
    ramping: bool,
}

impl ModalBank {
    /// Allocates room for `max_modes`. All later mutation is allocation-free as
    /// long as the mode count stays within this capacity.
    pub fn with_capacity(max_modes: usize) -> Self {
        let n = padded_len(max_modes);
        let z = || Vec::with_capacity(n);
        ModalBank {
            re: z(),
            im: z(),
            a_re: z(),
            a_im: z(),
            g_re: z(),
            g_im: z(),
            cos_w: z(),
            sin_w: z(),
            freq: z(),
            sigma_base: z(),
            sigma_extra: z(),
            r_base: z(),
            r_cur: z(),
            r_tgt: z(),
            len: 0,
            ramping: false,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn capacity(&self) -> usize {
        self.re.capacity()
    }

    /// Drops all modes, keeping the allocation.
    pub fn clear(&mut self) {
        for v in self.arrays_mut() {
            v.clear();
        }
        self.len = 0;
        self.ramping = false;
    }

    /// Appends a mode: `freq_hz` centre frequency, `sigma` decay rate in 1/s
    /// (T60 = 6.91 / sigma), `gain` the input gain g_k. Real, i.e. the mode
    /// starts at the phase the input hands it.
    pub fn push_mode(&mut self, freq_hz: f32, sigma: f32, gain: f32) {
        self.push_mode_complex(freq_hz, sigma, gain, 0.0);
    }

    /// Appends a mode with a complex input gain `g_re + i g_im`.
    pub fn push_mode_complex(&mut self, freq_hz: f32, sigma: f32, g_re: f32, g_im: f32) {
        let k = self.len;
        if k == self.re.len() {
            for v in self.arrays_mut() {
                v.resize(v.len() + LANES, 0.0);
            }
        }
        self.len = k + 1;
        self.re[k] = 0.0;
        self.im[k] = 0.0;
        self.sigma_extra[k] = 0.0;
        self.write_mode(k, freq_hz, sigma, g_re, g_im);
    }

    /// Redefines mode `k` in place, keeping its resonator state (a retuned
    /// string keeps ringing rather than restarting) and its extra damping.
    pub fn set_mode(&mut self, k: usize, freq_hz: f32, sigma: f32, gain: f32) {
        debug_assert!(k < self.len);
        self.write_mode(k, freq_hz, sigma, gain, 0.0);
    }

    pub fn mode_freq(&self, k: usize) -> f32 {
        self.freq[k]
    }

    pub fn mode_sigma(&self, k: usize) -> f32 {
        self.sigma_base[k]
    }

    /// The modulus of mode `k`'s pole, `|a| = r`, as the recurrence will
    /// actually use it — after the `f32` rounding of `exp(-sigma/SR)` and after
    /// any extra damping.
    ///
    /// A resonator is stable iff this is strictly under one. It is `pub` so
    /// that the construction that builds the eigenmodes can be *tested* on the
    /// property rather than on a proxy for it: `sigma > 0` is the mathematical
    /// condition, `r < 1` is the arithmetic one, and at 48 kHz they are not the
    /// same condition — every `sigma` under about `5.7e-3` rounds to `r = 1`
    /// and rings forever (`string.rs::MIN_MODE_SIGMA`).
    pub fn pole_radius(&self, k: usize) -> f32 {
        self.r_cur[k]
    }

    pub fn mode_gain(&self, k: usize) -> f32 {
        self.g_re[k]
    }

    /// Mode `k`'s present state magnitude `|s_k|` — the peak amplitude it still
    /// contributes to the bank's output, and the quantity [`ModalBank::cull`]
    /// tests.
    ///
    /// `pub` for the same reason [`ModalBank::pole_radius`] is: a threshold that
    /// decides when a note ends has to be checkable against the thing it is
    /// applied to rather than against a proxy for it
    /// (`forensics/src/bin/top_octave.rs`, `DECISIONS.md` 275-276).
    pub fn mode_amplitude(&self, k: usize) -> f32 {
        self.re[k].hypot(self.im[k])
    }

    /// The imaginary half of mode `k`'s input gain; zero on every bank built
    /// with [`ModalBank::push_mode`].
    pub fn mode_gain_im(&self, k: usize) -> f32 {
        self.g_im[k]
    }

    /// Rewrites mode `k`'s complex input gain, leaving its pole and its state
    /// alone — how a strike vector that has changed direction (una corda) is
    /// applied to a group whose eigenmodes have not.
    pub fn set_mode_gain_complex(&mut self, k: usize, g_re: f32, g_im: f32) {
        self.g_re[k] = g_re;
        self.g_im[k] = g_im;
    }

    /// Silences the resonators without changing the mode layout.
    pub fn reset_state(&mut self) {
        self.re.iter_mut().for_each(|v| *v = 0.0);
        self.im.iter_mut().for_each(|v| *v = 0.0);
    }

    /// Adds `extra_sigma` (1/s) to the base decay rate of the modes in
    /// `k_range`. Cheap: only the pole radius is recomputed, and only when the
    /// requested damping actually changed. The change is ramped in over one
    /// block by `process_add`.
    ///
    /// Negative values are clamped away: a mode may only ever be made to decay
    /// faster, never driven towards instability.
    pub fn set_damping_scale(&mut self, k_range: Range<usize>, extra_sigma: f32) {
        let extra = extra_sigma.max(0.0);
        for k in k_range.start..k_range.end.min(self.len) {
            self.write_damping(k, extra);
        }
    }

    /// Per-mode form of [`set_damping_scale`], for damper profiles that weight
    /// each partial differently. Extra entries beyond the bank are ignored.
    pub fn set_damping_profile(&mut self, extra_sigma: &[f32]) {
        let n = self.len.min(extra_sigma.len());
        for (k, &extra) in extra_sigma[..n].iter().enumerate() {
            self.write_damping(k, extra.max(0.0));
        }
    }

    /// Current extra damping on mode `k`, in 1/s.
    pub fn damping_scale(&self, k: usize) -> f32 {
        self.sigma_extra[k]
    }

    /// Runs one block. `input` is the common excitation x[n]; the summed mode
    /// output is **added** into `out`, which lets a caller accumulate several
    /// banks into one buffer without a scratch copy.
    pub fn process_add(&mut self, input: &[f32], out: &mut [f32]) {
        debug_assert_eq!(input.len(), out.len());
        let n = input.len();
        if n == 0 || self.len == 0 {
            return;
        }
        // Nothing drives the bank this block, so every mode can only get
        // quieter: the ones already below audibility may be dropped.
        let silent_input = input.iter().all(|&x| x == 0.0);
        let ramping = self.ramping;
        let inv_n = 1.0 / n as f32;

        for base in (0..self.re.len()).step_by(LANES) {
            if silent_input && self.cull(base) {
                continue;
            }

            let mut c = Chunk {
                re: read4(&self.re, base),
                im: read4(&self.im, base),
                a_re: read4(&self.a_re, base),
                a_im: read4(&self.a_im, base),
                g_re: read4(&self.g_re, base),
                g_im: read4(&self.g_im, base),
            };

            if ramping {
                // A step in pole radius is audible as a click, so the radius
                // slides linearly across the block to its new value.
                let mut d_re = [0.0f32; LANES];
                let mut d_im = [0.0f32; LANES];
                for l in 0..LANES {
                    let dr = (self.r_tgt[base + l] - self.r_cur[base + l]) * inv_n;
                    d_re[l] = dr * self.cos_w[base + l];
                    d_im[l] = dr * self.sin_w[base + l];
                }
                for (o, &x) in out.iter_mut().zip(input) {
                    for l in 0..LANES {
                        c.a_re[l] += d_re[l];
                        c.a_im[l] += d_im[l];
                    }
                    *o += c.step(x);
                }
                for l in 0..LANES {
                    self.snap_pole(base + l);
                }
            } else if silent_input {
                for o in out.iter_mut() {
                    *o += c.step_free();
                }
            } else {
                for (o, &x) in out.iter_mut().zip(input) {
                    *o += c.step(x);
                }
            }

            self.re[base..base + LANES].copy_from_slice(&c.re);
            self.im[base..base + LANES].copy_from_slice(&c.im);
        }
        self.ramping = false;
    }

    /// Sum of |s_k|^2 over the bank — a cheap proxy for stored energy.
    pub fn energy(&self) -> f32 {
        self.re
            .iter()
            .zip(&self.im)
            .map(|(&r, &i)| r * r + i * i)
            .sum()
    }

    /// True when the bank is too quiet to matter and can be skipped entirely.
    pub fn is_idle(&self) -> bool {
        self.energy() < IDLE_ENERGY
    }

    /// Zeroes a chunk whose every mode has decayed past audibility. Returns
    /// true when the chunk was dropped and needs no further processing.
    fn cull(&mut self, base: usize) -> bool {
        let mut peak = 0.0f32;
        for k in base..base + LANES {
            peak = peak.max(self.re[k] * self.re[k] + self.im[k] * self.im[k]);
        }
        if peak >= CULL_AMPLITUDE * CULL_AMPLITUDE {
            return false;
        }
        for k in base..base + LANES {
            self.re[k] = 0.0;
            self.im[k] = 0.0;
            // A skipped chunk misses the ramp, so its pole jumps straight to
            // the target. Silent modes cannot click.
            self.snap_pole(k);
        }
        true
    }

    fn write_mode(&mut self, k: usize, freq_hz: f32, sigma: f32, g_re: f32, g_im: f32) {
        let w = std::f32::consts::TAU * freq_hz / SAMPLE_RATE;
        self.cos_w[k] = w.cos();
        self.sin_w[k] = w.sin();
        self.g_re[k] = g_re;
        self.g_im[k] = g_im;
        self.freq[k] = freq_hz;
        self.sigma_base[k] = sigma;
        self.r_base[k] = (-sigma.max(0.0) / SAMPLE_RATE).exp();
        self.refresh_target(k);
        self.snap_pole(k);
    }

    fn write_damping(&mut self, k: usize, extra: f32) {
        if self.sigma_extra[k] == extra {
            return;
        }
        self.sigma_extra[k] = extra;
        self.refresh_target(k);
        if self.r_tgt[k] != self.r_cur[k] {
            self.ramping = true;
        }
    }

    fn refresh_target(&mut self, k: usize) {
        self.r_tgt[k] = self.r_base[k] * decay_factor(self.sigma_extra[k] / SAMPLE_RATE);
    }

    /// Moves mode `k`'s pole to its target radius immediately.
    fn snap_pole(&mut self, k: usize) {
        self.r_cur[k] = self.r_tgt[k];
        self.a_re[k] = self.r_tgt[k] * self.cos_w[k];
        self.a_im[k] = self.r_tgt[k] * self.sin_w[k];
    }

    fn arrays_mut(&mut self) -> [&mut Vec<f32>; 14] {
        [
            &mut self.re,
            &mut self.im,
            &mut self.a_re,
            &mut self.a_im,
            &mut self.g_re,
            &mut self.g_im,
            &mut self.cos_w,
            &mut self.sin_w,
            &mut self.freq,
            &mut self.sigma_base,
            &mut self.sigma_extra,
            &mut self.r_base,
            &mut self.r_cur,
            &mut self.r_tgt,
        ]
    }
}

fn padded_len(modes: usize) -> usize {
    modes.div_ceil(LANES) * LANES
}

#[inline(always)]
fn read4(v: &[f32], base: usize) -> [f32; LANES] {
    let mut out = [0.0; LANES];
    out.copy_from_slice(&v[base..base + LANES]);
    out
}

/// `exp(-u)` for the small `u = extra_sigma / SAMPLE_RATE` a damper produces.
/// Dampers add at most a few hundred 1/s, i.e. u < 0.01, where the cubic Taylor
/// form is exact to f32 precision — this keeps the whole per-partial damper
/// update off the transcendental path while the damper ramps.
fn decay_factor(u: f32) -> f32 {
    if u < 0.05 {
        1.0 - u * (1.0 - 0.5 * u * (1.0 - u / 3.0))
    } else {
        (-u).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::BLOCK;

    /// Impulse response of a bank, `n` samples long, rendered the way the
    /// engine renders: one `BLOCK` at a time, so damping ramps and per-block
    /// culling behave as they do in the instrument.
    fn impulse_response(bank: &mut ModalBank, n: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; n.div_ceil(BLOCK) * BLOCK];
        let mut x = vec![0.0f32; BLOCK];
        x[0] = 1.0;
        for out in y.chunks_mut(BLOCK) {
            bank.process_add(&x, out);
            x[0] = 0.0;
        }
        y.truncate(n);
        y
    }

    /// Peak magnitude of `y` in a window — a cheap envelope reading, accurate
    /// to well under 1 % as long as the window spans several periods.
    fn envelope(y: &[f32], start: usize, len: usize) -> f32 {
        y[start..start + len].iter().fold(0.0f32, |m, v| m.max(v.abs()))
    }

    #[test]
    fn resonator_frequency_is_accurate() {
        let f = 1000.0;
        let mut bank = ModalBank::with_capacity(1);
        bank.push_mode(f, 0.0, 1.0);

        let n = 48_000;
        let y = impulse_response(&mut bank, n);

        // Count positive-going zero crossings and interpolate the two ends.
        let mut first = None;
        let mut last = 0.0;
        let mut count = 0u32;
        for i in 1..n {
            if y[i - 1] <= 0.0 && y[i] > 0.0 {
                let frac = -y[i - 1] / (y[i] - y[i - 1]);
                let t = (i - 1) as f32 + frac;
                if first.is_none() {
                    first = Some(t);
                } else {
                    count += 1;
                    last = t;
                }
            }
        }
        let measured = count as f32 * SAMPLE_RATE / (last - first.unwrap());
        assert!(
            (measured - f).abs() < 0.1,
            "measured {measured} Hz, expected {f} Hz"
        );
    }

    #[test]
    fn measured_t60_matches_sigma() {
        for &t60 in &[0.2f32, 1.0, 5.0] {
            let sigma = 6.91 / t60;
            let mut bank = ModalBank::with_capacity(1);
            bank.push_mode(500.0, sigma, 1.0);

            // One period of 500 Hz is 96 samples; 480 spans five.
            let win = 480;
            let n = (SAMPLE_RATE * t60) as usize + win;
            let y = impulse_response(&mut bank, n);

            let a0 = envelope(&y, 0, win);
            let a1 = envelope(&y, n - win, win);
            let dt = (n - win) as f32 / SAMPLE_RATE;
            let measured = 6.91 / ((a0 / a1).ln() / dt);
            assert!(
                (measured / t60 - 1.0).abs() < 0.05,
                "T60 {measured} s, expected {t60} s"
            );
        }
    }

    #[test]
    fn extra_damping_reaches_the_requested_decay() {
        let extra = 40.0;
        let mut bank = ModalBank::with_capacity(1);
        bank.push_mode(500.0, 1.0, 1.0);
        bank.set_damping_scale(0..1, extra);

        let win = 480;
        // 40 dB of decay: far enough to measure, well above the culling floor.
        let n = (SAMPLE_RATE * 4.6 / (1.0 + extra)) as usize + win;
        let y = impulse_response(&mut bank, n);
        let a0 = envelope(&y, 0, win);
        let a1 = envelope(&y, n - win, win);
        let measured = (a0 / a1).ln() / ((n - win) as f32 / SAMPLE_RATE);
        assert!(
            (measured / (1.0 + extra) - 1.0).abs() < 0.05,
            "sigma {measured}, expected {}",
            1.0 + extra
        );
    }

    #[test]
    fn damping_profile_is_per_mode() {
        let mut bank = ModalBank::with_capacity(3);
        for f in [300.0, 600.0, 900.0] {
            bank.push_mode(f, 1.0, 1.0);
        }
        bank.set_damping_profile(&[0.0, 5.0, 50.0]);
        assert_eq!(bank.damping_scale(0), 0.0);
        assert_eq!(bank.damping_scale(1), 5.0);
        assert_eq!(bank.damping_scale(2), 50.0);
    }

    #[test]
    fn damping_ramps_within_a_block_without_a_jump() {
        // The pole slides across the block, so the output must stay smooth even
        // though the damping changed by a lot between blocks.
        let mut bank = ModalBank::with_capacity(1);
        bank.push_mode(440.0, 1.0, 1.0);
        let mut x = [0.0f32; BLOCK];
        x[0] = 1.0;
        let mut warm = [0.0f32; BLOCK];
        bank.process_add(&x, &mut warm);

        bank.set_damping_scale(0..1, 300.0);
        let mut y = [0.0f32; BLOCK];
        bank.process_add(&[0.0; BLOCK], &mut y);
        let step = y
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0f32, f32::max);
        // A 440 Hz sine at this amplitude moves at most ~0.06 per sample.
        assert!(step < 0.1, "discontinuity of {step} across the ramp");
    }

    #[test]
    fn silent_bank_reports_idle() {
        let mut bank = ModalBank::with_capacity(4);
        bank.push_mode(220.0, 5.0, 1.0);
        assert!(bank.is_idle());
        let mut x = [0.0f32; BLOCK];
        x[0] = 1.0;
        let mut y = [0.0f32; BLOCK];
        bank.process_add(&x, &mut y);
        assert!(!bank.is_idle());
    }

    #[test]
    fn decayed_modes_are_culled_to_exact_zero() {
        let mut bank = ModalBank::with_capacity(1);
        bank.push_mode(1000.0, 300.0, 1.0);
        let mut x = [0.0f32; BLOCK];
        x[0] = 1.0;
        let mut y = [0.0f32; BLOCK];
        bank.process_add(&x, &mut y);
        for _ in 0..40 {
            bank.process_add(&[0.0; BLOCK], &mut y);
        }
        assert_eq!(bank.energy(), 0.0);
        y.fill(0.0);
        bank.process_add(&[0.0; BLOCK], &mut y);
        assert!(y.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn a_partially_filled_vector_matches_separate_banks() {
        // Five modes leaves three padding lanes; they must contribute nothing.
        let freqs = [110.0, 221.0, 333.0, 444.0, 555.0];
        let mut bank = ModalBank::with_capacity(freqs.len());
        for (i, &f) in freqs.iter().enumerate() {
            bank.push_mode(f, 2.0 * i as f32, 0.5);
        }
        assert_eq!(bank.len(), freqs.len());
        let mixed = impulse_response(&mut bank, BLOCK);

        let mut sum = vec![0.0f32; BLOCK];
        for (i, &f) in freqs.iter().enumerate() {
            let mut one = ModalBank::with_capacity(1);
            one.push_mode(f, 2.0 * i as f32, 0.5);
            for (s, v) in sum.iter_mut().zip(impulse_response(&mut one, BLOCK)) {
                *s += v;
            }
        }
        for (a, b) in mixed.iter().zip(&sum) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn clear_and_refill_keeps_the_allocation() {
        let mut bank = ModalBank::with_capacity(80);
        let cap = bank.capacity();
        for k in 1..=80 {
            bank.push_mode(20.0 * k as f32, 1.0, 1.0);
        }
        bank.clear();
        assert!(bank.is_empty());
        bank.push_mode(440.0, 1.0, 1.0);
        assert_eq!(bank.len(), 1);
        assert_eq!(bank.capacity(), cap);
    }

    #[test]
    fn decay_factor_agrees_with_exp() {
        for &u in &[0.0f32, 1e-6, 1e-3, 0.049, 0.05, 0.5] {
            let want = (-u).exp();
            assert!((decay_factor(u) / want - 1.0).abs() < 1e-6, "u = {u}");
        }
    }
}


