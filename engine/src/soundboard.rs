//! Soundboard, body and master chain.
//!
//! Voices are accumulated at their stereo pan position (the direct sound) and
//! also summed to mono to drive the board. The board is the two-part model from
//! the spec:
//!
//! 1. **Body modes** — 24 fixed resonators between 40 and 400 Hz that colour the
//!    drive signal. These are the cabinet/soundboard eigenmodes: sparse, fairly
//!    damped, and only significant at low frequency.
//! 2. **Diffuse board field** — an 8-line feedback delay network with mutually
//!    prime 3-15 ms delays, orthogonal (Hadamard) feedback and a one-pole loss
//!    filter per line giving T60 ≈ 0.4 s at LF falling to ≈ 0.1 s at 8 kHz. Its
//!    two output taps use orthogonal sign patterns, so the board field is stereo
//!    decorrelated while the direct sound keeps its pan position.
//!
//! This is the soundboard's own diffuse field, not a room: it is short and dense
//! by construction, and the whole board path is normalised to unity broadband
//! gain so that `board_mix` is a true crossfade and does not change loudness.
//!
//! The master chain is output gain, a 10 Hz DC blocker, a gentle high shelf
//! (the board radiates less efficiently as frequency rises) and a soft-knee
//! safety limiter that is bit-transparent below -1 dBFS.

use crate::modal::ModalBank;
use crate::preset::SoundboardVoicing;
use crate::types::{db_to_amp, key_position, BLOCK, OUTPUT_GAIN, SAMPLE_RATE};

/// Maximum pan displacement; bass to the left, treble to the right.
const MAX_PAN: f32 = 0.6;

/// Largest displacement `voicing.polarization_pan_spread` may put between the
/// two polarizations of one key, either side of that key's own pan.
///
/// `MAX_PAN + MAX_PAN_SPREAD` is 1: at the ceiling the outer polarization of
/// the outermost key lands hard left or hard right, and no setting can ask
/// [`Soundboard::add_voice`] for a position off the stage.
pub const MAX_PAN_SPREAD: f32 = 0.4;

/// DC blocker corner frequency, Hz.
const DC_BLOCK_HZ: f32 = 10.0;

/// Level above which the safety limiter starts to bend the signal (-1 dBFS).
const LIMIT_THRESHOLD: f32 = 0.891_251;

/// Stereo position of a key: -1.0 hard left, +1.0 hard right.
pub fn pan_for_key(key: u8) -> f32 {
    (2.0 * key_position(key) - 1.0) * MAX_PAN
}

pub struct Soundboard {
    direct_l: [f32; BLOCK],
    direct_r: [f32; BLOCK],
    mono: [f32; BLOCK],
    board_l: [f32; BLOCK],
    board_r: [f32; BLOCK],
    /// Mono sum after the body modes have coloured it; the FDN's input.
    drive: [f32; BLOCK],
    body: ModalBank,
    fdn: Fdn,
    board_mix: f32,
    /// Linear gain of the master high shelf's upper band.
    shelf_gain: f32,
    dc_r_coeff: f32,
    dc_state: [(f32, f32); 2],
    shelf_b: f32,
    shelf_state: [f32; 2],
}

impl Soundboard {
    pub fn new(voicing: &SoundboardVoicing) -> Self {
        let mut body = ModalBank::with_capacity(voicing.body_modes.len());
        for mode in &voicing.body_modes {
            // Q = f / bandwidth and this resonator's -3 dB bandwidth is sigma/pi.
            let sigma = std::f32::consts::PI * mode.hz / mode.q;
            // A complex one-pole driven at its own frequency settles at
            // |s| = g / (2 (1 - r)), so this normalises the mode's peak to its
            // tabulated gain.
            let r = (-sigma / SAMPLE_RATE).exp();
            body.push_mode(mode.hz, sigma, 2.0 * (1.0 - r) * mode.gain * voicing.body_mix);
        }
        Soundboard {
            direct_l: [0.0; BLOCK],
            direct_r: [0.0; BLOCK],
            mono: [0.0; BLOCK],
            board_l: [0.0; BLOCK],
            board_r: [0.0; BLOCK],
            drive: [0.0; BLOCK],
            body,
            fdn: Fdn::new(voicing),
            board_mix: voicing.board_mix,
            shelf_gain: db_to_amp(voicing.shelf_gain_db),
            dc_r_coeff: (-std::f32::consts::TAU * DC_BLOCK_HZ / SAMPLE_RATE).exp(),
            dc_state: [(0.0, 0.0); 2],
            shelf_b: 1.0 - (-std::f32::consts::TAU * voicing.shelf_hz / SAMPLE_RATE).exp(),
            shelf_state: [0.0; 2],
        }
    }

    pub fn board_mix(&self) -> f32 {
        self.board_mix
    }

    pub fn set_board_mix(&mut self, mix: f32) {
        self.board_mix = mix.clamp(0.0, 1.0);
    }

    /// Clears the accumulators before the voices of a new block are added.
    pub fn begin_block(&mut self) {
        self.direct_l.fill(0.0);
        self.direct_r.fill(0.0);
        self.mono.fill(0.0);
    }

    /// Accumulates one voice's mono output at pan position `pan` (-1..1).
    pub fn add_voice(&mut self, mono: &[f32], pan: f32) {
        debug_assert_eq!(mono.len(), BLOCK);
        // Equal-power pan keeps the summed level constant across the compass.
        let angle = (pan.clamp(-1.0, 1.0) + 1.0) * std::f32::consts::FRAC_PI_4;
        let (gl, gr) = (angle.cos(), angle.sin());
        for (i, &x) in mono.iter().enumerate() {
            self.direct_l[i] += gl * x;
            self.direct_r[i] += gr * x;
            self.mono[i] += x;
        }
    }

    /// Mixes, applies the master chain, and writes the finished block.
    pub fn process(&mut self, out_l: &mut [f32], out_r: &mut [f32]) {
        debug_assert_eq!(out_l.len(), BLOCK);
        debug_assert_eq!(out_r.len(), BLOCK);
        let direct = 1.0 - self.board_mix;
        self.board();
        let shelf_g = self.shelf_gain;
        for ch in 0..2 {
            let (out, dry, wet) = if ch == 0 {
                (&mut *out_l, &self.direct_l, &self.board_l)
            } else {
                (&mut *out_r, &self.direct_r, &self.board_r)
            };
            let (mut prev_x, mut prev_y) = self.dc_state[ch];
            let mut shelf = self.shelf_state[ch];
            for i in 0..BLOCK {
                let x = (direct * dry[i] + self.board_mix * wet[i]) * OUTPUT_GAIN;
                let dc = x - prev_x + self.dc_r_coeff * prev_y;
                prev_x = x;
                prev_y = dc;
                // High shelf as a one-pole crossover: low band passes at unity,
                // the remainder (the high band) is scaled by the shelf gain.
                shelf += self.shelf_b * (dc - shelf);
                out[i] = soft_clip(shelf_g * dc + (1.0 - shelf_g) * shelf);
            }
            self.dc_state[ch] = (prev_x, prev_y);
            self.shelf_state[ch] = shelf;
        }
    }

    pub fn reset(&mut self) {
        self.begin_block();
        self.board_l.fill(0.0);
        self.board_r.fill(0.0);
        self.drive.fill(0.0);
        self.body.reset_state();
        self.fdn.clear();
        self.dc_state = [(0.0, 0.0); 2];
        self.shelf_state = [0.0; 2];
    }

    /// Renders the board's stereo response to the mono voice sum.
    fn board(&mut self) {
        self.drive.copy_from_slice(&self.mono);
        // `process_add` accumulates and the mode gains already carry BODY_MIX,
        // so the body resonances land straight on top of the dry drive.
        self.body.process_add(&self.mono, &mut self.drive);
        self.fdn
            .process(&self.drive, &mut self.board_l, &mut self.board_r);
    }
}

/// Safety limiter: transparent below -1 dBFS, tanh-compressed above, and
/// continuous in value and slope at the threshold so it cannot click.
fn soft_clip(x: f32) -> f32 {
    let a = x.abs();
    if a <= LIMIT_THRESHOLD {
        x
    } else {
        let head = 1.0 - LIMIT_THRESHOLD;
        x.signum() * (LIMIT_THRESHOLD + head * ((a - LIMIT_THRESHOLD) / head).tanh())
    }
}

/// Number of delay lines in the board's diffuse field.
const FDN_LINES: usize = 8;

/// Line lengths in samples: 3.1-14.2 ms, all prime so no two lines share a
/// period and the modal density of the network is maximal.
const FDN_DELAYS: [usize; FDN_LINES] = [149, 211, 263, 331, 401, 461, 541, 683];

/// Injection and tap sign patterns, three mutually orthogonal rows of the 8×8
/// Hadamard matrix. Orthogonal taps are what makes the two output channels
/// decorrelated.
const FDN_IN_SIGN: [f32; FDN_LINES] = [1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0];
const FDN_L_SIGN: [f32; FDN_LINES] = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
const FDN_R_SIGN: [f32; FDN_LINES] = [1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0];

/// 1/sqrt(FDN_LINES): keeps injection and tapping unitary.
const FDN_TAP_SCALE: f32 = 0.353_553_4;

/// Below this peak sample value the network is inaudible and, with no input,
/// can only get quieter — flush it so decayed state cannot linger as denormals.
const FDN_QUIET: f32 = 1.0e-20;

/// Feedback delay network: the soundboard's diffuse field.
struct Fdn {
    /// All lines concatenated into one allocation; line `i` occupies
    /// `start[i] .. start[i] + FDN_DELAYS[i]`.
    delay: Vec<f32>,
    start: [usize; FDN_LINES],
    pos: [usize; FDN_LINES],
    /// One-pole loss filter per line: `g * ((1-a) x + a y[n-1])`, unity at DC.
    loss_state: [f32; FDN_LINES],
    loss_a: [f32; FDN_LINES],
    loss_g: [f32; FDN_LINES],
    /// Broadband gain correction that makes the whole board path unity, so
    /// `board_mix` is a loudness-preserving crossfade.
    level: f32,
    /// Largest sample written during the previous block.
    peak: f32,
}

impl Fdn {
    fn new(voicing: &SoundboardVoicing) -> Self {
        let mut start = [0usize; FDN_LINES];
        let mut total = 0;
        for i in 0..FDN_LINES {
            start[i] = total;
            total += FDN_DELAYS[i];
        }
        let mut loss_a = [0.0f32; FDN_LINES];
        let mut loss_g = [0.0f32; FDN_LINES];
        for i in 0..FDN_LINES {
            let (g, a) = line_loss(FDN_DELAYS[i], voicing);
            loss_g[i] = g;
            loss_a[i] = a;
        }
        Fdn {
            delay: vec![0.0; total],
            start,
            pos: [0; FDN_LINES],
            loss_state: [0.0; FDN_LINES],
            loss_a,
            loss_g,
            level: voicing.board_level,
            peak: 0.0,
        }
    }

    fn clear(&mut self) {
        self.delay.iter_mut().for_each(|v| *v = 0.0);
        self.pos = [0; FDN_LINES];
        self.loss_state = [0.0; FDN_LINES];
        self.peak = 0.0;
    }

    fn process(&mut self, input: &[f32], out_l: &mut [f32], out_r: &mut [f32]) {
        debug_assert_eq!(input.len(), out_l.len());
        debug_assert_eq!(input.len(), out_r.len());
        if self.peak < FDN_QUIET && input.iter().all(|&x| x == 0.0) {
            self.clear();
            out_l.fill(0.0);
            out_r.fill(0.0);
            return;
        }

        let mut peak = 0.0f32;
        for n in 0..input.len() {
            let mut tap = [0.0f32; FDN_LINES];
            let mut fed = [0.0f32; FDN_LINES];
            for i in 0..FDN_LINES {
                let d = self.delay[self.start[i] + self.pos[i]];
                let a = self.loss_a[i];
                let y = (1.0 - a) * d + a * self.loss_state[i];
                self.loss_state[i] = y;
                tap[i] = d;
                fed[i] = self.loss_g[i] * y;
            }
            // Orthogonal feedback: unitary mixing plus per-line loss < 1 makes
            // the loop strictly contractive, so the network cannot blow up.
            hadamard8(&mut fed);
            let x = input[n] * FDN_TAP_SCALE;
            let (mut l, mut r) = (0.0f32, 0.0f32);
            for i in 0..FDN_LINES {
                let w = FDN_IN_SIGN[i] * x + fed[i];
                self.delay[self.start[i] + self.pos[i]] = w;
                self.pos[i] += 1;
                if self.pos[i] == FDN_DELAYS[i] {
                    self.pos[i] = 0;
                }
                peak = peak.max(w.abs());
                l += FDN_L_SIGN[i] * tap[i];
                r += FDN_R_SIGN[i] * tap[i];
            }
            out_l[n] = self.level * FDN_TAP_SCALE * l;
            out_r[n] = self.level * FDN_TAP_SCALE * r;
        }
        self.peak = peak;
    }
}

/// Per-pass loss for a line of `m` samples: the DC gain that yields the
/// preset's low-frequency T60 and the one-pole coefficient that bends the gain
/// down to its high-frequency T60 at `fdn_hf_hz`.
fn line_loss(m: usize, voicing: &SoundboardVoicing) -> (f32, f32) {
    // T60 means -60 dB, i.e. a factor exp(-6.907) over T60 seconds.
    let passes = |t60: f32| (-6.907 * m as f32 / (t60 * SAMPLE_RATE)).exp();
    let g_lf = passes(voicing.fdn_t60_lf);
    // How much *more* the line must lose at `fdn_hf_hz` than at DC. A board
    // whose treble outlives its bass is not a board, so the ratio is clamped
    // at unity rather than refused: `rho >= 1` asks for a one-pole gain that
    // rises with frequency, which this form cannot make.
    let rho = (passes(voicing.fdn_t60_hf) / g_lf).min(1.0);
    // Solve |(1-a) / (1 - a e^-jw)| = rho for a in (0, 1).
    let cw = (std::f32::consts::TAU * voicing.fdn_hf_hz / SAMPLE_RATE).cos();
    let d = 1.0 - rho * rho;
    // Equal T60s make the loss flat, and the closed form below 0/0. The pole
    // that realises a flat gain is a = 0, which is what the limit approaches.
    if d <= f32::EPSILON {
        return (g_lf, 0.0);
    }
    let b = 1.0 - rho * rho * cw;
    (g_lf, (b - (b * b - d * d).sqrt()) / d)
}

/// In-place Walsh-Hadamard transform of 8 values, scaled to be orthonormal.
fn hadamard8(v: &mut [f32; FDN_LINES]) {
    for half in [4usize, 2, 1] {
        let mut base = 0;
        while base < FDN_LINES {
            for j in base..base + half {
                let (a, b) = (v[j], v[j + half]);
                v[j] = a + b;
                v[j + half] = a - b;
            }
            base += 2 * half;
        }
    }
    for x in v.iter_mut() {
        *x *= FDN_TAP_SCALE;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preset::Preset;

    fn voicing() -> SoundboardVoicing {
        Preset::default().soundboard
    }

    fn rms(v: &[f32]) -> f32 {
        (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt()
    }

    /// Renders `blocks` blocks of the board fed by `voice`, returning the peak
    /// absolute output sample seen.
    fn render_peak(sb: &mut Soundboard, voice: &[f32], blocks: usize) -> f32 {
        let (mut l, mut r) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
        let mut peak = 0.0f32;
        for _ in 0..blocks {
            sb.begin_block();
            sb.add_voice(voice, 0.0);
            sb.process(&mut l, &mut r);
            for &v in l.iter().chain(r.iter()) {
                peak = peak.max(v.abs());
            }
        }
        peak
    }

    /// Samples until the RMS of the tail has fallen `drop_db` below the first
    /// window after the excitation stopped.
    fn decay_samples(tail: &[f32], drop_db: f32) -> usize {
        const WINDOW: usize = 1024;
        let reference = rms(&tail[..WINDOW]);
        let target = reference * 10.0f32.powf(-drop_db / 20.0);
        for (i, w) in tail.chunks_exact(WINDOW).enumerate() {
            if rms(w) < target {
                return i * WINDOW;
            }
        }
        tail.len()
    }

    /// Drives the bare FDN with a Hann-windowed sine burst (windowed so the
    /// burst does not splatter energy across the spectrum) and returns the tail.
    fn fdn_burst_tail(freq: f32, burst: usize, tail_len: usize) -> Vec<f32> {
        let mut fdn = Fdn::new(&voicing());
        let (mut l, mut r) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
        let mut tail = Vec::with_capacity(tail_len);
        let mut n = 0usize;
        while n < burst + tail_len {
            let mut input = [0.0f32; BLOCK];
            for (i, x) in input.iter_mut().enumerate() {
                let t = n + i;
                if t < burst {
                    let env = 0.5
                        - 0.5 * (std::f32::consts::TAU * t as f32 / burst as f32).cos();
                    *x = env * (std::f32::consts::TAU * freq * t as f32 / SAMPLE_RATE).sin();
                }
            }
            fdn.process(&input, &mut l, &mut r);
            if n >= burst {
                tail.extend_from_slice(&l);
            }
            n += BLOCK;
        }
        tail
    }

    #[test]
    fn pan_spreads_bass_left_and_treble_right() {
        assert!((pan_for_key(21) + MAX_PAN).abs() < 1e-6);
        assert!((pan_for_key(108) - MAX_PAN).abs() < 1e-6);
        assert!(pan_for_key(64).abs() < 0.05);
    }

    #[test]
    fn soft_clip_is_transparent_then_bounded() {
        // Bit-transparency below the threshold is what "engaged only above
        // -1 dBFS" has to mean for a limiter with no lookahead.
        for i in 0..1000 {
            let x = LIMIT_THRESHOLD * (i as f32 / 999.0);
            assert_eq!(soft_clip(x), x);
            assert_eq!(soft_clip(-x), -x);
        }
        assert!(soft_clip(20.0) <= 1.0);
        assert!(soft_clip(20.0) > LIMIT_THRESHOLD);
        assert!((soft_clip(LIMIT_THRESHOLD + 1e-5) - LIMIT_THRESHOLD).abs() < 1e-4);
    }

    #[test]
    fn silence_in_silence_out() {
        let mut sb = Soundboard::new(&voicing());
        let (mut l, mut r) = ([1.0f32; BLOCK], [1.0f32; BLOCK]);
        sb.begin_block();
        sb.process(&mut l, &mut r);
        assert!(l.iter().chain(r.iter()).all(|&v| v == 0.0));
    }

    #[test]
    fn dc_offset_is_removed() {
        let mut sb = Soundboard::new(&voicing());
        let (mut l, mut r) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
        let dc = [0.001f32; BLOCK];
        for _ in 0..200 {
            sb.begin_block();
            sb.add_voice(&dc, 0.0);
            sb.process(&mut l, &mut r);
        }
        let mean = l.iter().sum::<f32>() / BLOCK as f32;
        assert!(mean.abs() < 1e-3, "residual DC {mean}");
    }

    #[test]
    fn board_decays_and_stays_bounded_over_ten_seconds() {
        let mut sb = Soundboard::new(&voicing());
        let mut impulse = [0.0f32; BLOCK];
        impulse[0] = 1.0;
        let early = render_peak(&mut sb, &impulse, 1);

        let silence = [0.0f32; BLOCK];
        let blocks = (10.0 * SAMPLE_RATE / BLOCK as f32) as usize;
        let mut late = 0.0f32;
        let (mut l, mut r) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
        for b in 0..blocks {
            sb.begin_block();
            sb.add_voice(&silence, 0.0);
            sb.process(&mut l, &mut r);
            for &v in l.iter().chain(r.iter()) {
                assert!(v.is_finite(), "non-finite output at block {b}");
                if b > blocks / 2 {
                    late = late.max(v.abs());
                }
            }
        }
        assert!(early > 0.0);
        assert!(late < early * 1e-6, "tail {late} vs impulse {early}");
    }

    #[test]
    fn diffuse_field_decays_faster_at_high_frequency() {
        let burst = (0.2 * SAMPLE_RATE) as usize;
        let tail = (1.5 * SAMPLE_RATE) as usize;
        let lf = decay_samples(&fdn_burst_tail(100.0, burst, tail), 20.0);
        let hf = decay_samples(&fdn_burst_tail(voicing().fdn_hf_hz, burst, tail), 20.0);
        assert!(
            lf > 2 * hf,
            "T20 at 100 Hz {lf} samples vs at 8 kHz {hf} samples"
        );
    }

    /// The loss filter design is what sets the decay, so check it directly
    /// against the two T60 targets rather than only through the tail.
    #[test]
    fn line_loss_hits_both_t60_targets() {
        let voicing = voicing();
        for m in FDN_DELAYS {
            let (g, a) = line_loss(m, &voicing);
            assert!((0.0..1.0).contains(&a), "line {m}: pole {a}");
            let round_trips = |t60: f32| SAMPLE_RATE * t60 / m as f32;
            // DC: unity through the filter, so g alone must give T60_LF.
            let lf_db = 20.0 * g.log10() * round_trips(voicing.fdn_t60_lf);
            assert!((lf_db + 60.0).abs() < 0.5, "line {m}: {lf_db} dB over T60_LF");
            let w = std::f32::consts::TAU * voicing.fdn_hf_hz / SAMPLE_RATE;
            let mag = g * (1.0 - a) / (1.0 - 2.0 * a * w.cos() + a * a).sqrt();
            let hf_db = 20.0 * mag.log10() * round_trips(voicing.fdn_t60_hf);
            assert!((hf_db + 60.0).abs() < 0.5, "line {m}: {hf_db} dB over T60_HF");
        }
    }

    /// A preset that asks for the same T60 at both ends of the spectrum is
    /// legal, and used to render `NaN`.
    ///
    /// `rho` — how much more the line loses at `fdn_hf_hz` than at DC — is 1
    /// there, and the pole that realises it came out of a `0/0`. Every sample
    /// the board produced after the first block was `NaN`, and
    /// `Preset::validate` had no reason to object: both numbers are positive
    /// and either one alone is fine. Found by an end-to-end sweep, not by the
    /// unit tests, because nothing had ever asked for a flat diffuse field.
    #[test]
    fn a_flat_diffuse_field_is_a_flat_gain_rather_than_a_division_by_zero() {
        let mut voicing = voicing();
        for (lf, hf) in [(0.4f32, 0.4f32), (0.05, 0.05), (0.1, 0.4), (3.0, 3.0)] {
            voicing.fdn_t60_lf = lf;
            voicing.fdn_t60_hf = hf;
            for m in FDN_DELAYS {
                let (g, a) = line_loss(m, &voicing);
                assert!(
                    g.is_finite() && a.is_finite(),
                    "T60 {lf}/{hf}, line {m}: gain {g}, pole {a}"
                );
                assert!((0.0..1.0).contains(&a), "T60 {lf}/{hf}, line {m}: pole {a}");
                assert!((0.0..1.0).contains(&g), "T60 {lf}/{hf}, line {m}: gain {g}");
            }
            let mut sb = Soundboard::new(&voicing);
            sb.set_board_mix(1.0);
            let mut impulse = [0.0f32; BLOCK];
            impulse[0] = 1.0;
            let (mut l, mut r) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
            for b in 0..400 {
                sb.begin_block();
                sb.add_voice(if b == 0 { &impulse } else { &[0.0; BLOCK] }, 0.0);
                sb.process(&mut l, &mut r);
                assert!(
                    l.iter().chain(r.iter()).all(|v| v.is_finite()),
                    "T60 {lf}/{hf}: block {b} of the diffuse field is not finite"
                );
            }
        }
    }

    #[test]
    fn board_output_channels_are_decorrelated() {
        let mut sb = Soundboard::new(&voicing());
        let mut impulse = [0.0f32; BLOCK];
        impulse[0] = 1.0;
        sb.set_board_mix(1.0);
        let (mut l, mut r) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
        let (mut ll, mut rr, mut lr) = (0.0f32, 0.0f32, 0.0f32);
        for b in 0..200 {
            sb.begin_block();
            sb.add_voice(if b == 0 { &impulse } else { &[0.0; BLOCK] }, 0.0);
            sb.process(&mut l, &mut r);
            for i in 0..BLOCK {
                ll += l[i] * l[i];
                rr += r[i] * r[i];
                lr += l[i] * r[i];
            }
        }
        let correlation = lr / (ll * rr).sqrt();
        assert!(correlation.abs() < 0.3, "L/R correlation {correlation}");
    }

    #[test]
    fn board_path_preserves_broadband_loudness() {
        // A pure crossfade only leaves the level alone if the board path has
        // roughly unity broadband gain; `board_level` is what pins that down.
        let mut dry = Soundboard::new(&voicing());
        let mut wet = Soundboard::new(&voicing());
        dry.set_board_mix(0.0);
        wet.set_board_mix(1.0);
        let (mut l, mut r) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
        let (mut dry_energy, mut wet_energy) = (0.0f32, 0.0f32);
        // Deterministic broadband excitation: a linear-congruential noise burst,
        // quiet enough that the safety limiter stays out of the measurement.
        let level = 0.05 / OUTPUT_GAIN;
        let mut state = 0x2545_f491u32;
        for b in 0..400 {
            let mut noise = [0.0f32; BLOCK];
            if b < 200 {
                for x in noise.iter_mut() {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    *x = level * ((state >> 8) as f32 / (1 << 23) as f32 - 1.0);
                }
            }
            for (sb, energy) in [(&mut dry, &mut dry_energy), (&mut wet, &mut wet_energy)] {
                sb.begin_block();
                sb.add_voice(&noise, 0.0);
                sb.process(&mut l, &mut r);
                *energy += l.iter().map(|x| x * x).sum::<f32>();
            }
        }
        let ratio_db = 10.0 * (wet_energy / dry_energy).log10();
        assert!(ratio_db.abs() < 1.0, "board path is {ratio_db} dB off unity");
    }

    /// Steady-state amplitude of the body bank alone at `freq`, unit sine in.
    fn body_response(freq: f32) -> f32 {
        let mut body = Soundboard::new(&voicing()).body;
        let (mut y, mut peak) = ([0.0f32; BLOCK], 0.0f32);
        // The lowest mode has T60 ≈ 0.6 s; settle well past that before reading.
        let settle = 300;
        for b in 0..settle + 40 {
            let mut sine = [0.0f32; BLOCK];
            for (i, x) in sine.iter_mut().enumerate() {
                let t = (b * BLOCK + i) as f32;
                *x = (std::f32::consts::TAU * freq * t / SAMPLE_RATE).sin();
            }
            y.fill(0.0);
            body.process_add(&sine, &mut y);
            if b >= settle {
                peak = peak.max(y.iter().fold(0.0f32, |m, v| m.max(v.abs())));
            }
        }
        peak
    }

    #[test]
    fn body_modes_are_separate_resonances() {
        // Modal overlap must stay low enough that the table is audible as
        // resonances rather than as one broad low-frequency shelf: every
        // tabulated frequency has to be a local maximum.
        for w in voicing().body_modes.windows(2) {
            let (lo, hi) = (body_response(w[0].hz), body_response(w[1].hz));
            let mid = body_response(0.5 * (w[0].hz + w[1].hz));
            assert!(
                mid < 0.9 * lo.min(hi),
                "modes at {} and {} Hz merge: {lo}, {mid}, {hi}",
                w[0].hz,
                w[1].hz
            );
        }
    }

    #[test]
    fn body_modes_stay_in_the_low_frequency_range() {
        for mode in &voicing().body_modes {
            // Bounded gain: the body colours the board, it must not boom.
            assert!(body_response(mode.hz) < 1.0);
        }
        assert!(body_response(1_000.0) < 0.03, "body bank rings above its range");
    }

    #[test]
    fn master_shelf_tilts_the_treble_down() {
        let level = |freq: f32| {
            let mut sb = Soundboard::new(&voicing());
            sb.set_board_mix(0.0);
            let (mut l, mut r) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
            let mut energy = 0.0f32;
            for b in 0..100 {
                let mut sine = [0.0f32; BLOCK];
                for (i, x) in sine.iter_mut().enumerate() {
                    let t = (b * BLOCK + i) as f32;
                    *x = 0.01 * (std::f32::consts::TAU * freq * t / SAMPLE_RATE).sin();
                }
                sb.begin_block();
                sb.add_voice(&sine, 0.0);
                sb.process(&mut l, &mut r);
                if b >= 50 {
                    energy += l.iter().map(|x| x * x).sum::<f32>();
                }
            }
            energy
        };
        let tilt_db = 10.0 * (level(10_000.0) / level(200.0)).log10();
        assert!(
            (-4.5..-1.0).contains(&tilt_db),
            "shelf tilt {tilt_db} dB at 10 kHz"
        );
    }
}
