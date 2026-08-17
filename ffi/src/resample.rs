//! The host-rate boundary.
//!
//! The engine is 48 kHz and stays 48 kHz (`DECISIONS.md` 17, `DISTRIBUTION.md`
//! M0 option (b)): a plugin that voiced differently at 44.1 and 96 kHz would be
//! a different instrument at every rate, and every calibrated number in
//! `PHYSICS.md` was measured at one. So the rate conversion lives here, at the
//! boundary, and **at 48 kHz it does not run at all** — [`Boundary::Bypass`]
//! hands the host's buffers straight to `Engine::process`, which is what makes
//! a 48 kHz host bit-identical to `cargo run -- render`.
//!
//! The shape is `rubato`'s `SincFixedOut`: a fixed *output* block with a
//! variable input pull. That is exactly what `Engine::process` is good at — it
//! accepts any request length and produces the same stream however it is cut up
//! (`DECISIONS.md` 47) — so the resampler asks for however many engine frames
//! it needs and the engine answers, with no ring buffer between them.
//!
//! Because the source is a generator rather than a stream, the sinc's
//! look-ahead is **not latency**: it pulls further ahead instead of delaying.
//! The engine therefore runs about `sinc_len/2` engine frames (2.7 ms) ahead of
//! the host clock, which shows up in *when an event lands*, not as delay, and
//! the plugin reports zero added latency to the host.

use piano_emulator::SAMPLE_RATE;
use rubato::{
    Resampler, SincFixedOut, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

/// Taps in the polyphase sinc. `DISTRIBUTION.md` budgets 128-256; 256 is what
/// the -104 dB alias floor costs, and it comes to 0.70 % of one core at
/// 44.1 kHz and 1.17 % at 96 against the engine's own 31-40 % (`DECISIONS.md`
/// 25b, 37, 382).
pub const SINC_LEN: usize = 256;

/// Cutoff as a fraction of the lower of the two Nyquists. At 44.1 kHz that puts
/// the transition band at 20.9 kHz.
pub const F_CUTOFF: f32 = 0.95;

/// Intermediate sinc points per input sample. With cubic interpolation between
/// them this is what keeps the interpolation error under the filter's stopband
/// rather than over it.
pub const OVERSAMPLING: usize = 256;

/// Output frames the resampler is asked for in one call, when the host does not
/// state a block size worth using instead.
const FALLBACK_CHUNK: usize = 128;

/// Largest output chunk we will size buffers for. A host asking for more than
/// this per render still works — [`Boundary::render`] loops — it just does not
/// get a one-call render.
const MAX_CHUNK: usize = 8192;

/// A source of engine audio: `Engine::process`, or a test signal.
pub trait Source {
    /// Fills both channels with the next `l.len()` frames. Never asked for a
    /// length it can refuse.
    fn fill(&mut self, l: &mut [f32], r: &mut [f32]);
}

impl<F: FnMut(&mut [f32], &mut [f32])> Source for F {
    fn fill(&mut self, l: &mut [f32], r: &mut [f32]) {
        self(l, r)
    }
}

/// The rate conversion between the engine and the host, or nothing at all.
pub enum Boundary {
    /// Host rate == engine rate. The host's buffer *is* the engine's buffer.
    Bypass,
    Sinc(Box<Sinc>),
}

pub struct Sinc {
    src: SincFixedOut<f32>,
    /// Engine-rate scratch, sized once at `input_frames_max()`.
    input: [Vec<f32>; 2],
    /// Host-rate scratch, one resampler chunk.
    output: [Vec<f32>; 2],
    /// Read position in `output`; `chunk` means empty.
    pos: usize,
    chunk: usize,
    /// Host rate over engine rate.
    ratio: f64,
}

impl Boundary {
    /// Builds the boundary for a host rate, or returns `None` if the rate is not
    /// a number a resampler can be built for.
    ///
    /// `max_frames` is the host's largest render block: it becomes the
    /// resampler's output chunk, so a host that renders exactly its stated block
    /// size gets one resampler call per render and no remainder at all.
    pub fn new(host_sample_rate: f64, max_frames: u32) -> Option<Boundary> {
        if !host_sample_rate.is_finite() || host_sample_rate <= 0.0 {
            return None;
        }
        // Exact equality and not a tolerance: the bypass has to be the thing
        // that is bit-exact, and "48000.5 Hz is close enough to 48 kHz" is a
        // detuned instrument, which is the one thing item 17 refused.
        if host_sample_rate == SAMPLE_RATE as f64 {
            return Some(Boundary::Bypass);
        }
        Boundary::sinc(host_sample_rate, max_frames)
    }

    /// The same boundary with the bypass refused — the filter runs even at
    /// 48 kHz.
    ///
    /// Nothing ships this: it exists so that a test can measure what the bypass
    /// is worth, which is the only honest way to argue for it.
    pub fn sinc(host_sample_rate: f64, max_frames: u32) -> Option<Boundary> {
        if !host_sample_rate.is_finite() || host_sample_rate <= 0.0 {
            return None;
        }
        let ratio = host_sample_rate / SAMPLE_RATE as f64;
        let chunk = (max_frames as usize)
            .clamp(1, MAX_CHUNK)
            .max(FALLBACK_CHUNK);
        let parameters = SincInterpolationParameters {
            sinc_len: SINC_LEN,
            f_cutoff: F_CUTOFF,
            oversampling_factor: OVERSAMPLING,
            interpolation: SincInterpolationType::Cubic,
            window: WindowFunction::BlackmanHarris2,
        };
        // Relative ratio 1.0: the host's rate is fixed for the lifetime of the
        // engine (AUv3 re-allocates render resources when it changes), so
        // nothing ever calls `set_resample_ratio` and the input scratch can be
        // sized for the one ratio we have.
        let src = SincFixedOut::<f32>::new(ratio, 1.0, parameters, chunk, 2).ok()?;
        // `input_frames_next()` is recomputed after every call and rides a
        // fractional accumulator, so the bound is `input_frames_max()`; the
        // slack on top is there so that a bound that is ever off by one is a
        // wasted kilobyte rather than a dropped block.
        let capacity = src.input_frames_max() + SINC_LEN;
        Some(Boundary::Sinc(Box::new(Sinc {
            src,
            input: [vec![0.0; capacity], vec![0.0; capacity]],
            output: [vec![0.0; chunk], vec![0.0; chunk]],
            pos: chunk,
            chunk,
            ratio,
        })))
    }

    /// Engine frames the boundary is running ahead of the host clock, at the
    /// start of a stream. Reported for the record; it is not latency and is not
    /// declared to the host (see the module docs).
    pub fn lookahead_frames(&self) -> usize {
        match self {
            Boundary::Bypass => 0,
            Boundary::Sinc(s) => s
                .src
                .input_frames_next()
                .saturating_sub((s.chunk as f64 / s.ratio).ceil() as usize),
        }
    }

    /// Renders `l.len()` host-rate frames, pulling engine-rate frames from
    /// `source` as it needs them.
    ///
    /// Allocation-free and lock-free on both paths, for any length.
    pub fn render<S: Source>(&mut self, source: &mut S, l: &mut [f32], r: &mut [f32]) {
        debug_assert_eq!(l.len(), r.len());
        match self {
            Boundary::Bypass => source.fill(l, r),
            Boundary::Sinc(sinc) => sinc.render(source, l, r),
        }
    }

    /// Drops the filter's history. The engine is reset separately: this is the
    /// half of a reset that belongs to the boundary.
    pub fn reset(&mut self) {
        if let Boundary::Sinc(sinc) = self {
            sinc.src.reset();
            sinc.pos = sinc.chunk;
            for buf in sinc.output.iter_mut() {
                buf.fill(0.0);
            }
        }
    }
}

impl Sinc {
    fn render<S: Source>(&mut self, source: &mut S, l: &mut [f32], r: &mut [f32]) {
        let mut done = 0;
        while done < l.len() {
            if self.pos == self.chunk {
                self.pull(source);
            }
            let n = (self.chunk - self.pos).min(l.len() - done);
            let end = self.pos + n;
            l[done..done + n].copy_from_slice(&self.output[0][self.pos..end]);
            r[done..done + n].copy_from_slice(&self.output[1][self.pos..end]);
            self.pos = end;
            done += n;
        }
    }

    /// One resampler chunk: ask how many engine frames it wants, render exactly
    /// that many, hand them over.
    fn pull<S: Source>(&mut self, source: &mut S) {
        let needed = self.src.input_frames_next();
        debug_assert!(needed <= self.input[0].len());
        if needed > self.input[0].len() {
            // Cannot happen with the capacity above, and if it ever did the
            // audio thread's only sound answers are silence or a stale block.
            // Silence, and the debug build says so.
            self.output[0].fill(0.0);
            self.output[1].fill(0.0);
            self.pos = 0;
            return;
        }
        let (in_l, in_r) = self.input.split_at_mut(1);
        source.fill(&mut in_l[0][..needed], &mut in_r[0][..needed]);
        let ins = [&in_l[0][..needed], &in_r[0][..needed]];
        let (out_l, out_r) = self.output.split_at_mut(1);
        let mut outs = [out_l[0].as_mut_slice(), out_r[0].as_mut_slice()];
        match self.src.process_into_buffer(&ins, &mut outs, None) {
            Ok(_) => {}
            Err(_) => {
                // Same reasoning: the audio thread does not get to panic.
                outs[0].fill(0.0);
                outs[1].fill(0.0);
            }
        }
        self.pos = 0;
    }
}
