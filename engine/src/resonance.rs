//! Sympathetic resonance bus, and the bridge admittance it runs through.
//!
//! Every string's output is summed into a mono bus, the bus is filtered by the
//! bridge's admittance `B(f)`, and each *undamped* string is driven by
//! `coupling * (B(bus) - own_contribution)`. Subtracting a string's own
//! contribution is not an optimization — without it each string feeds back into
//! itself, which is indistinguishable from reducing its damping, and the decay
//! time drifts with how many other strings happen to be ringing.
//!
//! The bus a string reads is the previous block's sum. One block (2.7 ms) of
//! latency in a diffuse coupling path is inaudible, and it removes the
//! circular dependency between "sum all strings" and "drive all strings".
//!
//! # The bridge admittance
//!
//! A string terminates on a bridge with a complex admittance, not on a node
//! (`PHYSICS.md` §4). [`BridgeFilter`] is that admittance: a smooth *backbone*
//! (the mean driving-point mobility, ≈ 1.3e-3 s/kg over 100–1000 Hz, falling in
//! the treble) with discrete *peaks* on it (the plate modes, sharp and well
//! separated below ~500 Hz). Two things follow from putting it on the bus.
//! Sympathetic excitation stops being spectrally uniform — a partial that lands
//! on a bridge resonance reaches the rest of the instrument far louder than one
//! that lands in a trough — and, because the bus is the only path between
//! strings, the coupling becomes two-way: a pair of strings that meet on a peak
//! exchange energy fast enough that the struck one measurably loses it, which
//! is the decay-rate coupling of Weinreich and Cartling.
//!
//! Absent from the preset, `B` is the unity filter and [`ResonanceBus`] is the
//! flat bus the engine has always had, sample for sample.
//!
//! ## Where the filter runs, and what that costs
//!
//! `B` is evaluated **once per block on the mono bus**, not once per voice.
//! That is not only a cost decision but it is mostly a cost decision: 88
//! undamped strings under the pedal would need 88 filter states, and a
//! sixty-section filter at 48 kHz costs about 0.03 % of a core *per instance* —
//! 3 % for the instrument, against a worst case that is already at 39 % of the
//! spec's 50 % budget. One instance is 0.03 %, and it does not grow with
//! polyphony.
//!
//! The price is that `B(bus - own)` cannot be formed exactly: the filter is
//! linear, so `B(bus - own) = B(bus) - B(own)`, and `B(own)` is a different
//! signal for every voice. What the bus subtracts instead is `own` scaled by
//! [`ResonanceBus::own_gain`] — `|B|` averaged over that key's own partials,
//! weighted by the `1/k^2` energy law a struck string's bridge force roughly
//! follows, computed once when the instrument is built. The subtraction is
//! therefore exact in the mean over the frequencies the string actually puts on
//! the bus: a note whose whole series sits where `B` is 10 dB down has its
//! self-contribution removed 10 dB down with it, instead of leaving a
//! systematic `coupling * (B - 1) * own` of extra damping across the treble
//! merely because the backbone falls there. What survives is the *fluctuation*
//! of `B` across one note's partials, which has no consistent sign, does not
//! grow with polyphony, and is bounded by the same loop-gain contract as
//! everything else here. With `B` unity the gain is exactly 1 and the drive is
//! bit for bit what it always was.
//!
//! # Stability
//!
//! The bus carries the engine's signal unit — bridge force in newtons times
//! the per-note output scale in `string.rs` — and the strings are driven at
//! their force input. Driven steadily at one of its partials, a string answers
//! with at most `output_scale * sin(k pi x) / sigma_k` of signal: roughly 1 for
//! the slowest bass partials and far less everywhere else. A loop that runs
//! through `m` mutually coincident partials therefore has gain
//! `~ m * coupling * max|B|`, so what has to be bounded is the *effective*
//! coupling `coupling * max|B|` and not `coupling` alone — `B` is allowed gain
//! well over one at its resonances. [`MAX_COUPLING`] bounds the coupling by
//! itself and [`MAX_BRIDGE_LOOP_GAIN`] bounds the product, both in
//! `Preset::validate`, which measures `max|B|` on the *realised* filter rather
//! than trusting the numbers in the file — and the smaller of the two, together
//! with the segments' own bound, is carried into the bus as
//! [`ResonanceBus::ceiling`] so that every later change to the coupling is held
//! to the same contract the preset was admitted under.
//!
//! That argument depends on the string parameters, so it is backed by a hard
//! guarantee that does not: the drive is clamped to [`DRIVE_CEILING`]. Every
//! modal pole is strictly inside the unit circle, so a bounded excitation can
//! only produce a bounded output — with the clamp in place the coupling loop
//! cannot diverge no matter how the strings are retuned or how the bridge is
//! voiced.

use crate::preset::{BridgePeak, BridgeVoicing, Preset};
use crate::types::{index_to_note, BLOCK, CULL_AMPLITUDE, NUM_KEYS, SAMPLE_RATE};

/// Largest coupling [`ResonanceBus::set_coupling`] will accept. Well above the
/// spec's 0.005-0.03 range and well below the loop gain analysed above.
pub const MAX_COUPLING: f32 = 0.05;

/// Largest `resonance_coupling * max|B(f)|` a preset may ask for.
///
/// The tightest loop the bus can close is string → bus → `B` → string, whose
/// round-trip gain is `coupling * max|B| * A` with `A` the string's answer to a
/// steady drive at one of its own partials — at most about 1 signal unit per
/// unit drive, for the slowest bass partials, and far less everywhere else
/// (see the stability discussion above). It sustains itself at 1. A quarter of
/// that is 12 dB of margin against the very worst string in the instrument,
/// and four times as much again against the realistic case where no two
/// partials coincide exactly.
pub const MAX_BRIDGE_LOOP_GAIN: f32 = 0.25;

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

// ------------------------------------------------------ the bridge admittance

/// `cos(-w)`, `sin(-w)`, `cos(-2w)`, `sin(-2w)` for one frequency — everything
/// a section needs to report its magnitude there.
#[derive(Clone, Copy, Debug)]
struct Trig {
    c1: f64,
    s1: f64,
    c2: f64,
    s2: f64,
}

impl Trig {
    fn at_hz(hz: f64) -> Trig {
        Trig::at_radians(std::f64::consts::TAU * hz / SAMPLE_RATE as f64)
    }

    fn at_radians(w: f64) -> Trig {
        let (s1, c1) = (-w).sin_cos();
        Trig {
            c1,
            s1,
            // Double angle, rather than two more transcendentals.
            c2: c1 * c1 - s1 * s1,
            s2: 2.0 * s1 * c1,
        }
    }
}

/// One second-order section, in `f64`.
///
/// `f64` and not `f32` because the bridge's low modes are the sharpest filters
/// in the engine: a `Q`-50 resonance at 20 Hz is a pole 2.6e-5 of the way round
/// the unit circle, where an `f32` coefficient has about four significant
/// digits of pole position left. It costs nothing that matters — this filter
/// runs once per block on one mono signal, not once per voice.
#[derive(Clone, Copy, Debug, Default)]
struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    s1: f64,
    s2: f64,
}

impl Biquad {
    /// Transposed direct form II: one multiply-add chain, and the state is the
    /// filter's output history rather than its input history, which is the
    /// numerically better-behaved of the two for cascades.
    #[inline]
    fn step(&mut self, x: f64) -> f64 {
        let y = self.b0 * x + self.s1;
        self.s1 = self.b1 * x - self.a1 * y + self.s2;
        self.s2 = self.b2 * x - self.a2 * y;
        y
    }

    fn reset(&mut self) {
        self.s1 = 0.0;
        self.s2 = 0.0;
    }

    /// `|H(e^{jw})|`, for the response measurements. `w` is in radians per
    /// sample.
    fn magnitude(&self, w: f64) -> f64 {
        self.magnitude_at(&Trig::at_radians(w))
    }

    /// `|H|` from the four already-computed trigonometric terms. The maximum
    /// search below evaluates tens of thousands of frequencies through every
    /// section, and the sines and cosines depend only on the frequency: hoisting
    /// them out of the section loop is what keeps that search a few
    /// milliseconds.
    fn magnitude_at(&self, t: &Trig) -> f64 {
        let nr = self.b0 + self.b1 * t.c1 + self.b2 * t.c2;
        let ni = self.b1 * t.s1 + self.b2 * t.s2;
        let dr = 1.0 + self.a1 * t.c1 + self.a2 * t.c2;
        let di = self.a1 * t.s1 + self.a2 * t.s2;
        ((nr * nr + ni * ni) / (dr * dr + di * di)).sqrt()
    }

    /// The RBJ high shelf at its steepest non-resonant slope (`S = 1`): unity
    /// below `hz`, `gain_db` above it, half of `gain_db` exactly *at* `hz`, and
    /// symmetric in log frequency about it.
    ///
    /// Second order rather than first because the backbone is a *fit*: a
    /// first-order shelf spreads a step over three octaves whatever the step
    /// is, so a cascade of them cannot follow anchors an octave apart, while
    /// this one's transition is roughly as wide as the step it carries.
    fn high_shelf(hz: f64, gain_db: f64) -> Biquad {
        let a = 10.0f64.powf(gain_db / 40.0);
        let w0 = std::f64::consts::TAU * hz / SAMPLE_RATE as f64;
        let (sin, cos) = w0.sin_cos();
        let alpha = sin * std::f64::consts::FRAC_1_SQRT_2;
        let two_root = 2.0 * a.sqrt() * alpha;
        let a0 = (a + 1.0) - (a - 1.0) * cos + two_root;
        Biquad {
            b0: a * ((a + 1.0) + (a - 1.0) * cos + two_root) / a0,
            b1: -2.0 * a * ((a - 1.0) + (a + 1.0) * cos) / a0,
            b2: a * ((a + 1.0) + (a - 1.0) * cos - two_root) / a0,
            a1: 2.0 * ((a - 1.0) - (a + 1.0) * cos) / a0,
            a2: ((a + 1.0) - (a - 1.0) * cos - two_root) / a0,
            s1: 0.0,
            s2: 0.0,
        }
    }

    /// The RBJ peaking (bell) section: `gain_db` at `hz`, unity away from it,
    /// with `q = hz / bandwidth`.
    fn peaking(hz: f64, q: f64, gain_db: f64) -> Biquad {
        let a = 10.0f64.powf(gain_db / 40.0);
        let w0 = std::f64::consts::TAU * hz / SAMPLE_RATE as f64;
        let (sin, cos) = w0.sin_cos();
        let alpha = sin / (2.0 * q);
        let a0 = 1.0 + alpha / a;
        Biquad {
            b0: (1.0 + alpha * a) / a0,
            b1: (-2.0 * cos) / a0,
            b2: (1.0 - alpha * a) / a0,
            a1: (-2.0 * cos) / a0,
            a2: (1.0 - alpha / a) / a0,
            s1: 0.0,
            s2: 0.0,
        }
    }
}

/// How many refinement passes the backbone fit is allowed. It converges in a
/// handful; the cap is there because the loop is a fixed point and a
/// pathological anchor list need not have one.
const BACKBONE_FIT_PASSES: usize = 32;

/// Largest step any one shelf of the backbone fit may be pushed to, in dB.
/// The refinement below is free to move the steps to hit the anchors, and a
/// nearly vertical target (40 dB between two adjacent anchors) would otherwise
/// let it run away; the fit is then simply worse, which the realised-response
/// measurement sees.
const MAX_BACKBONE_STEP_DB: f64 = 80.0;

/// Points of the coarse log grid the realised response is measured on, and its
/// ends. The grid steps by 1.4 %; on its own it resolves only the *smooth* half
/// of the filter, and [`BridgeFilter::max_magnitude`] documents what covers the
/// rest.
const RESPONSE_GRID: usize = 512;
const RESPONSE_LOW_HZ: f64 = 20.0;
const RESPONSE_HIGH_HZ: f64 = 20_000.0;

/// How far either side of a resonance the fine scan runs, in −3 dB bandwidths.
/// A peaking section is within a decibel of its skirt asymptote by four
/// bandwidths out and has no curvature left worth resolving by eight, which is
/// where the coarse grid takes over.
const PEAK_WINDOW_BANDWIDTHS: f64 = 8.0;

/// Samples per −3 dB bandwidth inside that window. A resonance's log-magnitude
/// is quadratic about its centre with a curvature of `≈ 8 · gain_db` per squared
/// bandwidth, so sampling at `bw/64` leaves at most `gain_db / 512` dB of any
/// one section's peak unseen — a thousandth of a dB per section at the schema's
/// ceiling, and under a tenth of a dB with all forty of them stacked.
const SAMPLES_PER_BANDWIDTH: f64 = 64.0;

/// Iterations of the golden-section refinement run on the best sample of every
/// window. Sixty halvings of an interval of one sampling step is far past f64.
const REFINE_STEPS: usize = 60;

/// The bridge's driving-point admittance as a filter: a broadband gain, a
/// cascade of first-order shelves fitted to the backbone anchors, and one
/// peaking section per bridge resonance.
///
/// Built once, from a `[voicing.bridge]` section. The default — no section —
/// is [`BridgeFilter::unity`], which copies its input and is the flat bus.
pub struct BridgeFilter {
    gain: f64,
    sections: Vec<Biquad>,
    /// Centre frequency and −3 dB bandwidth of every peak, kept so that
    /// [`Self::max_magnitude`] can scan *through* them rather than merely near
    /// them. See the derivation there: these are the only places the response
    /// has curvature the coarse grid cannot see.
    peaks: Vec<(f64, f64)>,
}

impl BridgeFilter {
    /// The flat bus: `B(f) = 1` everywhere, and `process` is a copy.
    pub fn unity() -> BridgeFilter {
        BridgeFilter {
            gain: 1.0,
            sections: Vec::new(),
            peaks: Vec::new(),
        }
    }

    /// Builds the filter a `[voicing.bridge]` section describes.
    ///
    /// The backbone is a target curve — the anchors' gains interpolated in log
    /// frequency — and it is *fitted*, not realised exactly: one first-order
    /// high shelf per anchor interval, cornered at the interval's geometric
    /// mean, then a fixed-point refinement that pushes the shelf steps until
    /// the cascade passes through the anchors. A first-order shelf's transition
    /// is spread over a couple of octaves, so anchors closer together than that
    /// interact, and the refinement is what resolves it. Anything the fit
    /// cannot reach — a 40 dB cliff inside one interval — comes out smoothed,
    /// which for a mean-mobility curve is the right failure.
    ///
    /// Nothing downstream trusts the target: `max|B|`, the response tests and
    /// the validator all measure the sections that were actually built.
    pub fn new(bridge: &BridgeVoicing) -> BridgeFilter {
        let anchors: Vec<(f64, f64)> = bridge
            .backbone
            .iter()
            .map(|a| (a.hz as f64, a.gain_db as f64))
            .collect();
        let (gain_db, steps) = fit_backbone(&anchors);

        let mut sections = Vec::with_capacity(steps.len() + bridge.peaks.len());
        for (i, &step) in steps.iter().enumerate() {
            let corner = (anchors[i].0 * anchors[i + 1].0).sqrt();
            sections.push(Biquad::high_shelf(corner, step));
        }
        for &BridgePeak { hz, q, gain_db } in &bridge.peaks {
            sections.push(Biquad::peaking(hz as f64, q as f64, gain_db as f64));
        }
        BridgeFilter {
            gain: 10.0f64.powf(gain_db / 20.0),
            sections,
            peaks: bridge
                .peaks
                .iter()
                .map(|p| (p.hz as f64, (p.hz / p.q.max(1.0e-6)) as f64))
                .collect(),
        }
    }

    /// The board's discrete modes alone — the peaks with no backbone under
    /// them, which is the *fluctuation* of the bridge's mobility about its
    /// mean.
    ///
    /// This is what `Re Y` in a string's damping is proportional to
    /// ([`BridgeVoicing::radiated_share`]): the mean is already inside the
    /// fitted `sigma(f)`, and only the fluctuation is missing from it. Never
    /// used to filter anything — `string.rs` reads its magnitude at each
    /// partial when the instrument is built.
    pub fn peaks_only(bridge: &BridgeVoicing) -> BridgeFilter {
        let mut filter = BridgeFilter::unity();
        for &BridgePeak { hz, q, gain_db } in &bridge.peaks {
            filter
                .sections
                .push(Biquad::peaking(hz as f64, q as f64, gain_db as f64));
            filter.peaks.push((hz as f64, (hz / q.max(1.0e-6)) as f64));
        }
        filter
    }

    /// True when this filter is the flat bus, in which case `process` is a copy
    /// and every path through the bus is bit for bit the pre-admittance one.
    pub fn is_unity(&self) -> bool {
        self.sections.is_empty() && self.gain == 1.0
    }

    /// `|B(f)|` of the filter as realised, at a frequency in Hz.
    pub fn magnitude(&self, hz: f32) -> f32 {
        self.magnitude_at(hz as f64) as f32
    }

    fn magnitude_at(&self, hz: f64) -> f64 {
        self.magnitude_trig(&Trig::at_hz(hz))
    }

    fn magnitude_trig(&self, t: &Trig) -> f64 {
        self.sections
            .iter()
            .fold(self.gain, |m, s| m * s.magnitude_at(t))
    }

    /// The largest `|B(f)|` anywhere in the audio band, which is what the
    /// coupling loop is bounded against.
    ///
    /// # Why a grid alone will not do
    ///
    /// This filter is a **cascade**, so the sections' decibels *add*, and the
    /// maximum of two overlapping resonances lies strictly *between* their
    /// centres. Sampling the centres and a 1.4 % log grid therefore misses it:
    /// twenty `Q`-50 +20 dB peaks at 101.63 Hz and twenty at 102.32 Hz — every
    /// one inside the schema, and both centres inside one grid interval — read
    /// 654.6 dB when sampled that way and 670.3 dB when scanned densely.
    /// 15.6 dB hidden between the samples is more than the whole 12 dB margin
    /// [`MAX_BRIDGE_LOOP_GAIN`] is built from, and a preset fitted against the
    /// sampled figure would realise a loop gain of 1.5 — self-sustaining. The
    /// stability contract is a contract, so the measurement has to be one too.
    ///
    /// # What is scanned, and why that is enough
    ///
    /// The log-magnitude of the cascade is the *sum* of its sections' — a
    /// broadband gain, monotone shelves, and one bell per resonance. A maximum
    /// can only hide between two samples if the curve rises and falls inside
    /// that interval, which needs curvature at the interval's scale, and the
    /// only thing in the cascade with curvature at any scale finer than an
    /// octave is a peaking section near its own centre. So:
    ///
    /// * the coarse [`RESPONSE_GRID`] covers everything smooth — the shelves'
    ///   transitions and the peaks' far skirts, where a bell's contribution
    ///   changes by at most `≈ 0.12 dB` per grid step and monotonically;
    /// * around **every** peak, a fine linear scan runs
    ///   ±[`PEAK_WINDOW_BANDWIDTHS`] bandwidths at
    ///   [`SAMPLES_PER_BANDWIDTH`] samples per bandwidth. Two resonances close
    ///   enough to add appreciably are within a few bandwidths of each other, so
    ///   both windows cover the ground between them — including the 101.83 Hz
    ///   the construction above hides its maximum at;
    /// * the best sample of every window, and of the coarse grid, is then
    ///   refined by golden section over one sampling step either side, so the
    ///   answer is a *local maximum* of the realised response rather than the
    ///   largest of a set of samples.
    ///
    /// Measured against a 4-million-point dense scan the residue is under
    /// 0.01 dB on the adversarial constructions and 0.0000 dB on the shipped
    /// preset (`the_measured_maximum_cannot_be_hidden_between_grid_points`).
    /// The cost is bounded by the schema: 40 peaks × 1024 points ≈ 42 k
    /// evaluations, a few milliseconds, at preset-load time only.
    pub fn max_magnitude(&self) -> f32 {
        if self.is_unity() {
            return 1.0;
        }
        let mut max = 0.0f64;

        // The smooth half: one sweep of the whole band.
        let ratio = (RESPONSE_HIGH_HZ / RESPONSE_LOW_HZ).ln() / (RESPONSE_GRID - 1) as f64;
        let mut best = (0.0f64, RESPONSE_LOW_HZ);
        for i in 0..RESPONSE_GRID {
            let hz = RESPONSE_LOW_HZ * (i as f64 * ratio).exp();
            let m = self.magnitude_at(hz);
            if m > best.0 {
                best = (m, hz);
            }
        }
        max = max.max(self.refine(best.1, best.1 * ratio));

        // The modal half: one fine window per resonance, refined at its best
        // point. Every peak gets its own window even when another peak's window
        // already covers it — the windows are cheap and the overlap is exactly
        // the case that has to be scanned twice as densely, not less.
        for &(hz, bandwidth) in &self.peaks {
            let step = bandwidth / SAMPLES_PER_BANDWIDTH;
            if !step.is_finite() || step <= 0.0 {
                continue;
            }
            let lo = (hz - PEAK_WINDOW_BANDWIDTHS * bandwidth).max(RESPONSE_LOW_HZ);
            let hi = (hz + PEAK_WINDOW_BANDWIDTHS * bandwidth).min(RESPONSE_HIGH_HZ);
            let mut best = (self.magnitude_at(hz), hz);
            let points = ((hi - lo) / step).ceil() as usize;
            for i in 0..=points {
                let f = (lo + i as f64 * step).min(hi);
                let m = self.magnitude_at(f);
                if m > best.0 {
                    best = (m, f);
                }
            }
            max = max.max(self.refine(best.1, step));
        }
        max as f32
    }

    /// Golden-section maximisation of `|B|` over `centre ± half_width`, which is
    /// one sampling step of whatever grid found `centre`. Returns the largest
    /// magnitude seen, never less than the value at `centre` itself.
    fn refine(&self, centre: f64, half_width: f64) -> f64 {
        const INV_PHI: f64 = 0.618_033_988_749_894_9;
        let mut lo = (centre - half_width).max(RESPONSE_LOW_HZ);
        let mut hi = (centre + half_width).min(RESPONSE_HIGH_HZ);
        let mut best = self.magnitude_at(centre);
        let (mut c, mut d) = (hi - INV_PHI * (hi - lo), lo + INV_PHI * (hi - lo));
        let (mut fc, mut fd) = (self.magnitude_at(c), self.magnitude_at(d));
        for _ in 0..REFINE_STEPS {
            best = best.max(fc).max(fd);
            if fc > fd {
                hi = d;
                d = c;
                fd = fc;
                c = hi - INV_PHI * (hi - lo);
                fc = self.magnitude_at(c);
            } else {
                lo = c;
                c = d;
                fc = fd;
                d = lo + INV_PHI * (hi - lo);
                fd = self.magnitude_at(d);
            }
        }
        best.max(fc).max(fd)
    }

    /// Filters one block. `input` and `output` may not overlap; both are
    /// [`BLOCK`] long. No allocation, no branching per section beyond the loop.
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), BLOCK);
        debug_assert_eq!(output.len(), BLOCK);
        if self.is_unity() {
            output.copy_from_slice(input);
            return;
        }
        for (o, &x) in output.iter_mut().zip(input) {
            let mut y = self.gain * x as f64;
            for section in &mut self.sections {
                y = section.step(y);
            }
            *o = y as f32;
        }
    }

    pub fn reset(&mut self) {
        for section in &mut self.sections {
            section.reset();
        }
    }
}

/// Fits a shelf cascade to the backbone anchors, returning the broadband gain
/// in dB and one step in dB per anchor interval.
///
/// The starting guess is the obvious one — the cascade's steps are the target's
/// own steps — and each pass measures where the realised curve actually lands
/// at the anchors and moves the steps by the difference. The best pass wins, so
/// a target the fit cannot follow leaves the cascade at its closest approach
/// rather than wherever the last pass wandered to.
fn fit_backbone(anchors: &[(f64, f64)]) -> (f64, Vec<f64>) {
    let n = anchors.len();
    debug_assert!(n >= 2);
    let corners: Vec<f64> = (0..n - 1)
        .map(|i| (anchors[i].0 * anchors[i + 1].0).sqrt())
        .collect();

    let mut gain_db = anchors[0].1;
    let mut steps: Vec<f64> = (0..n - 1)
        .map(|i| anchors[i + 1].1 - anchors[i].1)
        .collect();
    let mut best = (gain_db, steps.clone(), f64::INFINITY);

    for _ in 0..BACKBONE_FIT_PASSES {
        // Where the cascade as it stands actually passes through the anchors.
        let shelves: Vec<Biquad> = corners
            .iter()
            .zip(&steps)
            .map(|(&hz, &step)| Biquad::high_shelf(hz, step))
            .collect();
        let errors: Vec<f64> = anchors
            .iter()
            .map(|&(hz, target)| {
                let w = std::f64::consts::TAU * hz / SAMPLE_RATE as f64;
                let realised = shelves.iter().fold(1.0, |m, s| m * s.magnitude(w));
                target - (gain_db + 20.0 * realised.max(1.0e-300).log10())
            })
            .collect();

        let worst = errors.iter().fold(0.0f64, |m, e| m.max(e.abs()));
        if worst < best.2 {
            best = (gain_db, steps.clone(), worst);
        }
        if worst < 1.0e-3 || !worst.is_finite() {
            break;
        }
        gain_db += errors[0];
        for i in 0..n - 1 {
            steps[i] = (steps[i] + errors[i + 1] - errors[i])
                .clamp(-MAX_BACKBONE_STEP_DB, MAX_BACKBONE_STEP_DB);
        }
    }
    (best.0, best.1)
}

// -------------------------------------------------------------- the bus

pub struct ResonanceBus {
    bus: [f32; BLOCK],
    accum: [f32; BLOCK],
    bridge: BridgeFilter,
    /// Per key, `|B|` averaged over that key's own partials — what the drive
    /// path subtracts the voice's own contribution with. Exactly 1 everywhere
    /// when the bridge is the flat bus. See the module documentation.
    own_gain: Vec<f32>,
    coupling: f32,
    /// Largest coupling this bus may run at: [`MAX_COUPLING`] on a flat bus,
    /// and the loop-gain bound of the realised bridge and duplex tables when
    /// there are any. See [`ResonanceBus::set_coupling`].
    ceiling: f32,
    /// Peak absolute value of `bus`, kept so the engine can decide cheaply
    /// whether a silent voice is worth waking.
    peak: f32,
}

impl ResonanceBus {
    /// A bus with the flat bridge: `coupling` is the preset's
    /// `resonance_coupling`, clamped to the stable range like every later
    /// change to it.
    pub fn new(coupling: f32) -> Self {
        let mut bus = ResonanceBus {
            bus: [0.0; BLOCK],
            accum: [0.0; BLOCK],
            bridge: BridgeFilter::unity(),
            own_gain: vec![1.0; NUM_KEYS],
            coupling: 0.0,
            ceiling: MAX_COUPLING,
            peak: 0.0,
        };
        bus.set_coupling(coupling);
        bus
    }

    /// The bus this preset asks for, bridge admittance and all.
    ///
    /// The per-key self-cancellation gains are measured here, off the realised
    /// filter and the preset's own partial layout, which is why this is a
    /// construction-time function and the audio path only ever indexes a table.
    pub fn from_preset(preset: &Preset) -> Self {
        let mut bus = ResonanceBus::new(preset.voicing.resonance_coupling);
        let Some(voicing) = &preset.voicing.bridge else {
            // No bridge: `MAX_COUPLING` *is* the loop bound, and a duplex table
            // without a bridge is bounded through a unity `max|B|`.
            bus.ceiling = preset.coupling_ceiling(1.0);
            bus.set_coupling(preset.voicing.resonance_coupling);
            return bus;
        };
        bus.bridge = BridgeFilter::new(voicing);
        // The ceiling is a property of the filter that was *built*, measured
        // here rather than taken from the file — the same number
        // `Preset::validate` refused against, and the reason `set_coupling` can
        // promise anything at all.
        bus.ceiling = preset.coupling_ceiling(bus.bridge.max_magnitude());
        bus.set_coupling(preset.voicing.resonance_coupling);
        for (i, gain) in bus.own_gain.iter_mut().enumerate() {
            let p = preset.string_params(index_to_note(i));
            // Weighted by 1/k^2: a struck string's bridge force falls away
            // with partial number, so the top of the series contributes almost
            // nothing to what this string puts on the bus and should not decide
            // how much of it is taken back off.
            let (mut sum, mut weight) = (0.0f64, 0.0f64);
            for k in 1..=p.partial_count().max(1) {
                let w = 1.0 / (k * k) as f64;
                sum += w * bus.bridge.magnitude_at(p.partial_freq(k) as f64);
                weight += w;
            }
            *gain = (sum / weight) as f32;
        }
        bus
    }

    pub fn coupling(&self) -> f32 {
        self.coupling
    }

    /// The realised bridge admittance, for tests and for the engine's own
    /// reporting. Never read on the audio path.
    pub fn bridge(&self) -> &BridgeFilter {
        &self.bridge
    }

    /// How much of its own contribution the voice at `index` takes back off the
    /// bus. 1.0 for every key on a flat bus.
    pub fn own_gain(&self, index: usize) -> f32 {
        self.own_gain[index]
    }

    /// Sets the coupling, clamped to `0..=`[`ResonanceBus::ceiling`]. A value
    /// that is not a number silences the bus rather than passing through —
    /// `f32::clamp` returns NaN for NaN, and a NaN here would reach every
    /// undamped string.
    ///
    /// The clamp is against *this bus's* ceiling and not against
    /// [`MAX_COUPLING`], because those are the same number only on a flat bus.
    /// `Preset::validate` bounds `resonance_coupling · max|B|`, which says
    /// nothing about what some *later* caller may set: on
    /// `presets/salamander-c5.toml`, whose bridge peaks at 26.7 dB,
    /// `set_coupling(MAX_COUPLING)` would realise a loop gain of 1.08 — past
    /// self-sustainment, and the very number `Preset::validate` refuses that
    /// preset for. So the bound the preset was validated against is stored when
    /// the bus is built (`Preset::max_safe_coupling`) and enforced here, which
    /// makes the invariant "this bus is inside its loop bound" true of every
    /// path into it rather than only of the load path. `DRIVE_CEILING` remains
    /// the hard backstop underneath, holding whatever the tables say.
    pub fn set_coupling(&mut self, coupling: f32) {
        self.coupling = if coupling.is_finite() {
            coupling.clamp(0.0, self.ceiling)
        } else {
            0.0
        };
    }

    /// The largest coupling this bus will accept — the loop bound of the bridge
    /// and the segments it was built with. For the REPL, which reports it, and
    /// for tests.
    pub fn ceiling(&self) -> f32 {
        self.ceiling
    }

    /// Publishes the block that just finished as the bus the next block reads,
    /// through the bridge admittance.
    pub fn begin_block(&mut self) {
        self.bridge.process(&self.accum, &mut self.bus);
        self.accum.fill(0.0);
        self.peak = self.bus.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    }

    /// True when the bus carries enough to make an otherwise silent undamped
    /// string audible, so the engine has to render it. Below this the coupled
    /// drive `coupling * bus` cannot lift any mode past [`CULL_AMPLITUDE`] even
    /// at a partial's exact centre frequency, and every silent voice can be
    /// skipped — which is what keeps a pedal-down chord from costing the same
    /// as the full 88-key worst case.
    ///
    /// The peak it tests is the peak of the bus *after* the bridge, so a
    /// resonance that lifts the halo into audibility wakes the voices that
    /// would hear it.
    pub fn is_active(&self) -> bool {
        self.peak * self.coupling * MAX_STRING_ADMITTANCE > CULL_AMPLITUDE
    }

    /// Bus contents visible during the current block, i.e. `B(bus)`.
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

    /// Adds `coupling * (B(bus) - own_gain * own_previous)` into `out`, where
    /// `own_previous` is the voice `index`'s output during the block the bus
    /// was summed from.
    pub fn drive(&self, index: usize, own_previous: &[f32], out: &mut [f32]) {
        debug_assert_eq!(own_previous.len(), BLOCK);
        debug_assert_eq!(out.len(), BLOCK);
        let own_gain = self.own_gain[index];
        for i in 0..BLOCK {
            let rest = self.bus[i] - own_gain * own_previous[i];
            out[i] += (self.coupling * rest).clamp(-DRIVE_CEILING, DRIVE_CEILING);
        }
    }

    pub fn reset(&mut self) {
        self.bus.fill(0.0);
        self.accum.fill(0.0);
        self.bridge.reset();
        self.peak = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preset::{BridgeAnchor, BridgeVoicing, MAX_BRIDGE_Q};

    /// A bus coupled as the instrument really runs it.
    fn bus() -> ResonanceBus {
        ResonanceBus::new(Preset::default().voicing.resonance_coupling)
    }
    use crate::modal::ModalBank;
    use crate::types::{amp_to_db, key_index};

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

    /// A bridge with one resonance, at `hz`, on a flat backbone.
    fn one_peak(hz: f32, q: f32, gain_db: f32) -> BridgeVoicing {
        BridgeVoicing {
            backbone: vec![
                BridgeAnchor {
                    hz: 20.0,
                    gain_db: 0.0,
                },
                BridgeAnchor {
                    hz: 16_000.0,
                    gain_db: 0.0,
                },
            ],
            peaks: vec![BridgePeak { hz, q, gain_db }],
            radiated_share: 0.0,
        }
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
        r.drive(0, &own, &mut out);
        assert!(out.iter().all(|&v| v.abs() < 1e-9));
    }

    #[test]
    fn coupling_is_clamped_to_the_stable_range() {
        let mut r = bus();
        r.set_coupling(10.0);
        assert_eq!(r.coupling(), MAX_COUPLING);
        r.set_coupling(-1.0);
        assert_eq!(r.coupling(), 0.0);
        r.set_coupling(f32::NAN);
        assert_eq!(r.coupling(), 0.0);
    }

    /// `set_coupling` is a `pub` method on a `pub` type, so its clamp is a
    /// promise made to every caller and not only to the load path.
    ///
    /// On a bus with a voiced bridge, [`MAX_COUPLING`] is *not* that promise:
    /// the loop gain is `coupling · max|B|`, so a bridge peaking at 26.7 dB
    /// turns the 0.05 the flat bus was safe at into a realised loop gain of
    /// 1.08 — past self-sustainment, and the number `Preset::validate` refuses
    /// that very preset for. The bus therefore carries the bound its own
    /// filter implies and clamps against that.
    #[test]
    fn set_coupling_cannot_walk_a_bridged_bus_past_its_loop_bound() {
        for gain_db in [6.0f32, 20.0] {
            let mut preset = Preset::default();
            preset.voicing.bridge = Some(one_peak(114.0, 18.0, gain_db));
            preset.voicing.resonance_coupling = 0.001;
            assert!(preset.validate().is_ok());
            let max_b = BridgeFilter::new(preset.voicing.bridge.as_ref().unwrap()).max_magnitude();

            let mut r = ResonanceBus::from_preset(&preset);
            assert_eq!(r.coupling(), 0.001, "the preset's own value was changed");
            for asked in [MAX_COUPLING, 1.0, 1.0e9] {
                r.set_coupling(asked);
                let realised = r.coupling() * max_b;
                assert!(
                    realised <= MAX_BRIDGE_LOOP_GAIN * 1.0001,
                    "set_coupling({asked}) on a {gain_db} dB bridge realised a loop gain of \
                     {realised}, past the {MAX_BRIDGE_LOOP_GAIN} the bus is stable under"
                );
            }
            // ... and it is not gratuitously tighter than the bound either: the
            // ceiling is the bound, to the last few bits.
            assert!((r.ceiling() - MAX_COUPLING.min(MAX_BRIDGE_LOOP_GAIN / max_b)).abs() < 1.0e-9);
        }
        // A flat bus is exactly as free as it always was.
        let mut r = ResonanceBus::from_preset(&Preset::default());
        assert_eq!(r.ceiling(), MAX_COUPLING);
        r.set_coupling(1.0);
        assert_eq!(r.coupling(), MAX_COUPLING);
    }

    /// The shipped preset must be *provably* legal, not marginally legal: a
    /// file fitted onto the exact edge of the refusal test becomes unloadable
    /// the first time the measurement it was fitted against gets sharper.
    #[test]
    fn the_shipped_presets_have_real_headroom_against_the_loop_bound() {
        for name in ["default.toml", "salamander-c5.toml"] {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../presets")
                .join(name);
            let preset = Preset::from_toml(&std::fs::read_to_string(&path).expect("preset file"))
                .expect("the shipped preset is valid");
            let ceiling = preset.max_safe_coupling();
            let used = preset.voicing.resonance_coupling / ceiling;
            println!(
                "{name}: coupling {:.9}, ceiling {ceiling:.9}, {:.1} % of the bound \
                 ({:.2} dB of headroom)",
                preset.voicing.resonance_coupling,
                100.0 * used,
                -20.0 * used.log10()
            );
            assert!(
                used <= 0.95,
                "{name} asks for {:.4} % of its loop bound: one extra grid point in the \
                 max|B| measurement and the engine refuses its own preset",
                100.0 * used
            );
        }
    }

    #[test]
    fn drive_is_bounded_however_loud_the_bus_gets() {
        let mut r = bus();
        let huge = [1.0e12f32; BLOCK];
        r.contribute(&huge);
        r.begin_block();
        let mut out = [0.0f32; BLOCK];
        r.drive(0, &[0.0; BLOCK], &mut out);
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
                r.drive(i, &previous[i], &mut input);
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
                r.drive(i, &previous[i], &mut input);
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

    // ------------------------------------------------ the bridge admittance

    #[test]
    fn no_bridge_section_is_the_flat_bus_sample_for_sample() {
        let mut filter = BridgeFilter::unity();
        assert!(filter.is_unity());
        assert_eq!(filter.max_magnitude(), 1.0);
        let mut input = [0.0f32; BLOCK];
        for (i, x) in input.iter_mut().enumerate() {
            *x = (i as f32 * 0.37).sin() * 0.5;
        }
        let mut output = [0.0f32; BLOCK];
        filter.process(&input, &mut output);
        assert_eq!(input, output);
        // ... and so is the bus built from a preset that has no section.
        let r = ResonanceBus::from_preset(&Preset::default());
        assert!(r.bridge().is_unity());
        assert!(r.own_gain.iter().all(|&g| g == 1.0));
    }

    /// The backbone is a fit, so what it has to do is pass through its own
    /// anchors — measured on the sections that were actually built.
    #[test]
    fn the_backbone_passes_through_its_anchors() {
        // The shape `PHYSICS.md` §4 describes: a mean mobility over the
        // midrange, a slight dip towards 2 kHz, a slight rise to 4 kHz, and a
        // fall above it.
        let anchors = [
            (30.0, -12.0),
            (100.0, -2.0),
            (300.0, 1.5),
            (1_100.0, 0.0),
            (2_000.0, -4.0),
            (4_000.0, -1.0),
            (8_000.0, -9.0),
            (14_000.0, -18.0),
        ];
        let bridge = BridgeVoicing {
            backbone: anchors
                .iter()
                .map(|&(hz, gain_db)| BridgeAnchor { hz, gain_db })
                .collect(),
            peaks: Vec::new(),
            radiated_share: 0.0,
        };
        let filter = BridgeFilter::new(&bridge);
        for (hz, want) in anchors {
            let got = amp_to_db(filter.magnitude(hz));
            assert!(
                (got - want).abs() < 1.0,
                "backbone at {hz} Hz is {got:.2} dB, anchored at {want:.2}"
            );
        }
        // Below the first anchor and above the last it is clamped, like every
        // other anchor table in the preset.
        assert!((amp_to_db(filter.magnitude(21.0)) - (-12.0)).abs() < 1.5);
    }

    /// A peak is a peak: the response at its centre is up by what the preset
    /// asked for, and an octave away it is not.
    #[test]
    fn a_bridge_peak_lands_where_it_was_put() {
        let filter = BridgeFilter::new(&one_peak(262.0, 20.0, 12.0));
        assert!((amp_to_db(filter.magnitude(262.0)) - 12.0).abs() < 0.2);
        assert!(amp_to_db(filter.magnitude(131.0)).abs() < 1.0);
        assert!(amp_to_db(filter.magnitude(524.0)).abs() < 1.0);
        // ... and the measured maximum finds it even though a Q-20 resonance
        // is narrower than the log grid's step.
        assert!((amp_to_db(filter.max_magnitude()) - 12.0).abs() < 0.2);
    }

    /// The measurement the stability contract rests on must not be fooled by a
    /// resonance narrower than the grid it is measured on.
    #[test]
    fn the_measured_maximum_sees_the_sharpest_resonance_the_schema_allows() {
        for hz in [20.0f32, 41.2, 261.6, 3_520.0, 15_000.0] {
            let filter = BridgeFilter::new(&one_peak(hz, MAX_BRIDGE_Q, 20.0));
            let got = amp_to_db(filter.max_magnitude());
            assert!(
                got > 19.0,
                "a Q-{MAX_BRIDGE_Q} +20 dB resonance at {hz} Hz measured as {got:.1} dB"
            );
        }
    }

    /// A dense sweep of the realised response — the ground truth
    /// [`BridgeFilter::max_magnitude`]'s sampling scheme is held against.
    fn dense_max(filter: &BridgeFilter, lo: f64, hi: f64, points: usize) -> f64 {
        let mut max = 0.0f64;
        for i in 0..points {
            let hz = lo + (hi - lo) * i as f64 / (points - 1) as f64;
            max = max.max(filter.magnitude_at(hz));
        }
        max
    }

    /// The measurement the whole stability contract rests on cannot be evaded
    /// by a preset that is *inside* the schema.
    ///
    /// A cascade adds decibels, so two clusters of resonances put their
    /// maximum **between** their centres. Twenty `Q`-50 +20 dB peaks at
    /// 101.63 Hz and twenty at 102.32 Hz sit inside one interval of the coarse
    /// log grid, and a measurement that sampled only that grid and the peaks'
    /// own centres read 15.6 dB less than the response really reaches — enough
    /// to certify a preset whose realised loop gain is 1.5, i.e. one that
    /// sustains itself. Every split below is checked against a dense scan of
    /// the same band: what the validator is told must be what the filter does.
    #[test]
    fn the_measured_maximum_cannot_be_hidden_between_grid_points() {
        for (a, b, count, gain_db) in [
            // The reviewer's construction, unchanged: both centres inside one
            // interval of the coarse grid, every field inside the schema.
            (101.63f32, 102.32f32, 20, 20.0f32),
            (101.63, 102.0, 20, 20.0),
            (101.63, 101.9, 10, 6.0),
            (41.2, 41.35, 10, 6.0),
            (3_520.0, 3_530.0, 10, 6.0),
        ] {
            let mut peaks = Vec::new();
            for hz in [a, b] {
                for _ in 0..count {
                    peaks.push(BridgePeak {
                        hz,
                        q: MAX_BRIDGE_Q,
                        gain_db,
                    });
                }
            }
            let filter = BridgeFilter::new(&BridgeVoicing {
                backbone: vec![
                    BridgeAnchor {
                        hz: 20.0,
                        gain_db: 0.0,
                    },
                    BridgeAnchor {
                        hz: 16_000.0,
                        gain_db: 0.0,
                    },
                ],
                peaks,
                radiated_share: 0.0,
            });
            let measured = amp_to_db(filter.max_magnitude()) as f64;
            let span = (b - a).max(a / MAX_BRIDGE_Q) as f64 * 8.0;
            let dense =
                20.0 * dense_max(&filter, a as f64 - span, b as f64 + span, 2_000_001).log10();
            assert!(
                measured >= dense - 0.01,
                "peaks at {a}/{b} Hz measure {measured:.3} dB but reach {dense:.3} dB: \
                 {:.3} dB hidden between the samples",
                dense - measured
            );
        }
    }

    /// ... and the same thing said as the property that matters: a preset the
    /// validator accepts really does have the loop gain it was accepted for.
    #[test]
    fn an_accepted_preset_realises_the_loop_gain_it_was_accepted_for() {
        let mut peaks = Vec::new();
        for hz in [101.63f32, 102.32] {
            for _ in 0..10 {
                peaks.push(BridgePeak {
                    hz,
                    q: MAX_BRIDGE_Q,
                    gain_db: 6.0,
                });
            }
        }
        let bridge = BridgeVoicing {
            backbone: vec![
                BridgeAnchor {
                    hz: 20.0,
                    gain_db: 0.0,
                },
                BridgeAnchor {
                    hz: 16_000.0,
                    gain_db: 0.0,
                },
            ],
            peaks,
            radiated_share: 0.0,
        };
        let filter = BridgeFilter::new(&bridge);
        let mut preset = Preset::default();
        // Exactly the coupling that puts the *measured* loop gain on the bound.
        preset.voicing.resonance_coupling = MAX_BRIDGE_LOOP_GAIN / filter.max_magnitude();
        preset.voicing.bridge = Some(bridge);
        assert!(preset.validate().is_ok(), "the probe preset is not legal");

        let dense = dense_max(&filter, 100.0, 104.0, 4_000_001) as f32;
        let realised = preset.voicing.resonance_coupling * dense;
        assert!(
            realised <= MAX_BRIDGE_LOOP_GAIN * 1.001,
            "an accepted preset realises a loop gain of {realised}, past the \
             {MAX_BRIDGE_LOOP_GAIN} it was certified for"
        );
    }

    /// A cascade of resonances multiplies, and the validator is what stands
    /// between a preset that stacks them and a bus that runs away.
    #[test]
    fn stacked_peaks_are_refused_by_the_loop_gain_check() {
        let mut preset = Preset::default();
        let peaks: Vec<BridgePeak> = (0..8)
            .map(|_| BridgePeak {
                hz: 261.6,
                q: 8.0,
                gain_db: 20.0,
            })
            .collect();
        preset.voicing.bridge = Some(BridgeVoicing {
            backbone: vec![
                BridgeAnchor {
                    hz: 20.0,
                    gain_db: 0.0,
                },
                BridgeAnchor {
                    hz: 16_000.0,
                    gain_db: 0.0,
                },
            ],
            peaks,
            radiated_share: 0.0,
        });
        assert!(preset.validate().is_err());
        // The realised filter really is that loud, i.e. the refusal is not an
        // artefact of the bound being too tight.
        let filter = BridgeFilter::new(preset.voicing.bridge.as_ref().unwrap());
        assert!(amp_to_db(filter.max_magnitude()) > 100.0);
    }

    /// The self-cancellation gain follows the note: a key whose partials all
    /// sit where the bridge is quiet has its own contribution taken off quietly
    /// too, so a tilted backbone does not become a global damping change.
    #[test]
    fn the_own_gain_follows_the_bridge_across_the_compass() {
        let mut preset = Preset::default();
        preset.voicing.bridge = Some(BridgeVoicing {
            backbone: vec![
                BridgeAnchor {
                    hz: 20.0,
                    gain_db: 6.0,
                },
                BridgeAnchor {
                    hz: 500.0,
                    gain_db: 6.0,
                },
                BridgeAnchor {
                    hz: 4_000.0,
                    gain_db: -18.0,
                },
                BridgeAnchor {
                    hz: 16_000.0,
                    gain_db: -18.0,
                },
            ],
            peaks: Vec::new(),
            radiated_share: 0.0,
        });
        assert!(preset.validate().is_ok());
        let r = ResonanceBus::from_preset(&preset);
        let bass = r.own_gain(key_index(28).unwrap());
        let treble = r.own_gain(key_index(105).unwrap());
        assert!(
            amp_to_db(bass) > 4.0,
            "bass own gain is {:.1} dB under a +6 dB backbone",
            amp_to_db(bass)
        );
        assert!(
            amp_to_db(treble) < -12.0,
            "treble own gain is {:.1} dB under a -18 dB backbone",
            amp_to_db(treble)
        );
    }

    /// Runs a pair of co-tuned one-partial strings of `key` through the bus for
    /// `seconds`, one of them struck, and returns the struck string's level and
    /// the partner's, in dB, over the last fifth of a second.
    ///
    /// The difference between them is what the struck string has put *into the
    /// bus* and got back out at another string, as a fraction of what it still
    /// holds: the rate at which it is shedding energy sideways.
    fn exchange(key: u8, bridge: Option<BridgeVoicing>, coupling: f32, seconds: f32) -> (f32, f32) {
        let mut preset = Preset::default();
        preset.voicing.resonance_coupling = coupling;
        preset.voicing.bridge = bridge;
        assert!(preset.validate().is_ok(), "the probe preset is not valid");
        let mut r = ResonanceBus::from_preset(&preset);

        let mut banks = [partial(key, 0.0), partial(key, 0.0)];
        let mut previous = [[0.0f32; BLOCK]; 2];
        let index = key_index(key).unwrap();
        let blocks = (seconds * SAMPLE_RATE / BLOCK as f32) as usize;
        let tail = (0.2 * SAMPLE_RATE / BLOCK as f32) as usize;
        let (mut struck, mut partner) = (0.0f32, 0.0f32);
        for b in 0..blocks {
            r.begin_block();
            for (i, bank) in banks.iter_mut().enumerate() {
                let mut input = [0.0f32; BLOCK];
                if b == 0 && i == 0 {
                    // Large enough that a -60 dB halo is still a hundred dB
                    // clear of `ModalBank`'s culling floor.
                    input[0] = 1.0e6;
                }
                r.drive(index, &previous[i], &mut input);
                let mut out = [0.0f32; BLOCK];
                bank.process_add(&input, &mut out);
                r.contribute(&out);
                previous[i].copy_from_slice(&out);
            }
            if b + tail >= blocks {
                struck = struck.max(peak(&previous[0]));
                partner = partner.max(peak(&previous[1]));
            }
        }
        (amp_to_db(struck), amp_to_db(partner))
    }

    /// The two-way half of `PHYSICS.md` §4: the coupling stops being
    /// spectrally uniform, so how fast a string sheds energy into the bus
    /// depends on where its partial sits on `B`.
    ///
    /// Two co-tuned one-partial strings, one struck; the only difference
    /// between the three runs is a ±12 dB bridge resonance under the note. What
    /// is measured is the partner's level *relative to the struck string's*,
    /// which is the fraction of its own energy the struck string has handed to
    /// the rest of the instrument — a rate, not a level, so it does not depend
    /// on how hard the probe was struck or how far the note has already
    /// decayed. Two keys 5¼ octaves apart, because a bridge that only works in
    /// one register is not a bridge.
    #[test]
    fn a_partial_on_a_bridge_peak_sheds_energy_into_the_bus_faster_than_one_in_a_trough() {
        const COUPLING: f32 = 0.02;
        for key in [21u8, 84] {
            let f0 = Preset::default().f0(key);
            let shed = |bridge| {
                let (struck, partner) = exchange(key, bridge, COUPLING, 2.0);
                partner - struck
            };
            let flat = shed(None);
            let on_peak = shed(Some(one_peak(f0, 12.0, 12.0)));
            let in_trough = shed(Some(one_peak(f0, 12.0, -12.0)));
            assert!(
                on_peak - flat > 9.0,
                "key {key}: a +12 dB bridge resonance only sped the transfer up by \
                 {:.1} dB ({flat:.1} flat, {on_peak:.1} on the peak)",
                on_peak - flat
            );
            assert!(
                flat - in_trough > 9.0,
                "key {key}: a -12 dB bridge trough only slowed the transfer by {:.1} dB",
                flat - in_trough
            );
            assert!(
                on_peak - in_trough > 20.0,
                "key {key}: peak {on_peak:.1} dB against trough {in_trough:.1} dB"
            );
        }
    }

    /// … and the half of `PHYSICS.md` §4 that does **not** fall out free.
    ///
    /// §4 expects the two-way coupling to bring a *decay-rate* coupling with
    /// it: a partial on a bridge resonance should also die faster. It does not,
    /// and it cannot, because the drive path subtracts the string's own
    /// contribution — nothing in the loop is proportional to the string's own
    /// motion with a phase that removes energy from it, so the bus can only
    /// ever *add* drive. What is left is the second-order path, the partner's
    /// answer coming back, and at these couplings its exchange time is two
    /// orders of magnitude longer than the note itself.
    ///
    /// This test pins that: at the maximum coupling the schema allows and with
    /// a resonance right on the fundamental, the struck string's own level
    /// moves by well under a decibel over five seconds, while the energy it
    /// hands to its neighbour moves by twelve. The decay-rate half is therefore
    /// **not** in this module. It is in `string.rs`, at the partial's own pole:
    /// `voicing.bridge.radiated_share` gives each partial the share of its
    /// fitted decay that is loss into the board and lets the board's own modes
    /// modulate it (`string::tests::a_partial_on_a_board_mode_decays_faster_\
    /// than_the_fitted_law`). What this test says is that the bus does not do
    /// it and must not be expected to, whatever the bridge is voiced like.
    #[test]
    fn the_bridge_does_not_move_the_struck_strings_own_decay() {
        const KEY: u8 = 21;
        let f0 = Preset::default().f0(KEY);
        let (flat, _) = exchange(KEY, None, MAX_COUPLING, 5.0);
        let (on_peak, _) = exchange(KEY, Some(one_peak(f0, 12.0, 12.0)), MAX_COUPLING, 5.0);
        let (in_trough, _) = exchange(KEY, Some(one_peak(f0, 12.0, -12.0)), MAX_COUPLING, 5.0);
        for (name, level) in [("peak", on_peak), ("trough", in_trough)] {
            assert!(
                (level - flat).abs() < 1.0,
                "the struck string's own level moved {:.2} dB on a bridge {name}: \
                 the decay-rate coupling would be a finding, not a regression",
                level - flat
            );
        }
    }
}
