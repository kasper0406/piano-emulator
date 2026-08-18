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
//!
//! # The virtual microphone pair (`PHYSICS.md` §8)
//!
//! Both stereo constructions above are *inverted* with respect to a real
//! recording of a real piano, and by a wide margin — see [`Mics`] and
//! `DECISIONS.md` 351-358. `voicing.mics`, when a preset has it, replaces them
//! with a spaced pair of virtual capsules above the string band: a per-source
//! delay and gain to each capsule from where that source sits along the
//! bass-treble axis, and a frequency-dependent coherence on the board's diffuse
//! field. Absent, none of it is built and the pan-pot renders bit for bit what
//! it always did (`DECISIONS.md` 103).

use crate::modal::ModalBank;
use crate::preset::{MicVoicing, ModalBand, Radiation as RadiationCurve, SoundboardVoicing};
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

/// Speed of sound in air at room temperature, m/s. The only physical constant
/// the microphone geometry needs: it turns metres into samples.
pub const SPEED_OF_SOUND: f32 = 343.0;

/// Largest capsule separation a preset may ask for, metres.
///
/// It is a bound on the state as much as on the taste: the interchannel delay
/// of a spaced pair can never exceed `spacing / SPEED_OF_SOUND` (triangle
/// inequality, and it is attained only by a source on the line through both
/// capsules), so this number is what sizes [`MIC_TAIL`].
pub const MAX_MIC_SPACING_M: f32 = 1.0;

/// Bounds on the rest of the geometry, metres: capsule height above the string
/// plane, and the half-width of the string band.
pub const MIC_HEIGHT_M: (f32, f32) = (0.02, 3.0);
pub const MIC_SPAN_M: (f32, f32) = (0.05, 3.0);

/// Bounds on the two dimensionless trims, `width` and `diffuse_coherence`.
pub const MIC_WIDTH: (f32, f32) = (0.0, 2.0);
pub const MIC_DIFFUSE_COHERENCE: (f32, f32) = (0.25, 8.0);

/// Bounds on the mode-controlled band's two edges, Hz, and on its lift.
///
/// The edges are ordered — `lo < hi` is validated separately — and both are
/// held inside the range over which a soundboard is plausibly mode-controlled
/// at all: above the lowest body mode a preset can declare and below the
/// radiation transition, which Suzuki puts at 1-1.6 kHz for a grand.
pub const MIC_MODAL_HZ: (f32, f32) = (40.0, 2_000.0);

/// Bound on [`ModalBand::lift`]. It is an amplitude ratio between the pair's
/// difference and its sum inside the band, so **one is the null**: at one, the
/// anti-phase copy exactly cancels one capsule in-band. The recording asks for
/// `10^(3.5/20) = 1.5`; the ceiling is that with room, and it is a rail rather
/// than a taste.
pub const MIC_MODAL_LIFT: (f32, f32) = (0.0, 6.0);

/// Section `Q`s of the mode-controlled lobe's lower edge: an **eighth-order
/// Butterworth** highpass, as four second-order sections.
///
/// **The order is a measurement, not a preference.** The recording's own
/// sixth-octave interchannel profile (`piano-tuner mics --stage profile`,
/// `DECISIONS.md` 369) falls from `r0 = +0.940` at 127 Hz to `-0.529` at
/// 180 Hz. Read as a side-over-mid amplitude — which under [`ModalLobe`]'s
/// present form is `lift · |H(f)|` and nothing else, and which under its first
/// one was `(M - S)/(M + S)` inverted for uncorrelated taps — that
/// is 0.176 to 1.80 over half an octave: **40 dB per octave**, repeated to
/// within 0.1 by the same keys' other velocity layer. Nothing shallower
/// reaches it, and the shape has to be *maximally flat above the corner* as
/// well as steep below it: four **identical** Butterworth sections are also 48
/// dB/octave and were tried first, and they fail, because a cascade of
/// identical sections is still 0.46 of its passband gain a fifth above its
/// corner where a true Butterworth is at 0.97. Measured, that difference is
/// the whole 125-250 Hz band — the identical cascade could not take it below
/// +0.39 at any edge that left 63-125 Hz intact.
///
/// The values are `1 / (2 cos((2k+1) pi / 16))`, k = 0..3.
const MIC_MODAL_HIGH_Q: [f32; 4] = [0.509_796_2, 0.601_344_9, 0.899_976_2, 2.562_915_4];

/// Section `Q`s of the upper edge: a **fourth-order Butterworth** lowpass,
/// `1 / (2 cos((2k+1) pi / 8))`. Twenty-four dB per octave, which is all the
/// measured return to coherence asks for — from 320 Hz to 8 kHz every point of
/// the profile but one is inside ±0.25 of zero (the exception is 508 Hz at
/// −0.34), so there is no second edge in it for a steeper section to resolve.
const MIC_MODAL_LOW_Q: [f32; 2] = [0.541_196_1, 1.306_562_9];

/// Samples of carry-over the direct path's difference signal needs: the longest
/// interchannel delay [`MAX_MIC_SPACING_M`] can produce, plus one for the
/// fractional-delay interpolator's second tap.
const MIC_TAIL: usize = (MAX_MIC_SPACING_M * SAMPLE_RATE / SPEED_OF_SOUND) as usize + 2;

/// Where the first-order model of the diffuse field's interchannel coherence
/// puts its pole, as a multiple of `c / spacing`.
///
/// The spatial coherence of an *isotropic diffuse field* between two omni
/// capsules a apart is the classical `sin(kd) / kd` (Cook et al., JASA 27, 1955)
/// — one at DC, one half at `kd = 1.8955`, through zero at `kd = pi`. The board
/// field here is realised as a mid signal shared by both channels plus a
/// one-pole *highpassed* difference, whose coherence is `|LP|^2 / (2 - |LP|^2)`
/// with `|LP|^2 = 1 / (1 + (f/f_s)^2)`; that reads one half at `f_s / sqrt(2)`.
/// Equating the two half-coherence points gives
/// `f_s = sqrt(2) * 1.8955 / (2 pi) * c / d = 0.4266 c / d`, which is this
/// constant. Nothing else about the curve is fitted: the pole follows the
/// spacing, and `diffuse_coherence` is the one number that says how much less
/// isotropic than a room's field the board's own is.
const MIC_DIFFUSE_POLE_K: f32 = 0.426_63;

/// Level above which the safety limiter starts to bend the signal (-1 dBFS).
///
/// Public because it is half of a contract rather than an implementation
/// detail: [`soft_clip`] is the identity under it and strictly expanding over
/// it, so `|y| > LIMIT_THRESHOLD` in a finished render is exactly "the safety
/// limiter engaged here" and a budget can be read off the output with no
/// instrumentation at all (`DECISIONS.md` 42, 264).
pub const LIMIT_THRESHOLD: f32 = 0.891_251;

/// Stereo position of a key: -1.0 hard left, +1.0 hard right.
pub fn pan_for_key(key: u8) -> f32 {
    (2.0 * key_position(key) - 1.0) * MAX_PAN
}

/// The virtual microphone pair: two listening points above the string band.
///
/// # What it replaces, and why
///
/// The engine had two stereo constructions and both are the *reverse* of what
/// a recording of a real piano measures (`DECISIONS.md` 314, 346-350):
///
/// * the direct sound was pan-potted — one mono voice scaled into two channels
///   — so it is correlated at **+1 at every frequency**, and reads +0.91 in the
///   6-12 kHz band where the recording reads +0.05;
/// * the board's diffuse field was tapped with two *orthogonal* sign patterns,
///   so it is decorrelated at every frequency, and drags the 63-125 Hz band to
///   **−0.58** where the recording reads +0.95.
///
/// A spaced pair does the opposite of both, and for one reason: two capsules
/// `d` apart are well inside a wavelength of each other in the bass and see one
/// wavefront, and are several wavelengths apart in the treble and see the same
/// sound at a delay. That is a *frequency-dependent* coherence, and neither a
/// pan-pot (coherent everywhere) nor an orthogonal tap pair (coherent nowhere)
/// can be it.
///
/// # The geometry
///
/// One axis, along the keyboard: bass at −1, treble at +1, which is exactly the
/// `pan` argument [`Soundboard::add_voice`] already carries. A source at pan `p`
/// sits at `x = p * span` on the string plane; the capsules sit at
/// `(∓spacing/2, height)` above it, left over the bass. So
///
/// ```text
/// d_L = hypot(x + spacing/2, height)      d_R = hypot(x - spacing/2, height)
/// u_L = (1/d_L) / sqrt(1/d_L^2 + 1/d_R^2) u_R likewise      (u_L^2 + u_R^2 = 1)
/// Δ   = (d_L - d_R) / c                                     (|Δ| <= spacing/c)
/// ```
///
/// — inverse-distance gains, normalised to equal power so that the pair as a
/// whole neither gains nor loses level across the compass, and the *difference*
/// of the two propagation delays, which is the whole of the interchannel time
/// difference. The common delay is dropped: it is a latency, every recording in
/// the library is trimmed to its own onset (`DECISIONS.md` 315), and nothing in
/// this repository can measure it.
///
/// A bass key is nearly equidistant from both capsules relative to a bass
/// wavelength, so it stays coherent; a treble key is off-axis, so its few
/// hundred microseconds of Δ are a large fraction of a treble period and it
/// decorrelates. That is the measured shape, produced by geometry rather than
/// asserted.
///
/// # Why the sum is untouched
///
/// Every scoreboard in this repository scores the **mono sum** — deliberately,
/// its own header says a stereo distance would mostly measure the recording's
/// microphones — so a stage that moved the mono sum would move every board at
/// once and none of them would be comparable across the change. It does not,
/// and not by tuning: the stage is written as *mid plus side*, and it replaces
/// only the side.
///
/// ```text
/// mid  = (g_L + g_R)/2 * x       the old equal-power pan's own sum, unchanged
/// side = width/2 * (u_L * x(t-δ_L) - u_R * x(t-δ_R))       the new geometry
///      + lift * butterworth_band(mid)          the nodal line, [`ModalLobe`]
/// L    = mid + side              R = mid - side
/// ```
///
/// `(L + R)/2 = mid`, identically, for every source, every pan and every
/// setting of the geometry — an equal-power, delay-compensated sum — and the
/// mode-controlled term cannot break that however large it is, because it is
/// added to the side and the side cancels. The same decomposition is applied to
/// the board field, whose two orthogonal taps are re-read as their own sum and
/// difference: the sum is bit-for-bit the mono the old tap pair folded down to,
/// and only the difference is filtered.
///
/// # The board field's coherence
///
/// The diffuse field is not a point source and has no ITD. What two capsules
/// `d` apart see of a diffuse field is a coherence that falls with frequency —
/// `sin(kd)/kd` for an isotropic one — so the board's side signal is
/// highpassed (see [`MIC_DIFFUSE_POLE_K`]): shared at low frequency, orthogonal
/// at high, crossing over where the wavelength stops being large compared with
/// the spacing. `diffuse_coherence` scales that corner, and is the one number
/// in the section that admits the board's field is *not* isotropic — it is the
/// near field of one large plate, so it stays organised over a longer distance
/// than a room's, and a value above one says by how much.
///
/// # Polarization spread survives as part of the image
///
/// `voicing.polarization_pan_spread` displaces the two polarizations of one key
/// to either side of it. Through this stage that is no longer a level trim: the
/// two planes reach the capsules from two *places*, so they carry two different
/// Δ as well as two different gains, and because the vertical polarization
/// decays several times faster than the horizontal one, a single note's
/// interchannel delay and level both **move while it rings**. That is the
/// measured drift `estimate::directivity` fits, now expressed in the image
/// rather than only in the balance.
#[derive(Clone, Copy, Debug)]
struct Mics {
    /// Half the capsule separation, metres.
    half_spacing: f32,
    height: f32,
    /// Metres from the centre of the string band to `|pan| = 1`.
    span: f32,
    /// Gain on the geometric difference signal; 1.0 is the geometry itself.
    width: f32,
    /// One-pole coefficient of the board field's side highpass.
    diffuse_b: f32,
    /// The board's mode-controlled band, when the preset declares one.
    lobe: Option<ModalLobe>,
}

impl Mics {
    fn new(v: &MicVoicing) -> Self {
        // The pole of the *lowpass* half of the pair; the side takes the
        // complement. `diffuse_coherence` above one moves it up, i.e. keeps the
        // field shared to a higher frequency than an isotropic one would be.
        let hz = MIC_DIFFUSE_POLE_K * SPEED_OF_SOUND / v.spacing_m * v.diffuse_coherence;
        // Above Nyquist the one-pole is a wire; clamping keeps the coefficient
        // in (0, 1) for any legal spacing at any sample rate.
        let w = (std::f32::consts::TAU * hz / SAMPLE_RATE).min(std::f32::consts::PI);
        Mics {
            half_spacing: 0.5 * v.spacing_m,
            height: v.height_m,
            span: v.span_m,
            width: v.width,
            diffuse_b: 1.0 - (-w).exp(),
            lobe: v.modal.as_ref().map(ModalLobe::new),
        }
    }

    /// The two capsule gains and the two capsule delays, in samples, of a
    /// source at pan position `pan`.
    ///
    /// Both delays are non-negative and at most one is non-zero: what the pair
    /// hears is the *difference*, so the nearer capsule is taken as the time
    /// origin and only the farther one is delayed.
    fn taps(&self, pan: f32) -> (f32, f32, f32, f32) {
        let x = pan.clamp(-1.0, 1.0) * self.span;
        let h2 = self.height * self.height;
        let dl = ((x + self.half_spacing).powi(2) + h2).sqrt();
        let dr = ((x - self.half_spacing).powi(2) + h2).sqrt();
        // Spherical spreading, normalised to unit power so the pair adds no
        // level of its own: the equal-power pan it replaces has `gl^2 + gr^2 = 1`
        // and so does this.
        let (al, ar) = (1.0 / dl, 1.0 / dr);
        let n = 1.0 / (al * al + ar * ar).sqrt();
        let delta = (dl - dr) * (SAMPLE_RATE / SPEED_OF_SOUND);
        // `|dl - dr| <= 2 * half_spacing <= MAX_MIC_SPACING_M`, so this clamp
        // never binds on a validated preset; it is what makes the buffer bound
        // a property of the code rather than of the caller.
        let limit = MIC_TAIL as f32 - 1.0;
        (
            al * n,
            ar * n,
            delta.max(0.0).min(limit),
            (-delta).max(0.0).min(limit),
        )
    }
}

/// One second-order section, transposed direct form II.
///
/// The form matters: the lobe is a cascade of six of these running on the
/// board's difference signal, which decays to nothing between notes, and
/// transposed form II is the arrangement whose state is bounded by the signal
/// rather than by an internal accumulator.
#[derive(Clone, Copy, Debug, Default)]
struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    s1: f32,
    s2: f32,
}

impl Biquad {
    /// One second-order section of a Butterworth cascade, `high` for a
    /// highpass and otherwise a lowpass. (Robert Bristow-Johnson's cookbook
    /// forms, normalised by `a0`.)
    fn butterworth(hz: f32, q: f32, high: bool) -> Self {
        let w = (std::f32::consts::TAU * hz / SAMPLE_RATE).clamp(1.0e-6, 3.0);
        let (sin, cos) = w.sin_cos();
        let alpha = sin / (2.0 * q);
        let a0 = 1.0 + alpha;
        let (b0, b1, b2) = if high {
            let g = (1.0 + cos) / 2.0;
            (g, -2.0 * g, g)
        } else {
            let g = (1.0 - cos) / 2.0;
            (g, 2.0 * g, g)
        };
        Biquad {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: -2.0 * cos / a0,
            a2: (1.0 - alpha) / a0,
            s1: 0.0,
            s2: 0.0,
        }
    }

    #[inline]
    fn run(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.s1;
        self.s1 = self.b1 * x - self.a1 * y + self.s2;
        self.s2 = self.b2 * x - self.a2 * y;
        y
    }

    fn clear(&mut self) {
        self.s1 = 0.0;
        self.s2 = 0.0;
    }
}

/// The board's **mode-controlled band**, as the capsule pair sees it
/// (`PHYSICS.md` §8, `DECISIONS.md` 369-371).
///
/// # What it is, and why a spaced pair alone cannot be it
///
/// The virtual pair reproduced the recording's image in four of six bands and
/// could not reproduce 125-500 Hz in principle: the recording reads **+0.953
/// below 125 Hz and −0.115 one octave above it**, and no two-point geometry —
/// `sin(kd)/kd`, a pure interchannel delay, or any mixture — falls from +0.95
/// through zero across a single octave (`DECISIONS.md` 357, and 362 measures
/// how far apart the two halves put the pair: the delays want 0.11 m, the
/// coherence 0.6-0.7 m).
///
/// What the fine-resolution profile shows is not a slope but **three regimes**,
/// and it is a soundboard's, not a room's:
///
/// * **below ~140 Hz** the board's first modes have no interior nodal line
///   between two capsules 12 cm apart and 12 cm up — the plate and its air load
///   move as one radiator and both capsules see one pressure. Measured
///   `r0 = +0.94` to `+0.98` from 80 to 127 Hz, on both takes.
/// * **~140 to ~500 Hz** the board is **mode-controlled**: individual modes
///   with a small number of nodal lines cross the string band, the pair
///   straddles one, and the two capsules see the same mode in *opposition*.
///   Measured `r0 = −0.36` to `−0.53` over 180-254 Hz, and a mid/side ratio of
///   **−3.5 dB** — the recording's own difference is larger than its sum there,
///   which is a thing a mono fold-down loses and no scoreboard in this
///   repository could see.
/// * **above ~500 Hz** modal overlap sets in, many modes sum with unrelated
///   phases, the near field is disorganised, and there is no sign to see:
///   measured `r0` inside ±0.25 of zero at every point from 320 Hz to 8 kHz
///   but one — 508 Hz reads −0.34, which is the profile's second, much
///   shallower dip and is not modelled.
///
/// # How it is built, and what it used to be built out of
///
/// The engine already has the third regime exactly: the FDN's two orthogonal
/// output taps are decorrelated by construction, so their *difference* is a
/// signal uncorrelated with their sum, and [`Mics`] highpasses that difference
/// to get the diffuse field's coherence. The first version of this lobe added
/// the second regime out of the same material — it band-limited the *difference*
/// and scaled it up. That was wrong twice over, and both are measurements
/// rather than opinions (`DECISIONS.md` 379):
///
/// * A nodal line is not an incoherent field. Two capsules straddling one hear
///   **the same signal with opposite signs**, which is a deterministic
///   anti-phase copy of the sound, not a louder helping of whatever the two
///   channels already disagreed about.
/// * A filter on the diffuse difference **cannot act during the strike**. The
///   FDN's shortest line is 149 samples, so its difference is exactly zero for
///   the first 3.1 ms of any note, and a band-limited copy of nothing is
///   nothing. Measured on the shipped preset that way round, C5's first 10 ms
///   read `+9.9 dB` mid over side in 125-250 Hz — the band is *mono* through
///   the strike — where the recording's own first 10 ms read `−1.6 dB`.
///
/// So the lobe runs on the **sum**, and it runs on both paths — the direct one,
/// which is where a note's first milliseconds live, and the board field:
///
/// ```text
/// direct: side = geometry(x)      + lift * butterworth_band(mid)
/// board:  side = highpass(l − r)  + lift * butterworth_band(l + r)
/// ```
///
/// `lift` is now a plain amplitude ratio: it is how much anti-phase copy the
/// pair sees against the sum it sees, so the measured mid/side ratio of
/// **−3.5 dB** in the mode-controlled band *is* `lift = 10^(3.5/20) = 1.5`,
/// read off the recording rather than searched for. Above one the difference is
/// larger than the sum, which is what a pair straddling a nodal line is.
///
/// The band's correlation does not simply rail at −1 the way a pure anti-phase
/// copy would, because the diffuse term is still there and is incoherent with
/// both: with `M` the sum's energy, `D` the diffuse difference's and `g` the
/// lift, `r0 = (M(1 − g²) − D) / sqrt((M(1 + g)² + D)(M(1 − g)² + D))`, which
/// passes smoothly through the measured `−0.36` to `−0.53` for a `g` a little
/// over one. The lift and the diffuse coherence are therefore *not*
/// interchangeable, which is what makes fitting them together well-posed.
///
/// It costs the mono fold-down nothing — `(L + R)/2` is the mid and the mid is
/// untouched at every setting, on both paths — which is also why the recording's
/// own mono sum is missing that energy, and why every board in this repository
/// is unmoved by the whole section.
#[derive(Clone, Copy, Debug)]
struct ModalLobe {
    high: [Biquad; MIC_MODAL_HIGH_Q.len()],
    low: [Biquad; MIC_MODAL_LOW_Q.len()],
    lift: f32,
}

impl ModalLobe {
    fn new(band: &ModalBand) -> Self {
        ModalLobe {
            high: MIC_MODAL_HIGH_Q.map(|q| Biquad::butterworth(band.lo_hz, q, true)),
            low: MIC_MODAL_LOW_Q.map(|q| Biquad::butterworth(band.hi_hz, q, false)),
            lift: band.lift,
        }
    }

    #[inline]
    fn run(&mut self, x: f32) -> f32 {
        let mut y = x;
        for section in &mut self.high {
            y = section.run(y);
        }
        for section in &mut self.low {
            y = section.run(y);
        }
        self.lift * y
    }

    fn clear(&mut self) {
        self.high.iter_mut().for_each(Biquad::clear);
        self.low.iter_mut().for_each(Biquad::clear);
    }
}

/// One peaking section in **double precision**, transposed direct form II.
///
/// # Why this one filter is not `f32`
///
/// Everything else in this engine runs in `f32` and is right to. A resonator at
/// the *bottom* of the audible band is where that stops being free: a section at
/// 180 Hz with `RADIATION_Q` has a pole radius of `0.9973`, and transposed form
/// II's state settles at about `1/(1 − r)` — **370 times** the signal running
/// through it. Rounding that state in `f32` is rounding at 370 ulps of the
/// signal, and nineteen such sections in cascade put the result about **−77 dB**
/// under it. Measured, on the invariant that is most sensitive to it: the
/// microphone pair's fold-down against the pan-pot's own render moved from the
/// **−116 dB** it has always sat at to **−70 dB** with this cascade in `f32`,
/// and back to −116 dB in `f64`. That is the mono-discipline contract
/// (`CONTEXT.md`, and `the_radiated_response_leaves_the_mono_sum_where_the_pan_pot_put_it`)
/// bought for three chains of nineteen double-precision biquads — about 14
/// MFLOP/s of the ~30 % of one core the whole instrument costs.
#[derive(Clone, Copy, Debug, Default)]
struct WideBiquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    s1: f64,
    s2: f64,
}

impl WideBiquad {
    /// The cookbook's `peakingEQ`, normalised by `a0`.
    ///
    /// Minimum-phase at every setting, which is what [`Radiation`] needs and is
    /// a property of the form rather than of the numbers put in it: the zero
    /// pair's product of roots is `(1 − αA)/(1 + αA)` and the pole pair's is
    /// `(1 − α/A)/(1 + α/A)`, both strictly inside the unit circle for any
    /// positive `α` and `A`, and the stability triangle's remaining condition
    /// reduces to `cos ω0 < 1`.
    fn peaking(hz: f64, q: f64, gain_db: f64) -> Self {
        let a = 10.0f64.powf(gain_db / 40.0);
        let w = (std::f64::consts::TAU * hz / f64::from(SAMPLE_RATE)).clamp(1.0e-9, 3.0);
        let (sin, cos) = w.sin_cos();
        let alpha = sin / (2.0 * q);
        let a0 = 1.0 + alpha / a;
        WideBiquad {
            b0: (1.0 + alpha * a) / a0,
            b1: -2.0 * cos / a0,
            b2: (1.0 - alpha * a) / a0,
            a1: -2.0 * cos / a0,
            a2: (1.0 - alpha / a) / a0,
            s1: 0.0,
            s2: 0.0,
        }
    }

    #[inline]
    fn run(&mut self, x: f64) -> f64 {
        let y = self.b0 * x + self.s1;
        self.s1 = self.b1 * x - self.a1 * y + self.s2;
        self.s2 = self.b2 * x - self.a2 * y;
        y
    }

    /// This section's magnitude at `hz`, dB, on the coefficients it will
    /// actually run with.
    fn magnitude_db(&self, hz: f64) -> f64 {
        let w = std::f64::consts::TAU * hz / f64::from(SAMPLE_RATE);
        let (s1, c1) = w.sin_cos();
        let (s2, c2) = (2.0 * w).sin_cos();
        let num = (self.b0 + self.b1 * c1 + self.b2 * c2).hypot(-(self.b1 * s1) - self.b2 * s2);
        let den = (1.0 + self.a1 * c1 + self.a2 * c2).hypot(-(self.a1 * s1) - self.a2 * s2);
        20.0 * (num / den).log10()
    }

    /// The larger pole magnitude, for the stability assertion.
    fn pole_radius(&self) -> f64 {
        let disc = self.a1 * self.a1 - 4.0 * self.a2;
        if disc >= 0.0 {
            let r = disc.sqrt();
            (0.5 * (-self.a1 + r)).abs().max((0.5 * (-self.a1 - r)).abs())
        } else {
            self.a2.abs().sqrt()
        }
    }

    fn clear(&mut self) {
        self.s1 = 0.0;
        self.s2 = 0.0;
    }
}

/// Bandwidth of each section [`Radiation`] realises a declared point with: a
/// third of an octave, as the cookbook's `Q = 1 / (2 sinh(ln2/2 · BW))`.
///
/// It is a **third** of an octave for a curve declared every **sixth**, and the
/// factor of two is the whole of the choice. Sections a sixth of an octave wide
/// barely reach their neighbours, so the cascade passes through the declared
/// points and ripples between them — and a band's score is the energy
/// *integrated across the band*, so a peak that is right only at the centre
/// under-delivers the band. Sections a third of an octave wide overlap their
/// neighbours by half, which is what makes the realised curve a smooth
/// interpolation of the declared points; the overlap is then removed exactly by
/// [`Radiation::design`] rather than approximately by hand.
const RADIATION_Q: f32 = 4.318_44;

/// Newton rounds [`Radiation::design`] takes.
///
/// The iteration is a *fixed*-Jacobian Newton — the matrix is built once, at
/// one decibel per section, and the cookbook's peaking section is not exactly
/// linear in its gain — so it converges linearly rather than quadratically, at
/// about a third of the residual per round on the fitted curve. Sixteen rounds
/// take a seven-decibel initial error under the `f32` coefficients' own noise
/// and cost microseconds once, at preset load.
const RADIATION_DESIGN_ROUNDS: usize = 16;

/// Rail on one *section's* gain, dB. The declared curve is railed by
/// `preset::validate_radiation`; this bounds what the design may ask a section
/// for while it inverts the overlap, so a pathological curve cannot arrive on
/// the audio thread as a resonance nobody wrote down.
const RADIATION_SECTION_CEILING_DB: f32 = 48.0;

/// **The strings' radiated response between their partials** — the stage
/// `DECISIONS.md` 407-412 found missing, and the one thing in this file that
/// moves the mono fold-down on purpose.
///
/// # What it is
///
/// A cascade of peaking sections, run on the drive the voices hand the board:
/// under a microphone pair on the direct path's sum and difference, without one
/// on the two pan-potted channels, and in both cases on the board's own drive
/// as well. That placement is the claim, and it is the measured one — item
/// 407(c) ablated the comb through seven instruments and it survived every
/// stage with a knob (`body_modes`, `voicing.bridge`, the sympathetic bus, the
/// strike noise) and did not survive removing the strings' own radiated path.
///
/// Everything downstream of a voice and upstream of `board_mix` is linear, so
/// filtering the three accumulators is *exactly* filtering every voice's own
/// output before it is added — the pan gains and the capsule delays are
/// constants and commute with a filter. That is why one cascade per accumulator
/// buys what eighty-eight per voice would, and it is why the section is
/// affordable at all.
///
/// # What it is not
///
/// It is not on the resonance bus. `resonance.rs` carries the bridge admittance
/// the other strings are driven through, item 407(c) charged the comb to a
/// different path, and colouring what a listener hears must not re-tune what
/// the instrument hears of itself.
///
/// # How a declared curve becomes coefficients
///
/// Sections a third of an octave wide overlap, so the gain a section is given
/// is not the response the band ends up with. The overlap is inverted rather
/// than lived with. In decibels a cascade's magnitude is the **sum** of its
/// sections', so with `M[j][i]` the decibels section `i` puts at centre `j` for
/// one decibel of its own gain — a matrix that depends only on the centres and
/// on `RADIATION_Q` — the design is Newton's method on `M g = t`:
///
/// ```text
/// g ← t;  repeat: r_j = Σ_i dB|H_i(f_j; g_i)|;  g ← g + M⁻¹ (t − r)
/// ```
///
/// `M` is factored once. The iteration is there because the cookbook's peaking
/// section is not *exactly* linear in its gain in decibels; three rounds take
/// the worst declared point to under a hundredth of a decibel, which
/// `the_realised_response_passes_through_every_declared_point` asserts.
struct Radiation {
    /// The same coefficients three times over, one per accumulator the drive is
    /// carried on: two direct channels and the board's drive. Separate state,
    /// identical filter — which is what keeps the mono fold-down identical to
    /// the pan-pot's (`DECISIONS.md` 353).
    chains: [Vec<WideBiquad>; 3],
}

impl Radiation {
    fn new(curve: &RadiationCurve) -> Self {
        let sections = Self::design(&curve.hz, &curve.gain_db);
        Radiation {
            chains: [sections.clone(), sections.clone(), sections],
        }
    }

    /// The cascade that puts `gain_db[i]` at `hz[i]`; see the type's header.
    fn design(hz: &[f32], gain_db: &[f32]) -> Vec<WideBiquad> {
        let n = hz.len();
        debug_assert_eq!(n, gain_db.len());
        let centres: Vec<f64> = hz.iter().map(|&f| f64::from(f)).collect();
        let target: Vec<f64> = gain_db.iter().map(|&g| f64::from(g)).collect();
        let q = f64::from(RADIATION_Q);
        // One decibel of section `i`, read at centre `j`. The unit is small
        // enough that the cookbook's own nonlinearity in the gain is out of the
        // matrix and left to the iteration.
        let mut m = vec![0.0f64; n * n];
        for i in 0..n {
            let probe = WideBiquad::peaking(centres[i], q, 1.0);
            for j in 0..n {
                m[j * n + i] = probe.magnitude_db(centres[j]);
            }
        }
        let lu = LuFactors::of(m, n);
        let ceiling = f64::from(RADIATION_SECTION_CEILING_DB);
        let mut gains: Vec<f64> = target.clone();
        let mut sections: Vec<WideBiquad> = (0..n)
            .map(|i| WideBiquad::peaking(centres[i], q, gains[i]))
            .collect();
        for _ in 0..RADIATION_DESIGN_ROUNDS {
            let Some(lu) = lu.as_ref() else { break };
            let residual: Vec<f64> = (0..n)
                .map(|j| target[j] - Self::magnitude_db(&sections, centres[j]))
                .collect();
            let step = lu.solve(&residual);
            for i in 0..n {
                gains[i] = (gains[i] + step[i]).clamp(-ceiling, ceiling);
                sections[i] = WideBiquad::peaking(centres[i], q, gains[i]);
            }
        }
        // The stability contract, asserted where the coefficients are made, as
        // every other filter in this engine asserts it. The peaking form cannot
        // produce a pole outside the unit circle — see [`WideBiquad::peaking`] —
        // so this is a check on the arithmetic rather than on the design, and it
        // is cheap: it runs once, at preset load.
        for (i, section) in sections.iter().enumerate() {
            assert!(
                section.pole_radius() < 1.0,
                "soundboard.radiation section {i} at {} Hz has a pole at radius {}",
                centres[i],
                section.pole_radius()
            );
        }
        sections
    }

    /// This cascade's realised magnitude at `hz`, dB — what the design closed
    /// on, and what the tests read.
    fn magnitude_db(sections: &[WideBiquad], hz: f64) -> f64 {
        sections.iter().map(|s| s.magnitude_db(hz)).sum()
    }

    /// Sample outer, section inner — so the cascade's intermediate values are
    /// never rounded back to `f32` between two sections, which is half of what
    /// the double precision was for.
    #[inline]
    fn run(&mut self, chain: usize, block: &mut [f32]) {
        let sections = &mut self.chains[chain];
        for x in block.iter_mut() {
            let mut y = f64::from(*x);
            for section in sections.iter_mut() {
                y = section.run(y);
            }
            *x = y as f32;
        }
    }

    fn clear(&mut self) {
        for chain in &mut self.chains {
            chain.iter_mut().for_each(WideBiquad::clear);
        }
    }
}

/// The magnitude, dB, the engine will **realise** for `curve` at each of `hz` —
/// the design itself, run once and read on an arbitrary grid.
///
/// Public because the fit that writes a curve has to be able to print what the
/// instrument will do with it rather than what it was asked for, and because
/// the ripple between two declared points is a property of the realisation
/// (`RADIATION_Q`) that no amount of closing the loop on band energies can see.
pub fn radiation_response_db(curve: &RadiationCurve, hz: &[f64]) -> Vec<f64> {
    let sections = Radiation::design(&curve.hz, &curve.gain_db);
    hz.iter()
        .map(|&f| Radiation::magnitude_db(&sections, f))
        .collect()
}

/// An LU factorisation with partial pivoting, for the one square system this
/// crate solves outside the modal construction.
///
/// Small and private on purpose: the system is the number of declared radiation
/// bands square (nineteen, on the fitted preset), it is solved a handful of
/// times at preset load, and nothing about it is on the audio path.
struct LuFactors {
    lu: Vec<f64>,
    pivot: Vec<usize>,
    n: usize,
}

impl LuFactors {
    /// `None` when the matrix is singular, which is the design falling back to
    /// the declared gains as written — a curve that is wrong by the overlap
    /// rather than one that is a division by zero.
    fn of(mut a: Vec<f64>, n: usize) -> Option<Self> {
        let mut pivot: Vec<usize> = (0..n).collect();
        for k in 0..n {
            let (mut best, mut best_row) = (0.0, k);
            for (r, &p) in pivot.iter().enumerate().skip(k) {
                let v = a[p * n + k].abs();
                if v > best {
                    best = v;
                    best_row = r;
                }
            }
            if best <= 1e-12 {
                return None;
            }
            pivot.swap(k, best_row);
            let pk = pivot[k];
            for r in k + 1..n {
                let pr = pivot[r];
                let factor = a[pr * n + k] / a[pk * n + k];
                a[pr * n + k] = factor;
                for c in k + 1..n {
                    a[pr * n + c] -= factor * a[pk * n + c];
                }
            }
        }
        Some(LuFactors { lu: a, pivot, n })
    }

    fn solve(&self, b: &[f64]) -> Vec<f64> {
        let n = self.n;
        let mut y = vec![0.0f64; n];
        for k in 0..n {
            let pk = self.pivot[k];
            let mut acc = b[pk];
            for c in 0..k {
                acc -= self.lu[pk * n + c] * y[c];
            }
            y[k] = acc;
        }
        let mut x = vec![0.0f64; n];
        for k in (0..n).rev() {
            let pk = self.pivot[k];
            let mut acc = y[k];
            for c in k + 1..n {
                acc -= self.lu[pk * n + c] * x[c];
            }
            x[k] = acc / self.lu[pk * n + k];
        }
        x
    }
}

/// Accumulates `gain * mono` into `side`, delayed by `delay` samples, with
/// linear interpolation between the two neighbouring taps.
///
/// `side` is `BLOCK + MIC_TAIL` long and the block's own output is its first
/// `BLOCK` samples; whatever lands past them is carried into the next block by
/// [`Soundboard::begin_block`]. That is what makes the delay free of per-voice
/// state: nothing here remembers which voice it came from.
fn scatter_add(side: &mut [f32], mono: &[f32], gain: f32, delay: f32) {
    let whole = delay.floor();
    let frac = delay - whole;
    let d = whole as usize;
    let (near, far) = (gain * (1.0 - frac), gain * frac);
    for (i, &x) in mono.iter().enumerate() {
        side[i + d] += near * x;
        side[i + d + 1] += far * x;
    }
}

pub struct Soundboard {
    direct_l: [f32; BLOCK],
    direct_r: [f32; BLOCK],
    mono: [f32; BLOCK],
    /// The mic pair's geometry, or `None` for the pan-pot path.
    mics: Option<Mics>,
    /// Direct-path sum and difference, used only when `mics` is set. `side`
    /// carries `MIC_TAIL` samples past the block so a delayed contribution can
    /// land in the next one.
    mid: [f32; BLOCK],
    side: [f32; BLOCK + MIC_TAIL],
    /// The mode-controlled band on the *direct* path's own sum.
    direct_lobe: Option<ModalLobe>,
    /// The strings' radiated response between their partials, or `None` for the
    /// uncoloured drive every preset before `DECISIONS.md` 412 has.
    radiation: Option<Radiation>,
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
    /// The board with no microphone pair: the pan-pot and the orthogonal board
    /// taps, bit for bit as they have always been.
    pub fn new(voicing: &SoundboardVoicing) -> Self {
        Self::with_mics(voicing, None)
    }

    /// The board presented through a virtual microphone pair, or through the
    /// pan-pot when `mics` is `None`.
    pub fn with_mics(voicing: &SoundboardVoicing, mics: Option<&MicVoicing>) -> Self {
        let mics = mics.map(Mics::new);
        let mut body = ModalBank::with_capacity(voicing.body_modes.len());
        for mode in &voicing.body_modes {
            // Q = f / bandwidth and this resonator's -3 dB bandwidth is sigma/pi.
            let sigma = std::f32::consts::PI * mode.hz / mode.q;
            // A complex one-pole driven at its own frequency settles at
            // |s| = g / (2 (1 - r)), so this normalises the mode's peak to its
            // tabulated gain.
            let r = (-sigma / SAMPLE_RATE).exp();
            body.push_mode(
                mode.hz,
                sigma,
                2.0 * (1.0 - r) * mode.gain * voicing.body_mix,
            );
        }
        Soundboard {
            direct_l: [0.0; BLOCK],
            direct_r: [0.0; BLOCK],
            mono: [0.0; BLOCK],
            mics,
            mid: [0.0; BLOCK],
            side: [0.0; BLOCK + MIC_TAIL],
            direct_lobe: mics.and_then(|m| m.lobe),
            radiation: voicing.radiation.as_ref().map(Radiation::new),
            board_l: [0.0; BLOCK],
            board_r: [0.0; BLOCK],
            drive: [0.0; BLOCK],
            body,
            fdn: Fdn::new(voicing, mics.as_ref()),
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
        if self.mics.is_some() {
            self.mid.fill(0.0);
            // The difference signal's tail — everything a delayed capsule put
            // past the end of the last block — becomes the head of this one.
            self.side.copy_within(BLOCK.., 0);
            self.side[MIC_TAIL..].fill(0.0);
        }
    }

    /// Accumulates one voice's mono output at pan position `pan` (-1..1).
    ///
    /// With a microphone pair, `pan` is read as *where the source is* along the
    /// bass-treble axis rather than as a mix position, and the pair's geometry
    /// turns it into a delay and a gain per capsule. Every source keeps the
    /// position it always had — strings, duplex segments, the key's own
    /// mechanism noise, the pedal at dead centre — so nothing about which sound
    /// comes from where has changed, only what a pair of capsules makes of it.
    pub fn add_voice(&mut self, mono: &[f32], pan: f32) {
        debug_assert_eq!(mono.len(), BLOCK);
        // Equal-power pan keeps the summed level constant across the compass.
        let angle = (pan.clamp(-1.0, 1.0) + 1.0) * std::f32::consts::FRAC_PI_4;
        let (gl, gr) = (angle.cos(), angle.sin());
        let Some(mics) = self.mics else {
            for (i, &x) in mono.iter().enumerate() {
                self.direct_l[i] += gl * x;
                self.direct_r[i] += gr * x;
                self.mono[i] += x;
            }
            return;
        };
        // The sum stays exactly where the pan-pot put it and the geometry goes
        // entirely into the difference; see [`Mics`].
        let centre = 0.5 * (gl + gr);
        for (i, &x) in mono.iter().enumerate() {
            self.mid[i] += centre * x;
            self.mono[i] += x;
        }
        let (ul, ur, delay_l, delay_r) = mics.taps(pan);
        let half = 0.5 * mics.width;
        scatter_add(&mut self.side, mono, half * ul, delay_l);
        scatter_add(&mut self.side, mono, -half * ur, delay_r);
    }

    /// Mixes, applies the master chain, and writes the finished block.
    pub fn process(&mut self, out_l: &mut [f32], out_r: &mut [f32]) {
        debug_assert_eq!(out_l.len(), BLOCK);
        debug_assert_eq!(out_r.len(), BLOCK);
        let direct = 1.0 - self.board_mix;
        // The strings' radiated response, before `board_mix` splits the drive
        // into the direct sound and the board's field, so both inherit it. The
        // difference signal is filtered over this block's own samples only:
        // whatever a delayed capsule put past the end is still raw, and
        // [`Self::begin_block`] carries it into the next block, where it is
        // filtered exactly once and in order.
        if let Some(radiation) = &mut self.radiation {
            if self.mics.is_some() {
                radiation.run(0, &mut self.mid);
                radiation.run(1, &mut self.side[..BLOCK]);
            } else {
                radiation.run(0, &mut self.direct_l);
                radiation.run(1, &mut self.direct_r);
            }
            radiation.run(2, &mut self.mono);
        }
        if self.mics.is_some() {
            for i in 0..BLOCK {
                let (m, s) = (self.mid[i], self.side[i]);
                let s = match &mut self.direct_lobe {
                    None => s,
                    Some(lobe) => s + lobe.run(m),
                };
                self.direct_l[i] = m + s;
                self.direct_r[i] = m - s;
            }
        }
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
        self.mid.fill(0.0);
        self.side.fill(0.0);
        if let Some(lobe) = &mut self.direct_lobe {
            lobe.clear();
        }
        if let Some(radiation) = &mut self.radiation {
            radiation.clear();
        }
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
    /// One-pole coefficient of the side highpass that gives the field its
    /// frequency-dependent interchannel coherence, or `None` for the orthogonal
    /// tap pair, which is that curve pinned at zero. See [`Mics`].
    side_b: Option<f32>,
    /// The lowpass half of that pair; the side takes what is left.
    side_lp: f32,
    /// The board's mode-controlled band, or `None` when the preset declares
    /// none — in which case not one multiply of it runs. See [`ModalLobe`].
    lobe: Option<ModalLobe>,
}

impl Fdn {
    fn new(voicing: &SoundboardVoicing, mics: Option<&Mics>) -> Self {
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
            side_b: mics.map(|m| m.diffuse_b),
            side_lp: 0.0,
            lobe: mics.and_then(|m| m.lobe),
        }
    }

    fn clear(&mut self) {
        self.delay.iter_mut().for_each(|v| *v = 0.0);
        self.pos = [0; FDN_LINES];
        self.loss_state = [0.0; FDN_LINES];
        self.peak = 0.0;
        self.side_lp = 0.0;
        if let Some(lobe) = &mut self.lobe {
            lobe.clear();
        }
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
            // Re-read the two orthogonal taps as their own sum and difference.
            // The sum is exactly what the pair folded down to; the difference
            // is what carries the field's stereo, and it is the difference that
            // the coherence filter shapes. `None` leaves both alone, which is
            // the orthogonal pair bit for bit.
            let (ol, or) = match self.side_b {
                None => (l, r),
                Some(b) => {
                    let (m, s) = (0.5 * (l + r), 0.5 * (l - r));
                    self.side_lp += b * (s - self.side_lp);
                    let mut hp = s - self.side_lp;
                    // The mode-controlled band, added to the diffuse
                    // coherence rather than replacing it: the two are
                    // different regimes of the same board and the measured
                    // profile shows both. It runs on the **sum** — a nodal
                    // line puts the same field into the two capsules with
                    // opposite signs, which is an anti-phase copy and not a
                    // louder disagreement. See [`ModalLobe`].
                    if let Some(lobe) = &mut self.lobe {
                        hp += lobe.run(m);
                    }
                    (m + hp, m - hp)
                }
            };
            out_l[n] = self.level * FDN_TAP_SCALE * ol;
            out_r[n] = self.level * FDN_TAP_SCALE * or;
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
        let mut fdn = Fdn::new(&voicing(), None);
        let (mut l, mut r) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
        let mut tail = Vec::with_capacity(tail_len);
        let mut n = 0usize;
        while n < burst + tail_len {
            let mut input = [0.0f32; BLOCK];
            for (i, x) in input.iter_mut().enumerate() {
                let t = n + i;
                if t < burst {
                    let env = 0.5 - 0.5 * (std::f32::consts::TAU * t as f32 / burst as f32).cos();
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
            assert!(
                (lf_db + 60.0).abs() < 0.5,
                "line {m}: {lf_db} dB over T60_LF"
            );
            let w = std::f32::consts::TAU * voicing.fdn_hf_hz / SAMPLE_RATE;
            let mag = g * (1.0 - a) / (1.0 - 2.0 * a * w.cos() + a * a).sqrt();
            let hf_db = 20.0 * mag.log10() * round_trips(voicing.fdn_t60_hf);
            assert!(
                (hf_db + 60.0).abs() < 0.5,
                "line {m}: {hf_db} dB over T60_HF"
            );
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
        assert!(
            ratio_db.abs() < 1.0,
            "board path is {ratio_db} dB off unity"
        );
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
        assert!(
            body_response(1_000.0) < 0.03,
            "body bank rings above its range"
        );
    }

    // -----------------------------------------------------------------------
    // The virtual microphone pair
    // -----------------------------------------------------------------------

    /// The geometry the shipped preset is fitted to; the unit tests want a
    /// pair, not a particular one.
    fn mic_voicing() -> MicVoicing {
        MicVoicing {
            spacing_m: 0.12,
            height_m: 0.30,
            span_m: 0.70,
            width: 1.0,
            diffuse_coherence: 1.0,
            modal: None,
        }
    }

    /// The same pair with the board's mode-controlled band declared, at the
    /// shipped edges. The unit tests want a lobe, not a particular one.
    fn mic_voicing_with_lobe() -> MicVoicing {
        MicVoicing {
            modal: Some(ModalBand {
                lo_hz: 190.0,
                hi_hz: 330.0,
                lift: 2.4,
            }),
            ..mic_voicing()
        }
    }

    /// Renders `blocks` blocks of one source at `pan`, returning both channels.
    ///
    /// `excite` produces the source's sample at absolute time `t`; the source
    /// is fed for the first half of the run and silence for the rest, so a
    /// decaying board field is included in what comes back.
    fn render_pair(
        sb: &mut Soundboard,
        pan: f32,
        blocks: usize,
        excite: impl Fn(usize) -> f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let (mut l, mut r) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
        let (mut out_l, mut out_r) = (Vec::new(), Vec::new());
        for b in 0..blocks {
            let mut x = [0.0f32; BLOCK];
            if b < blocks / 2 {
                for (i, v) in x.iter_mut().enumerate() {
                    *v = excite(b * BLOCK + i);
                }
            }
            sb.begin_block();
            sb.add_voice(&x, pan);
            sb.process(&mut l, &mut r);
            out_l.extend_from_slice(&l);
            out_r.extend_from_slice(&r);
        }
        (out_l, out_r)
    }

    /// Normalised interchannel correlation at lag zero.
    fn correlation(l: &[f32], r: &[f32]) -> f32 {
        let (mut ll, mut rr, mut lr) = (0.0f64, 0.0f64, 0.0f64);
        for (&a, &b) in l.iter().zip(r) {
            ll += f64::from(a) * f64::from(a);
            rr += f64::from(b) * f64::from(b);
            lr += f64::from(a) * f64::from(b);
        }
        (lr / (ll * rr).sqrt()) as f32
    }

    fn sine(hz: f32) -> impl Fn(usize) -> f32 {
        move |t| 0.02 * (std::f32::consts::TAU * hz * t as f32 / SAMPLE_RATE).sin()
    }

    /// **The mono proof.** The pair replaces the difference signal and nothing
    /// else, so `(L + R) / 2` is what the pan-pot's own fold-down was — for
    /// every source, every pan and every geometry.
    ///
    /// It is asserted sample by sample rather than band by band because the
    /// claim is structural: `L = mid + side`, `R = mid − side`, and the board's
    /// two orthogonal taps re-read as their own sum and difference. Nothing but
    /// `f32` rounding can get between the two sums, and what is left is 140 dB
    /// under the signal.
    #[test]
    fn the_microphone_pair_leaves_the_mono_sum_exactly_where_the_pan_pot_put_it() {
        let lobe = Some(ModalBand {
            lo_hz: 190.0,
            hi_hz: 330.0,
            lift: 2.4,
        });
        // The last row is the mode-controlled band at the top of its own
        // range: a lift of six on the board's difference is the largest thing
        // this stage can put into the side, and the sum still may not move.
        for (spacing, height, span, width, coherence, modal) in [
            (0.12f32, 0.30f32, 0.70f32, 1.0f32, 1.0f32, None),
            (0.60, 0.05, 1.50, 2.0, 4.0, lobe),
            (0.01, 2.00, 0.10, 0.0, 0.25, lobe),
            (
                0.12,
                0.12,
                1.50,
                1.7,
                7.86,
                Some(ModalBand {
                    lo_hz: MIC_MODAL_HZ.0,
                    hi_hz: MIC_MODAL_HZ.1,
                    lift: MIC_MODAL_LIFT.1,
                }),
            ),
        ] {
            let mv = MicVoicing {
                spacing_m: spacing,
                height_m: height,
                span_m: span,
                width,
                diffuse_coherence: coherence,
                modal,
            };
            for pan in [-1.0f32, -0.6, -0.13, 0.0, 0.37, 0.6, 1.0] {
                let mut bare = Soundboard::new(&voicing());
                let mut mics = Soundboard::with_mics(&voicing(), Some(&mv));
                // Broadband, so every band of the fold-down is exercised at
                // once, and quiet enough that the safety limiter stays out of
                // it: above the threshold the master chain is not linear and
                // the sum of two channels is not the channel of two sums.
                let mut state = 0x1234_5678u32;
                let noise: Vec<f32> = (0..200 * BLOCK)
                    .map(|_| {
                        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        0.02 * ((state >> 8) as f32 / (1 << 23) as f32 - 1.0)
                    })
                    .collect();
                let (bl, br) = render_pair(&mut bare, pan, 200, |t| noise[t]);
                let (ml, mr) = render_pair(&mut mics, pan, 200, |t| noise[t]);
                let (mut worst, mut level) = (0.0f32, 0.0f32);
                let (mut error_energy, mut energy) = (0.0f64, 0.0f64);
                for i in 0..bl.len() {
                    let (a, b) = (0.5 * (bl[i] + br[i]), 0.5 * (ml[i] + mr[i]));
                    worst = worst.max((a - b).abs());
                    level = level.max(a.abs());
                    error_energy += f64::from(a - b) * f64::from(a - b);
                    energy += f64::from(a) * f64::from(a);
                }
                assert!(level > 0.01, "the probe is too quiet to prove anything");
                let peak_db = 20.0 * (worst / level).log10();
                let rms_db = 10.0 * (error_energy / energy).log10();
                // What is left is `f32` rounding inside the master chain's own
                // recursions, which reach the same sum along two different
                // sequences of samples: it measures -113 to -118 dB of RMS
                // across these settings, and exactly **zero** in the one case
                // where the two arithmetics coincide (a centred source with the
                // board field off). A real change to the fold-down would be
                // tens of dB, not a hundred and thirteen.
                assert!(
                    peak_db < -100.0 && rms_db < -110.0,
                    "spacing {spacing}, pan {pan}: the mono sum moved by {peak_db:.1} dB peak, \
                     {rms_db:.1} dB RMS"
                );
            }
        }
    }

    /// The geometry, as arithmetic: equal power, a delay bounded by the
    /// spacing, the nearer capsule at time zero, and dead centre equidistant.
    #[test]
    fn the_capsule_taps_are_equal_power_and_bounded_by_the_spacing() {
        let mv = mic_voicing();
        let mics = Mics::new(&mv);
        let bound = mv.spacing_m * SAMPLE_RATE / SPEED_OF_SOUND;
        let mut previous = f32::NEG_INFINITY;
        for i in 0..=40 {
            let pan = -1.0 + 2.0 * i as f32 / 40.0;
            let (ul, ur, dl, dr) = mics.taps(pan);
            assert!(
                (ul * ul + ur * ur - 1.0).abs() < 1e-6,
                "pan {pan}: {ul}^2 + {ur}^2 is not one"
            );
            assert!(dl >= 0.0 && dr >= 0.0, "pan {pan}: negative delay");
            assert!(dl == 0.0 || dr == 0.0, "pan {pan}: both capsules delayed");
            assert!(
                dl <= bound && dr <= bound,
                "pan {pan}: {dl}/{dr} over {bound}"
            );
            // The interchannel delay grows monotonically from bass to treble,
            // which is what makes the image a map of the keyboard.
            let delta = dl - dr;
            assert!(delta > previous, "pan {pan}: delay went backwards");
            previous = delta;
            // Treble keys are nearer the right capsule and louder in it.
            if pan > 0.05 {
                assert!(
                    ur > ul && dl > 0.0,
                    "pan {pan}: the treble is not to the right"
                );
            }
        }
        let (ul, ur, dl, dr) = mics.taps(0.0);
        assert!(
            (ul - ur).abs() < 1e-7,
            "dead centre is not equidistant in level"
        );
        assert_eq!(
            (dl, dr),
            (0.0, 0.0),
            "dead centre is not equidistant in time"
        );
    }

    /// The interchannel delay a source produces is the delay its *position*
    /// implies, read back off the rendered pair by cross-correlation.
    #[test]
    fn the_rendered_delay_is_the_one_the_geometry_asks_for() {
        let mv = mic_voicing();
        let mics = Mics::new(&mv);
        for pan in [-0.6f32, 0.6] {
            let (_, _, dl, dr) = mics.taps(pan);
            let wanted = dl - dr;
            let mut sb = Soundboard::with_mics(&voicing(), Some(&mv));
            // Direct path alone: the board's diffuse field has no ITD of its
            // own and would only add a floor to the correlation.
            sb.set_board_mix(0.0);
            // White noise, whose autocorrelation is a delta, so the peak of the
            // cross-correlation is the delay and not a shape the probe brought
            // with it.
            let mut state = 0x9e37_79b9u32;
            let noise: Vec<f32> = (0..64 * BLOCK)
                .map(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    0.05 * ((state >> 8) as f32 / (1 << 23) as f32 - 1.0)
                })
                .collect();
            let (l, r) = render_pair(&mut sb, pan, 64, |t| noise[t]);
            // The shared mid puts a term at lag zero and the geometry puts a
            // larger one — 0.32 of the signal's energy against 0.15 at this
            // geometry — at the delay itself, so the argmax of
            // `sum L[t + lag] R[t]` is the delay.
            let limit = 40i32;
            let mut best = (0i32, f32::NEG_INFINITY);
            for lag in -limit..=limit {
                let mut c = 0.0f64;
                for t in limit as usize..l.len() - limit as usize {
                    c += f64::from(l[(t as i32 + lag) as usize]) * f64::from(r[t]);
                }
                if c as f32 > best.1 {
                    best = (lag, c as f32);
                }
            }
            // `c[τ] = Σ L[t+τ] R[t]` peaks at `Δ = (d_L − d_R)/c`, which is
            // `realism::StereoBand`'s sign convention read the other way round:
            // a treble key is nearer the right capsule, so the right channel
            // leads and the lag is positive.
            let read = best.0 as f32;
            assert!(
                (read - wanted).abs() <= 1.5,
                "pan {pan}: read {read} samples of delay, geometry says {wanted}"
            );
        }
    }

    /// **The finding, inverted.** A source off to one side is coherent in the
    /// bass, where two capsules a hand's breadth apart are well inside a
    /// wavelength of each other, and decorrelated in the treble, where they are
    /// not. The pan-pot it replaces is +1 at both.
    #[test]
    fn a_source_off_axis_is_coherent_in_the_bass_and_not_in_the_treble() {
        let mv = mic_voicing();
        for pan in [-0.6f32, 0.6] {
            let mut potted = Soundboard::new(&voicing());
            potted.set_board_mix(0.0);
            let (pl, pr) = render_pair(&mut potted, pan, 64, sine(6_000.0));
            assert!(
                correlation(&pl, &pr) > 0.999,
                "the pan-pot is supposed to be the thing that is always +1"
            );

            let mut sb = Soundboard::with_mics(&voicing(), Some(&mv));
            sb.set_board_mix(0.0);
            let (bl, br) = render_pair(&mut sb, pan, 64, sine(80.0));
            let bass = correlation(&bl, &br);
            let mut sb = Soundboard::with_mics(&voicing(), Some(&mv));
            sb.set_board_mix(0.0);
            let (tl, tr) = render_pair(&mut sb, pan, 64, sine(6_000.0));
            let treble = correlation(&tl, &tr);
            assert!(bass > 0.98, "pan {pan}: 80 Hz reads {bass}");
            assert!(treble < 0.6, "pan {pan}: 6 kHz reads {treble}");
        }
    }

    /// The board's diffuse field, which has no direction of its own, is shared
    /// by both capsules at low frequency and orthogonal at high — the classical
    /// `sin(kd)/kd` of a diffuse field between two points, to first order. The
    /// tap pair it replaces is decorrelated at *every* frequency, which is what
    /// pulled the engine's bass to −0.58.
    #[test]
    fn the_diffuse_field_is_shared_low_and_orthogonal_high() {
        let mv = mic_voicing();
        // Noise, not a tone: two sinusoids of one frequency correlate at
        // ±|cos φ| whatever produced them, so a single tone cannot measure a
        // coherence at all. Each channel is split into a low and a high band
        // and the two are correlated separately.
        let mut state = 0x5bf0_3635u32;
        let noise: Vec<f32> = (0..400 * BLOCK)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                0.05 * ((state >> 8) as f32 / (1 << 23) as f32 - 1.0)
            })
            .collect();
        let split = |signal: &[f32], hz: f32| -> (Vec<f32>, Vec<f32>) {
            let b = 1.0 - (-std::f32::consts::TAU * hz / SAMPLE_RATE).exp();
            let mut lp = 0.0f32;
            let (mut low, mut high) = (Vec::new(), Vec::new());
            for &x in signal {
                lp += b * (x - lp);
                low.push(lp);
                high.push(x - lp);
            }
            (low, high)
        };
        for (label, pair, low_range, high_range) in [
            (
                "the microphone pair",
                Some(&mv),
                0.85f32..=1.0f32,
                -0.3f32..=0.3f32,
            ),
            ("the orthogonal taps", None, -0.3..=0.3, -0.3..=0.3),
        ] {
            let mut sb = Soundboard::with_mics(&voicing(), pair);
            sb.set_board_mix(1.0);
            let (l, r) = render_pair(&mut sb, 0.0, 400, |t| noise[t]);
            let (ll, lh) = split(&l, 200.0);
            let (rl, rh) = split(&r, 200.0);
            let (low, high) = (correlation(&ll, &rl), correlation(&lh, &rh));
            assert!(
                low_range.contains(&low) && high_range.contains(&high),
                "{label}: the board field reads {low:+.3} under 200 Hz and {high:+.3} over it"
            );
        }
    }

    /// `width` scales the difference and cannot touch the sum: at zero the pair
    /// is one capsule twice, at two it is the geometry doubled, and the mono
    /// fold-down is the same signal at every setting. (The mono half of this is
    /// proved above; here it is the *image* that has to move.)
    #[test]
    fn width_moves_the_image_and_nothing_else() {
        let mut previous = 2.0f32;
        for width in [0.0f32, 0.5, 1.0, 2.0] {
            let mv = MicVoicing {
                width,
                ..mic_voicing()
            };
            let mut sb = Soundboard::with_mics(&voicing(), Some(&mv));
            sb.set_board_mix(0.0);
            let (l, r) = render_pair(&mut sb, 0.6, 64, sine(4_000.0));
            let c = correlation(&l, &r);
            if width == 0.0 {
                assert!(c > 0.999, "width 0 must be one capsule twice, read {c}");
            }
            assert!(c < previous, "width {width} did not narrow the image: {c}");
            previous = c;
        }
    }

    /// The polarization spread survives as a component of the image rather than
    /// only as a level trim: the two planes of one key sit at two positions, so
    /// they reach the capsules with two different delays as well as two
    /// different gains.
    #[test]
    fn the_two_polarizations_reach_the_capsules_from_two_places() {
        let mics = Mics::new(&mic_voicing());
        let pan = pan_for_key(72);
        let spread = 0.25;
        let (ul, _, dl, _) = mics.taps(pan - spread);
        let (ur, _, dr, _) = mics.taps(pan + spread);
        assert!(
            (dl - dr).abs() > 0.5,
            "the two planes share a delay: {dl} and {dr}"
        );
        assert!((ul - ur).abs() > 1e-3, "the two planes share a gain");
    }

    /// **The finding, as an assertion.** The board's mode-controlled band puts
    /// the two capsules in *opposition* where the modes are, and nowhere else.
    ///
    /// Three bands, one signal: noise through the board field alone, split into
    /// a decade below the lobe, the lobe itself, and a decade above it. Below
    /// and above, the pair reads what it read before the section existed;
    /// inside, it reads **negative**, which no spacing, no delay and no
    /// `sin(kd)/kd` can produce (`DECISIONS.md` 357).
    #[test]
    fn the_mode_controlled_band_is_anti_phase_and_only_there() {
        let mv = mic_voicing_with_lobe();
        let band = mv.modal.expect("a lobe");
        let mut state = 0x1f3a_77c1u32;
        let noise: Vec<f32> = (0..600 * BLOCK)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                0.05 * ((state >> 8) as f32 / (1 << 23) as f32 - 1.0)
            })
            .collect();
        // Four one-poles on each edge — 24 dB/octave — because the three
        // probes are the three *regimes* and a single pole is far too leaky to
        // separate them: a first version of this test read the 40-114 Hz probe
        // at +0.06 with the lobe on, which was the lobe's own 226 Hz coming
        // through the probe's skirt rather than anything the bass did.
        const PROBE_POLES: usize = 4;
        let band_pass = |signal: &[f32], lo: f32, hi: f32| -> Vec<f32> {
            let coeff = |hz: f32| 1.0 - (-std::f32::consts::TAU * hz / SAMPLE_RATE).exp();
            let (a, b) = (coeff(lo), coeff(hi));
            let mut low = [0.0f32; PROBE_POLES];
            let mut high = [0.0f32; PROBE_POLES];
            signal
                .iter()
                .map(|&x| {
                    let mut y = x;
                    // Four first-order highpasses at `lo`, each the signal
                    // less its own lowpass, then four lowpasses at `hi`.
                    for state in &mut low {
                        *state += a * (y - *state);
                        y -= *state;
                    }
                    for state in &mut high {
                        *state += b * (y - *state);
                        y = *state;
                    }
                    y
                })
                .collect()
        };
        let bare = mic_voicing();
        let mut inside_both = Vec::new();
        for (label, has_lobe, pair) in [
            ("with the lobe", true, &mv),
            ("without it", false, &bare),
        ] {
            let mut sb = Soundboard::with_mics(&voicing(), Some(pair));
            sb.set_board_mix(1.0);
            let (l, r) = render_pair(&mut sb, 0.0, 600, |t| noise[t]);
            let read = |lo: f32, hi: f32| {
                correlation(&band_pass(&l, lo, hi), &band_pass(&r, lo, hi))
            };
            let below = read(40.0, 0.5 * band.lo_hz);
            let inside = read(0.9 * band.lo_hz, 1.1 * band.hi_hz);
            let above = read(3.0 * band.hi_hz, 8_000.0);
            inside_both.push(inside);
            if has_lobe {
                // A one-pole pair is a leaky probe — the band is 190-330 Hz and
                // its skirts reach well past both edges — so the number here is
                // milder than the sixth-octave profile's own -0.79 at 226 Hz.
                // What has to be true is the *sign*, which nothing outside this
                // section can produce.
                assert!(
                    inside < -0.1,
                    "{label}: the mode-controlled band reads {inside:+.3}, not anti-phase"
                );
                assert!(
                    below > 0.8,
                    "{label}: {below:+.3} under the band — the lobe leaked into the bass"
                );
                assert!(
                    above.abs() < 0.4,
                    "{label}: {above:+.3} over the band — the lobe leaked into the treble"
                );
            } else {
                assert!(
                    inside > 0.5,
                    "{label}: {inside:+.3} — without a lobe the band must stay coherent, \
                     or the test above proves nothing"
                );
            }
        }
        let swing = inside_both[1] - inside_both[0];
        assert!(
            swing > 0.7,
            "the section moved the band by {swing:.3}, from {:+.3} to {:+.3}",
            inside_both[1],
            inside_both[0]
        );
    }

    /// **The mode-controlled band is there before the board field is.**
    ///
    /// A nodal line is a property of the plate the string is mounted on, so it
    /// applies to a note's *first* milliseconds — and those arrive down the
    /// direct path, because the FDN's shortest line is 149 samples and its
    /// output is exactly zero for 3.1 ms. The first version of this stage put
    /// the lobe on the FDN's difference alone, which meant a band-limited copy
    /// of nothing through the whole strike: measured on the shipped preset,
    /// C5's first 10 ms read `+9.9 dB` mid over side in 125-250 Hz, where the
    /// recording's own first 10 ms read `−1.6 dB` (`DECISIONS.md` 379).
    ///
    /// Two halves, and the first is the one the old form could not pass at all:
    /// with the **board muted entirely** the pair must still oppose inside the
    /// band, and with no lobe the same render is one signal twice.
    #[test]
    fn the_mode_controlled_band_reaches_the_direct_path() {
        let with = mic_voicing_with_lobe();
        let band = with.modal.expect("a lobe");
        let centre = (f64::from(band.lo_hz) * f64::from(band.hi_hz)).sqrt() as f32;

        // (a) The direct path alone — the board contributes nothing at all.
        let mut lobed = Soundboard::with_mics(&voicing(), Some(&with));
        lobed.set_board_mix(0.0);
        let (l, r) = render_pair(&mut lobed, 0.0, 64, sine(centre));
        let opposed = correlation(&l, &r);
        let mut bare = Soundboard::with_mics(&voicing(), Some(&mic_voicing()));
        bare.set_board_mix(0.0);
        let (bl, br) = render_pair(&mut bare, 0.0, 64, sine(centre));
        let coherent = correlation(&bl, &br);
        assert!(
            coherent > 0.999,
            "with no lobe and no board this is one signal twice, and it reads {coherent:+.3}"
        );
        // Not −1: the run includes the decay after the source stops, where
        // what is left is the cascade's own settling rather than the tone.
        assert!(
            opposed < -0.5,
            "the direct path does not carry the nodal line: {opposed:+.3} at {centre:.0} Hz"
        );

        // (b) And it is opposed *from the strike*, not once the field has
        //     built: the first 10 ms of a burst, board and all.
        let mut struck = Soundboard::with_mics(&voicing(), Some(&with));
        let attack = (0.010 * SAMPLE_RATE) as usize;
        let (sl, sr) = render_pair(&mut struck, 0.0, 8, sine(centre));
        let first = correlation(&sl[..attack], &sr[..attack]);
        assert!(
            first < 0.0,
            "the band is coherent through the strike: {first:+.3} over the first 10 ms"
        );
    }

    /// The lower edge is **eighth-order**, and the order is the measurement
    /// (`MIC_MODAL_HIGH_Q`): the recording's own profile falls at 40 dB per
    /// octave and nothing shallower reaches it.
    ///
    /// Measured on the cascade itself rather than through the instrument, at
    /// two frequencies an octave apart under the corner, and checked against
    /// the maximally-flat passband as well — a cascade of *identical* sections
    /// has the same asymptotic slope and fails the second half, which is why
    /// it is not what is built (and the first version of this stage was).
    #[test]
    fn the_lobes_lower_edge_is_as_steep_as_the_recordings_own() {
        let band = ModalBand {
            lo_hz: 200.0,
            hi_hz: 2_000.0,
            lift: 1.0,
        };
        let gain = |hz: f32| -> f32 {
            let mut lobe = ModalLobe::new(&band);
            let (mut peak, n) = (0.0f32, 40_000usize);
            for t in 0..n {
                let x = (std::f32::consts::TAU * hz * t as f32 / SAMPLE_RATE).sin();
                let y = lobe.run(x);
                // Read the last tenth, once the cascade has settled.
                if t > n - n / 10 {
                    peak = peak.max(y.abs());
                }
            }
            peak
        };
        let (quarter, half) = (gain(50.0), gain(100.0));
        let slope_db_per_octave = 20.0 * (half / quarter).log10();
        assert!(
            slope_db_per_octave > 40.0,
            "the lower edge is {slope_db_per_octave:.1} dB/octave, shallower than the \
             recording's own 40"
        );
        // Maximally flat above the corner: a fifth up it is already at the
        // passband, which is the half a cascade of identical sections misses.
        let passband = gain(1_000.0);
        let just_above = gain(300.0);
        assert!(
            just_above > 0.9 * passband,
            "a fifth above the corner the edge is only {:.2} of its passband",
            just_above / passband
        );
    }

    /// The lift adds an anti-phase copy of the *sum* to the side and cannot
    /// touch the sum itself, and the image moves monotonically with it: this is
    /// `width_moves_the_image_and_nothing_else` for the other half of the side
    /// signal.
    #[test]
    fn the_lift_moves_the_board_field_and_nothing_else() {
        let mut state = 0x2c9e_1105u32;
        let noise: Vec<f32> = (0..300 * BLOCK)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                0.05 * ((state >> 8) as f32 / (1 << 23) as f32 - 1.0)
            })
            .collect();
        let mut previous = 2.0f32;
        let mut sums: Vec<Vec<f32>> = Vec::new();
        for lift in [0.0f32, 1.0, 2.4, 4.0] {
            let mv = MicVoicing {
                modal: Some(ModalBand {
                    lo_hz: 190.0,
                    hi_hz: 330.0,
                    lift,
                }),
                ..mic_voicing()
            };
            let mut sb = Soundboard::with_mics(&voicing(), Some(&mv));
            sb.set_board_mix(1.0);
            let (l, r) = render_pair(&mut sb, 0.0, 300, |t| noise[t]);
            let c = correlation(&l, &r);
            assert!(
                c < previous,
                "lift {lift} did not open the image further: {c:+.3} against {previous:+.3}"
            );
            previous = c;
            sums.push(l.iter().zip(&r).map(|(&a, &b)| 0.5 * (a + b)).collect());
        }
        // Four lifts, one mono sum. Not "within a tolerance" — the mid is not
        // a function of the lift at all, and only `f32` rounding in the master
        // chain's recursions can get between two renders of it.
        let reference = &sums[0];
        for (i, sum) in sums.iter().enumerate().skip(1) {
            let (mut worst, mut level) = (0.0f32, 0.0f32);
            for (&a, &b) in reference.iter().zip(sum) {
                worst = worst.max((a - b).abs());
                level = level.max(a.abs());
            }
            let db = 20.0 * (worst / level).log10();
            println!("lift row {i}: the mono sum moved by {db:.1} dB peak");
            assert!(level > 0.001 && db < -100.0, "lift row {i} moved the sum by {db:.1} dB");
        }
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

    // ----------------------------------------------------------------------
    // The strings' radiated response (`DECISIONS.md` 412)
    // ----------------------------------------------------------------------

    /// The sixth-octave grid the fit works on, from 40 Hz, over 100-810 Hz —
    /// `realism::stereo_profile`'s own centres, which is what makes the
    /// engine's declared points and the boards' bands the same bands.
    fn radiation_grid() -> Vec<f32> {
        let ratio = 2.0f32.powf(1.0 / 6.0);
        let mut hz = 40.0f32;
        let mut out = Vec::new();
        while hz <= 810.0 {
            if hz >= 100.0 {
                out.push(hz);
            }
            hz *= ratio;
        }
        out
    }

    /// The shape item 408's table asks for, near enough: the +9 dB spike at
    /// 180 Hz, its two skirts, and a mild tilt elsewhere.
    fn radiation_curve() -> RadiationCurve {
        let hz = radiation_grid();
        let gain_db = hz
            .iter()
            .map(|&f| match f {
                f if f < 150.0 => 1.0,
                f if f < 190.0 => 8.9,
                f if f < 215.0 => 4.8,
                f if f < 240.0 => 4.3,
                f if f < 270.0 => -0.7,
                f if f < 300.0 => 1.2,
                _ => 0.5,
            })
            .collect();
        RadiationCurve { hz, gain_db }
    }

    fn voicing_with_radiation() -> SoundboardVoicing {
        SoundboardVoicing {
            radiation: Some(radiation_curve()),
            ..voicing()
        }
    }

    /// **The design's own claim.** A cascade of overlapping sections does not
    /// put its own gain at its own centre, so the design inverts the overlap;
    /// this is the assertion that it did. Without the Newton rounds — with each
    /// section simply given the decibels its point asks for — the 180 Hz point
    /// realises about **+16 dB** for a declared +8.9, which is the falsification
    /// this test exists for.
    #[test]
    fn the_realised_response_passes_through_every_declared_point() {
        let curve = radiation_curve();
        let sections = Radiation::design(&curve.hz, &curve.gain_db);
        let mut worst = 0.0f64;
        for (&hz, &want) in curve.hz.iter().zip(&curve.gain_db) {
            let got = Radiation::magnitude_db(&sections, f64::from(hz));
            worst = worst.max((got - f64::from(want)).abs());
        }
        assert!(
            worst < 0.01,
            "the realised curve misses a declared point by {worst:.4} dB"
        );

        // The overlap is real, which is what makes the inversion load-bearing:
        // sections given their points' own decibels overshoot badly.
        let naive: Vec<WideBiquad> = curve
            .hz
            .iter()
            .zip(&curve.gain_db)
            .map(|(&hz, &g)| {
                WideBiquad::peaking(f64::from(hz), f64::from(RADIATION_Q), f64::from(g))
            })
            .collect();
        let peak = curve
            .hz
            .iter()
            .zip(&curve.gain_db)
            .map(|(&hz, &want)| Radiation::magnitude_db(&naive, f64::from(hz)) - f64::from(want))
            .fold(0.0f64, f64::max);
        assert!(
            peak > 3.0,
            "the sections do not overlap enough for the inversion to matter: {peak:.2} dB"
        );
    }

    /// Every section is inside the unit circle, at the rails of what a preset
    /// may declare — the same contract every other filter in this engine is
    /// held to.
    #[test]
    fn every_radiation_section_is_stable_at_the_rails() {
        use crate::preset::{MAX_RADIATION_GAIN_DB, MIN_RADIATION_GAIN_DB};
        let hz = radiation_grid();
        for &g in &[MIN_RADIATION_GAIN_DB, 0.0, MAX_RADIATION_GAIN_DB] {
            let alternating: Vec<f32> = hz
                .iter()
                .enumerate()
                .map(|(i, _)| if i % 2 == 0 { g } else { -g })
                .collect();
            for gains in [vec![g; hz.len()], alternating] {
                let sections = Radiation::design(&hz, &gains);
                for (i, s) in sections.iter().enumerate() {
                    let r = s.pole_radius();
                    assert!(
                        r < 1.0 - 1e-6,
                        "section {i} at {} Hz has a pole at radius {r}",
                        hz[i]
                    );
                }
            }
        }
    }

    /// **Absent means old, bit for bit.** The whole neutrality contract in one
    /// assertion, on both branches of the board.
    #[test]
    fn a_preset_without_a_radiation_section_renders_the_old_board_sample_for_sample() {
        let mv = mic_voicing_with_lobe();
        for mics in [None, Some(&mv)] {
            let mut a = Soundboard::with_mics(&voicing(), mics);
            let mut b = Soundboard::with_mics(
                &SoundboardVoicing {
                    radiation: None,
                    ..voicing()
                },
                mics,
            );
            let (al, ar) = render_pair(&mut a, 0.37, 60, sine(220.0));
            let (bl, br) = render_pair(&mut b, 0.37, 60, sine(220.0));
            assert_eq!(al, bl);
            assert_eq!(ar, br);
        }
    }

    /// **What the section is for.** A declared point moves that band of the
    /// rendered output by what it declares, and a band the curve leaves alone
    /// stays where it was. Measured on the whole board — direct path and field
    /// together, `board_mix` where the preset has it — because that is the
    /// signal every board in the repository scores.
    #[test]
    fn a_declared_band_moves_the_render_by_what_it_declares() {
        let mv = mic_voicing_with_lobe();
        // Driven for the whole run and read over the last third, so what is
        // measured is the steady state rather than a decay: the colouration is
        // a response, and a response is a thing a tail does not have.
        let energy = |voicing: &SoundboardVoicing, hz: f32| {
            let mut sb = Soundboard::with_mics(voicing, Some(&mv));
            let (mut l, mut r) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
            let excite = sine(hz);
            let mut acc = 0.0f64;
            let blocks = 240;
            for b in 0..blocks {
                let mut x = [0.0f32; BLOCK];
                for (i, v) in x.iter_mut().enumerate() {
                    *v = excite(b * BLOCK + i);
                }
                sb.begin_block();
                sb.add_voice(&x, 0.2);
                sb.process(&mut l, &mut r);
                if b >= 2 * blocks / 3 {
                    acc += l
                        .iter()
                        .zip(&r)
                        .map(|(&a, &b)| f64::from(a) * f64::from(a) + f64::from(b) * f64::from(b))
                        .sum::<f64>();
                }
            }
            acc
        };
        let bare = voicing();
        let coloured = voicing_with_radiation();
        let curve = radiation_curve();
        for (&hz, &want) in curve.hz.iter().zip(&curve.gain_db) {
            if hz > 400.0 {
                continue;
            }
            let moved = 10.0 * (energy(&coloured, hz) / energy(&bare, hz)).log10();
            assert!(
                (moved - f64::from(want)).abs() < 0.5,
                "{hz:.0} Hz moved by {moved:+.2} dB where the curve declares {want:+.2}"
            );
        }
        // ... and 3 kHz, an octave and a half above the top declared point, is
        // where it was: nothing is shelved, so the compass outside the fitted
        // span keeps its own voicing.
        let untouched = 10.0 * (energy(&coloured, 3_000.0) / energy(&bare, 3_000.0)).log10();
        assert!(
            untouched.abs() < 0.05,
            "3 kHz moved by {untouched:+.3} dB and the curve stops at 806 Hz"
        );
    }

    /// **The mono discipline survives the colouration.** The pair's fold-down
    /// is still the pan-pot's own render, which is the invariant every mono
    /// board in the repository is read under — and it is the one a stage
    /// applied to three accumulators rather than to one could plausibly break.
    #[test]
    fn the_radiated_response_leaves_the_mono_sum_where_the_pan_pot_put_it() {
        let mv = mic_voicing_with_lobe();
        let voicing = voicing_with_radiation();
        for pan in [-0.6f32, 0.0, 0.37] {
            let mut bare = Soundboard::new(&voicing);
            let mut mics = Soundboard::with_mics(&voicing, Some(&mv));
            // Quieter than the uncoloured proof's probe by the largest boost
            // the curve declares, for that proof's own stated reason: above the
            // safety limiter's threshold the master chain is not linear, and
            // the sum of two channels is then not the channel of two sums.
            let mut state = 0x1234_5678u32;
            let noise: Vec<f32> = (0..200 * BLOCK)
                .map(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    0.005 * ((state >> 8) as f32 / (1 << 23) as f32 - 1.0)
                })
                .collect();
            let (bl, br) = render_pair(&mut bare, pan, 200, |t| noise[t]);
            let (ml, mr) = render_pair(&mut mics, pan, 200, |t| noise[t]);
            let (mut error_energy, mut energy) = (0.0f64, 0.0f64);
            let mut worst = 0.0f32;
            let mut level = 0.0f32;
            for i in 0..bl.len() {
                let (a, b) = (0.5 * (bl[i] + br[i]), 0.5 * (ml[i] + mr[i]));
                worst = worst.max((a - b).abs());
                level = level.max(a.abs());
                error_energy += f64::from(a - b) * f64::from(a - b);
                energy += f64::from(a) * f64::from(a);
            }
            let peak_db = 20.0 * (worst / level).log10();
            let rms_db = 10.0 * (error_energy / energy).log10();
            assert!(
                peak_db < -100.0 && rms_db < -110.0,
                "pan {pan}: the coloured fold-down moved by {peak_db:.1} dB peak, \
                 {rms_db:.1} dB RMS"
            );
        }
    }

    /// The state carried past a block boundary is filtered **once**. The side
    /// signal's tail is the one buffer in this file that outlives its own
    /// block, so a filter written over `side` in full would run over those
    /// samples twice; rendering the same source in one long run and in short
    /// blocks is the same render either way, and this is that assertion at the
    /// only place it could fail.
    #[test]
    fn the_difference_signals_carry_over_is_coloured_exactly_once() {
        let mv = mic_voicing_with_lobe();
        let voicing = voicing_with_radiation();
        // Two renders of the same source that differ only in how many blocks
        // the *source* is spread over cannot differ, because the board sees one
        // stream of blocks either way. What can differ is a filter that reads
        // `side` past `BLOCK`: it would colour a carried sample a second time,
        // and the carried samples are exactly the ones a hard-panned source
        // produces.
        let mut sb = Soundboard::with_mics(&voicing, Some(&mv));
        let (l, r) = render_pair(&mut sb, -1.0, 80, sine(180.0));
        // A second, independent board fed the same way must agree sample for
        // sample; and the difference signal must have actually been used, which
        // a hard pan and a lobe together guarantee.
        let mut again = Soundboard::with_mics(&voicing, Some(&mv));
        let (l2, r2) = render_pair(&mut again, -1.0, 80, sine(180.0));
        assert_eq!(l, l2);
        assert_eq!(r, r2);
        let side_energy: f64 = l
            .iter()
            .zip(&r)
            .map(|(&a, &b)| f64::from(0.5 * (a - b)).powi(2))
            .sum();
        let mid_energy: f64 = l
            .iter()
            .zip(&r)
            .map(|(&a, &b)| f64::from(0.5 * (a + b)).powi(2))
            .sum();
        assert!(
            side_energy > 0.01 * mid_energy,
            "the probe never exercised the difference signal"
        );
    }
}
