//! The coupled-eigenmode unison, built offline and measured against both the
//! shipped engine and the recording it is fitted to.
//!
//! ```text
//! cargo run --release -p forensics --bin eigenmode_prototype -- \
//!     [data/salamander] [presets/salamander-c5.toml] [renders/jitter]
//! ```
//!
//! # What this is
//!
//! `docs/history/FUNDAMENTALS.md` §5 derives a replacement for `engine/src/string.rs`'s
//! unison construction. The engine builds `2N` **free-running** sinusoids per
//! partial — one per unison string per polarization — at frequency offsets the
//! preset fixes once (`notes.detune_cents` as a ratio, `voicing.horizontal_offset_hz`
//! as a constant number of hertz) with decay rates written in by hand. The
//! physics (Weinreich 1977; Capleton 2004; Woodhouse 2021) says those `2N`
//! degrees of freedom are **coupled** through the bridge admittance, and that
//! the coupling coefficient is not free: it is the same number as the radiation
//! damping the preset has already fitted.
//!
//! This example builds the coupled system, solves it, renders C4/A2/C6 through
//! it, and runs `renders/jitter/JITTER.md`'s own measurement code on the result.
//! `engine/` is not touched: the eigen renderer is a second, offline modal
//! renderer, and `02_modal_shipped` is the same renderer with the *engine's*
//! construction in it so that `03_eigenmode − 02_modal_shipped` is the change
//! under test and not the change of renderer.
//!
//! # The construction, in full
//!
//! Per partial `k` of one key, with `N` strings and two polarizations:
//!
//! ```text
//!     a' = A_k a,    A_k = i Omega_k - sigma_int I - C_k
//!
//!     Omega_k = diag(omega_k * detune_j)      (polarization does not change omega)
//!     C_k     = [ c_v J_N,    0     ]         J_N = all-ones N x N
//!               [   0,    c_h J_N   ]         c_p = gamma_v (g_p + i beta_p)
//! ```
//!
//! `C_k` is block diagonal (the off-diagonal of the 2x2 bridge admittance is
//! second order — `docs/history/FUNDAMENTALS.md` §2.5, deliberately deferred), and each block
//! is `c_p` times an all-ones matrix, i.e. **rank one**. So the `2N x 2N`
//! eigenproblem is two `N x N` rank-one updates of a diagonal, and its
//! characteristic equation factorises into two scalar rational equations
//!
//! ```text
//!     1 + c_p * sum_j 1/(lambda - d_jp) = 0,     d_jp = i omega_j - sigma_int
//! ```
//!
//! whose `N` roots per block are the roots of a degree-`N` complex polynomial —
//! [`block_modes`]. `D - c J` is complex **symmetric**, so its left eigenvectors
//! are its right ones transposed (not conjugated) and the inverse eigenvector
//! matrix needed for the strike projection is one division per mode:
//! `v_jm = 1/(d_j - lambda_m)`, `n_m = sum_j v_jm^2`, `c_m = (v_m . u)/n_m`.
//! No LAPACK, no matrix inversion, and nothing in it depends on velocity.
//!
//! **Excitation.** `u_(j,v) = s_j g_k e^{-i omega_k d_j}` and
//! `u_(j,h) = eta_eps * u_(j,v)`: the existing per-string strike share, the
//! existing per-string timing skew turned into a phase rotation at the partial's
//! own frequency, and the existing per-partial `g_k` (comb, contact taper,
//! `notes.partial_gains`). Radiated gain `G_m = (w . v_m) c_m` with `w = 1`
//! inside a block.
//!
//! **Where the two shipped fields go.** Only the *product* of the hammer's
//! horizontal leak and the horizontal plane's radiation efficiency enters a
//! block-diagonal `C`, so [`HORIZONTAL_LEAK`] is that product and it is read
//! straight off `voicing.horizontal_gain_db`: at zero coupling the construction
//! then reduces to the engine's, bit for bit in structure. `horizontal_offset_hz`
//! is **not read at all** — the polarization split is now the reactive
//! anisotropy [`REACTIVE_ANISOTROPY`] and comes out proportional to omega, about
//! 0.014 Hz at C4 k=1 against the shipped 0.35 Hz. `unison_coupling` is not read
//! either; it is the coupling, and the coupling is `radiated_share * sigma_k`.
//! `unison_sigma_scale` is not read: the per-string decay split it was built for
//! is what the eigenproblem produces on its own.
//!
//! # The two normalisations, both forced
//!
//! 1. **`radiated_share` against `horizontal_decay_ratio`** — `docs/history/FUNDAMENTALS.md`
//!    §2.6 shows the shipped 0.5 and 0.172 make incompatible claims about the
//!    same quantity, and §6 lists reconciling them as the step that blocks this
//!    build. They are reconciled *here* by deriving one from the other. The
//!    slowest mode of the coupled system radiates nothing and therefore decays
//!    at `sigma_int = (1 - share) sigma_k`; the loudest decays at `sigma_k`; so
//!    the ratio of the note's aftersound decay to its prompt decay is exactly
//!    `1 - share`. The engine's fitted value for that ratio is
//!    `horizontal_decay_ratio`. Hence [`radiated_share`] `= 1 - 0.172 = 0.828`,
//!    which is also the side of the contradiction Woodhouse (2021) is on: body
//!    coupling exceeds air damping by ~20 dB across the midrange.
//!
//! 2. **The T60 anchors** — `notes.sigma0`/`sigma1` are fitted to *recorded*
//!    whole-note decays, so the composite of the `2N` modes must reach −60 dB at
//!    `6.91/sigma_k` or the whole compass retunes. The engine does this with one
//!    global closed form (`Voicing::vertical_decay_factor`, valid only for its
//!    own two-exponential construction); [`decay_scale`] does the same thing for
//!    an arbitrary mode set, per partial, by solving for the one factor on
//!    `sigma_int` and `gamma_v` that puts the composite's −60 dB crossing on the
//!    anchor. It is the same discipline, and it is what keeps `mu = pi Df/gamma`
//!    at its physical value instead of at 3x it.
//!
//! # What is measured
//!
//! `JITTER.md`'s statistics, on the identical code path, over 0.3–3.0 s of the
//! mono sum: the RMS instantaneous-frequency deviation inside 0.1–20 Hz
//! (`cents`), where in the note that deviation sits (`wRMS/raw` — near 1 means
//! it rides the loud part of the partial, a small fraction means it is a spike
//! at the null of a beat), the beat depth of the log envelope, and the
//! line-versus-continuum flatness of both. Added here, because the aftersound
//! must survive the change and because the honest negative needs it:
//!
//! * **the double decay** — a straight-line fit of the partial's log envelope
//!   over [`PROMPT_LO_S`]–[`PROMPT_HI_S`] and again over [`TAIL_LO_S`]–[`TAIL_HI_S`],
//!   reported as two slopes in dB/s and as the level the tail extrapolates back
//!   to at the strike, which is the aftersound level in dB below the prompt;
//! * **the AM–FM correlation** — the correlation between the band-limited
//!   frequency track and the band-limited log envelope, and its regression slope
//!   in cents per dB. A beat null forces the frequency to swing hardest where
//!   the amplitude is *lowest*; a frequency movement driven by the string's own
//!   amplitude does the opposite. This is the statistic that says what the
//!   recording is actually doing, and it is the reason §5's verdict below is a
//!   split one.

use std::f64::consts::TAU;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use piano_emulator::hammer::{Hammer, MAX_SKEW_SAMPLES};
use piano_emulator::modal::ModalBank;
use piano_emulator::preset::{Preset as EnginePreset, Voicing};
use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_emulator::resonance::BridgeFilter;
use piano_emulator::soundboard::{pan_for_key, Soundboard};
use piano_emulator::string::{contact_taper, PartialShaping, StringParams};
use piano_emulator::types::{db_to_amp, Event, BLOCK, SAMPLE_RATE as ENGINE_SR};
use piano_tuner::{audio, detect_onset, Sample, SampleLibrary, SAMPLE_RATE};
use rustfft::{num_complex::Complex64, FftPlanner};

const SR: f64 = SAMPLE_RATE as f64;

/// The three keys, in report order. The same three
/// `renders/timbre-ladder/ANALYSIS.md` measures a linewidth on, and three of the
/// four `JITTER.md` reports, so every number here has a published counterpart.
const KEYS: [(u8, &str); 3] = [(60, "C4"), (45, "A2"), (84, "C6")];

/// Partials reported per key, matching `JITTER.md`.
const MAX_PARTIAL: usize = 4;

/// Velocity every headline render is struck at, and the two extra velocities the
/// velocity-invariance columns are read from.
const VELOCITY: u8 = 90;
const EXTRA_VELOCITIES: [u8; 2] = [40, 120];

/// Silence before the strike in every render, in frames. Four engine blocks, so
/// the offline renderers' block grid lines up with the engine's.
const PREROLL: usize = 4 * BLOCK;
/// How long every renderer runs past the strike, and how long a written file is.
const RENDER_S: f64 = 4.5;
const NOTE_S: f64 = 4.0;

/// `JITTER.md`'s analysis window, in seconds since the strike.
const T0_S: f64 = 0.3;
const T1_S: f64 = 3.0;

/// The two windows the double decay is read from, in seconds since the strike.
/// The first is inside the prompt sound and past the hammer's own noise; the
/// second is past every crossing time `docs/history/FUNDAMENTALS.md` §2.4 tabulates.
const PROMPT_LO_S: f64 = 0.10;
const PROMPT_HI_S: f64 = 0.60;
const TAIL_LO_S: f64 = 1.50;
const TAIL_HI_S: f64 = 3.50;

/// Time-domain standard deviation of the Gaussian band-pass, i.e. the smoothing
/// the frequency track gets: 31.8 Hz wide, never wider than a quarter of the
/// carrier. `JITTER.md`'s value, unchanged, so the two files' numbers compare.
const SMOOTH_SIGMA_S: f64 = 0.005;

/// Rate the demodulated track is decimated to before the phase is
/// differentiated.
const TRACK_HZ: f64 = 1000.0;

/// The modulation band both spectra are reported over.
const MOD_LO_HZ: f64 = 0.1;
const MOD_HI_HZ: f64 = 20.0;

/// A frequency excursion this far from the partial's mean pitch is counted.
const EXCURSION_CENTS: f64 = 3.0;

/// Transform length the band-pass is applied in: 5.46 s at 48 kHz, longer than
/// anything analysed, so the filter never wraps into its own input.
const FFT_N: usize = 1 << 18;

/// Window every written file's level is matched over, in seconds since the
/// strike. `JITTER.md`'s window, so the new files sit at the level the existing
/// listening set is at.
const MATCH_LO_S: f64 = 0.2;
const MATCH_HI_S: f64 = 2.0;

/// Fades applied to every written file, so nothing can click at either edge.
const FADE_IN_S: f64 = 0.002;
const FADE_OUT_S: f64 = 0.030;

// --------------------------------------------------------- the bridge, as physics

/// `Im Y / Re Y` at the bridge: how reactive the termination is compared with
/// how lossy it is.
///
/// Weinreich's measured admittances make the two parts comparable, and Capleton
/// §III.B works his example at a ratio of order one. It sets the *frequency
/// pull* the coupling produces; the pull common to a whole partial is
/// pre-compensated away in [`partial_modes`] (a tuner would tune it out — it is
/// 0.9 cents at C4 k=1), so what this number is left doing is the *relative*
/// pull between the symmetric and antisymmetric modes, which is the anti-veering.
const REACTIVE_RATIO: f64 = 1.0;

/// The values of `Im Y / Re Y` the sensitivity sweep re-solves and re-renders
/// the whole construction at.
///
/// It decides whether the coupling **attracts** the group's frequencies or
/// **repels** them, and the literature does not pin it. A purely resistive
/// bridge (0.0) is Woodhouse's anti-veering case — "with anti-veering there
/// will be no beats"; a reactive-dominated one (3.0) veers and the beats
/// survive. Everything in this file's verdict is a claim about *all* of these,
/// not about the one the headline is rendered at, which is why they are all
/// measured rather than argued.
const REACTIVE_SWEEP: [f64; 4] = [0.0, 0.25, 1.0, 3.0];

/// How much less reactive the bridge is horizontally than vertically.
///
/// This one number replaces `voicing.horizontal_offset_hz` entirely. Capleton,
/// summarising Weinreich: "the angular variation of the reactive part of the
/// bridge admittance is at least a factor of 10 smaller than the variation in
/// the resistive part", and his worked example uses a reactive ratio of
/// 1 : 0.925. The resulting polarization split is `N gamma_v beta eps / 2 pi`,
/// which is **proportional to omega** and about 0.014 Hz at C4 k=1 — against the
/// shipped preset's 0.35 Hz, flat across every partial of every key.
const REACTIVE_ANISOTROPY: f64 = 0.075;

/// The bridge's resistive anisotropy `Re Y_h / Re Y_v`, read off
/// `voicing.horizontal_decay_ratio` — `docs/history/FUNDAMENTALS.md` §5.4's re-reading of the
/// field. The horizontal plane loses less into the board, which is why it
/// outlives the vertical one.
fn resistive_anisotropy(voicing: &Voicing) -> f64 {
    f64::from(voicing.horizontal_decay_ratio)
}

/// Fraction of a partial's decay rate that is loss **into the board**, i.e. the
/// coupling constant of `docs/history/FUNDAMENTALS.md` §1.1, derived rather than read.
///
/// See the module doc: the slowest mode of the coupled group radiates nothing
/// and decays at `(1 - share) sigma_k`, the loudest decays at `sigma_k`, so
/// `1 - share` *is* the fitted aftersound/prompt decay ratio. This resolves the
/// §2.6 contradiction in favour of `horizontal_decay_ratio`, which is the field
/// that was fitted to recordings; the shipped `radiated_share = 0.5` would cap
/// the aftersound at half the prompt decay rate and delete the double decay.
fn radiated_share(voicing: &Voicing) -> f64 {
    1.0 - resistive_anisotropy(voicing)
}

/// The hammer's horizontal leak times the horizontal plane's radiation
/// efficiency, in amplitude.
///
/// Because `C` is block diagonal the two factors are never separable — every
/// horizontal mode's gain carries both as one common scalar — so the product is
/// what the model has, and it is taken from the field that was fitted to it.
/// At zero coupling this makes the construction reduce exactly to the engine's
/// `horizontal_gain_db`, which is what makes the comparison a controlled one.
fn horizontal_leak(voicing: &Voicing) -> f64 {
    f64::from(db_to_amp(voicing.horizontal_gain_db))
}

// ------------------------------------------------------------ the eigenproblem

/// One mode of one partial of one key: a complex pole and a complex radiated
/// gain. `2N` of them per partial — the same count `ModalBank` already holds.
#[derive(Clone, Copy, Debug)]
struct EigenMode {
    /// `lambda = -sigma + i 2 pi f`, in rad/s.
    lambda: Complex64,
    /// `G_m = (w . v_m) c_m`, the complex input/output gain.
    gain: Complex64,
    /// Which polarization block this mode came out of — it decides which of the
    /// two stereo positions the mode radiates from, exactly as the engine's two
    /// banks do.
    horizontal: bool,
}

impl EigenMode {
    fn sigma(&self) -> f64 {
        -self.lambda.re
    }

    fn hz(&self) -> f64 {
        self.lambda.im / TAU
    }
}

/// Coefficients of `prod_j (z - r_j)`, low order first, monic.
fn poly_from_roots(roots: &[Complex64]) -> Vec<Complex64> {
    let mut coeff = vec![Complex64::new(1.0, 0.0)];
    for &r in roots {
        let mut next = vec![Complex64::new(0.0, 0.0); coeff.len() + 1];
        for (i, &c) in coeff.iter().enumerate() {
            next[i + 1] += c;
            next[i] -= c * r;
        }
        coeff = next;
    }
    coeff
}

/// All roots of a monic complex polynomial, by Durand–Kerner.
///
/// Degree is at most `MAX_UNISON` = 3 here and the caller has already shifted
/// the variable so that every root is `O(radius)`, which is what makes a plain
/// Weierstrass iteration converge in a few dozen steps without any scaling care.
fn roots_of(coeff: &[Complex64], radius: f64) -> Vec<Complex64> {
    let n = coeff.len() - 1;
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![-coeff[0] / coeff[1]];
    }
    let seed = Complex64::new(0.4, 0.9);
    let scale = radius.max(1e-12);
    let mut z: Vec<Complex64> = (0..n).map(|i| seed.powu(i as u32) * scale).collect();
    let eval = |x: Complex64| {
        // Horner, high order first.
        coeff
            .iter()
            .rev()
            .fold(Complex64::new(0.0, 0.0), |acc, &c| acc * x + c)
    };
    for _ in 0..500 {
        let mut moved = 0.0f64;
        for i in 0..n {
            let mut denom = coeff[n];
            for j in 0..n {
                if j != i {
                    denom *= z[i] - z[j];
                }
            }
            if denom.norm() < 1e-300 {
                continue;
            }
            let step = eval(z[i]) / denom;
            z[i] -= step;
            moved = moved.max(step.norm());
        }
        if moved < 1e-13 * scale {
            break;
        }
    }
    z
}

/// The `N` eigenmodes of one polarization block: `D - c J_N` with
/// `D = diag(i omega_j - sigma_int)`.
///
/// The rank-one structure gives both halves in closed form. Eigenvalues are the
/// roots of `prod_j (lambda - d_j) + c sum_i prod_{j != i} (lambda - d_j)`;
/// eigenvectors are `v_jm = 1/(d_j - lambda_m)`; and because `D - cJ` is complex
/// symmetric the row of `V^-1` needed for the strike projection is
/// `v_m / (v_m . v_m)` — *not* `v_m^H`, which would be the wrong basis for a
/// non-normal matrix (`docs/history/FUNDAMENTALS.md` §5.1).
fn block_modes(
    omegas: &[f64],
    sigma_int: f64,
    c: Complex64,
    u: &[Complex64],
    horizontal: bool,
) -> Vec<EigenMode> {
    let n = omegas.len();
    debug_assert_eq!(u.len(), n);
    // Degenerate poles make `1/(d_j - lambda)` singular in a way that does not
    // cancel, so a group whose strings are tuned to the same number is nudged
    // apart by an amount far below any audible or measurable frequency
    // (1e-6 rad/s is 1.6e-7 Hz). A real unison is never exactly in tune, and
    // `notes.detune_cents = 0` is a bisection rung, not an instrument.
    let mut omegas: Vec<f64> = omegas.to_vec();
    for i in 0..n {
        for j in 0..i {
            if (omegas[i] - omegas[j]).abs() < 1e-6 {
                omegas[i] += 1e-6 * (i - j) as f64;
            }
        }
    }
    let d: Vec<Complex64> = omegas
        .iter()
        .map(|&w| Complex64::new(-sigma_int, w))
        .collect();
    // Shift to the block's centre before forming the polynomial: the roots are
    // then O(detune spread + N|c|) instead of O(omega), which is the difference
    // between a well conditioned degree-3 solve and a hopeless one.
    let centre = d.iter().sum::<Complex64>() / n as f64;
    let e: Vec<Complex64> = d.iter().map(|&x| x - centre).collect();
    let mut coeff = poly_from_roots(&e);
    // `+ c * sum_i prod_{j != i} (z - e_j)`, which is `c` times the derivative
    // of `prod_j (z - e_j)`.
    for i in 1..coeff.len() {
        // Read before write: step `i` writes index `i-1` and reads index `i`,
        // which the previous steps have not touched.
        let above = coeff[i];
        coeff[i - 1] += c * above * i as f64;
    }
    let radius = e.iter().map(|x| x.norm()).fold(0.0, f64::max) + n as f64 * c.norm();
    let mut modes = Vec::with_capacity(n);
    for z in roots_of(&coeff, radius.max(1e-9)) {
        let lambda = centre + z;
        let v: Vec<Complex64> = d.iter().map(|&dj| 1.0 / (dj - lambda)).collect();
        let norm: Complex64 = v.iter().map(|x| x * x).sum();
        if norm.norm() < 1e-300 {
            continue;
        }
        let c_m: Complex64 = v.iter().zip(u).map(|(a, b)| a * b).sum::<Complex64>() / norm;
        let radiated: Complex64 = v.iter().sum();
        modes.push(EigenMode {
            lambda,
            gain: radiated * c_m,
            horizontal,
        });
    }
    modes
}

/// Everything about one key the eigen construction needs before a note is struck.
struct EigenKey {
    /// `2N` modes per partial, partial 1 first.
    partials: Vec<Vec<EigenMode>>,
    /// The scale [`decay_scale`] solved for on each partial, reported so the
    /// normalisation is visible rather than implied.
    scales: Vec<f64>,
    /// Which of the shipped preset's numbers the construction did *not* need.
    unison: usize,
}

/// The `2N` modes of partial `k`, at a given scale on the whole loss budget.
///
/// `scale` multiplies `sigma_int` and `gamma_v` together, which is the only free
/// parameter left once the physics has fixed their ratio; [`decay_scale`] picks
/// it so that the composite reaches −60 dB on the fitted anchor.
fn partial_modes(
    params: &StringParams,
    voicing: &Voicing,
    k: usize,
    sigma_hat: f64,
    gain_k: f64,
    scale: f64,
    beta: f64,
) -> Vec<EigenMode> {
    let n = params.unison.max(1);
    let share = radiated_share(voicing);
    let sigma = scale * sigma_hat;
    let sigma_int = (1.0 - share) * sigma;
    let gamma_v = share * sigma / n as f64;
    let c_v = Complex64::new(gamma_v, gamma_v * beta);
    let c_h = Complex64::new(
        gamma_v * resistive_anisotropy(voicing),
        gamma_v * beta * (1.0 - REACTIVE_ANISOTROPY),
    );
    // The pull common to the whole partial is a tuning offset, not a beat: the
    // symmetric vertical mode would sit `N gamma_v beta / 2 pi` flat (0.9 cents
    // at C4 k=1), and `notes.f0` is fitted to recordings that already contain
    // whatever pull the real bridge applies. Adding it back keeps this
    // prototype's pitch identical to the shipped engine's, so the A/B is about
    // the wobble and not about the tuning. What survives is the *difference*
    // between the modes, which is the anti-veering, and the difference between
    // the two blocks, which is the polarization split.
    let compensation = n as f64 * gamma_v * beta;
    let omegas: Vec<f64> = (0..n)
        .map(|j| {
            let detune = f64::from(voicing.detune_ratio(j, n, params.detune_cents));
            TAU * f64::from(params.partial_freq(k)) * detune + compensation
        })
        .collect();
    // The strike vector: per-string share, and the per-string timing skew as a
    // phase rotation at this partial's own frequency — `MAX_SKEW_SAMPLES` is the
    // same skew `PianoString`'s caller applies in the time domain.
    let u_v: Vec<Complex64> = (0..n)
        .map(|j| {
            let share = f64::from(voicing.strike_share(j, n));
            let skew = (j * MAX_SKEW_SAMPLES / n) as f64 / SR;
            Complex64::from_polar(share * gain_k, -omegas[j] * skew)
        })
        .collect();
    let leak = horizontal_leak(voicing);
    let u_h: Vec<Complex64> = u_v.iter().map(|x| x * leak).collect();
    let mut modes = block_modes(&omegas, sigma_int, c_v, &u_v, false);
    modes.extend(block_modes(&omegas, sigma_int, c_h, &u_h, true));
    modes
}

/// The composite envelope's −60 dB time for a mode set, in seconds.
///
/// Evaluated on a uniform grid by advancing each mode by a fixed complex ratio,
/// which is one complex multiply per mode per step instead of an `exp`. The
/// *last* time above the threshold is taken, not the first crossing: a
/// double-decay envelope with a beat on it can dip below and come back, and what
/// the anchor means is when the note has gone.
fn composite_t60(modes: &[EigenMode], t_max: f64, steps: usize) -> f64 {
    let dt = t_max / steps as f64;
    let mut state: Vec<Complex64> = modes.iter().map(|m| m.gain).collect();
    let step: Vec<Complex64> = modes.iter().map(|m| (m.lambda * dt).exp()).collect();
    let mut peak = 0.0f64;
    let mut last = 0.0f64;
    for i in 0..=steps {
        let sum: Complex64 = state.iter().sum();
        let a = sum.norm();
        if i < steps / 200 + 2 {
            peak = peak.max(a);
        }
        if peak > 0.0 && a > 1e-3 * peak {
            last = i as f64 * dt;
        }
        for (s, &z) in state.iter_mut().zip(&step) {
            *s *= z;
        }
    }
    last
}

/// The one scale on `sigma_int` and `gamma_v` that puts the composite's −60 dB
/// crossing on the fitted anchor `6.91 / sigma_hat`.
///
/// This is `Voicing::vertical_decay_factor` generalised: that closed form
/// assumes the engine's own two-exponential construction and is exact for it,
/// and there is no closed form for `2N` coupled modes, so it is solved. Damped
/// fixed point on `log scale`, which converges monotonically because a longer
/// T60 always wants a larger scale.
fn decay_scale(
    params: &StringParams,
    voicing: &Voicing,
    k: usize,
    sigma_hat: f64,
    gain_k: f64,
    beta: f64,
) -> f64 {
    let target = 6.91 / sigma_hat;
    let mut scale = 1.0f64;
    for _ in 0..48 {
        let modes = partial_modes(params, voicing, k, sigma_hat, gain_k, scale, beta);
        let t60 = composite_t60(&modes, 10.0 * target, 4000);
        if t60 <= 0.0 {
            break;
        }
        let ratio = t60 / target;
        if (ratio - 1.0).abs() < 1e-4 {
            break;
        }
        scale *= ratio.powf(0.6);
    }
    scale
}

/// `engine::string::radiated_damping`, which is private there: the per-partial
/// multiplier the bridge admittance's *fluctuation* puts on a partial's decay.
fn radiated_damping(params: &StringParams, voicing: &Voicing, partials: usize) -> Vec<f32> {
    let share = match &voicing.bridge {
        Some(bridge) if bridge.radiated_share > 0.0 => bridge.radiated_share,
        _ => return vec![1.0; partials],
    };
    let modes = BridgeFilter::peaks_only(voicing.bridge.as_ref().expect("checked above"));
    (1..=partials)
        .map(|k| {
            let excess = modes.magnitude(params.partial_freq(k)) - 1.0;
            (1.0 + share * excess).clamp(0.25, 4.0)
        })
        .collect()
}

/// The per-partial input gain `g_k` the engine builds, with the strike comb, the
/// contact taper and `notes.partial_gains` in it. Copied term for term from
/// `PianoString::new`, minus the per-string share, which the eigen construction
/// carries in `u` instead.
fn partial_gain(
    params: &StringParams,
    voicing: &Voicing,
    shaping: &PartialShaping<'_>,
    k: usize,
) -> f64 {
    const REFERENCE_F0: f32 = 261.6256;
    let output_scale = voicing.excitation_scale * params.bridge_gain * params.f0 / REFERENCE_F0;
    let comb = (k as f32 * std::f32::consts::PI * params.strike_position).sin();
    let comb = if params.comb_floor > 0.0 {
        comb.signum() * (comb * comb + params.comb_floor * params.comb_floor).sqrt()
    } else {
        comb
    };
    f64::from(
        output_scale * comb * contact_taper(k, params.contact_width) * shaping.gain_at(k)
            / ENGINE_SR,
    )
}

impl EigenKey {
    /// Solves every partial of one key. Nothing here depends on velocity, so in
    /// the engine this would run once at preset load and be cached.
    fn new(preset: &EnginePreset, key: u8, beta: f64) -> EigenKey {
        let params = preset.string_params(key);
        let voicing = &preset.voicing;
        let shaping = preset.partial_shaping(key);
        let count = params.partial_count();
        let radiated = radiated_damping(&params, voicing, count);
        let mut partials = Vec::with_capacity(count);
        let mut scales = Vec::with_capacity(count);
        for k in 1..=count {
            // The fitted whole-note rate of this partial, with M3's per-partial
            // table and the bridge's `Re Y` fluctuation on it — the same
            // `sigma_k` the engine starts from, before its own vertical /
            // horizontal / per-string factors, all three of which the
            // eigenproblem replaces.
            let sigma_hat =
                f64::from(params.partial_sigma(k) * shaping.sigma_scale_at(k) * radiated[k - 1]);
            let gain_k = partial_gain(&params, voicing, &shaping, k);
            let scale = decay_scale(&params, voicing, k, sigma_hat, gain_k, beta);
            partials.push(partial_modes(
                &params, voicing, k, sigma_hat, gain_k, scale, beta,
            ));
            scales.push(scale);
        }
        EigenKey {
            partials,
            scales,
            unison: params.unison,
        }
    }

    /// Every mode of the key, flattened, with the polarization tag kept.
    fn modes(&self) -> Vec<EigenMode> {
        self.partials.iter().flatten().copied().collect()
    }
}

// -------------------------------------------------------------- the renderers

type Stereo = (Vec<f32>, Vec<f32>);

fn total_frames() -> usize {
    PREROLL + (RENDER_S * SR) as usize
}

/// One complex one-pole per eigenmode, driven by a common real input.
///
/// `s[n] = a s[n-1] + G x[n]`, `y[n] = Im(s[n])` — the identical recurrence
/// `engine::modal::ModalBank` runs, with the one difference `docs/history/FUNDAMENTALS.md`
/// §5.2 costs out: the input gain is complex, so the update carries one extra
/// multiply-add. Held in `f64` here because this is a measurement rig, not the
/// hot loop; the engine's `f32` is unaffected by the change.
struct EigenBank {
    pole: Vec<Complex64>,
    gain: Vec<Complex64>,
    state: Vec<Complex64>,
    horizontal: Vec<bool>,
}

impl EigenBank {
    fn new(modes: &[EigenMode]) -> EigenBank {
        EigenBank {
            pole: modes.iter().map(|m| (m.lambda / SR).exp()).collect(),
            gain: modes.iter().map(|m| m.gain).collect(),
            state: vec![Complex64::new(0.0, 0.0); modes.len()],
            horizontal: modes.iter().map(|m| m.horizontal).collect(),
        }
    }

    /// One block, adding the vertical block's modes into `out_v` and the
    /// horizontal block's into `out_h` — the split `PianoString::process_split`
    /// makes, so the polarization stereo spread of the shipped preset survives.
    fn process(&mut self, input: &[f32], out_v: &mut [f32], out_h: &mut [f32]) {
        for (n, &x) in input.iter().enumerate() {
            let x = f64::from(x);
            let (mut v, mut h) = (0.0f64, 0.0f64);
            for i in 0..self.state.len() {
                let s = self.state[i] * self.pole[i] + self.gain[i] * x;
                self.state[i] = s;
                if self.horizontal[i] {
                    h += s.im;
                } else {
                    v += s.im;
                }
            }
            out_v[n] += v as f32;
            out_h[n] += h as f32;
        }
    }
}

/// Strikes a key through the eigen construction and radiates it through the
/// engine's own soundboard, hammer and panning.
fn render_eigen(preset: &EnginePreset, cached: &EigenKey, key: u8, vel: u8) -> Stereo {
    let mut bank = EigenBank::new(&cached.modes());
    render_through_board(preset, key, vel, |excitation, out_v, out_h| {
        bank.process(excitation, out_v, out_h);
    })
}

/// The engine's own modal construction, in the same offline rig — the control
/// that separates "the eigenproblem" from "not the engine's signal path".
///
/// Every formula is `PianoString::new`'s: the partial layout, the damping law
/// with `Voicing::vertical_decay_factor` on it, the per-string
/// `unison_sigma_scale`, the `horizontal_offset_hz` polarization split, the
/// `horizontal_gain_db` level, the bridge's `Re Y` correction, the strike comb
/// with its taper, the per-partial tables, and the one-block-late unison
/// coupling. This is `renders/timbre-ladder`'s rung `09` with M3's tables added.
struct ShippedModal {
    strings: Vec<(ModalBank, ModalBank)>,
    excitation: Vec<[f32; BLOCK]>,
    previous: Vec<[f32; BLOCK]>,
    group_previous: [f32; BLOCK],
    coupling: f32,
    shares: Vec<f32>,
}

impl ShippedModal {
    fn new(preset: &EnginePreset, key: u8) -> ShippedModal {
        const REFERENCE_F0: f32 = 261.6256;
        let params = preset.string_params(key);
        let voicing = &preset.voicing;
        let shaping = preset.partial_shaping(key);
        let partials = params.partial_count();
        let radiated = radiated_damping(&params, voicing, partials);
        let output_scale = voicing.excitation_scale * params.bridge_gain * params.f0 / REFERENCE_F0;
        let vertical_factor = voicing.vertical_decay_factor();
        let horizontal_gain = db_to_amp(voicing.horizontal_gain_db);
        let mut strings = Vec::with_capacity(params.unison);
        for (i, &offset) in voicing
            .horizontal_offset_hz
            .iter()
            .take(params.unison)
            .enumerate()
        {
            let detune = voicing.detune_ratio(i, params.unison, params.detune_cents);
            let sigma_scale = voicing.sigma_scale(i, params.unison);
            let mut vertical = ModalBank::with_capacity(partials);
            let mut horizontal = ModalBank::with_capacity(partials);
            for k in 1..=partials {
                let f = params.partial_freq(k) * detune;
                let sigma = params.partial_sigma(k)
                    * shaping.sigma_scale_at(k)
                    * vertical_factor
                    * sigma_scale
                    * radiated[k - 1];
                let g = partial_gain(&params, voicing, &shaping, k) as f32;
                vertical.push_mode(f, sigma, g);
                horizontal.push_mode(
                    f + offset,
                    sigma * voicing.horizontal_decay_ratio,
                    g * horizontal_gain,
                );
            }
            strings.push((vertical, horizontal));
        }
        let n = strings.len();
        ShippedModal {
            strings,
            excitation: vec![[0.0; BLOCK]; n],
            previous: vec![[0.0; BLOCK]; n],
            group_previous: [0.0; BLOCK],
            coupling: voicing.unison_coupling / output_scale,
            shares: (0..params.unison)
                .map(|i| voicing.strike_share(i, params.unison))
                .collect(),
        }
    }

    /// `PianoString::process_split`, term for term, including the one-block-late
    /// bridge coupling.
    fn process(&mut self, out_v: &mut [f32], out_h: &mut [f32]) {
        if self.strings.len() == 1 {
            let (v, h) = &mut self.strings[0];
            v.process_add(&self.excitation[0], out_v);
            h.process_add(&self.excitation[0], out_h);
            self.excitation[0].fill(0.0);
            return;
        }
        for (excitation, previous) in self.excitation.iter_mut().zip(&self.previous) {
            for ((e, &sum), &own) in excitation
                .iter_mut()
                .zip(&self.group_previous)
                .zip(previous.iter())
            {
                *e += self.coupling * (sum - own);
            }
        }
        self.group_previous.fill(0.0);
        let mut vertical = [0.0f32; BLOCK];
        for (s, (v, h)) in self.strings.iter_mut().enumerate() {
            vertical.fill(0.0);
            self.previous[s].fill(0.0);
            v.process_add(&self.excitation[s], &mut vertical);
            h.process_add(&self.excitation[s], &mut self.previous[s]);
            self.excitation[s].fill(0.0);
            for i in 0..BLOCK {
                out_v[i] += vertical[i];
                out_h[i] += self.previous[s][i];
                self.previous[s][i] += vertical[i];
                self.group_previous[i] += self.previous[s][i];
            }
        }
    }
}

fn render_shipped_modal(preset: &EnginePreset, key: u8, vel: u8) -> Stereo {
    let mut note = ShippedModal::new(preset, key);
    let mut hammer = Hammer::new(preset.hammer_params(key));
    let pan = pan_for_key(key);
    let spread = preset.pan_spread(key) * if key % 2 == 0 { 1.0 } else { -1.0 };
    let (pan_v, pan_h) = (pan - spread, pan + spread);
    let mut board = Soundboard::new(&preset.soundboard);
    let frames = total_frames();
    let (mut left, mut right) = (vec![0.0f32; frames], vec![0.0f32; frames]);
    let mut vertical = [0.0f32; BLOCK];
    let mut horizontal = [0.0f32; BLOCK];
    let strings = note.strings.len();
    let mut start = 0usize;
    while start < frames {
        let end = (start + BLOCK).min(frames);
        if start == PREROLL {
            hammer.strike_midi(u16::from(vel));
        }
        board.begin_block();
        if hammer.is_active() {
            for s in 0..strings {
                let skew = s * MAX_SKEW_SAMPLES / strings.max(1);
                hammer.add_pulse(&mut note.excitation[s], skew, note.shares[s]);
            }
            hammer.advance(BLOCK);
        }
        vertical.fill(0.0);
        horizontal.fill(0.0);
        note.process(&mut vertical, &mut horizontal);
        board.add_voice(&vertical, pan_v);
        board.add_voice(&horizontal, pan_h);
        let mut block_l = [0.0f32; BLOCK];
        let mut block_r = [0.0f32; BLOCK];
        board.process(&mut block_l, &mut block_r);
        left[start..end].copy_from_slice(&block_l[..end - start]);
        right[start..end].copy_from_slice(&block_r[..end - start]);
        start = end;
    }
    cut_preroll((left, right))
}

/// The block loop both offline renderers share: the engine's hammer, the
/// engine's soundboard, the engine's panning, and one closure that turns the
/// hammer's pulse into two polarization buses.
fn render_through_board(
    preset: &EnginePreset,
    key: u8,
    vel: u8,
    mut voice: impl FnMut(&[f32], &mut [f32], &mut [f32]),
) -> Stereo {
    let mut hammer = Hammer::new(preset.hammer_params(key));
    let mut board = Soundboard::new(&preset.soundboard);
    let pan = pan_for_key(key);
    let spread = preset.pan_spread(key) * if key % 2 == 0 { 1.0 } else { -1.0 };
    let (pan_v, pan_h) = (pan - spread, pan + spread);
    let frames = total_frames();
    let (mut left, mut right) = (vec![0.0f32; frames], vec![0.0f32; frames]);
    let mut excitation = [0.0f32; BLOCK];
    let mut vertical = [0.0f32; BLOCK];
    let mut horizontal = [0.0f32; BLOCK];
    let mut start = 0usize;
    while start < frames {
        let end = (start + BLOCK).min(frames);
        if start == PREROLL {
            hammer.strike_midi(u16::from(vel));
        }
        board.begin_block();
        excitation.fill(0.0);
        if hammer.is_active() {
            // One common, unskewed, unit-share pulse: the shares and the skew
            // are inside the modes' complex gains, which is the whole point of
            // the eigen construction — `N` input buffers collapse into `2N`
            // complex scalars.
            hammer.add_pulse(&mut excitation, 0, 1.0);
            hammer.advance(BLOCK);
        }
        vertical.fill(0.0);
        horizontal.fill(0.0);
        voice(&excitation, &mut vertical, &mut horizontal);
        board.add_voice(&vertical, pan_v);
        board.add_voice(&horizontal, pan_h);
        let mut block_l = [0.0f32; BLOCK];
        let mut block_r = [0.0f32; BLOCK];
        board.process(&mut block_l, &mut block_r);
        left[start..end].copy_from_slice(&block_l[..end - start]);
        right[start..end].copy_from_slice(&block_r[..end - start]);
        start = end;
    }
    cut_preroll((left, right))
}

/// Drops the preroll so that frame 0 of every signal is the strike, which is
/// what every measurement window below is counted from.
fn cut_preroll((left, right): Stereo) -> Stereo {
    let keep = (RENDER_S * SR) as usize;
    let cut = |c: Vec<f32>| -> Vec<f32> {
        let mut v: Vec<f32> = c.into_iter().skip(PREROLL).collect();
        v.resize(keep, 0.0);
        v
    };
    (cut(left), cut(right))
}

/// The shipped engine's render of one note through its public API.
fn render_engine(preset: &EnginePreset, key: u8, vel: u8) -> Stereo {
    let preroll_s = PREROLL as f32 / ENGINE_SR;
    let events = [RenderEvent::new(
        preroll_s,
        Event::NoteOn {
            key,
            vel: u16::from(vel),
        },
    )];
    let (left, right) = render_to_buffer(preset, &events, preroll_s + RENDER_S as f32);
    let mut left = left;
    let mut right = right;
    left.resize(total_frames(), 0.0);
    right.resize(total_frames(), 0.0);
    cut_preroll((left, right))
}

/// The library layer of `key` a strike at `velocity` would trigger.
fn layer_for(
    library: &SampleLibrary,
    key: u8,
    velocity: u8,
) -> Result<&Sample, Box<dyn std::error::Error>> {
    library
        .layers(key)
        .iter()
        .find(|s| (s.lovel..=s.hivel).contains(&velocity))
        .ok_or_else(|| format!("key {key} has no layer covering velocity {velocity}").into())
}

/// The recording, on the engine's clock, cut so that frame 0 is the strike.
fn recording(sample: &Sample) -> Result<Stereo, Box<dyn std::error::Error>> {
    let clip = audio::load_at(&sample.path, SAMPLE_RATE)?;
    let onset = detect_onset(&clip.mono(), SR);
    let start = (onset * SR).round() as usize;
    let frames = (RENDER_S * SR) as usize;
    let channel = |i: usize| -> Vec<f32> {
        let source = &clip.channels[i.min(clip.channel_count() - 1)];
        (0..frames)
            .map(|n| source.get(start + n).copied().unwrap_or(0.0))
            .collect()
    };
    Ok((channel(0), channel(1)))
}

// ------------------------------------------------------------ the measurement

/// One partial's demodulated track, on the [`TRACK_HZ`] grid. `JITTER.md`'s
/// structure, with the envelope kept over a longer span so the double decay can
/// be read off the same demodulation the jitter is.
struct Track {
    mean_hz: f64,
    peak_db: f64,
    cents: Vec<f64>,
    amp_db: Vec<f64>,
    weight: Vec<f64>,
    /// The log envelope over [`PROMPT_LO_S`]–[`TAIL_HI_S`], for the double decay.
    decay_db: Vec<f64>,
    decay_t0: f64,
}

const MIN_PEAK_DB: f64 = 10.0;

/// Everything one partial of one signal contributes to the tables.
struct PartialStats {
    mean_hz: f64,
    peak_db: f64,
    band_cents: f64,
    p95_cents: f64,
    /// Separate runs past ±[`EXCURSION_CENTS`] inside the band, per second.
    excursions_per_s: f64,
    raw_cents: f64,
    weighted_cents: f64,
    freq_flatness_db: f64,
    beat_depth_db: f64,
    amp_flatness_db: f64,
    /// Mean rate of the log envelope's own movement, from its sign changes, Hz.
    beat_rate_hz: f64,
    /// Slope of the log envelope over the prompt window, dB/s (negative).
    prompt_db_s: f64,
    /// Slope of the log envelope over the tail window, dB/s (negative).
    tail_db_s: f64,
    /// Where the tail's straight line extrapolates back to at the strike,
    /// relative to the prompt's — the aftersound level, in dB below the prompt.
    aftersound_db: f64,
    /// Correlation between the band-limited frequency track and the
    /// band-limited log envelope, and the regression slope in cents per dB.
    am_fm_r: f64,
    am_fm_cents_per_db: f64,
}

impl PartialStats {
    /// Where in the note the wobble sits: near 1 is a wobble that rides the loud
    /// part of the partial, a small fraction is a spike at the null of a beat.
    fn placement(&self) -> f64 {
        if self.raw_cents > 0.0 {
            self.weighted_cents / self.raw_cents
        } else {
            0.0
        }
    }
}

/// The forward transform of one signal, computed once and reused by every
/// partial of it. `JITTER.md`'s code, with a second demodulation window.
struct Spectrum {
    bins: Vec<Complex64>,
}

impl Spectrum {
    fn new(signal: &[f64], planner: &mut FftPlanner<f64>) -> Spectrum {
        let mut bins: Vec<Complex64> = (0..FFT_N)
            .map(|n| Complex64::new(signal.get(n).copied().unwrap_or(0.0), 0.0))
            .collect();
        planner.plan_fft_forward(FFT_N).process(&mut bins);
        Spectrum { bins }
    }

    fn hz(m: usize) -> f64 {
        m as f64 * SR / FFT_N as f64
    }

    /// The strongest bin within `+-half_width` of `nominal`, refined by a
    /// parabolic fit, and how far it stands over the band's median magnitude.
    fn peak_near(&self, nominal: f64, half_width: f64) -> (f64, f64) {
        let bin = |hz: f64| ((hz * FFT_N as f64 / SR).round() as isize).max(1) as usize;
        let lo = bin(nominal - half_width).max(1);
        let hi = bin(nominal + half_width).min(FFT_N / 2 - 2);
        if hi <= lo {
            return (nominal, 0.0);
        }
        let mag = |m: usize| self.bins[m].norm();
        let mut best = lo;
        for m in lo..=hi {
            if mag(m) > mag(best) {
                best = m;
            }
        }
        let mut band: Vec<f64> = (lo..=hi).map(mag).collect();
        band.sort_by(|a, b| a.partial_cmp(b).expect("magnitudes are finite"));
        let median = band[band.len() / 2].max(f64::MIN_POSITIVE);
        let peak_db = 20.0 * (mag(best) / median).log10();
        let (a, b, c) = (
            mag(best - 1).max(f64::MIN_POSITIVE).ln(),
            mag(best).max(f64::MIN_POSITIVE).ln(),
            mag(best + 1).max(f64::MIN_POSITIVE).ln(),
        );
        let denom = a - 2.0 * b + c;
        let delta = if denom.abs() > 1e-12 {
            (0.5 * (a - c) / denom).clamp(-0.5, 0.5)
        } else {
            0.0
        };
        (Spectrum::hz(best) + delta * SR / FFT_N as f64, peak_db)
    }

    /// The partial's analytic signal, band-passed by a Gaussian centred on
    /// `carrier`, demodulated to zero and decimated to [`TRACK_HZ`]; returned
    /// over `[t0, t1]` inclusive of one extra sample at each end.
    fn demodulate(
        &self,
        carrier: f64,
        t0: f64,
        t1: f64,
        planner: &mut FftPlanner<f64>,
    ) -> Vec<Complex64> {
        let sigma_f = (1.0 / (TAU * SMOOTH_SIGMA_S)).min(carrier / 4.0);
        let mut z = vec![Complex64::new(0.0, 0.0); FFT_N];
        let span = (6.0 * sigma_f * FFT_N as f64 / SR).ceil() as usize;
        let centre = (carrier * FFT_N as f64 / SR).round() as usize;
        let lo = centre.saturating_sub(span).max(1);
        let hi = (centre + span).min(FFT_N / 2 - 1);
        for (m, bin) in z.iter_mut().enumerate().take(hi + 1).skip(lo) {
            let u = (Spectrum::hz(m) - carrier) / sigma_f;
            *bin = self.bins[m] * (2.0 * (-0.5 * u * u).exp());
        }
        planner.plan_fft_inverse(FFT_N).process(&mut z);
        let scale = 1.0 / FFT_N as f64;
        let step = (SR / TRACK_HZ).round() as usize;
        let from = (t0 * SR) as usize;
        let to = ((t1 * SR) as usize + step).min(FFT_N - 1);
        (from..=to)
            .step_by(step)
            .map(|n| {
                let phase = -TAU * carrier * n as f64 / SR;
                z[n] * scale * Complex64::from_polar(1.0, phase)
            })
            .collect()
    }
}

/// Demodulates one partial and returns its track, or `None` if the partial is
/// not present over its own background.
fn track_partial(
    spectrum: &Spectrum,
    nominal_hz: f64,
    search_half_width: f64,
    planner: &mut FftPlanner<f64>,
) -> Option<Track> {
    let (carrier_hz, peak_db) = spectrum.peak_near(nominal_hz, search_half_width);
    if peak_db < MIN_PEAK_DB {
        return None;
    }
    let y = spectrum.demodulate(carrier_hz, T0_S, T1_S, planner);
    if y.len() < 3 {
        return None;
    }
    let mut inst = Vec::with_capacity(y.len() - 1);
    let mut weight = Vec::with_capacity(y.len() - 1);
    for j in 0..y.len() - 1 {
        let d = y[j + 1] * y[j].conj();
        inst.push(carrier_hz + d.arg() * TRACK_HZ / TAU);
        weight.push((y[j].norm_sqr() * y[j + 1].norm_sqr()).sqrt());
    }
    let total: f64 = weight.iter().sum();
    if total.is_nan() || total <= 0.0 {
        return None;
    }
    let mean_hz: f64 = inst
        .iter()
        .zip(&weight)
        .map(|(f, w)| f * w / total)
        .sum::<f64>();
    if mean_hz.is_nan() || mean_hz <= 0.0 {
        return None;
    }
    let cents: Vec<f64> = inst
        .iter()
        .map(|f| {
            if *f > 0.0 {
                1200.0 * (f / mean_hz).log2()
            } else {
                -1200.0 * (TRACK_HZ / 2.0 / mean_hz).log2().abs()
            }
        })
        .collect();
    let peak_power = weight.iter().copied().fold(0.0f64, f64::max);
    let amp_db: Vec<f64> = weight
        .iter()
        .map(|w| 10.0 * w.max(1e-300).log10())
        .collect();
    let weight: Vec<f64> = weight.iter().map(|w| w / peak_power).collect();
    let long = spectrum.demodulate(carrier_hz, PROMPT_LO_S, TAIL_HI_S, planner);
    let decay_db: Vec<f64> = long
        .iter()
        .map(|z| 20.0 * z.norm().max(1e-300).log10())
        .collect();
    Some(Track {
        mean_hz,
        peak_db,
        cents,
        amp_db,
        weight,
        decay_db,
        decay_t0: PROMPT_LO_S,
    })
}

/// Least-squares slope and intercept of `y` against `t`, over the samples whose
/// time falls in `[lo, hi]`. Returns `(slope per second, value at t = 0)`.
fn line_fit(y: &[f64], t0: f64, rate: f64, lo: f64, hi: f64) -> Option<(f64, f64)> {
    let mut n = 0.0f64;
    let (mut st, mut sy, mut stt, mut sty) = (0.0, 0.0, 0.0, 0.0);
    for (i, &v) in y.iter().enumerate() {
        let t = t0 + i as f64 / rate;
        if t < lo || t > hi || !v.is_finite() {
            continue;
        }
        n += 1.0;
        st += t;
        sy += v;
        stt += t * t;
        sty += t * v;
    }
    if n < 8.0 {
        return None;
    }
    let denom = n * stt - st * st;
    if denom.abs() < 1e-12 {
        return None;
    }
    let slope = (n * sty - st * sy) / denom;
    Some((slope, (sy - slope * st) / n))
}

/// Every statistic of one track.
fn statistics(track: &Track) -> PartialStats {
    let n = track.cents.len() as f64;
    let raw_cents = (track.cents.iter().map(|c| c * c).sum::<f64>() / n).sqrt();
    let wsum: f64 = track.weight.iter().sum();
    let weighted_cents = (track
        .cents
        .iter()
        .zip(&track.weight)
        .map(|(c, w)| c * c * w)
        .sum::<f64>()
        / wsum.max(f64::MIN_POSITIVE))
    .sqrt();

    let detrended_cents = detrended(&track.cents, 1);
    let freq_mod = mod_stats(&detrended_cents, TRACK_HZ);
    let band = band_limited(&detrended_cents, TRACK_HZ);
    let band_cents = (band.iter().map(|c| c * c).sum::<f64>() / n).sqrt();
    let mut abs: Vec<f64> = band.iter().map(|c| c.abs()).collect();
    abs.sort_by(|a, b| a.partial_cmp(b).expect("deviations are finite"));
    let p95_cents = abs[((abs.len() - 1) as f64 * 0.95) as usize];

    // Runs, not samples: one slow swing past the threshold and back is one
    // excursion however many milliseconds it spends out there.
    let mut runs = 0usize;
    let mut inside = true;
    for c in &band {
        if c.abs() > EXCURSION_CENTS {
            if inside {
                runs += 1;
            }
            inside = false;
        } else {
            inside = true;
        }
    }

    let residual = detrended(&track.amp_db, 3);
    let amp_mod = mod_stats(&residual, TRACK_HZ);
    let amp_band = band_limited(&residual, TRACK_HZ);
    let mut sorted = amp_band.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("envelope values are finite"));
    let at = |q: f64| sorted[((sorted.len() - 1) as f64 * q) as usize];

    // AM-FM: is the frequency moving because the amplitude is, and in which
    // direction? Both tracks are the same length, band-limited the same way and
    // already detrended, so this is a plain Pearson r plus its regression slope.
    let (mut sxx, mut syy, mut sxy) = (0.0f64, 0.0f64, 0.0f64);
    for (&f, &a) in band.iter().zip(&amp_band) {
        sxx += a * a;
        syy += f * f;
        sxy += a * f;
    }
    let am_fm_r = if sxx > 0.0 && syy > 0.0 {
        sxy / (sxx * syy).sqrt()
    } else {
        0.0
    };
    let am_fm_cents_per_db = if sxx > 0.0 { sxy / sxx } else { 0.0 };

    let prompt = line_fit(
        &track.decay_db,
        track.decay_t0,
        TRACK_HZ,
        PROMPT_LO_S,
        PROMPT_HI_S,
    );
    let tail = line_fit(
        &track.decay_db,
        track.decay_t0,
        TRACK_HZ,
        TAIL_LO_S,
        TAIL_HI_S,
    );
    let (prompt_db_s, prompt_at_0) = prompt.unwrap_or((f64::NAN, f64::NAN));
    let (tail_db_s, tail_at_0) = tail.unwrap_or((f64::NAN, f64::NAN));

    PartialStats {
        mean_hz: track.mean_hz,
        peak_db: track.peak_db,
        band_cents,
        p95_cents,
        excursions_per_s: runs as f64 * TRACK_HZ / n,
        raw_cents,
        weighted_cents,
        freq_flatness_db: freq_mod.flatness_db,
        beat_depth_db: at(0.95) - at(0.05),
        amp_flatness_db: amp_mod.flatness_db,
        beat_rate_hz: crossing_rate(&amp_band, TRACK_HZ),
        prompt_db_s,
        tail_db_s,
        aftersound_db: tail_at_0 - prompt_at_0,
        am_fm_r,
        am_fm_cents_per_db,
    }
}

/// `x` with everything outside [`MOD_LO_HZ`]–[`MOD_HI_HZ`] removed, zero phase.
fn band_limited(x: &[f64], rate: f64) -> Vec<f64> {
    let n = x.len();
    if n < 16 {
        return x.to_vec();
    }
    let mut planner = FftPlanner::<f64>::new();
    let mut buf: Vec<Complex64> = x.iter().map(|&v| Complex64::new(v, 0.0)).collect();
    planner.plan_fft_forward(n).process(&mut buf);
    let bin_hz = rate / n as f64;
    for (m, b) in buf.iter_mut().enumerate() {
        let hz = if m <= n / 2 {
            m as f64 * bin_hz
        } else {
            (n - m) as f64 * bin_hz
        };
        if !(MOD_LO_HZ..=MOD_HI_HZ).contains(&hz) {
            *b = Complex64::new(0.0, 0.0);
        }
    }
    planner.plan_fft_inverse(n).process(&mut buf);
    buf.iter().map(|c| c.re / n as f64).collect()
}

/// `x` with a least-squares polynomial of `degree` in time removed.
fn detrended(x: &[f64], degree: usize) -> Vec<f64> {
    let n = x.len();
    let cols = degree + 1;
    let u = |j: usize| 2.0 * j as f64 / (n - 1).max(1) as f64 - 1.0;
    let mut a = vec![0.0f64; cols * cols];
    let mut b = vec![0.0f64; cols];
    let mut p = vec![1.0f64; cols];
    for (j, &value) in x.iter().enumerate() {
        p[0] = 1.0;
        for c in 1..cols {
            p[c] = p[c - 1] * u(j);
        }
        for r in 0..cols {
            for c in 0..cols {
                a[r * cols + c] += p[r] * p[c];
            }
            b[r] += p[r] * value;
        }
    }
    for c in 0..cols {
        let mut pivot = c;
        for r in c + 1..cols {
            if a[r * cols + c].abs() > a[pivot * cols + c].abs() {
                pivot = r;
            }
        }
        if a[pivot * cols + c].abs() < 1e-12 {
            return x.to_vec();
        }
        for k in 0..cols {
            a.swap(c * cols + k, pivot * cols + k);
        }
        b.swap(c, pivot);
        for r in 0..cols {
            if r == c {
                continue;
            }
            let f = a[r * cols + c] / a[c * cols + c];
            for k in c..cols {
                a[r * cols + k] -= f * a[c * cols + k];
            }
            b[r] -= f * b[c];
        }
    }
    let coeff: Vec<f64> = (0..cols).map(|c| b[c] / a[c * cols + c]).collect();
    (0..n)
        .map(|j| {
            let mut p = 1.0;
            let mut fit = 0.0;
            for &c in &coeff {
                fit += c * p;
                p *= u(j);
            }
            x[j] - fit
        })
        .collect()
}

/// The line-versus-continuum statistics of one modulation spectrum, plus the
/// rate the band's strongest line sits at.
#[derive(Clone, Copy, Default)]
struct ModStats {
    /// Spectral flatness of the band, dB. Zero is a continuum, −20 dB and below
    /// is one or two discrete lines.
    flatness_db: f64,
}

/// Mean rate of an already band-limited, zero-mean track, from its own sign
/// changes: `crossings / (2 * span)`.
///
/// This is the statistic that tests *which* mechanism a beat comes from — a
/// unison mistuning is a frequency **ratio**, so partial `k` beats at `k` times
/// the fundamental's rate; a fixed-hertz split beats at the same rate on every
/// partial; a split from the string's own stiffness anisotropy grows faster
/// than `k` — and it is counted rather than transformed because the transform
/// cannot do it: a 2.7 s window resolves 0.37 Hz, so an FFT puts 0.7 Hz and
/// 1.0 Hz in the same bin and the whole comparison collapses. Counting sign
/// changes has no bin grid. What it costs is that a track with two comparable
/// lines returns something between them rather than either.
fn crossing_rate(x: &[f64], rate: f64) -> f64 {
    if x.len() < 4 {
        return 0.0;
    }
    let crossings = x
        .windows(2)
        .filter(|w| (w[0] < 0.0) != (w[1] < 0.0))
        .count();
    crossings as f64 * rate / (2.0 * (x.len() - 1) as f64)
}

/// The modulation statistics of an already-detrended track.
fn mod_stats(x: &[f64], rate: f64) -> ModStats {
    let n = x.len();
    if n < 16 {
        return ModStats::default();
    }
    let window: Vec<f64> = (0..n)
        .map(|j| 0.5 - 0.5 * (TAU * j as f64 / n as f64).cos())
        .collect();
    let mut buf: Vec<Complex64> = x
        .iter()
        .zip(&window)
        .map(|(&v, &w)| Complex64::new(v * w, 0.0))
        .collect();
    FftPlanner::<f64>::new()
        .plan_fft_forward(n)
        .process(&mut buf);
    let bin_hz = rate / n as f64;
    let lo = (MOD_LO_HZ / bin_hz).ceil().max(1.0) as usize;
    let hi = ((MOD_HI_HZ / bin_hz).floor() as usize).min(n / 2 - 1);
    if hi <= lo {
        return ModStats::default();
    }
    let power: Vec<f64> = (lo..=hi).map(|m| buf[m].norm_sqr()).collect();
    let total: f64 = power.iter().sum();
    if total.is_nan() || total <= 0.0 {
        return ModStats::default();
    }
    let mean = total / power.len() as f64;
    let log_mean = power.iter().map(|p| p.max(1e-300).ln()).sum::<f64>() / power.len() as f64;
    ModStats {
        flatness_db: 10.0 * (log_mean.exp() / mean).log10(),
    }
}

// ------------------------------------------------------------------- output

/// One thing to be measured and listened to.
struct Signal {
    label: String,
    stereo: Stereo,
    mono: Vec<f64>,
}

impl Signal {
    fn new(label: impl Into<String>, stereo: Stereo) -> Signal {
        let n = stereo.0.len().min(stereo.1.len());
        let mono = (0..n)
            .map(|i| 0.5 * (f64::from(stereo.0[i]) + f64::from(stereo.1[i])))
            .collect();
        Signal {
            label: label.into(),
            stereo,
            mono,
        }
    }
}

fn rms(left: &[f32], right: &[f32], from: usize, to: usize) -> f64 {
    let to = to.min(left.len()).min(right.len());
    if to <= from {
        return 0.0;
    }
    let sum: f64 = (from..to)
        .map(|i| f64::from(left[i]).powi(2) + f64::from(right[i]).powi(2))
        .sum();
    (sum / (2 * (to - from)) as f64).sqrt()
}

fn match_rms(audio: &Stereo) -> f64 {
    rms(
        &audio.0,
        &audio.1,
        (MATCH_LO_S * SR) as usize,
        (MATCH_HI_S * SR) as usize,
    )
}

/// Writes a listening set: every signal matched to the first one's level over
/// [`MATCH_LO_S`]–[`MATCH_HI_S`], then one common gain if anything would clip.
/// `JITTER.md`'s convention exactly, so the new files sit beside the old ones at
/// the same level.
fn write_set(dir: &Path, signals: &[Signal]) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let reference = match_rms(&signals[0].stereo);
    if reference.is_nan() || reference <= 0.0 {
        return Err(format!("{}: the reference is silent", dir.display()).into());
    }
    let gains: Vec<f64> = signals
        .iter()
        .map(|s| {
            let level = match_rms(&s.stereo);
            if level > 0.0 {
                reference / level
            } else {
                0.0
            }
        })
        .collect();
    let peak = signals
        .iter()
        .zip(&gains)
        .map(|(s, &g)| {
            s.stereo
                .0
                .iter()
                .chain(s.stereo.1.iter())
                .fold(0.0f64, |m, &v| m.max(f64::from(v).abs()))
                * g
        })
        .fold(0.0f64, f64::max);
    let common = if peak > 0.891 { 0.891 / peak } else { 1.0 };
    let mut applied = Vec::with_capacity(signals.len());
    for (i, (signal, &gain)) in signals.iter().zip(&gains).enumerate() {
        let path = dir.join(format!("{i:02}_{}.wav", signal.label));
        write_wav(&path, &signal.stereo, gain * common)?;
        applied.push(gain * common);
    }
    Ok(applied)
}

fn write_wav(path: &Path, (left, right): &Stereo, gain: f64) -> Result<(), hound::Error> {
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let frames = (NOTE_S * SR) as usize;
    let fade_in = (FADE_IN_S * SR) as usize;
    let fade_out = (FADE_OUT_S * SR) as usize;
    let mut writer = hound::WavWriter::create(path, spec)?;
    for n in 0..frames {
        let mut envelope = gain;
        if n < fade_in {
            envelope *= 0.5 - 0.5 * (std::f64::consts::PI * n as f64 / fade_in as f64).cos();
        }
        if n + fade_out > frames {
            let u = (n + fade_out - frames) as f64 / fade_out as f64;
            envelope *= 0.5 + 0.5 * (std::f64::consts::PI * u).cos();
        }
        let at = |c: &[f32]| f64::from(c.get(n).copied().unwrap_or(0.0)) * envelope;
        writer.write_sample(at(left) as f32)?;
        writer.write_sample(at(right) as f32)?;
    }
    writer.finalize()
}

/// A `| a | b | c |` row from a label and a formatted cell per partial.
fn row(label: &str, cells: &[String]) -> String {
    format!("| {label} | {} |", cells.join(" | "))
}

fn cell(stats: &[Option<PartialStats>], get: impl Fn(&PartialStats) -> f64) -> Vec<String> {
    stats
        .iter()
        .map(|s| s.as_ref().map_or("-".into(), |s| format!("{:.2}", get(s))))
        .collect()
}

/// Geometric mean of a list of positive ratios; `NaN` if it is empty.
fn geometric_mean(values: &[f64]) -> f64 {
    let usable: Vec<f64> = values.iter().copied().filter(|v| *v > 0.0).collect();
    if usable.is_empty() {
        return f64::NAN;
    }
    (usable.iter().map(|v| v.ln()).sum::<f64>() / usable.len() as f64).exp()
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        f64::NAN
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

// ---------------------------------------------------------------------- main

/// Index of a signal inside a key's signal list, so the summary tables can pick
/// rows out by name without matching strings.
const RECORDING: usize = 0;
const ENGINE: usize = 1;
const MODAL_SHIPPED: usize = 2;
const EIGENMODE: usize = 3;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let root = PathBuf::from(args.next().unwrap_or_else(|| "data/salamander".into()));
    let preset_path = args
        .next()
        .unwrap_or_else(|| "presets/salamander-c5.toml".into());
    let out = PathBuf::from(args.next().unwrap_or_else(|| "renders/jitter".into()));

    let library = SampleLibrary::from_sfz(root.join("SalamanderGrandPiano-V3+20200602.sfz"))?;
    let preset = EnginePreset::load(Path::new(&preset_path))?;
    preset.validate()?;
    let set_dir = out.join("eigenmode");
    std::fs::create_dir_all(&set_dir)?;

    let voicing = &preset.voicing;
    let mut planner = FftPlanner::<f64>::new();
    let mut report = String::new();
    writeln!(
        report,
        "# EIGENMODE.md - the coupled-eigenmode unison, prototyped and measured\n\n\
         Written by `cargo run --release -p forensics --bin eigenmode_prototype`. Preset \
         `{preset_path}`, velocity {VELOCITY} unless a row says otherwise. `engine/` is not \
         touched: `03_eigenmode` is an offline modal renderer with `docs/history/FUNDAMENTALS.md` §5's \
         construction in it, and `02_modal_shipped` is the *same* renderer with \
         `PianoString::new`'s construction in it, so the difference between those two rows \
         is the change under test and nothing else. `01_engine` is the shipped instrument \
         through its public API, and `00_recording` is the pass bar.\n\n\
         Every statistic is `renders/jitter/JITTER.md`'s, on the same code, over \
         {T0_S}-{T1_S} s of the mono sum, so the two files' numbers compare directly.\n\n\
         ## The construction, as built\n\n\
         | quantity | value | where it comes from |\n|:--|--:|:--|\n\
         | `Re Y_h / Re Y_v` | {:.4} | `voicing.horizontal_decay_ratio`, re-read as a bridge \
         property (§5.4) |\n\
         | `radiated_share` | **{:.4}** | **derived**, `= 1 - Re Y_h/Re Y_v`: the slowest \
         mode radiates nothing and decays at `(1-share) sigma_k`, so `1-share` *is* the \
         fitted aftersound/prompt ratio. This is §2.6's contradiction resolved; the shipped \
         0.5 would cap the aftersound at half the prompt rate. |\n\
         | `Im Y / Re Y` | {REACTIVE_RATIO:.2} | Weinreich / Capleton §III.B, order one |\n\
         | reactive anisotropy | {REACTIVE_ANISOTROPY:.3} | Capleton's 1 : 0.925 example; \
         **replaces `voicing.horizontal_offset_hz` outright** |\n\
         | horizontal leak x radiation | {:.5} ({:.2} dB) | `voicing.horizontal_gain_db`: \
         only the product enters a block-diagonal `C`, so the construction reduces to the \
         engine's at zero coupling |\n\n\
         Not read at all: `voicing.horizontal_offset_hz`, `voicing.unison_coupling`, \
         `voicing.unison_sigma_scale`, `Voicing::vertical_decay_factor`. Read unchanged: \
         `notes.detune_cents`, `voicing.unison_layout` (detune and share), \
         `notes.partial_gains`, `notes.partial_sigma_scale`, the strike comb, the contact \
         taper, the bridge's per-partial `Re Y` fluctuation, the hammer, the soundboard and \
         the panning.\n",
        resistive_anisotropy(voicing),
        radiated_share(voicing),
        horizontal_leak(voicing),
        voicing.horizontal_gain_db,
    )?;

    // Column A / Column B accumulators, over every key x partial cell.
    let mut a_engine: Vec<f64> = Vec::new();
    let mut a_eigen: Vec<f64> = Vec::new();
    let mut place_engine: Vec<f64> = Vec::new();
    let mut place_eigen: Vec<f64> = Vec::new();
    let mut place_rec: Vec<f64> = Vec::new();
    let mut depth_engine: Vec<f64> = Vec::new();
    let mut depth_eigen: Vec<f64> = Vec::new();
    let mut vel_engine: Vec<f64> = Vec::new();
    let mut vel_eigen: Vec<f64> = Vec::new();
    let mut vel_rec: Vec<f64> = Vec::new();
    // The recording's own statistics per key, kept so the sensitivity sweep
    // below can score against them without re-reading and re-transforming the
    // library four more times.
    let mut reference: std::collections::HashMap<&str, Vec<Option<PartialStats>>> =
        std::collections::HashMap::new();

    for (key, name) in KEYS {
        let params = preset.string_params(key);
        let f0 = f64::from(params.partial_freq(1));
        // Timed, because §5.3 costs the eigensolve out as a preset-load job and
        // the claim needs a number rather than an estimate. Nothing here depends
        // on velocity, so this is once per key for the life of the instrument.
        let built = std::time::Instant::now();
        let cached = EigenKey::new(&preset, key, REACTIVE_RATIO);
        let build_ms = built.elapsed().as_secs_f64() * 1e3;

        // The order the `RECORDING` / `ENGINE` / `MODAL_SHIPPED` / `EIGENMODE`
        // indices name, and the order the files are numbered in.
        let mut signals: Vec<Signal> = vec![
            Signal::new("recording", recording(layer_for(&library, key, VELOCITY)?)?),
            Signal::new("engine", render_engine(&preset, key, VELOCITY)),
            Signal::new(
                "modal_shipped",
                render_shipped_modal(&preset, key, VELOCITY),
            ),
            Signal::new("eigenmode", render_eigen(&preset, &cached, key, VELOCITY)),
        ];
        for vel in EXTRA_VELOCITIES {
            signals.push(Signal::new(
                format!("eigenmode_vel{vel:03}"),
                render_eigen(&preset, &cached, key, vel),
            ));
        }
        for vel in EXTRA_VELOCITIES {
            signals.push(Signal::new(
                format!("engine_vel{vel:03}"),
                render_engine(&preset, key, vel),
            ));
        }
        for vel in EXTRA_VELOCITIES {
            signals.push(Signal::new(
                format!("recording_vel{vel:03}"),
                recording(layer_for(&library, key, vel)?)?,
            ));
        }

        let dir = set_dir.join(name);
        std::fs::create_dir_all(&dir)?;
        let gains = write_set(&dir, &signals)?;
        // The file the task asks for, at the same gain it has inside the set.
        write_wav(
            &out.join(format!("eigenmode_{name}.wav")),
            &signals[EIGENMODE].stereo,
            gains[EIGENMODE],
        )?;

        // ---- what the construction produced, before anything is measured.
        writeln!(
            report,
            "\n## {name} (key {key}, {} string{}, detune {:.3} cents)\n",
            cached.unison,
            if cached.unison == 1 { "" } else { "s" },
            params.detune_cents,
        )?;
        writeln!(
            report,
            "**The eigenmodes the construction builds**, partial by partial, sorted by \
             radiated amplitude. `df` is against the nominal partial frequency, `T60` is \
             that mode alone, `dB` is against the loudest mode of the same partial, and \
             `scale` is the factor [`decay_scale`] solved for so the composite reaches \
             -60 dB on the fitted anchor.\n\n\
             | k | scale | mode | df Hz | sigma | T60 s | dB | plane |\n\
             |--:|--:|--:|--:|--:|--:|--:|:--|"
        )?;
        let _ = build_ms;
        for k in 1..=MAX_PARTIAL.min(cached.partials.len()) {
            let mut modes = cached.partials[k - 1].clone();
            modes.sort_by(|a, b| {
                b.gain
                    .norm()
                    .partial_cmp(&a.gain.norm())
                    .expect("gains are finite")
            });
            let nominal = f64::from(params.partial_freq(k));
            let peak = modes[0].gain.norm().max(f64::MIN_POSITIVE);
            for (i, m) in modes.iter().enumerate() {
                writeln!(
                    report,
                    "| {} | {} | {} | {:+.4} | {:.4} | {:.2} | {:+.1} | {} |",
                    if i == 0 { k.to_string() } else { String::new() },
                    if i == 0 {
                        format!("{:.3}", cached.scales[k - 1])
                    } else {
                        String::new()
                    },
                    i + 1,
                    m.hz() - nominal,
                    m.sigma(),
                    6.91 / m.sigma().max(1e-9),
                    20.0 * (m.gain.norm() / peak).log10(),
                    if m.horizontal { "h" } else { "v" },
                )?;
            }
        }

        // ---- the measurement.
        let measured: Vec<Vec<Option<PartialStats>>> = signals
            .iter()
            .map(|signal| {
                let spectrum = Spectrum::new(&signal.mono, &mut planner);
                (1..=MAX_PARTIAL)
                    .map(|k| {
                        let nominal = f64::from(params.partial_freq(k));
                        track_partial(&spectrum, nominal, 0.35 * f0, &mut planner)
                            .map(|t| statistics(&t))
                    })
                    .collect()
            })
            .collect();

        for (title, note, get) in [
            (
                "Frequency jitter",
                "RMS of the instantaneous-frequency deviation inside 0.1-20 Hz, in cents. \
                 The recording's row is the pass bar - in **both** directions: too still is \
                 as wrong as too wobbly.",
                (|s: &PartialStats| s.band_cents) as fn(&PartialStats) -> f64,
            ),
            (
                "Where the jitter sits (wRMS / raw)",
                "the power-weighted deviation over the plain one. About 1 is a wobble that \
                 rides the partial while it is loud; a small fraction is a spike at the null \
                 of a beat.",
                |s: &PartialStats| s.placement(),
            ),
            (
                "Beat depth",
                "peak-to-trough span of the log envelope inside the same band, dB.",
                |s: &PartialStats| s.beat_depth_db,
            ),
            (
                "Envelope flatness",
                "spectral flatness of the log envelope's modulation spectrum, dB. Near 0 is \
                 a continuum; -20 dB and below is one or two discrete lines.",
                |s: &PartialStats| s.amp_flatness_db,
            ),
            (
                "Frequency-track flatness",
                "the same question asked of the frequency track, dB. Read it beside the beat \
                 depth: a deep periodic beat spikes the track at every null and a train of \
                 spikes is broadband however regular it is.",
                |s: &PartialStats| s.freq_flatness_db,
            ),
            (
                "AM-FM correlation",
                "Pearson r between the band-limited frequency track and the band-limited log \
                 envelope. A beat null drives the frequency hardest where the amplitude is \
                 *lowest*; a frequency that follows the string's own amplitude does the \
                 opposite. This is the column the negative result is read off.",
                |s: &PartialStats| s.am_fm_r,
            ),
            (
                "AM-FM slope",
                "the same regression, in cents of pitch per dB of envelope.",
                |s: &PartialStats| s.am_fm_cents_per_db,
            ),
        ] {
            writeln!(
                report,
                "\n**{name} - {title}.** {note}\n\n| signal | k=1 | k=2 | k=3 | k=4 |\n\
                 |:--|--:|--:|--:|--:|"
            )?;
            for (signal, stats) in signals.iter().zip(&measured) {
                writeln!(report, "{}", row(&signal.label, &cell(stats, get)))?;
            }
        }

        // ---- which mechanism the beat comes from, read off its rate against k.
        writeln!(
            report,
            "\n**{name} - the dominant beat rate, and what it scales with.** The strongest \
             line of the log envelope's modulation spectrum, in Hz, and the same divided by \
             the partial number. A beat from a **unison mistuning** is a frequency ratio, so \
             its rate is proportional to `k` and `rate/k` is a constant; a beat from a fixed \
             *hertz* offset has the same rate on every partial and `rate/k` falls like `1/k`; \
             a beat from the string's own stiffness anisotropy grows faster than `k`. \
             Counted from the band-limited log envelope's sign changes, not from a \
             transform, because the {:.2} Hz a {:.1} s window resolves is coarser than the \
             difference under test.\n\n\
             | signal | k=1 | k=2 | k=3 | k=4 | rate/1 | rate/2 | rate/3 | rate/4 |\n\
             |:--|--:|--:|--:|--:|--:|--:|--:|--:|",
            1.0 / (T1_S - T0_S),
            T1_S - T0_S
        )?;
        for idx in [RECORDING, ENGINE, MODAL_SHIPPED, EIGENMODE] {
            let rate = |k: usize| measured[idx][k - 1].as_ref().map(|s| s.beat_rate_hz);
            let cells: Vec<String> = (1..=MAX_PARTIAL)
                .map(|k| rate(k).map_or("-".into(), |v| format!("{v:.2}")))
                .chain(
                    (1..=MAX_PARTIAL)
                        .map(|k| rate(k).map_or("-".into(), |v| format!("{:.2}", v / k as f64))),
                )
                .collect();
            writeln!(report, "{}", row(&signals[idx].label, &cells))?;
        }

        // ---- and the pair that envelope implies, which is the concrete target.
        writeln!(
            report,
            "\n**{name} - the companion each row implies.** Inverting the beat depth for the \
             amplitude ratio of a two-component pair (`D = 20 log10((1+r)/(1-r))`) and \
             quoting `r` in dB, beside the offset the rate above implies. This is what a \
             two-component model would have to contain to produce the measured envelope: \
             *how loud* the second component is and *how far away*. It is the cleanest \
             statement of the gap, because it is in the units the preset is written in.\n\n\
             | signal | k=1 dB / Hz | k=2 dB / Hz | k=3 dB / Hz | k=4 dB / Hz |\n\
             |:--|--:|--:|--:|--:|"
        )?;
        for idx in [RECORDING, ENGINE, MODAL_SHIPPED, EIGENMODE] {
            let cells: Vec<String> = (1..=MAX_PARTIAL)
                .map(|k| {
                    measured[idx][k - 1].as_ref().map_or("-".into(), |s| {
                        let x = 10f64.powf(s.beat_depth_db / 20.0);
                        let r = ((x - 1.0) / (x + 1.0)).clamp(1e-6, 1.0);
                        format!("{:.1} / {:.2}", 20.0 * r.log10(), s.beat_rate_hz)
                    })
                })
                .collect();
            writeln!(report, "{}", row(&signals[idx].label, &cells))?;
        }

        // ---- the full reading, partial by partial, for the four headline rows.
        writeln!(
            report,
            "\n**{name} - the full reading.** `S/N` is the partial's own bin over the median \
             bin of its neighbourhood, so a large `raw` beside a small `S/N` is a measurement \
             of the background and not of the partial; `exc/s` counts separate swings past \
             ±{EXCURSION_CENTS} cents per second.\n\n\
             | signal | k | mean Hz | S/N dB | cents | p95 | exc/s | raw | wRMS |\n\
             |:--|--:|--:|--:|--:|--:|--:|--:|--:|"
        )?;
        for idx in [RECORDING, ENGINE, MODAL_SHIPPED, EIGENMODE] {
            for k in 1..=MAX_PARTIAL {
                if let Some(s) = &measured[idx][k - 1] {
                    writeln!(
                        report,
                        "| {} | {k} | {:.2} | {:.0} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} |",
                        signals[idx].label,
                        s.mean_hz,
                        s.peak_db,
                        s.band_cents,
                        s.p95_cents,
                        s.excursions_per_s,
                        s.raw_cents,
                        s.weighted_cents,
                    )?;
                }
            }
        }

        writeln!(
            report,
            "\n**{name} - the double decay.** A straight line through the partial's log \
             envelope over {PROMPT_LO_S}-{PROMPT_HI_S} s (`prompt`) and again over \
             {TAIL_LO_S}-{TAIL_HI_S} s (`tail`), both in dB/s, and where the tail's line \
             extrapolates back to at the strike relative to the prompt's (`after`, dB - the \
             aftersound level). The construction must keep this: a unison that stops \
             beating but also stops having a tail has not solved anything.\n\n\
             | signal | k | prompt dB/s | tail dB/s | ratio | after dB |\n\
             |:--|--:|--:|--:|--:|--:|"
        )?;
        for idx in [RECORDING, ENGINE, MODAL_SHIPPED, EIGENMODE] {
            for k in 1..=MAX_PARTIAL {
                if let Some(s) = &measured[idx][k - 1] {
                    writeln!(
                        report,
                        "| {} | {k} | {:.2} | {:.2} | {:.2} | {:.1} |",
                        signals[idx].label,
                        s.prompt_db_s,
                        s.tail_db_s,
                        if s.tail_db_s.abs() > 1e-6 {
                            s.prompt_db_s / s.tail_db_s
                        } else {
                            f64::NAN
                        },
                        s.aftersound_db,
                    )?;
                }
            }
        }

        // ---- velocity invariance, the property no fixed-offset sum can have.
        writeln!(
            report,
            "\n**{name} - velocity spread.** The range of each cell over velocities \
             {:?} and {VELOCITY}: `cents` is the spread of the frequency jitter, `depth` \
             the spread of the beat depth. `docs/history/FUNDAMENTALS.md` §4.1 property 4 says a \
             fixed-offset sum must read zero here; the recording does not.\n\n\
             | signal | k=1 c | k=2 c | k=3 c | k=4 c | k=1 dB | k=2 dB | k=3 dB | k=4 dB |\n\
             |:--|--:|--:|--:|--:|--:|--:|--:|--:|",
            EXTRA_VELOCITIES
        )?;
        let velocity_sets: [(&str, [usize; 3]); 3] = [
            ("recording", [RECORDING, 8, 9]),
            ("engine", [ENGINE, 6, 7]),
            ("eigenmode", [EIGENMODE, 4, 5]),
        ];
        for (label, indices) in velocity_sets {
            let spread = |get: fn(&PartialStats) -> f64, k: usize| -> Option<f64> {
                let values: Vec<f64> = indices
                    .iter()
                    .filter_map(|&i| measured[i][k - 1].as_ref().map(get))
                    .collect();
                if values.len() < indices.len() {
                    return None;
                }
                let hi = values.iter().copied().fold(f64::MIN, f64::max);
                let lo = values.iter().copied().fold(f64::MAX, f64::min);
                Some(hi - lo)
            };
            let cents: Vec<Option<f64>> = (1..=MAX_PARTIAL)
                .map(|k| spread(|s| s.band_cents, k))
                .collect();
            let depth: Vec<Option<f64>> = (1..=MAX_PARTIAL)
                .map(|k| spread(|s| s.beat_depth_db, k))
                .collect();
            let cells: Vec<String> = cents
                .iter()
                .chain(&depth)
                .map(|v| v.map_or("-".into(), |v| format!("{v:.3}")))
                .collect();
            writeln!(report, "{}", row(label, &cells))?;
            let sink = match label {
                "recording" => &mut vel_rec,
                "engine" => &mut vel_engine,
                _ => &mut vel_eigen,
            };
            sink.extend(cents.iter().flatten());
        }

        // ---- feed the two scoreboard columns.
        for k in 1..=MAX_PARTIAL {
            let (rec, eng, eig) = (
                measured[RECORDING][k - 1].as_ref(),
                measured[ENGINE][k - 1].as_ref(),
                measured[EIGENMODE][k - 1].as_ref(),
            );
            if let (Some(r), Some(e)) = (rec, eng) {
                a_engine.push(ratio(r.band_cents, e.band_cents));
                depth_engine.push((r.beat_depth_db - e.beat_depth_db).abs());
                place_engine.push(e.placement());
                place_rec.push(r.placement());
            }
            if let (Some(r), Some(e)) = (rec, eig) {
                a_eigen.push(ratio(r.band_cents, e.band_cents));
                depth_eigen.push((r.beat_depth_db - e.beat_depth_db).abs());
                place_eigen.push(e.placement());
            }
        }

        reference.insert(
            name,
            measured.into_iter().next().expect("the recording is row 0"),
        );
        println!(
            "{name}: {} signals to {} ({} partials x {} modes solved in {build_ms:.1} ms)",
            signals.len(),
            dir.display(),
            cached.partials.len(),
            2 * cached.unison,
        );
    }

    // ---- the one constant the literature does not pin, swept.
    writeln!(
        report,
        "\n## Sensitivity to `Im Y / Re Y`\n\n\
         The whole construction re-solved and re-rendered at each value of the bridge's \
         reactive-to-resistive ratio in {REACTIVE_SWEEP:?}. It is the constant that decides \
         whether the coupling pulls the group's frequencies **together** (Woodhouse's \
         anti-veering, resistive-dominated, no beats) or **apart** (veering, beats survive), \
         and no source pins it, so the verdict has to hold across it. `A1` is the \
         instantaneous-frequency mismatch against the recording over all \
         {} cells - lower is closer, and the shipped engine scores {:.2}.\n\n\
         | Im Y / Re Y | A1 | C4 k=1 c | C4 k=2 c | A2 k=1 c | C6 k=1 c | C4 k=2 depth dB \
         | C4 k=3 depth dB |\n|--:|--:|--:|--:|--:|--:|--:|--:|",
        a_engine.len(),
        geometric_mean(&a_engine),
    )?;
    for beta in REACTIVE_SWEEP {
        let mut mismatch: Vec<f64> = Vec::new();
        let mut cents: Vec<String> = Vec::new();
        let mut depth: Vec<String> = Vec::new();
        for (key, name) in KEYS {
            let params = preset.string_params(key);
            let f0 = f64::from(params.partial_freq(1));
            let cached = EigenKey::new(&preset, key, beta);
            let signal = Signal::new("sweep", render_eigen(&preset, &cached, key, VELOCITY));
            let spectrum = Spectrum::new(&signal.mono, &mut planner);
            for k in 1..=MAX_PARTIAL {
                let nominal = f64::from(params.partial_freq(k));
                let here = track_partial(&spectrum, nominal, 0.35 * f0, &mut planner)
                    .map(|t| statistics(&t));
                if let (Some(a), Some(b)) = (&here, &reference[name][k - 1]) {
                    mismatch.push(ratio(a.band_cents, b.band_cents));
                }
                if matches!((name, k), ("C4", 1) | ("C4", 2) | ("A2", 1) | ("C6", 1)) {
                    cents.push(
                        here.as_ref()
                            .map_or("-".into(), |s| format!("{:.2}", s.band_cents)),
                    );
                }
                if name == "C4" && (k == 2 || k == 3) {
                    depth.push(
                        here.as_ref()
                            .map_or("-".into(), |s| format!("{:.2}", s.beat_depth_db)),
                    );
                }
            }
        }
        let cells: Vec<String> = std::iter::once(format!("{:.2}", geometric_mean(&mismatch)))
            .chain(cents)
            .chain(depth)
            .collect();
        writeln!(report, "{}", row(&format!("{beta:.2}"), &cells))?;
    }

    writeln!(
        report,
        "\n## The two scoreboard columns, over every key x partial cell\n\n\
         The perception review's Column A (instantaneous-frequency mismatch and placement) \
         and Column B (beat-depth error and velocity coherence), computed here on \
         {} cells.\n\n\
         | statistic | gate | engine | eigenmode | recording |\n|:--|--:|--:|--:|--:|\n\
         | A1 IF mismatch (geo-mean max/min per cell) | < 2.0 | {:.2} | **{:.2}** | 1.00 |\n\
         | A2 placement `wRMS/raw` (mean) | > 0.5 | {:.2} | **{:.2}** | {:.2} |\n\
         | B1 beat-depth error (mean abs dB) | < 3.0 | {:.2} | **{:.2}** | 0.00 |\n\
         | B2 velocity spread of jitter (mean cents) | > 0.25 x reference | {:.3} | \
         **{:.3}** | {:.3} |\n",
        a_engine.len(),
        geometric_mean(&a_engine),
        geometric_mean(&a_eigen),
        mean(&place_engine),
        mean(&place_eigen),
        mean(&place_rec),
        mean(&depth_engine),
        mean(&depth_eigen),
        mean(&vel_engine),
        mean(&vel_eigen),
        mean(&vel_rec),
    )?;

    writeln!(
        report,
        "\n## The listening set\n\n\
         `{}/<note>/NN_<label>.wav`, {NOTE_S} s, stereo, every file matched to the \
         recording's RMS over {MATCH_LO_S}-{MATCH_HI_S} s with one common headroom gain over \
         the set - `JITTER.md`'s convention, so these sit at the level the existing set \
         does. `{}/eigenmode_<note>.wav` is `03_eigenmode` again, at the same gain, where \
         the task asked for it.\n\n\
         - `00_recording` - the Salamander layer, the pass bar\n\
         - `01_engine` - the shipped instrument, public API\n\
         - `02_modal_shipped` - the offline renderer with `PianoString::new`'s construction: \
         the control that separates the eigenproblem from the signal path\n\
         - `03_eigenmode` - the same renderer with `docs/history/FUNDAMENTALS.md` §5's construction\n\
         - `04`/`05` - the eigen construction at velocities {:?}\n\
         - `06`/`07` - the shipped engine at the same velocities\n\
         - `08`/`09` - the recording's own layers at the same velocities\n",
        set_dir.display(),
        out.display(),
        EXTRA_VELOCITIES
    )?;

    let path = out.join("EIGENMODE.md");
    std::fs::write(&path, &report)?;
    println!("wrote {}", path.display());
    Ok(())
}

/// The larger of two positive numbers over the smaller — a symmetric mismatch
/// that reads 1 when they agree and does not care which way round they are.
fn ratio(a: f64, b: f64) -> f64 {
    let (a, b) = (a.abs().max(1e-6), b.abs().max(1e-6));
    if a > b {
        a / b
    } else {
        b / a
    }
}
