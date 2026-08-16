//! Two-exponential decay fitting: the double decay of a piano partial, and the
//! smooth `sigma(f)` curve the engine's string model wants.
//!
//! A struck string radiates in two polarizations. The vertical one is coupled
//! hard to the bridge and dies fast; the horizontal one starts ~12 dB down and
//! outlives it several times over. What a partial's envelope therefore looks
//! like is not one exponential but two,
//!
//! ```text
//!     a(t) = A_f exp(-sigma_f t) + A_s exp(-sigma_s t),      sigma_f > sigma_s,
//! ```
//! the "prompt" and "aftersound" of the piano literature. Fitting a single
//! exponential to it gives an answer that depends entirely on which part of the
//! tail was fitted, which is why the engine's parameters have never been
//! measurable before this.
//!
//! The fit is done on `ln a`, not on `a`. A decay spans 60 dB and a
//! least-squares fit in linear amplitude would be a fit to its first 10 dB;
//! equal weight per decibel is what "fits the decay" means.
//!
//! # The two beats
//!
//! That sum of two decaying exponentials is what a partial's envelope would be
//! if everything radiating at that partial sat at exactly one frequency. Nothing
//! on a piano does. The two polarizations of one string see slightly different
//! transverse stiffness and stand a fraction of a hertz apart; the two or three
//! strings of a unison stand a few tenths of a hertz apart by design. What the
//! tracker measures at partial `k` is therefore the *modulus of a sum* of
//! several close components, and it beats.
//!
//! Both beats have the same origin and different algebra. Writing the unison
//! strings' shares as `s_i` at offsets `d_i`, and giving every string the same
//! pair of polarizations (gain `h`, rate `rho sigma`, offset `o`), the sum
//! factorizes exactly:
//!
//! ```text
//!   a(t) = |sum_i s_i e^{i 2 pi d_i t}| * |A_f e^{-sigma_f t}
//!                                          + A_s e^{-sigma_s t} e^{i 2 pi o t}|
//!          \_______ unison modulation _______/ \____ polarization beat ____/
//! ```
//!
//! The unison factor multiplies the whole envelope and leaves its shape alone;
//! the polarization beat does not, because it beats the two *decays* against
//! each other and its depth therefore changes as the fast component dies. This
//! is why it has to be in the model rather than smoothed away: fitted without
//! it, a single-strung A1 rendered by the engine returns a T60 16 % short and a
//! three-string C4 anything between 35 % short and 29 % long, the error being
//! whatever the beat nulls happened to do inside the record (`DECISIONS.md`
//! item 81).
//!
//! The fitted envelope is reported *without* its modulation — `amplitude_at` is
//! the coherent sum, which is what the strike put into the partial at `t = 0`
//! and what the engine's `sigma` tables mean by a T60. The modulation is
//! reported alongside in [`EnvelopeBeats`], where the unison estimator reads the
//! beat rate it needs.

use crate::error::{Error, Result};
use crate::estimate::{level_floor, FitSpan};
use crate::numeric::{median, weighted_least_squares, NelderMead};
use crate::trajectory::{NoteTrajectories, PartialTrack};

/// `2 pi`, in the one place a beat's phase advance is written.
const TAU: f64 = std::f64::consts::TAU;

/// `20 / ln 10`: turns an RMS of natural-log residuals into decibels.
const NEPERS_TO_DB: f64 = 8.685_889_638_065_035;

/// Amplitude ratio corresponding to -60 dB, in natural logarithms: the constant
/// that turns a decay rate into a T60 and back.
pub const LN_1000: f64 = 6.907_755_278_982_137;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecayConfig {
    /// Fewest measurements a track needs before its envelope is fitted.
    pub min_points: usize,
    /// How far below the loudest partial a track may be and still be fitted, in
    /// dB. Above the note's highest real partial the tracker is following the
    /// noise floor, and a decay fitted to noise is a number with no meaning
    /// that the `sigma(f)` fit would nonetheless average in.
    pub min_level_db: f64,
    /// Points more than this far below the track's own peak are dropped: below
    /// the noise floor an envelope measurement is a measurement of the noise.
    pub range_db: f64,
    /// Budget for the nonlinear fit, in objective evaluations.
    pub max_evaluations: usize,
    /// Floor on either decay rate, 1/s. Zero would be an undamped partial.
    pub min_sigma: f64,
    /// A fitted slow component below this fraction of the total initial
    /// amplitude is not a polarization, it is the fit using its fourth
    /// parameter to shave the last decibel off the residual. Such a fit still
    /// describes the envelope, but it does not contribute to the polarization
    /// statistics.
    pub min_split_ratio: f64,
    /// How far past the end of the data a fitted T60 may reach, as a multiple
    /// of the time actually observed, before it stops counting as a
    /// measurement. A three-second recording of a note whose fundamental rings
    /// for twelve has not measured that partial's decay — it has seen the first
    /// few decibels of it and the rest is extrapolation, and one such partial
    /// in a `sigma(f)` fit is worth more than all the others put together. Such
    /// fits are still returned; they are left out of the curve and of the
    /// polarization statistics.
    pub max_t60_ratio: f64,
    /// Fastest beat the envelope model will look for, Hz. A unison is never
    /// mistuned by more than a few cents and two polarizations never by more
    /// than about a hertz, so at the eighth partial of a treble note a couple
    /// of dozen hertz is already generous; anything faster is another partial
    /// leaking into the tracker's window.
    pub max_beat_hz: f64,
    /// Fewest cycles of a beat the record must contain before it is fitted.
    /// Below one cycle a "beat" and a slow drift are the same curve.
    pub min_beat_cycles: f64,
    /// Largest second-string share the unison modulation may take. At 1 the
    /// modulation has exact nulls, `ln a` has poles, and the fit is decided by
    /// whichever measurement came closest to one.
    pub max_unison_depth: f64,
    /// Simplex budget for the beat-aware refinement, per starting point.
    pub beat_evaluations: usize,
    /// How much of the residual the beats have to explain before they are
    /// believed, as a fraction.
    ///
    /// Three more parameters can always shave something off a fit, and what
    /// they shave off a partial that is not beating is the shape of its noise —
    /// which then moves the amplitude the fit extrapolates back to the strike,
    /// and that is what the strike comb and the felt are read from. Measured on
    /// a synthetic tone with no beat in it at all, the beat model still finds a
    /// third of the residual to explain, and the mass fitted from the spectrum
    /// it leaves is out by 70 %. A real beat is not a marginal improvement:
    /// A1's fundamental, rendered by the engine, goes from a 1.4 dB residual to
    /// 1.0 and from a T60 10 % short to 4 %.
    pub min_beat_improvement: f64,
    /// Blocks the envelope is reduced to when looking for the floor of the
    /// recording. Fewer than four switches the search off, which is right for a
    /// signal known to have nothing under it.
    pub floor_blocks: usize,
    /// How much of the record has to lie below the envelope's quietest block,
    /// as a fraction, before that block is read as the floor rather than as the
    /// end of a decay.
    pub floor_tail_fraction: f64,
    /// Signal-to-floor ratio, in dB, below which a measurement is more floor
    /// than partial.
    pub floor_margin_db: f64,
    /// How much louder than the first frame that measured it a fit may claim
    /// the partial was at the strike, in dB.
    ///
    /// A frame is timestamped at the centre of its window, so the first usable
    /// measurement is half a window after the strike and nothing constrains the
    /// model before it. That leaves a degenerate direction: a fast component
    /// steep enough to be spent by the first frame can be given any amplitude
    /// at all, and it costs the residual nothing while multiplying the
    /// extrapolated `a(0)` — and dividing the T60, which is defined 60 dB below
    /// it. Rendered notes have been seen to come back with a tenth of their true
    /// T60 this way. A partial genuinely does lose 25 dB before the first frame
    /// at the top of its note's range, so the limit is loose; it only has to
    /// exclude a claim that rests on no data at all.
    pub max_extrapolation_db: f64,
}

impl Default for DecayConfig {
    fn default() -> Self {
        Self {
            min_points: 8,
            min_level_db: 60.0,
            range_db: 60.0,
            max_evaluations: 1_200,
            min_sigma: 1e-3,
            min_split_ratio: 1e-3,
            max_t60_ratio: 6.0,
            max_beat_hz: 24.0,
            min_beat_cycles: 1.5,
            max_unison_depth: 0.95,
            beat_evaluations: 3_000,
            min_beat_improvement: 0.4,
            floor_blocks: 8,
            floor_tail_fraction: 0.375,
            floor_margin_db: 8.0,
            max_extrapolation_db: 40.0,
        }
    }
}

/// The level at which this envelope stopped being the partial and became the
/// recording it was made in.
///
/// A real recording has a floor under it — the room, the microphone, the rest
/// of the instrument ringing sympathetically — and a partial that has reached
/// it goes on being tracked at a level that no longer falls. Fitting a decay
/// through that measures the floor rather than the string: the slow component
/// latches onto the flat tail and comes back undamped. Salamander's C8, whose
/// partial is gone inside a second, leaves three seconds of halo 25 dB down and
/// was fitted with a T60 of minutes.
///
/// Flatness cannot be the test: the slow half of a double decay is flat-ish
/// too, and cutting *that* off is the opposite mistake. What identifies a floor
/// is where the quietest part of the record is. The quietest part of a decay is
/// its end — that is what "decay" means — so a record whose quietest stretch is
/// followed by a third of a record that never goes below it has stopped
/// decaying, and the level it stopped at is the floor. Salamander's C3 is the
/// clear case: its fundamental falls 55 dB in six seconds and then sits between
/// −48 and −55 dB for another nine, and fitted through that it comes back
/// ringing for twenty seconds instead of six.
///
/// Block medians rather than raw points, so that one beat null does not read as
/// a floor, and a whole third of the record after the minimum rather than a
/// block or two, so that noise in the medians of a slow tail cannot trip it.
///
/// The level returned is the floor itself. Callers subtract it in *power*: a
/// partial and the room are uncorrelated, so what the tracker measures where
/// they overlap is `sqrt(signal^2 + floor^2)`, and taking the floor out of that
/// is a subtraction of squares rather than of amplitudes.
fn recorded_floor(points: &[(f64, f64)], config: &DecayConfig) -> f64 {
    let blocks = config.floor_blocks;
    if blocks < 4 || points.len() < 4 * blocks {
        return 0.0;
    }
    let medians: Vec<f64> = (0..blocks)
        .map(|b| {
            let lo = b * points.len() / blocks;
            let hi = (b + 1) * points.len() / blocks;
            let mut window: Vec<f64> = points[lo..hi].iter().map(|&(_, a)| a).collect();
            window.sort_by(f64::total_cmp);
            window[window.len() / 2]
        })
        .collect();
    let quietest = medians
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .expect("at least four blocks");
    let after = (blocks - 1 - quietest.0) as f64 / blocks as f64;
    if after < config.floor_tail_fraction || *quietest.1 <= 0.0 {
        return 0.0;
    }
    // The floor's *typical* level over the stretch it occupies, not its
    // quietest moment: a room wanders by several decibels, and subtracting its
    // minimum would leave most of it in the measurement.
    let mut tail: Vec<f64> = medians[quietest.0..].to_vec();
    tail.sort_by(f64::total_cmp);
    tail[tail.len() / 2]
}

/// One exponential component of an envelope: `amplitude exp(-sigma t)`, with
/// `t` measured from the strike.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Exponential {
    pub amplitude: f64,
    pub sigma: f64,
}

impl Exponential {
    pub fn amplitude_at(&self, t: f64) -> f64 {
        self.amplitude * (-self.sigma * t).exp()
    }

    /// Time for this component alone to fall 60 dB.
    pub fn t60(&self) -> f64 {
        LN_1000 / self.sigma
    }
}

/// The modulation a partial's envelope carries on top of its decay.
///
/// Both rates are frequency *differences* between things radiating at the same
/// partial: `unison_hz` between two strings of the group, `polarization_hz`
/// between the two polarizations of one string. Zero means the fit found none —
/// a single-strung note has no unison to beat, and a partial measured over less
/// than [`DecayConfig::min_beat_cycles`] of a beat has nothing to fit.
///
/// One unison rate, not one per string: a three-string group beats at three
/// rates at once, and over the two to four cycles a decaying note's record
/// contains, an envelope fit cannot take them apart — offering it the
/// parameters to try only lets it explain the measurement's own noise with
/// them (`DECISIONS.md` item 84). What is fitted is the deepest of the three.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EnvelopeBeats {
    /// Beat rate between the two loudest strings of the unison group, Hz.
    pub unison_hz: f64,
    /// Share of the second string relative to the loudest, in `[0, 1]`: how
    /// deep that modulation is. Zero is one string; one is two equal strings
    /// and a modulation that reaches silence.
    pub unison_depth: f64,
    /// Beat rate between the two polarizations, Hz.
    ///
    /// This beat has no phase of its own: one hammer blow excites both
    /// polarizations of a string at the same instant, so they start together
    /// however late the string was struck.
    pub polarization_hz: f64,
}

impl EnvelopeBeats {
    /// `|1 + d e^{i 2 pi f t}| / (1 + d)`: the group's modulation, normalized
    /// to 1 at the strike where its strings are still in phase.
    pub fn unison_modulation(&self, t: f64) -> f64 {
        unison_modulation(self.unison_hz, self.unison_depth, t)
    }
}

/// The fitted envelope of one partial.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecayFit {
    pub k: u32,
    /// The partial's frequency, carried so a `sigma(f)` fit does not have to go
    /// back to the trajectories for it.
    pub frequency_hz: f64,
    /// The faster-decaying, louder component: the vertical polarization.
    pub fast: Exponential,
    /// The slower, quieter component: the horizontal polarization.
    pub slow: Exponential,
    /// The modulation fitted alongside the decay.
    pub beats: EnvelopeBeats,
    /// RMS of the fit residual, in dB.
    pub residual_db: f64,
    pub points: usize,
    /// Seconds of envelope the fit actually saw.
    pub span_s: f64,
}

impl DecayFit {
    /// The envelope with its beats taken out: what the partial would decay like
    /// if everything radiating at it stood at one frequency. This is the
    /// quantity the engine's `sigma` tables describe, and `amplitude_at(0)` is
    /// the excitation the hammer delivered — at the strike every component is
    /// in phase, so the coherent sum is what the recording actually starts at.
    pub fn amplitude_at(&self, t: f64) -> f64 {
        self.fast.amplitude_at(t) + self.slow.amplitude_at(t)
    }

    /// The envelope as measured: the coherent decay through both beats.
    pub fn modulated_amplitude_at(&self, t: f64) -> f64 {
        polarization_beat(
            self.fast.amplitude_at(t),
            self.slow.amplitude_at(t),
            self.beats.polarization_hz,
            t,
        ) * self.beats.unison_modulation(t)
    }

    /// Amplitude the fit extrapolates back to the strike — the partial's share
    /// of the hammer's excitation, and the input to the strike-position and
    /// hammer estimators.
    ///
    /// The *modulated* envelope at the strike, not the coherent one: the group's
    /// strings do not quite reach full phase agreement even at `t = 0` (the
    /// hammer meets them a few tenths of a millisecond apart), and what the
    /// spectrum estimators want is the amplitude the recording actually starts
    /// at.
    pub fn initial_amplitude(&self) -> f64 {
        self.modulated_amplitude_at(0.0)
    }

    /// Time for the whole partial to fall 60 dB below its extrapolated initial
    /// amplitude. Solved rather than derived: with two components the -60 dB
    /// point has no closed form, and the sum is strictly decreasing so a
    /// bisection is exact to machine precision.
    pub fn t60(&self) -> f64 {
        let target = 1e-3 * self.initial_amplitude();
        if !(target.is_finite() && target > 0.0) {
            return f64::INFINITY;
        }
        // Bracket: the whole initial amplitude decaying at the *slow* rate is
        // an upper bound on the sum, so it reaches the target no earlier.
        let mut hi = if self.slow.amplitude > 0.0 {
            self.slow.t60()
        } else {
            self.fast.t60()
        };
        if !(hi.is_finite() && hi > 0.0) {
            return f64::INFINITY;
        }
        let mut lo = 0.0;
        for _ in 0..100 {
            let mid = 0.5 * (lo + hi);
            if self.amplitude_at(mid) > target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }

    /// The single decay rate that would give this partial the same T60 — the
    /// quantity the engine's `sigma0`/`sigma1` tables hold.
    pub fn effective_sigma(&self) -> f64 {
        LN_1000 / self.t60()
    }

    /// Whether the two components are far enough apart in level to be read as
    /// two polarizations rather than as one decay plus fitting slack.
    pub fn is_split(&self, config: &DecayConfig) -> bool {
        self.slow.amplitude > config.min_split_ratio * self.initial_amplitude()
            && self.slow.sigma < 0.95 * self.fast.sigma
    }

    /// Whether the recording was long enough for this T60 to be a measurement
    /// rather than an extrapolation.
    pub fn is_measured(&self, config: &DecayConfig) -> bool {
        let t60 = self.t60();
        t60.is_finite() && t60 <= config.max_t60_ratio * self.span_s
    }
}

/// How the two polarizations of a note compare: the engine's
/// `horizontal_gain_db` and `horizontal_decay_ratio`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolarizationSplit {
    /// Level of the slow component relative to the fast one, in dB (negative).
    pub gain_db: f64,
    /// `sigma_slow / sigma_fast`, below 1 by construction.
    pub decay_ratio: f64,
    /// Partials the medians were taken over.
    pub partials: usize,
}

/// The engine's damping law `sigma(f) = sigma0 + sigma1 (f/1000)^2`, fitted
/// across a note's partials.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecayCurve {
    pub sigma0: f64,
    pub sigma1: f64,
    /// RMS residual of the fitted rates, 1/s.
    pub residual: f64,
}

impl DecayCurve {
    pub fn sigma_at(&self, hz: f64) -> f64 {
        let khz = hz / 1000.0;
        self.sigma0 + self.sigma1 * khz * khz
    }

    pub fn t60_at(&self, hz: f64) -> f64 {
        LN_1000 / self.sigma_at(hz)
    }
}

/// Everything the decay stage produces for one note.
#[derive(Clone, Debug)]
pub struct DecayReport {
    pub partials: Vec<DecayFit>,
    pub curve: DecayCurve,
    pub polarization: PolarizationSplit,
}

impl DecayReport {
    /// The partials' amplitudes extrapolated back to the strike, ascending in
    /// `k`: what the strike-position and hammer estimators read.
    pub fn excitation_spectrum(&self) -> Vec<(u32, f64)> {
        self.partials
            .iter()
            .map(|fit| (fit.k, fit.initial_amplitude()))
            .collect()
    }

    pub fn fit(&self, k: u32) -> Option<&DecayFit> {
        self.partials.iter().find(|fit| fit.k == k)
    }
}

/// Fits every partial's envelope, then the `sigma(f)` curve and the
/// polarization split across them.
pub fn fit_decays(trajectories: &NoteTrajectories, config: &DecayConfig) -> Result<DecayReport> {
    let span = FitSpan::from_trajectories(trajectories);
    let floor = level_floor(trajectories, config.min_level_db);
    let partials: Vec<DecayFit> = trajectories
        .tracks
        .iter()
        .filter(|track| track.peak().is_some_and(|peak| peak.amplitude >= floor))
        .filter_map(|track| fit_two_exponential(track, span, config).ok())
        .collect();
    if partials.len() < 2 {
        return Err(Error::Estimate(format!(
            "decay fit needs 2 usable partials, got {}",
            partials.len()
        )));
    }
    let curve = fit_decay_curve(&partials, config)?;
    let polarization = polarization_split(&partials, config)?;
    Ok(DecayReport {
        partials,
        curve,
        polarization,
    })
}

/// Fits the beating double decay of the module header to one partial's
/// envelope.
///
/// Two stages, because the beats are the part with local minima in them. The
/// first fits the plain sum `A_f exp(-sigma_f t) + A_s exp(-sigma_s t)`: its
/// two rates enter nonlinearly and its two amplitudes linearly, so it starts
/// from a two-segment log-linear fit (early slope for the fast component, late
/// slope for the slow one) plus a linear solve for the amplitudes, and the
/// simplex refines all four against the log-domain residual. Parameterization
/// is `(ln A_f, ln A_s, ln sigma_s, ln(sigma_f - sigma_s))`, which keeps the
/// amplitudes and rates positive and the ordering `sigma_f > sigma_s` exact
/// rather than penalized.
///
/// The second stage adds the two beats. Their rates are read off the *spectrum*
/// of what the first stage could not explain — a beat is the only periodic
/// thing in a decay's residual — and the simplex then refines all seven
/// parameters together from each plausible assignment of the peaks it found to
/// the two beats. A start is only accepted if it leaves a smaller residual than
/// the beatless fit, so a partial with nothing beating in it keeps the answer
/// this function has always given.
pub fn fit_two_exponential(
    track: &PartialTrack,
    span: FitSpan,
    config: &DecayConfig,
) -> Result<DecayFit> {
    let peak = track
        .peak()
        .ok_or_else(|| Error::Estimate(format!("partial {} has no measurements", track.k)))?;
    let floor = peak.amplitude * 10f64.powf(-config.range_db / 20.0);
    let mut points: Vec<(f64, f64)> = track
        .points
        .iter()
        .filter(|p| p.time_s >= span.start_s && p.amplitude > floor && p.amplitude > 0.0)
        .map(|p| (p.time_s - span.onset_s, p.amplitude))
        .collect();
    // Where the recording has a floor, take it out of the measurements in
    // power and stop at the point where what is left is no longer the partial.
    // Truncation rather than a level filter: a floor wanders, so parts of it
    // stand above its own median, and what the floor marks is the *instant* the
    // record stopped being evidence about this string.
    let recorded_floor = recorded_floor(&points, config);
    if recorded_floor > 0.0 {
        let cut = recorded_floor * 10f64.powf(config.floor_margin_db / 20.0);
        let end = points
            .iter()
            .position(|&(_, a)| a <= cut)
            .unwrap_or(points.len());
        points.truncate(end);
        for point in points.iter_mut() {
            point.1 = (point.1 * point.1 - recorded_floor * recorded_floor)
                .max(0.0)
                .sqrt();
        }
    }
    if points.len() < config.min_points {
        return Err(Error::Estimate(format!(
            "partial {} has {} usable envelope points, need {}",
            track.k,
            points.len(),
            config.min_points
        )));
    }
    let duration = points[points.len() - 1].0 - points[0].0;
    if duration <= 0.0 {
        return Err(Error::Estimate(format!(
            "partial {} spans no time",
            track.k
        )));
    }

    let (mut sigma_fast, mut sigma_slow) = seed_rates(&points, duration, config);
    let (mut amplitude_fast, mut amplitude_slow) =
        seed_amplitudes(&points, sigma_fast, sigma_slow).unwrap_or((peak.amplitude, 0.0));
    if !(amplitude_fast.is_finite() && amplitude_fast > 0.0) {
        amplitude_fast = peak.amplitude;
    }
    if !(amplitude_slow.is_finite() && amplitude_slow > 0.0) {
        amplitude_slow = 1e-3 * amplitude_fast;
    }
    if sigma_slow < config.min_sigma {
        sigma_slow = config.min_sigma;
    }
    if sigma_fast <= sigma_slow {
        sigma_fast = sigma_slow * 1.5 + config.min_sigma;
    }

    let headroom = config.max_extrapolation_db / NEPERS_TO_DB;
    let smooth = NelderMead {
        max_evaluations: config.max_evaluations,
        tolerance: 1e-9,
        initial_step: 0.35,
    }
    .minimize(
        &[
            amplitude_fast.ln(),
            amplitude_slow.ln(),
            sigma_slow.ln(),
            (sigma_fast - sigma_slow).ln(),
        ],
        |p| envelope_residual(&Parameters::beatless(p), &points, headroom),
    );
    let mut best = Parameters::beatless(&smooth.point);
    let mut best_value = smooth.value;

    // The search over the beats is run on a thinned copy. Frames a hop apart
    // share almost all of their window and so almost all of their information;
    // what this fit needs is enough samples per cycle of the fastest beat it is
    // chasing, and ten is plenty. Thinning is what pays for two passes of a
    // seven-parameter simplex from each of several starting points.
    let peaks = beat_candidates(&points, &best, duration, config);
    let coarse = thin(&points, peaks.iter().copied().fold(0.0, f64::max));
    let mut candidate = None;
    let mut candidate_value = f64::MAX;
    let solver = NelderMead {
        max_evaluations: config.beat_evaluations,
        tolerance: 1e-9,
        initial_step: 0.15,
    };
    for start in beat_starts(&best, &peaks) {
        let start = start.pack(duration);
        // Two passes per start, and the first one holds the beat rates at what
        // the spectrum said they were. A residual as a function of a beat rate
        // oscillates with a period of one cycle over the record — half a cycle
        // out and the model's nulls sit where the measurement's peaks are — so
        // a simplex that is free to move the rates before the depths and decays
        // around them are anywhere near right walks straight out of the basin
        // the spectrum handed it. The spectral estimate is good to about that
        // one cycle to begin with.
        let fixed = solver.minimize(&reduce(&start), |p| {
            envelope_residual(
                &Parameters::unpack(&restore(p, &start), duration, config),
                &coarse,
                headroom,
            )
        });
        let refined = solver.minimize(&restore(&fixed.point, &start), |p| {
            envelope_residual(&Parameters::unpack(p, duration, config), &coarse, headroom)
        });
        if refined.value < candidate_value {
            candidate_value = refined.value;
            candidate = Some(refined.point);
        }
    }
    // The winner is then polished against every measurement, and only replaces
    // the beatless fit if it explains the whole record better — so a partial
    // with nothing beating in it keeps the answer it has always been given.
    if let Some(start) = candidate {
        let polished = NelderMead {
            max_evaluations: config.max_evaluations,
            tolerance: 1e-9,
            initial_step: 0.05,
        }
        .minimize(&start, |p| {
            envelope_residual(&Parameters::unpack(p, duration, config), &points, headroom)
        });
        let mut parameters = Parameters::unpack(&polished.point, duration, config);
        parameters.drop_unseen_beats(duration, config);
        // Judged after the unseen beats have been taken out again, so that what
        // is accepted is an improvement the *kept* beats explain.
        let value = envelope_residual(&parameters, &points, headroom);
        if value < (1.0 - config.min_beat_improvement) * best_value {
            best = parameters;
            best_value = value;
        }
    }

    let fit = DecayFit {
        k: track.k,
        frequency_hz: track
            .weighted_frequency()
            .or_else(|| track.median_frequency())
            .unwrap_or(0.0),
        fast: Exponential {
            amplitude: best.amplitude_fast,
            sigma: best.sigma_fast,
        },
        slow: Exponential {
            amplitude: best.amplitude_slow,
            sigma: best.sigma_slow,
        },
        beats: best.beats(),
        residual_db: NEPERS_TO_DB * best_value.sqrt(),
        points: points.len(),
        span_s: duration,
    };
    if !fit.initial_amplitude().is_finite() || fit.initial_amplitude() <= 0.0 {
        return Err(Error::Estimate(format!(
            "partial {} envelope fit diverged",
            track.k
        )));
    }
    Ok(fit)
}

/// Below this share a string of the group is inaudible and its offset is
/// unidentifiable: what the fit has found there is the shape of the noise.
const MIN_UNISON_DEPTH: f64 = 5.0e-3;

/// Simplex coordinate for a string that is not there. The depth is a logistic
/// of the coordinate, so zero is at minus infinity; this is far enough down (a
/// share of 5e-5) to be nothing and near enough that the simplex can climb back
/// out of it.
const ABSENT_STRING: f64 = -10.0;

/// The envelope model's coordinates with the two beat rates taken out, and put
/// back. Which coordinates they are is [`Parameters::pack`]'s business; these
/// two are the only other place that ordering is relied on.
fn reduce(full: &[f64]) -> Vec<f64> {
    vec![full[0], full[1], full[2], full[3], full[6]]
}

fn restore(reduced: &[f64], rates: &[f64]) -> Vec<f64> {
    vec![
        reduced[0], reduced[1], reduced[2], reduced[3], rates[4], rates[5], reduced[4],
    ]
}

/// Cycles over the record below which a fitted beat rate is no beat at all.
const NEGLIGIBLE_CYCLES: f64 = 0.05;

/// Share the second string is seeded with: a modulation a few decibels deep,
/// enough for the simplex to feel a beat and shallow enough not to invent one.
const SEED_DEPTH: f64 = 0.25;

/// The envelope model's parameters, in the units they mean rather than in the
/// simplex's coordinates.
#[derive(Clone, Copy, Debug)]
struct Parameters {
    amplitude_fast: f64,
    amplitude_slow: f64,
    sigma_fast: f64,
    sigma_slow: f64,
    /// Offset between the two polarizations, Hz.
    polarization_hz: f64,
    /// Offset of the second unison string from the loudest, Hz, and its share.
    unison_hz: f64,
    depth: f64,
}

impl Parameters {
    /// The four-parameter beatless model, as the simplex writes it.
    fn beatless(p: &[f64]) -> Self {
        let sigma_slow = p[2].exp();
        Self {
            amplitude_fast: p[0].exp(),
            amplitude_slow: p[1].exp(),
            sigma_fast: sigma_slow + p[3].exp(),
            sigma_slow,
            polarization_hz: 0.0,
            unison_hz: 0.0,
            depth: 0.0,
        }
    }

    /// The simplex's coordinates.
    ///
    /// The two beat rates are carried in *cycles over the record* rather than
    /// in hertz. The residual as a function of a beat rate oscillates with a
    /// period of one cycle over the record — half a cycle out and the model's
    /// nulls sit where the measurement's peaks are — so a simplex stepping in
    /// hertz steps over several of those basins at a bass note's rates and
    /// through none of them at a treble note's. In cycles the step means the
    /// same thing at both ends of the compass.
    fn pack(&self, duration: f64) -> Vec<f64> {
        vec![
            self.amplitude_fast.ln(),
            self.amplitude_slow.ln(),
            self.sigma_slow.ln(),
            (self.sigma_fast - self.sigma_slow).max(f64::MIN_POSITIVE).ln(),
            self.polarization_hz * duration,
            self.unison_hz * duration,
            if self.depth <= 0.0 {
                ABSENT_STRING
            } else {
                (self.depth / (1.0 - self.depth).max(f64::MIN_POSITIVE)).ln()
            },
        ]
    }

    fn unpack(p: &[f64], duration: f64, config: &DecayConfig) -> Self {
        let sigma_slow = p[2].exp();
        let per_cycle = 1.0 / duration.max(f64::MIN_POSITIVE);
        // Absolute value rather than a bound: a beat has no sign, and
        // reflecting keeps the simplex from having to stop at zero. A rate
        // under a twentieth of a cycle over the whole record is not a slow beat
        // but the absence of one — its cosine has moved by half a percent — and
        // saying so exactly is what lets the guard below tell "no beat" apart
        // from "a beat too slow to have been seen".
        let rate = |cycles: f64| {
            if cycles.abs() < NEGLIGIBLE_CYCLES {
                0.0
            } else {
                cycles.abs() * per_cycle
            }
        };
        let unison_hz = rate(p[5]);
        Self {
            amplitude_fast: p[0].exp(),
            amplitude_slow: p[1].exp(),
            sigma_fast: sigma_slow + p[3].exp(),
            sigma_slow,
            polarization_hz: rate(p[4]),
            unison_hz,
            // A logistic keeps the share inside `(0, max_unison_depth)` without
            // a wall for the simplex to press against. A modulation that does
            // not modulate has no depth either: at a standstill it is a constant
            // factor on the envelope, which the amplitudes already carry.
            depth: if unison_hz > 0.0 {
                config.max_unison_depth / (1.0 + (-p[6]).exp())
            } else {
                0.0
            },
        }
    }

    /// Removes any beat too slow for the record to have seen a cycle and a half
    /// of.
    ///
    /// The simplex is free to move a beat's rate anywhere once it has started,
    /// and the place it goes when there is nothing left to fit is *down*: a
    /// modulation slower than the record is a slow drift, and a slow drift can
    /// absorb the model error of anything. What it absorbs it takes out of the
    /// decay, and with it out of the amplitude extrapolated back to the strike.
    fn drop_unseen_beats(&mut self, duration: f64, config: &DecayConfig) {
        let lowest = config.min_beat_cycles / duration.max(f64::MIN_POSITIVE);
        if self.polarization_hz < lowest {
            self.polarization_hz = 0.0;
        }
        if self.unison_hz < lowest {
            self.unison_hz = 0.0;
            self.depth = 0.0;
        }
    }

    fn beats(&self) -> EnvelopeBeats {
        // A modulation with no depth in it has no rate worth reporting.
        let beating = self.depth > MIN_UNISON_DEPTH;
        EnvelopeBeats {
            unison_hz: if beating { self.unison_hz } else { 0.0 },
            unison_depth: if beating { self.depth } else { 0.0 },
            polarization_hz: self.polarization_hz,
        }
    }
}

/// `|A_f e^{-sigma_f t} + A_s e^{-sigma_s t} e^{i 2 pi f t}|`: the two
/// polarizations of one string beating against each other. Their beat is inside
/// the sum and not a factor on it, which is why its depth grows and fades as the
/// fast component dies, and why smoothing it away biases the rates.
fn polarization_beat(fast: f64, slow: f64, hz: f64, t: f64) -> f64 {
    (fast * fast + slow * slow + 2.0 * fast * slow * (TAU * hz * t).cos())
        .max(0.0)
        .sqrt()
}

/// `|1 + d e^{i 2 pi f t}| / (1 + d)`: the second string of the group beating
/// against the loudest, normalized to 1 where they are in phase.
fn unison_modulation(hz: f64, depth: f64, t: f64) -> f64 {
    if depth <= 0.0 {
        return 1.0;
    }
    let phase = TAU * hz * t;
    ((1.0 + depth * depth + 2.0 * depth * phase.cos()).max(0.0)).sqrt() / (1.0 + depth)
}

/// Mean square log-domain residual of the envelope model, in squared nepers —
/// the quantity every stage of the fit is judged on.
fn envelope_residual(p: &Parameters, points: &[(f64, f64)], headroom: f64) -> f64 {
    if !(p.amplitude_fast.is_finite()
        && p.amplitude_slow.is_finite()
        && p.sigma_fast.is_finite()
        && p.sigma_slow.is_finite())
    {
        return f64::MAX;
    }
    let mut sum = 0.0;
    for &(t, a) in points {
        let fast = p.amplitude_fast * (-p.sigma_fast * t).exp();
        let slow = p.amplitude_slow * (-p.sigma_slow * t).exp();
        let model = polarization_beat(fast, slow, p.polarization_hz, t)
            * unison_modulation(p.unison_hz, p.depth, t);
        if !(model.is_finite() && model > 0.0) {
            return f64::MAX;
        }
        let error = a.ln() - model.ln();
        sum += error * error;
    }
    let mut mean = sum / points.len() as f64;
    // The extrapolation back to the strike, penalized past `headroom`. Smooth
    // and one-sided, so it is inert on every fit the data supports.
    let (first, _) = points[0];
    let start = p.amplitude_fast + p.amplitude_slow;
    let measured = polarization_beat(
        p.amplitude_fast * (-p.sigma_fast * first).exp(),
        p.amplitude_slow * (-p.sigma_slow * first).exp(),
        p.polarization_hz,
        first,
    );
    if measured > 0.0 {
        let excess = (start / measured).ln() - headroom;
        if excess > 0.0 {
            mean += excess * excess;
        }
    }
    mean
}


/// Median spacing of a point series, in seconds.
fn points_hop(points: &[(f64, f64)]) -> f64 {
    let gaps: Vec<f64> = points.windows(2).map(|w| w[1].0 - w[0].0).collect();
    median(&gaps).filter(|&gap| gap > 0.0).unwrap_or(1.0)
}

/// Every n-th point, chosen so that the fastest beat still gets ten samples per
/// cycle.
fn thin(points: &[(f64, f64)], fastest_hz: f64) -> Vec<(f64, f64)> {
    let hop = points_hop(points);
    let wanted = if fastest_hz > 0.0 {
        1.0 / (10.0 * fastest_hz)
    } else {
        hop
    };
    let stride = ((wanted / hop).floor() as usize).max(1);
    // Never thin below ten measurements per fitted parameter.
    let stride = stride.min((points.len() / 100).max(1));
    points.iter().copied().step_by(stride).collect()
}

/// The beat rates a partial's envelope might carry: the periodic content of
/// what the beatless fit could not explain.
fn beat_candidates(
    points: &[(f64, f64)],
    seed: &Parameters,
    duration: f64,
    config: &DecayConfig,
) -> Vec<f64> {
    let residual: Vec<(f64, f64)> = points
        .iter()
        .map(|&(t, a)| {
            let model = seed.amplitude_fast * (-seed.sigma_fast * t).exp()
                + seed.amplitude_slow * (-seed.sigma_slow * t).exp();
            (t, a.ln() - model.max(f64::MIN_POSITIVE).ln())
        })
        .collect();
    // A beat the record has seen less than `min_beat_cycles` of is a drift, and
    // a drift is exactly what a beatless fit leaves behind.
    let lowest = config.min_beat_cycles / duration.max(f64::MIN_POSITIVE);
    modulation_peaks(&residual, lowest, config.max_beat_hz, 2)
}

/// Starting points for the beat-aware refinement.
///
/// Which peak belongs to which beat is not knowable from the spectrum alone —
/// the two kinds enter the model differently but both are just "a beat" in the
/// residual — so every plausible assignment of the peaks found is offered to
/// the simplex and the one that fits best wins.
fn beat_starts(seed: &Parameters, peaks: &[f64]) -> Vec<Parameters> {
    let mut starts = Vec::new();
    let mut push = |polarization: f64, unison: Option<f64>| {
        starts.push(Parameters {
            polarization_hz: polarization,
            unison_hz: unison.unwrap_or(0.0),
            depth: if unison.is_some() { SEED_DEPTH } else { 0.0 },
            ..*seed
        });
    };
    let Some(&first) = peaks.first() else {
        return starts;
    };
    // The strongest modulation, as either kind of beat ...
    push(first, None);
    push(0.0, Some(first));
    // ... and, when there are two, one of each kind either way round.
    if let Some(&second) = peaks.get(1) {
        push(second, Some(first));
        push(first, Some(second));
    }
    starts
}

/// The strongest periodic components of an unevenly sampled residual, in
/// descending order of strength and separated by more than the record can
/// resolve.
fn modulation_peaks(residual: &[(f64, f64)], lo_hz: f64, hi_hz: f64, count: usize) -> Vec<f64> {
    let (Some(first), Some(last)) = (residual.first(), residual.last()) else {
        return Vec::new();
    };
    let duration = last.0 - first.0;
    if duration <= 0.0 || hi_hz <= lo_hz {
        return Vec::new();
    }
    // A Hann taper over the record: without it the edges put a sinc pattern
    // around every peak whose sidelobes are 13 dB down and can outrank a real
    // second beat.
    let windowed: Vec<(f64, f64)> = residual
        .iter()
        .map(|&(t, r)| {
            let phase = TAU * (t - first.0) / duration;
            (t, r * (0.5 - 0.5 * phase.cos()))
        })
        .collect();
    let magnitude = |hz: f64| -> f64 {
        let (mut re, mut im) = (0.0, 0.0);
        for &(t, r) in &windowed {
            let phase = -TAU * hz * t;
            re += r * phase.cos();
            im += r * phase.sin();
        }
        (re * re + im * im).sqrt()
    };

    // Four points per resolution cell of the record, so no peak is missed and
    // its position is set by the data rather than by the grid.
    let resolution = 1.0 / duration;
    let steps = (4.0 * (hi_hz - lo_hz) / resolution).ceil().max(8.0) as usize;
    let step = (hi_hz - lo_hz) / steps as f64;
    let spectrum: Vec<f64> = (0..=steps).map(|i| magnitude(lo_hz + i as f64 * step)).collect();

    let mut peaks: Vec<(f64, f64)> = (1..steps)
        .filter(|&i| spectrum[i] > spectrum[i - 1] && spectrum[i] >= spectrum[i + 1])
        .map(|i| {
            let offset = crate::numeric::parabolic_offset(
                spectrum[i - 1],
                spectrum[i],
                spectrum[i + 1],
            );
            (spectrum[i], lo_hz + (i as f64 + offset) * step)
        })
        .collect();
    peaks.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut chosen: Vec<f64> = Vec::with_capacity(count);
    for (_, hz) in peaks {
        if chosen.len() == count {
            break;
        }
        // Two peaks closer than the record can resolve are one peak.
        if chosen.iter().all(|&other| (other - hz).abs() > 2.0 * resolution) {
            chosen.push(hz);
        }
    }
    chosen
}

/// Log-linear slopes over the first and last thirds of the track: the fast
/// component dominates early, the slow one late.
fn seed_rates(points: &[(f64, f64)], duration: f64, config: &DecayConfig) -> (f64, f64) {
    let first = points[0].0;
    let early: Vec<(f64, f64)> = points
        .iter()
        .copied()
        .filter(|&(t, _)| t <= first + duration / 3.0)
        .collect();
    let late: Vec<(f64, f64)> = points
        .iter()
        .copied()
        .filter(|&(t, _)| t >= first + 2.0 * duration / 3.0)
        .collect();
    let fast = log_slope(&early).unwrap_or(LN_1000 / duration);
    let slow = log_slope(&late).unwrap_or(fast * 0.3);
    (
        fast.max(config.min_sigma),
        slow.clamp(config.min_sigma, fast.max(config.min_sigma)),
    )
}

/// Negated least-squares slope of `ln a` against `t`: a single decay rate.
fn log_slope(points: &[(f64, f64)]) -> Option<f64> {
    if points.len() < 2 {
        return None;
    }
    let n = points.len() as f64;
    let mean_t = points.iter().map(|p| p.0).sum::<f64>() / n;
    let mean_y = points.iter().map(|p| p.1.ln()).sum::<f64>() / n;
    let (mut num, mut den) = (0.0, 0.0);
    for &(t, a) in points {
        num += (t - mean_t) * (a.ln() - mean_y);
        den += (t - mean_t) * (t - mean_t);
    }
    if den <= 0.0 {
        return None;
    }
    Some(-num / den)
}

/// Amplitudes of the two components with their rates held fixed, by weighted
/// linear least squares. Weight `1/a^2` makes the linear problem approximate
/// the log-domain one the simplex then minimizes exactly.
fn seed_amplitudes(points: &[(f64, f64)], sigma_fast: f64, sigma_slow: f64) -> Option<(f64, f64)> {
    let mut basis = Vec::with_capacity(points.len() * 2);
    let mut y = Vec::with_capacity(points.len());
    let mut weights = Vec::with_capacity(points.len());
    for &(t, a) in points {
        basis.push((-sigma_fast * t).exp());
        basis.push((-sigma_slow * t).exp());
        y.push(a);
        weights.push(1.0 / (a * a));
    }
    let solution = weighted_least_squares(&basis, &y, &weights, 2)?;
    Some((solution[0], solution[1]))
}

/// Fits `sigma(f) = sigma0 + sigma1 (f/1000)^2` to the partials' effective
/// decay rates.
///
/// Both coefficients are physical losses and neither may be negative: air and
/// internal friction only ever remove energy. A fit that wants a negative
/// coefficient is a fit whose data does not support that term, so the term is
/// dropped and the rest refitted rather than a negative damping written into a
/// preset.
pub fn fit_decay_curve(partials: &[DecayFit], config: &DecayConfig) -> Result<DecayCurve> {
    let rates: Vec<(f64, f64)> = partials
        .iter()
        .filter(|fit| fit.frequency_hz > 0.0 && fit.is_measured(config))
        .map(|fit| ((fit.frequency_hz / 1000.0).powi(2), fit.effective_sigma()))
        .filter(|&(_, sigma)| sigma.is_finite() && sigma > 0.0)
        .collect();
    if rates.len() < 2 {
        return Err(Error::Estimate(format!(
            "sigma(f) needs two partials whose decay the recording is long enough to \
             measure, got {} of {}",
            rates.len(),
            partials.len()
        )));
    }
    let basis: Vec<f64> = rates.iter().flat_map(|&(x, _)| [1.0, x]).collect();
    let y: Vec<f64> = rates.iter().map(|&(_, s)| s).collect();
    let weights = vec![1.0; rates.len()];
    let solution = weighted_least_squares(&basis, &y, &weights, 2)
        .ok_or_else(|| Error::Estimate("sigma(f) fit is singular".into()))?;
    let (mut sigma0, mut sigma1) = (solution[0], solution[1]);
    if sigma1 < 0.0 {
        sigma1 = 0.0;
        sigma0 = y.iter().sum::<f64>() / y.len() as f64;
    }
    if sigma0 < 0.0 {
        // Refit through the origin: sigma1 = sum(x y) / sum(x^2).
        sigma0 = 0.0;
        let num: f64 = rates.iter().map(|&(x, s)| x * s).sum();
        let den: f64 = rates.iter().map(|&(x, _)| x * x).sum();
        sigma1 = if den > 0.0 { num / den } else { 0.0 };
    }
    let curve = DecayCurve {
        sigma0,
        sigma1,
        residual: 0.0,
    };
    let residual = (rates
        .iter()
        .map(|&(x, s)| (s - (sigma0 + sigma1 * x)).powi(2))
        .sum::<f64>()
        / rates.len() as f64)
        .sqrt();
    Ok(DecayCurve { residual, ..curve })
}

/// Median level and rate ratio between the two components, over the partials
/// whose fit actually split. Medians and not means: a partial whose beat
/// pattern happened to fool the fit is one sample, not a bias.
pub fn polarization_split(partials: &[DecayFit], config: &DecayConfig) -> Result<PolarizationSplit> {
    let split: Vec<&DecayFit> = partials
        .iter()
        .filter(|fit| fit.is_split(config) && fit.is_measured(config))
        .collect();
    if split.is_empty() {
        return Err(Error::Estimate(
            "no partial showed a two-polarization decay".into(),
        ));
    }
    let gains: Vec<f64> = split
        .iter()
        .map(|fit| 20.0 * (fit.slow.amplitude / fit.fast.amplitude).log10())
        .collect();
    let ratios: Vec<f64> = split
        .iter()
        .map(|fit| fit.slow.sigma / fit.fast.sigma)
        .collect();
    Ok(PolarizationSplit {
        gain_db: median(&gains).expect("non-empty"),
        decay_ratio: median(&ratios).expect("non-empty"),
        partials: split.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trajectory::TrackPoint;

    /// A track sampled from a known envelope, on the recording's clock.
    fn track(k: u32, hz: f64, onset_s: f64, hop_s: f64, count: usize, envelope: impl Fn(f64) -> f64) -> PartialTrack {
        PartialTrack {
            k,
            points: (0..count)
                .map(|i| {
                    let t = i as f64 * hop_s;
                    TrackPoint {
                        time_s: onset_s + t,
                        frequency_hz: hz,
                        amplitude: envelope(t),
                    }
                })
                .collect(),
        }
    }

    const SPAN: FitSpan = FitSpan {
        onset_s: 0.0,
        start_s: 0.0,
    };

    #[test]
    fn a_double_decay_returns_both_of_its_rates() {
        let (a_f, s_f, a_s, s_s) = (1.0, 4.0, 0.25, 0.8);
        let envelope = |t: f64| a_f * (-s_f * t).exp() + a_s * (-s_s * t).exp();
        let fit =
            fit_two_exponential(&track(1, 110.0, 0.0, 0.01, 800, envelope), SPAN, &DecayConfig::default())
                .unwrap();
        assert!((fit.fast.sigma / s_f - 1.0).abs() < 0.01, "{fit:?}");
        assert!((fit.slow.sigma / s_s - 1.0).abs() < 0.01, "{fit:?}");
        assert!((fit.fast.amplitude / a_f - 1.0).abs() < 0.02, "{fit:?}");
        assert!((fit.slow.amplitude / a_s - 1.0).abs() < 0.02, "{fit:?}");
        assert!(fit.residual_db < 0.01, "{fit:?}");

        // ... and the T60 of the pair, which is what the preset stores.
        let truth = {
            let target = 1e-3 * (a_f + a_s);
            let mut t = 0.0;
            while envelope(t) > target {
                t += 1e-4;
            }
            t
        };
        assert!((fit.t60() / truth - 1.0).abs() < 0.05, "{} vs {truth}", fit.t60());
    }

    /// The envelope of one string of a piano: two polarizations a fraction of a
    /// hertz apart, so their sum beats. Fitted as a plain sum of exponentials
    /// this returns whatever the beat nulls inside the record happened to do;
    /// the T60 assertion here is the one that used to fail by 16 %.
    #[test]
    fn a_double_decay_that_beats_gives_back_its_rates_and_its_beat() {
        let (a_f, s_f, a_s, s_s, beat) = (1.0, 1.5, 0.25, 0.44, 0.35);
        let envelope = |t: f64| {
            let (fast, slow) = (a_f * (-s_f * t).exp(), a_s * (-s_s * t).exp());
            (fast * fast + slow * slow + 2.0 * fast * slow * (TAU * beat * t).cos()).sqrt()
        };
        let fit = fit_two_exponential(
            &track(1, 55.0, 0.0, 0.01, 1_800, envelope),
            SPAN,
            &DecayConfig::default(),
        )
        .unwrap();
        assert!((fit.fast.sigma / s_f - 1.0).abs() < 0.05, "{fit:?}");
        assert!((fit.slow.sigma / s_s - 1.0).abs() < 0.05, "{fit:?}");
        assert!(
            (fit.beats.polarization_hz - beat).abs() < 0.02,
            "beat {:.4} Hz vs {beat}: {fit:?}",
            fit.beats.polarization_hz
        );
        assert!(fit.residual_db < 0.05, "{fit:?}");
        // The reported envelope is the coherent one, which is what a T60 and an
        // excitation spectrum are read from.
        assert!((fit.initial_amplitude() / (a_f + a_s) - 1.0).abs() < 0.02, "{fit:?}");
        let coherent = |t: f64| a_f * (-s_f * t).exp() + a_s * (-s_s * t).exp();
        let truth = {
            let target = 1e-3 * coherent(0.0);
            let mut t = 0.0;
            while coherent(t) > target {
                t += 1e-3;
            }
            t
        };
        assert!(
            (fit.t60() / truth - 1.0).abs() < 0.05,
            "T60 {:.2} s vs {truth:.2}",
            fit.t60()
        );
    }

    /// A unison's beat multiplies the envelope instead of beating the two
    /// decays against each other, and it has to come back as the other kind.
    #[test]
    fn a_unison_beat_is_told_apart_from_a_polarization_beat() {
        let (depth, beat) = (0.35, 0.6);
        let envelope = |t: f64| {
            let decay = (-1.2 * t).exp() + 0.25 * (-0.4 * t).exp();
            decay * unison_modulation(beat, depth, t)
        };
        let fit = fit_two_exponential(
            &track(1, 65.0, 0.0, 0.01, 1_400, envelope),
            SPAN,
            &DecayConfig::default(),
        )
        .unwrap();
        assert!(
            (fit.beats.unison_hz - beat).abs() < 0.02
                && (fit.beats.unison_depth / depth - 1.0).abs() < 0.1,
            "{:?}",
            fit.beats
        );
        assert!((fit.fast.sigma / 1.2 - 1.0).abs() < 0.05, "{fit:?}");
        assert!((fit.slow.sigma / 0.4 - 1.0).abs() < 0.05, "{fit:?}");
    }

    #[test]
    fn a_partial_that_does_not_beat_is_not_given_a_beat() {
        // Three parameters can always shave something off a fit; on an envelope
        // with no modulation in it they may not, because what they would be
        // fitting is the measurement's noise and what it costs is the amplitude
        // the fit extrapolates back to the strike.
        let clean = |t: f64| 0.8 * (-3.0 * t).exp() + 0.2 * (-0.9 * t).exp();
        let mut points = track(1, 220.0, 0.0, 0.01, 600, clean);
        for (i, point) in points.points.iter_mut().enumerate() {
            point.amplitude *= 1.0 + 0.02 * ((i as f64 * 0.7).sin() + (i as f64 * 0.13).cos());
        }
        let fit = fit_two_exponential(&points, SPAN, &DecayConfig::default()).unwrap();
        assert_eq!(fit.beats, EnvelopeBeats::default(), "{fit:?}");
        assert!((fit.initial_amplitude() - 1.0).abs() < 0.02, "{fit:?}");
    }

    #[test]
    fn a_single_exponential_is_not_forced_into_two() {
        let envelope = |t: f64| 0.7 * (-2.5 * t).exp();
        let fit =
            fit_two_exponential(&track(3, 330.0, 0.0, 0.01, 400, envelope), SPAN, &DecayConfig::default())
                .unwrap();
        // Whatever the fit does with its spare component, the envelope it
        // describes and the T60 it reports must be the true ones.
        assert!((fit.t60() / (LN_1000 / 2.5) - 1.0).abs() < 0.05, "{fit:?}");
        assert!((fit.amplitude_at(1.0) / envelope(1.0) - 1.0).abs() < 0.02, "{fit:?}");
        assert!((fit.initial_amplitude() / 0.7 - 1.0).abs() < 0.02, "{fit:?}");
    }

    #[test]
    fn envelope_noise_does_not_move_the_t60_by_five_percent() {
        // 5 % multiplicative jitter on every point, sign alternating so it
        // cannot be averaged away by a lucky symmetric fit.
        let (a_f, s_f, a_s, s_s) = (1.0, 6.0, 0.15, 1.1);
        let clean = |t: f64| a_f * (-s_f * t).exp() + a_s * (-s_s * t).exp();
        let mut points = track(2, 220.0, 0.05, 0.01, 600, clean);
        for (i, point) in points.points.iter_mut().enumerate() {
            point.amplitude *= if i % 2 == 0 { 1.05 } else { 0.95 };
        }
        let fit = fit_two_exponential(&points, FitSpan::new(0.05, 0.05), &DecayConfig::default()).unwrap();
        let truth = {
            let target = 1e-3 * (a_f + a_s);
            let mut t = 0.0;
            while clean(t) > target {
                t += 1e-4;
            }
            t
        };
        assert!((fit.t60() / truth - 1.0).abs() < 0.05, "{} vs {truth}", fit.t60());
        assert!((fit.slow.sigma / s_s - 1.0).abs() < 0.05, "{fit:?}");
    }

    #[test]
    fn a_partial_that_has_reached_the_recordings_floor_is_fitted_without_it() {
        // A treble partial: two seconds of tape, a prompt decay of half a
        // second, and a room 30 dB down that wanders by a couple of decibels
        // and that the string is under after the first quarter second. What is
        // recoverable from that is the prompt rate, and only because the floor
        // is found and taken out; the whole envelope's T60 is defined 60 dB
        // down, which this record does not contain at all.
        let config = DecayConfig::default();
        let sigma = LN_1000 / 0.5;
        let floor = |t: f64| 0.03 * (1.0 + 0.3 * (TAU * t / 1.2).sin());
        let envelope = |t: f64| ((-sigma * t).exp().powi(2) + floor(t).powi(2)).sqrt();
        let track = track(1, 2000.0, 0.0, 0.005, 400, envelope);

        let fit = fit_two_exponential(&track, SPAN, &config).unwrap();
        assert!(
            (fit.fast.sigma / sigma - 1.0).abs() < 0.05,
            "prompt rate {:.3} /s, expected {sigma:.3}",
            fit.fast.sigma
        );
        // Nothing in this signal has an aftersound, and the fit does not claim
        // one: what is left over is a thousandth of the strike, not a
        // polarization.
        let aftersound = fit.slow.amplitude / fit.initial_amplitude();
        assert!(aftersound < 0.015, "aftersound {aftersound:.4} of the strike");
        assert!(fit.residual_db < 0.2, "residual {:.2} dB", fit.residual_db);

        // With the floor detector off the same fit reads the room as the
        // string's aftersound — at exactly the room's own level, 3 % of the
        // strike — which is the number the polarization split is the median of.
        // This is the failure the detector exists for, pinned so that switching
        // it off stays a choice.
        let blind = fit_two_exponential(
            &track,
            SPAN,
            &DecayConfig {
                floor_blocks: 0,
                ..config
            },
        )
        .unwrap();
        let aftersound = blind.slow.amplitude / blind.initial_amplitude();
        assert!((aftersound - 0.03).abs() < 0.005, "aftersound {aftersound:.4}");
        assert!(blind.residual_db > 1.0, "residual {:.2} dB", blind.residual_db);
    }

    #[test]
    fn a_decay_the_recording_is_too_short_to_see_is_not_counted_as_measured() {
        // Three seconds of a partial that rings for thirty: five decibels of
        // evidence for a sixty-decibel claim.
        let config = DecayConfig::default();
        let fits: Vec<DecayFit> = [110.0, 220.0]
            .iter()
            .enumerate()
            .map(|(index, &hz)| {
                let envelope = |t: f64| 0.5 * (-0.23 * t).exp();
                fit_two_exponential(
                    &track(index as u32 + 1, hz, 0.0, 0.01, 300, envelope),
                    SPAN,
                    &config,
                )
                .unwrap()
            })
            .collect();
        for fit in &fits {
            assert!(!fit.is_measured(&config), "{fit:?}");
            // The fit is still returned, and still describes what was seen.
            assert!((fit.amplitude_at(1.0) / (0.5 * (-0.23f64).exp()) - 1.0).abs() < 0.02);
        }
        // ... but a curve through nothing but extrapolations is refused.
        assert!(fit_decay_curve(&fits, &config).is_err());
    }

    #[test]
    fn the_sigma_curve_recovers_the_engines_damping_law() {
        // sigma_k = 0.7 + 1.2 (f_k/1000)^2 on a stiff string.
        let model = crate::trajectory::InharmonicModel::new(110.0, 4e-4);
        let partials: Vec<DecayFit> = (1..=16)
            .map(|k| {
                let f = model.partial(k);
                let sigma = 0.7 + 1.2 * (f / 1000.0).powi(2);
                DecayFit {
                    k,
                    frequency_hz: f,
                    fast: Exponential {
                        amplitude: 1.0,
                        sigma,
                    },
                    slow: Exponential {
                        amplitude: 0.0,
                        sigma,
                    },
                    beats: EnvelopeBeats::default(),
                    residual_db: 0.0,
                    points: 100,
                    span_s: 1e6,
                }
            })
            .collect();
        let curve = fit_decay_curve(&partials, &DecayConfig::default()).unwrap();
        assert!((curve.sigma0 - 0.7).abs() < 1e-6, "{curve:?}");
        assert!((curve.sigma1 - 1.2).abs() < 1e-6, "{curve:?}");
    }

    #[test]
    fn a_negative_damping_coefficient_is_refused() {
        // Rates that fall with frequency: physically impossible, so the
        // frequency term must be dropped rather than fitted negative.
        let partials: Vec<DecayFit> = (1..=8)
            .map(|k| DecayFit {
                k,
                frequency_hz: 100.0 * f64::from(k),
                fast: Exponential {
                    amplitude: 1.0,
                    sigma: 3.0 - 0.2 * f64::from(k),
                },
                slow: Exponential {
                    amplitude: 0.0,
                    sigma: 3.0,
                },
                beats: EnvelopeBeats::default(),
                residual_db: 0.0,
                points: 100,
                span_s: 1e6,
            })
            .collect();
        let curve = fit_decay_curve(&partials, &DecayConfig::default()).unwrap();
        assert_eq!(curve.sigma1, 0.0);
        assert!(curve.sigma0 > 0.0);
    }

    #[test]
    fn the_polarization_split_is_the_median_of_the_partials() {
        let config = DecayConfig::default();
        let partials: Vec<DecayFit> = (1..=5)
            .map(|k| DecayFit {
                k,
                frequency_hz: 100.0 * f64::from(k),
                fast: Exponential {
                    amplitude: 1.0,
                    sigma: 4.0,
                },
                slow: Exponential {
                    // -12 dB and a rate ratio of 0.3, as the default preset has.
                    amplitude: 0.251_189,
                    sigma: 1.2,
                },
                beats: EnvelopeBeats::default(),
                residual_db: 0.0,
                points: 100,
                span_s: 1e6,
            })
            .collect();
        let split = polarization_split(&partials, &config).unwrap();
        assert!((split.gain_db + 12.0).abs() < 0.01, "{split:?}");
        assert!((split.decay_ratio - 0.3).abs() < 1e-9, "{split:?}");
        assert_eq!(split.partials, 5);
    }
}

