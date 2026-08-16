//! Duplex and aliquot segments, from the recordings of what still rings when
//! the dampers land.
//!
//! `PHYSICS.md` §3: the string does not end at the bridge or at the agraffe.
//! The front segment (capo bar to tuning pin) and the rear one (bridge to hitch
//! pin) are short, high-pitched and have **no dampers**, so they go on sounding
//! after the speaking length has been stopped. §3 also names the material that
//! measures them and the method:
//!
//! > Estimation is the existing tracker with the inharmonic seed removed:
//! > peak-pick the residual, keep peaks with T60 > 0.3 s, write the strongest
//! > 2–6 per key.
//!
//! Salamander's `harmL*`, `harmS*` and `harmV3*` are three velocity tiers of
//! "release string resonances" — literally a recording of what is left when the
//! key comes up — and its `rel*` samples are the key-off thumps, whose *tail*
//! is the same thing under the noise of the damper landing.
//!
//! # Why the seed has to go, and what replaces it
//!
//! [`PartialTracker`](crate::tracker::PartialTracker) is seeded: the caller
//! says where partial `k` should be and association reduces to "the nearest
//! peak inside a window no other partial can reach into". That is exactly the
//! wrong instrument here, because the whole claim about a duplex is that it is
//! **not** at `k f0 sqrt(1 + B k^2)` — Öberg & Askenfelt found real rear-duplex
//! tuning sharp of nominal by an average approaching +50 cents with ~25 cents
//! of scatter within one trichord, and *that scatter is the sound*. A seeded
//! tracker asked to find them would either miss them or, worse, snap them onto
//! the harmonic grid and report ratios.
//!
//! So [`residual_modes`] tracks free: every frame is peak-picked, peaks are
//! chained across frames by proximity alone, and a chain becomes a candidate
//! only if it survives long enough to fit a decay to. The note's own partials
//! are then subtracted as a *frequency* exclusion — anything within
//! [`DuplexConfig::guard_cents`] of a measured partial of the struck note is
//! that partial, not a segment — and what is left, ranked by level and cut at
//! `min_t60_s`, is the row.
//!
//! # What the numbers mean
//!
//! `hz` and `t60_s` are measured directly. `gain_db` is not: the schema defines
//! it as the segment's response at its own frequency *per unit of the bridge
//! force driving it*, and a recording does not carry the bridge force. What it
//! carries is a level relative to a strike of the same key, which is the same
//! quantity the whole `harm*` table in `docs/history/TUNING_REPORT.md` §5 is quoted in. The
//! path from a segment's `gain_db` to that ratio is linear — one gain in a
//! chain of gains — so it is one constant, [`DUPLEX_LEVEL_OFFSET_DB`], measured
//! on the engine and pinned by `tuner/tests/calibration.rs` rather than
//! asserted here. This is the same discipline
//! [`directivity`](crate::estimate::directivity) uses for the pan spread.

use crate::error::{Error, Result};
use crate::preset::{
    DuplexMode, MAX_DUPLEX_GAIN_DB, MAX_DUPLEX_HZ, MAX_DUPLEX_MODES, MAX_DUPLEX_T60_S,
    MIN_DUPLEX_GAIN_DB, MIN_DUPLEX_HZ, MIN_DUPLEX_T60_S,
};
use crate::stft::{find_peaks, Peak, Stft, StftConfig};
use crate::tracker::hann_decay_gain;

/// dB between a segment's `gain_db` and the level it reaches in a render,
/// measured against the loudest sinusoid of a velocity-90 strike of the same
/// key ([`strongest_peak`]).
///
/// **Measured on the engine, not derived**: over 18 dB of `gain_db` the level a
/// render shows moves one dB for one dB to within 0.05, so the whole inversion
/// is this one subtraction. `tuner/tests/calibration.rs` re-measures it; if the
/// engine's gain staging moves, that test fails rather than this constant
/// quietly becoming wrong.
///
/// # Why it is 94 dB and not 30
///
/// The number is a finding, not a unit conversion. `gain_db` is normalised to
/// the segment's *steady* response at its own frequency, so the per-sample
/// input gain the engine builds is `2 G (1 − r)` — about one part in ten
/// thousand at a 1.4 s decay — and the mode has to be *driven up* over its own
/// time constant to reach `G`. `ModalBank`'s culling zeroes a state below
/// `CULL_AMPLITUDE` on every block, which is above where a segment starts, so
/// the mode is zeroed before the drive can raise it and what a render shows is
/// its impulse response rather than its resonant one. That is the 64 dB
/// between the two, and it is why a segment written from a measurement is
/// inaudible in the engine as it stands: see `DECISIONS.md`, and the
/// `(c)` block of the gate test, which fails the day the drive path changes.
pub const DUPLEX_LEVEL_OFFSET_DB: f64 = 93.7;

/// The `gain_db` a measured level asks for, before the schema's own range is
/// applied: `level + `[`DUPLEX_LEVEL_OFFSET_DB`].
///
/// Separate from [`duplex_row`] so that a caller can see how far past the
/// ceiling a whole instrument's measurements land — which, on the Salamander
/// recordings, is where they all land — and shift them together rather than
/// letting each row clamp on its own and lose the relative levels that are the
/// one thing the gate proves recoverable.
pub fn gain_for_level(level_db: f64) -> f64 {
    level_db + DUPLEX_LEVEL_OFFSET_DB
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DuplexConfig {
    /// Analysis window, samples. Short by the tuner's standards: a segment
    /// rings for 0.3–2 s and the fit needs several frames inside that, while
    /// the frequencies involved are 1–8 kHz where nothing crowds.
    pub window: usize,
    /// Advance between frames, samples.
    pub hop: usize,
    /// Peak-picking floor, dB relative to the loudest bin of the same frame
    /// (negative, as `stft::find_peaks` takes it).
    pub floor_db: f64,
    /// How near two peaks in consecutive frames must be to be the same mode,
    /// in cents. A duplex is a fixed resonance and does not glide; this is a
    /// tolerance on the parabolic peak fit, not on the physics.
    pub link_cents: f64,
    /// Frames a chain may go without a peak before it is closed.
    pub max_gap_frames: usize,
    /// Fewest frames a chain needs before a decay may be fitted to it.
    pub min_frames: usize,
    /// A mode must ring at least this long. `PHYSICS.md` §3's own cut: below it
    /// the peak is part of the transient the damper made, not a resonance.
    pub min_t60_s: f64,
    /// A segment starts ringing when the key is released, so its loudest frame
    /// is at the start of the recording. A chain whose peak arrives later than
    /// this is something else — a beat between two resonances, or a chain that
    /// broke and re-formed — and its level, which is read off the fitted line
    /// at the start of the recording, would be an extrapolation over the whole
    /// gap. Rejected rather than extrapolated.
    pub max_onset_s: f64,
    /// The band a segment may be found in.
    ///
    /// The floor is the discriminating parameter of this whole estimator. A
    /// release recording holds two things: the other *speaking lengths* ringing
    /// sympathetically, which live at the pitches of the instrument and which
    /// the resonance bus already models, and the short undamped lengths beyond
    /// the bridge and the agraffe, which are what `notes.duplex` is. Öberg &
    /// Askenfelt's survey and `PHYSICS.md` §3 both put the second at
    /// 1.5–8 kHz, and `docs/history/TUNING_REPORT.md` §5's own centroids for `harmLC3` and
    /// `harmLC5` (314 and 507 Hz) say the first dominates below ~1 kHz. A
    /// candidate below the floor is therefore the halo, not a segment, and
    /// writing it as one would model the same energy twice.
    pub min_hz: f64,
    pub max_hz: f64,
    /// A segment is a *short* piece of the same string, so it cannot sit near
    /// the speaking length's own fundamental however high the note.
    pub min_ratio_to_f0: f64,
    /// A candidate within this of one of the note's own partials is taken to
    /// *be* that partial.
    ///
    /// Deliberately narrow — the width of one analysis window and not a
    /// musical interval — because a real duplex lives exactly where a wide
    /// guard would throw it away. Öberg & Askenfelt's rear-duplex tuning is
    /// sharp of nominal by an average approaching +50 cents, so a guard of half
    /// a semitone excludes the median segment on the instrument they measured.
    /// What separates a segment from a partial here is not frequency but
    /// **decay**: this is a *release* recording, the speaking length is being
    /// damped in it, and `min_t60_s` is what keeps the damped partial out. At
    /// 8192 samples the window resolves 5.9 Hz, which is 10 cents at 1 kHz and
    /// 5 at 2 kHz — below that two peaks are one peak whatever one calls them.
    pub guard_cents: f64,
    /// Candidates more than this below the strongest surviving one are
    /// dropped, however many slots are left.
    pub range_db: f64,
    /// Worst RMS departure from a straight line in log amplitude a chain may
    /// have, dB. A segment is one resonator and fits a line to a hundredth of
    /// a dB; a chain that is really the sidelobe of a strong partial, or two
    /// resonances beating, does not. Cheap, and it is what separates a
    /// resonance from a piece of one.
    pub max_fit_db: f64,
    /// A uniform shift applied to every `gain_db` a row is written with, dB.
    ///
    /// Zero for a measurement that fits inside the schema. It exists because
    /// on real recordings the levels do not: `DUPLEX_LEVEL_OFFSET_DB` is 94 dB
    /// and the measured levels are −26 to −88 dB relative to a strike, so every
    /// segment asks for more than the +6 dB ceiling. Clamping each row on its
    /// own would flatten the instrument to one level and throw away the
    /// relative structure the gate proves is recoverable; one shift for the
    /// whole instrument keeps it.
    pub shift_db: f64,
    /// Longest decay a row may be written with. Shorter than the schema's own
    /// 3 s ceiling on purpose: `PHYSICS.md` §3 asks for 0.5–2 s because
    /// nothing damps these banks, and a release recording is 2–3 s long, so a
    /// fitted decay at the schema's ceiling means the recording ended before
    /// the mode did rather than that the mode rings that long.
    pub max_t60_s: f64,
    /// Most segments to write for one key.
    pub max_modes: usize,
}

impl Default for DuplexConfig {
    fn default() -> Self {
        Self {
            window: 8_192,
            hop: 2_048,
            floor_db: -60.0,
            link_cents: 40.0,
            max_gap_frames: 2,
            min_frames: 4,
            min_t60_s: 0.3,
            max_onset_s: 0.35,
            min_hz: 1_000.0,
            max_hz: 12_000.0,
            min_ratio_to_f0: 3.0,
            guard_cents: 25.0,
            range_db: 24.0,
            max_fit_db: 6.0,
            shift_db: 0.0,
            max_t60_s: 2.0,
            max_modes: MAX_DUPLEX_MODES,
        }
    }
}

/// Most a chain's level may be corrected back to the start of the recording,
/// dB. See [`Chain::fit`].
const MAX_EXTRAPOLATION_DB: f64 = 15.0;

/// One free-tracked resonance of a residual recording.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResidualMode {
    /// Amplitude-weighted mean frequency over the chain, Hz.
    pub hz: f64,
    /// Level at `t = 0` of the analysed signal — the instant the damper
    /// landed — in the units of that signal, read off the fitted decay line
    /// rather than off any one frame.
    ///
    /// It has to be a common instant and not each mode's own peak frame: two
    /// segments with the same level and different decays are already several
    /// dB apart by the time the first analysis window is centred, and ranking
    /// them by what a frame measured would rank them by their T60.
    pub amplitude: f64,
    /// The largest amplitude actually measured in a frame, before any
    /// extrapolation — the sanity check on the line above.
    pub peak_amplitude: f64,
    /// Fitted decay, seconds to −60 dB.
    pub t60_s: f64,
    /// Frames the chain was seen in, and the span it covered.
    pub frames: usize,
    pub span_s: f64,
    /// When the chain was loudest, seconds into the recording.
    pub onset_s: f64,
    /// Residual of the log-amplitude line, in dB — how exponential the decay
    /// was. A segment is a single resonator and fits well; a beat between two
    /// of them, or a peak that is really the tail of a partial, does not.
    pub fit_db: f64,
}

/// Free-tracks every resonance of `signal` and returns them strongest first.
///
/// `exclude_hz` is the struck note's own measured partials: anything within
/// `guard_cents` of one of them is dropped before the ranking, because it is
/// the note and not a segment.
pub fn residual_modes(
    signal: &[f32],
    sample_rate: f64,
    exclude_hz: &[f64],
    config: &DuplexConfig,
) -> Result<Vec<ResidualMode>> {
    residual_modes_above(signal, sample_rate, exclude_hz, 0.0, config)
}

/// [`residual_modes`], with the struck note's fundamental so that the band cut
/// can be relative to it as well as absolute.
pub fn residual_modes_above(
    signal: &[f32],
    sample_rate: f64,
    exclude_hz: &[f64],
    f0_hz: f64,
    config: &DuplexConfig,
) -> Result<Vec<ResidualMode>> {
    let stft = Stft::new(StftConfig::padded(config.window, config.hop, 2)?)?;
    let fft_size = stft.config().fft_size;
    let window_s = stft.config().window_s(sample_rate);

    let mut chains: Vec<Chain> = Vec::new();
    let mut peaks: Vec<Peak> = Vec::new();
    let mut frame = 0usize;
    stft.for_each_frame(signal, sample_rate, |time_s, magnitude| {
        find_peaks(magnitude, sample_rate, fft_size, config.floor_db, &mut peaks);
        // Nearest-peak association, cheapest first: every open chain takes the
        // closest unclaimed peak inside its tolerance, and peaks nobody claimed
        // start chains of their own. There is no seed to disagree with, so
        // "nearest in cents" is the whole rule.
        let mut claimed = vec![false; peaks.len()];
        for chain in chains.iter_mut().filter(|c| c.open(frame, config)) {
            let mut best: Option<(usize, f64)> = None;
            for (i, peak) in peaks.iter().enumerate() {
                if claimed[i] {
                    continue;
                }
                let distance = 1200.0 * (peak.frequency_hz / chain.last_hz).log2();
                if distance.abs() <= config.link_cents
                    && best.map_or(true, |(_, d)| distance.abs() < d)
                {
                    best = Some((i, distance.abs()));
                }
            }
            if let Some((i, _)) = best {
                claimed[i] = true;
                chain.push(frame, time_s, peaks[i].frequency_hz, peaks[i].amplitude);
            }
        }
        for (i, peak) in peaks.iter().enumerate() {
            if !claimed[i] {
                chains.push(Chain::new(frame, time_s, peak.frequency_hz, peak.amplitude));
            }
        }
        frame += 1;
    });

    let mut modes: Vec<ResidualMode> = chains
        .iter()
        .filter(|chain| chain.times.len() >= config.min_frames)
        .filter_map(|chain| chain.fit(window_s))
        .filter(|mode| mode.t60_s >= config.min_t60_s)
        .filter(|mode| mode.onset_s <= config.max_onset_s)
        .filter(|mode| (config.min_hz..=config.max_hz).contains(&mode.hz))
        .filter(|mode| mode.hz >= config.min_ratio_to_f0 * f0_hz)
        .filter(|mode| mode.fit_db <= config.max_fit_db)
        .filter(|mode| {
            exclude_hz
                .iter()
                .all(|&f| f <= 0.0 || (1200.0 * (mode.hz / f).log2()).abs() > config.guard_cents)
        })
        .collect();
    modes.sort_by(|a, b| b.amplitude.total_cmp(&a.amplitude));
    Ok(modes)
}

/// Turns the surviving residual modes of one key into the row a preset writes.
///
/// `reference` is the peak amplitude of a velocity-90 strike of the same key
/// **on the same scale as `signal` was analysed on** — the level every `harm*`
/// figure in `docs/history/TUNING_REPORT.md` §5 is quoted against. A row is capped at
/// `max_modes`, cut at `range_db` below its own strongest member, and every
/// number is clamped into the schema's range rather than being allowed to make
/// a preset the engine will refuse.
///
/// Returns an empty row when the recording has nothing that qualifies. That is
/// the point of the exercise: `PHYSICS.md` §3's survey starts at D4, the bottom
/// of the compass has no duplex worth the name, and a key with no measurement
/// gets no segments rather than a plausible-looking harmonic ratio.
pub fn duplex_row(modes: &[ResidualMode], reference: f64, config: &DuplexConfig) -> Vec<DuplexMode> {
    if reference <= 0.0 {
        return Vec::new();
    }
    let strongest = modes.iter().map(|m| m.amplitude).fold(0.0, f64::max);
    modes
        .iter()
        .filter(|m| m.amplitude > 0.0)
        .filter(|m| 20.0 * (m.amplitude / strongest).log10() >= -config.range_db)
        .filter(|m| (f64::from(MIN_DUPLEX_HZ)..=f64::from(MAX_DUPLEX_HZ)).contains(&m.hz))
        .take(config.max_modes.min(MAX_DUPLEX_MODES))
        .map(|m| DuplexMode {
            hz: m.hz as f32,
            gain_db: (gain_for_level(20.0 * (m.amplitude / reference).log10()) + config.shift_db)
                .clamp(f64::from(MIN_DUPLEX_GAIN_DB), f64::from(MAX_DUPLEX_GAIN_DB))
                as f32,
            t60_s: m
                .t60_s
                .clamp(f64::from(MIN_DUPLEX_T60_S), config.max_t60_s.min(f64::from(MAX_DUPLEX_T60_S)))
                as f32,
        })
        .collect()
}

/// One free-tracked chain of peaks, before it becomes a [`ResidualMode`].
struct Chain {
    last_frame: usize,
    last_hz: f64,
    times: Vec<f64>,
    frequencies: Vec<f64>,
    amplitudes: Vec<f64>,
}

impl Chain {
    fn new(frame: usize, time_s: f64, hz: f64, amplitude: f64) -> Chain {
        Chain {
            last_frame: frame,
            last_hz: hz,
            times: vec![time_s],
            frequencies: vec![hz],
            amplitudes: vec![amplitude],
        }
    }

    fn open(&self, frame: usize, config: &DuplexConfig) -> bool {
        frame - self.last_frame <= config.max_gap_frames + 1
    }

    fn push(&mut self, frame: usize, time_s: f64, hz: f64, amplitude: f64) {
        self.last_frame = frame;
        self.last_hz = hz;
        self.times.push(time_s);
        self.frequencies.push(hz);
        self.amplitudes.push(amplitude);
    }

    /// Amplitude-weighted frequency, and a straight line through the log
    /// amplitudes from the chain's own peak onwards.
    ///
    /// From the peak onwards because a segment is *driven* before it decays —
    /// the damper lands, the bridge hands it energy, and only then is it a free
    /// resonator. Fitting the rise as if it were part of the decay is what
    /// would turn a 1 s segment into a 3 s one.
    fn fit(&self, window_s: f64) -> Option<ResidualMode> {
        let peak = (0..self.amplitudes.len())
            .max_by(|&a, &b| self.amplitudes[a].total_cmp(&self.amplitudes[b]))?;
        let n = self.amplitudes.len() - peak;
        if n < 3 {
            return None;
        }
        let (times, amplitudes) = (&self.times[peak..], &self.amplitudes[peak..]);
        let t0 = times[0];
        // Weighted least squares on `ln a` against `t`, weighted by amplitude:
        // the loud end of a decay is where the measurement is, and the quiet
        // end is where the recording's own floor is.
        let (mut sw, mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for i in 0..n {
            if amplitudes[i] <= 0.0 {
                continue;
            }
            let x = times[i] - t0;
            let y = amplitudes[i].ln();
            let w = amplitudes[i];
            sw += w;
            sx += w * x;
            sy += w * y;
            sxx += w * x * x;
            sxy += w * x * y;
        }
        let denominator = sw * sxx - sx * sx;
        if sw <= 0.0 || denominator.abs() < 1.0e-30 {
            return None;
        }
        let slope = (sw * sxy - sx * sy) / denominator;
        let intercept = (sy - slope * sx) / sw;
        if !slope.is_finite() || slope >= 0.0 {
            return None;
        }
        // The Hann window reads a decaying sinusoid low; the correction is the
        // tracker's own, applied once at the fitted rate rather than per frame
        // because a free chain has no seed to re-estimate from.
        let sigma = -slope;
        let gain = hann_decay_gain(0.5 * sigma * window_s);
        let residual_db: f64 = (0..n)
            .filter(|&i| amplitudes[i] > 0.0)
            .map(|i| {
                let predicted = intercept + slope * (times[i] - t0);
                (8.685_889_638_065_035 * (amplitudes[i].ln() - predicted)).powi(2)
            })
            .sum::<f64>()
            / n as f64;

        let weight: f64 = amplitudes.iter().sum();
        Some(ResidualMode {
            hz: self
                .frequencies
                .iter()
                .zip(&self.amplitudes)
                .map(|(f, a)| f * a)
                .sum::<f64>()
                / self.amplitudes.iter().sum::<f64>().max(f64::MIN_POSITIVE),
            // Read off the fitted line at the start of the recording — a
            // common instant for every chain, because two segments of the same
            // level and different decays are already several dB apart by the
            // time the first analysis window is centred, and ranking them by
            // what a frame measured would rank them by their T60. The
            // correction is capped: `max_onset_s` bounds how far it reaches
            // back, and `MAX_EXTRAPOLATION_DB` bounds what it may claim, so a
            // chain that begins late and decays fast is under-ranked rather
            // than absurdly over-ranked.
            amplitude: (intercept.exp() / gain)
                * 10f64.powf(
                    (8.685_889_638_065_035 * -slope * t0).min(MAX_EXTRAPOLATION_DB) / 20.0,
                ),
            peak_amplitude: amplitudes[0] / gain,
            t60_s: 3.0 * std::f64::consts::LN_10 / sigma,
            frames: self.times.len(),
            span_s: times[n - 1] - t0,
            onset_s: t0,
            fit_db: residual_db.sqrt() * (weight / weight.max(f64::MIN_POSITIVE)),
        })
    }
}

/// The loudest sinusoid this analysis sees anywhere in a signal, in the units
/// [`residual_modes`] reports its own levels in.
///
/// This is what a duplex level is a ratio *to*. It has to be measured the same
/// way as the segment it is compared with — an STFT peak against an STFT peak —
/// because the alternative, a time-domain peak, is the sum of forty partials
/// arriving in phase and its ratio to one sinusoid is a property of the note's
/// spectrum rather than a unit conversion. `docs/history/TUNING_REPORT.md` §5 quotes its
/// `harm*` levels against a time-domain peak; the difference between the two
/// conventions is a constant of the note, and is why
/// [`DUPLEX_LEVEL_OFFSET_DB`] is measured on the engine through this same
/// routine rather than derived.
pub fn strongest_peak(signal: &[f32], sample_rate: f64, config: &DuplexConfig) -> Option<f64> {
    let stft = Stft::new(StftConfig::padded(config.window, config.hop, 2).ok()?).ok()?;
    let fft_size = stft.config().fft_size;
    let mut peaks: Vec<Peak> = Vec::new();
    let mut loudest = 0.0f64;
    stft.for_each_frame(signal, sample_rate, |_, magnitude| {
        find_peaks(magnitude, sample_rate, fft_size, config.floor_db, &mut peaks);
        for peak in &peaks {
            loudest = loudest.max(peak.amplitude);
        }
    });
    (loudest > 0.0).then_some(loudest)
}

/// The struck note's own partials, as frequencies to exclude.
///
/// A convenience for callers that have a preset rather than a measurement: the
/// two-parameter law with the signed fourth-order term, up to `count`.
pub fn partial_frequencies(f0_hz: f64, b: f64, b4: f64, count: u32) -> Vec<f64> {
    (1..=count)
        .map(|k| {
            let k = f64::from(k);
            let radicand = 1.0 + b * k * k + b4 * k * k * k * k;
            if radicand > 0.0 {
                k * f0_hz * radicand.sqrt()
            } else {
                0.0
            }
        })
        .collect()
}

/// How far one key's segments are from the frequencies a harmonic reading would
/// have put them at, in cents.
///
/// The finding `PHYSICS.md` §3 asks a measurement to confirm or refute, made
/// reportable: for each segment, the interval to the *nearest* partial of the
/// note. A duplex tuned to a ratio would return zeros.
pub fn detuning_cents(modes: &[DuplexMode], partials: &[f64]) -> Vec<f64> {
    modes
        .iter()
        .filter_map(|mode| {
            partials
                .iter()
                .filter(|&&f| f > 0.0)
                .map(|&f| 1200.0 * (f64::from(mode.hz) / f).log2())
                .min_by(|a, b| a.abs().total_cmp(&b.abs()))
        })
        .collect()
}

/// Rejects a row whose segments are all within `cents` of a partial, which is
/// what a fit that had quietly re-found the note rather than its duplex would
/// produce.
pub fn assert_not_harmonic(modes: &[DuplexMode], partials: &[f64], cents: f64) -> Result<()> {
    let detunings = detuning_cents(modes, partials);
    if !detunings.is_empty() && detunings.iter().all(|d| d.abs() < cents) {
        return Err(Error::Estimate(format!(
            "every segment landed within {cents} cents of a partial ({detunings:?}), which is \
             the note and not its duplex"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE: f64 = 48_000.0;

    /// A sum of decaying sinusoids at known frequencies, levels and T60s,
    /// starting at `t = 0`.
    fn ring(modes: &[(f64, f64, f64)], seconds: f64) -> Vec<f32> {
        let n = (SAMPLE_RATE * seconds) as usize;
        let mut signal = vec![0.0f32; n];
        for (i, sample) in signal.iter_mut().enumerate() {
            let t = i as f64 / SAMPLE_RATE;
            let mut x = 0.0;
            for &(hz, amplitude, t60) in modes {
                let sigma = 3.0 * std::f64::consts::LN_10 / t60;
                x += amplitude * (-sigma * t).exp() * (std::f64::consts::TAU * hz * t).sin();
            }
            *sample = x as f32;
        }
        signal
    }

    #[test]
    fn a_free_tracker_finds_resonances_nobody_told_it_about() {
        // Deliberately *not* harmonic: 3 f0 + 47 cents and 5 f0 - 31 cents on
        // a 523.25 Hz note, which is the scatter Öberg & Askenfelt measured.
        let f0 = 523.25;
        let a = 4.0 * f0 * 2f64.powf(47.0 / 1200.0);
        let b = 7.0 * f0 * 2f64.powf(-31.0 / 1200.0);
        let signal = ring(&[(a, 0.05, 1.2), (b, 0.02, 0.8)], 3.0);
        let modes = residual_modes(&signal, SAMPLE_RATE, &[], &DuplexConfig::default()).unwrap();
        assert!(modes.len() >= 2, "{modes:?}");
        assert!((modes[0].hz - a).abs() < 2.0, "{:?}", modes[0]);
        assert!((modes[1].hz - b).abs() < 2.0, "{:?}", modes[1]);
        assert!((modes[0].t60_s - 1.2).abs() < 0.15, "{:?}", modes[0]);
        assert!((modes[1].t60_s - 0.8).abs() < 0.15, "{:?}", modes[1]);
        // And the levels come back in the right order and the right ratio.
        let ratio = 20.0 * (modes[0].amplitude / modes[1].amplitude).log10();
        assert!((ratio - 20.0 * (0.05f64 / 0.02).log10()).abs() < 1.5, "{ratio}");
    }

    #[test]
    fn a_short_ring_is_not_a_segment_and_the_notes_own_partials_are_not_either() {
        let f0 = 523.25;
        let segment = 4.0 * f0 * 2f64.powf(47.0 / 1200.0);
        let signal = ring(
            &[
                (segment, 0.05, 1.2),
                // The note's fifth partial, louder than the segment.
                (5.0 * f0, 0.2, 1.0),
                // A thump: loud, and gone inside the cut.
                (3_100.0, 0.3, 0.12),
                // And the halo: another string, below the band a segment
                // lives in, ringing for as long as one would.
                (640.0, 0.4, 1.5),
            ],
            3.0,
        );
        let partials = partial_frequencies(f0, 0.0, 0.0, 8);
        let modes = residual_modes(&signal, SAMPLE_RATE, &partials, &DuplexConfig::default())
            .unwrap();
        assert_eq!(modes.len(), 1, "{modes:?}");
        assert!((modes[0].hz - segment).abs() < 2.0, "{:?}", modes[0]);
    }

    /// The two halves of the level convention: `gain_db` is the measured level
    /// re a strike shifted by the engine's own offset, and a row is capped and
    /// clamped into the schema rather than being allowed to make a preset the
    /// engine refuses.
    #[test]
    fn a_row_is_the_measured_levels_on_the_engines_scale() {
        let modes: Vec<ResidualMode> = [(3_000.0, 0.01), (4_000.0, 0.005), (5_000.0, 1.0e-6)]
            .into_iter()
            .map(|(hz, amplitude)| ResidualMode {
                hz,
                amplitude,
                peak_amplitude: amplitude,
                t60_s: 1.0,
                frames: 20,
                span_s: 1.0,
                onset_s: 0.0,
                fit_db: 0.1,
            })
            .collect();
        let config = DuplexConfig {
            shift_db: -60.0,
            ..DuplexConfig::default()
        };
        let row = duplex_row(&modes, 1.0, &config);
        // The third is 100 dB down on the first, past `range_db`.
        assert_eq!(row.len(), 2);
        // The measured levels are −40 and −46 dB relative to the strike; the
        // schema's ceiling is +6, so a shift of −60 puts them inside it and
        // keeps them 6 dB apart, which is what they were measured to be.
        assert!((f64::from(row[0].gain_db) - (-40.0 + DUPLEX_LEVEL_OFFSET_DB - 60.0)).abs() < 0.05);
        assert!((f64::from(row[1].gain_db) - (-46.02 + DUPLEX_LEVEL_OFFSET_DB - 60.0)).abs() < 0.05);
        assert!((f64::from(row[0].gain_db - row[1].gain_db) - 6.02).abs() < 0.05);
        // A key with no reference level cannot be measured, so it gets nothing.
        assert!(duplex_row(&modes, 0.0, &config).is_empty());
    }

    #[test]
    fn a_row_never_exceeds_what_the_schema_allows() {
        let modes: Vec<ResidualMode> = (0..12)
            .map(|i| ResidualMode {
                hz: 2_000.0 + 100.0 * f64::from(i),
                amplitude: 1.0,
                peak_amplitude: 1.0,
                // Longer than the ceiling, which exists because nothing damps
                // these banks.
                t60_s: 9.0,
                frames: 20,
                span_s: 3.0,
                onset_s: 0.0,
                fit_db: 0.1,
            })
            .collect();
        // A reference far below the segments: `gain_db` would run to +60 dB.
        let row = duplex_row(&modes, 1.0e-3, &DuplexConfig::default());
        assert_eq!(row.len(), MAX_DUPLEX_MODES);
        for mode in &row {
            assert!(mode.gain_db <= MAX_DUPLEX_GAIN_DB);
            assert!(mode.gain_db >= MIN_DUPLEX_GAIN_DB);
            assert!(mode.t60_s <= MAX_DUPLEX_T60_S);
            assert!(mode.hz >= MIN_DUPLEX_HZ && mode.hz <= MAX_DUPLEX_HZ);
        }
    }

    #[test]
    fn a_row_that_is_only_the_note_again_is_refused() {
        let f0 = 261.63;
        let partials = partial_frequencies(f0, 4.0e-4, 0.0, 20);
        let harmonic: Vec<DuplexMode> = partials[8..11]
            .iter()
            .map(|&hz| DuplexMode {
                hz: hz as f32,
                gain_db: -30.0,
                t60_s: 1.0,
            })
            .collect();
        assert!(assert_not_harmonic(&harmonic, &partials, 10.0).is_err());

        let scattered: Vec<DuplexMode> = harmonic
            .iter()
            .map(|m| DuplexMode {
                hz: m.hz * 2f32.powf(50.0 / 1200.0),
                ..*m
            })
            .collect();
        assert!(assert_not_harmonic(&scattered, &partials, 10.0).is_ok());
        let detunings = detuning_cents(&scattered, &partials);
        assert!(detunings.iter().all(|d| (d - 50.0).abs() < 1.0), "{detunings:?}");
    }
}
