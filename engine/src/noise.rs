//! Mechanism noise: the sounds the instrument makes that are not a string.
//!
//! A real piano answers a note-off and a pedal move with an audible thump. Ours
//! did not: a release was a change of damping and nothing else, and a pedal
//! press was silent. `TUNING_REPORT.md` §5 measured what is missing, on the
//! Salamander library's own mechanism recordings and at the level the SFZ plays
//! them: a key-off at **−25 to −39 dB** relative to a velocity-90 strike of the
//! same key, 165–285 ms long and centred at 143–261 Hz; a pedal-down at −36 dB
//! with a six-second 70 Hz rumble; a pedal-up at −42 dB over 0.3 s. That is one
//! event on every release and every pedal move — the most obviously missing
//! sound in playing, and the cheapest thing on the report's backlog.
//!
//! A fifth event was added later and is not an action sound at all: the
//! **strike** — the hammer, the felt and the string's own onset, which
//! `renders/realism/REALISM.md` and `renders/timbre-ladder/ANALYSIS.md` §8.3
//! both convict the engine of missing (attacks +5.2 dB too tonal over six
//! phrases; the first 30 ms 11-13 dB too tonal at all three ladder keys). It is
//! silent by default, because nothing in the library isolates a blow.
//!
//! # The model
//!
//! One [`Burst`] per event: white noise through a band-pass pair and a band
//! limit, under an exponential envelope. Askenfelt's dummy-mass measurements
//! (`PHYSICS.md` §5) are the justification for the four action events' limit —
//! the structure-borne spectrum of the action *ends* around 2 kHz
//! ([`BANDWIDTH_HZ`]) — and the preset's per-event `centroid_hz` places the
//! burst inside it. The band limit is a property of the [`EventShape`] rather
//! than of the module because the strike does not obey it: a hammer on a string
//! radiates directly instead of through the keybed, so `[noise.strike]` states
//! its own `bandwidth_hz` and reaches to 8 kHz.
//!
//! Nothing here allocates, locks or branches on time: a burst is a fixed struct
//! with a handful of floats, and a voice that has no burst running does not
//! execute a single instruction of this module.
//!
//! # Determinism
//!
//! The noise is *not* random from render to render. Each burst is seeded from
//! the key and the frame the event landed on ([`seed_of`]), so a given event
//! list renders the same samples every time, offline and live, on every thread.
//! There is no global RNG state on the audio path — each burst carries its own
//! 32-bit word — which is also what makes two voices sounding at once
//! independent rather than correlated.

use crate::preset::{EventNoise, NoiseAnchor, NoiseTables};
use crate::types::{db_to_amp, interp_anchors, key_position, SAMPLE_RATE};

/// Upper edge of the **action's** spectrum, Hz.
///
/// Askenfelt removed C4's strings, replaced them with a 4 kg dummy mass and
/// measured what the action alone puts into the bridge: the structure-borne
/// spectrum extends only to about 2 kHz (`PHYSICS.md` §5). Fixed rather than
/// tabulated for the four events it describes — it is a property of the
/// instrument's structure, not of the individual event, and every measured
/// centroid in `TUNING_REPORT.md` §5 sits an octave or more below it.
///
/// The strike is not one of those four: a hammer meeting a string radiates
/// directly rather than through the keybed, so `[noise.strike]` carries its own
/// `bandwidth_hz` and this constant does not apply to it.
pub const BANDWIDTH_HZ: f32 = 2_000.0;

/// Q of each of the two band-pass sections.
const BAND_Q: f32 = 1.1;

/// The band-pass pair is centred this far below the centroid the preset asks
/// for.
///
/// A spectral centroid is a first moment in *linear* frequency while a resonant
/// filter is symmetric in log frequency, so a band-pass at `f` returns a
/// centroid well above `f`: the skirt above the peak covers far more hertz than
/// the one below it. The factor is measured, not derived — see
/// `noise::tests::the_burst_lands_near_the_centroid_the_preset_asks_for`, which
/// checks the finished chain against the frequencies `TUNING_REPORT.md` §5
/// tabulates rather than against this constant.
const CENTROID_WARP: f32 = 0.92;

/// Envelope level at which a burst stops running, relative to its own start.
///
/// `decay_s` is the time to −40 dB (which is how `TUNING_REPORT.md` §5 measured
/// it), so this is another 50 dB below that: −90 dB of an event that is itself
/// 25–42 dB below a strike is far under the master chain's noise floor.
const BURST_FLOOR: f32 = 3.162e-5;

/// Peak amplitude below which an event is not rendered at all.
///
/// The mechanism is *always* on — `[noise]`'s default is a measurement, not a
/// neutral value — so the only way to hear the instrument without its action is
/// to write a preset that silences it, and that has to mean silence rather than
/// a very quiet thump: the gate on the sounding path
/// (`engine/tests/preset.rs`'s `the_sounding_path_is_what_it_was_before_the_
/// mechanism`) compares renders sample for sample, and 1e-12 is not 0. Nothing
/// audible is refused: a velocity-90 strike arrives at the board at a couple of
/// hundredths of full scale (`calibrate.rs` measures it per preset), so this
/// floor stands some 160 dB under one.
const SILENT_AMPLITUDE: f32 = 1.0e-9;

/// Drive at which a key event plays at its tabulated level: MIDI release
/// velocity 64, which is what a keyboard sends when it has nothing better and
/// what a MIDI file that carries no release velocity is read as.
pub const NOMINAL_KEY_DRIVE: f32 = 64.0 / 127.0;

/// Drive at which a pedal event plays at its tabulated level: the whole damper
/// rail moving at once, which is what the `pedalD1`/`pedalU1` recordings are.
pub const NOMINAL_PEDAL_DRIVE: f32 = 1.0;

/// Drive at which the hammer's own noise plays at its tabulated level: MIDI
/// velocity 90, the strike every mechanism level in `TUNING_REPORT.md` §5 — and
/// therefore every level in `[noise]` — is quoted against. So a `[noise.strike]`
/// level of −20 dB means −20 dB under the note it is part of, at the velocity
/// that note was measured at.
pub const NOMINAL_STRIKE_DRIVE: f32 =
    crate::preset::NOMINAL_STRIKE_VELOCITY as f32 / 127.0;

/// Seed for the event on `key` at `frame`.
///
/// Both halves are mixed before they meet so that neighbouring keys at the same
/// instant — a released chord — get uncorrelated bursts rather than the same
/// noise at different levels, which would sum as one louder thump.
pub fn seed_of(key: u8, frame: u64) -> u32 {
    let a = (key as u32).wrapping_add(1).wrapping_mul(0x9e37_79b1);
    let b = (frame as u32).wrapping_mul(0x85eb_ca6b);
    let mut x = a ^ b.rotate_left(13);
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    // A xorshift generator seeded with zero stays at zero for ever.
    x | 1
}

/// A transposed-direct-form-II biquad. Coefficients only; the state lives with
/// the burst, so one shape can drive several bursts.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
}

#[derive(Clone, Copy, Debug, Default)]
struct BiquadState {
    s1: f32,
    s2: f32,
}

impl Biquad {
    /// RBJ band-pass, unity gain at the peak.
    fn band_pass(hz: f32, q: f32) -> Biquad {
        let (w, alpha) = Biquad::geometry(hz, q);
        Biquad::normalize(alpha, 0.0, -alpha, 1.0 + alpha, -2.0 * w.cos(), 1.0 - alpha)
    }

    /// RBJ low-pass.
    fn low_pass(hz: f32, q: f32) -> Biquad {
        let (w, alpha) = Biquad::geometry(hz, q);
        let cos = w.cos();
        Biquad::normalize(
            (1.0 - cos) * 0.5,
            1.0 - cos,
            (1.0 - cos) * 0.5,
            1.0 + alpha,
            -2.0 * cos,
            1.0 - alpha,
        )
    }

    /// Digital frequency and bandwidth parameter, with the corner kept clear of
    /// both ends of the spectrum so no preset can put a pole on the unit circle.
    fn geometry(hz: f32, q: f32) -> (f32, f32) {
        let hz = hz.clamp(1.0, 0.45 * SAMPLE_RATE);
        let w = std::f32::consts::TAU * hz / SAMPLE_RATE;
        (w, w.sin() / (2.0 * q.max(0.05)))
    }

    fn normalize(b0: f32, b1: f32, b2: f32, a0: f32, a1: f32, a2: f32) -> Biquad {
        Biquad {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    #[inline]
    fn step(&self, state: &mut BiquadState, x: f32) -> f32 {
        let y = self.b0 * x + state.s1;
        state.s1 = self.b1 * x - self.a1 * y + state.s2;
        state.s2 = self.b2 * x - self.a2 * y;
        y
    }
}

/// The colour and length of one kind of event, resolved once from the preset.
///
/// Shared by every key that plays that event: only the *level* is per key.
#[derive(Clone, Copy, Debug)]
pub struct EventShape {
    band: Biquad,
    limit: Biquad,
    /// Envelope multiplier per sample.
    decay: f32,
    /// Makes a burst triggered at amplitude 1 peak at about 1.
    norm: f32,
}

impl EventShape {
    /// `bandwidth_hz` is the burst's own upper band limit: [`BANDWIDTH_HZ`] for
    /// the four structure-borne action events, and whatever `[noise.strike]`
    /// asks for for the hammer.
    pub fn new(centroid_hz: f32, decay_s: f32, bandwidth_hz: f32) -> EventShape {
        let band = Biquad::band_pass(centroid_hz * CENTROID_WARP, BAND_Q);
        let limit = Biquad::low_pass(bandwidth_hz, std::f32::consts::FRAC_1_SQRT_2);
        // `decay_s` is the time to -40 dB, so the per-sample multiplier is the
        // 1/(decay_s * fs)-th root of 1e-2.
        let decay = 10.0f32.powf(-2.0 / (decay_s.max(1.0e-3) * SAMPLE_RATE));
        let mut shape = EventShape {
            band,
            limit,
            decay,
            norm: 1.0,
        };
        shape.norm = 1.0 / shape.burst_peak();
        shape
    }

    /// Peak of a whole burst of this shape, triggered at amplitude 1.
    ///
    /// Setup-time only, and deterministic. The level a preset asks for is a
    /// *peak* level — that is the column `TUNING_REPORT.md` §5 reports — and
    /// the peak of a filtered, enveloped noise burst is not something the
    /// coefficients give you in closed form. It matters that the envelope is in
    /// here and not just the filter: a band-pass at 120 Hz needs a couple of
    /// cycles to ring up, by which time a 0.24 s envelope has already lost
    /// 2 dB, so the burst's real peak is several dB under its steady state and
    /// normalising against the steady state alone would leave every event
    /// quiet by an amount that depended on its own centroid.
    fn burst_peak(&self) -> f32 {
        // Long enough to contain the peak of any shape (the envelope only
        // falls), short enough that the six-second pedal rumble does not make
        // building an engine noticeably slower.
        const WINDOW: usize = 48_000;
        // The peak of one realization of a narrow noise band is itself a random
        // number — a 100 Hz band under a 0.2 s envelope only gets a couple of
        // dozen independent excursions, and their maximum scatters by 2-3 dB.
        // Averaging a few draws makes the *design* level exact even though each
        // individual event still lands where its own seed puts it, which is
        // what a mechanism whose thumps are all identical would get wrong.
        const DRAWS: usize = 8;
        let mut total = 0.0f32;
        for draw in 0..DRAWS {
            let mut burst = Burst::new();
            burst.rng = seed_of(0, draw as u64);
            let mut gain = 1.0f32;
            let mut peak = 0.0f32;
            for _ in 0..WINDOW {
                let x = burst.white();
                let y = self.band.step(&mut burst.band_a, x);
                let y = self.band.step(&mut burst.band_b, y);
                let y = self.limit.step(&mut burst.limit_state, y);
                peak = peak.max((gain * y).abs());
                gain *= self.decay;
            }
            total += peak;
        }
        (total / DRAWS as f32).max(1.0e-12)
    }
}

/// The five event shapes of one preset, built once.
///
/// A shape costs a few hundred thousand filter steps to normalise, and every
/// one of the 88 voices would otherwise build the same three — so they are built
/// once per engine and copied into the voices, which is also what the comment
/// on [`EventShape`] promises.
#[derive(Clone, Copy, Debug)]
pub struct NoiseShapes {
    pub key_off: EventShape,
    pub damper_lift: EventShape,
    pub pedal_down: EventShape,
    pub pedal_up: EventShape,
    pub strike: EventShape,
}

impl NoiseShapes {
    pub fn new(noise: &NoiseTables) -> NoiseShapes {
        // The four action events share the structure-borne ceiling; the hammer
        // does not, and says so in the file.
        let shape = |e: &EventNoise| EventShape::new(e.centroid_hz, e.decay_s, BANDWIDTH_HZ);
        NoiseShapes {
            key_off: shape(&noise.key_off),
            damper_lift: shape(&noise.damper_lift),
            pedal_down: shape(&noise.pedal_down),
            pedal_up: shape(&noise.pedal_up),
            strike: EventShape::new(
                noise.strike.centroid_hz,
                noise.strike.decay_s,
                noise.strike.bandwidth_hz,
            ),
        }
    }
}

/// One event's level law, resolved for one key (or, for the pedal events, for
/// the instrument as a whole).
#[derive(Clone, Copy, Debug)]
pub struct EventModel {
    shape: EventShape,
    /// Linear peak amplitude at [`EventModel::reference`], in the soundboard's
    /// input units.
    level: f32,
    velocity_db: f32,
    reference: f32,
}

impl EventModel {
    /// Resolves `spec` for `key` — which the pedal events, whose level table is
    /// a single global anchor, simply read past.
    ///
    /// `strike_reference` is the board-input amplitude that renders at the same
    /// output peak as a velocity-90 strike of this key, which is the level
    /// `TUNING_REPORT.md` §5's table calls 0 dB; `calibrate.rs` measures it per
    /// preset, per key and per event shape.
    pub fn new(
        spec: &EventNoise,
        shape: EventShape,
        key: u8,
        drive_reference: f32,
        strike_reference: f32,
    ) -> EventModel {
        EventModel::from_levels(
            &spec.level_db,
            spec.velocity_db,
            shape,
            key,
            drive_reference,
            strike_reference,
        )
    }

    /// The same, for an event whose spec is not an [`EventNoise`] — the strike,
    /// which carries a band limit of its own. Only the level law lives here, so
    /// there is one of it.
    pub fn from_levels(
        level_db: &[NoiseAnchor],
        velocity_db: f32,
        shape: EventShape,
        key: u8,
        drive_reference: f32,
        strike_reference: f32,
    ) -> EventModel {
        let anchors: Vec<(f32, f32)> = level_db
            .iter()
            .map(|a| (key_position(a.key), a.db))
            .collect();
        EventModel {
            shape,
            level: strike_reference * db_to_amp(interp_anchors(key_position(key), &anchors)),
            velocity_db,
            reference: drive_reference,
        }
    }

    /// An event of this shape at unit peak amplitude and with no velocity
    /// tracking: what the calibration drives the soundboard with to find out
    /// what the board does to a burst of this colour.
    pub fn unit(shape: EventShape) -> EventModel {
        EventModel {
            shape,
            level: 1.0,
            velocity_db: 0.0,
            reference: 0.0,
        }
    }

    /// Peak amplitude of the event at `drive` — release velocity over 127 for a
    /// key event, the fraction of the dampers that move for a pedal one.
    ///
    /// The tracking is a straight line in dB, `velocity_db` per unit of drive,
    /// through the tabulated level at the reference drive. Salamander's own
    /// release group tracks velocity (`amp_veltrack = 82`) and Pianoteq's
    /// Blüthner spans 12 dB over note-off velocity, which is where the default
    /// slope comes from.
    pub fn amplitude(&self, drive: f32) -> f32 {
        self.level * db_to_amp(self.velocity_db * (drive.clamp(0.0, 1.0) - self.reference))
    }

    pub fn shape(&self) -> &EventShape {
        &self.shape
    }
}

/// One filtered-noise event, running or not.
///
/// Retriggering keeps the filter state and only restarts the envelope and the
/// generator: a second event arriving while the first is still ringing is a
/// louder continuation, not a discontinuity, so a trill cannot click.
pub struct Burst {
    rng: u32,
    /// Current envelope amplitude; zero when the burst is not running.
    gain: f32,
    /// Level at which the envelope is switched off, fixed when the burst was
    /// triggered. Relative to the burst's own amplitude and not absolute, so a
    /// quiet event stops as promptly as a loud one.
    floor: f32,
    decay: f32,
    norm: f32,
    band: Biquad,
    limit: Biquad,
    band_a: BiquadState,
    band_b: BiquadState,
    limit_state: BiquadState,
}

impl Burst {
    pub fn new() -> Burst {
        Burst {
            rng: 1,
            gain: 0.0,
            floor: 0.0,
            decay: 0.0,
            norm: 1.0,
            band: Biquad::default(),
            limit: Biquad::default(),
            band_a: BiquadState::default(),
            band_b: BiquadState::default(),
            limit_state: BiquadState::default(),
        }
    }

    /// Starts the event described by `model` at `drive`, seeded with `seed`.
    ///
    /// A burst that is still ringing keeps its filter state, which is what
    /// makes a retrigger a louder continuation rather than a discontinuity. One
    /// that has already stopped does not: the filters run on *un-enveloped*
    /// white noise and only the output is scaled, so the state a finished burst
    /// leaves behind is of order one however quiet that burst had become when
    /// it stopped. Released into the next event it is a foreign transient
    /// exactly where that event's peak is — measured on the release of a
    /// silently pressed key, whose damper-lift burst is 0.2 s dead by then:
    /// 1.4 dB of extra peak on average, and a key-off that did not render like
    /// the one [`EventShape::new`] normalised.
    pub fn trigger(&mut self, model: &EventModel, drive: f32, seed: u32) {
        let amplitude = model.amplitude(drive);
        if !amplitude.is_finite() || amplitude < SILENT_AMPLITUDE {
            return;
        }
        if !self.is_active() {
            self.band_a = BiquadState::default();
            self.band_b = BiquadState::default();
            self.limit_state = BiquadState::default();
        }
        self.rng = seed;
        self.gain = amplitude;
        self.floor = BURST_FLOOR * amplitude;
        self.decay = model.shape.decay;
        self.norm = model.shape.norm;
        self.band = model.shape.band;
        self.limit = model.shape.limit;
    }

    /// True while the burst still contributes anything audible.
    pub fn is_active(&self) -> bool {
        self.gain > 0.0
    }

    /// **Adds** the burst's next `out.len()` samples into `out`.
    pub fn add(&mut self, out: &mut [f32]) {
        if !self.is_active() {
            return;
        }
        let (mut band_a, mut band_b, mut limit) = (self.band_a, self.band_b, self.limit_state);
        let mut gain = self.gain;
        for sample in out.iter_mut() {
            let x = self.white();
            let y = self.band.step(&mut band_a, x);
            let y = self.band.step(&mut band_b, y);
            let y = self.limit.step(&mut limit, y);
            *sample += self.norm * gain * y;
            gain *= self.decay;
        }
        self.band_a = band_a;
        self.band_b = band_b;
        self.limit_state = limit;
        self.gain = if gain > self.floor { gain } else { 0.0 };
    }

    pub fn reset(&mut self) {
        self.gain = 0.0;
        self.band_a = BiquadState::default();
        self.band_b = BiquadState::default();
        self.limit_state = BiquadState::default();
    }

    /// xorshift32, mapped to [-1, 1). Twenty-four bits is the whole of an
    /// `f32`'s mantissa; the rest of the word would not survive the conversion.
    #[inline]
    fn white(&mut self) -> f32 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 17;
        self.rng ^= self.rng << 5;
        (self.rng >> 8) as f32 * (2.0 / 16_777_216.0) - 1.0
    }
}

impl Default for Burst {
    fn default() -> Self {
        Burst::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibrate::MechanismCalibration;
    use crate::preset::Preset;
    use crate::types::{amp_to_db, BLOCK};

    fn render(model: &EventModel, drive: f32, seconds: f32) -> Vec<f32> {
        render_seeded(model, drive, seconds, seed_of(60, 0))
    }

    fn render_seeded(model: &EventModel, drive: f32, seconds: f32, seed: u32) -> Vec<f32> {
        let mut burst = Burst::new();
        burst.trigger(model, drive, seed);
        let mut out = vec![0.0f32; (seconds * SAMPLE_RATE) as usize];
        for block in out.chunks_mut(BLOCK) {
            burst.add(block);
        }
        out
    }

    fn peak(x: &[f32]) -> f32 {
        x.iter().fold(0.0f32, |m, &v| m.max(v.abs()))
    }

    /// Power-weighted mean frequency of `x`, by direct evaluation of the DFT on
    /// a coarse grid — enough to place a broad noise band.
    fn centroid(x: &[f32]) -> f32 {
        let n = x.len().min(8192);
        let (mut num, mut den) = (0.0f64, 0.0f64);
        let mut f = 20.0f64;
        while f < 8_000.0 {
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for (i, &v) in x[..n].iter().enumerate() {
                let phase = std::f64::consts::TAU * f * i as f64 / SAMPLE_RATE as f64;
                re += v as f64 * phase.cos();
                im -= v as f64 * phase.sin();
            }
            let power = re * re + im * im;
            num += f * power;
            den += power;
            f *= 1.03;
        }
        (num / den.max(1e-30)) as f32
    }

    fn model(key: u8) -> EventModel {
        let preset = Preset::default();
        let shapes = NoiseShapes::new(&preset.noise);
        let calibration = MechanismCalibration::new(&preset, &shapes);
        EventModel::new(
            &preset.noise.key_off,
            shapes.key_off,
            key,
            NOMINAL_KEY_DRIVE,
            calibration.key_off(key),
        )
    }

    #[test]
    fn an_untriggered_burst_is_bit_exact_silence() {
        let mut burst = Burst::new();
        let mut out = [1.0f32; BLOCK];
        assert!(!burst.is_active());
        burst.add(&mut out);
        assert!(out.iter().all(|&x| x == 1.0), "an idle burst wrote samples");
    }

    /// The whole point of seeding from the event: two renders of the same
    /// performance are the same samples, and two keys released together are not
    /// the same noise twice.
    #[test]
    fn bursts_are_deterministic_per_event_and_independent_across_keys() {
        let m = model(60);
        let mut a = Burst::new();
        let mut b = Burst::new();
        let (mut ya, mut yb) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
        a.trigger(&m, 0.5, seed_of(60, 4096));
        b.trigger(&m, 0.5, seed_of(60, 4096));
        a.add(&mut ya);
        b.add(&mut yb);
        assert_eq!(ya, yb, "the same event rendered differently");

        let mut c = Burst::new();
        let mut yc = [0.0f32; BLOCK];
        c.trigger(&m, 0.5, seed_of(61, 4096));
        c.add(&mut yc);
        assert!(ya != yc, "neighbouring keys got the same noise");
    }

    /// An event that follows a *finished* one has to sound like the first event
    /// of its life, and an event that interrupts a running one has not.
    ///
    /// The filters run on un-enveloped white noise and only the sum is scaled,
    /// so a burst that has stopped leaves states of order one behind however
    /// quiet it had become — released into the next event they are a foreign
    /// transient at exactly the moment its peak arrives. Measured before this
    /// was fixed: every release of a *silently pressed* key, whose damper-lift
    /// burst is long dead by then, peaked 1.4 dB above the same release on a key
    /// that had not been touched.
    #[test]
    fn a_burst_that_follows_a_finished_one_starts_from_silence() {
        let preset = Preset::default();
        let shapes = NoiseShapes::new(&preset.noise);
        let calibration = MechanismCalibration::new(&preset, &shapes);
        let lift = EventModel::new(
            &preset.noise.damper_lift,
            shapes.damper_lift,
            60,
            NOMINAL_KEY_DRIVE,
            calibration.damper_lift(60),
        );
        let key_off = model(60);
        let seed = seed_of(60, 4096);

        // A lift, run until it stops, and then a key-off on the same burst.
        let mut used = Burst::new();
        used.trigger(&lift, 1.0, seed_of(60, 0));
        let mut scratch = vec![0.0f32; (0.4 * SAMPLE_RATE) as usize];
        for block in scratch.chunks_mut(BLOCK) {
            used.add(block);
        }
        assert!(!used.is_active(), "the lift was still running");
        used.trigger(&key_off, NOMINAL_KEY_DRIVE, seed);

        let mut fresh = Burst::new();
        fresh.trigger(&key_off, NOMINAL_KEY_DRIVE, seed);

        let (mut a, mut b) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
        for _ in 0..8 {
            a.fill(0.0);
            b.fill(0.0);
            used.add(&mut a);
            fresh.add(&mut b);
            assert_eq!(a, b, "a burst after a finished one did not start from rest");
        }

        // ... while a burst that is retriggered *while it rings* keeps its
        // state, which is what stops a trill from clicking.
        let mut running = Burst::new();
        running.trigger(&key_off, NOMINAL_KEY_DRIVE, seed_of(60, 1));
        running.add(&mut a);
        assert!(running.is_active());
        running.trigger(&key_off, NOMINAL_KEY_DRIVE, seed);
        let mut still_fresh = Burst::new();
        still_fresh.trigger(&key_off, NOMINAL_KEY_DRIVE, seed);
        a.fill(0.0);
        b.fill(0.0);
        running.add(&mut a);
        still_fresh.add(&mut b);
        assert!(a != b, "the retrigger discarded a ringing burst's state");
    }

    /// The measured column the defaults come from: 165–285 ms to −40 dB for a
    /// key-off, 0.32 s for a pedal-up, 5.76 s for the pedal-down rumble.
    #[test]
    fn a_burst_decays_over_the_time_the_preset_asks_for() {
        let noise = Preset::default().noise;
        let shapes = NoiseShapes::new(&noise);
        for (spec, shape, reference) in [
            (&noise.key_off, shapes.key_off, NOMINAL_KEY_DRIVE),
            (&noise.pedal_up, shapes.pedal_up, NOMINAL_PEDAL_DRIVE),
        ] {
            let model = EventModel::new(spec, shape, 60, reference, 1.0);
            let y = render(&model, reference, 6.0 * spec.decay_s);
            // Energy from an instant to the end of the burst, which for an
            // exponential envelope is proportional to the envelope there
            // squared. Taken over the whole tail rather than over a short
            // window: a hundred-hertz noise band gets only a couple of dozen
            // independent excursions per envelope time constant, and a short
            // window would be measuring that scatter instead of the decay.
            let tail = |from: f32| -> f64 {
                let a = (from * spec.decay_s * SAMPLE_RATE) as usize;
                y[a..].iter().map(|&v| (v as f64) * (v as f64)).sum()
            };
            let db = (10.0 * (tail(1.5) / tail(0.5)).log10()) as f32;
            assert!(
                (-42.5..-37.5).contains(&db),
                "over decay_s = {} s the burst fell {db:.1} dB, expected -40",
                spec.decay_s
            );
        }
    }

    /// The spectral centroid the preset names is the one the burst plays at —
    /// which is what makes `TUNING_REPORT.md` §5's 143–261 Hz column a
    /// parameter rather than a decoration — and nothing survives above the
    /// action's own 2 kHz ceiling.
    #[test]
    fn the_burst_lands_near_the_centroid_the_preset_asks_for() {
        for &asked in &[77.0f32, 143.0, 190.0, 261.0, 500.0] {
            let shape = EventShape::new(asked, 0.25, BANDWIDTH_HZ);
            let m = EventModel {
                shape,
                level: 1.0,
                velocity_db: 0.0,
                reference: 0.0,
            };
            let y = render(&m, 1.0, 0.2);
            let measured = centroid(&y);
            let ratio = measured / asked;
            println!("centroid: asked {asked} got {measured:.1} ratio {ratio:.3}");
            assert!(
                (0.8..1.25).contains(&ratio),
                "asked for a centroid of {asked} Hz and got {measured:.0} Hz"
            );
        }
        // ... and the band limit is real: a burst asked for 500 Hz has almost
        // nothing an octave and a half above it.
        let y = render(
            &EventModel {
                shape: EventShape::new(500.0, 0.25, BANDWIDTH_HZ),
                level: 1.0,
                velocity_db: 0.0,
                reference: 0.0,
            },
            1.0,
            0.2,
        );
        let band = |lo: f32, hi: f32| {
            let mut power = 0.0f64;
            let mut f = lo;
            while f < hi {
                let (mut re, mut im) = (0.0f64, 0.0f64);
                for (i, &v) in y[..8192].iter().enumerate() {
                    let phase = std::f64::consts::TAU * f as f64 * i as f64 / SAMPLE_RATE as f64;
                    re += v as f64 * phase.cos();
                    im -= v as f64 * phase.sin();
                }
                power += re * re + im * im;
                f *= 1.03;
            }
            power
        };
        let inside = band(200.0, 1_000.0);
        let above = band(4_000.0, 16_000.0);
        assert!(
            above < inside * 1.0e-3,
            "the 2 kHz band limit leaked: {above:e} above against {inside:e} inside"
        );
    }

    /// Power of `x` between `lo` and `hz`, on the same coarse log grid the
    /// centroid uses.
    fn band_power(x: &[f32], lo: f32, hi: f32) -> f64 {
        let n = x.len().min(8192);
        let mut power = 0.0f64;
        let mut f = lo;
        while f < hi {
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for (i, &v) in x[..n].iter().enumerate() {
                let phase = std::f64::consts::TAU * f as f64 * i as f64 / SAMPLE_RATE as f64;
                re += v as f64 * phase.cos();
                im -= v as f64 * phase.sin();
            }
            power += re * re + im * im;
            f *= 1.03;
        }
        power
    }

    /// The band limit is per event, and the four action events keep the 2 kHz
    /// ceiling Askenfelt measured while the strike does not.
    ///
    /// The strike is the one event that is not structure-borne — a hammer on a
    /// string radiates directly — and the residual the timbre ladder finds
    /// missing is broadband at every key and centred *above* 2 kHz at C6. An
    /// event held to the action's ceiling cannot supply it, which is why
    /// `bandwidth_hz` is a field of the event rather than a constant of the
    /// module.
    #[test]
    fn the_strike_carries_its_own_band_limit_and_the_action_events_keep_theirs() {
        let mut preset = Preset::default();
        preset.noise.strike.centroid_hz = 1_500.0;
        preset.noise.strike.bandwidth_hz = 6_000.0;
        preset.noise.strike.decay_s = 0.06;
        assert!(preset.validate().is_ok());
        let shapes = NoiseShapes::new(&preset.noise);

        // Whatever the strike asks for, the action's four are the fixed 2 kHz.
        let action_limit = Biquad::low_pass(BANDWIDTH_HZ, std::f32::consts::FRAC_1_SQRT_2);
        for shape in [
            shapes.key_off,
            shapes.damper_lift,
            shapes.pedal_down,
            shapes.pedal_up,
        ] {
            assert_eq!(shape.limit, action_limit, "an action event moved its ceiling");
        }
        assert_ne!(shapes.strike.limit, action_limit);
        assert_eq!(
            shapes.strike.limit,
            Biquad::low_pass(6_000.0, std::f32::consts::FRAC_1_SQRT_2)
        );

        // ... and it is audible in the render, not only in the coefficients.
        let render_shape = |shape: EventShape| {
            render(
                &EventModel {
                    shape,
                    level: 1.0,
                    velocity_db: 0.0,
                    reference: 0.0,
                },
                1.0,
                0.2,
            )
        };
        let broad = render_shape(EventShape::new(3_000.0, 0.06, 6_000.0));
        let narrow = render_shape(EventShape::new(3_000.0, 0.06, BANDWIDTH_HZ));
        let above = |y: &[f32]| band_power(y, 3_500.0, 8_000.0) / band_power(y, 500.0, 2_000.0);
        let ratio = above(&broad) / above(&narrow);
        println!("strike band ratio {ratio:.1}x");
        // Measured 10.3x — two second-order sections' worth of difference over
        // 3.5-8 kHz. The threshold is well under it because what is being
        // asserted is that the ceiling moved, not by how much.
        assert!(
            ratio > 6.0,
            "the strike's 6 kHz band holds only {ratio:.1}x what a 2 kHz-limited \
             burst of the same colour does"
        );
        // The centroid the preset names is still the one it plays.
        let measured = centroid(&render_shape(shapes.strike));
        assert!(
            (0.8..1.25).contains(&(measured / 1_500.0)),
            "asked for a 1500 Hz centroid and got {measured:.0} Hz"
        );
    }

    /// The strike's level means what the other four events' levels mean: a peak
    /// in dB relative to a velocity-90 strike of the same key, measured through
    /// the same output-referenced calibration.
    #[test]
    fn the_strike_burst_plays_at_the_level_the_preset_asks_for() {
        const WANT_DB: f32 = -20.0;
        let mut preset = Preset::default();
        preset.noise.strike.centroid_hz = 1_500.0;
        preset.noise.strike.bandwidth_hz = 6_000.0;
        preset.noise.strike.decay_s = 0.06;
        preset.noise.strike.level_db = vec![crate::preset::NoiseAnchor {
            key: 21,
            db: WANT_DB,
        }];
        assert!(preset.validate().is_ok());

        let shapes = NoiseShapes::new(&preset.noise);
        let calibration = MechanismCalibration::new(&preset, &shapes);
        let reference = calibration.strike(60);
        let model = EventModel::from_levels(
            &preset.noise.strike.level_db,
            preset.noise.strike.velocity_db,
            shapes.strike,
            60,
            NOMINAL_STRIKE_DRIVE,
            reference,
        );
        // Averaged over seeds: the peak of one realization of a noise band is
        // itself a random number, and that scatter is the feature.
        let mean: f32 = (0..8)
            .map(|i| amp_to_db(peak(&render_seeded(&model, NOMINAL_STRIKE_DRIVE, 0.3, seed_of(60, i * 4096)))))
            .sum::<f32>()
            / 8.0;
        let re_strike = mean - amp_to_db(reference);
        println!("strike burst: {re_strike:.2} dB re a strike, asked for {WANT_DB}");
        assert!(
            (re_strike - WANT_DB).abs() < 2.5,
            "the strike noise is {re_strike:.1} dB re a strike, expected {WANT_DB}"
        );

        // Velocity reaches it: the tabulated level is the level at velocity 90,
        // and a fortissimo blow is louder.
        let soft = amp_to_db(peak(&render(&model, 40.0 / 127.0, 0.3)));
        let hard = amp_to_db(peak(&render(&model, 1.0, 0.3)));
        assert!(
            soft < hard - 6.0,
            "velocity barely moved the strike noise: {soft:.1} / {hard:.1} dB"
        );
    }

    /// The neutral level is silence, and silence has to be *bit-exact* silence:
    /// the default preset must render the note it always rendered.
    #[test]
    fn a_silent_strike_level_never_starts_a_burst() {
        let preset = Preset::default();
        assert_eq!(preset.noise.strike.level_db[0].db, crate::preset::SILENT_LEVEL_DB);
        let shapes = NoiseShapes::new(&preset.noise);
        let calibration = MechanismCalibration::new(&preset, &shapes);
        let model = EventModel::from_levels(
            &preset.noise.strike.level_db,
            preset.noise.strike.velocity_db,
            shapes.strike,
            60,
            NOMINAL_STRIKE_DRIVE,
            calibration.strike(60),
        );
        let mut burst = Burst::new();
        // The hardest blow there is, which is where the velocity law puts the
        // level highest.
        burst.trigger(&model, 1.0, seed_of(60, 0));
        assert!(!burst.is_active(), "a -200 dB strike started a burst");
        let mut out = [1.0f32; BLOCK];
        burst.add(&mut out);
        assert!(out.iter().all(|&x| x == 1.0), "a silent strike wrote samples");
    }

    /// Release velocity has to reach the level, or the key-off thump would be
    /// the same sound however the key was let go.
    #[test]
    fn release_velocity_moves_the_level_and_the_nominal_one_hits_the_table() {
        let m = model(60);
        let soft = amp_to_db(peak(&render(&m, 0.0, 0.3)));
        let nominal = amp_to_db(peak(&render(&m, NOMINAL_KEY_DRIVE, 0.3)));
        let hard = amp_to_db(peak(&render(&m, 1.0, 0.3)));
        assert!(
            soft < nominal - 4.0 && nominal < hard - 4.0,
            "release velocity barely moved the level: {soft:.1} / {nominal:.1} / {hard:.1} dB"
        );
        // The whole span is the preset's `velocity_db`.
        let span = hard - soft;
        assert!(
            (span - Preset::default().noise.key_off.velocity_db).abs() < 0.5,
            "velocity span {span:.1} dB"
        );
        // ... and at the nominal drive the burst peaks at the tabulated level
        // relative to a strike, which is the number `TUNING_REPORT.md` §5
        // reports for C4: -35.4 dB. Averaged over seeds, because the peak of
        // one realization of a noise band scatters by a couple of dB and that
        // scatter is a feature: no two releases of a real key are identical.
        // Two means over eight draws each still differ by a dB or so, which is
        // why the window below is wider than the arithmetic suggests.
        let mean: f32 = (0..8)
            .map(|i| {
                amp_to_db(peak(&render_seeded(
                    &m,
                    NOMINAL_KEY_DRIVE,
                    0.3,
                    seed_of(60, i * 4096),
                )))
            })
            .sum::<f32>()
            / 8.0;
        // ... where "the tabulated level" is measured against the amplitude
        // the calibration says a velocity-90 strike of this key comes to
        // (`calibrate.rs`), which is what the table's dB are relative to. That
        // the *rendered* thump then lands there through the board is
        // `acceptance::a_note_off_thumps_at_the_level_the_recordings_measured`.
        let preset = Preset::default();
        let shapes = NoiseShapes::new(&preset.noise);
        let reference = MechanismCalibration::new(&preset, &shapes).key_off(60);
        let re_strike = mean - amp_to_db(reference);
        assert!(
            (re_strike + 35.4).abs() < 2.5,
            "C4 key-off is {re_strike:.1} dB re a strike, expected -35.4"
        );
    }
}
