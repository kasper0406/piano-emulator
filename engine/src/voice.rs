//! One voice per key: the unison string group, its hammer, and the damper
//! lifecycle that ties them to the pedals.
//!
//! A voice is never stolen. Re-striking a ringing key must not reset the modal
//! banks — the new hammer pulse adds into the state that is still there, which
//! is what happens physically and what makes repeated notes under the sustain
//! pedal sound right.

use crate::calibrate::MechanismCalibration;
use crate::duplex::DuplexBank;
use crate::hammer::Hammer;
use crate::noise::{self, Burst, EventModel, NoiseShapes};
use crate::pedal::PedalState;
use crate::preset::Preset;
use crate::resonance::ResonanceBus;
use crate::soundboard::{pan_for_key, Soundboard};
use crate::string::PianoString;
use crate::types::{key_index, BLOCK, DEFAULT_RELEASE_VELOCITY, SAMPLE_RATE};

/// Time constant of the damper engage/release ramp at the nominal release
/// velocity. The felt takes a few milliseconds to settle onto the string;
/// stepping the damping would click.
const DAMPER_RAMP_SECONDS: f32 = 0.010;

/// How much slower the damper falls at the slowest release than at the nominal
/// one, and — the same factor the other way — how much faster at the fastest.
///
/// A release is not a switch: how fast the key comes back sets how fast the
/// damper meets the string, which is the difference between a chord that stops
/// and one that is *let* go (`PHYSICS.md` §6). The span is 50 ms at MIDI
/// release velocity 1 down to 2 ms at 127, through the 10 ms the engine used to
/// apply unconditionally.
const DAMPER_RAMP_RANGE: f32 = 5.0;

/// How far into the string's swing the felt reaches when the damper is fully
/// seated, as a fraction of the level the note had when the damper arrived.
///
/// A damper that is touching but not seated is not merely extra `σ`: Lehtonen,
/// Askenfelt & Välimäki measured the acoustic signal and the damper's own
/// acceleration through a part-pedalled note and found three intervals — free
/// vibration, damper-string interaction, free vibration again — where the
/// middle one decays fast *and changes the timbre*, because the felt limits the
/// string's deflection nonlinearly at its position (`PHYSICS.md` §6). A linear
/// damper cannot do that at any `σ`.
///
/// The limit travels geometrically with the damper's position — full swing when
/// the felt is clear of the string, this fraction of it when the damper is
/// down — so a half-pedal sits at the geometric mean, a quarter of the string's
/// swing, which is deep enough to fold the waveform and not so deep that it
/// gates the note.
///
/// The travel is *continuous through both of its ends*, and that is a
/// correctness property rather than a refinement (`DECISIONS.md` 218). The
/// damper's position advances once per block, so a threshold recomputed from it
/// and held for the block is a staircase: at the nominal release the position
/// moves 0.267 per block and the exponent turns that into a 13.3 dB jump in the
/// threshold at each of the ramp's block boundaries, with a fourth and larger
/// one where the old code stopped limiting at a seated damper and the output
/// snapped back from `soft_limit(x, 0.01 · reference)` to `x` between two
/// adjacent samples. [`Voice::felt_ramp`] walks the threshold geometrically
/// *within* the block instead, and it retires the limit over the damper's own
/// ramp once the damper has landed rather than switching it off — the
/// interaction interval ends where Lehtonen's third one begins, and an interval
/// that ends in 12 ms is a sound, where one that ends between two samples is a
/// click.
const FELT_CLEARANCE: f32 = 0.01;

/// Time constant of the peak follower that remembers how loud the note was when
/// the damper started to arrive.
const FELT_FOLLOWER_S: f32 = 0.05;

/// Mixed into the frame before the strike burst's seed is derived.
///
/// [`noise::seed_of`] answers "the event on this key at this frame", and a
/// note-on and a note-off of one key *can* land on the same block — a repeated
/// note under one MIDI file's quantisation does it routinely. Without a salt the
/// two bursts would then run the same noise at two levels and sum as one louder
/// event, which is the same failure the key half of `seed_of` exists to prevent
/// between neighbouring keys.
const STRIKE_SEED_SALT: u64 = 0x9e37_79b9_7f4a_7c15;

pub struct Voice {
    key: u8,
    index: usize,
    pan: f32,
    /// Where the two polarizations are placed when the preset spreads them.
    /// Both equal `pan` when it does not.
    pan_vertical: f32,
    pan_horizontal: f32,
    /// True when the polarizations are placed apart and have to be rendered
    /// apart. Decided at construction: the audio path only reads it.
    spread: bool,
    string: PianoString,
    /// The key's duplex and aliquot segments. Empty for every key of a preset
    /// that has no `notes.duplex` table, and then never touched.
    duplex: DuplexBank,
    duplex_out: [f32; BLOCK],
    hammer: Hammer,
    held: bool,
    damper_current: f32,
    damper_target: f32,
    damper_step: f32,
    /// Level the note had reached when the damper last started to arrive,
    /// followed by a slow peak follower for as long as the felt is not
    /// limiting. The felt's limiting threshold is a fraction of this, which is
    /// what makes the interaction interval end by itself as the note dies away.
    felt_reference: f32,
    /// Where the felt stood at the *end* of the block this voice rendered last,
    /// as the limiter sees it: 0 clear of the string, 1 seated on it. The
    /// limiter's threshold starts every block from here and walks to where the
    /// damper has moved to, so the waveshaper is continuous across the block
    /// boundary instead of stepping at it.
    felt_engage: f32,
    /// True once the damper has finished arriving, which is when the felt's
    /// limit starts to retire. Cleared the moment the damper moves off the
    /// string again, so a re-strike or a pedal starts the cycle over.
    felt_seated: bool,
    /// This key's *action* noise — the key-off thump and the damper lift. One
    /// burst: the key can only be in one place at a time, and a second event
    /// arriving while the first still rings retriggers it rather than layering.
    noise: Burst,
    noise_out: [f32; BLOCK],
    key_off_noise: EventModel,
    damper_lift_noise: EventModel,
    /// The hammer's own noise, on its own burst.
    ///
    /// Deliberately not the burst above. The action's events are alternatives —
    /// a key is falling or its damper is lifting, never both — but a strike and
    /// a release are not: a staccato is a note-off 80 ms after a note-on, and
    /// sharing one burst would have the key-off cut the attack noise of the note
    /// it belongs to. They mix rather than replace, at the same pan, into the
    /// same block.
    strike_noise: Burst,
    strike_noise_model: EventModel,
    /// Horizontal polarization's own block, used only while `spread`.
    horizontal_out: [f32; BLOCK],
    /// This voice's output during the block the resonance bus was summed from.
    previous_out: [f32; BLOCK],
    previous_silent: bool,
}

impl Voice {
    /// `calibration` carries what a velocity-90 strike of this key measures at
    /// the output, which is what the `[noise]` levels are quoted against
    /// (`calibrate.rs`). A [`MechanismCalibration::silent`] one builds a voice
    /// whose action makes no sound at all — which is what the calibration's own
    /// scratch voices are.
    pub fn new(
        key: u8,
        preset: &Preset,
        shapes: &NoiseShapes,
        calibration: &MechanismCalibration,
    ) -> Self {
        let hammer = Hammer::new(preset.hammer_params(key));
        let pan = pan_for_key(key);
        // The slow horizontal polarization goes to one side of the key's pan
        // position and the fast vertical one to the other, so the note's
        // balance travels from the second towards the first as it decays. Which
        // side is which alternates with the key, or the spread would tilt the
        // whole instrument: every voice would put its aftersound on the same
        // side of itself and the sum of 88 of them is a lateral bias.
        let spread = preset.pan_spread(key) * if key % 2 == 0 { 1.0 } else { -1.0 };
        let mut voice = Voice {
            key,
            index: key_index(key).expect("voice key must be within A0..C8"),
            pan,
            pan_vertical: pan - spread,
            pan_horizontal: pan + spread,
            spread: spread != 0.0,
            string: PianoString::new(
                preset.string_params(key),
                &preset.voicing,
                preset.partial_shaping(key),
            ),
            duplex: DuplexBank::new(preset.duplex_modes(key)),
            duplex_out: [0.0; BLOCK],
            hammer,
            held: false,
            damper_current: 0.0,
            damper_target: 0.0,
            damper_step: damper_step(DEFAULT_RELEASE_VELOCITY),
            felt_reference: 0.0,
            felt_engage: 0.0,
            felt_seated: false,
            noise: Burst::new(),
            noise_out: [0.0; BLOCK],
            key_off_noise: EventModel::new(
                &preset.noise.key_off,
                shapes.key_off,
                key,
                noise::NOMINAL_KEY_DRIVE,
                calibration.key_off(key),
            ),
            damper_lift_noise: EventModel::new(
                &preset.noise.damper_lift,
                shapes.damper_lift,
                key,
                noise::NOMINAL_KEY_DRIVE,
                calibration.damper_lift(key),
            ),
            strike_noise: Burst::new(),
            strike_noise_model: EventModel::from_levels(
                &preset.noise.strike.level_db,
                preset.noise.strike.velocity_db,
                shapes.strike,
                key,
                noise::NOMINAL_STRIKE_DRIVE,
                calibration.strike(key),
            ),
            horizontal_out: [0.0; BLOCK],
            previous_out: [0.0; BLOCK],
            previous_silent: true,
        };
        // Idle keys below G6 rest with their dampers down.
        voice.damper_current = if crate::pedal::has_damper(key) { 1.0 } else { 0.0 };
        voice.damper_target = voice.damper_current;
        voice.string.set_damper(voice.damper_current);
        voice
    }

    pub fn key(&self) -> u8 {
        self.key
    }

    pub fn index(&self) -> usize {
        self.index
    }

    pub fn pan(&self) -> f32 {
        self.pan
    }

    /// Stereo positions of the vertical and horizontal polarizations. Equal to
    /// [`Voice::pan`] unless the preset spreads them.
    pub fn polarization_pans(&self) -> (f32, f32) {
        (self.pan_vertical, self.pan_horizontal)
    }

    pub fn is_held(&self) -> bool {
        self.held
    }

    /// A voice makes no sound when nothing is ringing, no hammer pulse is in
    /// flight, and its undamped segments have gone quiet too.
    pub fn is_idle(&self) -> bool {
        self.strings_idle() && self.duplex.is_idle()
    }

    /// The strings' own half of [`Voice::is_idle`].
    ///
    /// Kept apart because the segments are never damped and the strings are:
    /// a duplex bank that is still ringing must not drag the note's eighty
    /// partials through another second of blocks with it.
    fn strings_idle(&self) -> bool {
        !self.hammer.is_active() && self.string.is_idle()
    }

    pub fn string(&self) -> &PianoString {
        &self.string
    }

    /// The key's undamped segments, for tests and reporting.
    pub fn duplex(&self) -> &DuplexBank {
        &self.duplex
    }

    /// A key press hard enough to reach escapement: the damper lifts, the
    /// hammer is thrown, and the hammer's own noise sounds with it.
    ///
    /// No lift noise here, though the damper does lift: the felt leaving the
    /// string under a hammer blow is inaudible — which is why no sample library
    /// records one — and a broadband burst on every note-on that nobody can
    /// hear is something every later measurement of a render would have to fit
    /// around. It is modelled where it is the whole sound: [`Voice::key_down`].
    ///
    /// The **strike** noise is the opposite case: it is inaudible only as a
    /// separate event, because it arrives inside the attack of the note it
    /// belongs to, and that is exactly what the measurements say is missing —
    /// the engine's attacks read +5.2 dB more tonal than the recordings'
    /// (`renders/realism/REALISM.md`) and its 30 ms residual 11–13 dB more
    /// tonal at every key (`renders/timbre-ladder/ANALYSIS.md` §8.3). It is
    /// triggered here, in the same block as the hammer pulse, so that the burst
    /// and the blow start on the same sample.
    pub fn note_on(&mut self, vel: u8, pedals: &PedalState, frame: u64) {
        self.press(pedals);
        self.hammer.set_una_corda(pedals.una_corda());
        // Both halves of the strike vector's direction, in one pass over the
        // key's modes: which strings the hammer reaches, and how hard it is
        // travelling. The second is the only way velocity enters the linear
        // string model at all (`FUNDAMENTALS.md` §7.5 step 3), and a preset
        // without a `[voicing.strike_direction]` ignores it.
        self.string.set_strike(pedals.una_corda(), vel);
        self.hammer.strike_midi(vel);
        // A re-strike into a burst that is still ringing keeps its filter state
        // and restarts the envelope — the same "louder continuation" a trill
        // gets from the action's own events, and the reason a repeated note
        // cannot click.
        self.strike_noise.trigger(
            &self.strike_noise_model,
            vel as f32 / 127.0,
            noise::seed_of(self.key, frame ^ STRIKE_SEED_SALT),
        );
    }

    /// A key press that does not reach escapement, so the damper lifts and
    /// nothing is struck. The damper leaving the string is the only sound, and
    /// the key counts as held — which is what lets sostenuto capture a silently
    /// prepared note, the reason the gesture exists.
    pub fn key_down(&mut self, vel: u8, pedals: &PedalState, frame: u64) {
        // Read before the target moves: the felt only makes a noise if it was
        // on the string to begin with. A key pressed again while it is still
        // held, or pressed with the pedal already down, lifts nothing.
        let lifting = self.damper_target > 0.0 && crate::pedal::has_damper(self.key);
        self.press(pedals);
        if lifting {
            self.noise.trigger(
                &self.damper_lift_noise,
                vel as f32 / 127.0,
                noise::seed_of(self.key, frame),
            );
        }
    }

    fn press(&mut self, pedals: &PedalState) {
        self.held = true;
        self.update_dampers(pedals);
        // The ramp a release velocity set belongs to that release. A key going
        // down takes its damper off the string at the nominal rate: the jack is
        // driving it, not the key's own weight.
        self.damper_step = damper_step(DEFAULT_RELEASE_VELOCITY);
    }

    /// A key release at release velocity `vel`, which sets both how fast the
    /// damper falls and how loud the key-off thump is.
    pub fn note_off(&mut self, vel: u8, pedals: &PedalState, frame: u64) {
        self.held = false;
        self.update_dampers(pedals);
        self.damper_step = damper_step(vel);
        // On every key, including the ones above the damper break: `rel76` is
        // C7, five semitones past it, and it is one of the loudest of the
        // measured releases. What stops at G6 is the damper, not the key.
        self.noise.trigger(
            &self.key_off_noise,
            vel as f32 / 127.0,
            noise::seed_of(self.key, frame),
        );
    }

    /// Recomputes the damper target from the current pedal state. Cheap; call
    /// it on every pedal or key change.
    pub fn update_dampers(&mut self, pedals: &PedalState) {
        self.damper_target = pedals.damper_amount(self.index, self.held);
    }

    /// Renders one block into `out` (overwritten) and accumulates it into
    /// `board` at this voice's stereo position, reading the sympathetic
    /// resonance bus and leaving `out` ready to be fed back into it.
    ///
    /// `out` is the voice's mono sum whether or not the polarizations are
    /// panned apart: the resonance bus is one mono signal and the halo must not
    /// depend on the stereo image. What the spread changes is only how the
    /// voice reaches the board — one `add_voice` at the key's pan, or two, one
    /// per polarization.
    ///
    /// The key's duplex segments join that mono sum, so they reach the board
    /// and the bus with the note; they are rendered whether or not the strings
    /// are, since nothing damps them.
    ///
    /// Returns false when the voice had nothing at all to render, in which case
    /// `out` was not touched — the caller must not feed it to the bus. The
    /// board may still have been added to: a key released long after its note
    /// died makes its mechanism noise from exactly that state, and the noise
    /// never travels through `out`. A voice that is silent still has to run its
    /// strings whenever its dampers are off them and the bus carries something
    /// — that is the whole mechanism of sympathetic resonance, and skipping it
    /// would mean a piano whose undamped strings never answer the ones being
    /// played.
    pub fn process(&mut self, out: &mut [f32], bus: &ResonanceBus, board: &mut Soundboard) -> bool {
        debug_assert_eq!(out.len(), BLOCK);
        // The strings run while they are ringing, while a hammer is in flight,
        // and whenever the bus can reach them.
        let strings_live = !self.strings_idle() || (bus.is_active() && self.damper_target < 1.0);
        // The segments have no damper at all, so the damper does not appear
        // here: they run while they still hold something, and whenever anything
        // can drive them — this key's own strings, or the bus with another
        // key's note on it. A key with no segments never takes this path at
        // all, which is what keeps a preset without them exactly the instrument
        // it was.
        let duplex_live = !self.duplex.is_empty()
            && (strings_live || bus.is_active() || !self.duplex.is_idle());
        if !strings_live && !duplex_live {
            // A key released long after its note died still thumps, and a
            // silently prepared key still lifts its damper audibly. Neither
            // wakes the strings: the noise reaches the board directly, so a
            // voice with nothing but a burst running costs a block of filtering
            // and no resonators. A voice with no burst either does not execute
            // any of this — that is what keeps 88 idle voices bit-exact silent.
            self.add_noise(board);
            if !self.previous_silent {
                self.previous_out.fill(0.0);
                self.previous_silent = true;
            }
            return false;
        }
        out.fill(0.0);

        if strings_live {
            if self.damper_current != self.damper_target {
                let delta = self.damper_target - self.damper_current;
                self.damper_current += delta.clamp(-self.damper_step, self.damper_step);
                self.string.set_damper(self.damper_current);
            }

            if self.hammer.is_active() {
                // One common, unskewed, unit-share pulse. The per-string shares
                // and the small timing skew — the hammer is not perfectly square
                // to the strings — are inside the modes' complex gains, which is
                // the whole point of the eigen construction: `N` input buffers
                // collapse into `2N` complex scalars. Under una corda the hammer
                // misses one string of the group, which is the strike vector's
                // *direction* changing, and `PianoString::set_una_corda` swaps
                // the gain table the note-on selected.
                self.hammer.add_pulse(self.string.excitation_mut(), 0, 1.0);
                self.hammer.advance(BLOCK);
            }
        }

        // One drive buffer, read by the strings and by the segments both: there
        // is one bus path through the bridge admittance and this is it. The
        // strings take it scaled by how far their damper is off them; the
        // segments take all of it, always, because nothing dampens them.
        let mut drive = [0.0f32; BLOCK];
        let strings_driven = strings_live && self.damper_current < 1.0;
        let duplex_driven = duplex_live && bus.is_active();
        if strings_driven || duplex_driven {
            bus.drive(self.index, &self.previous_out, &mut drive);
        }

        if strings_live {
            // Undamped strings pick up the rest of the instrument.
            if strings_driven {
                let gain = 1.0 - self.damper_current;
                self.string.add_excitation_all(&drive, gain);
            }

            // Once per rendered block, after the damper has moved and before
            // either polarization is limited: the ramp is this voice's felt
            // travelling from where it was to where the damper has just put it,
            // and computing it twice would advance it twice.
            let felt = self.felt_ramp();

            if self.spread {
                self.horizontal_out.fill(0.0);
                self.string.process_split(out, &mut self.horizontal_out);
                // The felt is one piece of cloth on one string: it sees the
                // whole of it, both planes at once, so the limiter is computed
                // from the sum and applied to the two halves as a common gain.
                self.felt_limit_split(out, felt);
                board.add_voice(out, self.pan_vertical);
                board.add_voice(&self.horizontal_out, self.pan_horizontal);
                // ... and the mono sum, for the resonance bus and the next
                // block's coupling, is the two of them back together.
                for (o, &h) in out.iter_mut().zip(&self.horizontal_out) {
                    *o += h;
                }
            } else {
                self.string.process(out);
                self.felt_limit(out, felt);
                board.add_voice(out, self.pan);
            }
            // Before the segments are added: what the felt limits against is
            // the speaking length's own swing, and the felt never touches them.
            self.track_felt_reference(out, felt);
        }

        if duplex_live {
            // Driven by the bridge force the key has just produced — `out`, the
            // whole mono sum, felt limiter and all — and by the bus. The
            // segments are read before they are added, so nothing here is a
            // path from a segment straight back into itself.
            self.duplex_out.fill(0.0);
            self.duplex.add(
                out,
                duplex_driven.then_some(&drive[..]),
                &mut self.duplex_out,
            );
            // They radiate from the key's own end of the bridge, and they are
            // one signal: the polarization spread is a property of the speaking
            // length's two planes and the segments have no part in it.
            board.add_voice(&self.duplex_out, self.pan);
            for (o, &d) in out.iter_mut().zip(&self.duplex_out) {
                *o += d;
            }
        }

        self.add_noise(board);
        self.previous_out.copy_from_slice(out);
        self.previous_silent = false;
        true
    }

    /// Renders this key's mechanism noise, if any is running, straight onto the
    /// board at the key's own pan.
    ///
    /// Deliberately not into `out`: `out` is what the resonance bus reads, and
    /// a key-off thump is a structure-borne sound that reaches the ear through
    /// the keybed and the board rather than a force that drives the other
    /// strings. Fontana et al. showed listeners can tell *which key* was played
    /// from the mechanical noise alone, which is why it is panned per note
    /// rather than centred (`PHYSICS.md` §5).
    fn add_noise(&mut self, board: &mut Soundboard) {
        if !self.noise.is_active() && !self.strike_noise.is_active() {
            return;
        }
        // One buffer and one `add_voice` for both bursts: they are the same
        // key's mechanism and they radiate from the same place.
        self.noise_out.fill(0.0);
        self.noise.add(&mut self.noise_out);
        self.strike_noise.add(&mut self.noise_out);
        board.add_voice(&self.noise_out, self.pan);
    }

    /// Applies the partly-engaged damper's soft limit to `mono` in place.
    ///
    /// Touches nothing — and costs one block peak — when the damper is fully
    /// off or fully on the string, which is every voice on the instrument
    /// except the handful being half-pedalled or released at that instant.
    ///
    /// The limit is on the string's force on the bridge, which is what the
    /// board radiates *and* what the sympathetic bus carries to the rest of the
    /// instrument, so the felt's correction does reach everything downstream of
    /// the string. It is deliberately not injected back into the string's own
    /// excitation: converting a force at the damper's position into a modal
    /// drive needs the string's admittance rather than a constant, and at the
    /// magnitudes involved — a tenth of a signal that is already deep into a
    /// damped decay — a constant-gain injection is 40 dB below anything
    /// audible. See `DECISIONS.md`.
    fn felt_limit(&mut self, mono: &mut [f32], felt: Option<FeltRamp>) {
        let Some(felt) = felt else { return };
        let peak = mono.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        if peak <= felt.lowest {
            return;
        }
        let mut threshold = felt.start;
        for x in mono.iter_mut() {
            *x = soft_limit(*x, threshold);
            threshold *= felt.step;
        }
    }

    /// The same limit computed from the two polarizations' sum and applied to
    /// both of them, for the voices whose polarizations are panned apart.
    fn felt_limit_split(&mut self, vertical: &mut [f32], felt: Option<FeltRamp>) {
        let Some(felt) = felt else { return };
        let peak = vertical
            .iter()
            .zip(&self.horizontal_out)
            .fold(0.0f32, |m, (&v, &h)| m.max((v + h).abs()));
        if peak <= felt.lowest {
            return;
        }
        let mut threshold = felt.start;
        for (v, h) in vertical.iter_mut().zip(&mut self.horizontal_out) {
            let x = *v + *h;
            let y = soft_limit(x, threshold);
            threshold *= felt.step;
            // A common gain, so the note's stereo image is not moved by the
            // damper: `x` is only zero where both planes are, and there the
            // scale is 1 exactly.
            let scale = if y == x { 1.0 } else { y / x };
            *v *= scale;
            *h *= scale;
        }
    }

    /// Where the felt travels during the block about to be rendered, as the
    /// limiter's threshold at the first sample and the per-sample factor that
    /// walks it to the last — or `None` when the felt is clear of the string
    /// for the whole of it, which is every voice on the instrument except the
    /// handful being released, half-pedalled or damped at that instant.
    ///
    /// Advances [`Voice::felt_engage`], so it is called exactly once per
    /// rendered block, after the damper has moved.
    ///
    /// Three properties, each of which is the answer to something measured:
    ///
    /// * **It is a ramp, not a value.** The damper moves once per block and the
    ///   threshold is exponential in its position, so a per-block threshold
    ///   steps by 13.3 dB at every boundary of a nominal release — audible as a
    ///   click at −18 to -23 dB under the phrase's own peak on every note-off
    ///   of every preset, the defect this shape exists to remove. The walk is
    ///   geometric because the position enters through `powf`: one `powf` for
    ///   the ratio, then a multiply per sample.
    /// * **It retires at a seated damper instead of stopping there.** The felt
    ///   is deepest into the string exactly when the damper lands, so switching
    ///   the limiter off at that moment put a single-sample jump from
    ///   `soft_limit(x, 0.01 · reference)` back to `x` — the largest of the four
    ///   steps. Holding it there instead is worse than the click: an absolute
    ///   limit anchored to the pre-release level clamps the whole tail to 2 % of
    ///   it, which measured out as a release that no longer answers the release
    ///   *velocity* at all (−80.8 / −81.0 / −81.6 dB at MIDI 1 / 64 / 127, where
    ///   the acceptance test wants 2 dB between the first two). What ends is the
    ///   *interaction*, and it ends over the damper's own ramp: engagement walks
    ///   back to zero at `damper_step` per block, so the note is free again
    ///   after the damper has been down for as long as it took to land, and
    ///   every sample in between is on a continuous curve. A note that has
    ///   already decayed under the limit leaves it earlier and for free, through
    ///   the `lowest` test below — `PHYSICS.md` §6's third interval, unchanged.
    /// * **It still tests the damper's direction.** The engine starts the
    ///   damper's lift and the hammer's blow at the same instant, while the
    ///   real action lifts the damper early in the key's travel and has it
    ///   clear before the hammer arrives. Limiting a *leaving* damper spends
    ///   the first ~10 ms of every re-strike — the whole attack transient —
    ///   clamped against a threshold left over from the previous note, which
    ///   measured 27–68 dB of choke on the first two blocks of a fortissimo C4
    ///   re-strike and made the attack depend on how loud the note before it
    ///   was. Lehtonen's measurement is of a damper *arriving*
    ///   (`DECISIONS.md` 116); nothing in it describes one leaving under a
    ///   hammer blow. A gesture that lifts a damper *during its own descent*
    ///   therefore still releases the limit in one step rather than a ramp —
    ///   the one discontinuity left on this path, and the one the alternative
    ///   is worse than (`DECISIONS.md` 218).
    fn felt_ramp(&mut self) -> Option<FeltRamp> {
        if self.damper_target < self.damper_current {
            // Leaving: the felt is clear of the string, and the block is not
            // limited at all. See the third point above.
            self.felt_seated = false;
            self.felt_engage = 0.0;
            return None;
        }
        let to = if self.damper_current < 1.0 {
            // Arriving, or held part way down by a half pedal: the felt is
            // exactly where the damper is, and stays there for as long as the
            // pedal holds it — which is the interaction interval this whole
            // mechanism exists to render.
            self.felt_seated = false;
            self.damper_current
        } else if !self.felt_seated {
            // The block the damper lands on: the arrival finishes at the
            // deepest the felt ever reaches.
            self.felt_seated = true;
            1.0
        } else {
            // Seated, and the interaction is over: the string is now decaying
            // under the damper's full linear grip and an absolute limit
            // anchored to the level the note had *before* the release has
            // nothing left to say about it. It retires at the rate the damper
            // arrived at — so a slow release keeps its felt as long as it keeps
            // its ramp — and the note is free again by the time the damper has
            // been down for as long as it took to land.
            (self.felt_engage - self.damper_step).max(0.0)
        };
        let from = self.felt_engage;
        self.felt_engage = to;
        if from <= 0.0 && to <= 0.0 {
            return None;
        }
        // `felt_reference` is where the note was when the damper started to
        // arrive, so the limit is an absolute level and the note stops meeting
        // it as it decays.
        if self.felt_reference <= 0.0 {
            return None;
        }
        let start = self.felt_reference * FELT_CLEARANCE.powf(from);
        let end = self.felt_reference * FELT_CLEARANCE.powf(to);
        Some(FeltRamp {
            start,
            step: FELT_CLEARANCE.powf((to - from) / (BLOCK - 1) as f32),
            lowest: start.min(end),
        })
    }

    /// Remembers how loud the note is while the felt is *not* limiting it, so
    /// the felt has an absolute level to limit against when it arrives.
    ///
    /// The follower runs whenever the limiter is disengaged rather than only at
    /// a fully lifted damper: what it must not read is its own output, which is
    /// what a limiting block gives it. A damper that is leaving the string —
    /// under a re-strike, or a pedal going down — is not limiting, and it is
    /// exactly then that the string's new free swing has to be picked up, so
    /// that a release arriving before the damper has finished lifting limits
    /// against *this* note rather than against the one before it.
    ///
    /// The gate is the block's own ramp rather than whether that ramp bit:
    /// every block the felt was on the string for has to be excluded, including
    /// the ones where the note was already under the limit, or the reference
    /// would chase the threshold down and squeeze a note that had fallen clear
    /// of the felt. The reference therefore holds from the moment the felt
    /// arrives until it leaves again — which is what "the level the note had
    /// when the damper arrived" means — instead of resuming its decay at a
    /// seated damper, as it did while a seated damper was counted as clear.
    fn track_felt_reference(&mut self, mono: &[f32], felt: Option<FeltRamp>) {
        if felt.is_some() {
            return;
        }
        const RELEASE: f32 = 1.0 - BLOCK as f32 / (FELT_FOLLOWER_S * SAMPLE_RATE);
        let peak = mono.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        self.felt_reference = peak.max(self.felt_reference * RELEASE);
    }

    /// Immediate silence, used by `AllOff`.
    pub fn reset(&mut self) {
        self.held = false;
        self.hammer.reset();
        self.string.reset();
        // The one thing that stops a segment. `AllOff` is a panic button, not a
        // gesture: no key and no pedal reaches this.
        self.duplex.reset();
        self.duplex_out.fill(0.0);
        self.noise.reset();
        self.strike_noise.reset();
        self.noise_out.fill(0.0);
        self.horizontal_out.fill(0.0);
        self.previous_out.fill(0.0);
        self.previous_silent = true;
        self.felt_reference = 0.0;
        self.felt_engage = 0.0;
        self.felt_seated = false;
        self.damper_step = damper_step(DEFAULT_RELEASE_VELOCITY);
        self.damper_current = if crate::pedal::has_damper(self.key) { 1.0 } else { 0.0 };
        self.damper_target = self.damper_current;
        self.string.set_damper(self.damper_current);
    }
}

/// Damper ramp rate, as a fraction of the way to the target per block, for a
/// release at MIDI velocity `vel`. Geometric around [`DAMPER_RAMP_SECONDS`] at
/// the nominal velocity, so the ramp the engine has always used is exactly what
/// a source with no release velocity still gets.
fn damper_step(vel: u8) -> f32 {
    let exponent =
        (DEFAULT_RELEASE_VELOCITY as f32 - vel as f32) / (127 - DEFAULT_RELEASE_VELOCITY) as f32;
    let seconds = DAMPER_RAMP_SECONDS * DAMPER_RAMP_RANGE.powf(exponent);
    BLOCK as f32 / (seconds * SAMPLE_RATE)
}

/// One block of the felt limiter's threshold: where it starts, the factor that
/// walks it from sample to sample, and the lower of its two ends.
///
/// Copied rather than borrowed so that one call to [`Voice::felt_ramp`] can
/// serve both polarizations and the peak follower without re-deriving it, and
/// so that the whole of the felt's per-block state is these three numbers.
#[derive(Clone, Copy)]
struct FeltRamp {
    /// Threshold at the block's first sample.
    start: f32,
    /// Multiplied into the threshold after every sample.
    step: f32,
    /// The smaller of the thresholds at the two ends of the block: a block
    /// whose peak is under it cannot reach the felt anywhere, which is the
    /// early-out that keeps a decayed note free (`PHYSICS.md` §6's third
    /// interval) and costs one block peak to find.
    lowest: f32,
}

/// Soft limiter: transparent below `threshold`, tanh-compressed above it, and
/// continuous in value and slope where they meet. The compression is what puts
/// harmonics of the string's own partials into the sound — the buzz of a half
/// pedal — rather than merely making it quieter.
fn soft_limit(x: f32, threshold: f32) -> f32 {
    let a = x.abs();
    if a <= threshold {
        x
    } else {
        x.signum() * threshold * (1.0 + ((a - threshold) / threshold).tanh())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preset::{DuplexMode, MAX_DUPLEX_GAIN_DB};
    use crate::types::{key_index, NUM_KEYS};

    fn voice(key: u8) -> Voice {
        voice_from(key, &Preset::default())
    }

    fn voice_from(key: u8, preset: &Preset) -> Voice {
        let shapes = NoiseShapes::new(&preset.noise);
        let calibration = MechanismCalibration::new(preset, &shapes);
        Voice::new(key, preset, &shapes, &calibration)
    }

    fn bus() -> ResonanceBus {
        ResonanceBus::new(Preset::default().voicing.resonance_coupling)
    }

    /// A board to render into. The tests below read the voice's own mono block,
    /// not the board, except where they say otherwise.
    fn board() -> Soundboard {
        Soundboard::new(&Preset::default().soundboard)
    }

    fn rms(v: &[f32]) -> f32 {
        (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt()
    }

    /// Renders `blocks` blocks and returns the RMS of the last one.
    fn render(voice: &mut Voice, blocks: usize) -> f32 {
        let bus = bus();
        let mut board = board();
        let mut out = [0.0f32; BLOCK];
        for _ in 0..blocks {
            if !voice.process(&mut out, &bus, &mut board) {
                out.fill(0.0);
            }
        }
        rms(&out)
    }

    #[test]
    fn a_fresh_voice_is_idle_and_silent() {
        let mut v = voice(60);
        assert!(v.is_idle());
        assert_eq!(render(&mut v, 10), 0.0);
    }

    #[test]
    fn note_on_makes_sound_and_note_off_stops_it() {
        let pedals = PedalState::new();
        let mut v = voice(60);
        v.note_on(90, &pedals, 0);
        let struck = render(&mut v, 200);
        assert!(struck > 0.0);
        v.note_off(DEFAULT_RELEASE_VELOCITY, &pedals, 0);
        // 0.5 s of damped decay must lose more than 40 dB.
        let after = render(&mut v, (0.5 * crate::types::SAMPLE_RATE / BLOCK as f32) as usize);
        assert!(
            after < struck * 0.005,
            "after note-off {after} vs struck {struck}"
        );
    }

    #[test]
    fn harder_strikes_are_louder() {
        let pedals = PedalState::new();
        let mut soft = voice(60);
        let mut hard = voice(60);
        soft.note_on(40, &pedals, 0);
        hard.note_on(110, &pedals, 0);
        assert!(render(&mut hard, 100) > render(&mut soft, 100) * 2.0);
    }

    #[test]
    fn undamped_treble_keeps_ringing_after_release() {
        let pedals = PedalState::new();
        let mut v = voice(96); // C7, above the damper break
        v.note_on(100, &pedals, 0);
        render(&mut v, 50);
        v.note_off(DEFAULT_RELEASE_VELOCITY, &pedals, 0);
        assert!(render(&mut v, 10) > 0.0);
    }

    /// A silent string must still run when its dampers are up and the bus has
    /// something to give it — that is what sympathetic resonance is — and must
    /// be skipped otherwise, which is what keeps the 88 voices affordable.
    #[test]
    fn a_silent_voice_runs_only_when_the_bus_can_reach_it() {
        let mut pedals = PedalState::new();
        let mut v = voice(60);
        let mut board = board();
        let mut out = [0.0f32; BLOCK];

        let mut quiet = bus();
        quiet.begin_block();
        assert!(!quiet.is_active());
        assert!(!v.process(&mut out, &quiet, &mut board));

        let mut loud = bus();
        loud.contribute(&[0.01; BLOCK]);
        loud.begin_block();
        assert!(loud.is_active());
        // Dampers still down: the bus cannot reach the string.
        assert!(!v.process(&mut out, &loud, &mut board));

        pedals.set_sustain(1.0);
        v.update_dampers(&pedals);
        assert!(v.process(&mut out, &loud, &mut board));
        assert!(out.iter().any(|&x| x != 0.0), "no sympathetic response");
    }

    /// Where the two polarizations sit, and that the default puts them in the
    /// same place — which is what keeps the shipped instrument's render the
    /// single-buffer one it has always been.
    #[test]
    fn the_pan_spread_places_the_polarizations_either_side_of_the_key() {
        for key in [21u8, 60, 61, 108] {
            let (v, h) = voice(key).polarization_pans();
            assert_eq!((v, h), (pan_for_key(key), pan_for_key(key)));
        }

        let mut preset = Preset::default();
        preset.voicing.polarization_pan_spread = 0.4;
        assert!(preset.validate().is_ok());
        for key in [21u8, 60, 61, 108] {
            let v = voice_from(key, &preset);
            let (pan_v, pan_h) = v.polarization_pans();
            let sign = if key % 2 == 0 { 1.0 } else { -1.0 };
            assert_eq!(pan_v, v.pan() - 0.4 * sign);
            assert_eq!(pan_h, v.pan() + 0.4 * sign);
            // Nothing lands off the stage: `MAX_PAN + MAX_PAN_SPREAD` is 1.
            assert!(pan_v.abs() <= 1.0 && pan_h.abs() <= 1.0);
        }
        // The spread alternates, so the instrument as a whole is not pulled to
        // one side: neighbouring keys put their aftersound on opposite sides.
        let (a, _) = voice_from(60, &preset).polarization_pans();
        let (b, _) = voice_from(61, &preset).polarization_pans();
        assert!(a < voice_from(60, &preset).pan() && b > voice_from(61, &preset).pan());
    }

    /// A per-key table overrides the global scalar key by key, and a preset
    /// without one is the compass the scalar describes. The measurement behind
    /// it: at the global ceiling the engine's drift is 0.24 dB at A0 and
    /// 8.67 dB at C5 against the recordings' 1.24 and 5.33
    /// (`TUNING_REPORT.md` §5), so one number cannot fit both ends.
    #[test]
    fn a_per_key_spread_overrides_the_global_one_key_by_key() {
        let mut preset = Preset::default();
        preset.voicing.polarization_pan_spread = 0.4;
        preset.notes.pan_spread = (0..NUM_KEYS).map(|i| 0.01 * (i % 8) as f32).collect();
        assert!(preset.validate().is_ok());
        for key in [21u8, 60, 61, 108] {
            let v = voice_from(key, &preset);
            let want = preset.notes.pan_spread[usize::from(key - 21)]
                * if key % 2 == 0 { 1.0 } else { -1.0 };
            let (pan_v, pan_h) = v.polarization_pans();
            assert_eq!(pan_v, v.pan() - want);
            assert_eq!(pan_h, v.pan() + want);
        }
        // A key the table gives zero is mono again even though the global
        // scalar is at its ceiling — the table is an override, not an offset.
        let zeroed = voice_from(21, &preset);
        assert_eq!(zeroed.polarization_pans(), (zeroed.pan(), zeroed.pan()));
    }

    /// The spread must not reach the sympathetic bus: the halo is one mono
    /// signal and a note's stereo image cannot be allowed to change what the
    /// rest of the instrument picks up from it.
    #[test]
    fn the_pan_spread_leaves_the_voices_mono_sum_alone() {
        let mut spread = Preset::default();
        spread.voicing.polarization_pan_spread = 0.4;
        let pedals = PedalState::new();
        let bus = bus();
        let mut plain = voice(60);
        let mut split = voice_from(60, &spread);
        plain.note_on(100, &pedals, 0);
        split.note_on(100, &pedals, 0);

        let (mut a, mut b) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
        let mut peak = 0.0f32;
        let mut worst = 0.0f32;
        for _ in 0..200 {
            assert!(plain.process(&mut a, &bus, &mut board()));
            assert!(split.process(&mut b, &bus, &mut board()));
            for i in 0..BLOCK {
                peak = peak.max(a[i].abs());
                worst = worst.max((a[i] - b[i]).abs());
            }
        }
        assert!(peak > 0.0);
        // The two paths add the same numbers in a different order, so this is a
        // rounding bound and not a bit-exactness one — and it is stated against
        // the note's own peak rather than against a running maximum, because the
        // group's modes cancel into small samples early in the attack.
        assert!(
            worst <= 1e-6 * peak,
            "the spread voice's mono sum differs by {worst:e} on a peak of {peak:e}"
        );
    }

    /// A voice with nothing to render must write nothing at all — the noise
    /// path added a second reason for a voice to be alive, and an idle key
    /// still has to cost exactly zero samples.
    #[test]
    fn an_idle_voice_writes_no_samples_at_all() {
        let mut v = voice(60);
        let mut board = board();
        let mut out = [7.0f32; BLOCK];
        assert!(!v.process(&mut out, &bus(), &mut board));
        assert!(out.iter().all(|&x| x == 7.0), "an idle voice touched its block");
        let (mut l, mut r) = ([1.0f32; BLOCK], [1.0f32; BLOCK]);
        board.begin_block();
        assert!(!v.process(&mut out, &bus(), &mut board));
        board.process(&mut l, &mut r);
        assert!(
            l.iter().chain(r.iter()).all(|&x| x == 0.0),
            "an idle voice put something on the board"
        );
    }

    /// ... and a key released long after its note has died still thumps, which
    /// is the case that forced the noise onto the idle path in the first place.
    #[test]
    fn a_note_off_sounds_even_when_the_string_is_long_since_silent() {
        let pedals = PedalState::new();
        let mut v = voice(60);
        assert!(v.is_idle());
        v.note_off(64, &pedals, 0);
        let mut board = board();
        let mut out = [0.0f32; BLOCK];
        let (mut l, mut r) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
        board.begin_block();
        // Still idle — no string is ringing — but it is not silent.
        assert!(!v.process(&mut out, &bus(), &mut board));
        board.process(&mut l, &mut r);
        assert!(
            l.iter().chain(r.iter()).any(|&x| x != 0.0),
            "the note-off made no sound"
        );
        assert!(
            out.iter().all(|&x| x == 0.0),
            "the thump reached the resonance bus; it must go to the board only"
        );
    }

    /// A press below escapement lifts the damper and strikes nothing. The
    /// damper leaving the string is the whole sound of the gesture.
    #[test]
    fn a_silent_press_lifts_the_damper_without_striking() {
        let pedals = PedalState::new();
        let mut v = voice(60);
        v.key_down(crate::types::ESCAPEMENT_VELOCITY, &pedals, 0);
        assert!(v.is_held());
        assert!(v.is_idle(), "a silent press threw the hammer");
        let mut lifted = board();
        let mut out = [0.0f32; BLOCK];
        let (mut l, mut r) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
        lifted.begin_block();
        v.process(&mut out, &bus(), &mut lifted);
        lifted.process(&mut l, &mut r);
        assert!(
            l.iter().chain(r.iter()).any(|&x| x != 0.0),
            "the damper lifted in silence"
        );
        assert_eq!(v.string().energy(), 0.0, "the string was struck");

        // A key above the damper break has no damper to lift, so it makes no
        // sound at all.
        let mut top = voice(96);
        top.key_down(crate::types::ESCAPEMENT_VELOCITY, &pedals, 0);
        let mut empty = board();
        empty.begin_block();
        top.process(&mut out, &bus(), &mut empty);
        empty.process(&mut l, &mut r);
        assert!(l.iter().chain(r.iter()).all(|&x| x == 0.0));
    }

    /// The damper ramp is the release velocity's, and the nominal velocity
    /// leaves it exactly where it has always been.
    #[test]
    fn release_velocity_sets_how_fast_the_damper_falls() {
        let nominal = damper_step(DEFAULT_RELEASE_VELOCITY);
        assert!(
            (nominal - BLOCK as f32 / (DAMPER_RAMP_SECONDS * SAMPLE_RATE)).abs() < 1e-6,
            "the nominal release moved off the ramp the engine has always used"
        );
        assert!(damper_step(1) < nominal, "a slow release did not slow the damper");
        assert!(damper_step(127) > nominal, "a fast release did not hurry it");
        // The span is the constant, both ways round the nominal.
        assert!((damper_step(127) / nominal - DAMPER_RAMP_RANGE).abs() < 0.05);
        assert!((nominal / damper_step(1) - DAMPER_RAMP_RANGE).abs() < 0.05);
    }

    // ------------------------------------------------- the duplex segments

    /// The key the segment tests use. C6 and not something above the damper
    /// break, because the whole question is what survives a damper landing.
    const DUPLEX_KEY: u8 = 84;

    /// Where this key's segments sit: an aliquot on the third partial, as a
    /// well-scaled rear duplex is, and a second segment a quarter of a semitone
    /// off the fifth, as Öberg & Askenfelt's scatter puts it.
    ///
    /// The placement matters more than it looks. A segment is a resonator with
    /// a bandwidth under two hertz, and the bridge force one string puts out is
    /// a sum of its own partials, so a segment that sits *between* them is
    /// driven only by the attack transient and answers 30–40 dB lower. That is
    /// what an aliquot is *for*, and it is why the two frequencies here are
    /// taken from the note's own series rather than written as round numbers.
    fn duplex_hz() -> [f32; 2] {
        let p = Preset::default().string_params(DUPLEX_KEY);
        [p.partial_freq(3), p.partial_freq(5) * 2.0f32.powf(25.0 / 1200.0)]
    }

    fn duplex_preset(gain_db: f32) -> Preset {
        let mut preset = Preset::default();
        preset.notes.duplex = vec![Vec::new(); NUM_KEYS];
        preset.notes.duplex[key_index(DUPLEX_KEY).unwrap()] = duplex_hz()
            .iter()
            .map(|&hz| DuplexMode {
                hz,
                gain_db,
                t60_s: 1.5,
            })
            .collect();
        preset
            .validate()
            .expect("a two-segment treble key is a legal preset");
        preset
    }

    /// The Öberg & Askenfelt signature: play a treble key staccato and the
    /// segments go on sounding after the damper has stopped the speaking
    /// length. Nothing damps them, so the note does not end when the key does.
    #[test]
    fn a_struck_treble_keys_duplex_rings_on_after_a_staccato_release() {
        let pedals = PedalState::new();
        // Plays the gesture and returns the note's own peak and what is still
        // sounding half a second after the key came up — by which time this
        // key's damper, whose release T60 is about a tenth of a second, has
        // long since finished with the speaking length.
        let staccato = |preset: &Preset| {
            let mut v = voice_from(DUPLEX_KEY, preset);
            let bus = bus();
            let mut board = board();
            let mut out = [0.0f32; BLOCK];
            v.note_on(110, &pedals, 0);
            let mut struck = 0.0f32;
            for _ in 0..20 {
                // ~50 ms: a staccato.
                if !v.process(&mut out, &bus, &mut board) {
                    out.fill(0.0);
                }
                struck = struck.max(out.iter().fold(0.0f32, |m, &x| m.max(x.abs())));
            }
            v.note_off(100, &pedals, 0);
            for _ in 0..(0.5 * SAMPLE_RATE / BLOCK as f32) as usize {
                if !v.process(&mut out, &bus, &mut board) {
                    out.fill(0.0);
                }
            }
            (struck, rms(&out), v)
        };
        let (_, plain, _) = staccato(&Preset::default());
        let (struck, voiced, v) = staccato(&duplex_preset(MAX_DUPLEX_GAIN_DB));
        assert!(
            voiced > plain * 30.0,
            "after a staccato release the segments left {voiced:e} against the \
             damped string's {plain:e}"
        );
        // ... and it is the segments that are left, not a string the damper
        // failed to stop.
        assert!(v.string.is_idle(), "the damper did not stop the string");
        assert!(!v.duplex.is_idle(), "the segments were damped with the string");
        assert!(!v.is_idle(), "a voice with ringing segments called itself idle");

        // The level, stated rather than merely ordered. `TUNING_REPORT.md` §5
        // measures the release resonances at -31 dB (C3) and -39 dB (C5)
        // relative to a strike of the same key, ringing 1-2 s. An aliquot at
        // the top of the schema's range measures -61 dB here, which is an RMS
        // against a peak (about 9 dB of the difference) half a second into the
        // segments' own 1.5 s decay (another 20): a band, not a number, because
        // what this pins is that the model reaches the right neighbourhood from
        // one string's bridge force alone.
        let level = crate::types::amp_to_db(voiced / struck);
        assert!(
            (-70.0..-40.0).contains(&level),
            "the segments are {level:.1} dB under the note that drove them"
        );
    }

    /// Neither of the two pedals that are *not* dampers may touch them: una
    /// corda moves the hammer sideways and sostenuto catches damper levers, and
    /// a duplex segment has no damper lever to catch.
    #[test]
    fn una_corda_and_sostenuto_do_not_damp_the_segments() {
        let preset = duplex_preset(MAX_DUPLEX_GAIN_DB);
        let mut plain = PedalState::new();
        let mut both = PedalState::new();
        both.set_una_corda(true);
        both.set_sostenuto(true, &[false; NUM_KEYS]);

        let ring = |pedals: &PedalState| {
            let mut v = voice_from(DUPLEX_KEY, &preset);
            v.note_on(110, pedals, 0);
            render(&mut v, 20);
            v.note_off(100, pedals, 0);
            render(&mut v, (0.5 * SAMPLE_RATE / BLOCK as f32) as usize);
            v.duplex.is_idle()
        };
        assert!(!ring(&plain), "the segments never rang");
        assert!(!ring(&both), "una corda or sostenuto damped the segments");
        plain.set_sustain(0.0);
    }

    /// ... and `AllOff` does, because it is a panic button and not a gesture.
    #[test]
    fn all_off_stops_the_segments() {
        let pedals = PedalState::new();
        let mut v = voice_from(DUPLEX_KEY, &duplex_preset(MAX_DUPLEX_GAIN_DB));
        v.note_on(110, &pedals, 0);
        render(&mut v, 20);
        assert!(!v.duplex.is_idle());
        v.reset();
        assert!(v.duplex.is_idle(), "AllOff left the segments ringing");
        assert!(v.is_idle());
        assert_eq!(render(&mut v, 4), 0.0);
    }

    /// The cost of never being damped, paid where it belongs. A voice whose
    /// segments are still ringing must not drag the note's eighty partials
    /// through the block with them — and a voice whose segments have gone quiet
    /// must be back to writing no samples at all, which is what keeps 88
    /// undamped banks affordable.
    #[test]
    fn ringing_segments_do_not_keep_the_strings_awake_and_go_quiet_by_themselves() {
        let pedals = PedalState::new();
        let mut v = voice_from(DUPLEX_KEY, &duplex_preset(MAX_DUPLEX_GAIN_DB));
        v.note_on(110, &pedals, 0);
        render(&mut v, 20);
        v.note_off(100, &pedals, 0);
        render(&mut v, (0.5 * SAMPLE_RATE / BLOCK as f32) as usize);
        assert!(!v.is_idle() && v.strings_idle(), "the strings should be asleep");

        // Four T60s of nothing at all: the only thing that can stop a segment
        // is its own decay, and it has to actually get there.
        let quiet = bus();
        let mut board = board();
        let mut out = [0.0f32; BLOCK];
        for _ in 0..(6.0 * SAMPLE_RATE / BLOCK as f32) as usize {
            v.process(&mut out, &quiet, &mut board);
        }
        assert!(v.is_idle(), "the segments never decayed to silence");
        // ... and the voice is back on the branch that touches nothing.
        let mut untouched = [7.0f32; BLOCK];
        assert!(!v.process(&mut untouched, &quiet, &mut board));
        assert!(
            untouched.iter().all(|&x| x == 7.0),
            "an idle voice with silent segments wrote samples"
        );
    }

    /// An instrument that is not being played is silent whether or not its keys
    /// have segments: the same bit-exact silence, from the same branch.
    #[test]
    fn an_idle_voice_with_silent_segments_renders_nothing_at_all() {
        let mut v = voice_from(DUPLEX_KEY, &duplex_preset(MAX_DUPLEX_GAIN_DB));
        assert!(v.is_idle());
        let mut board = board();
        let mut out = [7.0f32; BLOCK];
        assert!(!v.process(&mut out, &bus(), &mut board));
        assert!(out.iter().all(|&x| x == 7.0));
        let (mut l, mut r) = ([1.0f32; BLOCK], [1.0f32; BLOCK]);
        board.begin_block();
        assert!(!v.process(&mut out, &bus(), &mut board));
        board.process(&mut l, &mut r);
        assert!(l.iter().chain(r.iter()).all(|&x| x == 0.0));
    }

    /// The segments answer the *bus* — another key's note — with the key's own
    /// string damped and silent. This is most of what the feature is for, and
    /// it is the path that must not become a second bus: the drive is the one
    /// `ResonanceBus::drive` produces, admittance and all.
    #[test]
    fn a_damped_keys_segments_still_answer_the_bus() {
        let preset = duplex_preset(0.0);
        let mut v = voice_from(DUPLEX_KEY, &preset);
        assert_eq!(v.string.damper(), 1.0, "the key should rest damped");

        // A bus carrying one of the segments' own frequencies and nothing else.
        let mut loud = ResonanceBus::from_preset(&preset);
        let mut board = board();
        let mut out = [0.0f32; BLOCK];
        let mut n = 0usize;
        for _ in 0..40 {
            let mut tone = [0.0f32; BLOCK];
            for x in tone.iter_mut() {
                *x = 0.05 * (std::f32::consts::TAU * duplex_hz()[0] * n as f32 / SAMPLE_RATE).sin();
                n += 1;
            }
            loud.contribute(&tone);
            loud.begin_block();
            assert!(v.process(&mut out, &loud, &mut board), "the voice slept");
        }
        assert!(!v.duplex.is_idle(), "the segments did not answer the bus");
        assert!(v.string.is_idle(), "the damped string answered too");
        assert!(
            out.iter().any(|&x| x != 0.0),
            "the answer never reached the voice's output"
        );
    }

    // ------------------------------------------------- the strike's noise

    /// The same preset with the hammer's noise at `db` relative to a strike.
    /// Everything else — including the burst's colour, so the calibration's own
    /// board measurement is unchanged — is held fixed, which is what lets the
    /// two renders below be subtracted.
    fn strike_noise_preset(db: f32) -> Preset {
        let mut preset = Preset::default();
        preset.noise.strike.centroid_hz = 1_500.0;
        preset.noise.strike.bandwidth_hz = 6_000.0;
        preset.noise.strike.decay_s = 0.06;
        preset.noise.strike.level_db = vec![crate::preset::NoiseAnchor { key: 21, db }];
        preset.validate().expect("a voiced strike is a legal preset");
        preset
    }

    /// Renders `blocks` blocks of one key's note-on through the board, and
    /// returns the stereo pair.
    fn render_note(
        preset: &Preset,
        key: u8,
        vel: u8,
        blocks: usize,
        frame: u64,
    ) -> (Vec<f32>, Vec<f32>) {
        let shapes = NoiseShapes::new(&preset.noise);
        let calibration = MechanismCalibration::new(preset, &shapes);
        let mut voice = Voice::new(key, preset, &shapes, &calibration);
        let mut board = Soundboard::new(&preset.soundboard);
        let bus = ResonanceBus::new(0.0);
        voice.note_on(vel, &PedalState::new(), frame);
        let (mut left, mut right) = (Vec::new(), Vec::new());
        let mut out = [0.0f32; BLOCK];
        let (mut l, mut r) = ([0.0f32; BLOCK], [0.0f32; BLOCK]);
        for _ in 0..blocks {
            board.begin_block();
            voice.process(&mut out, &bus, &mut board);
            board.process(&mut l, &mut r);
            left.extend_from_slice(&l);
            right.extend_from_slice(&r);
        }
        (left, right)
    }

    /// The hammer's noise plays at the note-on, in the same block as the blow,
    /// at the level the preset asks for — and a preset that asks for silence
    /// gets silence rather than a very quiet thump.
    ///
    /// Measured by subtraction: the two presets differ in nothing but
    /// `[noise.strike].level_db`, so the strings render bit for bit alike and
    /// the difference between the two outputs *is* the burst. Its peak against
    /// the strike's own peak is the number the file states, which is the same
    /// convention the other four events are quoted in
    /// (`TUNING_REPORT.md` §5, `calibrate.rs`).
    #[test]
    fn a_note_on_sounds_the_hammers_noise_at_the_level_the_preset_asks_for() {
        const WANT_DB: f32 = -20.0;
        const KEY: u8 = 60;
        let blocks = (0.3 * SAMPLE_RATE / BLOCK as f32) as usize;
        let (quiet_l, quiet_r) = render_note(&strike_noise_preset(-200.0), KEY, 90, blocks, 0);

        let magnitude_peak = |l: &[f32], r: &[f32]| {
            l.iter()
                .zip(r)
                .fold(0.0f32, |m, (&a, &b)| m.max((a * a + b * b).sqrt()))
        };
        let difference = |a: &[f32], b: &[f32]| -> Vec<f32> {
            a.iter().zip(b).map(|(x, y)| x - y).collect()
        };
        let strike_peak = magnitude_peak(&quiet_l, &quiet_r);
        assert!(strike_peak > 0.0, "the note made no sound");

        // Averaged over sixteen blows, and averaged in *amplitude*, which is
        // the statistic `calibrate.rs` places the level with. The peak of one
        // realization of a filtered noise burst is itself a random number that
        // scatters by 5-7 dB (`DECISIONS.md` 114), so no single event lands on
        // the table and none should: no two blows of a real hammer are
        // identical. What the level fixes is where many of them land.
        let voiced = strike_noise_preset(WANT_DB);
        let mut total = 0.0f32;
        let draws = 16u64;
        let mut loud = (Vec::new(), Vec::new());
        for draw in 0..draws {
            loud = render_note(&voiced, KEY, 90, blocks, draw * 4096);
            total += magnitude_peak(
                &difference(&loud.0, &quiet_l),
                &difference(&loud.1, &quiet_r),
            );
        }
        let measured = crate::types::amp_to_db(total / draws as f32 / strike_peak);
        println!("the hammer's noise renders {measured:.2} dB under the strike");
        assert!(
            (measured - WANT_DB).abs() < 3.0,
            "the hammer's noise rendered {measured:.1} dB under the strike, \
             expected {WANT_DB}"
        );

        // It arrives *with* the blow, not after it: the first block already
        // carries it.
        let first = difference(&loud.0[..BLOCK], &quiet_l[..BLOCK]);
        assert!(
            first.iter().any(|&x| x != 0.0),
            "the strike noise did not start in the hammer's own block"
        );
    }

    /// A −200 dB strike is bit-exact silence, and the burst is never even
    /// started — which is what makes `presets/default.toml` the instrument it
    /// has always been.
    #[test]
    fn a_silent_strike_leaves_the_note_bit_for_bit_what_it_was() {
        let pedals = PedalState::new();
        let mut v = voice(60);
        v.note_on(100, &pedals, 0);
        assert!(
            !v.strike_noise.is_active(),
            "the default preset started a strike burst"
        );
        let blocks = (0.3 * SAMPLE_RATE / BLOCK as f32) as usize;
        let (a_l, a_r) = render_note(&Preset::default(), 60, 90, blocks, 0);
        let (b_l, b_r) = render_note(&strike_noise_preset(-200.0), 60, 90, blocks, 0);
        assert_eq!(a_l, b_l, "a silenced strike changed the left channel");
        assert_eq!(a_r, b_r, "a silenced strike changed the right channel");
    }

    /// A re-strike into a burst that is still ringing retriggers it, and a
    /// key-off in between does not take it away: the two bursts are separate
    /// because a staccato is a release 80 ms into an attack.
    #[test]
    fn a_restrike_retriggers_the_strike_noise_and_a_release_does_not_cut_it() {
        let pedals = PedalState::new();
        let preset = strike_noise_preset(-20.0);
        let mut v = voice_from(60, &preset);
        v.note_on(90, &pedals, 0);
        assert!(v.strike_noise.is_active());
        render(&mut v, 2);

        // A note-off starts the action's own burst and leaves the hammer's
        // alone.
        v.note_off(64, &pedals, 1024);
        assert!(v.noise.is_active(), "the release made no sound");
        assert!(
            v.strike_noise.is_active(),
            "the release cut the attack noise of the note it belongs to"
        );

        // ... and a re-strike restarts the envelope rather than layering a
        // second burst.
        render(&mut v, 40);
        let before = v.strike_noise.is_active();
        v.note_on(110, &pedals, 8192);
        assert!(v.strike_noise.is_active());
        assert!(before || v.strike_noise.is_active());
    }

    #[test]
    fn restrike_does_not_reset_the_ringing_string() {
        let pedals = PedalState::new();
        let mut v = voice(60);
        v.note_on(100, &pedals, 0);
        render(&mut v, 100);
        let before = v.string().energy();
        v.note_on(1, &pedals, 0);
        assert!(v.string().energy() >= before * 0.5);
    }
}

