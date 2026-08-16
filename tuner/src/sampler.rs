//! A sample player for the SFZ the recordings ship with: the benchmark's
//! ground truth.
//!
//! [`library`](crate::library) reads the same file to find out *which key a
//! recording is of*, and says of itself that it is not an SFZ player. This is
//! the player. It exists so that a phrase can be rendered twice from one event
//! list — once by the engine, once by the recordings the engine is fitted to —
//! and the two compared. That is what `TUNING.md`'s stage 2 loss is computed
//! between, so the reference has to be worth comparing against: every gain the
//! instrument definition states is applied, and every place where a sampler
//! *cannot* do what the model does is written down rather than approximated.
//!
//! What a sample player is, in full:
//!
//! - a note-on picks the regions whose key range, velocity range and random
//!   range all admit it, and plays each one from its first sample;
//! - the level of a region is `volume` plus SFZ's velocity law scaled by
//!   `amp_veltrack`, and nothing else;
//! - a note-off releases the sustaining voices over `ampeg_release` and fires
//!   the `trigger=release` regions of that key — Salamander's damper landing
//!   (`rel*`) and its string resonances (`harm*`), attenuated by `rt_decay`
//!   decibels for every second the key was held;
//! - the sustain pedal defers the note-off until the pedal comes up;
//! - voices sum, in stereo, with no limiter and no polyphony cap.
//!
//! ### What this reference cannot do, and so does not pretend to
//!
//! Documented here because a benchmark's error bar is the list of things its
//! reference gets wrong on purpose:
//!
//! 1. **Half-pedalling does not exist.** CC 64 is a switch at 0.5; a pedal at
//!    0.3 and a pedal at 0.9 render identically. No sampler can do better —
//!    there is no recording of a string against a damper that is touching it —
//!    while the engine models the damper's travel continuously.
//! 2. **Sostenuto and una corda do nothing.** The instrument definition maps
//!    no region and no gain to CC 66 or CC 67, so the events pass through the
//!    player and change nothing. A phrase that uses them is not comparable.
//! 3. **A silently pressed key makes no sound.** [`SamplerEvent::KeyDown`]
//!    (and a note-on at velocity 0) lifts no damper here because there is no
//!    damper: the key is remembered as held, so its release still fires the
//!    key-off group, and nothing else happens. Sympathetic resonance prepared
//!    that way — the whole point of the gesture — is absent.
//! 4. **Nothing rings in sympathy with anything.** The `harm*` samples are
//!    recordings of one key's release, played back at that key's release;
//!    a chord's halo is the sum of those, not a coupled instrument.
//! 5. **A re-struck key layers over itself.** The note regions declare no
//!    `off_by`, so the file says a second strike does not stop the first, and
//!    the file is followed.
//! 6. **Above the sampled compass the pitch is shifted, not resampled from a
//!    recording of that key.** Salamander records in minor thirds (`A0`, `C1`,
//!    `D#1`, …, see `readme.txt`), each recording covering its own key ±1
//!    semitone, so two keys in three are a resampled neighbour — the same
//!    "2 of every 3 notes are interpolated" `TUNING.md` flags for stage 1,
//!    here in the audio rather than in the parameters. The shift is done once
//!    per (file, interval) with the same band-limited sinc resampler the
//!    recordings are brought onto the engine's clock with ([`audio::resample`],
//!    `rubato`), so it costs a cached resample rather than quality. The
//!    key-off group sets `pitch_keytrack=0` and is played untransposed, as the
//!    file asks.
//! 7. **The release curve's shape is an interpretation.** `ampeg_release` is
//!    the time the voice takes to go silent; players differ on the curve
//!    between. This one falls linearly in decibels to
//!    [`RELEASE_FLOOR_DB`] and then stops, which is a damper-shaped fall and
//!    cannot click. A five-millisecond floor ([`MIN_RELEASE_S`]) applies to
//!    regions that declare no release at all, so nothing is ever cut mid-wave.
//! 8. **The velocity law is SFZ's, not the recording's.** `amp_veltrack=73`
//!    means a strike at velocity `v` inside a layer's band plays that layer
//!    `0.73 · 40 log10(v/127)` dB down. Between two layers the level therefore
//!    steps: the file specifies no crossfade (no `xfin_*`/`xfout_*` opcode
//!    appears in it), so none is invented here.
//!
//! Every opcode the file *does* use is honoured. What was skipped is counted
//! rather than guessed at: [`Instrument::ignored_opcodes`] returns the census,
//! and on `SalamanderGrandPiano-V3+20200602.sfz` it is empty.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::audio::{self, Audio};
use crate::error::{Error, Result};
use crate::library::{note_number, tokens, Token};

/// What this player *is*, for the benefit of anything that caches its output.
///
/// The reference renders the `compass` and `bench` subcommands measure the
/// engine against are a pure function of
/// three things: the SFZ (and the recordings it names), the events asked for,
/// and this module. The first two are hashed by content into every cache key
/// ([`crate::cache`]); this constant stands for the third, because no cheap
/// hash of a compiled module exists.
///
/// **Bump it in the same commit as any change that moves a rendered sample.**
/// That is: the mixing and gain law in [`Sampler::render`] and its helpers, the
/// release curve, the velocity law, the pitch shift, the voice selection, the
/// event grain, the seeded round-robin draw, or any constant above that feeds
/// them. Changing a doc comment, a `Debug` impl, an error message, or
/// [`Instrument::ignored_opcodes`] does not move a sample and does not need a
/// bump.
///
/// Getting this wrong is the one way the cache can lie, so the rule is: **if
/// you are unsure whether your change moves a sample, bump it.** A needless
/// bump costs one re-render of the reference (about 30 s for the compass, 40 s
/// for the phrase set); a missed one silently scores a new engine against a
/// stale piano.
pub const SAMPLER_VERSION: u32 = 1;

/// Level a released voice has fallen to when `ampeg_release` has elapsed, in
/// dB. Below −100 dB the voice is 5 ppm of full scale and stopping it is
/// inaudible by any measure this crate can take.
pub const RELEASE_FLOOR_DB: f64 = -100.0;

/// Shortest release any voice is given, in seconds. SFZ's default
/// `ampeg_release` is zero, which is a discontinuity; five milliseconds is
/// below the resolution of anything the benchmark measures and above the
/// threshold of a click.
pub const MIN_RELEASE_S: f64 = 0.005;

/// The engine applies an event at the start of the 128-frame block that
/// contains it (`engine::render::render_to_buffer`), which is the finest grain
/// it has, live or offline. The sampler quantises to the same grid by default
/// so that a phrase rendered both ways starts its notes on the same frames and
/// the comparison is not measuring a 1.3 ms mean offset.
pub const ENGINE_BLOCK: usize = 128;

/// The opcodes this player understands. Anything else in a file is counted by
/// [`Instrument::ignored_opcodes`] and has no effect.
const HONOURED: &[&str] = &[
    "sample",
    "default_path",
    "key",
    "lokey",
    "hikey",
    "pitch_keycenter",
    "pitch_keytrack",
    "lovel",
    "hivel",
    "lorand",
    "hirand",
    "trigger",
    "volume",
    "amp_veltrack",
    "ampeg_release",
    "rt_decay",
    "group",
    "off_by",
    "on_locc64",
    "on_hicc64",
];

/// Everything a caller can tell the player to do, spelled exactly as the
/// engine's `Event` spells it so that one event list drives both.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SamplerEvent {
    /// A key press. Velocity 0 is the silent press, as in the engine.
    NoteOn { key: u8, vel: u8 },
    /// A key release. `vel` is the release velocity, which this player has no
    /// recording to vary with and therefore ignores (limitation 3 above).
    NoteOff { key: u8, vel: u8 },
    /// A key held down without striking.
    KeyDown { key: u8 },
    /// Sustain pedal, 0.0 up to 1.0 down. Read as a switch at 0.5.
    Sustain(f32),
    /// Sostenuto. Nothing in the instrument definition responds to it.
    Sostenuto(bool),
    /// Una corda. Nothing in the instrument definition responds to it.
    UnaCorda(bool),
    /// Everything off: every sounding voice is released, no release samples
    /// fire (a panic is not a key coming up), the pedal is lifted.
    AllOff,
}

/// A [`SamplerEvent`] and the time it happens, in seconds from the start of
/// the render.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimedEvent {
    pub time_s: f64,
    pub event: SamplerEvent,
}

impl TimedEvent {
    pub fn new(time_s: f64, event: SamplerEvent) -> Self {
        TimedEvent { time_s, event }
    }

    /// Note on at `time_s`, released `dur_s` later.
    pub fn note(time_s: f64, key: u8, vel: u8, dur_s: f64) -> [TimedEvent; 2] {
        [
            TimedEvent::new(time_s, SamplerEvent::NoteOn { key, vel }),
            TimedEvent::new(time_s + dur_s, SamplerEvent::NoteOff { key, vel: 64 }),
        ]
    }
}

/// How the player is run. The defaults are what a comparison against the
/// engine wants; nothing here changes what the instrument definition says.
#[derive(Clone, Copy, Debug)]
pub struct SamplerConfig {
    /// Output sample rate. Recordings at any other rate are resampled on the
    /// way in, as everywhere else in this crate.
    pub sample_rate: u32,
    /// Event timing grain, in frames. [`ENGINE_BLOCK`] reproduces the engine's
    /// own quantisation; 1 is sample-accurate.
    pub event_grain: usize,
    /// Applied to every voice. 0 dB plays the recordings at the level they
    /// were recorded at.
    pub master_gain_db: f64,
    /// Seed for the `lorand`/`hirand` draw, so a render is reproducible.
    pub seed: u64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        SamplerConfig {
            sample_rate: crate::SAMPLE_RATE,
            event_grain: ENGINE_BLOCK,
            master_gain_db: 0.0,
            seed: 0x5341_4d50_4c45_5231,
        }
    }
}

/// When a region plays.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Trigger {
    /// The default: a key going down.
    Attack,
    /// A key coming up.
    Release,
    /// `trigger=first`/`legato`, which this player does not implement. Such a
    /// region is never played and its opcode is counted as ignored.
    Unsupported,
}

/// One playable region of the instrument definition, with every opcode
/// resolved down the `<global>`/`<master>`/`<group>`/`<region>` hierarchy.
#[derive(Clone, Debug)]
pub struct Region {
    pub sample: PathBuf,
    lokey: i32,
    hikey: i32,
    keycenter: i32,
    /// Cents of transposition per key away from `keycenter`; 0 for a recording
    /// of the action, which must not be transposed.
    keytrack: i32,
    lovel: u8,
    hivel: u8,
    lorand: f64,
    hirand: f64,
    trigger: Trigger,
    volume_db: f64,
    amp_veltrack: f64,
    ampeg_release: f64,
    rt_decay: f64,
    group: Option<u32>,
    off_by: Option<u32>,
    on_locc64: Option<i32>,
    on_hicc64: Option<i32>,
}

impl Region {
    fn matches(&self, key: u8, vel: u8, draw: f64) -> bool {
        let key = i32::from(key);
        key >= self.lokey
            && key <= self.hikey
            && vel >= self.lovel
            && vel <= self.hivel
            && draw >= self.lorand
            && draw < self.hirand
    }

    /// The pedal gate, where the region has one: a region triggered by CC 64
    /// entering a range rather than by a key.
    fn cc64_gate(&self) -> Option<(i32, i32)> {
        match (self.on_locc64, self.on_hicc64) {
            (None, None) => None,
            (lo, hi) => Some((lo.unwrap_or(0), hi.unwrap_or(127))),
        }
    }

    /// Gain the region plays at for a strike at `vel`, linear.
    ///
    /// SFZ's velocity law is `(v/127)^2` at full tracking — 40 dB per decade of
    /// velocity — scaled by `amp_veltrack` over a hundred, on top of `volume`.
    fn gain(&self, vel: u8) -> f64 {
        let vel = f64::from(vel.max(1)) / 127.0;
        let veltrack_db = self.amp_veltrack / 100.0 * 40.0 * vel.log10();
        db_to_amp(self.volume_db + veltrack_db)
    }

    /// Cents this region is transposed by to play `key`.
    fn transpose_cents(&self, key: u8) -> i32 {
        (i32::from(key) - self.keycenter) * self.keytrack
    }

    /// The pitch this region is a recording of, where it is a recording of one.
    ///
    /// `None` for a region that is told not to transpose (`pitch_keytrack=0`) —
    /// a damper landing or a pedal tray is a noise, not a note, and rerouting
    /// it would be meaningless.
    pub fn recorded_key(&self) -> Option<u8> {
        if self.keytrack == 0 || self.cc64_gate().is_some() {
            return None;
        }
        u8::try_from(self.keycenter).ok().filter(|k| (21..=108).contains(k))
    }

    /// The keys this region answers to.
    pub fn key_range(&self) -> (i32, i32) {
        (self.lokey, self.hikey)
    }

    fn matches_key(&self, key: u8) -> bool {
        let key = i32::from(key);
        key >= self.lokey && key <= self.hikey
    }
}

pub fn db_to_amp(db: f64) -> f64 {
    10.0f64.powf(db / 20.0)
}

/// An SFZ instrument definition, parsed into playable regions.
#[derive(Clone, Debug, Default)]
pub struct Instrument {
    regions: Vec<Region>,
    ignored: BTreeMap<String, usize>,
}

impl Instrument {
    /// Reads an SFZ file. Sample paths resolve against the file's own
    /// directory, and against `<control> default_path` where one is set.
    pub fn from_sfz(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let root = path.parent().unwrap_or(Path::new(".")).to_path_buf();
        Self::parse(&std::fs::read_to_string(path)?, &root)
    }

    fn parse(text: &str, root: &Path) -> Result<Self> {
        let mut global = Opcodes::default();
        let mut master = Opcodes::default();
        let mut group = Opcodes::default();
        let mut region = Opcodes::default();
        let mut control = Opcodes::default();
        let mut level = Level::Global;
        let mut raw: Vec<Opcodes> = Vec::new();
        let mut ignored: BTreeMap<String, usize> = BTreeMap::new();

        for token in tokens(text) {
            match token {
                Token::Header(name) => {
                    if level == Level::Region {
                        raw.push(std::mem::take(&mut region));
                    }
                    match name {
                        "global" => {
                            global = Opcodes::default();
                            master = Opcodes::default();
                            group = Opcodes::default();
                            level = Level::Global;
                        }
                        "master" => {
                            master = Opcodes::default();
                            group = Opcodes::default();
                            level = Level::Master;
                        }
                        "group" => {
                            group = Opcodes::default();
                            level = Level::Group;
                        }
                        "region" => {
                            region = global.merged(&master).merged(&group);
                            level = Level::Region;
                        }
                        "control" => level = Level::Control,
                        // <curve>, <effect>: nothing here reads them and none
                        // of them may contain a region.
                        _ => level = Level::Other,
                    }
                }
                Token::Opcode(name, value) => {
                    if !HONOURED.contains(&name) {
                        *ignored.entry(name.to_string()).or_default() += 1;
                        continue;
                    }
                    let target = match level {
                        Level::Global => &mut global,
                        Level::Master => &mut master,
                        Level::Group => &mut group,
                        Level::Region => &mut region,
                        Level::Control => &mut control,
                        Level::Other => continue,
                    };
                    target.set(name, value);
                }
            }
        }
        if level == Level::Region {
            raw.push(region);
        }

        let base = match control.default_path.as_deref() {
            Some(relative) => root.join(relative),
            None => root.to_path_buf(),
        };
        let mut regions = Vec::with_capacity(raw.len());
        for opcodes in raw {
            let Some(sample) = opcodes.sample.as_deref() else {
                continue;
            };
            if opcodes.trigger() == Trigger::Unsupported {
                *ignored.entry("trigger".to_string()).or_default() += 1;
                continue;
            }
            regions.push(opcodes.region(base.join(sample)));
        }
        Ok(Instrument { regions, ignored })
    }

    pub fn regions(&self) -> &[Region] {
        &self.regions
    }

    /// The same instrument with every pitched key played from a **different
    /// recording**: `route(key)` names the take each key is to be transposed
    /// from, and a key the route has no answer for is left exactly as the file
    /// maps it.
    ///
    /// This exists to put a number on the cost of transposition
    /// (`DECISIONS.md` 329). A library that samples every minor third plays two
    /// keys in three from a neighbour's recording, and there is always a second
    /// neighbour that could have been used instead. Both reconstructions are
    /// equally legitimate; the distance between them is how much of the
    /// reference at those keys is the resampler rather than the piano, and it
    /// cannot be argued, only measured.
    ///
    /// Unpitched regions — the damper landing, the pedal tray, anything with
    /// `pitch_keytrack=0` or a CC 64 gate — are carried over untouched: they are
    /// recorded per key and there is nothing to reroute.
    pub fn rerouted(&self, keys: std::ops::RangeInclusive<u8>, route: impl Fn(u8) -> Option<u8>) -> Instrument {
        let mut regions: Vec<Region> = self
            .regions
            .iter()
            .filter(|r| r.recorded_key().is_none())
            .cloned()
            .collect();
        for key in keys {
            let Some(take) = route(key) else {
                // No route: keep whatever the file already says for this key.
                regions.extend(self.regions.iter().filter(|r| {
                    r.recorded_key().is_some() && r.matches_key(key)
                }).cloned());
                continue;
            };
            for region in self.regions.iter().filter(|r| r.recorded_key() == Some(take)) {
                let mut moved = region.clone();
                moved.lokey = i32::from(key);
                moved.hikey = i32::from(key);
                regions.push(moved);
            }
        }
        Instrument {
            regions,
            ignored: self.ignored.clone(),
        }
    }

    /// Opcodes present in the file that this player does not implement, and
    /// how often each appeared. The benchmark's error bar: what the reference
    /// silently does not do.
    pub fn ignored_opcodes(&self) -> &BTreeMap<String, usize> {
        &self.ignored
    }
}

/// A decoded recording, at the output rate and at one transposition.
#[derive(Debug)]
struct Buffer {
    left: Vec<f32>,
    right: Vec<f32>,
}

impl Buffer {
    fn frames(&self) -> usize {
        self.left.len()
    }
}

/// One region playing once.
#[derive(Debug)]
struct Voice {
    buffer: Arc<Buffer>,
    /// Output frame the recording's first sample lands on.
    start: usize,
    gain: f32,
    /// Frame the release began at, if it has.
    released_at: Option<usize>,
    /// Seconds from the start of the release to silence.
    release_s: f64,
    /// The group whose triggering switches this voice off, where the file
    /// names one.
    off_by: Option<u32>,
}

/// A key that is sounding: the note-on that started it, and the voices it owns.
#[derive(Debug)]
struct Sounding {
    vel: u8,
    start: usize,
    voices: Vec<usize>,
    /// The key has been let go but the pedal is holding the note.
    key_up: bool,
}

/// The player.
pub struct Sampler {
    instrument: Instrument,
    config: SamplerConfig,
    cache: HashMap<(PathBuf, i32), Arc<Buffer>>,
}

impl Sampler {
    /// Reads an SFZ file and prepares to play it.
    pub fn new(sfz: impl AsRef<Path>) -> Result<Self> {
        Self::with_config(sfz, SamplerConfig::default())
    }

    pub fn with_config(sfz: impl AsRef<Path>, config: SamplerConfig) -> Result<Self> {
        Ok(Sampler {
            instrument: Instrument::from_sfz(sfz)?,
            config,
            cache: HashMap::new(),
        })
    }

    /// A player for an instrument that was built rather than read — the one
    /// caller is [`Instrument::rerouted`], which needs to play a definition no
    /// file contains.
    pub fn from_instrument(instrument: Instrument, config: SamplerConfig) -> Self {
        Sampler {
            instrument,
            config,
            cache: HashMap::new(),
        }
    }

    pub fn instrument(&self) -> &Instrument {
        &self.instrument
    }

    pub fn config(&self) -> &SamplerConfig {
        &self.config
    }

    /// Decoded recordings held from previous renders. Unbounded on purpose:
    /// this is an offline reference and decoding the library twice is the only
    /// cost worth avoiding.
    pub fn cached_buffers(&self) -> usize {
        self.cache.len()
    }

    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Renders `events` to a stereo buffer `duration_s` long.
    ///
    /// Events are applied in time order; simultaneous events keep the order
    /// they were given in, as they do in the engine's own schedule.
    pub fn render(&mut self, events: &[TimedEvent], duration_s: f64) -> Result<Audio> {
        let rate = f64::from(self.config.sample_rate);
        let frames = (duration_s.max(0.0) * rate) as usize;
        let grain = self.config.event_grain.max(1);

        let mut schedule: Vec<(usize, SamplerEvent)> = events
            .iter()
            .map(|e| {
                let frame = (e.time_s.max(0.0) * rate).round() as usize;
                (frame / grain * grain, e.event)
            })
            .collect();
        schedule.sort_by_key(|&(frame, _)| frame);

        let mut state = Render {
            voices: Vec::new(),
            sounding: HashMap::new(),
            cc64: 0,
            rng: SplitMix64::new(self.config.seed),
            master: db_to_amp(self.config.master_gain_db),
        };

        for (frame, event) in schedule {
            self.apply(&mut state, frame, event)?;
        }

        let mut left = vec![0.0f32; frames];
        let mut right = vec![0.0f32; frames];
        for voice in &state.voices {
            mix(voice, rate, &mut left, &mut right);
        }
        Audio::new(self.config.sample_rate, vec![left, right])
    }

    fn apply(&mut self, state: &mut Render, frame: usize, event: SamplerEvent) -> Result<()> {
        match event {
            SamplerEvent::NoteOn { key, vel: 0 } | SamplerEvent::KeyDown { key } => {
                // No damper to lift and no string to leave ringing: the key is
                // only remembered so that its release still plays the key-off
                // group, which is the one thing a silent press does produce.
                state.sounding.entry(key).or_default().push(Sounding {
                    vel: crate::sampler::SILENT_PRESS_VELOCITY,
                    start: frame,
                    voices: Vec::new(),
                    key_up: false,
                });
            }
            SamplerEvent::NoteOn { key, vel } => {
                let draw = state.rng.next_f64();
                let mut voices = Vec::new();
                for index in self.matching(key, vel, draw, Trigger::Attack) {
                    let voice = self.voice(state, index, Some(key), vel, frame, 0.0)?;
                    voices.push(voice);
                }
                state.sounding.entry(key).or_default().push(Sounding {
                    vel,
                    start: frame,
                    voices,
                    key_up: false,
                });
            }
            SamplerEvent::NoteOff { key, .. } => {
                if state.cc64 >= SUSTAIN_THRESHOLD {
                    for note in state.sounding.entry(key).or_default() {
                        note.key_up = true;
                    }
                } else {
                    let notes = state.sounding.remove(&key).unwrap_or_default();
                    for note in notes {
                        self.release(state, note, key, frame, true)?;
                    }
                }
            }
            SamplerEvent::Sustain(value) => {
                let cc = (f64::from(value).clamp(0.0, 1.0) * 127.0).round() as i32;
                let previous = state.cc64;
                state.cc64 = cc;
                if previous >= SUSTAIN_THRESHOLD && cc < SUSTAIN_THRESHOLD {
                    let keys = state.released_in_order();
                    for key in keys {
                        let notes = state.sounding.remove(&key).unwrap_or_default();
                        let (up, down): (Vec<_>, Vec<_>) =
                            notes.into_iter().partition(|n| n.key_up);
                        for note in up {
                            self.release(state, note, key, frame, true)?;
                        }
                        if !down.is_empty() {
                            state.sounding.insert(key, down);
                        }
                    }
                }
                self.pedal_noise(state, frame, previous, cc)?;
            }
            // The instrument definition maps nothing to either, so neither can
            // change a sample.
            SamplerEvent::Sostenuto(_) | SamplerEvent::UnaCorda(_) => {}
            SamplerEvent::AllOff => {
                let keys = state.released_in_order();
                for key in keys {
                    let notes = state.sounding.remove(&key).unwrap_or_default();
                    for note in notes {
                        self.release(state, note, key, frame, false)?;
                    }
                }
                state.cc64 = 0;
            }
        }
        Ok(())
    }

    /// Ends a sounding note: the sustaining voices fall over their own
    /// `ampeg_release`, and — unless this is a panic rather than a key coming
    /// up — the release group of that key fires.
    fn release(
        &mut self,
        state: &mut Render,
        note: Sounding,
        key: u8,
        frame: usize,
        key_came_up: bool,
    ) -> Result<()> {
        for index in note.voices {
            let voice = &mut state.voices[index];
            if voice.released_at.is_none() {
                voice.released_at = Some(frame.max(voice.start));
            }
        }
        if !key_came_up {
            return Ok(());
        }
        // SFZ selects a release region by the velocity of the *strike*, which
        // is what Salamander's loud/soft resonance split (`harmL` at
        // `lovel=45`, `harmS` at `hivel=44`) is asking about: how hard the note
        // was hit, not how fast the key came back.
        let held_s = frame.saturating_sub(note.start) as f64 / f64::from(self.config.sample_rate);
        let draw = state.rng.next_f64();
        for index in self.matching(key, note.vel, draw, Trigger::Release) {
            self.voice(state, index, Some(key), note.vel, frame, held_s)?;
        }
        Ok(())
    }

    /// The pedal's own recordings, which fire when CC 64 *enters* the range a
    /// region gates on — Salamander gates the tray going down on 126–127 and
    /// coming up on 0–1, so only a pedal moved to its stop makes a noise.
    fn pedal_noise(
        &mut self,
        state: &mut Render,
        frame: usize,
        previous: i32,
        cc: i32,
    ) -> Result<()> {
        if previous == cc {
            return Ok(());
        }
        let draw = state.rng.next_f64();
        let triggered: Vec<usize> = self
            .instrument
            .regions
            .iter()
            .enumerate()
            .filter(|(_, region)| {
                let Some((lo, hi)) = region.cc64_gate() else {
                    return false;
                };
                let entering = cc >= lo && cc <= hi;
                let was_inside = previous >= lo && previous <= hi;
                entering
                    && !was_inside
                    && draw >= region.lorand
                    && draw < region.hirand
            })
            .map(|(index, _)| index)
            .collect();
        for index in triggered {
            // `off_by` is the file's own note-stealing: the tray coming up
            // stops the tray going down.
            if let Some(group) = self.instrument.regions[index].group {
                for voice in state.voices.iter_mut() {
                    if voice.off_by == Some(group) && voice.released_at.is_none() {
                        voice.released_at = Some(frame.max(voice.start));
                    }
                }
            }
            // A pedal region is on no key: it is played at the pitch it was
            // recorded at, and at the velocity MIDI gives a controller move,
            // i.e. none — so `amp_veltrack` has nothing to scale and the
            // region plays at its `volume`.
            self.voice(state, index, None, 127, frame, 0.0)?;
        }
        Ok(())
    }

    fn matching(&self, key: u8, vel: u8, draw: f64, trigger: Trigger) -> Vec<usize> {
        self.instrument
            .regions
            .iter()
            .enumerate()
            .filter(|(_, region)| {
                region.trigger == trigger
                    && region.cc64_gate().is_none()
                    && region.matches(key, vel, draw)
            })
            .map(|(index, _)| index)
            .collect()
    }

    /// Starts one region and returns the voice's index.
    fn voice(
        &mut self,
        state: &mut Render,
        index: usize,
        key: Option<u8>,
        vel: u8,
        frame: usize,
        held_s: f64,
    ) -> Result<usize> {
        let region = &self.instrument.regions[index];
        let cents = key.map_or(0, |key| region.transpose_cents(key));
        // `rt_decay` is the file saying that a release recording is quieter
        // the longer the note has been ringing, in dB per second held.
        let gain = region.gain(vel) * db_to_amp(-region.rt_decay * held_s) * state.master;
        let release_s = region.ampeg_release.max(MIN_RELEASE_S);
        let off_by = region.off_by;
        let path = region.sample.clone();
        let buffer = self.buffer(&path, cents)?;
        state.voices.push(Voice {
            buffer,
            start: frame,
            gain: gain as f32,
            released_at: None,
            release_s,
            off_by,
        });
        Ok(state.voices.len() - 1)
    }

    /// The recording behind a region, at the output rate and transposed by
    /// `cents`, decoded once and kept.
    fn buffer(&mut self, path: &Path, cents: i32) -> Result<Arc<Buffer>> {
        let entry = (path.to_path_buf(), cents);
        if let Some(buffer) = self.cache.get(&entry) {
            return Ok(Arc::clone(buffer));
        }
        let decoded = audio::load_at(path, self.config.sample_rate)?;
        let mut channels = decoded.channels;
        if cents != 0 {
            // Playing a recording `cents` sharp is reading it that much faster,
            // which is the same signal as one resampled to a proportionally
            // lower rate and then played at the output rate. Stated as a rate
            // ratio so the crate's own band-limited sinc resampler can do it;
            // the 1e8 reference keeps the ratio's rounding error under a
            // hundred-thousandth of a cent.
            const REFERENCE_HZ: f64 = 1.0e8;
            let speed = 2.0f64.powf(f64::from(cents) / 1200.0);
            let target = (REFERENCE_HZ / speed).round() as u32;
            channels = audio::resample(&channels, REFERENCE_HZ as u32, target)?;
        }
        let (left, right) = match channels.len() {
            0 => return Err(Error::Unsupported(format!("{}: no channels", path.display()))),
            1 => (channels[0].clone(), channels[0].clone()),
            _ => (channels[0].clone(), channels[1].clone()),
        };
        let buffer = Arc::new(Buffer { left, right });
        self.cache.insert(entry, Arc::clone(&buffer));
        Ok(buffer)
    }
}

/// Velocity a silent press is remembered at, so that the key-off group it
/// eventually fires is selected the way the softest real strike would select
/// it. The engine spells the same idea as `ESCAPEMENT_VELOCITY`.
const SILENT_PRESS_VELOCITY: u8 = 1;

/// CC 64 at or above this is a pedal held down. Half-pedalling is limitation 1.
const SUSTAIN_THRESHOLD: i32 = 64;

/// Everything a render accumulates while the events are walked.
struct Render {
    voices: Vec<Voice>,
    sounding: HashMap<u8, Vec<Sounding>>,
    cc64: i32,
    rng: SplitMix64,
    master: f64,
}

impl Render {
    /// The sounding keys, **lowest first**, for the two events that let go of
    /// more than one key at once: the sustain pedal coming up, and `AllOff`.
    ///
    /// The order matters and used to be the hash map's. Releasing a key appends
    /// its release voices to `voices`, and [`Sampler::render`] sums that list
    /// into the output buffers in order, so the order keys are released in is
    /// the order a pedal-up chord's voices are *added together* — and floating
    /// point addition is not associative. `HashMap`'s iteration order is seeded
    /// randomly per process, so two runs of the same phrase produced reference
    /// audio differing by an ulp wherever the pedal lifted a chord: measured on
    /// `bench`'s six phrases, `chords_pedal` and `excerpt` (the two that
    /// pedal) came back with 60-90 thousand samples one ulp apart between runs,
    /// and the other four bit-identical. No metric in `REALISM.md` moved by so
    /// much as a printed digit — but a reference that is not reproducible cannot
    /// be cached, cannot be diffed between machines, and cannot be quoted, so
    /// the order is now the keyboard's (`DECISIONS.md` 284).
    fn released_in_order(&self) -> Vec<u8> {
        let mut keys: Vec<u8> = self.sounding.keys().copied().collect();
        keys.sort_unstable();
        keys
    }
}

/// Adds one voice to the output. The release envelope is exponential — linear
/// in dB — so it is stepped by a constant factor rather than evaluated, which
/// keeps a thirty-second phrase to one multiply per sample.
fn mix(voice: &Voice, rate: f64, left: &mut [f32], right: &mut [f32]) {
    let frames = left.len();
    if voice.start >= frames {
        return;
    }
    let count = voice.buffer.frames().min(frames - voice.start);
    let step = db_to_amp(RELEASE_FLOOR_DB / (voice.release_s * rate));
    let mut envelope = 1.0f64;
    for i in 0..count {
        let frame = voice.start + i;
        if let Some(released) = voice.released_at {
            if frame >= released {
                if envelope <= RELEASE_FLOOR {
                    return;
                }
                envelope *= step;
            }
        }
        let gain = voice.gain * envelope as f32;
        left[frame] += gain * voice.buffer.left[i];
        right[frame] += gain * voice.buffer.right[i];
    }
}

/// Linear amplitude of [`RELEASE_FLOOR_DB`].
const RELEASE_FLOOR: f64 = 1.0e-5;

/// The generator behind `lorand`/`hirand`. Seeded, so a render is
/// reproducible: two runs of the same phrase pick the same pedal recordings.
/// Same construction as [`crate::synth`]'s noise, for the same reason.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    fn next_f64(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Level {
    Global,
    Master,
    Group,
    Region,
    Control,
    Other,
}

/// The opcodes of one level of the hierarchy, before they are resolved.
#[derive(Clone, Debug, Default)]
struct Opcodes {
    sample: Option<String>,
    default_path: Option<String>,
    key: Option<i32>,
    lokey: Option<i32>,
    hikey: Option<i32>,
    pitch_keycenter: Option<i32>,
    pitch_keytrack: Option<i32>,
    lovel: Option<u8>,
    hivel: Option<u8>,
    lorand: Option<f64>,
    hirand: Option<f64>,
    trigger: Option<String>,
    volume: Option<f64>,
    amp_veltrack: Option<f64>,
    ampeg_release: Option<f64>,
    rt_decay: Option<f64>,
    group: Option<u32>,
    off_by: Option<u32>,
    on_locc64: Option<i32>,
    on_hicc64: Option<i32>,
}

impl Opcodes {
    fn merged(&self, other: &Opcodes) -> Opcodes {
        Opcodes {
            sample: other.sample.clone().or_else(|| self.sample.clone()),
            default_path: other
                .default_path
                .clone()
                .or_else(|| self.default_path.clone()),
            key: other.key.or(self.key),
            lokey: other.lokey.or(self.lokey),
            hikey: other.hikey.or(self.hikey),
            pitch_keycenter: other.pitch_keycenter.or(self.pitch_keycenter),
            pitch_keytrack: other.pitch_keytrack.or(self.pitch_keytrack),
            lovel: other.lovel.or(self.lovel),
            hivel: other.hivel.or(self.hivel),
            lorand: other.lorand.or(self.lorand),
            hirand: other.hirand.or(self.hirand),
            trigger: other.trigger.clone().or_else(|| self.trigger.clone()),
            volume: other.volume.or(self.volume),
            amp_veltrack: other.amp_veltrack.or(self.amp_veltrack),
            ampeg_release: other.ampeg_release.or(self.ampeg_release),
            rt_decay: other.rt_decay.or(self.rt_decay),
            group: other.group.or(self.group),
            off_by: other.off_by.or(self.off_by),
            on_locc64: other.on_locc64.or(self.on_locc64),
            on_hicc64: other.on_hicc64.or(self.on_hicc64),
        }
    }

    fn set(&mut self, name: &str, value: &str) {
        match name {
            "sample" => self.sample = Some(value.replace('\\', "/")),
            "default_path" => self.default_path = Some(value.replace('\\', "/")),
            "key" => self.key = note_number(value),
            "lokey" => self.lokey = note_number(value),
            "hikey" => self.hikey = note_number(value),
            "pitch_keycenter" => self.pitch_keycenter = note_number(value),
            "pitch_keytrack" => self.pitch_keytrack = value.parse().ok(),
            "lovel" => self.lovel = value.parse().ok(),
            "hivel" => self.hivel = value.parse().ok(),
            "lorand" => self.lorand = value.parse().ok(),
            "hirand" => self.hirand = value.parse().ok(),
            "trigger" => self.trigger = Some(value.to_ascii_lowercase()),
            "volume" => self.volume = value.parse().ok(),
            "amp_veltrack" => self.amp_veltrack = value.parse().ok(),
            "ampeg_release" => self.ampeg_release = value.parse().ok(),
            "rt_decay" => self.rt_decay = value.parse().ok(),
            "group" => self.group = value.parse().ok(),
            "off_by" => self.off_by = value.parse().ok(),
            "on_locc64" => self.on_locc64 = value.parse().ok(),
            "on_hicc64" => self.on_hicc64 = value.parse().ok(),
            _ => {}
        }
    }

    fn trigger(&self) -> Trigger {
        match self.trigger.as_deref() {
            None | Some("attack") => Trigger::Attack,
            Some("release") => Trigger::Release,
            _ => Trigger::Unsupported,
        }
    }

    fn region(&self, sample: PathBuf) -> Region {
        let keycenter = self.pitch_keycenter.or(self.key).unwrap_or(60);
        Region {
            sample,
            lokey: self.lokey.or(self.key).unwrap_or(0),
            hikey: self.hikey.or(self.key).unwrap_or(127),
            keycenter,
            keytrack: self.pitch_keytrack.unwrap_or(100),
            lovel: self.lovel.unwrap_or(0),
            hivel: self.hivel.unwrap_or(127),
            lorand: self.lorand.unwrap_or(0.0),
            hirand: self.hirand.unwrap_or(1.0),
            trigger: self.trigger(),
            volume_db: self.volume.unwrap_or(0.0),
            amp_veltrack: self.amp_veltrack.unwrap_or(100.0),
            ampeg_release: self.ampeg_release.unwrap_or(0.0),
            rt_decay: self.rt_decay.unwrap_or(0.0),
            group: self.group,
            off_by: self.off_by,
            on_locc64: self.on_locc64,
            on_hicc64: self.on_hicc64,
        }
    }
}

/// Driving the player from the engine's own event list.
///
/// The point of the reference is that one phrase renders both ways, so the
/// translation lives here rather than in whatever is doing the comparing —
/// there is exactly one mapping and it is tested. The engine is a
/// dev-dependency of this crate (`Cargo.toml` says why), so this module is
/// compiled for the crate's own tests and for anyone who asks for the
/// `engine-events` feature.
#[cfg(any(test, feature = "engine-events"))]
pub mod engine_events {
    use super::{SamplerEvent, TimedEvent};
    use crate::error::{Error, Result};
    use piano_emulator::{Event, PedalEvent, RenderEvent};
    use std::path::Path;

    /// The same performance, spelled for the sampler. Total: every engine
    /// event has a meaning here, including the ones this player answers with
    /// silence.
    pub fn from_render_events(events: &[RenderEvent]) -> Vec<TimedEvent> {
        events
            .iter()
            .map(|e| {
                let event = match e.event {
                    Event::NoteOn { key, vel } => SamplerEvent::NoteOn { key, vel },
                    Event::NoteOff { key, vel } => SamplerEvent::NoteOff { key, vel },
                    Event::KeyDown { key } => SamplerEvent::KeyDown { key },
                    Event::Pedal(PedalEvent::Sustain(v)) => SamplerEvent::Sustain(v),
                    Event::Pedal(PedalEvent::Sostenuto(on)) => SamplerEvent::Sostenuto(on),
                    Event::Pedal(PedalEvent::UnaCorda(on)) => SamplerEvent::UnaCorda(on),
                    Event::AllOff => SamplerEvent::AllOff,
                };
                TimedEvent::new(f64::from(e.time_s), event)
            })
            .collect()
    }

    /// The same performance, spelled for the engine. The exact inverse of
    /// [`from_render_events`], and the direction every comparison uses: a
    /// phrase is written in the sampler's event type because that is the one
    /// the tuner owns, and both sides then render the same list of gestures.
    /// Nothing is dropped or reinterpreted, which is the property that makes
    /// the two renders comparable at all.
    pub fn to_render_events(events: &[TimedEvent]) -> Vec<RenderEvent> {
        events
            .iter()
            .map(|e| {
                let event = match e.event {
                    SamplerEvent::NoteOn { key, vel } => Event::NoteOn { key, vel },
                    SamplerEvent::NoteOff { key, vel } => Event::NoteOff { key, vel },
                    SamplerEvent::KeyDown { key } => Event::KeyDown { key },
                    SamplerEvent::Sustain(v) => Event::Pedal(PedalEvent::Sustain(v)),
                    SamplerEvent::Sostenuto(v) => Event::Pedal(PedalEvent::Sostenuto(v)),
                    SamplerEvent::UnaCorda(v) => Event::Pedal(PedalEvent::UnaCorda(v)),
                    SamplerEvent::AllOff => Event::AllOff,
                };
                RenderEvent::new(e.time_s as f32, event)
            })
            .collect()
    }

    /// A standard MIDI file, read by the engine's own reader — tempo map, CC
    /// 64/66/67 and all — so the sampler and the engine cannot disagree about
    /// what the file says. Returns the events and the render length the engine
    /// would use for them.
    pub fn from_midi_file(path: impl AsRef<Path>) -> Result<(Vec<TimedEvent>, f64)> {
        let performance = piano_emulator::midi::load(path.as_ref())
            .map_err(|e| Error::Config(format!("{}: {e}", path.as_ref().display())))?;
        let duration = f64::from(performance.duration_s());
        Ok((from_render_events(&performance.events), duration))
    }
}

#[cfg(any(test, feature = "engine-events"))]
impl Sampler {
    /// Renders the engine's own event list.
    pub fn render_engine_events(
        &mut self,
        events: &[piano_emulator::RenderEvent],
        duration_s: f64,
    ) -> Result<Audio> {
        let events = engine_events::from_render_events(events);
        self.render(&events, duration_s)
    }

    /// Renders a standard MIDI file, for the length the engine would render it
    /// for unless `duration_s` says otherwise.
    pub fn render_midi(
        &mut self,
        path: impl AsRef<Path>,
        duration_s: Option<f64>,
    ) -> Result<Audio> {
        let (events, default_duration) = engine_events::from_midi_file(path)?;
        self.render(&events, duration_s.unwrap_or(default_duration))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    const RATE: u32 = 48_000;

    /// A fixture library: sine recordings of known amplitude and frequency,
    /// written as float WAV so that what the player reads back is bit for bit
    /// what these tests generated.
    struct Fixture {
        dir: PathBuf,
    }

    impl Fixture {
        fn new(name: &str) -> Fixture {
            let dir = std::env::temp_dir().join(format!("piano-tuner-sampler-{name}"));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(dir.join("samples")).unwrap();
            Fixture { dir }
        }

        /// A stereo sine, right channel at half the left channel's level so
        /// that channel routing is testable.
        fn sine(&self, name: &str, hz: f64, amplitude: f64, seconds: f64) -> Vec<f32> {
            let frames = (seconds * f64::from(RATE)) as usize;
            let left: Vec<f32> = (0..frames)
                .map(|i| (amplitude * (TAU * hz * i as f64 / f64::from(RATE)).sin()) as f32)
                .collect();
            let right: Vec<f32> = left.iter().map(|&x| x * 0.5).collect();
            let audio = Audio::new(RATE, vec![left.clone(), right]).unwrap();
            audio.write_wav(self.dir.join("samples").join(name)).unwrap();
            left
        }

        fn sfz(&self, text: &str) -> PathBuf {
            let path = self.dir.join("instrument.sfz");
            std::fs::write(&path, text).unwrap();
            path
        }
    }

    fn config() -> SamplerConfig {
        SamplerConfig {
            event_grain: 1,
            ..SamplerConfig::default()
        }
    }

    fn peak(x: &[f32]) -> f32 {
        x.iter().fold(0.0f32, |m, &v| m.max(v.abs()))
    }

    /// RMS of a window given in seconds.
    fn rms(x: &[f32], from_s: f64, to_s: f64) -> f64 {
        let from = (from_s * f64::from(RATE)) as usize;
        let to = ((to_s * f64::from(RATE)) as usize).min(x.len());
        if to <= from {
            return 0.0;
        }
        let sum: f64 = x[from..to].iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        (sum / (to - from) as f64).sqrt()
    }

    // ---------------------------------------------------------------------
    // The instrument definition

    #[test]
    fn the_salamander_opcodes_are_all_honoured_and_the_hierarchy_resolves() {
        // The groups and regions of the real file, in miniature: two velocity
        // layers under one group, a release group, the key-off group with its
        // `pitch_keytrack=0`, and the pedal's two gated groups.
        let sfz = "\
<group> amp_veltrack=73 ampeg_release=1
<region> sample=samples/C4v1.wav lokey=59 hikey=61 lovel=1 hivel=26 pitch_keycenter=60
<region> sample=samples/C4v2.wav lokey=59 hikey=61 lovel=27 pitch_keycenter=60
<group> trigger=release volume=-4 amp_veltrack=94 rt_decay=6
<region> sample=samples/harmC4.wav lokey=59 hikey=61 lovel=45 pitch_keycenter=60
<group> trigger=release pitch_keytrack=0 volume=-37 amp_veltrack=82 rt_decay=2
<region> sample=samples/rel40.wav lokey=60 hikey=60
<group> group=1 hikey=-1 lokey=-1 on_locc64=126 on_hicc64=127 off_by=2 volume=-20
<region> sample=samples/pedalD1.wav lorand=0 hirand=0.5
<region> sample=samples/pedalD2.wav lorand=0.5 hirand=1
";
        let instrument = Instrument::parse(sfz, Path::new("/lib")).unwrap();
        assert!(instrument.ignored_opcodes().is_empty());
        let regions = instrument.regions();
        assert_eq!(regions.len(), 6);

        // The note group's opcodes reached both its regions, and the velocity
        // split did not leak upwards.
        assert_eq!(regions[0].amp_veltrack, 73.0);
        assert_eq!(regions[0].ampeg_release, 1.0);
        assert_eq!(regions[0].hivel, 26);
        assert_eq!(regions[1].hivel, 127);
        assert_eq!(regions[0].sample, Path::new("/lib/samples/C4v1.wav"));

        // The release groups, told apart by what they say about pitch.
        assert_eq!(regions[2].trigger, Trigger::Release);
        assert_eq!(regions[2].keytrack, 100);
        assert_eq!(regions[2].rt_decay, 6.0);
        assert_eq!(regions[3].keytrack, 0);
        assert_eq!(regions[3].volume_db, -37.0);

        // The pedal: gated on the controller, on no key, and randomised.
        assert_eq!(regions[4].cc64_gate(), Some((126, 127)));
        assert_eq!(regions[4].off_by, Some(2));
        assert_eq!(regions[5].lorand, 0.5);
        assert!(!regions[4].matches(60, 100, 0.25), "a pedal region is on no key");
    }

    #[test]
    fn an_opcode_the_player_does_not_implement_is_counted_not_guessed() {
        let sfz = "<region> sample=a.wav key=60 cutoff=500 fil_type=lpf_2p ampeg_attack=0.1\n\
                   <region> sample=b.wav key=61 cutoff=800\n\
                   <group> trigger=first\n<region> sample=c.wav key=62\n";
        let instrument = Instrument::parse(sfz, Path::new(".")).unwrap();
        assert_eq!(instrument.ignored_opcodes()["cutoff"], 2);
        assert_eq!(instrument.ignored_opcodes()["fil_type"], 1);
        assert_eq!(instrument.ignored_opcodes()["ampeg_attack"], 1);
        // A trigger mode that is not implemented drops the region rather than
        // playing it at the wrong time.
        assert_eq!(instrument.ignored_opcodes()["trigger"], 1);
        assert_eq!(instrument.regions().len(), 2);
    }

    /// A library that samples every minor third, mapped over its neighbours the
    /// way an SFZ does it, rerouted onto the *other* neighbour's take.
    #[test]
    fn rerouting_plays_every_unrecorded_key_from_the_other_take() {
        let mut sfz = String::from("<group>\n");
        for centre in [57u8, 60, 63, 66] {
            sfz.push_str(&format!(
                "<region> sample=note{centre}.wav lokey={} hikey={} pitch_keycenter={centre}\n",
                centre - 1,
                centre + 1
            ));
        }
        // Damper landings: per key, unpitched, and none of this touches them.
        sfz.push_str("<group> trigger=release pitch_keytrack=0\n");
        for centre in [57u8, 60, 63, 66] {
            sfz.push_str(&format!(
                "<region> sample=rel{centre}.wav lokey={centre} hikey={centre}\n"
            ));
        }
        let instrument = Instrument::parse(&sfz, Path::new(".")).unwrap();
        let recorded = [57u8, 60, 63, 66];
        // Second-nearest take, or the key's own where it has one.
        let route = |key: u8| -> Option<u8> {
            if recorded.contains(&key) {
                return Some(key);
            }
            let nearest = *recorded.iter().min_by_key(|&&k| k.abs_diff(key))?;
            recorded
                .iter()
                .copied()
                .filter(|&k| k != nearest)
                .min_by_key(|&k| k.abs_diff(key))
        };
        let moved = instrument.rerouted(58..=65, route);

        let take_for = |inst: &Instrument, key: u8| -> Option<u8> {
            inst.regions()
                .iter()
                .find(|r| r.recorded_key().is_some() && r.matches_key(key))
                .and_then(Region::recorded_key)
        };
        // Before: the nearest take. After: the other one.
        for (key, before, after) in [(58u8, 57u8, 60u8), (59, 60, 57), (61, 60, 63), (62, 63, 60)] {
            assert_eq!(take_for(&instrument, key), Some(before), "key {key} before");
            assert_eq!(take_for(&moved, key), Some(after), "key {key} after");
        }
        // A recorded key keeps its own recording either way.
        for key in [60u8, 63] {
            assert_eq!(take_for(&moved, key), Some(key), "key {key} was rerouted");
        }
        // And the unpitched release regions are carried over untouched.
        let unpitched = |inst: &Instrument| {
            inst.regions()
                .iter()
                .filter(|r| r.recorded_key().is_none())
                .count()
        };
        assert_eq!(unpitched(&moved), unpitched(&instrument));
        assert_eq!(unpitched(&instrument), 4);
    }

    #[test]
    fn the_velocity_law_is_the_files_own() {
        let region = Instrument::parse(
            "<region> sample=a.wav key=60 amp_veltrack=73 volume=-4\n",
            Path::new("."),
        )
        .unwrap()
        .regions[0]
            .clone();
        // 0.73 * 40 log10(v/127) dB on top of volume.
        let expected = |v: u8| {
            10f64.powf((-4.0 + 0.73 * 40.0 * (f64::from(v) / 127.0).log10()) / 20.0)
        };
        for v in [1u8, 32, 64, 100, 127] {
            assert!((region.gain(v) - expected(v)).abs() < 1e-12, "velocity {v}");
        }
        // Full velocity is the region's `volume` and nothing else.
        assert!((region.gain(127) - db_to_amp(-4.0)).abs() < 1e-12);
    }

    // ---------------------------------------------------------------------
    // Playback

    #[test]
    fn a_single_note_is_its_recording_gain_adjusted() {
        let fixture = Fixture::new("single-note");
        let source = fixture.sine("C4v2.wav", 220.0, 0.5, 1.0);
        let path = fixture.sfz(
            "<group> amp_veltrack=73 ampeg_release=1\n\
             <region> sample=samples/C4v2.wav lokey=59 hikey=61 lovel=27 pitch_keycenter=60\n",
        );
        let mut sampler = Sampler::with_config(&path, config()).unwrap();
        let out = sampler
            .render(
                &[TimedEvent::new(0.0, SamplerEvent::NoteOn { key: 60, vel: 100 })],
                1.5,
            )
            .unwrap();

        let gain = 10f64.powf(0.73 * 40.0 * (100.0f64 / 127.0).log10() / 20.0) as f32;
        assert_eq!(out.sample_rate, RATE);
        assert_eq!(out.channel_count(), 2);
        for (i, &x) in source.iter().enumerate() {
            assert!(
                (out.channels[0][i] - x * gain).abs() < 1e-7,
                "frame {i}: {} vs {}",
                out.channels[0][i],
                x * gain
            );
            // The recording's own stereo balance survives untouched.
            assert!((out.channels[1][i] - 0.5 * x * gain).abs() < 1e-7);
        }
        // Nothing is invented past the end of the recording.
        assert_eq!(peak(&out.channels[0][source.len()..]), 0.0);
    }

    #[test]
    fn a_region_at_unity_gain_is_reproduced_bit_for_bit() {
        // The strongest form of "this is the recording": with nothing to scale
        // and nothing to transpose, the player is a copy. Anything that
        // resampled, dithered or normalised on the way through would show up
        // here as a last-bit difference.
        let fixture = Fixture::new("bit-exact");
        let source = fixture.sine("note.wav", 220.0, 0.4, 0.5);
        let path = fixture.sfz(
            "<region> sample=samples/note.wav key=60 amp_veltrack=0 volume=0\n",
        );
        let mut sampler = Sampler::with_config(&path, config()).unwrap();
        let out = sampler
            .render(
                &[TimedEvent::new(0.0, SamplerEvent::NoteOn { key: 60, vel: 127 })],
                0.5,
            )
            .unwrap();
        assert_eq!(out.channels[0], source);
    }

    #[test]
    fn the_layer_the_strike_lands_in_is_the_one_that_plays() {
        let fixture = Fixture::new("layers");
        fixture.sine("v1.wav", 220.0, 0.1, 0.5);
        fixture.sine("v2.wav", 440.0, 0.1, 0.5);
        let path = fixture.sfz(
            "<group> amp_veltrack=0\n\
             <region> sample=samples/v1.wav key=60 lovel=1 hivel=26\n\
             <region> sample=samples/v2.wav key=60 lovel=27 hivel=127\n",
        );
        let mut sampler = Sampler::with_config(&path, config()).unwrap();
        // Zero velocity tracking isolates the choice of layer from its gain.
        for (vel, hz) in [(20u8, 220.0f64), (90, 440.0)] {
            let out = sampler
                .render(
                    &[TimedEvent::new(0.0, SamplerEvent::NoteOn { key: 60, vel })],
                    0.5,
                )
                .unwrap();
            let measured = zero_crossings(&out.channels[0][..RATE as usize / 4]) as f64;
            assert!(
                (measured - hz).abs() < 4.0,
                "velocity {vel} played {measured} Hz, expected {hz}"
            );
        }
    }

    #[test]
    fn a_key_away_from_the_recordings_own_is_transposed_by_the_interval() {
        let fixture = Fixture::new("transpose");
        fixture.sine("C4.wav", 200.0, 0.2, 1.0);
        let path = fixture.sfz(
            "<region> sample=samples/C4.wav lokey=59 hikey=61 pitch_keycenter=60 amp_veltrack=0\n",
        );
        let mut sampler = Sampler::with_config(&path, config()).unwrap();
        for (key, expected) in [(59u8, 200.0 / 1.059_463), (60, 200.0), (61, 200.0 * 1.059_463)] {
            let out = sampler
                .render(
                    &[TimedEvent::new(0.0, SamplerEvent::NoteOn { key, vel: 80 })],
                    1.0,
                )
                .unwrap();
            // Measured over half a second, well inside the shortest of the
            // three transposed buffers.
            let window = &out.channels[0][..RATE as usize / 2];
            let measured = zero_crossings(window) as f64;
            assert!(
                (measured - expected).abs() < 2.0,
                "key {key} played {measured} Hz, expected {expected}"
            );
        }
    }

    #[test]
    fn a_note_off_releases_the_sustaining_sample_and_fires_the_release_group() {
        let fixture = Fixture::new("release");
        // The two voices are told apart by frequency, so each can be measured
        // while the other is sounding.
        fixture.sine("note.wav", 220.0, 0.4, 4.0);
        fixture.sine("harm.wav", 660.0, 0.4, 1.0);
        let path = fixture.sfz(
            "<group> amp_veltrack=0 ampeg_release=1\n\
             <region> sample=samples/note.wav key=60\n\
             <group> trigger=release amp_veltrack=0 volume=-6 rt_decay=6\n\
             <region> sample=samples/harm.wav key=60\n",
        );
        let mut sampler = Sampler::with_config(&path, config()).unwrap();
        let out = sampler
            .render(&TimedEvent::note(0.0, 60, 90, 1.0), 3.0)
            .unwrap();
        let left = &out.channels[0];

        // While the key is down: the recording, and no release sample.
        assert!((tone_level(left, 220.0, 0.2, 0.9) - 0.4).abs() < 0.01);
        assert!(tone_level(left, 660.0, 0.2, 0.9) < 1e-4);

        // The release group fires at the note-off, at `volume` minus
        // `rt_decay` decibels for the second the key was held.
        let expected = 0.4 * db_to_amp(-6.0 - 6.0 * 1.0);
        let fired = tone_level(left, 660.0, 1.02, 1.3);
        assert!(
            (fired - expected).abs() < 0.02 * expected,
            "release sample at {fired}, expected {expected}"
        );

        // And the sustaining voice falls over its `ampeg_release`: half of one
        // second into a fall to -100 dB is -50 dB, and it is gone at 1 s.
        let half = tone_level(left, 220.0, 1.49, 1.51);
        let predicted = 0.4 * db_to_amp(RELEASE_FLOOR_DB / 2.0);
        assert!(
            (half / predicted - 1.0).abs() < 0.15,
            "half a release in: {half}, expected {predicted}"
        );
        assert!(tone_level(left, 220.0, 2.1, 2.5) < 1e-6);
        // The release sample is a second long and nothing outlives it.
        assert_eq!(peak(&left[(2.1 * f64::from(RATE)) as usize..]), 0.0);
    }

    #[test]
    fn a_release_sample_is_quieter_the_longer_the_key_was_held() {
        let fixture = Fixture::new("rt-decay");
        fixture.sine("note.wav", 220.0, 0.2, 8.0);
        fixture.sine("harm.wav", 660.0, 0.4, 0.5);
        let path = fixture.sfz(
            "<group> amp_veltrack=0 ampeg_release=0.01\n\
             <region> sample=samples/note.wav key=60\n\
             <group> trigger=release amp_veltrack=0 rt_decay=6\n\
             <region> sample=samples/harm.wav key=60\n",
        );
        let mut sampler = Sampler::with_config(&path, config()).unwrap();
        let mut level = |held: f64| {
            let out = sampler
                .render(&TimedEvent::note(0.0, 60, 90, held), held + 1.0)
                .unwrap();
            rms(&out.channels[0], held + 0.1, held + 0.4)
        };
        let short = level(0.5);
        let long = level(2.5);
        // Two seconds more holding at 6 dB per second: 12 dB quieter.
        let ratio_db = 20.0 * (long / short).log10();
        assert!((ratio_db + 12.0).abs() < 0.3, "rt_decay gave {ratio_db:.2} dB");
    }

    #[test]
    fn the_sustain_pedal_holds_the_note_and_defers_its_release() {
        let fixture = Fixture::new("pedal");
        fixture.sine("note.wav", 220.0, 0.4, 6.0);
        fixture.sine("harm.wav", 660.0, 0.4, 0.5);
        let path = fixture.sfz(
            "<group> amp_veltrack=0 ampeg_release=0.05\n\
             <region> sample=samples/note.wav key=60\n\
             <group> trigger=release amp_veltrack=0\n\
             <region> sample=samples/harm.wav key=60\n",
        );
        let mut sampler = Sampler::with_config(&path, config()).unwrap();
        let events = vec![
            TimedEvent::new(0.0, SamplerEvent::Sustain(1.0)),
            TimedEvent::new(0.1, SamplerEvent::NoteOn { key: 60, vel: 90 }),
            TimedEvent::new(1.0, SamplerEvent::NoteOff { key: 60, vel: 64 }),
            TimedEvent::new(3.0, SamplerEvent::Sustain(0.0)),
        ];
        let out = sampler.render(&events, 5.0).unwrap();
        let left = &out.channels[0];

        // The key came up at 1.0 s and the note is still there at full level
        // two seconds later.
        let held = rms(left, 2.0, 2.9);
        assert!((held - 0.4 / 2f64.sqrt()).abs() < 0.02, "note faded while held: {held}");
        // The release sample did not fire at the note-off.
        assert!(rms(left, 1.05, 1.4) - held < 0.02);
        // The pedal coming up ends it: the note is gone and the release sample
        // is there instead.
        assert!(rms(left, 3.1, 3.4) > 0.2, "release sample missing after the pedal");
        assert_eq!(peak(&left[(3.6 * f64::from(RATE)) as usize..]), 0.0);

        // Half-pedalling is a documented limitation, and this is what it looks
        // like: a pedal at 0.5 is a pedal down.
        let half = vec![
            TimedEvent::new(0.0, SamplerEvent::Sustain(0.5)),
            TimedEvent::new(0.1, SamplerEvent::NoteOn { key: 60, vel: 90 }),
            TimedEvent::new(1.0, SamplerEvent::NoteOff { key: 60, vel: 64 }),
        ];
        let out = sampler.render(&half, 3.0).unwrap();
        assert!(rms(&out.channels[0], 2.0, 2.9) > 0.25);
    }

    #[test]
    fn the_pedal_recordings_fire_on_the_gate_the_file_states() {
        let fixture = Fixture::new("pedal-noise");
        fixture.sine("down.wav", 70.0, 0.3, 0.5);
        fixture.sine("up.wav", 90.0, 0.3, 0.5);
        let path = fixture.sfz(
            "<group> group=1 hikey=-1 lokey=-1 on_locc64=126 on_hicc64=127 off_by=2 volume=-20\n\
             <region> sample=samples/down.wav\n\
             <group> group=2 hikey=-1 lokey=-1 on_locc64=0 on_hicc64=1 volume=-19\n\
             <region> sample=samples/up.wav\n",
        );
        let mut sampler = Sampler::with_config(&path, config()).unwrap();

        // A pedal pressed halfway crosses no gate and makes no sound, which is
        // what the file says and not what a real tray does.
        let out = sampler
            .render(&[TimedEvent::new(0.1, SamplerEvent::Sustain(0.5))], 1.0)
            .unwrap();
        assert_eq!(peak(&out.channels[0]), 0.0);

        // Pressed to the stop and released to the stop, both recordings play,
        // at their own `volume`.
        let out = sampler
            .render(
                &[
                    TimedEvent::new(0.1, SamplerEvent::Sustain(1.0)),
                    TimedEvent::new(1.0, SamplerEvent::Sustain(0.0)),
                ],
                2.0,
            )
            .unwrap();
        let down = rms(&out.channels[0], 0.2, 0.5) * 2f64.sqrt();
        assert!((down - 0.3 * db_to_amp(-20.0)).abs() < 5e-4, "pedal down at {down}");
        // `off_by=2` stops the tray going down when the tray comes up, so the
        // 70 Hz recording is not still running under the 90 Hz one.
        let up = rms(&out.channels[0], 1.1, 1.4) * 2f64.sqrt();
        assert!((up - 0.3 * db_to_amp(-19.0)).abs() < 5e-4, "pedal up at {up}");
    }

    #[test]
    fn a_silent_press_makes_no_sound_and_still_lets_the_key_come_up() {
        let fixture = Fixture::new("silent-press");
        fixture.sine("note.wav", 220.0, 0.4, 2.0);
        fixture.sine("rel.wav", 660.0, 0.4, 0.5);
        let path = fixture.sfz(
            "<group> amp_veltrack=0\n\
             <region> sample=samples/note.wav key=60 lovel=1\n\
             <group> trigger=release amp_veltrack=0 pitch_keytrack=0\n\
             <region> sample=samples/rel.wav key=60\n",
        );
        let mut sampler = Sampler::with_config(&path, config()).unwrap();
        let out = sampler
            .render(
                &[
                    TimedEvent::new(0.0, SamplerEvent::KeyDown { key: 60 }),
                    TimedEvent::new(1.0, SamplerEvent::NoteOff { key: 60, vel: 64 }),
                ],
                2.0,
            )
            .unwrap();
        // Nothing while the key is down — no string was struck and no sampler
        // can produce the resonance a real one prepares.
        assert_eq!(peak(&out.channels[0][..RATE as usize]), 0.0);
        // The key still comes up, and the action still makes its noise.
        assert!(rms(&out.channels[0], 1.05, 1.4) > 0.2);
    }

    #[test]
    fn a_re_struck_key_layers_because_the_file_says_nothing_stops_it() {
        let fixture = Fixture::new("restrike");
        fixture.sine("note.wav", 220.0, 0.2, 4.0);
        let path = fixture.sfz("<region> sample=samples/note.wav key=60 amp_veltrack=0\n");
        let mut sampler = Sampler::with_config(&path, config()).unwrap();
        let out = sampler
            .render(
                &[
                    TimedEvent::new(0.0, SamplerEvent::NoteOn { key: 60, vel: 90 }),
                    TimedEvent::new(1.0, SamplerEvent::NoteOn { key: 60, vel: 90 }),
                ],
                3.0,
            )
            .unwrap();
        let one = rms(&out.channels[0], 0.2, 0.9);
        let two = rms(&out.channels[0], 1.2, 1.9);
        // The second strike lands on the same phase as the first (both start
        // at a zero of the same sine), so two voices are twice one.
        assert!((two / one - 2.0).abs() < 0.02, "{two} against {one}");
    }

    #[test]
    fn a_thirty_second_phrase_is_finite_and_free_of_clicks() {
        let fixture = Fixture::new("phrase");
        // Slow sines: any envelope discontinuity stands far above the material's
        // own slew, which is what makes the click test mean something.
        for (name, hz) in [("a.wav", 55.0), ("b.wav", 65.0), ("c.wav", 75.0)] {
            fixture.sine(name, hz, 0.25, 12.0);
        }
        fixture.sine("harm.wav", 300.0, 0.2, 2.0);
        let path = fixture.sfz(
            "<group> amp_veltrack=73 ampeg_release=1\n\
             <region> sample=samples/a.wav lokey=59 hikey=61 pitch_keycenter=60 lovel=1 hivel=64\n\
             <region> sample=samples/b.wav lokey=59 hikey=61 pitch_keycenter=60 lovel=65\n\
             <region> sample=samples/c.wav lokey=62 hikey=64 pitch_keycenter=63\n\
             <group> trigger=release amp_veltrack=94 volume=-4 rt_decay=6\n\
             <region> sample=samples/harm.wav lokey=59 hikey=64 pitch_keycenter=60\n",
        );
        let mut sampler = Sampler::with_config(&path, config()).unwrap();

        let mut events = Vec::new();
        let mut t = 0.0f64;
        let mut n = 0usize;
        while t < 29.0 {
            let key = [59u8, 60, 61, 62, 63, 64][n % 6];
            let vel = [30u8, 70, 110, 55][n % 4];
            events.extend(TimedEvent::note(t, key, vel, 0.45));
            if n % 8 == 0 {
                events.push(TimedEvent::new(t, SamplerEvent::Sustain(1.0)));
            }
            if n % 8 == 5 {
                events.push(TimedEvent::new(t, SamplerEvent::Sustain(0.0)));
            }
            t += 0.25;
            n += 1;
        }
        let out = sampler.render(&events, 30.0).unwrap();
        assert_eq!(out.frames(), 30 * RATE as usize);

        for channel in &out.channels {
            assert!(channel.iter().all(|x| x.is_finite()), "a sample is not finite");
            assert!(peak(channel) < 8.0, "runaway peak {}", peak(channel));
            // The material's own worst step is a 75 Hz sine at 0.25:
            // 0.25 * 2 pi * 75 / 48000 = 2.5e-3 per voice. Six keys of two
            // voices each cannot reach 0.05; a voice cut off mid-wave would be
            // a step of order 0.1.
            let step = channel
                .windows(2)
                .map(|w| (w[1] - w[0]).abs())
                .fold(0.0f32, f32::max);
            assert!(step < 0.05, "discontinuity of {step}");
        }
        // Something actually played, in both channels, all the way through.
        assert!(rms(&out.channels[0], 1.0, 29.0) > 0.05);
        assert!(rms(&out.channels[1], 1.0, 29.0) > 0.02);
    }

    #[test]
    fn a_render_is_reproducible_including_its_random_regions() {
        let fixture = Fixture::new("determinism");
        fixture.sine("d1.wav", 70.0, 0.3, 0.5);
        fixture.sine("d2.wav", 110.0, 0.3, 0.5);
        let path = fixture.sfz(
            "<group> hikey=-1 lokey=-1 on_locc64=126 on_hicc64=127\n\
             <region> sample=samples/d1.wav lorand=0 hirand=0.5\n\
             <region> sample=samples/d2.wav lorand=0.5 hirand=1\n",
        );
        let mut sampler = Sampler::with_config(&path, config()).unwrap();
        let events: Vec<TimedEvent> = (0..6)
            .flat_map(|i| {
                let t = i as f64 * 0.5;
                [
                    TimedEvent::new(t, SamplerEvent::Sustain(1.0)),
                    TimedEvent::new(t + 0.25, SamplerEvent::Sustain(0.0)),
                ]
            })
            .collect();
        let first = sampler.render(&events, 4.0).unwrap();
        let second = sampler.render(&events, 4.0).unwrap();
        assert_eq!(first.channels, second.channels);
        // Exactly one of the two recordings plays per press.
        assert!(peak(&first.channels[0]) > 0.0);
    }

    // ---------------------------------------------------------------------
    // The same event list the engine renders

    #[test]
    fn an_engine_event_list_drives_the_sampler_unchanged() {
        use piano_emulator::{Event, PedalEvent, RenderEvent};

        let engine_events = vec![
            RenderEvent::new(0.0, Event::Pedal(PedalEvent::Sustain(1.0))),
            RenderEvent::new(0.25, Event::NoteOn { key: 60, vel: 90 }),
            RenderEvent::new(0.5, Event::KeyDown { key: 62 }),
            RenderEvent::new(1.0, Event::NoteOff { key: 60, vel: 40 }),
            RenderEvent::new(1.5, Event::Pedal(PedalEvent::Sostenuto(true))),
            RenderEvent::new(1.6, Event::Pedal(PedalEvent::UnaCorda(true))),
            RenderEvent::new(2.0, Event::Pedal(PedalEvent::Sustain(0.0))),
            RenderEvent::new(2.5, Event::AllOff),
        ];
        let translated = engine_events::from_render_events(&engine_events);
        assert_eq!(translated.len(), engine_events.len());
        assert_eq!(
            translated[1],
            TimedEvent::new(0.25, SamplerEvent::NoteOn { key: 60, vel: 90 })
        );
        assert_eq!(translated[3].event, SamplerEvent::NoteOff { key: 60, vel: 40 });
        assert_eq!(translated[7].event, SamplerEvent::AllOff);

        let fixture = Fixture::new("engine-events");
        fixture.sine("note.wav", 220.0, 0.4, 4.0);
        let path = fixture.sfz(
            "<group> amp_veltrack=73 ampeg_release=1\n\
             <region> sample=samples/note.wav lokey=59 hikey=63 pitch_keycenter=60\n",
        );
        let mut sampler = Sampler::new(&path).unwrap();
        let out = sampler.render_engine_events(&engine_events, 4.0).unwrap();
        // The pedal held the note past its note-off, and the panic ended it.
        let held = rms(&out.channels[0], 1.2, 1.9);
        assert!(held > 0.15, "the pedal did not hold the note: {held}");
        assert!(rms(&out.channels[0], 3.0, 3.9) < 0.01 * held);
    }

    /// The other direction, which is the one every comparison uses: a phrase
    /// is written in the sampler's own event type and both instruments render
    /// it. The translation has to be total and lossless or the two renders are
    /// not of the same performance — which is the whole premise of
    /// `REALISM.md`, `COMPASS.md` and the melody gate.
    #[test]
    fn a_phrase_round_trips_through_the_engines_event_list_unchanged() {
        use piano_emulator::{Event, PedalEvent, RenderEvent};

        let phrase = vec![
            TimedEvent::new(0.0, SamplerEvent::Sustain(0.5)),
            TimedEvent::new(0.25, SamplerEvent::NoteOn { key: 60, vel: 90 }),
            TimedEvent::new(0.5, SamplerEvent::KeyDown { key: 62 }),
            TimedEvent::new(1.0, SamplerEvent::NoteOff { key: 60, vel: 40 }),
            TimedEvent::new(1.5, SamplerEvent::Sostenuto(true)),
            TimedEvent::new(1.625, SamplerEvent::UnaCorda(true)),
            TimedEvent::new(2.5, SamplerEvent::AllOff),
        ];
        let engine = engine_events::to_render_events(&phrase);
        assert_eq!(engine.len(), phrase.len());
        assert_eq!(
            engine[1],
            RenderEvent::new(0.25, Event::NoteOn { key: 60, vel: 90 })
        );
        // A continuous pedal survives as a continuous pedal: half-pedalling is
        // the one thing a boolean translation would quietly destroy.
        assert_eq!(engine[0].event, Event::Pedal(PedalEvent::Sustain(0.5)));
        assert_eq!(engine[6].event, Event::AllOff);
        // And back again, unchanged. Narrowing the time to `f32` is the one
        // lossy step in the round trip, so every time above is chosen to be
        // exact in binary — the equality is then about the *events*, which is
        // what has to be lossless.
        assert_eq!(engine_events::from_render_events(&engine), phrase);
    }

    #[test]
    fn the_default_grain_is_the_engines_own_block() {
        let fixture = Fixture::new("grain");
        fixture.sine("note.wav", 220.0, 0.4, 1.0);
        let path = fixture.sfz("<region> sample=samples/note.wav key=60 amp_veltrack=0\n");
        let mut sampler = Sampler::new(&path).unwrap();
        // An event 100 frames in lands at the start of its block, exactly where
        // `render_to_buffer` puts it.
        let out = sampler
            .render(
                &[TimedEvent::new(
                    100.0 / f64::from(RATE),
                    SamplerEvent::NoteOn { key: 60, vel: 127 },
                )],
                1.0,
            )
            .unwrap();
        assert_eq!(sampler.config().event_grain, ENGINE_BLOCK);
        assert_eq!(out.channels[0][0], 0.0);
        assert_ne!(out.channels[0][1], 0.0);
    }

    #[test]
    fn a_midi_file_renders_to_the_same_audio_as_the_events_it_holds() {
        // A note held across a pedal press, written as a standard MIDI file
        // and again as an event list, so the two paths into the player can be
        // compared on the audio rather than on the events.
        let mut track = Vec::new();
        push_event(0, &[0x90, 60, 90], &mut track); // note on at t = 0
        push_event(240, &[0xb0, 64, 127], &mut track); // CC 64 down at 0.25 s
        push_event(240, &[0x80, 60, 64], &mut track); // note off at 0.5 s
        push_event(480, &[0xb0, 64, 0], &mut track); // CC 64 up at 1.0 s
        let fixture = Fixture::new("midi");
        fixture.sine("note.wav", 220.0, 0.4, 4.0);
        let path = fixture.sfz(
            "<group> amp_veltrack=73 ampeg_release=0.5\n\
             <region> sample=samples/note.wav key=60\n",
        );
        let midi = fixture.dir.join("phrase.mid");
        std::fs::write(&midi, smf(480, &track)).unwrap();

        let mut sampler = Sampler::new(&path).unwrap();
        let from_file = sampler.render_midi(&midi, Some(3.0)).unwrap();
        let by_hand = sampler
            .render(
                &[
                    TimedEvent::new(0.0, SamplerEvent::NoteOn { key: 60, vel: 90 }),
                    TimedEvent::new(0.25, SamplerEvent::Sustain(1.0)),
                    TimedEvent::new(0.5, SamplerEvent::NoteOff { key: 60, vel: 64 }),
                    TimedEvent::new(1.0, SamplerEvent::Sustain(0.0)),
                ],
                3.0,
            )
            .unwrap();
        assert_eq!(from_file.channels, by_hand.channels);
        // And the pedal in the file did what a pedal does.
        assert!(rms(&from_file.channels[0], 0.6, 0.95) > 0.15);
        assert!(rms(&from_file.channels[0], 1.6, 2.5) < 1e-4);
    }

    // ---------------------------------------------------------------------
    // The real library, when it is on disk

    /// The Salamander library is 707 MiB and gitignored, so everything above
    /// runs on fixtures. When it *is* fetched, the one thing worth asserting is
    /// that the file the benchmark will actually read parses whole and asks for
    /// nothing this player does not implement.
    #[test]
    fn the_salamander_instrument_asks_for_nothing_this_player_ignores() {
        let path = Path::new("../data/salamander/SalamanderGrandPiano-V3+20200602.sfz");
        if !path.exists() {
            eprintln!("skipped: {} not fetched", path.display());
            return;
        }
        let instrument = Instrument::from_sfz(path).unwrap();
        assert!(
            instrument.ignored_opcodes().is_empty(),
            "unimplemented opcodes: {:?}",
            instrument.ignored_opcodes()
        );
        assert_eq!(instrument.regions().len(), 641);
        // 30 sampled keys x 16 layers of struck notes, and the rest is the
        // instrument not being struck.
        let attacks = instrument
            .regions()
            .iter()
            .filter(|r| r.trigger == Trigger::Attack && r.cc64_gate().is_none())
            .count();
        assert_eq!(attacks, 480);
        // Every key of the compass is playable, at every velocity.
        for key in 21..=108u8 {
            for vel in [1u8, 45, 90, 127] {
                let region = instrument
                    .regions()
                    .iter()
                    .find(|r| r.trigger == Trigger::Attack && r.matches(key, vel, 0.5));
                assert!(region.is_some(), "no region for key {key} at velocity {vel}");
            }
        }
    }

    /// A MIDI variable-length quantity, and one event with its delta time —
    /// the two pieces a standard MIDI file is made of.
    fn push_event(delta: u32, message: &[u8], out: &mut Vec<u8>) {
        let mut buffer = [0u8; 4];
        let mut count = 0;
        let mut value = delta;
        loop {
            buffer[count] = (value & 0x7f) as u8;
            count += 1;
            value >>= 7;
            if value == 0 {
                break;
            }
        }
        for i in (0..count).rev() {
            out.push(buffer[i] | if i > 0 { 0x80 } else { 0 });
        }
        out.extend_from_slice(message);
    }

    /// A single-track format-0 file with `division` ticks per quarter note; at
    /// the default 120 bpm, 480 ticks is half a second.
    fn smf(division: u16, track: &[u8]) -> Vec<u8> {
        let mut file = b"MThd".to_vec();
        file.extend_from_slice(&6u32.to_be_bytes());
        file.extend_from_slice(&0u16.to_be_bytes());
        file.extend_from_slice(&1u16.to_be_bytes());
        file.extend_from_slice(&division.to_be_bytes());
        let mut body = track.to_vec();
        push_event(0, &[0xff, 0x2f, 0x00], &mut body);
        file.extend_from_slice(b"MTrk");
        file.extend_from_slice(&(body.len() as u32).to_be_bytes());
        file.extend_from_slice(&body);
        file
    }

    /// Amplitude of one sinusoid inside a window, by projecting the signal
    /// onto it. A Hann window puts the leakage from a component hundreds of
    /// bins away below anything asserted here, so two voices at different
    /// frequencies can be measured while both are sounding.
    fn tone_level(x: &[f32], hz: f64, from_s: f64, to_s: f64) -> f64 {
        let from = (from_s * f64::from(RATE)) as usize;
        let to = ((to_s * f64::from(RATE)) as usize).min(x.len());
        if to <= from + 16 {
            return 0.0;
        }
        let n = to - from;
        let (mut re, mut im, mut weight) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..n {
            let w = 0.5 - 0.5 * (TAU * i as f64 / n as f64).cos();
            let phase = TAU * hz * (from + i) as f64 / f64::from(RATE);
            let sample = w * f64::from(x[from + i]);
            re += sample * phase.cos();
            im -= sample * phase.sin();
            weight += w;
        }
        2.0 * re.hypot(im) / weight
    }

    /// The one test that plays the recordings themselves: a pedalled phrase
    /// over a sampled key, its transposed neighbour and a key from another
    /// register. Skipped, loudly, when the library has not been fetched.
    #[test]
    fn the_salamander_library_renders_a_pedalled_phrase() {
        let sfz = Path::new("../data/salamander/SalamanderGrandPiano-V3+20200602.sfz");
        if !sfz.exists() {
            eprintln!("skipped: {} not fetched", sfz.display());
            return;
        }
        let mut sampler = Sampler::new(sfz).unwrap();
        let mut events = vec![TimedEvent::new(0.0, SamplerEvent::Sustain(1.0))];
        // C4 is recorded; C#4 is C4 transposed up a semitone; E3 comes from
        // another recording entirely.
        for (i, key) in [60u8, 61, 52].into_iter().enumerate() {
            events.extend(TimedEvent::note(0.5 + i as f64 * 0.6, key, 90, 0.4));
        }
        events.push(TimedEvent::new(2.6, SamplerEvent::Sustain(0.0)));
        let out = sampler.render(&events, 6.0).unwrap();

        for channel in &out.channels {
            assert!(channel.iter().all(|x| x.is_finite()));
            assert!(peak(channel) < 1.0, "clipping at {}", peak(channel));
        }
        // The pedal held all three notes to 2.6 s and the release ends them.
        assert!(rms(&out.channels[0], 2.0, 2.5) > 0.02);
        assert!(rms(&out.channels[0], 4.5, 5.9) < 1e-3);
        // Three struck notes, each of them three layers deep in release
        // recordings (`harmL`, `harmV3`, `rel`), from three distinct keys.
        assert!(sampler.cached_buffers() >= 12, "{}", sampler.cached_buffers());
    }

    /// Counts zero crossings per second in a window, which reads a sine's
    /// frequency without a transform.
    fn zero_crossings(x: &[f32]) -> usize {
        let crossings = x
            .windows(2)
            .filter(|w| (w[0] < 0.0) != (w[1] < 0.0))
            .count();
        crossings * RATE as usize / (2 * x.len())
    }
}
