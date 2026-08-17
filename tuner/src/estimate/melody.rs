//! The listener's test, made permanent: is any one note of a melody textured
//! unlike the rest of the line?
//!
//! This module exists because of the complaint that opened `DECISIONS.md` 284.
//! A listener played the Ode to Joy excerpt and heard **one note** — C4 — as
//! textured differently from the notes either side of it. The diagnosis was the
//! opposite of the complaint: C4 was the only key in the phrase carrying a
//! *measured* `notes.partial_gains` row and *measured* `notes.false_beat`
//! splits, and D4/E4/F4/G4 carried none at all. The seam was the defect and the
//! fitted note was the correct one.
//!
//! Item 284 closed that seam and item 288 measured it — but every statistic it
//! was measured with is a **compass** statistic: 88 keys struck alone at one
//! velocity, scored against their own neighbours. None of them is the thing the
//! listener actually did, which was to play a tune and hear one note of it stick
//! out. So the milestone's own trigger had no standing gate, and this is it.
//!
//! # What is rendered
//!
//! [`soprano`] is the melody line of [`realism::excerpt`] **alone**: the same
//! thirty notes at the same onsets and the same velocity, with the harmony and
//! the sustain pedal taken away. Both are built from
//! [`realism::ODE_MELODY`], so the gate cannot end up testing a line the
//! scoreboard does not play. The pedal goes because a pedalled line rings every
//! note into the next one, and a per-note measurement then reads the phrase
//! rather than the note; the harmony goes for the same reason and one more —
//! keys 36-55 would put their own texture into the window.
//!
//! # The numbers, per note
//!
//! Measured identically on the engine's render and on the recordings', over the
//! same window of the same note, which is what lets the two lines be quoted
//! against each other.
//!
//! | metric | definition | what the milestone put into it |
//! |---|---|---|
//! | `roughness` | [`Series::irregularity`] over the note's own partials, dB | the drawn `notes.partial_gains` row — a jagged series is a *table* by construction, `COMPASS.md`'s own argument |
//! | `wobble` | median over partials of the RMS of that partial's dB envelope about its own straight-line decay | the drawn `notes.false_beat` splits — a split makes one partial's envelope oscillate where nothing else in the engine can |
//! | `hf` | 2-6 kHz share of the note's total power, dB | brilliance, at absolute frequency (`DECISIONS.md` 292), because texture that changed a note's colour would be heard as one note being brighter |
//! | `strike` | attack tonality of the first 30 ms, dB — how loud the mechanism is against the note (`DECISIONS.md` 340-341) | a **balance**, scored against the recording rather than against the line's own trend |
//! | `channel` | [`pair_over_mono_db`]: `10 log10((E_L + E_R) / 2 E_M)` over the note's window | **the only column here that is not a mono fold-down** (`DECISIONS.md` 392-394) — what the two loudspeakers put in the room against what this note's own mono sum says they do |
//!
//! `channel` exists because the other four cannot exist without it. All four
//! are computed on `(L+R)/2`, as is every other board in this repository, and
//! so is every number the factory closes on — which means a stereo stage can
//! make one note of a melody **four decibels louder in the room than its
//! neighbours** and leave all of them unmoved. That is not hypothetical: the
//! virtual microphone pair's mode-controlled lobe read +6.42 dB at C4 against
//! +2.41 at F4, a listener picked C4 out of this very line three milestones
//! running, and 696 tests were green throughout.
//!
//! `wobble` is not [`motion::Motion::beat_depth_db`], which `COMPASS.md` uses,
//! and it cannot be: `motion.rs` measures over 0.3-3.0 s and a melody gives one
//! note 0.4 s before the next strike. It is the same physical quantity read in
//! the window a *tune* leaves, which is the window the complaint was made in.
//!
//! # Two windows
//!
//! Since `DECISIONS.md` 330 each of the three is measured twice: over
//! [`NOTE_WINDOW_S`], the prompt sound, and over [`TAIL_WINDOW_S`], what is left
//! of the note afterwards. The tail columns exist because the first regression
//! this gate failed to catch was a *decay* one — C4's `partial_sigma_scale` row
//! against drawn neighbours, and specifically its cells **under 2 kHz**, which
//! `estimate::tail`'s correction curve holds at one and which therefore existed
//! at the recorded keys and at none of the others until `DECISIONS.md` 335 —
//! and a window that closes at 0.40 s cannot see a decay. They are measured on [`slow_line`] — the melody's own pitches at its
//! own velocity, played slowly and legato — because at the melody's tempo the
//! late window of a note contains three later strikes, two of which on this
//! tune are that note's own third and fifth harmonics. [`TAIL_BEAT_S`] carries
//! the measurement that settles it.
//!
//! # The score, and which reference notes are allowed into it
//!
//! Five distinct pitches (C4, D4, E4, F4, G4). What is gated is the listener's
//! own act: they played the melody on the engine and heard one note that did
//! not belong. So the first gate is the largest departure of any note from the
//! **engine line's own** register trend. The engine renders all five from its
//! own tables, so all five are its own work and all five are scored.
//!
//! The bar it is held against is not. `DECISIONS.md` 328: the library records
//! one key every minor third, and in this line **only C4 is a recording**. D4
//! and E4 are both the D#4 take resampled a semitone down and up; F4 and G4 are
//! both the F#4 take. The reference line is therefore four clones and one note,
//! and how far its notes scatter about their own trend is a fact about a
//! resampler. Bars measured off it are not the piano's — they are far too
//! small, because two transpositions of one recording agree with each other far
//! better than two notes of a piano do.
//!
//! So every bar here is rebuilt from a **recorded-key population**: [`ladder`]
//! plays the recorded keys of the melody's own register as a line, at the same
//! tempo, the same velocity and through the same window, and the bar is
//!
//! * how far *those* notes go from *their* trend — a real instrument's
//!   note-to-note scatter in this register — held against
//! * the **per-take scatter**, `|reference − neighbouring velocity layer|` at
//!   the same recorded key, which is the same device `REALISM.md`'s noise floor
//!   is, generalised from a phrase distance to a per-note metric: below it, two
//!   recordings of one key disagree by that much and nothing can be concluded.
//!
//! The larger of the two, times [`ALLOWANCE`].
//!
//! Both terms are read at the **same order statistic** the gate's own number
//! is — the median of the largest of five draws, not the largest of nine — so
//! the comparison is between instruments and not between sample sizes. See
//! [`worst_of_n`], which on the column that matters takes the bar from 8.44 dB
//! to 5.32 and is the difference between passing C4 and failing it.
//!
//! Reported beside it, and **not** gated, is the **seam**: `error = engine −
//! reference` and its departure from the register's median error. Item 297
//! could not set a bar for it because the melody holds exactly one key whose
//! reference is that key; on the recorded-key ladder every note is that key, so
//! half of that objection is answered, and the other half is not — the engine's
//! absolute distance from the piano moves by several dB across a register for
//! reasons no per-note floor covers. It carries its own floor
//! ([`Column::seam_floor`]: the same statistic with the neighbouring velocity
//! layer standing in for the engine) so a reader can see how far above it the
//! number is. What the line itself contributes is the one number it honestly
//! can: C4's own error, reported as [`Column::line_error`]. Its other four
//! notes are marked `transposed — unscored`, rendered and listened to and never
//! scored.

use crate::audio::Audio;
use crate::estimate::brilliance::{band, FULL, HF1};
use crate::realism::{self, Phrase, RecordedKeys};
use crate::sampler::{SamplerEvent, TimedEvent};
use crate::series::Series;
use crate::stft::{Stft, StftConfig};

/// The window one note is measured over, seconds from its own onset.
///
/// Starts past the hammer's noise and the attack transient, and ends before the
/// key is released — the melody's quarter notes are held 0.45 s, so 0.40 s is
/// entirely inside the sounding note and none of it is a damper falling.
pub const NOTE_WINDOW_S: (f64, f64) = (0.03, 0.40);

/// The **late** window, seconds from the same onset: what is left of the note
/// after the prompt sound has gone.
///
/// `DECISIONS.md` 330. Everything above measures the first 0.40 s of a note,
/// which is where a `partial_gains` row lives and where a hammer's colour is
/// decided. It is not where a *decay* lives. The C4 the listener called "quite
/// off the rest" of the M8 melody carries a fitted `partial_sigma_scale` row
/// 41 partials deep while D4/E4/F4/G4 carry drawn ones (`notes.synthesized_decay`
/// names all four), and a row that only changes how fast a partial dies changes
/// nothing a 0.03-0.40 s window can see. The gate windowed the attack and the
/// seam lived in the tail, which is exactly how a regression walks past a gate
/// nobody widened.
pub const TAIL_WINDOW_S: (f64, f64) = (0.5, 2.0);

/// Notes shorter than this are not measured at all.
///
/// The line has two half-beat passing notes (0.20 s of held key). Measuring
/// them over a shorter window than everything else would put a different
/// frequency resolution into the same column; leaving them out costs nothing,
/// because both of their pitches sound five more times in the line at full
/// length.
pub const MIN_NOTE_S: f64 = 0.42;

/// Partials `wobble` is taken over.
///
/// The splits `notes.false_beat` writes live on k1-k8 (`DECISIONS.md` 285), and
/// by k6 a melody note's partial is far enough down that its envelope is mostly
/// the note's neighbours; six is where both of those stop being true.
pub const WOBBLE_PARTIALS: usize = 6;

/// Milliseconds trimmed from each end of a partial's envelope before its
/// residual is taken: the heterodyne's boxcar is still filling at the start and
/// running out at the end, and neither is the note moving.
pub const WOBBLE_TRIM_MS: usize = 20;

/// The band `hf` reads, and the band the whole note's power is measured over.
pub const HF_BAND: (f64, f64) = HF1;

/// FFT the `hf` band powers are summed over. 2048 at 48 kHz is 23.4 Hz per bin
/// and 43 ms per frame — fine enough to place a band edge, short enough that
/// several frames fit inside [`NOTE_WINDOW_S`].
pub const HF_WINDOW: usize = 2048;

/// Where a note's own strike is looked for, seconds either side of the onset
/// the phrase gives it.
///
/// Not a nicety: the sampler plays each recording from its own start, and every
/// recording begins with however much silence there was between the engineer's
/// trigger and the hammer. That offset is a property of one *sample file*, so in
/// a phrase it differs from note to note — and a per-note window that ignored it
/// would measure one key 40 ms into its decay and the next one 10 ms before its
/// attack. The engine's own offset is zero and it is searched anyway, because a
/// gate whose two sides are windowed by different rules is not comparing them.
pub const ONSET_SEARCH_S: (f64, f64) = (-0.03, 0.12);

/// Seconds of the soprano line. The last note is three beats and starts at beat
/// 30, so the line is over at 16.7 s.
pub const SOPRANO_S: f64 = 17.5;

/// Seconds between strikes on the line the tail is read off, and how long each
/// of its notes is held.
///
/// The tail cannot be read at the melody's own tempo and this is not a
/// preference, it is arithmetic. At 0.5 s a beat, the window 0.5-2.0 s after a
/// note's strike contains three later strikes; two of them, on this tune, are
/// the note's own third and fifth harmonics (G4 is C4's third partial to within
/// a hertz, E4 sits under its fifth), so a metric that reads C4's partial
/// levels in that window is reading G4 and E4. Measured: on the line played at
/// tempo with the pedal held, the five pitches' tail `hf` spans 2.7 dB where
/// the same five keys struck alone span 15, because the number is the *chord*
/// and not the note.
///
/// So the tail is read off the same tune played slowly and legato — the line's
/// own pitches, at the line's own velocity, in the order the tune introduces
/// them, each held long enough to have a tail and struck after the one before
/// it has been let go. No pedal: a pedal would put the previous notes back.
pub const TAIL_BEAT_S: f64 = 2.5;
/// How long a note of the slow line is held: past [`TAIL_WINDOW_S`], so no
/// damper is inside the window.
pub const TAIL_HOLD_S: f64 = 2.2;
/// How many times the slow line goes through the melody's pitches. Two, so
/// [`per_key`]'s median has something to be a median of.
pub const TAIL_PASSES: usize = 2;

/// How far either side of the line's own span the recorded-key population that
/// sets the bars is drawn from, in semitones.
///
/// The line spans C4-G4, seven semitones, and holds exactly **one** recorded
/// key. A bar has to be measured off more than one note, so it is measured off
/// the register: nine semitones each way reaches D#3 to D#5, which on a library
/// that samples every minor third is nine takes. Wider would start comparing
/// the melody's register against the bass's, and every one of these three
/// metrics has a register trend.
pub const LADDER_REACH: u8 = 9;

/// How much of the measured bar a note is allowed to stand out by.
///
/// One, plus a quarter, because the bar is itself a measurement off five notes
/// and a gate that tripped whenever a measurement landed a hair over its own
/// noise floor would be a coin flip rather than a gate. It is not a tolerance
/// on the defect and nothing here is near it: on the shipped preset the six
/// columns read 0.64, 0.32, 0.51, 0.26, 0.80 and **0.71** of their bars, and on
/// the instrument of `DECISIONS.md` 331 — the same preset with the drawn rows'
/// sub-2 kHz cells put back to one — the tail `hf` column alone read **1.02**
/// (items 298, 336).
pub const ALLOWANCE: f64 = 1.25;

/// Names in the order [`NoteTexture::values`] returns them.
pub const METRICS: [&str; 5] = ["roughness", "wobble", "hf", "strike", "channel"];

/// Which of [`METRICS`] is a **balance** rather than an evenness, and is
/// therefore gated on a different question.
///
/// `DECISIONS.md` 341, 394. The first three ask *does one note of the line
/// stand out from the rest*, which is the listener's complaint of item 284 and
/// needs no recording of the note to answer. `strike` asks *is the mechanism as
/// loud against the note as the piano's is* and `channel` asks *do the two
/// loudspeakers play this note as the piano's two channels do* — both are
/// comparisons with a recording and only mean anything at a key the library
/// recorded, so both are scored on the recorded ladder and not on the line, and
/// their bar is built out of two takes of one recorded key rather than out of
/// the scatter of the register.
pub const METRIC_IS_BALANCE: [bool; 5] = [false, false, false, true, true];

/// What `DECISIONS.md` 340's refit moved `[noise.strike]` by, kept here so that
/// the instrument the `strike` column fails on can be built from the shipped
/// preset without a hand-edited file — `melody --before-noise` and
/// `tests/melody.rs`'s falsification are the same two numbers.
///
/// Added to every anchor: the level the event carried before the refit is the
/// shipped one plus this.
pub const STRIKE_REFIT_LEVEL_DB: f32 = 7.292_264;
/// And the velocity law it carried before it.
pub const STRIKE_VELOCITY_DB_BEFORE: f32 = 24.401_855;

/// The index of `strike` in [`METRICS`].
pub const STRIKE_METRIC: usize = 3;

/// The index of `channel` in [`METRICS`].
pub const CHANNEL_METRIC: usize = 4;

// ---------------------------------------------------------------------------
// The line
// ---------------------------------------------------------------------------

/// One note of the line: which key, when it is struck, and how long it is held.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LineNote {
    pub key: u8,
    pub onset_s: f64,
    pub held_s: f64,
}

impl LineNote {
    /// Whether this note is long enough to measure ([`MIN_NOTE_S`]).
    pub fn measurable(&self) -> bool {
        self.held_s >= MIN_NOTE_S
    }
}

/// The melody line of [`realism::excerpt`], on its own: no harmony, no pedal.
pub fn soprano() -> Phrase {
    Phrase {
        name: "ode_soprano",
        description: "the Ode to Joy melody line of `excerpt`, alone and unpedalled",
        duration_s: SOPRANO_S,
        events: line_events(&line_notes()),
    }
}

/// The line the tail is read off: the melody's own pitches, at the melody's own
/// velocity, in the order the tune introduces them, played slowly and legato.
///
/// [`soprano`] is the tune at tempo and it is the right material for everything
/// measured over [`NOTE_WINDOW_S`]. It is also why the gate could not see the
/// C4 the listener heard, and it fails in two ways at once: with the dampers
/// working the melody's 0.45 s notes are *over* before [`TAIL_WINDOW_S`] opens,
/// and with the pedal held instead they are not over but they are not alone
/// either. See [`TAIL_BEAT_S`] for the measurement that settles it.
pub fn slow_line() -> Phrase {
    let notes = slow_line_notes();
    Phrase {
        name: "ode_pitches_slow",
        description: "the melody's own pitches at its own velocity, slowly and legato",
        duration_s: line_duration(&notes),
        events: line_events(&notes),
    }
}

/// The notes [`slow_line`] plays: the line's distinct pitches, ordered by where
/// the tune first sounds them, [`TAIL_PASSES`] times through.
pub fn slow_line_notes() -> Vec<LineNote> {
    let mut order: Vec<u8> = Vec::new();
    for note in line_notes() {
        if note.measurable() && !order.contains(&note.key) {
            order.push(note.key);
        }
    }
    spaced(&order.repeat(TAIL_PASSES), TAIL_BEAT_S, TAIL_HOLD_S)
}

/// The recorded keys of the melody's register, played as a line in the same
/// material the window they set the bar for is measured in.
///
/// This is the population every bar in this module is now measured off
/// (`DECISIONS.md` 328-329). The line's own five pitches cannot set one: four
/// of them are the *same two recordings* resampled — D4 and E4 are both the
/// D#4 take, F4 and G4 are both the F#4 take — so the reference line is four
/// clones and one note, and its scatter about its own trend is a property of a
/// resampler rather than of a piano. Nine takes of nine different keys, played
/// in the same music and measured through the same window, are what a real
/// instrument's note-to-note scatter looks like.
///
/// At the melody's tempo it goes up and back down, so that every key is
/// measured in two different contexts — the same reason [`per_key`] takes a
/// median over a pitch's occurrences. Slowly, every note is alone and there is
/// no context to average over, so it goes up once.
pub fn ladder(keys: &[u8], window: Window) -> Phrase {
    let notes = ladder_notes(keys, window);
    Phrase {
        name: match window {
            Window::Head => "recorded_ladder",
            Window::Tail => "recorded_ladder_slow",
        },
        description: "the recorded keys of the melody's register, played as a line",
        duration_s: line_duration(&notes),
        events: line_events(&notes),
    }
}

/// The notes [`ladder`] plays, in time order.
pub fn ladder_notes(keys: &[u8], window: Window) -> Vec<LineNote> {
    let mut ascending: Vec<u8> = keys.to_vec();
    ascending.dedup();
    match window {
        Window::Head => {
            let order: Vec<u8> = ascending
                .iter()
                .copied()
                .chain(ascending.iter().rev().skip(1).copied())
                .collect();
            spaced(&order, realism::ODE_BEAT, realism::ODE_BEAT - 0.05)
        }
        Window::Tail => spaced(&ascending, TAIL_BEAT_S, TAIL_HOLD_S),
    }
}

/// The line one column of the gate is measured on.
pub fn line_for(window: Window) -> Phrase {
    match window {
        Window::Head => soprano(),
        Window::Tail => slow_line(),
    }
}

/// The notes of [`line_for`], which is what [`measure_line`] windows.
pub fn line_notes_for(window: Window) -> Vec<LineNote> {
    match window {
        Window::Head => line_notes(),
        Window::Tail => slow_line_notes(),
    }
}

fn spaced(keys: &[u8], beat_s: f64, held_s: f64) -> Vec<LineNote> {
    keys.iter()
        .enumerate()
        .map(|(i, &key)| LineNote {
            key,
            onset_s: realism::ODE_START + i as f64 * beat_s,
            held_s,
        })
        .collect()
}

fn line_duration(notes: &[LineNote]) -> f64 {
    notes
        .last()
        .map_or(0.0, |n| n.onset_s + n.held_s.max(TAIL_WINDOW_S.1))
        + 0.5
}

/// The recorded keys [`ladder`] is played on, for a library and a line.
pub fn ladder_keys(recorded: &RecordedKeys, line: &[u8]) -> Vec<u8> {
    let lo = line.iter().copied().min().unwrap_or(60).saturating_sub(LADDER_REACH);
    let hi = line.iter().copied().max().unwrap_or(60).saturating_add(LADDER_REACH);
    recorded.in_range(lo, hi)
}

/// Note-ons and note-offs, and nothing else: no pedal on any line here.
///
/// The tail line does without one for the same reason `soprano` does. A pedal
/// would put every note that has already sounded back into the window the next
/// note's tail is read in, which is the contamination [`TAIL_BEAT_S`] exists to
/// avoid; holding the key instead keeps the damper off this note's own strings
/// and nothing else's.
fn line_events(notes: &[LineNote]) -> Vec<TimedEvent> {
    let mut events = Vec::with_capacity(2 * notes.len());
    for note in notes {
        events.push(TimedEvent::new(
            note.onset_s,
            SamplerEvent::NoteOn {
                key: note.key,
                vel: realism::ODE_MELODY_VEL,
            },
        ));
        events.push(TimedEvent::new(
            note.onset_s + note.held_s,
            SamplerEvent::NoteOff {
                key: note.key,
                vel: 64,
            },
        ));
    }
    events
}

/// The line's notes in time order, with the same onsets and lengths
/// [`realism::excerpt`] gives them.
pub fn line_notes() -> Vec<LineNote> {
    realism::ODE_MELODY
        .iter()
        .map(|&(at, key, len)| LineNote {
            key,
            onset_s: realism::ODE_START + at * realism::ODE_BEAT,
            held_s: (len * realism::ODE_BEAT - 0.05).max(0.08),
        })
        .collect()
}

/// The distinct pitches of the line, ascending.
pub fn line_keys() -> Vec<u8> {
    let mut keys: Vec<u8> = line_notes()
        .iter()
        .filter(|n| n.measurable())
        .map(|n| n.key)
        .collect();
    keys.sort_unstable();
    keys.dedup();
    keys
}

// ---------------------------------------------------------------------------
// The numbers, per note
// ---------------------------------------------------------------------------

/// What one note of the line sounds like, in the three ways this gate reads.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NoteTexture {
    pub key: u8,
    pub onset_s: f64,
    /// Mean absolute step between adjacent partial levels, dB.
    pub roughness_db: f64,
    /// Median over partials of the RMS deviation of that partial's dB envelope
    /// from its own straight line, dB.
    pub wobble_db: f64,
    /// 2-6 kHz share of the note's total power, dB.
    pub hf_db: f64,
    /// The note's **noise-to-tone ratio**: the attack tonality of the first
    /// [`realism::ATTACK_WINDOW_S`] from its own strike
    /// ([`crate::estimate::attack::noise_to_tone_db`]), in dB. Large is a line
    /// spectrum, zero is a continuum, so a hammer that is too loud against its
    /// own note reads *low*.
    ///
    /// Read from the strike itself and not from the window the other three use:
    /// [`NOTE_WINDOW_S`] starts at 0.03 s precisely because that is "past the
    /// hammer's noise", so the one span the three texture metrics deliberately
    /// exclude is the only span this one is about. It is therefore the same
    /// number in both windows and only the head column is scored — a tail has
    /// no strike in it.
    pub strike_db: f64,
    /// **What the two loudspeakers do with this note, against what its own mono
    /// sum does**: `10 log10((E_L + E_R) / 2 E_M)` over the note's window,
    /// where `E_M` is the energy of `(L+R)/2`.
    ///
    /// `DECISIONS.md` 392-394. Zero for any signal whose two channels are its
    /// mono sum scaled — a pan-potted note reads 0.00 at every key — and the
    /// recording's own is not zero, because two capsules over a real
    /// soundboard hear a note at two levels. It is the **loudness** column the
    /// other four do not have: `roughness`, `wobble` and `hf` are shapes and
    /// `strike` is a ratio, all four are computed on the mono fold-down, and
    /// "that note stands out" is a statement about level in the room. On the
    /// instrument this column was written for it read **+6.42 dB at C4 against
    /// +2.41 at F4** where the recording's own five pitches sit inside a
    /// decibel of each other.
    pub channel_db: f64,
}

impl NoteTexture {
    pub fn values(&self) -> [f64; 5] {
        [
            self.roughness_db,
            self.wobble_db,
            self.hf_db,
            self.strike_db,
            self.channel_db,
        ]
    }
}

/// Which part of a note a column reads.
///
/// The same metrics, the same code, two windows. [`Window::Head`] is the
/// prompt sound — where a `partial_gains` row and a hammer's colour live.
/// [`Window::Tail`] is what is left of the note afterwards, which is where a
/// `partial_sigma_scale` row lives and where the C4 of `DECISIONS.md` 330 was
/// hiding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Window {
    Head,
    Tail,
}

impl Window {
    pub fn span_s(self) -> (f64, f64) {
        match self {
            Window::Head => NOTE_WINDOW_S,
            Window::Tail => TAIL_WINDOW_S,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Window::Head => "head",
            Window::Tail => "tail",
        }
    }

}

/// Measures every measurable note of the line off one rendered signal.
///
/// `partial_hz` gives each note's partial frequencies — the same table for both
/// signals, taken from the preset, exactly as `compass` does it: the
/// recording is measured at the frequencies the model says the note has, so a
/// difference between the two columns is never a difference in where they
/// looked.
pub fn measure_line(
    audio: &Audio,
    sample_rate: f64,
    notes: &[LineNote],
    partial_hz: &dyn Fn(u8) -> Vec<f64>,
    window: Window,
) -> Vec<NoteTexture> {
    let mono = audio.mono();
    let stereo = (audio.channel_count() >= 2)
        .then(|| (&audio.channels[0], &audio.channels[1]));
    let (from_s, to_s) = window.span_s();
    let stft = Stft::new(StftConfig::new(HF_WINDOW, HF_WINDOW / 4, HF_WINDOW).expect("valid"))
        .expect("valid");
    notes
        .iter()
        .filter(|n| n.measurable())
        .map(|note| {
            let strike = note_onset(&mono, sample_rate, note.onset_s);
            let lo = ((strike + from_s) * sample_rate) as usize;
            let hi = (((strike + to_s) * sample_rate) as usize).min(mono.len());
            let (lo, hi) = (lo.min(hi), hi);
            let slice = &mono[lo..hi];
            let hz = partial_hz(note.key);
            let series = Series::measure(slice, &hz, sample_rate);
            NoteTexture {
                key: note.key,
                onset_s: strike,
                roughness_db: series.irregularity(),
                wobble_db: wobble(slice, sample_rate, &hz, &series),
                hf_db: hf_share_db(&stft, slice, sample_rate),
                strike_db: crate::estimate::attack::noise_to_tone_db(&mono, strike, sample_rate),
                channel_db: match stereo {
                    // A signal with one channel *is* its own mono sum, so the
                    // column is zero rather than absent: it is the same
                    // statement a pan-pot makes.
                    None => 0.0,
                    Some((left, right)) => pair_over_mono_db(
                        &left[lo.min(left.len())..hi.min(left.len())],
                        &right[lo.min(right.len())..hi.min(right.len())],
                        slice,
                    ),
                },
            }
        })
        .collect()
}

/// `10 log10((E_L + E_R) / 2 E_M)` over one window — the column
/// [`NoteTexture::channel_db`] carries.
///
/// The factor of two is what makes a mono-equivalent pair read **0.00**:
/// `E_L = E_R = E_M` there, so the ratio is one. Above zero the two
/// loudspeakers between them are radiating more than the mono fold-down says
/// they are, which is what a side signal is; and every mono board in this
/// repository is a function of that fold-down, so this is precisely the part
/// none of them can see.
pub fn pair_over_mono_db(left: &[f32], right: &[f32], mono: &[f32]) -> f64 {
    let energy = |s: &[f32]| s.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>();
    let (pair, mid) = (energy(left) + energy(right), energy(mono));
    if mid.is_nan() || pair.is_nan() || mid <= 0.0 || pair <= 0.0 {
        return f64::NAN;
    }
    10.0 * (pair / (2.0 * mid)).log10()
}

/// Where the strike actually is, near where the phrase says it is.
///
/// The largest rise in a 1 ms RMS envelope over [`ONSET_SEARCH_S`]. A rise
/// rather than a level, because a melody note is struck **into the tail of the
/// note before it** and any threshold on level would fire on that tail; a piano
/// strike is the one thing in the window that goes up.
/// One primitive, two boards: [`realism::strike_near`] is the same search, and
/// `DECISIONS.md` 338 gives the phrase board's `attack` column the same
/// treatment this gate has had since it was written.
pub fn note_onset(mono: &[f32], sample_rate: f64, nominal_s: f64) -> f64 {
    realism::strike_near(
        mono,
        sample_rate,
        nominal_s,
        -ONSET_SEARCH_S.0,
        ONSET_SEARCH_S.1,
    )
}

/// How much a note's partials move while it sounds, in dB.
///
/// Each present partial's envelope comes out of the same heterodyne
/// `estimate::brilliance` uses, the straight line through it in dB is its own
/// decay, and what is left is the note *moving*. The median over partials,
/// because one partial sitting on a neighbour's sidelobe is not the note.
pub fn wobble(window: &[f32], sample_rate: f64, partial_hz: &[f64], series: &Series) -> f64 {
    let mut residuals: Vec<f64> = Vec::new();
    for (k, _) in series.sounding() {
        if k > WOBBLE_PARTIALS || k > partial_hz.len() {
            continue;
        }
        let env = crate::estimate::brilliance::narrowband_db(window, partial_hz[k - 1], sample_rate);
        if env.len() <= 2 * WOBBLE_TRIM_MS + 40 {
            continue;
        }
        let pts = &env[WOBBLE_TRIM_MS..env.len() - WOBBLE_TRIM_MS];
        residuals.push(line_rms(pts));
    }
    median(&mut residuals)
}

/// RMS of `y` about its own least-squares straight line in index.
fn line_rms(y: &[f64]) -> f64 {
    let n = y.len() as f64;
    let mx = (n - 1.0) / 2.0;
    let my = y.iter().sum::<f64>() / n;
    let (mut num, mut den) = (0.0, 0.0);
    for (i, &v) in y.iter().enumerate() {
        let dx = i as f64 - mx;
        num += dx * (v - my);
        den += dx * dx;
    }
    let slope = if den > 0.0 { num / den } else { 0.0 };
    let sum: f64 = y
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let r = v - (my + slope * (i as f64 - mx));
            r * r
        })
        .sum();
    (sum / n).sqrt()
}

/// The note's 2-6 kHz power as a ratio of its whole power, in dB.
fn hf_share_db(stft: &Stft, window: &[f32], sample_rate: f64) -> f64 {
    let mut power = vec![0.0f64; stft.bins()];
    stft.for_each_frame(window, sample_rate, |_, magnitude| {
        for (s, &m) in power.iter_mut().zip(magnitude.iter()) {
            *s += f64::from(m) * f64::from(m);
        }
    });
    10.0 * (band(&power, sample_rate, HF_BAND) / band(&power, sample_rate, FULL))
        .max(1e-30)
        .log10()
}

// ---------------------------------------------------------------------------
// Per pitch, and the line's own trend
// ---------------------------------------------------------------------------

/// The line's five pitches, each metric taken as the **median** over that
/// pitch's own occurrences.
///
/// Median rather than mean because the notes are played into the tail of the
/// one before them and how much tail there is depends on where in the phrase a
/// note falls; the median over six occurrences is the note, not the phrasing.
pub fn per_key(textures: &[NoteTexture]) -> Vec<(u8, [f64; 5])> {
    let mut keys: Vec<u8> = textures.iter().map(|t| t.key).collect();
    keys.sort_unstable();
    keys.dedup();
    keys.into_iter()
        .map(|key| {
            let mut out = [0.0; 5];
            for (m, slot) in out.iter_mut().enumerate() {
                let mut values: Vec<f64> = textures
                    .iter()
                    .filter(|t| t.key == key)
                    .map(|t| t.values()[m])
                    .filter(|v| v.is_finite())
                    .collect();
                *slot = median(&mut values);
            }
            (key, out)
        })
        .collect()
}

/// What one metric says about one note of the line.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LineNoteScore {
    pub key: u8,
    /// Whether the library has a recording of **this key**. `false` means the
    /// reference note is a neighbour's take resampled: it is still rendered and
    /// still listened to, and it carries no per-note score
    /// (`DECISIONS.md` 328).
    pub recorded: bool,
    /// The engine's value.
    pub engine: f64,
    /// The recordings' value for the same note. Present whether or not the note
    /// is recorded, because it is what a listener hears; scored only when it is.
    pub reference: f64,
    /// The same note out of the recordings' neighbouring velocity layer.
    pub layer: f64,
    /// How far this note stands from the engine line's own register trend. The
    /// engine renders every key from its own tables, so this is a fact at every
    /// note of the line whatever the library recorded.
    pub engine_residual: f64,
    /// The same, on the recordings' line. Kept for the report and no longer
    /// used for anything: on this line it is four clones and one note.
    pub reference_residual: f64,
    /// `engine - reference`, or `NaN` where the reference note is transposed.
    pub error: f64,
}

/// One recorded key of the register, measured the same way.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PopulationScore {
    pub key: u8,
    pub engine: f64,
    pub reference: f64,
    /// The same recorded key out of the neighbouring velocity layer: a second
    /// take of the same note by the same piano.
    pub layer: f64,
    /// How far the *recording* stands from the recordings' own register trend
    /// across the recorded keys — the piano's own note-to-note scatter.
    pub reference_residual: f64,
    /// `|reference - layer|`: how far one recorded key moves between two of its
    /// own takes.
    pub take_delta: f64,
    /// `engine - reference` at a key where both words mean the same note.
    pub error: f64,
    /// That error's departure from the register's median error.
    pub seam: f64,
}

/// One metric's picture of the line in one window, and its two verdicts.
#[derive(Clone, Debug, PartialEq)]
pub struct Column {
    pub metric: &'static str,
    pub window: Window,
    pub notes: Vec<LineNoteScore>,
    /// The recorded keys of the register, which is where every bar below comes
    /// from.
    pub population: Vec<PopulationScore>,
    // --- gate 1: the listener's own question -----------------------------
    /// The largest [`LineNoteScore::engine_residual`] on the line: how far the
    /// engine's worst note stands from the engine's own trend.
    pub standout: f64,
    pub standout_key: u8,
    /// How far a piano's own notes go, over the **recorded keys of the
    /// register**, at the same order statistic [`Column::standout`] is: the
    /// median of the largest of `n` draws, where `n` is the number of pitches
    /// in the line. This is the honest half of the bar.
    pub population_bar: f64,
    /// The single worst recorded key of the register, for the report. Not the
    /// bar: a maximum over nine keys is systematically larger than a maximum
    /// over the line's five and holding one against the other would be a
    /// comparison of sample sizes.
    pub population_standout: f64,
    pub population_standout_key: u8,
    /// The largest [`PopulationScore::take_delta`]: how far one recorded key
    /// moves between two takes of itself. The floor under everything here, and
    /// the same device `REALISM.md`'s velocity-layer floor is.
    pub take_scatter: f64,
    pub take_scatter_key: u8,
    /// What the bar used to be, kept so the change is visible: the same
    /// statistic on the *line's own* reference notes, four fifths of which are
    /// one of two recordings resampled.
    pub clone_standout: f64,
    pub clone_layer_standout: f64,
    /// What [`Column::standout`] had to come in under.
    pub bar: f64,
    pub pass: bool,
    // --- the balance verdict, for a metric that has one --------------------
    /// Whether this column is gated on [`Column::balance`] instead of on
    /// [`Column::standout`] — see [`METRIC_IS_BALANCE`].
    pub gated_on_balance: bool,
    /// The engine's median distance from the piano over the **recorded keys of
    /// the register**, signed. For `strike` that is how much louder or quieter
    /// the engine's mechanism is against its own note than the piano's is
    /// against the same note, at keys where both words mean the same note.
    pub balance: f64,
    /// What it has to come in under: the median distance between **two takes of
    /// one recorded key** — the same key out of the neighbouring velocity layer
    /// — times [`ALLOWANCE`]. The same device `REALISM.md`'s velocity-layer
    /// floor is, as a median rather than a maximum because the statistic it
    /// bounds is one.
    pub balance_bar: f64,
    // --- gate 2: does the engine track the piano, key by key --------------
    /// The largest [`PopulationScore::seam`]: how far the engine's distance
    /// from one recorded key's own recording departs from the register's median
    /// distance. Item 288's S1, measured on material where both sides are the
    /// same note.
    pub seam: f64,
    pub seam_key: u8,
    /// The seam's own noise floor: the same statistic with the neighbouring
    /// velocity layer standing in for the engine. **Reported, not gated** —
    /// item 297's objection is half answered (both sides are now the same note)
    /// and half not: the engine's absolute distance from the piano moves by
    /// several dB across a register for reasons no per-note floor covers, and
    /// on the shipped preset every column's seam is many times this number.
    pub seam_floor: f64,
    /// The one note of the *line* that carries a per-note score, and its error.
    /// `None` when the line holds no recorded key at all.
    pub line_error: Option<(u8, f64)>,
}

impl Column {
    /// The standout as a fraction of the bar. At or under 1 it passes.
    pub fn ratio(&self) -> f64 {
        self.standout / self.bar
    }

    /// How many times its own noise floor the seam is.
    pub fn seam_ratio(&self) -> f64 {
        self.seam / self.seam_floor
    }

    /// The keys of the line whose reference note is a transposition, in the
    /// order the line reads them — what a report marks `transposed — unscored`.
    pub fn transposed_keys(&self) -> Vec<u8> {
        self.notes.iter().filter(|n| !n.recorded).map(|n| n.key).collect()
    }
}

/// Scores the engine's line against the recordings' line and against the same
/// line out of the recordings' neighbouring velocity layer.
///
/// **What is gated** is the listener's own act: they played the melody on the
/// engine and heard one note that did not belong. So the gate is on the engine's
/// line alone — the largest departure of any note from the line's own register
/// trend — and the bar is the same statistic on the recordings, because a real
/// piano's melody is not even either and the question is only whether the
/// engine's line stands out *where the piano's does not*.
///
/// The trend is removed with a **Theil-Sen** line in key, the median of the ten
/// pairwise slopes: a least-squares line through five points partly absorbs the
/// very outlier this is looking for, and C4 — the note the complaint was about —
/// is an endpoint, where least squares absorbs most.
///
/// The bar is the larger of the recordings' own standout and the standout of the
/// **neighbouring velocity layer**, which is two recordings of one piano and
/// therefore the smallest difference this measurement can resolve. The floor is
/// not a formality: four of the line's five pitches are *transpositions* — the
/// library samples every minor third, so D4 and E4 come out of one recording and
/// F4 and G4 out of another — and their scatter against each other is therefore
/// smaller than a real piano's would be.
///
/// **What is reported and not gated** is the seam: `error = engine - reference`
/// per note, and its departure from the line's median error. That is item 288's
/// S1 restricted to the melody, and it answers the other question — whether the
/// engine *tracks* the piano note by note — which no bar measured off this line
/// can set honestly, because the melody contains exactly one key whose texture
/// was ever fitted to a recording.
pub fn compare(
    window: Window,
    line: &Lines,
    population: &Lines,
    recorded: &RecordedKeys,
) -> Vec<Column> {
    METRICS
        .iter()
        .enumerate()
        // A strike has no tail: the balance metric is a property of the note's
        // first 30 ms and is the same number in both windows, so it is scored
        // once and in the window that contains it.
        .filter(|&(m, _)| !(METRIC_IS_BALANCE[m] && window == Window::Tail))
        .map(|(m, &metric)| {
            let residuals = |rows: &[(u8, [f64; 5])]| -> Vec<f64> {
                let points: Vec<(f64, f64)> =
                    rows.iter().map(|&(k, v)| (f64::from(k), v[m])).collect();
                let (slope, intercept) = theil_sen(&points);
                points
                    .iter()
                    .map(|&(x, y)| y - (intercept + slope * x))
                    .collect()
            };

            // ---- the line: the engine's own evenness, note by note ----------
            let engine_resid = residuals(&line.engine);
            let reference_resid = residuals(&line.reference);
            let layer_resid = residuals(&line.layer);
            let notes: Vec<LineNoteScore> = (0..line.engine.len())
                .map(|i| {
                    let key = line.engine[i].0;
                    let is_recorded = recorded.is_recorded(key);
                    LineNoteScore {
                        key,
                        recorded: is_recorded,
                        engine: line.engine[i].1[m],
                        reference: line.reference[i].1[m],
                        layer: line.layer[i].1[m],
                        engine_residual: engine_resid[i],
                        reference_residual: reference_resid[i],
                        error: if is_recorded {
                            line.engine[i].1[m] - line.reference[i].1[m]
                        } else {
                            f64::NAN
                        },
                    }
                })
                .collect();
            let (standout_key, standout) = worst_by(&notes, |n| (n.key, n.engine_residual));
            let line_error = notes
                .iter()
                .filter(|n| n.recorded && n.error.is_finite())
                .map(|n| (n.key, n.error))
                .next();

            // ---- the population: the recorded keys of the register ----------
            let pop_reference_resid = residuals(&population.reference);
            let mut pop_errors: Vec<f64> = Vec::with_capacity(population.engine.len());
            for i in 0..population.engine.len() {
                pop_errors.push(population.engine[i].1[m] - population.reference[i].1[m]);
            }
            let pop_centre = median(&mut pop_errors.clone());
            // The seam's own noise floor, computed like for like: the same
            // statistic with the neighbouring velocity layer standing in for the
            // engine. Two takes of one piano differ from key to key too, and how
            // much they differ is the only honest thing to hold `seam` against —
            // `take_scatter` alone would be the floor of a *difference*, not of
            // a difference's departure from its own centre.
            let take_signed: Vec<f64> = (0..population.reference.len())
                .map(|i| population.reference[i].1[m] - population.layer[i].1[m])
                .collect();
            let take_centre = median(&mut take_signed.clone());
            let seam_floor = take_signed
                .iter()
                .map(|d| (d - take_centre).abs())
                .fold(0.0f64, f64::max);
            let population_rows: Vec<PopulationScore> = (0..population.engine.len())
                .map(|i| PopulationScore {
                    key: population.engine[i].0,
                    engine: population.engine[i].1[m],
                    reference: population.reference[i].1[m],
                    layer: population.layer[i].1[m],
                    reference_residual: pop_reference_resid[i],
                    take_delta: (population.reference[i].1[m] - population.layer[i].1[m]).abs(),
                    error: pop_errors[i],
                    seam: pop_errors[i] - pop_centre,
                })
                .collect();
            let (population_standout_key, population_standout) =
                worst_by(&population_rows, |p| (p.key, p.reference_residual));
            let (take_scatter_key, take_scatter) =
                worst_by(&population_rows, |p| (p.key, p.take_delta));
            let (seam_key, seam) = worst_by(&population_rows, |p| (p.key, p.seam));
            let population_bar = worst_of_n(
                &population_rows
                    .iter()
                    .map(|p| p.reference_residual)
                    .collect::<Vec<f64>>(),
                notes.len(),
            );

            let bar = population_bar.max(take_scatter) * ALLOWANCE;

            // The balance: the engine's median distance from the piano over the
            // recorded keys, against the median distance between two takes of
            // one of them. Both are medians over the same nine keys, so the
            // comparison is like for like.
            let mut errors: Vec<f64> = population_rows
                .iter()
                .map(|p| p.error)
                .filter(|e| e.is_finite())
                .collect();
            let balance = median(&mut errors);
            let mut takes: Vec<f64> = population_rows
                .iter()
                .map(|p| p.take_delta)
                .filter(|d| d.is_finite())
                .collect();
            // **Two terms, and the second one is why this bar is honest for a
            // statistic the recording repeats exactly.** The take-to-take
            // distance is the right floor when it is the limit of the
            // measurement — `strike` reads 1.64 dB between two velocity layers
            // of one key, because how much hammer noise a take has is a
            // property of that take. `channel` does not: the two layers are the
            // *same two microphones on the same key*, so the ratio of what they
            // hear repeats to **0.03 dB**, and a bar of 0.04 dB would be a bar
            // on the recording's dither. What the question can actually be
            // asked to is the second term — how well nine keys pin a median
            // that moves by `sigma` across them — which is
            // `realism::StereoColumn`'s `uncertainty` exactly, and for the same
            // reason (`DECISIONS.md` 348, 394). The larger of the two governs,
            // so nothing about `strike` moves: its register sigma over nine
            // keys is smaller than its 1.64 dB take floor.
            let mut spread: Vec<f64> = population_rows
                .iter()
                .map(|p| p.reference_residual.abs())
                .filter(|d| d.is_finite())
                .collect();
            let n = spread.len().max(1) as f64;
            let register_sigma = 1.4826 * median(&mut spread) / n.sqrt();
            let balance_bar = median(&mut takes).max(register_sigma) * ALLOWANCE;
            let gated_on_balance = METRIC_IS_BALANCE[m];

            Column {
                metric,
                window,
                pass: if gated_on_balance {
                    balance.abs() <= balance_bar
                } else {
                    standout <= bar
                },
                notes,
                population: population_rows,
                standout,
                standout_key,
                population_bar,
                population_standout,
                population_standout_key,
                take_scatter,
                take_scatter_key,
                clone_standout: biggest(&reference_resid),
                clone_layer_standout: biggest(&layer_resid),
                bar,
                gated_on_balance,
                balance,
                balance_bar,
                seam,
                seam_key,
                seam_floor,
                line_error,
            }
        })
        .collect()
}

/// One metric measured on the same music through three players: the engine, the
/// recordings, and the recordings' neighbouring velocity layer.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Lines {
    pub engine: Vec<(u8, [f64; 5])>,
    pub reference: Vec<(u8, [f64; 5])>,
    pub layer: Vec<(u8, [f64; 5])>,
}

impl Lines {
    pub fn new(
        engine: Vec<(u8, [f64; 5])>,
        reference: Vec<(u8, [f64; 5])>,
        layer: Vec<(u8, [f64; 5])>,
    ) -> Self {
        Lines {
            engine,
            reference,
            layer,
        }
    }
}

/// The largest `|value|` in a set, and the key it belongs to.
fn worst_by<T>(rows: &[T], pick: impl Fn(&T) -> (u8, f64)) -> (u8, f64) {
    rows.iter().fold((0u8, 0.0f64), |best, row| {
        let (key, value) = pick(row);
        let value = value.abs();
        if value.is_finite() && value > best.1 {
            (key, value)
        } else {
            best
        }
    })
}

fn biggest(values: &[f64]) -> f64 {
    values.iter().map(|v| v.abs()).fold(0.0f64, f64::max)
}

/// How large the largest of `n` draws from `values` typically is.
///
/// The gate's own statistic is a **maximum over the line's five pitches**, and
/// the bar it is held against has to be the same order statistic or the
/// comparison is one of sample sizes rather than of instruments: the largest of
/// nine recorded keys is systematically bigger than the largest of five notes,
/// so a bar set on it forgives the engine an amount that grows with how many
/// keys the library happened to sample.
///
/// The matched statistic is the **median of the maximum of `n` draws**, which
/// is the `0.5^(1/n)` quantile of the distribution — 0.871 for five. Read off
/// the sorted magnitudes by linear interpolation, which is the ordinary
/// definition of a sample quantile and needs no distribution assumed.
///
/// Measured on the instrument this gate was widened for, this is what the
/// choice is worth: on the tail `hf` column the maximum over the register is
/// 6.75 dB and the matched quantile is 4.26, and the engine's own worst note
/// stood at 5.43 — so the maximum would have passed the note the listener
/// complained about and the matched statistic failed it. That is the whole
/// argument for taking the trouble, and the fix that followed it
/// (`DECISIONS.md` 335) took the same column to 3.76.
pub fn worst_of_n(values: &[f64], n: usize) -> f64 {
    let mut magnitudes: Vec<f64> = values
        .iter()
        .map(|v| v.abs())
        .filter(|v| v.is_finite())
        .collect();
    if magnitudes.is_empty() {
        return 0.0;
    }
    magnitudes.sort_by(f64::total_cmp);
    if magnitudes.len() == 1 || n == 0 {
        return magnitudes[magnitudes.len() - 1];
    }
    let quantile = 0.5f64.powf(1.0 / n as f64);
    let position = quantile * (magnitudes.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = (lower + 1).min(magnitudes.len() - 1);
    let t = position - lower as f64;
    magnitudes[lower] + t * (magnitudes[upper] - magnitudes[lower])
}

/// A line through `points` resistant to one bad point: the median of every
/// pairwise slope, and the median offset about it.
pub fn theil_sen(points: &[(f64, f64)]) -> (f64, f64) {
    let mut slopes: Vec<f64> = Vec::new();
    for (i, &(x0, y0)) in points.iter().enumerate() {
        for &(x1, y1) in &points[i + 1..] {
            if (x1 - x0).abs() > 1e-9 {
                slopes.push((y1 - y0) / (x1 - x0));
            }
        }
    }
    let slope = median(&mut slopes);
    let slope = if slope.is_finite() { slope } else { 0.0 };
    let mut offsets: Vec<f64> = points.iter().map(|&(x, y)| y - slope * x).collect();
    let intercept = median(&mut offsets);
    (slope, if intercept.is_finite() { intercept } else { 0.0 })
}

/// A printable table of one comparison — the text the gate prints when it fails
/// and the driver prints always.
pub fn report(columns: &[Column]) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    for c in columns {
        if c.gated_on_balance {
            let _ = writeln!(
                out,
                "{:<10} {:<4} balance {:+6.2} over the recorded keys, bar {:5.2} \
(one key's two takes {:4.2}, x{:.2}) {}  |  evenness {:5.2} at {:<3} of {:5.2}",
                c.metric,
                c.window.name(),
                c.balance,
                c.balance_bar,
                c.balance_bar / ALLOWANCE,
                ALLOWANCE,
                if c.pass { "pass" } else { "FAIL" },
                c.standout,
                note_name(c.standout_key),
                c.bar,
            );
        } else {
        let _ = writeln!(
            out,
            "{:<10} {:<4} stands out {:5.2} at {:<3} bar {:5.2} (register {:4.2}, worst {:4.2} at {}, \
take {:4.2} at {}, x{:.2}) {}  |  seam {:5.2} at {:<3} floor {:4.2}",
            c.metric,
            c.window.name(),
            c.standout,
            note_name(c.standout_key),
            c.bar,
            c.population_bar,
            c.population_standout,
            note_name(c.population_standout_key),
            c.take_scatter,
            note_name(c.take_scatter_key),
            ALLOWANCE,
            if c.pass { "pass" } else { "FAIL" },
            c.seam,
            note_name(c.seam_key),
            c.seam_floor,
        );
        }
        let _ = writeln!(
            out,
            "{:<15}   line: key    engine  (resid)  recording  (resid)    layer     error",
            ""
        );
        for n in &c.notes {
            let _ = writeln!(
                out,
                "{:<15}         {:<4} {:7.2}  {:6.2}   {:8.2} {:7.2}  {:7.2}  {}",
                "",
                note_name(n.key),
                n.engine,
                n.engine_residual,
                n.reference,
                n.reference_residual,
                n.layer,
                if n.recorded {
                    format!("{:8.2}", n.error)
                } else {
                    "transposed — unscored".to_string()
                },
            );
        }
        let _ = writeln!(
            out,
            "{:<15}   recorded keys: key    engine  recording  (resid)    layer   take   error   seam",
            ""
        );
        for p in &c.population {
            let _ = writeln!(
                out,
                "{:<15}                  {:<4} {:7.2}  {:8.2} {:7.2}  {:7.2} {:6.2} {:7.2} {:6.2}",
                "",
                note_name(p.key),
                p.engine,
                p.reference,
                p.reference_residual,
                p.layer,
                p.take_delta,
                p.error,
                p.seam
            );
        }
    }
    out
}

/// `C4` for 60.
pub use crate::realism::note_name;

/// A rendered line cut so that its first note starts where the phrase says it
/// does.
///
/// Kept here so the gate and its driver align the reference the same way: the
/// sampler's recordings begin with their own pre-attack silence, and a per-note
/// window that did not remove it would read the note before.
pub fn align_reference(rendered: &Audio, first_onset_s: f64) -> Audio {
    let mono = rendered.mono();
    let sr = f64::from(rendered.sample_rate);
    let onset = crate::detect_onset(&mono, sr);
    let shift = onset - first_onset_s;
    let skip = (shift * sr).round().max(0.0) as usize;
    let channels: Vec<Vec<f32>> = rendered
        .channels
        .iter()
        .map(|c| c.iter().skip(skip).copied().collect())
        .collect();
    Audio::new(rendered.sample_rate, channels).unwrap_or_else(|_| rendered.clone())
}

fn median(v: &mut Vec<f64>) -> f64 {
    v.retain(|x| x.is_finite());
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_soprano_line_is_the_excerpts_own_melody() {
        let excerpt = realism::excerpt();
        let soprano = soprano();
        // Every note the soprano plays is a note the excerpt plays, at the same
        // instant, at the same velocity.
        let mut excerpt_notes: Vec<(u64, u8, u8)> = excerpt
            .events
            .iter()
            .filter_map(|e| match e.event {
                SamplerEvent::NoteOn { key, vel } => {
                    Some(((e.time_s * 1e6).round() as u64, key, vel))
                }
                _ => None,
            })
            .collect();
        excerpt_notes.sort_unstable();
        for e in &soprano.events {
            if let SamplerEvent::NoteOn { key, vel } = e.event {
                let want = ((e.time_s * 1e6).round() as u64, key, vel);
                assert!(
                    excerpt_notes.contains(&want),
                    "the soprano plays {key} at {:.3} s and the excerpt does not",
                    e.time_s
                );
            }
        }
        assert_eq!(soprano.note_count(), realism::ODE_MELODY.len());
        // And nothing else: no pedal, no harmony.
        assert!(soprano
            .events
            .iter()
            .all(|e| matches!(e.event, SamplerEvent::NoteOn { .. } | SamplerEvent::NoteOff { .. })));
        assert_eq!(line_keys(), vec![60, 62, 64, 65, 67]);
    }

    #[test]
    fn the_two_passing_notes_are_the_only_ones_left_out() {
        let notes = line_notes();
        let short: Vec<u8> = notes
            .iter()
            .filter(|n| !n.measurable())
            .map(|n| n.key)
            .collect();
        assert_eq!(short, vec![62, 60]);
        assert_eq!(notes.iter().filter(|n| n.measurable()).count(), 28);
    }

    /// The library this crate is measured against: one key every minor third.
    fn salamander_keys() -> RecordedKeys {
        RecordedKeys::from_keys(&(21u8..=108).step_by(3).collect::<Vec<u8>>())
    }

    /// The columns gated on evenness, which is every one but `strike`: the
    /// three texture metrics answer "does a note stand out" and the balance
    /// column answers "is the mechanism as loud as the piano's", so a fixture
    /// built to exercise one of them says nothing about the other.
    fn evenness(columns: Vec<Column>) -> Vec<Column> {
        columns.into_iter().filter(|c| !c.gated_on_balance).collect()
    }

    /// Five pitches, every metric alike, on a rising register trend of `slope` per
    /// semitone; `bump` is added to the engine's C4 alone.
    fn lines(bump: f64, slope: f64) -> Lines {
        let keys = [60u8, 62, 64, 65, 67];
        let reference: Vec<(u8, [f64; 5])> = keys
            .iter()
            .map(|&k| (k, [slope * f64::from(k - 60); 5]))
            .collect();
        // The neighbouring velocity layer: the piano again, on its own line.
        let layer = reference.clone();
        // The engine is the same line 3 dB smoother — a voicing offset — plus a
        // bump on one note.
        let mut engine: Vec<(u8, [f64; 5])> = reference
            .iter()
            .map(|&(k, v)| (k, v.map(|x| x - 3.0)))
            .collect();
        engine[0].1 = engine[0].1.map(|x| x + bump);
        Lines::new(engine, reference, layer)
    }

    /// The recorded-key population that sets the bars: nine takes on a straight
    /// register trend, the engine parallel to them `offset` dB below, with
    /// `scatter` on one recording and `take` between one key's two takes.
    fn population(slope: f64, offset: f64, scatter: f64, take: f64) -> Lines {
        // Eight keys off the trend by `scatter` and one on it, so that the
        // matched quantile of the magnitudes is `scatter` exactly however the
        // interpolation lands, and the median offset is still zero.
        const PATTERN: [f64; 9] = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 0.0];
        let keys: Vec<u8> = (51u8..=75).step_by(3).collect();
        let engine: Vec<(u8, [f64; 5])> = keys
            .iter()
            .map(|&k| (k, [slope * (f64::from(k) - 60.0) - offset; 5]))
            .collect();
        let reference: Vec<(u8, [f64; 5])> = keys
            .iter()
            .zip(PATTERN)
            .map(|(&k, p)| (k, [slope * (f64::from(k) - 60.0) + scatter * p; 5]))
            .collect();
        let mut layer = reference.clone();
        layer[5].1 = layer[5].1.map(|x| x + take);
        Lines::new(engine, reference, layer)
    }

    #[test]
    fn a_uniform_offset_and_a_register_trend_are_not_a_note_standing_out() {
        let line = lines(0.0, 0.7);
        let pop = population(0.7, 3.0, 0.0, 0.0);
        for c in evenness(compare(Window::Head, &line, &pop, &salamander_keys())) {
            assert!(
                c.standout < 1e-9 && c.seam.abs() < 1e-9,
                "{} reads {:.3} on a line that is 3 dB down and perfectly parallel",
                c.metric,
                c.standout
            );
        }
    }

    #[test]
    fn one_note_off_the_trend_is_the_note_that_is_named() {
        let line = lines(4.0, 0.7);
        // A piano whose own register scatters 1 dB and whose takes agree to
        // 0.5 dB: a 4 dB note on the engine's line does not belong.
        let pop = population(0.7, 3.0, 1.0, 0.5);
        for c in evenness(compare(Window::Head, &line, &pop, &salamander_keys())) {
            assert_eq!(c.standout_key, 60, "{}", c.metric);
            assert!(
                (c.standout - 4.0).abs() < 1e-9,
                "{} {:.3}",
                c.metric,
                c.standout
            );
            assert!(!c.pass, "{} passed a 4 dB bump at a bar of {:.2}", c.metric, c.bar);
        }
    }

    #[test]
    fn the_bar_is_the_recorded_registers_own_scatter_and_never_under_one_keys_two_takes() {
        let line = lines(4.0, 0.0);
        // The register itself has a 5 dB note in it: the engine having a 4 dB
        // one is not news.
        for c in evenness(compare(
            Window::Head,
            &line,
            &population(0.0, 3.0, 5.0, 0.0),
            &salamander_keys(),
        )) {
            assert!(
                (c.population_bar - 5.0).abs() < 1e-6,
                "{} {:.3}",
                c.metric,
                c.population_bar
            );
            assert!(c.pass, "{} failed at a bar of {:.2}", c.metric, c.bar);
        }
        // With an even register, one key's two takes are what is left.
        for c in evenness(compare(
            Window::Head,
            &line,
            &population(0.0, 3.0, 0.0, 6.0),
            &salamander_keys(),
        )) {
            assert!((c.take_scatter - 6.0).abs() < 1e-9, "{} {:.3}", c.metric, c.take_scatter);
            assert!((c.bar - 6.0 * ALLOWANCE).abs() < 1e-9, "{} {:.3}", c.metric, c.bar);
            assert!(c.pass, "{} failed 4 dB against a 6 dB take floor", c.metric);
        }
    }

    #[test]
    fn the_balance_column_is_scored_on_the_recorded_keys_and_not_on_the_line() {
        // The line is 4 dB out at C4 and the register is even, so the evenness
        // gate has something to say and the balance gate has nothing: the
        // engine sits exactly on the recordings at every recorded key.
        // Every recorded key one decibel from its own other take, which is what
        // sets this column's bar; `population`'s own fixture moves one key, and
        // a median over nine reads that as zero.
        let takes = |offset: f64, take: f64| -> Lines {
            let keys: Vec<u8> = (51u8..=75).step_by(3).collect();
            let engine: Vec<(u8, [f64; 5])> =
                keys.iter().map(|&k| (k, [-offset; 5])).collect();
            let reference: Vec<(u8, [f64; 5])> = keys.iter().map(|&k| (k, [0.0; 5])).collect();
            let layer: Vec<(u8, [f64; 5])> = keys.iter().map(|&k| (k, [take; 5])).collect();
            Lines::new(engine, reference, layer)
        };
        let line = lines(4.0, 0.0);
        let pop = takes(0.0, 1.0);
        let strike = compare(Window::Head, &line, &pop, &salamander_keys())
            .into_iter()
            .find(|c| c.metric == "strike")
            .expect("the balance column");
        assert!(strike.gated_on_balance);
        assert!(strike.balance.abs() < 1e-9, "{:.3}", strike.balance);
        assert!((strike.balance_bar - 1.0 * ALLOWANCE).abs() < 1e-9);
        assert!(strike.pass, "an engine on top of the recordings failed");
        // The 4 dB note of the *line* is the evenness number and it is printed
        // rather than gated: the balance column does not fail on it.
        assert!((strike.standout - 4.0).abs() < 1e-9);

        // Now the engine is 3 dB from the piano at every recorded key with the
        // same 1 dB between two takes of one of them. That is the balance, and
        // it fails.
        let off = compare(Window::Head, &line, &takes(3.0, 1.0), &salamander_keys())
            .into_iter()
            .find(|c| c.metric == "strike")
            .expect("the balance column");
        assert!((off.balance + 3.0).abs() < 1e-9, "{:.3}", off.balance);
        assert!(!off.pass, "3 dB from the piano passed a bar of {:.2}", off.balance_bar);
        // And the sign is kept: which way the mechanism is wrong is the whole
        // attribution.
        assert!(off.balance < 0.0);
    }

    /// **A tail carries none of the balance metrics**, and the count is
    /// derived from [`METRIC_IS_BALANCE`] rather than written down, because
    /// the version of this test that wrote `METRICS.len() - 1` went red the
    /// day item 394 added a second balance metric — `channel` — and stayed red
    /// while saying nothing about the rule it was there to pin.
    ///
    /// The rule is the one at [`compare`]'s own filter: a balance metric is
    /// scored against the recording's own value at the same note, and the tail
    /// window is 0.5-2.0 s after a strike, where neither a hammer's burst nor
    /// two capsules' view of one is still the thing being measured.
    #[test]
    fn a_tail_has_no_strike_in_it() {
        let line = lines(0.0, 0.0);
        let pop = population(0.0, 0.0, 0.0, 0.0);
        let head = compare(Window::Head, &line, &pop, &salamander_keys());
        let tail = compare(Window::Tail, &line, &pop, &salamander_keys());
        let balances = METRIC_IS_BALANCE.iter().filter(|&&b| b).count();
        assert!(balances >= 2, "this test stops meaning anything at one");
        assert_eq!(head.len(), METRICS.len());
        assert_eq!(tail.len(), METRICS.len() - balances);
        for (m, &is_balance) in METRIC_IS_BALANCE.iter().enumerate() {
            assert_eq!(
                tail.iter().any(|c| c.metric == METRICS[m]),
                !is_balance,
                "{} is in the tail and should not be, or is not and should be",
                METRICS[m]
            );
        }
    }

    #[test]
    fn the_clone_line_would_have_set_a_bar_the_recorded_register_refuses() {
        // Four of the line's five reference notes are transpositions of two
        // recordings, so the reference line is smooth by construction: its own
        // worst note is 0 dB off its trend. The old bar was that number.
        let line = lines(4.0, 0.7);
        let pop = population(0.7, 3.0, 1.0, 0.5);
        for c in evenness(compare(Window::Head, &line, &pop, &salamander_keys())) {
            assert!(
                c.clone_standout < 1e-9,
                "{} clone line stands out {:.3}",
                c.metric,
                c.clone_standout
            );
            assert!(
                c.bar > 1.0,
                "{} bar {:.3} did not come off the recorded register",
                c.metric,
                c.bar
            );
        }
    }

    #[test]
    fn only_the_lines_recorded_key_carries_a_per_note_error() {
        let line = lines(4.0, 0.7);
        let pop = population(0.7, 3.0, 1.0, 0.5);
        for c in evenness(compare(Window::Head, &line, &pop, &salamander_keys())) {
            assert_eq!(c.transposed_keys(), vec![62, 64, 65, 67], "{}", c.metric);
            let scored: Vec<u8> = c
                .notes
                .iter()
                .filter(|n| n.recorded)
                .map(|n| n.key)
                .collect();
            assert_eq!(scored, vec![60], "{}", c.metric);
            for n in &c.notes {
                assert_eq!(
                    n.error.is_finite(),
                    n.recorded,
                    "{} at {}: error {:?} on a {} note",
                    c.metric,
                    note_name(n.key),
                    n.error,
                    if n.recorded { "recorded" } else { "transposed" }
                );
            }
            assert_eq!(c.line_error.map(|(k, _)| k), Some(60), "{}", c.metric);
        }
    }

    #[test]
    fn the_seam_is_measured_where_both_sides_are_the_same_note() {
        let line = lines(0.0, 0.0);
        let mut pop = population(0.0, 3.0, 0.0, 0.5);
        // One recorded key the engine gets wrong by 4 dB more than the rest.
        pop.engine[3].1 = pop.engine[3].1.map(|x| x + 4.0);
        for c in evenness(compare(Window::Head, &line, &pop, &salamander_keys())) {
            assert_eq!(c.seam_key, 60, "{}", c.metric);
            assert!((c.seam - 4.0).abs() < 1e-9, "{} {:.3}", c.metric, c.seam);
            assert!(
                c.seam > 4.0 * c.seam_floor,
                "{} reads a 4 dB seam as {:.2} against a floor of {:.2}",
                c.metric,
                c.seam,
                c.seam_floor
            );
        }
    }

    #[test]
    fn the_recorded_keys_of_the_melodys_register_are_the_ladders_own() {
        let keys = ladder_keys(&salamander_keys(), &line_keys());
        assert_eq!(keys, vec![51, 54, 57, 60, 63, 66, 69, 72, 75]);
        // At tempo, up and back down: two contexts for every key.
        let fast = ladder(&keys, Window::Head);
        assert_eq!(fast.note_count(), 2 * keys.len() - 1);
        // Slowly, once up: with every note alone there is no context to average.
        let slow = ladder(&keys, Window::Tail);
        assert_eq!(slow.note_count(), keys.len());
        for phrase in [&fast, &slow] {
            assert!(phrase.events.iter().all(|e| matches!(
                e.event,
                SamplerEvent::NoteOn { .. } | SamplerEvent::NoteOff { .. }
            )));
        }
    }

    /// A tail is only a tail when it is the note's own. On the slow line every
    /// note is let go before the next one is struck, so nothing else is
    /// sounding inside [`TAIL_WINDOW_S`] — which is the property the melody's
    /// own tempo cannot have.
    #[test]
    fn no_note_of_the_slow_line_sounds_inside_another_ones_tail() {
        let notes = slow_line_notes();
        assert_eq!(notes.len(), TAIL_PASSES * line_keys().len());
        assert!(notes.iter().all(LineNote::measurable));
        // The pitches, in the order the tune first sounds them.
        assert_eq!(
            notes.iter().take(5).map(|n| n.key).collect::<Vec<u8>>(),
            vec![64, 65, 67, 62, 60]
        );
        for pair in notes.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            assert!(
                a.onset_s + a.held_s <= b.onset_s,
                "{} is still held when {} is struck",
                note_name(a.key),
                note_name(b.key)
            );
            assert!(
                b.onset_s >= a.onset_s + TAIL_WINDOW_S.1,
                "{}'s tail window runs past the next strike",
                note_name(a.key)
            );
        }
        // And the window is entirely inside the held note, so no damper is in it.
        assert!(TAIL_HOLD_S >= TAIL_WINDOW_S.1);
        assert!(slow_line().duration_s >= notes.last().unwrap().onset_s + TAIL_WINDOW_S.1);
        assert!(slow_line().events.iter().all(|e| matches!(
            e.event,
            SamplerEvent::NoteOn { .. } | SamplerEvent::NoteOff { .. }
        )));
    }

    #[test]
    fn the_take_a_transposed_note_is_made_of_is_named() {
        let keys = salamander_keys();
        assert!(keys.is_recorded(60));
        for (key, take) in [(62u8, 63u8), (64, 63), (65, 66), (67, 66)] {
            assert!(!keys.is_recorded(key), "{key}");
            assert_eq!(keys.take_for(key), Some(take), "{key}");
        }
        // And the alternative route, which is what `bench` measures the cost of
        // transposition with.
        assert_eq!(keys.alternate_take(62), Some(60));
        assert_eq!(keys.alternate_take(64), Some(66));
        assert_eq!(keys.alternate_take(60), None);
    }

    #[test]
    fn the_bar_reads_the_order_statistic_the_gate_is_and_not_the_population_maximum() {
        // Nine draws, one of them far out. The maximum is the outlier; the
        // largest of five typically is not.
        let values = [0.0, 0.2, 0.4, 0.4, 0.5, 0.6, 0.7, 1.2, 6.8];
        assert!((worst_of_n(&values, 5) - 1.18).abs() < 0.02, "{}", worst_of_n(&values, 5));
        // Asking for the largest of many draws walks back up towards it.
        assert!(worst_of_n(&values, 100) > 3.0);
        // Signs do not matter, and one sample is its own answer.
        assert_eq!(worst_of_n(&[-3.0], 5), 3.0);
        assert!(worst_of_n(&[], 5).abs() < 1e-12);
    }

    #[test]
    fn theil_sen_ignores_one_bad_point_where_least_squares_does_not() {
        let mut points: Vec<(f64, f64)> =
            (0..7).map(|i| (f64::from(i), 2.0 * f64::from(i))).collect();
        points[2].1 += 50.0;
        let (slope, intercept) = theil_sen(&points);
        assert!((slope - 2.0).abs() < 1e-9, "slope {slope}");
        assert!(intercept.abs() < 1e-9, "intercept {intercept}");
        let n = points.len() as f64;
        let mx = points.iter().map(|p| p.0).sum::<f64>() / n;
        let my = points.iter().map(|p| p.1).sum::<f64>() / n;
        let ls = points.iter().map(|(x, y)| (x - mx) * (y - my)).sum::<f64>()
            / points.iter().map(|(x, _)| (x - mx) * (x - mx)).sum::<f64>();
        assert!((ls - 2.0).abs() > 1.0, "least squares {ls} was not fooled");
    }

    #[test]
    fn wobble_reads_a_modulated_partial_and_not_a_plain_decay() {
        let sr = 48_000.0;
        let n = (0.37 * sr) as usize;
        let hz = 261.6;
        let mut plain = Vec::with_capacity(n);
        let mut beating = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f64 / sr;
            let carrier = (std::f64::consts::TAU * hz * t).sin();
            let decay = (-1.5 * t).exp();
            plain.push((carrier * decay) as f32);
            // 6 dB peak-to-peak at 2 Hz on top of the same decay.
            let am = 1.0 + 0.5 * (std::f64::consts::TAU * 2.0 * t).sin();
            beating.push((carrier * decay * am) as f32);
        }
        let hzs = vec![hz];
        let plain_series = Series::measure(&plain, &hzs, sr);
        let beat_series = Series::measure(&beating, &hzs, sr);
        let quiet = wobble(&plain, sr, &hzs, &plain_series);
        let loud = wobble(&beating, sr, &hzs, &beat_series);
        // A plain decay *is* the line, so it reads 0.03 dB; the beating partial
        // reads 1.46. It is not 6 dB, and the reason is the window this gate is
        // allowed: 0.37 s of a 2 Hz beat is three quarters of one cycle, and a
        // straight line through three quarters of a cycle absorbs most of it.
        // That is not a defect of the metric — it is exactly how much of a slow
        // beat a listener gets to hear inside a quarter note.
        assert!(quiet < 0.5, "a plain decay wobbles {quiet:.2} dB");
        assert!(loud > 1.0, "a beating partial wobbles only {loud:.2} dB");
        assert!(loud > 10.0 * quiet, "quiet {quiet:.3}, loud {loud:.3}");
    }

    #[test]
    fn hf_share_hears_a_bright_note_as_brighter() {
        let sr = 48_000.0;
        let n = (0.37 * sr) as usize;
        let stft = Stft::new(StftConfig::new(HF_WINDOW, HF_WINDOW / 4, HF_WINDOW).unwrap()).unwrap();
        let tone = |hz: f64| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let t = i as f64 / sr;
                    (0.3 * (std::f64::consts::TAU * hz * t).sin()) as f32
                })
                .collect()
        };
        let dark = hf_share_db(&stft, &tone(500.0), sr);
        let bright = hf_share_db(&stft, &tone(3_000.0), sr);
        assert!(bright > dark + 20.0, "dark {dark:.1}, bright {bright:.1}");
    }
}
