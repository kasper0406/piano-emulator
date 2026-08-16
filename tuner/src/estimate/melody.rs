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
//! # The three numbers, per note
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
//!
//! `wobble` is not [`motion::Motion::beat_depth_db`], which `COMPASS.md` uses,
//! and it cannot be: `motion.rs` measures over 0.3-3.0 s and a melody gives one
//! note 0.4 s before the next strike. It is the same physical quantity read in
//! the window a *tune* leaves, which is the window the complaint was made in.
//!
//! # The score
//!
//! Five distinct pitches (C4, D4, E4, F4, G4). What is gated is the listener's
//! own act: they played the melody on the engine and heard one note that did
//! not belong. So the gate is the largest departure of any note from the
//! **engine line's own** register trend, held against the same statistic on the
//! recordings — because a real piano's melody is not even either (these five
//! notes read 7.3 / 6.7 / 7.4 / 8.1 / 9.6 dB of roughness on the recordings)
//! and the question is only whether the engine's line stands out *where the
//! piano's does not*.
//!
//! What is **not** gated is the engine line's own smoothness. An engine line
//! smoother than the piano's is precisely the defect item 284 was opened on, so
//! a gate that rewarded smoothness would reward the disease: the pre-284 line
//! is the *flatter* of the two here, and it is the wrong one.
//!
//! Reported beside the gate is the **seam**: `error = engine - reference` per
//! note and its departure from the line's median error, which is item 288's S1
//! restricted to the melody. It is not gated because no bar measured off this
//! line can be set for it honestly — the melody contains exactly one key whose
//! texture was ever fitted to a recording. See [`compare`] for both.

use crate::audio::Audio;
use crate::estimate::brilliance::{band, FULL, HF1};
use crate::realism::{self, Phrase};
use crate::sampler::{SamplerEvent, TimedEvent};
use crate::series::Series;
use crate::stft::{Stft, StftConfig};

/// The window one note is measured over, seconds from its own onset.
///
/// Starts past the hammer's noise and the attack transient, and ends before the
/// key is released — the melody's quarter notes are held 0.45 s, so 0.40 s is
/// entirely inside the sounding note and none of it is a damper falling.
pub const NOTE_WINDOW_S: (f64, f64) = (0.03, 0.40);

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

/// How much of the measured bar a note is allowed to stand out by.
///
/// One, plus a quarter, because the bar is itself a measurement off five notes
/// and a gate that tripped whenever a measurement landed a hair over its own
/// noise floor would be a coin flip rather than a gate. It is not a tolerance
/// on the defect and nothing here is near it: on the shipped preset the three
/// columns read 0.92, **4.94** and 0.18 of their bars, and on the instrument
/// item 284 started from 0.51, 2.08 and 0.37 (`DECISIONS.md` 298).
pub const ALLOWANCE: f64 = 1.25;

/// Names in the order [`NoteTexture::values`] returns them.
pub const METRICS: [&str; 3] = ["roughness", "wobble", "hf"];

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
    let mut events = Vec::with_capacity(2 * realism::ODE_MELODY.len());
    for note in line_notes() {
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
    Phrase {
        name: "ode_soprano",
        description: "the Ode to Joy melody line of `excerpt`, alone and unpedalled",
        duration_s: SOPRANO_S,
        events,
    }
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
// The three numbers
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
}

impl NoteTexture {
    pub fn values(&self) -> [f64; 3] {
        [self.roughness_db, self.wobble_db, self.hf_db]
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
    mono: &[f32],
    sample_rate: f64,
    notes: &[LineNote],
    partial_hz: &dyn Fn(u8) -> Vec<f64>,
) -> Vec<NoteTexture> {
    let stft = Stft::new(StftConfig::new(HF_WINDOW, HF_WINDOW / 4, HF_WINDOW).expect("valid"))
        .expect("valid");
    notes
        .iter()
        .filter(|n| n.measurable())
        .map(|note| {
            let strike = note_onset(mono, sample_rate, note.onset_s);
            let lo = ((strike + NOTE_WINDOW_S.0) * sample_rate) as usize;
            let hi = (((strike + NOTE_WINDOW_S.1) * sample_rate) as usize).min(mono.len());
            let window = &mono[lo.min(hi)..hi];
            let hz = partial_hz(note.key);
            let series = Series::measure(window, &hz, sample_rate);
            NoteTexture {
                key: note.key,
                onset_s: strike,
                roughness_db: series.irregularity(),
                wobble_db: wobble(window, sample_rate, &hz, &series),
                hf_db: hf_share_db(&stft, window, sample_rate),
            }
        })
        .collect()
}

/// Where the strike actually is, near where the phrase says it is.
///
/// The largest rise in a 1 ms RMS envelope over [`ONSET_SEARCH_S`]. A rise
/// rather than a level, because a melody note is struck **into the tail of the
/// note before it** and any threshold on level would fire on that tail; a piano
/// strike is the one thing in the window that goes up.
pub fn note_onset(mono: &[f32], sample_rate: f64, nominal_s: f64) -> f64 {
    let block = ((sample_rate * 0.001) as usize).max(1);
    let from = (((nominal_s + ONSET_SEARCH_S.0) * sample_rate) as isize).max(0) as usize;
    let to = ((((nominal_s + ONSET_SEARCH_S.1) * sample_rate) as usize) + block).min(mono.len());
    if from + 4 * block >= to {
        return nominal_s;
    }
    let envelope: Vec<f64> = mono[from..to]
        .chunks(block)
        .map(|c| (c.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>() / c.len() as f64).sqrt())
        .collect();
    // Over three milliseconds: one is inside the hammer's own contact time at
    // every key of this line, and a single block is noise.
    let step = 3usize;
    let mut best = (0usize, f64::MIN);
    for i in 0..envelope.len().saturating_sub(step) {
        let rise = envelope[i + step] - envelope[i];
        if rise > best.1 {
            best = (i, rise);
        }
    }
    from as f64 / sample_rate + best.0 as f64 * block as f64 / sample_rate
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
pub fn per_key(textures: &[NoteTexture]) -> Vec<(u8, [f64; 3])> {
    let mut keys: Vec<u8> = textures.iter().map(|t| t.key).collect();
    keys.sort_unstable();
    keys.dedup();
    keys.into_iter()
        .map(|key| {
            let mut out = [0.0; 3];
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
    /// The engine's value.
    pub engine: f64,
    /// The recordings' value for the same note.
    pub reference: f64,
    /// The same note out of the recordings' neighbouring velocity layer.
    pub layer: f64,
    /// How far this note stands from the engine line's own register trend.
    pub engine_residual: f64,
    /// The same, on the recordings' line.
    pub reference_residual: f64,
    /// `engine - reference`: how far this note's texture is from the piano's.
    pub error: f64,
    /// That error's departure from the line's median error.
    pub seam: f64,
}

/// One metric's picture of the line, and its verdict.
#[derive(Clone, Debug, PartialEq)]
pub struct Column {
    pub metric: &'static str,
    pub notes: Vec<LineNoteScore>,
    // --- the gate: the listener's own question --------------------------
    /// The largest [`LineNoteScore::engine_residual`] on the line: how far the
    /// engine's worst note stands from the engine's own trend.
    pub standout: f64,
    pub standout_key: u8,
    /// The same statistic on the recordings' line, and on the recordings'
    /// neighbouring velocity layer; the larger of the two sets the bar.
    pub reference_standout: f64,
    pub layer_standout: f64,
    /// What [`Column::standout`] had to come in under.
    pub bar: f64,
    pub pass: bool,
    // --- reported, not gated ---------------------------------------------
    /// The largest [`LineNoteScore::seam`]: how far one note's distance from
    /// its own recording departs from the line's median distance. Item 288's
    /// S1, restricted to the melody.
    pub seam: f64,
    pub seam_key: u8,
}

impl Column {
    /// The standout as a fraction of the bar. At or under 1 it passes.
    pub fn ratio(&self) -> f64 {
        self.standout / self.bar
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
    engine: &[(u8, [f64; 3])],
    reference: &[(u8, [f64; 3])],
    layer: &[(u8, [f64; 3])],
) -> Vec<Column> {
    METRICS
        .iter()
        .enumerate()
        .map(|(m, &metric)| {
            let residuals = |rows: &[(u8, [f64; 3])]| -> Vec<f64> {
                let points: Vec<(f64, f64)> =
                    rows.iter().map(|&(k, v)| (f64::from(k), v[m])).collect();
                let (slope, intercept) = theil_sen(&points);
                points
                    .iter()
                    .map(|&(x, y)| y - (intercept + slope * x))
                    .collect()
            };
            let engine_resid = residuals(engine);
            let reference_resid = residuals(reference);
            let layer_resid = residuals(layer);
            let errors: Vec<f64> = engine
                .iter()
                .zip(reference)
                .map(|(e, r)| e.1[m] - r.1[m])
                .collect();
            let centre = median(&mut errors.clone());

            let notes: Vec<LineNoteScore> = (0..engine.len())
                .map(|i| LineNoteScore {
                    key: engine[i].0,
                    engine: engine[i].1[m],
                    reference: reference[i].1[m],
                    layer: layer[i].1[m],
                    engine_residual: engine_resid[i],
                    reference_residual: reference_resid[i],
                    error: errors[i],
                    seam: errors[i] - centre,
                })
                .collect();
            let worst = |pick: &dyn Fn(&LineNoteScore) -> f64| -> (u8, f64) {
                notes.iter().fold((0u8, 0.0f64), |best, n| {
                    let v = pick(n).abs();
                    if v > best.1 {
                        (n.key, v)
                    } else {
                        best
                    }
                })
            };
            let (standout_key, standout) = worst(&|n| n.engine_residual);
            let (seam_key, seam) = worst(&|n| n.seam);
            let reference_standout = biggest(&reference_resid);
            let layer_standout = biggest(&layer_resid);
            let bar = reference_standout.max(layer_standout) * ALLOWANCE;
            Column {
                metric,
                pass: standout <= bar,
                notes,
                standout,
                standout_key,
                reference_standout,
                layer_standout,
                bar,
                seam,
                seam_key,
            }
        })
        .collect()
}

fn biggest(values: &[f64]) -> f64 {
    values.iter().map(|v| v.abs()).fold(0.0f64, f64::max)
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
        let _ = writeln!(
            out,
            "{:<10} stands out {:5.2} at {:<3} bar {:5.2} (piano {:4.2}, layer {:4.2}, x{:.2})  {}   \
seam {:5.2} at {}",
            c.metric,
            c.standout,
            note_name(c.standout_key),
            c.bar,
            c.reference_standout,
            c.layer_standout,
            ALLOWANCE,
            if c.pass { "pass" } else { "FAIL" },
            c.seam,
            note_name(c.seam_key),
        );
        let _ = writeln!(
            out,
            "{:<10}   key    engine  (resid)  recording  (resid)    layer    error   seam",
            ""
        );
        for n in &c.notes {
            let _ = writeln!(
                out,
                "{:<10}   {:<4} {:7.2}  {:6.2}   {:8.2} {:7.2}  {:7.2}  {:7.2} {:6.2}",
                "",
                note_name(n.key),
                n.engine,
                n.engine_residual,
                n.reference,
                n.reference_residual,
                n.layer,
                n.error,
                n.seam
            );
        }
    }
    out
}

/// `C4` for 60.
pub fn note_name(key: u8) -> String {
    const NAMES: [&str; 12] = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    format!(
        "{}{}",
        NAMES[usize::from(key) % 12],
        i32::from(key) / 12 - 1
    )
}

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

    /// Five pitches, three metrics, on a rising register trend of `slope` per
    /// semitone; `bump` is added to the engine's C4 alone.
    fn lines(bump: f64, slope: f64) -> [Vec<(u8, [f64; 3])>; 3] {
        let keys = [60u8, 62, 64, 65, 67];
        let reference: Vec<(u8, [f64; 3])> = keys
            .iter()
            .map(|&k| (k, [slope * f64::from(k - 60); 3]))
            .collect();
        // The neighbouring velocity layer: the piano again, on its own line.
        let layer = reference.clone();
        // The engine is the same line 3 dB smoother — a voicing offset — plus a
        // bump on one note.
        let mut engine: Vec<(u8, [f64; 3])> = reference
            .iter()
            .map(|&(k, v)| (k, v.map(|x| x - 3.0)))
            .collect();
        engine[0].1 = engine[0].1.map(|x| x + bump);
        [engine, reference, layer]
    }

    #[test]
    fn a_uniform_offset_and_a_register_trend_are_not_a_note_standing_out() {
        let [engine, reference, layer] = lines(0.0, 0.7);
        for c in compare(&engine, &reference, &layer) {
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
        let [engine, reference, layer] = lines(4.0, 0.7);
        for c in compare(&engine, &reference, &layer) {
            assert_eq!(c.standout_key, 60, "{}", c.metric);
            assert!(
                (c.standout - 4.0).abs() < 1e-9,
                "{} {:.3}",
                c.metric,
                c.standout
            );
            assert!(!c.pass, "{} passed a 4 dB bump at a bar of {:.2}", c.metric, c.bar);
            // And the seam sees it too, since the piano did not move.
            assert_eq!(c.seam_key, 60, "{}", c.metric);
        }
    }

    #[test]
    fn the_bar_is_the_pianos_own_worst_note_and_never_smaller_than_the_layers() {
        let keys = [60u8, 62, 64, 65, 67];
        let flat: Vec<(u8, [f64; 3])> = keys.iter().map(|&k| (k, [0.0; 3])).collect();
        // A piano whose own line has a 4 dB note tolerates the engine having one.
        let mut reference = flat.clone();
        reference[3].1 = [4.0; 3];
        let mut engine = flat.clone();
        engine[0].1 = [4.0; 3];
        for c in compare(&engine, &reference, &flat) {
            assert!(
                (c.reference_standout - 4.0).abs() < 1e-9,
                "{} {:.3}",
                c.metric,
                c.reference_standout
            );
            assert!(c.pass, "{} failed at a bar of {:.2}", c.metric, c.bar);
        }
        // With a flat piano, the velocity layer is what is left.
        let mut layer = flat.clone();
        layer[2].1 = [2.0; 3];
        for c in compare(&engine, &flat, &layer) {
            assert!((c.layer_standout - 2.0).abs() < 1e-9, "{} {:.3}", c.metric, c.layer_standout);
            assert!((c.bar - 2.0 * ALLOWANCE).abs() < 1e-9, "{} {:.3}", c.metric, c.bar);
            assert!(!c.pass, "{} passed 4 dB against a 2 dB piano", c.metric);
        }
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
