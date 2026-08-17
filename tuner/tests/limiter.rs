//! The limiter budget: how far into its two nonlinearities the instrument goes
//! while it is being played normally.
//!
//! The engine has exactly two waveshapers and neither is meant to be a sound.
//! `soundboard::soft_clip` is the **master safety limiter** — bit-transparent
//! under −1 dBFS, `tanh` over it — and `DECISIONS.md` 42 states its contract as
//! a level: only the loudest thing a pianist can do reaches it, and ordinary
//! playing never does. `voice::soft_limit` is the **damper felt**, the
//! nonlinear contact of `PHYSICS.md` §6, and its contract is a *gesture*: it
//! colours a half pedal and a slow release, and an ordinary note-off is not
//! either of those.
//!
//! Both were being reported by ear before they were measured (`DECISIONS.md`
//! 262–264), and one of the two readings that started that investigation was
//! not a limiter at all: `realism::level_match`'s `PEAK_CEILING` is a **linear**
//! scale applied to both members of a pair, so a benchmark render peaking at
//! exactly 0.98 says that the *guard* engaged and says nothing whatever about
//! the engine. What the guard measures is crest, not distortion, and the tests
//! here are on the raw render for that reason.
//!
//! Neither gate needs the Salamander corpus. Both carry the number the corpus
//! gave, which is what makes them honest rather than arbitrary, and
//! `forensics/src/bin/limiter_probe.rs` is the harness that re-derives it.

use std::path::Path;

use piano_emulator::preset::Preset;
use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_emulator::soundboard::LIMIT_THRESHOLD;
use piano_emulator::types::{Event, FIRST_UNDAMPED_KEY, HIGHEST_KEY, LOWEST_KEY};
use piano_tuner::realism::{self, Phrase};
use piano_tuner::sampler::{engine_events, SamplerEvent};

/// The two shipped voicings. A budget that only holds on the preset it was
/// measured on is not a budget.
const PRESETS: [&str; 2] = ["presets/default.toml", "presets/salamander-c5.toml"];

fn preset(path: &str) -> Preset {
    Preset::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(path))
        .expect("a shipped preset loads")
}

fn db(x: f32) -> f64 {
    20.0 * f64::from(x).max(1e-30).log10()
}

fn render(preset: &Preset, phrase: &Phrase) -> (Vec<f32>, Vec<f32>) {
    render_to_buffer(
        preset,
        &engine_events::to_render_events(&phrase.events),
        phrase.duration_s as f32,
    )
}

/// Samples the master limiter shaped, and the render's peak.
///
/// No instrumentation: `soft_clip` is the identity below `LIMIT_THRESHOLD` and
/// strictly increasing above it, so a finished sample is over the threshold
/// exactly when the sample that produced it was.
fn master_limiter(l: &[f32], r: &[f32]) -> (usize, f32) {
    let over = l
        .iter()
        .chain(r.iter())
        .filter(|x| x.abs() > LIMIT_THRESHOLD)
        .count();
    let peak = l
        .iter()
        .chain(r.iter())
        .fold(0.0f32, |m, &x| m.max(x.abs()));
    (over, peak)
}

// ---------------------------------------------------------------------------
// The master safety limiter
// ---------------------------------------------------------------------------

/// `DECISIONS.md` 42's contract, on the music: normal playing never reaches the
/// safety limiter.
///
/// Measured on both shipped presets, all six benchmark phrases: **not one
/// sample** of the twelve renders is over the threshold, and the least
/// headroom any of them leaves is **6.92 dB** (`arpeggio_dynamics` on the
/// measured preset, whose fortissimo arpeggio is the loudest gesture in the
/// set; `presets/default.toml`'s worst is the same phrase at 15.23 dB). The
/// gate is a decibel of headroom rather than zero samples alone, so that a
/// change which merely *grazes* the limiter fails here instead of passing and
/// then failing on the next louder phrase somebody writes.
///
/// The 1.60 dB this comment carried until `DECISIONS.md` 343 was measured
/// before the master-gain recalibration of item 277 took the whole instrument
/// 5.19 dB down; it was a stale number under a live gate, which is the one
/// thing a doc comment beside an assertion must not be.
#[test]
fn no_benchmark_phrase_reaches_the_master_safety_limiter() {
    for path in PRESETS {
        let preset = preset(path);
        for phrase in realism::phrase_set() {
            let (l, r) = render(&preset, &phrase);
            let (over, peak) = master_limiter(&l, &r);
            let headroom = db(LIMIT_THRESHOLD) - db(peak);
            assert_eq!(
                over,
                0,
                "{path} {}: {over} samples inside the safety limiter, peak {:.2} dBFS",
                phrase.name,
                db(peak)
            );
            assert!(
                headroom > 1.0,
                "{path} {}: only {headroom:.2} dB under the safety limiter",
                phrase.name
            );
        }
    }
}

/// ... and on **both shipped presets** a single note never reaches it at any
/// velocity a player can ask for, anywhere on the keyboard.
///
/// This is the clause of `DECISIONS.md` 42 that is a statement about *one
/// note*: the limiter is for chords. Every fourth key at three velocities, the
/// loudest of them the loudest a MIDI file can carry. Over the keys this gate
/// samples the loudest is **A7 at velocity 127 on `presets/default.toml`,
/// -9.86 dBFS, and F6 on `presets/salamander-c5.toml`, -9.20** in the channel
/// it is panned into — 8.86 and 8.20 dB under the threshold. Over all 88 keys
/// (`forensics/src/bin/output_gain.rs`, which is not on the every-fourth grid)
/// it is C8 at -9.42 and A#7 at -7.52, 8.42 and 6.52 dB under, and that second
/// pair is unmoved by `DECISIONS.md` 334-341.
///
/// **`presets/salamander-c5.toml` used to be deliberately excluded, and both
/// halves of the reason are now gone.** Under the same engine three keys of the
/// fitted preset — 87, 96 and 99 — used to put a single fortissimo strike
/// inside the limiter (17, 2525 and 1487 samples, all three peaking at 0.00
/// dBFS), which item 265 attributed to `notes.partial_gains` rather than to the
/// construction. The disciplined refit (`DECISIONS.md` 273-274) took the level
/// out of those rows, and the master-gain recalibration (277) moved the whole
/// instrument 5.19 dB down from a threshold it was driving past; the fitted
/// preset's loudest key is still a treble one (A#7) but it now sits 6.5 dB
/// under the threshold instead of on it. Both presets are in the gate now,
/// which is where a claim about "the instrument" belongs.
#[test]
fn a_single_note_never_reaches_the_master_safety_limiter() {
    for path in ["presets/default.toml", "presets/salamander-c5.toml"] {
        let preset = preset(path);
        let mut loudest = (0.0f32, 0u8, 0u16);
        for key in (LOWEST_KEY..=HIGHEST_KEY).step_by(4) {
            for vel in [80, 110, 127] {
                let (l, r) = render_to_buffer(
                    &preset,
                    &[RenderEvent::new(0.0, Event::NoteOn { key, vel })],
                    0.6,
                );
                let (over, peak) = master_limiter(&l, &r);
                assert_eq!(
                    over, 0,
                    "{path}: key {key} at velocity {vel} put {over} samples inside \
                     the safety limiter"
                );
                if peak > loudest.0 {
                    loudest = (peak, key, vel);
                }
            }
        }
        let headroom = db(LIMIT_THRESHOLD) - db(loudest.0);
        println!(
            "{path}: loudest single strike is key {} at velocity {}, {:.2} dBFS in \
             its channel, {headroom:.2} dB under the threshold",
            loudest.1,
            loudest.2,
            db(loudest.0)
        );
        assert!(
            headroom > 2.0,
            "{path}: the loudest single note (key {}, velocity {}) leaves only \
             {headroom:.2} dB under the safety limiter",
            loudest.1,
            loudest.2
        );
    }
}

// ---------------------------------------------------------------------------
// The damper felt
// ---------------------------------------------------------------------------

/// Note-offs whose reading is about the damper and not about something else in
/// the window: the key has a damper at all, and nothing is struck and no pedal
/// moves while the two windows [`realism::note_off_hf`] compares are open.
fn readable_note_offs(phrase: &Phrase) -> Vec<f64> {
    let (from, to) = realism::NOTE_OFF_WINDOW_S;
    let interruptions: Vec<f64> = phrase
        .events
        .iter()
        .filter(|e| {
            matches!(e.event, SamplerEvent::NoteOn { vel, .. } if vel > 0)
                || matches!(e.event, SamplerEvent::Sustain(_))
        })
        .map(|e| e.time_s)
        .collect();
    phrase
        .events
        .iter()
        .filter_map(|e| match e.event {
            SamplerEvent::NoteOff { key, .. } if key < FIRST_UNDAMPED_KEY => Some(e.time_s),
            _ => None,
        })
        .filter(|&t| {
            !interruptions
                .iter()
                .any(|&s| s > t + realism::NOTE_OFF_REFERENCE_S.0 && s < t + to)
                && from < to
        })
        .collect()
}

/// `PHYSICS.md` §6's felt is a *correction* on the string's swing, and an
/// ordinary note-off must not hear it as a fuzz box.
///
/// The statistic is [`realism::note_off_hf`] — how much energy above 10 kHz a
/// damper's landing adds — pooled over the readable note-offs of all six
/// benchmark phrases on the preset the benchmark is voiced from. A damper takes
/// energy out of a string, so the honest expectation is a *negative* number,
/// and the Salamander recordings of the same six phrases give it: over the same
/// 73 note-offs they pool at **mean −1.16 dB, p90 +0.63, worst +1.00, and not
/// one of them over +6 dB**.
///
/// Before `DECISIONS.md` 262 the engine pooled at **mean +9.36 dB, p90 +27.08,
/// worst +35.79, with 43.8 % of note-offs over +6** — the felt's threshold dove
/// through the whole 40 dB of `FELT_CLEARANCE` inside the damper's 10 ms
/// arrival while the string lost two of them, so `soft_limit` ran 20 to 41 dB
/// past its knee and hard-clipped the note. It now pools at **mean −1.27, p90
/// +3.70, worst +11.13, 6.8 % over +6** (`DECISIONS.md` 343; before the hammer
/// refit of item 340 it was −1.56 / +3.14 / +11.14 / 4.1 %, and the whole of
/// that move is `[noise.strike]` coming down out of the *reference* half of
/// this ratio — the tail refit of items 334-336 moves it by nothing at all).
/// `presets/default.toml` is not in this gate and would not pass it: the
/// hand-tuned dampers pool at **+19.11 / +36.00 / +66.86, 89.0 % over +6**.
///
/// The two gates are the two halves of that: a damper may not add energy up
/// there *on average*, and it may not do so on more than a tenth of the notes.
/// A ceiling on the single worst reading is deliberately not one of them —
/// `excerpt` carries a +13.4 dB cell that this milestone did not move by a
/// hundredth of a decibel, so it is not the felt and a gate that included it
/// would be measuring something else.
#[test]
fn a_damper_landing_does_not_add_high_frequency_energy() {
    let preset = preset("presets/salamander-c5.toml");
    let mut pooled: Vec<f64> = Vec::new();
    for phrase in realism::phrase_set() {
        let (l, r) = render(&preset, &phrase);
        let mono: Vec<f32> = l.iter().zip(&r).map(|(a, b)| a + b).collect();
        pooled.extend(realism::note_off_hf(
            &mono,
            f64::from(piano_tuner::SAMPLE_RATE),
            &readable_note_offs(&phrase),
        ));
    }
    assert!(
        pooled.len() > 60,
        "only {} readable note-offs in the phrase set",
        pooled.len()
    );
    let mean = pooled.iter().sum::<f64>() / pooled.len() as f64;
    let loud = pooled.iter().filter(|&&x| x > 6.0).count();
    let share = 100.0 * loud as f64 / pooled.len() as f64;
    assert!(
        mean <= 0.0,
        "a damper landing adds {mean:+.2} dB above 10 kHz on average over {} note-offs; \
         the recordings of the same phrases give -1.16",
        pooled.len()
    );
    assert!(
        share <= 10.0,
        "{loud} of {} note-offs ({share:.1} %) add more than 6 dB above 10 kHz; \
         the recordings of the same phrases have none",
        pooled.len()
    );
}
