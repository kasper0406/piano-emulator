//! `DECISIONS.md` 42's calibration, run as a procedure rather than quoted as a
//! claim: the four clauses measured through the finished chain, and the factor
//! that would put the one clause that *is* the calibration — the loudest thing a
//! pianist can do, ten notes at fortissimo struck exactly together — on the
//! safety limiter's threshold.
//!
//! The other three clauses (mezzo-forte C4, a single fortissimo note, the
//! loudest single strike anywhere on the instrument) are **consequences** of
//! that one number and are reported, not solved for. Item 42 set the anchor and
//! wrote the consequences down; this re-runs both halves.
//!
//! **Why the peak has to be read per channel.** `soundboard::soft_clip` is
//! applied to each channel independently, so `LIMIT_THRESHOLD` is a per-channel
//! number. The mono sum of a centred note is 3 dB over its channel peak, so a
//! mono reading compared against the threshold is 3 dB pessimistic before
//! anything is wrong — and once the limiter saturates, the mono sum of two
//! channels both pinned near 1.0 reads +6.02 dBFS *whatever* the drive was, which
//! is what `DECISIONS.md` 266 read as "6 dB past the threshold". Past the
//! threshold it was, but that number is the saturation and not the size of the
//! overshoot.
//!
//! **The procedure is iterative and it converges in one step**, because
//! everything between the voices and `soft_clip` is linear. While the chord
//! clips, the render cannot say how far over it is; so lower `OUTPUT_GAIN` until
//! it does not clip, at which point the render *is* the pre-limiter signal and
//! the exact answer is `OUTPUT_GAIN * LIMIT_THRESHOLD / peak`.
//!
//! ```text
//! cargo run --release -p forensics --bin output_gain -- [preset.toml]
//! ```

use std::path::PathBuf;

use piano_emulator::preset::Preset;
use piano_emulator::render::{demo_sequence, render_to_buffer, RenderEvent, DEMO_DURATION_S};
use piano_emulator::soundboard::LIMIT_THRESHOLD;
use piano_emulator::types::{Event, PedalEvent, OUTPUT_GAIN};

fn db(x: f32) -> f32 {
    20.0 * x.max(1e-30).log10()
}

/// (mono-sum peak, per-channel peak, samples the safety limiter shaped)
fn strike(preset: &Preset, keys: &[u8], vel: u8, seconds: f32) -> (f32, f32, usize) {
    let events: Vec<RenderEvent> = keys
        .iter()
        .map(|&key| RenderEvent::new(0.0, Event::NoteOn { key, vel }))
        .collect();
    let (l, r) = render_to_buffer(preset, &events, seconds);
    let mono = l
        .iter()
        .zip(&r)
        .fold(0.0f32, |m, (&a, &b)| m.max((a + b).abs()));
    let chan = l.iter().chain(r.iter()).fold(0.0f32, |m, &x| m.max(x.abs()));
    let over = l
        .iter()
        .chain(r.iter())
        .filter(|x| x.abs() > LIMIT_THRESHOLD)
        .count();
    (mono, chan, over)
}

fn report(name: &str, (mono, chan, over): (f32, f32, usize)) {
    println!(
        "  {name:<34} mono {:7.2} dBFS   channel {:7.2}   headroom {:6.2} dB   limiter samples {over}",
        db(mono),
        db(chan),
        db(LIMIT_THRESHOLD) - db(chan),
    );
}

/// The ten notes of `DECISIONS.md` 42's chord: two hands, ff, struck together.
const CHORD: [u8; 10] = [36, 43, 48, 52, 55, 60, 64, 67, 72, 76];

/// Thirty seconds of dense pseudo-random playing with the pedal pumped — the
/// same sequence `acceptance::thirty_seconds_of_dense_playing_stays_safe`
/// builds, so item 42's last two numbers come off the gate's own material.
fn dense_playing() -> Vec<RenderEvent> {
    let mut events = vec![RenderEvent::new(
        0.0,
        Event::Pedal(PedalEvent::Sustain(1.0)),
    )];
    let mut state = 0x9e37_79b9u32;
    for i in 0..1200 {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let key = 21 + (state >> 16) as u8 % 88;
        let vel = 20 + (state >> 8) as u8 % 107;
        let t = i as f32 * 0.024;
        events.push(RenderEvent::new(t, Event::NoteOn { key, vel }));
        events.push(RenderEvent::new(t + 0.35, Event::NoteOff { key, vel: 64 }));
        if i % 100 == 0 {
            let pedal = if (i / 100) % 2 == 0 { 1.0 } else { 0.0 };
            events.push(RenderEvent::new(t, Event::Pedal(PedalEvent::Sustain(pedal))));
        }
    }
    events
}

/// Peak, limiter count and worst channel DC of a rendered passage.
fn passage(preset: &Preset, events: &[RenderEvent], seconds: f32) -> (f32, usize, f32) {
    let (l, r) = render_to_buffer(preset, events, seconds);
    let peak = l.iter().chain(r.iter()).fold(0.0f32, |m, &x| m.max(x.abs()));
    let over = l
        .iter()
        .chain(r.iter())
        .filter(|x| x.abs() > LIMIT_THRESHOLD)
        .count();
    let dc = [&l, &r]
        .into_iter()
        .map(|c| (c.iter().sum::<f32>() / c.len() as f32).abs())
        .fold(0.0f32, f32::max);
    (peak, over, dc)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let preset_path = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "presets/default.toml".into()),
    );
    let preset = Preset::load(&preset_path)?;
    println!(
        "DECISIONS 42 on {} at OUTPUT_GAIN {OUTPUT_GAIN}  \
         (limiter threshold {:.2} dBFS, per channel)",
        preset_path.display(),
        db(LIMIT_THRESHOLD)
    );

    let mf = strike(&preset, &[60], 80, 0.6);
    report("(1) mezzo-forte C4 (vel 80)", mf);
    let ff = strike(&preset, &[60], 127, 0.6);
    report("(2) single fortissimo C4 (vel 127)", ff);

    // (3) The loudest single strike anywhere on the instrument, found rather
    // than assumed: item 266 names C8, which is a claim about this preset.
    let mut loudest = (0u8, (0.0f32, 0.0f32, 0usize));
    for key in 21u8..=108 {
        let s = strike(&preset, &[key], 127, 0.6);
        if s.1 > loudest.1 .1 {
            loudest = (key, s);
        }
    }
    report(
        &format!("(3) loudest single strike (key {})", loudest.0),
        loudest.1,
    );

    let chord = strike(&preset, &CHORD, 127, 0.6);
    report("(4) ten-note ff chord", chord);

    // The two passages item 42 also reports, on the same chain.
    let (peak, over, _) = passage(&preset, &demo_sequence(), DEMO_DURATION_S);
    println!(
        "  {:<34} peak {:7.2} dBFS   limiter samples {over}",
        "(5) the built-in demo", db(peak)
    );
    let (peak, over, dc) = passage(&preset, &dense_playing(), 30.0);
    println!(
        "  {:<34} peak {:7.2} dBFS   limiter samples {over}   worst channel DC {:7.1} dBFS",
        "(6) 30 s of dense random playing",
        db(peak),
        db(dc)
    );

    println!();
    if chord.2 == 0 {
        let want = OUTPUT_GAIN * LIMIT_THRESHOLD / chord.1;
        println!(
            "the chord does not clip, so the render is the pre-limiter signal and the \
             calibration is exact:\n  OUTPUT_GAIN {OUTPUT_GAIN} -> {want:.4}  \
             ({:+.2} dB); every level in this report moves with it",
            db(want / OUTPUT_GAIN)
        );
    } else {
        println!(
            "the chord is inside the limiter ({} samples), so its peak is the \
             saturation and not the drive.\nLower OUTPUT_GAIN until it clears and \
             re-run: the answer is then exact in one step.",
            chord.2
        );
    }
    Ok(())
}
