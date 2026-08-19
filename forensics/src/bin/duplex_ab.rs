//! The duplex milestone's own evidence: what the segments cost the renders that
//! were not supposed to move, and what they add to the ones that were
//! (`DECISIONS.md` 481-483).
//!
//! Three jobs in one instrument, because they need the same two renders:
//!
//! * **the pins** — the Ode melody and the recorded-key ladder, mono, hashed,
//!   with `notes.duplex` stripped. `DECISIONS.md` 453-457's C4 repair lives in
//!   those renders and 481 must not move it, so the honest statement is a hash
//!   of the same phrase with the segments taken out: identical means the repair
//!   did not move, and anything the segments *do* move is then visibly theirs.
//! * **the A/B** — a treble phrase rendered twice, segments in and segments
//!   out, written to `renders/duplex/` for a listener, with the level of the
//!   difference between them printed.
//! * **the census** — per key, how much energy the segments add to the note's
//!   own render and to its tail, so that "the segments are audible now" is a
//!   number rather than a claim.
//!
//! ```sh
//! cargo run --release -p forensics --bin duplex_ab -- presets/salamander-c5.toml
//! cargo run --release -p forensics --bin duplex_ab -- presets/salamander-c5.toml --write
//! ```

use std::path::PathBuf;

use piano_emulator::preset::Preset;
use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_emulator::types::{Event, SAMPLE_RATE};

/// The nine keys of the recorded ladder, D#3 to D#5 (`DECISIONS.md` 457).
const LADDER: [u8; 9] = [51, 54, 57, 60, 63, 66, 69, 72, 75];
/// The Ode line's own pitches and velocity (`piano_tuner::realism`).
const ODE: [(f64, u8, f64); 16] = [
    (0.0, 64, 1.0),
    (1.0, 64, 1.0),
    (2.0, 65, 1.0),
    (3.0, 67, 1.0),
    (4.0, 67, 1.0),
    (5.0, 65, 1.0),
    (6.0, 64, 1.0),
    (7.0, 62, 1.0),
    (8.0, 60, 1.0),
    (9.0, 60, 1.0),
    (10.0, 62, 1.0),
    (11.0, 64, 1.0),
    (12.0, 64, 1.5),
    (13.5, 62, 0.5),
    (14.0, 62, 2.0),
    (16.0, 60, 2.0),
];
const ODE_VEL: u16 = 88;
const BEAT_S: f64 = 0.5;
/// The treble phrase the A/B is judged on: a rising arpeggio in the top two
/// octaves, staccato, where `PHYSICS.md` §3 predicts the shimmer lives.
const TREBLE_PHRASE: [(f64, u8); 12] = [
    (0.00, 72),
    (0.18, 76),
    (0.36, 79),
    (0.54, 84),
    (0.72, 88),
    (0.90, 91),
    (1.20, 96),
    (1.50, 91),
    (1.68, 88),
    (1.86, 84),
    (2.04, 79),
    (2.22, 72),
];

fn fingerprint(left: &[f32], right: &[f32]) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for value in left.iter().chain(right.iter()) {
        for byte in value.to_bits().to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x1000_0000_01b3);
        }
    }
    format!("{hash:016x}")
}

fn rms(x: &[f32]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    (x.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>() / x.len() as f64).sqrt()
}

fn db(x: f64) -> f64 {
    20.0 * x.max(1.0e-30).log10()
}

fn without_duplex(preset: &Preset) -> Preset {
    let mut bare = preset.clone();
    bare.notes.duplex = Vec::new();
    bare
}

fn mono(l: &[f32], r: &[f32]) -> Vec<f32> {
    l.iter().zip(r).map(|(&a, &b)| 0.5 * (a + b)).collect()
}

fn ode_events() -> Vec<RenderEvent> {
    let mut events = Vec::new();
    for (at, key, len) in ODE {
        let t = (at * BEAT_S) as f32;
        events.push(RenderEvent::new(t, Event::NoteOn { key, vel: ODE_VEL }));
        events.push(RenderEvent::new(
            t + (len * BEAT_S - 0.05).max(0.08) as f32,
            Event::NoteOff { key, vel: 64 },
        ));
    }
    events
}

fn ladder_events() -> Vec<RenderEvent> {
    let mut events = Vec::new();
    for (i, key) in LADDER.into_iter().enumerate() {
        let t = 0.6 * i as f32;
        events.push(RenderEvent::new(t, Event::NoteOn { key, vel: 90 }));
        events.push(RenderEvent::new(t + 0.4, Event::NoteOff { key, vel: 64 }));
    }
    events
}

fn treble_events() -> Vec<RenderEvent> {
    let mut events = Vec::new();
    for (at, key) in TREBLE_PHRASE {
        events.push(RenderEvent::new(at as f32, Event::NoteOn { key, vel: 96 }));
        events.push(RenderEvent::new(
            at as f32 + 0.12,
            Event::NoteOff { key, vel: 64 },
        ));
    }
    events
}

fn write_wav(path: &std::path::Path, l: &[f32], r: &[f32]) -> std::io::Result<()> {
    use std::io::Write;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let frames = l.len().min(r.len());
    let data_len = (frames * 4) as u32;
    let mut out = Vec::with_capacity(44 + data_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&(SAMPLE_RATE as u32).to_le_bytes());
    out.extend_from_slice(&((SAMPLE_RATE as u32) * 4).to_le_bytes());
    out.extend_from_slice(&4u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for i in 0..frames {
        for &v in &[l[i], r[i]] {
            let s = (v.clamp(-1.0, 1.0) * 32_767.0).round() as i16;
            out.extend_from_slice(&s.to_le_bytes());
        }
    }
    std::fs::File::create(path)?.write_all(&out)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "presets/salamander-c5.toml".into()),
    );
    let write = std::env::args().any(|a| a == "--write");
    let preset = Preset::from_toml(&std::fs::read_to_string(&path)?)?;
    let bare = without_duplex(&preset);
    println!(
        "{}: {} keys carry segments, {} segments in all",
        path.display(),
        preset.notes.duplex.iter().filter(|r| !r.is_empty()).count(),
        preset.notes.duplex.iter().flatten().count()
    );

    // ------------------------------------------------------------- the pins
    println!("\nthe renders D453-457's C4 repair lives in, with the segments stripped:");
    for (name, events, seconds) in [
        ("melody", ode_events(), 12.0f32),
        ("ladder", ladder_events(), 8.0),
    ] {
        let (l, r) = render_to_buffer(&bare, &events, seconds);
        let m = mono(&l, &r);
        println!(
            "  {name:<7} mono {} (stereo {}), rms {:.2} dBFS",
            fingerprint(&m, &[]),
            fingerprint(&l, &r),
            db(rms(&m))
        );
    }
    println!("\n... and the same renders with the segments in:");
    for (name, events, seconds) in [
        ("melody", ode_events(), 12.0f32),
        ("ladder", ladder_events(), 8.0),
    ] {
        let (l, r) = render_to_buffer(&preset, &events, seconds);
        let (bl, br) = render_to_buffer(&bare, &events, seconds);
        let m = mono(&l, &r);
        let bm = mono(&bl, &br);
        let diff: Vec<f32> = m.iter().zip(&bm).map(|(&a, &b)| a - b).collect();
        println!(
            "  {name:<7} mono {}, rms {:.2} dBFS, the segments' own share {:+.2} dB",
            fingerprint(&m, &[]),
            db(rms(&m)),
            db(rms(&diff)) - db(rms(&bm))
        );
    }

    // ------------------------------------------------------------ the census
    println!("\nper key: what the segments add, head (0-0.3 s) and tail (1-3 s), dB re the note");
    println!("  key   n   segments Hz            head      tail");
    for (i, row) in preset.notes.duplex.iter().enumerate() {
        if row.is_empty() {
            continue;
        }
        let key = 21 + i as u8;
        let events = [
            RenderEvent::new(0.05, Event::NoteOn { key, vel: 90 }),
            RenderEvent::new(0.35, Event::NoteOff { key, vel: 64 }),
        ];
        let (l, r) = render_to_buffer(&preset, &events, 3.5);
        let (bl, br) = render_to_buffer(&bare, &events, 3.5);
        let m = mono(&l, &r);
        let bm = mono(&bl, &br);
        let diff: Vec<f32> = m.iter().zip(&bm).map(|(&a, &b)| a - b).collect();
        let at = |from: f64, to: f64| {
            let a = (from * f64::from(SAMPLE_RATE)) as usize;
            let b = ((to * f64::from(SAMPLE_RATE)) as usize).min(bm.len());
            (
                db(rms(&diff[a.min(b)..b])) - db(rms(&bm[a.min(b)..b])),
                db(rms(&bm[a.min(b)..b])),
            )
        };
        let (head, _) = at(0.05, 0.35);
        let (tail, tail_level) = at(1.0, 3.0);
        println!(
            "  {key:>3} {:>3}   {:<20}  {head:>+7.2}  {tail:>+8.2}  (tail at {tail_level:.1} dBFS)",
            row.len(),
            row.iter()
                .take(2)
                .map(|m| format!("{:.0}", m.hz))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }

    // -------------------------------------------------- the halo's own targets
    // `docs/history/TUNING_REPORT.md` §4's between-partial census, which is
    // where the treble sympathetic halo's ~21 dB shortfall is read. The
    // segments live in the same band, so the honest question is whether they
    // narrow it; this measures the same quantity the halo fit closes on, with
    // the segments in and out.
    println!("\nthe between-partial census the treble halo shortfall is read in:");
    println!("  target          key    recording    duplex off     duplex on    narrowed by");
    let survey = piano_tuner::survey::SurveyConfig::default();
    let tuner_preset = piano_tuner::preset::Preset::from_toml(&std::fs::read_to_string(&path)?)
        .map_err(|e| e.to_string())?;
    for target in piano_tuner::estimate::halo::salamander_targets() {
        if target.name.starts_with("harm") {
            continue;
        }
        let index = (target.key - 21) as usize;
        let f0 = f64::from(tuner_preset.notes.f0_hz[index]);
        let Ok(note_config) = survey.note_config(f0) else {
            continue;
        };
        let level = |p: &Preset| -> f64 {
            let (l, r) = render_to_buffer(
                p,
                &[RenderEvent::new(0.0, Event::NoteOn { key: target.key, vel: 90 })],
                5.0,
            );
            piano_tuner::estimate::halo::between_partials(
                &mono(&l, &r),
                f64::from(SAMPLE_RATE),
                f0,
                &note_config,
                &piano_tuner::estimate::halo::HaloConfig::default(),
            )
            .map(|b| b.at_late_db)
            .unwrap_or(f64::NAN)
        };
        let off = level(&bare);
        let on = level(&preset);
        println!(
            "  {:<14} {:>3}   {:>+9.2}    {:>+10.2}    {:>+10.2}    {:>+11.2}",
            target.name,
            target.key,
            target.target_db,
            off,
            on,
            (target.target_db - off).abs() - (target.target_db - on).abs()
        );
    }

    // --------------------------------------------------------------- the A/B
    let events = treble_events();
    let (l, r) = render_to_buffer(&preset, &events, 6.0);
    let (bl, br) = render_to_buffer(&bare, &events, 6.0);
    let m = mono(&l, &r);
    let bm = mono(&bl, &br);
    let diff: Vec<f32> = m.iter().zip(&bm).map(|(&a, &b)| a - b).collect();
    let tail = (3.0 * f64::from(SAMPLE_RATE)) as usize;
    println!(
        "\ntreble phrase (12 staccato notes, C5 to C8): the segments are {:+.2} dB of the whole \
         phrase and {:+.2} dB of what is left after 3 s",
        db(rms(&diff)) - db(rms(&bm)),
        db(rms(&diff[tail.min(diff.len())..])) - db(rms(&bm[tail.min(bm.len())..]))
    );
    if write {
        write_wav(std::path::Path::new("renders/duplex/treble-duplex-on.wav"), &l, &r)?;
        write_wav(std::path::Path::new("renders/duplex/treble-duplex-off.wav"), &bl, &br)?;
        let dl: Vec<f32> = l.iter().zip(&bl).map(|(&a, &b)| a - b).collect();
        let dr: Vec<f32> = r.iter().zip(&br).map(|(&a, &b)| a - b).collect();
        write_wav(
            std::path::Path::new("renders/duplex/treble-segments-only.wav"),
            &dl,
            &dr,
        )?;
        println!("wrote renders/duplex/treble-duplex-{{on,off}}.wav and -segments-only.wav");
    }
    Ok(())
}
