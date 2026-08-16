//! The limiter budget, measured: what `tuner/tests/limiter.rs` asserts and where
//! its numbers come from (`DECISIONS.md` 262-265).
//!
//! Renders the six benchmark phrases through the engine and writes the **raw**
//! master output — before `realism::level_match` touches it, because that
//! function's `PEAK_CEILING` is a linear scale on both members of a pair and a
//! render peaking at exactly 0.98 is that guard and not a limiter. For each
//! phrase it prints the peak, the headroom under `soundboard::LIMIT_THRESHOLD`,
//! how many samples the safety limiter shaped, and `realism::note_off_hf` over
//! the note-offs nothing else lands in, beside the same statistic taken on
//! `renders/realism/<phrase>_reference.wav` if the benchmark has been run. Then
//! it reads `DECISIONS.md` 42's calibration anchors off the same chain.
//!
//! ```text
//! cargo run --release -p forensics --bin limiter_probe -- <out-dir> [preset.toml]
//! ```

use std::path::{Path, PathBuf};

use piano_emulator::preset::Preset;
use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_emulator::soundboard::LIMIT_THRESHOLD;
use piano_emulator::types::{Event, FIRST_UNDAMPED_KEY};
use piano_tuner::audio;
use piano_tuner::realism;
use piano_tuner::sampler::{engine_events, SamplerEvent, TimedEvent};

/// Note-off times of the keys that have a damper — the only ones the felt can
/// reach.
fn damped_note_offs(events: &[TimedEvent]) -> Vec<f64> {
    let (from, to) = realism::NOTE_OFF_WINDOW_S;
    let strikes: Vec<f64> = events
        .iter()
        .filter(|e| {
            matches!(e.event, SamplerEvent::NoteOn { vel, .. } if vel > 0)
                || matches!(e.event, SamplerEvent::Sustain(_))
        })
        .map(|e| e.time_s)
        .collect();
    events
        .iter()
        .filter_map(|e| match e.event {
            SamplerEvent::NoteOff { key, .. } if key < FIRST_UNDAMPED_KEY => Some(e.time_s),
            _ => None,
        })
        // A window with a strike in it is measuring the strike.
        .filter(|&t| !strikes.iter().any(|&s| s > t + from - 0.010 && s < t + to))
        .collect()
}

fn summary(v: &[f64]) -> (f64, f64) {
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    let worst = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (mean, worst)
}


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
    let mono = l.iter().zip(&r).fold(0.0f32, |m, (&a, &b)| m.max((a + b).abs()));
    let chan = l.iter().chain(r.iter()).fold(0.0f32, |m, &x| m.max(x.abs()));
    let over = l.iter().chain(r.iter()).filter(|x| x.abs() > LIMIT_THRESHOLD).count();
    (mono, chan, over)
}

fn report(name: &str, contract: f32, (mono, chan, over): (f32, f32, usize)) {
    println!(
        "  {name:<32} mono {:7.2} dBFS (contract {contract:.1})  channel {:7.2}  limiter samples {over}",
        db(mono),
        db(chan)
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let out = PathBuf::from(args.next().unwrap_or_else(|| "/tmp/limiter_probe".into()));
    let preset_path =
        PathBuf::from(args.next().unwrap_or_else(|| "presets/salamander-c5.toml".into()));
    std::fs::create_dir_all(&out)?;
    let preset = Preset::load(&preset_path)?;
    let mut pooled: Vec<f64> = Vec::new();
    let mut pooled_ref: Vec<f64> = Vec::new();

    for phrase in realism::phrase_set() {
        let (l, r) = render_to_buffer(
            &preset,
            &engine_events::to_render_events(&phrase.events),
            phrase.duration_s as f32,
        );
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 48_000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let path = out.join(format!("{}_raw.wav", phrase.name));
        let mut w = hound::WavWriter::create(&path, spec)?;
        for i in 0..l.len() {
            w.write_sample(l[i])?;
            w.write_sample(r[i])?;
        }
        w.finalize()?;
        let peak = l.iter().chain(r.iter()).fold(0.0f32, |m, &x| m.max(x.abs()));
        let over = l
            .iter()
            .chain(r.iter())
            .filter(|x| x.abs() > LIMIT_THRESHOLD)
            .count();
        let mono: Vec<f32> = l.iter().zip(&r).map(|(a, b)| a + b).collect();
        let offs = damped_note_offs(&phrase.events);
        let readings = realism::note_off_hf(&mono, 48_000.0, &offs);
        let (mean, worst) = summary(&readings);
        pooled.extend(readings);
        let reference = audio::load_at(
            Path::new("renders/realism").join(format!("{}_reference.wav", phrase.name)),
            48_000,
        )
        .ok()
        .map(|a| {
            let r = realism::note_off_hf(&a.mono(), 48_000.0, &offs);
            pooled_ref.extend(r.iter().copied());
            summary(&r)
        });
        println!(
            "{:<18} peak {:.5} ({:6.2} dBFS)  headroom {:5.2} dB  limiter samples {}  \
             note-off HF jump mean {:+6.1} worst {:+6.1}  (reference {})",
            phrase.name,
            peak,
            db(peak),
            db(LIMIT_THRESHOLD) - db(peak),
            over,
            mean,
            worst,
            reference
                .map(|(m, w)| format!("mean {m:+6.1} worst {w:+6.1}"))
                .unwrap_or_else(|| "not rendered".into())
        );
    }

    let show = |name: &str, v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        println!(
            "  {name:<10} n={:3}  mean {mean:+6.2}  p90 {:+6.2}  worst {:+6.2}  over +6 dB: {} ({:.1}%)",
            v.len(),
            v[(v.len() as f64 * 0.9) as usize],
            v[v.len() - 1],
            v.iter().filter(|&&x| x > 6.0).count(),
            100.0 * v.iter().filter(|&&x| x > 6.0).count() as f64 / v.len() as f64
        );
    };
    println!("\npooled note-off HF jump over all six phrases");
    show("engine", &mut pooled);
    show("recording", &mut pooled_ref);

    println!("\nDECISIONS 42 anchors on {}", preset_path.display());
    report("mezzo-forte C4 (vel 80)", -19.5, strike(&preset, &[60], 80, 0.6));
    report("single fortissimo C4 (vel 127)", -10.0, strike(&preset, &[60], 127, 0.6));
    report("single forte C4 (vel 95)", -10.0, strike(&preset, &[60], 95, 0.6));
    let chord: Vec<u8> = [36, 43, 48, 52, 55, 60, 64, 67, 72, 76].to_vec();
    report("ten-note ff chord", -1.0, strike(&preset, &chord, 127, 0.6));
    Ok(())
}
