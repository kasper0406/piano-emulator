//! Independent-audit renderer: renders the audio the verification measures.
//! Not part of the milestone; writes only into the target/ scratch directory.
//!
//! ```text
//! cargo run --release -p forensics --bin audit_render -- <out_dir> [stage]
//! ```

use std::path::{Path, PathBuf};

use piano_emulator::preset::{Preset, StrikeDirection};
use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_emulator::types::Event;

const SR: u32 = 48_000;
const START: f32 = 0.05;

fn write_wav(path: &Path, l: &[f32], r: &[f32]) {
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: SR,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut w = hound::WavWriter::create(path, spec).expect("wav create");
    for (&a, &b) in l.iter().zip(r) {
        w.write_sample(a).unwrap();
        w.write_sample(b).unwrap();
    }
    w.finalize().unwrap();
}

fn render_held(preset: &Preset, key: u8, vel: u8, dur: f32) -> (Vec<f32>, Vec<f32>) {
    let events = vec![RenderEvent::new(START, Event::NoteOn { key, vel })];
    render_to_buffer(preset, &events, dur)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let out = PathBuf::from(args.next().expect("out dir"));
    let stage = args.next().unwrap_or_else(|| "all".into());
    std::fs::create_dir_all(&out).unwrap();

    let fitted = Preset::load(Path::new("presets/salamander-c5.toml")).expect("fitted preset");
    let default = Preset::load(Path::new("presets/default.toml")).expect("default preset");

    if stage == "all" || stage == "jitter" {
        for &key in &[45u8, 60, 69, 84] {
            for &vel in &[40u8, 90, 120] {
                let (l, r) = render_held(&fitted, key, vel, 12.2);
                write_wav(&out.join(format!("fit_k{key:03}_v{vel:03}.wav")), &l, &r);
            }
        }
        println!("jitter renders done");
    }

    if stage == "all" || stage == "dom" {
        for key in (21u8..=108).step_by(3) {
            let (l, r) = render_held(&fitted, key, 90, 2.0);
            write_wav(&out.join(format!("dom_k{key:03}.wav")), &l, &r);
        }
        println!("dominant-partial renders done");
    }

    if stage == "all" || stage == "default" {
        for &key in &[21u8, 33, 45, 57, 60, 69, 81, 93, 105] {
            let (l, r) = render_held(&default, key, 90, 16.0);
            write_wav(&out.join(format!("def_k{key:03}.wav")), &l, &r);
        }
        println!("default renders done");
    }

    if stage == "all" || stage == "params" {
        // The preset's own asked-for frequencies and per-partial decay anchors,
        // for both presets, as JSON on stdout lines starting with PARAMS.
        for (name, preset) in [("fitted", &fitted), ("default", &default)] {
            for key in (21u8..=108).step_by(3) {
                let p = preset.string_params(key);
                let freqs: Vec<f32> = (1..=10).map(|k| p.partial_freq(k)).collect();
                let sigmas: Vec<f32> = (1..=6).map(|k| p.partial_sigma(k)).collect();
                println!(
                    "PARAMS {{\"preset\":\"{name}\",\"key\":{key},\"freq\":{freqs:?},\"sigma\":{sigmas:?}}}"
                );
            }
        }
    }

    if stage == "all" || stage == "det" {
        // Two fully independent constructions of the same render.
        let pa = Preset::load(Path::new("presets/salamander-c5.toml")).unwrap();
        let pb = Preset::load(Path::new("presets/salamander-c5.toml")).unwrap();
        let (la, ra) = render_held(&pa, 60, 90, 4.5);
        let (lb, rb) = render_held(&pb, 60, 90, 4.5);
        let same = la
            .iter()
            .zip(&lb)
            .chain(ra.iter().zip(&rb))
            .all(|(a, b)| a.to_bits() == b.to_bits());
        println!("DETERMINISM bit_identical={same}");
        write_wav(&out.join("det_a.wav"), &la, &ra);
        write_wav(&out.join("det_b.wav"), &lb, &rb);
    }

    if stage == "all" || stage == "clicks" {
        // Staccato: 80 ms notes with real releases across the compass.
        let keys = [48u8, 60, 72, 84, 55, 67, 79, 91];
        let mut events = Vec::new();
        let mut offs = Vec::new();
        for (i, &key) in keys.iter().enumerate() {
            let at = 0.25 + i as f32 * 0.5;
            events.push(RenderEvent::new(at, Event::NoteOn { key, vel: 100 }));
            events.push(RenderEvent::new(at + 0.08, Event::NoteOff { key, vel: 64 }));
            offs.push(at + 0.08);
        }
        let (l, r) = render_to_buffer(&fitted, &events, 6.0);
        write_wav(&out.join("clicks.wav"), &l, &r);
        println!("CLICK_OFFS {offs:?}");
    }

    if stage == "all" || stage == "fuzz" {
        // Boundary presets: every new-mechanism field at a schema rail.
        let mut b1 = fitted.clone();
        for row in &mut b1.notes.false_beat {
            for fb in row.iter_mut() {
                fb.hz = 3.0;
                fb.db = 0.0;
            }
        }
        b1.voicing.strike_direction = Some(StrikeDirection {
            vh_db_at_pp: 12.0,
            vh_db_at_ff: -12.0,
            share_tilt: 0.2,
        });
        if let Some(bridge) = &mut b1.voicing.bridge {
            bridge.radiated_share = 0.9;
        }
        b1.voicing.horizontal_decay_ratio = 0.95;

        let mut b2 = fitted.clone();
        for row in &mut b2.notes.false_beat {
            for fb in row.iter_mut() {
                fb.hz = 0.2;
                fb.db = -40.0;
            }
        }
        b2.voicing.strike_direction = Some(StrikeDirection {
            vh_db_at_pp: -12.0,
            vh_db_at_ff: 12.0,
            share_tilt: -0.2,
        });
        if let Some(bridge) = &mut b2.voicing.bridge {
            bridge.radiated_share = 0.0;
        }
        b2.voicing.horizontal_decay_ratio = 0.01;

        for (name, preset) in [("b1", &b1), ("b2", &b2)] {
            match preset.validate() {
                Ok(()) => println!("FUZZ {name} validates"),
                Err(e) => println!("FUZZ {name} INVALID: {e}"),
            }
            let events = vec![
                RenderEvent::new(START, Event::NoteOn { key: 21, vel: 127 }),
                RenderEvent::new(START, Event::NoteOn { key: 60, vel: 127 }),
                RenderEvent::new(START, Event::NoteOn { key: 64, vel: 127 }),
                RenderEvent::new(START, Event::NoteOn { key: 67, vel: 127 }),
                RenderEvent::new(START, Event::NoteOn { key: 96, vel: 127 }),
            ];
            let (l, r) = render_to_buffer(preset, &events, 60.0);
            let finite = l.iter().chain(r.iter()).all(|x| x.is_finite());
            let peak = l
                .iter()
                .chain(r.iter())
                .fold(0.0f32, |m, &x| m.max(x.abs()));
            let tail_start = (59.0 * SR as f32) as usize;
            let tail_peak = l[tail_start..]
                .iter()
                .chain(r[tail_start..].iter())
                .fold(0.0f32, |m, &x| m.max(x.abs()));
            println!("FUZZ {name} finite={finite} peak={peak:.6} tail_peak={tail_peak:.8}");
            write_wav(&out.join(format!("fuzz_{name}.wav")), &l, &r);
        }
    }
}
