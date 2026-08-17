//! Which of the engine's two mechanism events the melody render's noise is —
//! the strike on every note-on or the key-off on every note-off — and how far
//! the sampler's own lead-in displaces a window placed on the reference's onset.
//!
//! The instrument behind `DECISIONS.md` 338-339. Two questions, one binary,
//! because both are answered off the same renders.
//!
//! **1. Which event.** The Ode soprano line, rendered through the engine four
//! ways — as shipped, with `[noise.strike]` silenced, with `[noise.key_off]`
//! silenced, with both — and through the recordings. Both events are additive
//! in `Voice::process`, so the sample-wise difference of two of these renders
//! *is* that event through the board, and its level against the music is exact
//! rather than inferred. The release step at every note-off — the level over the
//! 60 ms after it against the 60 ms before — is what a key-off thump moves and
//! what a listener would hear it in.
//!
//! **2. The lead-in.** The sampler plays each recording from the file's own
//! start, so the audible strike is late by however much silence there was
//! between the engineer's trigger and the hammer. `estimate::melody` has always
//! searched for each side's own strike; `realism`'s attack column did not, and
//! read the engine at the *reference's* onset. This measures the displacement
//! that caused, at the recorded keys, at five velocities.
//!
//! ```text
//! cargo run --release -p forensics --bin line_noise -- [preset]
//! ```

use piano_emulator::preset::{NoiseAnchor, Preset, SILENT_LEVEL_DB};
use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_emulator::types::Event;
use piano_tuner::estimate::brilliance::band;
use piano_tuner::estimate::melody;
use piano_tuner::realism::RecordedKeys;
use piano_tuner::sampler::engine_events;
use piano_tuner::{SampleLibrary, Sampler, TimedEvent, SAMPLE_RATE};

use rustfft::num_complex::Complex32;
use rustfft::FftPlanner;

const SFZ: &str = "data/salamander/SalamanderGrandPiano-V3+20200602.sfz";
const VELOCITIES: [u8; 5] = [24, 48, 72, 88, 110];

fn mono(channels: &[Vec<f32>]) -> Vec<f32> {
    (0..channels[0].len())
        .map(|i| channels.iter().map(|c| c[i]).sum::<f32>() / channels.len() as f32)
        .collect()
}

fn silent() -> Vec<NoiseAnchor> {
    vec![NoiseAnchor {
        key: 21,
        db: SILENT_LEVEL_DB,
    }]
}

fn rms(signal: &[f32], from_s: f64, to_s: f64) -> f64 {
    let sr = f64::from(SAMPLE_RATE);
    let lo = ((from_s * sr).max(0.0) as usize).min(signal.len());
    let hi = ((to_s * sr).max(0.0) as usize).min(signal.len());
    if hi <= lo {
        return 1e-30;
    }
    (signal[lo..hi].iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>()
        / (hi - lo) as f64)
        .sqrt()
        .max(1e-30)
}

fn spectrum(slice: &[f32]) -> Vec<f64> {
    let n = slice.len().next_power_of_two().max(4096);
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n);
    let mut buffer = vec![Complex32::new(0.0, 0.0); n];
    for (i, slot) in buffer.iter_mut().take(slice.len()).enumerate() {
        let w = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / slice.len() as f64).cos();
        *slot = Complex32::new(slice[i] * w as f32, 0.0);
    }
    fft.process(&mut buffer);
    buffer[..=n / 2].iter().map(|c| f64::from(c.norm_sqr())).collect()
}

fn octave_bands(signal: &[f32], from_s: f64, to_s: f64) -> Vec<(f64, f64)> {
    let sr = f64::from(SAMPLE_RATE);
    let lo = ((from_s * sr) as usize).min(signal.len());
    let hi = ((to_s * sr) as usize).min(signal.len());
    if hi <= lo + 32 {
        return Vec::new();
    }
    let power = spectrum(&signal[lo..hi]);
    let mut out = Vec::new();
    let mut c = 62.5f64;
    while c < 16_000.0 {
        out.push((c, band(&power, sr, (c / 1.414, c * 1.414))));
        c *= 2.0;
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args
        .first()
        .cloned()
        .unwrap_or_else(|| "presets/salamander-c5.toml".to_string());
    let preset = Preset::load(std::path::Path::new(&path)).expect("preset");

    let mut no_strike = preset.clone();
    no_strike.noise.strike.level_db = silent();
    let mut no_key_off = preset.clone();
    no_key_off.noise.key_off.level_db = silent();
    let mut bare = no_strike.clone();
    bare.noise.key_off.level_db = silent();

    let phrase = melody::soprano();
    let events = engine_events::to_render_events(&phrase.events);
    let render = |p: &Preset| {
        let (l, r) = render_to_buffer(p, &events, phrase.duration_s as f32);
        mono(&[l, r])
    };
    let shipped = render(&preset);
    let quiet_strike = render(&no_strike);
    let quiet_key_off = render(&no_key_off);
    let bare_render = render(&bare);

    let mut sampler = Sampler::new(SFZ).expect("sampler");
    let reference_audio = sampler
        .render(&phrase.events, phrase.duration_s)
        .expect("reference");
    let reference =
        mono(&melody::align_reference(&reference_audio, phrase.events[0].time_s).channels);

    let strike_burst: Vec<f32> = shipped
        .iter()
        .zip(&quiet_strike)
        .map(|(&a, &b)| a - b)
        .collect();
    let key_off_burst: Vec<f32> = shipped
        .iter()
        .zip(&quiet_key_off)
        .map(|(&a, &b)| a - b)
        .collect();
    let both: Vec<f32> = shipped
        .iter()
        .zip(&bare_render)
        .map(|(&a, &b)| a - b)
        .collect();

    let whole = (0.0, phrase.duration_s);
    println!("the Ode soprano line, {} s, engine on {path}\n", phrase.duration_s);
    let level = 20.0 * rms(&shipped, whole.0, whole.1).log10();
    println!("whole-line RMS {level:.1} dB; each event alone, against it:");
    for (name, signal) in [
        ("the strike burst ", &strike_burst),
        ("the key-off burst", &key_off_burst),
        ("both together    ", &both),
    ] {
        println!(
            "  {name}  {:.1} dB below the line",
            level - 20.0 * rms(signal, whole.0, whole.1).log10()
        );
    }

    let match_db =
        20.0 * (rms(&reference, whole.0, whole.1) / rms(&shipped, whole.0, whole.1)).log10();
    println!(
        "\noctave by octave over the whole line, engine level-matched to the reference; \
         the last two columns are each event against the engine's own content in that octave\n"
    );
    println!(
        "{:>8} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "Hz", "engine", "ref", "eng-ref", "strike", "key-off"
    );
    let e = octave_bands(&shipped, whole.0, whole.1);
    let r = octave_bands(&reference, whole.0, whole.1);
    let s = octave_bands(&strike_burst, whole.0, whole.1);
    let k = octave_bands(&key_off_burst, whole.0, whole.1);
    let db = |p: f64| 10.0 * p.max(1e-30).log10();
    for i in 0..e.len() {
        println!(
            "{:>8.0} {:>9.1} {:>9.1} {:>9.1} {:>9.1} {:>9.1}",
            e[i].0,
            db(e[i].1) + match_db,
            db(r[i].1),
            db(e[i].1) + match_db - db(r[i].1),
            db(s[i].1) - db(e[i].1),
            db(k[i].1) - db(e[i].1),
        );
    }

    println!("\nat each note-off — level over the 60 ms after it, against the 60 ms before:");
    println!(
        "{:>7} {:>4} {:>9} {:>9} {:>9} {:>9}",
        "at", "key", "engine", "ref", "no-keyoff", "eng-ref"
    );
    let notes = melody::line_notes();
    let mut deltas: Vec<f64> = Vec::new();
    let mut deltas_quiet: Vec<f64> = Vec::new();
    for note in notes.iter().filter(|n| n.measurable()) {
        let off = note.onset_s + note.held_s;
        let step =
            |sig: &[f32]| 20.0 * (rms(sig, off, off + 0.06) / rms(sig, off - 0.06, off)).log10();
        let (a, b, c) = (step(&shipped), step(&reference), step(&quiet_key_off));
        deltas.push(a - b);
        deltas_quiet.push(c - b);
        println!(
            "{:>7.2} {:>4} {:>9.2} {:>9.2} {:>9.2} {:>+9.2}",
            off,
            melody::note_name(note.key),
            a,
            b,
            c,
            a - b
        );
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let mean_abs = |v: &[f64]| v.iter().map(|x: &f64| x.abs()).sum::<f64>() / v.len() as f64;
    println!(
        "\nrelease step, engine minus reference: mean {:+.2} dB (|·| {:.2}) as shipped, \
         {:+.2} dB (|·| {:.2}) with [noise.key_off] silenced",
        mean(&deltas),
        mean_abs(&deltas),
        mean(&deltas_quiet),
        mean_abs(&deltas_quiet)
    );

    // The sampler's own lead-in: what a window placed on the reference's onset
    // does to the engine's.
    println!("\nthe sampler's lead-in at the recorded keys — reference onset minus engine onset:");
    let library = SampleLibrary::from_sfz(SFZ).expect("library");
    let recorded = RecordedKeys::from_library(&library).expect("recorded keys");
    let sr = f64::from(SAMPLE_RATE);
    let mut lead: Vec<f64> = Vec::new();
    for &key in recorded.keys() {
        let mut row = Vec::new();
        for &vel in &VELOCITIES {
            let engine_events = [
                RenderEvent::new(0.05, Event::NoteOn { key, vel: u16::from(vel) }),
                RenderEvent::new(0.55, Event::NoteOff { key, vel: 64 }),
            ];
            let (l, r) = render_to_buffer(&preset, &engine_events, 0.8);
            let engine = mono(&[l, r]);
            let played = sampler
                .render(&TimedEvent::note(0.05, key, vel, 0.5), 0.8)
                .expect("reference");
            let played = mono(&played.channels);
            let ms = (melody::note_onset(&played, sr, 0.05)
                - melody::note_onset(&engine, sr, 0.05))
                * 1e3;
            row.push(ms);
            lead.push(ms);
        }
        println!(
            "  {:<4} {}",
            melody::note_name(key),
            row.iter()
                .map(|ms| format!("{ms:+6.0}"))
                .collect::<Vec<String>>()
                .join(" ")
        );
    }
    lead.sort_by(f64::total_cmp);
    println!(
        "  n {}, median {:+.0} ms, mean {:+.0} ms, range {:+.0} .. {:+.0} ms — against a \
         30 ms attack window",
        lead.len(),
        lead[lead.len() / 2],
        lead.iter().sum::<f64>() / lead.len() as f64,
        lead[0],
        lead[lead.len() - 1]
    );
}
