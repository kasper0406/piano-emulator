//! **How a partial's decay slope is fitted**, decided by measurement rather than
//! by taste: five conventions, six spans, both signals, and the spread of each
//! convention's answer across the spans as the score.
//!
//! `DECISIONS.md` 453 stopped at the observation that the correct
//! `notes.partial_sigma_scale[C4][k=1]` is **span-convention-sensitive** (0.87
//! by endpoint arithmetic, 1.0-1.2 by a slope over the law, 1.5 by a
//! floor-excluded fit) and instructed that a re-fit decide its convention
//! first. This is that decision.
//!
//! # What is scored
//!
//! What the fit writes is a **ratio** — how much faster the recording's partial
//! falls than the engine's — so the quantity whose stability matters is the
//! ratio and not either side's rate. All three are printed: a convention can be
//! unstable on both signals and stable on their ratio (a common bias cancels)
//! and that is a perfectly good convention, which is why the ratio is the
//! score.
//!
//! Stability is `max/min` of the answer over [`SPANS`], per cell, and the score
//! is the **median** of it over the cells — with the 90th percentile beside it,
//! because a convention that is excellent at the median and catastrophic at one
//! cell in ten is not one a fit can use.
//!
//! # The conventions
//!
//! | name | what it is |
//! |---|---|
//! | `endpt` | the shipped one: `env(t0) - env(t1)`, two instants of the dB envelope, `tail::levels_at_instants` |
//! | `cen0.30` | the same two instants, each read as the **power average** of the envelope over a 0.30 s window centred on it and *narrowed* where it would reach past the strike ([`power_window`]) |
//! | `shf0.30` / `shf0.45` / `shf0.60` | the same, with the window *slid* instead of narrowed, so both instants keep one width ([`power_shifted`]) |
//! | `ls` | least squares through the dB envelope over the whole span, floor-limited points only |
//! | `upper` | least squares through the running **maximum** over [`AVG_S`] — the beat's crests, which is where the string is |
//!
//! A three-string unison's partial is amplitude-modulated at the beat rate, so
//! an instant is a sample of a modulation the fit does not want and an average
//! over more than a beat period is not. The library's measured unison beats run
//! 0.7-4 Hz, so a third of a second is at least one cycle of all but the
//! slowest.
//!
//! ```sh
//! cargo run --release -p forensics --bin span_convention -- \
//!     data/salamander presets/salamander-c5.toml 51 54 57 60 63 66 69 72 75
//! ```

use std::path::PathBuf;

use piano_emulator::preset::Preset;
use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_emulator::types::Event;
use piano_tuner::estimate::tail::{partial_envelopes, FLOOR_FROM_S, FLOOR_MARGIN_DB, HOP_S};
use piano_tuner::{realism, Audio, Sampler, SamplerEvent, TimedEvent, SAMPLE_RATE};

const VELOCITY: u8 = realism::ODE_MELODY_VEL;
const RENDER_S: f64 = 4.2;
const PREROLL_S: f64 = 0.05;
/// Partials scored. The question is the sub-2 kHz band and at C4 that is k <= 7.
const PARTIALS: usize = 8;
/// Window the running maximum uses.
const AVG_S: f64 = 0.30;

/// The spans a convention's answer has to be the same over.
///
/// Two families, scored separately, because they ask different questions and a
/// piano partial is a **double decay** — a prompt component and an aftersound —
/// so a slope over 0.1-0.7 s and one over 0.3-2.0 s are genuinely different
/// quantities and no convention can make them equal. `wide` therefore measures
/// how much of the answer is the double decay plus the beat, and `edges`
/// perturbs only the ends of one nominal span, which is the part that is
/// arbitrary and the part a convention is allowed to be judged on.
const SPANS: [(f64, f64); 6] = [
    (0.10, 0.70),
    (0.10, 1.00),
    (0.10, 1.50),
    (0.15, 1.20),
    (0.20, 1.60),
    (0.30, 2.00),
];

/// The same nominal span with its two edges moved. See [`SPANS`].
const EDGES: [(f64, f64); 6] = [
    (0.10, 1.30),
    (0.10, 1.50),
    (0.10, 1.70),
    (0.15, 1.50),
    (0.20, 1.50),
    (0.15, 1.65),
];

const CONVENTIONS: [&str; 7] = [
    "endpt", "cen0.30", "shf0.30", "shf0.45", "shf0.60", "ls", "upper",
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let data = PathBuf::from(args.next().unwrap_or_else(|| "data/salamander".into()));
    let preset_path = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "presets/salamander-c5.toml".into()),
    );
    let keys: Vec<u8> = args.filter_map(|a| a.parse().ok()).collect();
    let keys = if keys.is_empty() {
        vec![51, 54, 57, 60, 63, 66, 69, 72, 75]
    } else {
        keys
    };
    let sfz = data.join("SalamanderGrandPiano-V3+20200602.sfz");
    let preset = Preset::load(&preset_path)?;
    let mut sampler = Sampler::new(&sfz)?;

    // Per cell and per convention: the rate on each side over each span, and
    // the ratio. Collected first, scored after, so that one pass over the audio
    // answers every convention. `[band][convention]` with band 0 the sub-2 kHz
    // half this milestone owns and band 1 everything.
    let mut spread_ref: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); CONVENTIONS.len()]; 2];
    let mut spread_eng: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); CONVENTIONS.len()]; 2];
    let mut spread_ratio: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); CONVENTIONS.len()]; 2];
    let mut spread_edges: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); CONVENTIONS.len()]; 2];
    // The same two scores kept cell by cell, so that the conventions can also
    // be compared on the cells **all** of them resolved: they do not refuse the
    // same material — `ls` fits a span half of whose points are above the
    // floor, `pavg` needs both of its readings to be — and a score taken over
    // each convention's own population rewards the one that answers where it
    // should not.
    let mut cells_wide: Vec<Vec<Vec<f64>>> = vec![Vec::new(); 2];
    let mut cells_edge: Vec<Vec<Vec<f64>>> = vec![Vec::new(); 2];

    println!(
        "per-cell ratio (recording rate / engine rate), median over the {} wide spans\n\
         key   k       hz | {} | spread x across the wide spans",
        SPANS.len(),
        CONVENTIONS
            .iter()
            .map(|c| format!("{c:>8}"))
            .collect::<String>(),
    );
    for &key in &keys {
        let params = preset.string_params(key);
        let n = usize::from(params.partial_count()).min(PARTIALS);
        let hz: Vec<f64> = (1..=n).map(|k| f64::from(params.partial_freq(k))).collect();
        let f0 = hz[0];
        let engine = render(&preset, key).mono();
        let Ok(reference) = reference_mono(&mut sampler, key) else {
            eprintln!("  key {key}: no recording");
            continue;
        };
        let sr = f64::from(SAMPLE_RATE);
        let e_env = partial_envelopes(&engine, &hz, f0, sr);
        let r_env = partial_envelopes(&reference, &hz, f0, sr);
        for k in 1..=n {
            let (e, r) = (&e_env[k - 1], &r_env[k - 1]);
            let (ef, rf) = (resolvable(e), resolvable(r));
            let low = hz[k - 1] < 2_000.0;
            let mut ratios = Vec::new();
            let mut edges_here = Vec::new();
            for (c, _) in CONVENTIONS.iter().enumerate() {
                let er: Vec<f64> = SPANS.iter().map(|&s| rate(e, ef, c, s)).collect();
                let rr: Vec<f64> = SPANS.iter().map(|&s| rate(r, rf, c, s)).collect();
                let ratio = |er: &[f64], rr: &[f64]| -> Vec<f64> {
                    er.iter()
                        .zip(rr)
                        .map(|(&a, &b)| if a > 0.0 { b / a } else { f64::NAN })
                        .collect()
                };
                let ee: Vec<f64> = EDGES.iter().map(|&s| rate(e, ef, c, s)).collect();
                let re: Vec<f64> = EDGES.iter().map(|&s| rate(r, rf, c, s)).collect();
                let s = spread(&ratio(&er, &rr));
                let bands: &[usize] = if low { &[0, 1] } else { &[1] };
                for &b in bands {
                    push(&mut spread_eng[b][c], spread(&er));
                    push(&mut spread_ref[b][c], spread(&rr));
                    push(&mut spread_ratio[b][c], s);
                    push(&mut spread_edges[b][c], spread(&ratio(&ee, &re)));
                }
                ratios.push(s);
                edges_here.push(spread(&ratio(&ee, &re)));
            }
            for &b in if low { &[0usize, 1][..] } else { &[1usize][..] } {
                cells_wide[b].push(ratios.clone());
                cells_edge[b].push(edges_here.clone());
            }
            println!(
                "{key:>3} {k:>3} {:>8.1} | {} | {}",
                hz[k - 1],
                CONVENTIONS
                    .iter()
                    .enumerate()
                    .map(|(c, _)| {
                        let er = median(SPANS.iter().map(|&s| rate(e, ef, c, s)).collect());
                        let rr = median(SPANS.iter().map(|&s| rate(r, rf, c, s)).collect());
                        format!("{:>8.3}", rr / er)
                    })
                    .collect::<String>(),
                ratios
                    .iter()
                    .map(|s| format!("{s:>7.2}"))
                    .collect::<String>(),
            );
        }
    }

    for (b, what) in ["under 2 kHz — the band this milestone owns", "every partial"]
        .iter()
        .enumerate()
    {
        println!(
            "\nstability across spans, {what} \
             (max/min of the answer; 1.00 is a convention, 2.00 is a coin)"
        );
        println!(
            "{:>8} | {:>15} | {:>15} | {:>15} | {:>15}",
            "", "recording rate", "engine rate", "wide-span ratio", "edge-span ratio"
        );
        println!(
            "{:>8} | {:>7} {:>7} | {:>7} {:>7} | {:>7} {:>7} | {:>7} {:>7} | {:>5}",
            "", "median", "p90", "median", "p90", "median", "p90", "median", "p90", "cells"
        );
        for (c, name) in CONVENTIONS.iter().enumerate() {
            println!(
                "{name:>8} | {:>7.2} {:>7.2} | {:>7.2} {:>7.2} | {:>7.2} {:>7.2} | \
                 {:>7.2} {:>7.2} | {:>5}",
                median(spread_ref[b][c].clone()),
                percentile(spread_ref[b][c].clone(), 0.9),
                median(spread_eng[b][c].clone()),
                percentile(spread_eng[b][c].clone(), 0.9),
                median(spread_ratio[b][c].clone()),
                percentile(spread_ratio[b][c].clone(), 0.9),
                median(spread_edges[b][c].clone()),
                percentile(spread_edges[b][c].clone(), 0.9),
                spread_ratio[b][c].len(),
            );
        }
        let common_w: Vec<&Vec<f64>> = cells_wide[b]
            .iter()
            .filter(|r| r.iter().all(|v| v.is_finite()))
            .collect();
        let common_e: Vec<&Vec<f64>> = cells_edge[b]
            .iter()
            .filter(|r| r.iter().all(|v| v.is_finite()))
            .collect();
        println!(
            "  on the {} wide / {} edge cells every convention resolved:",
            common_w.len(),
            common_e.len()
        );
        for (c, name) in CONVENTIONS.iter().enumerate() {
            let w: Vec<f64> = common_w.iter().map(|r| r[c]).collect();
            let e: Vec<f64> = common_e.iter().map(|r| r[c]).collect();
            println!(
                "{name:>8} | {:>15} | {:>15} | {:>7.2} {:>7.2} | {:>7.2} {:>7.2} |",
                "", "",
                median(w.clone()),
                percentile(w, 0.9),
                median(e.clone()),
                percentile(e, 0.9),
            );
        }
    }
    Ok(())
}

fn push(into: &mut Vec<f64>, v: f64) {
    if v.is_finite() {
        into.push(v);
    }
}

/// Fewest spans that must resolve before a cell's spread is a measurement of a
/// convention rather than of which spans happened to survive the floor.
const MIN_SPANS: usize = 4;

/// `max/min` of the finite entries — the spread a convention leaves across the
/// spans. `NaN` where fewer than [`MIN_SPANS`] resolved.
///
/// A cell is dropped for **too few spans** and not for any missing one, because
/// dropping a cell the moment one span fails scores the conventions on
/// different populations: `ls` refuses a floor-limited span outright where
/// `endpt` returns a number from two readings that are both inside the floor,
/// so the strict rule would reward the convention that answers where it should
/// not.
fn spread(values: &[f64]) -> f64 {
    let ok: Vec<f64> = values
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v > 0.0)
        .collect();
    if ok.len() < MIN_SPANS {
        return f64::NAN;
    }
    let hi = ok.iter().copied().fold(f64::MIN, f64::max);
    let lo = ok.iter().copied().fold(f64::MAX, f64::min);
    hi / lo
}

/// The signal's own floor plus the margin nothing is fitted inside.
fn resolvable(env: &[f64]) -> f64 {
    let from = ((FLOOR_FROM_S / HOP_S) as usize).min(env.len());
    if from >= env.len() {
        return f64::NEG_INFINITY;
    }
    let mut tail: Vec<f64> = env[from..].to_vec();
    tail.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    tail[tail.len() / 2] + FLOOR_MARGIN_DB
}

/// One convention's answer for one partial over one span, in dB per second.
fn rate(env: &[f64], floor: f64, convention: usize, span: (f64, f64)) -> f64 {
    match convention {
        0 => {
            let (a, b) = (at(env, span.0), at(env, span.1));
            if a < floor || b < floor {
                return f64::NAN;
            }
            (a - b) / (span.1 - span.0)
        }
        1 => {
            let (a, b) = (power_window(env, span.0, 0.30), power_window(env, span.1, 0.30));
            if a < floor || b < floor {
                return f64::NAN;
            }
            (a - b) / (span.1 - span.0)
        }
        2..=4 => {
            let w = [0.30, 0.45, 0.60][convention - 2];
            let (a, b) = (power_shifted(env, span.0, w), power_shifted(env, span.1, w));
            if a < floor || b < floor {
                return f64::NAN;
            }
            (a - b) / (span.1 - span.0)
        }
        5 => -slope(env, floor, span),
        6 => -slope(&running_max(env), floor, span),
        _ => f64::NAN,
    }
}

/// The envelope's power average over a window of width `w` centred on `t`,
/// **narrowed symmetrically** where `t` is closer to the strike than half of it.
///
/// Narrowed and not clamped: a window clamped at zero is no longer centred on
/// its instant, and a window that reaches back past the onset averages in the
/// strike, which is the one part of the record the engine and the recording
/// differ in most. Narrowing keeps the window centred and keeps the **same**
/// width at both instants, which is what makes the smoothing exact: for a
/// single exponential the average over a symmetric window is the instantaneous
/// value times `sinh(sigma w) / (sigma w)`, the same factor at both instants,
/// so it cancels out of their difference and only the beat is removed.
fn power_window(env: &[f64], t: f64, w: f64) -> f64 {
    let half = (0.5 * w).min(t).max(0.0);
    power_forward(env, t - half, 2.0 * half)
}

/// The same window of width `w`, **slid** forward rather than narrowed where
/// the instant is closer to the strike than half of it.
///
/// This is the one that keeps the exactness: the correction a power average
/// puts on a single exponential is a function of the window's *width*, so two
/// windows of the same width carry the same correction whatever their centres
/// are and it cancels out of their difference. Narrowing one of them does not.
/// What sliding costs is a little of the nominal separation between the two
/// instants — the same amount on both signals, so it cancels out of the ratio
/// the fit writes.
fn power_shifted(env: &[f64], t: f64, w: f64) -> f64 {
    power_forward(env, (t - 0.5 * w).max(0.0), w)
}

/// The envelope's power average over `w` seconds **starting** at `t`, in dB.
fn power_forward(env: &[f64], t: f64, w: f64) -> f64 {
    let lo = (t / HOP_S).round() as usize;
    let hi = (lo + (w / HOP_S).round() as usize).min(env.len());
    if lo >= hi {
        return f64::NEG_INFINITY;
    }
    let n = (hi - lo) as f64;
    let sum: f64 = env[lo..hi].iter().map(|&db| 10f64.powf(db / 10.0)).sum();
    10.0 * (sum / n).max(1e-30).log10()
}

fn at(env: &[f64], t: f64) -> f64 {
    let i = (t / HOP_S).round() as usize;
    env.get(i).copied().unwrap_or(f64::NEG_INFINITY)
}

fn running_max(env: &[f64]) -> Vec<f64> {
    let half = (0.5 * AVG_S / HOP_S).round() as usize;
    (0..env.len())
        .map(|i| {
            let lo = i.saturating_sub(half);
            let hi = (i + half + 1).min(env.len());
            env[lo..hi].iter().copied().fold(f64::MIN, f64::max)
        })
        .collect()
}

/// Least squares in dB per second over the span, over the points that stand
/// clear of the signal's own floor.
fn slope(env: &[f64], floor: f64, span: (f64, f64)) -> f64 {
    let (a, b) = (
        (span.0 / HOP_S) as usize,
        ((span.1 / HOP_S) as usize + 1).min(env.len()),
    );
    if a >= b {
        return f64::NAN;
    }
    let pts: Vec<(f64, f64)> = env[a..b]
        .iter()
        .enumerate()
        .filter(|(_, &v)| v.is_finite() && v > floor)
        .map(|(i, &v)| ((a + i) as f64 * HOP_S, v))
        .collect();
    // Half the span's points, or the fit is of something else.
    if pts.len() * 2 < b - a {
        return f64::NAN;
    }
    let n = pts.len() as f64;
    let mx = pts.iter().map(|p| p.0).sum::<f64>() / n;
    let my = pts.iter().map(|p| p.1).sum::<f64>() / n;
    let num: f64 = pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
    let den: f64 = pts.iter().map(|p| (p.0 - mx).powi(2)).sum();
    if den <= 0.0 {
        f64::NAN
    } else {
        num / den
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.retain(|x| x.is_finite());
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    v[v.len() / 2]
}

fn percentile(mut v: Vec<f64>, p: f64) -> f64 {
    v.retain(|x| x.is_finite());
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    v[(((v.len() - 1) as f64) * p).round() as usize]
}

fn render(preset: &Preset, key: u8) -> Audio {
    let events = [RenderEvent::new(
        PREROLL_S as f32,
        Event::NoteOn {
            key,
            vel: u16::from(VELOCITY),
        },
    )];
    let (left, right) = render_to_buffer(preset, &events, (PREROLL_S + RENDER_S) as f32);
    let skip = (PREROLL_S * f64::from(SAMPLE_RATE)) as usize;
    Audio::new(
        SAMPLE_RATE,
        vec![left[skip..].to_vec(), right[skip..].to_vec()],
    )
    .expect("the engine renders stereo")
}

fn reference_mono(sampler: &mut Sampler, key: u8) -> Result<Vec<f32>, piano_tuner::Error> {
    let events = [TimedEvent::new(
        0.0,
        SamplerEvent::NoteOn {
            key,
            vel: VELOCITY,
        },
    )];
    let rendered = sampler.render(&events, RENDER_S + 0.3)?;
    let mono = rendered.mono();
    let onset = piano_tuner::detect_onset(&mono, f64::from(SAMPLE_RATE));
    let skip = ((onset * f64::from(SAMPLE_RATE)).round() as usize).min(mono.len());
    Ok(mono[skip..].to_vec())
}
