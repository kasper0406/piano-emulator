//! Where the engine's upper partials decay at a rate nobody measured, by how
//! much, and the stage that fixes it — `DECISIONS.md` 304-309 and 319-321.
//!
//! `DECISIONS.md` 295 convicted the high partials' decay from the *band* side:
//! over 0.1 -> 1 s the engine's 2-6 kHz falls 11.8 dB less than the recording's
//! in the tenor and its 6-12 kHz 15.3 dB less, against a velocity-layer floor of
//! 0.42 / 0.86 dB, and that is why the strike-side brightness fix of the same
//! item was falsified — a continuation fitted to the strike carries its lift
//! through a tail that is already far too long, and a phrase integrates the
//! tail. This asks the same question of the **partial**, which is the thing the
//! preset can move: `notes.partial_sigma_scale` is a per-`k` multiplier on the
//! fitted `sigma(f)` law, and above the reach the tracker gave a key every cell
//! of it is exactly 1.0.
//!
//! [`piano_tuner::estimate::tail`] holds the measurement, its rules and the
//! draw; this is the driver.
//!
//! # What it prints
//!
//! 1. **Per key**: the key's bank, the reach its row has, how many of its
//!    partials measured a fall at all, the median correction inside its reach,
//!    above it and over 2 kHz, where its row came from, its two band decay gaps
//!    with the bound its floors leave on them, and what each band's stop says
//!    now.
//! 2. **What the recordings say**, key by key: the population every unsampled
//!    key's target is read off, printed so that the curve through it can be
//!    checked against the points.
//! 3. **Per register and band**: the median `fall(recording)/fall(engine)`,
//!    which is the factor a partial's `sigma` is out by — one is right — over
//!    every trusted partial and again over only those whose *late* readings are
//!    both measurements rather than floor bounds, which above 6 kHz is a
//!    different set and a different answer (`DECISIONS.md` 319).
//! 4. **The band decay gap** over the same two instants, counted over the
//!    partial bins alone and over the whole band, against the recording's own
//!    velocity-layer floor and against each signal's own late-time floor —
//!    `estimate::brilliance::band_decay`, the column this milestone is gated on,
//!    with `≥` where the reference side of it has fallen into the recording's
//!    own floor and the number is a bound rather than a measurement.
//! 5. **The fitted keys against the drawn ones**, per register, which is the
//!    seam item 320 was opened on.
//! 6. **The seam**: the same statistic inside and outside each key's own fitted
//!    reach.
//!
//! # The stage
//!
//! `--passes <n> --out <file>` runs the fit. Each pass measures what is *left*
//! on the render and multiplies it into the row, which is a fixed-point
//! iteration and not a prediction — the pattern of items 137, 199, 211, 264,
//! 273 and 300. A **band** is closed against the key's own recording where that
//! recording resolves it and against a target read off
//! [`DecayModel`](piano_tuner::estimate::tail::DecayModel) — the compass's own
//! line through the sampled keys plus their interpolated departure from it —
//! where it does not; the keys whose rows carry a target rather than a
//! measurement are named in `notes.synthesized_decay`.
//!
//! It is idempotent: the keys the provenance list names are cleared before the
//! stage runs again, the sampled keys' loop has a stop wide enough to be a fixed
//! point rather than a ratchet (`tail::BandFall::partial_median_ratio_error`),
//! and a second run over its own output reproduces the same 61 rows, the same 37
//! drawn keys and **every cell to the last digit**.
//!
//! ```sh
//! cargo run --release -p piano-tuner -- tail \
//!     data/salamander presets/salamander-c5.toml
//! cargo run --release -p piano-tuner -- tail \
//!     data/salamander presets/salamander-c5.toml \
//!     --passes 8 --out /tmp/fitted.toml
//! ```

use std::cell::RefCell;
use std::path::{Path, PathBuf};

use rayon::prelude::*;

use piano_emulator::preset::Preset;
use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_emulator::types::Event;
use piano_tuner::cache;
use piano_tuner::estimate::tail::{
    engine_band_fall, extend_row, fit_tail_to, levels_at_instants, measurable_db,
    bank_owns_band, partial_band_fall, partial_envelopes, reach, reach_to, BandFall,
    DecayModel, DecayPoint,
    DrawnDecay, PartialBandFall, PartialTail, SideFall, TailCorrection, FLOOR_FROM_S, HOP_S,
};
use piano_tuner::estimate::brilliance::{band_decay, band_decay_gap, BandDecay, HF1, HF2};
use piano_tuner::realism::VelocityLayers;
use piano_tuner::sampler::SAMPLER_VERSION;
use piano_tuner::stft::{Stft, StftConfig};
use piano_tuner::{Audio, SampleLibrary, Sampler, SamplerEvent, TimedEvent, SAMPLE_RATE};

const VELOCITY: u8 = 90;
/// Seconds of note — `compass`'s own length, so the reference cache is
/// shared byte for byte.
const RENDER_S: f64 = 3.6;
const PREROLL_S: f64 = 0.05;
const FIRST_KEY: u8 = 21;
const LAST_KEY: u8 = 108;
const SR: f64 = SAMPLE_RATE as f64;
const MAX_CACHED_BUFFERS: usize = 8;

/// Share of one pass's correction actually applied. See `TailCorrection::damped`.
const DAMPING: f64 = 0.7;

/// Least width the gate's own floor is ever given, dB.
///
/// The velocity-layer floor is a *measurement* and at a few keys it comes back
/// under a tenth of a decibel, which would ask this loop to chase a band decay
/// to a precision the render's own strike alignment does not have. A quarter of
/// a decibel is under every register's floor in `DECISIONS.md` 292's table
/// (0.42 / 0.86 in the tenor, 1.17 / 1.87 in the treble) and is where a
/// half-millisecond of onset lands.
const MIN_GATE_FLOOR_DB: f64 = 0.25;

thread_local! {
    static SAMPLER: RefCell<Option<Sampler>> = const { RefCell::new(None) };
}

fn with_sampler<T>(
    sfz: &Path,
    body: impl FnOnce(&mut Sampler) -> Result<T, piano_tuner::Error>,
) -> Result<T, piano_tuner::Error> {
    SAMPLER.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            *slot = Some(Sampler::new(sfz)?);
        }
        let sampler = slot.as_mut().expect("a player was just built");
        let out = body(sampler);
        if sampler.cached_buffers() > MAX_CACHED_BUFFERS {
            sampler.clear_cache();
        }
        out
    })
}

fn render_engine(preset: &Preset, key: u8) -> Audio {
    let events = [RenderEvent::new(
        PREROLL_S as f32,
        Event::NoteOn { key, vel: VELOCITY },
    )];
    let (left, right) = render_to_buffer(preset, &events, (PREROLL_S + RENDER_S) as f32);
    let skip = (PREROLL_S * SR) as usize;
    Audio::new(
        SAMPLE_RATE,
        vec![left[skip..].to_vec(), right[skip..].to_vec()],
    )
    .expect("the engine renders stereo")
}

fn render_reference(sampler: &mut Sampler, key: u8, vel: u8) -> Result<Audio, piano_tuner::Error> {
    let events = [TimedEvent::new(0.0, SamplerEvent::NoteOn { key, vel })];
    let rendered = sampler.render(&events, RENDER_S + 0.2)?;
    let mono = rendered.mono();
    let onset = piano_tuner::detect_onset(&mono, SR);
    let skip = (onset * SR).round() as usize;
    let frames = (RENDER_S * SR) as usize;
    let cut = |c: &Vec<f32>| -> Vec<f32> {
        (0..frames)
            .map(|n| c.get(skip + n).copied().unwrap_or(0.0))
            .collect()
    };
    Audio::new(SAMPLE_RATE, rendered.channels.iter().map(cut).collect())
}

fn note_name(key: u8) -> String {
    const NAMES: [&str; 12] = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    format!("{}{}", NAMES[usize::from(key) % 12], i32::from(key) / 12 - 1)
}

fn median(xs: impl Iterator<Item = f64>) -> f64 {
    let mut v: Vec<f64> = xs.filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

struct KeyRow {
    key: u8,
    /// The 2-6 kHz band decay over the two instants counted **only over the
    /// partial bins**, engine minus reference, in dB — `band_decay_gap`'s own
    /// statistic with everything that is not a partial of this note left out.
    partial_band_gap: [f64; 2],
    /// The same over the whole band, straight off the two spectra, with each
    /// signal's own late-time floor beside it — `estimate::brilliance::band_decay`,
    /// the column the milestone is gated on and the direction it is bounded in.
    full_band: [BandDecay; 2],
    /// The same taken between the recording and the *velocity layer next door*:
    /// two recordings of one piano, and therefore what this statistic cannot
    /// resolve. A band already inside it is not corrected.
    layer_band_gap: [f64; 2],
    /// What each band's own partials did on both signals — the numbers a key the
    /// library sampled is closed on. `None` for a band its recording could not
    /// measure, which is a band that gets drawn for like an unsampled key's.
    measured: [Option<PartialBandFall>; 2],
    /// The engine's side alone, over the partials the engine still resolves:
    /// what a *drawn* band is closed on, since it has no recording to trust.
    engine_fall: [Option<SideFall>; 2],
    /// The length of this key's `notes.partial_sigma_scale` row as the preset
    /// carries it.
    reach: usize,
    /// The highest partial this key's *recording* still measured a fall on, and
    /// that partial's frequency: what a sampled key's row is licensed to, and
    /// the population `tail::Ceiling` is fitted from.
    measured_reach: usize,
    ceiling_hz: f64,
    tails: Vec<PartialTail>,
}

/// One key's two bands as the fit sees them: measured against its own recording
/// where that recording said anything, drawn against [`DecayModel`] where it did
/// not, and `None` where neither is available.
///
/// A **band** rule and not a key rule: the six sampled keys of the top octave,
/// and the upper band of several keys below them, resolve too few partials for a
/// recording to say anything, and item 307 routed every sampled key down the
/// measured branch and so left them with no row at all (`DECISIONS.md` 321).
///
/// The second return is which bands were drawn for, which is what item 291's
/// provenance rule is written from.
fn band_falls(
    r: &KeyRow,
    drawn: &DrawnDecay,
    is_sampled: bool,
    may_draw: bool,
) -> ([Option<BandFall>; 2], [bool; 2]) {
    let mut drew = [false; 2];
    let falls = [0usize, 1].map(|b| {
        let floor_db = r.layer_band_gap[b].abs().max(MIN_GATE_FLOOR_DB);
        if is_sampled {
            if let Some(m) = r.measured[b] {
                return Some(BandFall {
                    engine_db: m.engine.band_db,
                    reference_db: m.reference.band_db,
                    partial_share: 1.0,
                    partial_median_ratio: m.median_ratio,
                    floor_db,
                    partial_median_ratio_error: m.median_ratio_ln_error,
                });
            }
        }
        if !may_draw {
            return None;
        }
        // No recording of this band, so the target is drawn and the *engine's*
        // side of the comparison is still measured — the close is on the render
        // either way. Both halves of the stop are median partial falls, which is
        // the statistic the measured branch stops on.
        let engine = r.engine_fall[b]?;
        drew[b] = true;
        Some(BandFall {
            engine_db: engine.band_db,
            reference_db: drawn.target_fall_db[b],
            partial_share: 1.0,
            partial_median_ratio: drawn.target_partial_fall_db[b]
                / engine.median_db.max(f64::MIN_POSITIVE),
            floor_db,
            // The target is a curve through the compass and carries no error of
            // its own that this loop can read; what it does carry is the error
            // of the median it is divided by, which is the same denominator the
            // measured branch has.
            partial_median_ratio_error: engine.median_ln_error,
        })
    });
    (falls, drew)
}

pub fn run(args: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
    let mut args = args.into_iter();
    let data = PathBuf::from(args.next().unwrap_or_else(|| "data/salamander".into()));
    let preset_path = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "presets/salamander-c5.toml".into()),
    );
    let rest: Vec<String> = args.collect();
    let flag = |name: &str| -> Option<&String> {
        rest.iter()
            .position(|a| a == name)
            .and_then(|i| rest.get(i + 1))
    };
    let detail: Option<u8> = flag("--key").and_then(|v| v.parse().ok());
    let out: Option<PathBuf> = flag("--out").map(PathBuf::from);
    let passes: usize = flag("--passes").and_then(|v| v.parse().ok()).unwrap_or(0);
    let sfz = data.join("SalamanderGrandPiano-V3+20200602.sfz");
    if !sfz.exists() {
        eprintln!("the reference piano is not here: {}", sfz.display());
        std::process::exit(2);
    }
    let mut preset = Preset::load(&preset_path)?;
    let library = SampleLibrary::from_sfz(&sfz)?;
    let mut sampled: Vec<u8> = library.samples().map(|s| s.key).collect();
    sampled.sort_unstable();
    sampled.dedup();
    // Every key, not only the thirty the library sampled: the fit runs on the
    // sampled ones and the draw on the rest, and both are closed in the same
    // loop because the sympathetic halo couples them (`DECISIONS.md` 289).
    let mut keys: Vec<u8> = (FIRST_KEY..=LAST_KEY).collect();

    let reference_cache = cache::reference_dir(&data);
    let mut compass_key = cache::Fingerprint::new();
    compass_key
        .str("compass-scan/reference")
        .u64(u64::from(SAMPLER_VERSION))
        .file(&sfz)?
        .u64(u64::from(SAMPLE_RATE))
        .u64(u64::from(VELOCITY))
        .f64(RENDER_S);

    let layers = VelocityLayers::from_library(&library)?;
    let alt_velocity = layers.alternate(VELOCITY);
    // Byte-for-byte `brilliance.rs`'s own key, so the two tools share a cache.
    let mut alt_key = cache::Fingerprint::new();
    alt_key
        .str("brilliance/alt-layer")
        .u64(u64::from(SAMPLER_VERSION))
        .file(&sfz)?
        .u64(u64::from(SAMPLE_RATE))
        .u64(u64::from(alt_velocity))
        .f64(RENDER_S);

    println!(
        "per-partial tail audit: engine on {}, reference {}\n{} sampled keys at velocity {VELOCITY}, {RENDER_S} s",
        preset_path.display(),
        sfz.display(),
        keys.len()
    );

    if let Some(k) = detail {
        keys.retain(|&x| x == k);
    }
    let measure = |preset: &Preset, keys: &[u8]| -> Result<Vec<KeyRow>, piano_tuner::Error> {
        keys.par_iter()
        .map(|&key| -> Result<KeyRow, piano_tuner::Error> {
            let engine = render_engine(preset, key).mono();
            let mut k = compass_key;
            k.u64(u64::from(key));
            let path = reference_cache.join(format!("compass-key{key:03}-{}.wav", k.hex()));
            let reference =
                cache::audio(&path, || with_sampler(&sfz, |s| render_reference(s, key, VELOCITY)))?
                    .mono();
            let mut a = alt_key;
            a.u64(u64::from(key));
            let alt_path =
                reference_cache.join(format!("brilliance-alt-key{key:03}-{}.wav", a.hex()));
            let alt = cache::audio(&alt_path, || {
                with_sampler(&sfz, |s| render_reference(s, key, alt_velocity))
            })?
            .mono();
            let params = preset.string_params(key);
            let n = params.partial_count();
            let partial_hz: Vec<f64> = (1..=n).map(|i| f64::from(params.partial_freq(i))).collect();
            let f0 = partial_hz.first().copied().unwrap_or(440.0);
            let e = partial_envelopes(&engine, &partial_hz, f0, SR);
            let r = partial_envelopes(&reference, &partial_hz, f0, SR);
            let tails: Vec<PartialTail> = partial_hz
                .iter()
                .enumerate()
                .map(|(i, &hz)| {
                    let (me, mr) = (measurable_db(&e[i], HOP_S), measurable_db(&r[i], HOP_S));
                    let drop = match (me, mr) {
                        (Some(a), Some(b)) => a.min(b),
                        _ => f64::NAN,
                    };
                    if detail == Some(key) {
                        let fe = fit_tail_to(&e[i], HOP_S, drop);
                        let fr = fit_tail_to(&r[i], HOP_S, drop);
                        let fall = |v: piano_tuner::estimate::tail::Levels| {
                            Some(v.at[0]? - v.late_or_bound())
                        };
                        let (ae, ar) = (
                            fall(levels_at_instants(&e[i], HOP_S)),
                            fall(levels_at_instants(&r[i], HOP_S)),
                        );
                        println!(
                            "  k={:>3} {:>8.0} Hz  meas e/r {:>6.1}/{:>6.1}  fall e/r {:>6}/{:>6} \
                             corr {:>5}  T60 e/r {:>6}/{:>6}",
                            i + 1,
                            hz,
                            me.unwrap_or(f64::NAN),
                            mr.unwrap_or(f64::NAN),
                            ae.map_or("-".into(), |v| format!("{v:.1}")),
                            ar.map_or("-".into(), |v| format!("{v:.1}")),
                            match (ae, ar) {
                                (Some(a), Some(b)) if a > 0.5 && b > 0.5 =>
                                    format!("{:.2}", b / a),
                                _ => "-".into(),
                            },
                            fe.map_or("-".into(), |f| format!("{:.2}", f.t60_s)),
                            fr.map_or("-".into(), |f| format!("{:.2}", f.t60_s)),
                        );
                    }
                    PartialTail {
                        k: i + 1,
                        hz,
                        drop_db: drop,
                        engine_db: levels_at_instants(&e[i], HOP_S),
                        reference_db: levels_at_instants(&r[i], HOP_S),
                        engine: fit_tail_to(&e[i], HOP_S, drop),
                        reference: fit_tail_to(&r[i], HOP_S, drop),
                    }
                })
                .collect();
            let gap = |blo: f64, bhi: f64| -> f64 {
                let sum = |env: &[Vec<f64>], t: f64| -> f64 {
                    let i = (t / HOP_S).round() as usize;
                    partial_hz
                        .iter()
                        .zip(env)
                        .filter(|(&hz, _)| hz >= blo && hz < bhi)
                        .filter_map(|(_, e)| e.get(i))
                        .map(|&db| 10f64.powf(db / 10.0))
                        .sum()
                };
                let drop = |env: &[Vec<f64>]| {
                    10.0 * (sum(env, 1.0).max(1e-30) / sum(env, 0.1).max(1e-30)).log10()
                };
                drop(&e) - drop(&r)
            };
            let spectrum = |sig: &[f32], t: f64| -> Vec<f64> {
                let n = 4096usize;
                let start = (t * SR) as usize;
                if start + n > sig.len() {
                    return Vec::new();
                }
                let stft = Stft::new(StftConfig::new(n, n, n).expect("geometry")).expect("plan");
                stft.analyze(&sig[start..start + n], SR).frames[0]
                    .magnitude
                    .iter()
                    .map(|&m| f64::from(m) * f64::from(m))
                    .collect()
            };
            // A signal's own floor, per bin: the median over the frames from
            // `FLOOR_FROM_S` to the end of the render — what is left when the
            // note is over. A median and not one frame because a single late
            // frame of a recording can catch a room event, and half the frames
            // would have to be wrong before this moved.
            let floor_spectrum = |sig: &[f32]| -> Vec<f64> {
                let frames: Vec<Vec<f64>> = (0..6)
                    .map(|i| spectrum(sig, FLOOR_FROM_S + 0.1 * f64::from(i)))
                    .filter(|f| !f.is_empty())
                    .collect();
                let Some(bins) = frames.first().map(Vec::len) else {
                    return Vec::new();
                };
                (0..bins)
                    .map(|b| {
                        let mut v: Vec<f64> = frames.iter().map(|f| f[b]).collect();
                        v.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
                        v[v.len() / 2]
                    })
                    .collect()
            };
            let (e0, e1) = (spectrum(&engine, 0.1), spectrum(&engine, 1.0));
            let (r0, r1) = (spectrum(&reference, 0.1), spectrum(&reference, 1.0));
            let (a0, a1) = (spectrum(&alt, 0.1), spectrum(&alt, 1.0));
            let (ef, rf) = (floor_spectrum(&engine), floor_spectrum(&reference));
            let full_band =
                [HF1, HF2].map(|b| band_decay(&e0, &e1, &ef, &r0, &r1, &rf, SR, b));
            let layer_band_gap =
                [HF1, HF2].map(|b| band_decay_gap(&a0, &a1, &r0, &r1, SR, b));
            // What the key's own partials did in each band, on both signals:
            // the quantity a `notes.partial_sigma_scale` row can actually move.
            let measured = [HF1, HF2].map(|b| partial_band_fall(&tails, b));
            let engine_fall = [HF1, HF2].map(|b| engine_band_fall(&tails, b));
            let measured_reach = reach(&tails);
            Ok(KeyRow {
                key,
                partial_band_gap: [gap(HF1.0, HF1.1), gap(HF2.0, HF2.1)],
                full_band,
                layer_band_gap,
                measured,
                engine_fall,
                reach: preset.notes.partial_sigma_scale[usize::from(key - FIRST_KEY)].len(),
                measured_reach,
                ceiling_hz: tails
                    .iter()
                    .find(|t| t.k == measured_reach)
                    .map_or(f64::NAN, |t| t.hz),
                tails,
            })
        })
        .collect::<Result<Vec<_>, _>>()
    };

    let mut rows = measure(&preset, &keys)?;
    // The close on the render: a correction is measured off the render it is
    // going to change, so writing it once is a prediction and not a fit. Each
    // pass measures what is *left* and multiplies it into the row, which is a
    // fixed-point iteration on a quantity that is monotone in the cell — the
    // pattern of items 137, 199, 211, 264, 273 and 300.
    // The distribution the unsampled keys draw from, fitted from the thirty the
    // library did sample and from nothing else. What is drawn is the recording's
    // own band **fall** — a statement about the piano — and never a correction,
    // which is a statement about a render that changes every time the preset
    // does.
    let model = DecayModel::fit(
        &rows
            .iter()
            .filter(|r| sampled.contains(&r.key))
            .map(|r| {
                // A band the key's own series ends inside is corrected like any
                // other — its partials are still its partials — but it does not
                // stand for that band in the population: see `bank_owns_band`.
                let owns = [HF1, HF2].map(|b| bank_owns_band(&r.tails, b));
                DecayPoint {
                    key: r.key,
                    reference_fall_db: [0usize, 1]
                        .map(|b| r.measured[b].filter(|_| owns[b]).map(|f| f.reference.band_db)),
                    reference_partial_fall_db: [0usize, 1]
                        .map(|b| r.measured[b].filter(|_| owns[b]).map(|f| f.reference.median_db)),
                    ceiling_hz: r.ceiling_hz,
                }
            })
            .collect::<Vec<_>>(),
    );
    println!(
        "\ndrawn from the sampled keys, band sum / median partial:\n  \
         2-6k fall exp({:+.4}{:+.5}·key) dB x{:.2} (r {:+.3}, n {}), \
         partial exp({:+.4}{:+.5}·key) x{:.2} (r {:+.3}, n {})\n  \
         6-12k fall exp({:+.4}{:+.5}·key) dB x{:.2} (r {:+.3}, n {}), \
         partial exp({:+.4}{:+.5}·key) x{:.2} (r {:+.3}, n {})\n  \
         ceiling {:.0} Hz x{:.2} over {} keys — a frequency, not a partial index, and one number\n  \
         the two lines' residuals move together at r {:+.3} (2-6k) and {:+.3} (6-12k), which is \
         the assumption under interpolating each of them on its own",
        model.fall[0].intercept, model.fall[0].slope, model.fall[0].sigma.exp(),
        model.fall[0].correlation, model.fall[0].points,
        model.partial_fall[0].intercept, model.partial_fall[0].slope,
        model.partial_fall[0].sigma.exp(), model.partial_fall[0].correlation,
        model.partial_fall[0].points,
        model.fall[1].intercept, model.fall[1].slope, model.fall[1].sigma.exp(),
        model.fall[1].correlation, model.fall[1].points,
        model.partial_fall[1].intercept, model.partial_fall[1].slope,
        model.partial_fall[1].sigma.exp(), model.partial_fall[1].correlation,
        model.partial_fall[1].points,
        model.ceiling.hz, model.ceiling.spread, model.ceiling.points,
        model.residual_correlation[0], model.residual_correlation[1],
    );
    // What the recordings themselves say, key by key: the population the draw is
    // fitted from, printed so that a line through it can be checked against the
    // points rather than against its own correlation coefficient.
    println!(
        "\nwhat the recordings say, over the sampled keys (fall over 0.1 -> 1 s, dB)\n\
         {:>4} {:>5} {:>10} {:>10} {:>10} {:>10} {:>9} {:>10}",
        "key", "note", "band 2-6k", "part 2-6k", "band 6-12k", "part 6-12k", "ceiling", "in the fit"
    );
    for r in rows.iter().filter(|r| sampled.contains(&r.key)) {
        let cell = |b: usize, pick: fn(&PartialBandFall) -> f64| {
            r.measured[b].map_or("-".to_string(), |m| format!("{:.2}", pick(&m)))
        };
        // Which of the two bands this key stands for in the population: a band
        // its own series ends inside is corrected like any other and is not a
        // measurement of what that band does — `tail::bank_owns_band`.
        let owns = [HF1, HF2].map(|b| bank_owns_band(&r.tails, b));
        println!(
            "{:>4} {:>5} {:>10} {:>10} {:>10} {:>10} {:>9.0} {:>10}",
            r.key,
            note_name(r.key),
            cell(0, |m| m.reference.band_db),
            cell(0, |m| m.reference.median_db),
            cell(1, |m| m.reference.band_db),
            cell(1, |m| m.reference.median_db),
            r.ceiling_hz,
            format!(
                "{} {}",
                if owns[0] { "2-6k" } else { "-" },
                if owns[1] { "6-12k" } else { "-" }
            ),
        );
    }

    // How far the drawn stop is from the measured one it stands in for: a
    // sampled key stops on the median of its partials' own ratios and a drawn
    // key can only take the ratio of the two medians. Measured on the sampled
    // keys, where both exist, so that the substitution is a number and not an
    // assumption.
    let stop_agreement: Vec<f64> = rows
        .iter()
        .filter(|r| sampled.contains(&r.key))
        .flat_map(|r| r.measured.into_iter())
        .flatten()
        .filter(|m| m.engine.median_db > 0.0)
        .map(|m| (m.reference.median_db / m.engine.median_db) / m.median_ratio)
        .collect();
    println!(
        "  the drawn stop against the measured one, on the keys that have both: \
         median x{:.3} over {} bands (1.0 is the same statistic)",
        median(stop_agreement.iter().copied()),
        stop_agreement.len(),
    );
    // The other half of the substitution, and the one that bit: a drawn band's
    // denominator is `engine_band_fall`, over every partial the *engine* still
    // shows at 0.1 s, while a measured band's is over the partials the
    // *recording* also shows. The extra partials are the weak ones high in the
    // band, which are deep under the render's own floor and therefore report the
    // largest falls the bound allows.
    let denominator: Vec<f64> = rows
        .iter()
        .filter(|r| sampled.contains(&r.key))
        .flat_map(|r| [0usize, 1].map(|b| Some((r.measured[b]?, r.engine_fall[b]?))))
        .flatten()
        .filter(|(m, _)| m.engine.median_db > 0.0)
        .map(|(m, e)| e.median_db / m.engine.median_db)
        .collect();
    println!(
        "  the drawn stop's denominator against the measured one's: median x{:.3} over {} bands",
        median(denominator.iter().copied()),
        denominator.len(),
    );

    // Idempotence, item 291's rule: a stage that draws clears exactly the keys
    // its own provenance list names before it draws again, so that a second run
    // on its own output is the first run and not a second correction on top of
    // one. The *sampled* keys need no clearing — their loop is a fixed point on
    // the render, so a row that is already right measures a correction of one
    // and does not move.
    // Only when there is a fit to run: with `--passes 0` this tool is an audit
    // of the preset as it stands, and an audit that had emptied the provenance
    // list would report every drawn row as a fitted one.
    if passes > 0 {
        for key in std::mem::take(&mut preset.notes.synthesized_decay) {
            preset.notes.partial_sigma_scale[usize::from(key - FIRST_KEY)].clear();
        }
    }
    // Which rows this stage owns outright, read *after* the clearing and before
    // the first pass. The provenance list is also the clearing list, so a key
    // may only be declared drawn if there is nothing else in its row to lose:
    // `estimate::shaping` writes measured cells for the sampled keys up to the
    // tracker's own reach (items 200-201) and this stage cannot reproduce them.
    let stage_owns: Vec<bool> = preset
        .notes
        .partial_sigma_scale
        .iter()
        .map(Vec::is_empty)
        .collect();
    let mut drawn_row = vec![false; stage_owns.len()];
    if passes > 0 {
        preset.validate()?;
        rows = measure(&preset, &keys)?;
    }
    for pass in 1..=passes {
        for r in &rows {
            let i = usize::from(r.key - FIRST_KEY);
            let params = preset.string_params(r.key);
            let n = params.partial_count();
            let partial_hz: Vec<f64> =
                (1..=n).map(|j| f64::from(params.partial_freq(j))).collect();
            let is_sampled = sampled.contains(&r.key);
            // A band is closed against this key's own recording where that
            // recording measured it, and against a drawn target where nothing
            // did — which is a *band* rule and not a key rule, because the six
            // sampled keys of the top octave and the bands above them resolve
            // too few partials for a recording to say anything, and item 307
            // routed every sampled key down the measured branch and so left
            // them with no row at all (`DECISIONS.md` 321).
            let drawn = model.draw(r.key);
            let may_draw = !is_sampled || stage_owns[i];
            let (falls, drew) = band_falls(r, &drawn, is_sampled, may_draw);
            // A sampled key's row stops where its own recording stopped; a key
            // with no recording of its own stops at the library's ceiling
            // carried down its own series.
            let row_reach = if is_sampled && r.measured_reach > 0 {
                r.measured_reach
            } else {
                reach_to(&partial_hz, drawn.ceiling_hz)
            };
            let correction =
                TailCorrection::from_band_falls(falls, row_reach).damped(DAMPING);
            if correction.is_empty() {
                continue;
            }
            // Item 291's rule: a key is declared drawn only where a drawn number
            // was actually written into its row.
            drawn_row[i] |= (0..2).any(|b| drew[b] && correction.band[b].is_some());
            preset.notes.partial_sigma_scale[i] =
                extend_row(&preset.notes.partial_sigma_scale[i], &partial_hz, &correction);
        }
        // The list is what the tables say at the end of the pass, not what this
        // pass happened to touch: a drawn key whose correction has converged to
        // one still carries a drawn row.
        preset.notes.synthesized_decay = keys
            .iter()
            .copied()
            .filter(|k| {
                let i = usize::from(k - FIRST_KEY);
                drawn_row[i] && !preset.notes.partial_sigma_scale[i].is_empty()
            })
            .collect();
        preset.validate()?;
        rows = measure(&preset, &keys)?;
        let gate = |lo: u8, hi: u8, b: usize| {
            median(
                rows.iter()
                    .filter(|r| (lo..=hi).contains(&r.key))
                    .map(|r| r.full_band[b].gap_db),
            )
        };
        println!(
            "  pass {pass}: band decay gap (whole band, engine - reference, dB) — \
             bass {:+.2}/{:+.2}  tenor {:+.2}/{:+.2}  treble {:+.2}/{:+.2}  top {:+.2}/{:+.2}",
            gate(21, 47, 0),
            gate(21, 47, 1),
            gate(48, 71, 0),
            gate(48, 71, 1),
            gate(72, 83, 0),
            gate(72, 83, 1),
            gate(84, 108, 0),
            gate(84, 108, 1),
        );
    }
    let rows = rows;
    if let Some(path) = &out {
        println!(
            "{} keys carry a decay row, {} of them drawn",
            preset.notes.partial_sigma_scale.iter().filter(|r| !r.is_empty()).count(),
            preset.notes.synthesized_decay.len()
        );
        preset.save(path)?;
        println!("wrote {}", path.display());
    }

    // ---- per key -----------------------------------------------------------
    //
    // `row` says where the key's row came from: `fit` for a key closed against
    // its own recording, `drawn` for one closed against `tail::DecayModel`, and
    // `-` for a key that carries no row at all. The two gap columns are the
    // gate's own statistic with the mark its floors leave on it: `≥` where the
    // *reference* side has fallen into the recording's own floor and the number
    // is a lower bound on the truth, `≤` where the engine has, `?` where both
    // have and it is not evidence in either direction.
    println!(
        "\n{:>4} {:>5} {:>6} {:>6} {:>6} {:>7} {:>7} {:>7} {:>7} {:>7} {:>6} {:>10} {:>10} {:>11} {:>11}",
        "key", "note", "bank", "reach", "meas", "top k", "top Hz", "c<=rch", "c>rch", "c>2kHz",
        "row", "gap 2-6k", "gap 6-12k", "stop 2-6k", "stop 6-12k",
    );
    // What each band's stop would say now, and where it came from: `m` for a
    // key closed against its own recording, `d` for one closed against the draw.
    // A ratio at or under one is a band this table has done what it can for, and
    // this column is where a key that stopped early says so.
    let stop = |r: &KeyRow, b: usize| -> String {
        let is_sampled = sampled.contains(&r.key);
        let i = usize::from(r.key - FIRST_KEY);
        let (falls, drew) = band_falls(
            r,
            &model.draw(r.key),
            is_sampled,
            !is_sampled || stage_owns[i],
        );
        falls[b].map_or_else(
            || "        -".to_string(),
            |f| format!("{:>8.2}{}", f.partial_median_ratio, if drew[b] { "d" } else { "m" }),
        )
    };
    for r in &rows {
        let measured: Vec<&PartialTail> =
            r.tails.iter().filter(|t| t.correction().is_some()).collect();
        let top = measured.last();
        let provenance = if preset.notes.synthesized_decay.contains(&r.key) {
            "drawn"
        } else if r.reach > 0 {
            "fit"
        } else {
            "-"
        };
        println!(
            "{:>4} {:>5} {:>6} {:>6} {:>6} {:>7} {:>7.0} {:>7} {:>7} {:>7} {:>6} {:>9.2}{} {:>9.2}{} {:>11} {:>11}",
            r.key,
            note_name(r.key),
            r.tails.len(),
            r.reach,
            measured.len(),
            top.map_or(0, |t| t.k),
            top.map_or(0.0, |t| t.hz),
            fmt(median(
                measured
                    .iter()
                    .filter(|t| t.k <= r.reach)
                    .filter_map(|t| t.correction())
            )),
            fmt(median(
                measured
                    .iter()
                    .filter(|t| t.k > r.reach)
                    .filter_map(|t| t.correction())
            )),
            fmt(median(
                measured
                    .iter()
                    .filter(|t| t.hz >= 2_000.0)
                    .filter_map(|t| t.correction())
            )),
            provenance,
            r.full_band[0].gap_db,
            r.full_band[0].mark(),
            r.full_band[1].gap_db,
            r.full_band[1].mark(),
            stop(r, 0),
            stop(r, 1),
        );
    }

    // ---- where the law departs, by register and by band --------------------
    let registers: [(&str, u8, u8); 4] = [
        ("A0-B2 bass", 21, 47),
        ("C3-B4 tenor", 48, 71),
        ("C5-B5 treble", 72, 83),
        ("C6-C8 top", 84, 108),
    ];
    let bands: [(&str, f64, f64); 4] = [
        ("under 1k", 0.0, 1_000.0),
        ("1-2k", 1_000.0, 2_000.0),
        ("2-6k", 2_000.0, 6_000.0),
        ("6-12k", 6_000.0, 12_000.0),
    ];
    // Two rows per register, because most of the 6-12 kHz evidence is a bound
    // and not a measurement: a partial whose *late* reading is inside its own
    // signal's floor contributes a fall that is a lower bound, so the ratio it
    // reports is bounded too and the median of a set of bounds is not a
    // measurement of the population's median. The second row keeps only the
    // partials both of whose late readings are real (`DECISIONS.md` 319).
    println!(
        "\nmedian fall(recording)/fall(engine) over 0.1-1 s per register and band — over 1 means the engine rings too long"
    );
    println!("  `bounded` is every trusted partial; `measured` keeps only those whose late readings are both real");
    print!("{:<14}{:<10}", "register", "cells");
    for (name, _, _) in bands {
        print!("{:>16}", name);
    }
    println!("{:>10}", "all");
    for (name, lo, hi) in registers {
        for (label, measured_only, group) in [
            ("bounded", false, 2u8),
            ("measured", true, 2),
            ("meas fit", true, 0),
            ("meas drawn", true, 1),
        ] {
            print!("{name:<14}{label:<10}");
            let cells = |blo: f64, bhi: f64| -> Vec<f64> {
                rows.iter()
                    .filter(|r| (lo..=hi).contains(&r.key))
                    .filter(|r| {
                        let drawn = preset.notes.synthesized_decay.contains(&r.key);
                        group == 2 || (group == 1) == drawn
                    })
                    .flat_map(|r| r.tails.iter())
                    .filter(|t| t.hz >= blo && t.hz < bhi)
                    .filter(|t| {
                        !measured_only
                            || (!t.engine_db.late_is_bound() && !t.reference_db.late_is_bound())
                    })
                    .filter_map(|t| t.correction())
                    .collect()
            };
            for (_, blo, bhi) in bands {
                let sel = cells(blo, bhi);
                print!(
                    "{:>16}",
                    if sel.is_empty() {
                        "        -    ".to_string()
                    } else {
                        format!("{:.2} (n {})", median(sel.iter().copied()), sel.len())
                    }
                );
            }
            println!("{:>10}", fmt(median(cells(0.0, 1e9).into_iter())));
        }
    }

    // ---- the partials against the band they live in ------------------------
    //
    // The statistic the gate is scored on is a **band**'s decay, and a band
    // holds more than this note's partials: the board's diffuse field, the
    // strike noise, the sympathetic halo and whatever the comb floor put
    // between them. Counting the same two instants over the partial bins alone
    // and over the whole band says how much of the gap the partials can
    // possibly own.
    //
    // `head e/r` is how far each signal's own late reading stands over that
    // signal's own late-time floor. Under `FLOOR_MARGIN_DB` the reading is the
    // floor's and the gap beside it is a bound: this is the discipline item 304
    // put on every partial and left off the band, which is how a reference side
    // sitting inside its own floor came to be read as an engine overshoot
    // (`DECISIONS.md` 319).
    println!("\n2-6k / 6-12k band decay gap over 0.1-1 s (dB, engine - reference)");
    println!(
        "{:<14} {:>18} {:>18} {:>18} {:>18} {:>18} {:>14}",
        "register", "partials only", "whole band", "layer floor |.|", "head e 2-6k/6-12k",
        "head r 2-6k/6-12k", "measured cells"
    );
    for (name, lo, hi) in registers {
        let sel = || rows.iter().filter(|r| (lo..=hi).contains(&r.key));
        let measured_gap = |b: usize| -> String {
            let v: Vec<f64> = sel()
                .filter(|r| r.full_band[b].is_measured())
                .map(|r| r.full_band[b].gap_db)
                .collect();
            if v.is_empty() {
                "  -".to_string()
            } else {
                format!("{:+.2} (n {})", median(v.iter().copied()), v.len())
            }
        };
        println!(
            "{name:<14} {:>8.2} {:>8.2}  {:>7.2}{} {:>7.2}{}  {:>8.2} {:>8.2}  {:>8.1} {:>8.1}  {:>8.1} {:>8.1}  {:>13} {:>13}",
            median(sel().map(|r| r.partial_band_gap[0])),
            median(sel().map(|r| r.partial_band_gap[1])),
            median(sel().map(|r| r.full_band[0].gap_db)),
            bound_mark(sel().map(|r| r.full_band[0]).collect()),
            median(sel().map(|r| r.full_band[1].gap_db)),
            bound_mark(sel().map(|r| r.full_band[1]).collect()),
            median(sel().map(|r| r.layer_band_gap[0].abs())),
            median(sel().map(|r| r.layer_band_gap[1].abs())),
            median(sel().map(|r| r.full_band[0].engine_headroom_db)),
            median(sel().map(|r| r.full_band[1].engine_headroom_db)),
            median(sel().map(|r| r.full_band[0].reference_headroom_db)),
            median(sel().map(|r| r.full_band[1].reference_headroom_db)),
            measured_gap(0),
            measured_gap(1),
        );
    }

    // ---- the fitted keys against the drawn ones ----------------------------
    //
    // The seam item 320 measured: a key closed against its own recording and a
    // key closed against a drawn target are the same statistic on the same
    // register, so a difference between them is this stage's own doing and not
    // the piano's. Reported per register and per band, over the whole band.
    println!("\nthe fitted keys against the drawn ones, whole-band gap (dB, engine - reference)");
    println!(
        "{:<14} {:>22} {:>22} {:>22}",
        "register", "fitted 2-6k / 6-12k", "drawn 2-6k / 6-12k", "no row 2-6k / 6-12k"
    );
    for (name, lo, hi) in registers {
        let group = |which: u8| -> String {
            let sel: Vec<&KeyRow> = rows
                .iter()
                .filter(|r| (lo..=hi).contains(&r.key))
                .filter(|r| {
                    let drawn = preset.notes.synthesized_decay.contains(&r.key);
                    match which {
                        0 => r.reach > 0 && !drawn,
                        1 => drawn,
                        _ => r.reach == 0,
                    }
                })
                .collect();
            if sel.is_empty() {
                return format!("{:>22}", "-");
            }
            format!(
                "{:>10}",
                format!(
                    "{:+.2} / {:+.2} (n {})",
                    median(sel.iter().map(|r| r.full_band[0].gap_db)),
                    median(sel.iter().map(|r| r.full_band[1].gap_db)),
                    sel.len()
                )
            )
        };
        println!(
            "{name:<14} {:>22} {:>22} {:>22}",
            group(0),
            group(1),
            group(2)
        );
    }

    // ---- inside and outside the fitted reach -------------------------------
    println!("\nthe seam: the same statistic below and above each key's own fitted reach");
    for (name, lo, hi) in registers {
        let sel = |above: bool| -> Vec<f64> {
            rows.iter()
                .filter(|r| (lo..=hi).contains(&r.key))
                .flat_map(|r| r.tails.iter().map(move |t| (r.reach, t)))
                .filter(|(reach, t)| (t.k > *reach) == above)
                .filter_map(|(_, t)| t.correction())
                .collect()
        };
        let (inside, outside) = (sel(false), sel(true));
        println!(
            "{name:<14} inside {:>6} (n {:>4})   outside {:>6} (n {:>4})",
            fmt(median(inside.iter().copied())),
            inside.len(),
            fmt(median(outside.iter().copied())),
            outside.len(),
        );
    }
    Ok(())
}

/// The bound the *median* cell of a register's column carries, so that the mark
/// beside a median gap is read off the same cells the median is.
fn bound_mark(cells: Vec<BandDecay>) -> &'static str {
    BandDecay {
        gap_db: f64::NAN,
        engine_headroom_db: median(cells.iter().map(|c| c.engine_headroom_db)),
        reference_headroom_db: median(cells.iter().map(|c| c.reference_headroom_db)),
    }
    .mark()
}

fn fmt(v: f64) -> String {
    if v.is_finite() {
        format!("{v:.2}")
    } else {
        "-".to_string()
    }
}
