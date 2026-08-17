//! The realism scoreboard: how far the engine is from a real piano, on a fixed
//! set of phrases, measured the way `TUNING.md`'s stage 2 will measure it.
//!
//! Six phrases (`piano_tuner::realism::phrase_set`) are rendered twice from
//! **one event list** — once through the engine on `presets/salamander-c5.toml`,
//! once through the Salamander recordings played by `piano_tuner::sampler` —
//! and every metric in `piano_tuner::realism` is run over the pair.
//!
//! Every distance is also measured a third time, between the reference and
//! *itself played out of the neighbouring velocity layer*. That pair is two
//! recordings of the same piano playing the same music, so whatever it reads is
//! the metric's own noise: a difference the engine shows that is smaller than
//! it says nothing at all. Both numbers go into `REALISM.md` side by side, and
//! the second one is what makes the first readable.
//!
//! Outputs, all into `renders/realism/`:
//!
//! * `<phrase>_engine.wav`, `<phrase>_reference.wav` — the level-matched pair.
//! * `<phrase>_mel.png` — engine, reference and their signed difference as
//!   log-mel spectrograms on one page.
//! * `REALISM.md` — the scoreboard.
//!
//! ```text
//! cargo run --release -p piano-tuner -- bench \
//!     [data/salamander] [renders/realism] [preset.toml]
//! ```

use std::cell::RefCell;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rayon::prelude::*;

use piano_emulator::preset::Preset;
use piano_emulator::render::{render_to_buffer, RenderEvent};
use piano_emulator::types::Event;
use piano_tuner::audio::Audio;
use piano_tuner::cache;
use piano_tuner::realism::{
    self, MelDiff, MotionCell, MotionColumns, Phrase, RealismMetrics, RecordedKeys, ReleaseDelta,
    StereoColumn, StereoItem, VelocityLayers, A1_GATE, A2_GATE, B1_GATE_DB, B2_GATE, MEL_BANDS,
    MEL_FLOOR_DB, MEL_F_MAX, MEL_F_MIN, MOTION_KEYS, MOTION_PARTIALS, MOTION_REFERENCE_VELOCITY,
    MOTION_VELOCITIES, MULTI_RES_WINDOWS, PHRASE_SET_VERSION, STEREO_ALLOWANCE,
    STEREO_BAND_FLOOR_DB, STEREO_MAX_LAG_S,
};
use piano_tuner::sampler::{
    engine_events, Sampler, SamplerConfig, SamplerEvent, TimedEvent, SAMPLER_VERSION,
};
use piano_tuner::{audio, detect_onset};
use piano_tuner::{SampleLibrary, SAMPLE_RATE};

/// The preset the engine is voiced from. The measured one: the whole point is
/// how close the *estimated* instrument is to the piano it was estimated from.
const DEFAULT_PRESET: &str = "presets/salamander-c5.toml";

thread_local! {
    /// One reference player per worker thread.
    ///
    /// [`Sampler::render`] reseeds its round-robin draw from the config at the
    /// top of every call and its decoded-buffer cache is a speed device nothing
    /// reads, so a phrase rendered on a fresh player is the same bytes as the
    /// same phrase rendered sixth on a shared one. The `clear_cache()` this
    /// replaces was there to stop six phrases' worth of recordings — a few
    /// gigabytes — piling up; per phrase and per thread it is the same bound.
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
        // Each phrase touches a few dozen recordings; keeping every one of them
        // decoded across all six would be gigabytes for no gain.
        sampler.clear_cache();
        out
    })
}

pub fn run(args: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
    let mut args = args.into_iter();
    let data = PathBuf::from(args.next().unwrap_or_else(|| "data/salamander".into()));
    let out = PathBuf::from(args.next().unwrap_or_else(|| "renders/realism".into()));
    let preset_path = PathBuf::from(args.next().unwrap_or_else(|| DEFAULT_PRESET.into()));

    let sfz = data.join("SalamanderGrandPiano-V3+20200602.sfz");
    if !sfz.exists() {
        eprintln!(
            "the reference piano is not here: {}\nrun data/fetch_salamander.sh first (707 MiB).",
            sfz.display()
        );
        std::process::exit(2);
    }
    std::fs::create_dir_all(&out)?;

    let preset = Preset::load(&preset_path)?;
    let library = SampleLibrary::from_sfz(&sfz)?;
    let layers = VelocityLayers::from_library(&library)?;
    let recorded = RecordedKeys::from_library(&library)?;
    let sample_rate = f64::from(SAMPLE_RATE);

    // The same recordings, with every key the library did **not** sample played
    // from its other neighbour's take instead of its nearest one. Both are
    // legitimate reconstructions of a note nobody recorded, so the distance
    // between them is how much of "the reference" at those keys is the resampler
    // (`DECISIONS.md` 329).
    let rerouted = Sampler::new(&sfz)?
        .instrument()
        .rerouted(FIRST_KEY..=LAST_KEY, recorded.routing());

    // The reference and its noise-floor partner are a function of the sampler,
    // the SFZ and the phrase set, none of which move when the engine does — so
    // they are cached to disk and an iteration on the engine pays for the engine
    // side alone. The key carries all three (`piano_tuner::cache`).
    let reference_cache = cache::reference_dir(&data);
    let mut reference_key = cache::Fingerprint::new();
    reference_key
        .str("realism-bench/reference")
        .u64(u64::from(SAMPLER_VERSION))
        .file(&sfz)?
        .u64(u64::from(SAMPLE_RATE))
        .u64(u64::from(PHRASE_SET_VERSION));

    println!(
        "phrase set v{PHRASE_SET_VERSION}, engine on {}, reference {}",
        preset_path.display(),
        sfz.display()
    );
    println!("reference cache: {}", reference_cache.display());

    // A phrase is an independent measurement — two renders, a level match and a
    // page of pictures — so the six run across the cores. Every line they print
    // and every row they contribute is emitted below in phrase-set order, so the
    // console and `REALISM.md` read the same at any thread count.
    let phrases: Vec<Phrase> = realism::phrase_set();
    let done: Vec<(Row, String)> = phrases
        .into_par_iter()
        // Errors cross a thread boundary here, so they travel as their printed
        // form: every one of them ends up in front of a person anyway.
        .map(|phrase| -> Result<(Row, String), String> {
            let started = Instant::now();

            let engine_raw = render_engine(&preset, &phrase);
            let cached = |name: &str, events: &[TimedEvent]| -> Result<Audio, piano_tuner::Error> {
                let mut key = reference_key;
                key.str(name).str(phrase.name).f64(phrase.duration_s);
                let path = reference_cache.join(format!(
                    "realism-{}-{name}-{}.wav",
                    phrase.name,
                    key.hex()
                ));
                cache::audio(&path, || {
                    with_sampler(&sfz, |s| s.render(events, phrase.duration_s))
                })
            };
            let say = |e: piano_tuner::Error| e.to_string();
            let reference_raw = cached("reference", &phrase.events).map_err(say)?;
            // The noise-floor partner: the same music out of the layer next door.
            let alt_raw = cached("alt-layer", &layers.shift(&phrase.events)).map_err(say)?;
            // The transposition partner: the same music, the same recordings,
            // the other route onto every key nobody recorded.
            let rerouted_raw = {
                let mut key = reference_key;
                key.str("rerouted").str(phrase.name).f64(phrase.duration_s);
                let path = reference_cache.join(format!(
                    "realism-{}-rerouted-{}.wav",
                    phrase.name,
                    key.hex()
                ));
                cache::audio(&path, || {
                    let mut player =
                        Sampler::from_instrument(rerouted.clone(), SamplerConfig::default());
                    player.render(&phrase.events, phrase.duration_s)
                })
                .map_err(say)?
            };

            let (engine, reference) =
                realism::level_match(&engine_raw, &reference_raw).map_err(say)?;
            let (reference_b, alt) = realism::level_match(&reference_raw, &alt_raw).map_err(say)?;
            let (reference_c, rerouted_matched) =
                realism::level_match(&reference_raw, &rerouted_raw).map_err(say)?;

            engine
                .write_wav(out.join(format!("{}_engine.wav", phrase.name)))
                .map_err(say)?;
            reference
                .write_wav(out.join(format!("{}_reference.wav", phrase.name)))
                .map_err(say)?;

            let ons = phrase.note_on_times();
            let offs = phrase.note_off_times();
            let measured =
                realism::compare(&engine.mono(), &reference.mono(), sample_rate, &ons, &offs)
                    .map_err(say)?;
            let floor =
                realism::compare(&alt.mono(), &reference_b.mono(), sample_rate, &ons, &offs)
                    .map_err(say)?;
            let transposition = realism::compare(
                &rerouted_matched.mono(),
                &reference_c.mono(),
                sample_rate,
                &ons,
                &offs,
            )
            .map_err(say)?;
            let (struck, transposed_notes) = phrase_keys(&phrase, &recorded);

            // The STEREO columns. Measured on the two channels rather than on
            // the mono sum every metric above is computed from, which is the
            // whole reason they exist (`DECISIONS.md` 314, 317 (a)). The
            // level-matched renders are used so that the mid/side ratio — the
            // one stereo number that is *not* invariant to a gain — is read on
            // the same signals the mono columns are.
            let stereo = StereoItem {
                label: phrase.name.to_string(),
                engine: realism::stereo_image_of(&engine).map_err(say)?,
                reference: realism::stereo_image_of(&reference).map_err(say)?,
                alternate: realism::stereo_image_of(&alt).map_err(say)?,
            };

            draw_page(
                &out.join(format!("{}_mel.png", phrase.name)),
                &phrase,
                &engine.mono(),
                &reference.mono(),
                sample_rate,
            )
            .map_err(|e| e.to_string())?;

            let line = format!(
                "{:<18}  mel {:5.2} dB (floor {:4.2}, transposition {:4.2})   mod {:5.2} dB \
(floor {:4.2})   {}/{} notes transposed   {:.1} s\n",
                phrase.name,
                measured.mel.mean,
                floor.mel.mean,
                transposition.mel.mean,
                measured.modulation.mean,
                floor.modulation.mean,
                transposed_notes,
                struck,
                started.elapsed().as_secs_f64()
            );
            Ok((
                Row {
                    phrase,
                    measured,
                    floor,
                    transposition,
                    stereo,
                    struck,
                    transposed_notes,
                },
                line,
            ))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut rows: Vec<Row> = Vec::with_capacity(done.len());
    for (row, line) in done {
        print!("{line}");
        rows.push(row);
    }

    // ---- Columns S: the stereo image ---------------------------------------
    // Arithmetic on images that were already measured beside every phrase, so
    // this costs nothing and is printed where a reader looks for it.
    let stereo_items: Vec<StereoItem> = rows.iter().map(|r| r.stereo.clone()).collect();
    let stereo_columns = realism::stereo_columns(&stereo_items);
    println!("\nSTEREO columns (not a mono sum) — interchannel r at lag 0, per band:");
    println!(
        "  {:>8}  {:>9}  {:>11}  {:>7}  {:>7}  {:>6}  {:>4}",
        "band", "engine", "reference", "|err|", "bar", "", "n"
    );
    for c in &stereo_columns {
        println!(
            "  {:>8}  {:+9.3}  {:+11.3}  {:7.3}  {:7.3}  {:>6}  {:4}",
            c.name,
            c.engine_r0,
            c.reference_r0,
            c.error,
            c.bar,
            if c.pass { "pass" } else { "RED" },
            c.items
        );
    }

    // ---- Columns A and B: the sixteen cells, at three velocities -----------
    print!("{:<18}", "motion columns");
    let started = Instant::now();
    let library = SampleLibrary::from_sfz(&sfz)?;
    let cells = motion_cells(&preset, &library);
    let columns = realism::motion_columns(&cells);
    println!(
        "  A1 {:5.2}   A2 {:5.2}   B1 {:5.2} dB   B2 {:5.3}   {:.1} s",
        columns.if_mismatch,
        columns.if_placement,
        columns.beat_depth_error_db,
        columns.velocity_coherence,
        started.elapsed().as_secs_f64()
    );

    let report = out.join("REALISM.md");
    std::fs::write(&report, scoreboard(&rows, &columns, &preset_path, &sfz))?;
    println!("\n{}", report.display());
    Ok(())
}

struct Row {
    phrase: Phrase,
    measured: RealismMetrics,
    floor: RealismMetrics,
    /// The reference against **itself, transposed the other way**: the same
    /// recordings reaching every unrecorded key from its second-nearest take
    /// instead of its nearest. `DECISIONS.md` 329.
    transposition: RealismMetrics,
    /// The phrase's stereo image, three ways: engine, reference and the
    /// reference's neighbouring velocity layer. Not a mono sum and marked
    /// STEREO everywhere it is printed.
    stereo: StereoItem,
    /// How many of the phrase's strikes there are, and how many of them land on
    /// a key the library never recorded.
    struck: usize,
    transposed_notes: usize,
}

/// The lowest and highest MIDI key of an 88-key piano.
const FIRST_KEY: u8 = 21;
const LAST_KEY: u8 = 108;

/// How many notes a phrase strikes, and how many of those are keys the library
/// plays by transposing a neighbour's recording.
fn phrase_keys(phrase: &Phrase, recorded: &RecordedKeys) -> (usize, usize) {
    let keys: Vec<u8> = phrase
        .events
        .iter()
        .filter_map(|e| match e.event {
            SamplerEvent::NoteOn { key, vel } if vel > 0 => Some(key),
            _ => None,
        })
        .collect();
    let transposed = keys.iter().filter(|&&k| !recorded.is_recorded(k)).count();
    (keys.len(), transposed)
}

// ---------------------------------------------------------------------------
// Driving the engine from the phrase set
// ---------------------------------------------------------------------------

fn render_engine(preset: &Preset, phrase: &Phrase) -> Audio {
    let (left, right) = render_to_buffer(
        preset,
        &engine_events::to_render_events(&phrase.events),
        phrase.duration_s as f32,
    );
    Audio::new(SAMPLE_RATE, vec![left, right]).expect("the engine renders stereo")
}

// ---------------------------------------------------------------------------
// The scoreboard
// ---------------------------------------------------------------------------

/// Every Column A/B cell, measured on the engine and on the recording layer a
/// strike at the same velocity would trigger.
/// Every `(key, velocity)` pair is one render and one recording, measured the
/// same way and sharing nothing, so the grid runs across the cores; `flat_map`
/// over an indexed parallel iterator keeps the cells in the order
/// [`realism::motion_columns`] reads them.
fn motion_cells(preset: &Preset, library: &SampleLibrary) -> Vec<MotionCell> {
    let sample_rate = f64::from(SAMPLE_RATE);
    MOTION_KEYS
        .par_iter()
        .flat_map_iter(|&(key, _)| MOTION_VELOCITIES.iter().map(move |&v| (key, v)))
        .flat_map(|(key, velocity)| {
            let params = preset.string_params(key);
            let partial_hz: Vec<f64> = (1..=MOTION_PARTIALS)
                .map(|k| f64::from(params.partial_freq(k as usize)))
                .collect();
            let engine = measure_render(preset, key, velocity, &partial_hz);
            let reference = library
                .layers(key)
                .iter()
                .find(|s| (s.lovel..=s.hivel).contains(&velocity))
                .and_then(|sample| {
                    let audio = audio::load_at(&sample.path, SAMPLE_RATE).ok()?;
                    let mono = audio.mono();
                    let onset = detect_onset(&mono, sample_rate);
                    let start = (onset * sample_rate).round() as usize;
                    let frames = (MOTION_RENDER_S * sample_rate) as usize;
                    let cut: Vec<f64> = (0..frames)
                        .map(|n| f64::from(mono.get(start + n).copied().unwrap_or(0.0)))
                        .collect();
                    Some(realism::measure_partials(&cut, &partial_hz))
                })
                .unwrap_or_else(|| vec![None; partial_hz.len()]);
            (1..=MOTION_PARTIALS)
                .map(|k| MotionCell {
                    key,
                    k,
                    velocity,
                    engine: engine[k as usize - 1],
                    reference: reference[k as usize - 1],
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Seconds of note the motion columns are measured over: past the analysis
/// window with room for the band-pass's own tail.
const MOTION_RENDER_S: f64 = 4.5;
const MOTION_PREROLL_S: f64 = 0.05;

fn measure_render(
    preset: &Preset,
    key: u8,
    velocity: u8,
    partial_hz: &[f64],
) -> Vec<Option<piano_tuner::motion::Motion>> {
    let events = [RenderEvent::new(
        MOTION_PREROLL_S as f32,
        Event::NoteOn {
            key,
            vel: u16::from(velocity),
        },
    )];
    let (left, right) =
        render_to_buffer(preset, &events, (MOTION_PREROLL_S + MOTION_RENDER_S) as f32);
    let skip = (MOTION_PREROLL_S * f64::from(SAMPLE_RATE)) as usize;
    let mono: Vec<f64> = left
        .iter()
        .zip(&right)
        .skip(skip)
        .map(|(&l, &r)| 0.5 * (f64::from(l) + f64::from(r)))
        .collect();
    realism::measure_partials(&mono, partial_hz)
}

fn scoreboard(rows: &[Row], columns: &MotionColumns, preset: &Path, sfz: &Path) -> String {
    let mut s = String::new();
    let _ = writeln!(
        s,
        "# REALISM.md — the engine against the piano it was measured from\n"
    );
    let _ = writeln!(
        s,
        "Written by `cargo run --release -p piano-tuner -- bench`. \
Six fixed phrases (set v{PHRASE_SET_VERSION}), each rendered twice from **one event list**: \
through the engine on `{}`, and through the recordings of the Yamaha C5 that preset was \
estimated from (`{}`), played by `piano_tuner::sampler`. Every pair is level-matched on \
whole-phrase RMS. **Every column of the scoreboard and of Columns A and B is measured on the \
mono sum**; the one section that is not is *Columns S*, which is marked STEREO throughout and \
is the only place in this file where the two channels are looked at separately.\n",
        preset.display(),
        sfz.display()
    );
    let _ = writeln!(
        s,
        "Each cell is **engine-vs-reference (noise floor)**. The floor is the same metric \
between the reference and *itself played out of the neighbouring velocity layer* — two \
recordings of the same piano playing the same music. A distance at or below its floor is not \
evidence of anything; the gap between the two numbers is the whole content of this file.\n"
    );
    let _ = writeln!(
        s,
        "**Every note stays in these renders and every note is in the mel score.** \
`DECISIONS.md` 328 takes transposed reference notes out of the *per-note* boards — \
`COMPASS.md`'s `match` column and the melody gate's bars — because a note the library never \
recorded cannot be a measurement of the piano at that key. A phrase distance is a different \
statistic: it is a whole performance against a whole performance, and dropping two notes in \
three from a piece of music would not make it more honest, it would make it a different \
piece. So the policy shows up here as a **number in the floor commentary instead of a hole \
in the table** — see *What the transposition costs*, below, which measures it rather than \
asserting it.\n"
    );

    // ---- the scoreboard ----
    let _ = writeln!(s, "## Scoreboard\n");
    let _ = writeln!(
        s,
        "| phrase | notes | mel dB | modulation dB | attack dB | r bass | r mid | r treble | release dB |"
    );
    let _ = writeln!(s, "|:--|--:|--:|--:|--:|--:|--:|--:|--:|");
    for r in rows {
        let _ = writeln!(
            s,
            "| `{}` | {} | {} | {} | {} | {} | {} | {} | {} |",
            r.phrase.name,
            r.phrase.note_count(),
            cell(r.measured.mel.mean, r.floor.mel.mean),
            cell(r.measured.modulation.mean, r.floor.modulation.mean),
            cell(r.measured.attack.mean_abs_db, r.floor.attack.mean_abs_db),
            cell3(r.measured.bands.r[0], r.floor.bands.r[0]),
            cell3(r.measured.bands.r[1], r.floor.bands.r[1]),
            cell3(r.measured.bands.r[2], r.floor.bands.r[2]),
            release_cell(&r.measured.release, &r.floor.release),
        );
    }
    let mean = |f: fn(&Row) -> f64| rows.iter().map(f).sum::<f64>() / rows.len() as f64;
    let _ = writeln!(
        s,
        "| **mean** | {} | {} | {} | {} | {} | {} | {} | {} |",
        rows.iter().map(|r| r.phrase.note_count()).sum::<usize>(),
        cell(mean(|r| r.measured.mel.mean), mean(|r| r.floor.mel.mean)),
        cell(
            mean(|r| r.measured.modulation.mean),
            mean(|r| r.floor.modulation.mean)
        ),
        cell(
            mean(|r| r.measured.attack.mean_abs_db),
            mean(|r| r.floor.attack.mean_abs_db)
        ),
        cell3(
            mean(|r| r.measured.bands.r[0]),
            mean(|r| r.floor.bands.r[0])
        ),
        cell3(
            mean(|r| r.measured.bands.r[1]),
            mean(|r| r.floor.bands.r[1])
        ),
        cell3(
            mean(|r| r.measured.bands.r[2]),
            mean(|r| r.floor.bands.r[2])
        ),
        cell(
            mean(|r| r.measured.release.mean_abs_db),
            mean(|r| r.floor.release.mean_abs_db)
        ),
    );
    let _ = writeln!(
        s,
        "\n`mel` is the multi-resolution log-mel distance (windows {:?}, {MEL_BANDS} mel bands \
{MEL_F_MIN:.0} Hz–{:.0} kHz, {MEL_FLOOR_DB:.0} dB range, mean |ΔdB|) — the number \
`TUNING.md` stage 2 minimises. `modulation` is the distance between the band envelopes' \
modulation spectra over 0.5–50 Hz. `attack` is the mean absolute difference in spectral \
tonality of the first 30 ms of every detected onset, each side windowed on **its own** \
strike (`DECISIONS.md` 338: the onsets are detected on the reference, and the sampler plays \
every recording from the file's own start, so an engine read at the reference's onset is \
read a median of 19 ms and a mean of 27 ms into its own attack). `r` is the Pearson \
correlation of the \
energy envelopes of bass (20–250 Hz), mid (250 Hz–2 kHz) and treble (2–16 kHz). `release` \
is the mean absolute level difference over the 0.5 s after every note-off nothing \
interrupts, with the number of such windows in brackets.\n",
        MULTI_RES_WINDOWS,
        MEL_F_MAX / 1000.0
    );

    // The check that would invalidate every other number in the file if it
    // came out wrong.
    let worst_lag = rows
        .iter()
        .map(|r| (r.phrase.name, r.measured.lag_s))
        .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
        .unwrap();
    let mean_lag = rows.iter().map(|r| r.measured.lag_s).sum::<f64>() / rows.len() as f64;
    let rise_engine =
        rows.iter().map(|r| r.measured.attack.rise_s.0).sum::<f64>() / rows.len() as f64;
    let rise_reference =
        rows.iter().map(|r| r.measured.attack.rise_s.1).sum::<f64>() / rows.len() as f64;
    let _ = writeln!(
        s,
        "**Alignment.** Both renders are driven by the same event list quantised to the same \
128-frame block, so every strike *begins* on the same sample by construction, and nothing in \
this file is a scheduling offset in disguise. The lag at which a pair's broadband energy \
envelope correlates best is nevertheless not zero: {:+.1} ms on average, {:+.1} ms at the \
extreme (`{}`), measured to an envelope frame of {:.1} ms — and the sign is the same \
everywhere, the engine's energy arriving **earlier**. Since the strikes coincide, that is \
envelope *shape*: some of it the attack, the rest the decay. The attack part is measurable \
on its own — the mean time from each side's own strike to the loudest part of the note is \
**{:.1} ms in the engine against {:.1} ms in the recordings** — and it accounts for only a \
few of those \
milliseconds, so most of the lead is the engine's energy leaving sooner rather than arriving \
sooner. That is the same story the `release` column tells, and it is a model difference \
rather than a bug.\n",
        mean_lag * 1000.0,
        worst_lag.1 * 1000.0,
        worst_lag.0,
        1000.0 * realism::ENVELOPE_HOP as f64 / f64::from(SAMPLE_RATE),
        rise_engine * 1000.0,
        rise_reference * 1000.0,
    );

    // ---- Columns S: the stereo image ----
    let stereo: Vec<StereoItem> = rows.iter().map(|r| r.stereo.clone()).collect();
    let stereo_columns = realism::stereo_columns(&stereo);
    let _ = writeln!(
        s,
        "## Columns S — the stereo image  *(STEREO, not a mono sum)*\n"
    );
    let _ = writeln!(
        s,
        "**Every other number in this file is a mono sum and these are not.** `DECISIONS.md` \
314 measured the interchannel behaviour of the recording against the engine's and found the \
largest single difference in the chain experiment sitting in a place no column could see: the \
recording's two channels are **+0.945 correlated below 125 Hz** and fall to about zero \
through the mid and treble, with a peak |r| of 0.57-0.65 at lags under two milliseconds — a \
spaced pair of microphones, two capsules inside a wavelength of each other in the bass seeing \
one wavefront and seeing the same sound about 60 % coherent a fraction of a millisecond apart \
above it. The engine was **inverted in every band**: the soundboard FDN's two opposite-sign \
output taps decorrelated the bass to −0.55 on these six phrases, and `soundboard::pan_for_key` \
scaled one mono voice into two channels, which is a pan-pot and correlated at +0.79 where the \
recording reads −0.07. Item 317 (a) asked for the loss to get this term *before* the \
two-microphone geometry of `PHYSICS.md` §8 was built, because a stage built to fix something \
nothing scores is a stage nobody can regress; that is what this section is. §8 is now built — \
`voicing.mics`, a spaced pair of virtual capsules over the string band with a per-source delay \
and gain and a frequency-dependent coherence on the board's diffuse field (`DECISIONS.md` \
351-358), its five numbers since fitted off the recording rather than swept (`DECISIONS.md` \
359-367) — and the sign of every band became the recording's. What that could not reach was \
125 Hz to 500 Hz, and *no* two-point geometry can: the recording's own +0.95 at 100 Hz and \
about zero one octave up is a step no spatial coherence function has. What closed it is the \
board's **mode-controlled band** (`[voicing.mics.modal]`, `DECISIONS.md` 368-377), measured \
off the recording's sixth-octave interchannel curve — +0.94 at 127 Hz, −0.53 at 180, back \
inside ±0.2 of zero above 500, and repeated by the same keys' other velocity layer. That is a \
plate whose modes put a nodal line between two capsules, not a pair of microphones. Since \
`DECISIONS.md` 379 the band is built as an **anti-phase copy of the pair's own sum** rather \
than as a gain on the board's difference, on the direct path as well as the board's, so it is \
there from a note's first sample instead of once the diffuse field has built — and the whole \
section was refitted at a window that opens where the note does, which is `DECISIONS.md` 378 \
and is the reason the numbers on this board moved. It is \
still **not a room**: §9 is refused by measurement in item 315 and is out of scope. These are \
six *phrases*; the per-key gate is `tuner/tests/stereo.rs` and `renders/stereo/STEREO.md` is \
the A/B you can listen to.\n"
    );
    let _ = writeln!(
        s,
        "`r@0` is the normalised interchannel correlation at lag zero, **signed** — a pan-pot \
pins it at +1 and an anti-phase pair reads −1, and a metric that reported |r| would call those \
two the same thing. `peak |r| @ lag` is the largest |r| over ±{:.0} ms and where it sits, \
positive meaning the *right* channel leads. `M/S` is 10·log10 of mid over side energy in the \
band, which is the same fact as `r@0` for a level-balanced pair and is not the same fact when \
the two channels differ in level. A band is read only where it holds more than \
{STEREO_BAND_FLOOR_DB:.0} dB under the whole signal's energy in **all three** of engine, \
reference and floor partner, so the three sides are always the same set; `n` is how many of \
the {} phrases survived that.\n",
        STEREO_MAX_LAG_S * 1000.0,
        rows.len()
    );
    let _ = write!(s, "{}", realism::stereo_report(&stereo_columns));
    let _ = writeln!(
        s,
        "\n`|err|` is |engine r@0 − reference r@0|, both **medians over the phrases**, and is \
the score. `floor` is that same distance between the reference and *itself played out of the \
neighbouring velocity layer* — the identical statistic on a second recording of the same \
music. `scatter` is the robust sigma of the reference's own r@0 across the phrases, and it is \
**not** the bar: material moving is a thing the engine is meant to reproduce, not to be \
excused from. It enters as `scatter/sqrt(n)`, the precision with which n phrases pin a \
median. The **bar** is the larger of those two times {STEREO_ALLOWANCE}, built out of the \
reference and never out of the engine — the property that makes it a bar rather than a \
description. `per-item |err| / floor` is the stricter, ungated statement: the median of the \
per-phrase distances, beside the same median between the two takes. A model that fixes a \
band's median without fixing its image shows up as a small `|err|` next to a large per-phrase \
one.\n\n`tuner/tests/stereo.rs` gates these same columns on **solo notes at the 30 recorded \
keys**, which is the material with a real take pair under it (a transposed key's two velocity \
layers are two resamplings of one take, not two takes — `DECISIONS.md` 328's argument). The \
phrases are the held-out reading; the notes are the gate.\n"
    );
    let reds: Vec<&StereoColumn> = stereo_columns.iter().filter(|c| !c.pass).collect();
    let _ = writeln!(
        s,
        "**{} of {} bands are red.**{}\n",
        reds.len(),
        stereo_columns.len(),
        if reds.is_empty() {
            String::new()
        } else {
            let worst = reds
                .iter()
                .max_by(|a, b| (a.error / a.bar).total_cmp(&(b.error / b.bar)))
                .expect("a red band");
            format!(
                " The worst is `{}`, {:.3} against a bar of {:.3} ({:.0}x), engine {:+.3} \
where the recording reads {:+.3}; its worst phrase is `{}`.",
                worst.name,
                worst.error,
                worst.bar,
                worst.error / worst.bar.max(1e-12),
                worst.engine_r0,
                worst.reference_r0,
                worst.worst.as_ref().map(|w| w.0.as_str()).unwrap_or("?"),
            )
        }
    );

    // ---- Columns A and B ----
    let _ = writeln!(s, "## Columns A and B — the motion of a single partial\n");
    let _ = writeln!(
        s,
        "The six columns above are functionals of *energy*, and `docs/history/FUNDAMENTALS.md` Part II is \
the argument that what the instrument still fails on is not one. These four are that review's \
own answer (§II.3), measured over {} key x partial cells — {} at partials 1..{} — at \
velocities {:?}, on single held notes rather than phrases. Every per-cell frequency deviation \
is clamped at the measurement's own 0.05-cent floor before any ratio is taken, which is the \
verification errata's pinned spec.\n",
        MOTION_KEYS.len() * MOTION_PARTIALS as usize,
        MOTION_KEYS
            .iter()
            .map(|&(_, name)| name)
            .collect::<Vec<_>>()
            .join(", "),
        MOTION_PARTIALS,
        MOTION_VELOCITIES
    );
    let _ = writeln!(
        s,
        "| column | what it measures | gate | reading | verdict |"
    );
    let _ = writeln!(s, "|:--|:--|--:|--:|:--|");
    let verdict = |pass: bool| if pass { "**pass**" } else { "fail" };
    let _ = writeln!(
        s,
        "| `A1` IF mismatch | geometric mean of `max(J_eng, J_ref) / min(...)`, symmetric so \
too dead fails as loudly as too spiky | ≤ {A1_GATE:.1} | {:.2} | {} |",
        columns.if_mismatch,
        verdict(columns.if_mismatch <= A1_GATE)
    );
    let _ = writeln!(
        s,
        "| `A2` IF placement | median of `L_eng / L_ref`: does the wobble ride the loud part \
of the partial or spike at a null | ≥ {A2_GATE:.1} | {:.2} | {} |",
        columns.if_placement,
        verdict(columns.if_placement >= A2_GATE)
    );
    let _ = writeln!(
        s,
        "| `B1` beat-depth error | mean absolute difference of the two beat depths, \
dB | ≤ {B1_GATE_DB:.0} dB | {:.2} | {} |",
        columns.beat_depth_error_db,
        verdict(columns.beat_depth_error_db <= B1_GATE_DB)
    );
    let _ = writeln!(
        s,
        "| `B2` velocity coherence | the engine's mean per-cell spread across the three \
velocities over the recording's, pooled over `J` and `D` | ≥ {B2_GATE:.2} | {:.3} | {} |",
        columns.velocity_coherence,
        verdict(columns.velocity_coherence >= B2_GATE)
    );
    let _ = writeln!(
        s,
        "\n`A1`, `A2` and `B1` are taken over the {} cells at velocity {}; `B2` over the {} of \
them that measured at all three velocities on both sides. What `B2` is made of, which is what \
says *which* half moved: frequency spread {:.3} cents engine against {:.3} recorded (ratio \
{:.3}), beat-depth spread {:.2} dB against {:.2} (ratio {:.3}).\n",
        columns.cells,
        MOTION_REFERENCE_VELOCITY,
        columns.velocity_cells,
        columns.spread_cents.0,
        columns.spread_cents.1,
        columns.velocity_coherence_freq,
        columns.spread_depth_db.0,
        columns.spread_depth_db.1,
        columns.velocity_coherence_depth
    );

    // ---- signed detail ----
    let _ = writeln!(s, "## Which way, and where\n");
    let _ = writeln!(
        s,
        "Signed, engine minus reference: positive means the engine has more of it.\n"
    );
    let _ = writeln!(
        s,
        "| phrase | worst band | ΔdB there | worst instant | ΔdB there | attack Δ | rise ms eng/ref | release Δ | worst modulation band | worst rate | envelope lag |"
    );
    let _ = writeln!(s, "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|");
    for r in rows {
        let d = r.measured.mel.detail();
        let (band_hz, band_db) = d.worst_band();
        let (time_s, time_db) = d.worst_time();
        let signed = signed_at(d, band_hz);
        let (mod_hz, mod_db) = r.measured.modulation.worst_band();
        let (rate_hz, rate_db) = r.measured.modulation.worst_rate();
        let _ = writeln!(
            s,
            "| `{}` | {} | {:+.2} ({:.2}) | {:.2} s | {:.2} | {:+.2} | {:.1} / {:.1} | {} | {} ({:.2}) | {:.1} Hz ({:.2}) | {:+.1} ms |",
            r.phrase.name,
            hz(band_hz),
            signed,
            band_db,
            time_s,
            time_db,
            r.measured.attack.mean_signed_db,
            r.measured.attack.rise_s.0 * 1000.0,
            r.measured.attack.rise_s.1 * 1000.0,
            match r.measured.release.windows {
                0 => "—".to_string(),
                _ => format!("{:+.2}", r.measured.release.mean_signed_db),
            },
            hz(mod_hz),
            mod_db,
            rate_hz,
            rate_db,
            r.measured.lag_s * 1000.0,
        );
    }

    // ---- resolution breakdown ----
    let _ = writeln!(s, "\n## The three resolutions\n");
    let _ = writeln!(
        s,
        "The multi-resolution distance is the mean of these. A phrase whose short-window \
column is the largest differs in its attacks; one whose long-window column is the largest \
differs in its partials.\n"
    );
    let _ = write!(s, "| phrase |");
    for w in MULTI_RES_WINDOWS {
        let _ = write!(s, " {w} |");
    }
    let _ = writeln!(s, " mean |");
    let _ = writeln!(s, "|:--|--:|--:|--:|--:|");
    for r in rows {
        let _ = write!(s, "| `{}` |", r.phrase.name);
        for res in &r.measured.mel.resolutions {
            let _ = write!(s, " {:.2} |", res.mean);
        }
        let _ = writeln!(s, " {:.2} |", r.measured.mel.mean);
    }

    // ---- the reading ----
    let _ = writeln!(s, "\n## Reading\n");
    s.push_str(&reading(rows));
    s.push_str(&transposition_reading(rows));

    // ---- phrases ----
    let _ = writeln!(s, "\n## The phrases\n");
    let _ = writeln!(s, "| phrase | s | notes | what it is for |");
    let _ = writeln!(s, "|:--|--:|--:|:--|");
    for r in rows {
        let _ = writeln!(
            s,
            "| `{}` | {:.0} | {} | {} |",
            r.phrase.name,
            r.phrase.duration_s,
            r.phrase.note_count(),
            r.phrase.description
        );
    }

    let _ = writeln!(
        s,
        "\nThe phrase set is fixed in `tuner/src/realism.rs` (`PHRASE_SET_VERSION = \
{PHRASE_SET_VERSION}`) and is versioned with it: a `REALISM.md` written at a different \
version is not comparable with this one, row for row.\n"
    );

    // ---- images ----
    let _ = writeln!(s, "## Images\n");
    let _ = writeln!(
        s,
        "`<phrase>_mel.png` is one page per phrase: engine, reference, and their signed \
difference, as log-mel spectrograms on a common colour scale ({MEL_FLOOR_DB:.0} dB under the \
loudest cell of the pair). The difference panel is blue where the engine is quieter than the \
piano and red where it is louder, saturating at ±18 dB; a panel that is mostly grey is a \
phrase the engine gets right.\n"
    );

    // ---- regenerate ----
    let _ = writeln!(s, "## Regenerating\n");
    let _ = writeln!(
        s,
        "```sh\ndata/fetch_salamander.sh          # once; 707 MiB into the gitignored data/\ncargo run --release -p piano-tuner -- bench \\\n    data/salamander renders/realism {}\n```\n",
        preset.display()
    );
    let _ = writeln!(
        s,
        "Everything in `renders/realism/` is rewritten in place. The format of this file is \
meant to be diffable: same phrases in the same order, same columns, fixed decimals — a \
change to the engine or the preset should show up as a column of numbers moving, and nothing \
else."
    );
    s
}

/// `value (floor)`, two decimals, for a distance in dB.
fn cell(value: f64, floor: f64) -> String {
    format!("{value:.2} ({floor:.2})")
}

/// `value (floor)`, three decimals, for a correlation.
fn cell3(value: f64, floor: f64) -> String {
    format!("{value:.3} ({floor:.3})")
}

fn release_cell(measured: &ReleaseDelta, floor: &ReleaseDelta) -> String {
    if measured.windows == 0 {
        return "— (0)".to_string();
    }
    format!(
        "{:.2} ({:.2}) ×{}",
        measured.mean_abs_db, floor.mean_abs_db, measured.windows
    )
}

fn hz(v: f64) -> String {
    if v >= 1000.0 {
        format!("{:.1} kHz", v / 1000.0)
    } else {
        format!("{v:.0} Hz")
    }
}

fn signed_at(diff: &MelDiff, centre_hz: f64) -> f64 {
    diff.centres_hz
        .iter()
        .position(|&c| (c - centre_hz).abs() < 1e-6)
        .map(|i| diff.signed_per_band[i])
        .unwrap_or(0.0)
}

/// What the reference's own transposition is worth, measured.
///
/// `DECISIONS.md` 329. Two keys in three on this library are played by
/// resampling a neighbour's take, and there is always a *second* neighbour that
/// could have been resampled instead — D4 is the D#4 take a semitone down, or
/// equally the C4 take two semitones up. Both are legitimate reconstructions of
/// a note nobody recorded. Rendering the whole phrase set both ways and
/// measuring the two against each other puts a number on how much of "the
/// reference" at those keys is the resampler rather than the piano, and it is a
/// number rather than an argument.
///
/// It is an **estimate of the ambiguity, not a bound on the error**: both routes
/// stretch, one by more than the other, so the disagreement between them is of
/// the same order as either one's own distance from the note that was never
/// recorded, but it is not that distance. Quoted as what it is.
fn transposition_reading(rows: &[Row]) -> String {
    let mut s = String::new();
    let mean = |f: fn(&Row) -> f64| rows.iter().map(f).sum::<f64>() / rows.len() as f64;
    let struck: usize = rows.iter().map(|r| r.struck).sum();
    let transposed: usize = rows.iter().map(|r| r.transposed_notes).sum();
    let _ = writeln!(s, "\n## What the transposition costs\n");
    let _ = writeln!(
        s,
        "Of the **{struck} strikes** in the phrase set, **{transposed}** ({:.0} %) land on a \
key the library never recorded and are played by resampling a neighbour's take. Each of those \
keys has a second take within reach that could have been resampled instead. The column below \
is the whole phrase set rendered **both ways through the same recordings** and measured \
against itself: same player, same event list, same level match, only the choice of which take \
to stretch is different.\n",
        100.0 * transposed as f64 / struck.max(1) as f64
    );
    let _ = writeln!(
        s,
        "| phrase | notes | transposed | mel: engine-vs-reference | velocity-layer floor | \
**transposition** | modulation: transposition |"
    );
    let _ = writeln!(s, "|:--|--:|--:|--:|--:|--:|--:|");
    for r in rows {
        let _ = writeln!(
            s,
            "| `{}` | {} | {} | {:.2} | {:.2} | **{:.2}** | {:.2} |",
            r.phrase.name,
            r.struck,
            r.transposed_notes,
            r.measured.mel.mean,
            r.floor.mel.mean,
            r.transposition.mel.mean,
            r.transposition.modulation.mean,
        );
    }
    let _ = writeln!(
        s,
        "| **mean** | {struck} | {transposed} | {:.2} | {:.2} | **{:.2}** | {:.2} |\n",
        mean(|r| r.measured.mel.mean),
        mean(|r| r.floor.mel.mean),
        mean(|r| r.transposition.mel.mean),
        mean(|r| r.transposition.modulation.mean),
    );
    let transposition = mean(|r| r.transposition.mel.mean);
    let floor = mean(|r| r.floor.mel.mean);
    let measured = mean(|r| r.measured.mel.mean);
    let _ = writeln!(
        s,
        "**{transposition:.2} dB of mel is the reference disagreeing with itself about notes \
nobody played.** Against a velocity-layer floor of {floor:.2} dB and an engine distance of \
{measured:.2} dB, that is {:.0} % of the number this scoreboard is minimised on and {:.1}x the \
floor the table quotes beside every cell. It is not subtracted from anything and it is not a \
correction: it is the size of the thing the per-note boards refuse to score against, stated \
where a reader of the phrase board can see it.\n",
        100.0 * transposition / measured.max(1e-9),
        transposition / floor.max(1e-9),
    );
    let _ = writeln!(
        s,
        "The two ends of the range say the same thing twice. `{}` — {} of its {} strikes \
transposed — reads {:.2} dB; `{}`, at {} of {}, reads {:.2}.\n",
        rows.iter()
            .max_by(|a, b| transposed_share(a).total_cmp(&transposed_share(b)))
            .map_or("-", |r| r.phrase.name),
        rows.iter()
            .max_by(|a, b| transposed_share(a).total_cmp(&transposed_share(b)))
            .map_or(0, |r| r.transposed_notes),
        rows.iter()
            .max_by(|a, b| transposed_share(a).total_cmp(&transposed_share(b)))
            .map_or(0, |r| r.struck),
        rows.iter()
            .max_by(|a, b| transposed_share(a).total_cmp(&transposed_share(b)))
            .map_or(0.0, |r| r.transposition.mel.mean),
        rows.iter()
            .min_by(|a, b| transposed_share(a).total_cmp(&transposed_share(b)))
            .map_or("-", |r| r.phrase.name),
        rows.iter()
            .min_by(|a, b| transposed_share(a).total_cmp(&transposed_share(b)))
            .map_or(0, |r| r.transposed_notes),
        rows.iter()
            .min_by(|a, b| transposed_share(a).total_cmp(&transposed_share(b)))
            .map_or(0, |r| r.struck),
        rows.iter()
            .min_by(|a, b| transposed_share(a).total_cmp(&transposed_share(b)))
            .map_or(0.0, |r| r.transposition.mel.mean),
    );
    s
}

fn transposed_share(row: &Row) -> f64 {
    row.transposed_notes as f64 / row.struck.max(1) as f64
}

/// The prose the scoreboard exists to support: where the biggest distances are,
/// stated as facts read off the tables above rather than as an opinion.
fn reading(rows: &[Row]) -> String {
    let mut s = String::new();

    // Excess over the floor is the honest ranking: a phrase can be far from the
    // reference simply because the reference is far from itself there.
    let mut ranked: Vec<(&Row, f64)> = rows
        .iter()
        .map(|r| (r, r.measured.mel.mean - r.floor.mel.mean))
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let _ = writeln!(
        s,
        "**Ranked by how far the spectral distance sits above its own noise floor**, which is \
the only ordering that means anything:\n"
    );
    for (place, (r, excess)) in ranked.iter().enumerate() {
        let d = r.measured.mel.detail();
        let (band_hz, band_db) = d.worst_band();
        let (time_s, _) = d.worst_time();
        let _ = writeln!(
            s,
            "{}. `{}` — {:.2} dB against a floor of {:.2}, an excess of **{:+.2} dB**. Worst \
band {} ({:.2} dB, engine {} the piano there); worst instant {:.2} s.",
            place + 1,
            r.phrase.name,
            r.measured.mel.mean,
            r.floor.mel.mean,
            excess,
            hz(band_hz),
            band_db,
            if signed_at(d, band_hz) > 0.0 {
                "above"
            } else {
                "below"
            },
            time_s,
        );
    }

    let worst_band = ranked[0].0.measured.mel.detail().worst_band();
    let worst_mod = rows
        .iter()
        .max_by(|a, b| {
            (a.measured.modulation.mean - a.floor.modulation.mean)
                .partial_cmp(&(b.measured.modulation.mean - b.floor.modulation.mean))
                .unwrap()
        })
        .unwrap();
    let worst_corr = rows
        .iter()
        .min_by(|a, b| {
            a.measured
                .bands
                .worst()
                .1
                .partial_cmp(&b.measured.bands.worst().1)
                .unwrap()
        })
        .unwrap();
    // At least three clean windows, or the "worst" is one accident of one tail.
    let worst_release = rows
        .iter()
        .filter(|r| r.measured.release.windows >= 3)
        .max_by(|a, b| {
            a.measured
                .release
                .mean_abs_db
                .partial_cmp(&b.measured.release.mean_abs_db)
                .unwrap()
        });

    let _ = writeln!(
        s,
        "\n**The three worst discrepancies, with where they are:**\n"
    );
    let _ = writeln!(
        s,
        "1. **Spectral — `{}` at {}.** {:.2} dB of mean absolute difference in that band \
against a whole-phrase distance of {:.2} dB; the engine is {} the piano there. The worst \
instant of the phrase is {:.2} s.",
        ranked[0].0.phrase.name,
        hz(worst_band.0),
        worst_band.1,
        ranked[0].0.measured.mel.mean,
        if signed_at(ranked[0].0.measured.mel.detail(), worst_band.0) > 0.0 {
            "louder than"
        } else {
            "quieter than"
        },
        ranked[0].0.measured.mel.detail().worst_time().0,
    );
    let (mb_hz, mb_db) = worst_mod.measured.modulation.worst_band();
    let (mr_hz, _) = worst_mod.measured.modulation.worst_rate();
    let _ = writeln!(
        s,
        "2. **Modulation — `{}` at {}, around {:.1} Hz.** {:.2} dB in that band against a \
floor of {:.2} dB for the whole phrase. This is the axis the timbre ladder found most \
diagnostic: it is how the level *moves*, which is beating, uneven decay and the liveliness \
no envelope model reproduces.",
        worst_mod.phrase.name,
        hz(mb_hz),
        mr_hz,
        mb_db,
        worst_mod.floor.modulation.mean,
    );
    let (cname, cval) = worst_corr.measured.bands.worst();
    let _ = writeln!(
        s,
        "3. **Envelope — `{}`, {} register.** The two renders' energy envelopes correlate \
{:.3} there, against {:.3} for the reference against itself. An envelope correlation below \
its floor is a decay-rate or a pedal-timing disagreement, not a timbre one.",
        worst_corr.phrase.name,
        cname,
        cval,
        worst_corr
            .floor
            .bands
            .r
            .get(
                worst_corr
                    .measured
                    .bands
                    .names
                    .iter()
                    .position(|&n| n == cname)
                    .unwrap()
            )
            .copied()
            .unwrap_or(0.0),
    );

    if let Some(r) = worst_release {
        let _ = writeln!(
            s,
            "\nThe releases: `{}` is the phrase whose tails disagree most — {:.2} dB mean \
absolute ({:+.2} dB signed) over {} clean half-second window{}, floor {:.2} dB.",
            r.phrase.name,
            r.measured.release.mean_abs_db,
            r.measured.release.mean_signed_db,
            r.measured.release.windows,
            if r.measured.release.windows == 1 {
                ""
            } else {
                "s"
            },
            r.floor.release.mean_abs_db,
        );
    }

    let attack_bias = rows
        .iter()
        .map(|r| r.measured.attack.mean_signed_db)
        .sum::<f64>()
        / rows.len() as f64;
    let _ = writeln!(
        s,
        "\nAcross all six phrases the engine's attacks read {:+.2} dB of spectral tonality \
against the piano's — {} than the recordings.",
        attack_bias,
        if attack_bias > 0.0 {
            "more tonal, i.e. less noisy"
        } else {
            "noisier"
        }
    );
    s
}

// ---------------------------------------------------------------------------
// Images
// ---------------------------------------------------------------------------

/// Mel bands drawn. Three pixel rows each, so the panel height is exact and no
/// band is drawn wider than its neighbour.
const IMAGE_BANDS: usize = MEL_BANDS;
const BAND_PX: usize = 3;
const PANEL_H: usize = IMAGE_BANDS * BAND_PX;
const PLOT_W: usize = 1080;
const MARGIN_L: usize = 76;
const MARGIN_R: usize = 132;
const MARGIN_T: usize = 40;
const PANEL_GAP: usize = 32;
const MARGIN_B: usize = 44;
/// Saturation of the difference panel, in dB.
const DIFF_RANGE_DB: f64 = 18.0;

const INK: [u8; 3] = [0xd6, 0xd8, 0xdd];
const DIM: [u8; 3] = [0x8a, 0x8e, 0x96];
const PAPER: [u8; 3] = [0x10, 0x11, 0x16];
const GRID: [u8; 3] = [0x3a, 0x3d, 0x46];

struct Canvas {
    w: usize,
    h: usize,
    px: Vec<u8>,
}

impl Canvas {
    fn new(w: usize, h: usize, fill: [u8; 3]) -> Self {
        let mut px = Vec::with_capacity(w * h * 3);
        for _ in 0..w * h {
            px.extend_from_slice(&fill);
        }
        Canvas { w, h, px }
    }

    fn set(&mut self, x: usize, y: usize, c: [u8; 3]) {
        if x >= self.w || y >= self.h {
            return;
        }
        let i = (y * self.w + x) * 3;
        self.px[i..i + 3].copy_from_slice(&c);
    }

    fn rect(&mut self, x: usize, y: usize, w: usize, h: usize, c: [u8; 3]) {
        for yy in y..(y + h).min(self.h) {
            for xx in x..(x + w).min(self.w) {
                self.set(xx, yy, c);
            }
        }
    }

    /// Uppercase 5x7 text. `scale` is an integer pixel multiplier.
    fn text(&mut self, x: usize, y: usize, s: &str, scale: usize, c: [u8; 3]) {
        let mut cx = x;
        for ch in s.chars() {
            let glyph = glyph(ch.to_ascii_uppercase());
            for (col, bits) in glyph.iter().enumerate() {
                for row in 0..7 {
                    if bits & (1 << row) != 0 {
                        self.rect(cx + col * scale, y + row * scale, scale, scale, c);
                    }
                }
            }
            cx += 6 * scale;
        }
    }

    /// Text right-aligned at `x`.
    fn text_right(&mut self, x: usize, y: usize, s: &str, scale: usize, c: [u8; 3]) {
        let w = s.chars().count() * 6 * scale;
        self.text(x.saturating_sub(w), y, s, scale, c);
    }

    fn write_png(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let file = std::fs::File::create(path)?;
        let mut encoder =
            png::Encoder::new(std::io::BufWriter::new(file), self.w as u32, self.h as u32);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.write_header()?.write_image_data(&self.px)?;
        Ok(())
    }
}

/// A 5x7 uppercase font, five columns per glyph, bit 0 the top row. Enough of
/// ASCII to label an axis; anything else draws as a blank.
fn glyph(ch: char) -> [u8; 5] {
    match ch {
        '!' => [0x00, 0x00, 0x5f, 0x00, 0x00],
        '(' => [0x00, 0x1c, 0x22, 0x41, 0x00],
        ')' => [0x00, 0x41, 0x22, 0x1c, 0x00],
        '+' => [0x08, 0x08, 0x3e, 0x08, 0x08],
        ',' => [0x00, 0x50, 0x30, 0x00, 0x00],
        '-' => [0x08, 0x08, 0x08, 0x08, 0x08],
        '.' => [0x00, 0x60, 0x60, 0x00, 0x00],
        '/' => [0x20, 0x10, 0x08, 0x04, 0x02],
        '0' => [0x3e, 0x51, 0x49, 0x45, 0x3e],
        '1' => [0x00, 0x42, 0x7f, 0x40, 0x00],
        '2' => [0x42, 0x61, 0x51, 0x49, 0x46],
        '3' => [0x21, 0x41, 0x45, 0x4b, 0x31],
        '4' => [0x18, 0x14, 0x12, 0x7f, 0x10],
        '5' => [0x27, 0x45, 0x45, 0x45, 0x39],
        '6' => [0x3c, 0x4a, 0x49, 0x49, 0x30],
        '7' => [0x01, 0x71, 0x09, 0x05, 0x03],
        '8' => [0x36, 0x49, 0x49, 0x49, 0x36],
        '9' => [0x06, 0x49, 0x49, 0x29, 0x1e],
        ':' => [0x00, 0x36, 0x36, 0x00, 0x00],
        '<' => [0x08, 0x14, 0x22, 0x41, 0x00],
        '>' => [0x00, 0x41, 0x22, 0x14, 0x08],
        'A' => [0x7e, 0x11, 0x11, 0x11, 0x7e],
        'B' => [0x7f, 0x49, 0x49, 0x49, 0x36],
        'C' => [0x3e, 0x41, 0x41, 0x41, 0x22],
        'D' => [0x7f, 0x41, 0x41, 0x22, 0x1c],
        'E' => [0x7f, 0x49, 0x49, 0x49, 0x41],
        'F' => [0x7f, 0x09, 0x09, 0x01, 0x01],
        'G' => [0x3e, 0x41, 0x49, 0x49, 0x7a],
        'H' => [0x7f, 0x08, 0x08, 0x08, 0x7f],
        'I' => [0x00, 0x41, 0x7f, 0x41, 0x00],
        'J' => [0x20, 0x40, 0x41, 0x3f, 0x01],
        'K' => [0x7f, 0x08, 0x14, 0x22, 0x41],
        'L' => [0x7f, 0x40, 0x40, 0x40, 0x40],
        'M' => [0x7f, 0x02, 0x0c, 0x02, 0x7f],
        'N' => [0x7f, 0x04, 0x08, 0x10, 0x7f],
        'O' => [0x3e, 0x41, 0x41, 0x41, 0x3e],
        'P' => [0x7f, 0x09, 0x09, 0x09, 0x06],
        'Q' => [0x3e, 0x41, 0x51, 0x21, 0x5e],
        'R' => [0x7f, 0x09, 0x19, 0x29, 0x46],
        'S' => [0x46, 0x49, 0x49, 0x49, 0x31],
        'T' => [0x01, 0x01, 0x7f, 0x01, 0x01],
        'U' => [0x3f, 0x40, 0x40, 0x40, 0x3f],
        'V' => [0x1f, 0x20, 0x40, 0x20, 0x1f],
        'W' => [0x7f, 0x20, 0x18, 0x20, 0x7f],
        'X' => [0x63, 0x14, 0x08, 0x14, 0x63],
        'Y' => [0x03, 0x04, 0x78, 0x04, 0x03],
        'Z' => [0x61, 0x51, 0x49, 0x45, 0x43],
        '_' => [0x40, 0x40, 0x40, 0x40, 0x40],
        _ => [0x00, 0x00, 0x00, 0x00, 0x00],
    }
}

/// A viridis-like sequential map: dark blue-purple through teal and green to
/// yellow. Perceptually ordered, and it survives being printed in grey.
fn viridis(t: f64) -> [u8; 3] {
    const STOPS: [[f64; 3]; 9] = [
        [68.0, 1.0, 84.0],
        [72.0, 40.0, 120.0],
        [62.0, 74.0, 137.0],
        [49.0, 104.0, 142.0],
        [38.0, 130.0, 142.0],
        [31.0, 158.0, 137.0],
        [53.0, 183.0, 121.0],
        [110.0, 206.0, 88.0],
        [253.0, 231.0, 37.0],
    ];
    ramp(&STOPS, t)
}

/// Diverging map for the signed difference: blue where the engine is quieter,
/// near-neutral grey at zero, red where it is louder.
fn diverging(t: f64) -> [u8; 3] {
    const STOPS: [[f64; 3]; 5] = [
        [38.0, 88.0, 190.0],
        [90.0, 140.0, 205.0],
        [60.0, 62.0, 70.0],
        [214.0, 128.0, 96.0],
        [190.0, 48.0, 40.0],
    ];
    ramp(&STOPS, t)
}

fn ramp(stops: &[[f64; 3]], t: f64) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0) * (stops.len() - 1) as f64;
    let i = (t.floor() as usize).min(stops.len() - 2);
    let f = t - i as f64;
    let mut out = [0u8; 3];
    for (c, slot) in out.iter_mut().enumerate() {
        *slot = (stops[i][c] + f * (stops[i + 1][c] - stops[i][c]))
            .round()
            .clamp(0.0, 255.0) as u8;
    }
    out
}

/// One page: engine, reference, and their signed difference.
fn draw_page(
    path: &Path,
    phrase: &Phrase,
    engine: &[f32],
    reference: &[f32],
    sample_rate: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let window = 1024usize;
    let hop = 256usize;
    let a = realism::mel_spectrogram(
        engine,
        sample_rate,
        window,
        hop,
        IMAGE_BANDS,
        MEL_F_MIN,
        MEL_F_MAX,
    )?;
    let b = realism::mel_spectrogram(
        reference,
        sample_rate,
        window,
        hop,
        IMAGE_BANDS,
        MEL_F_MIN,
        MEL_F_MAX,
    )?;
    let frames = a.frames.len().min(b.frames.len());
    let peak = a.peak_db().max(b.peak_db());
    let floor = peak + MEL_FLOOR_DB;
    let db = |e: f64| {
        if e > 0.0 {
            (10.0 * e.log10()).max(floor)
        } else {
            floor
        }
    };

    // Columns: the mean, in dB, of the frames that fall in each column. The
    // mean rather than the maximum, so that a column is the level over that
    // slice of time rather than the loudest thing in it.
    let column = |spec: &realism::MelSpec, x: usize, band: usize| -> f64 {
        let lo = x * frames / PLOT_W;
        let hi = (((x + 1) * frames) / PLOT_W).max(lo + 1).min(frames);
        let mut sum = 0.0;
        for t in lo..hi {
            sum += db(spec.frames[t][band]);
        }
        sum / (hi - lo) as f64
    };

    let width = MARGIN_L + PLOT_W + MARGIN_R;
    let height = MARGIN_T + 3 * (PANEL_GAP + PANEL_H) + MARGIN_B;
    let mut c = Canvas::new(width, height, PAPER);

    c.text(
        MARGIN_L,
        12,
        &format!(
            "{}  -  LOG-MEL {} BANDS {:.0} HZ TO {:.0} KHZ  -  WINDOW {} HOP {}",
            phrase.name.replace('_', " "),
            IMAGE_BANDS,
            MEL_F_MIN,
            MEL_F_MAX / 1000.0,
            window,
            hop
        ),
        2,
        INK,
    );

    let titles = [
        "ENGINE".to_string(),
        "REFERENCE  SALAMANDER C5".to_string(),
        format!("DIFFERENCE  ENGINE MINUS REFERENCE  PLUS/MINUS {DIFF_RANGE_DB:.0} DB"),
    ];
    for (panel, title) in titles.iter().enumerate() {
        let top = MARGIN_T + panel * (PANEL_GAP + PANEL_H) + PANEL_GAP;
        c.text(MARGIN_L, top - 22, title, 2, INK);
        for x in 0..PLOT_W {
            for band in 0..IMAGE_BANDS {
                let value = match panel {
                    0 => (column(&a, x, band) - floor) / (peak - floor),
                    1 => (column(&b, x, band) - floor) / (peak - floor),
                    _ => {
                        let d = column(&a, x, band) - column(&b, x, band);
                        0.5 + 0.5 * (d / DIFF_RANGE_DB).clamp(-1.0, 1.0)
                    }
                };
                let colour = if panel == 2 {
                    diverging(value)
                } else {
                    viridis(value)
                };
                // Band 0 is the bottom of the panel.
                let y = top + (IMAGE_BANDS - 1 - band) * BAND_PX;
                c.rect(MARGIN_L + x, y, 1, BAND_PX, colour);
            }
        }
        // Frequency ticks, placed by the band whose apex is nearest. The mel
        // scale packs the bottom two octaves into a handful of bands, so a tick
        // that would land within a glyph's height of the last one is dropped
        // rather than drawn on top of it.
        let mut last_y = usize::MAX;
        for &tick in &[
            50.0f64, 100.0, 200.0, 500.0, 1_000.0, 2_000.0, 4_000.0, 8_000.0,
        ] {
            let band = a
                .centres_hz
                .iter()
                .enumerate()
                .min_by(|x, y| (x.1 - tick).abs().partial_cmp(&(y.1 - tick).abs()).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            let y = top + (IMAGE_BANDS - 1 - band) * BAND_PX + BAND_PX / 2;
            if last_y != usize::MAX && last_y.saturating_sub(y) < 10 {
                continue;
            }
            last_y = y;
            c.rect(MARGIN_L - 5, y, 4, 1, DIM);
            c.text_right(MARGIN_L - 8, y.saturating_sub(3), &hz_label(tick), 1, DIM);
        }
        // Second ticks.
        let seconds = phrase.duration_s;
        let mut t = 0.0;
        while t <= seconds {
            let x = MARGIN_L + ((t / seconds) * PLOT_W as f64) as usize;
            if x < MARGIN_L + PLOT_W {
                for y in (top..top + PANEL_H).step_by(6) {
                    c.set(x, y, GRID);
                }
                if panel == 2 {
                    c.rect(x, top + PANEL_H + 1, 1, 4, DIM);
                    c.text(
                        x.saturating_sub(4),
                        top + PANEL_H + 8,
                        &format!("{t:.0}"),
                        1,
                        DIM,
                    );
                }
            }
            t += 2.0;
        }
    }
    c.text(
        MARGIN_L + PLOT_W / 2 - 30,
        MARGIN_T + 3 * (PANEL_GAP + PANEL_H) + 20,
        "SECONDS",
        1,
        DIM,
    );

    // Colour bars, in the right margin: one for the two spectrograms, one for
    // the difference.
    let bar_x = MARGIN_L + PLOT_W + 24;
    let bar_w = 14;
    let top_a = MARGIN_T + PANEL_GAP;
    draw_bar(
        &mut c,
        bar_x,
        top_a,
        bar_w,
        PANEL_H,
        viridis,
        &[
            (1.0, format!("{peak:.0} DB")),
            (0.5, format!("{:.0}", floor + 0.5 * (peak - floor))),
            (0.0, format!("{floor:.0}")),
        ],
    );
    let top_c = MARGIN_T + 2 * (PANEL_GAP + PANEL_H) + PANEL_GAP;
    draw_bar(
        &mut c,
        bar_x,
        top_c,
        bar_w,
        PANEL_H,
        diverging,
        &[
            (1.0, format!("+{DIFF_RANGE_DB:.0} DB")),
            (0.5, "0".to_string()),
            (0.0, format!("-{DIFF_RANGE_DB:.0} DB")),
        ],
    );

    c.write_png(path)?;
    Ok(())
}

fn hz_label(hz: f64) -> String {
    if hz >= 1000.0 {
        format!("{:.0}K", hz / 1000.0)
    } else {
        format!("{hz:.0}")
    }
}

fn draw_bar(
    c: &mut Canvas,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    map: fn(f64) -> [u8; 3],
    ticks: &[(f64, String)],
) {
    for row in 0..h {
        let t = 1.0 - row as f64 / (h - 1) as f64;
        c.rect(x, y + row, w, 1, map(t));
    }
    for (t, label) in ticks {
        let row = ((1.0 - t) * (h - 1) as f64).round() as usize;
        c.rect(x + w, y + row, 4, 1, DIM);
        c.text(x + w + 7, (y + row).saturating_sub(3), label, 1, DIM);
    }
}
