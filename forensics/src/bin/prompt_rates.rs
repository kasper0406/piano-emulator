//! What the **prompt** decay curve is actually fitted to, partial by partial.
//!
//! `NoteSurvey::decay_curve` does not read a partial's T60. It reads the
//! *fast* component of the two-exponential envelope fit — the prompt sound —
//! divides it by the engine's vertical split factor, and puts the result
//! through a weighted line in `(f/1000)^2`. Everything that decides the
//! `notes.sigma0` / `notes.sigma1` a key ships with is therefore in the four
//! numbers of `DecayFit::fast` and `DecayFit::slow`, and none of those are
//! printed by `piano-tuner estimate`, which reports the *whole-partial* T60
//! and `a(0)`.
//!
//! This prints them. Per partial: both components' rate and amplitude, the
//! fast component's own T60, its share of the amplitude the fit extrapolates
//! back to the strike, the envelope hop the fit had to resolve that T60 with
//! (`span_s / points`), and the two facts the curve's filter turns on — whether
//! the partial is admitted and what rate it would contribute. Then the curve
//! itself, from `survey::prompt_decay_curve` rather than from a copy of it.
//!
//! It was built to convict one of two degeneracies on `upright-parlour`'s G5,
//! whose shipped `notes.sigma1` is 3750 where every other key in every other
//! preset is under 0.53: either the fast component is a **slack term** with no
//! amplitude, whose rate means nothing, or it is a real component whose rate is
//! **shorter than the hop**, i.e. the analysis window rather than the string.
//!
//! **The geometry is half the question, so it is an argument.** `--geometry
//! survey` (the default) is what the factory runs: `SurveyConfig`'s per-note
//! window — twelve periods of the fundamental, so 4096 samples at G5, hopped
//! sixteen ways — with the library's tail trimmed and its length capped, which
//! is the analysis that wrote the preset. `--geometry estimate` is the fixed
//! 65536-sample window of the `piano-tuner estimate` subcommand. G5's vl1 layer
//! is healthy at the second and blown at the first, and that contrast is the
//! reason the subcommand could not find this defect.
//!
//! ```sh
//! cargo run --release -p forensics --bin prompt_rates -- \
//!     data/vcsl-knight-upright/Sustains/Player_vl1_rr1_G4.wav --f0 785.26
//! ```

use piano_tuner::estimate::decay::LN_1000;
use piano_tuner::pipeline::track_refined;
use piano_tuner::preset::Preset;
use piano_tuner::stft::StftConfig;
use piano_tuner::survey::{load_signal, prompt_decay_curve, SurveyConfig};
use piano_tuner::trajectory::InharmonicModel;
use piano_tuner::{
    analyze_trajectories, audio, NoteConfig, TrackerConfig, SAMPLE_RATE,
};

/// The transform geometry `piano-tuner estimate` uses, for `--geometry estimate`.
const ESTIMATE_WINDOW: usize = 1 << 16;
const ESTIMATE_HOP: usize = 480;
const ESTIMATE_PAD: usize = 2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut input: Option<String> = None;
    let mut f0: Option<f64> = None;
    let mut preset_path = "presets/default.toml".to_string();
    let mut survey_geometry = true;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--f0" => f0 = args.next().and_then(|v| v.parse().ok()),
            "--preset" => preset_path = args.next().unwrap_or(preset_path),
            "--geometry" => survey_geometry = args.next().as_deref() != Some("estimate"),
            other => input = Some(other.to_string()),
        }
    }
    let input = input.ok_or("usage: prompt_rates <file.wav> --f0 <hz> [--geometry estimate]")?;
    let f0 = f0.ok_or("--f0 is required")?;

    let survey = SurveyConfig::default();
    let (signal, note_config) = if survey_geometry {
        // Exactly `survey::analyze_sample`'s two steps, with the recording cut
        // the way the factory cuts it: a library fades its own tail, and a
        // decay fitted through the fade measures the fade.
        (
            load_signal(std::path::Path::new(&input), &survey)?,
            survey.note_config(f0)?,
        )
    } else {
        (
            audio::load_at(&input, SAMPLE_RATE)?.mono(),
            NoteConfig {
                tracker: TrackerConfig {
                    stft: StftConfig::padded(ESTIMATE_WINDOW, ESTIMATE_HOP, ESTIMATE_PAD)?,
                    max_partials: 80,
                    ..NoteConfig::default().tracker
                },
                ..NoteConfig::default()
            },
        )
    };
    let (trajectories, _) = track_refined(
        &signal,
        f64::from(SAMPLE_RATE),
        InharmonicModel::harmonic(f0),
        &note_config,
    )?;
    let analysis = analyze_trajectories(trajectories, &note_config)?;

    // The survey divides every prompt rate by this before fitting: the engine
    // builds both polarizations from one table entry and its global split. A
    // survey uses its own measured split when it has one; a single file has no
    // library behind it, so the base preset's is what is on offer.
    let base = Preset::load(&preset_path)?;
    let factor = base.voicing.vertical_decay_factor();
    let decay = note_config.decay;

    let stft = note_config.tracker.stft;
    println!(
        "{input}\n\
         {} geometry: window {} hop {} pad {}  |  {:.2} s analysed, onset {:.3} s, \
         {} partials fitted",
        if survey_geometry { "survey" } else { "estimate" },
        stft.window,
        stft.hop,
        stft.fft_size / stft.window,
        signal.len() as f64 / f64::from(SAMPLE_RATE),
        analysis.trajectories.onset_s,
        analysis.decays.partials.len()
    );
    println!(
        "vertical factor {factor:.4} from {preset_path}  \
         (max_t60_ratio {:.1}, min_split_ratio {:.0e})",
        decay.max_t60_ratio, decay.min_split_ratio
    );
    println!(
        "   k   frequency Hz   fast sigma   fast T60 s     fast amp      share |  \
         slow sigma     slow amp | span s   pts    hop s | slow  fast    rate in\n\
         (share = fast amp / a(0); hop s = span_s / points, the envelope's own resolution;\n\
         `slow`/`fast` are the curve's two guards: T60 inside max_t60_ratio spans, \
         T60 over one hop)"
    );
    for fit in &analysis.decays.partials {
        let a0 = fit.initial_amplitude();
        let share = if a0 > 0.0 {
            fit.fast.amplitude / a0
        } else {
            f64::NAN
        };
        let hop = if fit.points > 0 {
            fit.span_s / fit.points as f64
        } else {
            f64::NAN
        };
        // The curve's two filters, restated here so the row can be read against
        // its own verdict: a prompt decay slower than the record is long was
        // never seen to finish, and one shorter than the record's own hop was
        // never seen at all.
        let usable = fit.frequency_hz > 0.0 && fit.fast.sigma > 0.0;
        let not_too_slow = usable && LN_1000 / fit.fast.sigma <= decay.max_t60_ratio * fit.span_s;
        let not_too_fast = usable && fit.points > 0 && LN_1000 / fit.fast.sigma >= hop;
        println!(
            "{:4}  {:12.3}  {:11.4e}  {:11.4e}  {:11.4e}  {:9.3e} |  {:10.4}  {:11.4e} | \
             {:6.3}  {:4}  {:7.5} | {:>4}  {:>4}  {:10.4e}",
            fit.k,
            fit.frequency_hz,
            fit.fast.sigma,
            LN_1000 / fit.fast.sigma,
            fit.fast.amplitude,
            share,
            fit.slow.sigma,
            fit.slow.amplitude,
            fit.span_s,
            fit.points,
            hop,
            if not_too_slow { "pass" } else { "DROP" },
            if not_too_fast { "pass" } else { "DROP" },
            fit.fast.sigma / factor,
        );
    }
    match prompt_decay_curve(&analysis.decays.partials, factor, &decay) {
        Some(curve) => println!(
            "prompt curve   sigma0 {:.4} /s, sigma1 {:.4} /s, residual {:.4} /s",
            curve.sigma0, curve.sigma1, curve.residual
        ),
        None => println!("prompt curve   refused: fewer than two admitted partials"),
    }
    // What `piano-tuner estimate` prints, for the two tables to be read side by
    // side: the *other* curve, fitted to the whole partial's T60.
    let whole = analysis.decays.curve;
    println!(
        "whole curve    sigma0 {:.4} /s, sigma1 {:.4} /s  (fit_decay_curve, T60 ordinate)",
        whole.sigma0, whole.sigma1
    );
    Ok(())
}
